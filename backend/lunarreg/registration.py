from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from .balancing import spatial_balance
from .config import RegistrationConfig
from .geometry import estimate_best_model, refit_model, reprojection_residuals, transform_points, warp_image
from .geospatial import geospatial_pixel_prior
from .io import image_from_array, read_image
from .local_warp import rbf_residual_warp
from .matchers import LightGlueMatcher, LoFTRMatcher, SIFTMatcher
from .metrics import registration_metrics
from .preprocess import feature_views, percentile_normalize
from .refine import corner_subpixel, local_ncc_refine
from .types import ImageData, MatchSet, RegistrationResult
from .visualization import draw_matches, make_checkerboard, make_overlay


class RegistrationError(RuntimeError):
    pass


def _combine_match_sets(match_sets: list[MatchSet]) -> MatchSet:
    usable = [matches for matches in match_sets if len(matches)]
    if not usable:
        return MatchSet(
            np.empty((0, 2), dtype=np.float32),
            np.empty((0, 2), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=object),
        )
    source = np.vstack([matches.source_points for matches in usable])
    reference = np.vstack([matches.reference_points for matches in usable])
    scores = np.concatenate([matches.scores for matches in usable])
    labels = np.concatenate(
        [
            matches.labels if matches.labels is not None else np.full(len(matches), "unknown", dtype=object)
            for matches in usable
        ]
    )
    order = np.argsort(-scores)
    retained: list[int] = []
    seen: set[tuple[int, int, int, int]] = set()
    for index in order:
        key = tuple(round(value / 1.5) for value in (*source[index], *reference[index]))
        if key in seen:
            continue
        seen.add(key)
        retained.append(int(index))
    indices = np.asarray(retained, dtype=int)
    return MatchSet(source[indices], reference[indices], scores[indices], labels[indices])


def _run_matcher(
    config: RegistrationConfig,
    source_views: dict[str, np.ndarray],
    reference_views: dict[str, np.ndarray],
) -> MatchSet:
    if config.matcher == "sift":
        return SIFTMatcher(config).match(source_views, reference_views)
    if config.matcher == "lightglue":
        return LightGlueMatcher(config).match(source_views, reference_views)
    if config.matcher == "loftr":
        return LoFTRMatcher(config).match(source_views, reference_views)
    match_sets = [SIFTMatcher(config).match(source_views, reference_views)]
    for matcher_type in (LightGlueMatcher, LoFTRMatcher):
        try:
            match_sets.append(matcher_type(config).match(source_views, reference_views))
        except (RuntimeError, OSError):
            continue
    return _combine_match_sets(match_sets)


def _trim_and_refit(
    name: str,
    source: np.ndarray,
    reference: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray | None,
    config: RegistrationConfig,
) -> tuple[MatchSet, np.ndarray]:
    estimate = refit_model(name, source, reference, config)
    residuals = reprojection_residuals(source, reference, estimate.matrix)
    robust_limit = max(
        config.ransac_threshold_px,
        float(np.median(residuals) + 3.5 * max(np.median(np.abs(residuals - np.median(residuals))), 0.15)),
    )
    keep = residuals <= robust_limit
    if keep.sum() >= config.min_inliers and keep.sum() < len(keep):
        source = source[keep]
        reference = reference[keep]
        scores = scores[keep]
        labels = None if labels is None else labels[keep]
        estimate = refit_model(name, source, reference, config)
    return MatchSet(source, reference, scores, labels), estimate.matrix


def register_images(
    source: ImageData | np.ndarray,
    reference: ImageData | np.ndarray,
    config: RegistrationConfig | None = None,
) -> RegistrationResult:
    cfg = config or RegistrationConfig()
    cfg.validate()
    np.random.seed(cfg.random_seed)
    source_data = source if isinstance(source, ImageData) else image_from_array(source, "source")
    reference_data = (
        reference if isinstance(reference, ImageData) else image_from_array(reference, "reference")
    )
    started = time.perf_counter()

    geospatial_prior = (
        geospatial_pixel_prior(source_data, reference_data) if cfg.use_geospatial_prior else None
    )
    working_source = source_data
    if geospatial_prior is not None:
        working_array = warp_image(
            source_data.array.astype(np.float32), geospatial_prior, reference_data.shape
        )
        working_mask = None
        if source_data.valid_mask is not None:
            working_mask = warp_image(
                source_data.valid_mask.astype(np.uint8),
                geospatial_prior,
                reference_data.shape,
                interpolation=0,
            ).astype(bool)
        working_source = ImageData(
            array=working_array,
            name=f"{source_data.name}_geospatial_prior",
            valid_mask=working_mask,
            metadata={"geospatial_prior_applied": True},
        )

    source_views = feature_views(working_source.array, working_source.valid_mask, cfg.use_gradient_view)
    reference_views = feature_views(reference_data.array, reference_data.valid_mask, cfg.use_gradient_view)
    tentative = _run_matcher(cfg, source_views, reference_views)
    if len(tentative) < cfg.min_tentative_matches:
        raise RegistrationError(
            f"Only {len(tentative)} tentative matches were found; at least {cfg.min_tentative_matches} are required."
        )

    try:
        initial = estimate_best_model(tentative.source_points, tentative.reference_points, cfg)
    except ValueError as exc:
        raise RegistrationError(str(exc)) from exc
    geometric_indices = np.flatnonzero(initial.inlier_mask)
    geometric = tentative.subset(geometric_indices)
    if len(geometric) < cfg.min_inliers:
        raise RegistrationError(f"Only {len(geometric)} geometric inliers were found")

    balanced, _ = spatial_balance(
        geometric,
        working_source.shape,
        reference_data.shape,
        cfg.grid_rows,
        cfg.grid_cols,
        cfg.max_matches_per_cell,
    )
    if len(balanced) < cfg.min_inliers:
        balanced = geometric

    source_points = balanced.source_points.copy()
    reference_points = balanced.reference_points.copy()
    if cfg.enable_corner_subpixel:
        source_points = corner_subpixel(source_views["intensity"], source_points, cfg.corner_window)
        reference_points = corner_subpixel(reference_views["intensity"], reference_points, cfg.corner_window)

    final_matches, matrix = _trim_and_refit(
        initial.name,
        source_points,
        reference_points,
        balanced.scores,
        balanced.labels,
        cfg,
    )
    correlations = None
    if cfg.enable_local_ncc_refinement and len(final_matches) >= cfg.min_inliers:
        ncc_source_points = final_matches.source_points.copy()
        ncc_targets, accepted, correlations_all = local_ncc_refine(
            source_views.get("gradient", source_views["intensity"]),
            reference_views.get("gradient", reference_views["intensity"]),
            final_matches.source_points,
            matrix,
            cfg.local_patch_size,
            cfg.local_search_radius,
            cfg.local_min_correlation,
        )
        if int(accepted.sum()) >= cfg.min_inliers:
            updated_reference = final_matches.reference_points.copy()
            updated_reference[accepted] = ncc_targets[accepted]
            final_matches, matrix = _trim_and_refit(
                initial.name,
                final_matches.source_points,
                updated_reference,
                final_matches.scores,
                final_matches.labels,
                cfg,
            )
            # Recompute correlations for the retained coordinate set by nearest source point.
            correlations = np.full(len(final_matches), np.nan, dtype=np.float32)
            for index, point in enumerate(final_matches.source_points):
                distances = np.linalg.norm(ncc_source_points - point, axis=1)
                nearest = int(np.argmin(distances))
                if accepted[nearest]:
                    correlations[index] = correlations_all[nearest]

    if geospatial_prior is not None:
        inverse_prior = np.linalg.inv(geospatial_prior)
        original_tentative = MatchSet(
            transform_points(tentative.source_points, inverse_prior),
            tentative.reference_points,
            tentative.scores,
            tentative.labels,
        )
        original_final = MatchSet(
            transform_points(final_matches.source_points, inverse_prior),
            final_matches.reference_points,
            final_matches.scores,
            final_matches.labels,
        )
        matrix = matrix @ geospatial_prior
    else:
        original_tentative = tentative
        original_final = final_matches

    registered = warp_image(source_data.array.astype(np.float32), matrix, reference_data.shape)
    if cfg.enable_local_rbf_warp and len(original_final) >= cfg.local_warp_min_points:
        predicted = transform_points(original_final.source_points, matrix)
        registered = rbf_residual_warp(registered, predicted, original_final.reference_points)

    metrics = registration_metrics(
        tentative_count=len(tentative),
        geometric_inlier_count=len(geometric),
        source_points=original_final.source_points,
        reference_points=original_final.reference_points,
        matrix=matrix,
        source_shape=source_data.shape,
        reference_shape=reference_data.shape,
        grid_rows=cfg.grid_rows,
        grid_cols=cfg.grid_cols,
        correlations=correlations,
    )
    metrics["model"] = initial.name
    metrics["runtime_seconds"] = float(time.perf_counter() - started)
    metrics["matcher"] = cfg.matcher
    metrics["geospatial_prior_applied"] = geospatial_prior is not None
    coverage = float(metrics["uniform_grid_coverage"])
    status = "success" if coverage >= cfg.minimum_grid_coverage else "warning"
    message = (
        "Registration passed the configured quality gate."
        if status == "success"
        else "Registration completed, but control-point coverage is below the configured target."
    )
    reference_preview = percentile_normalize(reference_data.array, reference_data.valid_mask)
    registered_preview = percentile_normalize(registered)
    overlay = make_overlay(registered_preview, reference_preview)
    checkerboard = make_checkerboard(registered_preview, reference_preview)
    match_visualization = draw_matches(
        percentile_normalize(source_data.array, source_data.valid_mask),
        reference_views["intensity"],
        original_final.source_points,
        original_final.reference_points,
    )
    return RegistrationResult(
        status=status,
        message=message,
        model_name=initial.name,
        matrix=matrix,
        tentative_matches=original_tentative,
        final_matches=original_final,
        registered_image=registered,
        reference_preview=reference_preview,
        overlay=overlay,
        checkerboard=checkerboard,
        match_visualization=match_visualization,
        metrics=metrics,
        source=source_data,
        reference=reference_data,
    )


def register_files(
    source_path: str | Path,
    reference_path: str | Path,
    config: RegistrationConfig | None = None,
) -> RegistrationResult:
    return register_images(read_image(source_path), read_image(reference_path), config)

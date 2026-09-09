from __future__ import annotations

import cv2
import numpy as np

from .config import RegistrationConfig
from .types import ModelEstimate

MODEL_COMPLEXITY = {"similarity": 4, "affine": 6, "homography": 8}


def as_homography(matrix: np.ndarray) -> np.ndarray:
    value = np.asarray(matrix, dtype=np.float64)
    if value.shape == (3, 3):
        return value
    if value.shape == (2, 3):
        return np.vstack([value, [0.0, 0.0, 1.0]])
    raise ValueError(f"Unexpected transform shape {value.shape}")


def transform_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 1, 2)
    projected = cv2.perspectiveTransform(pts, as_homography(matrix))
    return projected.reshape(-1, 2).astype(np.float32)


def reprojection_residuals(source: np.ndarray, reference: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    predicted = transform_points(source, matrix)
    return np.linalg.norm(predicted - np.asarray(reference, dtype=np.float32), axis=1)


def _estimate_specific(
    name: str,
    source: np.ndarray,
    reference: np.ndarray,
    config: RegistrationConfig,
    robust: bool = True,
) -> ModelEstimate | None:
    if len(source) < (4 if name == "homography" else 3):
        return None
    mask = None
    matrix = None
    threshold = config.ransac_threshold_px
    confidence = config.ransac_confidence
    max_iters = config.ransac_max_iterations

    if name == "similarity":
        method = cv2.RANSAC if robust else cv2.LMEDS
        affine, mask = cv2.estimateAffinePartial2D(
            source,
            reference,
            method=method,
            ransacReprojThreshold=threshold,
            maxIters=max_iters,
            confidence=confidence,
            refineIters=15,
        )
        if affine is not None:
            matrix = as_homography(affine)
    elif name == "affine":
        method = cv2.RANSAC if robust else cv2.LMEDS
        affine, mask = cv2.estimateAffine2D(
            source,
            reference,
            method=method,
            ransacReprojThreshold=threshold,
            maxIters=max_iters,
            confidence=confidence,
            refineIters=15,
        )
        if affine is not None:
            matrix = as_homography(affine)
    elif name == "homography":
        if robust:
            method = getattr(cv2, "USAC_MAGSAC", cv2.RANSAC)
            matrix, mask = cv2.findHomography(
                source,
                reference,
                method=method,
                ransacReprojThreshold=threshold,
                maxIters=max_iters,
                confidence=confidence,
            )
        else:
            matrix, mask = cv2.findHomography(source, reference, method=0)
    else:
        raise ValueError(name)

    if matrix is None:
        return None
    matrix = as_homography(matrix)
    if not np.all(np.isfinite(matrix)) or abs(np.linalg.det(matrix)) < 1e-12:
        return None
    residuals = reprojection_residuals(source, reference, matrix)
    if mask is None:
        inliers = residuals <= threshold
    else:
        inliers = np.asarray(mask).reshape(-1).astype(bool)
    if inliers.sum() == 0:
        return None
    median = float(np.median(residuals[inliers]))
    ratio = float(inliers.mean())
    # Lower is better. Complexity only breaks close ties.
    score = median + 0.035 * MODEL_COMPLEXITY[name] + 0.45 * (1.0 - ratio)
    return ModelEstimate(name, matrix, inliers, residuals, score)


def estimate_best_model(
    source: np.ndarray, reference: np.ndarray, config: RegistrationConfig
) -> ModelEstimate:
    candidates = (
        [config.transform_model]
        if config.transform_model != "auto"
        else ["similarity", "affine", "homography"]
    )
    estimates = [
        estimate
        for name in candidates
        if (estimate := _estimate_specific(name, source, reference, config, robust=True)) is not None
        and int(estimate.inlier_mask.sum()) >= config.min_inliers
    ]
    if not estimates:
        raise ValueError("No transformation model produced enough inliers")
    if len(estimates) == 1:
        return estimates[0]

    best_accuracy = min(
        estimates, key=lambda estimate: float(np.median(estimate.residuals[estimate.inlier_mask]))
    )
    best_median = float(np.median(best_accuracy.residuals[best_accuracy.inlier_mask]))
    best_count = int(best_accuracy.inlier_mask.sum())
    tolerance = max(best_median * 1.28, best_median + 0.35)

    for name in ["similarity", "affine", "homography"]:
        for estimate in estimates:
            median = float(np.median(estimate.residuals[estimate.inlier_mask]))
            count = int(estimate.inlier_mask.sum())
            if (
                estimate.name == name
                and median <= tolerance
                and count >= max(config.min_inliers, int(0.82 * best_count))
            ):
                return estimate
    return min(estimates, key=lambda estimate: estimate.score)


def refit_model(
    name: str,
    source: np.ndarray,
    reference: np.ndarray,
    config: RegistrationConfig,
) -> ModelEstimate:
    estimate = _estimate_specific(name, source, reference, config, robust=False)
    if estimate is None:
        estimate = _estimate_specific(name, source, reference, config, robust=True)
    if estimate is None:
        raise ValueError(f"Could not refit {name} model")
    return estimate


def warp_image(
    image: np.ndarray, matrix: np.ndarray, output_shape: tuple[int, int], interpolation=cv2.INTER_CUBIC
) -> np.ndarray:
    height, width = output_shape
    return cv2.warpPerspective(
        image,
        as_homography(matrix),
        (width, height),
        flags=interpolation,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

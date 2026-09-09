from __future__ import annotations

from typing import Any

import numpy as np

from .balancing import grid_coverage
from .geometry import reprojection_residuals, transform_points


def summarize_residuals(residuals: np.ndarray) -> dict[str, float]:
    values = np.asarray(residuals, dtype=np.float64)
    if values.size == 0:
        return {
            "reprojection_rmse_px": float("nan"),
            "reprojection_median_px": float("nan"),
            "reprojection_p95_px": float("nan"),
            "reprojection_max_px": float("nan"),
        }
    return {
        "reprojection_rmse_px": float(np.sqrt(np.mean(values**2))),
        "reprojection_median_px": float(np.median(values)),
        "reprojection_p95_px": float(np.percentile(values, 95)),
        "reprojection_max_px": float(np.max(values)),
    }


def registration_metrics(
    tentative_count: int,
    geometric_inlier_count: int,
    source_points: np.ndarray,
    reference_points: np.ndarray,
    matrix: np.ndarray,
    source_shape: tuple[int, int],
    reference_shape: tuple[int, int],
    grid_rows: int,
    grid_cols: int,
    correlations: np.ndarray | None = None,
) -> dict[str, Any]:
    residuals = reprojection_residuals(source_points, reference_points, matrix)
    source_coverage = grid_coverage(source_points, source_shape, grid_rows, grid_cols)
    reference_coverage = grid_coverage(reference_points, reference_shape, grid_rows, grid_cols)
    metrics: dict[str, Any] = {
        "tentative_match_count": int(tentative_count),
        "geometric_inlier_count": int(geometric_inlier_count),
        "final_control_point_count": len(source_points),
        "inlier_ratio": float(geometric_inlier_count / max(tentative_count, 1)),
        "source_grid_coverage": source_coverage,
        "reference_grid_coverage": reference_coverage,
        "uniform_grid_coverage": min(source_coverage, reference_coverage),
        **summarize_residuals(residuals),
    }
    if correlations is not None:
        finite = correlations[np.isfinite(correlations)]
        metrics["subpixel_refined_count"] = int(finite.size)
        metrics["subpixel_mean_correlation"] = float(finite.mean()) if finite.size else None
    return metrics


def ground_truth_transform_error(
    estimated: np.ndarray,
    ground_truth: np.ndarray,
    shape: tuple[int, int],
    rows: int = 12,
    cols: int = 16,
    margin: float = 0.08,
) -> dict[str, float]:
    height, width = shape
    xs = np.linspace(width * margin, width * (1.0 - margin), cols)
    ys = np.linspace(height * margin, height * (1.0 - margin), rows)
    grid_x, grid_y = np.meshgrid(xs, ys)
    points = np.column_stack([grid_x.ravel(), grid_y.ravel()]).astype(np.float32)
    predicted = transform_points(points, estimated)
    truth = transform_points(points, ground_truth)
    errors = np.linalg.norm(predicted - truth, axis=1)
    return {
        "ground_truth_grid_rmse_px": float(np.sqrt(np.mean(errors**2))),
        "ground_truth_grid_median_px": float(np.median(errors)),
        "ground_truth_grid_p95_px": float(np.percentile(errors, 95)),
    }

from __future__ import annotations

import cv2
import numpy as np


def rbf_residual_warp(
    globally_warped: np.ndarray,
    predicted_points: np.ndarray,
    target_points: np.ndarray,
    smoothing: float = 2.0,
    grid_step: int = 48,
) -> np.ndarray:
    """Apply a smooth residual displacement field after a global transform.

    The global model remains the authoritative reported transform. This optional
    stage is intended for orthorectified imagery with small spatially varying
    residuals, not for replacing sensor-model geometry.
    """
    try:
        from scipy.interpolate import RBFInterpolator
    except ImportError as exc:
        raise RuntimeError("scipy is required for local RBF warping") from exc

    predicted = np.asarray(predicted_points, dtype=np.float64)
    target = np.asarray(target_points, dtype=np.float64)
    displacement = target - predicted
    if len(predicted) < 6:
        return globally_warped

    # Anchor the image boundary with zero displacement to avoid extrapolation.
    height, width = globally_warped.shape[:2]
    anchors = np.asarray(
        [
            [0, 0],
            [width / 2, 0],
            [width - 1, 0],
            [0, height / 2],
            [width - 1, height / 2],
            [0, height - 1],
            [width / 2, height - 1],
            [width - 1, height - 1],
        ],
        dtype=np.float64,
    )
    nodes = np.vstack([predicted, anchors])
    values = np.vstack([displacement, np.zeros((len(anchors), 2), dtype=np.float64)])
    model = RBFInterpolator(
        nodes,
        values,
        kernel="thin_plate_spline",
        smoothing=smoothing,
        neighbors=min(40, len(nodes)),
    )

    coarse_x = np.arange(0, width + grid_step, grid_step, dtype=np.float64)
    coarse_y = np.arange(0, height + grid_step, grid_step, dtype=np.float64)
    coarse_x[-1] = width - 1
    coarse_y[-1] = height - 1
    grid_x, grid_y = np.meshgrid(coarse_x, coarse_y)
    query = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    field = model(query).reshape(len(coarse_y), len(coarse_x), 2).astype(np.float32)
    field = cv2.resize(field, (width, height), interpolation=cv2.INTER_CUBIC)
    field = np.clip(field, -24.0, 24.0)

    yy, xx = np.indices((height, width), dtype=np.float32)
    map_x = xx - field[..., 0]
    map_y = yy - field[..., 1]
    return cv2.remap(
        globally_warped,
        map_x,
        map_y,
        interpolation=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

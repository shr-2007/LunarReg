from __future__ import annotations

import numpy as np

from .geometry import transform_points
from .types import ImageData


def _affine_matrix(transform) -> np.ndarray:
    return np.asarray(
        [
            [float(transform.a), float(transform.b), float(transform.c)],
            [float(transform.d), float(transform.e), float(transform.f)],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _bbox(points: np.ndarray) -> tuple[float, float, float, float]:
    return (
        float(points[:, 0].min()),
        float(points[:, 1].min()),
        float(points[:, 0].max()),
        float(points[:, 1].max()),
    )


def _intersection_fraction(box: tuple[float, float, float, float], shape: tuple[int, int]) -> float:
    x0, y0, x1, y1 = box
    height, width = shape
    ix0, iy0 = max(0.0, x0), max(0.0, y0)
    ix1, iy1 = min(float(width), x1), min(float(height), y1)
    intersection = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    area = max(1.0, (x1 - x0) * (y1 - y0))
    return float(intersection / area)


def geospatial_pixel_prior(source: ImageData, reference: ImageData) -> np.ndarray | None:
    """Return a source-pixel to reference-pixel affine prior when safe.

    This uses only same-CRS affine transforms. Different planetary projections
    should first be harmonized with ISIS or rasterio.warp.reproject.
    """
    if source.transform is None or reference.transform is None:
        return None
    if source.crs is None or reference.crs is None or str(source.crs) != str(reference.crs):
        return None
    try:
        prior = np.linalg.inv(_affine_matrix(reference.transform)) @ _affine_matrix(source.transform)
    except (AttributeError, np.linalg.LinAlgError, TypeError, ValueError):
        return None
    height, width = source.shape
    corners = np.asarray([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)
    projected = transform_points(corners, prior)
    if _intersection_fraction(_bbox(projected), reference.shape) < 0.02:
        return None
    return prior / prior[2, 2]

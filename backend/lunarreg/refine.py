from __future__ import annotations

import cv2
import numpy as np

from .geometry import transform_points, warp_image


def corner_subpixel(image_u8: np.ndarray, points: np.ndarray, window: int = 4) -> np.ndarray:
    if len(points) == 0:
        return np.asarray(points, dtype=np.float32)
    refined = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2).copy()
    height, width = image_u8.shape
    valid = (
        (refined[:, 0, 0] >= window + 1)
        & (refined[:, 0, 0] < width - window - 1)
        & (refined[:, 0, 1] >= window + 1)
        & (refined[:, 0, 1] < height - window - 1)
    )
    if valid.any():
        subset = refined[valid].copy()
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001)
        try:
            cv2.cornerSubPix(
                image_u8.astype(np.float32) / 255.0,
                subset,
                (window, window),
                (-1, -1),
                criteria,
            )
            movement = np.linalg.norm(subset[:, 0] - refined[valid, 0], axis=1)
            safe = movement <= max(1.5, window * 0.75)
            valid_indices = np.flatnonzero(valid)
            refined[valid_indices[safe]] = subset[safe]
        except cv2.error:
            pass
    return refined.reshape(-1, 2)


def _quadratic_peak(values: np.ndarray, x: int, y: int) -> tuple[float, float]:
    height, width = values.shape
    if x <= 0 or x >= width - 1 or y <= 0 or y >= height - 1:
        return 0.0, 0.0

    def vertex(left: float, center: float, right: float) -> float:
        denominator = left - 2.0 * center + right
        if abs(denominator) < 1e-8:
            return 0.0
        return float(np.clip(0.5 * (left - right) / denominator, -0.75, 0.75))

    return (
        vertex(values[y, x - 1], values[y, x], values[y, x + 1]),
        vertex(values[y - 1, x], values[y, x], values[y + 1, x]),
    )


def local_ncc_refine(
    source_feature: np.ndarray,
    reference_feature: np.ndarray,
    source_points: np.ndarray,
    matrix: np.ndarray,
    patch_size: int,
    search_radius: int,
    minimum_correlation: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    predicted = transform_points(source_points, matrix)
    warped = warp_image(source_feature, matrix, reference_feature.shape, interpolation=cv2.INTER_LINEAR)
    half = patch_size // 2
    search_size = patch_size + 2 * search_radius
    refined = predicted.copy()
    accepted = np.zeros(len(source_points), dtype=bool)
    correlations = np.full(len(source_points), np.nan, dtype=np.float32)
    height, width = reference_feature.shape

    for index, (x, y) in enumerate(predicted):
        if (
            x < half + search_radius + 2
            or x >= width - half - search_radius - 2
            or y < half + search_radius + 2
            or y >= height - half - search_radius - 2
        ):
            continue
        patch = cv2.getRectSubPix(warped, (patch_size, patch_size), (float(x), float(y)))
        search = cv2.getRectSubPix(reference_feature, (search_size, search_size), (float(x), float(y)))
        if float(patch.std()) < 2.0 or float(search.std()) < 2.0:
            continue
        response = cv2.matchTemplate(search, patch, cv2.TM_CCOEFF_NORMED)
        _, maximum, _, location = cv2.minMaxLoc(response)
        if not np.isfinite(maximum) or maximum < minimum_correlation:
            continue
        peak_x, peak_y = location
        dx, dy = _quadratic_peak(response, peak_x, peak_y)
        offset_x = peak_x + dx - search_radius
        offset_y = peak_y + dy - search_radius
        if abs(offset_x) > search_radius + 0.8 or abs(offset_y) > search_radius + 0.8:
            continue
        refined[index] = [x + offset_x, y + offset_y]
        accepted[index] = True
        correlations[index] = maximum

    return refined, accepted, correlations

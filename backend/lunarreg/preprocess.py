from __future__ import annotations

import cv2
import numpy as np


def percentile_normalize(
    image: np.ndarray,
    valid_mask: np.ndarray | None = None,
    lower: float = 1.0,
    upper: float = 99.0,
) -> np.ndarray:
    data = np.asarray(image, dtype=np.float32)
    valid = np.isfinite(data)
    if valid_mask is not None:
        valid &= valid_mask.astype(bool)
    values = data[valid]
    if values.size == 0:
        return np.zeros(data.shape, dtype=np.uint8)
    low, high = np.percentile(values, [lower, upper])
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        low = float(np.nanmin(values))
        high = float(np.nanmax(values))
    if high <= low:
        return np.zeros(data.shape, dtype=np.uint8)
    normalized = np.clip((data - low) / (high - low), 0.0, 1.0)
    normalized[~valid] = 0.0
    return np.round(normalized * 255.0).astype(np.uint8)


def local_contrast(image_u8: np.ndarray) -> np.ndarray:
    height, width = image_u8.shape
    tile = max(4, min(12, round(min(height, width) / 180)))
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(tile, tile))
    enhanced = clahe.apply(image_u8)
    return cv2.GaussianBlur(enhanced, (3, 3), 0.6)


def gradient_magnitude(image_u8: np.ndarray) -> np.ndarray:
    smooth = cv2.GaussianBlur(image_u8, (0, 0), 1.0)
    gx = cv2.Scharr(smooth, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(smooth, cv2.CV_32F, 0, 1)
    magnitude = cv2.magnitude(gx, gy)
    return percentile_normalize(magnitude, lower=2.0, upper=98.5)


def feature_views(
    image: np.ndarray, valid_mask: np.ndarray | None = None, include_gradient: bool = True
) -> dict[str, np.ndarray]:
    normalized = percentile_normalize(image, valid_mask)
    enhanced = local_contrast(normalized)
    views = {"intensity": enhanced}
    if include_gradient:
        views["gradient"] = gradient_magnitude(enhanced)
    return views


def resize_for_matching(image: np.ndarray, max_dimension: int) -> tuple[np.ndarray, float, float]:
    height, width = image.shape[:2]
    largest = max(height, width)
    if largest <= max_dimension:
        return image, 1.0, 1.0
    scale = max_dimension / float(largest)
    new_width = max(1, round(width * scale))
    new_height = max(1, round(height * scale))
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return resized, new_width / width, new_height / height


def valid_feature_mask(
    valid_mask: np.ndarray | None, shape: tuple[int, int], erosion: int = 3
) -> np.ndarray | None:
    if valid_mask is None:
        return None
    mask = valid_mask.astype(np.uint8) * 255
    if mask.shape != shape:
        mask = cv2.resize(mask, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    if erosion > 0:
        kernel = np.ones((erosion, erosion), dtype=np.uint8)
        mask = cv2.erode(mask, kernel)
    return mask

from __future__ import annotations

import cv2
import numpy as np

from .preprocess import percentile_normalize


def as_rgb(image: np.ndarray) -> np.ndarray:
    data = percentile_normalize(image) if image.dtype != np.uint8 else image
    if data.ndim == 2:
        return cv2.cvtColor(data, cv2.COLOR_GRAY2RGB)
    return data[..., :3].copy()


def make_overlay(registered: np.ndarray, reference: np.ndarray) -> np.ndarray:
    reg = percentile_normalize(registered)
    ref = percentile_normalize(reference)
    output = np.zeros((*ref.shape, 3), dtype=np.uint8)
    output[..., 0] = reg
    output[..., 1] = ref
    output[..., 2] = ref
    return output


def make_checkerboard(registered: np.ndarray, reference: np.ndarray, tiles: int = 12) -> np.ndarray:
    reg = percentile_normalize(registered)
    ref = percentile_normalize(reference)
    height, width = ref.shape
    yy, xx = np.indices((height, width))
    cell_h = max(1, height // tiles)
    cell_w = max(1, width // tiles)
    selector = ((yy // cell_h) + (xx // cell_w)) % 2 == 0
    output = ref.copy()
    output[selector] = reg[selector]
    return output


def draw_matches(
    source: np.ndarray,
    reference: np.ndarray,
    source_points: np.ndarray,
    reference_points: np.ndarray,
    max_draw: int = 180,
) -> np.ndarray:
    source_rgb = as_rgb(source)
    reference_rgb = as_rgb(reference)
    height = max(source_rgb.shape[0], reference_rgb.shape[0])
    width = source_rgb.shape[1] + reference_rgb.shape[1]
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[: source_rgb.shape[0], : source_rgb.shape[1]] = source_rgb
    offset = source_rgb.shape[1]
    canvas[: reference_rgb.shape[0], offset:] = reference_rgb
    if len(source_points) > max_draw:
        indices = np.linspace(0, len(source_points) - 1, max_draw).astype(int)
    else:
        indices = np.arange(len(source_points))
    palette = [(36, 198, 220), (40, 100, 220), (92, 211, 134), (255, 179, 71)]
    for count, index in enumerate(indices):
        color = palette[count % len(palette)]
        p0 = tuple(np.round(source_points[index]).astype(int))
        p1_raw = np.round(reference_points[index]).astype(int)
        p1 = (int(p1_raw[0] + offset), int(p1_raw[1]))
        cv2.circle(canvas, p0, 3, color, -1, cv2.LINE_AA)
        cv2.circle(canvas, p1, 3, color, -1, cv2.LINE_AA)
        cv2.line(canvas, p0, p1, color, 1, cv2.LINE_AA)
    return canvas

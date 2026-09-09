from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter


@dataclass(slots=True)
class SyntheticPair:
    source: np.ndarray
    reference: np.ndarray
    ground_truth_matrix: np.ndarray
    elevation: np.ndarray


def _lunar_elevation(shape: tuple[int, int], seed: int, crater_count: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    height, width = shape
    terrain = gaussian_filter(rng.normal(size=shape), 18) * 2.5
    terrain += gaussian_filter(rng.normal(size=shape), 65) * 7.0
    yy, xx = np.indices(shape, dtype=np.float32)
    for _ in range(crater_count):
        radius = float(np.exp(rng.uniform(np.log(5.0), np.log(min(shape) * 0.105))))
        cx = rng.uniform(-0.05 * width, 1.05 * width)
        cy = rng.uniform(-0.05 * height, 1.05 * height)
        distance = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        depth = rng.uniform(0.8, 2.4) * np.sqrt(radius)
        bowl = -depth * np.exp(-(distance**2) / (2.0 * (0.50 * radius) ** 2))
        rim = 0.55 * depth * np.exp(-((distance - radius) ** 2) / (2.0 * (0.13 * radius) ** 2))
        ejecta = 0.12 * depth * np.exp(-((distance - 1.35 * radius) ** 2) / (2.0 * (0.30 * radius) ** 2))
        terrain += bowl + rim + ejecta
    return terrain.astype(np.float32)


def _hillshade(elevation: np.ndarray, azimuth_deg: float, elevation_deg: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    dy, dx = np.gradient(elevation)
    normal = np.dstack([-dx, -dy, np.ones_like(elevation)])
    normal /= np.maximum(np.linalg.norm(normal, axis=2, keepdims=True), 1e-6)
    azimuth = np.deg2rad(azimuth_deg)
    altitude = np.deg2rad(elevation_deg)
    light = np.asarray(
        [
            np.cos(altitude) * np.sin(azimuth),
            np.cos(altitude) * np.cos(azimuth),
            np.sin(altitude),
        ]
    )
    shade = np.clip(np.sum(normal * light, axis=2), -0.3, 1.0)
    albedo = 0.82 + 0.18 * gaussian_filter(rng.normal(size=elevation.shape), 5)
    image = (shade + 0.30) / 1.30 * albedo
    image += gaussian_filter(rng.normal(scale=0.018, size=elevation.shape), 0.7)
    low, high = np.percentile(image, [0.5, 99.5])
    return np.round(np.clip((image - low) / (high - low), 0, 1) * 255).astype(np.uint8)


def generate_synthetic_pair(
    shape: tuple[int, int] = (720, 960),
    seed: int = 42,
    illumination_change: float = 20.0,
    perspective: bool = True,
) -> SyntheticPair:
    elevation = _lunar_elevation(shape, seed, crater_count=115)
    source = _hillshade(elevation, azimuth_deg=48.0, elevation_deg=34.0, seed=seed + 1)
    alternate = _hillshade(
        elevation,
        azimuth_deg=48.0 + illumination_change,
        elevation_deg=29.0,
        seed=seed + 2,
    )
    height, width = shape
    center = (width / 2.0, height / 2.0)
    affine = cv2.getRotationMatrix2D(center, 4.2, 1.025)
    affine[:, 2] += [22.4, -15.7]
    matrix = np.vstack([affine, [0.0, 0.0, 1.0]]).astype(np.float64)
    if perspective:
        perspective_term = np.asarray(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.6e-5, -1.15e-5, 1.0]],
            dtype=np.float64,
        )
        matrix = perspective_term @ matrix
    reference = cv2.warpPerspective(
        alternate,
        matrix,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return SyntheticPair(source, reference, matrix / matrix[2, 2], elevation)

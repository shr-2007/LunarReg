from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(slots=True)
class ImageData:
    array: np.ndarray
    name: str = "image"
    path: Path | None = None
    valid_mask: np.ndarray | None = None
    transform: Any | None = None
    crs: Any | None = None
    profile: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def shape(self) -> tuple[int, int]:
        return int(self.array.shape[0]), int(self.array.shape[1])


@dataclass(slots=True)
class MatchSet:
    source_points: np.ndarray
    reference_points: np.ndarray
    scores: np.ndarray
    labels: np.ndarray | None = None

    def __len__(self) -> int:
        return int(self.source_points.shape[0])

    def subset(self, indices: np.ndarray) -> MatchSet:
        labels = None if self.labels is None else self.labels[indices]
        return MatchSet(
            source_points=self.source_points[indices],
            reference_points=self.reference_points[indices],
            scores=self.scores[indices],
            labels=labels,
        )


@dataclass(slots=True)
class ModelEstimate:
    name: str
    matrix: np.ndarray
    inlier_mask: np.ndarray
    residuals: np.ndarray
    score: float


@dataclass(slots=True)
class RegistrationResult:
    status: str
    message: str
    model_name: str
    matrix: np.ndarray
    tentative_matches: MatchSet
    final_matches: MatchSet
    registered_image: np.ndarray
    reference_preview: np.ndarray
    overlay: np.ndarray
    checkerboard: np.ndarray
    match_visualization: np.ndarray
    metrics: dict[str, Any]
    source: ImageData
    reference: ImageData

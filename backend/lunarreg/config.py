from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class RegistrationConfig:
    """Configuration for the end-to-end registration pipeline."""

    matcher: str = "sift"
    transform_model: str = "auto"
    use_geospatial_prior: bool = True
    max_dimension: int = 2600
    sift_features: int = 14000
    sift_contrast_threshold: float = 0.015
    sift_edge_threshold: float = 12.0
    ratio_test: float = 0.80
    require_mutual: bool = True
    use_gradient_view: bool = True
    min_tentative_matches: int = 12
    min_inliers: int = 8
    ransac_threshold_px: float = 4.0
    ransac_confidence: float = 0.999
    ransac_max_iterations: int = 15000
    grid_rows: int = 8
    grid_cols: int = 8
    max_matches_per_cell: int = 6
    minimum_grid_coverage: float = 0.08
    enable_corner_subpixel: bool = True
    corner_window: int = 4
    enable_local_ncc_refinement: bool = True
    local_patch_size: int = 31
    local_search_radius: int = 5
    local_min_correlation: float = 0.18
    enable_local_rbf_warp: bool = False
    local_warp_min_points: int = 20
    random_seed: int = 42

    def validate(self) -> None:
        if self.matcher not in {"sift", "lightglue", "loftr", "ensemble"}:
            raise ValueError(f"Unsupported matcher: {self.matcher}")
        if self.transform_model not in {"auto", "similarity", "affine", "homography"}:
            raise ValueError(f"Unsupported transform model: {self.transform_model}")
        if self.max_dimension < 256:
            raise ValueError("max_dimension must be at least 256")
        if not 0.0 < self.ratio_test < 1.0:
            raise ValueError("ratio_test must be between 0 and 1")
        if self.local_patch_size % 2 == 0 or self.local_patch_size < 9:
            raise ValueError("local_patch_size must be an odd integer >= 9")
        if self.grid_rows < 1 or self.grid_cols < 1:
            raise ValueError("grid dimensions must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_config(
    path: str | Path | None = None, overrides: dict[str, Any] | None = None
) -> RegistrationConfig:
    values: dict[str, Any] = {}
    if path is not None:
        with Path(path).open("r", encoding="utf-8") as stream:
            loaded = yaml.safe_load(stream) or {}
        if not isinstance(loaded, dict):
            raise ValueError("Configuration file must contain a YAML mapping")
        values.update(loaded)
    if overrides:
        values.update({key: value for key, value in overrides.items() if value is not None})

    allowed = {field.name for field in fields(RegistrationConfig)}
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError(f"Unknown configuration fields: {', '.join(unknown)}")

    config = RegistrationConfig(**values)
    config.validate()
    return config

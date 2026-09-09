from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from .geometry import reprojection_residuals
from .io import save_png, save_registered_geotiff
from .types import RegistrationResult


def _json_safe(value: Any):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def export_result(
    result: RegistrationResult, output_directory: str | Path, make_zip: bool = True
) -> dict[str, Path]:
    output = Path(output_directory).resolve()
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "registered_png": output / "registered_source.png",
        "reference_png": output / "reference_preview.png",
        "overlay_png": output / "registration_overlay.png",
        "checkerboard_png": output / "registration_checkerboard.png",
        "matches_png": output / "corresponding_match_points.png",
        "matches_csv": output / "match_points.csv",
        "metrics_json": output / "metrics.json",
        "transform_json": output / "transformation.json",
    }
    save_png(paths["registered_png"], result.registered_image)
    save_png(paths["reference_png"], result.reference_preview)
    save_png(paths["overlay_png"], result.overlay)
    save_png(paths["checkerboard_png"], result.checkerboard)
    save_png(paths["matches_png"], result.match_visualization)

    residuals = reprojection_residuals(
        result.final_matches.source_points,
        result.final_matches.reference_points,
        result.matrix,
    )
    transform = result.reference.transform
    with paths["matches_csv"].open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "source_x",
                "source_y",
                "reference_x",
                "reference_y",
                "confidence",
                "feature_view",
                "residual_px",
                "map_x",
                "map_y",
            ]
        )
        labels = result.final_matches.labels
        for index, (source, reference, score, residual) in enumerate(
            zip(
                result.final_matches.source_points,
                result.final_matches.reference_points,
                result.final_matches.scores,
                residuals,
            )
        ):
            map_x = map_y = ""
            if transform is not None:
                try:
                    map_x, map_y = transform * (float(reference[0]), float(reference[1]))
                except Exception:  # noqa: BLE001 - optional third-party affine transform protocol.
                    map_x = map_y = ""
            writer.writerow(
                [
                    float(source[0]),
                    float(source[1]),
                    float(reference[0]),
                    float(reference[1]),
                    float(score),
                    "" if labels is None else str(labels[index]),
                    float(residual),
                    map_x,
                    map_y,
                ]
            )

    paths["metrics_json"].write_text(
        json.dumps(
            _json_safe(
                {
                    "status": result.status,
                    "message": result.message,
                    **result.metrics,
                }
            ),
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    paths["transform_json"].write_text(
        json.dumps(
            _json_safe(
                {
                    "model": result.model_name,
                    "source_to_reference_matrix": result.matrix,
                    "coordinate_order": "x, y",
                }
            ),
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    if save_registered_geotiff(
        output / "registered_source.tif", result.registered_image.astype(np.float32), result.reference
    ):
        paths["registered_geotiff"] = output / "registered_source.tif"

    if make_zip:
        archive_path = Path(shutil.make_archive(str(output), "zip", root_dir=output))
        paths["archive"] = archive_path
    return paths

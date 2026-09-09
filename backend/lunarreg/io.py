from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from .types import ImageData

RASTER_EXTENSIONS = {".tif", ".tiff", ".img", ".jp2", ".vrt", ".xml"}


def _to_grayscale(array: np.ndarray) -> np.ndarray:
    data = np.asarray(array)
    if data.ndim == 2:
        return data
    if data.ndim == 3 and data.shape[0] <= 8 and data.shape[0] < data.shape[-1]:
        data = np.moveaxis(data, 0, -1)
    if data.ndim != 3:
        raise ValueError(f"Expected a 2-D or 3-D image, got shape {data.shape}")
    if data.shape[2] == 1:
        return data[..., 0]
    rgb = data[..., :3].astype(np.float32)
    return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]


def _read_with_rasterio(path: Path) -> ImageData | None:
    try:
        import rasterio
    except ImportError:
        return None

    try:
        with rasterio.open(path) as dataset:
            array = dataset.read()
            profile = dict(dataset.profile)
            nodata = dataset.nodata
            tags: dict[str, Any] = dict(dataset.tags())
            gray = _to_grayscale(array).astype(np.float32)
            valid = np.isfinite(gray)
            if nodata is not None:
                valid &= gray != nodata
            return ImageData(
                array=gray,
                name=path.stem,
                path=path,
                valid_mask=valid,
                transform=dataset.transform,
                crs=dataset.crs,
                profile=profile,
                metadata={
                    "driver": dataset.driver,
                    "width": dataset.width,
                    "height": dataset.height,
                    "count": dataset.count,
                    "dtype": str(dataset.dtypes[0]),
                    "nodata": nodata,
                    "tags": tags,
                },
            )
    except Exception:  # noqa: BLE001 - fall back to OpenCV/Pillow when a raster driver rejects a file.
        return None


def read_image(path: str | Path) -> ImageData:
    image_path = Path(path).expanduser().resolve()
    if not image_path.exists():
        raise FileNotFoundError(image_path)

    if image_path.suffix.lower() in RASTER_EXTENSIONS:
        raster = _read_with_rasterio(image_path)
        if raster is not None:
            return raster

    raw = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        try:
            with Image.open(image_path) as image:
                raw = np.asarray(image)
        except Exception as exc:
            raise ValueError(
                f"Could not read {image_path}. Install rasterio/GDAL for planetary PDS or GeoTIFF products."
            ) from exc

    if raw.ndim == 3:
        if raw.shape[2] == 4:
            raw = cv2.cvtColor(raw, cv2.COLOR_BGRA2GRAY)
        elif raw.shape[2] >= 3:
            raw = cv2.cvtColor(raw[..., :3], cv2.COLOR_BGR2GRAY)
    gray = _to_grayscale(raw).astype(np.float32)
    valid = np.isfinite(gray)
    return ImageData(
        array=gray,
        name=image_path.stem,
        path=image_path,
        valid_mask=valid,
        metadata={
            "width": int(gray.shape[1]),
            "height": int(gray.shape[0]),
            "dtype": str(raw.dtype),
            "source_format": image_path.suffix.lower(),
        },
    )


def image_from_array(array: np.ndarray, name: str = "image") -> ImageData:
    gray = _to_grayscale(np.asarray(array)).astype(np.float32)
    return ImageData(array=gray, name=name, valid_mask=np.isfinite(gray))


def save_png(path: str | Path, array: np.ndarray) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    data = np.asarray(array)
    if data.dtype != np.uint8:
        finite = data[np.isfinite(data)]
        if finite.size == 0:
            data = np.zeros(data.shape, dtype=np.uint8)
        else:
            low, high = np.percentile(finite, [1, 99])
            if high <= low:
                high = low + 1.0
            data = np.clip((data - low) / (high - low), 0, 1)
            data = np.round(data * 255).astype(np.uint8)
    if data.ndim == 3 and data.shape[2] == 3:
        data = cv2.cvtColor(data, cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(output), data):
        raise OSError(f"Failed to write {output}")


def save_registered_geotiff(path: str | Path, image: np.ndarray, reference: ImageData) -> bool:
    if reference.profile is None:
        return False
    try:
        import rasterio
    except ImportError:
        return False

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    profile = dict(reference.profile)
    profile.update(
        driver="GTiff",
        height=int(image.shape[0]),
        width=int(image.shape[1]),
        count=1,
        dtype=str(image.dtype),
        compress="deflate",
    )
    with rasterio.open(output, "w", **profile) as dataset:
        dataset.write(image, 1)
        dataset.update_tags(REGISTRATION_MODEL="LunarReg")
    return True


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)

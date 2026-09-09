from __future__ import annotations

import os
import shutil
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from lunarreg.config import load_config
from lunarreg.io import save_png
from lunarreg.outputs import export_result
from lunarreg.registration import RegistrationError, register_files, register_images
from lunarreg.synthetic import generate_synthetic_pair

APP_VERSION = "1.0.0"
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_MB", "50")) * 1024 * 1024
RESULT_TTL_SECONDS = int(os.getenv("RESULT_TTL_SECONDS", "7200"))
RESULTS_ROOT = Path(os.getenv("RESULTS_ROOT", tempfile.gettempdir())) / "lunarreg-results"
ALLOWED_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".jp2", ".img"}

app = FastAPI(
    title="LunarReg API",
    version=APP_VERSION,
    description="Sub-pixel registration for Chandrayaan-2 and lunar reference imagery.",
)


def _cleanup_expired_jobs() -> None:
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    cutoff = time.time() - RESULT_TTL_SECONDS
    for path in RESULTS_ROOT.iterdir():
        try:
            if path.is_dir() and path.stat().st_mtime < cutoff:
                shutil.rmtree(path, ignore_errors=True)
        except OSError:
            continue


def _json_safe(value: Any) -> Any:
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
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def _result_payload(job_id: str, result: Any) -> dict[str, Any]:
    prefix = f"/result/{job_id}"
    return _json_safe(
        {
            "job_id": job_id,
            "status": result.status,
            "message": result.message,
            "model": result.model_name,
            "metrics": result.metrics,
            "matrix": result.matrix,
            "assets": {
                "registered": f"{prefix}/result/registered_source.png",
                "reference": f"{prefix}/result/reference_preview.png",
                "overlay": f"{prefix}/result/registration_overlay.png",
                "checkerboard": f"{prefix}/result/registration_checkerboard.png",
                "matches": f"{prefix}/result/corresponding_match_points.png",
            },
            "download": f"{prefix}/result.zip",
        }
    )


def _run_job(source_path: Path, reference_path: Path, options: dict[str, Any]) -> dict[str, Any]:
    job_id = source_path.parent.name
    config = load_config(None, options)
    result = register_files(source_path, reference_path, config)
    export_result(result, source_path.parent / "result")
    return _result_payload(job_id, result)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "version": APP_VERSION, "engine": "opencv-sift"}


@app.post("/register")
async def register_endpoint(
    source: UploadFile = File(...),
    reference: UploadFile = File(...),
    matcher: str = Form("sift"),
    model: str = Form("auto"),
    max_dimension: int = Form(2200),
    subpixel: bool = Form(True),
) -> dict[str, Any]:
    _cleanup_expired_jobs()
    for upload in (source, reference):
        suffix = Path(upload.filename or "").suffix.lower()
        if suffix not in ALLOWED_SUFFIXES:
            raise HTTPException(status_code=415, detail=f"Unsupported image format: {suffix or 'unknown'}")

    source_bytes, reference_bytes = await source.read(), await reference.read()
    if not source_bytes or not reference_bytes:
        raise HTTPException(status_code=422, detail="Both source and reference images are required.")
    if len(source_bytes) > MAX_UPLOAD_BYTES or len(reference_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"Each image must be smaller than {MAX_UPLOAD_BYTES // 1024 // 1024} MB.")

    job_id = uuid.uuid4().hex[:12]
    job_dir = RESULTS_ROOT / job_id
    job_dir.mkdir(parents=True, exist_ok=False)
    source_path = job_dir / f"source{Path(source.filename or '.tif').suffix.lower()}"
    reference_path = job_dir / f"reference{Path(reference.filename or '.tif').suffix.lower()}"
    source_path.write_bytes(source_bytes)
    reference_path.write_bytes(reference_bytes)
    try:
        return _run_job(
            source_path,
            reference_path,
            {
                "matcher": matcher,
                "transform_model": model,
                "max_dimension": max(512, min(int(max_dimension), 4000)),
                "enable_corner_subpixel": subpixel,
            },
        )
    except (RegistrationError, ValueError, RuntimeError, OSError) as exc:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/demo")
def demo_endpoint() -> dict[str, Any]:
    _cleanup_expired_jobs()
    job_id = uuid.uuid4().hex[:12]
    job_dir = RESULTS_ROOT / job_id
    job_dir.mkdir(parents=True, exist_ok=False)
    try:
        synthetic = generate_synthetic_pair(shape=(720, 960), seed=2026)
        source_path = job_dir / "source.png"
        reference_path = job_dir / "reference.png"
        save_png(source_path, synthetic.source)
        save_png(reference_path, synthetic.reference)
        result = register_images(
            synthetic.source,
            synthetic.reference,
            load_config(None, {"matcher": "sift", "transform_model": "auto", "max_dimension": 1800}),
        )
        export_result(result, job_dir / "result")
        payload = _result_payload(job_id, result)
        payload["message"] = "Synthetic lunar pair registered successfully. Upload mission imagery when ready."
        return payload
    except (RegistrationError, ValueError, RuntimeError, OSError, cv2.error) as exc:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/result/{job_id}/{file_path:path}")
def result_file(job_id: str, file_path: str) -> FileResponse:
    if not job_id.isalnum() or ".." in file_path:
        raise HTTPException(status_code=404, detail="Result not found.")
    job_dir = (RESULTS_ROOT / job_id).resolve()
    requested = (job_dir / file_path).resolve()
    if job_dir not in requested.parents or not requested.is_file():
        raise HTTPException(status_code=404, detail="Result not found or expired.")
    media_type = "application/zip" if requested.suffix == ".zip" else None
    return FileResponse(requested, media_type=media_type, filename=requested.name)

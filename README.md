<<<<<<< HEAD
# LunarReg · SIH 2026

An end-to-end web application for registering Chandrayaan-2 optical imagery against lunar reference images. It is designed for the Smart India Hackathon 2026 image-registration statement and reports sub-pixel correspondence quality with auditable outputs.

The repository contains:

- a responsive Next.js/Vinext registration workspace;
- a FastAPI + OpenCV registration engine;
- a one-click synthetic lunar demonstration;
- SIFT matching, robust geometric model selection, spatial balancing, and sub-pixel refinement;
- overlay, checkerboard, match-point, and registered-image views;
- RMSE, inlier ratio, control-point count, coverage, model, and runtime metrics;
- ZIP export with PNG products, CSV match points, metrics, and transformation matrix;
- a Render Blueprint that deploys the UI and API as connected services.

## Architecture

```text
Browser → Vinext UI / API proxy → FastAPI engine → OpenCV pipeline → temporary result bundle
```

The UI talks to the backend through same-origin proxy routes. On Render, `LUNARREG_API_HOSTPORT` is populated from the backend service's private network address, so no public API URL or CORS configuration is needed.

## Local development

Requirements:

- Node.js 22.13+
- Python 3.10+ (3.12 recommended)

Terminal 1 — backend:

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Terminal 2 — frontend:

```bash
npm ci
npm run dev
```

Open `http://localhost:3000`. If the UI uses another backend address, copy `.env.example` to `.env.local` and change `LUNARREG_API_HOSTPORT`.

## Deploy from GitHub to Render

1. Create a new GitHub repository.
2. Push this project to the repository's default branch.
3. In Render, choose **New → Blueprint**.
4. Connect the GitHub repository.
5. Render reads `render.yaml` and creates:
   - `lunarreg-api-sih-2026` — Python/FastAPI service;
   - `lunarreg-ui-sih-2026` — Node/Vinext service.
6. Approve the Blueprint and wait for both health checks to pass.
7. Open the UI service URL and run **Synthetic demo** before uploading mission data.

The Blueprint uses Render's free plan by default. Free web services can spin down while idle, so the first request after inactivity may take longer. For judging, move both services to a paid instance or warm them shortly before the demonstration.

## Input guidance

Accepted UI formats: PNG, JPEG, TIFF, JP2, and IMG, up to 50 MB per image.

- **Source / moving image:** Chandrayaan-2 OHRC, TMC-2, or IIRS optical image.
- **Reference / fixed image:** LRO NAC, SELENE/KAGUYA TC, or another lunar orthoimage.
- Crop or select images with overlapping lunar coordinates.
- Use similar map projection and ground sampling distance when available.
- Radiometrically calibrated or map-projected products are preferred to browse thumbnails.

Open archives:

- [Chandrayaan Data Explorer](https://chmapbrowse.issdc.gov.in/)
- [LROC curated downloads](https://www.lroc.asu.edu/images/downloads)
- [LROC QuickMap](https://quickmap.lroc.asu.edu/)
- [KAGUYA/SELENE archive](https://darts.isas.jaxa.jp/app/pdap/selene/)

The specific SIH evaluation-image link is still listed as **TBD**. Add the released pairs without changing the application contract: source is moving; reference is fixed.

## Output bundle

Each successful job produces:

| File | Purpose |
|---|---|
| `registered_source.png` | Source warped into the reference frame |
| `registration_overlay.png` | False-color visual alignment check |
| `registration_checkerboard.png` | Alternating-tile alignment check |
| `corresponding_match_points.png` | Spatially distributed control points |
| `match_points.csv` | Source/reference coordinates, confidence, and residual |
| `metrics.json` | RMSE, ratios, coverage, counts, model, and runtime |
| `transformation.json` | Source-to-reference transformation matrix |

Result files are temporary and expire after two hours by default. Change `RESULT_TTL_SECONDS` in `render.yaml` if needed.

## Algorithm

1. Read and convert inputs to validated grayscale arrays.
2. Normalize robust intensity and gradient representations.
3. Detect and match SIFT features with ratio and mutual checks.
4. Estimate similarity, affine, or homography transforms with RANSAC.
5. Balance control points across an 8×8 grid.
6. Refine corners and local correlation to sub-pixel coordinates.
7. Refit the transform, warp the source, compute metrics, and export artifacts.

`ensemble` currently retains SIFT as the reliable CPU baseline and automatically uses learned matchers only when their optional packages are installed. Do not select GPU methods on Render's free CPU service.

## Production notes

- The free backend uses one worker to stay within 512 MB RAM.
- Large mission rasters should be cropped to the overlap region before upload.
- For PDS labels, geospatial GeoTIFF metadata, or very large products, add GDAL/rasterio and use a larger Render instance.
- For multi-user production use, move result storage to an object store and run registration as queued background jobs.
- Validate final accuracy against independent expert or mission-provided control points; reprojection RMSE alone is not absolute ground-truth error.

## License

Code is provided for the SIH 2026 proof of concept. Mission imagery remains subject to the terms and citation rules of the supplying archive.
=======
# LunarReg
>>>>>>> 0d07a57f2361b215fa19439f4a201dae49bbd88a

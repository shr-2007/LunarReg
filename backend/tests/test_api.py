from fastapi.testclient import TestClient

from main import app


client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["engine"] == "opencv-sift"


def test_synthetic_demo_and_result_download():
    response = client.post("/demo")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] in {"success", "warning"}
    assert payload["metrics"]["geometric_inlier_count"] >= 8
    assert payload["metrics"]["reprojection_rmse_px"] < 4.0

    archive = client.get(payload["download"])
    assert archive.status_code == 200
    assert archive.headers["content-type"] == "application/zip"
    assert len(archive.content) > 10_000

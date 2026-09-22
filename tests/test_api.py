from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_health_exposes_data_provenance() -> None:
    response = client.get("/api/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["data_mode"] in {"wm811k", "demo"}


def test_wafer_round_trip() -> None:
    wafers = client.get("/api/wafers?limit=1").json()
    assert len(wafers) == 1
    detail = client.get(f"/api/wafers/{wafers[0]['wafer_id']}")
    assert detail.status_code == 200
    assert detail.json()["wafer_map"]


def test_missing_wafer_returns_404() -> None:
    assert client.get("/api/wafers/not-found").status_code == 404


def test_model_status_is_explicit() -> None:
    response = client.get("/api/models/baseline/status")
    assert response.status_code == 200
    assert response.json()["artifact_path"] == "artifacts/baseline.joblib"


def test_prediction_without_artifact_is_not_faked() -> None:
    wafers = client.get("/api/wafers?limit=1").json()
    response = client.get(f"/api/wafers/{wafers[0]['wafer_id']}/prediction")
    assert response.status_code == 503


def test_similarity_and_investigation_workflow() -> None:
    wafer_id = client.get("/api/wafers?limit=1").json()[0]["wafer_id"]
    similar = client.get(f"/api/wafers/{wafer_id}/similar?limit=3")
    assert similar.status_code == 200
    assert len(similar.json()) == 3
    assert all(item["wafer_id"] != wafer_id for item in similar.json())

    investigation = client.get(f"/api/wafers/{wafer_id}/investigation")
    assert investigation.status_code == 200
    body = investigation.json()
    assert body["wafer_id"] == wafer_id
    assert body["recommended_checks"]
    assert "not equipment telemetry" in body["disclaimer"]

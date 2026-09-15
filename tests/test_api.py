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


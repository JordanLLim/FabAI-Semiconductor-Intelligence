from collections import Counter
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse

from app.schemas import DatasetSummary, HealthResponse, WaferDetail, WaferSummary
from src.data.repository import WaferRecord, WaferRepository
from src.features.wafer import extract_features

DATASET_PATH = Path("data/raw/LSWMD.pkl")
repository = WaferRepository(DATASET_PATH)
app = FastAPI(title="FAB.AI Semiconductor Intelligence", version="0.1.0")


def summarize(record: WaferRecord) -> WaferSummary:
    features = extract_features(record.wafer_map)
    return WaferSummary(
        wafer_id=record.wafer_id,
        failure_type=record.failure_type,
        split=record.split,
        shape=list(record.wafer_map.shape),
        die_count=int(features["die_count"]),
        failing_die_count=int(features["failing_die_count"]),
        failure_rate=features["failure_rate"],
        data_source=record.data_source,
    )


@app.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        dataset_loaded=repository.mode == "wm811k",
        wafer_count=len(repository.records),
        data_mode=repository.mode,
    )


@app.get("/api/dataset/summary", response_model=DatasetSummary)
def dataset_summary() -> DatasetSummary:
    return DatasetSummary(
        wafer_count=len(repository.records),
        data_mode=repository.mode,
        class_distribution=dict(Counter(r.failure_type for r in repository.records)),
        split_distribution=dict(Counter(r.split for r in repository.records)),
    )


@app.get("/api/wafers", response_model=list[WaferSummary])
def list_wafers(
    failure_type: str | None = None,
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
) -> list[WaferSummary]:
    records = repository.records
    if failure_type:
        records = [r for r in records if r.failure_type.lower() == failure_type.lower()]
    return [summarize(record) for record in records[offset : offset + limit]]


@app.get("/api/wafers/{wafer_id}", response_model=WaferDetail)
def get_wafer(wafer_id: str) -> WaferDetail:
    record = repository.get(wafer_id)
    if not record:
        raise HTTPException(status_code=404, detail="Wafer not found")
    summary = summarize(record)
    return WaferDetail(
        **summary.model_dump(),
        wafer_map=record.wafer_map.tolist(),
        features=extract_features(record.wafer_map),
    )


@app.get("/", response_class=HTMLResponse)
def command_center() -> str:
    return Path("app/static/index.html").read_text(encoding="utf-8")


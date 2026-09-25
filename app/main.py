from collections import Counter
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse

from app.schemas import (
    DatasetSummary,
    HealthResponse,
    InvestigationResponse,
    ModelEvaluation,
    ModelStatus,
    PredictionResponse,
    SimilarWaferResponse,
    WaferDetail,
    WaferSummary,
)
from src.data.repository import WaferRecord, WaferRepository
from src.features.wafer import extract_features
from src.investigation import WaferSimilarityIndex, build_investigation
from src.models.experiments import load_experiment_summary
from src.models.registry import ModelRegistry

DATASET_PATH = Path(os.getenv("FABAI_DATASET_PATH", "data/raw/LSWMD.pkl"))
MODEL_PATH = Path(os.getenv("FABAI_MODEL_PATH", "artifacts/baseline.joblib"))
CNN_METRICS_PATH = Path(os.getenv("FABAI_CNN_METRICS_PATH", "artifacts/cnn/metrics.json"))
repository = WaferRepository(DATASET_PATH)
model_registry = ModelRegistry(MODEL_PATH)
similarity_index = WaferSimilarityIndex(repository.records)
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


@app.get("/api/models/baseline/status", response_model=ModelStatus)
def model_status() -> ModelStatus:
    return ModelStatus(
        available=model_registry.available,
        artifact_path=model_registry.artifact_path.as_posix(),
    )


@app.get("/api/models/baseline/evaluation", response_model=ModelEvaluation)
def model_evaluation() -> ModelEvaluation:
    if not model_registry.available:
        raise HTTPException(
            status_code=503,
            detail="No trained model artifact. Run python -m scripts.train on WM-811K first.",
        )
    return ModelEvaluation(**model_registry.evaluation())


@app.get("/api/models/experiments")
def model_experiments() -> dict:
    baseline_evaluation = model_registry.evaluation() if model_registry.available else None
    return load_experiment_summary(baseline_evaluation, CNN_METRICS_PATH)


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


@app.get("/api/showcase", response_model=list[WaferSummary])
def showcase_wafers(per_class: int = Query(2, ge=1, le=5)) -> list[WaferSummary]:
    """Return a compact, class-balanced queue for the command-center demo."""
    selected: list[WaferRecord] = []
    counts: Counter[str] = Counter()
    for record in repository.records:
        label = record.failure_type
        if label.lower() == "unknown" or counts[label] >= per_class:
            continue
        selected.append(record)
        counts[label] += 1
    return [summarize(record) for record in selected]


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


@app.get("/api/wafers/{wafer_id}/prediction", response_model=PredictionResponse)
def predict_wafer(wafer_id: str) -> PredictionResponse:
    record = repository.get(wafer_id)
    if not record:
        raise HTTPException(status_code=404, detail="Wafer not found")
    if not model_registry.available:
        raise HTTPException(
            status_code=503,
            detail="No trained model artifact. Run python -m scripts.train on WM-811K first.",
        )
    return PredictionResponse(wafer_id=wafer_id, **model_registry.predict(record))


@app.get("/api/wafers/{wafer_id}/similar", response_model=list[SimilarWaferResponse])
def similar_wafers(
    wafer_id: str, limit: int = Query(5, ge=1, le=20)
) -> list[SimilarWaferResponse]:
    if not repository.get(wafer_id):
        raise HTTPException(status_code=404, detail="Wafer not found")
    return [
        SimilarWaferResponse(
            wafer_id=match.record.wafer_id,
            failure_type=match.record.failure_type,
            similarity=match.similarity,
            feature_distance=match.distance,
            failure_rate=extract_features(match.record.wafer_map)["failure_rate"],
            data_source=match.record.data_source,
        )
        for match in similarity_index.search(wafer_id, limit)
    ]


@app.get("/api/wafers/{wafer_id}/investigation", response_model=InvestigationResponse)
def investigate_wafer(wafer_id: str) -> InvestigationResponse:
    record = repository.get(wafer_id)
    if not record:
        raise HTTPException(status_code=404, detail="Wafer not found")
    return InvestigationResponse(**build_investigation(record, similarity_index.search(wafer_id, 5)))


@app.get("/", response_class=HTMLResponse)
def command_center() -> str:
    return Path("app/static/index.html").read_text(encoding="utf-8")

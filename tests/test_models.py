import numpy as np

from src.data.repository import build_demo_records
from src.models.baseline import train_baseline
from src.models.registry import ModelRegistry
from src.models.split import make_split


def test_official_split_is_preferred_when_complete() -> None:
    labels = np.asarray(["a", "b", "a", "b"])
    splits = np.asarray(["Training", "Training", "Test", "Test"])
    result = make_split(labels, splits)
    assert result.strategy == "wm811k_official"
    assert result.train_indices.tolist() == [0, 1]
    assert result.test_indices.tolist() == [2, 3]


def test_baseline_artifact_can_serve_predictions(tmp_path) -> None:
    records = build_demo_records(count=50)
    metrics = train_baseline(records, tmp_path)
    registry = ModelRegistry(tmp_path / "baseline.joblib")
    prediction = registry.predict(records[0])
    assert metrics["split_strategy"] == "stratified_random_fallback"
    assert prediction["predicted_failure_type"]
    assert 0 <= prediction["confidence"] <= 1
    assert abs(sum(prediction["class_probabilities"].values()) - 1) < 1e-9

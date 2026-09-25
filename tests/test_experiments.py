from pathlib import Path

from src.models.experiments import load_experiment_summary


def test_missing_cnn_metrics_are_explicit() -> None:
    result = load_experiment_summary(
        baseline_evaluation=None,
        cnn_metrics_path=Path("artifacts/test-fixtures/not-present.json"),
    )

    assert result["baseline"] is None
    assert result["cnn"]["status"] == "not_run"
    assert result["comparison"] is None


def test_cnn_comparison_is_metric_delta(tmp_path: Path) -> None:
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(
        '{"model_type":"WaferCNN","accuracy":0.80,"macro_f1":0.50,'
        '"balanced_accuracy":0.55,"test_samples":100,"split_strategy":"official"}',
        encoding="utf-8",
    )

    result = load_experiment_summary(
        baseline_evaluation={
            "model_type": "RandomForest",
            "accuracy": 0.78,
            "macro_f1": 0.45,
            "balanced_accuracy": 0.50,
            "test_samples": 100,
            "split_strategy": "official",
        },
        cnn_metrics_path=metrics_path,
    )

    assert result["cnn"]["status"] == "available"
    assert result["comparison"]["macro_f1_delta"] == 0.05
    assert result["comparison"]["balanced_accuracy_delta"] == 0.05

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def load_experiment_summary(
    baseline_evaluation: dict[str, Any] | None,
    cnn_metrics_path: Path = Path("artifacts/cnn/metrics.json"),
) -> dict[str, Any]:
    """Return evidence-backed model results without inventing missing experiments."""
    cnn = _read_json(cnn_metrics_path)

    baseline = None
    if baseline_evaluation is not None:
        baseline = {
            "model_type": baseline_evaluation.get("model_type", "RandomForest"),
            "status": "available",
            "accuracy": baseline_evaluation.get("accuracy"),
            "macro_f1": baseline_evaluation.get("macro_f1"),
            "balanced_accuracy": baseline_evaluation.get("balanced_accuracy"),
            "test_samples": baseline_evaluation.get("test_samples"),
            "split_strategy": baseline_evaluation.get("split_strategy"),
        }

    cnn_summary = None
    if cnn is not None:
        cnn_summary = {
            "model_type": cnn.get("model_type", "WaferCNN"),
            "status": "available",
            "accuracy": cnn.get("accuracy"),
            "macro_f1": cnn.get("macro_f1"),
            "balanced_accuracy": cnn.get("balanced_accuracy"),
            "test_samples": cnn.get("test_samples"),
            "split_strategy": cnn.get("split_strategy"),
            "device": cnn.get("device"),
            "history": cnn.get("history", []),
            "classification_report": cnn.get("classification_report", {}),
        }
    else:
        cnn_summary = {
            "model_type": "WaferCNN",
            "status": "not_run",
            "message": "Run python -m scripts.train_cnn to populate artifacts/cnn/metrics.json.",
        }

    comparison = None
    if baseline and cnn_summary["status"] == "available":
        comparison = {
            "macro_f1_delta": cnn_summary["macro_f1"] - baseline["macro_f1"],
            "balanced_accuracy_delta": cnn_summary["balanced_accuracy"] - baseline["balanced_accuracy"],
            "accuracy_delta": cnn_summary["accuracy"] - baseline["accuracy"],
            "metric_basis": "CNN minus Random Forest on the same held-out test split",
        }

    return {
        "baseline": baseline,
        "cnn": cnn_summary,
        "comparison": comparison,
    }

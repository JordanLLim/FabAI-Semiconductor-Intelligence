from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np

from src.data.repository import WaferRecord
from src.features.wafer import extract_features


class ModelRegistry:
    """Lazy model loader that keeps training artifacts outside the source tree."""

    def __init__(self, artifact_path: Path):
        self.artifact_path = artifact_path
        self._bundle: dict | None = None

    @property
    def available(self) -> bool:
        return self.artifact_path.exists()

    def _load(self) -> dict:
        if self._bundle is None:
            self._bundle = joblib.load(self.artifact_path)
        return self._bundle

    def evaluation(self) -> dict:
        bundle = self._load()
        metadata = bundle.get("training_metadata", {})
        report = metadata.get("classification_report", {})
        labels = metadata.get("labels", list(bundle["model"].classes_))
        per_class = {
            str(label): {
                "precision": float(report.get(label, {}).get("precision", 0.0)),
                "recall": float(report.get(label, {}).get("recall", 0.0)),
                "f1": float(report.get(label, {}).get("f1-score", 0.0)),
                "support": int(report.get(label, {}).get("support", 0)),
            }
            for label in labels
        }
        return {
            "model_type": bundle.get("model_type", type(bundle["model"]).__name__),
            "split_strategy": metadata.get("split_strategy", "unknown"),
            "train_samples": int(metadata.get("train_samples", 0)),
            "test_samples": int(metadata.get("test_samples", 0)),
            "accuracy": float(metadata.get("accuracy", 0.0)),
            "macro_f1": float(metadata.get("macro_f1", 0.0)),
            "balanced_accuracy": float(metadata.get("balanced_accuracy", 0.0)),
            "weighted_f1": float(metadata.get("weighted_f1", 0.0)),
            "labels": [str(label) for label in labels],
            "per_class": per_class,
            "confusion_matrix": metadata.get("confusion_matrix", []),
        }

    def predict(self, record: WaferRecord) -> dict:
        bundle = self._load()
        feature_values = extract_features(record.wafer_map)
        vector = np.asarray([[feature_values[name] for name in bundle["feature_names"]]])
        model = bundle["model"]
        probabilities = model.predict_proba(vector)[0]
        ranked = sorted(
            zip(model.classes_, probabilities, strict=True), key=lambda item: item[1], reverse=True
        )
        predicted = str(ranked[0][0])
        return {
            "ground_truth": record.failure_type,
            "predicted_failure_type": predicted,
            "is_correct": predicted.lower() == record.failure_type.lower(),
            "confidence": float(ranked[0][1]),
            "class_probabilities": {str(label): float(score) for label, score in ranked},
            "model_type": bundle.get("model_type", type(model).__name__),
            "training_metadata": bundle.get("training_metadata", {}),
        }


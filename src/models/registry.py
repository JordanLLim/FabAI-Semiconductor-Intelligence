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

    def predict(self, record: WaferRecord) -> dict:
        bundle = self._load()
        feature_values = extract_features(record.wafer_map)
        vector = np.asarray([[feature_values[name] for name in bundle["feature_names"]]])
        model = bundle["model"]
        probabilities = model.predict_proba(vector)[0]
        ranked = sorted(
            zip(model.classes_, probabilities, strict=True), key=lambda item: item[1], reverse=True
        )
        return {
            "predicted_failure_type": str(ranked[0][0]),
            "confidence": float(ranked[0][1]),
            "class_probabilities": {str(label): float(score) for label, score in ranked},
            "model_type": bundle.get("model_type", type(model).__name__),
            "training_metadata": bundle.get("training_metadata", {}),
        }


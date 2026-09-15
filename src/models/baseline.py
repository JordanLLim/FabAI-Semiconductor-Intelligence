from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix

from src.data.repository import WaferRecord
from src.features.wafer import extract_features
from src.models.split import make_split


def feature_matrix(records: list[WaferRecord]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    rows = [extract_features(record.wafer_map) for record in records]
    columns = sorted(rows[0])
    features = np.asarray([[row[column] for column in columns] for row in rows])
    labels = np.asarray([record.failure_type for record in records])
    return features, labels, columns


def train_baseline(records: list[WaferRecord], output_dir: Path) -> dict:
    labelled = [record for record in records if record.failure_type.lower() not in {"", "unknown"}]
    if len(labelled) < 20:
        raise ValueError("At least 20 labelled wafers are required for a meaningful split.")
    features, labels, columns = feature_matrix(labelled)
    counts = dict(zip(*np.unique(labels, return_counts=True), strict=True))
    rare = {label for label, count in counts.items() if count < 2}
    keep = np.asarray([label not in rare for label in labels])
    features, labels = features[keep], labels[keep]
    if len(np.unique(labels)) < 2:
        raise ValueError("At least two classes with two samples each are required.")

    split_labels = np.asarray([record.split for record, selected in zip(labelled, keep, strict=True) if selected])
    split = make_split(labels, split_labels)
    x_train, x_test = features[split.train_indices], features[split.test_indices]
    y_train, y_test = labels[split.train_indices], labels[split.test_indices]
    model = RandomForestClassifier(
        n_estimators=300, class_weight="balanced_subsample", random_state=42, n_jobs=-1
    )
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    report = classification_report(y_test, predictions, output_dict=True, zero_division=0)
    result = {
        "train_samples": len(y_train),
        "test_samples": len(y_test),
        "split_strategy": split.strategy,
        "feature_names": columns,
        "macro_f1": report["macro avg"]["f1-score"],
        "balanced_accuracy": balanced_accuracy_score(y_test, predictions),
        "classification_report": report,
        "labels": model.classes_.tolist(),
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=model.classes_).tolist(),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "model_type": "RandomForestClassifier",
            "feature_names": columns,
            "training_metadata": {
                "split_strategy": split.strategy,
                "train_samples": len(y_train),
                "test_samples": len(y_test),
                "macro_f1": result["macro_f1"],
                "balanced_accuracy": result["balanced_accuracy"],
            },
        },
        output_dir / "baseline.joblib",
    )
    return result

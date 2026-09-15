from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from src.data.repository import WaferRecord
from src.features.wafer import extract_features


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

    x_train, x_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=labels
    )
    model = RandomForestClassifier(
        n_estimators=300, class_weight="balanced_subsample", random_state=42, n_jobs=-1
    )
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    report = classification_report(y_test, predictions, output_dict=True, zero_division=0)
    result = {
        "train_samples": len(y_train),
        "test_samples": len(y_test),
        "feature_names": columns,
        "macro_f1": report["macro avg"]["f1-score"],
        "balanced_accuracy": balanced_accuracy_score(y_test, predictions),
        "classification_report": report,
        "labels": model.classes_.tolist(),
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=model.classes_).tolist(),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "feature_names": columns}, output_dir / "baseline.joblib")
    return result


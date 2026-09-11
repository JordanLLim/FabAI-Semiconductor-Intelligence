import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = [
    "active_dies",
    "failed_dies",
    "failure_rate",
    "edge_failure_rate",
    "center_failure_rate",
    "x_centroid",
    "y_centroid",
]


def build_model():
    return Pipeline(
        [
            ("impute", SimpleImputer()),
            ("scale", StandardScaler()),
            (
                "clf",
                HistGradientBoostingClassifier(
                    max_iter=150,
                    learning_rate=0.08,
                    random_state=42,
                ),
            ),
        ]
    )


def matrix(rows):
    return np.asarray([[row[feature] for feature in FEATURES] for row in rows], dtype=float)

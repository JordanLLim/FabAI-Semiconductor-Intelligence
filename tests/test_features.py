import numpy as np

from src.features.wafer import extract_features


def test_feature_extraction_counts_tested_and_failed_dies() -> None:
    wafer = np.asarray([[0, 1, 0], [1, 2, 1], [0, 1, 0]], dtype=np.uint8)
    features = extract_features(wafer)
    assert features["die_count"] == 5
    assert features["failing_die_count"] == 1
    assert features["failure_rate"] == 0.2


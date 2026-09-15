from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class DatasetSplit:
    train_indices: np.ndarray
    test_indices: np.ndarray
    strategy: str


def make_split(labels: np.ndarray, split_labels: np.ndarray, test_size: float = 0.2) -> DatasetSplit:
    """Use WM-811K's published split when viable, otherwise use a deterministic fallback."""
    normalized = np.char.lower(split_labels.astype(str))
    train_mask = np.isin(normalized, ["training", "train"])
    test_mask = np.isin(normalized, ["test", "testing"])

    train_classes = set(labels[train_mask])
    test_classes = set(labels[test_mask])
    all_classes = set(labels)
    if train_mask.sum() and test_mask.sum() and all_classes <= train_classes & test_classes:
        return DatasetSplit(np.flatnonzero(train_mask), np.flatnonzero(test_mask), "wm811k_official")

    indices = np.arange(len(labels))
    train_indices, test_indices = train_test_split(
        indices, test_size=test_size, random_state=42, stratify=labels
    )
    return DatasetSplit(train_indices, test_indices, "stratified_random_fallback")


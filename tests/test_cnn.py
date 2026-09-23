import numpy as np
import torch

from src.models.cnn import WaferCNN
from scripts.train_cnn import normalize_map


def test_normalize_map_preserves_discrete_wafer_labels() -> None:
    wafer = np.asarray([[0, 1], [2, 2]], dtype=np.uint8)
    normalized = normalize_map(wafer, size=4)
    assert normalized.shape == (4, 4)
    assert set(np.unique(normalized)) <= {0.0, 0.5, 1.0}


def test_cnn_forward_shape() -> None:
    model = WaferCNN(num_classes=9)
    output = model(torch.zeros(2, 1, 32, 32))
    assert output.shape == (2, 9)

import numpy as np


def normalize_wafer(arr, size=32):
    """Nearest-neighbour resize without adding heavyweight CV dependencies."""
    x = np.asarray(arr, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError("wafer map must be 2D")
    yi = np.linspace(0, x.shape[0] - 1, size).round().astype(int)
    xi = np.linspace(0, x.shape[1] - 1, size).round().astype(int)
    return x[np.ix_(yi, xi)]


def engineer_features(arr):
    x = np.asarray(arr)
    active = x > 0
    fail = x == 2
    n = max(int(active.sum()), 1)
    h, w = x.shape
    yy, xx = np.indices(x.shape)
    cy, cx = (h - 1) / 2, (w - 1) / 2
    radius = np.sqrt(((yy - cy) / (h / 2 + 1e-6)) ** 2 + ((xx - cx) / (w / 2 + 1e-6)) ** 2)
    return {
        "active_dies": int(active.sum()),
        "failed_dies": int(fail.sum()),
        "failure_rate": float(fail.sum() / n),
        "edge_failure_rate": float((fail & (radius > 0.7)).sum() / max(int((active & (radius > 0.7)).sum()), 1)),
        "center_failure_rate": float((fail & (radius < 0.35)).sum() / max(int((active & (radius < 0.35)).sum()), 1)),
        "x_centroid": float(xx[fail].mean() / max(w - 1, 1)) if fail.any() else 0.5,
        "y_centroid": float(yy[fail].mean() / max(h - 1, 1)) if fail.any() else 0.5,
    }

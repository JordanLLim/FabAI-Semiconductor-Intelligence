from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

LABELS = ["Center", "Donut", "Edge-Loc", "Edge-Ring", "Loc", "Random", "Scratch", "Near-full", "none"]


def _unwrap(value):
    while isinstance(value, (list, tuple, np.ndarray)) and np.asarray(value, dtype=object).size == 1:
        value = np.asarray(value, dtype=object).reshape(-1)[0]
    return str(value) if value is not None else ""


def normalize_label(value):
    label = _unwrap(value).strip()
    return label if label in LABELS else ("none" if label.lower() == "none" else label)


def load_wm811k(path: str | Path) -> pd.DataFrame:
    df = pd.read_pickle(path)
    required = {"waferMap", "lotName", "waferIndex", "failureType"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing WM-811K columns: {sorted(missing)}")

    out = df.copy()
    out["failureType_norm"] = out["failureType"].map(normalize_label)
    out["map_h"] = out["waferMap"].map(lambda x: int(np.asarray(x).shape[0]))
    out["map_w"] = out["waferMap"].map(lambda x: int(np.asarray(x).shape[1]))
    out["defect_dies"] = out["waferMap"].map(lambda x: int((np.asarray(x) == 2).sum()))
    out["valid_dies"] = out["waferMap"].map(lambda x: int((np.asarray(x) > 0).sum()))
    out["defect_rate"] = out["defect_dies"] / out["valid_dies"].clip(lower=1)
    return out


def wafer_payload(row):
    arr = np.asarray(row["waferMap"], dtype=np.uint8)
    ys, xs = np.where(arr > 0)
    vals = arr[ys, xs]
    return {
        "lot": str(row["lotName"]),
        "wafer_index": int(row["waferIndex"]),
        "label": normalize_label(row["failureType"]),
        "h": int(arr.shape[0]),
        "w": int(arr.shape[1]),
        "cells": [[int(x), int(y), int(v)] for x, y, v in zip(xs, ys, vals)],
        "defect_rate": float((arr == 2).sum() / max(1, (arr > 0).sum())),
    }

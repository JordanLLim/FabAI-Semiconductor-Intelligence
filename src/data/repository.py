from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class WaferRecord:
    wafer_id: str
    wafer_map: np.ndarray
    failure_type: str
    split: str
    data_source: str


def _unwrap(value: Any, default: str = "unknown") -> str:
    """Normalize WM-811K scalar values, which are often nested arrays."""
    if value is None:
        return default
    array = np.asarray(value, dtype=object).reshape(-1)
    if not len(array):
        return default
    text = str(array[0]).strip()
    return text if text and text.lower() != "nan" else default


def load_wm811k(path: Path) -> list[WaferRecord]:
    frame = pd.read_pickle(path)
    required = {"waferMap", "failureType", "trianTestLabel"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"WM-811K file is missing columns: {sorted(missing)}")

    records: list[WaferRecord] = []
    for position, (_, row) in enumerate(frame.iterrows()):
        wafer_map = np.asarray(row["waferMap"], dtype=np.uint8)
        if wafer_map.ndim != 2 or wafer_map.size == 0:
            continue
        records.append(
            WaferRecord(
                wafer_id=f"wm-{position}",
                wafer_map=wafer_map,
                failure_type=_unwrap(row["failureType"]),
                split=_unwrap(row["trianTestLabel"], "unassigned").lower(),
                data_source="WM-811K",
            )
        )
    return records


def build_demo_records(count: int = 24, seed: int = 17) -> list[WaferRecord]:
    """Create clearly labelled simulated maps so the UI can run without the dataset."""
    rng = np.random.default_rng(seed)
    labels = ["Center", "Donut", "Edge-Loc", "Scratch", "none"]
    records: list[WaferRecord] = []
    y, x = np.ogrid[-1:1:32j, -1:1:32j]
    wafer_mask = x * x + y * y <= 0.95

    for index in range(count):
        wafer_map = np.zeros((32, 32), dtype=np.uint8)
        wafer_map[wafer_mask] = 1
        label = labels[index % len(labels)]
        if label == "Center":
            defects = x * x + y * y < 0.12
        elif label == "Donut":
            radius = x * x + y * y
            defects = (radius > 0.28) & (radius < 0.42)
        elif label == "Edge-Loc":
            defects = (x > 0.55) & wafer_mask
        elif label == "Scratch":
            defects = (np.abs(y - 0.45 * x) < 0.07) & wafer_mask
        else:
            defects = rng.random((32, 32)) < 0.015
        wafer_map[defects] = 2
        records.append(
            WaferRecord(
                wafer_id=f"demo-{index:03d}",
                wafer_map=wafer_map,
                failure_type=label,
                split="demo",
                data_source="simulated_demo",
            )
        )
    return records


class WaferRepository:
    def __init__(self, dataset_path: Path):
        self.dataset_path = dataset_path
        self.mode = "wm811k" if dataset_path.exists() else "demo"
        self.records = load_wm811k(dataset_path) if dataset_path.exists() else build_demo_records()
        self._by_id = {record.wafer_id: record for record in self.records}

    def get(self, wafer_id: str) -> WaferRecord | None:
        return self._by_id.get(wafer_id)


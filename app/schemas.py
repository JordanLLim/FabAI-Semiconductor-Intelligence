from typing import Literal

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: Literal["ok"]
    dataset_loaded: bool
    wafer_count: int
    data_mode: Literal["wm811k", "demo"]


class WaferSummary(BaseModel):
    wafer_id: str
    failure_type: str
    split: str
    shape: list[int]
    die_count: int
    failing_die_count: int
    failure_rate: float = Field(ge=0, le=1)
    data_source: Literal["WM-811K", "simulated_demo"]


class WaferDetail(WaferSummary):
    wafer_map: list[list[int]]
    features: dict[str, float]


class DatasetSummary(BaseModel):
    wafer_count: int
    data_mode: Literal["wm811k", "demo"]
    class_distribution: dict[str, int]
    split_distribution: dict[str, int]


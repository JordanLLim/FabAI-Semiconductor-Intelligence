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


class ModelStatus(BaseModel):
    available: bool
    artifact_path: str


class ClassEvaluation(BaseModel):
    precision: float
    recall: float
    f1: float
    support: int


class ModelEvaluation(BaseModel):
    model_type: str
    split_strategy: str
    train_samples: int
    test_samples: int
    accuracy: float = Field(ge=0, le=1)
    macro_f1: float = Field(ge=0, le=1)
    balanced_accuracy: float = Field(ge=0, le=1)
    weighted_f1: float = Field(ge=0, le=1)
    labels: list[str]
    per_class: dict[str, ClassEvaluation]
    confusion_matrix: list[list[int]]


class PredictionResponse(BaseModel):
    wafer_id: str
    ground_truth: str
    predicted_failure_type: str
    is_correct: bool
    confidence: float = Field(ge=0, le=1)
    class_probabilities: dict[str, float]
    model_type: str
    training_metadata: dict


class SimilarWaferResponse(BaseModel):
    wafer_id: str
    failure_type: str
    similarity: float = Field(ge=0, le=1)
    feature_distance: float = Field(ge=0)
    failure_rate: float = Field(ge=0, le=1)
    data_source: Literal["WM-811K", "simulated_demo"]


class InvestigationResponse(BaseModel):
    wafer_id: str
    pattern: str
    severity: Literal["review", "high"]
    observation: str
    recommended_checks: list[str]
    evidence: dict[str, float | int]
    similar_case_ids: list[str]
    disclaimer: str

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.data.repository import WaferRecord
from src.features.wafer import extract_features


@dataclass(frozen=True)
class SimilarWafer:
    record: WaferRecord
    distance: float
    similarity: float


class WaferSimilarityIndex:
    """Small in-memory feature index for explainable case retrieval.

    This intentionally indexes engineered inspection features rather than claiming
    image-embedding or fab-process similarity that the project has not measured.
    """

    def __init__(self, records: list[WaferRecord], max_index_records: int = 20_000):
        self._all_by_id = {record.wafer_id: record for record in records}
        if len(records) > max_index_records:
            positions = np.linspace(0, len(records) - 1, max_index_records, dtype=int)
            self.records = [records[int(position)] for position in positions]
        else:
            self.records = records
        rows = [extract_features(record.wafer_map) for record in self.records]
        self.feature_names = sorted(rows[0]) if rows else []
        matrix = np.asarray(
            [[row[name] for name in self.feature_names] for row in rows], dtype=float
        )
        if matrix.size:
            self.mean = matrix.mean(axis=0)
            self.scale = matrix.std(axis=0)
            self.scale[self.scale == 0] = 1.0
            self.matrix = (matrix - self.mean) / self.scale
        else:
            self.mean = np.empty(0)
            self.scale = np.empty(0)
            self.matrix = matrix
    def search(self, wafer_id: str, limit: int = 5) -> list[SimilarWafer]:
        query_record = self._all_by_id.get(wafer_id)
        if query_record is None or not self.feature_names:
            return []
        query_features = extract_features(query_record.wafer_map)
        query = np.asarray([query_features[name] for name in self.feature_names], dtype=float)
        query = (query - self.mean) / self.scale
        distances = np.linalg.norm(self.matrix - query, axis=1)
        order = np.argsort(distances)
        matches: list[SimilarWafer] = []
        for candidate in order:
            if self.records[int(candidate)].wafer_id == wafer_id:
                continue
            distance = float(distances[candidate])
            matches.append(
                SimilarWafer(
                    record=self.records[int(candidate)],
                    distance=distance,
                    similarity=1.0 / (1.0 + distance),
                )
            )
            if len(matches) == limit:
                break
        return matches


PATTERN_GUIDANCE = {
    "center": (
        "Defects are concentrated near the wafer centre.",
        ["Review centre-zone process uniformity", "Compare the same recipe and chamber cohort"],
    ),
    "donut": (
        "Defects form a radial ring pattern.",
        ["Check radial process-uniformity signals", "Compare recent ring-pattern cases"],
    ),
    "edge-loc": (
        "Defects are concentrated near one edge region.",
        ["Inspect edge handling and exclusion settings", "Compare orientation across nearby lots"],
    ),
    "edge-ring": (
        "Defects are concentrated around the wafer perimeter.",
        ["Review edge process conditions", "Check whether the pattern persists by chamber"],
    ),
    "scratch": (
        "Defects follow an elongated path consistent with a scratch-like map pattern.",
        ["Review wafer-handling checkpoints", "Inspect orientation consistency across cases"],
    ),
    "loc": (
        "Defects are locally clustered.",
        ["Inspect the affected spatial region", "Compare local clusters in the same cohort"],
    ),
    "random": (
        "Defects are spatially dispersed without a dominant geometric pattern.",
        ["Check for broad process instability", "Compare failure rate against the recent baseline"],
    ),
    "near-full": (
        "A large portion of active dies are failing.",
        ["Escalate for high-severity review", "Validate map ingestion and process excursion context"],
    ),
    "none": (
        "No labelled systematic defect pattern is present.",
        ["Retain as a healthy comparison case", "Monitor for distribution drift"],
    ),
}


def build_investigation(record: WaferRecord, similar: list[SimilarWafer]) -> dict:
    features = extract_features(record.wafer_map)
    pattern = record.failure_type.lower()
    observation, checks = PATTERN_GUIDANCE.get(
        pattern,
        (
            "The map has no supported pattern-specific investigation template.",
            ["Review the wafer map manually", "Compare the closest feature-based cases"],
        ),
    )
    return {
        "wafer_id": record.wafer_id,
        "pattern": record.failure_type,
        "severity": "high" if features["failure_rate"] >= 0.25 else "review",
        "observation": observation,
        "recommended_checks": checks,
        "evidence": {
            "failure_rate": features["failure_rate"],
            "failing_die_count": int(features["failing_die_count"]),
            "similar_case_count": len(similar),
        },
        "similar_case_ids": [match.record.wafer_id for match in similar],
        "disclaimer": (
            "Decision-support hypothesis only. WM-811K contains wafer maps and labels, "
            "not equipment telemetry or confirmed physical root causes."
        ),
    }

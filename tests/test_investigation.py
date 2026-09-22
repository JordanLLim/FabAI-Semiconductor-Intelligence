from src.data.repository import build_demo_records
from src.investigation import WaferSimilarityIndex, build_investigation


def test_similarity_excludes_query_and_is_ordered() -> None:
    records = build_demo_records()
    index = WaferSimilarityIndex(records)
    matches = index.search(records[0].wafer_id, limit=4)
    assert len(matches) == 4
    assert all(match.record.wafer_id != records[0].wafer_id for match in matches)
    assert [match.distance for match in matches] == sorted(match.distance for match in matches)


def test_investigation_is_evidence_bounded() -> None:
    records = build_demo_records()
    index = WaferSimilarityIndex(records)
    result = build_investigation(records[0], index.search(records[0].wafer_id, limit=3))
    assert result["pattern"] == "Center"
    assert result["evidence"]["similar_case_count"] == 3
    assert "not equipment telemetry" in result["disclaimer"]

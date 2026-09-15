from __future__ import annotations

import numpy as np


def extract_features(wafer_map: np.ndarray) -> dict[str, float]:
    tested = wafer_map > 0
    failures = wafer_map == 2
    die_count = int(tested.sum())
    fail_count = int(failures.sum())
    rows, columns = np.nonzero(failures)

    if fail_count:
        row_center = float(rows.mean() / max(wafer_map.shape[0] - 1, 1))
        column_center = float(columns.mean() / max(wafer_map.shape[1] - 1, 1))
        row_spread = float(rows.std() / max(wafer_map.shape[0] - 1, 1))
        column_spread = float(columns.std() / max(wafer_map.shape[1] - 1, 1))
    else:
        row_center = column_center = row_spread = column_spread = 0.0

    border = np.zeros_like(failures)
    border[[0, -1], :] = True
    border[:, [0, -1]] = True
    edge_failures = int((failures & border).sum())

    return {
        "die_count": float(die_count),
        "failing_die_count": float(fail_count),
        "failure_rate": fail_count / die_count if die_count else 0.0,
        "failure_row_center": row_center,
        "failure_column_center": column_center,
        "failure_row_spread": row_spread,
        "failure_column_spread": column_spread,
        "edge_failure_ratio": edge_failures / fail_count if fail_count else 0.0,
        "aspect_ratio": wafer_map.shape[1] / wafer_map.shape[0],
    }


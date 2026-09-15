import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from src.data.repository import load_wm811k


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile a local WM-811K pickle file.")
    parser.add_argument("--data", type=Path, default=Path("data/raw/LSWMD.pkl"))
    args = parser.parse_args()
    records = load_wm811k(args.data)
    summary = {
        "wafer_count": len(records),
        "class_distribution": Counter(record.failure_type for record in records),
        "split_distribution": Counter(record.split for record in records),
        "shape_distribution": Counter(str(tuple(record.wafer_map.shape)) for record in records),
        "total_tested_dies": int(sum(np.count_nonzero(r.wafer_map) for r in records)),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()


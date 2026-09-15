import argparse
import json
from pathlib import Path

from src.data.repository import load_wm811k
from src.models.baseline import train_baseline


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the engineered-feature baseline.")
    parser.add_argument("--data", type=Path, default=Path("data/raw/LSWMD.pkl"))
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    result = train_baseline(load_wm811k(args.data), args.output)
    metrics_path = args.output / "metrics.json"
    metrics_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"macro_f1": result["macro_f1"], "balanced_accuracy": result["balanced_accuracy"]}, indent=2))


if __name__ == "__main__":
    main()


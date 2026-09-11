from pathlib import Path
import json
from src.wm811k import load_wm811k

ROOT = Path(__file__).resolve().parents[1]
df = load_wm811k(ROOT / "data/raw/LSWMD.pkl")
report = {
    "rows": len(df),
    "columns": list(df.columns),
    "memory_mb": round(df.memory_usage(deep=True).sum() / 1024**2, 2),
    "class_counts": df["failureType_norm"].value_counts().to_dict(),
}
(ROOT / "artifacts").mkdir(exist_ok=True)
(ROOT / "artifacts/data_profile.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))

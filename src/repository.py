from __future__ import annotations
from pathlib import Path
from .wm811k import load_wm811k, wafer_payload


class WaferRepository:
    def __init__(self, data_path: str):
        self.path = Path(data_path)
        self.df = load_wm811k(self.path) if self.path.exists() else None

    @property
    def ready(self):
        return self.df is not None

    def summary(self):
        if not self.ready:
            return {"ready": False, "message": "Place LSWMD.pkl at data/raw/LSWMD.pkl"}
        counts = self.df["failureType_norm"].value_counts().to_dict()
        return {
            "ready": True,
            "wafers": len(self.df),
            "lots": int(self.df["lotName"].nunique()),
            "classes": counts,
        }

    def incident(self, label="Edge-Ring"):
        if not self.ready:
            return None
        q = self.df[self.df.failureType_norm == label]
        if q.empty:
            q = self.df[self.df.failureType_norm != "none"]
        return wafer_payload(q.iloc[0])

    def recent(self, n=12):
        if not self.ready:
            return []
        data = self.df[self.df.failureType_norm != "none"].head(n)
        return [
            {
                "lot": str(row.lotName),
                "wafer_index": int(row.waferIndex),
                "label": row.failureType_norm,
                "defect_rate": float(row.defect_rate),
            }
            for _, row in data.iterrows()
        ]

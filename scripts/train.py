"""Train an interpretable WM-811K baseline after data/raw/LSWMD.pkl is present."""
import json
from pathlib import Path
import joblib
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from src.ml.features import engineer_features
from src.ml.model import build_model, matrix
from src.wm811k import load_wm811k

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts"
OUT.mkdir(exist_ok=True)

df = load_wm811k(ROOT / "data/raw/LSWMD.pkl")
rows, labels = [], []
for _, row in df.iterrows():
    label = row["failureType_norm"]
    if label not in {"Center", "Donut", "Edge-Loc", "Edge-Ring", "Loc", "Random", "Scratch", "Near-full"}:
        continue
    rows.append(engineer_features(row["waferMap"]))
    labels.append(label)

X = matrix(rows)
y = labels
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
model = build_model()
model.fit(X_train, y_train)
pred = model.predict(X_test)
metrics = {
    "macro_f1": f1_score(y_test, pred, average="macro"),
    "balanced_accuracy": balanced_accuracy_score(y_test, pred),
    "classification_report": classification_report(y_test, pred, output_dict=True),
    "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
}
joblib.dump(model, OUT / "wm811k_baseline.joblib")
(OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))
print(json.dumps({k: metrics[k] for k in ("macro_f1", "balanced_accuracy")}, indent=2))

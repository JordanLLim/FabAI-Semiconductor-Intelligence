# FAB.AI — Semiconductor Intelligence

AI-powered semiconductor wafer defect intelligence and fab investigation platform built around the public WM-811K wafer-map dataset.

The project separates **real wafer evidence** from **simulated operational storytelling**. WM-811K is used for wafer maps and defect labels; live equipment telemetry, maintenance histories, and intervention outcomes are not claimed as measured WM-811K data.

## Implemented now

- Validated WM-811K pickle loader and repository layer
- FastAPI REST backend with health, dataset, list and detail endpoints
- Interactive wafer-map command center that runs in labelled demo mode when data is absent
- Nine-feature Random Forest baseline with class balancing
- Reproducible train/test evaluation pipeline and persisted artifacts
- Macro-F1, balanced accuracy, per-class report and confusion matrix
- Dataset profiling CLI
- Docker / Docker Compose and GitHub Actions CI
- API and feature extraction tests
- Explainable similar-wafer retrieval over standardized engineered features
- Evidence-bounded investigation reports with explicit root-cause limitations

## Planned path

`WM-811K -> validation -> baseline -> similarity retrieval -> investigation workflow -> deployment`

CNN experiments and a richer React/Three.js interface are possible extensions, not current claims.

## Data setup

Download the WM-811K dataset separately and place:

```text
data/raw/LSWMD.pkl
```

Raw data and trained artifacts are intentionally not committed.

## Run

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Then open `http://localhost:8000`. Without the dataset, the application intentionally
starts in `demo` mode using visibly labelled simulated wafer maps. API documentation is
available at `http://localhost:8000/docs`.

## API

| Endpoint | Purpose |
| --- | --- |
| `GET /api/health` | Service health, dataset availability and provenance mode |
| `GET /api/dataset/summary` | Class and split distributions |
| `GET /api/wafers` | Paginated wafer summaries with optional class filter |
| `GET /api/wafers/{wafer_id}` | Wafer map and engineered inspection signals |
| `GET /api/models/baseline/status` | Whether a trained artifact is available |
| `GET /api/wafers/{wafer_id}/prediction` | Predicted class, confidence and probabilities |
| `GET /api/wafers/{wafer_id}/similar` | Closest explainable feature-space cases |
| `GET /api/wafers/{wafer_id}/investigation` | Pattern observation, evidence and recommended checks |

## Train baseline

```bash
python scripts/profile_dataset.py
python scripts/train.py
```

Model metrics are written to `artifacts/metrics.json` and the fitted pipeline to
`artifacts/baseline.joblib`. Metrics are generated from the actual local dataset and are
never hard-coded.

The evaluator prefers WM-811K's provided training/test labels when both partitions cover
all retained classes. If those labels are incomplete, it records use of a deterministic
stratified fallback in both `metrics.json` and the serialized model metadata.

## Test

```bash
pip install -r requirements-dev.txt
ruff check .
pytest -q
```

## Evaluation principles

WM-811K is class-imbalanced, so accuracy alone is not enough. This project reports macro-F1, balanced accuracy, per-class precision/recall/F1 and a confusion matrix. Data splitting happens before any future augmentation/resampling to reduce leakage risk.

Similarity is computed from standardized engineered wafer-map features and is presented as
case retrieval, not proof that two wafers share a physical root cause. Investigation output is
deterministic decision support grounded in the observed map and retrieved cases; it does not
invent equipment telemetry, maintenance history or intervention outcomes absent from WM-811K.

## Disclaimer

Independent portfolio / hackathon project. Not affiliated with Micron, Samsung, SK hynix, TSMC, UCI, or the original WM-811K data owner.

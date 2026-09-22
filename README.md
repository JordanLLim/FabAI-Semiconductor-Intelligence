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

## Real-data baseline results

The current baseline was trained and evaluated locally on the real WM-811K file using its
official split: 54,355 training wafers and 118,595 test wafers. Unlabelled records were excluded
from supervised training and evaluation.

| Metric | Result |
| --- | ---: |
| Accuracy | 0.8594 |
| Macro-F1 | 0.4128 |
| Balanced accuracy | 0.4391 |

Accuracy is not treated as the headline metric: 110,701 of the 118,595 test wafers are labelled
`none`. Macro-F1 and per-class recall expose the baseline's weak performance on rare spatial
patterns. See [`docs/MODEL_RESULTS.md`](docs/MODEL_RESULTS.md) for the class-level results,
confusion-matrix analysis and the next modelling hypothesis.

## Data setup

Download the WM-811K dataset separately and place:

```text
data/raw/LSWMD.pkl
```

Raw data and trained artifacts are intentionally not committed.

## Run

```bash
pip install -r requirements.txt
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
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
| `GET /api/showcase` | Compact class-balanced inspection queue |
| `GET /api/wafers/{wafer_id}` | Wafer map and engineered inspection signals |
| `GET /api/models/baseline/status` | Whether a trained artifact is available |
| `GET /api/wafers/{wafer_id}/prediction` | Predicted class, confidence and probabilities |
| `GET /api/wafers/{wafer_id}/similar` | Closest explainable feature-space cases |
| `GET /api/wafers/{wafer_id}/investigation` | Pattern observation, evidence and recommended checks |

## Train baseline

```bash
python -m scripts.profile_dataset
python -m scripts.train
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

## Current boundary and next experiment

This version proves the end-to-end system contract: validated ingestion, leakage-aware evaluation,
persisted model metadata, API inference, explainable case retrieval and a human-facing review
workflow. The nine-feature Random Forest is deliberately retained as an interpretable baseline,
not presented as the final classifier. The next experiment is a spatial model/CNN comparison on
the same official split, with class weighting and per-class error analysis. Process drift, SPC and
root-cause attribution require separate process or equipment data and are not inferred from wafer
maps alone.

## Disclaimer

Independent portfolio / hackathon project. Not affiliated with Micron, Samsung, SK hynix, TSMC, UCI, or the original WM-811K data owner.

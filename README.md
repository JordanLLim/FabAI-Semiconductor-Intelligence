# FAB.AI — Semiconductor Intelligence

AI-powered semiconductor wafer defect intelligence and fab investigation platform built around the public WM-811K wafer-map dataset.

The project separates **real wafer evidence** from **simulated operational storytelling**. WM-811K is used for wafer maps and defect labels; live equipment telemetry, maintenance histories, and intervention outcomes are not claimed as measured WM-811K data.

## Current scope

- WM-811K loader and repository layer
- FastAPI backend
- Interactive wafer-map command center
- Engineered-feature ML baseline
- Reproducible training/evaluation pipeline
- Macro-F1, balanced accuracy, per-class report and confusion matrix
- Dataset profiling
- Docker / Docker Compose
- GitHub Actions CI
- Architecture and interview decision notes

## Planned path

`WM-811K -> validation -> baseline -> CNN -> similarity retrieval -> AI investigation -> React/TypeScript -> Three.js digital twin -> deployment`

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

Then open `http://localhost:8000`.

## Train baseline

```bash
python scripts/profile_dataset.py
python scripts/train.py
```

Model metrics are generated from the actual local dataset and are never hard-coded.

## Evaluation principles

WM-811K is class-imbalanced, so accuracy alone is not enough. This project reports macro-F1, balanced accuracy, per-class precision/recall/F1 and a confusion matrix. Data splitting happens before any future augmentation/resampling to reduce leakage risk.

## Disclaimer

Independent portfolio / hackathon project. Not affiliated with Micron, Samsung, SK hynix, TSMC, UCI, or the original WM-811K data owner.
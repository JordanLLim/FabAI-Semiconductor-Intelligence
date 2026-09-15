# Resume-ready project entry (truthful current version)

## FAB.AI — Semiconductor Wafer Intelligence Platform

**Python, FastAPI, scikit-learn, pandas, NumPy, Docker, GitHub Actions**

- Engineered a production-style wafer defect investigation service around the public WM-811K dataset, with validated dataset ingestion, REST APIs, wafer-map visualization, and explicit provenance controls separating measured wafer evidence from simulated demo data.
- Built a reproducible feature-engineering and Random Forest baseline pipeline for imbalanced defect classification, reporting macro-F1, balanced accuracy, per-class metrics, and confusion matrices instead of relying on accuracy alone.
- Containerized the service with Docker and added automated linting, API tests, and feature tests in GitHub Actions; designed the architecture for later CNN classification, similarity retrieval, and AI-assisted root-cause investigation.

Do not add a metric until `python scripts/train.py` has run successfully on the real local dataset.


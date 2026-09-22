# Resume-ready project entry (truthful current version)

## FAB.AI — Semiconductor Wafer Intelligence Platform

**Python, FastAPI, scikit-learn, pandas, NumPy, Docker, GitHub Actions**

- Engineered a production-style wafer defect investigation service around the public WM-811K dataset, with validated dataset ingestion, REST APIs, wafer-map visualization, and explicit provenance controls separating measured wafer evidence from simulated demo data.
- Evaluated a class-balanced Random Forest on the official 54,355/118,595 train/test split,
  reporting 0.4128 macro-F1 and 0.4391 balanced accuracy alongside class-level confusion analysis
  rather than relying on imbalance-inflated accuracy.
- Implemented explainable similar-wafer retrieval and an evidence-bounded investigation workflow
  that separates observed map patterns from unavailable equipment telemetry and causal claims.
- Containerized the FastAPI service and enforced automated linting, API tests and feature tests in
  GitHub Actions, with persisted feature contracts and model metadata for reproducible serving.

import os


# Keep API tests deterministic and fast even when a developer has the full
# WM-811K dataset and a trained artifact in the repository working tree.
os.environ["FABAI_DATASET_PATH"] = "data/test-fixtures/not-present.pkl"
os.environ["FABAI_MODEL_PATH"] = "artifacts/test-fixtures/not-present.joblib"

# Architecture

## Current request path

1. `WaferRepository` validates and normalizes the local WM-811K pickle.
2. FastAPI exposes dataset summaries and individual wafer maps.
3. `extract_features` converts each variable-size map into nine deterministic signals.
4. `ModelRegistry` lazily loads the versioned local artifact and aligns features by name.
5. The prediction endpoint returns the winning class, confidence and all class probabilities.
6. The command center renders the wafer map and shows model output only when an artifact exists.

## Training path

1. Remove records without usable labels and classes that cannot support evaluation.
2. Prefer the dataset's published training/test assignment when class coverage is complete.
3. Otherwise record use of a seeded stratified fallback.
4. Fit the class-balanced baseline on training data only.
5. Calculate macro-F1, balanced accuracy, per-class metrics and the confusion matrix on test data.
6. Persist the estimator, ordered feature contract and evaluation metadata together.

## Truth boundary

- WM-811K supplies wafer maps, failure labels and its available split labels.
- Generated demo maps exist only to exercise the application without distributing raw data.
- The API returns HTTP 503 when no model artifact exists; it never substitutes a fake prediction.
- Equipment telemetry, maintenance events and causal root-cause claims are outside the current data.

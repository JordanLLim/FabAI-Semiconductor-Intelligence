# FAB.AI architecture

## Data truth boundary
WM-811K supplies real wafer maps and defect labels. It does **not** supply live fab telemetry, maintenance logs, tool/chamber causal histories, or intervention outcomes. Those must never be presented as measured WM-811K facts.

## Production path
`WM-811K -> validation -> feature/CNN pipeline -> evaluation -> model artifact -> FastAPI -> investigation UI`

Future synthetic/demo event streams are isolated behind a simulation adapter and labelled simulated in the UI.

## Evaluation
Report macro-F1, per-class precision/recall/F1, balanced accuracy, confusion matrix, and inference latency. Accuracy alone is insufficient because defect classes are imbalanced. Split before augmentation/resampling.

## Planned visual layer
React + TypeScript command center; Three.js only for the fab/digital-twin scene. Plotly/ECharts for quantitative charts. The 3D layer is presentation, not evidence of physical simulation.

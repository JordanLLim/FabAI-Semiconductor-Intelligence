# Engineering decisions

- Real wafer evidence and simulated operational storytelling are separated.
- A lightweight engineered-feature baseline exists before CNN experimentation.
- Macro-F1 and balanced accuracy are first-class metrics because WM-811K is imbalanced.
- FastAPI separates model serving from the visual client.
- Root-cause claims require telemetry/maintenance evidence; wafer morphology alone supports defect-pattern classification, not equipment causality.

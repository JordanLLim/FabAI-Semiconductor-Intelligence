from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, classification_report
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.data.repository import load_wm811k
from src.models.cnn import WaferCNN
from src.models.split import make_split


SEED = 42
TARGET_SIZE = 32


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def normalize_map(wafer_map: np.ndarray, size: int = TARGET_SIZE) -> np.ndarray:
    """Resize a wafer map with nearest-neighbour sampling while preserving its labels."""
    height, width = wafer_map.shape
    rows = np.clip(np.rint(np.linspace(0, height - 1, size)).astype(int), 0, height - 1)
    cols = np.clip(np.rint(np.linspace(0, width - 1, size)).astype(int), 0, width - 1)
    return wafer_map[np.ix_(rows, cols)].astype(np.float32) / 2.0


def prepare(records):
    labelled = [r for r in records if r.failure_type.lower() not in {"", "unknown"}]
    labels = np.asarray([r.failure_type for r in labelled])
    counts = dict(zip(*np.unique(labels, return_counts=True), strict=True))
    rare = {label for label, count in counts.items() if count < 2}
    keep = np.asarray([label not in rare for label in labels])
    labelled = [r for r, selected in zip(labelled, keep, strict=True) if selected]
    labels = labels[keep]

    split_labels = np.asarray([r.split for r in labelled])
    split = make_split(labels, split_labels)
    classes = sorted(np.unique(labels).tolist())
    class_to_id = {label: index for index, label in enumerate(classes)}
    encoded = np.asarray([class_to_id[label] for label in labels], dtype=np.int64)

    maps = np.stack([normalize_map(r.wafer_map) for r in labelled], axis=0)[:, None, :, :]
    return maps, encoded, classes, split


def evaluate(model, loader, device, criterion=None):
    model.eval()
    losses = []
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            if criterion is not None:
                losses.append(float(criterion(logits, y).item()))
            y_true.extend(y.cpu().numpy().tolist())
            y_pred.extend(logits.argmax(dim=1).cpu().numpy().tolist())
    report = classification_report(
        y_true, y_pred, labels=list(range(model.classifier[-1].out_features)),
        target_names=None, output_dict=True, zero_division=0,
    )
    return {
        "loss": float(np.mean(losses)) if losses else None,
        "accuracy": float(np.mean(np.asarray(y_true) == np.asarray(y_pred))),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "classification_report": report,
        "y_true": y_true,
        "y_pred": y_pred,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a spatial CNN on WM-811K.")
    parser.add_argument("--data", type=Path, default=Path("data/raw/LSWMD.pkl"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/cnn"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    records = load_wm811k(args.data)
    maps, labels, classes, split = prepare(records)

    x_train, y_train = maps[split.train_indices], labels[split.train_indices]
    x_test, y_test = maps[split.test_indices], labels[split.test_indices]

    train_counts = np.bincount(y_train, minlength=len(classes)).astype(np.float32)
    class_weights = np.sqrt(train_counts.sum() / np.maximum(train_counts, 1))
    class_weights /= class_weights.mean()

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
        batch_size=args.batch_size, shuffle=True, num_workers=0,
    )
    test_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test)),
        batch_size=args.batch_size, shuffle=False, num_workers=0,
    )

    model = WaferCNN(len(classes)).to(device)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))

        metrics = evaluate(model, test_loader, device, criterion)
        metrics.pop("classification_report", None)
        metrics.pop("y_true", None)
        metrics.pop("y_pred", None)
        metrics["epoch"] = epoch
        metrics["train_loss"] = float(np.mean(train_losses))
        history.append(metrics)
        print(json.dumps(metrics))

    final = evaluate(model, test_loader, device)
    report = final["classification_report"]
    per_class = {
        classes[int(label)]: report[str(label)]
        for label in range(len(classes))
    }

    args.output.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "classes": classes,
            "target_size": TARGET_SIZE,
            "model_type": "WaferCNN",
            "training_metadata": {
                "split_strategy": split.strategy,
                "train_samples": len(y_train),
                "test_samples": len(y_test),
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.lr,
                "seed": args.seed,
                "device": str(device),
                "class_weights": class_weights.tolist(),
            },
        },
        args.output / "model.pt",
    )

    result = {
        "model_type": "WaferCNN",
        "split_strategy": split.strategy,
        "train_samples": len(y_train),
        "test_samples": len(y_test),
        "accuracy": final["accuracy"],
        "macro_f1": final["macro_f1"],
        "balanced_accuracy": final["balanced_accuracy"],
        "classification_report": per_class,
        "history": history,
        "device": str(device),
    }
    (args.output / "metrics.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "accuracy": final["accuracy"],
        "macro_f1": final["macro_f1"],
        "balanced_accuracy": final["balanced_accuracy"],
    }, indent=2))


if __name__ == "__main__":
    main()

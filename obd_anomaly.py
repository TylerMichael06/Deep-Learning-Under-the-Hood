#!/usr/bin/env python3
"""Train and use a binary PyTorch classifier for potential engine issues.

EngineFaultDB labels are grouped as follows:
    0 -> no issue
    1, 2, 3 -> potential issue

The model's validation scores are calibrated with logistic (Platt) scaling. A
prediction is YES when the calibrated issue probability reaches the configured
threshold (0.70 by default), and NO otherwise.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


DEFAULT_DATASET = "EngineFaultDB_Final.csv"
FAULT_NAMES = {
    0: "no_fault",
    1: "rich_mixture",
    2: "lean_mixture",
    3: "low_ignition_voltage",
}
ALL_FEATURES = [
    "MAP",
    "TPS",
    "Force",
    "Power",
    "RPM",
    "Consumption L/H",
    "Consumption L/100KM",
    "Speed",
    "CO",
    "HC",
    "CO2",
    "O2",
    "Lambda",
    "AFR",
]
OBD_FEATURES = ["MAP", "TPS", "RPM", "Speed", "Lambda", "AFR"]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        device = torch.device(requested)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        if device.type == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is not available.")
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_dataset_path(value: str | None) -> Path:
    if value:
        path = Path(value).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Dataset not found: {path}")
        return path

    script_dir = Path(__file__).resolve().parent
    candidates = [
        script_dir / DEFAULT_DATASET,
        script_dir.parent / DEFAULT_DATASET,
        Path.cwd() / DEFAULT_DATASET,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    searched = "\n  - ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        f"Could not locate {DEFAULT_DATASET}. Searched:\n  - {searched}\n"
        "Pass the location explicitly with --data."
    )


def load_csv(path: Path, drop_duplicates: bool) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame.columns = [str(column).strip() for column in frame.columns]
    frame.insert(0, "source_row", np.arange(2, len(frame) + 2, dtype=np.int64))
    if drop_duplicates:
        duplicate_mask = frame.drop(columns="source_row").duplicated(keep="first")
        if duplicate_mask.any():
            print(f"Removing {int(duplicate_mask.sum()):,} exact duplicate row(s).")
            frame = frame.loc[~duplicate_mask].copy()
    return frame.reset_index(drop=True)


def validate_features(frame: pd.DataFrame, feature_names: Iterable[str]) -> None:
    feature_names = list(feature_names)
    missing = [name for name in feature_names if name not in frame.columns]
    if missing:
        raise ValueError(f"CSV is missing required feature columns: {missing}")
    for name in feature_names:
        converted = pd.to_numeric(frame[name], errors="coerce")
        invalid = converted.isna() & frame[name].notna()
        if invalid.any():
            raise ValueError(f"Feature {name!r} contains non-numeric values.")
        frame[name] = converted
    if frame[feature_names].isna().any().any():
        missing_counts = frame[feature_names].isna().sum()
        missing_counts = missing_counts[missing_counts > 0].to_dict()
        raise ValueError(f"Missing feature values are not supported: {missing_counts}")
    values = frame[feature_names].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Features contain infinite or non-finite values.")


def read_fault_labels(frame: pd.DataFrame, required: bool = True) -> np.ndarray | None:
    if "Fault" not in frame.columns:
        if required:
            raise ValueError("CSV must contain a Fault column for training.")
        return None
    numeric = pd.to_numeric(frame["Fault"], errors="coerce")
    if numeric.isna().any() or not np.allclose(numeric, np.round(numeric)):
        raise ValueError("Fault must contain integer labels 0, 1, 2, or 3.")
    labels = numeric.to_numpy(dtype=np.int64)
    unknown = sorted(set(labels) - set(FAULT_NAMES))
    if unknown:
        raise ValueError(f"Unknown Fault labels: {unknown}; expected 0, 1, 2, or 3.")
    return labels


def binary_labels(fault_labels: np.ndarray) -> np.ndarray:
    return (fault_labels != 0).astype(np.int64)


def split_indices(
    original_labels: np.ndarray, split_method: str, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    indices = np.arange(len(original_labels))
    if split_method == "random":
        train_idx, remainder_idx = train_test_split(
            indices,
            test_size=0.30,
            random_state=seed,
            stratify=original_labels,
        )
        val_idx, test_idx = train_test_split(
            remainder_idx,
            test_size=0.50,
            random_state=seed,
            stratify=original_labels[remainder_idx],
        )
        return np.sort(train_idx), np.sort(val_idx), np.sort(test_idx)

    train_parts: list[np.ndarray] = []
    val_parts: list[np.ndarray] = []
    test_parts: list[np.ndarray] = []
    for label in sorted(np.unique(original_labels)):
        label_idx = indices[original_labels == label]
        train_end = int(math.floor(0.70 * len(label_idx)))
        val_end = train_end + int(math.floor(0.15 * len(label_idx)))
        train_parts.append(label_idx[:train_end])
        val_parts.append(label_idx[train_end:val_end])
        test_parts.append(label_idx[val_end:])
    return (
        np.sort(np.concatenate(train_parts)),
        np.sort(np.concatenate(val_parts)),
        np.sort(np.concatenate(test_parts)),
    )


class FaultClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple[int, ...] = (64, 32),
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        current_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(current_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, 2))
        self.network = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


def make_loader(
    features: np.ndarray,
    labels: np.ndarray | None,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    feature_tensor = torch.from_numpy(features.astype(np.float32, copy=False))
    if labels is None:
        dataset = TensorDataset(feature_tensor)
    else:
        label_tensor = torch.from_numpy(labels.astype(np.int64, copy=False))
        dataset = TensorDataset(feature_tensor, label_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def evaluate_logits(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module | None = None,
) -> tuple[np.ndarray, float | None]:
    model.eval()
    logits_parts: list[np.ndarray] = []
    total_loss = 0.0
    total_rows = 0
    with torch.inference_mode():
        for batch in loader:
            inputs = batch[0].to(device)
            logits = model(inputs)
            logits_parts.append(logits.detach().cpu().numpy())
            if criterion is not None and len(batch) == 2:
                targets = batch[1].to(device)
                loss = criterion(logits, targets)
                total_loss += float(loss.item()) * len(inputs)
                total_rows += len(inputs)
    logits_array = np.concatenate(logits_parts, axis=0)
    average_loss = total_loss / total_rows if total_rows else None
    return logits_array, average_loss


def raw_issue_scores(logits: np.ndarray) -> np.ndarray:
    return logits[:, 1] - logits[:, 0]


def fit_calibrator(scores: np.ndarray, labels: np.ndarray, seed: int) -> tuple[float, float]:
    calibrator = LogisticRegression(C=1e6, solver="lbfgs", random_state=seed)
    calibrator.fit(scores.reshape(-1, 1), labels)
    return float(calibrator.coef_[0, 0]), float(calibrator.intercept_[0])


def calibrated_probabilities(scores: np.ndarray, coefficient: float, intercept: float) -> np.ndarray:
    scores = scores.astype(np.float64, copy=False)
    scaled = np.clip(coefficient * scores + intercept, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-scaled))


def calculate_metrics(
    labels: np.ndarray, probabilities: np.ndarray, threshold: float
) -> tuple[dict[str, Any], np.ndarray]:
    predictions = (probabilities >= threshold).astype(np.int64)
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    metrics: dict[str, Any] = {
        "issue_threshold": threshold,
        "accuracy": accuracy_score(labels, predictions),
        "issue_precision": precision_score(labels, predictions, zero_division=0),
        "issue_recall": recall_score(labels, predictions, zero_division=0),
        "issue_f1": f1_score(labels, predictions, zero_division=0),
        "macro_f1": f1_score(labels, predictions, average="macro", zero_division=0),
        "roc_auc": roc_auc_score(labels, probabilities),
        "average_precision": average_precision_score(labels, probabilities),
        "brier_score": brier_score_loss(labels, probabilities),
        "confusion_matrix": matrix.tolist(),
        "classification_report": classification_report(
            labels,
            predictions,
            labels=[0, 1],
            target_names=["NO: no issue", "YES: potential issue"],
            output_dict=True,
            zero_division=0,
        ),
    }
    return metrics, predictions


def make_prediction_frame(
    source_rows: np.ndarray,
    probabilities: np.ndarray,
    predictions: np.ndarray,
    threshold: float,
    split: str,
    original_labels: np.ndarray | None = None,
) -> pd.DataFrame:
    output = pd.DataFrame(
        {
            "source_row": source_rows,
            "split": split,
            "potential_issue": np.where(predictions == 1, "YES", "NO"),
            "issue_probability": probabilities,
            "no_issue_probability": 1.0 - probabilities,
            "decision_confidence": np.where(
                predictions == 1, probabilities, 1.0 - probabilities
            ),
            "alert_threshold": threshold,
        }
    )
    if original_labels is not None:
        output.insert(1, "true_fault_label", original_labels)
        output.insert(
            2,
            "true_fault_class",
            [FAULT_NAMES[int(label)] for label in original_labels],
        )
        output.insert(
            3,
            "true_potential_issue",
            np.where(binary_labels(original_labels) == 1, "YES", "NO"),
        )
    return output


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def inspect_command(args: argparse.Namespace) -> None:
    data_path = resolve_dataset_path(args.data)
    frame = load_csv(data_path, drop_duplicates=not args.keep_duplicates)
    labels = read_fault_labels(frame, required=True)
    assert labels is not None
    validate_features(frame, ALL_FEATURES)

    print(f"Dataset: {data_path}")
    print(f"Rows after duplicate handling: {len(frame):,}")
    print(f"Columns ({len(frame.columns) - 1}): {', '.join(frame.columns[1:])}")
    print("\nOriginal fault classes:")
    for label, count in zip(*np.unique(labels, return_counts=True)):
        print(f"  {label} ({FAULT_NAMES[int(label)]}): {int(count):,}")
    issue_labels = binary_labels(labels)
    no_count = int((issue_labels == 0).sum())
    yes_count = int((issue_labels == 1).sum())
    print("\nBinary target:")
    print(f"  NO  (Fault = 0): {no_count:,}")
    print(f"  YES (Fault = 1, 2, or 3): {yes_count:,}")
    print(f"Exact missing cells: {int(frame.isna().sum().sum()):,}")


def train_command(args: argparse.Namespace) -> None:
    if not 0.0 < args.issue_threshold < 1.0:
        raise ValueError("--issue-threshold must be strictly between 0 and 1.")
    if args.epochs < 1 or args.patience < 1 or args.batch_size < 1:
        raise ValueError("epochs, patience, and batch size must be positive.")

    set_seed(args.seed)
    device = choose_device(args.device)
    data_path = resolve_dataset_path(args.data)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_names = ALL_FEATURES if args.feature_set == "all" else OBD_FEATURES

    frame = load_csv(data_path, drop_duplicates=not args.keep_duplicates)
    validate_features(frame, feature_names)
    original_labels = read_fault_labels(frame, required=True)
    assert original_labels is not None
    labels = binary_labels(original_labels)
    features = frame[feature_names].to_numpy(dtype=np.float32)
    train_idx, val_idx, test_idx = split_indices(original_labels, args.split_method, args.seed)

    mean = features[train_idx].mean(axis=0, dtype=np.float64)
    scale = features[train_idx].std(axis=0, dtype=np.float64)
    scale[scale < 1e-12] = 1.0
    standardized = ((features - mean) / scale).astype(np.float32)

    train_loader = make_loader(
        standardized[train_idx], labels[train_idx], args.batch_size, shuffle=True
    )
    val_loader = make_loader(
        standardized[val_idx], labels[val_idx], args.batch_size, shuffle=False
    )
    test_loader = make_loader(
        standardized[test_idx], labels[test_idx], args.batch_size, shuffle=False
    )

    model = FaultClassifier(len(feature_names), dropout=args.dropout).to(device)
    counts = np.bincount(labels[train_idx], minlength=2).astype(np.float64)
    weights = counts.sum() / (2.0 * counts)
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(weights, dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    print(f"Dataset: {data_path}")
    print(f"Device: {device}")
    print(f"Features ({len(feature_names)}): {', '.join(feature_names)}")
    print(
        f"Split ({args.split_method}): train={len(train_idx):,}, "
        f"validation={len(val_idx):,}, test={len(test_idx):,}"
    )
    print(f"YES threshold: {args.issue_threshold:.0%}")

    history: list[dict[str, float | int]] = []
    best_val_loss = math.inf
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    stale_epochs = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_rows = 0
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(inputs)
            total_rows += len(inputs)

        train_loss = total_loss / total_rows
        val_logits, val_loss = evaluate_logits(model, val_loader, device, criterion)
        assert val_loss is not None
        val_predictions = np.argmax(val_logits, axis=1)
        val_accuracy = accuracy_score(labels[val_idx], val_predictions)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": val_loss,
                "validation_accuracy_raw_0_5": val_accuracy,
            }
        )
        print(
            f"Epoch {epoch:03d} | train loss {train_loss:.5f} | "
            f"validation loss {val_loss:.5f} | raw validation accuracy {val_accuracy:.4f}"
        )

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print(f"Early stopping after epoch {epoch}; best epoch was {best_epoch}.")
                break

    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint.")
    model.load_state_dict(best_state)

    val_logits, _ = evaluate_logits(model, val_loader, device)
    coefficient, intercept = fit_calibrator(
        raw_issue_scores(val_logits), labels[val_idx], args.seed
    )
    test_logits, _ = evaluate_logits(model, test_loader, device)
    test_probabilities = calibrated_probabilities(
        raw_issue_scores(test_logits), coefficient, intercept
    )
    test_metrics, test_predictions = calculate_metrics(
        labels[test_idx], test_probabilities, args.issue_threshold
    )

    checkpoint = {
        "model_state_dict": best_state,
        "feature_names": feature_names,
        "feature_set": args.feature_set,
        "hidden_dims": [64, 32],
        "dropout": args.dropout,
        "scaler_mean": mean.tolist(),
        "scaler_scale": scale.tolist(),
        "calibration_coefficient": coefficient,
        "calibration_intercept": intercept,
        "issue_threshold": args.issue_threshold,
        "fault_names": FAULT_NAMES,
        "binary_mapping": {0: "NO", 1: "YES", 2: "YES", 3: "YES"},
        "seed": args.seed,
        "split_method": args.split_method,
        "best_epoch": best_epoch,
    }
    torch.save(checkpoint, output_dir / "model.pt")
    pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False)

    prediction_frame = make_prediction_frame(
        frame.loc[test_idx, "source_row"].to_numpy(),
        test_probabilities,
        test_predictions,
        args.issue_threshold,
        "test",
        original_labels[test_idx],
    )
    prediction_frame.to_csv(output_dir / "predictions.csv", index=False)
    pd.DataFrame(
        test_metrics["confusion_matrix"],
        index=["actual_NO", "actual_YES"],
        columns=["predicted_NO", "predicted_YES"],
    ).to_csv(output_dir / "confusion_matrix.csv")

    metrics_payload = {
        "test": test_metrics,
        "calibration": {
            "method": "Platt scaling on validation split",
            "coefficient": coefficient,
            "intercept": intercept,
        },
    }
    write_json(output_dir / "metrics.json", metrics_payload)
    summary = {
        "dataset": str(data_path),
        "rows_after_duplicate_handling": len(frame),
        "feature_names": feature_names,
        "binary_target": {"NO": "Fault = 0", "YES": "Fault in 1, 2, or 3"},
        "issue_threshold": args.issue_threshold,
        "split_method": args.split_method,
        "split_sizes": {
            "train": len(train_idx),
            "validation": len(val_idx),
            "test": len(test_idx),
        },
        "best_epoch": best_epoch,
        "device": str(device),
        "test_metrics": test_metrics,
    }
    write_json(output_dir / "summary.json", summary)

    print("\nUntouched test results at the saved threshold:")
    print(f"  Accuracy:        {test_metrics['accuracy']:.4f}")
    print(f"  Issue precision: {test_metrics['issue_precision']:.4f}")
    print(f"  Issue recall:    {test_metrics['issue_recall']:.4f}")
    print(f"  Issue F1:        {test_metrics['issue_f1']:.4f}")
    print(f"  ROC AUC:         {test_metrics['roc_auc']:.4f}")
    print(f"  Brier score:     {test_metrics['brier_score']:.4f}")
    print(f"Artifacts written to: {output_dir}")


def predict_command(args: argparse.Namespace) -> None:
    model_path = Path(args.model).expanduser().resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    feature_names = list(checkpoint["feature_names"])
    threshold = (
        float(args.issue_threshold)
        if args.issue_threshold is not None
        else float(checkpoint["issue_threshold"])
    )
    if not 0.0 < threshold < 1.0:
        raise ValueError("--issue-threshold must be strictly between 0 and 1.")

    data_path = resolve_dataset_path(args.data)
    frame = load_csv(data_path, drop_duplicates=False)
    validate_features(frame, feature_names)
    original_labels = read_fault_labels(frame, required=False)
    features = frame[feature_names].to_numpy(dtype=np.float64)
    mean = np.asarray(checkpoint["scaler_mean"], dtype=np.float64)
    scale = np.asarray(checkpoint["scaler_scale"], dtype=np.float64)
    standardized = ((features - mean) / scale).astype(np.float32)

    device = choose_device(args.device)
    model = FaultClassifier(
        input_dim=len(feature_names),
        hidden_dims=tuple(checkpoint["hidden_dims"]),
        dropout=float(checkpoint["dropout"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    loader = make_loader(standardized, None, args.batch_size, shuffle=False)
    logits, _ = evaluate_logits(model, loader, device)
    probabilities = calibrated_probabilities(
        raw_issue_scores(logits),
        float(checkpoint["calibration_coefficient"]),
        float(checkpoint["calibration_intercept"]),
    )
    predictions = (probabilities >= threshold).astype(np.int64)
    output = make_prediction_frame(
        frame["source_row"].to_numpy(),
        probabilities,
        predictions,
        threshold,
        "inference",
        original_labels,
    )
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)

    yes_count = int(predictions.sum())
    print(f"Scored {len(output):,} row(s) on {device}.")
    print(f"YES (potential issue): {yes_count:,}")
    print(f"NO  (threshold not reached): {len(output) - yes_count:,}")
    print(f"Threshold: {threshold:.0%}")
    print(f"Predictions written to: {output_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Binary potential-engine-issue classifier for EngineFaultDB."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect", help="Inspect labels and columns.")
    inspect_parser.add_argument("--data", help="Path to EngineFaultDB CSV.")
    inspect_parser.add_argument(
        "--keep-duplicates", action="store_true", help="Do not remove exact duplicates."
    )
    inspect_parser.set_defaults(func=inspect_command)

    train_parser = subparsers.add_parser("train", help="Train and evaluate the model.")
    train_parser.add_argument("--data", help="Path to EngineFaultDB CSV.")
    train_parser.add_argument("--output-dir", default="artifacts", help="Artifact directory.")
    train_parser.add_argument(
        "--feature-set", choices=["all", "obd"], default="all"
    )
    train_parser.add_argument(
        "--split-method",
        choices=["block", "random"],
        default="block",
        help="Conservative ordered blocks per source class, or random stratified split.",
    )
    train_parser.add_argument("--issue-threshold", type=float, default=0.70)
    train_parser.add_argument("--epochs", type=int, default=100)
    train_parser.add_argument("--patience", type=int, default=12)
    train_parser.add_argument("--batch-size", type=int, default=512)
    train_parser.add_argument("--learning-rate", type=float, default=1e-3)
    train_parser.add_argument("--weight-decay", type=float, default=1e-4)
    train_parser.add_argument("--dropout", type=float, default=0.15)
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument(
        "--device", default="auto", help="auto, cpu, cuda, or mps"
    )
    train_parser.add_argument("--keep-duplicates", action="store_true")
    train_parser.set_defaults(func=train_command)

    predict_parser = subparsers.add_parser("predict", help="Score a CSV with a checkpoint.")
    predict_parser.add_argument("--data", required=True, help="CSV to score.")
    predict_parser.add_argument("--model", default="artifacts/model.pt")
    predict_parser.add_argument("--output", default="artifacts/new_predictions.csv")
    predict_parser.add_argument(
        "--issue-threshold",
        type=float,
        help="Optional override; otherwise use the checkpoint threshold.",
    )
    predict_parser.add_argument("--batch-size", type=int, default=1024)
    predict_parser.add_argument("--device", default="auto")
    predict_parser.set_defaults(func=predict_command)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

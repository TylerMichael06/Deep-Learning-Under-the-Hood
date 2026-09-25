import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from obdfault.data import dieselobd, enginead, enginefaultdb
from obdfault.metrics import binary_metrics, pick_threshold
from obdfault.models import CNN1d, MLP, baseline, predict_proba, train_torch, window_features

DATASETS = {"enginefaultdb": enginefaultdb.folds, "dieselobd": dieselobd.folds, "enginead": enginead.folds}
MODELS = ("logreg", "gbm", "nn")


def fit_predict(model_name, fold, seed):
    windows = fold.X_train.ndim == 3
    if model_name in ("logreg", "gbm"):
        prep = window_features if windows else (lambda X: X)
        clf = baseline(model_name, seed).fit(prep(fold.X_train), fold.y_train)
        return clf.predict_proba(prep(fold.X_val))[:, 1], clf.predict_proba(prep(fold.X_test))[:, 1]
    torch.manual_seed(seed)  # before construction, so weight init is seeded too
    model = (CNN1d if windows else MLP)(fold.X_train)
    train_torch(model, fold, seed=seed)
    return predict_proba(model, fold.X_val), predict_proba(model, fold.X_test)


def run(dataset, model, data_root="data", out_root="artifacts", seed=0):
    if model not in MODELS:
        raise ValueError(f"unknown model {model!r}; choose from {MODELS}")
    rows, thresholds = [], []
    for k, fold in enumerate(DATASETS[dataset](Path(data_root))):
        p_val, p_test = fit_predict(model, fold, seed)
        threshold = pick_threshold(fold.y_val, p_val)
        thresholds.append(threshold)
        rows.append(pd.DataFrame({"fold": k, "group": fold.test_groups, "y": fold.y_test,
                                  "p": p_test, "pred": (p_test >= threshold).astype(int)}))
    preds = pd.concat(rows, ignore_index=True)
    metrics = binary_metrics(preds["y"].to_numpy(), preds["p"].to_numpy(), preds["pred"].to_numpy())
    metrics["thresholds"] = thresholds
    metrics["per_group"] = (
        preds.groupby("group")
        .agg(n=("y", "size"), fault_rate=("y", "mean"), mean_p=("p", "mean"), predicted_fault_rate=("pred", "mean"))
        .reset_index()
        .to_dict(orient="records")
    )
    out = Path(out_root) / dataset / model
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, default=lambda o: o.item()))
    preds.to_csv(out / "predictions.csv", index=False)
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate a binary engine-fault classifier.")
    parser.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    parser.add_argument("--model", required=True, choices=MODELS)
    parser.add_argument("--data", default="data")
    parser.add_argument("--out", default="artifacts")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    m = run(args.dataset, args.model, args.data, args.out, args.seed)
    fmt = lambda v: "n/a" if v is None else f"{v:.3f}"
    print(f"{args.dataset}/{args.model}: F1 {fmt(m['f1'])}  bal-acc {fmt(m['balanced_accuracy'])}  "
          f"AUROC {fmt(m['auroc'])}  AUPRC {fmt(m['auprc'])}  TN {m['tn']} FP {m['fp']} FN {m['fn']} TP {m['tp']}")


if __name__ == "__main__":
    main()

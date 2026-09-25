import json

import numpy as np
import pandas as pd
import pytest

from obdfault import run as runner
from obdfault.splits import leave_one_group_out, make_fold


def toy_folds(root):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 3)).astype(np.float32)
    y = (X[:, 0] > 0).astype(np.int64)
    groups = np.repeat(["a", "b", "c"], 100)
    return [make_fold(X, y, groups, *idx) for idx in leave_one_group_out(groups)]


@pytest.mark.parametrize("model", ["logreg", "gbm", "nn"])
def test_run_writes_metrics_and_predictions(tmp_path, monkeypatch, model):
    monkeypatch.setitem(runner.DATASETS, "toy", toy_folds)
    metrics = runner.run("toy", model, data_root=tmp_path, out_root=tmp_path / "artifacts")
    out = tmp_path / "artifacts" / "toy" / model
    saved = json.loads((out / "metrics.json").read_text())
    preds = pd.read_csv(out / "predictions.csv")
    assert saved["auroc"] == metrics["auroc"]
    if model != "nn":  # nn gets one batch per epoch on this toy set; learning is covered in test_models
        assert metrics["auroc"] > 0.9
    assert len(saved["thresholds"]) == 3
    assert {g["group"] for g in saved["per_group"]} == {"a", "b", "c"}
    assert list(preds.columns) == ["fold", "group", "y", "p", "pred"] and len(preds) == 300


def test_unknown_model_raises(tmp_path, monkeypatch):
    monkeypatch.setitem(runner.DATASETS, "toy", toy_folds)
    with pytest.raises(ValueError, match="model"):
        runner.run("toy", "svm", data_root=tmp_path, out_root=tmp_path)

import numpy as np

from obdfault.metrics import binary_metrics, pick_threshold


def test_pick_threshold_separates_perfectly_separable_scores():
    y = np.array([0, 0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.3, 0.7, 0.9])
    threshold = pick_threshold(y, p)
    assert 0.3 < threshold <= 0.7
    assert ((p >= threshold).astype(int) == y).all()


def test_pick_threshold_defaults_to_half_for_single_class():
    assert pick_threshold(np.array([1, 1, 1]), np.array([0.2, 0.6, 0.9])) == 0.5


def test_binary_metrics_values():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.6, 0.4, 0.9])
    pred = np.array([0, 1, 0, 1])
    m = binary_metrics(y, p, pred)
    assert (m["tn"], m["fp"], m["fn"], m["tp"]) == (1, 1, 1, 1)
    assert m["f1"] == 0.5
    assert m["balanced_accuracy"] == 0.5
    assert m["auroc"] == 0.75
    assert m["n"] == 4 and m["positives"] == 2


def test_binary_metrics_single_class_has_no_ranking_metrics():
    m = binary_metrics(np.array([0, 0]), np.array([0.2, 0.7]), np.array([0, 1]))
    assert m["auroc"] is None and m["auprc"] is None and m["balanced_accuracy"] is None
    assert m["fp"] == 1

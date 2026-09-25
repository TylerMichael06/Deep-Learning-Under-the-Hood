import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from obdfault.models import CNN1d, MLP, baseline, predict_proba, train_torch, window_features
from obdfault.splits import contiguous_split, make_fold


def tabular_fold(n=600):
    rng = np.random.default_rng(0)
    X = (rng.normal(size=(n, 3)) * [1, 10, 100]).astype(np.float32)
    y = (X[:, 0] > 0).astype(np.int64)
    return make_fold(X, y, np.zeros(n).astype(str), *contiguous_split(n))


def window_fold(n=300):
    rng = np.random.default_rng(0)
    y = (np.arange(n) % 2).astype(np.int64)
    X = rng.normal(size=(n, 40, 2)).astype(np.float32)
    X[y == 1, 10:20, 0] += 3.0
    return make_fold(X, y, np.zeros(n).astype(str), *contiguous_split(n))


def test_window_features_shape():
    assert window_features(np.zeros((5, 40, 2), dtype=np.float32)).shape == (5, 8)


def test_forward_shapes():
    tab, win = tabular_fold(), window_fold()
    assert MLP(tab.X_train)(torch.zeros(4, 3)).shape == (4,)
    assert CNN1d(win.X_train)(torch.zeros(4, 40, 2)).shape == (4,)


def test_mlp_learns_separable_data():
    fold = tabular_fold()
    model = MLP(fold.X_train)
    history = train_torch(model, fold, epochs=20, batch_size=64)
    assert len(history) >= 1
    assert roc_auc_score(fold.y_test, predict_proba(model, fold.X_test)) > 0.95


def test_cnn_learns_separable_windows():
    fold = window_fold()
    model = CNN1d(fold.X_train)
    train_torch(model, fold, epochs=20, batch_size=32)
    assert roc_auc_score(fold.y_test, predict_proba(model, fold.X_test)) > 0.95


def test_baselines_fit_and_predict():
    fold = tabular_fold()
    for name in ("logreg", "gbm"):
        p = baseline(name, seed=0).fit(fold.X_train, fold.y_train).predict_proba(fold.X_test)[:, 1]
        assert roc_auc_score(fold.y_test, p) > 0.95

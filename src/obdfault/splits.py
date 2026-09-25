from dataclasses import dataclass

import numpy as np


@dataclass
class Fold:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    test_groups: np.ndarray


def contiguous_split(n, train=0.70, val=0.15):
    a = int(n * train)
    b = a + int(n * val)
    idx = np.arange(n)
    return idx[:a], idx[a:b], idx[b:]


def block_split_by_class(labels, train=0.70, val=0.15):
    parts = ([], [], [])
    for label in np.unique(labels):
        idx = np.flatnonzero(labels == label)
        for part, sel in zip(parts, contiguous_split(len(idx), train, val)):
            part.append(idx[sel])
    return tuple(np.concatenate(p) for p in parts)


def group_split(groups, seed=0, val_frac=0.15, test_frac=0.15):
    names = np.unique(groups)
    if len(names) < 3:
        raise ValueError(f"need at least 3 groups, got {len(names)}")
    np.random.default_rng(seed).shuffle(names)
    n_test = max(1, round(len(names) * test_frac))
    n_val = max(1, round(len(names) * val_frac))
    chosen = names[:n_test], names[n_test:n_test + n_val], names[n_test + n_val:]
    test, val, train = (np.flatnonzero(np.isin(groups, c)) for c in chosen)
    return train, val, test


def leave_one_group_out(groups, val_frac=0.15):
    names = np.unique(groups)
    folds = []
    for held_out in names:
        train, val = [], []
        for name in names:
            if name == held_out:
                continue
            idx = np.flatnonzero(groups == name)
            cut = len(idx) - int(len(idx) * val_frac)
            train.append(idx[:cut])
            val.append(idx[cut:])
        folds.append((np.concatenate(train), np.concatenate(val), np.flatnonzero(groups == held_out)))
    return folds


def make_fold(X, y, groups, train, val, test):
    return Fold(X[train], y[train], X[val], y[val], X[test], y[test], groups[test])

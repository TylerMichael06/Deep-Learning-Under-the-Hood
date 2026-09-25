import os
import pickle

import numpy as np
import pandas as pd
import pytest

from obdfault.data import enginead


def window(label_steps, truck, steps=300):
    df = pd.DataFrame(np.random.default_rng(0).normal(size=(steps, 8)), columns=enginead.CHANNELS)
    df["label"] = 0
    if label_steps:
        df.loc[: label_steps - 1, "label"] = 1
    df["vehicle_id"] = pd.Series([truck] * steps, dtype=object)  # the real files use object dtype
    return df


def write_trucks(root, n=4):
    folder = root / "EngineAD"
    folder.mkdir(parents=True)
    for k in range(1, n + 1):
        windows = [window(0, f"truck {k}"), window(5, f"truck {k}"), window(0, f"truck {k}")]
        with open(folder / f"truck_{k}-1.pickle", "wb") as f:
            pickle.dump(windows, f)


def test_read_truck_labels_windows_by_any_anomalous_step(tmp_path):
    write_trucks(tmp_path, n=1)
    X, y = enginead.read_truck(tmp_path / "EngineAD" / "truck_1-1.pickle")
    assert X.shape == (3, 300, 8) and X.dtype == np.float32
    assert y.tolist() == [0, 1, 0]


def test_load_caches_and_groups_by_truck(tmp_path):
    write_trucks(tmp_path)
    X, y, truck = enginead.load(tmp_path)
    assert X.shape == (12, 300, 8)
    assert sorted(set(truck)) == [f"truck_{k}-1" for k in range(1, 5)]
    assert (tmp_path / "EngineAD" / "cache" / "truck_1-1.npz").exists()
    X2, _, _ = enginead.load(tmp_path)
    assert np.array_equal(X, X2)


def test_folds_split_by_truck(tmp_path):
    write_trucks(tmp_path)
    [fold] = enginead.folds(tmp_path)
    assert len(fold.y_train) + len(fold.y_val) + len(fold.y_test) == 12
    assert fold.X_test.shape[1:] == (300, 8)


class Evil:
    def __reduce__(self):
        return (os.getcwd, ())


def test_unexpected_pickle_class_is_blocked(tmp_path):
    folder = tmp_path / "EngineAD"
    folder.mkdir(parents=True)
    with open(folder / "truck_1-1.pickle", "wb") as f:
        pickle.dump([Evil()], f)
    with pytest.raises(pickle.UnpicklingError, match="blocked"):
        enginead.read_truck(folder / "truck_1-1.pickle")


def test_too_few_trucks_raises(tmp_path):
    write_trucks(tmp_path, n=2)
    with pytest.raises(FileNotFoundError, match="download_data"):
        enginead.load(tmp_path)

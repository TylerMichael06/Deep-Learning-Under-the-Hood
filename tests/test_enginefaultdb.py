import numpy as np
import pandas as pd
import pytest

from obdfault.data import enginefaultdb


def write_csv(root, rows_per_class=20):
    folder = root / "EngineFaultDB"
    folder.mkdir(parents=True)
    rng = np.random.default_rng(0)
    frames = []
    for fault in range(4):
        df = pd.DataFrame(rng.normal(size=(rows_per_class, 14)), columns=enginefaultdb.FEATURES)
        df.insert(0, "Fault", fault)
        frames.append(df)
    pd.concat(frames).to_csv(folder / "EngineFaultDB_Final.csv", index=False)


def test_folds_single_fold_binary_labels_all_classes_in_test(tmp_path):
    write_csv(tmp_path)
    [fold] = enginefaultdb.folds(tmp_path)
    assert fold.X_train.shape == (56, 14) and fold.X_train.dtype == np.float32
    assert len(fold.y_val) == 12 and len(fold.y_test) == 12
    assert set(fold.test_groups) == set(enginefaultdb.FAULT_NAMES.values())
    no_fault = fold.test_groups == "no_fault"
    assert (fold.y_test[no_fault] == 0).all() and (fold.y_test[~no_fault] == 1).all()


def test_missing_column_raises(tmp_path):
    write_csv(tmp_path)
    path = tmp_path / "EngineFaultDB" / "EngineFaultDB_Final.csv"
    pd.read_csv(path).drop(columns=["AFR"]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="AFR"):
        enginefaultdb.folds(tmp_path)


def test_unknown_fault_label_raises(tmp_path):
    write_csv(tmp_path)
    path = tmp_path / "EngineFaultDB" / "EngineFaultDB_Final.csv"
    df = pd.read_csv(path)
    df.loc[0, "Fault"] = 7
    df.to_csv(path, index=False)
    with pytest.raises(ValueError, match="Fault"):
        enginefaultdb.folds(tmp_path)

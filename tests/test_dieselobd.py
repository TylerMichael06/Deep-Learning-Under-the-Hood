import numpy as np
import pandas as pd
import pytest

from obdfault.data import dieselobd

CODES = ["P0000", "P0403", "P0113", "P0562", "P0107", "P0234"]
BLOCKS = [["P0403"], ["P0000"], ["P0113"], [], ["P0562"], ["P0107"], ["P0234"], ["P0000"]]
LABELS = [1, 0, 1, 0, 1, 1, 1, 0]


def make_frame(rows_per_vehicle, blocks=BLOCKS):
    rng = np.random.default_rng(0)
    frames = []
    for codes in blocks:
        df = pd.DataFrame(rng.normal(size=(rows_per_vehicle, 11)), columns=dieselobd.PIDS)
        for c in CODES:
            df[c] = int(c in codes)
        df["Mode"] = 0
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def write_files(root, blocks=BLOCKS):
    folder = root / "OBD-Dataset"
    folder.mkdir(parents=True)
    make_frame(20, blocks).to_excel(folder / "MASTER_TRAIN_X.xlsx", sheet_name="Train", index=False)
    make_frame(10, blocks).to_excel(folder / "MASTER_TEST_X.xlsx", sheet_name="Test", index=False)


def test_load_merges_files_and_labels_vehicles(tmp_path):
    write_files(tmp_path)
    X, y, vehicle = dieselobd.load(tmp_path)
    assert X.shape == (240, 11) and X.dtype == np.float32
    for k, label in enumerate(LABELS, start=1):
        rows = vehicle == f"vehicle_{k}"
        assert rows.sum() == 30
        assert (y[rows] == label).all()


def test_folds_leave_one_vehicle_out(tmp_path):
    write_files(tmp_path)
    folds = dieselobd.folds(tmp_path)
    assert len(folds) == 8
    for fold in folds:
        assert len(set(fold.test_groups)) == 1
        assert len(fold.y_test) == 30
        assert len(fold.y_train) + len(fold.y_val) == 210


def test_wrong_vehicle_count_raises(tmp_path):
    write_files(tmp_path, blocks=BLOCKS[:7])
    with pytest.raises(ValueError, match="8 vehicle blocks"):
        dieselobd.load(tmp_path)

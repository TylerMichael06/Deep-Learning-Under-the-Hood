import re
from pathlib import Path

import numpy as np
import pandas as pd

from obdfault.splits import leave_one_group_out, make_fold

PIDS = ["LOAD_PCT", "ECT", "MAP", "RPM", "VSS", "IAT", "MAF", "FRP", "BARO", "VPWR", "AAT"]
NO_FAULT_CODE = "P0000"
N_VEHICLES = 8
FILES = ["MASTER_TRAIN_X.xlsx", "MASTER_TEST_X.xlsx"]


def load_file(path):
    df = pd.read_excel(path)
    missing = [c for c in PIDS if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name} is missing columns: {missing}")
    dtcs = [c for c in df.columns if re.fullmatch(r"P\d{4}", str(c))]
    # ponytail: the files have no vehicle column; each vehicle is one contiguous block of identical DTC flags
    signature = df[dtcs].astype(str).agg("".join, axis=1)
    block = (signature != signature.shift()).cumsum().to_numpy(np.int64)
    if block.max() != N_VEHICLES:
        raise ValueError(f"{path.name}: expected {N_VEHICLES} vehicle blocks, found {block.max()}")
    faults = [c for c in dtcs if c != NO_FAULT_CODE]
    y = (df[faults].sum(axis=1) > 0).to_numpy(np.int64)
    return df[PIDS].to_numpy(np.float32), y, block


def load(root):
    parts = [load_file(Path(root) / "OBD-Dataset" / name) for name in FILES]
    X = np.concatenate([p[0] for p in parts])
    y = np.concatenate([p[1] for p in parts])
    vehicle = np.array([f"vehicle_{b}" for b in np.concatenate([p[2] for p in parts])])
    for name in np.unique(vehicle):
        if len(np.unique(y[vehicle == name])) != 1:
            raise ValueError(f"{name} has different labels in the train and test files")
    return X, y, vehicle


def folds(root):
    X, y, vehicle = load(root)
    return [make_fold(X, y, vehicle, *idx) for idx in leave_one_group_out(vehicle)]

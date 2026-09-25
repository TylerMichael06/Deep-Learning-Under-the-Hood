from pathlib import Path

import numpy as np
import pandas as pd

from obdfault.splits import Fold, block_split_by_class

FEATURES = [
    "MAP", "TPS", "Force", "Power", "RPM", "Consumption L/H", "Consumption L/100KM",
    "Speed", "CO", "HC", "CO2", "O2", "Lambda", "AFR",
]
FAULT_NAMES = {0: "no_fault", 1: "rich_mixture", 2: "lean_mixture", 3: "low_voltage"}


def load(path):
    df = pd.read_csv(path)
    missing = [c for c in ["Fault", *FEATURES] if c not in df.columns]
    if missing:
        raise ValueError(f"EngineFaultDB is missing columns: {missing}")
    if not df["Fault"].isin(list(FAULT_NAMES)).all():
        raise ValueError("Fault must only contain 0, 1, 2, 3")
    fault = df["Fault"].to_numpy(np.int64)
    return df[FEATURES].to_numpy(np.float32), (fault != 0).astype(np.int64), fault


def folds(root):
    X, y, fault = load(Path(root) / "EngineFaultDB" / "EngineFaultDB_Final.csv")
    train, val, test = block_split_by_class(fault)
    names = np.array([FAULT_NAMES[f] for f in fault[test]])
    return [Fold(X[train], y[train], X[val], y[val], X[test], y[test], names)]

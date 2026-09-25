# Public-Data Binary Fault Classifiers Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train and evaluate "engine fault: yes/no" classifiers on EngineFaultDB, OBD-Dataset (DieselOBD) and EngineAD, with group-safe splits and comparable metrics.

**Architecture:** A small `obdfault` package under `src/`. Each dataset has one loader that returns a list of `Fold`s (train/val/test arrays plus test group names). Everything downstream is shared: models (logistic regression, gradient boosting, PyTorch MLP/1D-CNN), threshold picking, metrics, and one CLI that writes `artifacts/<dataset>/<model>/`.

**Tech Stack:** Python 3.14, numpy 2.5.3, pandas 3.0.6, scikit-learn 1.9.1, torch 2.14.0, openpyxl 3.1.5, pytest 9.1.1.

**Spec:** `OBD2-Predictive-Maintenance-Design.md` (§3–§4, §6, §8). The car stage (§5) is out of scope; it gets its own plan once the dongle is tested.

## Global Constraints

- Work on branch `alex-ye` in `/Users/alexye/Desktop/CSCI566/Deep-Learning-Under-the-Hood`. Do not push without asking the user.
- Commit messages: **no** `Co-Authored-By` or other attribution trailer (user rule in the local `CLAUDE.md`).
- Never commit `data/`, `artifacts/`, `.venv/`, or `CLAUDE.md`.
- Independent implementation: do not read or copy code from teammates' branches (e.g. `anthony-mvp`).
- Use the project venv: `.venv/bin/python`, `.venv/bin/pytest`, `.venv/bin/pip` (already created; packages above already installed).
- Splits are never random rows: EngineFaultDB uses contiguous blocks per class, DieselOBD uses leave-one-vehicle-out, EngineAD splits by truck.
- Every model reports the same metrics: F1, balanced accuracy, AUROC, AUPRC, and a confusion matrix, with the threshold chosen on the validation set (max F1; 0.5 if validation has one class).
- Tests use synthetic data only; real data is exercised by the "check on real data" steps.

## Data facts (verified 2026-09-25)

- **EngineFaultDB** `data/EngineFaultDB/EngineFaultDB_Final.csv`: 55,999 rows. Columns are `Fault` plus 14 features: `MAP, TPS, Force, Power, RPM, Consumption L/H, Consumption L/100KM, Speed, CO, HC, CO2, O2, Lambda, AFR`. `Fault` ∈ {0,1,2,3} with counts 16000/10998/15000/14001. Each class is **one contiguous block** of time-ordered rows, and neighboring rows are near-identical. Label semantics: 0 no fault, 1 rich mixture, 2 lean mixture, 3 low voltage (supported by the lambda and CO means per class).
- **DieselOBD** `data/OBD-Dataset/MASTER_TRAIN_X.xlsx` (sheet `Train`, 118,477 rows) and `MASTER_TEST_X.xlsx` (sheet `Test`, 33,838 rows). Columns: 11 PIDs `LOAD_PCT, ECT, MAP, RPM, VSS, IAT, MAF, FRP, BARO, VPWR, AAT`, then 0/1 DTC columns `P0000 P0562 P0113 P0102 P0403 P0404 P2562 P0234 P2015 P2009 P0107 P0069 P0089 P0406`, then `Mode` (0/1/2 driving mode; not used as a feature). There is **no vehicle column**. Each file has exactly 8 contiguous blocks of identical DTC signature, one per vehicle, in the same order in both files:

  | block | DTCs | vehicle (README) | label |
  |---|---|---|---|
  | 1 | P0403 P0404 P2015 P2009 | Kia Sportage | 1 |
  | 2 | P0000 | fault-free (one of three) | 0 |
  | 3 | P0113 P0102 P2562 | Citroën Berlingo | 1 |
  | 4 | (none set) | fault-free (one of three) | 0 |
  | 5 | P0562 | Peugeot 2008 | 1 |
  | 6 | P0107 P0069 P0089 | Chevrolet Captiva | 1 |
  | 7 | P0113 P0102 P0234 P0406 | Peugeot Expert (idle only, Mode 0) | 1 |
  | 8 | P0000 | fault-free (one of three) | 0 |

  Label = 1 if any DTC column other than `P0000` is set. The published train/test files share all 8 vehicles, so the loader merges them.
- **EngineAD** `data/EngineAD/truck_<k>-1.pickle`, k = 1..25, 104–489 MB each (~6.4 GB total). Each file is a pickled `list` of `DataFrame`s. Each DataFrame is one 300-step window with columns `PC1..PC8` (float64), `label` (0/1 per timestep) and `vehicle_id` (e.g. `"truck 7"`). Windows don't overlap. Truck 7 has 4,595 windows, 220 (4.8%) with any anomalous step, and 3 with mixed labels. The pickle needs only these classes: `builtins.slice`, `numpy.dtype`, `numpy.ndarray`, `numpy._core.multiarray._reconstruct`, `pandas._libs.internals._unpickle_block`, `pandas.core.frame.DataFrame`, `pandas.core.indexes.base.Index`, `pandas.core.indexes.base._new_Index`, `pandas.core.indexes.range.RangeIndex`, `pandas.core.internals.managers.BlockManager`. Only `truck_7-1.pickle` is downloaded so far.

## File Structure

```
pyproject.toml                      # package + pinned deps + pytest config
.gitignore                          # data/, artifacts/, .venv/, caches
scripts/download_data.py            # fetch all three datasets into data/
src/obdfault/__init__.py
src/obdfault/splits.py              # Fold + split helpers (no dataset knowledge)
src/obdfault/metrics.py             # threshold picking + metric dict
src/obdfault/data/__init__.py
src/obdfault/data/enginefaultdb.py  # CSV → folds (contiguous block per class)
src/obdfault/data/dieselobd.py      # 2 xlsx → vehicle blocks → leave-one-vehicle-out folds
src/obdfault/data/enginead.py       # safe unpickle → npz cache → split by truck
src/obdfault/models.py              # baselines, MLP, CNN1d, train_torch, predict_proba
src/obdfault/run.py                 # CLI: fit, threshold, metrics, write artifacts
tests/test_splits.py
tests/test_metrics.py
tests/test_enginefaultdb.py
tests/test_dieselobd.py
tests/test_enginead.py
tests/test_models.py
tests/test_run.py
```

---

### Task 1: Project scaffold, download script, and splits

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `scripts/download_data.py`, `src/obdfault/__init__.py`, `src/obdfault/data/__init__.py`, `src/obdfault/splits.py`
- Test: `tests/test_splits.py`

**Interfaces:**
- Produces:
  - `Fold` dataclass with fields `X_train, y_train, X_val, y_val, X_test, y_test, test_groups` (all `np.ndarray`; X is float32 of shape `[n, F]` or `[n, T, C]`, y is int64 0/1, test_groups is str)
  - `contiguous_split(n: int, train=0.70, val=0.15) -> tuple[np.ndarray, np.ndarray, np.ndarray]`
  - `block_split_by_class(labels: np.ndarray, train=0.70, val=0.15) -> tuple[idx, idx, idx]`
  - `group_split(groups: np.ndarray, seed: int = 0, val_frac=0.15, test_frac=0.15) -> tuple[idx, idx, idx]`
  - `leave_one_group_out(groups: np.ndarray, val_frac=0.15) -> list[tuple[idx, idx, idx]]`
  - `make_fold(X, y, groups, train, val, test) -> Fold`

- [ ] **Step 1: Write project files**

`pyproject.toml`:
```toml
[project]
name = "obdfault"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "numpy==2.5.3",
    "pandas==3.0.6",
    "scikit-learn==1.9.1",
    "torch==2.14.0",
    "openpyxl==3.1.5",
]

[project.optional-dependencies]
dev = ["pytest==9.1.1"]

[build-system]
requires = ["setuptools>=61"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

`.gitignore`:
```
/data/
/artifacts/
.venv/
__pycache__/
*.egg-info/
.pytest_cache/
```

`src/obdfault/__init__.py` and `src/obdfault/data/__init__.py`: empty files.

`scripts/download_data.py`:
```python
"""Fetch the three labeled datasets into data/. Usage: python scripts/download_data.py [--trucks N]"""
import argparse
import json
import re
import subprocess
import urllib.request
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"
REPOS = {
    "EngineFaultDB": "https://github.com/Leo-Thomas/EngineFaultDB.git",
    "OBD-Dataset": "https://github.com/AbouAbdallah-Lounis/OBD-Dataset.git",
}
ENGINEAD_API = "https://borealisdata.ca/api/datasets/:persistentId/?persistentId=doi:10.5683/SP3/TX13P1"
ENGINEAD_FILE = "https://borealisdata.ca/api/access/datafile/{}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trucks", type=int, default=25, help="EngineAD trucks to fetch, lowest numbers first (25 = all, ~6.4 GB)")
    args = parser.parse_args()
    DATA.mkdir(exist_ok=True)
    for name, url in REPOS.items():
        if not (DATA / name).exists():
            subprocess.run(["git", "clone", "--depth", "1", url, str(DATA / name)], check=True)

    with urllib.request.urlopen(ENGINEAD_API) as response:
        files = [f["dataFile"] for f in json.load(response)["data"]["latestVersion"]["files"]]
    files.sort(key=lambda d: int(re.search(r"truck_(\d+)", d["filename"]).group(1)))
    out = DATA / "EngineAD"
    out.mkdir(exist_ok=True)
    for d in files[: args.trucks]:
        dest = out / d["filename"]
        if dest.exists():
            continue
        part = dest.with_suffix(".part")
        urllib.request.urlretrieve(ENGINEAD_FILE.format(d["id"]), part)
        part.rename(dest)
        print("downloaded", dest.name)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Install the package in editable mode**

Run: `.venv/bin/pip install -q -e ".[dev]"`
Expected: exits 0.

- [ ] **Step 3: Write the failing tests**

`tests/test_splits.py`:
```python
import numpy as np
import pytest

from obdfault.splits import (
    block_split_by_class,
    contiguous_split,
    group_split,
    leave_one_group_out,
    make_fold,
)


def test_contiguous_split_is_ordered_blocks():
    train, val, test = contiguous_split(100)
    assert train.tolist() == list(range(70))
    assert val.tolist() == list(range(70, 85))
    assert test.tolist() == list(range(85, 100))


def test_block_split_by_class_keeps_each_class_in_time_order():
    labels = np.array([0] * 20 + [1] * 40)
    train, val, test = block_split_by_class(labels)
    assert sorted(np.concatenate([train, val, test]).tolist()) == list(range(60))
    assert train.tolist() == list(range(14)) + list(range(20, 48))
    assert val.tolist() == list(range(14, 17)) + list(range(48, 54))
    assert test.tolist() == list(range(17, 20)) + list(range(54, 60))


def test_group_split_has_no_group_overlap_and_is_deterministic():
    groups = np.repeat([f"g{i}" for i in range(10)], 5)
    train, val, test = group_split(groups, seed=0)
    sets = [set(groups[i]) for i in (train, val, test)]
    assert not (sets[0] & sets[1]) and not (sets[0] & sets[2]) and not (sets[1] & sets[2])
    assert len(sets[1]) == 2 and len(sets[2]) == 2
    assert sorted(np.concatenate([train, val, test]).tolist()) == list(range(50))
    assert [a.tolist() for a in group_split(groups, seed=0)] == [train.tolist(), val.tolist(), test.tolist()]


def test_group_split_needs_three_groups():
    with pytest.raises(ValueError):
        group_split(np.array(["a", "a", "b"]))


def test_leave_one_group_out_holds_out_each_group_and_uses_tails_for_val():
    groups = np.repeat(["a", "b", "c"], 20)
    folds = leave_one_group_out(groups)
    assert len(folds) == 3
    train, val, test = folds[0]
    assert set(groups[test]) == {"a"}
    assert "a" not in set(groups[train]) | set(groups[val])
    assert val.tolist() == list(range(37, 40)) + list(range(57, 60))
    assert train.tolist() == list(range(20, 37)) + list(range(40, 57))


def test_make_fold_slices_arrays():
    X = np.arange(10, dtype=np.float32).reshape(5, 2)
    y = np.array([0, 1, 0, 1, 0])
    groups = np.array(list("abcde"))
    fold = make_fold(X, y, groups, np.array([0, 1]), np.array([2]), np.array([3, 4]))
    assert fold.X_train.shape == (2, 2)
    assert fold.y_val.tolist() == [0]
    assert fold.test_groups.tolist() == ["d", "e"]
```

- [ ] **Step 4: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_splits.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'obdfault.splits'`

- [ ] **Step 5: Implement `src/obdfault/splits.py`**

```python
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
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_splits.py -q`
Expected: `6 passed`

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml .gitignore scripts/download_data.py src/obdfault/__init__.py src/obdfault/data/__init__.py src/obdfault/splits.py tests/test_splits.py
git status --short   # confirm data/, .venv/, CLAUDE.md are NOT listed as staged
git commit -m "Add obdfault package scaffold, download script, and group-safe splits"
```

---

### Task 2: Threshold and metrics

**Files:**
- Create: `src/obdfault/metrics.py`
- Test: `tests/test_metrics.py`

**Interfaces:**
- Produces:
  - `pick_threshold(y_val: np.ndarray, p_val: np.ndarray) -> float`: the threshold with max F1 on validation; 0.5 if `y_val` has one class.
  - `binary_metrics(y: np.ndarray, p: np.ndarray, pred: np.ndarray) -> dict` with keys `n, positives, f1, balanced_accuracy, auroc, auprc, tn, fp, fn, tp`. The last four are ints; `balanced_accuracy`, `auroc` and `auprc` are `None` when `y` has one class.

- [ ] **Step 1: Write the failing tests**

`tests/test_metrics.py`:
```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_metrics.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'obdfault.metrics'`

- [ ] **Step 3: Implement `src/obdfault/metrics.py`**

```python
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)


def pick_threshold(y_val, p_val):
    if len(np.unique(y_val)) < 2:
        return 0.5
    precision, recall, thresholds = precision_recall_curve(y_val, p_val)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    return float(thresholds[np.argmax(f1[:-1])])


def binary_metrics(y, p, pred):
    both = len(np.unique(y)) == 2
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "n": int(len(y)),
        "positives": int(np.sum(y)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)) if both else None,
        "auroc": float(roc_auc_score(y, p)) if both else None,
        "auprc": float(average_precision_score(y, p)) if both else None,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_metrics.py -q`
Expected: `4 passed`

- [ ] **Step 5: Commit**

```bash
git add src/obdfault/metrics.py tests/test_metrics.py
git commit -m "Add validation threshold picking and binary metrics"
```

---

### Task 3: EngineFaultDB loader

**Files:**
- Create: `src/obdfault/data/enginefaultdb.py`
- Test: `tests/test_enginefaultdb.py`

**Interfaces:**
- Consumes: `Fold`, `block_split_by_class` (Task 1)
- Produces:
  - `FEATURES: list[str]` (14 names), `FAULT_NAMES: dict[int, str]`
  - `load(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]`: `(X float32 [n,14], y int64 0/1, fault int64 0..3)`
  - `folds(root: Path) -> list[Fold]`: one fold; `test_groups` are fault names.

- [ ] **Step 1: Write the failing tests**

`tests/test_enginefaultdb.py`:
```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_enginefaultdb.py -q`
Expected: FAIL with `ImportError: cannot import name 'enginefaultdb'`

- [ ] **Step 3: Implement `src/obdfault/data/enginefaultdb.py`**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_enginefaultdb.py -q`
Expected: `3 passed`

- [ ] **Step 5: Check on real data**

Run: `.venv/bin/python -c "from obdfault.data import enginefaultdb as e; [f]=e.folds('data'); print(f.X_train.shape, f.X_val.shape, f.X_test.shape, f.y_test.mean().round(3))"`
Expected: `(39198, 14) (8399, 14) (8402, 14) 0.714`

- [ ] **Step 6: Commit**

```bash
git add src/obdfault/data/enginefaultdb.py tests/test_enginefaultdb.py
git commit -m "Add EngineFaultDB loader with contiguous per-class split"
```

---

### Task 4: DieselOBD loader (leave-one-vehicle-out)

**Files:**
- Create: `src/obdfault/data/dieselobd.py`
- Test: `tests/test_dieselobd.py`

**Interfaces:**
- Consumes: `leave_one_group_out`, `make_fold` (Task 1)
- Produces:
  - `PIDS: list[str]` (11 names), `N_VEHICLES = 8`, `FILES = ["MASTER_TRAIN_X.xlsx", "MASTER_TEST_X.xlsx"]`
  - `load_file(path: Path) -> tuple[X float32 [n,11], y int64, block int64 1..8]`
  - `load(root: Path) -> tuple[X, y, vehicle str array like "vehicle_1"]`
  - `folds(root: Path) -> list[Fold]`: 8 folds; each fold's `test_groups` is a single vehicle.

- [ ] **Step 1: Write the failing tests**

`tests/test_dieselobd.py`:
```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_dieselobd.py -q`
Expected: FAIL with `ImportError: cannot import name 'dieselobd'`

- [ ] **Step 3: Implement `src/obdfault/data/dieselobd.py`**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_dieselobd.py -q`
Expected: `3 passed`

- [ ] **Step 5: Check on real data**

Run: `.venv/bin/python -c "from obdfault.data import dieselobd as d; import numpy as np; X,y,v=d.load('data'); print(X.shape); [print(n,(v==n).sum(),y[v==n][0]) for n in np.unique(v)]"`
Expected: `(152315, 11)`, then 8 lines whose labels are, for vehicle_1..8: 1,0,1,0,1,1,1,0, with row counts 34845, 20126, 25124, 5487, 20203, 19948, 5094, 21488.

- [ ] **Step 6: Commit**

```bash
git add src/obdfault/data/dieselobd.py tests/test_dieselobd.py
git commit -m "Add DieselOBD loader with vehicle recovery and leave-one-vehicle-out folds"
```

---

### Task 5: EngineAD loader (safe unpickle + cache)

**Files:**
- Create: `src/obdfault/data/enginead.py`
- Test: `tests/test_enginead.py`

**Interfaces:**
- Consumes: `group_split`, `make_fold` (Task 1)
- Produces:
  - `CHANNELS = ["PC1", ..., "PC8"]`
  - `read_truck(path: Path) -> tuple[X float32 [n,300,8], y int64]` (window label = 1 if any step is anomalous)
  - `load(root: Path) -> tuple[X, y, truck str array like "truck_7-1"]` (caches each truck as `data/EngineAD/cache/<stem>.npz`)
  - `folds(root: Path) -> list[Fold]`: one fold, split by truck with `group_split(seed=0)`.

- [ ] **Step 1: Write the failing tests**

`tests/test_enginead.py`:
```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_enginead.py -q`
Expected: FAIL with `ImportError: cannot import name 'enginead'`

- [ ] **Step 3: Implement `src/obdfault/data/enginead.py`**

```python
import pickle
from pathlib import Path

import numpy as np

from obdfault.splits import group_split, make_fold

CHANNELS = [f"PC{i}" for i in range(1, 9)]
# Loading a pickle can run arbitrary code, so only the classes these files actually use are allowed.
ALLOWED = {
    ("builtins", "slice"),
    ("numpy", "dtype"),
    ("numpy", "ndarray"),
    ("numpy._core.multiarray", "_reconstruct"),
    ("numpy.core.multiarray", "_reconstruct"),
    ("pandas._libs.internals", "_unpickle_block"),
    ("pandas.core.frame", "DataFrame"),
    ("pandas.core.indexes.base", "Index"),
    ("pandas.core.indexes.base", "_new_Index"),
    ("pandas.core.indexes.range", "RangeIndex"),
    ("pandas.core.internals.managers", "BlockManager"),
}


class _SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) not in ALLOWED:
            raise pickle.UnpicklingError(f"blocked {module}.{name}")
        return super().find_class(module, name)


def read_truck(path):
    with open(path, "rb") as f:
        windows = _SafeUnpickler(f).load()
    X = np.stack([w[CHANNELS].to_numpy(np.float32) for w in windows])
    y = np.array([int(w["label"].max()) for w in windows], dtype=np.int64)
    return X, y


def load(root):
    folder = Path(root) / "EngineAD"
    paths = sorted(folder.glob("truck_*.pickle"))
    if len(paths) < 3:
        raise FileNotFoundError(f"need at least 3 EngineAD trucks in {folder}; run scripts/download_data.py")
    cache = folder / "cache"
    cache.mkdir(exist_ok=True)
    Xs, ys, trucks = [], [], []
    for path in paths:
        cached = cache / f"{path.stem}.npz"
        if not cached.exists():
            X, y = read_truck(path)
            np.savez(cached, X=X, y=y)
        with np.load(cached) as d:
            Xs.append(d["X"])
            ys.append(d["y"])
        trucks.append(np.full(len(ys[-1]), path.stem))
    return np.concatenate(Xs), np.concatenate(ys), np.concatenate(trucks)


def folds(root):
    X, y, truck = load(root)
    return [make_fold(X, y, truck, *group_split(truck, seed=0))]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_enginead.py -q`
Expected: `5 passed`

- [ ] **Step 5: Check the real file loads through the safe unpickler**

Run: `.venv/bin/python -c "from obdfault.data import enginead as e; X,y=e.read_truck(__import__('pathlib').Path('data/EngineAD/truck_7-1.pickle')); print(X.shape, int(y.sum()))"`
Expected: `(4595, 300, 8) 220`

- [ ] **Step 6: Commit**

```bash
git add src/obdfault/data/enginead.py tests/test_enginead.py
git commit -m "Add EngineAD loader with restricted unpickler, npz cache, and truck split"
```

---

### Task 6: Models and training loop

**Files:**
- Create: `src/obdfault/models.py`
- Test: `tests/test_models.py`

**Interfaces:**
- Consumes: `Fold` (Task 1)
- Produces:
  - `window_features(X: [n,T,C]) -> [n,4C]` (per-channel mean, std, min, max)
  - `baseline(name: "logreg" | "gbm", seed: int)`: an unfitted sklearn classifier with `fit` / `predict_proba`
  - `MLP(X_train, hidden=(64, 32), dropout=0.1)`: `forward([n,F]) -> logits [n]`
  - `CNN1d(X_train, channels=32, dropout=0.1)`: `forward([n,T,C]) -> logits [n]`
  - `train_torch(model, fold, *, epochs=30, batch_size=512, lr=1e-3, patience=5, seed=0) -> list[float]` (val loss per epoch; restores the best weights)
  - `predict_proba(model, X, batch_size=1024) -> np.ndarray [n]`
  - Both torch models standardize inputs internally with training-set mean/std (stored as buffers), so the caller never copies the large EngineAD arrays.

- [ ] **Step 1: Write the failing tests**

`tests/test_models.py`:
```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_models.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'obdfault.models'`

- [ ] **Step 3: Implement `src/obdfault/models.py`**

```python
import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn


def window_features(X):
    return np.concatenate([X.mean(1), X.std(1), X.min(1), X.max(1)], axis=1)


def baseline(name, seed):
    if name == "logreg":
        return make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=1000))
    if name == "gbm":
        return HistGradientBoostingClassifier(class_weight="balanced", random_state=seed)
    raise ValueError(f"unknown baseline {name!r}")


class Standardize(nn.Module):
    def __init__(self, X_train):
        super().__init__()
        flat = torch.as_tensor(X_train.reshape(-1, X_train.shape[-1]), dtype=torch.float32)
        self.register_buffer("mean", flat.mean(0))
        self.register_buffer("std", flat.std(0).clamp_min(1e-6))

    def forward(self, x):
        return (x - self.mean) / self.std


class MLP(nn.Module):
    def __init__(self, X_train, hidden=(64, 32), dropout=0.1):
        super().__init__()
        layers, width = [Standardize(X_train)], X_train.shape[1]
        for h in hidden:
            layers += [nn.Linear(width, h), nn.ReLU(), nn.Dropout(dropout)]
            width = h
        layers.append(nn.Linear(width, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class CNN1d(nn.Module):
    def __init__(self, X_train, channels=32, dropout=0.1):
        super().__init__()
        self.norm = Standardize(X_train)
        self.conv = nn.Sequential(
            nn.Conv1d(X_train.shape[-1], channels, 7, padding=3), nn.ReLU(),
            nn.Conv1d(channels, channels, 7, padding=3, stride=2), nn.ReLU(),
            nn.Conv1d(channels, channels, 7, padding=3, stride=2), nn.ReLU(),
            nn.AdaptiveMaxPool1d(1),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(channels, 1))

    def forward(self, x):
        return self.head(self.conv(self.norm(x).transpose(1, 2))).squeeze(-1)


def _device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _batches(X, y, batch_size, device, order=None):
    order = np.arange(len(X)) if order is None else order
    for i in range(0, len(order), batch_size):
        idx = order[i:i + batch_size]
        yield (torch.as_tensor(X[idx], dtype=torch.float32, device=device),
               torch.as_tensor(y[idx], dtype=torch.float32, device=device))


def train_torch(model, fold, *, epochs=30, batch_size=512, lr=1e-3, patience=5, seed=0):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    device = _device()
    model.to(device)
    pos = int(fold.y_train.sum())
    pos_weight = torch.tensor((len(fold.y_train) - pos) / max(pos, 1), dtype=torch.float32, device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    best, best_state, stale, history = float("inf"), None, 0, []
    for _ in range(epochs):
        model.train()
        for xb, yb in _batches(fold.X_train, fold.y_train, batch_size, device, rng.permutation(len(fold.X_train))):
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            total = sum(loss_fn(model(xb), yb).item() * len(xb)
                        for xb, yb in _batches(fold.X_val, fold.y_val, batch_size, device))
        history.append(total / len(fold.X_val))
        if history[-1] < best - 1e-4:
            best, stale = history[-1], 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= patience:
                break
    model.load_state_dict(best_state)
    return history


def predict_proba(model, X, batch_size=1024):
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        return np.concatenate([
            torch.sigmoid(model(torch.as_tensor(X[i:i + batch_size], dtype=torch.float32, device=device))).cpu().numpy()
            for i in range(0, len(X), batch_size)
        ])
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_models.py -q`
Expected: `5 passed`

- [ ] **Step 5: Commit**

```bash
git add src/obdfault/models.py tests/test_models.py
git commit -m "Add baseline, MLP, and 1D-CNN fault classifiers with class-weighted training"
```

---

### Task 7: Run CLI and artifacts

**Files:**
- Create: `src/obdfault/run.py`
- Test: `tests/test_run.py`

**Interfaces:**
- Consumes: `enginefaultdb.folds`, `dieselobd.folds`, `enginead.folds` (Tasks 3–5); `pick_threshold`, `binary_metrics` (Task 2); `baseline`, `window_features`, `MLP`, `CNN1d`, `train_torch`, `predict_proba` (Task 6)
- Produces:
  - `DATASETS: dict[str, Callable[[Path], list[Fold]]]` with keys `enginefaultdb`, `dieselobd`, `enginead`
  - `MODELS = ("logreg", "gbm", "nn")`: `nn` means MLP for tabular folds and CNN1d for window folds
  - `run(dataset: str, model: str, data_root="data", out_root="artifacts", seed=0) -> dict`: writes `artifacts/<dataset>/<model>/metrics.json` and `predictions.csv` (columns `fold, group, y, p, pred`)
  - CLI: `python -m obdfault.run --dataset <name> --model <name> [--data data] [--out artifacts] [--seed 0]`

- [ ] **Step 1: Write the failing tests**

`tests/test_run.py`:
```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_run.py -q`
Expected: FAIL with `ImportError: cannot import name 'run'`

- [ ] **Step 3: Implement `src/obdfault/run.py`**

```python
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_run.py -q`
Expected: `4 passed`

- [ ] **Step 5: Run the full suite**

Run: `.venv/bin/pytest -q`
Expected: `30 passed`

- [ ] **Step 6: Commit**

```bash
git add src/obdfault/run.py tests/test_run.py
git commit -m "Add run CLI that writes metrics and predictions per dataset and model"
```

---

### Task 8: Real runs and results table

**Files:**
- Modify: `README.md` (add Setup, Usage, Results sections)

**Interfaces:**
- Consumes: the `python -m obdfault.run` CLI (Task 7) and `scripts/download_data.py` (Task 1)

- [ ] **Step 1: Download the remaining EngineAD trucks**

Ask the user before running this: it downloads ~6.3 GB more. If they want a smaller first run, use `--trucks 8`.
Run: `.venv/bin/python scripts/download_data.py`
Expected: `downloaded truck_<k>-1.pickle` lines. `truck_7-1.pickle` is skipped because it already exists.

- [ ] **Step 2: Run all nine combinations**

```bash
for d in enginefaultdb dieselobd enginead; do
  for m in logreg gbm nn; do
    .venv/bin/python -m obdfault.run --dataset "$d" --model "$m"
  done
done
```
Expected: nine summary lines, and `artifacts/<dataset>/<model>/metrics.json` for each. EngineAD's first run builds the npz cache, which is slow. DieselOBD `nn` trains 8 folds.

- [ ] **Step 3: Sanity-check the results before writing them down**

- EngineFaultDB: a near-perfect score from every model is plausible (a lab rig with strong gas-analyzer signals). But if `logreg` AUROC is ≥ 0.999, re-check that `block_split_by_class` is used and not a random split.
- DieselOBD: open `artifacts/dieselobd/*/metrics.json` → `per_group`. Each vehicle should have `fault_rate` 0 or 1. Look at which vehicles are misclassified; vehicle_7 (Peugeot Expert) only has idle data.
- EngineAD: `positives` in the test set must be > 0. If it is 0, the held-out trucks had no anomalies; report it and don't tune around it.

- [ ] **Step 4: Write `README.md`**

Replace the file with the following, filling the results table from the nine `metrics.json` files (values rounded to 3 decimals):

````markdown
# Deep-Learning-Under-the-Hood

Binary engine-fault classifiers ("fault: yes/no") on three public datasets. Design: `OBD2-Predictive-Maintenance-Design.md`.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/python scripts/download_data.py        # add --trucks 8 for a smaller EngineAD download
.venv/bin/pytest -q
```

## Usage

```bash
.venv/bin/python -m obdfault.run --dataset {enginefaultdb,dieselobd,enginead} --model {logreg,gbm,nn}
```

Writes `artifacts/<dataset>/<model>/metrics.json` and `predictions.csv`.

## Evaluation protocol

| Dataset | Split | Why |
|---|---|---|
| EngineFaultDB | per fault class, contiguous 70/15/15 blocks | rows are time-ordered; random splits leak near-duplicate neighbors |
| DieselOBD | leave-one-vehicle-out (8 folds), val = last 15% of each training vehicle | labels are per vehicle; the published train/test files share all vehicles |
| EngineAD | by truck, 70/15/15, seed 0 | windows from one truck are correlated |

Threshold: max F1 on validation. Metrics are pooled over folds.

## Results

| Dataset | Model | F1 | Balanced acc. | AUROC | AUPRC |
|---|---|---|---|---|---|
| EngineFaultDB | logreg | | | | |
| EngineFaultDB | gbm | | | | |
| EngineFaultDB | nn (MLP) | | | | |
| DieselOBD | logreg | | | | |
| DieselOBD | gbm | | | | |
| DieselOBD | nn (MLP) | | | | |
| EngineAD | logreg | | | | |
| EngineAD | gbm | | | | |
| EngineAD | nn (1D CNN) | | | | |

## Limitations

- DieselOBD has 8 vehicles, so each fold's test set is one car; results say more about vehicle-level separability than about fault detection in general.
- EngineAD inputs are PCA components and can't be mapped back to OBD-II signals.
- EngineFaultDB is a lab rig; most features (gas analyzer, dynamometer) won't exist on a real car.
````

- [ ] **Step 5: Commit**

```bash
git add README.md
git commit -m "Add setup, usage, and results for public-data fault classifiers"
```

Stop here and report the results table to the user. Do not push.

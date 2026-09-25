import pickle
from pathlib import Path

import numpy as np

from obdfault.splits import group_split, make_fold

CHANNELS = [f"PC{i}" for i in range(1, 9)]
# Loading a pickle can run arbitrary code, so only DataFrame-building classes are allowed:
# the dataset's own layout (older pandas) plus the layout pandas 3 writes.
ALLOWED = {
    ("builtins", "slice"),
    ("numpy", "dtype"),
    ("numpy", "ndarray"),
    ("numpy._core.multiarray", "_reconstruct"),
    ("numpy.core.multiarray", "_reconstruct"),
    ("numpy._core.numeric", "_frombuffer"),
    ("pandas", "DataFrame"),
    ("pandas", "Index"),
    ("pandas", "RangeIndex"),
    ("pandas", "StringDtype"),
    ("pandas.arrays", "StringArray"),
    ("pandas._libs.arrays", "__pyx_unpickle_NDArrayBacked"),
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

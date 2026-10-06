"""Refactor safety net: every pipeline stage must give the same output as Hoda's original scripts (bca09e1).
Regenerate golden.pkl only from the ORIGINAL code:  python tests/test_golden.py
Deleted after the refactor (later model changes are supposed to change these numbers).

If this fails on your machine BEFORE you change anything, the reference file is the likely cause: golden.pkl was
made on Python 3.14 / pandas 3 / Apple Silicon; older pandas (Python 3.10) can't read it and other CPUs can round
differently. Rebuild it from the original code on your machine (don't commit the rebuilt file):
    git worktree add ../orig d4aa878
    (cd ../orig && python tests/test_golden.py)
    cp ../orig/tests/golden.pkl tests/ && pytest
    git worktree remove ../orig && git checkout -- tests/golden.pkl
"""
import os, tempfile
import numpy as np, pandas as pd, torch
from synthdata import EVAL_ARGS, SYN_ARGS, make_trips, write_kaggle_csvs

# --- the code under test (the refactor changes only this block) ---
from types import SimpleNamespace
from obdfault import config, data, features, model, train, scoring, faults, evaluate
from obdfault import synthetic as S
# ponytail: one namespace with every module's names, so compute() calls stay exactly as they were before the split
M = SimpleNamespace(**{k: v for mod in (config, data, features, model, train, scoring, faults, evaluate)
                       for k, v in vars(mod).items() if not k.startswith("__")})
# ------------------------------------------------------------------

GOLDEN = os.path.join(os.path.dirname(__file__), "golden.pkl")


def compute(folder):
    M.PHYSICS.clear()                                   # final model: no physics inputs (main() does the same)
    torch.set_num_threads(1); torch.manual_seed(0); np.random.seed(0)
    write_kaggle_csvs(folder)
    out = {f"kaggle_{i}_{k}": t[k] for i, t in enumerate(M.load_kaggle(folder)) for k in ("unit", "order", "X", "ctx", "code")}
    trips = make_trips()
    R, summ = M.evaluate_car("car2", trips, EVAL_ARGS)
    out.update(faults=R, summary=summ, binary=M.binary_report(R))
    torch.manual_seed(0)
    rows, key = S.test_car("car2", trips, SYN_ARGS, np.random.RandomState(42))
    rows, key = S.decide(rows, key, 5, 0.3, S.FINAL_VARIANT)
    minute, drive = S.scores(rows, key)
    out.update(syn_rows=rows, syn_key=key, syn_scores=pd.DataFrame([minute, drive]))
    return out


def test_same_as_original(tmp_path):
    got, want = compute(str(tmp_path)), pd.read_pickle(GOLDEN)
    assert got.keys() == want.keys()
    for k, w in want.items():
        g = got[k]
        if isinstance(w, pd.DataFrame): pd.testing.assert_frame_equal(g, w, check_exact=False, rtol=1e-5, atol=1e-6, obj=k)
        elif isinstance(w, np.ndarray): np.testing.assert_allclose(g, w, rtol=1e-5, atol=1e-6, equal_nan=True, err_msg=k)
        else: assert g == w, k


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as d: pd.to_pickle(compute(d), GOLDEN)
    print("wrote", GOLDEN)

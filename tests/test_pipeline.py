"""Smoke tests on made-up data: they fail if the pipeline breaks, not when its numbers improve."""
import numpy as np, torch
from synthdata import EVAL_ARGS, SYN_ARGS, make_trips, write_kaggle_csvs
from obdfault.config import CH
from obdfault.data import load_kaggle
from obdfault.features import PHYSICS
from obdfault.evaluate import evaluate_car
from obdfault.synthetic import FINAL_VARIANT, decide, test_car as run_synthetic_car   # renamed so pytest doesn't collect it


def test_load_kaggle_reads_the_raw_format(tmp_path):
    write_kaggle_csvs(str(tmp_path))
    trips = load_kaggle(str(tmp_path))
    assert [t["unit"] for t in trips] == ["A", "A", "B", "exp2_car", "exp2_car", "exp3_car"]   # gaps / runtime resets split drives
    assert trips[0]["X"].shape == (40, len(CH))
    assert not np.isnan(trips[0]["X"]).all(0).any()          # every unit format ('2124RPM', '48,60%', ...) was parsed
    assert np.nanmax(trips[0]["X"][:, CH.index("RPM")]) <= 8000   # the 9999 RPM glitch was dropped
    assert trips[2]["code"].any() and not trips[0]["code"].any()


def test_pipeline_runs_end_to_end():
    PHYSICS.clear(); torch.manual_seed(0)
    trips = make_trips()
    R, summary = evaluate_car("car2", trips, EVAL_ARGS)
    assert set(R.method) == {"pretrained only", "fine-tuned (ours)", "ridge baseline"}
    assert R.AUROC.notna().any() and R.AUROC.dropna().between(0, 1).all()
    assert summary.false_alarm_rate.between(0, 1).all()
    rows, key = run_synthetic_car("car2", trips, SYN_ARGS, np.random.RandomState(42))
    _, key = decide(rows, key, 5, 0.3, FINAL_VARIANT)
    assert set(key.model_says) <= {"HEALTHY", "FAULT", "TOO SHORT"}

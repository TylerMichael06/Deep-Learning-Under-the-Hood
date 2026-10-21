"""Smoke tests on made-up data: they fail if the pipeline breaks, not when its numbers improve."""
import numpy as np, torch
from synthdata import EVAL_ARGS, SYN_ARGS, make_trips, write_kaggle_csvs
from obdfault.config import CH
from obdfault.data import load_kaggle
from obdfault.features import PHYSICS
from obdfault.evaluate import evaluate_car, binary_report
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
    assert set(R.scoring) == {"shared cutoff", "stuck check", "two-speed + stuck check", "stuck check + bias"}
    assert R.AUROC.notna().any() and R.AUROC.dropna().between(0, 1).all()
    assert summary.false_alarm_rate.between(0, 1).all()
    plain = R[R.scoring != "stuck check + bias"]               # no bias part: confirming bias alarms changes nothing
    assert (plain.scoped_TP == plain.drive_TP).all() and (plain.scoped_FP == plain.drive_FP).all()
    assert (R.confirmed_TP <= R.drive_TP).all() and (R.confirmed_FP <= R.drive_FP).all()
    levels = set(binary_report(R).level)
    assert {"drive (>30%), bias alarms confirmed on 2 of last 3 drives",
            "drive (>30%), all alarms confirmed (reported only) on 2 of last 3 drives"} <= levels
    rows, key = run_synthetic_car("car2", trips, SYN_ARGS, np.random.RandomState(42))
    _, key = decide(rows, key, 5, 0.3, FINAL_VARIANT)
    assert set(key.model_says) <= {"HEALTHY", "FAULT", "TOO SHORT"}
    _, key_bias = decide(rows, key, 5, 0.3, "bias")
    assert set(key_bias.model_says) <= {"HEALTHY", "FAULT", "TOO SHORT"}

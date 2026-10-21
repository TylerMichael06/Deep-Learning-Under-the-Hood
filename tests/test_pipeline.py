"""Smoke tests on made-up data: they fail if the pipeline breaks, not when its numbers improve."""
import numpy as np, torch
from synthdata import EVAL_ARGS, SYN_ARGS, make_trips, write_kaggle_csvs
from obdfault.config import CH
from obdfault.data import load_kaggle, load_kit
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


def test_load_kit_reads_the_zip(tmp_path):
    import zipfile
    head = ("Time,Engine Coolant Temperature [\u00b0C],Intake Manifold Absolute Pressure [kPa],Engine RPM [RPM],"
            "Vehicle Speed Sensor [km/h],Intake Air Temperature [\u00b0C],Air Flow Rate from Mass Flow Sensor [g/s],"
            "Absolute Throttle Position [%],Ambient Air Temperature [\u00b0C],Accelerator Pedal Position D [%],"
            "Accelerator Pedal Position E [%]")
    def drive(start_s, n=600):                                 # one signal per row, like the app logs them
        rows = []
        for i in range(n):
            t = (start_s + i) % 86400; v = [""] * 10; v[i % 7] = str(50 + i % 30)
            rows.append(f"{int(t // 3600):02d}:{int(t % 3600 // 60):02d}:{t % 60:06.3f}," + ",".join(v))
        return "\n".join([head] + rows)
    path = tmp_path / "kit.zip"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("OBD-II-Dataset/2017-07-05_Seat_Leon_KA_KA_Normal.csv", drive(86400 - 300).encode("latin-1"))  # past midnight
        z.writestr("OBD-II-Dataset/2017-07-05_Seat_Leon_A_B_Frei.csv", drive(8 * 3600).encode("latin-1"))
    trips = load_kit(str(path))
    assert len(trips) == 2 and all(t["unit"] == "kit_leon" and t["test_only"] for t in trips)
    assert all(len(t["X"]) == 120 for t in trips)              # 600 s on a 5 s grid: midnight did not cut the drive
    have = ~np.isnan(trips[0]["X"]).all(0)
    assert [c for c, h in zip(CH, have) if h] == ["RPM", "SPEED", "THROTTLE", "ECT", "IAT", "MAP", "MAF"]
    assert trips[1]["order"] - trips[0]["order"] == 86400 - 300 - 8 * 3600   # order = date + start time of day


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

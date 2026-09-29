"""Synthetic fault test for the final model (obd_model.py), with an answer key.

For each held-out car: the model is pretrained WITHOUT that car, fine-tuned on its first 40% of driving and
calibrated on the next 20% (same steps as obd_model.py). Its last 40% of drives are then used as the test:
~30% are left clean, the rest get one designed fault. A fault can start partway through the drive, so the true
label is known minute by minute. The model's HEALTHY/FAULT labels are compared with that answer key.

Final settings (same as obd_model.py): the 2-minute score plus the stuck-sensor check, Medium cutoff (95th
percentile of calibration scores), no alarms in the first 5 minutes of a drive, and a drive is FAULT if more than
30% of its minutes are flagged. The other scorings (before the fixes, two-speed alarm) are printed for comparison.

Usage (from the Results folder):
  python synthetic_test.py --data "..\\Datasets\\OBD-II datasets"
Writes synthetic_results.csv, synthetic_answer_key.csv and synthetic_minutes.csv next to this script (figures: make_figures.py).
"""
import argparse, os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import obd_model as M

HERE = os.path.dirname(os.path.abspath(__file__))
STEPS_PER_MIN = 60 // M.DT

# name, sensor, kind, size, start minute (0 = from the start of the drive)
SCENARIOS = [
    ("MAF reads 15% low",                    "MAF",      "scale",  0.85, 0),
    ("Coolant sensor stuck from minute 10",  "ECT",      "stuck",  None, 10),
    ("MAP reads +8 kPa from minute 5",       "MAP",      "offset", 8,    5),
    ("Fuel trim +8% (vacuum leak)",          "STFT1",    "offset", 8,    0),
    ("Throttle signal noisy (+/-5%)",        "THROTTLE", "noise",  5,    0),
    ("Intake air temp drifts up to +20 C",   "IAT",      "drift",  20,   0),
    # realistic versions of the air faults: the engine computer reacts, so load and fuel trim change too (see obd_model.py)
    ("MAF reads 15% low, engine reacts",           "MAF", "maf_reacts", 0.85, 0),
    ("MAP reads +8 kPa from minute 5, engine reacts", "MAP", "map_reacts", 8,  5),
]

def apply_fault(X, sensor, kind, size, start, rng):
    """Returns the faulty copy of one drive and the per-row truth (True = sensor is faulty at this row)."""
    j = M.CH.index(sensor); X = X.copy(); v = X[:, j].copy(); n = len(v)
    on = (np.arange(n) >= start) & ~np.isnan(v)
    if kind in ("maf_reacts", "map_reacts"):              # several sensors change, from `start` onwards
        react = (M.maf_reacts if kind == "maf_reacts" else M.map_reacts)(size)
        X[start:] = react(X[start:])
        return X, on
    if kind == "scale":  v[on] = v[on] * size
    if kind == "offset": v[on] = v[on] + size
    if kind == "noise":  v[on] = v[on] + rng.normal(0, size, on.sum())
    if kind == "stuck":  v[on] = v[on][0]
    if kind == "drift":  v[on] = v[on] + size * np.clip((np.arange(n)[on] - start) / max(n - start, 1), 0, 1)
    X[:, j] = v
    return X, on

def usable(tr, sensor, start):
    """The drive must have the sensor and at least 5 minutes of it after the fault starts."""
    v = tr["X"][start:, M.CH.index(sensor)]
    return len(tr["X"]) >= start + 5 * STEPS_PER_MIN and (~np.isnan(v)).sum() >= 5 * STEPS_PER_MIN

def test_car(unit, trips, a, rng):
    P = M.prepare_car(unit, trips, a)
    if P is None: print(f"{unit}: skipped (not enough data)"); return None, None
    # --- build the answer key: each test drive is clean or gets one fault
    test, key = [], []
    for k, tr in enumerate(P["test_trips"]):
        options = [s for s in SCENARIOS if usable(tr, s[1], s[4] * STEPS_PER_MIN)]
        if rng.rand() < a.clean_share or not options:
            test.append(dict(tr, truth=np.zeros(len(tr["X"]), bool))); key.append(dict(drive=k, fault="clean", sensor="", start_min=np.nan))
            continue
        name, sensor, kind, size, start_min = options[rng.randint(len(options))]
        X, truth = apply_fault(tr["X"], sensor, kind, size, start_min * STEPS_PER_MIN, rng)
        test.append(dict(tr, X=X, truth=truth)); key.append(dict(drive=k, fault=name, sensor=sensor, start_min=start_min))
    # --- score with the final settings. With a warm-up grace period of g minutes, the first g minutes of every drive
    #     never raise an alarm, and they are also left out when the alarm level is set from calibration drives.
    D = M.windows(test, P["mu"], P["sd"], stride=2); Dc = P["Dcal"]
    E_cal, E = M.channel_errors(P["tuned"], Dc), M.channel_errors(P["tuned"], D)
    cal_min = (Dc["start"] + M.W) / STEPS_PER_MIN
    # current: score averaged over the last 2 minutes
    score, s_cal = M.calibrate(E_cal, Dc["trip"], a.smooth)
    s, who = score(E, D["trip"]); s_cal, _ = score(E_cal, Dc["trip"])
    # two-speed: also a slow score averaged over the last 10 minutes; each rescaled on calibration, then the larger wins
    score_l, _ = M.calibrate(E_cal, Dc["trip"], a.smooth_long)
    sl, who_l = score_l(E, D["trip"]); sl_cal, _ = score_l(E_cal, Dc["trip"])
    warm = cal_min >= max(a.graces)
    n_s, n_l = M.unit_scale(s_cal[warm], s), M.unit_scale(sl_cal[warm], sl)
    n_s_cal, n_l_cal = M.unit_scale(s_cal[warm], s_cal), M.unit_scale(sl_cal[warm], sl_cal)
    two, two_cal = np.maximum(n_s, n_l), np.maximum(n_s_cal, n_l_cal)
    who_two = np.where(n_l > n_s, who_l, who)
    # stuck check: minutes a slow sensor has been frozen, relative to the longest freeze seen on this car's calibration drives
    fl, fl_cal = M.flat_minutes(test, D), M.flat_minutes(P["cal_trips"], Dc)
    longest = np.maximum(fl_cal.max(0), a.min_flat)
    n_f, n_f_cal = (fl / longest).max(1), (fl_cal / longest).max(1)
    stuck, stuck_cal = np.maximum(two, n_f), np.maximum(two_cal, n_f_cal)
    flat_who = np.array([M.CH.index(c) for c in M.FLAT_SENSORS])[(fl / longest).argmax(1)]
    who_stuck = np.where(n_f > two, flat_who, who_two)
    # stuck check on its own (no slow score): the usual 2-minute score or the stuck check, whichever is larger
    only, only_cal = np.maximum(n_s, n_f), np.maximum(n_s_cal, n_f_cal)
    who_only = np.where(n_f > n_s, flat_who, who)
    truth = np.array([test[t]["truth"][st:st + M.W].mean() >= 0.5 for t, st in zip(D["trip"], D["start"])])
    name = lambda w: [M.CH[i] if i >= 0 else "" for i in w]
    rows = pd.DataFrame(dict(car=unit, drive=D["trip"], minute=(D["start"] + M.W) / STEPS_PER_MIN, truth=truth,
                             score=s, blamed=name(who), score_two=two, blamed_two=name(who_two),
                             score_stuck=stuck, blamed_stuck=name(who_stuck), score_only=only, blamed_only=name(who_only)))
    for g in a.graces:                                  # alarm lines, one per variant and warm-up length
        ok = cal_min >= g
        rows[f"thr_{g}"] = np.quantile(s_cal[ok], a.cutoff)
        rows[f"thr_two_{g}"] = np.quantile(two_cal[ok], a.cutoff)
        rows[f"thr_stuck_{g}"] = np.quantile(stuck_cal[ok], a.cutoff)
        rows[f"thr_only_{g}"] = np.quantile(only_cal[ok], a.cutoff)
    key = pd.DataFrame(key).assign(car=unit, truth=lambda k: np.where(k.fault == "clean", "HEALTHY", "FAULT"))
    print(f"{unit}: {len(key)} test drives ({(key.fault != 'clean').sum()} with a fault)", flush=True)
    return rows, key

VARIANTS = {"current (2-min score)": "", "two-speed (2 + 10 min)": "two", "two-speed + stuck check": "stuck",
            "stuck check (no two-speed)": "only"}
FINAL_VARIANT = "only"      # final model: 2-minute score + stuck-sensor check

def decide(rows, key, g, rule, variant=""):
    """Apply the alarm with a g-minute warm-up grace period; fill in the model's verdict for every drive."""
    sfx = f"_{variant}" if variant else ""
    rows = rows.assign(score=rows["score" + sfx], blamed=rows["blamed" + sfx], threshold=rows[f"thr{sfx}_{g}"])
    rows["flagged"] = (rows.score > rows.threshold) & (rows.minute >= g)
    key = key.copy(); ids = list(zip(key.car, key.drive))
    share = rows.groupby(["car", "drive"]).flagged.mean()
    key["model_says"] = ["FAULT" if share.get(i, 0) > rule else "HEALTHY" for i in ids]
    hit = rows[rows.flagged & rows.truth].groupby(["car", "drive"])
    blamed, first = hit.blamed.agg(lambda b: b.value_counts().index[0]), hit.minute.min()   # first correct alarm
    key["model_blames"] = [blamed.get(i, "") for i in ids]
    key["minutes_to_alarm"] = [first.get(i, np.nan) for i in ids] - key.start_min.fillna(0)
    return rows, key

def metrics(truth, pred):
    tp, fn = int((pred & truth).sum()), int((~pred & truth).sum()); fp, tn = int((pred & ~truth).sum()), int((~pred & ~truth).sum())
    p = tp / (tp + fp) if tp + fp else np.nan; r = tp / (tp + fn) if tp + fn else np.nan
    return dict(TP=tp, FN=fn, FP=fp, TN=tn, accuracy=(tp + tn) / (tp + fn + fp + tn), precision=p, recall=r,
                F1=2 * p * r / (p + r) if p + r else np.nan, false_alarm_rate=fp / (fp + tn) if fp + tn else np.nan)

def timelines(rows, key, path, grace=0, n=4):
    """Plot a few drives: model score over time, the alarm line, and when the fault really started."""
    length = rows.groupby(["car", "drive"]).minute.max()
    key = key.assign(length=[length.get((c, d), 0) for c, d in zip(key.car, key.drive)])
    faulty = key[key.fault != "clean"].sort_values(["model_says", "start_min", "length"], ascending=[True, False, False])
    faulty = faulty.drop_duplicates("fault").head(n - 1)                       # different faults, caught ones first
    picks = pd.concat([faulty, key[key.fault == "clean"].sort_values("length", ascending=False).head(1)])
    fig, axes = plt.subplots(len(picks), 1, figsize=(10, 2.6 * len(picks)), squeeze=False)
    for ax, (_, k) in zip(axes[:, 0], picks.iterrows()):
        r = rows[(rows.car == k.car) & (rows.drive == k.drive)]
        ax.plot(r.minute, r.score, color="#2a6fdb", lw=1.5, label="model score")
        ax.axhline(r.threshold.iloc[0], color="#d62728", ls="--", lw=1, label="alarm line")
        ax.fill_between(r.minute, r.score.min(), r.score.max(), where=r.truth, color="#ff9896", alpha=.3, label="fault really present")
        ax.axvspan(r.minute.min(), grace, color="#bbbbbb", alpha=.35, label="warm-up (no alarms)") if grace else None
        ax.scatter(r.minute[r.flagged], r.score[r.flagged], s=10, color="#d62728", zorder=3, label="FAULT minute")
        ax.set_title(f"{k.car}, drive {k.drive}: {k.fault}   ->  model says {k.model_says}"
                     + (f", blames {k.model_blames}" if k.model_blames else ""), fontsize=9)
        ax.set_xlabel("minutes into drive"); ax.set_ylabel("score")
    axes[0, 0].legend(fontsize=7, loc="upper left"); fig.tight_layout()
    try: fig.savefig(path, dpi=130)
    except OSError:                                    # the old image is open in a viewer -> save next to it
        path = path.replace(".png", "_new.png"); fig.savefig(path, dpi=130)
    plt.close(fig); print(f"figure -> {os.path.basename(path)}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="folder with the three Kaggle exp*.csv files")
    ap.add_argument("--cars", nargs="*", default=["car11", "car9", "car8", "exp2_car"], help="held-out test cars")
    ap.add_argument("--clean-share", type=float, default=0.3, help="share of test drives left without a fault")
    ap.add_argument("--cutoff", type=float, default=0.95); ap.add_argument("--drive-rule", type=float, default=0.3)
    ap.add_argument("--epochs", type=int, default=40); ap.add_argument("--ft-epochs", type=int, default=20)
    ap.add_argument("--smooth", type=int, default=12); ap.add_argument("--quick", action="store_true")
    ap.add_argument("--graces", type=int, nargs="*", default=[5, 0, 3],
                    help="warm-up grace periods (minutes) to compare; the first one is used for the saved files")
    ap.add_argument("--smooth-long", type=int, default=60, help="slow score: average over this many windows (60 x 10 s = 10 min)")
    ap.add_argument("--min-flat", type=float, default=5, help="stuck check: never flag a freeze shorter than this (minutes)")
    ap.add_argument("--physics", action="store_true", help="add the physics inputs (not used in the final model)")
    ap.add_argument("--tag", default="", help="added to output file names, e.g. _physics")
    a = ap.parse_args()
    if not a.physics: M.PHYSICS.clear()
    if a.quick: a.epochs, a.ft_epochs, a.cars = 3, 2, a.cars[:1]
    rng = np.random.RandomState(42); trips = M.load_kaggle(a.data)
    out = [test_car(u, trips, a, rng) for u in a.cars]
    rows0 = pd.concat([r for r, _ in out if r is not None]); key0 = pd.concat([k for _, k in out if k is not None])
    pd.set_option("display.width", 220); pd.set_option("display.max_columns", 20)
    auc = roc_auc_score(rows0.truth, rows0[f"score_{FINAL_VARIANT}"])
    print(f"\n=== synthetic test: {len(key0)} drives on {key0.car.nunique()} held-out cars, minute-level AUROC {auc:.3f} (final model) ===")
    table = []
    for g in sorted(a.graces):
        r, k = decide(rows0, key0, g, a.drive_rule, FINAL_VARIANT)
        table += [dict(grace_min=g, level="minute", **metrics(r.truth.to_numpy(), r.flagged.to_numpy())),
                  dict(grace_min=g, level=f"drive (>{a.drive_rule:.0%})", **metrics((k.truth == "FAULT").to_numpy(), (k.model_says == "FAULT").to_numpy()))]
    res = pd.DataFrame(table); print(res.round(3).to_string(index=False))
    # --- the slow-fault fixes, side by side at the chosen warm-up
    g0, comp, byfault = a.graces[0], [], []
    for vname, v in VARIANTS.items():
        r, k = decide(rows0, key0, g0, a.drive_rule, v)
        comp += [dict(variant=vname, level="minute", AUROC=roc_auc_score(r.truth, r.score), **metrics(r.truth.to_numpy(), r.flagged.to_numpy())),
                 dict(variant=vname, level=f"drive (>{a.drive_rule:.0%})", AUROC=np.nan,
                      **metrics((k.truth == "FAULT").to_numpy(), (k.model_says == "FAULT").to_numpy()))]
        byfault.append(k.groupby("fault").apply(lambda g: (g.model_says == g.truth).sum()).rename(vname))
    comp = pd.DataFrame(comp)
    print(f"\n=== slow-fault fixes compared ({g0}-min warm-up) ===\n", comp.round(3).to_string(index=False))
    print("\n=== drives labelled correctly, by fault ===\n", pd.concat(byfault, axis=1).assign(
        drives=key0.groupby("fault").size()).to_string())
    rows, key = decide(rows0, key0, a.graces[0], a.drive_rule, FINAL_VARIANT)
    print(f"\n(files and the table below use the final model: stuck check, {a.graces[0]}-minute warm-up)")
    per = key.groupby("fault").apply(lambda g: pd.Series(dict(
        drives=len(g), labelled_correctly=(g.model_says == g.truth).mean(),
        right_sensor=(g.model_blames == g.sensor)[g.model_says == "FAULT"].mean() if g.sensor.iloc[0] else np.nan,
        median_minutes_to_alarm=g.minutes_to_alarm.median())))
    print("\n=== by scenario ===\n", per.round(2).to_string())
    res.assign(minute_AUROC=auc).to_csv(os.path.join(M.RESULTS, f"synthetic_results{a.tag}.csv"), index=False)
    key[["car", "drive", "fault", "sensor", "start_min", "truth", "model_says", "model_blames", "minutes_to_alarm"]].to_csv(
        os.path.join(M.RESULTS, f"synthetic_answer_key{a.tag}.csv"), index=False)
    rows.to_csv(os.path.join(M.RESULTS, f"synthetic_minutes{a.tag}.csv"), index=False)       # every scored minute, for make_figures.py
    print("\nsaved synthetic_results.csv, synthetic_answer_key.csv, synthetic_minutes.csv  (figures: run make_figures.py)")

if __name__ == "__main__":
    main()

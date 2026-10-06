"""OBD-II "virtual sensor" fault model (PyTorch).

Idea: on a healthy engine the sensors agree with each other (MAF follows RPM x load, MAP follows
throttle, coolant follows warm-up time, ...). The network learns those relations. At test time each
sensor is hidden in turn and predicted from the others; a sensor that keeps disagreeing with its
prediction is flagged, and the model names it.

Only standard OBD-II PIDs that a phone app / ELM327 dongle can log are used, so the same model can be
fine-tuned on your own car later:
    RPM, SPEED, LOAD, THROTTLE, ECT (coolant), IAT (intake air), MAP, MAF, STFT1 (short-term fuel trim)
Cars that don't report a sensor just have it masked out.

Pipeline
  1. pretrain  on every healthy car except the one being tested       (Kaggle exp1 + exp2 + exp3)
  2. fine-tune on the first 40% of the test car's drives (its "normal") (what you will do on your car)
  3. calibrate the alarm line on the next 20% of its drives
  4. test on the last 40%: false alarms on clean drives, detection of 10 sensor faults injected into those
     same drives (incl. realistic MAF/MAP faults where engine load and fuel trim react), and real
     trouble-code events where a car has them (car13)
Compared against: the pretrained model without fine-tuning, and a per-car linear (ridge) virtual sensor.

Final settings (see model_build_guide.md, Step 7)
  - minute score = worst sensor's prediction error (averaged over 2 minutes) or the stuck-sensor check, whichever is larger
  - alarm line = 90th percentile of the car's calibration scores ("Relaxed")
  - no alarms in the first 5 minutes of a drive (warm-up)
  - a drive is FAULT if more than 30% of its minutes are flagged; drives with under 1 minute after the warm-up are
    "too short to judge" (reported separately, not counted as caught or missed)
  - no physics inputs (tested with --physics; not kept)

Usage (from the repo root, after `pip install -e .`):
  python -m obdfault.evaluate                # full evaluation + final model (data from data/public/kaggle-obd2)
  python -m obdfault.evaluate --final-only   # only retrain/save models/obd_pretrained.pt
  python -m obdfault.evaluate --quick        # fast smoke test
Output is binary: every minute (and every drive) is HEALTHY or FAULT; for a FAULT it also names the likely sensor.
Writes obd_results_faults.csv, obd_results_summary.csv, obd_results_binary.csv to results/ and obd_pretrained.pt to models/.
"""
import argparse, os, time
import numpy as np, pandas as pd
from .config import CH, DT, W, FAULTY_CARS, FINAL_SCORING, KAGGLE_DATA, RESULTS, add_ltft
from .data import load_kaggle, load_ved
from .features import PHYSICS, windows
from .model import channel_errors, ridge_errors
from .scoring import calibrate, slow_stuck_scorer, auroc
from .faults import ALL_FAULTS, inject
from .train import prepare_car, save_final

def evaluate_car(unit, trips, a):
    P = prepare_car(unit, trips, a)
    if P is None: return None, None
    base, tuned, mu, sd, Dtr, Dcal, Dclean, test_trips = (P[k] for k in ("base", "tuned", "mu", "sd", "Dtr", "Dcal", "Dclean", "test_trips"))
    has = Dclean["m"].numpy().mean(0) > 0.5
    methods = {"pretrained only": lambda Ds: [channel_errors(base, d) for d in Ds],
               "fine-tuned (ours)": lambda Ds: [channel_errors(tuned, d) for d in Ds],
               "ridge baseline": lambda Ds: ridge_errors(Dtr, Ds)}
    fault_trips = {f: inject(test_trips, f) for f in ALL_FAULTS if has[CH.index(ALL_FAULTS[f][0])]}
    fault_sets = {f: windows(t, mu, sd, stride=2) for f, t in fault_trips.items()}
    # warm-up grace: minutes before `a.grace` never raise an alarm and are not used to set the cutoff (cold starts)
    minute = lambda D_: (D_["start"] + W) * DT / 60
    warm_cal, warm_clean = minute(Dcal) >= a.grace, minute(Dclean) >= a.grace
    rows, summ = [], []
    for name, errs in methods.items():
        Es = errs([Dcal, Dclean] + list(fault_sets.values()))
        clean_ok = ~Dclean["code"]
        # "stuck check" is the final scoring; the other two are compared in the results (two-speed = EXPERIMENT, not kept)
        for scoring in ("shared cutoff", "stuck check", "two-speed + stuck check"):
            if scoring == "shared cutoff":
                sc, s_cal = calibrate(Es[0], Dcal["trip"], a.smooth); score = lambda E, D_, trips_: sc(E, D_["trip"])
            else:
                score, s_cal = slow_stuck_scorer(Es[0], Dcal, P["cal_trips"], a, use_slow=scoring != "stuck check")
            s_clean, _ = score(Es[1], Dclean, test_trips)
            real = auroc(Dclean["code"].astype(int), s_clean) if Dclean["code"].any() else np.nan
            scored = [(f, Df, *score(E, Df, fault_trips[f]), minute(Df) >= a.grace) for (f, Df), E in zip(fault_sets.items(), Es[2:])]
            # Only judged data is scored: minutes after the warm-up, and drives with at least a.min_judge windows
            # (1 minute) after the warm-up. Shorter drives are "too short to judge" and counted separately.
            warm_c, trip_c = warm_clean[clean_ok], Dclean["trip"][clean_ok]
            judged_c = pd.Series(warm_c).groupby(trip_c).sum() >= a.min_judge
            for q in a.cutoffs:                               # every cutoff comes from calibration drives only
                thr = np.quantile(s_cal[warm_cal], q)
                fc = (s_clean[clean_ok] > thr) & warm_c                        # FAULT calls on clean minutes
                fa = fc[warm_c].mean()
                summ.append(dict(car=unit, method=name, scoring=scoring, cutoff=q, grace_min=a.grace,
                                 sensors=",".join(c for c, h in zip(CH, has) if h),
                                 test_windows=int(warm_c.sum()), false_alarm_rate=fa,
                                 clean_drives_judged=int(judged_c.sum()), clean_drives_too_short=int((~judged_c).sum()),
                                 real_code_windows=int(Dclean["code"].sum()), real_code_AUROC=real))
                clean_trip_hit = pd.Series(fc).groupby(trip_c).mean()[judged_c]
                for f, Df, s_f, who, warm_f in scored:
                    ch = CH.index(ALL_FAULTS[f][0]); ff = (s_f > thr) & warm_f          # FAULT calls on faulty minutes
                    y = np.r_[np.zeros(clean_ok.sum()), np.ones(len(s_f))]
                    judged_f = pd.Series(warm_f).groupby(Df["trip"]).sum() >= a.min_judge
                    trip_hit = pd.Series(ff).groupby(Df["trip"]).mean()[judged_f]   # share of each drive's minutes flagged
                    for rule in a.drive_rules:        # a whole drive is FAULT if more than `rule` of its minutes are flagged
                        rows.append(dict(car=unit, method=name, scoring=scoring, cutoff=q, grace_min=a.grace, drive_rule=rule,
                                         fault=f, sensor=ALL_FAULTS[f][0], AUROC=auroc(y, np.r_[s_clean[clean_ok], s_f]),
                                         window_detection=ff[warm_f].mean(), drive_detection=(trip_hit > rule).mean(),
                                         blamed_right_sensor=(who[ff] == ch).mean() if ff.any() else np.nan,
                                         false_alarm_rate=fa,
                                         # binary HEALTHY/FAULT counts on judged data: same drives, clean copy vs faulty copy (1:1)
                                         TP=int(ff.sum()), FN=int((~ff & warm_f).sum()), FP=int(fc.sum()), TN=int((~fc & warm_c).sum()),
                                         drive_TP=int((trip_hit > rule).sum()), drive_FN=int((trip_hit <= rule).sum()),
                                         drive_FP=int((clean_trip_hit > rule).sum()), drive_TN=int((clean_trip_hit <= rule).sum()),
                                         drives_too_short=int((~judged_f).sum())))
    return pd.DataFrame(rows), pd.DataFrame(summ)

TAG = ""   # added to output file names (set by --tag, e.g. "_physics")

def save_csv(df, name):
    """Save to results/; if the file is open in Excel (locked), save as <name>_new.csv instead."""
    path = os.path.join(RESULTS, name.replace(".csv", TAG + ".csv"))
    try: df.to_csv(path, index=False)
    except PermissionError:
        path = path.replace(".csv", "_new.csv"); df.to_csv(path, index=False)
    return os.path.basename(path)

def binary_report(R):
    """Confusion matrix + accuracy / precision / recall / F1 for each method, scoring and cutoff:
    per 1-minute window, and per whole drive for each drive rule."""
    def metrics(tp, fn, fp, tn):
        p = tp / (tp + fp) if tp + fp else np.nan; r = tp / (tp + fn) if tp + fn else np.nan
        return dict(TP=tp, FN=fn, FP=fp, TN=tn, accuracy=(tp + tn) / (tp + fn + fp + tn), precision=p, recall=r,
                    F1=2 * p * r / (p + r) if p + r else np.nan, false_alarm_rate=fp / (fp + tn))
    out, keys = [], ["method", "scoring", "cutoff"]
    first = R[R.drive_rule == R.drive_rule.min()]                         # minute counts are the same for every rule
    for k, g in first.groupby(keys):
        out.append(dict(zip(keys, k), level="minute", **metrics(*g[["TP", "FN", "FP", "TN"]].sum().to_numpy())))
    for k, g in R.groupby(keys + ["drive_rule"]):
        out.append(dict(zip(keys, k[:3]), level=f"drive (>{k[3]:.0%} of minutes)",
                        **metrics(*g[["drive_TP", "drive_FN", "drive_FP", "drive_TN"]].sum().to_numpy())))
    return pd.DataFrame(out).sort_values(keys + ["level"]).reset_index(drop=True)

# =================================== MAIN ======================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=KAGGLE_DATA, help="folder with the three Kaggle exp*.csv files (default: data/public/kaggle-obd2)")
    ap.add_argument("--cars", nargs="*", help="which cars to hold out (default: every healthy car with enough data)")
    ap.add_argument("--epochs", type=int, default=40); ap.add_argument("--ft-epochs", type=int, default=20)
    ap.add_argument("--smooth", type=int, default=12, help="average scores over this many windows (12 x 10 s = 2 min)")
    ap.add_argument("--cutoffs", type=float, nargs="*", default=[0.90, 0.95, 0.99, 0.975, 0.80],
                    help="alarm cutoffs = percentiles of each car's calibration scores; the first one is the chosen "
                         "operating point (0.90 = 'Relaxed')")
    ap.add_argument("--min-judge", type=int, default=6, help="drives need this many windows (6 = 1 minute) after the warm-up to be judged")
    ap.add_argument("--drive-rules", type=float, nargs="*", default=[0.3, 0.2, 0.4, 0.5],
                    help="a whole drive is FAULT if more than this share of its minutes is flagged; "
                         "the first one is the chosen rule (0.3 = more than 30%% of minutes)")
    ap.add_argument("--grace", type=float, default=5, help="warm-up grace period (minutes): no alarms at the start of a drive")
    ap.add_argument("--smooth-long", type=int, default=60, help="slow score: average over this many windows (60 x 10 s = 10 min)")
    ap.add_argument("--min-flat", type=float, default=5, help="stuck check: never flag a freeze shorter than this (minutes)")
    # ---- EXPERIMENT options (all off by default; the final model uses none of them) ----
    ap.add_argument("--physics", action="store_true",
                    help="add the physics inputs (tested: slightly better on air faults, more false alarms; not used in the final model)")
    ap.add_argument("--tag", default="", help="added to output file names, e.g. _physics")
    ap.add_argument("--skip-final", action="store_true", help="don't retrain/save obd_pretrained.pt")
    ap.add_argument("--final-only", action="store_true", help="skip the evaluation; only train and save obd_pretrained.pt")
    ap.add_argument("--ved", help="experiment: folder with the Vehicle Energy Dataset (VED_Static_Data_*.xlsx + dynamic/*.csv)")
    ap.add_argument("--ved-pretrain", type=int, default=60, help="VED gasoline cars added to pretraining")
    ap.add_argument("--ved-test", type=int, default=4, help="VED cars held out as extra test cars (never pretrained on)")
    ap.add_argument("--ltft", action="store_true", help="experiment: add long-term fuel trim (bank 1) as a 10th sensor")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    if a.quick: a.epochs, a.ft_epochs = 3, 2
    if not a.physics: PHYSICS.clear()                    # final model: no physics inputs
    if a.ltft: add_ltft()
    global TAG; TAG = a.tag
    t0 = time.time(); trips = load_kaggle(a.data)
    size = pd.Series({u: sum(len(t["X"]) for t in trips if t["unit"] == u) for u in {t["unit"] for t in trips}})
    print(f"{len(trips)} drives, {len(size)} cars; 5-s rows per car:\n{size.sort_values(ascending=False).to_string()}")
    cars = a.cars or [u for u in size.index if size[u] >= 1500 and u not in FAULTY_CARS]
    if a.ved:
        ved = load_ved(a.ved, a.ved_pretrain, a.ved_test); trips += ved
        if not a.cars: cars += sorted({t["unit"] for t in ved if t["test_only"]})
    if a.quick and not a.cars: cars = cars[:1] + [c for c in cars if c.startswith("ved")][:1]
    if a.final_only: return save_final(trips, a, t0)
    all_rows, all_summ = [], []
    for u in sorted(cars):
        t1 = time.time(); r, s = evaluate_car(u, trips, a)
        if r is None: print(f"{u}: skipped (not enough data)"); continue
        all_rows.append(r); all_summ.append(s); print(f"{u}: done in {time.time() - t1:.0f}s", flush=True)
        R, S = pd.concat(all_rows), pd.concat(all_summ)                       # save after every car
        save_csv(R, "obd_results_faults.csv"); save_csv(S, "obd_results_summary.csv")
    pd.set_option("display.width", 220); pd.set_option("display.max_columns", 20); pd.set_option("display.max_colwidth", 50)
    q0 = a.cutoffs[0]
    R0 = R[(R.cutoff == q0) & (R.scoring == FINAL_SCORING) & (R.drive_rule == a.drive_rules[0])]
    S0 = S[(S.cutoff == q0) & (S.scoring == FINAL_SCORING)]
    print(f"\n=== per-car false alarms and real trouble codes (cutoff {q0}) ===\n", S0.round(3).to_string(index=False))
    print(f"\n=== injected faults, mean over held-out cars (cutoff {q0}) ===\n",
          R0.groupby(["fault", "method"])[["AUROC", "window_detection", "drive_detection", "blamed_right_sensor"]].mean().round(3).to_string())
    print(f"\n=== overall (cutoff {q0}) ===\n", R0.groupby("method")[["AUROC", "window_detection", "drive_detection", "blamed_right_sensor", "false_alarm_rate"]].mean().round(3).to_string())
    print("\n=== binary classifier (FAULT = positive), all cars and faults pooled, every cutoff ===")
    B = binary_report(R); print(B.round(3).to_string(index=False))
    print("saved", save_csv(R, "obd_results_faults.csv"), save_csv(S, "obd_results_summary.csv"), save_csv(B, "obd_results_binary.csv"))
    if not a.skip_final: save_final(trips, a, t0)

if __name__ == "__main__":
    main()

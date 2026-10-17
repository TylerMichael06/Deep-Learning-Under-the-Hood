"""Opt-in thermostat coverage audit and paired warm-up experiment.

The default detector and official result files are not modified by this module.
"""
import numpy as np
import pandas as pd
from .config import CH, DT, W, MIN_CH, FAULTY_CARS

def thermostat_audit(trips, cold_max=50.):
    """Describe recording coverage; candidate flags are exploratory, not health labels.

    Window availability and the chronological 40/20/40 boundaries mirror the
    default pipeline without fitting a model. A low initial ECT with unknown
    runtime is only a provisional cold start. Whole trips with codes are excluded
    from candidates (stricter than the baseline's window-level code filtering).
    """
    rows = []
    for trip_id, tr in enumerate(trips):
        X = tr["X"]; ect = X[:, CH.index("ECT")]
        finite = np.flatnonzero(np.isfinite(ect))
        starts = np.arange(0, len(X) - W + 1, 2)
        if len(starts):
            present = np.lib.stride_tricks.sliding_window_view(~np.isnan(X), W, axis=0)[::2].all(-1)
            starts = starts[present.sum(1) >= MIN_CH]
        clean_windows = sum(not tr["code"][s:s + W].any() for s in starts)
        judged = sum((s + W) * DT / 60 >= 5 for s in starts)
        source = tr.get("runtime_source", "unknown")
        runtime0 = float(tr["ctx"][0])
        if source != "engine_runtime":
            evidence = "unknown"
        elif not np.isfinite(runtime0) or runtime0 < 0:
            evidence = "invalid_runtime"
        elif runtime0 <= 1:
            evidence = "observed_near_start"
        else:
            evidence = "late_recording"
        first = float(ect[finite[0]]) if len(finite) else np.nan
        delay = float(finite[0] * DT / 60) if len(finite) else np.nan
        delta = float(np.max(np.abs(ect[finite] - first)) * .5) if len(finite) >= W else np.nan
        candidate = (tr["unit"] not in FAULTY_CARS and not tr["code"].any()
                     and first < cold_max and delay <= 1 and len(finite) / len(X) >= .8
                     and judged >= 6 and evidence in ("unknown", "observed_near_start"))
        rows.append(dict(trip_id=trip_id, car=tr["unit"], order=tr["order"], samples=len(X),
                         duration_min=len(X) * DT / 60, windows=len(starts), clean_windows=clean_windows,
                         judged_windows=judged, runtime_source=source, runtime_start_min=runtime0,
                         start_evidence=evidence, ect_first=first, ect_first_min=delay,
                         ect_coverage=len(finite) / len(X), codes=bool(tr["code"].any()),
                         rpm_load_coverage=np.isfinite(X[:, [CH.index("RPM"), CH.index("LOAD")]]).all(1).mean(),
                         iat_first=float(X[0, CH.index("IAT")]),
                         legacy_max_change=delta, legacy_unchanged=np.isfinite(delta) and delta <= 1e-6,
                         cold_candidate=bool(candidate),
                         confirmed_start_candidate=bool(candidate and evidence == "observed_near_start")))
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["split"] = "unused"; df["baseline_evaluable"] = False
    for unit, g in df.groupby("car"):
        mine = g[g.windows > 0].sort_values("order", kind="stable")
        if len(mine) < 3 or unit in FAULTY_CARS:
            continue
        lens = mine.samples.cumsum().to_numpy()
        c1, c2 = (np.searchsorted(lens, q * lens[-1]) + 1 for q in (.4, .6))
        ft, cal, test = mine.iloc[:c1], mine.iloc[c1:c2], mine.iloc[c2:]
        for name, part in (("adaptation", ft), ("calibration", cal), ("test", test)):
            df.loc[part.index, "split"] = name
        # Same default car-size and minimum-window checks as evaluate/prepare_car.
        usable = (g.samples.sum() >= 1500 and ft.clean_windows.sum() >= 50
                  and cal.clean_windows.sum() >= 20 and test.windows.sum() > 0)
        df.loc[g.index, "baseline_evaluable"] = bool(usable)
    return df


def period_decision(score, minute, threshold, start=2., stop=5., mask=None):
    """Judge only finite, eligible windows in a fixed period; six are required."""
    score, minute = np.asarray(score), np.asarray(minute)
    ok = (minute >= start) & (minute < stop) & np.isfinite(score)
    if mask is not None:
        ok &= mask
    judged = ok.sum() >= 6
    flags = (score > threshold) & ok
    share = float(flags.sum() / ok.sum()) if judged else np.nan
    return dict(judged=bool(judged), eligible_windows=int(ok.sum()), flagged_share=share,
                decision="FAULT" if judged and share > .3 else "HEALTHY" if judged else "NO_CALL",
                first_alarm_min=float(minute[flags][0]) if judged and flags.any() else np.nan)


def combine_early_decision(baseline, early):
    """An early alarm adds evidence; an early no-call never clears a baseline alarm.

    Keep the baseline's judged cohort for this paired experiment. An early-only
    verdict for a short trip is reported separately, not as full-drive health.
    """
    combined = dict(baseline)
    if baseline["judged"] and early["decision"] == "FAULT":
        combined["decision"] = "FAULT"
    combined["flagged_share"] = np.nan  # OR of period verdicts, not a single window fraction
    alarms = [d["first_alarm_min"] for d in (baseline, early) if d["decision"] == "FAULT"]
    combined["first_alarm_min"] = min(alarms) if baseline["judged"] and alarms else np.nan
    return combined


def run_experiment(trips, audit, args, score_rows=None):
    """Paired comparison on a fixed 5-10 minute horizon and eligible windows.

    The baseline is refitted with the existing pipeline. Its historical training
    protocol is preserved; metrics here must not be compared directly to the
    official full-drive table. Warm starts are controls, never injected faults.
    """
    import torch
    from .features import windows
    from .train import prepare_car
    from .model import channel_errors
    from .scoring import WarmupRise, slow_stuck_scorer
    from .faults import delay_coolant

    rows, coverage = [], []
    for car, group in audit[audit.baseline_evaluable].groupby("car"):
        selected = {part: group[(group.split == part) & group.cold_candidate]
                    for part in ("adaptation", "calibration", "test")}
        record = dict(car=car, **{f"{k}_candidates": len(v) for k, v in selected.items()})
        if len(selected["adaptation"]) < 3 or len(selected["calibration"]) < 2 or not len(selected["test"]):
            coverage.append(dict(record, status="insufficient cold-state trips")); continue
        rise = WarmupRise()
        try:
            rise.fit([trips[i] for i in selected["adaptation"].trip_id])
        except ValueError as exc:
            coverage.append(dict(record, status=str(exc))); continue
        calibration, early_calibration = [], []
        for i in selected["calibration"].trip_id:
            tr = trips[i]; score = rise.score(tr)
            # Same 10-second endpoint grid as the network's windows.
            D = windows([tr], np.zeros(len(CH)), np.ones(len(CH)), stride=2)
            end = D["start"] + W - 1
            minute = (end + 1) * DT / 60
            values = score[end][(minute >= 5) & (minute <= 10) & np.isfinite(score[end])]
            if len(values) >= 6:
                calibration.append(values)
            early_values = score[end][(minute >= 2) & (minute < 5) & np.isfinite(score[end])]
            if len(early_values) >= 6:
                early_calibration.append(early_values)
        if len(calibration) < 2:
            coverage.append(dict(record, status="insufficient usable calibration trips")); continue
        threshold = float(np.quantile(np.concatenate(calibration), .90))
        early_threshold = (float(np.quantile(np.concatenate(early_calibration), .90))
                           if len(early_calibration) >= 2 else np.nan)
        print(f"{car}: fitting baseline; rise uses {rise.fit_trips} adaptation / {len(calibration)} calibration trips", flush=True)
        torch.manual_seed(args.seed)
        P = prepare_car(car, trips, args)
        if P is None:
            coverage.append(dict(record, status="baseline unavailable")); continue
        Ecal = channel_errors(P["tuned"], P["Dcal"])
        baseline_score, cal_score = slow_stuck_scorer(Ecal, P["Dcal"], P["cal_trips"], args, use_slow=False)
        warm = (P["Dcal"]["start"] + W) * DT / 60 >= 5
        baseline_threshold = float(np.quantile(cal_score[warm], .90))
        coverage.append(dict(record, status="evaluated", fit_trips=rise.fit_trips,
                             calibration_trips=len(calibration), rise_threshold=threshold,
                             baseline_threshold=baseline_threshold,
                             early_calibration_trips=len(early_calibration), early_threshold=early_threshold))
        cold_ids = set(selected["test"].trip_id)
        test_info = group[group.split == "test"].sort_values("order", kind="stable")
        assert [trips[i]["order"] for i in test_info.trip_id] == [t["order"] for t in P["test_trips"]]
        for info in test_info.itertuples():
            tr = trips[info.trip_id]
            cold = info.trip_id in cold_ids
            if not cold and (info.codes or not np.isfinite(info.ect_first) or info.ect_first < 50):
                continue
            cases = [("clean_cold" if cold else "clean_warm", tr, False)]
            if cold:
                cases += [(f"delay_{rate}", delay_coolant(tr, rate), True) for rate in (.75, .5)]
            Dclean = windows([tr], P["mu"], P["sd"], stride=2)
            for scenario, changed, fault in cases:
                D = windows([changed], P["mu"], P["sd"], stride=2)
                np.testing.assert_array_equal(D["start"], Dclean["start"])
                end = D["start"] + W - 1; minute = (end + 1) * DT / 60
                residual = rise.score(changed)[end]
                original_residual = rise.score(tr)[end]
                base, _ = baseline_score(channel_errors(P["tuned"], D), D, [changed])
                eligible = (minute >= 5) & (minute <= 10) & np.isfinite(base)
                if cold:
                    eligible &= np.isfinite(original_residual) & np.isfinite(residual)
                difference = np.abs(changed["X"][:, CH.index("ECT")] - tr["X"][:, CH.index("ECT")])
                changed_in_horizon = bool(np.any(difference[:int(10 * 60 / DT)] > 1e-6))
                excluded = fault and not changed_in_horizon
                if score_rows is not None:
                    score_rows.append(pd.DataFrame(dict(car=car, trip_id=info.trip_id,
                        scenario=scenario, fault=fault, cold=cold, minute=minute,
                        baseline_score=base, baseline_threshold=baseline_threshold,
                        rise_score=residual, rise_threshold=threshold, early_threshold=early_threshold,
                        original_rise_finite=np.isfinite(original_residual), code=D["code"])))
                flags = {"baseline": base > baseline_threshold,
                         "rise_only": residual > threshold,
                         "baseline_plus_rise": (base > baseline_threshold) | (residual > threshold)}
                for method, flagged in flags.items():
                    judged = eligible.sum() >= 6 and not excluded and (cold or method != "rise_only")
                    share = float(flagged[eligible].mean()) if judged else np.nan
                    first = minute[eligible & flagged]
                    rows.append(dict(car=car, trip_id=info.trip_id, scenario=scenario, fault=fault,
                                     start_evidence=info.start_evidence, method=method, judged=bool(judged),
                                     eligible_windows=int(eligible.sum()), flagged_share=share,
                                     decision="FAULT" if judged and share > .3 else "HEALTHY" if judged else "NO_CALL",
                                     first_alarm_min=float(first[0]) if judged and len(first) else np.nan,
                                     unchanged_fault=bool(excluded)))
                if args.early_warmup:
                    baseline = rows[-3]
                    early = period_decision(residual, minute, early_threshold,
                                            mask=np.isfinite(original_residual))
                    if not cold or excluded or not np.isfinite(early_threshold):
                        early.update(judged=False, decision="NO_CALL", flagged_share=np.nan, first_alarm_min=np.nan)
                    combined = combine_early_decision(baseline, early)
                    for method, decision in (("early_rise_only", early), ("baseline_plus_early_rise", combined)):
                        rows.append(dict(car=car, trip_id=info.trip_id, scenario=scenario, fault=fault,
                                         start_evidence=info.start_evidence, unchanged_fault=bool(excluded),
                                         **{k: decision[k] for k in ("judged", "eligible_windows", "flagged_share", "decision", "first_alarm_min")},
                                         method=method, early_eligible_windows=early["eligible_windows"],
                                         early_available=early["judged"]))
        print(f"{car}: paired comparison complete", flush=True)
    return pd.DataFrame(rows), pd.DataFrame(coverage)


def main():
    import argparse, hashlib, json, subprocess
    from pathlib import Path
    import torch
    from .config import KAGGLE_DATA
    from .data import load_kaggle
    from .features import PHYSICS

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=KAGGLE_DATA)
    parser.add_argument("--output", required=True, help="new experiment directory; existing directories are refused")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--ft-epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--early-warmup", action="store_true",
                        help="also evaluate a separately calibrated thermostat check from minute 2 up to minute 5")
    parser.add_argument("--save-scores", action="store_true",
                        help="save per-window scores for offline horizon and full-drive audits")
    a = parser.parse_args()
    if min(a.epochs, a.ft_epochs, a.threads) < 1:
        parser.error("epochs and threads must be positive")
    out = Path(a.output); out.mkdir(parents=True, exist_ok=False)
    a.smooth, a.smooth_long, a.grace, a.min_flat = 12, 60, 5, 5
    torch.set_num_threads(a.threads); PHYSICS.clear()
    trips = load_kaggle(a.data); audit = thermostat_audit(trips)
    audit.to_csv(out / "coverage.csv", index=False)
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    source_root = Path(__file__).parent
    metadata = dict(settings=vars(a), revision=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                    source_sha256={p.name: digest(p) for p in sorted(source_root.glob("*.py"))},
                    data_sha256={p.name: digest(p) for p in sorted(Path(a.data).glob("exp*.csv"))},
                    protocol="paired cold-state sensitivity test; 5-10 min; eligible-window denominator; cutoff .90; drive rule .30",
                    early_protocol="optional [2,5) min rise score; separate .90 calibration; >.30 period share; OR period verdict with unchanged baseline",
                    limitation="unknown runtime allowed; synthetic time warp; very few independent trips; no real-fault validation")
    (out / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    score_rows = [] if a.save_scores else None
    rows, coverage = run_experiment(trips, audit, a, score_rows=score_rows)
    if score_rows:
        pd.concat(score_rows, ignore_index=True).to_csv(out / "window_scores.csv", index=False)
    coverage.to_csv(out / "cars.csv", index=False)
    if rows.empty:
        print("No evaluable cars; see cars.csv."); return
    rows.to_csv(out / "decisions.csv", index=False)
    report = rows.groupby(["method", "scenario"]).agg(total=("judged", "size"), judged=("judged", "sum"))
    usable = rows[rows.judged].assign(alarm=lambda x: x.decision == "FAULT")
    report["alarms"] = usable.groupby(["method", "scenario"]).alarm.sum()
    report["alarm_rate"] = report.alarms / report.judged.replace(0, np.nan)
    report.to_csv(out / "summary.csv")
    clean = rows[rows.scenario == "clean_cold"][["car", "trip_id", "method", "decision", "judged"]]
    paired = rows[rows.fault].merge(clean, on=["car", "trip_id", "method"], suffixes=("_fault", "_clean"), validate="many_to_one")
    paired["new_detection"] = (paired.decision_clean == "HEALTHY") & (paired.decision_fault == "FAULT")
    paired["already_flagged_clean"] = (paired.decision_clean == "FAULT") & (paired.decision_fault == "FAULT")
    paired.to_csv(out / "paired_changes.csv", index=False)
    print(report.to_string())
    print(f"\nSaved exploratory results to {out}")


if __name__ == "__main__":
    main()

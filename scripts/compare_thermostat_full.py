"""Paired full-battery regression audit of the frozen, opt-in early thermostat rule.

Uses the final fine-tuned model and operating point only. Reuses one trained model
per car for both policies; never writes official results or model checkpoints.
Eligibility is frozen from the original recording, as in the earlier retrospective
audit. This is not validation of an online eligibility policy.
"""
import argparse
import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from obdfault.config import CH, DT, W, KAGGLE_DATA
from obdfault.data import load_kaggle
from obdfault.features import PHYSICS, windows
from obdfault.faults import ALL_FAULTS, inject
from obdfault.model import channel_errors
from obdfault.scoring import WarmupRise, slow_stuck_scorer, auroc
from obdfault.synthetic import SCENARIOS, STEPS_PER_MIN, usable, apply_fault, metrics
from obdfault.thermostat import thermostat_audit, period_decision
from obdfault.train import prepare_car

SYNTHETIC_CARS = ['car11', 'car9', 'car8', 'exp2_car']


def verdict(scores, minute, threshold, valid):
    """Match the official full-drive denominator and six-late-window abstention."""
    late = minute >= 5
    judged = int((valid & late).sum()) >= 6
    flagged = (scores > threshold) & late
    return judged, bool(judged and flagged[valid].mean() > .3)


def fit_early(trips, group):
    selected = group[group.cold_candidate]
    ids = {s: selected.loc[selected.split.eq(s), 'trip_id'].tolist()
           for s in ['adaptation', 'calibration', 'test']}
    if len(ids['adaptation']) < 3 or len(ids['calibration']) < 2 or not ids['test']:
        return None, np.nan, 'insufficient cold-state trips'
    try:
        model = WarmupRise().fit([trips[i] for i in ids['adaptation']])
    except ValueError as exc:
        return None, np.nan, str(exc)
    values = []
    for i in ids['calibration']:
        D = windows([trips[i]], np.zeros(len(CH)), np.ones(len(CH)), stride=2)
        end = D['start'] + W - 1
        minute = (end + 1) * DT / 60
        score = model.score(trips[i])[end]
        for early in [False, True]:
            mask = ((minute >= 2) & (minute < 5)) if early else ((minute >= 5) & (minute <= 10))
            v = score[mask & np.isfinite(score)]
            if len(v) >= 6:
                values.append((early, v))
    # Preserve both calibration-availability requirements of the original experiment.
    if any(sum(e == period for e, _ in values) < 2 for period in [False, True]):
        return None, np.nan, 'insufficient usable calibration trips'
    return model, float(np.quantile(np.concatenate([v for e, v in values if e]), .9)), 'available'


def synthetic_cases(trips, audit):
    """Use the existing synthetic suite's car order, RNG, and scenario assignment."""
    rng = np.random.RandomState(42)
    out = {}
    for car in SYNTHETIC_CARS:
        rows = audit[audit.car.eq(car) & audit.split.eq('test')].sort_values('order', kind='stable')
        cases = []
        for row in rows.itertuples():
            tr = trips[row.trip_id]
            options = [s for s in SCENARIOS if usable(tr, s[1], s[4] * STEPS_PER_MIN)]
            if rng.rand() < .3 or not options:
                cases.append(('clean', tr, False))
            else:
                name, sensor, kind, size, start = options[rng.randint(len(options))]
                X, _ = apply_fault(tr['X'], sensor, kind, size, start * STEPS_PER_MIN, rng)
                cases.append((name, dict(tr, X=X), True))
        out[car] = cases
    return out


def summarize(df):
    judged = df[df.judged]
    summary = judged.groupby(['suite', 'scenario'], sort=False).agg(
        n=('baseline', 'size'), baseline_alarms=('baseline', 'sum'),
        combined_alarms=('combined', 'sum'), added_alarms=('added', 'sum'),
        unchanged_injections=('unchanged_injection', 'sum'))
    return summary.reset_index()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data', default=KAGGLE_DATA)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--ft-epochs', type=int, default=20)
    ap.add_argument('--threads', type=int, default=2)
    a = ap.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    a.smooth, a.smooth_long, a.grace, a.min_flat = 12, 60, 5, 5
    torch.set_num_threads(a.threads)
    PHYSICS.clear()
    start = time.perf_counter()
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    manifest = dict(settings={k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()},
        revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        source_sha256={str(p): digest(p) for p in [Path(__file__), *sorted(Path('src/obdfault').glob('*.py'))]},
        data_sha256={p.name: digest(p) for p in sorted(Path(a.data).glob('exp*.csv'))},
        python=platform.python_version(), torch=torch.__version__, numpy=np.__version__, pandas=pd.__version__,
        protocol='CPU; seed reset per car; shared fitted baseline; cutoff .90; >.30 full-drive rule; six late windows; early [2,5) period OR; original-recording candidate and finite-score masks frozen; no tuning',
        scope='All evaluable Kaggle cars and available ALL_FAULTS; existing synthetic suite with RNG 42 and original car order. Drive decisions only; no new continuous combined score or sensor attribution.')
    (a.output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    trips = load_kaggle(a.data)
    audit = thermostat_audit(trips)
    audit.to_csv(a.output/'coverage.csv', index=False)
    synthetic = synthetic_cases(trips, audit)
    records, coverage, code_rows = [], [], []
    for car, group in audit[audit.baseline_evaluable].groupby('car', sort=True):
        t0 = time.perf_counter()
        torch.manual_seed(a.seed)
        P = prepare_car(car, trips, a)
        assert P is not None
        model, early_threshold, status = fit_early(trips, group)
        score, s_cal = slow_stuck_scorer(channel_errors(P['tuned'], P['Dcal']), P['Dcal'], P['cal_trips'], a, use_slow=False)
        cal_minute = (P['Dcal']['start'] + W) * DT / 60
        threshold = float(np.quantile(s_cal[cal_minute >= 5], .9))
        info = group[group.split.eq('test')].sort_values('order', kind='stable')
        originals = P['test_trips']
        assert [trips[i]['order'] for i in info.trip_id] == [tr['order'] for tr in originals]
        has = P['Dclean']['m'].numpy().mean(0) > .5
        scenarios = [('clean', originals, False)] + [(f, inject(originals, f), True)
            for f in ALL_FAULTS if has[CH.index(ALL_FAULTS[f][0])]]
        def evaluate(suite, names, test, faults):
            D = windows(test, P['mu'], P['sd'], stride=2)
            base_score, _ = score(channel_errors(P['tuned'], D), D, test)
            minute_all = (D['start'] + W) * DT / 60
            for k, row in enumerate(info.itertuples()):
                mask = D['trip'] == k
                minute, bs = minute_all[mask], base_score[mask]
                valid = ~D['code'][mask] if suite == 'full' and not faults[k] else np.ones(mask.sum(), bool)
                judged, baseline = verdict(bs, minute, threshold, valid)
                early_alarm, early_available = False, False
                if model is not None and row.cold_candidate:
                    end = D['start'][mask] + W - 1
                    original_finite = np.isfinite(model.score(originals[k])[end])
                    early = period_decision(model.score(test[k])[end], minute, early_threshold, mask=original_finite)
                    early_available = early['judged']
                    early_alarm = early['decision'] == 'FAULT'
                combined = baseline or (judged and early_alarm)
                unchanged = bool(faults[k] and np.array_equal(originals[k]['X'], test[k]['X'], equal_nan=True))
                records.append(dict(car=car, trip_id=row.trip_id, suite=suite, scenario=names[k], fault=faults[k],
                    judged=judged, baseline=baseline, combined=combined, added=combined and not baseline,
                    early_available=early_available, early_alarm=early_alarm, cold_candidate=row.cold_candidate,
                    unchanged_injection=unchanged, windows=int(mask.sum()), valid_windows=int(valid.sum())))
            if suite == 'full' and names[0] == 'clean':
                code_rows.append(dict(car=car, code_windows=int(D['code'].sum()),
                    baseline_code_AUROC=auroc(D['code'].astype(int), base_score)))
        for name, test, fault in scenarios:
            evaluate('full', [name]*len(test), test, [fault]*len(test))
        if car in synthetic:
            cases = synthetic[car]
            evaluate('synthetic', [c[0] for c in cases], [c[1] for c in cases], [c[2] for c in cases])
        coverage.append(dict(car=car, early_status=status, early_threshold=early_threshold,
            baseline_threshold=threshold, test_recordings=len(info), fault_types=len(scenarios)-1,
            seconds=time.perf_counter()-t0))
        df = pd.DataFrame(records)
        assert not (df.baseline & ~df.combined).any()
        df.to_csv(a.output/'decisions.csv', index=False)
        pd.DataFrame(coverage).to_csv(a.output/'cars.csv', index=False)
        summarize(df).to_csv(a.output/'summary.csv', index=False)
        pd.DataFrame(code_rows).to_csv(a.output/'code_events.csv', index=False)
        print(f'{car}: {status}; {len(info)} test recordings; {time.perf_counter()-t0:.1f}s', flush=True)
    manifest['elapsed_seconds'] = time.perf_counter()-start
    (a.output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    syn = df[(df.suite == 'synthetic') & df.judged]
    result = {method: metrics(syn.fault.to_numpy(), syn[method].to_numpy()) for method in ['baseline', 'combined']}
    (a.output/'synthetic_metrics.json').write_text(json.dumps(result, indent=2)+'\n')
    print(summarize(df).to_string(index=False))
    print(json.dumps(result, indent=2))
    print(f'Total: {manifest["elapsed_seconds"]:.1f}s', flush=True)


if __name__ == '__main__':
    main()

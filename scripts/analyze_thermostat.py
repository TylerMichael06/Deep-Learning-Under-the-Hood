"""Fixed-protocol robustness audit. Reuses test recordings: exploratory, not confirmation.

Run after obdfault.thermostat --early-warmup. Does not write official results.
"""
import argparse
import hashlib
import io
import json
import subprocess
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from obdfault.config import CH, DT, W, KAGGLE_DATA
from obdfault.data import load_kaggle
from obdfault.features import PHYSICS, windows
from obdfault.faults import delay_coolant
from obdfault.scoring import WarmupRise
from obdfault.thermostat import thermostat_audit, period_decision


def grid(tr):
    d = windows([tr], np.zeros(len(CH)), np.ones(len(CH)), stride=2)
    end = d['start'] + W - 1
    return end, (end + 1) * DT / 60


def threshold(model, trips, ids, early=True):
    scores = []
    for i in ids:
        end, minute = grid(trips[i]); s = model.score(trips[i])[end]
        mask = ((minute >= 2) & (minute < 5)) if early else ((minute >= 5) & (minute <= 10))
        v = s[mask & np.isfinite(s)]
        if len(v) >= 6:
            scores.append(v)
    if len(scores) < 2:
        raise ValueError('Fewer than two usable calibration recordings')
    return float(np.quantile(np.concatenate(scores), .9))


def descriptive_interval(k, n):
    ci = binomtest(int(k), int(n)).proportion_ci(method='exact')
    return dict(count=int(k), total=int(n), rate=k/n, low=ci.low, high=ci.high)


def paired_comparison(base, added):
    """Each input has one row per source trip for one scenario; no pooled copies."""
    keys = ['car', 'trip_id', 'scenario']
    pair = base.merge(added, on=keys, suffixes=('_base', '_added'), validate='one_to_one')
    pair = pair[pair.judged_base & pair.judged_added]
    b, a = pair.decision_base.eq('FAULT'), pair.decision_added.eq('FAULT')
    gains, losses = int((~b & a).sum()), int((b & ~a).sum())
    discordant = gains + losses
    return dict(n=len(pair), baseline_alarms=int(b.sum()), combined_alarms=int(a.sum()),
                gains=gains, losses=losses,
                descriptive_p=binomtest(gains, discordant).pvalue if discordant else 1.)


def full_drive_decision(g):
    """Preserve evaluate.py's denominator (including suppressed warm-up windows).

    Cold scenarios share their original eligibility. This is an offline policy
    comparison, not a causal start-qualification or latency claim.
    """
    clean = not bool(g.fault.iloc[0])
    valid = ~g.code if clean else np.ones(len(g), dtype=bool)
    warm = g.minute >= 5
    judged = int((valid & warm).sum()) >= 6
    flags = (g.baseline_score > g.baseline_threshold) & warm
    baseline = 'FAULT' if judged and flags[valid].mean() > .3 else 'HEALTHY' if judged else 'NO_CALL'
    early = period_decision(g.rise_score.to_numpy(),g.minute.to_numpy(),g.early_threshold.iloc[0],
                            mask=g.original_rise_finite.to_numpy())
    if not g.cold.iloc[0] or not np.isfinite(g.early_threshold.iloc[0]):
        early['decision'] = 'NO_CALL'
    combined = 'FAULT' if judged and (baseline=='FAULT' or early['decision']=='FAULT') else baseline
    return dict(judged=judged,baseline=baseline,combined=combined,early=early['decision'])


def team_log_audit(ref):
    paths = subprocess.check_output(['git', 'ls-tree', '-r', '--name-only', ref, 'data'], text=True).splitlines()
    rows = []
    for path in paths:
        if not path.endswith('.csv'):
            continue
        raw = subprocess.check_output(['git', 'show', ref + ':' + path])
        text = raw.decode('utf-16') if raw[:2] in (b'\xff\xfe', b'\xfe\xff') else raw.decode('utf-8-sig')
        df = pd.read_csv(io.StringIO(text), skiprows=2, skipinitialspace=True)
        df.columns = df.columns.str.strip()
        names = dict(ect='Engine Coolant Temperature (°C)', iat='Intake Air Temperature (°C)',
                     rpm='Engine RPM (rpm)', load='Calculated Engine Load Value (%)')
        t = pd.to_numeric(df['Time(s)'], errors='coerce')
        record = dict(file=path, sha256=hashlib.sha256(raw).hexdigest(), rows=len(df), duration_min=(t.max()-t.min())/60)
        for key, name in names.items():
            v = pd.to_numeric(df[name], errors='coerce') if name in df else pd.Series(np.nan,index=df.index)
            valid = v.dropna()
            record[key+'_coverage'] = float(v.notna().mean())
            record[key+'_first'] = float(valid.iloc[0]) if len(valid) else np.nan
            record[key+'_first_sec'] = float(t.loc[valid.index[0]]) if len(valid) else np.nan
        record['has_runtime'] = any('run time' in c.lower() or 'runtime' in c.lower() for c in df.columns)
        rows.append(record)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data', default=KAGGLE_DATA)
    ap.add_argument('--experiment', required=True, type=Path)
    ap.add_argument('--output', required=True, type=Path)
    ap.add_argument('--team-ref', default='origin/main')
    ap.add_argument('--seed-root', type=Path, help='collect seed*/ results and audit saved full-drive scores')
    a = ap.parse_args(); a.output.mkdir(parents=True, exist_ok=False)
    PHYSICS.clear(); trips=load_kaggle(a.data); audit=thermostat_audit(trips)
    audit.to_csv(a.output/'coverage.csv',index=False)
    decisions=pd.read_csv(a.experiment/'decisions.csv')
    cars=pd.read_csv(a.experiment/'cars.csv')
    sensitivities=[]; paired=[]; intervals=[]; coverage=[]
    for car in cars.loc[cars.status.eq('evaluated'),'car']:
        chosen=audit[(audit.car==car)&audit.cold_candidate]
        parts={s: chosen.loc[chosen.split.eq(s),'trip_id'].tolist() for s in ['adaptation','calibration','test']}
        variants=[('original',None,parts['adaptation'],parts['calibration'])]
        variants += [('omit_adaptation',i,[j for j in parts['adaptation'] if j!=i],parts['calibration']) for i in parts['adaptation']]
        variants += [('omit_calibration',i,parts['adaptation'],[j for j in parts['calibration'] if j!=i]) for i in parts['calibration']]
        for kind, omitted, fit_ids, cal_ids in variants:
            model=WarmupRise().fit([trips[i] for i in fit_ids])
            for period in ['early','late']:
                thr=threshold(model,trips,cal_ids,early=period=='early')
                for i in parts['test']:
                    for scenario,rate in [('clean_cold',1.),('delay_0.75',.75),('delay_0.5',.5)]:
                        tr=trips[i] if rate==1 else delay_coolant(trips[i],rate)
                        end,minute=grid(tr); score=model.score(tr)[end]
                        d=period_decision(score,minute,thr,start=2 if period=='early' else 5,
                                          stop=5 if period=='early' else np.nextafter(10.,np.inf))
                        base=decisions[(decisions.car==car)&(decisions.trip_id==i)&(decisions.scenario==scenario)&decisions.method.eq('baseline')].iloc[0]
                        sensitivities.append(dict(car=car,variant=kind,omitted=omitted,period=period,trip_id=i,
                            scenario=scenario,threshold=thr,rise_decision=d['decision'],baseline=base.decision,
                            combined='FAULT' if base.decision=='FAULT' or d['decision']=='FAULT' else base.decision))
        for info in audit[audit.car.eq(car)&audit.split.eq('test')].itertuples():
            model=WarmupRise().fit([trips[i] for i in parts['adaptation']])
            end,minute=grid(trips[info.trip_id]);score=model.score(trips[info.trip_id])[end]
            coverage.append(dict(car=car,trip_id=info.trip_id,cold_candidate=info.cold_candidate,
                start_evidence=info.start_evidence,early_finite_windows=int(((minute>=2)&(minute<5)&np.isfinite(score)).sum())))
    for scenario,g in decisions.groupby('scenario'):
        b=g[g.method.eq('baseline')]; e=g[g.method.eq('baseline_plus_early_rise')]
        paired.append(dict(scenario=scenario,**paired_comparison(b,e)))
        for method in ['baseline','baseline_plus_early_rise']:
            valid=g[g.method.eq(method)&g.judged]
            if len(valid):
                intervals.append(dict(scenario=scenario,method=method,**descriptive_interval(valid.decision.eq('FAULT').sum(),len(valid))))
    sens=pd.DataFrame(sensitivities);sens.to_csv(a.output/'sensitivity.csv',index=False)
    summary=sens.assign(alarm=sens.combined.eq('FAULT'),rise_alarm=sens.rise_decision.eq('FAULT')).groupby(
        ['car','variant','omitted','period','scenario'],dropna=False).agg(n=('trip_id','size'),combined_alarms=('alarm','sum'),rise_alarms=('rise_alarm','sum'),threshold=('threshold','first')).reset_index()
    summary.to_csv(a.output/'sensitivity_summary.csv',index=False)
    pd.DataFrame(paired).to_csv(a.output/'paired_statistics.csv',index=False)
    pd.DataFrame(intervals).to_csv(a.output/'descriptive_intervals.csv',index=False)
    pd.DataFrame(coverage).to_csv(a.output/'test_score_coverage.csv',index=False)
    team_log_audit(a.team_ref).to_csv(a.output/'team_log_coverage.csv',index=False)
    if a.seed_root:
        seed_reports, full_rows = [], []
        for run in sorted(a.seed_root.glob('seed[0-9]*')):
            if not run.is_dir():
                continue
            report = pd.read_csv(run/'summary.csv').assign(seed=int(run.name[4:]))
            seed_reports.append(report)
            if (run/'window_scores.csv').exists():
                scores=pd.read_csv(run/'window_scores.csv')
                for (car,trip_id,scenario),g in scores.groupby(['car','trip_id','scenario']):
                    full_rows.append(dict(seed=int(run.name[4:]),car=car,trip_id=trip_id,scenario=scenario,
                                          **full_drive_decision(g)))
        pd.concat(seed_reports,ignore_index=True).to_csv(a.output/'seed_summary.csv',index=False)
        full=pd.DataFrame(full_rows)
        if not full.empty:
            full.to_csv(a.output/'full_drive_decisions.csv',index=False)
            full[full.judged].assign(baseline_alarm=lambda d:d.baseline.eq('FAULT'),
                combined_alarm=lambda d:d.combined.eq('FAULT')).groupby(['seed','scenario']).agg(
                    judged=('trip_id','size'),baseline_alarms=('baseline_alarm','sum'),combined_alarms=('combined_alarm','sum')).to_csv(a.output/'full_drive_summary.csv')
    metadata=dict(caveat='All checks reuse four source recordings in one car. Binomial intervals/p-values assume independent recordings and are descriptive only, not confirmatory population inference.',
        protocol='Unchanged model and .90 thresholds; delete one training or calibration recording, never tune against test',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        team_ref=subprocess.check_output(['git','rev-parse',a.team_ref],text=True).strip(),
        decisions_sha256=hashlib.sha256((a.experiment/'decisions.csv').read_bytes()).hexdigest())
    (a.output/'manifest.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(pd.DataFrame(paired).to_string(index=False))
    print(summary.groupby(['variant','period','scenario']).agg(min_alarms=('combined_alarms','min'),max_alarms=('combined_alarms','max')).to_string())


if __name__=='__main__':
    main()

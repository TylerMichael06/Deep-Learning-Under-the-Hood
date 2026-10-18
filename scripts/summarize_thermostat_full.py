"""Summarize saved full-battery decisions and render the report figure; no training."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from obdfault.synthetic import metrics
from obdfault.faults import ALL_FAULTS


def summarize(audit, figure):
    df = pd.read_csv(audit/'decisions.csv')
    # Fail on incomplete runs, duplicated cases, or unexpected policy regressions.
    manifest = json.loads((audit/'manifest.json').read_text())
    if 'elapsed_seconds' not in manifest:
        raise ValueError('The full comparison has not completed')
    if df.duplicated(['suite', 'car', 'trip_id', 'scenario']).any():
        raise ValueError('Duplicate source/scenario decisions')
    assert not (df.baseline & ~df.combined).any()
    full = df[df.suite.eq('full') & df.judged]
    clean = full[~full.fault]
    faults = full[full.fault]
    per_fault = faults.groupby('scenario', sort=False).agg(n=('trip_id', 'size'),
        baseline=('baseline', 'sum'), combined=('combined', 'sum'),
        added=('added', 'sum'), unchanged=('unchanged_injection', 'sum'))
    per_car = clean.groupby('car').agg(n=('trip_id', 'size'),
        baseline=('baseline', 'sum'), combined=('combined', 'sum'))
    tables = []
    # Match evaluate.binary_report: repeat each clean control once for every
    # fault type evaluated on its car. This is a benchmark weighting, not prevalence.
    fault_counts = df[df.suite.eq('full') & df.fault].groupby('car').scenario.nunique()
    repeated = clean.loc[clean.index.repeat(clean.car.map(fault_counts))]
    pooled = pd.concat([faults, repeated])
    synthetic = df[df.suite.eq('synthetic') & df.judged]
    for suite, data in [('full_fault_weighted', pooled), ('synthetic', synthetic)]:
        for method in ['baseline', 'combined']:
            tables.append(dict(suite=suite, method=method,
                **metrics(data.fault.to_numpy(), data[method].to_numpy())))
    metric_table = pd.DataFrame(tables)
    per_fault.to_csv(audit/'by_fault.csv')
    per_car.to_csv(audit/'clean_by_car.csv')
    metric_table.to_csv(audit/'metrics.csv', index=False)
    changes = df[df.added].copy()
    originals = df[df.suite.eq('full') & ~df.fault][['car', 'trip_id', 'baseline', 'combined']]
    changes = changes.merge(originals, on=['car', 'trip_id'], suffixes=('', '_clean'), validate='many_to_one')
    changes.to_csv(audit/'added_alarm_details.csv', index=False)
    labels = ['MAF −25%', 'MAF drift to −30%', 'MAP +10 kPa', 'Legacy thermostat injection',
              'Coolant sensor stuck', 'IAT +15 °C', 'Fuel trim +10%', 'Throttle noise',
              'MAF −25%, engine reacts', 'MAP +10 kPa, engine reacts']
    label_map = dict(zip(ALL_FAULTS, labels))
    labels = [label_map[name] for name in per_fault.index]
    if len(per_fault) != len(ALL_FAULTS):
        raise ValueError('Expected all ten fault types')
    plt.rcParams.update({'font.size':10, 'axes.spines.top':False, 'axes.spines.right':False,
                         'figure.facecolor':'white', 'savefig.facecolor':'white'})
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5), gridspec_kw={'width_ratios':[1.5, 1]})
    for ax, table, names in [(axes[0], per_fault, labels), (axes[1], per_car, per_car.index)]:
        y = np.arange(len(table))
        for offset, method, color, label in [(-.18,'baseline','#2563a6','Current model'),(.18,'combined','#d45a21','With early check')]:
            values = table[method].to_numpy()/table.n.to_numpy()*100
            ax.barh(y+offset, values, height=.33, color=color, label=label)
            for yy, v, k, n in zip(y+offset, values, table[method], table.n):
                ax.text(v+.8, yy, f'{k}/{n}', va='center', fontsize=8)
        ax.set_yticks(y, names); ax.invert_yaxis()
        ax.set_xlim(0, 119 if ax is axes[0] else max(35, (table[['baseline','combined']].max(axis=1)/table.n*100).max()+12))
        ax.grid(axis='x', alpha=.15); ax.set_axisbelow(True)
        ax.set_xlabel('Share of judged recordings (%)')
    axes[0].set_title('Detection by injected fault\nHigher means more detections')
    axes[1].set_title('False alarms on unique clean controls\nHigher means more false alarms')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=2, frameon=False)
    fig.suptitle('Full-battery regression comparison at the frozen operating point', fontweight='bold')
    fig.tight_layout(rect=(0,.05,1,.95))
    figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure, dpi=180, bbox_inches='tight'); plt.close(fig)
    evidence = dict(decisions_sha256=hashlib.sha256((audit/'decisions.csv').read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        figure_sha256=hashlib.sha256(figure.read_bytes()).hexdigest())
    (audit/'report_manifest.json').write_text(json.dumps(evidence, indent=2)+'\n')
    print('BY FAULT\n'+per_fault.to_string())
    print('\nUNIQUE CLEAN CONTROLS\n'+per_car.to_string())
    print('\nBENCHMARK METRICS\n'+metric_table.to_string(index=False))
    print('\nADDED ALARMS\n'+changes[['suite','car','trip_id','scenario','baseline_clean','combined_clean']].to_string(index=False))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--audit', type=Path, required=True)
    ap.add_argument('--figure', type=Path, default=Path('docs/assets/thermostat_final/06_full_battery.png'))
    args = ap.parse_args()
    summarize(args.audit, args.figure)

"""Inspect BlueDriver logs with the pretrained model (no calibrated verdicts)."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from .config import CH, DT, W, RANGE, MODELS
from .data import resample
from .features import PHYSICS, windows
from .model import VirtualSensorNet, channel_errors

COLUMNS = {
    'RPM': 'Engine RPM (rpm)', 'SPEED': 'Vehicle Speed (km/h)',
    'LOAD': 'Calculated Engine Load Value (%)',
    'THROTTLE': 'Absolute Throttle Position (%)',
    'ECT': 'Engine Coolant Temperature (°C)',
    'IAT': 'Intake Air Temperature (°C)',
    'MAP': 'Intake Manifold Pressure (kPa)',
    'MAF': 'Mass Air Flow Rate (g/s)',
    'STFT1': 'Short Term Fuel Trim Bank 1 (%)',
}

def load_bluedriver(path, runtime_offset=None):
    """Keep missing channels masked; split gaps and clock resets before resampling."""
    path = Path(path)
    with path.open('rb') as handle:
        encoding = 'utf-16' if handle.read(2) in (b'\xff\xfe', b'\xfe\xff') else 'utf-8-sig'
    raw = pd.read_csv(path, encoding=encoding, skiprows=2, skipinitialspace=True)
    raw.columns = raw.columns.str.strip()
    if 'Time(s)' not in raw:
        raise ValueError(f'{path.name}: expected a BlueDriver Time(s) header on line 3')
    d = pd.DataFrame({'t': pd.to_numeric(raw['Time(s)'], errors='coerce')})
    metrics = {}
    for channel in CH:
        column = COLUMNS[channel]
        values = pd.to_numeric(raw[column], errors='coerce') if column in raw else pd.Series(np.nan, index=raw.index)
        valid = values.between(*RANGE[channel])
        d[channel] = values.where(valid)
        metrics[channel] = dict(column_present=column in raw, valid_rows=int(valid.sum()),
                                missing_or_unparseable_rows=int(values.isna().sum()),
                                out_of_range_rows=int((values.notna() & ~valid).sum()))
    moving = d.SPEED > 5
    rpm_zero_moving = int(((d.RPM == 0) & moving).sum())
    # Without an engine-start time, hold context at its 30-minute cap.
    # This is an explicit exploratory assumption, not inferred engine runtime.
    d['runtime'] = 1800.0 if runtime_offset is None else d.t + runtime_offset
    d['code'] = 0.0  # format has no labels; placeholder for the window builder only
    d['row'] = np.arange(len(d))
    invalid_time = int(d.t.isna().sum())
    d = d.dropna(subset=['t'])
    boundary = d.t.diff().isna() | (d.t.diff() <= 0) | (d.t.diff() > 60)
    trips = []
    for _, group in d.groupby(boundary.cumsum()):
        trip = resample(group)
        if trip is not None:
            if runtime_offset is None:
                trip['ctx'][:] = 30
            trips.append(trip)
    return trips, dict(source_file=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                       raw_rows=len(raw), invalid_time_rows=invalid_time,
                       duration_seconds=float(d.t.max() - d.t.min()) if len(d) else None,
                       usable_segments=len(trips), sensors=metrics,
                       rpm_zero_while_speed_above_5_kph_rows=rpm_zero_moving,
                       runtime_context='assumed 30 minutes (unknown engine start)' if runtime_offset is None else f'log time + {runtime_offset} seconds',
                       ground_truth_available=False)


def load_checkpoint(path):
    # Only load trusted checkpoints: this legacy repository checkpoint includes NumPy objects.
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    if checkpoint['channels'] != CH or checkpoint['physics'] or checkpoint['dt_seconds'] != DT or checkpoint['window'] != W:
        raise ValueError('Expected the standard nine-channel, no-physics, 5-second checkpoint')
    mu, sd = np.asarray(checkpoint['mu']), np.asarray(checkpoint['sd'])
    if mu.shape != (len(CH),) or sd.shape != mu.shape or not np.isfinite(mu).all() or not np.isfinite(sd).all() or (sd <= 0).any():
        raise ValueError('Invalid checkpoint normalization')
    model = VirtualSensorNet(C=len(CH), K=0)
    model.load_state_dict(checkpoint['state_dict'])
    return model, mu, sd


def analyze(paths, checkpoint_path, output, runtime_offset=None, condition='unknown'):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    previous_physics = PHYSICS[:]
    PHYSICS.clear()
    try:
        model, mu, sd = load_checkpoint(checkpoint_path)
        reports, frames = [], []
        for file_index, path in enumerate(paths):
            trips, report = load_bluedriver(path, runtime_offset)
            report['user_reported_condition'] = condition
            data = windows(trips, mu, sd, stride=2)
            report['usable_windows'] = 0 if data is None else len(data['x'])
            if data is not None:
                errors = channel_errors(model, data)
                frame = pd.DataFrame(errors, columns=[f'{c}_normalized_mse' for c in CH])
                frame.insert(0, 'window_end_seconds', (data['start'] + W) * DT)
                frame.insert(0, 'segment', data['trip'])
                frame.insert(0, 'file_index', file_index)
                frame.insert(0, 'source_file', Path(path).name)
                frames.append(frame)
                report['median_normalized_mse'] = {c: float(np.nanmedian(errors[:, j])) if np.isfinite(errors[:, j]).any() else None for j, c in enumerate(CH)}
                fig, ax = plt.subplots(figsize=(10, 4))
                for segment in np.unique(data['trip']):
                    mask = data['trip'] == segment
                    for j, channel in enumerate(CH):
                        if np.isfinite(errors[mask, j]).any():
                            ax.plot(frame.loc[mask, 'window_end_seconds'] / 60, np.log10(errors[mask, j] + 1e-6), label=f'{channel} (segment {segment})')
                ax.set(xlabel='Minutes since segment start', ylabel='log10 normalized reconstruction MSE',
                       title=f'{Path(path).name}\nExploratory errors; no calibrated alarm threshold')
                ax.legend(fontsize=7, ncol=4)
                fig.tight_layout()
                fig.savefig(output / f'log_{file_index + 1}_errors.png', dpi=140)
                plt.close(fig)
            reports.append(report)
        result = dict(mode='exploratory pretrained reconstruction; no fine-tuning or calibration',
                      checkpoint_sha256=hashlib.sha256(Path(checkpoint_path).read_bytes()).hexdigest(),
                      accuracy=None, precision=None, recall=None, F1=None, AUROC=None,
                      limitation='No verified labels or independent healthy calibration logs; errors are not fault probabilities.',
                      logs=reports)
        (output / 'report.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
        pd.concat(frames, ignore_index=True).to_csv(output / 'window_errors.csv', index=False) if frames else (output / 'window_errors.csv').write_text('source_file,file_index,segment,window_end_seconds\n')
        return result
    finally:
        PHYSICS[:] = previous_physics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('logs', nargs='+', type=Path)
    parser.add_argument('--checkpoint', type=Path, default=Path(MODELS) / 'obd_pretrained.pt')
    parser.add_argument('--output', type=Path, default=Path('artifacts/real_data'))
    parser.add_argument('--runtime-offset', type=float, help='Known engine runtime in seconds at the start of each log; omit if unknown')
    parser.add_argument('--condition', choices=['unknown', 'healthy', 'fault'], default='unknown', help='User-reported condition for all supplied logs; recorded as metadata only')
    args = parser.parse_args()
    if args.runtime_offset is not None and args.runtime_offset < 0:
        parser.error('--runtime-offset must be nonnegative')
    result = analyze(args.logs, args.checkpoint, args.output, args.runtime_offset, args.condition)
    for report in result['logs']:
        print(f"{report['source_file']}: {report['usable_windows']} windows; {report['rpm_zero_while_speed_above_5_kph_rows']} moving rows with zero RPM")
    print(f'Exploratory outputs: {args.output}; accuracy/AUROC unavailable without ground truth.')


if __name__ == '__main__':
    main()

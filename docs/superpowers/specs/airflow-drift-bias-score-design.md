# Airflow drift: signed-error (bias) score — design

Issue #33, branch `alex/33-airflow-drift`.

## Problem

The model misses small, steady airflow errors. Current results (fine-tuned model, final "stuck check" scoring,
90th-percentile cutoff, drive rule 30%, mean over the 8 held-out cars):

| Fault | AUROC | Drives caught |
|---|---|---|
| MAF drifts to -30% over the drive | 0.61 | 36% |
| MAP reads +10 kPa (sensor only) | 0.58 | 12% |
| MAF under-reads 25% (sensor only) | 0.78 | 58% |

Cause: each window is scored by the **squared** prediction error (`channel_errors`, `model.py`). A steady one-way
bias then looks like ordinary noise, and MAF/MAP are noisy sensors. The 2-minute average is too short to separate
the two. The 10-minute "two-speed" average was tried and dropped (it averaged squared error too, doubled false alarms,
and hurt when a fault started mid-drive).

## Goal and success bar

Catch more airflow drift/offset faults **without** more false alarms. The new scoring wins if, compared with
"stuck check" **in the same run** (identical trained models):

- MAF drift and MAP offset (sensor-only) AUROC and drive detection each improve by more than 0.03, and
- false alarms and per-drive F1 hold (no worse than 0.03) on both the main evaluation and the synthetic test.

Out of scope: which sensor is blamed for the realistic air faults, physics inputs (follow-up option B), any change to
training.

## Design

### 1. Signed error per window

For each 1-minute window and each sensor (hidden and predicted from the others, as now), also compute the mean
**signed** error, prediction minus reading, in normalized units. `channel_errors` and `ridge_errors` return it next to
the squared error, so the ridge baseline is scored the same way. Callers that only use the squared error keep working.

### 2. Per-car calibration

On the car's calibration drives (minutes after the warm-up), take each sensor's mean and standard deviation of the
signed error. New errors become z = (error − mean) / std. This removes a steady error the model has on that car
when it is healthy.

### 3. Two-sided CUSUM through the drive

Per drive and per sensor, two running totals, both starting at 0 and starting only after the warm-up grace:

    up   = max(0, up   + z − k)      # sensor reads lower than predicted (prediction − reading > 0)
    down = max(0, down − z − k)      # sensor reads higher than predicted

`k` is the allowance per window (`--cusum-k`, default 0.5). A sensor's bias score is max(up, down); the window's bias
score is the worst sensor, which is also the sensor blamed. Missing sensor values leave the totals unchanged.

### 4. Combined score

The bias score is rescaled with the existing `unit_scale` on calibration scores (median 0, 99th percentile 1), like
the stuck check. Window score = max(usual 2-minute score, stuck check, bias score). The alarm line (90th percentile of
calibration scores), warm-up grace and drive rule are unchanged.

Risk: neighbouring windows overlap (stride 10 s, length 1 min), so their errors are correlated and the totals can
creep up on healthy drives. Rescaling on calibration drives sets the line from each car's own normal growth; if false
alarms still rise, raise `k`.

## Code changes

- `model.py`: `channel_errors` and `ridge_errors` also return the per-window signed error.
- `scoring.py`: a small `cusum` function (running totals for one drive), and a `use_bias` option on
  `slow_stuck_scorer` that adds the bias score to the maximum. Off = today's behaviour.
- `evaluate.py`: a fourth scoring, `"stuck check + bias"`, next to the existing three; `--cusum-k` option.
- `synthetic.py`: a matching variant `"bias"`; `--cusum-k` option.
- `config.py`: `FINAL_SCORING` stays `"stuck check"` until the results pass the bar (separate decision).

## Testing

- Unit tests for `cusum`: a constant offset grows steadily; zero-mean noise stays small; an offset that starts
  mid-drive grows only after it starts; NaN values leave the totals unchanged.
- The existing smoke tests (`pytest`) pass and run the new scoring.

## Experiment

1. Quick check on the cars with MAF/MAP: `python -m obdfault.evaluate --cars car1 exp2_car car8 --skip-final --tag _bias3`.
   Catch obvious problems; tune `k` if false alarms jump.
2. Full main evaluation: `python -m obdfault.evaluate --skip-final --tag _bias`.
3. Synthetic test: `python -m obdfault.synthetic --tag _bias`.
4. Compare "stuck check + bias" with "stuck check" in the same files, against the success bar above. The `_mine`
   baseline files are a check that nothing else moved.

If it wins: a separate change switches `FINAL_SCORING` and updates the README and the build guide. If not: record
why in the guide (Step 10), then try option B (bias score + physics inputs).

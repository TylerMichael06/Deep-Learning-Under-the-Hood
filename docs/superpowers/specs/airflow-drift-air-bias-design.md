# Airflow drift: air-sensor bias score with a leave-one-out alarm line — design

Issue #33, branch `alex/33-airflow-drift`. Follow-up to `airflow-drift-bias-score-design.md` (that version was dropped:
see the build guide, Step 10, item 5).

## Why the first version failed

The first bias score (CUSUM of the signed error over all 9 sensors) caught far more airflow faults but raised false
alarms on clean drives from 9% to 35% per drive (main evaluation). A diagnostic on 3 cars (car1, car8, exp2_car) found:

- every healthy drive has its own steady signed error: the spread of per-drive averages is 0.3–2.0 times the
  window-to-window spread, so a running total grows on healthy drives too;
- the largest healthy offsets were on sensors #33 is not about (IAT, throttle, coolant, load), and the score took the
  worst of all 9 sensors;
- the calibration scores used for the alarm line came from the same drives the mean/std were fitted on, so they were
  too small and the line too low.

This matches the literature: residual offsets learned on one dataset do not hold on others (Berton, Frisk & Nyberg 2004,
truck engine air path), and CUSUM limits computed on the fitting data give too many false alarms (Masarotto & Capizzi;
bootstrap / held-out limits).

## Spike evidence (throwaway, 3 cars, same trained models for every row)

| | stuck check | CUSUM all sensors (dropped) | CUSUM air only, leave-one-out line |
|---|---|---|---|
| mean AUROC, all faults | 0.81 | 0.76 | 0.89 |
| MAP +10 kPa, drives caught | 25% | 63% | 73% |
| MAF drift, drives caught | 76% | 95% | 95% |
| per-drive F1 | 0.72 | 0.76 | 0.79 |
| per-drive false alarms | 18% | 35% | 24% |
| per-minute false alarms | 15% | 38% | 21% |

## Goal and success bar (unchanged)

The new scoring wins if, compared with "stuck check" in the same run (identical trained models):

- MAF drift and MAP offset (sensor-only) AUROC and drive detection each improve by more than 0.03, and
- false alarms and per-drive F1 hold (no worse than 0.03) on both the main evaluation and the synthetic test.

Out of scope: the drive-average score (spike variant 2), physics inputs, which sensor is blamed for the realistic air
faults, any change to training.

## Design

### 1. Air sensors only

`scoring.py` gets `BIAS_SENSORS = ["MAF", "MAP", "LOAD", "STFT1"]` (next to `FLAT_SENSORS`). `bias_scorer` only scores
those sensors: other sensors never add to the bias score and are never blamed by it. Sensors in the list that a car
does not report are skipped, as now.

### 2. Leave-one-drive-out calibration scores

`bias_scorer` still returns `(score, s_cal)`:

- `score(B, D)` (test drives) standardizes with the mean/std of **all** calibration drives, as now.
- `s_cal` (calibration drives) is computed leave-one-out: each calibration drive is scored with the mean/std learned
  from the **other** calibration drives. With only one calibration drive, `s_cal` falls back to the in-sample scores.

`s_cal` feeds `unit_scale` and therefore the shared 90th-percentile alarm line, so the line now reflects how the score
behaves on drives it was not fitted on.

### 3. Unchanged

The CUSUM (`cusum`), the allowance `k = 0.5` (`--cusum-k`), starting after the warm-up grace, and folding the rescaled
bias score into the maximum with the stuck check. The scoring keeps its name `"stuck check + bias"` (evaluate) and
variant `"bias"` (synthetic): it is replaced in place, both pick up the change through `bias_scorer`. `FINAL_SCORING`
stays `"stuck check"`.

## Code changes

- `scoring.py`: `BIAS_SENSORS`; `bias_scorer` restricted to those sensors and returning leave-one-out calibration scores.
- `tests/test_bias.py`: the `bias_scorer` tests move to full 9-sensor arrays (the old ones use 2 columns, which no longer
  maps onto sensor names); new tests below.
- No change to `evaluate.py`, `synthetic.py`, `model.py`.

## Testing

- An offset on MAF is flagged and MAF is blamed (existing test, rewritten for 9 sensors).
- An offset on a non-air sensor (IAT) gives a bias score of 0.
- A calibration drive with its own steady offset scores higher leave-one-out than in-sample.
- With one calibration drive, `s_cal` equals the in-sample scores.
- The warm-up test and the `cusum` tests still pass; the smoke tests (`pytest`) pass.

## Experiment

1. Quick check: `python -m obdfault.evaluate --cars car1 exp2_car car8 --skip-final --tag _airbias3`.
2. Full main evaluation: `python -m obdfault.evaluate --skip-final --tag _airbias`.
3. Synthetic test: `python -m obdfault.synthetic --tag _airbias`.
4. Compare "stuck check + bias" with "stuck check" in the same files against the success bar.

Record the outcome in the build guide (Step 10) either way. Switching `FINAL_SCORING` is a separate decision.

## Follow-ups if false alarms are still too high

- Calibrate the bias score's mean/std per operating region (e.g. by MAF or load level) instead of once per car, as the
  "statistical charts" of Berton, Frisk & Nyberg (2004) did for a truck air path (about 30% smaller detectable faults at
  the same false-alarm rate there; expect less here, since the network already sees the operating state).
- The drive-average score (spike variant 2): best separation in the spike (mean AUROC 0.92) but its line cannot be set
  reliably from ~7 calibration drives; needs a bootstrap line or a drive-level alarm rule.

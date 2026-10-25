# Airflow drift: bias score with drive-to-drive confirmation — design

Issue #33, branch `alex/33-airflow-drift`. Follow-up to `airflow-drift-air-bias-design.md` (that version was not
adopted: see the build guide, Step 10, item 6). Goal of this round: **fewer false alarms**, keeping the airflow gains.

## What the spikes showed (throwaway, cached models, 2 training seeds, 8 Kaggle cars, 74 clean test drives)

- The bias score's misses and false alarms had three causes: MAF errors in g/s (a dirty MAF reads a fixed *share* low,
  so the error is tiny at idle), short-term fuel trim (shifts between healthy drives; one clean car9 drive sat 1.9 sd
  off), and cars with 2 calibration drives (car4: 71% minute false alarms).
- Most remaining false-alarm drives come from the regular 2-minute score, not the bias score (5 of 8).
- Real OBD-II systems confirm a fault on a later drive before lighting the lamp ("two-trip" logic: pending code first;
  Toyota service manual, CARB 1968.2). Confirming on 2 of the last 3 drives cut false-alarm drives to 1 of 74.
- Not kept: scoring only steady/warm windows (production MAF monitors' enable conditions) and a one-sided MAF check
  (no gain); a per-car allowance from drive-to-drive spread (failed on the second seed).

| clean drives wrongly flagged / F1 / MAF drift drives / MAP +10 drives (seed 0; seed 1) | |
|---|---|
| stuck check (final model today) | 10/74, 0.74, 39%, 17% ; 9/74, 0.74, 25%, 18% |
| new bias score, one-drive rule | 8/74, 0.81, 60%, 55% ; 9/74, 0.80, 57%, 56% |
| **new bias score, bias alarms confirmed (this design)** | **7/74, 0.80, 56%, 44% ; 7/74, 0.79, 53%, 46%** |
| new bias score, every alarm confirmed (reported only) | 1/74, 0.77, 47%, 37% ; 1/74, 0.76, 47%, 43% |

## Design

### 1. Bias score (replaces "stuck check + bias" in place)

- `BIAS_SENSORS = ["MAF", "MAP", "LOAD"]` (fuel trim dropped).
- MAF's signed error is a share of the predicted airflow: window mean of (prediction − reading) in g/s divided by the
  window mean prediction in g/s, floored at 2 g/s (about idle). `channel_errors` and `ridge_errors` take
  `scale=(mu, sd)` to do this; other sensors are unchanged.
- `MIN_BIAS_DRIVES = 3`: with fewer calibration drives `bias_scorer` returns `(None, None)` and the scoring falls back
  to exactly the stuck check (no clipping of scores).
- Unchanged: CUSUM with k = 0.5 after the 5-minute warm-up, leave-one-drive-out calibration scores, one shared alarm
  line (90th percentile of calibration scores) for the regular and bias parts.

### 2. Confirmation scoped to bias alarms

Per drive (judged drives only, in driving order):
- **regular verdict**: FAULT if > 30% of its minutes have the regular part (2-minute score / stuck check) over the
  shared line;
- **combined verdict**: FAULT if > 30% of its minutes have the combined score over the line (today's rule);
- **final verdict = regular OR (combined AND combined was FAULT on at least 2 of the last 3 drives)**.

A fault the regular score sees is called on the first drive, as today; a drive that is FAULT only because of the bias
score waits for a second drive. For scorings without a bias part, regular = combined, so the verdict is the one-drive
rule. `score(E, D, trips)` called without `B` returns the regular part.

`scoring.py` gets `confirm(flags, n=3, k=2)` and `confirm_bias(regular, combined)`.

### 3. Reported, not used

The evaluation also reports "every alarm confirmed on 2 of the last 3 drives" (`confirm(combined)`) for all scorings,
so the team can decide on it later. `FINAL_SCORING` stays "stuck check"; the saved model is unchanged.

### 4. KIT car (opt-in, local only)

`load_kit(path)` reads the KIT Automotive OBD-II Dataset zip (Marc Weber, KIT 2023, DOI 10.35097/1130, CC BY 4.0):
one car, 81 drives over 9 months, 7 of our 9 sensors (no load, no fuel trim). It lives in
`data/public/kit-obd2/OBD-II-Dataset.zip` (git-ignored, never committed); `data/README.md` says where to download it.
`python -m obdfault.evaluate --kit` adds it as a held-out test car (`test_only`, never pretrained on), so the Kaggle cars'
results do not change. Each CSV is one drive; rows hold one signal each and Time is time of day (unwrapped past midnight).

## Success bar (compared with "stuck check" in the same run, Kaggle cars only)

Main evaluation, using the confirmed verdict for the new scoring and the one-drive rule for "stuck check":
- per-drive false alarms **lower** than "stuck check";
- per-drive F1 no worse than 0.03 below "stuck check";
- MAF drift and MAP +10 drive detection each up by more than 0.03; their minute AUROC each up by more than 0.03.

Synthetic test (faults change from drive to drive, so confirmation cannot be measured; one-drive rule): per-drive false
alarms and F1 no worse than 0.03.

The KIT car is reported separately (its own false alarms and detection), not part of the bar.

## Testing

- MAF share: equals (mean signed error × sd) / max(mean prediction in g/s, 2) for MAF; other sensors unchanged.
- `bias_scorer`: flags a MAF offset and blames MAF; ignores IAT and STFT1 offsets; LOO calibration scores;
  `(None, None)` below 3 calibration drives; waits for the warm-up.
- `slow_stuck_scorer`: with too few drives the bias variant equals the stuck check; without `B`, score = regular part.
- `confirm`: [1,0,1,1,0] → [0,0,1,1,0]; first drive never confirmed. `confirm_bias`: regular drives pass straight
  through; bias-only drives need confirmation.
- `load_kit`: a small zip with two drives (one past midnight) loads as two trips with the right sensors and order.
- Smoke tests (`pytest`) cover the new report levels.

## Experiment

1. `python -m obdfault.evaluate --kit --device mps --skip-final --tag _confirm`
2. `python -m obdfault.synthetic --device mps --tag _confirm`
3. Compare against the bar; record in the build guide (Step 10) either way.

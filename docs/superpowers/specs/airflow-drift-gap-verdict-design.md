# Airflow drift: MAF drive gap in the drive verdict — design

Issue #33, branch `alex/33-airflow-drift` (with Tyler's PR #42 merged). Follow-up to
`airflow-drift-confirmed-bias-design.md`.

## Why

PR #42 added a per-drive MAF check (`drive_gaps`, `gap_z`: measured ÷ predicted airflow over a drive, compared with the
car's calibration drives, FAULT if more than `GAP_Z = 3` normal spreads low) and a trend check across drives. Both only
feed a side report (`obd_results_drive_gap.csv`); the HEALTHY/FAULT drive verdict still comes from the minute score.

A throwaway spike (cached models, 2 training seeds, 8 Kaggle cars + the KIT car) adding the gap check to the drive
verdict of the final scoring ("stuck check"):

| seed 0 ; seed 1 | stuck check | stuck check OR MAF gap |
|---|---|---|
| MAF drift caught | 37% ; 29% | 76% ; 74% |
| MAF dirt buildup (across drives) caught | 37% ; 24% | 55% ; 51% |
| clean drives flagged, main / synthetic / KIT | 9/74, 3/15, 3/31 ; 8/74, 2/15, 3/31 | 9/74, 3/15, 3/31 ; 9/74, 2/15, 3/31 |

The MAF parts of the earlier bias score (MAF error as a share of predicted airflow, MAF in `BIAS_SENSORS`) do the same
job less well (KIT: 9-10 of 31 clean drives flagged) and are removed so MAF is judged one way.
Out of scope: MAP (issue #32), the season / air-density effect (next step), changing `FINAL_SCORING`.

## Design

### 1. MAF gap in the drive verdict

- `scoring.py`: `MIN_GAP_DRIVES = 3`; the gap check needs at least 3 calibration drives with a usable gap (PR #42 used 2),
  for both the verdict and Tyler's report.
- `evaluate.py`: per car (fine-tuned model, cars with MAF), compute the drive gap z-scores once for calibration, clean and
  every fault set (`maf_gap_scores`); `drive_gap_report` reuses them instead of recomputing.
- New drive verdict, per fault and drive rule, fine-tuned method only: **FAULT if the minute rule says FAULT or the MAF gap
  z-score is above `GAP_Z`**. Drives without a gap value (no MAF, too few windows) keep the minute verdict. Saved as
  `gap_TP/FN/FP/TN` and `gap_detection` columns; `binary_report` adds the level "drive (>30%) or MAF drive gap" where
  those columns exist (other methods leave them empty).
- `synthetic.py`: each test drive also gets its MAF gap verdict (`gap_fault` in the answer key); a new variant
  "stuck check + MAF drive gap" calls a drive FAULT if the stuck-check minute rule or the gap check says so. Minute-level
  numbers are unchanged (the gap check is per drive).

### 2. Remove the MAF bias pieces

- `model.py`: drop `signed_mean`, `MAF_FLOOR` and the `scale` argument of `channel_errors` / `ridge_errors`.
- `scoring.py`: `BIAS_SENSORS = ["MAP", "LOAD"]`. The "stuck check + bias" scoring and its confirmation stay as an
  experiment (MAP belongs to issue #32).
- `evaluate.py`, `synthetic.py`: stop passing `scale`.

## Success bar (same run, compared with the stuck check's one-drive verdict)

- MAF drift and MAF dirt buildup drive detection each up by more than 0.03 (Kaggle cars);
- per-drive false alarms and F1 no worse than 0.03 on the main test (Kaggle cars) and the synthetic test;
- KIT car reported separately.

## Testing

- The gap verdict ORs with the minute verdict and ignores drives without a gap value (unit test on a helper).
- `MIN_GAP_DRIVES`: with 2 calibration drives there is no gap check.
- Bias tests move from MAF to MAP; `BIAS_SENSORS == ["MAP", "LOAD"]`; the MAF-share test is removed.
- Smoke tests: the new binary level exists for the fine-tuned method; the synthetic variant runs.

## Experiment

`python -m obdfault.evaluate --kit --device mps --skip-final --tag _gap` and
`python -m obdfault.synthetic --device mps --tag _gap`; record in the build guide (Step 10).

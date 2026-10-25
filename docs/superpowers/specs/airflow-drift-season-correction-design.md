# Airflow drift: season (air-density) correction for the MAF drive gap — design

Issue #33, branch `alex/33-airflow-drift`. Follow-up to `airflow-drift-gap-verdict-design.md` (the MAF drive gap from
PR #42 now counts toward the drive verdict).

## Problem

Cold air is denser, so at the same engine state more air mass flows and the MAF reads higher. The pretraining cars never
see cold air (intake temperature 27–58 °C, 5th–95th percentile, on every Kaggle car); the KIT car goes down to −4.5 °C.
On the KIT car the drive gap (measured ÷ predicted airflow − 1) falls by 0.27–0.37% per °C of intake temperature (two
training seeds), close to the ideal-gas value of 0.35% per °C: the model barely captures the effect.

The gap check only alarms on "reads low", so winter drives (MAF reads high) cause no false alarms today. The reverse
case does: a car calibrated in winter and driven in summer reads low. Throwaway spike, KIT fine-tuned and calibrated on
Feb–Mar 2018 drives (median intake 10 °C) and tested on Jul–Aug 2017 drives (27 °C): **5 of 23 and 2 of 23** clean
summer drives flagged (seeds 0 / 1), and the trend check false-alarms by drive 3–4 on both seeds.

## Spike results (each window's measured airflow × (T_window / T_ref) ** α, T in Kelvin, T_ref = calibration median)

Winter calibration → summer test, seed 0 ; seed 1:

| | α = 0 (today) | **α = 0.5** | α = 1 (full ideal gas) |
|---|---|---|---|
| clean drives flagged | 5/23 ; 2/23 | **0/23 ; 0/23** | 0/23 ; 0/23 |
| trend check false alarm | yes ; yes | yes ; no | no ; no |
| dirt buildup caught | 83% ; 87% | **78% ; 78%** | 52% ; 74% |
| MAF drift caught | 100% ; 100% | **87% ; 96%** | 35% ; 83% |

The full correction over-corrects (healthy summer drives end up ~2.5 spreads "high"): after fine-tuning, the model has
already learned part of the effect. With α = 0.5, KIT in normal order catches more dirt (buildup 68% → 71% and
45% → 68%; drift 87% → 87% and 39% → 90%) with no clean drive flagged, and the Kaggle MAF cars are unchanged.
α = 0.5 is chosen on the only car with cold data; it is presented as "half the ideal-gas correction".

**Outcome:** the bar failed (see `docs/model_build_guide.md`, Step 10, item 9). The correction ships off
(`SEASON_ALPHA = 0.0`); `--season-alpha 0.5` reproduces the test.

## Design

### 1. Correction in `drive_gaps` (`scoring.py`)

- `SEASON_ALPHA = 0.5` and `IAT_KELVIN = 273.15`.
- `intake_ref(D, mu, sd, use)`: median intake temperature in Kelvin over the windows in `use` that have an intake
  reading (from `D["x"]` / `D["m"]`, column `IAT`), or `None` if there are none.
- `drive_gaps(P, D, c, mu, sd, use, min_windows, floor=0.5, t_ref=None, alpha=SEASON_ALPHA)`: if `t_ref` is given and
  `alpha != 0`, every reading of sensor c is multiplied by ((IAT + 273.15) / t_ref) ** alpha, row by row; rows of windows
  without an intake reading keep factor 1. With `t_ref=None` (or `alpha=0`) the result is exactly today's.

### 2. Wiring (fine-tuned model, everywhere the gap is used)

- `evaluate.maf_gap_scores`: `t_ref = intake_ref(Dcal, ...)` over the calibration windows after the warm-up, passed
  to every `drive_gaps` call (calibration, clean, fault sets). The drive verdict, Tyler's report and the trend check all
  use these corrected gaps.
- `synthetic.test_car`: the same for its `gap_fault`.
- Both CLIs get `--season-alpha` (default `SEASON_ALPHA`; 0 = no correction), so the same run can be repeated without
  the correction. Runs with the same seed and device give identical models, so the two runs compare fairly.

### 3. Winter-calibrated KIT car (`--kit-winter`)

- `data.kit_winter_first(kit_trips, unit="kit_winter")`: copies of the KIT drives where Feb–Mar 2018 drives come first,
  April drives are dropped, and the latest Jul–Aug 2017 drives are kept (dropping the earliest) until their driving time
  is at most 0.6 × the winter driving time; the kept summer drives are moved one year later (`order` + 365 days), so
  `prepare_car` fine-tunes and calibrates on winter drives and tests on summer drives. Unit renamed, `test_only`.
- `evaluate --kit-winter` adds this car (independent of `--kit`; both can be used together). Reported separately.

## Success bar (run with `--season-alpha 0.5` against `--season-alpha 0`, otherwise identical)

Using the "drive (>30%) or MAF drive gap" verdict of the fine-tuned model:
- Kaggle cars (main test) and the synthetic test: clean drives flagged no more than today; MAF fault detection no more
  than 0.03 below today;
- KIT car: no more clean drives flagged than today;
- winter-calibrated KIT car: fewer clean drives flagged than today.

## Testing

- `drive_gaps` with `t_ref` equal to the drives' intake temperature: unchanged; `alpha=0` or `t_ref=None`: unchanged.
- A drive with cold intake air and MAF reading high by the density ratio: corrected gap closer to 0 than uncorrected.
- `intake_ref`: median Kelvin of the used windows; `None` without intake readings.
- `kit_winter_first`: winter drives first, no April drives, summer drives after the winter ones, summer time ≤ 0.6 ×
  winter time, unit renamed, `test_only`.
- Smoke tests pass with the new arguments.

## Experiment

1. `python -m obdfault.evaluate --kit --kit-winter --device mps --skip-final --season-alpha 0 --tag _season0`
2. `python -m obdfault.evaluate --kit --kit-winter --device mps --skip-final --tag _season`
3. `python -m obdfault.synthetic --device mps --season-alpha 0 --tag _season0` and `... --tag _season`
4. Compare against the bar; record in the build guide (Step 10) either way.

Out of scope: MAP (issue #32), learning the temperature slope per car, model changes (density as an input), changing
`FINAL_SCORING`.

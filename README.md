# OBD-II Fault Detection (DSCI 566)

A small PyTorch **Transformer** that watches a car's OBD-II sensor readings and labels every minute, and every drive,
**HEALTHY** or **FAULT**. When it raises an alarm, it also names the sensor that looks wrong.
Model by Hoda; step-by-step details in [`docs/model_build_guide.md`](docs/model_build_guide.md).

## How it works

1. It reads 1 minute of driving at a time: 9 standard OBD-II sensors (RPM, speed, engine load, throttle, coolant temp,
   intake air temp, manifold pressure, mass air flow, short-term fuel trim) plus minutes since engine start.
2. It hides one sensor and predicts it from the others, for every sensor in turn ("virtual sensors").
3. It learned what "normal" looks like from 15 healthy cars in the Kaggle OBD-II dataset.
4. A prediction that is far off means something doesn't fit. That gap is the alarm score.
5. A stuck-sensor check flags coolant/intake-temp readings frozen for longer than the car ever showed normally.
6. A minute is FAULT if its score crosses the car's alarm line (the "Relaxed" line: the 90th percentile of the car's
   own normal scores). A drive is FAULT if more than 30% of its minutes are flagged, ignoring the first 5 minutes
   (warm-up). Drives with less than 1 minute after the warm-up (~6 min in total) are marked TOO SHORT to judge.

Each car is compared with itself: the model fine-tunes on the car's first 40% of driving, sets its alarm line on the
next 20%, and is tested on the last 40%.

## Results (cars the model never trained on, with faults added)

| Test | AUROC | Precision | Recall | F1 | False alarms |
|---|---|---|---|---|---|
| Main evaluation (8 cars), per drive | 0.81 | 0.88 | 0.67 | 0.76 | 9% |
| Synthetic test (67 drives), per drive | 0.75 | 0.89 | 0.74 | 0.81 | 20%* |

Scores are on drives long enough to judge (about 1 in 4 test drives is too short).
\* 3 of only 15 judged clean drives in the synthetic test.

Baselines (same network without fine-tuning, per-car linear model): AUROC 0.77. Results vary by about ±0.03 between runs.
Strongest on vacuum leaks and stuck coolant sensors (AUROC 0.96-0.97); weakest on small, steady airflow/pressure errors.

Full details, settings, assumptions and limitations: [`docs/model_build_guide.md`](docs/model_build_guide.md).

## Project layout

```
src/obdfault/              the code (a Python package)
  config.py                  sensors, valid ranges, timing, folders, final settings
  data.py                    read the Kaggle CSVs into drives on a 5-s grid   (guide Step 2)
  features.py                1-minute windows + normalization                  (Step 3)
  model.py                   the Transformer, training loop, ridge baseline    (Steps 4-5)
  train.py                   pretrain + fine-tune per car; the final model     (Step 5)
  scoring.py                 errors -> minute scores, alarm line, stuck check  (Steps 6-7)
  faults.py                  faults injected for testing                       (Step 8)
  evaluate.py                main evaluation                                   (Step 8)
  synthetic.py               synthetic test with an answer key                 (Step 9)
  figures.py                 report figures and the summary table
scripts/                   exploration scripts (start with explore_data.py)
tests/                     pytest, runs on made-up data (no download needed)
models/obd_pretrained.pt   the trained model (starting point for fine-tuning on a new car)
results/                   the current official numbers (spreadsheets)
figures/                   report figures
docs/model_build_guide.md  step-by-step guide to the model
data/                      public/ (downloaded, not in git) and collected/ (our car logs); see data/README.md
```

## Setup

Python 3.10 or newer.

```
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest                             # about a minute, no data needed
```

Data: download the Kaggle "OBD-II datasets" by cephasax (<https://www.kaggle.com/datasets/cephasax/obdii-ds3>) and put the
three `exp*.csv` files in `data/public/kaggle-obd2/`. That folder is ignored by git; see [`data/README.md`](data/README.md).

## Running it (from the repo root)

```
python -m obdfault.evaluate     # main evaluation + final model (~30 min on CPU)
python -m obdfault.synthetic    # synthetic fault test (~15 min)
python -m obdfault.figures      # figures and summary table
```

They read `data/public/kaggle-obd2/` unless you pass `--data <folder>`. `--quick` (evaluate, synthetic) does a fast smoke run on one car. `--skip-final` (evaluate) leaves `models/obd_pretrained.pt` alone.
Every run overwrites `results/`, `figures/` and `models/`. Use `git checkout -- results figures models` to get the committed versions back.

## Working on it as a team

- `main` is the shared base. Start every piece of work from it:
  `git switch main && git pull && git switch -c <name>/<issue>-<topic>` (e.g. `tyler/33-maf-drift`).
- Open a pull request into `main`, link its issue (`Closes #33`), get one teammate's review, and make sure `pytest` passes.
- `results/`, `figures/` and `models/` hold the numbers the report uses. Only commit new ones in a PR that changes
  the model, and say in the PR what moved.
- Exploring? Put scripts in `scripts/` and import what you need from `obdfault`
  (`python scripts/explore_data.py` is a starting point). Once something there is
  reused by the pipeline, move it into `src/obdfault/`.
- Where the open issues live in the code:

| Issue | Start in |
|---|---|
| #30 Model architecture investigation | `model.py` (new models next to `VirtualSensorNet`) |
| #32 Manifold pressure, #33 Airflow drift, #34 Thermostat warm-up | `scoring.py`, `faults.py` |
| #35 Continued vehicle data collection | `data.py` (a loader for our own car logs) |

## Next step

Test the model on our own cars' OBD-II logs: fine-tune on healthy drives, then check later drives.

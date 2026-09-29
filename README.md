# OBD-II Fault Detection (DSCI 566)

A small PyTorch **Transformer** that watches a car's OBD-II sensor readings and labels every minute, and every drive,
**HEALTHY** or **FAULT**. When it raises an alarm, it also names the sensor that looks wrong.

## How it works

1. It reads 1 minute of driving at a time: 9 standard OBD-II sensors (RPM, speed, engine load, throttle, coolant temp,
   intake air temp, manifold pressure, mass air flow, short-term fuel trim) plus minutes since engine start.
2. It hides one sensor and predicts it from the others, for every sensor in turn ("virtual sensors").
3. It learned what "normal" looks like from 15 healthy cars in the Kaggle OBD-II dataset.
4. A prediction that is far off means something doesn't fit. That gap is the alarm score.
5. A stuck-sensor check flags coolant/intake-temp readings frozen for longer than the car ever showed normally.
6. A minute is FAULT if its score crosses the car's alarm line. A drive is FAULT if more than 30% of its minutes are
   flagged, ignoring the first 5 minutes (warm-up).

Each car is compared with itself: the model fine-tunes on the car's first 40% of driving, sets its alarm line on the
next 20%, and is tested on the last 40%.

## Results (cars the model never trained on, with faults added)

| Test | AUROC | Precision | Recall | F1 | False alarms |
|---|---|---|---|---|---|
| Main evaluation (8 cars), per drive | 0.81 | 0.91 | 0.44 | 0.59 | 4% |
| Synthetic test (67 drives), per drive | 0.75 | 0.92 | 0.55 | 0.69 | 7% |

Baselines (same network without fine-tuning, per-car linear model): AUROC 0.77. Results vary by about ±0.03 between runs.
Strongest on vacuum leaks and stuck coolant sensors (AUROC 0.96-0.97); weakest on small, steady airflow/pressure errors.

Full details, settings, assumptions and limitations: [`model_build_guide.md`](model_build_guide.md).

## Files

| Path | What it is |
|---|---|
| `obd_model.py` | data loading, model, training, main evaluation, saves the final model |
| `synthetic_test.py` | synthetic fault test with an answer key |
| `make_figures.py` | all report figures and the summary table (`figures/`) |
| `obd_pretrained.pt` | the trained model (starting point for fine-tuning on a new car) |
| `*.csv` | final results |
| `model_build_guide.md` | step-by-step guide |

## Running it

Data (not included in this repo): download the Kaggle "OBD-II datasets" by cephasax,
<https://www.kaggle.com/datasets/cephasax/obdii-ds3>, and extract it to `Datasets/OBD-II datasets/` inside the repo folder (ignored by git).
Needs Python 3 (`pip install -r requirements.txt`). From the repo folder:

```
python obd_model.py --data "Datasets/OBD-II datasets"        # main evaluation + final model (~30 min on CPU)
python synthetic_test.py --data "Datasets/OBD-II datasets"   # synthetic fault test (~15 min)
python make_figures.py --data "Datasets/OBD-II datasets"     # figures and summary table
```

## Next step

Test the model on our own cars' OBD-II logs: fine-tune on healthy drives, then check later drives.

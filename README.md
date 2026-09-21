# Deep Learning Under the Hood

This project trains a PyTorch model to answer one question from engine sensor
data:

> Is there potentially a car issue: **YES** or **NO**?

The current dataset is `EngineFaultDB_Final.csv`. Found here https://github.com/leoxthomas/EngineFaultDB/blob/main/EngineFaultDB_Final.csv Its original four labels are
collapsed into a binary target:

| Original `Fault` | Meaning | Binary target |
| --- | --- | --- |
| 0 | No fault | NO |
| 1 | Rich mixture | YES |
| 2 | Lean mixture | YES |
| 3 | Low ignition voltage | YES |

The script calibrates the model's output on the validation split. It returns
**YES** when the calibrated `issue_probability` is at least **70%**, and **NO**
otherwise. The threshold can be changed, but 70% is the default saved in the
model checkpoint.

## Important interpretation

The 70% value is an **alert threshold**, not a guarantee that the model is 70%
correct in every vehicle or situation. A NO means only that the model did not
reach the threshold; it does not prove the vehicle is healthy. This prototype
must not replace a mechanic, scan-tool diagnosis, or vehicle safety procedure.

## Dataset location

The code automatically finds the downloaded CSV one directory above this
repository:

```text
/Users/anthonymartinez/Desktop/deeplearnProject/EngineFaultDB_Final.csv
```

You can also pass any location explicitly with `--data /path/to/file.csv`.

## Setup

From the repository directory:

```bash
cd /Users/anthonymartinez/Desktop/deeplearnProject/Deep-Learning-Under-the-Hood
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Apple Silicon, PyTorch uses the Mac GPU through MPS when it is available.
Otherwise the script automatically chooses CUDA or CPU.

## Inspect the data

```bash
python obd_anomaly.py inspect
```

This reports the source fault classes and the new YES/NO class counts.

## Train the binary model

```bash
python obd_anomaly.py train
```

The default training setup:

- uses all 14 numeric engine features;
- removes exact duplicate rows;
- creates 70% training, 15% validation, and 15% untouched test partitions;
- splits ordered blocks within every original fault class;
- calibrates issue probabilities on validation data;
- predicts YES at `issue_probability >= 0.70`;
- applies class-weighted loss and early stopping.

The default block split is intentionally more conservative than randomly
mixing nearby rows. Because the CSV does not include a trip/run identifier, it
still cannot guarantee that related measurements are isolated. Compare it with
a random stratified experiment if needed:

```bash
python obd_anomaly.py train --split-method random --output-dir artifacts/random
```

To use only fields closer to a typical OBD-II reader:

```bash
python obd_anomaly.py train --feature-set obd --output-dir artifacts/obd
```

The OBD feature set is `MAP`, `TPS`, `RPM`, `Speed`, `Lambda`, and `AFR`. Confirm
that a real reader supplies the same units and scaling before using its data.

## Training outputs

Training writes these files to `artifacts/` by default:

- `model.pt` — model weights, scaling values, calibration values, and threshold
- `summary.json` — configuration and headline test results
- `metrics.json` — binary metrics and classification report
- `training_history.csv` — loss and validation history by epoch
- `predictions.csv` — untouched test rows with YES/NO and probabilities
- `confusion_matrix.csv` — actual versus predicted binary decisions

The most important failure metric is `issue_recall`: the proportion of known
issue rows that produced a YES. `issue_precision` shows how often YES was right.
Accuracy alone can hide unsafe false negatives.

## Score another CSV

After training:

```bash
python obd_anomaly.py predict \
  --data /path/to/new_obd_data.csv \
  --model artifacts/model.pt \
  --output artifacts/new_predictions.csv
```

The new CSV does not need a `Fault` column, but it must contain the same feature
columns used for training. The output includes:

- `potential_issue`: YES or NO
- `issue_probability`: calibrated probability used for the decision
- `no_issue_probability`: one minus the issue probability
- `decision_confidence`: probability associated with the returned decision
- `alert_threshold`: threshold used for YES

You can test another operating threshold without retraining:

```bash
python obd_anomaly.py predict \
  --data /path/to/new_obd_data.csv \
  --model artifacts/model.pt \
  --issue-threshold 0.60 \
  --output artifacts/predictions_at_60_percent.csv
```

Lower thresholds usually catch more real issues but also create more false
alerts. Choose the final threshold using validation results and the cost of a
missed issue, not accuracy alone.

## What still needs to happen before real-world use

EngineFaultDB is a useful controlled benchmark, but it represents one lab
engine and three simulated fault types. It has no timestamps, trip identifiers,
vehicle identifiers, maintenance outcomes, or manufacturer specifications.
This model therefore detects patterns similar to this dataset; it does not yet
predict every kind of car failure.

Next, collect representative OBD trips with timestamps and vehicle/run IDs,
hold out entire vehicles or trips during testing, validate sensor units, and add
manufacturer limits as explicit features or post-model rules. Recalibrate and
retest the 70% threshold on real vehicles before treating it as meaningful.

See `plan.md` for the broader training and data-integration roadmap.

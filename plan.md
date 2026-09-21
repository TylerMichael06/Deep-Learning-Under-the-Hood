# PyTorch OBD Anomaly Detection Plan

## Goal

Build a reproducible PyTorch pipeline that combines OBD telemetry with a separate
manufacturer-specification dataset and produces explainable anomaly alerts.

The first version will be an unsupervised detector because the current OBD data
does not contain confirmed mechanical-fault labels. Its alerts will mean either:

1. The telemetry is unusual relative to learned healthy behavior.
2. A reading violates a supplied manufacturer operating limit.

The first version must not claim to diagnose a specific mechanical failure.

## Current state

- Three raw OBD experiments are available as CSV and Excel files in `archive (2)`.
- The CSV files contain approximately 57,500 nonblank observations.
- The files contain mixed decimal formats, embedded units, inconsistent column
  names, and substantial missing sensor coverage in experiment 1.
- Experiments 2 and 3 contain multiple drivers using one car. Their
  `VEHICLE_ID` values must not automatically be treated as distinct cars.
- Manufacturer specifications will be supplied as a separate dataset.
- The project contains a starter `obd_anomaly.py`, `requirements.txt`, and an
  empty manufacturer-specification template. These still need tests and a full
  training run before they should be considered complete.
- The original OBD files must remain read-only.

## Recommended approach

Use a hybrid detector:

```text
OBD telemetry ──> learned PyTorch anomaly score ──┐
                                                  ├─> final alert + explanation
Specifications ─> range/constraint score ─────────┘
```

Begin with a small row-level autoencoder to validate the pipeline. Move to a
windowed temporal model only after the baseline, data split, and evaluation
processes work correctly.

## Phase 1: Environment and reproducibility

- [ ] Create a project-local virtual environment named `.venv`.
- [ ] Install the pinned packages from `requirements.txt`.
- [ ] Verify that PyTorch detects the Apple Metal device (`mps`).
- [ ] Record Python, PyTorch, pandas, and scikit-learn versions in training output.
- [ ] Use a fixed random seed for splitting and model initialization.
- [ ] Keep generated checkpoints and score files under `artifacts/`.
- [ ] Add automated tests for parsing, splitting, specification joins, and scoring.

Suggested setup:

```bash
cd /Users/anthonymartinez/Desktop/deeplearnProject
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Phase 2: Define the data contract

Create one canonical OBD schema regardless of the source experiment.

### Identity and timing fields

```text
source_file
source_row
timestamp
trip_id
actual_vehicle_id
driver_id
make
model
model_year
engine
transmission
```

`actual_vehicle_id` and `driver_id` must be distinct. For experiments 2 and 3,
assign the shared car an actual vehicle identifier and map the existing IDs to
drivers or sessions after verifying the dataset documentation.

### Initial sensor set

```text
barometric_pressure_kpa
engine_coolant_temp_c
fuel_level_pct
engine_load_pct
ambient_air_temp_c
engine_rpm
intake_manifold_pressure_kpa
maf_g_per_s
air_intake_temp_c
speed_kph
short_term_fuel_trim_bank_1_pct
engine_runtime_seconds
throttle_position_pct
timing_advance
equivalence_ratio
dtc_count
```

Do not use latitude or longitude in the first model. Location could cause the
model to memorize routes and creates unnecessary privacy exposure.

## Phase 3: Clean and normalize the OBD files

- [ ] Remove completely blank records without modifying the raw source files.
- [ ] Convert decimal commas to decimal points.
- [ ] Strip units such as `%`, `RPM`, `km/h`, `kPa`, `C`, and `g/s`.
- [ ] Rename equivalent columns to the canonical names.
- [ ] Convert engine runtime from `HH:MM:SS` to seconds.
- [ ] Parse timestamps and preserve the original source row for traceability.
- [ ] Convert every sensor to the canonical unit.
- [ ] Treat missing readings as missing, never as numeric zero.
- [ ] Add an observed/missing mask for each model feature.
- [ ] Create trip boundaries using explicit trip identifiers when available;
  otherwise use engine-runtime resets and large timestamp gaps.
- [ ] Flag impossible values separately from manufacturer violations, such as
  negative RPM or percentages far outside their representable range.
- [ ] Produce a data-quality report by experiment, vehicle, and feature.

Acceptance checks:

- Every retained row has a source file, source row, and identity/session key.
- At least 99% of nonblank values in selected model features parse successfully.
- Units are explicit and consistent after cleaning.
- Missingness and dropped-row counts reconcile to the raw files.

## Phase 4: Prepare the manufacturer-specification dataset

Use a long-format specification table so one vehicle configuration can have many
feature limits.

```text
make
model
model_year
engine
transmission
feature
unit
minimum
maximum
condition_name
condition_min
condition_max
source_url
source_version
```

Examples of conditions include a warmed engine, an RPM band, ambient-temperature
range, or idle state. A single unconditional minimum and maximum is often too
broad for useful anomaly detection.

- [ ] Normalize make, model, engine, transmission, and units.
- [ ] Preserve the specification source and version.
- [ ] Reject duplicate specification keys unless conditions make them distinct.
- [ ] Validate that minimum is less than maximum when both exist.
- [ ] Verify that specification units match canonical OBD units.
- [ ] Report unmatched OBD vehicle configurations instead of inventing limits.
- [ ] Version the specification file used for every training run.

## Phase 5: Join specifications and engineer features

Join on the most specific reliable vehicle configuration key:

```text
make + model + model_year + engine + transmission
```

For every sensor with applicable specifications, calculate:

```text
range_position = (reading - minimum) / (maximum - minimum)
below_minimum
above_maximum
normalized_distance_outside_range
specification_available
```

Additional OBD features should include:

- First differences and rates of change.
- Rolling mean, standard deviation, minimum, and maximum.
- Engine state such as off, startup, idle, acceleration, cruise, and deceleration.
- Relationships such as RPM versus speed, MAF versus RPM/load, coolant warm-up
  rate, and fuel-trim behavior.
- A missingness mask and time since the previous valid reading.

Fit all learned transformations, including medians and scaling statistics, using
the training split only.

## Phase 6: Establish simple baselines

Before relying on a neural model, implement and save results for:

1. Manufacturer-limit violations.
2. Robust per-vehicle or per-configuration z-scores.
3. Isolation Forest on cleaned numeric features.

These baselines provide a comparison and can expose preprocessing mistakes. If a
neural model cannot outperform them on a meaningful test, keep the simpler model.

## Phase 7: Construct leakage-safe datasets

Do not randomly split individual rows.

- [ ] Resolve the true vehicle and driver identifiers first.
- [ ] Hold out complete actual vehicles when testing generalization.
- [ ] Within a vehicle, keep later trips after earlier trips.
- [ ] Keep every trip entirely within one split.
- [ ] Prevent rolling windows from crossing trip boundaries.
- [ ] Fit imputers, scalers, and thresholds using training data only.
- [ ] Record the exact vehicle and trip membership of each split.

Recommended split purposes:

```text
Training: fit preprocessing and model weights
Validation: early stopping and threshold selection
Test: final evaluation only
```

Because the data has no confirmed anomaly labels, prefer training on periods that
can reasonably be treated as healthy. Exclude known DTC periods and obvious
specification violations from healthy training data, but retain them for review.

## Phase 8: Train the first PyTorch model

### Version 1: row-level masked autoencoder

Inputs:

- Standardized sensor values.
- Observed/missing masks.
- Specification-relative features.
- Basic operating-state features.

Model:

```text
input -> dense layer -> dense bottleneck -> dense layer -> reconstructed sensors
```

Training requirements:

- Use a masked Huber or masked mean-squared reconstruction loss so missing values
  do not contribute to the loss.
- Use early stopping on held-out vehicles or trips.
- Save the best checkpoint rather than only the final epoch.
- Save feature order, scaling statistics, model hyperparameters, split IDs,
  threshold, code version, and specification version with the checkpoint.
- Track training and validation loss for every epoch.

### Version 2: temporal model

After version 1 is validated, create 30- to 60-second windows and test a small
temporal convolutional network or GRU autoencoder. Include both reconstruction
error and next-step prediction error. Use a transformer only if the dataset grows
substantially and simpler temporal models fail.

## Phase 9: Define anomaly scoring and explanations

Keep learned and rule-based scores separately visible.

```text
model_score = masked reconstruction/prediction error
spec_score = maximum normalized distance outside applicable limits
final_alert = calibrated model alert OR specification violation
```

- [ ] Choose the model threshold on validation data.
- [ ] Calibrate it to an acceptable false-alert rate per driving hour.
- [ ] Aggregate consecutive anomalous rows into one event.
- [ ] Require a minimum event duration when appropriate.
- [ ] Report the top sensor contributions to model error.
- [ ] Report the exact specification and source for every limit violation.
- [ ] Preserve separate severities for warning and critical events.

Do not hide whether an alert came from the learned model, a specification rule,
or both.

## Phase 10: Evaluate the system

The current dataset cannot support a valid mechanical-fault accuracy claim because
it lacks repair-confirmed labels.

### Evaluation possible now

- False alerts per driving hour on presumed-healthy validation trips.
- Percentage of trips with at least one alert.
- Score stability across vehicles, drivers, and operating states.
- Sensitivity to controlled synthetic faults added only to test copies.
- Manual review of the highest-scoring events.
- Comparison with the rule, z-score, and Isolation Forest baselines.

Synthetic tests should include stuck sensors, spikes, drift, increased noise,
dropouts, and physically inconsistent sensor relationships. Synthetic performance
measures sensitivity to those transformations, not real-world fault accuracy.

### Evaluation once labels are available

- Event-level precision, recall, and F1.
- False alerts per driving hour.
- Detection delay before a confirmed fault.
- Performance by make, model, engine, and fault type.
- Generalization to completely unseen vehicles.
- Calibration of alert severity.

## Phase 11: Score new OBD data

- [ ] Load the saved preprocessing state and model checkpoint.
- [ ] Validate the incoming schema and units.
- [ ] Match the appropriate manufacturer specifications.
- [ ] Score data in the same trip/window format used for training.
- [ ] Produce an event report with timestamps, vehicle identity, anomaly score,
  rule violations, top contributing sensors, and model/specification versions.
- [ ] Refuse or clearly mark scoring when required identity or unit information is
  missing.
- [ ] Monitor changes in feature distributions and missingness over time.

## Phase 12: Add supervised learning later

When repair-confirmed events or reliable fault labels are available:

- Preserve the unsupervised score and specification-rule features.
- Train a supervised event classifier to assign likely fault categories.
- Split by actual vehicle and time to avoid leakage.
- Address class imbalance with sampling or loss weighting.
- Compare the classifier against the unsupervised detector rather than replacing
  it automatically.

## Expected project outputs

```text
artifacts/<run_id>/
  model.pt
  preprocessing.json
  split_manifest.csv
  training_history.csv
  threshold.json
  scores.csv
  events.csv
  evaluation.json
  data_quality.json
```

Each run should be reproducible from its saved configuration and should identify
the exact OBD inputs and manufacturer-specification version used.

## Definition of done for the first model

The first model is ready for demonstration when all of the following are true:

- [ ] Raw data is reproducibly converted into one canonical schema.
- [ ] Driver IDs and actual vehicle IDs are correctly separated.
- [ ] Manufacturer specifications join without silent duplicate or unit errors.
- [ ] Training, validation, and test splits are vehicle/trip safe.
- [ ] A simple baseline and PyTorch autoencoder have both been evaluated.
- [ ] The selected threshold has a documented false-alert rate.
- [ ] Consecutive row alerts are aggregated into understandable events.
- [ ] Each alert states whether it came from the model, specifications, or both.
- [ ] Checkpoints contain all preprocessing and version information needed to
  reproduce scoring.
- [ ] Limitations caused by missing ground-truth fault labels are clearly reported.

## Immediate next steps

1. Verify the meaning of `VEHICLE_ID` in each experiment and create a driver/car
   mapping table.
2. Create the project-local environment and run the inspection command.
3. Test the parser against representative values from all three experiments.
4. Add and validate the manufacturer-specification dataset.
5. Generate the canonical cleaned dataset and data-quality report.
6. Establish the rule and Isolation Forest baselines.
7. Train the row-level PyTorch autoencoder.
8. Review its highest-scoring events before attempting a temporal model.

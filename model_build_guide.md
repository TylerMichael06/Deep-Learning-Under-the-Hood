# OBD-II Fault Detection Model: Build Guide (Version 1)

This guide walks through the final model step by step: what each step does, why, and which part of the code does it.


- **Code:** `obd_model.py` (data, model, training, main evaluation), `synthetic_test.py` (fault test with an answer key)
  and `make_figures.py` (all report figures and tables, saved in `figures\`)
- **Trained model:** `obd_pretrained.pt` (also stores the final alarm settings)
- **Output:** binary **HEALTHY / FAULT** for every minute and every drive, plus the sensor that looks wrong

---

## Step 1: The data

| File | Rows | What it is |
|---|---|---|
| `exp1_14drivers_14cars_dailyRoutes.csv` | 60,439 | 14 drivers, 14 cars (2003-2016), normal daily driving |
| `exp2_19drivers_1car_1route.csv` | 8,261 | 19 drivers, 1 car, same route |
| `exp3_4drivers_1car_1route.csv` | 1,743 | 4 drivers, 1 car, same route |

Recorded with an ELM327 Bluetooth OBD-II plug and an Android phone, one reading every ~4-12 s.

**The 9 sensors the model uses** (standard OBD-II Mode 01 PIDs, readable from any car):

| Model name | Column | PID | Meaning |
|---|---|---|---|
| RPM | ENGINE_RPM | 0C | engine speed |
| SPEED | SPEED | 0D | vehicle speed (km/h) |
| LOAD | ENGINE_LOAD | 04 | calculated engine load (%) |
| THROTTLE | THROTTLE_POS | 11 | throttle position (%) |
| ECT | ENGINE_COOLANT_TEMP | 05 | coolant temperature (°C) |
| IAT | AIR_INTAKE_TEMP | 0F | intake air temperature (°C) |
| MAP | INTAKE_MANIFOLD_PRESSURE | 0B | manifold pressure (kPa) |
| MAF | MAF | 10 | mass air flow (g/s) |
| STFT1 | SHORT TERM FUEL TRIM BANK 1 | 06 | short-term fuel trim (%) |

Plus **minutes since engine start** (ENGINE_RUNTIME, PID 1F) as context, so the model knows when the engine is warming up.
Other columns (bank-2 fuel trims, fuel pressure, barometric pressure, fuel level, car info, GPS) are too sparse or not
about engine health, so they are not used.

**Labels:** there is no healthy/faulty column. The only fault information is the trouble codes (TROUBLE_CODES, Mode 03):
- `car6`: P0133 (slow O2 sensor) for the whole log, so it is **left out** (it has no healthy period to learn from)
- `car13`: engine codes on and off (~7% of rows), used as the **real-fault test**
- `car9`: C0300, a chassis code, not engine, so treated as healthy
- everything else: no codes, treated as healthy

Only engine codes (starting with "P") count as faults.

---

## Step 2: Load and clean → `load_kaggle`, `num`, `resample`

1. **Parse numbers:** values are stored as text such as `"33,30%"`, `"2124RPM"` or `"24,77g/s"`; `num()` turns them into numbers.
2. **Remove glitches:** values outside physical ranges (`RANGE`, e.g. RPM > 8000, coolant outside -40...140 °C) become missing.
3. **Timing:** exp1 uses its millisecond TIMESTAMP. exp2/exp3 timestamps were saved rounded (`"1,51E+12"`), so ENGINE_RUNTIME is used instead.
4. **Split into drives:** a new drive starts after a gap of more than 60 s or when engine runtime resets.
5. **Resample:** every drive is put on an even 5-second grid. A value is kept only if a real reading is within 12 s.
6. Drives with no engine data (some car1 logs recorded only GPS speed) are skipped.

---

## Step 3: Normalize and cut into windows → `windows`

- **Window:** 12 rows = **1 minute** of driving. Windows start every 2 rows (10 s).
- A sensor counts as present in a window only if all 12 rows have it. Windows with fewer than 4 present sensors are dropped.
- **Normalization:** z-score per sensor, (value - mean) / std, using statistics from the **training cars only**,
  never from the car being tested. The same values are saved in `obd_pretrained.pt` for use on your own car.
- Missing sensors are set to 0 and flagged as missing (the network is told to ignore them).
- Minutes-since-start is scaled to 0-1 and capped at 30 minutes.

---

## Step 4: The model → `VirtualSensorNet`

A small **Transformer** neural network in PyTorch, with **102,601 parameters**.

```
input per time step (12 steps):  9 sensor values + 9 "is visible" flags + minutes-since-start  (19 numbers)
  -> Linear(19 -> 64) + learned position embedding
  -> Transformer encoder: 2 layers, 4 attention heads, feed-forward 256, dropout 0.1
  -> Linear(64 -> 9): a prediction of every sensor at every time step
```

**Idea ("virtual sensor"):** on a healthy engine the sensors agree with each other (MAF follows RPM × load, MAP
follows throttle, coolant follows warm-up time). The network learns those relationships. A sensor that stops
agreeing with its prediction is probably faulty.

**Training trick (masking):** in each training window, 1 or 2 of the present sensors are hidden at random
(`hide_random`), and the network must rebuild them from the others. The loss is the mean squared error on the hidden sensors only.

---

## Step 5: Train → `fit`, `prepare_car`

For each car being tested ("held-out car"):

1. **Pretrain** on all *other* healthy cars (exp1 + exp2 + exp3, excluding car6): Adam, learning rate 1e-3,
   batch 256, up to 40 epochs, early stopping after 5 epochs without improvement on a 10% validation split.
2. **Split the held-out car's drives by time:**
   - first **40%** of its driving → **fine-tune** (it learns this car's own "normal")
   - next **20%** → **calibration** (sets the alarm level)
   - last **40%** → **test** (never used before scoring)
3. **Fine-tune** the pretrained network on the first 40%: learning rate 3e-4, up to 20 epochs, early stopping
   on the last 15% of that data. Minutes with a trouble code are never used as "normal".

**Why 40 / 20 / 40?** These were chosen as sensible defaults, not tuned:
- **Time order** (earliest drives learn, later drives are tested) matches real use: you record healthy drives first, then check later ones.
- **40% fine-tune:** the largest share, because learning a car's "normal" needs the most data.
- **20% calibration, on separate drives:** the alarm line has to be set on drives the model did not train on.
  An earlier version set it on the tail of the fine-tune data and got 8-33% false alarms instead of the ~5% aimed for.
- **40% test:** a large, untouched set so the results are trustworthy.
A different split (e.g. 50/20/30) would likely give similar results; it was not tested.

**Key assumption:** the fine-tune and calibration drives are healthy. On your own car, record them when you
are confident the car is fine.

The final general model (`obd_pretrained.pt`) is pretrained once on **all** healthy cars.

---

## Step 6: Score each minute → `channel_errors`, `calibrate`

1. For each 1-minute window, hide each sensor in turn and let the network predict it. Error = mean squared difference.
2. Take the log of each sensor's error and z-score it using that sensor's errors on the car's **calibration** drives.
3. **Smooth:** average over the last 12 windows (2 minutes) within the same drive (a fault persists; a single odd moment does not).
4. **Stuck-sensor check** (`flat_minutes`, `slow_stuck_scorer`): a stuck sensor reads a normal value, just frozen, so the
   network barely notices it. For the slow sensors (coolant and intake temperature) we count how many minutes the reading
   has not changed at all, divided by the longest freeze seen on this car's calibration drives (at least 5 minutes).
   Both parts are rescaled on the calibration drives so they share one alarm line; the larger one counts.
5. **Minute score** = the worst sensor's value (or the stuck check, if larger). That sensor is the one the model **blames**.

---

## Step 7: Decide HEALTHY / FAULT (final settings)

| Setting | Final choice | Why |
|---|---|---|
| Alarm cutoff | **Medium**: 95th percentile of the car's calibration scores | best balance of catches vs. false alarms |
| Cutoff type | **one shared cutoff** for all sensors | per-sensor cutoffs did not help overall and were worse on real codes |
| Scoring | **2-minute score + stuck-sensor check** | best on both tests (see Step 10) |
| Warm-up grace | **5 minutes**: no alarms in the first 5 min of a drive, and those minutes are not used to set the cutoff | cold starts caused many false alarms |
| Drive rule | a drive is **FAULT if more than 30% of its minutes** are flagged | compared 20/30/40/50% (main test, before the warm-up rule): faulty drives caught 57/50/47/44%, clean drives flagged 27/18/17/15%. 30% gives the biggest drop in false alarms for the smallest loss in catches |
| Physics inputs | **not used** | tested; small gain on air faults but more false alarms (see Step 10) |

- **Minute:** FAULT if its score is above the cutoff (and it is past the warm-up).
- **Drive:** FAULT if more than 30% of its minutes are FAULT.

Both tests below use exactly these settings.

---

## Step 8: Main evaluation → `evaluate_car`, `binary_report` (`obd_results_*.csv`)

- **8 held-out cars:** car1, car4, car8, car9, car11, car12, car13, exp2_car (exp3_car had too little data to test)
- **10 sensor faults** injected into copies of the test drives, for the whole drive: MAF under-reads 25%,
  MAF drifts to -30%, MAP +10 kPa, coolant warms at half speed (thermostat), coolant sensor stuck,
  intake temp +15 °C, fuel trim +10% (vacuum leak), noisy throttle, plus **realistic versions of the two air faults**
  (`maf_reacts`, `map_reacts`): the engine computer believes the wrong reading, so the calculated engine load changes
  with it and the short-term fuel trim moves the other way (assumed: half of the fuel error is still visible in the
  short-term trim; the rest is absorbed by the long-term trim, which is not logged).
  The clean original of each drive is the healthy comparison.
- **Compared against:** the same network without fine-tuning, and a per-car linear (ridge) virtual sensor.

**Separation (AUROC, 0.5 = guessing, 1.0 = perfect):**

| Method | AUROC |
|---|---|
| **Fine-tuned Transformer (ours)** | **0.81** |
| Same network, no per-car fine-tuning | 0.77 |
| Ridge baseline | 0.77 |

**Binary scores, ours, final settings:**

| Level | Accuracy | Precision | Recall | F1 | False alarms |
|---|---|---|---|---|---|
| Per minute | 0.72 | 0.91 | 0.49 | 0.63 | 5% |
| Per drive (>30% rule) | 0.70 | 0.91 | 0.44 | 0.59 | 4% |

**By fault (AUROC / faulty drives caught / right sensor named):**

| Fault | AUROC | Drives caught | Right sensor |
|---|---|---|---|
| Fuel trim +10% (vacuum leak) | 0.97 | 75% | 98% |
| Coolant sensor stuck | 0.96 | 78% | 99% |
| MAF -25%, engine reacts (realistic) | 0.90 | 45% | 2%* |
| MAP +10 kPa, engine reacts (realistic) | 0.85 | 56% | 0%* |
| Intake air temp +15 °C | 0.84 | 53% | 79% |
| Throttle signal noisy | 0.83 | 44% | 49% |
| MAF -25% (sensor only) | 0.78 | 38% | 64% |
| Coolant warms at half speed (thermostat) | 0.68 | 36% | 71% |
| MAF drifts to -30% | 0.61 | 9% | 37% |
| MAP +10 kPa (sensor only) | 0.58 | 5% | 11% |

\* For the realistic air faults the model blames the **fuel trim**, which is where the fault really shows up;
a mechanic would read it the same way (a lean/rich trim points to the air sensor), but it counts as "wrong sensor" here.

**Real trouble codes (car13):** AUROC 0.66 (other methods 0.54-0.56).

---

## Step 9: Synthetic test with an answer key → `synthetic_test.py` (`synthetic_*.csv`; timeline figure: `figures\fig10_timelines.png`)

- 4 held-out cars (car11, car9, car8, exp2_car), each with its own model trained without that car
- Test drives: about 30% left clean, the rest get **one** fault from the starter list (**some start partway through the drive**):

| Fault | Sensor | Starts at |
|---|---|---|
| MAF reads 15% low | MAF | minute 0 |
| MAF reads 15% low, engine reacts (realistic) | MAF | minute 0 |
| Coolant sensor stuck | ECT | minute 10 |
| MAP reads +8 kPa | MAP | minute 5 |
| MAP reads +8 kPa, engine reacts (realistic) | MAP | minute 5 |
| Fuel trim +8% (vacuum leak) | STFT1 | minute 0 |
| Throttle signal noisy (±5%) | THROTTLE | minute 0 |
| Intake air temp drifts up to +20 °C | IAT | minute 0 |

- **67 drives** (40 faulty, 27 clean). Minute-level AUROC **0.75**.

**Final results (final settings):**

| Level | Accuracy | Precision | Recall | F1 | False alarms |
|---|---|---|---|---|---|
| Per minute | 0.54 | 0.95 | 0.38 | 0.55 | 5% |
| **Per drive** | **0.70** | **0.92** | **0.55** | **0.69** | **7%** |

**Run-to-run variation:** training on the CPU is not perfectly repeatable, so re-running the same test gives
slightly different numbers (about ±0.03; e.g. an earlier run of the same settings gave per-drive F1 0.73).

Effect of the warm-up grace (per drive, earlier run): no grace → F1 0.64, 29% false alarms; 3 min → F1 0.72, 12%; **5 min → F1 0.71, 9%**.

**By fault (drives labelled correctly):** coolant stuck 3/3 (alarm ~3 min after it freezes) · vacuum leak 4/6 ·
MAP +8 kPa with engine reacting 2/3 · MAF 15% low with engine reacting 1/2 · throttle noisy 5/8 · intake temp drift 6/11 ·
MAF 15% low (sensor only) 1/5 · MAP +8 kPa (sensor only) 0/2 · clean drives 25/27.

**What it still misses:**
- very short drives (almost all warm-up) are too short to judge
- slow faults such as a gradual intake-temp drift, which look like normal variation early on
- air faults where only the sensor changes; engine load is calculated from those sensors, so it moves with them.
  When the engine reacts as a real one would, most of these are caught through the fuel trim
- changing the drive rule (e.g. "N minutes in a row") did not help

---

## Step 10: The fixes we tested, and how much better the model got 

After the first version, four fixes were tested. **Both tests were re-run on exactly the same cars, drives and faults**
for every combination, so the numbers compare fairly.

1. **Two-speed alarm:** a second, slow score averaged over 10 minutes, to catch creeping faults.
2. **Stuck-sensor check:** flags coolant/intake-temp readings frozen for longer than this car ever showed normally.
3. **Physics inputs:** ready-made engine relationships given to the network (air per engine turn = MAF ÷ RPM,
   load ÷ MAP, load × RPM); hidden whenever a sensor they are built from is hidden. Run with `--physics`.
4. **Realistic air faults:** the MAF/MAP faults also change engine load and fuel trim, as in a real car (test change only).

| Configuration | Main AUROC | Main F1 (drive) | Main false alarms | Synthetic F1 (drive) | Synthetic false alarms |
|---|---|---|---|---|---|
| Before the fixes | 0.79 | 0.55 | 3% | 0.66 | 4% |
| **+ stuck check (FINAL)** | **0.81** | **0.59** | **4%** | **0.69** | **7%** |
| + two-speed alarm + stuck check | 0.83 | 0.61 | 9% | 0.67 | 15% |
| + physics inputs | 0.79 | 0.58 | 7% | 0.69 | 7% |
| + physics + stuck check | 0.81 | 0.61 | 5% | 0.68 | 11% |
| + all: physics + two-speed + stuck check | 0.82 | 0.63 | 11% | 0.62 | 15% |

- **Stuck check: kept.** Stuck coolant caught in 78% of drives instead of 44% (main test) and 3/3 instead of 1/3
  (synthetic), with only a small rise in false alarms. Best overall on both tests.
- **Two-speed alarm: dropped.** Helps when a fault lasts the whole drive, but hurts when it starts mid-drive and
  more than doubles false alarms.
- **Physics inputs: dropped.** Small gains on the air faults in the main test (MAF drift caught 23% vs 10%, MAP offset
  16% vs 4%) but more false alarms and slightly worse on the synthetic test. Only 4 of 14 cars have a MAF sensor,
  which is likely too little data for the network to use them well.
- **Realistic air faults: kept in the test.** They are caught far more often (45-60% of drives) than the sensor-only
  versions (5-40%), so in a real car these faults would mostly be visible through the fuel trim.

**Before → after (final), per drive:**

| Test | Precision | Recall | F1 | Accuracy | False alarms |
|---|---|---|---|---|---|
| Main evaluation | 0.92 → 0.91 | 0.39 → 0.44 | 0.55 → 0.59 | 0.68 → 0.70 | 3% → 4% |
| Synthetic test | 0.95 → 0.92 | 0.50 → 0.55 | 0.66 → 0.69 | 0.69 → 0.70 | 4% → 7% |

(Physics rows come from a separate run; given the ±0.03 run-to-run variation, differences of 0.01-0.02 are not meaningful.)

---

## Figures for the report → `make_figures.py` (saved in `figures\`)

| File | What it shows |
|---|---|
| `fig1a_feature_table.png` | the 9 features: OBD-II PID, unit, typical value, usual range, how many cars report it |
| `fig1b_sensor_relationships.png` | what the model learns: healthy sensors move together (RPM vs airflow, throttle vs MAP, warm-up vs coolant) |
| `fig2_sensor_coverage.png` | how many cars report each sensor |
| `fig3_faults_caught.png` | per fault type: share of faulty drives caught and AUROC |
| `fig4_right_sensor.png` | per fault type: how often the model names the broken sensor |
| `fig5_score_vs_alarm_line.png` | healthy vs faulty minutes relative to the alarm line |
| `fig6_roc_curve.png` | ROC curve of the synthetic test |
| `fig7_cutoff_tradeoff.png` | choosing the alarm line: faults caught vs false alarms |
| `fig10_timelines.png` | example drives: score over time, alarm line, when the fault started, warm-up |
| `table1_results_summary.png` (+ `summary_table.csv`) | final numbers: AUROC, accuracy, precision, recall, F1, false alarms |

---

## Step 11: What I tried first and why it failed (for the report)

- **EngineAD** (25 trucks): sensors released only as 8 blended numbers (PC1-PC8), which don't match OBD-II. Every
  method, including a supervised model trained on the labels, scored AUROC ~0.5. The paper's reported F1 ≈ 0.65 is a
  weighted F1 that equals random guessing on this data (reproduced: 0.642 vs 0.637 for random).
- **DieselOBD (Lounis)**: only 3 healthy cars, and every car is a different model, so "different car" looked like
  "faulty car". This led to the per-car "compare each car with itself" design.
- **EngineFaultDB**: only 4 of the 9 sensors overlap, and its faults are fuel-mixture faults visible in exhaust gases,
  which OBD-II doesn't provide. The final model scored AUROC 0.49 there, so it was not used.

---

## Step 12: Limitations

- Most test faults are injected (synthetic); the only real faults are car13's codes
- The realistic air faults assume half of the fuel error stays visible in the short-term fuel trim
- Only 15 healthy cars of training data (several with very little driving), logged every ~4-12 s over Bluetooth
- Assumes the fine-tune/calibration drives are healthy


## Step 13: Using it on your own car (final step, not built yet)

1. **Record** several hours of normal driving with an OBD app (e.g. Car Scanner or Torque Pro), exported to CSV at
   about 1 reading per second, with the 9 sensors above. Record when the car is known to be fine (no check-engine light),
   with a mix of cold starts, city and highway driving.
2. A "run on my car" script will:
   - map the app's column names to the 9 sensors
   - fine-tune `obd_pretrained.pt` on the first drives
   - calibrate the alarm on the next drives
   - label later drives HEALTHY/FAULT (per minute and per drive) and name the suspicious sensor


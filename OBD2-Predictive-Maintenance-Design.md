# OBD-II Predictive Maintenance: High-Level Design

Status: draft for team review.

## 1. Goal

Two stages:

1. **Public data:** one binary classifier per labeled dataset that answers "engine fault: yes/no", evaluated against known labels.
2. **Team car:** learn the car's normal behavior from healthy OBD-II driving data and flag deviations. The car has no fault examples, so this stage stays unsupervised (reconstruction error), and the final report must say what we could and could not verify.

## 2. Decisions so far

| Decision | Choice |
|---|---|
| Framework | PyTorch (+ scikit-learn for baselines) |
| Public-data model | Separate binary classifier per dataset; inputs barely overlap, so no merged model |
| Car-stage model | Reconstruction-error anomaly detection (classifier can't train without fault examples) |
| Splits | By group (vehicle / session / engine), never random rows |
| Compute | Undecided; keep code environment-agnostic |

## 3. Datasets

| Dataset | Format / license | Binary label | Features | Split unit | Caveat |
|---|---|---|---|---|---|
| **EngineFaultDB** | 56k tabular rows, CSV, GPL-3.0 | `Fault` 0 = no fault; 1 rich mixture, 2 lean mixture, 3 low ignition voltage = fault | MAP, TPS, RPM, Speed, Lambda, AFR (OBD-like) + Force, Power, consumption, CO, HC, CO2, O2 | No group ID; split by row order within each class | Lab rig. 71% of rows are faults. Gas-analyzer features won't exist on the car. |
| **OBD-Dataset** (Lounis) | CSV/XLSX, CC BY 4.0 | Vehicle has DTCs (5 vehicles) vs. fault-free (3) | LOAD_PCT, ECT, MAP, RPM, VSS, IAT, MAF, FRP, BARO, VPWR, AAT | `vehicle_id` | Labels are per vehicle, so we must test on held-out vehicles (leave-one-vehicle-out). With only 8 vehicles, results will be noisy. Closest signal match to our dongle. |
| **EngineAD** | 25 pickle files, 135–304 MB each (~5 GB), CC BY-NC 4.0 | Expert normal / anomaly | 8 PCA components × 300-step windows | Vehicle (file) | Can't be mapped back to OBD-II signals, so it's a standalone benchmark. Download a few vehicles first. |
| **Kaggle obdii-ds3** | CSV/XLSX, CC0 | **None** | OBD-II PIDs from driver-behavior experiments | Trip / driver | Not usable for the classifier. Set aside as healthy-only data for the car stage. |

## 4. Classifier pipeline (public stage)

```
loader(dataset) → X, y (0/1), group ─► group-safe split (train/val/test)
                                        │
                     fit scaler on train only
                                        │
          baselines: logistic regression, gradient boosting
          PyTorch:   MLP (tabular) | 1D CNN (EngineAD windows)
                                        │
          threshold chosen on val ─► metrics on test ─► artifacts/<dataset>/<run>/
```

- **Loader contract:** each dataset loader returns `X` (features), `y` (0/1), and `group` (split key). A dataset's logic lives only in its loader; splitting, training and metrics are shared.
- **Class imbalance:** use a class-weighted loss, and pick the threshold on the validation set, not a fixed 0.5.
- **Metrics:** F1, AUROC, AUPRC and balanced accuracy, plus a confusion matrix. For OBD-Dataset, also report per-vehicle predictions, since each held-out vehicle is effectively one test case.
- **Independent implementation:** each team member builds their own version on their personal branch, and the team compares them later. This branch implements all three classifiers from scratch as a small `src/` package with one loader per dataset.
- **EngineFaultDB split:** there's no group ID, and rows are likely time-ordered. Split each class into contiguous blocks (70/15/15) rather than random rows, so neighboring samples don't leak into the test set.

## 5. Car stage (unchanged in intent)

- An ELM327 logger writes one CSV per drive. First task: probe which PIDs the car actually returns.
- Train a reconstruction model on healthy windows: a row-level masked autoencoder first, then a TCN/GRU over 30–60 s windows.
- Set the threshold from the false-alert rate per driving hour on held-out drives. DTC events are the only real ground truth.
- Transfer from the public stage is limited to recipe and code. Weights only transfer where signals overlap, mainly with OBD-Dataset.

## 6. Testing

pytest, kept small:
- each loader returns aligned `X`/`y`/`group` with labels in {0,1};
- no group appears in more than one split;
- scaler statistics come from the training split only;
- model forward pass shapes are correct;
- a short training run on tiny synthetic data beats chance.

## 7. Timeline mapping

| Weeks | Work |
|---|---|
| 1–3 | Dataset download + loaders; group-safe split; logger PID probe |
| 3–6 | Baselines + PyTorch classifiers on all three labeled datasets; internal demo |
| 4–8 | Car data collection |
| 7–10 | Car-stage reconstruction model + thresholds |
| 10–12 | Evaluation + limitations |
| 12–14 | Poster, report |

## 8. Open questions / risks

1. **EngineFaultDB labels:** the README doesn't define the fault types. Confirm 0 = no fault against the paper before training.
2. **OBD-Dataset sample size:** 8 vehicles make per-vehicle labels a weak benchmark; report the uncertainty honestly.
3. **EngineAD size and license:** ~5 GB, and CC BY-NC (fine for coursework).
4. **Dongle PIDs:** unknown until tested on the car.
5. **Comparing versions:** each branch should report the same metrics on the same kind of split, so the team meeting can compare like with like.

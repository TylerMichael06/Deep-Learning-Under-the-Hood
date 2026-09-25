# Deep-Learning-Under-the-Hood

Binary engine-fault classifiers ("fault: yes/no") on three public datasets. Design: `OBD2-Predictive-Maintenance-Design.md`. Plan: `docs/superpowers/plans/2026-09-25-public-fault-classifiers.md`.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/python scripts/download_data.py        # add --trucks 8 for a smaller EngineAD download (all 25 = ~6.4 GB)
.venv/bin/pytest -q
```

## Usage

```bash
.venv/bin/python -m obdfault.run --dataset {enginefaultdb,dieselobd,enginead} --model {logreg,gbm,nn} [--seed 0]
```

Writes `artifacts/<dataset>/<model>/metrics.json` (pooled metrics, thresholds, per-group breakdown) and `predictions.csv`. `nn` is an MLP for the tabular datasets and a 1D CNN for EngineAD windows.

## Evaluation protocol

| Dataset | Split | Why |
|---|---|---|
| EngineFaultDB | per fault class, contiguous 70/15/15 blocks | rows are time-ordered; random splits leak near-duplicate neighbors |
| DieselOBD | leave-one-vehicle-out (8 folds), val = last 15% of each training vehicle | labels are per vehicle; the published train/test files share all vehicles |
| EngineAD | by truck, 70/15/15, seed 0 (test trucks: 2, 4, 13, 19) | windows from one truck are correlated |

Threshold: max F1 on validation. Metrics are pooled over folds. Class-weighted losses throughout.

## Results (seed 0)

| Dataset | Model | F1 | Balanced acc. | AUROC | AUPRC |
|---|---|---|---|---|---|
| EngineFaultDB | logreg | 0.833 | 0.500 | 0.524 | 0.730 |
| EngineFaultDB | gbm | **0.998** | **0.998** | **1.000** | **1.000** |
| EngineFaultDB | nn (MLP) | 0.807 | 0.627 | 0.825 | 0.946 |
| DieselOBD | logreg | 0.469 | 0.284 | 0.145 | 0.507 |
| DieselOBD | gbm | 0.407 | 0.245 | 0.395 | 0.744 |
| DieselOBD | nn (MLP) | 0.280 | 0.210 | 0.114 | 0.504 |
| EngineAD | logreg | 0.221 | 0.556 | 0.587 | 0.144 |
| EngineAD | gbm | 0.206 | 0.513 | 0.527 | 0.127 |
| EngineAD | nn (1D CNN) | 0.203 | 0.528 | 0.550 | 0.137 |

Test-set fault rates: EngineFaultDB 71%, DieselOBD 69% of rows (5 of 8 vehicles), EngineAD 11%. F1 alone looks good on the first two only because faults are the majority class; read balanced accuracy and AUROC.

## What the numbers mean

- **EngineFaultDB:** within each fault class, the test block runs at much higher RPM than the training block (about 3,200 vs 1,900), so this split measures extrapolation to a new operating regime. Gradient boosting still separates the classes perfectly. Logistic regression can't: "no fault" lies *between* rich and lean mixture on lambda and CO, so it isn't linearly separable, and the model ends up flagging everything. The MLP is unstable across seeds: AUROC 0.825 / 0.808 / 0.604 for seeds 0 / 1 / 2. A random row split scores 1.000 for the MLP, but that only reflects leakage between near-duplicate rows.
- **DieselOBD:** no model transfers to an unseen vehicle. AUROC below 0.5 is a real finding, not a bug: every model scores fault-free vehicle 8 near 1.0 and faulty vehicle 6 near 0. The models pick up make-specific sensor baselines, not fault signatures. Eight vehicles with one label each can't support a fault detector.
- **EngineAD:** cross-truck detection is near chance. As a diagnostic, training on the first 70% of every truck's windows and testing on the rest gives gradient boosting AUROC 0.69. So the labels carry signal, but it doesn't generalize across trucks with these features. The dataset authors benchmark per-vehicle one-class detectors, which fits the car stage of our design better than a cross-fleet classifier.

## Limitations

- DieselOBD has 8 vehicles, so each fold's test set is one car; results say more about vehicle-level separability than about fault detection in general.
- EngineAD inputs are PCA components and can't be mapped back to OBD-II signals.
- EngineFaultDB is a lab rig; most features (gas analyzer, dynamometer) won't exist on a real car.
- One seed per model in the table; the EngineFaultDB MLP shows how much a single seed can mislead.

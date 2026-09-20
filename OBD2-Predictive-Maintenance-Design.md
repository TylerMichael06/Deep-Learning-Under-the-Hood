# OBD-II Predictive Maintenance: High-Level Design

Status: draft for team review.
## 1. Goal

Detect anomalous engine behavior from OBD-II telemetry. Validate the approach on public labeled fault datasets (EngineFaultDB, EngineAD, HIL), then fine-tune on healthy driving data from one team vehicle and flag deviations from that car's normal behavior.

Success:
- **Public data:** measurable detection quality against known labels.
- **Team car:** no false alarms on ordinary driving, and any real anomaly (e.g. a DTC) gets flagged. We have no labeled failures on this car, and the final report must say what we could and could not verify.

## 2. Decisions so far

| Decision | Choice |
|---|---|
| Scope | Whole pipeline, phased to match the proposal timeline |
| Framework | PyTorch (+ scikit-learn for the baseline) |
| Modeling | Reconstruction-error anomaly detection, staged: simple baseline (MVP) then sequence autoencoder |
| Repo style | Python package + thin scripts; notebooks for EDA only |
| Compute | Undecided, so keep code environment-agnostic (no cloud-specific code) |

Why reconstruction: the team car only has presumed-healthy data, so a supervised healthy/faulty classifier cannot be fine-tuned on it. A model that learns "normal" from healthy data alone works in both stages.

## 3. Architecture

```
Public datasets ──┐
                  ├─► loaders → common schema → windowing/normalization ─┐
Car OBD-II logger ┘                                                      │
                                                                         ▼
                                            pretrain on healthy public windows
                                                                         │
                                            evaluate vs. public labels   │
                                                                         ▼
                                            fine-tune on car healthy windows
                                                                         │
                                                                         ▼
                                            score drives → flag windows → report
```

```
src/
  data/        # one loader per dataset + car logs → common schema
  logging/     # OBD-II collection script (python-OBD wrapper) → raw CSV per drive
  features/    # windowing, per-source normalization, missing-channel handling
  models/
    baseline.py   # Isolation Forest / per-sample autoencoder (MVP)
    sequence.py   # LSTM or TCN autoencoder
  train.py     # modes: pretrain-public, finetune-car
  evaluate.py  # modes: public-metrics, car-report
notebooks/     # EDA only; anything reused moves into src/
tests/
```

Each module has one owner and a narrow interface (frames in, frames/tensors out), so teammates can work in parallel.

## 4. Data

**Common schema** (superset; missing signals are `NaN`, never dropped rows):
`timestamp`, `engine_rpm`, `vehicle_speed`, `throttle_pos`, `engine_load`, `coolant_temp`, `intake_air_temp`*, `maf`*, `fuel_trim_short`*, `fuel_trim_long`*, `dtc_codes`, `label` (`healthy`/`faulty`/`unknown`), `source` (`fault_db`/`engine_ad`/`hil`/`car`). (*may be unavailable from a consumer reader.)

- **Loaders** map each dataset's native format to the schema and validate on load.
- **Car logger** polls the ELM327 at a fixed rate (target 1–2 Hz, whatever the dongle sustains), one CSV per drive. First task: probe the real car to learn which PIDs the dongle returns. The schema is a target, not a guarantee.
- **Windowing:** fixed-length sliding windows (e.g. 30–60 s) within a continuous session, never across session boundaries.
- **Normalization:** z-score per source/vehicle so different baselines don't dominate.
- Car data is labeled `unknown`; presumed healthy unless a DTC is present.

## 5. Modeling & training

- **Stage 1 (MVP baseline):** Isolation Forest or per-sample autoencoder on window summary features. Target: internal demo in weeks 3–6.
- **Stage 2 (main model):** LSTM/TCN autoencoder over windows. Anomaly score = mean reconstruction error per window.
- **Pretrain:** on healthy windows from public data. Threshold from a held-out healthy validation split (e.g. a high percentile of healthy scores).
- **Fine-tune:** input/output adapters map the car's channel set to the shared latent size. Freeze the trunk first, then unfreeze at a low learning rate. Train on car healthy windows only; set the threshold on held-out car drives.

## 6. Evaluation

- **Public:** split by session/engine, not random windows (avoids leakage). Metrics: AUROC, AUPRC, F1 at the chosen threshold, per-fault-type recall.
- **Car:** false-positive rate per hour on held-out drives; manual review of flagged windows; DTC events as the only real ground truth. Optional sanity check: inject synthetic drift (e.g. coolant temp offset) into a held-out drive and confirm it gets flagged.

## 7. Errors & testing

- **Errors:** logger timeouts or dropped frames → gaps marked `NaN`, session split at long gaps; unsupported PID → column stays `NaN`; schema/unit validation at the loader boundary, failing loudly.
- **Tests (pytest, one file per module):** loaders return the schema; windows never cross sessions; splits have no session overlap; model forward/backward shapes; a tiny synthetic-data smoke test of pretrain → fine-tune → score.

## 8. Timeline mapping

| Weeks | Phase | Design pieces |
|---|---|---|
| 1–3 | Setup | repo, loaders skeleton, logger + PID probe, dataset access |
| 3–6 | Model dev | windowing, baseline MVP, public-data eval |
| 4–8 | Car collection | logger in use, varied drives |
| 7–10 | Real-world | sequence model, fine-tune, car thresholds |
| 10–12 | Evaluation | public metrics, car report, limitations |
| 12–14 | Finalize | poster, report |

## 9. Open questions / risks

1. **Dataset access and licensing:** not yet checked for any of the three.
2. **Channel overlap:** public datasets may share few signals with what the ELM327 exposes. If overlap is small, what transfers is the architecture and training recipe, not the weights. Check in weeks 1–3, before committing to the fine-tune story.
3. **Dongle PIDs:** unknown until tested on the car.
4. **Compute:** undecided; keep models small enough for CPU/Colab.
5. **Sampling-rate mismatch:** the public data rates vs. ~1–2 Hz from the dongle; resampling policy needed.

# Final report: thermostat slow warm-up detection

[Summary](#summary) · [Complete report](#complete-report) · [Table of contents](#table-of-contents)

## Summary

**Recommendation: keep the thermostat check experimental. The complete comparison did not show a clear overall improvement, so we should not enable it by default. Pause further tuning until we have independent cold-start recordings and confirmed fault cases.**

We added an optional check for slower-than-expected engine warming and compared the current model against the model with this check. The full evaluation covered **8 vehicles and 10 injected fault types**, with 584 judged faulty copies and 74 unique clean controls.

| Full-evaluation result | Current model | With thermostat check |
|---|---:|---:|
| Injected faults detected | 391/584 (66.95%) | 395/584 (67.64%) |
| False alarms on unique clean controls | 8/74 (10.81%) | 9/74 (12.16%) |
| Overall F1 — balance of precision and recall* | 0.74976 | 0.74953 |
| Existing thermostat fault scenario detected | 29/74 | 29/74 |

*F1 uses the repository's fault-weighted benchmark convention; clean controls are repeated for each evaluated fault type. It is not a fleet-level performance estimate.*

**What this means:** four additional injected-fault detections came with one additional false alarm, leaving F1 essentially unchanged. The existing thermostat scenario did not improve. A separate synthetic test appeared to improve, but its only additional detection came from an alarm **before the injected fault started**.

Earlier tests did detect more artificial temperature delays, but only on **four recordings from one vehicle**. Only that vehicle had enough data to calibrate the new check; the other seven retained the original model's decisions. This is promising local sensitivity, not evidence of general improvement or detection of real thermostat failures.

**Latest team-data check:** after pulling `main`, one new BMW recording passes the signal-availability checks, but it cannot supply separate training, calibration, and test sets. Its long zero-RPM periods also require further validation; see [section 10](#section-10).

**Delivery:** the default model and checkpoint remain unchanged. The full run took **8 min 16 s**, and **21 automated tests passed**. The detailed methods, results, and limitations follow below.

---

## Complete report

Detailed evidence, methods, results, and reproduction instructions supporting the summary above.

### Table of contents

1. [Conclusion and decision](#section-1)
2. [Problem investigated](#section-2)
3. [Available data and why they were used](#section-3)
4. [What was implemented and how it produces an alarm](#section-4)
5. [Fault scenarios tested and what they represent](#section-5)
6. [Experiments and checks performed](#section-6)
7. [Results: two comparisons that must remain distinct](#section-7)
8. [Robustness: dependence on a single recording](#section-8)
9. [Uncertainty and interpretation limits](#section-9)
10. [Can the team's vehicle recordings validate the detector?](#section-10)
11. [Full-battery regression comparison — October 18](#section-11)
12. [Recommendation and missing evidence](#section-12)
13. [Files, evidence, and reproduction](#section-13)

---

**Date:** October 16, 2026 · updated October 18, 2026

**Investigation:** [issue #34](https://github.com/TylerMichael06/Deep-Learning-Under-the-Hood/issues/34) · branch `cristian/34-thermostat-warmup` · base revision `c82fec0`

**Status:** experimental; not integrated into the default detector.

<a id="section-1"></a>

### 1. Conclusion and decision

**There is insufficient evidence to determine that the additional detector improves the model in general.** Detection of artificial delays, particularly strong delays, improves reproducibly within four recordings from a single vehicle. However, applying the original full-drive rule reveals an additional false alarm. Detection of real thermostat failures and generalization to another vehicle have not been demonstrated.

| Question | Answer supported by the experiments |
|---|---|
| Does it detect more artificial temporal-delay cases in the available data? | Yes: under the full-drive rule, detection increases from 0/4 to 2/4 for the mild delay and from 0/4 to 4/4 for the strong delay. |
| Is the improvement free of additional false alarms? | No: alarms on clean controls increase from 1/17 to 2/17. |
| Can we claim that it improves the model in general? | No: only one vehicle is evaluable, the faults are artificial, and the sample is very small. |
| Can we conclude that it is ineffective? | No: sensitivity to the strong delay is consistent across the checks performed. |
| Does the October 18 full fault battery justify integration? | No clear aggregate benefit: recall increases slightly, unique clean alarms rise from 8/74 to 9/74, F1 is essentially unchanged, and legacy thermostat detection stays at 29/74. |
| What is recommended? | Keep it as an optional experiment and pause tuning until independent data are available. |

**October 18 update:** the complete comparison does not justify default integration. Its apparent synthetic-suite improvement comes from a pre-fault alarm on a recording that also alarms when clean; see section 11.

This report consolidates the work, explains how the data were selected, and distinguishes observed results from conclusions they cannot yet support. The report tables and PNG figures are included in version control; reading the report does not require local experiment outputs. Generated CSV tables and JSON manifests stay local and are excluded from Git.

<a id="section-2"></a>

### 2. Problem investigated

Low coolant temperature alone does not imply a fault: the engine may simply have started cold. The useful question is whether **it is warming less than expected given its initial state and activity**.

The existing detector uses a Transformer to reconstruct hidden sensors and measure prediction errors. It adds a frozen-temperature-sensor check and suppresses general alarms during the first five minutes. This protection against false alarms can hide early differences between normal and slow warming. A sensor that changes slowly also need not trigger the frozen-sensor check.

We investigated an additional component, `WarmupRise`, that learns each vehicle's normal temperature rise. It does not replace the Transformer and is not a new neural-network architecture: it is a regularized regression that produces a complementary score.

Ford's thermostat monitor also uses expected temperature and enabling conditions, and accounts for low-load situations that can produce incorrect diagnoses. This motivates the expected-versus-observed comparison, but **does not validate our method or make its thresholds universal**. [Ford monitor description, pp. 123–124](https://www.fordservicecontent.dealerconnection.com/Ford_Content/Catalog/OBDII/OBDSM1300.pdf#page=123).

<a id="section-3"></a>

### 3. Available data and why they were used

#### 3.1 Source and unit of analysis

We used the three Kaggle CSV files already consumed by the repository:

- `exp1_14drivers_14cars_dailyRoutes.csv`.
- `exp2_19drivers_1car_1route.csv`.
- `exp3_4drivers_1car_1route.csv`.

The loader produces **472 recording segments from 15 vehicle identifiers**, on a five-second grid. A segment may begin with a warm engine or partway through a drive; it should not automatically be treated as a complete drive from ignition.

In **323 segments**, engine runtime is missing and elapsed recording time is used instead. An initial value of zero therefore does not confirm an engine start. We added `runtime_source` provenance to distinguish these cases without changing the original model's input values.

The relevant unit for evaluating detection is the source recording; the vehicle also matters when assessing generalization across vehicles. Overlapping windows, artificial copies, and repetitions with different random seeds **do not increase the number of vehicles or source recordings**.

#### 3.2 Selection criteria and rationale

| Fixed criterion | Rationale |
|---|---|
| Initial ECT below 50 °C | Study recordings that begin relatively cold. This is an exploratory condition, not a universal diagnostic threshold. |
| First ECT within the first minute and at least 80% ECT coverage | Observe enough of the initial state and temperature evolution. |
| At least six valid windows after minute five | Allow comparison with the original late-period protocol. This excludes some short drives. |
| No engine codes recorded anywhere in the segment; exclude `car6` | Avoid using segments with recorded problems as normal references. Absence of codes does not certify mechanical health. |
| When real runtime exists, reconstructed starting runtime must be between 0 and 1 minute | Exclude known late or invalid starts. Unknown-runtime recordings remain provisional. |
| Initial IAT available and at least 90% cumulative RPM/load coverage at scored points | Provide thermal context and engine activity to the warm-up model. |
| Three usable recordings for training and two for calibration | Operational minimum for the prototype; not enough to demonstrate generalization. |

The initial filters leave **37 cold-state candidates**, only **three** of which have runtime consistent with a recording near engine start. Within the eight vehicles in the main evaluation, 32 candidates remain: 14 for training, nine for calibration, and nine for testing. The other five belong to vehicles outside that main cohort.

![Cold-state recordings available by vehicle and partition](assets/thermostat_final/01_coverage.png)

**How to read the figure:** bars count candidates before requiring all necessary signals. `car8` appears to have enough recordings, but its RPM/load data do not support model fitting under the fixed rules. Only `car9` completes training, calibration, and testing. Zeros for other vehicles do not mean the detector fails on them: they mean it cannot be evaluated on those data.

Evidence: `candidate_counts.csv` (local generated evidence) and `coverage_totals.json` (local generated evidence).

#### 3.3 Separation of training, calibration, and testing

We retained the original chronological partitions of approximately 40%/20%/40% of recording time, keeping whole segments separate. Each vehicle is partitioned first; cold-state candidates are filtered afterward. Windows from the four test cases were not randomly distributed between training and evaluation.

| Role for `car9` | Recordings | Internal loader IDs |
|---|---:|---|
| Learn expected warming | 7 | 360, 362, 363, 366, 371, 372, 374 |
| Calibrate error tolerance | 5 | 381, 382, 383, 384, 386 |
| Evaluate | 4 | 389, 390, 392, 394 |

These IDs are not VINs or external vehicle identifiers. All cases belong to the **same individual vehicle**, not to different vehicles of the same model. None of the four test cases has a runtime-confirmed engine start; they were recorded on three different UTC dates.

The Transformer retains its protocol: pretraining on other vehicles and fine-tuning on the target vehicle's initial partition. The warm-up regression instead learns and calibrates from the `car9` recordings listed above.

<a id="section-4"></a>

### 4. What was implemented and how it produces an alarm

#### 4.1 Expected temperature

`WarmupRise` uses:

- Initial ECT: the coolant's thermal state when observation begins.
- Initial IAT: approximate thermal context; not necessarily ambient temperature.
- Elapsed time since the first ECT reading.
- Accumulated activity: an approximate time integral of `(RPM/1000) × (load/100)`.

The prediction uses the initial state and activity history available at the scored point; it does not use future ECT. Current ECT is used to measure the prediction deficit. The one-minute average uses consecutive raw time rows, so missing readings are not compressed out of the time axis.

It learns ECT rise using logarithmic terms, squared terms, and interactions with initial temperature. The regression is regularized, and each recording has equal total weight. Both coefficients and scaling are learned only from the training partition.

```text
Observed rise = current ECT − initial ECT
Expected temperature = initial ECT + predicted rise
Deficit = maximum(expected temperature − observed ECT, 0)
Score = mean deficit over one minute
```

The fit uses 762 valid points from seven recordings. These are correlated observations within seven recordings, not 762 independent drives. Points with insufficient signals produce no score, not a declaration of normality.

#### 4.2 Calibration and decision rules

The threshold is set at the 90th percentile of normal calibration scores from the relevant period. The original thresholds were approximately **2.893 °C of mean deficit for the early period** and **7.027 °C for the late period**. The early threshold uses 90 overlapping windows from the five calibration recordings. These are not absolute ECT thresholds and do not guarantee a 10% false-alarm rate per drive. Neither the fitted coefficients nor the threshold has been validated for direct reuse on another vehicle; each vehicle needs its own training and calibration recordings.

| Variant | Interval | Combination |
|---|---|---|
| Original late check | Minutes 5–10, including both endpoints | OR the baseline and new-score window flags, then apply the >30% fraction rule. |
| Early check | From minute 2 up to, but not including, minute 5 | Judge the period with at least six valid windows and more than 30% above threshold; OR its verdict with the baseline. |
| Late control with the same combination | Minutes 5–10 | OR period verdicts to separate the effect of observation timing from the change in combination rule. |

OR means an alarm from either component is sufficient. The new component never clears a baseline alarm. When it lacks evidence, it preserves the previous verdict; it does not turn a drive with insufficient data into a healthy drive.

The first window exceeding the threshold is not necessarily the time at which the period verdict can be finalized. This study is retrospective and **does not demonstrate an online diagnosis time of two minutes**.

<a id="section-5"></a>

### 5. Fault scenarios tested and what they represent

Each original recording was retained, and two copies were created in which only the temporal evolution of ECT changes:

```text
Modified ECT at t = original ECT at rate × t
```

The formula is applied relative to the beginning of the observed signal. Other signals and missingness are preserved. The injection does not interpolate across ECT gaps longer than 15 seconds; unsupported points retain their original values.

| Scenario | Rate | Interpretation |
|---|---:|---|
| Clean original | 1.00 | Control without an injection; health is assumed from the available data. |
| Mild delay | 0.75 | The curve advances at 75% of its recorded temporal rate. |
| Strong delay | 0.50 | The curve advances at half its recorded temporal rate. |

For example, at rate 0.5, a temperature originally observed at minute three appears at approximately minute six. This **does not** represent a “50% thermostat failure” and is not a physical simulation of all engine responses. In particular, load, RPM, and other signals remain unchanged.

The repository's historical injection halved the increase from the initial temperature: a rise from 20 to 90 °C became a rise from 20 to 55 °C. We investigated a temporal change to distinguish slow warming from a permanent reduction in temperature rise. A constant signal can remain unchanged under the historical injection; this edge case was reproduced artificially but was not found among the eligible audited cases.

![Original, delayed, and expected temperature curves for the four cases](assets/thermostat_final/02_curves.png)

**What the figure shows:** the shaded region is the early 2–5 minute period, where several delayed curves fall below the expectation. Later, they may approach the original curve and lose separation. The expectation also does not perfectly describe every normal recording: recording 394 triggers the early check even in its original version.

| Test recording | Initial ECT | Recording duration |
|---|---:|---:|
| 389 | 45 °C | 25.4 minutes |
| 390 | 36 °C | 14.1 minutes |
| 392 | 43 °C | 21.4 minutes |
| 394 | 43 °C | 26.4 minutes |

All four supply 18 valid early windows. Recording 390 illustrates the decision rule: the original has 0/18 windows above threshold, the mild delay has 6/18, and the strong delay has 18/18. The latter two exceed the >30% rule; a single threshold crossing would not suffice. We also retained 17 warm recordings without injections as controls; 13 are evaluable and four are too short. The eight artificial copies still come from **four** recordings, not eight independent faults.

<a id="section-6"></a>

### 6. Experiments and checks performed

| Check | What was held fixed or compared | Main result |
|---|---|---|
| Data audit | Coverage criteria, runtime, partitions, and signals | Only `car9` is evaluable; rules were not relaxed to include other vehicles. |
| Late detector | Add the score during minutes 5–10 | No additional detections in the available scenarios. |
| Early detector | Score during [2,5), calibrated in the same period | Adds detections of artificial delays. |
| Seed-0 reproduction | Maximum 40/20 epochs, CPU, two threads, original rules | Reproduces historical decisions and fractions within 1e-12. |
| Seeds 1 and 2 | Same protocol, data, and auxiliary-component thresholds | All three seeds produce the same alarm counts in the restricted experiment. |
| Full-drive rule | Preserve the original denominator and add the early verdict | An additional false alarm appears; results match for seeds 1 and 2. |
| Remove one training recording | Seven fits without changing the test cases | Strong-delay improvement remains; mild-delay improvement can disappear. |
| Remove one calibration recording | Five calibrations without changing the test cases | The same sensitivity pattern. |
| Early/late control with period-verdict OR | Same combination rule, separately calibrated periods | The late component still does not improve detection; changing the combination alone does not explain the difference. |
| Team-log inspection | Four CSV files from `origin/main`, without training or new health labels | They do not yet provide independent cold-state validation under this protocol. |
| Automated tests | Score causality, missing data, abstention, pairing, and denominators | 18 tests passed. |

The three runs use up to 40 pretraining epochs and 20 fine-tuning epochs with early stopping. The additional regression is solved directly, without epochs. Seed 0 reproduced the historical experiment; only seeds 1 and 2 exported per-window scores for the subsequent full-drive audit.

The early detector had been proposed after examining these same cases in the initial investigation. Although its thresholds come from calibration data, the new repetitions **do not constitute an independent confirmation set**.

<a id="section-7"></a>

### 7. Results: two comparisons that must remain distinct

#### 7.1 Restricted experimental comparison

Here, the baseline judges eligible windows during minutes 5–10. The combined method adds the early decision.

| Scenario | Restricted baseline | With early check |
|---|---:|---:|
| Cold originals | 1/4 alarms | 1/4 alarms |
| Evaluable warm originals | 1/13 alarms | 1/13 alarms |
| Mild delay | 1/4 detected | 2/4 detected |
| Strong delay | 1/4 detected | 4/4 detected |

The initial 1/4 detection corresponds to a recording that already alarmed when clean. The gains attributable to the delay are one new detection for the mild case and three for the strong case. In this specific comparison, no alarms are added on clean controls.

#### 7.2 Comparison using the original full-drive rule

We reused saved scores and applied the repository's original rule: suppress flags before minute five, require six later windows, and calculate the flagged fraction over the drive's windows, **including suppressed warm-up windows in the denominator**. The audit preserves that behavior; it does not correct or replace it.

| Scenario | Full-drive baseline | With early check |
|---|---:|---:|
| Cold originals | 0/4 alarms | 1/4 alarms |
| Evaluable warm originals | 1/13 alarms | 1/13 alarms |
| Mild delay | 0/4 detected | 2/4 detected |
| Strong delay | 0/4 detected | 4/4 detected |
| All evaluable clean controls | **1/17: 5.9%** | **2/17: 11.8%** |

Results are identical for seeds 1 and 2. Recording 394 produces the new false alarm: its flags in the 5–10 minute window were sufficient for the restricted comparison but not for the full-drive baseline. The early component flags it regardless.

![Alarm comparison under the restricted window and full-drive rules](assets/thermostat_final/03_results.png)

**What the figure shows:** in delayed scenarios, more alarms mean more detections. In clean controls, more alarms are a cost. The left-hand comparison could suggest a cost-free improvement; the right-hand comparison shows that this interpretation does not hold for the model's original rule.

Warm controls provide little challenge to the new component because it abstains on them. Their unchanged results do not demonstrate robustness to new cold-start conditions. The 11.8% rate must not be extrapolated to an entire fleet either: it comes from these 17 `car9` controls.

The October 16 audit above did not recalculate a global AUROC/F1 comparable to the official table; section 11 adds the subsequent full-battery drive-level comparison. The cohort, scenarios, and artificial prevalence differ; all other fault classes were not revalidated here either. Evidence: `seed_summary.csv` (local generated evidence), `full_drive_summary.csv` (local generated evidence), and `full_drive_decisions.csv` (local generated evidence).

<a id="section-8"></a>

### 8. Robustness: dependence on a single recording

We performed 12 checks: seven removing one training recording and five removing one calibration recording. The four source test recordings, their copies, periods, and rules remained fixed. We did not select the best-performing variant.

![Alarms after removing one training or calibration recording](assets/thermostat_final/04_sensitivity.png)

**How to read the figure:** each cell shows alarms on the same four originals or copies. This is the original restricted comparison, not the full-drive comparison. A darker color means more alarms; that is favorable only in the artificial-fault columns.

- Strong-delay detection remains **4/4** in all 12 checks.
- Mild-delay detection ranges from **1/4 to 2/4**.
- Removing training recording 362 or calibration recording 383 eliminates the additional mild-delay detection.
- The combination retains one alarm among the four cold originals in that restricted window.

This supports stable local sensitivity to strong delays and reveals fragility for mild delays. It does not demonstrate generalization: all robustness checks reuse the same test set. `sensitivity_summary.csv` (local generated evidence).

<a id="section-9"></a>

### 9. Uncertainty and interpretation limits

![Descriptive intervals for results based on four recordings](assets/thermostat_final/05_uncertainty.png)

Exact 95% binomial intervals illustrate the sample-size problem: 4/4 detections corresponds to approximately **39.8%–100%**; 2/4 to **6.8%–93.2%**; and 1/4 false alarms to **0.6%–80.6%**. They were calculated using [SciPy's exact binomial interval](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binomtest.html).

These intervals assume independence between recordings. They do not adjust for dependence between drives from the same vehicle or exploratory detector selection, so they are **descriptive**, not confirmatory intervals for performance across vehicles. Repeating seeds or injections does not remove that limitation.

In the restricted strong-delay comparison, there are only three favorable discordant pairs and no unfavorable pairs; the two-sided exact binomial test on those pairs gives p=0.25. For the mild delay, there is only one pair and p=1. These values do not prove absence of benefit and are not an equivalence test. With only one evaluated vehicle, variability across vehicles cannot be meaningfully estimated. `descriptive_intervals.csv` (local generated evidence) and `paired_statistics.csv` (local generated evidence).

Other relevant limitations:

1. This evaluation contains no real thermostat faults confirmed by inspection or repair.
2. Control health is assumed from the records; it is not mechanically certified.
3. Runtime is unknown for all four cases: we observe warming from recording start, not necessarily from ignition.
4. Retrospective selection requires segment duration, coverage, and absence of codes. A fully online eligibility policy and performance on short drives have not been validated.
5. The thermal model and injection do not represent all responses to weather, cabin heating, low load, stop/start, or hybrid operation.
6. OR combination can increase both detection and false alarms. An agreed criterion for their acceptable cost is still needed.

<a id="section-10"></a>

### 10. Can the team's vehicle recordings validate the detector?

**After pulling the latest `main`, there is one potentially usable cold BMW recording, but still no independent per-vehicle training/calibration/test set.** The new file improves signal coverage; it does not change the model-comparison results or establish general improvement.

#### Repository snapshot and inspection method

On October 18, `git pull --ff-only origin main` in an isolated worktree returned **Already up to date** at `188e23e`. This preserved the uncommitted investigation changes. The current `data/` directory contains six BlueDriver CSVs: five BMW recordings and one Porsche recording. The two October 17 BMW files are the latest additions; no newer files appeared in this pull.

We inspected all six files and repeated the availability checks using the existing five-second resampler, channel validity ranges, pipeline window requirements, and `WarmupRise.inputs`. Missing sensors remained missing. Oxygen-sensor-specific trim and long-term fuel trim were not substituted for the missing STFT1 PID. No recording was trimmed to move engine activity into the early period, and missing initial IAT was not backfilled.

These files do not export engine runtime or trouble-code status. A neutral code array was used **only to calculate potential signal/window availability** with the existing audit API; it is not evidence that the vehicles are code-free or healthy. The counts below are potential windows, not fitted scores, alarms, or validation results. No model was fitted or scored on these recordings.

| Recording | Initial ECT | Signal/window finding | Role under the frozen protocol |
|---|---:|---|---|
| Porsche, October 3 | 96 °C | IAT and calculated load absent. | Warm; cannot train the cold-state check. |
| BMW, September 30 | 90 °C | IAT and calculated load absent. | Warm; cannot train the cold-state check. |
| BMW, October 11 at 16:50 | 36 °C | First IAT at 131 s; zero valid warm-up feature points under the initial-IAT rule. | Initially cold but unusable for this check. |
| BMW, October 11 at 16:58 | 88 °C | Initial context present. | Warm; the check abstains. |
| BMW, October 17 at 18:09 | **44 °C** | **109 valid feature points; 18 potential early windows and 31 potential late windows.** | **One provisional cold-state recording with adequate signals.** |
| BMW, October 17 at 19:12 | 75 °C | Initial context present; duration 11.25 minutes. | Warm; the check abstains. |

The October 16 inspection covered the first four files at revision `6b72f3b`. The initial October 18 review checked the two added files' raw columns. This follow-up goes further by checking their resampled inputs and usable windows. The later recording at 75 °C cannot be counted as another cold-state calibration or test example merely because its columns are complete.

#### What the new 44 °C recording adds—and what it cannot establish

The October 17, 18:09 recording lasts **13.38 minutes**, with ECT, IAT, RPM, and calculated load available from the beginning. Joint RPM/load coverage is **100% on the five-second grid**. It passes the prototype's signal and duration requirements, conditional on separately establishing code status and suitability as a healthy reference. Its 109 feature points remain observations within **one recording**, not 109 independent examples.

However, the first raw positive RPM reading occurs at **267 seconds (4 min 27 s)**; the first positive resampled point is at 270 seconds. RPM is zero in approximately **78.9% of the observed five-second samples**. The first row reports vehicle speed of 73 km/h while RPM is zero. These observations do not establish continuous combustion-engine operation from recording start. They also do not establish a thermostat fault or identify the reason for the zero-RPM periods.

![New BMW recording: coolant temperature and recorded RPM](assets/thermostat_final/07_new_bmw_recording.png)

**What the figure shows:** much of the current [2,5) early-check interval precedes the first observed positive RPM. Good column coverage is therefore not the same as observing three minutes of active engine warming. The check includes an RPM/load activity proxy, but its behavior on this recording and this operating pattern has not been validated. Shifting the clock or redefining eligibility now would be a separate protocol change, not a reproduction of the frozen experiment.

The prototype requires at least **three usable training recordings and two separate calibration recordings**, followed by an untouched test recording. One provisional recording cannot fill those roles. Dividing its windows into multiple apparent recordings would not create independent evidence. Those operational minima are also not a sufficient sample-size plan for general validation.

**Conclusion after the pull:** retain the new recording as useful material for future data collection and protocol validation, but do not train and evaluate on that same file or transfer `car9`'s fitted coefficients and threshold without validation. More separate cold-state recordings, engine-operation context, and health/fault evidence are still needed before revisiting default integration.

The source-file hashes, pinned main revision, inspection script, coverage table, and five-second traces are preserved locally in `results/thermostat_team_data_review_2026-10-18/`. CSV/JSON files remain ignored by Git. The Kaggle comparison and its numerical results are unchanged.

<a id="section-11"></a>

### 11. Full-battery regression comparison — October 18

#### 11.1 Purpose and frozen protocol

This final check asks whether adding the early verdict changes the **existing model's complete fault battery**, rather than testing more variations of the thermostat detector on the same four cases. It uses the final fine-tuned Transformer plus stuck-sensor check, the 90th-percentile calibration cutoff, five-minute baseline grace, six-late-window minimum, and >30% full-drive rule. The early component retains its separately calibrated [2,5) period and OR combination. No thresholds, features, or eligibility rules were tuned after inspecting these results.

Each held-out vehicle's baseline was trained once, then reused for both policies. The CPU run reset seed 0 per vehicle, used two threads and maximum 40/20 epochs with early stopping, and took **495.9 seconds (8 minutes 16 seconds)** including loading, training, scoring, and local output generation, but excluding initial Python imports and subsequent report rendering. This is a fresh paired reference run, not a claim of bit-for-bit reproduction of previously committed aggregate CSV results. The official checkpoint was not retrained or replaced.

The scope includes:

- **All eight evaluable Kaggle vehicles:** 99 source test recordings, of which 74 have enough code-free late windows for clean-control evaluation; 25 remain unjudged.
- **All ten existing fault types**, restricted to vehicles with the required sensor using the original availability rule: 796 injected copies, of which 584 are judged. None of the judged injections is an unchanged copy. Different faults have different denominators because sensor availability differs.
- **The existing synthetic suite:** its original four-vehicle order, scenario RNG seed 42, and 30% clean probability. It assigns one scenario per source recording: 67 recordings, 49 judged and 18 unjudged. The judged set contains 34 injected faults and 15 clean controls.
- **Existing code-event coverage:** `car13` has 162 coded windows; its baseline code-event AUROC is approximately 0.588. The early component is unavailable on that vehicle, so it supplies no new evidence about real code events or real thermostat failures.

The auxiliary model is available only on `car9`, and only four original test recordings have an eligible early verdict. The other seven vehicles retain baseline decisions because the auxiliary model cannot be fitted or calibrated. Their unchanged behavior verifies fallback behavior, not cross-vehicle robustness of the new detector. Eligibility and finite-score masks remain based on the original recording, as in the previous retrospective audit; this does not validate a deployable online qualification policy.

#### 11.2 Full fault-battery results

| Injected fault | Current model: detected/judged | With early check: detected/judged |
|---|---:|---:|
| MAF under-reads 25% | 28/42 | 28/42 |
| MAF drifts to −30% | 15/42 | 15/42 |
| MAP reads +10 kPa | 8/53 | 9/53 |
| Legacy thermostat injection | **29/74** | **29/74** |
| Coolant sensor stuck | 71/74 | 71/74 |
| IAT reads +15 °C | 60/74 | 63/74 |
| Throttle noise | 59/74 | 59/74 |
| MAF under-reads 25%, engine reacts | 31/42 | 31/42 |
| MAP reads +10 kPa, engine reacts | 35/53 | 35/53 |
| Fuel trim +10% | 55/56 | 55/56 |

The legacy thermostat scenario halves the temperature rise; it is **not** the temporal-delay scenario in sections 5–9. Its detection does not improve in this full comparison. The four additional fault alarms are one MAP case (recording 394) and three IAT cases (390, 392, 394), all on `car9`.

![Full-battery detections and false alarms by vehicle](assets/thermostat_final/06_full_battery.png)

**How to read the figure:** higher bars on the left mean more injected-fault detections; higher bars on the right mean more false alarms. Only MAP and IAT detection counts change. The additional clean alarm is on `car9`; unchanged results elsewhere mostly reflect abstention by the additional component.

| Overall measure | Current model | With early check |
|---|---:|---:|
| Detected injected copies | 391/584 | 395/584 |
| Recall across injected copies | 66.95% | 67.64% |
| False alarms on **unique clean controls** | **8/74 (10.81%)** | **9/74 (12.16%)** |
| Fault-weighted precision | 85.19% | 84.04% |
| Fault-weighted F1 | **0.74976** | **0.74953** |
| Fault-weighted accuracy | 77.65% | 77.40% |

“Fault-weighted” follows the repository's aggregate binary-report convention: repeat each vehicle's clean controls once for each evaluated fault type. The confusion matrices are TP/FN/FP/TN = **391/193/68/516** and **395/189/75/509**. The one extra unique clean alarm becomes seven extra weighted false positives because `car9` has seven available fault types. The weighted false-alarm rates are 11.64% and 12.84%; these differ from the unique-control rates above. This is a benchmark weighting, not real-world fault prevalence or 584 independent healthy recordings.

The recall gain is approximately 0.68 percentage points, while precision falls and F1 is practically unchanged, with a very small decrease. That difference is not evidence of a statistically established population-level degradation or equivalence. It does show that this complete run offers **no clear aggregate benefit that justifies enabling the component**.

Two of the four added fault alarms use recording 394, which the combined policy also flags when clean. Those alarms do not establish additional discrimination caused by the injected fault. The two remaining gains are IAT cases, not thermostat cases; this check does not validate correct sensor attribution. No baseline alarm is removed, as guaranteed by OR, and no unjudged recording is promoted into the judged cohort.

#### 11.3 Synthetic-suite results and the pre-fault alarm

| Measure on judged synthetic recordings | Current model | With early check |
|---|---:|---:|
| Detected injected faults | 25/34 | 26/34 |
| False alarms on assigned clean controls | 2/15 | 2/15 |
| Precision | 92.59% | 92.86% |
| Recall | 73.53% | 76.47% |
| F1 | 0.81967 | 0.83871 |
| Accuracy | 77.55% | 79.59% |

The only changed verdict is recording **394**, assigned “MAP reads +8 kPa from minute 5.” The added signal comes from the early [2,5) period, **before the injected MAP fault starts**. The same recording already triggers the check when clean, and its original and injected early deficit scores are identical: both exceed the threshold in 6 of 18 early windows. This direct check is saved locally as `synthetic_394_early_check.csv`. The drive-level answer key nevertheless counts the new alarm as a true positive because that recording eventually contains a fault.

Consequently, the synthetic F1 increase does **not** demonstrate a new response to the MAP fault. The random assignment places this false-alarm-prone original in the faulty subset, so it also does not appear among the 15 assigned clean controls. This is why unchanged synthetic false positives do not contradict the extra false alarm found when all original recordings are evaluated clean. These synthetic scenarios reuse the same source data and are not independent real-fault validation.

#### 11.4 Decision after the complete comparison

**Keep the early check experimental and pause further tuning until independent data are available.** The complete fault battery adds four alarms on injected copies and one alarm on a unique clean control, leaves legacy thermostat detection unchanged, and does not improve aggregate F1. The apparent synthetic-suite gain is explained by a pre-fault alarm on the same problematic original recording.

The earlier temporal-delay sensitivity result remains valid within its narrow scope. The full comparison supplies a stronger reason not to promote that result into a general model-improvement claim. This is a single-seed regression check at the chosen operating point, not a search over architectures or a new population-level study. A combined continuous AUROC, minute-level policy, sensor-attribution rule, and online diagnosis latency are not defined or validated by this drive-verdict OR experiment.

Local evidence is in `results/thermostat_full_2026-10-18/`: source/data hashes and settings in `manifest.json`, coverage and thresholds in `coverage.csv`/`cars.csv`, all paired outcomes in `decisions.csv`, per-fault and per-car summaries, aggregate `metrics.csv`, and `added_alarm_details.csv`. CSV/JSON files remain ignored by Git; this report, the chart, scripts, and tests are the reviewable repository artifacts.

<a id="section-12"></a>

### 12. Recommendation and missing evidence

**Keep the component as an optional experiment.** The evidence supports communicating “greater sensitivity to strong artificial delays in one vehicle”; it does not support “general improvement in thermostat fault detection.”

Before another confirmatory evaluation:

1. Freeze the eligibility policy, periods, combination rule, primary metric, and maximum acceptable increase in false alarms.
2. Collect independent cold starts from multiple vehicles with ECT, IAT, RPM, load, runtime or ignition evidence from the beginning.
3. Keep separate recordings for training, calibration, and testing within each vehicle. Leave new test results unexamined until the method is frozen.
4. Include healthy low-load controls, different temperatures, warm restarts, short drives, and hybrid vehicles where relevant.
5. Obtain real fault cases confirmed by diagnosis or repair to evaluate real-fault detection, rather than sensitivity to simulations alone.
6. Analyze paired differences and uncertainty clustered by vehicle; do not count windows or copies as independent drives.

The minimum of three training recordings and two calibration recordings allows the prototype to run, but does not determine the sample size needed to validate it. That size must be planned around the minimum relevant improvement, tolerable false-alarm rate, and dependence between recordings from the same vehicle.

<a id="section-13"></a>

### 13. Files, evidence, and reproduction

#### Related code

| File | Responsibility |
|---|---|
| [`data.py`](../src/obdfault/data.py) | Preserves engine-runtime provenance without changing the original arrays. |
| [`scoring.py`](../src/obdfault/scoring.py) | Implements the optional `WarmupRise` regression. |
| [`faults.py`](../src/obdfault/faults.py) | Implements the copy with delayed ECT evolution. |
| [`thermostat.py`](../src/obdfault/thermostat.py) | Audits coverage, runs comparisons, and optionally exports scores. |
| [`explore_data.py`](../scripts/explore_data.py) | Audits cold-state coverage without training. |
| [`analyze_thermostat.py`](../scripts/analyze_thermostat.py) | Analyzes robustness, uncertainty, team logs, and full-drive decisions. |
| [`compare_thermostat_full.py`](../scripts/compare_thermostat_full.py) | Runs the paired full fault battery and existing synthetic scenarios at the frozen operating point. |
| [`summarize_thermostat_full.py`](../scripts/summarize_thermostat_full.py) | Summarizes the full comparison and generates its chart without retraining. |
| [`test_thermostat_full.py`](../tests/test_thermostat_full.py) | Checks official verdict/scoring parity, code exclusions, and synthetic scenario assignment. |
| [`build_thermostat_report_figures.py`](../scripts/build_thermostat_report_figures.py) | Generates the five figures and local evidence tables. |
| [`test_thermostat.py`](../tests/test_thermostat.py) and [`test_thermostat_analysis.py`](../tests/test_thermostat_analysis.py) | Verify experiment and analysis behavior. |

#### Running the analysis

With the environment activated, the package installed using `python -m pip install -e ".[dev]"`, and the three Kaggle CSV files in `data/public/kaggle-obd2/`, run the following commands from the repository root. They do not require Bash-specific variable assignments.

To inspect coverage without training, or reproduce the original late-only experiment:

```text
python scripts/explore_data.py --thermostat
python -m obdfault.thermostat --output results/thermostat_late_reproduction --epochs 40 --ft-epochs 20
```

To reproduce the deeper audit:

```text
python -m obdfault.thermostat --early-warmup --save-scores --seed 0 --output results/thermostat_reproduction/seed0
python -m obdfault.thermostat --early-warmup --save-scores --seed 1 --output results/thermostat_reproduction/seed1
python -m obdfault.thermostat --early-warmup --save-scores --seed 2 --output results/thermostat_reproduction/seed2
python scripts/analyze_thermostat.py --experiment results/thermostat_reproduction/seed0 --seed-root results/thermostat_reproduction --output results/thermostat_reproduction/final_audit --team-ref 6b72f3b
python -m pytest -q
```

Each result directory must be new; experiment commands refuse to overwrite an existing directory. Outputs include coverage, exclusions, decisions, summaries, `paired_changes.csv`, and source/data hashes. `--save-scores` exports existing per-window scores without changing training or verdicts. For the early combined method, `first_alarm_min` is the first flagged window of a component with a FAULT period verdict, not the time an online verdict was finalized. To regenerate the documentation figures from the reproduced outputs:

```text
python scripts/build_thermostat_report_figures.py --audit results/thermostat_reproduction/final_audit
```

Figures are regenerated in `docs/assets/thermostat_final/`; this command replaces report assets, not the model's official results. Seeds and library versions may affect numerical values across environments.

To reproduce the October 18 full-battery comparison and its figure:

```text
python scripts/compare_thermostat_full.py --output results/thermostat_full_reproduction --seed 0 --epochs 40 --ft-epochs 20 --threads 2
python scripts/summarize_thermostat_full.py --audit results/thermostat_full_reproduction
```

This trains each held-out vehicle's baseline once and uses that same fit for both policies. It does not retrain or replace the deployed checkpoint. The synthetic suite reuses its original four-vehicle order, seed 42 for scenario assignment, and 30% clean probability. It compares the final fine-tuned model at the chosen operating point, without rerunning alternative architectures, cutoff searches, or experimental scoring variants.

#### Preserved evidence

Complete local results are in `results/thermostat_deep_2026-10-16/`, a directory ignored by Git. Tables used in this report are copied into `docs/assets/thermostat_final/evidence/`, together with audit provenance and hashes. This entire evidence directory is ignored by Git; its CSV and JSON files are not included in commits. The report contains the key results as Markdown tables and includes the PNG charts. Run the reproduction commands above to recreate local analysis tables and audit manifests. Charts were generated from those experimental outputs; curves are reconstructed from the original data and the same seven training recordings, without retraining the Transformer.

This document is the consolidated report for issue #34, including implementation details, the earlier restricted comparison, and the subsequent full-drive audit.

**Scope of completion:** 21 automated tests passed after adding the full-comparison checks; the default model, checkpoint, official tables, and official figures remain unchanged. This report and its assets are documentation for review, not promotion of the experiment to production.

[Back to summary](#summary) · [Back to table of contents](#table-of-contents)

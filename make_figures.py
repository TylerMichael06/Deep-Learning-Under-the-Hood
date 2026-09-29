"""Report figures and the summary table for the final model. Reads the data and the result files in this folder.

Usage (from the Results folder, after obd_model.py and synthetic_test.py have been run):
  python make_figures.py --data "..\\Datasets\\OBD-II datasets"
Writes PNGs and summary_table.csv into Results\\figures\\
"""
import argparse, os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
import obd_model as M
import synthetic_test as S

HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "figures")
BLUE, ORANGE, AQUA, INK, INK2, GRID, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#52514e", "#e4e3df", "#b4b2a9"
NICE = {"RPM": "RPM", "SPEED": "Speed (km/h)", "LOAD": "Engine load (%)", "THROTTLE": "Throttle (%)",
        "ECT": "Coolant temp (Â°C)", "IAT": "Intake air temp (Â°C)", "MAP": "Manifold pressure (kPa)",
        "MAF": "Mass air flow (g/s)", "STFT1": "Short-term fuel trim (%)"}
FINAL = dict(method="fine-tuned (ours)", scoring="stuck check", cutoff=0.95, drive_rule=0.3, grace=5)
SYN_VARIANT = {"shared cutoff": "", "stuck check": "only", "two-speed + stuck check": "stuck"}   # same scorings, as named in synthetic_test.py

plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": .6, "axes.axisbelow": True,
                     "axes.titleweight": "bold", "axes.titlesize": 10, "figure.facecolor": "white"})

def latest(name):
    """Use <name>_new.csv if the script had to save there because the original was open in Excel."""
    new = os.path.join(HERE, name.replace(".csv", "_new.csv"))
    old = os.path.join(HERE, name)
    return new if os.path.exists(new) and (not os.path.exists(old) or os.path.getmtime(new) > os.path.getmtime(old)) else old

def save(fig, name):
    try: fig.savefig(os.path.join(OUT, name), dpi=160, bbox_inches="tight")
    except OSError:                                    # the old image is open in a viewer -> save a copy next to it
        name = name.replace(".png", "_new.png"); fig.savefig(os.path.join(OUT, name), dpi=160, bbox_inches="tight")
    plt.close(fig); print("saved figures\\" + name)

# ---------------------------------------------------------------- 1. the features
PID = {"RPM": "0C", "SPEED": "0D", "LOAD": "04", "THROTTLE": "11", "ECT": "05", "IAT": "0F", "MAP": "0B", "MAF": "10", "STFT1": "06"}

def healthy(trips):
    return [t for t in trips if t["unit"] not in M.FAULTY_CARS]

def table_features(trips):
    """Feature table: what each sensor is, its typical range on healthy cars, and how many cars report it."""
    H = healthy(trips); units = sorted({t["unit"] for t in H})
    X = np.concatenate([t["X"] for t in H])
    has = {u: (~np.isnan(np.concatenate([t["X"] for t in H if t["unit"] == u]))).mean(0) > .5 for u in units}
    rows = []
    for j, c in enumerate(M.CH):
        v = X[:, j][~np.isnan(X[:, j])]; lo, med, hi = np.percentile(v, [5, 50, 95])
        name, unit = (NICE[c].split(" (") + [""])[:2]
        rows.append([name, PID[c], unit.rstrip(")") or "rpm", f"{med:.0f}", f"{lo:.0f} to {hi:.0f}", f"{sum(has[u][j] for u in units)} of {len(units)}"])
    cols = ["Sensor", "PID", "Unit", "Median", "Usual range*", "Cars with it"]
    fig, ax = plt.subplots(figsize=(10, 3.8)); ax.axis("off")
    tb = ax.table(cellText=rows, colLabels=cols, loc="upper center", cellLoc="center", colWidths=[.26, .08, .08, .1, .18, .14])
    fig.text(.08, .02, "* 5th to 95th percentile of healthy driving. PID = the standard OBD-II code used to request the reading.",
             fontsize=8, color=INK2)
    tb.auto_set_font_size(False); tb.set_fontsize(9); tb.scale(1, 1.55)
    for (i, j), cell in tb.get_celld().items():
        cell.set_edgecolor(GRID)
        if i == 0: cell.set_text_props(weight="bold", color=INK); cell.set_facecolor("#f1f0ec")
    ax.set_title("The 9 OBD-II features the model uses (healthy driving, all training cars)", loc="left", fontweight="bold", fontsize=10)
    save(fig, "fig1a_feature_table.png")

def fig_relationships(trips):
    """What the model learns: on a healthy engine the sensors move together."""
    X = np.concatenate([t["X"] for t in healthy(trips)]); ctx = np.concatenate([t["ctx"] for t in healthy(trips)])
    ix = {c: j for j, c in enumerate(M.CH)}
    panels = [(X[:, ix["RPM"]], X[:, ix["MAF"]], "RPM", "Mass air flow (g/s)", "More RPM -> more air in"),
              (X[:, ix["THROTTLE"]], X[:, ix["MAP"]], "Throttle (%)", "Manifold pressure (kPa)", "Open throttle -> higher pressure"),
              (np.minimum(ctx, 40), X[:, ix["ECT"]], "Minutes since engine start", "Coolant temp (Â°C)", "Engine warms up, then levels off")]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, (x, y, xl, yl, title) in zip(axes, panels):
        ok = ~np.isnan(x) & ~np.isnan(y)
        ax.hexbin(x[ok], y[ok], gridsize=40, cmap="Blues", bins="log", mincnt=1, linewidths=0)
        ax.set_xlabel(xl); ax.set_ylabel(yl); ax.set_title(title, loc="left", fontsize=9.5); ax.grid(False)
    fig.suptitle("What the model learns: healthy sensors move together (darker = more readings)", fontweight="bold", x=.01, ha="left", y=1.02)
    fig.text(.01, -.04, "A faulty sensor breaks its usual pattern (e.g. a dirty MAF reads too little air for the RPM), and that is what the model flags.",
             fontsize=8.5, color=INK2)
    fig.tight_layout(); save(fig, "fig1b_sensor_relationships.png")

def fig_coverage_bars(trips):
    """How many cars report each sensor (simple version of the coverage heatmap)."""
    H = healthy(trips); units = sorted({t["unit"] for t in H})
    share = pd.Series({c: np.mean([(~np.isnan(np.concatenate([t["X"] for t in H if t["unit"] == u])[:, j])).mean() > .5 for u in units])
                       for j, c in enumerate(M.CH)}).sort_values()
    fig, ax = plt.subplots(figsize=(7, 3.8)); y = np.arange(len(share))
    ax.barh(y, share * 100, color=BLUE, height=.6); ax.set_yticks(y, [NICE[c].split(" (")[0] for c in share.index]); ax.set_xlim(0, 105)
    for yi, (c, v) in zip(y, share.items()): ax.text(v * 100 + 1.5, yi, f"{round(v * len(units))} of {len(units)} cars", va="center", fontsize=8, color=INK)
    ax.set_xlabel("% of cars that report the sensor"); ax.set_title("Not every car reports every sensor, so the model works with whatever is available", loc="left", fontsize=9.5)
    fig.tight_layout(); save(fig, "fig2_sensor_coverage.png")


# ------------------------------------------------------ 2. which faults / sensors are caught
def fig_faults(R):
    """Per injected fault: share of faulty drives caught and AUROC (final settings, 8 held-out cars)."""
    r = R[(R.method == FINAL["method"]) & (R.scoring == FINAL["scoring"]) & (R.cutoff == FINAL["cutoff"]) & (R.drive_rule == FINAL["drive_rule"])]
    g = r.groupby("fault").agg(drive=("drive_detection", "mean"), auroc=("AUROC", "mean"), right=("blamed_right_sensor", "mean")).sort_values("drive")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True, gridspec_kw=dict(wspace=.08))
    y = np.arange(len(g))
    a1.barh(y, g.drive * 100, color=BLUE, height=.6); a1.set_yticks(y, g.index); a1.set_xlim(0, 100)
    a1.set_xlabel("% of faulty drives flagged FAULT"); a1.set_title("Faulty drives caught", loc="left")
    for yi, v in zip(y, g.drive * 100): a1.text(v + 1.5, yi, f"{v:.0f}%", va="center", fontsize=8, color=INK)
    a2.barh(y, g.auroc, color=BLUE, height=.6); a2.axvline(.5, color=INK2, ls="--", lw=1); a2.set_xlim(.4, 1)
    a2.text(.505, len(g) - .45, "coin flip", fontsize=7, color=INK2); a2.set_xlabel("AUROC (faulty vs clean minutes)")
    a2.set_title("Separation", loc="left")
    for yi, v in zip(y, g.auroc): a2.text(v + .008, yi, f"{v:.2f}", va="center", fontsize=8, color=INK)
    fig.suptitle("Which faults the model catches (final settings, 8 held-out cars)", fontweight="bold", x=.01, ha="left", y=1.02)
    save(fig, "fig3_faults_caught.png")

def fig_right_sensor(R):
    """When the model raised an alarm on a faulty drive, did it name the sensor that was really broken?"""
    r = R[(R.method == FINAL["method"]) & (R.scoring == FINAL["scoring"]) & (R.cutoff == FINAL["cutoff"]) & (R.drive_rule == FINAL["drive_rule"])]
    g = r.groupby("fault").blamed_right_sensor.mean().dropna().sort_values()
    fig, ax = plt.subplots(figsize=(9, 3.8)); y = np.arange(len(g))
    ax.barh(y, g * 100, color=BLUE, height=.6); ax.set_yticks(y, g.index); ax.set_xlim(0, 105)
    for yi, v in zip(y, g * 100): ax.text(v + 1.5, yi, f"{v:.0f}%", va="center", fontsize=8, color=INK)
    ax.set_xlabel("% of FAULT minutes where the model named the broken sensor")
    ax.set_title("Does the model point to the right sensor?", loc="left"); fig.tight_layout(); save(fig, "fig4_right_sensor.png")


# ---------------------------------------------------------- 3. alarms and the threshold
def fig_score_hist(rows):
    """Score distribution of healthy vs faulty minutes relative to each car's alarm line (0 = the line)."""
    m = rows[rows.minute >= FINAL["grace"]]; d = m.score - m.threshold
    bins = np.linspace(np.percentile(d, .5), np.percentile(d, 99.5), 50)
    fig, ax = plt.subplots(figsize=(9, 4.2))
    for sel, col, lab in ((~m.truth, BLUE, "minutes with NO fault (healthy)"), (m.truth, ORANGE, "minutes WITH a fault")):
        ax.hist(d[sel], bins=bins, color=col, alpha=.25, density=True)
        ax.hist(d[sel], bins=bins, color=col, histtype="step", lw=2, density=True, label=lab)
    top = ax.get_ylim()[1]
    ax.axvline(0, color=INK, ls="--", lw=1.2); ax.axvspan(0, bins[-1], color="#f1f0ec", zorder=0)
    ax.text(bins[-1] * .02, top * .95, "alarm line: minutes to the right\nare labelled FAULT", fontsize=8, color=INK, va="top")
    fa, caught = (d[~m.truth] > 0).mean(), (d[m.truth] > 0).mean()
    ax.text(bins[-1], top * .55, f"after the {FINAL['grace']}-min warm-up:\n{caught:.0%} of faulty minutes are to the right (caught)\n"
                                 f"{fa:.0%} of healthy minutes are to the right (false alarms)", fontsize=8.5, color=INK, ha="right")
    ax.set_xlabel("model score minus the car's alarm line   (0 = the alarm line)"); ax.set_ylabel("how common (density)")
    ax.legend(frameon=False, loc="upper left")
    ax.set_title("Healthy vs faulty minutes: how far each is from the alarm line (synthetic test, after warm-up)", loc="left")
    fig.tight_layout(); save(fig, "fig5_score_vs_alarm_line.png")

def fig_roc(rows):
    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    fpr, tpr, _ = roc_curve(rows.truth, rows.score); auc = roc_auc_score(rows.truth, rows.score)
    ax.plot(fpr, tpr, color=BLUE, lw=2, label=f"our model (AUROC {auc:.2f})"); ax.plot([0, 1], [0, 1], color=INK2, ls="--", lw=1, label="coin flip (0.50)")
    ax.set_xlabel("false-alarm rate"); ax.set_ylabel("share of faulty minutes caught (recall)"); ax.set_aspect("equal")
    ax.legend(frameon=False, loc="lower right"); ax.set_title("ROC curve (synthetic test, per minute)", loc="left")
    fig.tight_layout(); save(fig, "fig6_roc_curve.png")

def fig_tradeoff(B):
    """Cutoff trade-off: catches vs false alarms per minute, final method, shared cutoff."""
    b = B[(B.method == FINAL["method"]) & (B.scoring == FINAL["scoring"]) & (B.level == "minute")].sort_values("cutoff")
    names = {0.8: "most relaxed", 0.9: "relaxed", 0.95: "Medium (chosen)", 0.975: "strict", 0.99: "strictest"}
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(b.false_alarm_rate * 100, b.recall * 100, color=BLUE, lw=2, marker="o", ms=7)
    for _, r in b.iterrows():
        ax.annotate(names.get(round(r.cutoff, 3), r.cutoff), (r.false_alarm_rate * 100, r.recall * 100), textcoords="offset points",
                    xytext=(7, -3), fontsize=8, color=INK, fontweight="bold" if r.cutoff == FINAL["cutoff"] else "normal")
    ax.set_xlabel("false alarms (% of clean minutes)"); ax.set_ylabel("faulty minutes caught (%)")
    ax.set_title("Choosing the alarm line: catches vs false alarms", loc="left"); fig.tight_layout(); save(fig, "fig7_cutoff_tradeoff.png")

# ------------------------------------------------------------------ 4. summary table
def summary_table(B, R, syn):
    """One table with the headline numbers for the report (PNG + CSV)."""
    rows = []
    auc = R[(R.scoring == FINAL["scoring"]) & (R.cutoff == FINAL["cutoff"]) & (R.drive_rule == FINAL["drive_rule"])].groupby("method").AUROC.mean()
    for name, label in (("fine-tuned (ours)", "Ours (fine-tuned per car)"), ("pretrained only", "Same net, no fine-tuning"),
                        ("ridge baseline", "Linear (ridge) baseline")):
        for level, lab in (("minute", "per minute"), ("drive (>30% of minutes)", "per drive")):
            b = B[(B.method == name) & (B.scoring == FINAL["scoring"]) & (B.cutoff == FINAL["cutoff"]) & (B.level == level)].iloc[0]
            rows.append(dict(test="Main eval (8 cars)", model=label, level=lab, AUROC=auc[name], accuracy=b.accuracy,
                             precision=b.precision, recall=b.recall, F1=b.F1, false_alarms=b.false_alarm_rate))
    for lab, s in syn.items():
        rows.append(dict(test="Synthetic test (4 cars)", model="Ours (fine-tuned per car)", level=lab, AUROC=s["AUROC"],
                         accuracy=s["accuracy"], precision=s["precision"], recall=s["recall"], F1=s["F1"], false_alarms=s["false_alarm_rate"]))
    T = pd.DataFrame(rows); T.to_csv(os.path.join(OUT, "summary_table.csv"), index=False); print("saved figures\\summary_table.csv")
    show = T.copy()
    for c in ("AUROC", "accuracy", "precision", "recall", "F1"): show[c] = show[c].map(lambda v: f"{v:.2f}")
    show["false_alarms"] = show.false_alarms.map(lambda v: f"{v:.0%}")
    fig, ax = plt.subplots(figsize=(12, .34 * len(show) + .6)); ax.axis("off")
    widths = [.16, .2, .09] + [.07] * 6
    tb = ax.table(cellText=show.values, colLabels=[c.replace("_", " ") for c in show.columns], loc="upper center",
                  cellLoc="center", colLoc="center", colWidths=widths)
    tb.auto_set_font_size(False); tb.set_fontsize(8.5); tb.scale(1, 1.5)
    for (i, j), cell in tb.get_celld().items():
        cell.set_edgecolor(GRID)
        if i == 0: cell.set_text_props(weight="bold", color=INK); cell.set_facecolor("#f1f0ec")
        elif "ours" in show.iloc[i - 1].model.lower(): cell.set_text_props(weight="bold")
    ax.set_title(f"Results summary (final settings: {FINAL['scoring']}, Medium cutoff, 5-min warm-up, drive = FAULT if >30% of minutes flagged)",
                 loc="left", fontweight="bold", fontsize=10)
    save(fig, "table1_results_summary.png")

# ------------------------------------------------------------------ 5. helpers for the final scores
def synthetic_scores(tag, scoring):
    """Re-apply the final alarm rules to a saved synthetic run. Returns (minute rows, drive key, metrics per level)."""
    rows, key = pd.read_csv(os.path.join(HERE, f"synthetic_minutes{tag}.csv")), pd.read_csv(os.path.join(HERE, f"synthetic_answer_key{tag}.csv"))
    rows, key = S.decide(rows, key, FINAL["grace"], FINAL["drive_rule"], SYN_VARIANT[scoring])
    auc = roc_auc_score(rows.truth, rows.score)
    return rows, key, {"per minute": dict(AUROC=auc, **S.metrics(rows.truth.to_numpy(), rows.flagged.to_numpy())),
                       "per drive": dict(AUROC=auc, **S.metrics((key.truth == "FAULT").to_numpy(), (key.model_says == "FAULT").to_numpy()))}

def main_scores(B, R, scoring):
    auc = R[(R.method == FINAL["method"]) & (R.scoring == scoring) & (R.cutoff == FINAL["cutoff"]) & (R.drive_rule == FINAL["drive_rule"])].AUROC.mean()
    out = {}
    for level, lab in (("minute", "per minute"), ("drive (>30% of minutes)", "per drive")):
        b = B[(B.method == FINAL["method"]) & (B.scoring == scoring) & (B.cutoff == FINAL["cutoff"]) & (B.level == level)].iloc[0]
        out[lab] = dict(AUROC=auc, accuracy=b.accuracy, precision=b.precision, recall=b.recall, F1=b.F1, false_alarm_rate=b.false_alarm_rate)
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True, help="folder with the three Kaggle exp*.csv files")
    ap.add_argument("--scoring", default=FINAL["scoring"], choices=list(SYN_VARIANT), help="the final scoring to show")
    a = ap.parse_args(); os.makedirs(OUT, exist_ok=True); FINAL["scoring"] = a.scoring
    trips = M.load_kaggle(a.data)
    R, B = pd.read_csv(latest("obd_results_faults.csv")), pd.read_csv(latest("obd_results_binary.csv"))
    rows, key, syn = synthetic_scores("", a.scoring)
    for old in ("fig1_sensors_example_drive.png", "fig4_blamed_sensor.png"):   # replaced by clearer versions
        if os.path.exists(os.path.join(OUT, old)): os.remove(os.path.join(OUT, old))
    table_features(trips); fig_relationships(trips); fig_coverage_bars(trips)
    fig_faults(R); fig_right_sensor(R)
    fig_score_hist(rows); fig_roc(rows); fig_tradeoff(B)
    S.timelines(rows, key, os.path.join(OUT, "fig10_timelines.png"), grace=FINAL["grace"])
    summary_table(B, R, syn)

if __name__ == "__main__":
    main()

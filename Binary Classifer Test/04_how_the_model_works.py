"""Step 4: a picture of how the model works, on two real test windows.

data (100 readings) -> summarize (average + spread per sensor) -> model -> prediction

Run:  python 04_how_the_model_works.py   ->  charts/4_how_the_model_works.png
Uses the Approach B random forest. The two example windows are the MIDDLE test
window of Rich mixture and of Lean mixture, not picked for looking good.
"""
import importlib

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

from common import FEATURES, INK, INK2, MODE_COLORS, MODES, OUT, WINDOW, headline, load, test_mask

w = importlib.import_module("03_windowed_model")

SHOWN = ["HC", "CO", "Consumption L/H"]  # three of the 14 sensors, to keep the picture readable
SHORT = {"HC": "Hydrocarbons, HC (ppm)", "CO": "Carbon monoxide, CO (%)", "Consumption L/H": "Fuel use (L/h)"}
EXAMPLES = [1, 2]  # true failure modes to illustrate

df = load()
X, y = df[list(FEATURES)], df["Fault"]
test = test_mask(y)

# Train the Approach B model.
F_tr, L_tr = w.make_windows(X, y, ~test, w.TRAIN_STRIDE)
model = w.make_models()["Random\nforest"].fit(F_tr, L_tr)
forest = model.steps[-1][1]
share_avg = forest.feature_importances_[:14].sum()

# What a "normal" window's spread looks like, per sensor (median over No-fault training windows).
normal_rows = X[(y == 0).to_numpy() & ~test]
normal_std = np.median([normal_rows.iloc[s:s + WINDOW].std().to_numpy()
                        for s in range(0, len(normal_rows) - WINDOW + 1, WINDOW)], axis=0)
col = {c: i for i, c in enumerate(FEATURES)}

fig = plt.figure(figsize=(15.5, 9.6))
bg = fig.add_axes([0, 0, 1, 1])
bg.set_xlim(0, 1)
bg.set_ylim(0, 1)
bg.axis("off")
headline(fig, "How the model turns raw sensor readings into a failure-mode prediction",
         f"Two real test windows the model never saw in training. Each is {WINDOW} consecutive readings from one failure mode.")

for cx, txt, sub in [(0.17, "1  The data", "3 of the 14 sensors shown"),
                     (0.455, "2  Summarize each sensor", "the model gets all 14: 28 numbers per window"),
                     (0.665, "3  The model", ""), (0.865, "4  The prediction", "")]:
    bg.text(cx, 0.885, txt, ha="center", fontsize=12.5, fontweight="bold")
    bg.text(cx, 0.86, sub, ha="center", fontsize=9, color=INK2)

row_top = {0: 0.83, 1: 0.42}
for r, true_mode in enumerate(EXAMPLES):
    rows = X[(y == true_mode).to_numpy() & test]
    k = (len(rows) // WINDOW) // 2
    win = rows.iloc[k * WINDOW:(k + 1) * WINDOW]
    summary = w.summarize(win)                     # 14 averages then 14 spreads
    proba = model.predict_proba(summary.reshape(1, -1))[0]
    pred = int(forest.classes_[proba.argmax()])
    top = row_top[r]

    bg.text(0.015, top, f"Example {'AB'[r]}", fontsize=11.5, fontweight="bold", va="top")
    bg.text(0.015, top - 0.022, f"really:\n{true_mode}  {MODES[true_mode][0]}", fontsize=10.5, va="top",
            color=INK2, linespacing=1.4)

    for s, feat in enumerate(SHOWN):
        a_top = top - 0.075 - s * 0.10
        ax = fig.add_axes([0.10, a_top - 0.07, 0.165, 0.07])
        ax.plot(range(WINDOW), win[feat].to_numpy(), color=MODE_COLORS[true_mode], lw=1.4)
        ax.set_title(SHORT[feat], fontsize=8.5, color=INK2, loc="left", pad=6)
        ax.tick_params(labelsize=7.5, length=0)
        ax.set_xticks([0, WINDOW - 1], ["1st reading", "100th"] if s == 2 else ["", ""])
        ax.spines[["left", "bottom"]].set_color("#e6e5e1")
        avg, std = win[feat].mean(), win[feat].std()
        mult = std / normal_std[col[feat]]
        cy = a_top - 0.035
        bg.text(0.35, cy + 0.017, f"average  {avg:.2f}", fontsize=10, color=INK2, va="center")
        bg.text(0.35, cy - 0.013, f"spread  {std:.2f}", fontsize=11.5, fontweight="bold", va="center")
        bg.text(0.455, cy - 0.013, f"= {mult:.1f}x a normal window's", fontsize=9, color=INK2, va="center")
        bg.annotate("", xy=(0.343, cy), xytext=(0.272, cy), arrowprops=dict(arrowstyle="->", color="#b9b8b0", lw=1.2))

    mid = top - 0.075 - 0.10 - 0.035
    bg.add_patch(FancyBboxPatch((0.585, mid - 0.095), 0.16, 0.19, boxstyle="round,pad=0.005,rounding_size=0.012",
                                fc="#eef1f6", ec="#b9b8b0", lw=1.2))
    bg.text(0.665, mid + 0.03, "Random forest", ha="center", fontsize=11.5, fontweight="bold")
    bg.text(0.665, mid - 0.035, "300 decision trees.\nEach one looks at the\n28 numbers and votes\nfor a failure mode.",
            ha="center", va="center", fontsize=8.8, color=INK2, linespacing=1.4)
    for x0, x1 in [(0.555, 0.583), (0.747, 0.775)]:
        bg.annotate("", xy=(x1, mid), xytext=(x0, mid), arrowprops=dict(arrowstyle="->", color="#7c7b74", lw=1.8))

    axp = fig.add_axes([0.87, mid - 0.09, 0.10, 0.16])
    ms = list(MODES)
    vals = [proba[list(forest.classes_).index(m)] * 100 for m in ms]
    bars = axp.barh(range(4), vals, color=[MODE_COLORS[m] for m in ms], height=0.62)
    bars[pred].set_edgecolor(INK)
    bars[pred].set_linewidth(2)
    for i, v in enumerate(vals):
        axp.text(v + 2, i, f"{v:.0f}%", va="center", fontsize=9, fontweight="bold" if i == pred else "normal")
    axp.set_yticks(range(4), [f"{m} {MODES[m][0]}" for m in ms], fontsize=8.5)
    axp.invert_yaxis()
    axp.set_xlim(0, 118)
    axp.set_xticks([])
    axp.spines[["left", "bottom"]].set_visible(False)
    axp.tick_params(length=0)
    axp.set_title("share of the 300 trees\nvoting for each mode", fontsize=8, color=INK2, pad=6)
    ok = pred == true_mode
    bg.text(0.865, mid - 0.125, f"Model says: {pred}  {MODES[pred][0]}", ha="center", fontsize=10.5, fontweight="bold")
    bg.text(0.865, mid - 0.155, "✓ correct" if ok else f"✗ wrong (truth: {true_mode}  {MODES[true_mode][0]})",
            ha="center", fontsize=10.5, fontweight="bold", color="#184f95" if ok else "#a3360f")

bg.text(0.015, 0.035, "Spread = standard deviation: how much a sensor's readings jump around within the window. "
        "'Normal window' = typical No-fault window in the training data.", fontsize=8.5, color=INK2)
bg.text(0.015, 0.012, f"In this trained forest, about {share_avg:.0%} of the decision weight is on the 14 averages "
        f"and {1 - share_avg:.0%} on the 14 spreads.", fontsize=8.5, color=INK2)
fig.savefig(OUT / "4_how_the_model_works.png", dpi=150)
print("wrote", OUT / "4_how_the_model_works.png")

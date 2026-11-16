"""Draw the report's pipeline figure: how one drive goes from raw sensors to HEALTHY / FAULT / TOO SHORT.

  python scripts/make_pipeline_figure.py      # saves figures/fig0_pipeline.png and .pdf
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from obdfault.config import FIGURES

# same palette as obdfault.figures
BLUE, ORANGE, AQUA, INK, INK2, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#52514e", "#b4b2a9"
W, H = 7.2, 3.0

fig = plt.figure(figsize=(W, H))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")

def box(x, y, w, h, title, body, edge=INK2, fill="#f6f6f4", dashed=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.06", lw=1.1,
                                ec=edge, fc=fill, ls=(0, (4, 2)) if dashed else "-"))
    ax.text(x + w / 2, y + h - 0.13, title, ha="center", va="top", fontsize=8.5, fontweight="bold", color=INK)
    ax.text(x + w / 2, y + h - 0.36, body, ha="center", va="top", fontsize=7, color=INK2, linespacing=1.3)

def arrow(p, q, color=INK2, dashed=False):
    ax.annotate("", q, p, arrowprops=dict(arrowstyle="-|>", color=color, lw=1.1, mutation_scale=9,
                                          ls=(0, (4, 2)) if dashed else "-", shrinkA=0, shrinkB=0))

def pill(x, y, w, h, label, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.12", lw=1.3, ec=color, fc=color + "22"))
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=8.5, fontweight="bold", color=color)

# ---- row 1: every 1-minute window ------------------------------------------------------------
ax.text(0.08, 2.88, "For every minute of driving", fontsize=8, fontweight="bold", color=INK2, va="center")
bw, bh, gap, y1 = 1.25, 0.95, 0.2, 1.78
xs = [0.08 + i * (bw + gap) for i in range(5)]
row1 = [("1. Read", "9 engine sensors\nfor one minute", INK2, "#f6f6f4"),
        ("2. Guess", "the model guesses\neach sensor from\nthe other 8", BLUE, BLUE + "14"),
        ("3. Compare", "guess vs. actual\nreading; also, is a\nsensor stuck?", INK2, "#f6f6f4"),
        ("4. Unusual?", "compared with this\ncar's own healthy\ndrives", INK2, "#f6f6f4"),
        ("5. Flag", "if the gap is bigger\nthan in 9 of 10\nhealthy minutes", INK2, "#f6f6f4")]
for x, (t, b, e, f) in zip(xs, row1):
    box(x, y1, bw, bh, t, b, e, f)
for a, b in zip(xs, xs[1:]):
    arrow((a + bw, y1 + bh / 2), (b, y1 + bh / 2))

# wrap-around connector from the alarm threshold down to the drive rule
yc = 1.55; xr = xs[-1] + bw / 2; xd = 0.08 + 1.2
ax.plot([xr, xr, xd], [y1, yc, yc], color=INK2, lw=1.1)
arrow((xd, yc), (xd, 1.30))
ax.text(xr - 0.08, yc + 0.07, "flagged minutes", ha="right", va="bottom", fontsize=7, color=INK2, style="italic")

# ---- row 2: every drive ----------------------------------------------------------------------
ax.text(0.08, 1.42, "For every drive", fontsize=8, fontweight="bold", color=INK2, va="center")
dx, dy, dw, dh = 0.08, 0.1, 2.4, 1.2
box(dx, dy, dw, dh, "6. Judge the whole drive",
    "skip the first 5 min (engine warming up)\nunder 1 min left → TOO SHORT\nover 30% of minutes flagged → FAULT\notherwise → HEALTHY")
pw, ph, px = 1.05, 0.3, dx + dw + 0.45
outs = [("TOO SHORT", GREY, 0.95), ("FAULT", ORANGE, 0.55), ("HEALTHY", AQUA, 0.15)]
for label, color, py in outs:
    pill(px, py, pw, ph, label, "#8a8880" if label == "TOO SHORT" else color)
    arrow((dx + dw, dy + dh / 2), (px, py + ph / 2))

mx = px + pw + 1.0; mw = W - mx - 0.08
box(mx, dy, mw, dh, "Extra airflow (MAF) check",
    "Is the airflow sensor reading low\nfor the whole drive, and getting\nworse drive after drive?",
    edge=ORANGE, fill="#ffffff", dashed=True)
arrow((mx, 0.55 + ph / 2), (px + pw, 0.55 + ph / 2), color=ORANGE, dashed=True)
ax.text((px + pw + mx) / 2, 0.55 + ph / 2 - 0.07, "shown separately;\ncan also count\nas FAULT", ha="center", va="top",
        fontsize=6.5, color=ORANGE, linespacing=1.2)

os.makedirs(FIGURES, exist_ok=True)
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(FIGURES, f"fig0_pipeline.{ext}"), dpi=300, facecolor="white")
print("saved figures/fig0_pipeline.png and .pdf")

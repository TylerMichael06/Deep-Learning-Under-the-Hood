"""Step 1: look at the raw data, grouped by failure mode.

Run:  python 01_explore_data.py   ->  charts/1_raw_data_by_failure_mode.png
"""
import matplotlib.pyplot as plt

from common import FEATURES, MODE_COLORS, MODES, OUT, INK2, headline, load

df = load()

fig, axes = plt.subplots(4, 4, figsize=(14, 12.5))
fig.subplots_adjust(top=0.87, left=0.04, right=0.985, bottom=0.05, hspace=0.55, wspace=0.28)

for ax, (col, label) in zip(axes.flat, FEATURES.items()):
    groups = [df.loc[df.Fault == m, col] for m in MODES]
    bp = ax.boxplot(groups, positions=range(4), widths=0.6, whis=(5, 95), showfliers=False,
                    patch_artist=True, medianprops=dict(color="#0b0b0b", lw=1.8),
                    whiskerprops=dict(color=INK2, lw=1), capprops=dict(color=INK2, lw=1))
    for patch, m in zip(bp["boxes"], MODES):
        patch.set(facecolor=MODE_COLORS[m], edgecolor=MODE_COLORS[m], alpha=0.85)
    ax.set_xticks(range(4), [str(m) for m in MODES])
    ax.set_title(label, fontsize=10, fontweight="bold", loc="left")
    ax.yaxis.grid(True, color="#e6e5e1", lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    ax.spines[["left", "bottom"]].set_visible(False)

# Last two panels: the key, and how to read a box.
key = axes.flat[14]
key.axis("off")
key.set_title("Failure modes (x-axis number)", fontsize=10, fontweight="bold", loc="left")
for i, (m, (name, desc)) in enumerate(MODES.items()):
    y = 0.80 - i * 0.22
    key.add_patch(plt.Rectangle((0, y - 0.07), 0.09, 0.14, color=MODE_COLORS[m], transform=key.transAxes))
    key.text(0.14, y + 0.03, f"{m}  {name}", fontsize=10.5, fontweight="bold", va="center", transform=key.transAxes)
    key.text(0.14, y - 0.06, desc, fontsize=9, color=INK2, va="center", transform=key.transAxes)

how = axes.flat[15]
how.axis("off")
how.set_title("How to read a box", fontsize=10, fontweight="bold", loc="left")
how.text(0, 0.62, "Box = the middle 50% of readings.\nBlack line = the median.\n"
         "Whiskers = 5th to 95th percentile.\n\nLess overlap between boxes means\nthat sensor tells the modes apart.",
         fontsize=9.5, color=INK2, va="center", transform=how.transAxes, linespacing=1.5)

headline(fig, "What each of the 14 sensors reads under each failure mode",
         f"All {len(df):,} rows of EngineFaultDB, grouped by labeled failure mode. "
         "Each panel is one sensor; each box is one failure mode.")
fig.savefig(OUT / "1_raw_data_by_failure_mode.png", dpi=150)
print("wrote", OUT / "1_raw_data_by_failure_mode.png")

"""Shared bits: data loading, the four failure modes, the train/test split,
the results-table drawing, and the chart style."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
DATA = HERE / "data" / "EngineFaultDB_Final.csv"
OUT = HERE / "charts"
OUT.mkdir(exist_ok=True)

SEED = 0
WINDOW = 100     # readings per test (one test = 100 consecutive rows of one failure mode)
RIGHT_AT = 0.90  # a model "got it right" on a mode if >= 90% of that mode's tests are named correctly

# From the EngineFaultDB paper.
MODES = {
    0: ("No fault", "engine running normally"),
    1: ("Rich mixture", "too much fuel, too little air"),
    2: ("Lean mixture", "too much air, too little fuel"),
    3: ("Low voltage", "electrical system under-voltage"),
}

# Column -> label with unit (units from the dataset README).
FEATURES = {
    "MAP": "Manifold pressure, MAP (kPa)",
    "TPS": "Throttle position, TPS (%)",
    "Force": "Force (N)",
    "Power": "Power (kW)",
    "RPM": "Engine speed (RPM)",
    "Consumption L/H": "Fuel use (L/h)",
    "Consumption L/100KM": "Fuel use (L/100 km)",
    "Speed": "Speed (km/h)",
    "CO": "Carbon monoxide, CO (%)",
    "HC": "Hydrocarbons, HC (ppm)",
    "CO2": "Carbon dioxide, CO2 (%)",
    "O2": "Oxygen, O2 (%)",
    "Lambda": "Air-fuel equivalence, Lambda",
    "AFR": "Air-fuel ratio, AFR",
}

# One color per failure mode (fixed order, same in every chart).
MODE_COLORS = {0: "#2a78d6", 1: "#eb6834", 2: "#1baf7a", 3: "#eda100"}
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.edgecolor": GRID, "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
})


def load():
    """Return the dataset with the exact duplicate row removed."""
    return pd.read_csv(DATA).drop_duplicates().reset_index(drop=True)


def test_mask(y):
    """True for the LAST 20% of each failure mode's block of rows.

    The file stores each failure mode as one continuous block of near-identical
    neighbouring rows, so we hold out the end of each block (a shuffled split
    would let the model memorize neighbours of test rows)."""
    mask = np.zeros(len(y), dtype=bool)
    for m in MODES:
        idx = np.flatnonzero(np.asarray(y) == m)
        mask[idx[int(len(idx) * 0.8):]] = True
    return mask


def headline(fig, title, sub):
    fig.text(0.02, 0.985, title, fontsize=15, fontweight="bold", ha="left", va="top")
    fig.text(0.02, 0.945, sub, fontsize=10.5, color=INK2, ha="left", va="top", linespacing=1.5)


def draw_table(results, title, sub, footer, path):
    """Rows = failure modes, columns = models.
    results[model name][mode] = (tests right, tests total, most common wrong answer or None)."""
    names = list(results)
    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    fig.subplots_adjust(top=0.80, left=0.03, right=0.985, bottom=0.07)
    ax.set_xlim(0, 1 + len(names) * 1.25)
    ax.set_ylim(len(MODES) - 0.4, -1.05)
    ax.axis("off")
    left = 1.0  # width of the row-label column
    for j, n in enumerate(names):
        ax.text(left + j * 1.25 + 0.625, -0.72, n, ha="center", va="center", fontsize=11, fontweight="bold")
    ax.text(0, -0.72, "Engine failure mode\n(the truth)", ha="left", va="center", fontsize=10, color=INK2)
    for i, m in enumerate(MODES):
        name, desc = MODES[m]
        ax.text(0, i, f"{m}  {name}", ha="left", va="center", fontsize=11.5, fontweight="bold")
        ax.text(0, i + 0.27, desc, ha="left", va="center", fontsize=8.5, color=INK2)
        for j, n in enumerate(names):
            hit, total, confused = results[n][m]
            share = hit / total
            ok = share >= RIGHT_AT
            x0 = left + j * 1.25
            ax.add_patch(plt.Rectangle((x0 + 0.03, i - 0.44), 1.19, 0.88, edgecolor="none",
                                       facecolor="#dceaf9" if ok else "#fbdccd"))
            ax.text(x0 + 0.625, i - 0.20, "✓ RIGHT" if ok else "✗ WRONG", ha="center", va="center", fontsize=9,
                    fontweight="bold", color="#184f95" if ok else "#a3360f")
            ax.text(x0 + 0.625, i + 0.06, f"{hit} of {total}", ha="center", va="center", fontsize=17,
                    fontweight="bold")
            note = f"tests named right ({share:.0%})" if ok or confused is None else (
                f"wrong ones mostly said: {confused} ({MODES[confused][0]})")
            ax.text(x0 + 0.625, i + 0.31, note, ha="center", va="center", fontsize=8, color=INK2)
    headline(fig, title, sub)
    fig.text(0.02, 0.02, footer, fontsize=8.5, color=INK2)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print("wrote", path)

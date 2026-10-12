"""Figures for the power plant case study, using the results reported in the study.

Run from the repository root: python src/make_study_figures.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT.parent / "figures"
DATA = ROOT.parent / "data" / "ccpp.csv"

INK, MUTED, GRID, ACCENT, SOFT = "#1d1d1f", "#6e6e73", "#e5e5ea", "#2a78d6", "#b9c6d6"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "text.color": INK,
    "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "axes.titlesize": 11,
    "axes.titleweight": "bold", "axes.titlelocation": "left", "savefig.dpi": 170,
    "savefig.bbox": "tight", "savefig.facecolor": "white",
})

df = pd.read_csv(DATA)

# 1. Two peaks in output and temperature (summer / winter regimes)
fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))
for ax, col, label in ((axes[0], "PE", "Net electrical output (MW)"), (axes[1], "AT", "Ambient temperature (°C)")):
    ax.hist(df[col], bins=60, color=ACCENT, alpha=0.9)
    ax.set_xlabel(label)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
axes[0].set_title("Output")
axes[1].set_title("Temperature")
fig.tight_layout()
fig.savefig(OUT / "pp_distributions.png")

# 2. Model comparison: test R² reported in the study
models = ["Neural network\n(2 hidden layers)", "Linear regression\n(final model)", "Lasso", "Ridge", "Polynomial\n(degree 2)"]
r2 = [0.912, 0.921, 0.926, 0.926, 0.930]
fig, ax = plt.subplots(figsize=(8, 3.2))
ys = range(len(models))
ax.scatter(r2, ys, s=[140 if "final" in m else 80 for m in models],
           color=[ACCENT if "final" in m else SOFT for m in models], zorder=3)
for y, v in zip(ys, r2):
    ax.text(v + 0.0012, y, f"{v:.3f}", va="center", fontsize=9, color=INK)
ax.set_yticks(list(ys), models)
ax.set_xlim(0.905, 0.936)
ax.set_xlabel("R² on the test split (higher is better)")
ax.xaxis.grid(True, color=GRID)
ax.set_axisbelow(True)
ax.spines["left"].set_visible(False)
ax.tick_params(axis="y", length=0)
fig.tight_layout()
fig.savefig(OUT / "pp_model_comparison.png")

# 3. Five-fold cross-validation of the final linear model
folds = [0.9235, 0.9276, 0.9278, 0.9341, 0.9230]
fig, ax = plt.subplots(figsize=(8, 2.6))
ax.plot(range(1, 6), folds, "o-", color=ACCENT, lw=2, ms=7)
ax.axhline(sum(folds) / 5, color=MUTED, lw=1, ls="--")
ax.text(0.8, sum(folds) / 5 + 0.0012, f"mean R² {sum(folds) / 5:.3f}", color=MUTED, fontsize=9)
ax.set_xticks(range(1, 6), [f"Fold {i}" for i in range(1, 6)])
ax.set_xlim(0.6, 5.4)
ax.set_ylim(0.915, 0.94)
ax.set_ylabel("R²")
ax.yaxis.grid(True, color=GRID)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(OUT / "pp_cross_validation.png")
print("figures written to", OUT)

"""Presentation charts for the EE52037 viva deck.

Sized and weighted for projection, not print: large type, minimal ink,
recessive grid, direct value labels. Palette slots 1 and 2 of the validated
categorical order (validator: all checks pass, light mode).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

OUT = Path(__file__).parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, MUTED, GRID, SURFACE = "#1a1a1a", "#5a5a5a", "#e6e6e3", "#fcfcfb"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 15,
    "text.color": INK,
    "axes.labelcolor": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
})


def style(ax, ymax=100, ylabel="Accuracy (%)"):
    ax.set_ylim(0, ymax)
    ax.set_ylabel(ylabel, fontsize=15, color=MUTED)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)
    ax.yaxis.grid(True, color=GRID, linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def label(ax, bars, fmt="{:.1f}", dy=1.5, size=15, weight="bold"):
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + dy,
                fmt.format(b.get_height()), ha="center", va="bottom",
                fontsize=size, fontweight=weight, color=INK)


# ---------------------------------------------------------------- 1. collapse
fig, ax = plt.subplots(figsize=(10, 5.0))
conds = ["Trained clean\nTested clean", "Trained clean\nTested noisy", "Trained noisy\nTested noisy"]
vals = [94.00, 19.98, 75.81]
cols = [BLUE, ORANGE, BLUE]
bars = ax.bar(conds, vals, color=cols, width=0.55, zorder=3)
label(ax, bars)
ax.axhline(20, color=MUTED, linewidth=1.5, linestyle=(0, (4, 4)), zorder=2)
ax.text(2.42, 21.5, "chance (20%)", fontsize=13, color=MUTED, ha="right")
style(ax)
ax.set_title("It works in simulation, and fails on realistic measurements",
             fontsize=18, fontweight="bold", color=INK, pad=16, loc="left")
fig.tight_layout()
fig.savefig(OUT / "p_collapse.png", dpi=200)
plt.close(fig)

# ------------------------------------------------------- 2. protocol inversion
fig, ax = plt.subplots(figsize=(10, 5.0))
comp = ["Gaussian\nnoise", "Contact\nimpedance", "Electrode\nbias", "Quantisation"]
matched = [96.61, 90.11, 78.68, 92.89]
deploy = [20.88, 19.72, 57.79, 14.03]
x = np.arange(len(comp))
w = 0.38
b1 = ax.bar(x - w / 2 - 0.01, matched, w, label="Tested on the same noise it trained on",
            color=BLUE, zorder=3)
b2 = ax.bar(x + w / 2 + 0.01, deploy, w, label="Tested on the noise a real device meets",
            color=ORANGE, zorder=3)
label(ax, b1, size=13)
label(ax, b2, size=13)
ax.set_xticks(x)
ax.set_xticklabels(comp)
style(ax, ymax=118)
ax.legend(frameon=False, fontsize=14, loc="upper center",
          bbox_to_anchor=(0.5, 1.02), ncol=1, labelcolor=MUTED)
ax.set_title("The same experiment, two protocols, opposite answers",
             fontsize=18, fontweight="bold", color=INK, pad=16, loc="left")
fig.tight_layout()
fig.savefig(OUT / "p_inversion.png", dpi=200)
plt.close(fig)

# ------------------------------------------------------------- 3. permutation
fig, ax = plt.subplots(figsize=(10, 5.0))
models = ["CNN", "Random\nForest", "MLP", "SVM"]
orig = [76.21, 60.03, 56.96, 52.53]
perm = [45.97, 59.12, 56.99, 52.53]
x = np.arange(len(models))
b1 = ax.bar(x - w / 2 - 0.01, orig, w, label="Original measurement order", color=BLUE, zorder=3)
b2 = ax.bar(x + w / 2 + 0.01, perm, w, label="Measurements shuffled", color=ORANGE, zorder=3)
label(ax, b1, size=13)
label(ax, b2, size=13)
ax.set_xticks(x)
ax.set_xticklabels(models)
style(ax, ymax=104)
ax.legend(frameon=False, fontsize=14, loc="upper right", labelcolor=MUTED)
ax.set_title("Only the CNN depends on measurement order",
             fontsize=18, fontweight="bold", color=INK, pad=26, loc="left")
ax.text(0, 1.015, "CNN falls 30.2 points; the other three move by at most 0.9",
        transform=ax.transAxes, fontsize=14, color=MUTED, va="bottom")
fig.tight_layout()
fig.savefig(OUT / "p_permutation.png", dpi=200)
plt.close(fig)

# ---------------------------------------------------------------- 4. rho sweep
fig, ax = plt.subplots(figsize=(10, 5.0))
rho = [0, 25, 50, 75, 90, 100]
acc = [75.80, 76.46, 77.47, 78.53, 82.85, 96.99]
ax.plot(rho, acc, color=BLUE, linewidth=2.5, marker="o", markersize=9,
        markerfacecolor=BLUE, markeredgecolor=SURFACE, markeredgewidth=2, zorder=3)
for xi, yi in zip(rho, acc):
    ax.text(xi, yi + 2.6, f"{yi:.1f}", ha="center", fontsize=13,
            fontweight="bold", color=INK)
ax.set_xlabel("Share of electrode error that cancels between the two frames (%)",
              fontsize=15, color=MUTED)
style(ax, ymax=110)
ax.set_ylim(70, 110)
ax.set_xticks(rho)
ax.set_title("The worst case holds until cancellation is near-perfect",
             fontsize=18, fontweight="bold", color=INK, pad=16, loc="left")
fig.tight_layout()
fig.savefig(OUT / "p_rho.png", dpi=200)
plt.close(fig)

print("wrote:", *[p.name for p in sorted(OUT.glob("*.png"))])

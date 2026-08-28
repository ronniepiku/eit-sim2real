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
style(ax)
ax.set_title("Classification accuracy under varying noise configurations",
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
ax.set_title("Matched vs Deployment Accuracy",
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

# ---------------------------------------------------------- 5. EIT sensing diagram
# Schematic of the sensing principle (mirrors dissertation Fig. 2.2 geometry:
# 16-electrode boundary, current-injection pair, voltage-measurement pair)
# extended rightward into the classification pipeline. Landscape layout to
# fit a wide, short column on the "problem" slide.
fig, ax = plt.subplots(figsize=(7.8, 4.5))
ax.set_aspect("equal")
ax.axis("off")

RED = "#c0392b"
R, cx, cy = 1.45, -2.55, 0.05

domain = plt.Circle((cx, cy), R, facecolor="#eef4fc", edgecolor=INK, linewidth=1.6, zorder=2)
ax.add_patch(domain)

# touch perturbation (placed away from both electrode pairs, below-right of centre)
touch = plt.Circle((cx + 0.15, cy - 0.15), 0.48, facecolor=ORANGE, alpha=0.55, edgecolor="none", zorder=3)
ax.add_patch(touch)
ax.annotate("touch", xy=(cx + 0.45, cy - 0.42), xytext=(cx + 0.75, cy - 1.55),
            fontsize=11.5, color=INK, ha="left", zorder=7,
            arrowprops=dict(arrowstyle="-", color=MUTED, lw=1.0))

# 16 boundary electrodes
n = 16
angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
ex = cx + R * np.cos(angles)
ey = cy + R * np.sin(angles)
ax.scatter(ex, ey, s=36, color="#8a8a86", edgecolor=SURFACE, linewidth=0.8, zorder=5)


def label_pair(idx0, idx1, text, color, dashed):
    ax.scatter([ex[idx0], ex[idx1]], [ey[idx0], ey[idx1]], s=46, color=color,
               edgecolor=SURFACE, linewidth=0.8, zorder=6)
    style = dict(arrowstyle="-", color=color, lw=1.6, linestyle=(0, (3, 2))) if dashed \
        else dict(arrowstyle="-|>", color=color, lw=1.8)
    ax.annotate("", xy=(ex[idx1], ey[idx1]), xytext=(ex[idx0], ey[idx0]), zorder=6,
                arrowprops=dict(**style, connectionstyle="arc3,rad=0.4", shrinkA=5, shrinkB=5))
    mid = (angles[idx0] + angles[idx1]) / 2
    lr = R * 1.30
    ax.text(cx + lr * np.cos(mid), cy + lr * np.sin(mid), text, fontsize=13,
            color=color, fontweight="bold", ha="center", va="center", zorder=7)


# current injection pair: upper-left. voltage pair: lower-left. Both clear of
# the touch, the pipeline exit (right), and the caption (bottom).
label_pair(5, 6, "I", RED, dashed=False)
label_pair(9, 10, "ΔV", BLUE, dashed=True)

ax.text(cx, cy - R - 0.28, "16-electrode ring\nionic hydrogel skin",
        fontsize=11, color=MUTED, ha="center", va="top", zorder=7)

# pipeline: disc -> classifier -> class output, left to right
gap0, gap1 = 0.22, 0.22
arrow1_x0, arrow1_x1 = cx + R + gap0, 0.35
ax.annotate("", xy=(arrow1_x1, cy), xytext=(arrow1_x0, cy),
            arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.8))
ax.text((arrow1_x0 + arrow1_x1) / 2, cy + 0.24, "voltage vector", fontsize=10.5,
        color=MUTED, ha="center", va="bottom")

box_w, box_h = 1.55, 0.85
box_left = arrow1_x1 + gap1
ax.add_patch(plt.Rectangle((box_left, cy - box_h / 2), box_w, box_h, facecolor="white",
                            edgecolor=INK, lw=1.5, zorder=4))
ax.text(box_left + box_w / 2, cy, "Classifier\n(1D-CNN)", ha="center", va="center",
        fontsize=11.5, fontweight="bold", color=INK, zorder=5)

arrow2_x0, arrow2_x1 = box_left + box_w + gap1, box_left + box_w + gap1 + 0.38
ax.annotate("", xy=(arrow2_x1, cy), xytext=(arrow2_x0, cy),
            arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.8))

classes = ["No contact", "Light touch", "Firm press", "Point contact", "Distributed"]
chip_w, chip_h, gap = 1.65, 0.335, 0.08
chip_left = arrow2_x1 + 0.12
total_h = len(classes) * chip_h + (len(classes) - 1) * gap
y0 = cy + total_h / 2
for i, c in enumerate(classes):
    yy = y0 - i * (chip_h + gap)
    active = c == "Light touch"
    fc = BLUE if active else "white"
    tc = "white" if active else INK
    ax.add_patch(plt.Rectangle((chip_left, yy - chip_h), chip_w, chip_h,
                                facecolor=fc, edgecolor=BLUE if active else INK,
                                lw=1.2, zorder=4))
    ax.text(chip_left + chip_w / 2, yy - chip_h / 2, c, ha="center", va="center",
            fontsize=9.8, color=tc, fontweight="bold" if active else "normal", zorder=5)

x_lo = cx - R - 0.20
x_hi = chip_left + chip_w + 0.15
y_lo = min(cy - R - 0.75, cy - 1.55 - 0.35, y0 - total_h - 0.15)
y_hi = cy + R + 0.15
ax.set_xlim(x_lo, x_hi)
ax.set_ylim(y_lo, y_hi)

fig.tight_layout()
fig.savefig(OUT / "p_eit_diagram.png", dpi=200)
plt.close(fig)

# ---------------------------------------------------------- 6. confusion matrix
# Test-set confusion matrix for the noise-aware CNN (Noisy->Noisy, one
# representative seed). Raw counts transcribed from the dissertation's own
# Appendix figure (results/figures/raw/cnn1d/noisy_train_noisy_eval/
# cnn1d_noisy_train_noisy_eval_cm_test.png, Figure A: fig:app-cm-noisy),
# restyled to match the deck. 750 test samples per class; row-normalised to
# recall (%) for readability.
cm_classes = ["No contact", "Light touch", "Firm press", "Point contact", "Distributed"]
cm_counts = np.array([
    [750,   0,   0,   0,   0],
    [  0, 501,   0, 249,   0],
    [  0,   0, 630,  23,  97],
    [  0, 367,  23, 354,   6],
    [  0,   0, 145,  12, 593],
])
cm_pct = cm_counts / cm_counts.sum(axis=1, keepdims=True) * 100

from matplotlib.colors import LinearSegmentedColormap
blues = LinearSegmentedColormap.from_list("deck_blues", ["#ffffff", BLUE])

fig, ax = plt.subplots(figsize=(6.6, 5.6))
im = ax.imshow(cm_pct, cmap=blues, vmin=0, vmax=100, aspect="equal")
ax.set_xticks(range(5))
ax.set_yticks(range(5))
ax.set_xticklabels(cm_classes, fontsize=11.5, rotation=28, ha="right", color=MUTED)
ax.set_yticklabels(cm_classes, fontsize=11.5, color=MUTED)
ax.set_xlabel("Predicted class", fontsize=13, color=MUTED)
ax.set_ylabel("True class", fontsize=13, color=MUTED)
for spine in ax.spines.values():
    spine.set_visible(False)
ax.set_xticks(np.arange(-0.5, 5, 1), minor=True)
ax.set_yticks(np.arange(-0.5, 5, 1), minor=True)
ax.grid(which="minor", color=SURFACE, linewidth=2.5)
ax.tick_params(which="minor", length=0)
ax.tick_params(which="major", length=0)
for i in range(5):
    for j in range(5):
        v = cm_pct[i, j]
        if v < 0.5:
            continue
        txt_color = "white" if v > 55 else INK
        ax.text(j, i, f"{v:.0f}%", ha="center", va="center", fontsize=12.5,
                fontweight="bold" if i == j else "normal", color=txt_color)
ax.set_title("Per-class recall, noise-aware CNN", fontsize=17, fontweight="bold",
             color=INK, pad=14, loc="left")
fig.tight_layout()
fig.savefig(OUT / "p_confusion.png", dpi=200)
plt.close(fig)

print("wrote:", *[p.name for p in sorted(OUT.glob("*.png"))])

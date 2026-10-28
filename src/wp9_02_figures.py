"""Manuscript figures (direction-vs-magnitude framing). Static, print-ready PNGs.

  Figure 1  within-patient change: laboratory vs ECG-derived, three models
  Figure 2  published-style selections: group trajectories and rmcorr vs individual r
  Figure 3  the same change read across two patients vs within one patient

Palette: reference categorical slots 1-3 (validated all-pairs, light mode);
models also differ by marker shape so identity is never colour-alone.
"""
import json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from paths import INTERIM, OUT
from wp_stats import deming

FIG = OUT / "manuscript"; FIG.mkdir(exist_ok=True)
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
LAB = "#52514e"
MODELS = [("In-house ResNet", "potassium", "#2a78d6", "o"),
          ("ECGFounder", "ecgfounder", "#eb6834", "s"),
          ("Stockholm model 3", "stockholm_3", "#1baf7a", "^")]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
                     "savefig.dpi": 300, "savefig.facecolor": "white"})


def grid(ax, axis="both"):
    ax.grid(True, axis=axis, color=GRID, lw=0.6, zorder=0); ax.set_axisbelow(True)


def panel(ax, s):
    ax.text(-0.13, 1.04, s, transform=ax.transAxes, fontsize=10, fontweight="bold", color=INK)


cal = json.load(open(OUT / "stockholm_comparison.json"))
d = pd.read_parquet(INTERIM / "delta_scored_stockholm_z.parquet")
d["stockholm_3"] = d.stockholm_3 * cal["calibration"]["stockholm_3"]["slope_mEq_per_z"]

# ------------------------------------------------------------------ Figure 1
fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 3.6), gridspec_kw={"wspace": 0.3})
lim = (-4, 3)
hb = a.hexbin(d.dy_lab, d.potassium, gridsize=46, extent=(*lim, *lim), mincnt=1,
              cmap="Blues", bins="log", linewidths=0.2, edgecolors="white")
xs = np.array(lim)
a.plot(xs, xs, ls="--", color=MUTED, lw=1)
sl, ic = deming(d.dy_lab, d.potassium)
a.plot(xs, ic + sl * xs, color="#184f95", lw=2)
a.text(-3.85, -3.2, "Perfect tracking", ha="left", va="bottom", color=INK2, fontsize=7.5, rotation=45,
       rotation_mode="anchor")
a.text(-3.85, 2.45, f"Solid: Deming slope {sl:.2f}", ha="left", va="center", color="#184f95", fontsize=7.5)
a.set(xlim=lim, ylim=lim, aspect="equal", xlabel="Laboratory change in potassium (mEq/L)",
      ylabel="ECG-derived change (mEq/L)")
a.set_title("In-house ResNet, 9,154 pairs", fontsize=8, color=INK2, loc="left")
cax = a.inset_axes([0.58, 0.15, 0.36, 0.035])
cb = fig.colorbar(hb, cax=cax, orientation="horizontal"); cb.set_label("Pairs per cell", fontsize=6.5)
cb.ax.tick_params(labelsize=6.5); cb.outline.set_visible(False)
panel(a, "A")

edges = np.array([-4, -3, -2.5, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 3])
d["bin"] = pd.cut(d.dy_lab, edges)
b.plot(xs, xs, ls="--", color=MUTED, lw=1, zorder=1)
for i, (name, col, c, m) in enumerate(MODELS):
    g = d.groupby("bin", observed=True)
    mx, my = g.dy_lab.mean(), g[col].mean()
    se = g[col].std() / np.sqrt(g.size())
    off = (i - 1) * 0.06
    b.errorbar(mx + off, my, yerr=1.96 * se, fmt=m + "-", color=c, ms=4.5, lw=1.4,
               elinewidth=0.8, capsize=0, mec="white", mew=0.6, zorder=3, label=name)
b.text(-3.85, -3.2, "Perfect tracking", ha="left", va="bottom", color=INK2, fontsize=7.5, rotation=45,
       rotation_mode="anchor")
b.set(xlim=lim, ylim=lim, aspect="equal", xlabel="Laboratory change in potassium (mEq/L)",
      ylabel="Mean ECG-derived change (mEq/L)")
b.set_title("Binned means, three models", fontsize=8, color=INK2, loc="left")
b.legend(frameon=False, loc="upper left", fontsize=7, handlelength=1.2)
for ax in (a, b):
    grid(ax); ax.axhline(0, color=MUTED, lw=0.5); ax.axvline(0, color=MUTED, lw=0.5)
panel(b, "B")
fig.savefig(FIG / "fig1_within_patient_change.png", bbox_inches="tight")
plt.close(fig)

# ------------------------------------------------------------------ Figure 2
t = pd.read_csv(OUT / "table27_direction_magnitude.csv")
subsets = list(dict.fromkeys(t.subset))
npairs = [t[t.subset == x].n_pairs.iloc[0] for x in subsets]
labels = [f"All pairs\n(n={npairs[0]:,})", f"K ≤2.5\nbefore\n(n={npairs[1]})",
          f"K ≥6.5\nbefore\n(n={npairs[2]})", f"K ≥6.5 +\ndialysis\n(n={npairs[3]})"]
fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 3.3), gridspec_kw={"wspace": 0.38})
w = 0.19
for j, sub in enumerate(subsets):
    s = t[t.subset == sub]
    a.bar(j - 1.5 * w, s.mean_dK_lab.iloc[0], w, color=LAB, zorder=2, edgecolor="white", lw=0.8)
    for i, (name, _, c, m) in enumerate(MODELS):
        a.bar(j + (i - 0.5) * w, s[s.model == name].mean_dK_ecg.iloc[0], w, color=c, zorder=2,
              edgecolor="white", lw=0.8)
a.axhline(0, color=MUTED, lw=0.6)
a.set_xticks(range(4)); a.set_xticklabels(labels, fontsize=6.5)
a.set_ylabel("Mean change in potassium (mEq/L)"); a.set_ylim(-2.4, 1.2)
a.set_title("Group trajectories", fontsize=8, color=INK2, loc="left")
grid(a, "y"); panel(a, "A")

ypos = np.arange(4)[::-1]
for i, (name, _, c, m) in enumerate(MODELS):
    s = t[t.model == name].set_index("subset").loc[subsets]
    y = ypos + (1 - i) * 0.24
    for col, filled, dy in (("r_individual", False, 0.05), ("rmcorr", True, -0.05)):
        b.errorbar(s[col], y + dy, xerr=[s[col] - s[col + "_lo"], s[col + "_hi"] - s[col]], fmt=m,
                   mfc=c if filled else "white", mec=c if not filled else "white", mew=0.8 if not filled else 0.5,
                   color=c, ms=4.5, elinewidth=0.7, capsize=0, zorder=3)
b.axvline(0, color=MUTED, lw=0.6)
b.set_yticks(ypos); b.set_yticklabels(labels, fontsize=6.5)
b.set_xlim(-0.2, 0.85); b.set_xlabel("Correlation with laboratory change")
b.set_title("Open = individual r;  filled = rmcorr", fontsize=8, color=INK2, loc="left")
grid(b, "x"); panel(b, "B")
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
hand = [Patch(color=LAB, label="Laboratory")] + [Line2D([], [], color=c, marker=m, ls="", ms=5, label=n)
                                                  for n, _, c, m in MODELS]
fig.legend(handles=hand, ncol=4, frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, 1.06))
fig.savefig(FIG / "fig2_published_style_selections.png", bbox_inches="tight")
plt.close(fig)

# ------------------------------------------------------------------ Figure 3
ms = {r["model"]: r for r in cal["models"]}
fig, ax = plt.subplots(figsize=(4.6, 2.2))
for i, (name, key, c, m) in enumerate(MODELS):
    r = ms[key]; y = 2 - i
    for v, lo, hi, filled in ((r["delta_r"], r["delta_r_ci_low"], r["delta_r_ci_high"], False),
                              (r["matched_cross_r"], r["matched_cross_mc_low"], r["matched_cross_mc_high"], True)):
        ax.errorbar(v, y, xerr=[[v - lo], [hi - v]], fmt=m, mfc=c if filled else "white",
                    mec="white" if filled else c, mew=0.5 if filled else 0.9, color=c, ms=5.5,
                    elinewidth=0.8, capsize=0, zorder=3)
ax.set_yticks([2, 1, 0]); ax.set_yticklabels([m[0] for m in MODELS])
ax.set_ylim(-0.6, 2.9); ax.set_xlim(0.28, 0.55)
ax.text(0.345, 2.5, "Within one patient\n(open)", ha="center", fontsize=7, color=INK2)
ax.text(0.475, 2.5, "Across two patients, same\nchange distribution (filled)", ha="center", fontsize=7, color=INK2)
ax.set_xlabel("Correlation of ECG-derived with laboratory change")
grid(ax, "x")
fig.savefig(FIG / "fig3_within_vs_across.png", bbox_inches="tight")
plt.close(fig)
print("figures written to", FIG)

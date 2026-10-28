"""Figure 5 - the delta-fusion hypothesis test."""
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from paths import OUT

plt.rcParams.update({"figure.dpi": 150, "font.size": 8.5, "axes.spines.top": False,
                     "axes.spines.right": False})
C_M, C_P, C_C, C_N = "#2b6cb0", "#dd6b20", "#a0aec0", "#c53030"

dh = pd.read_csv(OUT / "table14_delta_task_heads.csv")
ab = pd.read_csv(OUT / "table15_stack_ablation.csv")
dt = pd.read_csv(OUT / "table19_fusion_by_dt.csv")

fig = plt.figure(figsize=(13.4, 4.6))
gs = fig.add_gridspec(1, 3, width_ratios=[1.18, 1.05, 1.0], wspace=0.40)

# --- A: the delta-model ladder, and the regression-to-the-mean trap --------
ax = fig.add_subplot(gs[0])
sd = pd.read_csv(OUT / "table24_siamese_delta.csv")
key = {"implied Δ  f_post − f_pre": "implied Δ\nf_post−f_pre",
       "exp1  ridge on frozen embeddings": "exp1\nfrozen emb",
       "exp2a siamese head, frozen trunk": "exp2a\nsiamese\n(frozen)",
       "exp2b siamese, trunk fine-tuned": "exp2b\nsiamese\n(unfrozen)"}
sd = sd.set_index("delta_predictor").loc[list(key)]
lab = [key[k] for k in sd.index] + ["Δ head\n+K_pre", "NULL\nK_pre only"]
vals = list(sd.r_ensemble) + [float(dh.loc[dh.delta_predictor.str.contains(r"K_pre\)"), "r"].iloc[0]),
                              float(dh.loc[dh.delta_predictor.str.contains("NULL"), "r"].iloc[0])]
errlo = list(sd.r_ensemble - sd.ci_lo) + [0, 0]
errhi = list(sd.ci_hi - sd.r_ensemble) + [0, 0]
col = [C_M, "#7fb3d9", "#4c9bd6", "#1a4f80", C_P, C_N]
ax.bar(range(6), vals, 0.66, color=col, yerr=[errlo, errhi], capsize=2.5,
       error_kw={"lw": 0.9})
for i, v in enumerate(vals):
    ax.text(i, v + 0.045, f"{v:.3f}", ha="center", fontsize=7.0)
ax.axvline(3.5, color="#a0aec0", lw=0.9, ls=":")
ax.text(1.5, 0.80, "ECG-only Δ models", ha="center", fontsize=7.0, color="#1a202c")
ax.text(4.5, 0.80, "conditioned on K_pre", ha="center", fontsize=7.0, color="#c53030")
ax.annotate("", xy=(5, 0.667), xytext=(4, 0.692),
            arrowprops=dict(arrowstyle="<->", color="#1a202c", lw=1.0))
ax.text(4.5, 0.755, "gap 0.025", ha="center", fontsize=6.6)
ax.axhline(0.50, color="#2f855a", lw=1.0, ls="--")
ax.text(0.0, 0.515, "H1 threshold 0.50", fontsize=6.6, color="#2f855a")
ax.set_xticks(range(6)); ax.set_xticklabels(lab, fontsize=6.0)
ax.set_ylabel("r with laboratory ΔK"); ax.set_ylim(0, 0.88)
ax.set_title("A  A siamese read-out beats differencing, but stays\n"
             "     below H1; a K_pre-conditioned head is regression to the mean",
             fontsize=8.6, loc="left")

# --- B: the ladder isolating what the second ECG contributes ---------------
ax = fig.add_subplot(gs[1])
lad = pd.read_csv(OUT / "table22_contrast_ladder.csv")
order = ["A anchor only", "B anchor + current ECG", "C anchor + both ECGs",
         "D + delta head", "E ECG only"]
short = ["A\nanchor\nonly", "B\n+current\nECG", "C\n+second\nECG", "D\n+Δ head", "E\nECG\nonly"]
piv = lad.pivot(index="model", columns="source", values="MAE").loc[order]
x = np.arange(len(order)); w = 0.38
ax.bar(x - w / 2, piv["inhouse"], w, color=C_M, label="in-house ResNet")
ax.bar(x + w / 2, piv["stockholm"], w, color="#68a4d8", label="published checkpoint")
for i, v in enumerate(piv["inhouse"]):
    ax.text(i - w / 2, v + 0.006, f"{v:.3f}", ha="center", fontsize=6.4)
ax.set_xticks(x); ax.set_xticklabels(short, fontsize=6.8)
ax.set_ylabel("MAE for K_post (mEq/L)"); ax.set_ylim(0, 0.60)
ax.legend(fontsize=6.6, frameon=False, loc="upper right")
# annotate the decisive steps
ax.annotate("", xy=(1, 0.505), xytext=(0, 0.505),
            arrowprops=dict(arrowstyle="<->", color="#2f855a", lw=1.3))
ax.text(0.5, 0.515, "−0.068\nthe whole gain", ha="center", fontsize=6.6, color="#2f855a")
ax.annotate("", xy=(3, 0.455), xytext=(1, 0.455),
            arrowprops=dict(arrowstyle="<->", color="#c53030", lw=1.1))
ax.text(2.0, 0.463, "−0.005  n.s.", ha="center", fontsize=6.6, color="#c53030")
ax.set_title("B  All of the gain is the anchor + the CURRENT ECG;\n"
             "     the second ECG adds ~nothing", fontsize=8.8, loc="left")

# --- C: when is the ECG worth adding? --------------------------------------
ax = fig.add_subplot(gs[2])
d = dt[dt.dt_bin != "all"]
x = np.arange(len(d))
ax.plot(x, d.MAE_anchor, "o-", color=C_P, ms=5, label="anchor only (K_pre)")
ax.plot(x, d.MAE_ecg, "s-", color=C_M, ms=5, label="ECG only")
ax.plot(x, d.MAE_fused, "^-", color="#2f855a", ms=6, lw=2, label="anchor + ECG")
ax.set_xticks(x); ax.set_xticklabels(d.dt_bin, fontsize=7.6)
ax.set_xlabel("interval between the two ECGs"); ax.set_ylabel("MAE for K_post (mEq/L)")
ax.legend(fontsize=7, frameon=False)
ax.set_title("C  The anchor decays with time,\n     the ECG does not", fontsize=8.8, loc="left")

fig.savefig(OUT / "fig6_delta_fusion.svg", bbox_inches="tight")
print("wrote fig6_delta_fusion.svg")

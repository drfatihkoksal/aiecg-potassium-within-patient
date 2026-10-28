"""Manuscript figures, in narrative order."""
import json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy import stats
from paths import INTERIM, OUT
from wp_stats import deming

plt.rcParams.update({"figure.dpi": 150, "font.size": 8.5, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.labelsize": 8.5})
NICE = {"potassium": "Potassium\n(mEq/L)", "creatinine": "Creatinine\n(mg/dL)",
        "hemoglobin": "Haemoglobin\n(g/dL)", "calcium_total": "Calcium, total\n(mg/dL)",
        "calcium_corrected": "Calcium, corrected\n(mg/dL)"}
C_MODEL, C_PAT, C_CONST = "#2b6cb0", "#dd6b20", "#a0aec0"

# ============ FIG 1 - the metric is broken =================================
b = pd.read_csv(OUT / "table2b_baselines_same_testset.csv")
fig = plt.figure(figsize=(12.2, 4.0))
gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 1.05], wspace=0.40)

axA = fig.add_subplot(gs[0])
k = b[b.analyte == "potassium"].iloc[0]
# published MAE values (see table10_published_comparators.md for provenance)
PUB = [(0.262, "von Bachmann 2024"), (0.285, ""), (0.364, "Lin 2022"),
       (0.450, "Moon 2022"), (0.490, "MIMIC-IV published"), (0.527, "Kardio-Net 12-lead"),
       (0.599, "Kardio-Net external"), (0.740, "Kardio-Net ext. 2")]
axA.add_patch(Rectangle((-0.6, 0.262), 4.2, 0.740 - 0.262, color="#c53030",
                        alpha=0.09, zorder=0))
for v, lab in PUB:
    axA.plot([-0.55, 3.55], [v, v], color="#c53030", lw=0.7, ls=":", zorder=1)
    if lab:
        axA.text(3.58, v, f" {lab}", va="center", fontsize=5.9, color="#c53030")
axA.text(-0.5, 0.755, "range of published MAEs, 0.262–0.740", ha="left", va="bottom",
         fontsize=7.0, color="#c53030")
ef = b[b.analyte == "ecgfounder"].iloc[0]
vals = [k.model_MAE_on_history_subset, ef.model_MAE_on_history_subset,
        k.patient_mean_MAE, k.constant_MAE_on_history_subset]
bars = axA.bar(["in-house\nResNet", "ECGFounder\nfine-tuned",
                "patient's\nhistorical\nmean", "a single\nconstant"],
               vals, color=[C_MODEL, "#4c9bd6", C_PAT, C_CONST], width=0.66, zorder=3)
for r, v in zip(bars, vals):
    axA.text(r.get_x() + r.get_width() / 2, v + 0.012, f"{v:.3f}", ha="center", fontsize=7.6)
axA.set_ylabel("Test-set MAE (mEq/L)"); axA.set_ylim(0, 0.80)
axA.set_title("A  Potassium: both models lose to the patient's own mean;\n     a constant lands inside the published MAE range", fontsize=9, loc="left")

axB = fig.add_subplot(gs[1])
bb = b[b.analyte.isin(["potassium", "creatinine", "hemoglobin", "calcium_total"])]
bb = bb.set_index("analyte").loc[["potassium", "creatinine", "hemoglobin", "calcium_total"]].reset_index()
x = np.arange(len(bb)); w = 0.27
for i, (col, c, lab) in enumerate([("model_MAE_on_history_subset", C_MODEL, "AI-ECG model"),
                                   ("patient_mean_MAE", C_PAT, "patient's historical mean"),
                                   ("constant_MAE_on_history_subset", C_CONST, "single constant")]):
    axB.bar(x + (i - 1) * w, bb[col] / bb.constant_MAE_on_history_subset, w, color=c, label=lab)
axB.axhline(1.0, color="#4a5568", lw=0.9, ls="--")
for i, r in enumerate(bb.itertuples()):
    axB.text(i - w, r.model_MAE_on_history_subset / r.constant_MAE_on_history_subset + 0.02,
             f"{r.model_MAE_on_history_subset/r.constant_MAE_on_history_subset:.2f}",
             ha="center", fontsize=7)
    axB.text(i, r.patient_mean_MAE / r.constant_MAE_on_history_subset - 0.06,
             f"{r.patient_mean_MAE/r.constant_MAE_on_history_subset:.2f}",
             ha="center", fontsize=7, color="white", weight="bold")
axB.set_xticks(x); axB.set_xticklabels([NICE[a].split("\n")[0] for a in bb.analyte], fontsize=7.8)
axB.set_ylabel("MAE relative to a single constant"); axB.set_ylim(0, 1.18)
axB.legend(fontsize=7.4, frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.30))
axB.set_title("B  A patient's own historical mean beats the network,\n     for every analyte", fontsize=9, loc="left")
fig.subplots_adjust(bottom=0.28, top=0.84, left=0.075, right=0.985)
fig.savefig(OUT / "fig1_metric_is_broken.svg", bbox_inches="tight"); plt.close(fig)

# ============ FIG 2 - where the accuracy comes from ========================
J = {a: json.load(open(OUT / f"wp3_5_{a}.json"))
     for a in NICE if (OUT / f"wp3_5_{a}.json").exists()}
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.2, 3.3), gridspec_kw={"wspace": 0.3})
labs = [a for a in NICE if a in J and a != "calcium_corrected"]
x = np.arange(len(labs))
rb = [J[a]["wp3"]["R2_between"] for a in labs]; rw = [J[a]["wp3"]["R2_within"] for a in labs]
ax1.bar(x, rb, 0.6, label="between-subject R²", color=C_MODEL)
ax1.bar(x, rw, 0.6, bottom=rb, label="within-subject R²", color="#dd6b20")
for i, a in enumerate(labs):
    ax1.text(i, rb[i] + rw[i] + 0.012,
             f"within\n{J[a]['wp3']['frac_explained_within']:.1%}", ha="center", fontsize=7)
ax1.set_xticks(x); ax1.set_xticklabels([NICE[a].split("\n")[0] for a in labs], fontsize=7.6)
ax1.set_ylabel("explained variance (R²)"); ax1.set_ylim(0, 0.72)
ax1.legend(fontsize=7.2, frameon=False)
ax1.set_title("A  Accuracy is between-subject", fontsize=8.8, loc="left")

ip = [J[a]["wp3"]["icc_pred"]["icc"] for a in labs]
il = [J[a]["wp3"]["icc_lab"]["icc"] for a in labs]
ax2.plot([0, 1], [0, 1], color="#a0aec0", lw=0.9, ls="--")
ax2.scatter(il, ip, s=52, color=C_MODEL, zorder=3)
for a, xx, yy in zip(labs, il, ip):
    ax2.annotate(NICE[a].split("\n")[0], (xx, yy), textcoords="offset points",
                 xytext=(7, -3), fontsize=7.2)
ax2.set_xlabel("ICC of the measured analyte"); ax2.set_ylabel("ICC of the model's prediction")
ax2.set_xlim(0.3, 1.0); ax2.set_ylim(0.3, 1.0)
ax2.set_title("B  Above the line = the network imposes more\n     subject structure than physiology has",
              fontsize=8.8, loc="left")
fig.savefig(OUT / "fig2_between_vs_within.svg", bbox_inches="tight"); plt.close(fig)

# ============ FIG 3 - performance on the task that matters =================
d = pd.read_parquet(INTERIM / "delta_scored_potassium.parquet")
mg = pd.read_csv(OUT / "table4b_delta_by_magnitude.csv")
mg = mg[mg.analyte == "potassium"]
dt = pd.read_csv(OUT / "table7_delta_by_dt.csv"); dt = dt[dt.analyte == "potassium"]

fig = plt.figure(figsize=(9.4, 3.4))
gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 1, 1], wspace=0.34)
ax = fig.add_subplot(gs[0])
ax.scatter(d.dy_lab, d.dy_ecg, s=3, alpha=0.10, color=C_MODEL, edgecolors="none")
sl, ic = deming(d.dy_lab.values, d.dy_ecg.values)
xs = np.linspace(-4, 4, 10)
ax.plot(xs, xs, color="#4a5568", ls="--", lw=1.0, label="identity")
ax.plot(xs, sl * xs + ic, color="#c53030", lw=1.5, label=f"Deming, slope {sl:.2f}")
ax.set_xlim(-4, 4); ax.set_ylim(-2.2, 2.2)
ax.set_xlabel("laboratory ΔK (mEq/L)"); ax.set_ylabel("ECG-derived ΔK (mEq/L)")
ax.legend(fontsize=7, frameon=False, loc="upper left")
r = stats.pearsonr(d.dy_lab, d.dy_ecg)[0]
ax.set_title(f"A  Within-subject Δ tracking\n     r = {r:.3f}, n = {len(d):,} pairs",
             fontsize=8.8, loc="left")

ax = fig.add_subplot(gs[1])
ax.bar(range(len(mg)), mg.pearson, 0.62, color=C_MODEL)
ax.set_xticks(range(len(mg)))
ax.set_xticklabels([s.replace("|dY| ", "").replace("-99", "+") for s in mg.stratum],
                   fontsize=7.2, rotation=15)
for i, (v, n) in enumerate(zip(mg.pearson, mg.n)):
    ax.text(i, v + 0.008, f"{v:.2f}\nn={n:,}", ha="center", fontsize=6.8)
ax.set_xlabel("|ΔK| stratum (mEq/L)"); ax.set_ylabel("Pearson r"); ax.set_ylim(0, 0.50)
ax.set_title("B  Tracking appears only for\n     large shifts", fontsize=8.8, loc="left")

ax = fig.add_subplot(gs[2])
dd = dt.dropna(subset=["pearson"])
ax.errorbar(range(len(dd)), dd.pearson,
            yerr=[dd.pearson - dd.ci_lo, dd.ci_hi - dd.pearson],
            fmt="o-", color=C_MODEL, capsize=3, ms=5, label="all pairs")
dm = dt.dropna(subset=["pearson_matched"])
ax.errorbar(range(len(dm)), dm.pearson_matched,
            yerr=[dm.pearson_matched - dm.matched_ci_lo, dm.matched_ci_hi - dm.pearson_matched],
            fmt="s--", color=C_PAT, capsize=3, ms=4.5, label="|ΔK| ≥ 0.5 only")
ax.set_xticks(range(len(dd))); ax.set_xticklabels(dd.dt_bin, fontsize=7.4)
ax.set_xlabel("interval between the two ECGs"); ax.set_ylabel("Pearson r"); ax.set_ylim(0, 0.58)
ax.legend(fontsize=7, frameon=False, loc="upper left")
ax.set_title("C  Shorter windows track worse,\n     not better", fontsize=8.8, loc="left")
fig.savefig(OUT / "fig3_delta_tracking.svg", bbox_inches="tight"); plt.close(fig)

# ============ FIG 4 - negative-control panel (heat map) ====================
mods = [a for a in ["potassium", "creatinine", "hemoglobin", "calcium_total"]
        if (OUT / f"wp6_{a}.json").exists()]
labs_o = ["potassium", "creatinine", "hemoglobin", "calcium_total"]
M = np.full((len(mods), len(labs_o)), np.nan)
N = np.zeros_like(M)
for i, a in enumerate(mods):
    ca = json.load(open(OUT / f"wp6_{a}.json"))["delta_ecg_vs_other_analyte_deltas"]
    for j, o in enumerate(labs_o):
        if o in ca:
            M[i, j] = ca[o]["pearson"]; N[i, j] = ca[o]["n"]
fig, ax = plt.subplots(figsize=(5.4, 4.0))
im = ax.imshow(M, cmap="RdBu_r", vmin=-0.4, vmax=0.4)
ax.set_xticks(range(len(labs_o)))
ax.set_xticklabels([NICE[o].split("\n")[0] for o in labs_o], rotation=22, ha="right", fontsize=7.8)
ax.set_yticks(range(len(mods)))
ax.set_yticklabels([NICE[a].split("\n")[0] for a in mods], fontsize=7.8)
for i in range(len(mods)):
    for j in range(len(labs_o)):
        if np.isfinite(M[i, j]):
            ax.text(j, i, f"{M[i,j]:.3f}", ha="center", va="center", fontsize=8.2,
                    color="white" if abs(M[i, j]) > 0.25 else "#1a202c",
                    weight="bold" if i == j else "normal")
        if i == j:
            ax.add_patch(Rectangle((j - .5, i - .5), 1, 1, fill=False, ec="#1a202c", lw=2))
ax.set_xlabel("laboratory Δ of analyte"); ax.set_ylabel("model producing the ECG-derived Δ")
ax.set_title("Negative-control panel: does each model's ΔECG\ntrack its own analyte? (H5)",
             fontsize=9, loc="left")
fig.colorbar(im, ax=ax, shrink=0.8, label="Pearson r")
fig.tight_layout(); fig.savefig(OUT / "fig4_negative_control_panel.svg"); plt.close(fig)
print("wrote fig1_metric_is_broken, fig2_between_vs_within, fig3_delta_tracking, "
      "fig4_negative_control_panel")

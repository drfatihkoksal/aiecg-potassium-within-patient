"""Assemble the Section 9 output tables and figures from the per-WP JSON."""
import json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from paths import INTERIM, OUT

ANALYTES = ["potassium", "creatinine", "hemoglobin", "calcium_total"]
have = [a for a in ANALYTES if (OUT / f"wp3_5_{a}.json").exists()]
J = {a: json.load(open(OUT / f"wp3_5_{a}.json")) for a in have}
A6 = {a: json.load(open(OUT / f"wp6_{a}.json")) for a in have if (OUT / f"wp6_{a}.json").exists()}
MS = {a: json.load(open(OUT / f"wp6_matched_shuffle_{a}.json")) for a in have
      if (OUT / f"wp6_matched_shuffle_{a}.json").exists()}

def ci(t):
    return f"{t[0]:.3f} ({t[1]:.3f} to {t[2]:.3f})"

# ---- table2: cross-sectional performance ---------------------------------
rows = []
for a in have:
    t = json.load(open(OUT / f"wp2_test_{a}.json"))
    sp = pd.read_parquet(INTERIM / f"split_{a}.parquet")
    tr, te = sp[sp.split == "train"].y, sp[sp.split == "test"].y
    rows.append({"analyte": a, "n_test": t["n"], "MAE": round(t["mae"], 4),
                 "RMSE": round(t["rmse"], 4), "r": round(t["r"], 4),
                 "AUROC_high": round(t.get("auroc_hyper", np.nan), 3),
                 "AUROC_low": round(t.get("auroc_hypo", np.nan), 3),
                 "MAE_mean_predictor": round(float(np.abs(te - tr.mean()).mean()), 4),
                 "test_SD": round(float(te.std()), 4)})
pd.DataFrame(rows).to_csv(OUT / "table2_crosssectional_performance.csv", index=False)

# ---- table3: variance decomposition --------------------------------------
rows = []
for a, j in J.items():
    w = j["wp3"]
    rows.append({"analyte": a, "n_obs": w["n_obs"], "n_patients": w["n_patients"],
                 "ICC_pred": round(w["icc_pred"]["icc"], 3),
                 "ICC_lab": round(w["icc_lab"]["icc"], 3),
                 "r_between": round(w["r_between"], 3), "r_within": round(w["r_within"], 3),
                 "R2_between": round(w["R2_between"], 3), "R2_within": round(w["R2_within"], 3),
                 "within_share_of_explained_variance": round(w["frac_explained_within"], 3)})
pd.DataFrame(rows).to_csv(OUT / "table3_variance_decomposition.csv", index=False)

# ---- table4: delta tracking ----------------------------------------------
rows = []
for a, j in J.items():
    w = j["wp4"]
    rows.append({"analyte": a, "n_pairs": w["n_pairs"], "n_patients": w["n_patients"],
                 "pearson_r": ci(w["overall"]["pearson"]),
                 "spearman": ci(w["overall"]["spearman"]),
                 "deming_slope": ci(w["overall"]["deming_slope"]),
                 "r_attenuation_corrected": round(w["r_attenuation_corrected"], 3),
                 "delta_reliability": round(w["delta_reliability"], 3),
                 "BA_bias": round(w["bland_altman"]["bias"], 3),
                 "BA_LoA_lower": round(w["bland_altman"]["loa_lower"], 3),
                 "BA_LoA_upper": round(w["bland_altman"]["loa_upper"], 3)})
pd.DataFrame(rows).to_csv(OUT / "table4_delta_tracking.csv", index=False)
pd.concat([pd.DataFrame(j["wp4"]["by_magnitude"]).assign(analyte=a) for a, j in J.items()]
          ).to_csv(OUT / "table4b_delta_by_magnitude.csv", index=False)

# ---- table5: MDC ----------------------------------------------------------
pd.DataFrame([{"analyte": a, **j["wp5"]} for a, j in J.items()]).to_csv(
    OUT / "table5_mdc.csv", index=False)

# ---- table6: ablations / falsification -----------------------------------
rows = []
for a, r in A6.items():
    row = {"analyte": a, "n_pairs": r["n_pairs"],
           "delta_model_MAE": round(r["delta_task"]["model"]["mae"], 4),
           "delta_model_r": round(r["delta_task"]["model"]["pearson"], 3),
           "delta_trivial_MAE": round(r["delta_task"]["patient_mean_baseline"]["mae"], 4),
           "abs_model_MAE": round(r["absolute_task_post_ecg"]["model"]["mae"], 4),
           "abs_patientmean_MAE": round(r["absolute_task_post_ecg"]["patient_mean"]["mae"], 4),
           "abs_popmean_MAE": round(r["absolute_task_post_ecg"]["population_mean"]["mae"], 4),
           "cross_subject_shuffle_r": round(r["cross_subject_shuffle"]["pearson"], 3),
           "hd_r": round(r["hd_vs_nonhd"]["hd"].get("pearson", np.nan), 3),
           "nonhd_r": round(r["hd_vs_nonhd"]["non_hd"].get("pearson", np.nan), 3)}
    if a in MS:
        row["shuffle_dist_matched_r"] = round(MS[a]["cross_subject_distribution_matched_r"][0], 3)
    for o, v in r["delta_ecg_vs_other_analyte_deltas"].items():
        row[f"dECG_vs_d{o}_r"] = round(v["pearson"], 3)
    rows.append(row)
pd.DataFrame(rows).to_csv(OUT / "table6_ablations_falsification.csv", index=False)

# ================================ figures ==================================
plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False})

# fig2: Bland-Altman on the delta scale
d = pd.read_parquet(INTERIM / "delta_scored_potassium.parquet")
diff = d.dy_ecg - d.dy_lab; mean = (d.dy_ecg + d.dy_lab) / 2
fig, ax = plt.subplots(figsize=(5.2, 4))
ax.scatter(mean, diff, s=3, alpha=0.12, color="#2b6cb0", edgecolors="none")
b, s = diff.mean(), diff.std(ddof=1)
for v, ls, lab in ((b, "-", f"bias {b:+.2f}"), (b + 1.96 * s, "--", f"+1.96 SD {b+1.96*s:+.2f}"),
                   (b - 1.96 * s, "--", f"-1.96 SD {b-1.96*s:+.2f}")):
    ax.axhline(v, ls=ls, color="#c53030", lw=1.2)
    ax.text(ax.get_xlim()[1], v, f" {lab}", va="center", fontsize=7.5, color="#c53030")
ax.axhspan(-0.5, 0.5, color="#38a169", alpha=0.10)
ax.set_xlabel("mean of ECG-derived and laboratory ΔK (mEq/L)")
ax.set_ylabel("ECG-derived ΔK − laboratory ΔK (mEq/L)")
ax.set_title(f"Bland–Altman on the Δ scale, n={len(d):,} pairs\n"
             "green band = ±0.5 mEq/L clinical action threshold", fontsize=9)
fig.tight_layout(); fig.savefig(OUT / "fig2_bland_altman_delta.svg"); plt.close(fig)

# fig3: between vs within
fig, ax = plt.subplots(figsize=(5.4, 3.6))
labs = list(J); x = np.arange(len(labs)); w = 0.38
ax.bar(x - w / 2, [J[a]["wp3"]["R2_between"] for a in labs], w, label="between-subject R²", color="#2b6cb0")
ax.bar(x + w / 2, [J[a]["wp3"]["R2_within"] for a in labs], w, label="within-subject R²", color="#dd6b20")
for i, a in enumerate(labs):
    ax.text(i, 0.02, f"within share\n{J[a]['wp3']['frac_explained_within']:.1%}",
            ha="center", va="bottom", fontsize=7)
ax.set_xticks(x); ax.set_xticklabels(labs, rotation=12); ax.set_ylabel("R²")
ax.set_title("Explained variance: between- vs within-subject", fontsize=9); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "fig3_between_vs_within_r2.svg"); plt.close(fig)

# fig4: negative-control analytes
if "potassium" in A6:
    ca = A6["potassium"]["delta_ecg_vs_other_analyte_deltas"]
    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    ks = list(ca); v = [ca[k]["pearson"] for k in ks]
    ax.bar(ks, v, color=["#2b6cb0" if k == "potassium" else "#a0aec0" for k in ks])
    for i, (k, y) in enumerate(zip(ks, v)):
        ax.text(i, y + 0.008, f"{y:.3f}\nn={ca[k]['n']:,}", ha="center", fontsize=7)
    ax.set_ylabel("Pearson r"); ax.axhline(0, color="k", lw=0.8)
    ax.set_title("ECG-derived ΔK vs laboratory Δ of each analyte\n(H5 negative control)", fontsize=9)
    fig.tight_layout(); fig.savefig(OUT / "fig4_negative_control_analytes.svg"); plt.close(fig)

print("wrote tables 2-6 and figs 2-4 to outputs/")

"""WP3 (variance decomposition), WP4 (delta tracking), WP5 (MDC) for one analyte.

  python wp3_02_analysis.py potassium
"""
import argparse, json
import numpy as np, pandas as pd
from scipy import stats
from paths import INTERIM, OUT
from wp_stats import (cluster_bootstrap, deming, icc_oneway,
                      attenuation_correct, bland_altman, mdc95)

# analytical CV of the assay, EFLM biological-variation database
CV_ANALYTICAL = {"potassium": 0.020, "creatinine": 0.027,
                 "hemoglobin": 0.013, "calcium_total": 0.021,
                 "calcium_corrected": 0.021, "calcium_matched": 0.021,
                 "ecgfounder": 0.020, "ecgfounder_linprob": 0.020}
ACTION = {"potassium": 0.5, "creatinine": 0.3, "hemoglobin": 1.0, "calcium_total": 0.5, "calcium_corrected": 0.5, "calcium_matched": 0.5,
          "ecgfounder": 0.5, "ecgfounder_linprob": 0.5}

ap = argparse.ArgumentParser(); ap.add_argument("analyte"); ap.add_argument("--boot", type=int, default=2000)
A = ap.parse_args(); a = A.analyte
pred = pd.read_parquet(INTERIM / f"pred_{a}.parquet")[["study_id", f"pred_{a}"]]
pred.columns = ["study_id", "yhat"]

# ============================ WP3 - variance decomposition ==================
obs = pd.read_parquet(INTERIM / f"audit_{a}.parquet")
audit_pat = set(obs.subject_id)
xs = pd.read_parquet(INTERIM / f"train_{a}.parquet")  # training-pool patients (disjoint)
allobs = pd.concat([
    obs[["subject_id", "pre_study", "y_pre"]].rename(columns={"pre_study": "study_id", "y_pre": "y"}),
    obs[["subject_id", "post_study", "y_post"]].rename(columns={"post_study": "study_id", "y_post": "y"}),
]).drop_duplicates("study_id").merge(pred, on="study_id")
rep = allobs.groupby("subject_id").filter(lambda g: len(g) >= 2)

wp3 = {"n_obs": len(rep), "n_patients": rep.subject_id.nunique(),
       "icc_pred":  icc_oneway(rep.yhat.values, rep.subject_id.values),
       "icc_lab":   icc_oneway(rep.y.values,    rep.subject_id.values)}

# Snijders-Bosker style between/within split of explained variance
g = rep.groupby("subject_id")
rep = rep.assign(y_bar=g.y.transform("mean"), yhat_bar=g.yhat.transform("mean"))
rep = rep.assign(y_w=rep.y - rep.y_bar, yhat_w=rep.yhat - rep.yhat_bar)
pm = rep.groupby("subject_id")[["y_bar", "yhat_bar"]].first()
r_between = float(np.corrcoef(pm.y_bar, pm.yhat_bar)[0, 1])
r_within = float(np.corrcoef(rep.y_w, rep.yhat_w)[0, 1])
var_b_lab = float(pm.y_bar.var(ddof=1)); var_w_lab = float(rep.y_w.var(ddof=1))
expl_b = r_between ** 2 * var_b_lab
expl_w = r_within ** 2 * var_w_lab
wp3.update({"r_between": r_between, "r_within": r_within,
            "R2_between": r_between ** 2, "R2_within": r_within ** 2,
            "var_between_lab": var_b_lab, "var_within_lab": var_w_lab,
            "explained_between": expl_b, "explained_within": expl_w,
            "frac_explained_within": expl_w / (expl_b + expl_w)})
print(f"WP3 {a}: ICC_pred={wp3['icc_pred']['icc']:.3f}  ICC_lab={wp3['icc_lab']['icc']:.3f}  "
      f"r_between={r_between:.3f} r_within={r_within:.3f}  "
      f"within share of explained variance={wp3['frac_explained_within']:.3f}")

# ================================ WP4 - delta tracking ======================
d = obs.merge(pred.rename(columns={"study_id": "pre_study", "yhat": "yhat_pre"}), on="pre_study")
d = d.merge(pred.rename(columns={"study_id": "post_study", "yhat": "yhat_post"}), on="post_study")
d["dy_ecg"] = d.yhat_post - d.yhat_pre
d = d[np.isfinite(d.dy_ecg) & np.isfinite(d.dy_lab)]

def metrics(df):
    return {"pearson":  float(stats.pearsonr(df.dy_lab, df.dy_ecg)[0]),
            "spearman": float(stats.spearmanr(df.dy_lab, df.dy_ecg)[0]),
            "deming_slope": deming(df.dy_lab.values, df.dy_ecg.values)[0],
            "mae": float(np.abs(df.dy_ecg - df.dy_lab).mean())}

wp4 = {"n_pairs": len(d), "n_patients": int(d.subject_id.nunique())}
wp4["overall"] = cluster_bootstrap(d, metrics, n_boot=A.boot)
wp4["bland_altman"] = bland_altman(d.dy_ecg, d.dy_lab)
r_raw = wp4["overall"]["pearson"][0]
r_corr, rel = attenuation_correct(r_raw, CV_ANALYTICAL[a], d.dy_lab.std(ddof=1),
                                  float(np.mean([d.y_pre.mean(), d.y_post.mean()])))
wp4.update({"r_attenuation_corrected": r_corr, "delta_reliability": rel})

strata = []
for lo, hi in [(0, 0.25), (0.25, 0.5), (0.5, 1.0), (1.0, 99)]:
    s = d[(d.dy_lab.abs() >= lo) & (d.dy_lab.abs() < hi)]
    if len(s) >= 50:
        strata.append({"stratum": f"|dY| {lo}-{hi}", "n": len(s),
                       "n_pat": int(s.subject_id.nunique()), **metrics(s)})
wp4["by_magnitude"] = strata
print(f"WP4 {a}: n={len(d):,} pairs  r={r_raw:.3f} "
      f"[{wp4['overall']['pearson'][1]:.3f},{wp4['overall']['pearson'][2]:.3f}]  "
      f"Deming slope={wp4['overall']['deming_slope'][0]:.3f}  r_corrected={r_corr:.3f}")

# ==================================== WP5 - MDC =============================
nl = pd.read_parquet(INTERIM / f"null_{a}.parquet")
nl = nl.merge(pred.rename(columns={"study_id": "pre_study", "yhat": "yhat_pre"}), on="pre_study")
nl = nl.merge(pred.rename(columns={"study_id": "post_study", "yhat": "yhat_post"}), on="post_study")
nl["dy_ecg"] = nl.yhat_post - nl.yhat_pre
sd_w = float(nl.dy_ecg.std(ddof=1) / np.sqrt(2))     # SD of a single measurement
cv = CV_ANALYTICAL[a]; lvl = float(np.mean([nl.y_pre.mean(), nl.y_post.mean()]))
wp5 = {"n_pairs": len(nl), "n_patients": int(nl.subject_id.nunique()),
       "sd_delta_ecg": float(nl.dy_ecg.std(ddof=1)), "sd_within": sd_w,
       "mdc95": mdc95(sd_w),
       "lab_rcv_analytical_only": float(1.96 * np.sqrt(2) * cv * lvl),
       "clinical_action_threshold": ACTION[a],
       "mean_abs_delta_hd": float(d[(d.hd_icu | d.hd_icd) & ~d.crrt].dy_lab.abs().mean())
                            if ((d.hd_icu | d.hd_icd) & ~d.crrt).sum() else None}
print(f"WP5 {a}: MDC95={wp5['mdc95']:.3f}  vs action threshold {ACTION[a]}  "
      f"(n={len(nl):,} null-change pairs)")

out = {"analyte": a, "wp3": wp3, "wp4": wp4, "wp5": wp5}
(OUT / f"wp3_5_{a}.json").write_text(json.dumps(out, indent=2, default=float))
d.to_parquet(INTERIM / f"delta_scored_{a}.parquet", index=False)
nl.to_parquet(INTERIM / f"null_scored_{a}.parquet", index=False)
print(f"wrote outputs/wp3_5_{a}.json")

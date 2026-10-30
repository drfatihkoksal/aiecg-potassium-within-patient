"""Delta-t stratification of delta tracking (robustness to the time between ECGs).

How much of the weak within-subject tracking is the model, and how much is the
<=24 h window letting other things change? Two readings are needed, because dt
and |dY| are confounded (longer gaps carry larger shifts):

  raw      - r within each dt bin, as observed
  matched  - r within each dt bin restricted to |dY_lab| >= 0.5, so the bins are
             compared at a common shift magnitude

Also reports SD_w (and hence MDC95) by dt bin from the null-change stratum: if
the repeatability floor is flat across dt, elapsed time is not what is driving
the result.
"""
import json
import numpy as np, pandas as pd
from scipy import stats
from paths import INTERIM, OUT
from wp_stats import deming, cluster_bootstrap, mdc95

BINS = [(0, 2, "<2 h"), (2, 6, "2-6 h"), (6, 12, "6-12 h"), (12, 24, "12-24 h")]
ANALYTES = ["potassium", "creatinine", "hemoglobin", "calcium_total"]
MIN_D = {"potassium": 0.5, "creatinine": 0.3, "hemoglobin": 1.0, "calcium_total": 0.5, "calcium_corrected": 0.5,
         "ecgfounder": 0.5}

rows, null_rows = [], []
for a in ANALYTES:
    d = pd.read_parquet(INTERIM / f"delta_scored_{a}.parquet")
    nl = pd.read_parquet(INTERIM / f"null_scored_{a}.parquet")
    for lo, hi, lab in BINS:
        s = d[(d.dt_h >= lo) & (d.dt_h < hi)]
        rec = {"analyte": a, "dt_bin": lab, "n": len(s),
               "n_pat": int(s.subject_id.nunique()) if len(s) else 0,
               "mean_abs_dY": float(s.dy_lab.abs().mean()) if len(s) else np.nan,
               "sd_dY_lab": float(s.dy_lab.std(ddof=1)) if len(s) > 1 else np.nan}
        if len(s) >= 100:
            r = cluster_bootstrap(s, lambda g: float(stats.pearsonr(g.dy_lab, g.dy_ecg)[0]),
                                  n_boot=800)
            rec.update({"pearson": r[0], "ci_lo": r[1], "ci_hi": r[2],
                        "spearman": float(stats.spearmanr(s.dy_lab, s.dy_ecg)[0]),
                        "deming_slope": deming(s.dy_lab.values, s.dy_ecg.values)[0]})
        # magnitude-matched: same bins, restricted to clinically relevant shifts
        m = s[s.dy_lab.abs() >= MIN_D[a]]
        if len(m) >= 100:
            rm = cluster_bootstrap(m, lambda g: float(stats.pearsonr(g.dy_lab, g.dy_ecg)[0]),
                                   n_boot=800)
            rec.update({"n_matched": len(m), "pearson_matched": rm[0],
                        "matched_ci_lo": rm[1], "matched_ci_hi": rm[2],
                        "deming_matched": deming(m.dy_lab.values, m.dy_ecg.values)[0]})
        rows.append(rec)

        # repeatability floor by dt, from the null-change stratum
        q = nl[(nl.dt_h >= lo) & (nl.dt_h < hi)]
        if len(q) >= 50:
            sd_w = float(q.dy_ecg.std(ddof=1) / np.sqrt(2))
            null_rows.append({"analyte": a, "dt_bin": lab, "n": len(q),
                              "sd_within": sd_w, "mdc95": mdc95(sd_w)})

df = pd.DataFrame(rows); nf = pd.DataFrame(null_rows)
df.to_csv(OUT / "table7_delta_by_dt.csv", index=False)
nf.to_csv(OUT / "table8_mdc_by_dt.csv", index=False)
pd.set_option("display.width", 240)
print(df[df.analyte == "potassium"].to_string(index=False))
print("\nMDC by dt (potassium):")
print(nf[nf.analyte == "potassium"].to_string(index=False))
print("\nAll analytes, raw r by dt:")
print(df.pivot_table(index="analyte", columns="dt_bin", values="pearson").round(3).to_string())
print("\nAll analytes, magnitude-matched r by dt:")
print(df.pivot_table(index="analyte", columns="dt_bin", values="pearson_matched").round(3).to_string())

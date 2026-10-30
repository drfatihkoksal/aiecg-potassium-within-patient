"""Within-patient potassium change tracking by sex (JACC: CEP asks for sex-specific data).

Sex is the `gender` field recorded in MIMIC-IV (F/M), taken at the first ECG of each
pair. For each model and sex the script reports the main change-agreement measures
of the manuscript, with patient-level bootstrap CIs:

  r                  - Pearson r of estimated and measured change
  deming_slope       - fraction of the measured change the estimate captures
  ratio_of_means     - mean estimated / mean measured change
  mae_reduction_pct  - error reduction of the estimated change vs assuming no change

and the female-minus-male difference in r and in the error reduction (bootstrap over
patients of both sexes together). Stockholm model 3 uses the MIMIC-training-only
calibration slope, as in wp9_01.
"""
import json
import duckdb, numpy as np, pandas as pd
from paths import INTERIM, OUT
from wp_stats import cluster_bootstrap, deming

N_BOOT = 2000
cal = json.load(open(OUT / "stockholm_comparison.json"))["calibration"]["stockholm_3"]
d = pd.read_parquet(INTERIM / "delta_scored_stockholm_z.parquet")
con = duckdb.connect(str(INTERIM / "wp0.duckdb"), read_only=True)
sex = con.execute("select study_id as pre_study, gender from obs_potassium").df()
d = d.merge(sex, on="pre_study", how="left", validate="many_to_one")
assert d.gender.isin(["F", "M"]).all(), d.gender.value_counts(dropna=False)
MODELS = {"In-house ResNet": d.potassium, "ECGFounder": d.ecgfounder,
          "Stockholm model 3": d.stockholm_3 * cal["slope_mEq_per_z"]}


def agreement(x, y):
    return {"r": np.corrcoef(x, y)[0, 1],
            "ratio_of_means": y.mean() / x.mean(),
            "mae_reduction_pct": 100 * (1 - np.abs(y - x).mean() / np.abs(x).mean())}


def by_sex(g):
    out = {}
    for s in ("F", "M"):
        h = g[g.gender == s]
        for k, v in agreement(h.dy_lab.to_numpy(), h.dy_ecg.to_numpy()).items():
            out[f"{k}_{s}"] = v
    out["r_diff_F_minus_M"] = out["r_F"] - out["r_M"]
    out["mae_reduction_diff_F_minus_M"] = out["mae_reduction_pct_F"] - out["mae_reduction_pct_M"]
    return out


rows = []
for model, dy in MODELS.items():
    s = d[["subject_id", "gender", "dy_lab"]].assign(dy_ecg=dy.to_numpy())
    est = cluster_bootstrap(s, by_sex, n_boot=N_BOOT)
    for sx, label in (("F", "Female"), ("M", "Male")):
        h = s[s.gender == sx]
        rec = {"model": model, "sex": label, "n_pairs": len(h),
               "n_patients": int(h.subject_id.nunique()),
               "deming_slope": deming(h.dy_lab, h.dy_ecg)[0]}
        for k in ("r", "ratio_of_means", "mae_reduction_pct"):
            pt, lo, hi = est[f"{k}_{sx}"]
            rec.update({k: pt, f"{k}_lo": lo, f"{k}_hi": hi})
        rows.append(rec)
    for k in ("r_diff_F_minus_M", "mae_reduction_diff_F_minus_M"):
        pt, lo, hi = est[k]
        rows.append({"model": model, "sex": k, "n_pairs": len(s),
                     "n_patients": int(s.subject_id.nunique()),
                     "estimate": pt, "estimate_lo": lo, "estimate_hi": hi})
    print(model, {k: tuple(round(float(x), 3) for x in v) for k, v in est.items()})

pd.DataFrame(rows).to_csv(OUT / "table28_sex_stratified.csv", index=False)

"""Direction versus magnitude of within-patient potassium change (manuscript core).

Published AI-ECG "monitoring" studies select extreme, treated cohorts and report
group trajectories or repeated-measures correlation (rmcorr). With two
measurements per pair, rmcorr equals the *uncentred* correlation of the two
changes, so it rewards a shared direction of change, not its size or ranking.
This script reproduces those selections inside the audit cohort for all three
models and reports, per subset:

  mean_dK_lab / mean_dK_ecg - group trajectory, as the monitoring papers report
  ratio_of_means            - fraction of the laboratory change the ECG reproduces
  r_individual              - Pearson r of the changes (individual ranking)
  rmcorr                    - uncentred correlation of the changes
  direction_agree           - share of pairs with |dK_lab| >= 0.5 whose ECG change
                              has the same sign

Stockholm model 3 is put on the mEq/L scale with the MIMIC-training-only
calibration slope from stockholm_02_compare.py; correlations are scale-free.
"""
import json
import numpy as np, pandas as pd
from paths import INTERIM, OUT
from wp_stats import cluster_bootstrap, deming

N_BOOT = 2000
cal = json.load(open(OUT / "stockholm_comparison.json"))["calibration"]["stockholm_3"]
d = pd.read_parquet(INTERIM / "delta_scored_stockholm_z.parquet")
d["dial"] = d.hd_icu.astype(bool) | d.hd_icd.astype(bool)
MODELS = {"In-house ResNet": d.potassium, "ECGFounder": d.ecgfounder,
          "Stockholm model 3": d.stockholm_3 * cal["slope_mEq_per_z"]}
SUBSETS = [("All serial pairs", d.y_pre > 0),
           ("Severe hypokalaemia (K_pre <= 2.5)", d.y_pre <= 2.5),
           ("Severe hyperkalaemia (K_pre >= 6.5)", d.y_pre >= 6.5),
           ("Severe hyperkalaemia with dialysis", (d.y_pre >= 6.5) & d.dial)]


def metrics(g):
    x, y = g.dy_lab.to_numpy(), g.dy_ecg.to_numpy()
    big = np.abs(x) >= 0.5
    return {"mean_dK_lab": x.mean(), "mean_dK_ecg": y.mean(),
            "ratio_of_means": y.mean() / x.mean(),
            "r_individual": np.corrcoef(x, y)[0, 1],
            "rmcorr": (x * y).sum() / np.sqrt((x ** 2).sum() * (y ** 2).sum()),
            "direction_agree": np.mean(np.sign(y[big]) == np.sign(x[big])) if big.any() else np.nan}


rows = []
for model, dy in MODELS.items():
    for name, mask in SUBSETS:
        s = d.loc[mask, ["subject_id", "dy_lab"]].assign(dy_ecg=dy[mask].to_numpy())
        est = cluster_bootstrap(s, metrics, n_boot=N_BOOT)
        rec = {"model": model, "subset": name, "n_pairs": len(s),
               "n_patients": int(s.subject_id.nunique()),
               "deming_slope": deming(s.dy_lab, s.dy_ecg)[0]}
        for k, (pt, lo, hi) in est.items():
            rec.update({k: pt, f"{k}_lo": lo, f"{k}_hi": hi})
        rows.append(rec)
        print(f"{model:18s} {name:38s} n={len(s):5d} lab={rec['mean_dK_lab']:+.2f} "
              f"ecg={rec['mean_dK_ecg']:+.2f} ratio={rec['ratio_of_means']:.2f} "
              f"r={rec['r_individual']:.3f} rmcorr={rec['rmcorr']:.3f} "
              f"dir={rec['direction_agree']:.3f}")

pd.DataFrame(rows).to_csv(OUT / "table27_direction_magnitude.csv", index=False)

"""WP8d - when is the ECG worth adding to the last known value?

The anchor K_pre goes stale as the interval grows, so the value of fusing an
ECG-derived level estimate should rise with dt. Stratified on the delta-test
patients; paired bootstrap clustered by patient within each stratum.
"""
import numpy as np, pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor
from paths import INTERIM, OUT

SEED = 23
a = pd.read_parquet(INTERIM / "audit_potassium.parquet")
sp = pd.read_parquet(INTERIM / "wp8_delta_split.parquet")
pr = pd.read_parquet(INTERIM / "pred_potassium.parquet")[["study_id", "pred_potassium"]]
a = (a.merge(sp, on=["subject_id", "pre_study", "post_study"])
       .merge(pr.rename(columns={"study_id": "pre_study", "pred_potassium": "f_pre"}), on="pre_study")
       .merge(pr.rename(columns={"study_id": "post_study", "pred_potassium": "f_post"}), on="post_study"))
tr, te = a[a.split == "train"], a[a.split == "test"]

def fit(cols):
    m = HistGradientBoostingRegressor(max_depth=3, max_iter=300, learning_rate=0.05,
                                      random_state=SEED)
    return m.fit(tr[cols].values.astype(np.float32), tr.y_post.values)

M = {"anchor only": fit(["y_pre", "dt_h"]),
     "ECG only": fit(["f_post", "f_pre", "dt_h"]),
     "anchor + ECG": fit(["y_pre", "f_post", "f_pre", "dt_h"])}
COLS = {"anchor only": ["y_pre", "dt_h"], "ECG only": ["f_post", "f_pre", "dt_h"],
        "anchor + ECG": ["y_pre", "f_post", "f_pre", "dt_h"]}

def paired(e1, e2, sid, n_boot=3000, seed=5):
    rng = np.random.default_rng(seed)
    g = {k: np.asarray(v) for k, v in pd.Series(range(len(sid))).groupby(sid).indices.items()}
    keys = np.array(list(g.keys()))
    obs = e1.mean() - e2.mean(); d = []
    for _ in range(n_boot):
        idx = np.concatenate([g[k] for k in rng.choice(keys, len(keys), replace=True)])
        d.append(e1[idx].mean() - e2[idx].mean())
    return obs, float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))

BINS = [(0, 6, "<6 h"), (6, 12, "6-12 h"), (12, 24, "12-24 h")]
rows = []
for lo, hi, lab in BINS + [(0, 24, "all")]:
    s = te[(te.dt_h >= lo) & (te.dt_h < hi)] if lab != "all" else te
    if len(s) < 60:
        continue
    y = s.y_post.values; sid = s.subject_id.values
    e = {k: np.abs(M[k].predict(s[COLS[k]].values.astype(np.float32)) - y) for k in M}
    o, l, h = paired(e["anchor + ECG"], e["anchor only"], sid)
    o2, l2, h2 = paired(e["anchor + ECG"], e["ECG only"], sid)
    rows.append({"dt_bin": lab, "n": len(s), "n_pat": s.subject_id.nunique(),
                 "MAE_anchor": e["anchor only"].mean(), "MAE_ecg": e["ECG only"].mean(),
                 "MAE_fused": e["anchor + ECG"].mean(),
                 "gain_vs_anchor": o, "gv_lo": l, "gv_hi": h,
                 "gain_vs_ecg": o2, "ge_lo": l2, "ge_hi": h2})
r = pd.DataFrame(rows)
pd.set_option("display.width", 250)
print(r.round(4).to_string(index=False))
r.to_csv(OUT / "table19_fusion_by_dt.csv", index=False)

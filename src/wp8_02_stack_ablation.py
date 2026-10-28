"""WP8b - what does the ECG actually contribute to the fused predictor?

The stacked model beats the cross-sectional model, but it has access to K_pre,
and an anchor plus shrinkage improves K_post without any ECG information. This
ablates the stack's feature groups and tests every contrast with a PAIRED
bootstrap of the MAE difference, clustered by patient.
"""
import json
import numpy as np, pandas as pd
from scipy import stats
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import HistGradientBoostingRegressor
from paths import INTERIM, OUT

SEED = 23
a = pd.read_parquet(INTERIM / "audit_potassium.parquet")
sp = pd.read_parquet(INTERIM / "wp8_delta_split.parquet")
widx = pd.read_parquet(INTERIM / "wave_index.parquet", columns=["study_id", "row"])
pr = pd.read_parquet(INTERIM / "pred_potassium.parquet")[["study_id", "pred_potassium"]]
E = np.asarray(np.load(INTERIM / "emb_potassium.npy", mmap_mode="r"), dtype=np.float32)

a = (a.merge(sp, on=["subject_id", "pre_study", "post_study"])
       .merge(widx.rename(columns={"study_id": "pre_study", "row": "row_pre"}), on="pre_study")
       .merge(widx.rename(columns={"study_id": "post_study", "row": "row_post"}), on="post_study")
       .merge(pr.rename(columns={"study_id": "pre_study", "pred_potassium": "f_pre"}), on="pre_study")
       .merge(pr.rename(columns={"study_id": "post_study", "pred_potassium": "f_post"}), on="post_study"))
tr, te = a[a.split == "train"], a[a.split == "test"]

def dfeat(df):
    return np.hstack([E[df.row_post.values] - E[df.row_pre.values],
                      df[["dt_h"]].values.astype(np.float32)])
dh = RidgeCV(alphas=np.logspace(-1, 4, 24)).fit(dfeat(tr), tr.dy_lab.values)
dh_tr, dh_te = dh.predict(dfeat(tr)), dh.predict(dfeat(te))

GROUPS = {
    "anchor only            [K_pre, dt]":              (["y_pre", "dt_h"], False),
    "ECG only               [f_post, f_pre, dt]":      (["f_post", "f_pre", "dt_h"], False),
    "anchor + ECG level     [K_pre, f_post, dt]":      (["y_pre", "f_post", "dt_h"], False),
    "anchor + ECG level+pre [K_pre, f_post, f_pre, dt]": (["y_pre", "f_post", "f_pre", "dt_h"], False),
    "anchor + delta head    [K_pre, dhat, dt]":        (["y_pre", "dt_h"], True),
    "FULL                   [K_pre, f_post, f_pre, dhat, dt]": (["y_pre", "f_post", "f_pre", "dt_h"], True),
}
preds = {}
for name, (cols, use_dh) in GROUPS.items():
    Xtr = tr[cols].values.astype(np.float32); Xte = te[cols].values.astype(np.float32)
    if use_dh:
        Xtr = np.column_stack([Xtr, dh_tr]); Xte = np.column_stack([Xte, dh_te])
    m = HistGradientBoostingRegressor(max_depth=3, max_iter=300, learning_rate=0.05,
                                      random_state=SEED).fit(Xtr, tr.y_post.values)
    preds[name] = m.predict(Xte)

y = te.y_post.values
sid = te.subject_id.values
rows = [{"model": k, "MAE": float(np.abs(v - y).mean()),
         "RMSE": float(np.sqrt(((v - y) ** 2).mean())),
         "r": float(stats.pearsonr(v, y)[0])} for k, v in preds.items()]
res = pd.DataFrame(rows).sort_values("MAE")
pd.set_option("display.width", 220)
print("=== stack ablation, predicting K_post (held-out patients) ===")
print(res.round(4).to_string(index=False))

# paired bootstrap of MAE differences, clustered by patient
def paired(a_pred, b_pred, n_boot=3000, seed=5):
    rng = np.random.default_rng(seed)
    groups = {k: np.asarray(v) for k, v in pd.Series(range(len(sid))).groupby(sid).indices.items()}
    keys = np.array(list(groups.keys()))
    ea, eb = np.abs(a_pred - y), np.abs(b_pred - y)
    obs = ea.mean() - eb.mean()
    d = []
    for _ in range(n_boot):
        idx = np.concatenate([groups[k] for k in rng.choice(keys, len(keys), replace=True)])
        d.append(ea[idx].mean() - eb[idx].mean())
    d = np.asarray(d)
    return obs, float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))

CONTRASTS = [
    ("does the ECG add anything to the anchor?",
     "anchor + ECG level     [K_pre, f_post, dt]", "anchor only            [K_pre, dt]"),
    ("does the DELTA HEAD add to the anchor?",
     "anchor + delta head    [K_pre, dhat, dt]",   "anchor only            [K_pre, dt]"),
    ("does the DELTA HEAD add on top of ECG levels?",
     "FULL                   [K_pre, f_post, f_pre, dhat, dt]",
     "anchor + ECG level+pre [K_pre, f_post, f_pre, dt]"),
    ("does the anchor add to the ECG model?",
     "anchor + ECG level     [K_pre, f_post, dt]", "ECG only               [f_post, f_pre, dt]"),
]
print("\n=== paired MAE differences (negative favours the FIRST model) ===")
out = []
for label, m1, m2 in CONTRASTS:
    o, lo, hi = paired(preds[m1], preds[m2])
    sig = "yes" if hi < 0 or lo > 0 else "no"
    out.append({"contrast": label, "dMAE": round(o, 4),
                "ci_lo": round(lo, 4), "ci_hi": round(hi, 4), "significant": sig})
    print(f"{label:48s} dMAE={o:+.4f} [{lo:+.4f},{hi:+.4f}]  significant={sig}")

res.to_csv(OUT / "table15_stack_ablation.csv", index=False)
pd.DataFrame(out).to_csv(OUT / "table16_stack_contrasts.csv", index=False)

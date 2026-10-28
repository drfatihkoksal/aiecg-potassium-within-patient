"""WP8f - EXPLORATORY contrast ladder isolating what the second ECG contributes.

Design note. The audit deliberately held out every patient with a short-interval
ECG pair, so no such patient was ever seen by the cross-sectional models. To
train a change model those patients are now re-partitioned into a NEW
patient-level train/validation/test split (60/20/20). Because results for this
cohort were already inspected during the audit, everything here is reported as
EXPLORATORY, not confirmatory.

The ladder separates three sources of gain that the earlier ablation conflated:

  A  anchor only              [K_pre, dt]
  B  anchor + CURRENT ECG     [K_pre, f_post, dt]          <- new control
  C  anchor + BOTH ECGs       [K_pre, f_post, f_pre, dt]
  D  C + explicit delta head  [.., dhat]
  E  ECG only                 [f_post, f_pre, dt]

  B - A : does the current ECG add to the last known value?
  C - B : does COMPARING TWO ECGs add beyond the current ECG?   <- the question
  D - C : does an explicit delta read-out add on top?

Run for the in-house model and for the frozen published Stockholm checkpoint.
"""
import json
import numpy as np, pandas as pd
from scipy import stats
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import HistGradientBoostingRegressor
from paths import INTERIM, OUT

SEED = 23
a0 = pd.read_parquet(INTERIM / "audit_potassium.parquet")
sp = pd.read_parquet(INTERIM / "wp8_delta_split.parquet")
widx = pd.read_parquet(INTERIM / "wave_index.parquet", columns=["study_id", "row"])
E = np.asarray(np.load(INTERIM / "emb_potassium.npy", mmap_mode="r"), dtype=np.float32)

def build(source):
    a = a0.merge(sp, on=["subject_id", "pre_study", "post_study"])
    if source == "inhouse":
        pr = pd.read_parquet(INTERIM / "pred_potassium.parquet")[["study_id", "pred_potassium"]]
        pr = pr.rename(columns={"pred_potassium": "f"})
    else:
        cal = json.load(open(OUT / "stockholm_comparison.json"))["calibration"]["stockholm_3"]
        pr = pd.read_parquet(INTERIM / "pred_stockholm.parquet")[["study_id", "stockholm_3_z"]]
        pr["f"] = cal["intercept_mEq"] + cal["slope_mEq_per_z"] * pr["stockholm_3_z"]
        pr = pr[["study_id", "f"]]
    a = (a.merge(pr.rename(columns={"study_id": "pre_study", "f": "f_pre"}), on="pre_study")
           .merge(pr.rename(columns={"study_id": "post_study", "f": "f_post"}), on="post_study")
           .merge(widx.rename(columns={"study_id": "pre_study", "row": "row_pre"}), on="pre_study")
           .merge(widx.rename(columns={"study_id": "post_study", "row": "row_post"}), on="post_study"))
    return a[a.split == "train"], a[a.split == "test"]

def paired(e1, e2, sid, n_boot=4000, seed=5):
    rng = np.random.default_rng(seed)
    g = {k: np.asarray(v) for k, v in pd.Series(range(len(sid))).groupby(sid).indices.items()}
    keys = np.array(list(g.keys())); obs = e1.mean() - e2.mean(); d = []
    for _ in range(n_boot):
        idx = np.concatenate([g[k] for k in rng.choice(keys, len(keys), replace=True)])
        d.append(e1[idx].mean() - e2[idx].mean())
    return obs, float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))

LADDER = {"A anchor only": ["y_pre", "dt_h"],
          "B anchor + current ECG": ["y_pre", "f_post", "dt_h"],
          "C anchor + both ECGs": ["y_pre", "f_post", "f_pre", "dt_h"],
          "E ECG only": ["f_post", "f_pre", "dt_h"]}
CONTRASTS = [("B - A  current ECG adds to anchor", "B anchor + current ECG", "A anchor only"),
             ("C - B  SECOND ECG adds beyond current", "C anchor + both ECGs", "B anchor + current ECG"),
             ("D - C  explicit delta head adds", "D + delta head", "C anchor + both ECGs")]

all_rows, all_con = [], []
for src in ("inhouse", "stockholm"):
    tr, te = build(src)
    y, sid = te.y_post.values, te.subject_id.values
    dfeat = lambda d: np.hstack([E[d.row_post.values] - E[d.row_pre.values],
                                 d[["dt_h"]].values.astype(np.float32)])
    dh = RidgeCV(alphas=np.logspace(-1, 4, 24)).fit(dfeat(tr), tr.dy_lab.values)
    preds = {}
    for name, cols in LADDER.items():
        m = HistGradientBoostingRegressor(max_depth=3, max_iter=300, learning_rate=0.05,
                                          random_state=SEED).fit(tr[cols].values.astype(np.float32),
                                                                 tr.y_post.values)
        preds[name] = m.predict(te[cols].values.astype(np.float32))
    cols = LADDER["C anchor + both ECGs"]
    Xtr = np.column_stack([tr[cols].values.astype(np.float32), dh.predict(dfeat(tr))])
    Xte = np.column_stack([te[cols].values.astype(np.float32), dh.predict(dfeat(te))])
    m = HistGradientBoostingRegressor(max_depth=3, max_iter=300, learning_rate=0.05,
                                      random_state=SEED).fit(Xtr, tr.y_post.values)
    preds["D + delta head"] = m.predict(Xte)

    for k, v in preds.items():
        all_rows.append({"source": src, "model": k, "n_test": len(te),
                         "MAE": float(np.abs(v - y).mean()),
                         "r": float(stats.pearsonr(v, y)[0])})
    e = {k: np.abs(v - y) for k, v in preds.items()}
    for lab, m1, m2 in CONTRASTS:
        o, lo, hi = paired(e[m1], e[m2], sid)
        all_con.append({"source": src, "contrast": lab, "dMAE": o, "ci_lo": lo, "ci_hi": hi,
                        "significant": "yes" if hi < 0 or lo > 0 else "no"})

res = pd.DataFrame(all_rows); con = pd.DataFrame(all_con)
pd.set_option("display.width", 250)
print("=== EXPLORATORY: predicting K_post on the new delta-test split ===")
print(res.pivot(index="model", columns="source", values="MAE").round(4).to_string())
print("\n=== paired MAE differences (negative favours the first model) ===")
for src in ("inhouse", "stockholm"):
    print(f"\n-- {src} --")
    for r in con[con.source == src].itertuples():
        print(f"  {r.contrast:42s} {r.dMAE:+.4f} [{r.ci_lo:+.4f},{r.ci_hi:+.4f}]  sig={r.significant}")
res.to_csv(OUT / "table22_contrast_ladder.csv", index=False)
con.to_csv(OUT / "table23_contrast_ladder_paired.csv", index=False)

"""WP8 - does a purpose-built delta-K model improve estimation of the K level?

Framed as the decision-relevant serial-monitoring task: predict K_post when the
previous laboratory value K_pre is already known.

Predictors of K_post, all scored on the SAME held-out patients:
  xs            f(ECG_post)                      - the standard cross-sectional model
  lvcf          K_pre                            - last value carried forward
  patient_mean  the patient's historical mean
  xs_anchored   K_pre + [f(ECG_post) - f(ECG_pre)]   - the implied delta of the existing model
  delta_ecg     K_pre + dhat(emb_pre, emb_post)      - purpose-trained delta head, ECG only
  delta_ecg_kpre K_pre + dhat(emb, K_pre)            - delta head allowed to see K_pre
  regress_mean  K_pre + dhat(K_pre only)             - NULL CONTROL: regression to the mean,
                                                       no ECG information at all
  stack         learned combination of everything

The regress_mean control is essential: a delta head given K_pre can lower the
error simply by shrinking toward the population mean, which improves K_post
without reading the ECG at all. Any claimed benefit must beat this control.

Embeddings come from the cross-sectional model, which never saw these patients,
so they are out-of-sample throughout.
"""
import json
import numpy as np, pandas as pd
from scipy import stats
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import HistGradientBoostingRegressor
from paths import INTERIM, OUT
from wp_stats import cluster_bootstrap

SEED = 23
rng = np.random.default_rng(SEED)

a = pd.read_parquet(INTERIM / "audit_potassium.parquet")
widx = pd.read_parquet(INTERIM / "wave_index.parquet", columns=["study_id", "row"])
pred = pd.read_parquet(INTERIM / "pred_potassium.parquet")[["study_id", "pred_potassium"]]
emb = np.load(INTERIM / "emb_potassium.npy", mmap_mode="r")

a = (a.merge(widx.rename(columns={"study_id": "pre_study", "row": "row_pre"}), on="pre_study")
       .merge(widx.rename(columns={"study_id": "post_study", "row": "row_post"}), on="post_study")
       .merge(pred.rename(columns={"study_id": "pre_study", "pred_potassium": "f_pre"}), on="pre_study")
       .merge(pred.rename(columns={"study_id": "post_study", "pred_potassium": "f_post"}), on="post_study"))

# patient-level split of the delta cohort (60/20/20)
pids = a.subject_id.unique(); rng.shuffle(pids)
n1, n2 = int(0.60 * len(pids)), int(0.80 * len(pids))
grp = {p: "train" for p in pids[:n1]}
grp |= {p: "val" for p in pids[n1:n2]}
grp |= {p: "test" for p in pids[n2:]}
a["split"] = a.subject_id.map(grp)
print(a.groupby("split").agg(pairs=("dy_lab", "size"), patients=("subject_id", "nunique")).to_string())
assert not (set(a[a.split == "train"].subject_id) & set(a[a.split == "test"].subject_id))

E = np.asarray(emb, dtype=np.float32)
def feats(df, with_kpre):
    d = E[df.row_post.values] - E[df.row_pre.values]
    X = [d, df[["dt_h"]].values.astype(np.float32)]
    if with_kpre:
        X.append(df[["y_pre"]].values.astype(np.float32))
    return np.hstack(X)

tr, va, te = (a[a.split == s] for s in ("train", "val", "test"))

def fit_delta(with_kpre, ecg=True):
    if ecg:
        Xtr, Xte = feats(tr, with_kpre), feats(te, with_kpre)
    else:  # null control: K_pre (and dt) only, no ECG at all
        Xtr = tr[["y_pre", "dt_h"]].values.astype(np.float32)
        Xte = te[["y_pre", "dt_h"]].values.astype(np.float32)
    m = RidgeCV(alphas=np.logspace(-1, 4, 24)).fit(Xtr, tr.dy_lab.values)
    return m.predict(Xte)

dhat_ecg      = fit_delta(with_kpre=False, ecg=True)
dhat_ecg_kpre = fit_delta(with_kpre=True,  ecg=True)
dhat_null     = fit_delta(with_kpre=True,  ecg=False)

# patient historical mean, from draws before the pre ECG
import duckdb
con = duckdb.connect(str(INTERIM / "wp0.duckdb")); con.register("te", te)
pm = con.execute("""
 select te.pre_study, te.post_study,
   (select avg(valuenum) from labs l where l.subject_id=te.subject_id
      and l.series='k_chem' and l.charttime < te.pre_ecg) as pmean
 from te""").df()
te = te.merge(pm, on=["pre_study", "post_study"])

P = {
    "xs  f(ECG_post)":                       te.f_post.values,
    "lvcf  K_pre":                           te.y_pre.values,
    "patient_mean":                          te.pmean.values,
    "xs_anchored  K_pre + [f_post - f_pre]": te.y_pre.values + (te.f_post - te.f_pre).values,
    "delta_ecg  K_pre + dhat(ECG)":          te.y_pre.values + dhat_ecg,
    "delta_ecg_kpre  K_pre + dhat(ECG,K_pre)": te.y_pre.values + dhat_ecg_kpre,
    "regress_mean  K_pre + dhat(K_pre)  [NULL]": te.y_pre.values + dhat_null,
}
# stacked combiner, fitted on train, using everything
def stack_feats(df, dh):
    return np.column_stack([df.f_post.values, df.f_pre.values, df.y_pre.values,
                            dh, df.dt_h.values])
dh_tr = RidgeCV(alphas=np.logspace(-1, 4, 24)).fit(feats(tr, False), tr.dy_lab.values)
st = HistGradientBoostingRegressor(max_depth=3, max_iter=300, learning_rate=0.05,
                                   random_state=SEED)
st.fit(stack_feats(tr, dh_tr.predict(feats(tr, False))), tr.y_post.values)
P["stack  GBM(all features)"] = st.predict(stack_feats(te, dhat_ecg))

y = te.y_post.values
rows = []
for name, p in P.items():
    ok = np.isfinite(p) & np.isfinite(y)
    d = pd.DataFrame({"subject_id": te.subject_id.values[ok], "p": p[ok], "y": y[ok]})
    mae = cluster_bootstrap(d, lambda g: float(np.abs(g.p - g.y).mean()), n_boot=1500)
    rows.append({"predictor": name, "n": int(ok.sum()),
                 "MAE": mae[0], "MAE_lo": mae[1], "MAE_hi": mae[2],
                 "RMSE": float(np.sqrt(((p[ok] - y[ok]) ** 2).mean())),
                 "r": float(stats.pearsonr(p[ok], y[ok])[0])})
res = pd.DataFrame(rows).sort_values("MAE")
pd.set_option("display.width", 220)
print("\n=== predicting K_post on held-out patients ===")
print(res.round(4).to_string(index=False))

# --- the delta task itself: does a purpose-trained head beat the difference of predictions?
dl = te.dy_lab.values
dt_rows = []
for name, dh in [("f_post - f_pre (existing model)", (te.f_post - te.f_pre).values),
                 ("dhat(ECG) purpose-trained", dhat_ecg),
                 ("dhat(ECG, K_pre)", dhat_ecg_kpre),
                 ("dhat(K_pre) [NULL, no ECG]", dhat_null)]:
    d = pd.DataFrame({"subject_id": te.subject_id.values, "a": dh, "b": dl})
    r = cluster_bootstrap(d, lambda g: float(stats.pearsonr(g.a, g.b)[0]), n_boot=1500)
    dt_rows.append({"delta_predictor": name, "r": r[0], "ci_lo": r[1], "ci_hi": r[2],
                    "MAE_delta": float(np.abs(dh - dl).mean())})
dres = pd.DataFrame(dt_rows)
print("\n=== the delta task itself (correlation with lab dK) ===")
print(dres.round(4).to_string(index=False))

res.to_csv(OUT / "table13_delta_fusion_kpost.csv", index=False)
dres.to_csv(OUT / "table14_delta_task_heads.csv", index=False)
a[["subject_id", "pre_study", "post_study", "split"]].to_parquet(
    INTERIM / "wp8_delta_split.parquet", index=False)

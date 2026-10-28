"""WP8i - final siamese comparison, seed-ensembled, with a leakage-free D-C test.

Run-to-run variance was the same size as the effect, so each mode is run over 5
seeds and the seed-mean prediction is used. The K_post rung is tested without
touching test: rung C is fitted on Δ-train, the combiner [C_pred, Δ̂] is fitted on
Δ-VAL, and only the final scoring happens on Δ-test.
"""
import json
import numpy as np, pandas as pd
from scipy import stats
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import HistGradientBoostingRegressor
from paths import INTERIM, OUT

SEEDS = [0, 1, 2, 3, 4]
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
tr, va, te = (a[a.split == s] for s in ("train", "val", "test"))
lab, sid = te.dy_lab.values, te.subject_id.values

ens = lambda mode, part: np.mean(
    [np.load(INTERIM / f"dhat_siamese_{mode}{part}_s{s}.npy") for s in SEEDS], axis=0)
Xb = lambda d: np.hstack([E[d.row_pre.values], E[d.row_post.values]])
exp1_model = RidgeCV(alphas=np.logspace(-1, 4, 30)).fit(Xb(tr), tr.dy_lab.values)

P = {"implied Δ  f_post − f_pre": (te.f_post - te.f_pre).values,
     "exp1  ridge on frozen embeddings": exp1_model.predict(Xb(te)),
     "exp2a siamese head, frozen trunk": ens("head", ""),
     "exp2b siamese, trunk fine-tuned":  ens("full", "")}
PER_SEED = {m: [float(stats.pearsonr(np.load(INTERIM / f"dhat_siamese_{m}_s{s}.npy"), lab)[0])
                for s in SEEDS] for m in ("head", "full")}

G = {k: np.asarray(v) for k, v in pd.Series(range(len(sid))).groupby(sid).indices.items()}
KEYS = np.array(list(G.keys()))
def bidx(rng): return np.concatenate([G[k] for k in rng.choice(KEYS, len(KEYS), replace=True)])
def ci_r(p, n=3000, seed=9):
    rng = np.random.default_rng(seed)
    d = [float(stats.pearsonr(p[i], lab[i])[0]) for i in (bidx(rng) for _ in range(n))]
    return float(stats.pearsonr(p, lab)[0]), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))
def paired_r(p1, p2, n=3000, seed=9):
    rng = np.random.default_rng(seed)
    obs = float(stats.pearsonr(p1, lab)[0] - stats.pearsonr(p2, lab)[0]); d = []
    for _ in range(n):
        i = bidx(rng)
        d.append(float(stats.pearsonr(p1[i], lab[i])[0] - stats.pearsonr(p2[i], lab[i])[0]))
    return obs, float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))

rows = []
for k, v in P.items():
    r, lo, hi = ci_r(v)
    m = "head" if "2a" in k else ("full" if "2b" in k else None)
    rows.append({"delta_predictor": k, "r_ensemble": r, "ci_lo": lo, "ci_hi": hi,
                 "MAE_delta": float(np.abs(v - lab).mean()),
                 "per_seed_mean": np.mean(PER_SEED[m]) if m else np.nan,
                 "per_seed_sd": np.std(PER_SEED[m]) if m else np.nan})
res = pd.DataFrame(rows)
pd.set_option("display.width", 240)
print("=== Δ task, seed-ensembled (n=%d pairs / %d patients) ===" % (len(te), te.subject_id.nunique()))
print(res.round(4).to_string(index=False))

print("\n=== paired Δr, clustered by patient ===")
con = []
base = P["implied Δ  f_post − f_pre"]
for k in list(P)[1:]:
    o, lo, hi = paired_r(P[k], base)
    con.append({"contrast": f"{k} − implied Δ", "dr": o, "lo": lo, "hi": hi,
                "sig": "yes" if lo > 0 or hi < 0 else "no"})
o, lo, hi = paired_r(P["exp2b siamese, trunk fine-tuned"], P["exp2a siamese head, frozen trunk"])
con.append({"contrast": "exp2b − exp2a (unfreezing the trunk)", "dr": o, "lo": lo, "hi": hi,
            "sig": "yes" if lo > 0 or hi < 0 else "no"})
for c in con:
    print(f"  {c['contrast']:44s} Δr={c['dr']:+.4f} [{c['lo']:+.4f},{c['hi']:+.4f}]  sig={c['sig']}")

# ---- leakage-free K_post rung: C fitted on train, combiner fitted on VAL -----
cols = ["y_pre", "f_post", "f_pre", "dt_h"]
mC = HistGradientBoostingRegressor(max_depth=3, max_iter=300, learning_rate=0.05,
                                   random_state=SEED).fit(tr[cols].values.astype(np.float32),
                                                          tr.y_post.values)
Cva, Cte = mC.predict(va[cols].values.astype(np.float32)), mC.predict(te[cols].values.astype(np.float32))
print("\n=== K_post: does the seed-ensembled siamese Δ̂ add to rung C? ===")
print("    (rung C fitted on Δ-train; combiner fitted on Δ-val; scored on Δ-test)")
kp = []
for mode, name in (("head", "exp2a siamese head"), ("full", "exp2b siamese fine-tuned")):
    dva, dte = ens(mode, "_val"), ens(mode, "")
    comb = RidgeCV(alphas=np.logspace(-2, 3, 20)).fit(
        np.column_stack([Cva, dva]), va.y_post.values)
    pte = comb.predict(np.column_stack([Cte, dte]))
    e1, e2 = np.abs(pte - te.y_post.values), np.abs(Cte - te.y_post.values)
    rng = np.random.default_rng(5); d = []
    for _ in range(4000):
        i = bidx(rng); d.append(e1[i].mean() - e2[i].mean())
    o = e1.mean() - e2.mean()
    lo, hi = float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))
    kp.append({"delta_model": name, "MAE_C": float(e2.mean()), "MAE_C_plus_delta": float(e1.mean()),
               "dMAE": o, "lo": lo, "hi": hi, "sig": "yes" if lo > 0 or hi < 0 else "no"})
    print(f"  {name:28s} C={e2.mean():.4f} -> C+Δ̂={e1.mean():.4f}  "
          f"ΔMAE={o:+.4f} [{lo:+.4f},{hi:+.4f}]  sig={kp[-1]['sig']}")

res.to_csv(OUT / "table24_siamese_delta.csv", index=False)
pd.DataFrame(con).to_csv(OUT / "table25_siamese_paired.csv", index=False)
pd.DataFrame(kp).to_csv(OUT / "table26_siamese_kpost_rung.csv", index=False)

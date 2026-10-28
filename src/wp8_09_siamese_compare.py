"""WP8h - do the siamese Δ models beat differencing the level model?

Paired bootstrap of the correlation difference, clustered by patient, on the
Δ-test split. Then re-runs the contrast ladder's D-C rung with the best siamese
Δ̂ to see whether an explicit Δ read-out now contributes to K_post.
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
lab, sid = te.dy_lab.values, te.subject_id.values

# experiment 1 winner, refitted identically (ridge on both embeddings)
Xb = lambda d: np.hstack([E[d.row_pre.values], E[d.row_post.values]])
exp1 = RidgeCV(alphas=np.logspace(-1, 4, 30)).fit(Xb(tr), tr.dy_lab.values).predict(Xb(te))

P = {"implied Δ  f_post − f_pre": (te.f_post - te.f_pre).values,
     "exp1  ridge on frozen embeddings": exp1,
     "exp2a siamese head, frozen trunk": np.load(INTERIM / "dhat_siamese_head.npy"),
     "exp2b siamese, trunk fine-tuned":  np.load(INTERIM / "dhat_siamese_full.npy")}
for k, v in P.items():
    assert len(v) == len(lab), (k, len(v), len(lab))

G = {k: np.asarray(v) for k, v in pd.Series(range(len(sid))).groupby(sid).indices.items()}
KEYS = np.array(list(G.keys()))
def boot_idx(rng):
    return np.concatenate([G[k] for k in rng.choice(KEYS, len(KEYS), replace=True)])

def ci_r(p, n_boot=3000, seed=9):
    rng = np.random.default_rng(seed)
    d = [float(stats.pearsonr(p[i], lab[i])[0]) for i in (boot_idx(rng) for _ in range(n_boot))]
    return float(stats.pearsonr(p, lab)[0]), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))

def paired_r(p1, p2, n_boot=3000, seed=9):
    rng = np.random.default_rng(seed)
    obs = float(stats.pearsonr(p1, lab)[0] - stats.pearsonr(p2, lab)[0])
    d = []
    for _ in range(n_boot):
        i = boot_idx(rng)
        d.append(float(stats.pearsonr(p1[i], lab[i])[0] - stats.pearsonr(p2[i], lab[i])[0]))
    return obs, float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))

rows = []
for k, v in P.items():
    r, lo, hi = ci_r(v)
    rows.append({"delta_predictor": k, "r": r, "ci_lo": lo, "ci_hi": hi,
                 "MAE_delta": float(np.abs(v - lab).mean())})
res = pd.DataFrame(rows)
pd.set_option("display.width", 220)
print("=== Δ task on the held-out Δ-test patients (n=%d pairs, %d patients) ===" % (len(te), te.subject_id.nunique()))
print(res.round(4).to_string(index=False))

base = P["implied Δ  f_post − f_pre"]
print("\n=== paired difference in r vs differencing the level model ===")
con = []
for k in list(P)[1:]:
    o, lo, hi = paired_r(P[k], base)
    sig = "yes" if lo > 0 or hi < 0 else "no"
    con.append({"contrast": f"{k}  −  implied Δ", "dr": o, "ci_lo": lo, "ci_hi": hi,
                "significant": sig})
    print(f"  {k:38s} Δr={o:+.4f} [{lo:+.4f},{hi:+.4f}]  significant={sig}")
o, lo, hi = paired_r(P["exp2b siamese, trunk fine-tuned"], P["exp2a siamese head, frozen trunk"])
con.append({"contrast": "exp2b − exp2a (unfreezing the trunk)", "dr": o, "ci_lo": lo,
            "ci_hi": hi, "significant": "yes" if lo > 0 or hi < 0 else "no"})
print(f"  {'exp2b − exp2a (unfreezing the trunk)':38s} Δr={o:+.4f} [{lo:+.4f},{hi:+.4f}]  "
      f"significant={con[-1]['significant']}")

# --- does the best siamese Δ̂ now contribute to K_post? (ladder rung D − C) ---
best = max(P, key=lambda k: stats.pearsonr(P[k], lab)[0] if k != "implied Δ  f_post − f_pre" else -9)
print(f"\n=== ladder rung D − C using {best} ===")
# refit the same siamese head predictions are test-only, so fit the stack on train
# using the exp1 ridge (available for train) and report D-C for that; for the
# siamese heads only test predictions exist, so the rung is evaluated by adding
# dhat as a feature to a stack trained WITHOUT it is not possible -> use a
# leakage-free proxy: compare C vs C+dhat via a ridge on test residuals is invalid.
# Instead: report the incremental correlation of dhat with the residual of C.
cols = ["y_pre", "f_post", "f_pre", "dt_h"]
mC = HistGradientBoostingRegressor(max_depth=3, max_iter=300, learning_rate=0.05,
                                   random_state=SEED).fit(tr[cols].values.astype(np.float32),
                                                          tr.y_post.values)
residC = te.y_post.values - mC.predict(te[cols].values.astype(np.float32))
for k, v in P.items():
    r, lo, hi = (lambda t: t)(ci_r(v))  # reuse CI helper shape
    rr = float(stats.pearsonr(v, residC)[0])
    print(f"  corr(Δ̂, residual of rung C) for {k:38s} = {rr:+.4f}")
print("\n(Positive residual correlation is the headroom an explicit Δ read-out could")
print(" still recover; near zero means rung C already contains that information.)")

res.to_csv(OUT / "table24_siamese_delta.csv", index=False)
pd.DataFrame(con).to_csv(OUT / "table25_siamese_paired.csv", index=False)

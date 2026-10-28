"""WP8e - does the fusion result hold for a PUBLISHED potassium checkpoint?

Uses the frozen Stockholm model (primary member stockholm_3) with the user's own
MIMIC-train-only affine calibration from outputs/stockholm_comparison.json, on
the same delta-test patients as WP8. No embeddings are available for a frozen
checkpoint, so this tests the contrasts that matter operationally: anchor alone,
ECG level alone, and their fusion, plus whether the checkpoint's implied delta
adds anything on top of its levels.
"""
import json
import numpy as np, pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor
from paths import INTERIM, OUT

SEED = 23
PRIMARY = "stockholm_3"
cal = json.load(open(OUT / "stockholm_comparison.json"))["calibration"][PRIMARY]
sl, ic = cal["slope_mEq_per_z"], cal["intercept_mEq"]
print(f"{PRIMARY}: {sl:.4f} mEq/z, intercept {ic:.4f} (fit: {cal['fit']})")

a = pd.read_parquet(INTERIM / "audit_potassium.parquet")
sp = pd.read_parquet(INTERIM / "wp8_delta_split.parquet")
st = pd.read_parquet(INTERIM / "pred_stockholm.parquet")[["study_id", f"{PRIMARY}_z"]]
st["f"] = ic + sl * st[f"{PRIMARY}_z"]
a = (a.merge(sp, on=["subject_id", "pre_study", "post_study"])
       .merge(st[["study_id", "f"]].rename(columns={"study_id": "pre_study", "f": "f_pre"}), on="pre_study")
       .merge(st[["study_id", "f"]].rename(columns={"study_id": "post_study", "f": "f_post"}), on="post_study"))
tr, te = a[a.split == "train"], a[a.split == "test"]
print(f"pairs retained with Stockholm scores on both ECGs: train {len(tr):,}, test {len(te):,} "
      f"({te.subject_id.nunique():,} patients)")

COLS = {"anchor only": ["y_pre", "dt_h"],
        "ECG only": ["f_post", "f_pre", "dt_h"],
        "anchor + ECG level": ["y_pre", "f_post", "f_pre", "dt_h"]}
pred = {}
for k, c in COLS.items():
    m = HistGradientBoostingRegressor(max_depth=3, max_iter=300, learning_rate=0.05,
                                      random_state=SEED).fit(tr[c].values.astype(np.float32),
                                                             tr.y_post.values)
    pred[k] = m.predict(te[c].values.astype(np.float32))

y = te.y_post.values; sid = te.subject_id.values
def paired(e1, e2, n_boot=3000, seed=5):
    rng = np.random.default_rng(seed)
    g = {k: np.asarray(v) for k, v in pd.Series(range(len(sid))).groupby(sid).indices.items()}
    keys = np.array(list(g.keys())); obs = e1.mean() - e2.mean(); d = []
    for _ in range(n_boot):
        idx = np.concatenate([g[k] for k in rng.choice(keys, len(keys), replace=True)])
        d.append(e1[idx].mean() - e2[idx].mean())
    return obs, float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))

rows = [{"model": k, "MAE": float(np.abs(v - y).mean()),
         "r": float(stats.pearsonr(v, y)[0])} for k, v in pred.items()]
print("\n=== published Stockholm checkpoint, predicting K_post ===")
print(pd.DataFrame(rows).sort_values("MAE").round(4).to_string(index=False))

e = {k: np.abs(v - y) for k, v in pred.items()}
out = []
for lab, m1, m2 in [("does the ECG add to the anchor?", "anchor + ECG level", "anchor only"),
                    ("does the anchor add to the ECG model?", "anchor + ECG level", "ECG only")]:
    o, lo, hi = paired(e[m1], e[m2])
    out.append({"contrast": lab, "dMAE": o, "ci_lo": lo, "ci_hi": hi,
                "significant": "yes" if hi < 0 or lo > 0 else "no"})
    print(f"{lab:42s} dMAE={o:+.4f} [{lo:+.4f},{hi:+.4f}] significant={out[-1]['significant']}")

# does the checkpoint's implied delta track lab delta, and does it beat no-change?
dimp = (te.f_post - te.f_pre).values; dl = te.dy_lab.values
r = float(stats.pearsonr(dimp, dl)[0])
print(f"\nimplied Δ of the published checkpoint: r = {r:.4f}; "
      f"MAE(Δ) = {np.abs(dimp-dl).mean():.4f} vs no-change MAE = {np.abs(dl).mean():.4f}")

pd.DataFrame(rows).to_csv(OUT / "table20_stockholm_fusion.csv", index=False)
pd.DataFrame(out).to_csv(OUT / "table21_stockholm_fusion_contrasts.csv", index=False)

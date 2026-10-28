"""Corrected cross-subject shuffle (WP6).

The Section 6 shuffle as written cannot fail: pairing pre from A with post from B
inflates SD(dY_lab) (0.95 -> 1.35 mEq/L for potassium) so the contrast reverts to
the cross-sectional task, and its correlation reproduces the cross-sectional r
for any model with cross-sectional skill. It therefore measures the
between-subject CEILING, not whether the model reads delta physiology.

This version resamples cross-subject pseudo-pairs so their dY_lab distribution
matches the real within-subject one decile-for-decile, putting both contrasts on
the same footing.
"""
import argparse, json
import numpy as np, pandas as pd
from scipy import stats
from paths import INTERIM, OUT

ap = argparse.ArgumentParser(); ap.add_argument("analyte"); ap.add_argument("--reps", type=int, default=200)
A = ap.parse_args(); a = A.analyte
d = pd.read_parquet(INTERIM / f"delta_scored_{a}.parquet")
sid = d.subject_id.values
n = len(d)

edges = np.quantile(d.dy_lab, np.linspace(0, 1, 21))
edges[0], edges[-1] = -np.inf, np.inf
target = np.histogram(d.dy_lab, bins=edges)[0]

rng = np.random.default_rng(7)
raw_r, matched_r = [], []
for _ in range(A.reps):
    # large pool of cross-subject pseudo-pairs
    i = rng.integers(0, n, n * 6); j = rng.integers(0, n, n * 6)
    ok = sid[i] != sid[j]
    i, j = i[ok], j[ok]
    dl = d.y_post.values[j] - d.y_pre.values[i]
    de = d.yhat_post.values[j] - d.yhat_pre.values[i]
    raw_r.append(stats.pearsonr(dl, de)[0])
    # resample the pool to the real dY_lab histogram
    b = np.digitize(dl, edges) - 1
    sel = []
    for k, want in enumerate(target):
        cand = np.flatnonzero(b == k)
        if len(cand) and want:
            sel.append(rng.choice(cand, size=want, replace=len(cand) < want))
    sel = np.concatenate(sel)
    matched_r.append(stats.pearsonr(dl[sel], de[sel])[0])

real_r = float(stats.pearsonr(d.dy_lab, d.dy_ecg)[0])
out = {"analyte": a, "n_pairs": n,
       "within_subject_r": real_r,
       "cross_subject_raw_r": [float(np.mean(raw_r)),
                               float(np.percentile(raw_r, 2.5)), float(np.percentile(raw_r, 97.5))],
       "cross_subject_distribution_matched_r": [float(np.mean(matched_r)),
                               float(np.percentile(matched_r, 2.5)), float(np.percentile(matched_r, 97.5))],
       "sd_dy_lab_within": float(d.dy_lab.std(ddof=1))}
(OUT / f"wp6_matched_shuffle_{a}.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))

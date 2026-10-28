"""WP6 - ablations and falsification (S3, S5).

  python wp6_01_ablations.py potassium
"""
import argparse, json
import numpy as np, pandas as pd, duckdb
from scipy import stats
from paths import INTERIM, OUT

ap = argparse.ArgumentParser(); ap.add_argument("analyte")
A = ap.parse_args(); a = A.analyte
SER = {"potassium": "k_chem", "creatinine": "creat_chem",
       "hemoglobin": "hb_heme", "calcium_total": "ca_tot_chem",
       "calcium_corrected": "ca_tot_chem",
       "ecgfounder": "k_chem", "ecgfounder_linprob": "k_chem"}

d = pd.read_parquet(INTERIM / f"delta_scored_{a}.parquet")
con = duckdb.connect(str(INTERIM / "wp0.duckdb"))
con.execute("pragma threads=16")
con.register("d", d)
res = {"analyte": a, "n_pairs": len(d), "n_patients": int(d.subject_id.nunique())}

def rmet(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    return {"pearson": float(stats.pearsonr(x, y)[0]),
            "mae": float(np.abs(x - y).mean())}

# ---- 1/2. patient-mean and population-mean baselines (S3) -----------------
# the patient's own historical mean, from draws strictly BEFORE the pre ECG
pm = con.execute(f"""
select d.pre_study, d.post_study,
       (select avg(valuenum) from labs l
        where l.subject_id = d.subject_id and l.series = '{SER[a]}'
          and l.charttime < d.pre_ecg) as patient_mean
from d
""").df()
d = d.merge(pm, on=["pre_study", "post_study"], how="left")
pop_mean = float(pd.read_parquet(INTERIM / f"train_{a}.parquet").y.mean())

res["delta_task"] = {
    "model": rmet(d.dy_ecg, d.dy_lab),
    # a predictor emitting a per-patient constant necessarily gives delta = 0
    "patient_mean_baseline":    {"pearson": None, "mae": float(np.abs(d.dy_lab).mean())},
    "population_mean_baseline": {"pearson": None, "mae": float(np.abs(d.dy_lab).mean())},
}
ok = d.patient_mean.notna()
res["absolute_task_post_ecg"] = {
    "n": int(ok.sum()),
    "model":        rmet(d.yhat_post[ok], d.y_post[ok]),
    "patient_mean": rmet(d.patient_mean[ok], d.y_post[ok]),
    "population_mean": {"pearson": None,
                        "mae": float(np.abs(d.y_post[ok] - pop_mean).mean())},
}

# ---- 3. cross-subject delta shuffle ---------------------------------------
rng = np.random.default_rng(5)
sid = d.subject_id.values
perm = rng.permutation(len(d))
for _ in range(50):
    bad = sid == sid[perm]
    if not bad.any():
        break
    perm[bad] = rng.permutation(len(d))[: bad.sum()]
keep = sid != sid[perm]
res["cross_subject_shuffle"] = {
    "n": int(keep.sum()),
    **rmet(d.yhat_post.values[perm][keep] - d.yhat_pre.values[keep],
           d.y_post.values[perm][keep] - d.y_pre.values[keep])}

# magnitude-matched variant: partner drawn from the same delta decile
d["_bin"] = pd.qcut(d.dy_lab, 10, labels=False, duplicates="drop")
p2 = np.arange(len(d))
for b, g in d.groupby("_bin"):
    idx = g.index.values
    p2[idx] = rng.permutation(idx)
keep2 = sid != sid[p2]
res["cross_subject_shuffle_magnitude_matched"] = {
    "n": int(keep2.sum()),
    **rmet(d.yhat_post.values[p2][keep2] - d.yhat_pre.values[keep2],
           d.y_post.values[p2][keep2] - d.y_pre.values[keep2])}

# ---- 4. does this analyte's delta-ECG track the OTHER analytes' deltas? ----
# H5: if delta-ECG predicts delta-creatinine as well as delta-K, the model is
# detecting the clinical event, not the electrolyte.
cross_an = {}
for other, oser in SER.items():
    o = con.execute(f"""
    select d.pre_study, d.post_study,
      (select valuenum from labs l where l.subject_id = d.subject_id and l.series='{oser}'
         and abs(date_diff('minute', l.charttime, d.pre_ecg)) <= 60
       order by abs(date_diff('second', l.charttime, d.pre_ecg)) limit 1) as o_pre,
      (select valuenum from labs l where l.subject_id = d.subject_id and l.series='{oser}'
         and abs(date_diff('minute', l.charttime, d.post_ecg)) <= 60
       order by abs(date_diff('second', l.charttime, d.post_ecg)) limit 1) as o_post
    from d
    """).df()
    o = o.merge(d[["pre_study", "post_study", "dy_ecg"]], on=["pre_study", "post_study"])
    o["dy_other"] = o.o_post - o.o_pre
    m = o.dy_other.notna() & o.dy_ecg.notna()
    if m.sum() >= 100:
        cross_an[other] = {"n": int(m.sum()), **rmet(o.dy_ecg[m], o.dy_other[m])}
res["delta_ecg_vs_other_analyte_deltas"] = cross_an

# ---- 5. HD-exposed vs non-dialysis pairs ----------------------------------
hd = d[(d.hd_icu | d.hd_icd) & ~d.crrt]
non = d[~(d.hd_icu | d.hd_icd) & ~d.crrt]
res["hd_vs_nonhd"] = {
    "hd":     {"n": len(hd),  **(rmet(hd.dy_ecg, hd.dy_lab)  if len(hd)  >= 50 else {})},
    "non_hd": {"n": len(non), **(rmet(non.dy_ecg, non.dy_lab) if len(non) >= 50 else {})}}

(OUT / f"wp6_{a}.json").write_text(json.dumps(res, indent=2, default=float))
print(json.dumps(res, indent=2, default=float))

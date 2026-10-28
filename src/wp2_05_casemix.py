"""Does case mix explain the AUROC gap against published cohorts?

Published hyperkalaemia AUROCs are largely produced in ESRD/dialysis-enriched
cohorts. This scores the SAME model on progressively more enriched strata of our
own test set, so the case-mix argument is demonstrated rather than asserted.
"""
import duckdb, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from paths import INTERIM, OUT

con = duckdb.connect(str(INTERIM / "wp0.duckdb")); con.execute("pragma threads=16")
sp = pd.read_parquet(INTERIM / "split_potassium.parquet")
pr = pd.read_parquet(INTERIM / "pred_potassium.parquet")[["study_id", "pred_potassium"]]
te = sp[sp.split == "test"].merge(pr, on="study_id")
con.register("te", te[["subject_id", "study_id"]])
fl = con.execute("""
 select te.study_id, coalesce(f.esrd,0) esrd, coalesce(f.ckd,0) ckd,
   case when exists(select 1 from hd_icd h where h.subject_id=te.subject_id) then 1 else 0 end hd
 from te left join flags f on f.subject_id = te.subject_id""").df()
te = te.merge(fl, on="study_id")

def auroc(d, thr=5.5):
    m = d.y >= thr
    if m.sum() < 20 or (~m).sum() < 20:
        return np.nan, int(m.sum())
    return roc_auc_score(m, d.pred_potassium), int(m.sum())

rows = []
strata = [("Full test set", te),
          ("No CKD diagnosis", te[te.ckd == 0]),
          ("CKD (any stage)", te[te.ckd == 1]),
          ("ESRD (N18.6)", te[te.esrd == 1]),
          ("Any HD procedure code", te[te.hd == 1])]
for name, d in strata:
    if not len(d):
        continue
    a55, n55 = auroc(d, 5.5); a60, n60 = auroc(d, 6.0)
    rows.append({"stratum": name, "n_ecg": len(d), "n_patients": d.subject_id.nunique(),
                 "K_SD": round(float(d.y.std()), 3),
                 "pct_K_ge_5.5": round(100 * float((d.y >= 5.5).mean()), 2),
                 "AUROC_K_ge_5.5": round(a55, 3) if np.isfinite(a55) else None, "n_pos_5.5": n55,
                 "AUROC_K_ge_6.0": round(a60, 3) if np.isfinite(a60) else None, "n_pos_6.0": n60,
                 "MAE": round(float(np.abs(d.pred_potassium - d.y).mean()), 4)})
t = pd.DataFrame(rows)
t.to_csv(OUT / "table2c_casemix_auroc.csv", index=False)
pd.set_option("display.width", 250)
print(t.to_string(index=False))

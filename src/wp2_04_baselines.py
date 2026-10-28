"""Model vs trivial baselines on the IDENTICAL held-out test set, per analyte.

The patient-mean figures reported for WP6 are computed on the audit cohort, so
they cannot be placed beside test-set MAEs. This recomputes all three predictors
on the same test split:
  constant      - the training-pool mean
  patient-mean  - that patient's own mean from draws strictly before the ECG
  model         - the trained network
"""
import json
import numpy as np, pandas as pd, duckdb
from paths import INTERIM, OUT

SER = {"potassium": "k_chem", "creatinine": "creat_chem", "hemoglobin": "hb_heme",
       "calcium_total": "ca_tot_chem", "calcium_corrected": "ca_tot_chem",
       "ecgfounder": "k_chem", "ecgfounder_linprob": "k_chem"}
con = duckdb.connect(str(INTERIM / "wp0.duckdb")); con.execute("pragma threads=16")

rows = []
for a, ser in SER.items():
    sp_p = INTERIM / f"split_{a}.parquet"
    if not sp_p.exists():
        continue
    sp = pd.read_parquet(sp_p)
    pr = pd.read_parquet(INTERIM / f"pred_{a}.parquet")[["study_id", f"pred_{a}"]]
    te = sp[sp.split == "test"].merge(pr, on="study_id")
    const = float(sp[sp.split == "train"].y.mean())
    con.register("te", te[["subject_id", "study_id", "ecg_time"]])
    pm = con.execute(f"""
      select te.study_id,
             (select avg(valuenum) from labs l
              where l.subject_id = te.subject_id and l.series = '{ser}'
                and l.charttime < te.ecg_time) as patient_mean
      from te""").df()
    te = te.merge(pm, on="study_id")
    ok = te.patient_mean.notna()
    rows.append({
        "analyte": a, "n_test": len(te), "n_with_history": int(ok.sum()),
        "model_MAE": float(np.abs(te[f"pred_{a}"] - te.y).mean()),
        "constant_MAE": float(np.abs(te.y - const).mean()),
        "patient_mean_MAE": float(np.abs(te.patient_mean[ok] - te.y[ok]).mean()),
        # model restricted to the same rows the patient-mean baseline can score
        "model_MAE_on_history_subset": float(np.abs(te[f"pred_{a}"][ok] - te.y[ok]).mean()),
        "constant_MAE_on_history_subset": float(np.abs(te.y[ok] - const).mean()),
        "test_SD": float(te.y.std()),
    })
    print(f"{a:18s} model={rows[-1]['model_MAE']:.4f}  constant={rows[-1]['constant_MAE']:.4f}  "
          f"patient-mean={rows[-1]['patient_mean_MAE']:.4f}  (n_hist={int(ok.sum()):,}/{len(te):,})")

df = pd.DataFrame(rows)
df.to_csv(OUT / "table2b_baselines_same_testset.csv", index=False)

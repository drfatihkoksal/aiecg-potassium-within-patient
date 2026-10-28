"""Table 1 - cohort characteristics, and the potassium case-mix that the
AUROC comparison against published cohorts turns on."""
import duckdb, numpy as np, pandas as pd
from paths import HOSP, ICU, INTERIM, OUT

con = duckdb.connect(str(INTERIM / "wp0.duckdb")); con.execute("pragma threads=16")
con.execute(f"""create or replace table dx as
  select distinct subject_id, icd_code, icd_version
  from read_csv_auto('{HOSP}/diagnoses_icd.csv.gz')""")
con.execute(f"""create or replace table icustay as
  select distinct subject_id from read_csv_auto('{ICU}/icustays.csv.gz')""")
con.execute("""create or replace table flags as
 select distinct subject_id,
   max(case when (icd_version=10 and icd_code like 'N186%')
              or (icd_version=9 and icd_code='5856') then 1 else 0 end) esrd,
   max(case when (icd_version=10 and icd_code like 'N18%')
              or (icd_version=9 and icd_code like '585%') then 1 else 0 end) ckd
 from dx group by subject_id""")

def describe(tbl, name, ycol="y"):
    d = con.execute(f"""
    select t.subject_id, t.{ycol} as y, t.age, t.gender,
           coalesce(f.esrd,0) esrd, coalesce(f.ckd,0) ckd,
           case when exists(select 1 from icustay i where i.subject_id=t.subject_id) then 1 else 0 end icu,
           case when exists(select 1 from hd_icd h where h.subject_id=t.subject_id) then 1 else 0 end hd
    from {tbl} t left join flags f using (subject_id)""").df()
    q = np.percentile(d.y, [1, 5, 25, 50, 75, 95, 99])
    return {"cohort": name, "n_ecg": len(d), "n_patients": d.subject_id.nunique(),
            "age_median": round(float(d.age.median()), 1),
            "female_pct": round(100 * float((d.gender == "F").mean()), 1),
            "K_mean": round(float(d.y.mean()), 3), "K_SD": round(float(d.y.std()), 3),
            "K_p1": q[0], "K_p5": q[1], "K_p25": q[2], "K_median": q[3],
            "K_p75": q[4], "K_p95": q[5], "K_p99": q[6],
            "pct_K_ge_5.5": round(100 * float((d.y >= 5.5).mean()), 2),
            "pct_K_ge_6.0": round(100 * float((d.y >= 6.0).mean()), 2),
            "pct_K_le_3.5": round(100 * float((d.y <= 3.5).mean()), 2),
            "pct_ESRD": round(100 * float(d.groupby("subject_id").esrd.first().mean()), 1),
            "pct_CKD": round(100 * float(d.groupby("subject_id").ckd.first().mean()), 1),
            "pct_any_HD_code": round(100 * float(d.groupby("subject_id").hd.first().mean()), 1),
            "pct_ever_ICU": round(100 * float(d.groupby("subject_id").icu.first().mean()), 1)}

rows = [describe("obs_potassium", "All ECG–K observations"),
        describe("train_potassium", "Training pool (cross-sectional)")]
# audit cohort: describe at the pre ECG
con.execute("""create or replace table audit_obs as
  select subject_id, y_pre as y, 0 as age, '' as gender from audit_potassium""")
d = con.execute("""
 select a.subject_id, a.y_pre y, o.age, o.gender from audit_potassium a
 join obs_potassium o on o.study_id = a.pre_study""").df()
con.register("audit_pre", d)
rows.append(describe("audit_pre", "Audit cohort (Δ pairs, at pre-ECG)"))

t1 = pd.DataFrame(rows)
t1.to_csv(OUT / "table1_cohort.csv", index=False)
pd.set_option("display.width", 250)
print(t1[["cohort", "n_ecg", "n_patients", "age_median", "female_pct", "K_mean", "K_SD",
          "K_p95", "K_p99", "pct_K_ge_5.5", "pct_K_ge_6.0", "pct_K_le_3.5",
          "pct_ESRD", "pct_CKD", "pct_any_HD_code", "pct_ever_ICU"]].to_string(index=False))

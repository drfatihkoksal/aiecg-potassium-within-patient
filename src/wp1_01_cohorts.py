"""WP1 - cohort build.

Emits the audit cohorts and the WP2 training pool with a STROBE-style funnel.
Patient-level disjointness between the training pool and every audit cohort is a
hard constraint (concept Section 5/WP1) and is asserted at the end: the script
fails loudly on violation.
"""
import json
import duckdb, pandas as pd
from paths import ECG, HOSP, INTERIM, OUT

LAB_W    = 60    # min, ECG <-> lab
DT_MAX   = 24    # h, primary delta window
NULL_TOL = 0.15  # mEq/L, null-change stratum
K_LO, K_HI = 1.5, 9.5

# ventricular pacing masks repolarisation and is excluded (Section 5.4).
# Atrial-only pacing does not and is retained.
VPACE = r'(ventricular pac|v-paced|pacemaker rhythm|a-v sequential|dual chamber pacemaker|demand pacing|because of pacemaker)'
APACE = r'(atrial pac|atrial-paced)'
# acute cardiac events that invalidate a pair (Section 5.4)
ACUTE = {9: ("9960", "9961", "9962"), 10: ("5A2204Z", "5A12012")}

con = duckdb.connect(str(INTERIM / "wp0.duckdb"))
con.execute("pragma threads=16")
R = ", ".join(f"report_{i}" for i in range(18))
funnel = []

def step(label, n_obs, n_pat):
    funnel.append({"step": label, "ecg_obs": n_obs, "patients": n_pat})
    print(f"{label:<58s} {n_obs:>9,}  {n_pat:>7,}")

# ------------------------------------------------------------------ ECG meta
con.execute(f"""
create or replace table ecg_meta as
select subject_id, study_id,
       lower(concat_ws(' | ', {R})) as report,
       rr_interval, qrs_onset, qrs_end, t_end, p_axis, qrs_axis, t_axis
from read_csv_auto('{ECG}/machine_measurements.csv', ignore_errors=true)
""")
con.execute(f"""
create or replace table ecg_flag as
select subject_id, study_id, report, rr_interval,
       (qrs_end - qrs_onset) as qrs_dur, (t_end - qrs_end) as jt_int,
       regexp_matches(report, '{VPACE}') as vpaced,
       (regexp_matches(report, '{APACE}')
        and not regexp_matches(report, '{VPACE}')) as apaced_only,
       report like '%atrial fibrillation%' as afib,
       (report like '%pacer detection suspended%' or report like '%artifact%'
        or report like '%baseline wander%' or report like '%technically%'
        or report like '%no further analysis%') as report_quality_flag
from ecg_meta
""")

# ------------------------------------------------------------- demographics
con.execute(f"""
create or replace table pat as
select subject_id, gender, anchor_age, anchor_year, dod
from read_csv_auto('{HOSP}/patients.csv.gz')
""")

# ------------------------------------------------- acute cardiac event dates
con.execute(f"""
create or replace table acute as
select distinct subject_id, chartdate::date as ev_date
from read_csv_auto('{HOSP}/procedures_icd.csv.gz')
where (icd_version = 9  and icd_code in {ACUTE[9]})
   or (icd_version = 10 and icd_code in {ACUTE[10]})
""")

# ------------------------------------------- base ECG-K observation table
con.execute(f"""
create or replace table obs0 as
select e.subject_id, e.study_id, e.ecg_time, e.path,
       k.k, k.charttime as k_time, k.hemolysed,
       abs(date_diff('minute', k.charttime, e.ecg_time)) as gap_min
from ecg e
join lateral (
  select k, charttime, hemolysed from k_lab
  where k_lab.subject_id = e.subject_id and k_lab.series = 'k_chem'
    and valuenum_ok and abs(date_diff('minute', charttime, e.ecg_time)) <= {LAB_W}
  order by abs(date_diff('second', charttime, e.ecg_time)) limit 1) k on true
""".replace("valuenum_ok", f"k between {K_LO} and {K_HI}"))

step("ECGs in MIMIC-IV-ECG v1.0",
     *con.execute("select count(*), count(distinct subject_id) from ecg").fetchone())
step(f"  with serum K within +-{LAB_W} min",
     *con.execute("select count(*), count(distinct subject_id) from obs0").fetchone())

con.execute("""
create or replace table obs1 as
select o.*, p.gender,
       p.anchor_age + (year(o.ecg_time) - p.anchor_year) as age,
       f.vpaced, f.apaced_only, f.afib, f.report_quality_flag,
       f.rr_interval, f.qrs_dur, f.jt_int, f.report
from obs0 o join pat p using (subject_id)
     left join ecg_flag f using (subject_id, study_id)
""")
con.execute("create or replace table obs2 as select * from obs1 where age >= 18")
step("  age >= 18",
     *con.execute("select count(*), count(distinct subject_id) from obs2").fetchone())
con.execute("create or replace table obs3 as select * from obs2 where not coalesce(vpaced, false)")
step("  no ventricular pacing on the ECG",
     *con.execute("select count(*), count(distinct subject_id) from obs3").fetchone())
con.execute("""create or replace table obs as
              select * from obs3 where report is not null""")
step("  machine overread present (QC prerequisite)",
     *con.execute("select count(*), count(distinct subject_id) from obs").fetchone())

# haemolysed samples are held out as a sensitivity stratum, not discarded (5.5)
con.execute("create or replace table obs_clean as select * from obs where not hemolysed")
step("  non-haemolysed (haemolysed kept as sensitivity stratum)",
     *con.execute("select count(*), count(distinct subject_id) from obs_clean").fetchone())

# ------------------------------------------------------- delta pair cohorts
con.execute(f"""
create or replace table dp as
select subject_id, gender, age,
       study_id as pre_study, ecg_time as pre_ecg, k as k_pre, k_time as k_pre_time,
       path as pre_path, afib as pre_afib, report_quality_flag as pre_qflag,
       lead(study_id) over w as post_study, lead(ecg_time) over w as post_ecg,
       lead(k) over w as k_post, lead(k_time) over w as k_post_time,
       lead(path) over w as post_path, lead(afib) over w as post_afib,
       lead(report_quality_flag) over w as post_qflag
from obs_clean window w as (partition by subject_id order by ecg_time)
qualify post_study is not null
""")
con.execute(f"""
create or replace table dpf as
select *, k_post - k_pre as dk_lab,
       date_diff('minute', pre_ecg, post_ecg) / 60.0 as dt_h,
       exists (select 1 from hd_ok s where s.subject_id = dp.subject_id
               and s.hd_start >= dp.pre_ecg and s.hd_end <= dp.post_ecg) as hd_icu,
       exists (select 1 from hd_icd h where h.subject_id = dp.subject_id
               and h.hd_date between dp.pre_ecg::date and dp.post_ecg::date) as hd_icd,
       exists (select 1 from crrt c where c.subject_id = dp.subject_id
               and c.c_start <= dp.post_ecg and c.c_end >= dp.pre_ecg) as crrt,
       exists (select 1 from acute a where a.subject_id = dp.subject_id
               and a.ev_date between dp.pre_ecg::date and dp.post_ecg::date) as acute_event
from dp
""")
n0 = con.execute("select count(*) from dpf").fetchone()
step("Consecutive ECG-K pairs (all dt)", n0[0],
     con.execute("select count(distinct subject_id) from dpf").fetchone()[0])

con.execute("create or replace table dpg as select * from dpf where k_pre_time <> k_post_time")
step("  distinct lab draws (shared-K artefact removed)",
     *con.execute("select count(*), count(distinct subject_id) from dpg").fetchone())
con.execute("create or replace table dph as select * from dpg where not acute_event")
step("  no CPR / cardioversion between the ECGs",
     *con.execute("select count(*), count(distinct subject_id) from dph").fetchone())
con.execute(f"create or replace table cohort_delta as select * from dph where dt_h <= {DT_MAX}")
step(f"  dt <= {DT_MAX} h  -> COHORT_DELTA (primary)",
     *con.execute("select count(*), count(distinct subject_id) from cohort_delta").fetchone())

con.execute(f"""
create or replace table cohort_null_change as
select * from cohort_delta
where abs(dk_lab) < {NULL_TOL} and not hd_icu and not hd_icd and not crrt
""")
step(f"  |dK| < {NULL_TOL}, no dialysis -> COHORT_NULL_CHANGE",
     *con.execute("select count(*), count(distinct subject_id) from cohort_null_change").fetchone())

con.execute("""
create or replace table cohort_hd_pairs as
select * from cohort_delta where (hd_icu or hd_icd) and not crrt
""")
step("  HD between the ECGs -> COHORT_HD_PAIRS (subgroup)",
     *con.execute("select count(*), count(distinct subject_id) from cohort_hd_pairs").fetchone())

# ---------------------------------- cross-sectional training pool (disjoint)
con.execute("""
create or replace table audit_patients as
select distinct subject_id from cohort_delta
""")
con.execute("""
create or replace table cohort_crosssectional as
select * from obs_clean
where subject_id not in (select subject_id from audit_patients)
""")
step("Cross-sectional pool, audit patients removed -> COHORT_CROSSSECTIONAL",
     *con.execute("select count(*), count(distinct subject_id) from cohort_crosssectional").fetchone())

# ------------------------------------------------ hard disjointness assertion
leak = con.execute("""
select count(*) from cohort_crosssectional
where subject_id in (select subject_id from audit_patients)""").fetchone()[0]
assert leak == 0, f"PATIENT LEAKAGE: {leak} cross-sectional rows belong to audit patients"
for c in ("cohort_delta", "cohort_null_change", "cohort_hd_pairs"):
    l = con.execute(f"""select count(*) from {c}
        where subject_id in (select subject_id from cohort_crosssectional)""").fetchone()[0]
    assert l == 0, f"PATIENT LEAKAGE: {c} shares {l} rows with the training pool"
print("\npatient-level disjointness asserted: training pool n training/audit overlap = 0")

for t in ("cohort_delta", "cohort_null_change", "cohort_hd_pairs", "cohort_crosssectional"):
    con.execute(f"copy {t} to '{INTERIM}/{t}.parquet' (format parquet)")
pd.DataFrame(funnel).to_csv(OUT / "wp1_strobe_funnel.csv", index=False)
print(f"\nwrote 4 cohort parquets + outputs/wp1_strobe_funnel.csv")

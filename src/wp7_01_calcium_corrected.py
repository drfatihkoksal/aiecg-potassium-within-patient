"""Albumin-corrected total calcium (concept Section 5.6: 'albumin-correct total only').

corrected Ca = measured Ca + 0.8 * (4.0 - albumin[g/dL])

Albumin moves slowly relative to the ECG-lab pairing window, so the nearest
albumin within +-24 h of the ECG is used; pairs without albumin at BOTH time
points are dropped rather than imputed.
"""
import duckdb, pandas as pd
from paths import INTERIM, OUT

LAB_W, ALB_W, DT_MAX, TOL = 60, 1440, 24, 0.15
con = duckdb.connect(str(INTERIM / "wp0.duckdb"))
con.execute("pragma threads=16")

con.execute(f"""
create or replace table obs_calcium_corrected as
select o.subject_id, o.study_id, o.ecg_time, o.path, o.age, o.gender,
       c.valuenum + 0.8 * (4.0 - alb.valuenum) as y,
       c.charttime as y_time, c.valuenum as ca_raw, alb.valuenum as albumin
from obs o join wqc q using (study_id)
join lateral (
  select valuenum, charttime from labs
  where labs.subject_id = o.subject_id and labs.series = 'ca_tot_chem'
    and valuenum between 4 and 16
    and abs(date_diff('minute', charttime, o.ecg_time)) <= {LAB_W}
  order by abs(date_diff('second', charttime, o.ecg_time)) limit 1) c on true
join lateral (
  select valuenum, charttime from labs
  where labs.subject_id = o.subject_id and labs.series = 'alb_chem'
    and valuenum between 0.5 and 7
    and abs(date_diff('minute', charttime, o.ecg_time)) <= {ALB_W}
  order by abs(date_diff('second', charttime, o.ecg_time)) limit 1) alb on true
where q.qc_pass
""")

con.execute(f"""
create or replace table delta_calcium_corrected as
with p as (
  select subject_id, study_id as pre_study, ecg_time as pre_ecg, y as y_pre,
         y_time as t_pre, path as pre_path,
         lead(study_id) over w as post_study, lead(ecg_time) over w as post_ecg,
         lead(y) over w as y_post, lead(y_time) over w as t_post,
         lead(path) over w as post_path
  from obs_calcium_corrected window w as (partition by subject_id order by ecg_time)
  qualify post_study is not null)
select *, y_post - y_pre as dy_lab,
       date_diff('minute', pre_ecg, post_ecg) / 60.0 as dt_h,
       exists (select 1 from hd_ok h where h.subject_id = p.subject_id
               and h.hd_start >= p.pre_ecg and h.hd_end <= p.post_ecg) as hd_icu,
       exists (select 1 from hd_icd h where h.subject_id = p.subject_id
               and h.hd_date between p.pre_ecg::date and p.post_ecg::date) as hd_icd,
       exists (select 1 from crrt c where c.subject_id = p.subject_id
               and c.c_start <= p.post_ecg and c.c_end >= p.pre_ecg) as crrt,
       exists (select 1 from acute e where e.subject_id = p.subject_id
               and e.ev_date between p.pre_ecg::date and p.post_ecg::date) as acute_event
from p where t_pre <> t_post
""")
con.execute(f"""create or replace table audit_calcium_corrected as
  select * from delta_calcium_corrected where dt_h <= {DT_MAX} and not acute_event""")
con.execute(f"""create or replace table null_calcium_corrected as
  select * from audit_calcium_corrected
  where abs(dy_lab) < {TOL} and not hd_icu and not hd_icd and not crrt""")

# training pool must still exclude every audit patient of every analyte
con.execute("""
create or replace table audit_patients_all2 as
select distinct subject_id from audit_patients_all
union select distinct subject_id from audit_calcium_corrected
""")
con.execute("""
create or replace table train_calcium_corrected as
select * from obs_calcium_corrected
where subject_id not in (select subject_id from audit_patients_all2)
""")
leak = con.execute("""select count(*) from train_calcium_corrected
  where subject_id in (select subject_id from audit_patients_all2)""").fetchone()[0]
assert leak == 0, f"PATIENT LEAKAGE: {leak}"

for t in ("train_calcium_corrected", "audit_calcium_corrected", "null_calcium_corrected"):
    con.execute(f"copy {t} to '{INTERIM}/{t.replace('train_','train_').replace('audit_','audit_').replace('null_','null_')}.parquet' (format parquet)")

for t in ("obs_calcium_corrected", "audit_calcium_corrected",
          "null_calcium_corrected", "train_calcium_corrected"):
    n, p = con.execute(f"select count(*), count(distinct subject_id) from {t}").fetchone()
    print(f"{t:26s} {n:>8,} / {p:>7,} patients")
print("\ncorrection effect:")
print(con.execute("""select round(avg(ca_raw),3) mean_raw, round(avg(y),3) mean_corrected,
  round(corr(ca_raw, y),3) r_raw_vs_corrected, round(avg(albumin),3) mean_albumin
  from obs_calcium_corrected""").df().to_string(index=False))

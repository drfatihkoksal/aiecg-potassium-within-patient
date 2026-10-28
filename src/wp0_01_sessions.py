"""Build the HD-session table (Section 5.1) and the ECG/K index used by WP0.

Timestamped, discrete IHD sessions come from icu.procedureevents (itemid 225441).
CRRT/SCUF/PD events are kept only as an EXCLUSION mask (no discrete delta event).
Date-level hosp.procedures_icd HD codes are kept separately: they identify HD
patients/days but carry no session timing, so they cannot support the +-6 h ECG
window on their own.
"""
import duckdb
from paths import ECG, HOSP, ICU, INTERIM

IHD  = (225441,)                                   # Hemodialysis (intermittent)
CONT = (225802, 225803, 225809, 225955, 225805)    # CRRT, CVVHD, CVVHDF, SCUF, PD
ICD_HD = {9: ("3995",),
          10: ("5A1D00Z", "5A1D60Z", "5A1D70Z", "5A1D80Z", "5A1D90Z")}

con = duckdb.connect(str(INTERIM / "wp0.duckdb"))
con.execute("pragma threads=16")

# ---------------------------------------------------------------- ECG index
con.execute(f"""
create or replace table ecg as
select subject_id, study_id, ecg_time::timestamp as ecg_time, path
from read_csv_auto('{ECG}/record_list.csv')
""")

# ------------------------------------------------------- raw dialysis events
con.execute(f"""
create or replace table pe_raw as
select subject_id, hadm_id, stay_id, itemid, linkorderid,
       starttime::timestamp as starttime, endtime::timestamp as endtime,
       value, valueuom, statusdescription
from read_csv_auto('{ICU}/procedureevents.csv.gz')
where itemid in {IHD + CONT}
""")

# IHD sessions: collapse rows sharing an order, then merge runs separated by <1 h
con.execute(f"""
create or replace table ihd_ord as
select subject_id, hadm_id, stay_id,
       min(starttime) as starttime, max(endtime) as endtime
from pe_raw where itemid in {IHD}
group by subject_id, hadm_id, stay_id, linkorderid
""")
con.execute("""
create or replace table ihd_flag as
select *, case when starttime <= lag(endtime) over w + interval 60 minute
               then 0 else 1 end as is_new
from ihd_ord window w as (partition by subject_id order by starttime)
""")
con.execute("""
create or replace table hd_session as
select subject_id, any_value(hadm_id) as hadm_id, any_value(stay_id) as stay_id,
       min(starttime) as hd_start, max(endtime) as hd_end,
       date_diff('minute', min(starttime), max(endtime)) / 60.0 as hd_hours
from (select *, sum(is_new) over (partition by subject_id order by starttime) as grp
      from ihd_flag)
group by subject_id, grp
""")

# continuous-RRT mask (used to veto pairs that straddle a CRRT run)
con.execute(f"""
create or replace table crrt as
select subject_id, min(starttime) as c_start, max(endtime) as c_end
from pe_raw where itemid in {CONT}
group by subject_id, linkorderid
""")

# ------------------------------------------------ date-level HD (ward, no time)
con.execute(f"""
create or replace table hd_icd as
select distinct subject_id, hadm_id, chartdate::date as hd_date
from read_csv_auto('{HOSP}/procedures_icd.csv.gz')
where (icd_version = 9  and icd_code in {ICD_HD[9]})
   or (icd_version = 10 and icd_code in {ICD_HD[10]})
""")

# ---------------------------------------------------------------- lab series
con.execute(f"""
create or replace table labs as
select subject_id, itemid, charttime::timestamp as charttime, valuenum,
       valueuom, flag, comments,
       case when itemid in (50971, 52610)        then 'k_chem'
            when itemid in (50822, 52452)        then 'k_bg'
            when itemid in (50893)               then 'ca_tot_chem'
            when itemid in (52034, 52035)        then 'ca_tot_bg'
            when itemid in (51624)               then 'ca_ion_chem'
            when itemid in (50808)               then 'ca_ion_bg'
            when itemid in (51222)               then 'hb_heme'
            when itemid in (50811)               then 'hb_bg'
            when itemid in (50912, 52546)        then 'creat_chem'
            when itemid in (52024)               then 'creat_bg'
            when itemid in (50862, 53085, 53138) then 'alb_chem'
            when itemid in (50882)               then 'bicarb_chem'
            when itemid in (50820)               then 'ph_bg'
            when itemid in (50954)               then 'ld_chem'
       end as series
from '{INTERIM}/labs_subset.parquet'
where valuenum is not null and charttime is not null
""")

con.execute("""
create or replace table k_lab as
select subject_id, charttime, valuenum as k, series,
       -- Section 5.5. The comments field is dominated by a boilerplate assay
       -- disclaimer ("Hemolysis falsely elevates this test", ~100k rows) that is
       -- preferentially appended to elevated results; matching it would strip the
       -- hyperkalaemic observations this study is about. Only an actual specimen
       -- finding counts, and the explicit negation must be excluded.
       coalesce(lower(comments) like '%hemolyzed%'
                and lower(comments) not like '%not hemolyzed%', false) as hemolysed
from labs
where series in ('k_chem', 'k_bg') and valuenum between 1.5 and 9.5
""")

for t in ("ecg", "hd_session", "crrt", "hd_icd", "k_lab"):
    n, = con.execute(f"select count(*) from {t}").fetchone()
    s, = con.execute(f"select count(distinct subject_id) from {t}").fetchone()
    print(f"{t:12s} rows={n:>10,}  subjects={s:>8,}")

print("\nIHD session duration (h):")
print(con.execute("""select quantile_cont(hd_hours,[.05,.25,.5,.75,.95]) q,
                             round(avg(hd_hours),2) mean from hd_session""").df().to_string())
print("\nECG subjects overlapping IHD sessions:",
      con.execute("""select count(distinct s.subject_id) from hd_session s
                     where exists (select 1 from ecg e where e.subject_id=s.subject_id)""").fetchone()[0])
print("ECG subjects overlapping ICD-HD:",
      con.execute("""select count(distinct h.subject_id) from hd_icd h
                     where exists (select 1 from ecg e where e.subject_id=h.subject_id)""").fetchone()[0])

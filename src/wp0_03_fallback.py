"""WP0 part 2: the fallback cohort the Section 6/WP0 decision rule routes to when
HD-pairs are insufficient -- ALL within-subject paired ECG-K observations, with
HD status as a high-delta subgroup covariate. Also the Section 5.3 null-change
stratum and the WP2 cross-sectional training pool."""
import duckdb, pandas as pd
from paths import INTERIM, OUT

con = duckdb.connect(str(INTERIM / "wp0.duckdb"))
con.execute("pragma threads=16")
pd.set_option("display.width", 250)

# ---------------- ECG-K observations (cross-sectional pool, WP2) --------------
for lw in (30, 60, 120):
    con.execute(f"""
    create or replace table ecg_k_{lw} as
    select e.subject_id, e.study_id, e.ecg_time, k.k, k.charttime as k_time,
           abs(date_diff('minute', k.charttime, e.ecg_time)) as gap_min, k.hemolysed
    from ecg e
    join lateral (
      select k, charttime, hemolysed from k_lab
      where k_lab.subject_id = e.subject_id and k_lab.series = 'k_chem'
        and abs(date_diff('minute', charttime, e.ecg_time)) <= {lw}
      order by abs(date_diff('second', charttime, e.ecg_time)) limit 1) k on true
    """)
    n, s = con.execute(f"select count(*), count(distinct subject_id) from ecg_k_{lw}").fetchone()
    print(f"cross-sectional ECG-K, +-{lw:>3} min: {n:>8,} ECGs / {s:>7,} patients")

# ---------------- within-subject consecutive delta pairs ---------------------
con.execute("""
create or replace table dpair as
select subject_id,
       study_id  as pre_study,  ecg_time  as pre_ecg,  k as k_pre,
       lead(study_id) over w as post_study,
       lead(ecg_time) over w as post_ecg,
       lead(k)        over w as k_post
from ecg_k_60 window w as (partition by subject_id order by ecg_time)
qualify post_study is not null
""")
con.execute("""
create or replace table dpair2 as
select *, k_post - k_pre as dk_lab,
       date_diff('minute', pre_ecg, post_ecg) / 60.0 as dt_h,
       exists (select 1 from hd_ok s where s.subject_id = dpair.subject_id
               and s.hd_start >= dpair.pre_ecg and s.hd_end <= dpair.post_ecg) as hd_icu_between,
       exists (select 1 from hd_icd h where h.subject_id = dpair.subject_id
               and h.hd_date between pre_ecg::date and post_ecg::date) as hd_icd_between,
       exists (select 1 from crrt c where c.subject_id = dpair.subject_id
               and c.c_start <= dpair.post_ecg and c.c_end >= dpair.pre_ecg) as crrt_between
from dpair
""")

print("\n--- within-subject consecutive ECG-K delta pairs (lab +-60 min) ---")
print(con.execute("""
select case when dt_h <= 6 then 'a <=6h' when dt_h <= 12 then 'b 6-12h'
            when dt_h <= 24 then 'c 12-24h' when dt_h <= 72 then 'd 1-3d'
            else 'e >3d' end as dt_bin,
       count(*) n_pairs, count(distinct subject_id) n_pat,
       round(avg(abs(dk_lab)), 3) mean_abs_dk,
       sum(case when abs(dk_lab) >= 0.5 then 1 else 0 end) n_dk_ge_05,
       sum(case when abs(dk_lab) >= 1.0 then 1 else 0 end) n_dk_ge_10,
       sum(case when hd_icu_between or hd_icd_between then 1 else 0 end) n_hd_between
from dpair2 group by 1 order by 1""").df().to_string(index=False))

print("\n--- HD-exposed subgroup (any HD between the two ECGs, dt <= 24 h) ---")
print(con.execute("""
select case when hd_icu_between then 'ICU IHD session' else 'ICD-coded HD day' end src,
       count(*) n_pairs, count(distinct subject_id) n_pat,
       round(avg(dk_lab), 3) mean_dk, round(avg(abs(dk_lab)), 3) mean_abs_dk
from dpair2 where dt_h <= 24 and (hd_icu_between or hd_icd_between) and not crrt_between
group by 1""").df().to_string(index=False))

# ---------------- Section 5.3 null-change control stratum --------------------
print("\n--- null-change control stratum (|dK_lab| < 0.15, dt < 24 h, no dialysis) ---")
print(con.execute("""
select case when dt_h <= 6 then 'a <=6h' when dt_h <= 12 then 'b 6-12h'
            else 'c 12-24h' end dt_bin,
       count(*) n_pairs, count(distinct subject_id) n_pat
from dpair2
where abs(dk_lab) < 0.15 and dt_h <= 24
  and not hd_icu_between and not hd_icd_between and not crrt_between
group by 1 order by 1""").df().to_string(index=False))

con.execute(f"copy dpair2 to '{INTERIM}/wp0_dpairs.parquet' (format parquet)")

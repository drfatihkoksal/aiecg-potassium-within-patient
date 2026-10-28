"""WP0 feasibility gate: HD-pair yield across the Section 5.2 sensitivity grid,
plus the Section 5.3 null-change control stratum."""
import itertools, json
import duckdb, pandas as pd
from paths import INTERIM, OUT

ECG_W  = [3, 6, 12]     # hours around the session
LAB_W  = [30, 60, 120]  # minutes ECG<->lab
K_SERIES = "k_chem"     # serum chemistry potassium; blood-gas kept separate

con = duckdb.connect(str(INTERIM / "wp0.duckdb"))
con.execute("pragma threads=16")

# plausible discrete IHD sessions only (1-8 h); longer runs are continuous therapy
con.execute("""
create or replace table hd_ok as
select * from hd_session where hd_hours between 1 and 8
""")
print("discrete IHD sessions (1-8 h):",
      *con.execute("select count(*), count(distinct subject_id) from hd_ok").fetchone())


def pairs(ecg_w, lab_w, k_series, crrt_veto=True):
    """One grid cell -> the HD-pair table."""
    con.execute(f"""
    create or replace temp table kk as
      select subject_id, charttime, k from k_lab where series = '{k_series}'
    """)
    con.execute(f"""
    create or replace temp table cand as
    with pre as (
      select s.subject_id, s.hd_start, s.hd_end, e.study_id as pre_study,
             e.ecg_time as pre_ecg,
             row_number() over (partition by s.subject_id, s.hd_start
                                order by e.ecg_time desc) as rn
      from hd_ok s join ecg e using (subject_id)
      where e.ecg_time <  s.hd_start
        and e.ecg_time >= s.hd_start - interval {ecg_w} hour
    ), post as (
      select s.subject_id, s.hd_start, e.study_id as post_study,
             e.ecg_time as post_ecg,
             row_number() over (partition by s.subject_id, s.hd_start
                                order by e.ecg_time asc) as rn
      from hd_ok s join ecg e using (subject_id)
      where e.ecg_time >  s.hd_end
        and e.ecg_time <= s.hd_end + interval {ecg_w} hour
    )
    select pre.subject_id, pre.hd_start, pre.hd_end,
           pre.pre_study, pre.pre_ecg, post.post_study, post.post_ecg
    from pre join post
      on pre.subject_id = post.subject_id and pre.hd_start = post.hd_start
    where pre.rn = 1 and post.rn = 1
    """)
    con.execute(f"""
    create or replace temp table hdp as
    select c.*, kpre.k as k_pre, kpre.charttime as k_pre_time,
                kpost.k as k_post, kpost.charttime as k_post_time,
           kpost.k - kpre.k as dk_lab
    from cand c
    left join lateral (
      select k, charttime from kk
      where kk.subject_id = c.subject_id
        and abs(date_diff('minute', charttime, c.pre_ecg)) <= {lab_w}
      order by abs(date_diff('second', charttime, c.pre_ecg)) limit 1) kpre on true
    left join lateral (
      select k, charttime from kk
      where kk.subject_id = c.subject_id
        and abs(date_diff('minute', charttime, c.post_ecg)) <= {lab_w}
      order by abs(date_diff('second', charttime, c.post_ecg)) limit 1) kpost on true
    where kpre.k is not null and kpost.k is not null
    """)
    if crrt_veto:
        con.execute("""
        delete from hdp where exists (
          select 1 from crrt c
          where c.subject_id = hdp.subject_id
            and c.c_start <= hdp.post_ecg and c.c_end >= hdp.pre_ecg)
        """)
    return con.execute("""
      select count(*) n_pairs, count(distinct subject_id) n_subjects,
             avg(dk_lab) mean_dk, avg(abs(dk_lab)) mean_abs_dk,
             stddev(dk_lab) sd_dk
      from hdp""").df().iloc[0]


rows = []
for ew, lw in itertools.product(ECG_W, LAB_W):
    r = pairs(ew, lw, K_SERIES)
    rows.append({"ecg_window_h": ew, "lab_window_min": lw, **r.to_dict()})
    print(f"ECG +-{ew:>2}h  lab +-{lw:>3}min -> "
          f"{int(r.n_pairs):>5} pairs / {int(r.n_subjects):>4} patients  "
          f"mean|dK|={r.mean_abs_dk:.2f}" if r.n_pairs else
          f"ECG +-{ew:>2}h  lab +-{lw:>3}min -> 0 pairs")

grid = pd.DataFrame(rows)
grid.to_csv(OUT / "wp0_grid_hd_pairs.csv", index=False)

# blood-gas potassium as an alternative (much denser sampling in the ICU)
rows_bg = []
for ew, lw in itertools.product(ECG_W, LAB_W):
    r = pairs(ew, lw, "k_bg")
    rows_bg.append({"ecg_window_h": ew, "lab_window_min": lw, **r.to_dict()})
pd.DataFrame(rows_bg).to_csv(OUT / "wp0_grid_hd_pairs_kbg.csv", index=False)
print("\nblood-gas K grid:")
print(pd.DataFrame(rows_bg).to_string(index=False))

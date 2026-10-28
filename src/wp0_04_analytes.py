"""WP0 part 3: within-subject delta-pair yield for the S4/S5 analytes."""
import duckdb, pandas as pd
from paths import INTERIM, OUT

SERIES = {"potassium (chem)": "k_chem", "potassium (blood gas)": "k_bg",
          "calcium, total (chem)": "ca_tot_chem", "calcium, ionised (blood gas)": "ca_ion_bg",
          "haemoglobin (haem)": "hb_heme", "haemoglobin (blood gas)": "hb_bg",
          "creatinine (chem)": "creat_chem", "albumin (chem)": "alb_chem"}
RANGE = {"k_chem": (1.5, 9.5), "k_bg": (1.5, 9.5), "ca_tot_chem": (4, 16),
         "ca_ion_bg": (0.4, 2.2), "hb_heme": (3, 22), "hb_bg": (3, 22),
         "creat_chem": (0.1, 25), "alb_chem": (0.5, 7)}

con = duckdb.connect(str(INTERIM / "wp0.duckdb"))
con.execute("pragma threads=16")
rows = []
for name, s in SERIES.items():
    lo, hi = RANGE[s]
    con.execute(f"""
    create or replace temp table ek as
    select e.subject_id, e.study_id, e.ecg_time, l.valuenum as v, l.charttime as v_time
    from ecg e join lateral (
      select valuenum, charttime from labs
      where labs.subject_id = e.subject_id and labs.series = '{s}'
        and valuenum between {lo} and {hi}
        and abs(date_diff('minute', charttime, e.ecg_time)) <= 60
      order by abs(date_diff('second', charttime, e.ecg_time)) limit 1) l on true
    """)
    con.execute("""
    create or replace temp table dp as
    select subject_id, v as v_pre, v_time as t_pre,
           lead(v) over w as v_post, lead(v_time) over w as t_post,
           date_diff('minute', ecg_time, lead(ecg_time) over w) / 60.0 as dt_h
    from ek window w as (partition by subject_id order by ecg_time)
    qualify v_post is not null
    """)
    r = con.execute("""
    select (select count(*) from ek) n_xs,
           (select count(distinct subject_id) from ek) n_xs_pat,
           count(*) n_pairs_all, count(distinct subject_id) n_pat_all,
           sum(case when dt_h <= 24 then 1 else 0 end) n_pairs_24h,
           round(avg(abs(v_post - v_pre)), 3) mean_abs_d
    from dp where t_pre <> t_post""").df().iloc[0]
    rows.append({"analyte": name, **r.to_dict()})
    print(f"{name:30s} xs={int(r.n_xs):>7,}  pairs={int(r.n_pairs_all):>7,}  "
          f"<=24h={int(r.n_pairs_24h):>6,}  mean|d|={r.mean_abs_d}")

df = pd.DataFrame(rows)
df.to_csv(OUT / "wp0_analyte_yield.csv", index=False)

"""WP1b - multi-analyte labels, per-analyte delta cohorts, and the GLOBAL audit
patient set.

WP2 trains one model per analyte (not a shared backbone): the H5 falsification
test contrasts delta-K against delta-creatinine, and a shared trunk would couple
their errors and blunt exactly the comparison the test rests on.

Patient-level disjointness is therefore enforced against the UNION of every
analyte's audit patients, so no analyte's audit cohort can leak into any
analyte's training pool.
"""
import duckdb, pandas as pd
from paths import INTERIM, OUT

LAB_W, DT_MAX, NULL_TOL = 60, 24, 0.15
ANALYTES = {                       # series -> (plausible range, null-change tol)
    "k_chem":      ((1.5, 9.5),  0.15),
    "creat_chem":  ((0.1, 25.0), 0.10),
    "hb_heme":     ((3.0, 22.0), 0.30),
    "ca_tot_chem": ((4.0, 16.0), 0.15),
}
NAME = {"k_chem": "potassium", "creat_chem": "creatinine",
        "hb_heme": "hemoglobin", "ca_tot_chem": "calcium_total"}

con = duckdb.connect(str(INTERIM / "wp0.duckdb"))
con.execute("pragma threads=16")

# QC verdict per ECG, from the waveform pass
con.execute(f"""
create or replace table wqc as
select study_id, qc_pass from '{INTERIM}/wave_index.parquet'
""")

rows = []
for s, ((lo, hi), tol) in ANALYTES.items():
    a = NAME[s]
    # ECG-level labels: nearest draw within +-60 min, QC-passing ECGs only
    con.execute(f"""
    create or replace table obs_{a} as
    select o.subject_id, o.study_id, o.ecg_time, o.path, o.age, o.gender,
           l.valuenum as y, l.charttime as y_time
    from obs o join wqc q using (study_id)
    join lateral (
      select valuenum, charttime from labs
      where labs.subject_id = o.subject_id and labs.series = '{s}'
        and valuenum between {lo} and {hi}
        and abs(date_diff('minute', charttime, o.ecg_time)) <= {LAB_W}
      order by abs(date_diff('second', charttime, o.ecg_time)) limit 1) l on true
    where q.qc_pass
    """)
    # delta pairs; both ECGs QC-pass by construction, distinct draws enforced
    con.execute(f"""
    create or replace table delta_{a} as
    with p as (
      select subject_id, study_id as pre_study, ecg_time as pre_ecg,
             y as y_pre, y_time as t_pre, path as pre_path,
             lead(study_id) over w as post_study, lead(ecg_time) over w as post_ecg,
             lead(y) over w as y_post, lead(y_time) over w as t_post,
             lead(path) over w as post_path
      from obs_{a} window w as (partition by subject_id order by ecg_time)
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
    from p
    where t_pre <> t_post
    """)
    con.execute(f"""create or replace table audit_{a} as
                    select * from delta_{a} where dt_h <= {DT_MAX} and not acute_event""")
    con.execute(f"""create or replace table null_{a} as
                    select * from audit_{a}
                    where abs(dy_lab) < {tol} and not hd_icu and not hd_icd and not crrt""")
    r = {"analyte": a,
         "xs_ecgs":  con.execute(f"select count(*) from obs_{a}").fetchone()[0],
         "xs_pat":   con.execute(f"select count(distinct subject_id) from obs_{a}").fetchone()[0],
         "audit_pairs": con.execute(f"select count(*) from audit_{a}").fetchone()[0],
         "audit_pat":   con.execute(f"select count(distinct subject_id) from audit_{a}").fetchone()[0],
         "null_pairs":  con.execute(f"select count(*) from null_{a}").fetchone()[0],
         "hd_pairs":    con.execute(f"select count(*) from audit_{a} where (hd_icu or hd_icd) and not crrt").fetchone()[0]}
    rows.append(r)
    print(f"{a:14s} xs={r['xs_ecgs']:>7,}/{r['xs_pat']:>6,}pat  "
          f"audit={r['audit_pairs']:>6,}/{r['audit_pat']:>5,}pat  "
          f"null={r['null_pairs']:>5,}  hd={r['hd_pairs']:>4,}")

# ---------------------------- GLOBAL audit patient set (union over analytes)
con.execute("create or replace table audit_patients_all as " +
            " union ".join(f"select distinct subject_id from audit_{NAME[s]}" for s in ANALYTES))
n_audit = con.execute("select count(*) from audit_patients_all").fetchone()[0]
print(f"\nglobal audit patient set (union over analytes): {n_audit:,}")

for s in ANALYTES:
    a = NAME[s]
    con.execute(f"""
    create or replace table train_{a} as
    select * from obs_{a}
    where subject_id not in (select subject_id from audit_patients_all)
    """)
    n, p = con.execute(f"select count(*), count(distinct subject_id) from train_{a}").fetchone()
    leak = con.execute(f"""select count(*) from train_{a}
        where subject_id in (select subject_id from audit_patients_all)""").fetchone()[0]
    assert leak == 0, f"PATIENT LEAKAGE in train_{a}: {leak}"
    print(f"train_{a:12s} {n:>7,} ECGs / {p:>6,} patients   (leakage asserted 0)")
    con.execute(f"copy train_{a} to '{INTERIM}/train_{a}.parquet' (format parquet)")
    con.execute(f"copy audit_{a} to '{INTERIM}/audit_{a}.parquet' (format parquet)")
    con.execute(f"copy null_{a}  to '{INTERIM}/null_{a}.parquet'  (format parquet)")

pd.DataFrame(rows).to_csv(OUT / "wp1_analyte_cohorts.csv", index=False)
print("\nwrote per-analyte train/audit/null parquets + outputs/wp1_analyte_cohorts.csv")

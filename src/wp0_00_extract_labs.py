"""One-pass extraction of the analyte subset of hosp.labevents to parquet.

Itemids are resolved programmatically from d_labitems (Section 5.6): serum/chemistry
and blood-gas assays are kept as SEPARATE series and never pooled.
"""
import json
import duckdb
from paths import HOSP, INTERIM, OUT

# analyte -> {assay series -> [itemid]}; every id asserted against d_labitems below
ANALYTES = {
    "potassium":        {"chem":  [50971, 52610], "bg": [50822, 52452]},
    "calcium_total":    {"chem":  [50893],        "bg": [52034, 52035]},
    "calcium_ionised":  {"chem":  [51624],        "bg": [50808]},
    "hemoglobin":       {"heme":  [51222],        "bg": [50811], "chem": [51640, 51645]},
    "creatinine":       {"chem":  [50912, 52546], "bg": [52024]},
    "albumin":          {"chem":  [50862, 53085, 53138], "bg": [52022]},
    "bicarbonate":      {"chem":  [50882],        "bg": [50803, 52039]},
    "ph":               {"bg":    [50820]},
    "ld":               {"chem":  [50954]},
}
ALL_IDS = sorted({i for a in ANALYTES.values() for s in a.values() for i in s})

con = duckdb.connect()
d = con.execute(f"select * from read_csv_auto('{HOSP}/d_labitems.csv.gz')").df().set_index("itemid")

# assert every id exists, is a blood/serum assay, and log the resolved map
resolved = {}
for analyte, series in ANALYTES.items():
    resolved[analyte] = {}
    for s, ids in series.items():
        rows = []
        for i in ids:
            assert i in d.index, f"{analyte}/{s}: itemid {i} absent from d_labitems"
            r = d.loc[i]
            assert r.fluid in ("Blood", "Fluid"), f"{analyte}/{s}: itemid {i} fluid={r.fluid}"
            rows.append({"itemid": int(i), "label": r.label, "fluid": r.fluid,
                         "category": r.category})
        resolved[analyte][s] = rows

(OUT / "itemid_map.json").write_text(json.dumps(resolved, indent=2))
print(f"resolved {len(ALL_IDS)} itemids -> outputs/itemid_map.json")

con.execute("pragma threads=16")
con.execute(f"""
copy (
  select labevent_id, subject_id, hadm_id, itemid, charttime, storetime,
         value, valuenum, valueuom, ref_range_lower, ref_range_upper, flag,
         priority, comments
  from read_csv_auto('{HOSP}/labevents.csv.gz')
  where itemid in ({','.join(map(str, ALL_IDS))})
) to '{INTERIM}/labs_subset.parquet' (format parquet, compression zstd)
""")
n = con.execute(f"select count(*) from '{INTERIM}/labs_subset.parquet'").fetchone()[0]
print(f"labs_subset.parquet rows: {n:,}")

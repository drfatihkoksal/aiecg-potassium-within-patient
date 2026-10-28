from pathlib import Path

BASE = Path("/media/fatih/48bd54fc-58ee-46af-8646-96be97c8d33c")
ECG  = BASE / "mimic-iv-ecg" / "1.0"
HOSP = BASE / "mimiciv" / "3.1" / "hosp"
ICU  = BASE / "mimiciv" / "3.1" / "icu"
NOTE = BASE / "mimicNote" / "mimic-iv-note" / "note"

PROJ    = Path("/home/fatih/projects/deltaKdeltaKr")
OUT     = PROJ / "outputs"
INTERIM = PROJ / "data" / "interim"
for _p in (OUT, INTERIM):
    _p.mkdir(parents=True, exist_ok=True)

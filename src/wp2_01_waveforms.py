"""Extract cohort waveforms to a float16 memmap and compute Section 7.1 signal QC.

Headers were verified uniform across a 3,000-record sample (12 leads, 5000
samples, 500 Hz, format 16, gain 200/mV, baseline 0), so the flat .dat is read
directly; every record is still asserted on file size and header shape.
"""
import os, sys
import numpy as np, pandas as pd, duckdb
from multiprocessing import Pool
from scipy import signal as sps
from paths import ECG, INTERIM

FS, NSAMP, NLEAD = 500, 5000, 12
GAIN = 200.0
NBYTES = NSAMP * NLEAD * 2
WAVE_DIR = INTERIM.parent / "waveforms"
WAVE_DIR.mkdir(parents=True, exist_ok=True)
MM_PATH = WAVE_DIR / "ecg_raw_f16.npy"
LEADS = ["I", "II", "III", "aVR", "aVF", "aVL", "V1", "V2", "V3", "V4", "V5", "V6"]

_mm = None
def _init(n):
    global _mm
    _mm = np.memmap(MM_PATH, dtype=np.float16, mode="r+", shape=(n, NLEAD, NSAMP))

def _qc(x):
    """x: (12, 5000) float32 mV. Returns QC metrics; symmetric across leads."""
    sd = x.std(axis=1)
    flat = int((sd < 0.01).sum())                      # lead-off / disconnected
    sat  = float(np.mean(np.abs(x) > 32000 / GAIN))    # ADC saturation
    # baseline wander: power < 0.7 Hz vs total, on lead II
    f, p = sps.welch(x[1], fs=FS, nperseg=1024)
    tot = p.sum() + 1e-12
    wander = float(p[f < 0.7].sum() / tot)
    hf     = float(p[f > 40].sum() / tot)              # noise / EMG power
    # beat detection on lead II after a crude 5-15 Hz bandpass
    b, a = sps.butter(3, [5 / (FS / 2), 15 / (FS / 2)], btype="band")
    e = sps.filtfilt(b, a, x[1]) ** 2
    e = np.convolve(e, np.ones(int(0.12 * FS)) / (0.12 * FS), mode="same")
    pk, _ = sps.find_peaks(e, height=np.percentile(e, 90) * 0.4, distance=int(0.25 * FS))
    hr = 60.0 * len(pk) / (NSAMP / FS) if len(pk) else 0.0
    return dict(flat_leads=flat, sat_frac=sat, wander_frac=wander, hf_frac=hf,
                n_beats=int(len(pk)), hr_det=hr,
                amp_p99=float(np.percentile(np.abs(x), 99)))

def _one(job):
    i, path = job
    dat = ECG / (path + ".dat")
    try:
        if os.path.getsize(dat) != NBYTES:
            return i, dict(ok=False, err="size")
        raw = np.fromfile(dat, dtype="<i2")
        if raw.size != NSAMP * NLEAD:
            return i, dict(ok=False, err="shape")
        x = raw.reshape(NSAMP, NLEAD).T.astype(np.float32)
        miss = raw == -32768                            # WFDB missing-sample marker
        mfrac = float(miss.mean())
        if mfrac > 0:
            xm = x.copy(); xm[miss.reshape(NSAMP, NLEAD).T] = np.nan
            for L in range(NLEAD):
                v = xm[L]; bad = np.isnan(v)
                if bad.all(): v[:] = 0
                elif bad.any(): v[bad] = np.interp(np.flatnonzero(bad), np.flatnonzero(~bad), v[~bad])
            x = xm
        x /= GAIN                                       # -> mV
        q = _qc(x)
        _mm[i] = x.astype(np.float16)
        return i, dict(ok=True, miss_frac=mfrac, **q)
    except Exception as exc:                            # noqa: BLE001
        return i, dict(ok=False, err=type(exc).__name__)

if __name__ == "__main__":
    con = duckdb.connect(str(INTERIM / "wp0.duckdb"))
    idx = con.execute("""
        select distinct subject_id, study_id, ecg_time, path from obs order by study_id
    """).df()
    n = len(idx)
    print(f"{n:,} cohort ECGs -> {MM_PATH} "
          f"({n * NLEAD * NSAMP * 2 / 1e9:.1f} GB float16)", flush=True)
    np.memmap(MM_PATH, dtype=np.float16, mode="w+", shape=(n, NLEAD, NSAMP)).flush()

    jobs = list(enumerate(idx.path.tolist()))
    res = [None] * n
    with Pool(16, initializer=_init, initargs=(n,)) as pool:
        for c, (i, r) in enumerate(pool.imap_unordered(_one, jobs, chunksize=256), 1):
            res[i] = r
            if c % 20000 == 0:
                print(f"  {c:,}/{n:,}", flush=True)

    qc = pd.DataFrame(res)
    out = pd.concat([idx.reset_index(drop=True), qc], axis=1)
    out["row"] = np.arange(n)
    out.to_parquet(INTERIM / "wave_index.parquet", index=False)
    print("\nread failures:", int((~out.ok.astype(bool)).sum()))
    print(out[["flat_leads", "sat_frac", "wander_frac", "hf_frac", "n_beats",
               "hr_det", "amp_p99", "miss_frac"]].describe().T.to_string())

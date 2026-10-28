"""ECGFounder-specific preprocessing, replicating dataset.py + util.filter_bandpass
exactly. The upstream README warns that deviating makes results unreproducible,
so this mirrors their pipeline step for step:

  1. lead REORDER: MIMIC-IV-ECG stores I,II,III,aVR,aVF,aVL,V1..V6 but ECGFounder
     was trained on I,II,III,aVR,aVL,aVF,V1..V6 -> aVL and aVF are swapped.
     Their dataset.py does this via self.lead_indices; getting it wrong degrades
     the model silently.
  2. 50 Hz notch (iirnotch, Q=30), zero-phase
  3. Butterworth order 4 bandpass 0.67-40 Hz, zero-phase
  4. baseline removal by subtracting a median filter, kernel = int(0.4*fs)+1 = 201
     (scipy.signal.medfilt, i.e. zero-padded edges - NOT ndimage's 'nearest')
  5. GLOBAL z-score over the whole 12x5000 array (not per lead)
"""
import numpy as np, pandas as pd
from multiprocessing import Pool
from scipy.signal import medfilt, iirnotch, filtfilt, butter
from paths import INTERIM

FS, NSAMP, NLEAD = 500, 5000, 12
WAVE = INTERIM.parent / "waveforms"
SRC = WAVE / "ecg_raw_f16.npy"
DST = WAVE / "ecg_ecgfounder_f16.npy"

MIMIC_LEADS = ["I", "II", "III", "aVR", "aVF", "aVL", "V1", "V2", "V3", "V4", "V5", "V6"]
ECGF_LEADS  = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
LEAD_IDX = [MIMIC_LEADS.index(l) for l in ECGF_LEADS]
assert LEAD_IDX == [0, 1, 2, 3, 5, 4, 6, 7, 8, 9, 10, 11], LEAD_IDX

B_NOTCH, A_NOTCH = iirnotch(50, 30, FS)
B_BP, A_BP = butter(N=4, Wn=[0.67, 40], btype="bandpass", fs=FS)
KERNEL = int(0.4 * FS) + 1
if KERNEL % 2 == 0:
    KERNEL += 1

_src = _dst = None
def _init(n):
    global _src, _dst
    _src = np.memmap(SRC, dtype=np.float16, mode="r",  shape=(n, NLEAD, NSAMP))
    _dst = np.memmap(DST, dtype=np.float16, mode="r+", shape=(n, NLEAD, NSAMP))

def _one(i):
    x = np.asarray(_src[i], dtype=np.float64)
    x = np.nan_to_num(x, nan=0.0)
    x = x[LEAD_IDX, :]
    y = filtfilt(B_NOTCH, A_NOTCH, x, axis=-1)
    y = filtfilt(B_BP, A_BP, y, axis=-1)
    base = np.stack([medfilt(y[c], kernel_size=KERNEL) for c in range(NLEAD)])
    y = y - base
    y = (y - y.mean()) / (y.std() + 1e-8)          # global z-score, as upstream
    _dst[i] = y.astype(np.float16)
    return 1

if __name__ == "__main__":
    n = len(pd.read_parquet(INTERIM / "wave_index.parquet", columns=["row"]))
    print(f"ECGFounder preprocessing for {n:,} ECGs -> {DST}", flush=True)
    np.memmap(DST, dtype=np.float16, mode="w+", shape=(n, NLEAD, NSAMP)).flush()
    done = 0
    with Pool(16, initializer=_init, initargs=(n,)) as pool:
        for _ in pool.imap_unordered(_one, range(n), chunksize=128):
            done += 1
            if done % 20000 == 0:
                print(f"  {done:,}/{n:,}", flush=True)
    print("done", flush=True)

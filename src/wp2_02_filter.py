"""Standard filtering pass (WP2 input spec): zero-phase 3rd-order Butterworth
bandpass 0.5-40 Hz per lead, 500 Hz retained. Writes a second float16 memmap."""
import numpy as np, pandas as pd
from multiprocessing import Pool
from scipy import signal as sps
from paths import INTERIM

FS, NSAMP, NLEAD = 500, 5000, 12
WAVE = INTERIM.parent / "waveforms"
SRC, DST = WAVE / "ecg_raw_f16.npy", WAVE / "ecg_filt_f16.npy"
SOS = sps.butter(3, [0.5 / (FS / 2), 40.0 / (FS / 2)], btype="band", output="sos")

_src = _dst = None
def _init(n):
    global _src, _dst
    _src = np.memmap(SRC, dtype=np.float16, mode="r",  shape=(n, NLEAD, NSAMP))
    _dst = np.memmap(DST, dtype=np.float16, mode="r+", shape=(n, NLEAD, NSAMP))

def _blk(rng):
    lo, hi = rng
    x = np.asarray(_src[lo:hi], dtype=np.float32)
    y = sps.sosfiltfilt(SOS, x, axis=-1)
    _dst[lo:hi] = np.clip(y, -12, 12).astype(np.float16)
    return hi - lo

if __name__ == "__main__":
    n = len(pd.read_parquet(INTERIM / "wave_index.parquet", columns=["row"]))
    print(f"filtering {n:,} ECGs -> {DST}", flush=True)
    np.memmap(DST, dtype=np.float16, mode="w+", shape=(n, NLEAD, NSAMP)).flush()
    blocks = [(i, min(i + 512, n)) for i in range(0, n, 512)]
    done = 0
    with Pool(16, initializer=_init, initargs=(n,)) as pool:
        for d in pool.imap_unordered(_blk, blocks):
            done += d
            if done % 40960 < 512:
                print(f"  {done:,}/{n:,}", flush=True)
    print("done", flush=True)

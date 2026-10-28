"""Score a trained reference model over every cohort ECG.

Predictions are written once and every downstream work package (WP3 variance
decomposition, WP4 delta tracking, WP5 MDC, WP6 ablations) slices from them, so
all packages share one identical set of model outputs.
"""
import argparse
import numpy as np, pandas as pd, torch
from paths import INTERIM
from wp2_model import ECGResNet

NSAMP, NLEAD = 5000, 12
WAVE = INTERIM.parent / "waveforms" / "ecg_filt_f16.npy"

ap = argparse.ArgumentParser(); ap.add_argument("analyte"); ap.add_argument("--bs", type=int, default=512)
a = ap.parse_args()

widx = pd.read_parquet(INTERIM / "wave_index.parquet")
n = len(widx)
mm = np.memmap(WAVE, dtype=np.float16, mode="r", shape=(n, NLEAD, NSAMP))

model = ECGResNet().cuda().eval()
model.load_state_dict(torch.load(INTERIM / f"model_{a.analyte}.pt"))

pred = np.zeros(n, dtype=np.float32)
emb  = np.zeros((n, 1024), dtype=np.float16)   # kept for the identity ablations (WP6)
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    for i in range(0, n, a.bs):
        x = torch.from_numpy(np.asarray(mm[i:i + a.bs], dtype=np.float32)).cuda()
        z = model.embed(x)
        pred[i:i + a.bs] = model.head_reg(z).squeeze(-1).float().cpu().numpy()
        emb[i:i + a.bs] = z.float().cpu().numpy().astype(np.float16)
        if i % 51200 == 0:
            print(f"  {i:,}/{n:,}", flush=True)

out = widx[["subject_id", "study_id", "ecg_time", "row"]].copy()
out[f"pred_{a.analyte}"] = pred
out.to_parquet(INTERIM / f"pred_{a.analyte}.parquet", index=False)
np.save(INTERIM / f"emb_{a.analyte}.npy", emb)
print(f"wrote pred_{a.analyte}.parquet ({len(out):,}) and emb_{a.analyte}.npy")

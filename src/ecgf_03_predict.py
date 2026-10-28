"""Score the fine-tuned ECGFounder over every cohort ECG (ECGFounder preprocessing)."""
import argparse
import numpy as np, pandas as pd, torch
from paths import INTERIM, PROJ
from ecgf_model import ft_12lead_ECGFounder

NSAMP, NLEAD = 5000, 12
ap = argparse.ArgumentParser()
ap.add_argument("--tag", default="ecgfounder"); ap.add_argument("--bs", type=int, default=256)
a = ap.parse_args()

WAVE = INTERIM.parent / "waveforms" / "ecg_ecgfounder_f16.npy"
widx = pd.read_parquet(INTERIM / "wave_index.parquet")
n = len(widx)
mm = np.memmap(WAVE, dtype=np.float16, mode="r", shape=(n, NLEAD, NSAMP))

model = ft_12lead_ECGFounder("cuda", str(PROJ / "checkpoint" / "12_lead_ECGFounder_tensors.pth"),
                             n_classes=3)
model.load_state_dict(torch.load(INTERIM / f"model_{a.tag}.pt", weights_only=True))
model.eval()

pred = np.zeros(n, dtype=np.float32)
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    for i in range(0, n, a.bs):
        x = torch.from_numpy(np.asarray(mm[i:i + a.bs], dtype=np.float32)).cuda()
        pred[i:i + a.bs] = model(x)[:, 0].float().cpu().numpy()
        if i % 51200 == 0:
            print(f"  {i:,}/{n:,}", flush=True)

out = widx[["subject_id", "study_id", "ecg_time", "row"]].copy()
out[f"pred_{a.tag}"] = pred
out.to_parquet(INTERIM / f"pred_{a.tag}.parquet", index=False)
print(f"wrote pred_{a.tag}.parquet ({len(out):,})")

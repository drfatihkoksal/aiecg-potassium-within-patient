"""Download the official ECGFounder checkpoint and persist a tensors-only copy.

The upstream .pth is a full training checkpoint containing a numpy scalar, which
torch>=2.6 rejects under weights_only=True. We unpickle it once (official
MIT-licensed PKUDigitalHealth release, fetched over HTTPS from Hugging Face) and
re-save just the 509 weight tensors, so every downstream load is weights_only.
"""
import torch
from huggingface_hub import hf_hub_download
from paths import PROJ

CKDIR = PROJ / "checkpoint"; CKDIR.mkdir(exist_ok=True)
src = hf_hub_download("PKUDigitalHealth/ECGFounder", "12_lead_ECGFounder.pth",
                      local_dir=str(CKDIR))
ck = torch.load(src, map_location="cpu", weights_only=False)
sd = {k: v for k, v in ck["state_dict"].items() if torch.is_tensor(v)}
torch.save(sd, CKDIR / "12_lead_ECGFounder_tensors.pth")
print(f"pretrain val_auroc={float(ck['val_auroc']):.4f}; saved {len(sd)} tensors")

"""Fine-tune ECGFounder for potassium on the SAME split as our ResNet.

Comparability: the patient-level split is read from split_potassium.parquet, so
train/val/test membership is identical to the in-house model and the audit
patients remain excluded. The objective is also identical (MSE + 0.2 * BCE on the
two dyskalaemia heads, n_classes=3), which leaves the pretrained backbone as the
only difference between the two arms.

Amplitude jitter is dropped here: ECGFounder's preprocessing ends in a global
z-score, so rescaling afterwards would violate the normalisation contract the
checkpoint was trained under. The time shift is retained.

  python ecgf_02_train.py [--linear-prob] [--epochs 12]
"""
import argparse, json, time
import numpy as np, pandas as pd, torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
from paths import INTERIM, OUT, PROJ
from ecgf_model import ft_12lead_ECGFounder

NSAMP, NLEAD = 5000, 12
WAVE = INTERIM.parent / "waveforms" / "ecg_ecgfounder_f16.npy"
CKPT = PROJ / "checkpoint" / "12_lead_ECGFounder_tensors.pth"
LO, HI = 3.5, 5.5


class ECGSet(Dataset):
    def __init__(self, rows, y, n_total, train):
        self.rows, self.y, self.n, self.train, self.mm = rows, y, n_total, train, None

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        if self.mm is None:
            self.mm = np.memmap(WAVE, dtype=np.float16, mode="r", shape=(self.n, NLEAD, NSAMP))
        x = np.asarray(self.mm[self.rows[i]], dtype=np.float32)
        if self.train:
            x = np.roll(x, np.random.randint(-250, 251), axis=-1)
        return torch.from_numpy(x), torch.tensor(self.y[i], dtype=torch.float32)


def evaluate(model, dl, dev):
    model.eval(); P, Y = [], []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for x, y in dl:
            P.append(model(x.to(dev, non_blocking=True))[:, 0].float().cpu()); Y.append(y)
    p = torch.cat(P).numpy(); y = torch.cat(Y).numpy()
    m = {"n": len(y), "mae": float(np.abs(p - y).mean()),
         "rmse": float(np.sqrt(((p - y) ** 2).mean())), "r": float(np.corrcoef(p, y)[0, 1])}
    m["auroc_hyper"] = float(roc_auc_score(y >= HI, p))
    m["auroc_hypo"] = float(roc_auc_score(y <= LO, -p))
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--linear-prob", action="store_true")
    a = ap.parse_args()
    tag = "ecgfounder_linprob" if a.linear_prob else "ecgfounder"

    df = pd.read_parquet(INTERIM / "split_potassium.parquet")   # identical split
    n_total = len(pd.read_parquet(INTERIM / "wave_index.parquet", columns=["row"]))
    print(f"{tag}: {len(df):,} ECGs / {df.subject_id.nunique():,} patients")
    print(df.groupby("split").agg(n=("y", "size"), pat=("subject_id", "nunique")).to_string())

    dev = "cuda"
    dls = {}
    for s in ("train", "val", "test"):
        d = df[df.split == s]
        dls[s] = DataLoader(ECGSet(d.row.values, d.y.values, n_total, s == "train"),
                            batch_size=a.bs, shuffle=(s == "train"), num_workers=10,
                            pin_memory=True, drop_last=(s == "train"),
                            persistent_workers=True, prefetch_factor=4)

    model = ft_12lead_ECGFounder(dev, str(CKPT), n_classes=3, linear_prob=a.linear_prob)
    params = [p for p in model.parameters() if p.requires_grad]
    print(f"trainable params: {sum(p.numel() for p in params)/1e6:.2f} M "
          f"of {sum(p.numel() for p in model.parameters())/1e6:.2f} M")
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, epochs=a.epochs,
                                                steps_per_epoch=len(dls["train"]), pct_start=0.25)
    best, hist = 1e9, []
    ck = INTERIM / f"model_{tag}.pt"
    for ep in range(a.epochs):
        model.train(); t0 = time.time(); tot = nb = 0
        for x, y in dls["train"]:
            x = x.to(dev, non_blocking=True); y = y.to(dev, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                o = model(x)
                loss = (F.mse_loss(o[:, 0].float(), y)
                        + 0.2 * F.binary_cross_entropy_with_logits(
                            o[:, 1:].float(),
                            torch.stack([(y <= LO).float(), (y >= HI).float()], 1)))
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            opt.step(); sched.step()
            tot += loss.item(); nb += 1
        vm = evaluate(model, dls["val"], dev)
        hist.append({"epoch": ep, "train_loss": tot / nb, **vm})
        print(f"ep{ep:>2} loss={tot/nb:.4f} val MAE={vm['mae']:.4f} r={vm['r']:.3f} "
              f"({time.time()-t0:.0f}s)", flush=True)
        if vm["mae"] < best:
            best = vm["mae"]; torch.save(model.state_dict(), ck)

    model.load_state_dict(torch.load(ck, weights_only=True))
    tm = evaluate(model, dls["test"], dev)
    print(f"\nTEST {tag}: {json.dumps(tm, indent=2)}")
    pd.DataFrame(hist).to_csv(OUT / f"wp2_history_{tag}.csv", index=False)
    (OUT / f"wp2_test_{tag}.json").write_text(json.dumps(tm, indent=2))


if __name__ == "__main__":
    main()

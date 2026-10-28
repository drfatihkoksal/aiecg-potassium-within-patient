"""WP8g - EXPERIMENT 2 (exploratory): siamese Δ model.

Experiment 1 learned Δ from frozen embedding differences with linear/tree/MLP
read-outs and gained nothing over differencing the level model. That leaves two
possibilities: the read-out was too weak, or the representation itself does not
carry within-patient change. A siamese model separates them.

  --mode head    shared projection trained on the FROZEN trunk's embeddings
                 (a genuine siamese read-out, still no gradient into the trunk)
  --mode full    the same head with the trunk UNFROZEN, initialised from the
                 potassium model - tests whether task-specific fine-tuning can
                 create Δ information the frozen features lack

Antisymmetry augmentation: every pair is shown in both orders with the label
negated, which doubles the data and enforces dhat(a,b) = -dhat(b,a).

Same patient-level Δ split as WP8. Exploratory, per the section 8 design note.
"""
import argparse, json, time
import numpy as np, pandas as pd, torch, torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from scipy import stats
from paths import INTERIM, OUT
from wp2_model import ECGResNet

NSAMP, NLEAD, EMB = 5000, 12, 1024
WAVE = INTERIM.parent / "waveforms" / "ecg_filt_f16.npy"


class SiameseHead(nn.Module):
    """Shared per-ECG projection, then a head on [diff, pre, post]."""
    def __init__(self, din=EMB, dh=256, p=0.3):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(din, dh), nn.ReLU(), nn.Dropout(p))
        self.head = nn.Sequential(nn.Linear(3 * dh + 1, 128), nn.ReLU(), nn.Dropout(p),
                                  nn.Linear(128, 1))

    def forward(self, z_pre, z_post, dt):
        a, b = self.proj(z_pre), self.proj(z_post)
        return self.head(torch.cat([b - a, a, b, dt], dim=1)).squeeze(-1)


class PairSet(Dataset):
    def __init__(self, df, n_total, train, waveforms):
        self.pre = df.row_pre.values; self.post = df.row_post.values
        self.y = df.dy_lab.values.astype(np.float32)
        self.dt = df.dt_h.values.astype(np.float32)
        self.n, self.train, self.wave, self.mm = n_total, train, waveforms, None

    def __len__(self):
        return len(self.y)

    def _get(self, r, jitter):
        if self.mm is None:
            self.mm = np.memmap(WAVE, dtype=np.float16, mode="r", shape=(self.n, NLEAD, NSAMP))
        x = np.asarray(self.mm[r], dtype=np.float32)
        if jitter:
            x = np.roll(x, np.random.randint(-250, 251), axis=-1)
        return torch.from_numpy(x)

    def __getitem__(self, i):
        flip = self.train and np.random.rand() < 0.5  # seeded via np.random.seed in run()          # antisymmetry augmentation
        a, b = (self.post[i], self.pre[i]) if flip else (self.pre[i], self.post[i])
        y = -self.y[i] if flip else self.y[i]
        if self.wave:
            return self._get(a, self.train), self._get(b, self.train), \
                   torch.tensor([self.dt[i]]), torch.tensor(y)
        return torch.tensor(a), torch.tensor(b), torch.tensor([self.dt[i]]), torch.tensor(y)


def run(mode, epochs, lr, bs, seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    a = pd.read_parquet(INTERIM / "audit_potassium.parquet")
    sp = pd.read_parquet(INTERIM / "wp8_delta_split.parquet")
    widx = pd.read_parquet(INTERIM / "wave_index.parquet", columns=["study_id", "row"])
    n_total = len(widx)
    a = (a.merge(sp, on=["subject_id", "pre_study", "post_study"])
           .merge(widx.rename(columns={"study_id": "pre_study", "row": "row_pre"}), on="pre_study")
           .merge(widx.rename(columns={"study_id": "post_study", "row": "row_post"}), on="post_study"))
    tr, va, te = (a[a.split == s] for s in ("train", "val", "test"))
    dev = "cuda"
    full = mode == "full"

    head = SiameseHead().to(dev)
    if full:
        trunk = ECGResNet().to(dev)
        trunk.load_state_dict(torch.load(INTERIM / "model_potassium.pt", weights_only=True))
        params = list(head.parameters()) + list(trunk.parameters())
    else:
        trunk = None
        E = torch.tensor(np.asarray(np.load(INTERIM / "emb_potassium.npy", mmap_mode="r"),
                                    dtype=np.float32), device=dev)
        params = list(head.parameters())

    dls = {s: DataLoader(PairSet(d, n_total, s == "train", full), batch_size=bs,
                         shuffle=(s == "train"), num_workers=8 if full else 0,
                         pin_memory=full, drop_last=(s == "train" and full))
           for s, d in (("train", tr), ("val", va), ("test", te))}

    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, epochs=epochs,
                                                steps_per_epoch=max(len(dls["train"]), 1),
                                                pct_start=0.3)

    def emb(x):
        return trunk.embed(x) if full else E[x]

    def evaluate(dl):
        head.eval()
        if full: trunk.eval()
        P, Y = [], []
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for xa, xb, dt, y in dl:
                xa, xb, dt = xa.to(dev), xb.to(dev), dt.to(dev)
                P.append(head(emb(xa), emb(xb), dt).float().cpu()); Y.append(y)
        p, yy = torch.cat(P).numpy(), torch.cat(Y).numpy()
        return float(stats.pearsonr(p, yy)[0]), float(np.abs(p - yy).mean()), p, yy

    best, best_state, hist = -9, None, []
    for ep in range(epochs):
        head.train()
        if full: trunk.train()
        t0 = time.time(); tot = nb = 0
        for xa, xb, dt, y in dls["train"]:
            xa, xb, dt, y = xa.to(dev), xb.to(dev), dt.to(dev), y.to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = F.mse_loss(head(emb(xa), emb(xb), dt).float(), y)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            opt.step(); sched.step(); tot += loss.item(); nb += 1
        rv, mv, _, _ = evaluate(dls["val"])
        hist.append({"epoch": ep, "loss": tot / nb, "val_r": rv, "val_mae": mv})
        if rv > best:
            best = rv
            best_state = ({k: v.detach().clone() for k, v in head.state_dict().items()},
                          {k: v.detach().clone() for k, v in trunk.state_dict().items()} if full else None)
        print(f"  ep{ep:>2} loss={tot/nb:.4f} val r={rv:.4f} mae={mv:.4f} ({time.time()-t0:.0f}s)",
              flush=True)

    head.load_state_dict(best_state[0])
    if full: trunk.load_state_dict(best_state[1])
    rt, mt, p, y = evaluate(dls["test"])
    rv2, mv2, pv, _ = evaluate(dls["val"])
    print(f"\nTEST siamese_{mode}: r={rt:.4f}  MAE(Δ)={mt:.4f}  (best val r={best:.4f})")
    np.save(INTERIM / f"dhat_siamese_{mode}_s{seed}.npy", p)
    np.save(INTERIM / f"dhat_siamese_{mode}.npy", p)
    # val predictions are needed to fit the K_post combiner without touching test
    np.save(INTERIM / f"dhat_siamese_{mode}_val_s{seed}.npy", pv)
    np.save(INTERIM / f"dhat_siamese_{mode}_val.npy", pv)
    pd.DataFrame(hist).to_csv(OUT / f"wp8_siamese_{mode}_history.csv", index=False)
    return {"mode": mode, "test_r": rt, "test_mae_delta": mt, "best_val_r": best,
            "n_train_pairs": len(tr), "n_test_pairs": len(te)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["head", "full"], default="head")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    A = ap.parse_args()
    r = run(A.mode, A.epochs, A.lr, A.bs, A.seed)
    r["seed"] = A.seed
    (OUT / f"wp8_siamese_{A.mode}_s{A.seed}.json").write_text(json.dumps(r, indent=2))

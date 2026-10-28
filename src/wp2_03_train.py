"""WP2 - train the reference model for one analyte.

Split is patient-level (Section 5/WP2): stratified on the patient's mean analyte
value so the K distribution matches across folds. Audit patients were already
removed upstream, so no audit patient can appear here at all.

  python wp2_03_train.py potassium [--epochs 18]
"""
import argparse, json, time
import numpy as np, pandas as pd, torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
from paths import INTERIM, OUT
from wp2_model import ECGResNet

NSAMP, NLEAD = 5000, 12
WAVE = INTERIM.parent / "waveforms" / "ecg_filt_f16.npy"
# auxiliary binary heads: (low, high) clinical thresholds per analyte
AUX = {"potassium": (3.5, 5.5), "creatinine": (0.6, 1.5),
       "hemoglobin": (8.0, 12.0), "calcium_total": (8.4, 10.2),
       "calcium_corrected": (8.4, 10.2), "calcium_matched": (8.4, 10.2)}


class ECGSet(Dataset):
    def __init__(self, rows, y, n_total, train):
        self.rows, self.y, self.n, self.train, self.mm = rows, y, n_total, train, None

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        if self.mm is None:  # open per worker
            self.mm = np.memmap(WAVE, dtype=np.float16, mode="r",
                                shape=(self.n, NLEAD, NSAMP))
        x = np.asarray(self.mm[self.rows[i]], dtype=np.float32)
        if self.train:
            s = np.random.randint(-250, 251)                    # +-0.5 s time shift
            x = np.roll(x, s, axis=-1)
            x *= np.float32(np.random.uniform(0.9, 1.1))        # amplitude jitter
        return torch.from_numpy(x), torch.tensor(self.y[i], dtype=torch.float32)


def patient_split(df, seed=17):
    """80/10/10 by patient, stratified on the patient's mean value."""
    pm = df.groupby("subject_id").y.mean()
    q = pd.qcut(pm.rank(method="first"), 10, labels=False)
    rng = np.random.default_rng(seed)
    split = {}
    for s in range(10):
        pids = pm.index[q.values == s].to_numpy()
        rng.shuffle(pids)
        a, b = int(0.8 * len(pids)), int(0.9 * len(pids))
        for p in pids[:a]:  split[p] = "train"
        for p in pids[a:b]: split[p] = "val"
        for p in pids[b:]:  split[p] = "test"
    return df.subject_id.map(split).values


def evaluate(model, dl, dev, lo, hi):
    model.eval(); P, Y = [], []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for x, y in dl:
            p, _ = model(x.to(dev, non_blocking=True))
            P.append(p.float().cpu()); Y.append(y)
    p = torch.cat(P).numpy(); y = torch.cat(Y).numpy()
    m = {"n": len(y), "mae": float(np.abs(p - y).mean()),
         "rmse": float(np.sqrt(((p - y) ** 2).mean())),
         "r": float(np.corrcoef(p, y)[0, 1])}
    for tag, mask in (("hyper", y >= hi), ("hypo", y <= lo)):
        if 0 < mask.sum() < len(y):
            m[f"auroc_{tag}"] = float(roc_auc_score(mask, p if tag == "hyper" else -p))
    return m, p, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("analyte"); ap.add_argument("--epochs", type=int, default=18)
    ap.add_argument("--bs", type=int, default=256); ap.add_argument("--lr", type=float, default=2e-3)
    a = ap.parse_args()
    lo, hi = AUX[a.analyte]

    df = pd.read_parquet(INTERIM / f"train_{a.analyte}.parquet")
    widx = pd.read_parquet(INTERIM / "wave_index.parquet", columns=["study_id", "row"])
    n_total = len(widx)
    df = df.merge(widx, on="study_id", how="inner")
    df["split"] = patient_split(df)
    print(f"{a.analyte}: {len(df):,} ECGs / {df.subject_id.nunique():,} patients")
    print(df.groupby("split").agg(n=("y", "size"), pat=("subject_id", "nunique"),
                                  mean=("y", "mean")).to_string())

    dev = "cuda"
    dls = {}
    for s in ("train", "val", "test"):
        d = df[df.split == s]
        dls[s] = DataLoader(ECGSet(d.row.values, d.y.values, n_total, s == "train"),
                            batch_size=a.bs, shuffle=(s == "train"), num_workers=10,
                            pin_memory=True, drop_last=(s == "train"),
                            persistent_workers=True, prefetch_factor=4)

    model = ECGResNet().to(dev).to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, epochs=a.epochs,
                                                steps_per_epoch=len(dls["train"]), pct_start=0.25)
    best, hist = 1e9, []
    ck = INTERIM / f"model_{a.analyte}.pt"
    for ep in range(a.epochs):
        model.train(); t0 = time.time(); tot = nb = 0
        for x, y in dls["train"]:
            x = x.to(dev, non_blocking=True); y = y.to(dev, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                p, aux = model(x)
                loss = (F.mse_loss(p.float(), y)
                        + 0.2 * F.binary_cross_entropy_with_logits(
                            aux.float(), torch.stack([(y <= lo).float(), (y >= hi).float()], 1)))
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step(); sched.step()
            tot += loss.item(); nb += 1
        vm, _, _ = evaluate(model, dls["val"], dev, lo, hi)
        hist.append({"epoch": ep, "train_loss": tot / nb, **vm})
        print(f"ep{ep:>2} loss={tot/nb:.4f} val MAE={vm['mae']:.4f} r={vm['r']:.3f} "
              f"({time.time()-t0:.0f}s)", flush=True)
        if vm["mae"] < best:
            best = vm["mae"]; torch.save(model.state_dict(), ck)

    model.load_state_dict(torch.load(ck))
    tm, p, y = evaluate(model, dls["test"], dev, lo, hi)
    print(f"\nTEST {a.analyte}: {json.dumps(tm, indent=2)}")
    df.to_parquet(INTERIM / f"split_{a.analyte}.parquet", index=False)
    pd.DataFrame(hist).to_csv(OUT / f"wp2_history_{a.analyte}.csv", index=False)
    (OUT / f"wp2_test_{a.analyte}.json").write_text(json.dumps(tm, indent=2))


if __name__ == "__main__":
    main()

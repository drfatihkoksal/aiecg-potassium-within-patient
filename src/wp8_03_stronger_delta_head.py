"""WP8c - give the delta head its best shot before concluding.

The hypothesis stands or falls on whether a purpose-built delta model can extract
delta information the level model misses, so the head is not left as a single
ridge. Four families are fitted on the delta-train split, selected on the
delta-VAL split, and only the winner is scored on test.
"""
import numpy as np, pandas as pd, torch, torch.nn as nn
from scipy import stats
from sklearn.linear_model import RidgeCV
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingRegressor
from paths import INTERIM, OUT

SEED = 23
torch.manual_seed(SEED); np.random.seed(SEED)
a = pd.read_parquet(INTERIM / "audit_potassium.parquet")
sp = pd.read_parquet(INTERIM / "wp8_delta_split.parquet")
widx = pd.read_parquet(INTERIM / "wave_index.parquet", columns=["study_id", "row"])
pr = pd.read_parquet(INTERIM / "pred_potassium.parquet")[["study_id", "pred_potassium"]]
E = np.asarray(np.load(INTERIM / "emb_potassium.npy", mmap_mode="r"), dtype=np.float32)
a = (a.merge(sp, on=["subject_id", "pre_study", "post_study"])
       .merge(widx.rename(columns={"study_id": "pre_study", "row": "row_pre"}), on="pre_study")
       .merge(widx.rename(columns={"study_id": "post_study", "row": "row_post"}), on="post_study")
       .merge(pr.rename(columns={"study_id": "pre_study", "pred_potassium": "f_pre"}), on="pre_study")
       .merge(pr.rename(columns={"study_id": "post_study", "pred_potassium": "f_post"}), on="post_study"))
tr, va, te = (a[a.split == s] for s in ("train", "val", "test"))

def X_diff(df):  return E[df.row_post.values] - E[df.row_pre.values]
def X_both(df):  return np.hstack([E[df.row_pre.values], E[df.row_post.values]])

def r_of(p, df): return float(stats.pearsonr(p, df.dy_lab.values)[0])

cands = {}

# 1. ridge on the embedding difference
m = RidgeCV(alphas=np.logspace(-1, 4, 30)).fit(X_diff(tr), tr.dy_lab.values)
cands["ridge(diff)"] = (r_of(m.predict(X_diff(va)), va), m.predict(X_diff(te)))

# 2. ridge on both embeddings concatenated (can learn an asymmetric mapping)
m = RidgeCV(alphas=np.logspace(-1, 4, 30)).fit(X_both(tr), tr.dy_lab.values)
cands["ridge(pre,post)"] = (r_of(m.predict(X_both(va)), va), m.predict(X_both(te)))

# 3. GBM on PCA of the difference
pca = PCA(n_components=64, random_state=SEED).fit(X_diff(tr))
g = HistGradientBoostingRegressor(max_depth=3, max_iter=400, learning_rate=0.05,
                                  random_state=SEED).fit(pca.transform(X_diff(tr)), tr.dy_lab.values)
cands["gbm(pca64 diff)"] = (r_of(g.predict(pca.transform(X_diff(va))), va),
                            g.predict(pca.transform(X_diff(te))))

# 4. MLP on the difference
def mlp(Xtr, ytr, Xva, yva, Xte, epochs=300):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    T = lambda x: torch.tensor((x - mu) / sd, dtype=torch.float32).cuda()
    net = nn.Sequential(nn.Linear(Xtr.shape[1], 256), nn.ReLU(), nn.Dropout(0.3),
                        nn.Linear(256, 64), nn.ReLU(), nn.Dropout(0.3),
                        nn.Linear(64, 1)).cuda()
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2)
    xtr, yt = T(Xtr), torch.tensor(ytr, dtype=torch.float32).cuda()
    xva, xte = T(Xva), T(Xte)
    best, best_state = -9, None
    for ep in range(epochs):
        net.train(); opt.zero_grad()
        loss = nn.functional.mse_loss(net(xtr).squeeze(-1), yt)
        loss.backward(); opt.step()
        if ep % 10 == 0:
            net.eval()
            with torch.no_grad(): pv = net(xva).squeeze(-1).cpu().numpy()
            rv = float(stats.pearsonr(pv, yva)[0])
            if rv > best:
                best = rv; best_state = {k: v.clone() for k, v in net.state_dict().items()}
    net.load_state_dict(best_state); net.eval()
    with torch.no_grad(): return best, net(xte).squeeze(-1).cpu().numpy()

rv, pte = mlp(X_diff(tr), tr.dy_lab.values, X_diff(va), va.dy_lab.values, X_diff(te))
cands["mlp(diff)"] = (rv, pte)

print("=== delta-head families, selected on the delta-VAL split ===")
for k, (rv, _) in sorted(cands.items(), key=lambda kv: -kv[1][0]):
    print(f"  {k:20s} val r = {rv:.4f}")
best_name = max(cands, key=lambda k: cands[k][0])
best_te = cands[best_name][1]
print(f"\nselected: {best_name}")

base = (te.f_post - te.f_pre).values
lab = te.dy_lab.values
sid = te.subject_id.values

def cb(p, n_boot=2000, seed=9):
    rng = np.random.default_rng(seed)
    groups = {k: np.asarray(v) for k, v in pd.Series(range(len(sid))).groupby(sid).indices.items()}
    keys = np.array(list(groups.keys()))
    d = [float(stats.pearsonr(p[np.concatenate([groups[k] for k in
         rng.choice(keys, len(keys), replace=True)])],
         lab[np.concatenate([groups[k] for k in rng.choice(keys, len(keys), replace=True)])])[0])
         for _ in range(0)]  # placeholder, replaced below
    d = []
    for _ in range(n_boot):
        idx = np.concatenate([groups[k] for k in rng.choice(keys, len(keys), replace=True)])
        d.append(float(stats.pearsonr(p[idx], lab[idx])[0]))
    return float(stats.pearsonr(p, lab)[0]), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))

# paired bootstrap of the r difference
def paired_r(p1, p2, n_boot=2000, seed=9):
    rng = np.random.default_rng(seed)
    groups = {k: np.asarray(v) for k, v in pd.Series(range(len(sid))).groupby(sid).indices.items()}
    keys = np.array(list(groups.keys()))
    obs = float(stats.pearsonr(p1, lab)[0] - stats.pearsonr(p2, lab)[0])
    d = []
    for _ in range(n_boot):
        idx = np.concatenate([groups[k] for k in rng.choice(keys, len(keys), replace=True)])
        d.append(float(stats.pearsonr(p1[idx], lab[idx])[0] - stats.pearsonr(p2[idx], lab[idx])[0]))
    return obs, float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))

rb = cb(base); rd = cb(best_te)
o, lo, hi = paired_r(best_te, base)
print(f"\nexisting model implied delta : r = {rb[0]:.4f} [{rb[1]:.4f}, {rb[2]:.4f}]")
print(f"best purpose-trained head    : r = {rd[0]:.4f} [{rd[1]:.4f}, {rd[2]:.4f}]")
print(f"paired difference            : {o:+.4f} [{lo:+.4f}, {hi:+.4f}]  "
      f"significant={'yes' if lo > 0 or hi < 0 else 'no'}")

pd.DataFrame([{"delta_head": k, "val_r": v[0]} for k, v in cands.items()]).to_csv(
    OUT / "table17_delta_head_families.csv", index=False)
pd.DataFrame([{"predictor": "f_post - f_pre", "r": rb[0], "lo": rb[1], "hi": rb[2]},
              {"predictor": f"purpose-trained {best_name}", "r": rd[0], "lo": rd[1], "hi": rd[2]},
              {"predictor": "paired difference", "r": o, "lo": lo, "hi": hi}]).to_csv(
    OUT / "table18_delta_head_vs_implied.csv", index=False)

"""Step 3: the virtual-sensor Transformer, its training loop, and the per-car ridge baseline."""
import numpy as np, torch, torch.nn as nn
from .config import CH, W
from .features import PHYSICS, visible

# ======================= STEP 3: MODEL =========================================
class VirtualSensorNet(nn.Module):
    """Transformer over a 1-minute window. Input per time step: sensor values and physics inputs (hidden ones zeroed),
    which of them are visible, and minutes since engine start. Output: every sensor at every step, including the hidden ones."""
    def __init__(self, C=None, K=None, d=64, layers=2, heads=4):
        super().__init__()
        C = len(CH) if C is None else C; K = len(PHYSICS) if K is None else K
        self.inp = nn.Linear(2 * (C + K) + 1, d); self.pos = nn.Parameter(torch.zeros(1, W, d))
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.1, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, layers); self.out = nn.Linear(d, C)
    def forward(self, x, vis, ctx):
        v = vis[:, None, :].expand_as(x)
        return self.out(self.enc(self.inp(torch.cat([x * v, v, ctx[..., None]], -1)) + self.pos))

def hide_random(m):
    """Hide 1 or 2 of the sensors each window actually has (the model must rebuild them from the rest).
    Returns the visibility of every input (sensors + physics) and which sensors were hidden."""
    ms = m[:, :len(CH)]
    r = torch.rand_like(ms); r[ms == 0] = -1
    rank = r.argsort(-1, descending=True).argsort(-1)
    k = torch.randint(1, 3, (len(ms), 1))
    hide = ((rank < k) & (ms > 0)).float()
    return visible(ms * (1 - hide), m), hide

def fit(model, D, idx_tr, idx_va, epochs, lr, bs=256, patience=5):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    rng = np.random.RandomState(0); best, state, bad = np.inf, None, 0
    def loss_on(b):
        vis, hide = hide_random(D["m"][b])
        pred = model(D["x"][b], vis, D["ctx"][b])
        return (((pred - D["x"][b][:, :, :len(CH)]) ** 2) * hide[:, None, :]).sum() / (hide.sum() * W + 1e-6)
    for ep in range(epochs):
        model.train(); perm = idx_tr[rng.permutation(len(idx_tr))]
        for i in range(0, len(perm), bs):
            loss = loss_on(perm[i:i + bs]); opt.zero_grad(); loss.backward(); opt.step()
        model.eval(); torch.manual_seed(1)                                   # same hidden sensors every val pass
        with torch.no_grad(): va = np.nanmean([loss_on(idx_va[i:i + 1024]).item() for i in range(0, len(idx_va), 1024)] or [np.nan])
        if state is None or va < best - 1e-4:                               # (always keep the first epoch)
            best, state, bad = va, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience: break
    model.load_state_dict(state); return model

@torch.no_grad()
def channel_errors(model, D, bs=1024):
    """(N, C): error of each sensor when it alone is hidden and predicted from the others. NaN if the window lacks it."""
    model.eval(); N = len(D["x"]); E = np.full((N, len(CH)), np.nan)
    for c in range(len(CH)):
        idx = np.where(D["m"][:, c].numpy() > 0)[0]
        for i in range(0, len(idx), bs):
            b = idx[i:i + bs]; vis = D["m"][b][:, :len(CH)].clone(); vis[:, c] = 0
            pred = model(D["x"][b], visible(vis, D["m"][b]), D["ctx"][b])
            E[b, c] = ((pred[:, :, c] - D["x"][b][:, :, c]) ** 2).mean(1).numpy()
    return E

# ---- baseline: per-car linear virtual sensor ----------------------------------
def ridge_errors(D_fit, D_list, lam=1.0):
    """For each sensor, a ridge regression from the other sensors (+ warm-up time) at the same instant,
    fitted on this car's own fine-tune data (sensors only, no physics inputs). Returns an (N, C) error array for each dataset."""
    C = len(CH); xf, mf = D_fit["x"].numpy()[:, :, :C], D_fit["m"].numpy()[:, :C]; present = mf.mean(0) > 0.5
    outs = [np.full((len(D["x"]), len(CH)), np.nan) for D in D_list]
    for c in np.where(present)[0]:
        others = [j for j in np.where(present)[0] if j != c]
        rows = mf[:, others + [c]].all(1)
        if rows.sum() < 20 or not others: continue
        A = np.concatenate([xf[rows][:, :, others], D_fit["ctx"].numpy()[rows][..., None], np.ones((rows.sum(), W, 1))], -1).reshape(-1, len(others) + 2)
        y = xf[rows][:, :, c].reshape(-1)
        w = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ y)
        for D, E in zip(D_list, outs):
            x, m = D["x"].numpy()[:, :, :C], D["m"].numpy()[:, :C]; ok = m[:, others + [c]].all(1)
            B = np.concatenate([x[ok][:, :, others], D["ctx"].numpy()[ok][..., None], np.ones((ok.sum(), W, 1))], -1)
            E[ok, c] = ((B @ w - x[ok][:, :, c]) ** 2).mean(1)
    return outs

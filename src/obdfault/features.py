"""Step 2: cut drives into 1-minute windows and normalize them (plus the physics-input experiment)."""
import numpy as np, torch
from .config import CH, W, MIN_CH

# ---- EXPERIMENT (off in the final model; only used with --physics) -----------------------------------------
# Tested: slightly better on air faults but more false alarms, so main() empties PHYSICS unless --physics is given.
# With PHYSICS empty, full() returns the 9 sensors unchanged and visible() returns the sensor visibility unchanged.
# Physics inputs: simple engine relationships handed to the network ready-made. They are inputs only (never
# predicted); one is hidden whenever a sensor it is built from is hidden, so it can't give away the answer.
PHYSICS = [("AIR_PER_REV",  ("MAF", "RPM"),  lambda d: d["MAF"] / np.maximum(d["RPM"], 300) * 1000),  # air per engine turn ~ load
           ("LOAD_PER_MAP", ("LOAD", "MAP"), lambda d: d["LOAD"] / np.maximum(d["MAP"], 10)),          # load follows MAP (speed-density)
           ("LOAD_X_RPM",   ("LOAD", "RPM"), lambda d: d["LOAD"] * d["RPM"] / 1000)]                  # engine air demand ~ MAF

def full(X):
    """Raw sensors (T, C) -> sensors + physics inputs (T, C + K). Final model: no physics inputs, returns X unchanged."""
    d = {c: X[:, j] for j, c in enumerate(CH)}
    P = np.stack([f(d) for _, _, f in PHYSICS], 1) if PHYSICS else np.zeros((len(X), 0))
    return np.concatenate([X, P.astype(np.float32)], 1)

def visible(vis_s, m):
    """Sensor visibility (B, C) -> visibility of sensors + physics inputs (B, C + K). A physics input is hidden whenever a
    sensor it is built from is hidden (no peeking). Final model: no physics inputs, returns vis_s unchanged."""
    if not PHYSICS: return vis_s
    dep = torch.tensor([[c in srcs for c in CH] for _, srcs, _ in PHYSICS], dtype=torch.float32)   # (K, C)
    blocked = ((1 - vis_s) @ dep.T).clamp(max=1)
    return torch.cat([vis_s, m[:, len(CH):] * (1 - blocked)], 1)

# ====================== STEP 2: WINDOWS + NORMALISE ============================
def windows(trips, mu, sd, stride):
    """Cut trips into 1-minute windows. A channel counts as present in a window only if all 12 rows have it;
    windows with fewer than 4 present channels are dropped. Returns dict of arrays."""
    xs, ms, cs, codes, tid, st = [], [], [], [], [], []
    for k, tr in enumerate(trips):
        Xn = (full(tr["X"]) - mu) / sd
        for a in range(0, len(Xn) - W + 1, stride):
            x = Xn[a:a + W]; m = ~np.isnan(x).any(0)
            if m[:len(CH)].sum() < MIN_CH: continue
            xs.append(np.nan_to_num(x) * m); ms.append(m); cs.append(np.minimum(tr["ctx"][a:a + W], 30) / 30)
            codes.append(tr["code"][a:a + W].any()); tid.append(k); st.append(a)
    if not xs: return None
    return dict(x=torch.tensor(np.stack(xs), dtype=torch.float32), m=torch.tensor(np.stack(ms), dtype=torch.float32),
                ctx=torch.tensor(np.stack(cs), dtype=torch.float32), code=np.array(codes), trip=np.array(tid), start=np.array(st))

def subset(D, idx):
    return {k: v[idx] for k, v in D.items()}

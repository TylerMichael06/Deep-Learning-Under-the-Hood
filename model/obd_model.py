"""OBD-II "virtual sensor" fault model (PyTorch).

Idea: on a healthy engine the sensors agree with each other (MAF follows RPM x load, MAP follows
throttle, coolant follows warm-up time, ...). The network learns those relations. At test time each
sensor is hidden in turn and predicted from the others; a sensor that keeps disagreeing with its
prediction is flagged, and the model names it.

Only standard OBD-II PIDs that a phone app / ELM327 dongle can log are used, so the same model can be
fine-tuned on your own car later:
    RPM, SPEED, LOAD, THROTTLE, ECT (coolant), IAT (intake air), MAP, MAF, STFT1 (short-term fuel trim)
Cars that don't report a sensor just have it masked out.

Pipeline
  1. pretrain  on every healthy car except the one being tested       (Kaggle exp1 + exp2 + exp3)
  2. fine-tune on the first 40% of the test car's drives (its "normal") (what you will do on your car)
  3. calibrate the alarm line on the next 20% of its drives
  4. test on the last 40%: false alarms on clean drives, detection of 10 sensor faults injected into those
     same drives (incl. realistic MAF/MAP faults where engine load and fuel trim react), and real
     trouble-code events where a car has them (car13)
Compared against: the pretrained model without fine-tuning, and a per-car linear (ridge) virtual sensor.

Final settings (see model_build_guide.md, Step 7)
  - minute score = worst sensor's prediction error (averaged over 2 minutes) or the stuck-sensor check, whichever is larger
  - alarm line = 95th percentile of the car's calibration scores ("Medium")
  - no alarms in the first 5 minutes of a drive (warm-up)
  - a drive is FAULT if more than 30% of its minutes are flagged
  - no physics inputs (tested with --physics; not kept)

Usage (from the Results folder):
  python obd_model.py --data "..\\Datasets\\OBD-II datasets"               # full evaluation + final model
  python obd_model.py --data "..\\Datasets\\OBD-II datasets" --final-only  # only retrain/save obd_pretrained.pt
  python obd_model.py --data "..\\Datasets\\OBD-II datasets" --quick       # fast smoke test
Output is binary: every minute (and every drive) is HEALTHY or FAULT; for a FAULT it also names the likely sensor.
Writes obd_results_faults.csv, obd_results_summary.csv, obd_results_binary.csv and obd_pretrained.pt next to this script.
"""
import argparse, os, time
import numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.metrics import roc_auc_score

torch.manual_seed(0); np.random.seed(0)
HERE = os.path.dirname(os.path.abspath(__file__))
# GitHub layout: code in model/, spreadsheets in results/, figures in figures/. Anywhere else: everything next to the script.
_repo = os.path.basename(HERE).lower() == "model"
RESULTS = os.path.join(os.path.dirname(HERE), "results") if _repo else HERE
FIGURES = os.path.join(os.path.dirname(HERE), "figures") if _repo else os.path.join(HERE, "figures")

CH = ["RPM", "SPEED", "LOAD", "THROTTLE", "ECT", "IAT", "MAP", "MAF", "STFT1"]
SRC = {"RPM": "ENGINE_RPM", "SPEED": "SPEED", "LOAD": "ENGINE_LOAD", "THROTTLE": "THROTTLE_POS",
       "ECT": "ENGINE_COOLANT_TEMP", "IAT": "AIR_INTAKE_TEMP", "MAP": "INTAKE_MANIFOLD_PRESSURE",
       "MAF": "MAF", "STFT1": "SHORT TERM FUEL TRIM BANK 1"}
RANGE = {"RPM": (0, 8000), "SPEED": (0, 250), "LOAD": (0, 100), "THROTTLE": (0, 100), "ECT": (-40, 140),
         "IAT": (-40, 90), "MAP": (10, 255), "MAF": (0, 400), "STFT1": (-50, 50)}   # outside = logging glitch
DT, W = 5, 12            # resample every car to one row per 5 s; a window is 12 rows = 1 minute

# Physics inputs: simple engine relationships handed to the network ready-made. They are inputs only (never
# predicted); one is hidden whenever a sensor it is built from is hidden, so it can't give away the answer.
PHYSICS = [("AIR_PER_REV",  ("MAF", "RPM"),  lambda d: d["MAF"] / np.maximum(d["RPM"], 300) * 1000),  # air per engine turn ~ load
           ("LOAD_PER_MAP", ("LOAD", "MAP"), lambda d: d["LOAD"] / np.maximum(d["MAP"], 10)),          # load follows MAP (speed-density)
           ("LOAD_X_RPM",   ("LOAD", "RPM"), lambda d: d["LOAD"] * d["RPM"] / 1000)]                  # engine air demand ~ MAF

def full(X):
    """Raw sensors (T, C) -> sensors + physics inputs (T, C + K)."""
    d = {c: X[:, j] for j, c in enumerate(CH)}
    P = np.stack([f(d) for _, _, f in PHYSICS], 1) if PHYSICS else np.zeros((len(X), 0))
    return np.concatenate([X, P.astype(np.float32)], 1)

def visible(vis_s, m):
    """Sensor visibility (B, C) -> visibility of sensors + physics inputs (B, C + K)."""
    if not PHYSICS: return vis_s
    dep = torch.tensor([[c in srcs for c in CH] for _, srcs, _ in PHYSICS], dtype=torch.float32)   # (K, C)
    blocked = ((1 - vis_s) @ dep.T).clamp(max=1)
    return torch.cat([vis_s, m[:, len(CH):] * (1 - blocked)], 1)
FAULTY_CARS = {"car6"}   # P0133 for the whole log: never used as "healthy"

# =========================== STEP 1: LOAD ======================================
def num(s):
    """'48,60%' -> 48.6, '2124RPM' -> 2124, '24,77g/s' -> 24.77, 80.0 -> 80.0"""
    if pd.api.types.is_numeric_dtype(s): return s.astype(float)
    s = s.astype(str).str.replace(",", ".")
    return pd.to_numeric(s.str.extract(r"(-?\d+\.?\d*)")[0], errors="coerce")

def load_kaggle(folder):
    """Returns a list of trips: dict(unit, order, X (T,C) raw units with NaN, ctx (T,) minutes since engine start,
    code (T,) 1 if a trouble code was logged)."""
    trips = []
    for fname, unit_of in (("exp1_14drivers_14cars_dailyRoutes.csv", None),
                           ("exp2_19drivers_1car_1route.csv", "exp2_car"),   # 19 drivers, one car
                           ("exp3_4drivers_1car_1route.csv", "exp3_car")):   # 4 drivers, one car
        df = pd.read_csv(os.path.join(folder, fname), low_memory=False)
        df.columns = [c.strip().upper() for c in df.columns]
        df = df.dropna(subset=["VEHICLE_ID"]).reset_index(drop=True)
        d = pd.DataFrame({"src": df.VEHICLE_ID, "row": np.arange(len(df))})
        for c, s in SRC.items():
            v = num(df[s]) if s in df else pd.Series(np.nan, index=df.index)
            lo, hi = RANGE[c]; d[c] = v.where(v.between(lo, hi))
        d["runtime"] = pd.to_timedelta(df.ENGINE_RUNTIME, errors="coerce").dt.total_seconds()
        codes = df.TROUBLE_CODES.astype(str).str.strip() if "TROUBLE_CODES" in df else pd.Series("nan", index=df.index)
        d["code"] = codes.str.upper().str.startswith("P").astype(float)   # engine (P) codes only; e.g. car9's C0300 is chassis
        if unit_of is None:   # exp1: real millisecond timestamps
            d["t"] = pd.to_numeric(df.TIMESTAMP, errors="coerce") / 1000; d = d.sort_values(["src", "t"])
        else:                 # exp2/3: timestamps were saved rounded to 3 digits -> use engine runtime, file order
            d["t"] = d.runtime
        for src, g in d.groupby("src", sort=False):
            unit = unit_of or src
            gap = g.t.diff()
            new = gap.isna() | (gap <= 0 if unit_of else gap < 0) | (gap > 60) | (g.runtime.diff() < 0)
            for _, tg in g.groupby(new.cumsum()):
                tr = resample(tg)
                if tr is not None: trips.append(dict(unit=unit, order=float(tg.row.iloc[0]) if unit_of else float(tg.t.iloc[0]), **tr))
    return trips

def resample(g):
    """Put one trip on a 5 s grid. A value is kept only if a real reading is within 12 s of the grid point."""
    g = g.dropna(subset=["t"]); t = g.t.to_numpy(float)
    if len(t) < 3: return None
    keep = np.r_[True, np.diff(t) > 0]; g, t = g[keep], t[keep]; t = t - t[0]
    grid = np.arange(0, t[-1] + 1e-9, DT)
    if len(grid) < W: return None
    X = np.full((len(grid), len(CH)), np.nan, np.float32)
    for j, c in enumerate(CH):
        v = g[c].to_numpy(float); ok = ~np.isnan(v)
        if ok.sum() < 2: continue
        tx, vx = t[ok], v[ok]; i = np.clip(np.searchsorted(tx, grid), 1, len(tx) - 1)
        near = np.minimum(np.abs(tx[i] - grid), np.abs(tx[i - 1] - grid))
        X[:, j] = np.where((near <= 12) & (grid >= tx[0]) & (grid <= tx[-1]), np.interp(grid, tx, vx), np.nan)
    rt = g.runtime.to_numpy(float); ok = ~np.isnan(rt)
    start = (rt[ok][0] - t[ok][0]) if ok.any() else 0.0            # engine runtime at the trip's first row
    code = np.interp(grid, t, g.code.to_numpy(float)) > 0
    return dict(X=X, ctx=((start + grid) / 60).astype(np.float32), code=code)

# ====================== STEP 2: WINDOWS + NORMALISE ============================
def windows(trips, mu, sd, stride):
    """Cut trips into 1-minute windows. A channel counts as present in a window only if all 12 rows have it;
    windows with fewer than 4 present channels are dropped. Returns dict of arrays."""
    xs, ms, cs, codes, tid, st = [], [], [], [], [], []
    for k, tr in enumerate(trips):
        Xn = (full(tr["X"]) - mu) / sd
        for a in range(0, len(Xn) - W + 1, stride):
            x = Xn[a:a + W]; m = ~np.isnan(x).any(0)
            if m[:len(CH)].sum() < 4: continue
            xs.append(np.nan_to_num(x) * m); ms.append(m); cs.append(np.minimum(tr["ctx"][a:a + W], 30) / 30)
            codes.append(tr["code"][a:a + W].any()); tid.append(k); st.append(a)
    if not xs: return None
    return dict(x=torch.tensor(np.stack(xs), dtype=torch.float32), m=torch.tensor(np.stack(ms), dtype=torch.float32),
                ctx=torch.tensor(np.stack(cs), dtype=torch.float32), code=np.array(codes), trip=np.array(tid), start=np.array(st))

def subset(D, idx):
    return {k: v[idx] for k, v in D.items()}

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

# ====================== STEP 4: SCORE + CALIBRATE ==============================
def calibrate(E_cal, trip_cal, smooth, per_sensor=False):
    """On this car's clean calibration data, learn each sensor's usual (log) error -> z-score per sensor.
    Faults persist, so z-scores are averaged over the last `smooth` windows of the same drive (causal).
    Window score = worst sensor. Returns the scoring function and the calibration scores; the alarm cutoff is a
    percentile of those (99th -> ~1% false alarms on normal driving, 95th -> ~5%, ...).
    per_sensor=True gives every sensor its own cutoff: each sensor's score is rescaled so that its own 99th
    percentile on calibration = 1, so a naturally jumpy sensor (throttle) can't dominate the alarms."""
    ok = (~np.isnan(E_cal)).sum(0) >= 10                                  # sensors seen enough to calibrate
    L = np.log(E_cal + 1e-6); m = np.where(ok, np.nanmean(np.where(ok, L, 0), 0), np.nan)
    s = np.where(ok, np.nanstd(np.where(ok, L, 0), 0) + 1e-6, np.nan)
    def smoothed(E, trip):
        Z = pd.DataFrame((np.log(E + 1e-6) - m) / s)
        return Z.groupby(trip).transform(lambda c: c.rolling(smooth, min_periods=1).mean()).to_numpy()
    if per_sensor:
        Zc = smoothed(E_cal, trip_cal)
        med, top = np.nanmedian(Zc, 0), np.nanquantile(Zc, 0.99, axis=0)
        scale = np.where(top - med > 1e-6, top - med, np.nan)
    def score(E, trip):
        Z = smoothed(E, trip)
        if per_sensor: Z = (Z - med) / scale
        Z = np.where(np.isnan(Z), -np.inf, Z); sc, who = Z.max(1), Z.argmax(1)
        return np.where(np.isfinite(sc), sc, -10.0), np.where(np.isfinite(sc), who, -1)   # no usable sensor -> low score
    return score, score(E_cal, trip_cal)[0]

FLAT_SENSORS = ["ECT", "IAT"]          # slow sensors a "stuck" fault can hide in

def flat_minutes(trips, D):
    """For every window: how many minutes each slow sensor has been reading exactly the same value (ending at the window)."""
    runs = []
    for tr in trips:
        X = tr["X"][:, [CH.index(c) for c in FLAT_SENSORS]]; r = np.zeros_like(X)
        same = np.vstack([np.zeros((1, X.shape[1]), bool), np.abs(np.diff(X, axis=0)) < 1e-6])
        for i in range(1, len(X)): r[i] = np.where(same[i], r[i - 1] + DT / 60, 0)
        runs.append(r)
    return np.array([runs[t][st + W - 1] for t, st in zip(D["trip"], D["start"])]).reshape(-1, len(FLAT_SENSORS))

def unit_scale(x_cal, x):
    """Rescale a score so that on calibration data its median is 0 and its 99th percentile is 1."""
    med, top = np.median(x_cal), np.quantile(x_cal, .99)
    return (x - med) / max(top - med, 1e-6)

def slow_stuck_scorer(E_cal, Dcal, cal_trips, a, use_slow=True):
    """Fixes for slow faults. Score = the largest of
       (1) the usual 2-minute score, (2) a slow 10-minute score (catches creeping drift), and
       (3) a stuck check: minutes a slow sensor has been frozen ÷ the longest freeze seen on this car's calibration drives.
    Each part is rescaled on calibration data so they share one alarm line. Returns score(E, D, trips) and calibration scores."""
    fast, fast_cal = calibrate(E_cal, Dcal["trip"], a.smooth)
    slow, slow_cal = calibrate(E_cal, Dcal["trip"], a.smooth_long)
    warm = (Dcal["start"] + W) * DT / 60 >= a.grace
    longest = np.maximum(flat_minutes(cal_trips, Dcal).max(0), a.min_flat)
    def score(E, D, trips):
        (s, who), (l, who_l) = fast(E, D["trip"]), slow(E, D["trip"])
        ns, nl = unit_scale(fast_cal[warm], s), unit_scale(slow_cal[warm], l)
        if not use_slow: nl = np.full_like(nl, -np.inf)                            # stuck check only
        f = flat_minutes(trips, D) / longest; nf = f.max(1)
        who = np.where(nl > ns, who_l, who)
        who = np.where(nf > np.maximum(ns, nl), np.array([CH.index(c) for c in FLAT_SENSORS])[f.argmax(1)], who)
        return np.maximum.reduce([ns, nl, nf]), who
    return score, score(E_cal, Dcal, cal_trips)[0]

# ====================== STEP 5: FAULTS TO INJECT ===============================
# Realistic single-sensor faults, applied to whole drives in raw units (before the model sees them).
def _ramp(n): return np.linspace(0, 1, n)[:, None]
FAULTS = {
    "MAF under-reads 25% (dirty MAF, P0101)":        ("MAF", lambda v: v * 0.75),
    "MAF drifts to -30% over the drive":            ("MAF", lambda v: v * (1 - 0.3 * _ramp(len(v))[:, 0])),
    "MAP reads +10 kPa (P0106)":                    ("MAP", lambda v: v + 10),
    "Coolant warms at half speed (thermostat, P0128)": ("ECT", lambda v: v[np.where(~np.isnan(v))[0][0]] + 0.5 * (v - v[np.where(~np.isnan(v))[0][0]]) if (~np.isnan(v)).any() else v),
    "Coolant sensor stuck":                         ("ECT", lambda v: np.where(np.isnan(v), np.nan, v[np.where(~np.isnan(v))[0][0]]) if (~np.isnan(v)).any() else v),
    "Intake air temp reads +15 C":                  ("IAT", lambda v: v + 15),
    "Fuel trim +10% (vacuum leak / lean)":          ("STFT1", lambda v: v + 10),
    "Throttle signal noisy (+/-5%)":                ("THROTTLE", lambda v: v + np.random.RandomState(0).normal(0, 5, len(v))),
}

# Realistic air-sensor faults: in a real car the engine computer believes the wrong reading, so the knock-on effects
# show up in other sensors too. Engine load is calculated from the air sensor (MAF if the car has one, else MAP), and
# the wrong fuelling makes the O2 sensor push the short-term fuel trim the other way. Only part of that fuel error is
# still visible in the short-term trim; the rest is absorbed by the long-term trim, which is not logged (assumed 50%).
TRIM_SHARE = 0.5
_j = lambda c: CH.index(c)

def maf_reacts(f):
    def apply(X):
        X = X.copy(); X[:, _j("MAF")] *= f; X[:, _j("LOAD")] *= f                      # load is computed from MAF
        X[:, _j("STFT1")] += TRIM_SHARE * (1 / f - 1) * 100                               # too little fuel -> lean -> trim up
        return X
    return apply

def map_reacts(k):
    def apply(X):
        X = X.copy(); mp = X[:, _j("MAP")].copy(); X[:, _j("MAP")] = mp + k
        if np.isnan(X[:, _j("MAF")]).all(): X[:, _j("LOAD")] *= (mp + k) / np.maximum(mp, 10)   # no MAF: load from MAP
        X[:, _j("STFT1")] -= TRIM_SHARE * k / np.maximum(mp, 10) * 100                    # too much fuel -> rich -> trim down
        return X
    return apply

FAULTS_X = {
    "MAF under-reads 25%, engine reacts (realistic)": ("MAF", maf_reacts(0.75)),
    "MAP reads +10 kPa, engine reacts (realistic)":   ("MAP", map_reacts(10)),
}
ALL_FAULTS = {**FAULTS, **FAULTS_X}

def inject(trips, fault):
    c, f = ALL_FAULTS[fault]; j = CH.index(c); out = []
    for tr in trips:
        X = tr["X"].copy()
        if (~np.isnan(X[:, j])).sum() >= W:
            if fault in FAULTS_X: X = f(X)                   # several sensors react
            else: X[:, j] = f(X[:, j])                       # only the faulty sensor changes
        out.append(dict(tr, X=X))
    return out

def auroc(y, s):
    return roc_auc_score(y, s) if 0 < y.sum() < len(y) else np.nan

# ====================== STEP 6: EVALUATE ONE HELD-OUT CAR ======================
def prepare_car(unit, trips, a):
    """Steps 1-3 for one held-out car: pretrain on the other healthy cars, fine-tune on this car's first 40% of
    drive time, keep the next 20% for calibration and the last 40% as untouched test drives."""
    pre = [t for t in trips if t["unit"] != unit and t["unit"] not in FAULTY_CARS]
    allX = np.concatenate([full(t["X"]) for t in pre]); mu, sd = np.nanmean(allX, 0), np.nanstd(allX, 0) + 1e-6
    mine = sorted([t for t in trips if t["unit"] == unit and windows([t], mu, sd, stride=2) is not None],
                  key=lambda t: t["order"])                          # skip drives with no engine data (GPS-only logs)
    if len(mine) < 3: return None
    # --- pretrain
    D = windows(pre, mu, sd, stride=2); n = len(D["x"]); perm = np.random.RandomState(0).permutation(n)
    base = fit(VirtualSensorNet(), D, perm[: int(.9 * n)], perm[int(.9 * n):], a.epochs, 1e-3)
    # --- this car, in time order: first 40% of drive time fine-tunes, next 20% (other drives) sets the
    #     thresholds, last 40% is the test. Nothing from the test drives is used before scoring.
    lens = np.cumsum([len(t["X"]) for t in mine])
    c1, c2 = (np.searchsorted(lens, q * lens[-1]) + 1 for q in (0.4, 0.6))
    ft_trips, cal_trips, test_trips = mine[:c1], mine[c1:c2], mine[c2:]
    Dft, Dcal, Dclean = (windows(ts, mu, sd, stride=2) if ts else None for ts in (ft_trips, cal_trips, test_trips))
    if Dft is None or Dcal is None or Dclean is None: return None
    Dft, Dcal = (subset(D_, np.where(~D_["code"])[0]) for D_ in (Dft, Dcal))   # never learn "normal" from coded windows
    if len(Dft["x"]) < 50 or len(Dcal["x"]) < 20: return None
    k = int(.85 * len(Dft["x"]))                                                  # early-stopping slice of the fine-tune data
    Dtr = subset(Dft, np.arange(k))
    tuned = VirtualSensorNet(); tuned.load_state_dict(base.state_dict())
    tuned = fit(tuned, Dft, np.arange(k), np.arange(k, len(Dft["x"])), a.ft_epochs, 3e-4)
    return dict(base=base, tuned=tuned, mu=mu, sd=sd, Dtr=Dtr, Dcal=Dcal, Dclean=Dclean, test_trips=test_trips, cal_trips=cal_trips)

def evaluate_car(unit, trips, a):
    P = prepare_car(unit, trips, a)
    if P is None: return None, None
    base, tuned, mu, sd, Dtr, Dcal, Dclean, test_trips = (P[k] for k in ("base", "tuned", "mu", "sd", "Dtr", "Dcal", "Dclean", "test_trips"))
    has = Dclean["m"].numpy().mean(0) > 0.5
    methods = {"pretrained only": lambda Ds: [channel_errors(base, d) for d in Ds],
               "fine-tuned (ours)": lambda Ds: [channel_errors(tuned, d) for d in Ds],
               "ridge baseline": lambda Ds: ridge_errors(Dtr, Ds)}
    fault_trips = {f: inject(test_trips, f) for f in ALL_FAULTS if has[CH.index(ALL_FAULTS[f][0])]}
    fault_sets = {f: windows(t, mu, sd, stride=2) for f, t in fault_trips.items()}
    # warm-up grace: minutes before `a.grace` never raise an alarm and are not used to set the cutoff (cold starts)
    minute = lambda D_: (D_["start"] + W) * DT / 60
    warm_cal, warm_clean = minute(Dcal) >= a.grace, minute(Dclean) >= a.grace
    rows, summ = [], []
    for name, errs in methods.items():
        Es = errs([Dcal, Dclean] + list(fault_sets.values()))
        clean_ok = ~Dclean["code"]
        for scoring in ("shared cutoff", "stuck check", "two-speed + stuck check"):
            if scoring == "shared cutoff":
                sc, s_cal = calibrate(Es[0], Dcal["trip"], a.smooth); score = lambda E, D_, trips_: sc(E, D_["trip"])
            else:
                score, s_cal = slow_stuck_scorer(Es[0], Dcal, P["cal_trips"], a, use_slow=scoring != "stuck check")
            s_clean, _ = score(Es[1], Dclean, test_trips)
            real = auroc(Dclean["code"].astype(int), s_clean) if Dclean["code"].any() else np.nan
            scored = [(f, Df, *score(E, Df, fault_trips[f]), minute(Df) >= a.grace) for (f, Df), E in zip(fault_sets.items(), Es[2:])]
            for q in a.cutoffs:                               # every cutoff comes from calibration drives only
                thr = np.quantile(s_cal[warm_cal], q)
                fc = ((s_clean > thr) & warm_clean)[clean_ok]                # FAULT calls on clean minutes
                fa = fc.mean()
                summ.append(dict(car=unit, method=name, scoring=scoring, cutoff=q, grace_min=a.grace,
                                 sensors=",".join(c for c, h in zip(CH, has) if h),
                                 test_windows=int(clean_ok.sum()), false_alarm_rate=fa,
                                 real_code_windows=int(Dclean["code"].sum()), real_code_AUROC=real))
                clean_trip_hit = pd.Series(fc).groupby(Dclean["trip"][clean_ok]).mean()
                for f, Df, s_f, who, warm_f in scored:
                    ch = CH.index(ALL_FAULTS[f][0]); ff = (s_f > thr) & warm_f          # FAULT calls on faulty minutes
                    y = np.r_[np.zeros(clean_ok.sum()), np.ones(len(s_f))]
                    trip_hit = pd.Series(ff).groupby(Df["trip"]).mean()   # share of each drive's minutes flagged
                    for rule in a.drive_rules:        # a whole drive is FAULT if more than `rule` of its minutes are flagged
                        rows.append(dict(car=unit, method=name, scoring=scoring, cutoff=q, grace_min=a.grace, drive_rule=rule,
                                         fault=f, sensor=ALL_FAULTS[f][0], AUROC=auroc(y, np.r_[s_clean[clean_ok], s_f]),
                                         window_detection=ff.mean(), drive_detection=(trip_hit > rule).mean(),
                                         blamed_right_sensor=(who[ff] == ch).mean() if ff.any() else np.nan,
                                         false_alarm_rate=fa,
                                         # binary HEALTHY/FAULT counts: same drives, clean copy vs faulty copy (1:1)
                                         TP=int(ff.sum()), FN=int((~ff).sum()), FP=int(fc.sum()), TN=int((~fc).sum()),
                                         drive_TP=int((trip_hit > rule).sum()), drive_FN=int((trip_hit <= rule).sum()),
                                         drive_FP=int((clean_trip_hit > rule).sum()), drive_TN=int((clean_trip_hit <= rule).sum())))
    return pd.DataFrame(rows), pd.DataFrame(summ)

TAG = ""   # added to output file names (set by --tag, e.g. "_physics")
FINAL_SCORING = "stuck check"   # chosen after comparing all fixes: usual 2-minute score + stuck-sensor check

def save_csv(df, name):
    """Save next to this script; if the file is open in Excel (locked), save as <name>_new.csv instead."""
    path = os.path.join(RESULTS, name.replace(".csv", TAG + ".csv"))
    try: df.to_csv(path, index=False)
    except PermissionError:
        path = path.replace(".csv", "_new.csv"); df.to_csv(path, index=False)
    return os.path.basename(path)

def binary_report(R):
    """Confusion matrix + accuracy / precision / recall / F1 for each method, scoring and cutoff:
    per 1-minute window, and per whole drive for each drive rule."""
    def metrics(tp, fn, fp, tn):
        p = tp / (tp + fp) if tp + fp else np.nan; r = tp / (tp + fn) if tp + fn else np.nan
        return dict(TP=tp, FN=fn, FP=fp, TN=tn, accuracy=(tp + tn) / (tp + fn + fp + tn), precision=p, recall=r,
                    F1=2 * p * r / (p + r) if p + r else np.nan, false_alarm_rate=fp / (fp + tn))
    out, keys = [], ["method", "scoring", "cutoff"]
    first = R[R.drive_rule == R.drive_rule.min()]                         # minute counts are the same for every rule
    for k, g in first.groupby(keys):
        out.append(dict(zip(keys, k), level="minute", **metrics(*g[["TP", "FN", "FP", "TN"]].sum().to_numpy())))
    for k, g in R.groupby(keys + ["drive_rule"]):
        out.append(dict(zip(keys, k[:3]), level=f"drive (>{k[3]:.0%} of minutes)",
                        **metrics(*g[["drive_TP", "drive_FN", "drive_FP", "drive_TN"]].sum().to_numpy())))
    return pd.DataFrame(out).sort_values(keys + ["level"]).reset_index(drop=True)

# =================================== MAIN ======================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="folder with the three Kaggle exp*.csv files")
    ap.add_argument("--cars", nargs="*", help="which cars to hold out (default: every healthy car with enough data)")
    ap.add_argument("--epochs", type=int, default=40); ap.add_argument("--ft-epochs", type=int, default=20)
    ap.add_argument("--smooth", type=int, default=12, help="average scores over this many windows (12 x 10 s = 2 min)")
    ap.add_argument("--cutoffs", type=float, nargs="*", default=[0.95, 0.99, 0.975, 0.90, 0.80],
                    help="alarm cutoffs = percentiles of each car's calibration scores; the first one is the chosen "
                         "operating point (0.95 = 'Medium': precision ~0.85, ~8%% false alarms in testing)")
    ap.add_argument("--drive-rules", type=float, nargs="*", default=[0.3, 0.2, 0.4, 0.5],
                    help="a whole drive is FAULT if more than this share of its minutes is flagged; "
                         "the first one is the chosen rule (0.3 = more than 30%% of minutes)")
    ap.add_argument("--grace", type=float, default=5, help="warm-up grace period (minutes): no alarms at the start of a drive")
    ap.add_argument("--smooth-long", type=int, default=60, help="slow score: average over this many windows (60 x 10 s = 10 min)")
    ap.add_argument("--min-flat", type=float, default=5, help="stuck check: never flag a freeze shorter than this (minutes)")
    ap.add_argument("--physics", action="store_true",
                    help="add the physics inputs (tested: slightly better on air faults, more false alarms; not used in the final model)")
    ap.add_argument("--tag", default="", help="added to output file names, e.g. _physics")
    ap.add_argument("--skip-final", action="store_true", help="don't retrain/save obd_pretrained.pt")
    ap.add_argument("--final-only", action="store_true", help="skip the evaluation; only train and save obd_pretrained.pt")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    if a.quick: a.epochs, a.ft_epochs = 3, 2
    if not a.physics: PHYSICS.clear()                    # final model: no physics inputs
    global TAG; TAG = a.tag
    t0 = time.time(); trips = load_kaggle(a.data)
    size = pd.Series({u: sum(len(t["X"]) for t in trips if t["unit"] == u) for u in {t["unit"] for t in trips}})
    print(f"{len(trips)} drives, {len(size)} cars; 5-s rows per car:\n{size.sort_values(ascending=False).to_string()}")
    cars = a.cars or [u for u in size.index if size[u] >= 1500 and u not in FAULTY_CARS]
    if a.quick and not a.cars: cars = cars[:1]
    if a.final_only: return save_final(trips, a, t0)
    all_rows, all_summ = [], []
    for u in sorted(cars):
        t1 = time.time(); r, s = evaluate_car(u, trips, a)
        if r is None: print(f"{u}: skipped (not enough data)"); continue
        all_rows.append(r); all_summ.append(s); print(f"{u}: done in {time.time() - t1:.0f}s", flush=True)
        R, S = pd.concat(all_rows), pd.concat(all_summ)                       # save after every car
        save_csv(R, "obd_results_faults.csv"); save_csv(S, "obd_results_summary.csv")
    pd.set_option("display.width", 220); pd.set_option("display.max_columns", 20); pd.set_option("display.max_colwidth", 50)
    q0 = a.cutoffs[0]
    R0 = R[(R.cutoff == q0) & (R.scoring == FINAL_SCORING) & (R.drive_rule == a.drive_rules[0])]
    S0 = S[(S.cutoff == q0) & (S.scoring == FINAL_SCORING)]
    print(f"\n=== per-car false alarms and real trouble codes (cutoff {q0}) ===\n", S0.round(3).to_string(index=False))
    print(f"\n=== injected faults, mean over held-out cars (cutoff {q0}) ===\n",
          R0.groupby(["fault", "method"])[["AUROC", "window_detection", "drive_detection", "blamed_right_sensor"]].mean().round(3).to_string())
    print(f"\n=== overall (cutoff {q0}) ===\n", R0.groupby("method")[["AUROC", "window_detection", "drive_detection", "blamed_right_sensor", "false_alarm_rate"]].mean().round(3).to_string())
    print("\n=== binary classifier (FAULT = positive), all cars and faults pooled, every cutoff ===")
    B = binary_report(R); print(B.round(3).to_string(index=False))
    print("saved", save_csv(R, "obd_results_faults.csv"), save_csv(S, "obd_results_summary.csv"), save_csv(B, "obd_results_binary.csv"))
    if not a.skip_final: save_final(trips, a, t0)

def save_final(trips, a, t0):
    """Final model for your own cars: pretrain on every healthy car and save it with its settings."""
    pre = [t for t in trips if t["unit"] not in FAULTY_CARS]
    allX = np.concatenate([full(t["X"]) for t in pre]); mu, sd = np.nanmean(allX, 0), np.nanstd(allX, 0) + 1e-6
    D = windows(pre, mu, sd, stride=2); n = len(D["x"]); perm = np.random.RandomState(0).permutation(n)
    final = fit(VirtualSensorNet(), D, perm[: int(.9 * n)], perm[int(.9 * n):], a.epochs, 1e-3)
    torch.save(dict(state_dict=final.state_dict(), channels=CH, physics=[p[0] for p in PHYSICS], mu=mu, sd=sd, dt_seconds=DT, window=W,
                    settings=dict(scoring=FINAL_SCORING, cutoff=0.95, grace_min=a.grace, drive_rule=0.3, smooth=a.smooth, min_flat=a.min_flat)),
               os.path.join(HERE, "obd_pretrained.pt"))
    print(f"\nsaved obd_pretrained.pt   ({time.time() - t0:.0f}s total)")

if __name__ == "__main__":
    main()

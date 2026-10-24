"""Steps 6-7: turn prediction errors into minute scores and alarm lines (calibration, stuck-sensor check)."""
import numpy as np, pandas as pd, torch
from sklearn.metrics import roc_auc_score
from .config import CH, DT, W
from .features import visible
from .model import device_batch

# ====================== STEP 4: SCORE + CALIBRATE ==============================
def calibrate(E_cal, trip_cal, smooth, per_sensor=False):    # per_sensor=True: EXPERIMENT, not used in the final model
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

def slow_stuck_scorer(E_cal, Dcal, cal_trips, a, use_slow=True):   # final model: use_slow=False ("stuck check")
    """Fixes for slow faults. Score = the largest of
       (1) the usual 2-minute score, (2) a slow 10-minute score (catches creeping drift), and
       (3) a stuck check: minutes a slow sensor has been frozen ÷ the longest freeze seen on this car's calibration drives.
    Each part is rescaled on calibration data so they share one alarm line. Returns score(E, D, trips) and calibration scores."""
    fast, fast_cal = calibrate(E_cal, Dcal["trip"], a.smooth)
    slow, slow_cal = calibrate(E_cal, Dcal["trip"], a.smooth_long)
    warm = (Dcal["start"] + W) * DT / 60 >= a.grace
    longest = np.maximum(flat_minutes(cal_trips, Dcal).max(0), a.min_flat)
    def score(E, D, trips, B=None):                                                   # B: only used by map_scorer
        (s, who), (l, who_l) = fast(E, D["trip"]), slow(E, D["trip"])
        ns, nl = unit_scale(fast_cal[warm], s), unit_scale(slow_cal[warm], l)
        if not use_slow: nl = np.full_like(nl, -np.inf)                            # stuck check only
        f = flat_minutes(trips, D) / longest; nf = f.max(1)
        who = np.where(nl > ns, who_l, who)
        who = np.where(nf > np.maximum(ns, nl), np.array([CH.index(c) for c in FLAT_SENSORS])[f.argmax(1)], who)
        return np.maximum.reduce([ns, nl, nf]), who
    return score, score(E_cal, Dcal, cal_trips)[0]

# ---- MAP direction check (issue #32; part of the final scoring, "stuck check + MAP direction") ----------------------
# A MAP sensor reading a few kPa high still looks normal to the 2-minute score: squaring the prediction error throws
# away its direction, and while the throttle moves the healthy error is large anyway. This check keeps the direction:
# MAP's signed error (logged - predicted), averaged over 2 minutes. Healthy errors swing both ways and cancel out; an
# offset doesn't. It only counts beyond the 99th percentile of this car's calibration minutes (and then it is >= 1,
# above the alarm line), so it puts at most ~1% of calibration minutes above the line and leaves the line for the other
# sensors where it was. (An idle-MAP check was also tried: it doubled false alarms on car1 and car13, so it was dropped.)
MAP_SCORING = "stuck check + MAP direction"
MAP_Q = 0.99

@torch.no_grad()
def map_bias(model, D, bs=1024):
    """(N,): MAP's mean signed error (logged - predicted, normalized units) when MAP alone is hidden, as in
    channel_errors. NaN if the window lacks MAP."""
    model.eval(); c = CH.index("MAP"); B = np.full(len(D["x"]), np.nan)
    device = next(model.parameters()).device
    idx = np.where(D["m"][:, c].numpy() > 0)[0]
    for i in range(0, len(idx), bs):
        b = idx[i:i + bs]
        x, m, ctx = device_batch(D, b, device)
        vis = m[:, :len(CH)].clone(); vis[:, c] = 0
        B[b] = (x[:, :, c] - model(x, visible(vis, m), ctx)[:, :, c]).mean(1).cpu().numpy()
    return B

def map_direction(B_cal, Dcal, smooth, grace):
    """Calibrates the direction check on this car's calibration drives (after the warm-up). Returns check(B, trip) -> (N,):
    MAP's signed error, z-scored and averaged over the last `smooth` windows of the drive, in units of its calibration
    99th percentile, only where it is beyond it (>= 1); -inf elsewhere (nothing unusual, or no MAP)."""
    off = lambda B, trip: np.full(len(trip), -np.inf)
    if np.isnan(B_cal).all(): return off                                 # car has no MAP
    mb, sb = np.nanmean(B_cal), np.nanstd(B_cal) + 1e-6
    def avg(B, trip):
        Z = pd.Series((B - mb) / sb)
        return np.abs(Z.groupby(trip).transform(lambda c: c.rolling(smooth, min_periods=1).mean()).to_numpy())
    warm = (Dcal["start"] + W) * DT / 60 >= grace
    a_cal = avg(B_cal, Dcal["trip"])[warm]; a_cal = a_cal[~np.isnan(a_cal)]
    if len(a_cal) < 20: return off                                       # too little calibration data: check off
    tol = max(np.quantile(a_cal, MAP_Q), 1e-6)
    def check(B, trip):
        r = avg(B, trip) / tol
        return np.where(r >= 1, r, -np.inf)                              # NaN (window lacks MAP) -> -inf
    return check

def add_map_direction(s, who, p):
    """Score and blamed sensor with the direction check p added: the larger counts, and MAP is blamed when it wins."""
    return np.maximum(s, p), np.where(p > s, CH.index("MAP"), who)

def map_scorer(score0, s_cal0, B_cal, Dcal, a):
    """An existing scorer (the stuck check) plus the MAP direction check. score(E, D, trips, B) needs B = map_bias(model, D)."""
    check = map_direction(B_cal, Dcal, a.smooth, a.grace)
    def score(E, D, trips, B=None):
        return add_map_direction(*score0(E, D, trips), check(B, D["trip"]))
    return score, np.maximum(s_cal0, check(B_cal, Dcal["trip"]))

def auroc(y, s):
    return roc_auc_score(y, s) if 0 < y.sum() < len(y) else np.nan

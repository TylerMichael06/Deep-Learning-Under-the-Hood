"""Steps 6-7: turn prediction errors into minute scores and alarm lines (calibration, stuck-sensor check)."""
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from .config import CH, DT, W

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

# ---- EXPERIMENT (issue #33): bias score for slow air-sensor drift; off in the final model ------------------------------
def running_cusum(z, k):
    """Two-sided CUSUM through one drive. z: (N, C) standardized signed errors in time order (NaN = missing).
    'up' adds up readings lower than predicted, 'down' higher; each total forgets k per window and never drops below 0,
    so noise stays near 0 while a steady offset keeps growing. Returns (N, C) = max(up, down), NaN where z is missing."""
    up, down, out = np.zeros(z.shape[1]), np.zeros(z.shape[1]), np.full(z.shape, np.nan)
    for i, zi in enumerate(z):
        ok = ~np.isnan(zi)
        up[ok] = np.maximum(0, up[ok] + zi[ok] - k); down[ok] = np.maximum(0, down[ok] - zi[ok] - k)
        out[i, ok] = np.maximum(up, down)[ok]
    return out

BIAS_SENSORS = ["MAF", "MAP", "LOAD"]   # air-path sensors the bias score watches; others (fuel trim too) shift between healthy drives
MIN_BIAS_DRIVES = 3                      # fewer calibration drives can't show how much healthy drives differ: bias score off

def bias_scorer(B_cal, Dcal, k, grace):
    """Signed-error (bias) score on the air-path sensors (BIAS_SENSORS). On this car's calibration windows after the warm-up,
    learn each sensor's usual signed error (mean, std) -> z-score; then run a two-sided CUSUM per drive, starting after
    `grace` minutes. Window score = worst air sensor's total (0 if none has a value).
    Returns score(B, D), standardized with all calibration drives, and the calibration scores, where each calibration drive
    is scored with the mean/std of the OTHER calibration drives (leave-one-out), so the alarm line reflects how the score
    behaves on drives it was not fitted on. With fewer than MIN_BIAS_DRIVES calibration drives returns (None, None): no bias score."""
    warm = lambda D: (D["start"] + W) * DT / 60 >= grace
    air = np.isin(np.arange(len(CH)), [CH.index(c) for c in BIAS_SENSORS])
    def fit(B, D):                                                     # air sensors seen enough to calibrate
        Bw = np.where(air, B, np.nan)[warm(D)]; ok = (~np.isnan(Bw)).sum(0) >= 10; Bw = np.where(ok, Bw, 0)
        return np.where(ok, np.nanmean(Bw, 0), np.nan), np.where(ok, np.nanstd(Bw, 0) + 1e-6, np.nan)
    def scorer(m, s):
        def score(B, D):
            Z = np.where(warm(D)[:, None], (B - m) / s, np.nan); T = np.full(Z.shape, np.nan)
            for t in np.unique(D["trip"]): i = D["trip"] == t; T[i] = running_cusum(Z[i], k)
            T = np.where(np.isnan(T), -np.inf, T); sc, who = T.max(1), T.argmax(1)
            return np.where(np.isfinite(sc), sc, 0.0), np.where(np.isfinite(sc), who, -1)
        return score
    trips = np.unique(Dcal["trip"])
    if len(trips) < MIN_BIAS_DRIVES: return None, None
    score = scorer(*fit(B_cal, Dcal))
    s_cal = np.zeros(len(B_cal)); sub = lambda i: dict(trip=Dcal["trip"][i], start=Dcal["start"][i])
    for t in trips:                                                    # leave this calibration drive out
        i = Dcal["trip"] == t; s_cal[i] = scorer(*fit(B_cal[~i], sub(~i)))(B_cal[i], sub(i))[0]
    return score, s_cal

def slow_stuck_scorer(E_cal, Dcal, cal_trips, a, use_slow=True, B_cal=None):   # final model: use_slow=False, B_cal=None ("stuck check")
    """Fixes for slow faults. Score = the largest of
       (1) the usual 2-minute score, (2) a slow 10-minute score (catches creeping drift), and
       (3) a stuck check: minutes a slow sensor has been frozen ÷ the longest freeze seen on this car's calibration drives,
       (4) EXPERIMENT (issue #33), only if B_cal (calibration signed errors) is given: the bias score (bias_scorer).
    Each part is rescaled on calibration data so they share one alarm line. With B_cal, the calibration scores mix in-sample
    fast/slow/stuck parts with the leave-one-out bias part (only the bias score showed in-sample optimism). Returns score(E, D, trips, B) and calibration scores; score(E, D, trips) without B gives the parts other than the bias score."""
    fast, fast_cal = calibrate(E_cal, Dcal["trip"], a.smooth)
    slow, slow_cal = calibrate(E_cal, Dcal["trip"], a.smooth_long)
    warm = (Dcal["start"] + W) * DT / 60 >= a.grace
    longest = np.maximum(flat_minutes(cal_trips, Dcal).max(0), a.min_flat)
    bias, bias_cal = bias_scorer(B_cal, Dcal, a.cusum_k, a.grace) if B_cal is not None else (None, None)
    def base(E, D, trips):
        (s, who), (l, who_l) = fast(E, D["trip"]), slow(E, D["trip"])
        ns, nl = unit_scale(fast_cal[warm], s), unit_scale(slow_cal[warm], l)
        if not use_slow: nl = np.full_like(nl, -np.inf)                            # stuck check only
        f = flat_minutes(trips, D) / longest; nf = f.max(1)
        who = np.where(nl > ns, who_l, who)
        who = np.where(nf > np.maximum(ns, nl), np.array([CH.index(c) for c in FLAT_SENSORS])[f.argmax(1)], who)
        return np.maximum.reduce([ns, nl, nf]), who
    def score(E, D, trips, B=None):
        sc, who = base(E, D, trips)
        if bias is not None and B is not None:                                     # + bias score (without B: regular part only)
            b, who_b = bias(B, D); nb = unit_scale(bias_cal[warm], b)
            who = np.where(nb > sc, who_b, who); sc = np.maximum(sc, nb)
        return sc, who
    if bias is None: return score, score(E_cal, Dcal, cal_trips)[0]
    # the bias part of the calibration scores is leave-one-out, so the alarm line reflects drives the bias fit has not seen
    return score, np.maximum(base(E_cal, Dcal, cal_trips)[0], unit_scale(bias_cal[warm], bias_cal))

def confirm(flags, n=3, k=2):
    """EXPERIMENT (issue #33): OBD-II-style confirmation across drives. flags: one FAULT/HEALTHY verdict per judged drive,
    in driving order. A drive is confirmed FAULT only if it is FAULT and at least k of the last n drives (itself included)
    were FAULT, like a car's pending code that lights the check-engine lamp only when the fault comes back."""
    f = np.asarray(flags, bool)
    return f & (np.convolve(f, np.ones(n, int))[:len(f)] >= k)

def confirm_bias(regular, combined):
    """EXPERIMENT (issue #33): per-drive verdicts in driving order. A drive the regular score flags (regular) is FAULT at
    once, as before; a drive that is FAULT only with the bias score added (combined) must be confirmed on 2 of the last 3
    drives (confirm), because the bias score catches slow faults that persist from drive to drive."""
    regular, combined = np.asarray(regular, bool), np.asarray(combined, bool)
    return regular | confirm(combined)

# ---- slow faults (dirty MAF): one signed number per drive, tracked from drive to drive ----
GAP_Z = 3                  # a drive is FAULT if its gap is more than 3 of this car's normal spreads below its usual level
MIN_SPREAD = 0.02          # assumed: a healthy car's drive-to-drive gap varies by at least 2% (few calibration drives)

def drive_gaps(P, D, c, mu, sd, use, min_windows, floor=0.5):
    """Per drive: sum(measured) / sum(predicted) - 1 for sensor c over the windows in `use`. Signed and taken over the
    whole drive, so random prediction misses cancel out but a steady bias doesn't. `floor` (g/s for MAF) keeps idle
    predictions near 0 from blowing up the ratio. Drives with fewer than `min_windows` usable windows are left out."""
    ok = use & ~np.isnan(P[:, 0])
    meas = D["x"][ok][:, :, c].numpy() * sd[c] + mu[c]; pred = np.maximum(P[ok] * sd[c] + mu[c], floor)
    g = pd.DataFrame(dict(trip=D["trip"][ok], m=meas.sum(1), p=pred.sum(1), n=1)).groupby("trip").sum()
    g = g[g.n >= min_windows]; return g.m / g.p - 1

def gap_z(gap_cal):
    """From this car's calibration drives: how many normal drive-to-drive spreads a gap is BELOW the car's usual level.
    Spread = median absolute deviation (one odd drive can't stretch it), never below MIN_SPREAD."""
    med = gap_cal.median(); s = max(1.4826 * (gap_cal - med).abs().median(), MIN_SPREAD)
    return lambda gap: (med - gap) / s

def cusum(z, k=0.5, h=4):
    """Trend check over drives in time order (one-sided CUSUM, textbook k=0.5, h=4): position of the first drive where
    the running sum of (z - k) passes h, or None if it never does."""
    S = 0
    for i, v in enumerate(z):
        S = max(0, S + v - k)
        if S > h: return i
    return None

def auroc(y, s):
    return roc_auc_score(y, s) if 0 < y.sum() < len(y) else np.nan

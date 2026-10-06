"""The sensor faults injected into test drives."""
import numpy as np
from .config import CH, W

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
        if "LTFT1" in CH: X[:, _j("LTFT1")] += (1 - TRIM_SHARE) * (1 / f - 1) * 100       # the rest shows in the long-term trim
        return X
    return apply

def map_reacts(k):
    def apply(X):
        X = X.copy(); mp = X[:, _j("MAP")].copy(); X[:, _j("MAP")] = mp + k
        if np.isnan(X[:, _j("MAF")]).all(): X[:, _j("LOAD")] *= (mp + k) / np.maximum(mp, 10)   # no MAF: load from MAP
        X[:, _j("STFT1")] -= TRIM_SHARE * k / np.maximum(mp, 10) * 100                    # too much fuel -> rich -> trim down
        if "LTFT1" in CH: X[:, _j("LTFT1")] -= (1 - TRIM_SHARE) * k / np.maximum(mp, 10) * 100
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

"""Step 1: read the Kaggle OBD-II files (and the VED experiment) into drives on a 5-second grid."""
import glob, os
import numpy as np, pandas as pd
from .config import CH, SRC, RANGE, DT, W

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

# ---- EXPERIMENT (off in the final model; only runs with --ved) --------------------------------------------
# Vehicle Energy Dataset: 60 extra cars tried for pretraining. It has only 4 of our 9 sensors and did not improve
# the model, so the final model is trained on the Kaggle data only.
VED_COLS = {"RPM": "Engine RPM[RPM]", "SPEED": "Vehicle Speed[km/h]", "MAF": "MAF[g/sec]",
            "STFT1": "Short Term Fuel Trim Bank 1[%]", "LTFT1": "Long Term Fuel Trim Bank 1[%]"}
# VED's "Absolute Load" is a different quantity from Kaggle's calculated load (goes up to ~180%), so it is not used.

def load_ved(folder, n_pre, n_test, rows_pre=2000, rows_test=6000, seed=0):
    """Vehicle Energy Dataset (gasoline cars only). Returns trips like load_kaggle.
    n_pre cars (up to rows_pre 5-s rows each) are added to pretraining; n_test other cars (up to rows_test rows each)
    are held-out test cars that are never used for pretraining ("test_only")."""
    static = pd.read_excel(os.path.join(folder, "VED_Static_Data_ICE&HEV.xlsx"))
    ice = set(static.loc[static["Vehicle Type"] == "ICE", "VehId"])
    files = sorted(glob.glob(os.path.join(folder, "dynamic", "*.csv")))
    use = ["VehId", "Trip", "Timestamp(ms)"] + [VED_COLS[c] for c in VED_COLS if c in CH]
    parts = []
    for f in files:
        d = pd.read_csv(f, usecols=use, low_memory=False); parts.append(d[d.VehId.isin(ice)])
    df = pd.concat(parts, ignore_index=True)
    # cars with airflow and fuel trims (the signals VED adds), ranked by amount of driving
    ok = df.groupby("VehId").agg(n=("Trip", "size"), maf=(VED_COLS["MAF"], lambda s: s.notna().mean()),
                                 trim=(VED_COLS["STFT1"], lambda s: s.notna().mean()))
    ok = ok[(ok.maf > .8) & (ok.trim > .8) & (ok.n > 20000)].index.to_numpy()
    rng = np.random.RandomState(seed); rng.shuffle(ok)
    test_ids, pre_ids = set(ok[:n_test]), set(ok[n_test:n_test + n_pre])
    trips = []
    for vid, g in df[df.VehId.isin(test_ids | pre_ids)].groupby("VehId"):
        test = vid in test_ids; cap, total = (rows_test if test else rows_pre), 0
        for trip, tg in g.sort_values(["Trip", "Timestamp(ms)"]).groupby("Trip"):
            t = tg["Timestamp(ms)"].to_numpy(float) / 1000
            d = pd.DataFrame({"t": t - t[0], "runtime": t - t[0], "code": 0.0, "row": np.arange(len(tg))})
            for c in CH:
                v = pd.to_numeric(tg[VED_COLS[c]], errors="coerce").to_numpy() if c in VED_COLS else np.full(len(tg), np.nan)
                lo, hi = RANGE[c]; d[c] = np.where((v >= lo) & (v <= hi), v, np.nan)
            tr = resample(d)
            if tr is None: continue
            trips.append(dict(unit=f"ved{vid}", order=float(trip), test_only=test, **tr)); total += len(tr["X"])
            if total >= cap: break
    print(f"VED: {len(pre_ids)} pretraining cars, {len(test_ids)} held-out test cars, {len(trips)} drives")
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

"""Small made-up data for the tests: drives in the pipeline's own format, and raw files in the Kaggle CSV format."""
import argparse, os
import numpy as np, pandas as pd

# settings for a fast run: 1 training epoch; two cutoffs / drive rules so the comparison loops still run
EVAL_ARGS = argparse.Namespace(epochs=1, ft_epochs=1, smooth=12, cutoffs=[0.90, 0.95], min_judge=6, drive_rules=[0.3, 0.5],
                               grace=5, smooth_long=60, min_flat=5)
SYN_ARGS = argparse.Namespace(epochs=1, ft_epochs=1, smooth=12, smooth_long=60, graces=[5, 0], min_flat=5, cutoff=0.90,
                              clean_share=0.3)
KAGGLE_FILES = ("exp1_14drivers_14cars_dailyRoutes.csv", "exp2_19drivers_1car_1route.csv", "exp3_4drivers_1car_1route.csv")


def make_trips(n_cars=3, n_trips=8, T=200, seed=0):
    """Drives as load_kaggle returns them: X (T, 9 sensors in CH order), ctx (minutes since start), code (trouble code).
    Sensors follow each other the way a healthy engine's do. car1 has no MAF sensor (all NaN);
    car2's last drive logs a trouble code from row 100 to 130."""
    rng, trips = np.random.RandomState(seed), []
    for c in range(n_cars):
        for k in range(n_trips):
            t = np.arange(T) * 5 / 60
            rpm = 1500 + 800 * np.sin(t / 2 + c + k) + rng.normal(0, 50, T)
            throttle = 15 + 0.01 * (rpm - 1500) + rng.normal(0, 1, T)
            load = 30 + 0.02 * (rpm - 1500) + rng.normal(0, 1, T)
            speed = rpm / 40 + rng.normal(0, 1, T)
            ect = 90 - 70 * np.exp(-t / 4) + rng.normal(0, .3, T)
            iat = 25 + 5 * np.exp(-t / 10) + rng.normal(0, .3, T)
            map_ = 30 + 0.8 * throttle + rng.normal(0, 1, T)
            maf = rpm * load / 4000 + rng.normal(0, .5, T)
            if c == 1: maf[:] = np.nan
            stft = rng.normal(0, 2, T)
            code = np.zeros(T, bool)
            if c == 2 and k == n_trips - 1: code[100:130] = True
            X = np.stack([rpm, speed, load, throttle, ect, iat, map_, maf, stft], 1).astype(np.float32)
            trips.append(dict(unit=f"car{c}", order=float(k), X=X, ctx=t.astype(np.float32), code=code))
    return trips


def _drive(rng, vid, n=40, t0=0, codes=()):
    """n readings 5 s apart, written the way the Kaggle files store them ('2124RPM', '48,60%', '24,77g/s').
    Row 3 has an RPM logging glitch (9999, above the 8000 limit), which the loader must drop."""
    rpm = rng.randint(800, 3000, n); rpm[3] = 9999
    pct = lambda lo, hi: [f"{v:.2f}%".replace(".", ",") for v in rng.uniform(lo, hi, n)]
    return pd.DataFrame({
        "VEHICLE_ID": vid, "TIMESTAMP": (t0 + 5 * np.arange(n)) * 1000,
        "ENGINE_RUNTIME": [f"00:{5 * i // 60:02d}:{5 * i % 60:02d}" for i in range(n)],   # restarts at 0 every drive
        "ENGINE_RPM": [f"{v}RPM" for v in rpm], "SPEED": rng.randint(0, 120, n),
        "ENGINE_LOAD": pct(10, 80), "THROTTLE_POS": pct(5, 60),
        "ENGINE_COOLANT_TEMP": [f"{v}C" for v in rng.randint(60, 95, n)],
        "AIR_INTAKE_TEMP": [f"{v}C" for v in rng.randint(15, 40, n)],
        "INTAKE_MANIFOLD_PRESSURE": [f"{v}kPa" for v in rng.randint(25, 100, n)],
        "MAF": [f"{v:.2f}g/s".replace(".", ",") for v in rng.uniform(2, 40, n)],
        "SHORT TERM FUEL TRIM BANK 1": pct(-10, 10),
        "TROUBLE_CODES": ["P0133" if i in codes else "" for i in range(n)]})


def write_kaggle_csvs(folder, seed=0):
    """exp1: car A with two drives (13 minutes apart) and car B with one drive that logs a trouble code;
    exp2 and exp3: one car each, drives separated by the engine runtime restarting."""
    rng = np.random.RandomState(seed)
    frames = ([_drive(rng, "A"), _drive(rng, "A", t0=1000), _drive(rng, "B", codes=(10, 11, 12))],
              [_drive(rng, "x"), _drive(rng, "x")],
              [_drive(rng, "y")])
    for name, parts in zip(KAGGLE_FILES, frames):
        pd.concat(parts).to_csv(os.path.join(folder, name), index=False)

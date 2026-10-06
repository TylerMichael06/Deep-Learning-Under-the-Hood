"""Steps 4-5: pretrain + fine-tune for one held-out car, and train/save the final model."""
import os, time
import numpy as np, torch
from .config import CH, DT, W, FAULTY_CARS, FINAL_SCORING, MODELS
from .features import PHYSICS, full, windows, subset
from .model import VirtualSensorNet, fit

# ====================== STEP 6: EVALUATE ONE HELD-OUT CAR ======================
def prepare_car(unit, trips, a):
    """Steps 1-3 for one held-out car: pretrain on the other healthy cars, fine-tune on this car's first 40% of
    drive time, keep the next 20% for calibration and the last 40% as untouched test drives."""
    pre = [t for t in trips if t["unit"] != unit and t["unit"] not in FAULTY_CARS and not t.get("test_only")]
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

def save_final(trips, a, t0):
    """Final model for your own cars: pretrain on every healthy car and save it with its settings."""
    pre = [t for t in trips if t["unit"] not in FAULTY_CARS and not t.get("test_only")]
    allX = np.concatenate([full(t["X"]) for t in pre]); mu, sd = np.nanmean(allX, 0), np.nanstd(allX, 0) + 1e-6
    D = windows(pre, mu, sd, stride=2); n = len(D["x"]); perm = np.random.RandomState(0).permutation(n)
    final = fit(VirtualSensorNet(), D, perm[: int(.9 * n)], perm[int(.9 * n):], a.epochs, 1e-3)
    torch.save(dict(state_dict=final.state_dict(), channels=CH, physics=[p[0] for p in PHYSICS], mu=mu, sd=sd, dt_seconds=DT, window=W,
                    settings=dict(scoring=FINAL_SCORING, cutoff=a.cutoffs[0], min_judge_windows=a.min_judge, grace_min=a.grace, drive_rule=0.3, smooth=a.smooth, min_flat=a.min_flat)),
               os.path.join(MODELS, "obd_pretrained.pt"))
    print(f"\nsaved obd_pretrained.pt   ({time.time() - t0:.0f}s total)")

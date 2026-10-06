"""Settings shared by every step: the sensors, their raw-data columns and valid ranges, timing, folders, final settings."""
import os

# ponytail: repo root found from this file's location; works because we always install with `pip install -e .`
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS, FIGURES, MODELS = (os.path.join(ROOT, d) for d in ("results", "figures", "models"))
DATA = os.path.join(ROOT, "data")                             # public/ (git-ignored downloads) + collected/ (our car logs)
KAGGLE_DATA = os.path.join(DATA, "public", "kaggle-obd2")     # default --data for every CLI

CH = ["RPM", "SPEED", "LOAD", "THROTTLE", "ECT", "IAT", "MAP", "MAF", "STFT1"]
SRC = {"RPM": "ENGINE_RPM", "SPEED": "SPEED", "LOAD": "ENGINE_LOAD", "THROTTLE": "THROTTLE_POS",
       "ECT": "ENGINE_COOLANT_TEMP", "IAT": "AIR_INTAKE_TEMP", "MAP": "INTAKE_MANIFOLD_PRESSURE",
       "MAF": "MAF", "STFT1": "SHORT TERM FUEL TRIM BANK 1"}
RANGE = {"RPM": (0, 8000), "SPEED": (0, 250), "LOAD": (0, 100), "THROTTLE": (0, 100), "ECT": (-40, 140),
         "IAT": (-40, 90), "MAP": (10, 255), "MAF": (0, 400), "STFT1": (-50, 50)}   # outside = logging glitch
DT, W = 5, 12            # resample every car to one row per 5 s; a window is 12 rows = 1 minute
MIN_CH = 4               # a window needs at least this many sensors present

# ---- EXPERIMENT (off in the final model; only runs with --ltft) -------------------------------------------
def add_ltft():
    """EXPERIMENT (--ltft, used with --ved): long-term fuel trim (bank 1) as a 10th sensor. VED logs it; the Kaggle cars don't (masked)."""
    if "LTFT1" not in CH:
        CH.append("LTFT1"); SRC["LTFT1"] = "LONG TERM FUEL TRIM BANK 1"; RANGE["LTFT1"] = (-50, 50)

FAULTY_CARS = {"car6"}   # P0133 for the whole log: never used as "healthy"
FINAL_SCORING = "stuck check"   # chosen after comparing all fixes: usual 2-minute score + stuck-sensor check

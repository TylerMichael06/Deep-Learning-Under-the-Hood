"""Explore the Kaggle OBD-II data: per car, how much driving there is, which sensors it reports, and trouble codes.

  python scripts/explore_data.py          # reads data/public/kaggle-obd2 unless --data is given
"""
import argparse
import numpy as np, pandas as pd
from obdfault.config import CH, DT, FAULTY_CARS, KAGGLE_DATA
from obdfault.data import load_kaggle

ap = argparse.ArgumentParser(); ap.add_argument("--data", default=KAGGLE_DATA, help="folder with the three Kaggle exp*.csv files")
trips = load_kaggle(ap.parse_args().data)
rows = []
for unit in sorted({t["unit"] for t in trips}):
    mine = [t for t in trips if t["unit"] == unit]
    X = np.concatenate([t["X"] for t in mine])
    rows.append(dict(car=unit, drives=len(mine), hours=len(X) * DT / 3600,
                     code_minutes=sum(t["code"].sum() for t in mine) * DT / 60, used_as_healthy=unit not in FAULTY_CARS,
                     **{c: f"{(~np.isnan(X[:, j])).mean():.0%}" for j, c in enumerate(CH)}))
pd.set_option("display.width", 200)
print(pd.DataFrame(rows).round(2).to_string(index=False))
print("\nsensor columns = share of 5-s rows where the car reports that sensor")

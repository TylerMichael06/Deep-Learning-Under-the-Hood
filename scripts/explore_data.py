"""Explore the Kaggle OBD-II data: per car, how much driving there is, which sensors it reports, and trouble codes.

  python scripts/explore_data.py          # reads data/public/kaggle-obd2 unless --data is given
  python scripts/explore_data.py --thermostat --output /tmp/thermostat-trips.csv
"""
import argparse
import numpy as np, pandas as pd
from obdfault.config import CH, DT, FAULTY_CARS, KAGGLE_DATA
from obdfault.data import load_kaggle

from obdfault.thermostat import thermostat_audit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=KAGGLE_DATA, help="folder with the three Kaggle exp*.csv files")
    ap.add_argument("--thermostat", action="store_true", help="audit cold-start coverage without training")
    ap.add_argument("--cold-max", type=float, default=50., help="exploratory initial ECT bound in C, not a detector threshold")
    ap.add_argument("--output", help="optional per-trip audit CSV (requires --thermostat)")
    a = ap.parse_args()
    if a.output and not a.thermostat:
        ap.error("--output requires --thermostat")
    trips = load_kaggle(a.data)
    pd.set_option("display.width", 220)
    if a.thermostat:
        df = thermostat_audit(trips, cold_max=a.cold_max)
        if df.empty:
            print("No usable recordings found."); return
        print(f"{len(df)} recordings, {df.car.nunique()} vehicles; no model was trained.")
        print("Candidates: initial ECT below the exploratory bound, first ECT within 1 min, >=80% ECT coverage,")
        print("at least 6 valid windows after minute 5, no codes, not car6, no observed late/invalid start.")
        print("Unknown runtime remains provisional; absence of codes does not establish health.")
        print(df.groupby("car").agg(recordings=("car", "size"),
              runtime_unknown=("start_evidence", lambda x: (x == "unknown").sum()),
              cold_candidates=("cold_candidate", "sum"),
              confirmed_starts=("confirmed_start_candidate", "sum"),
              baseline_evaluable=("baseline_evaluable", "first")).to_string())
        used = df[df.baseline_evaluable & (df.split != "unused")]
        print("\nDefault evaluation cars, chronological splits:")
        print(used.groupby(["car", "split"]).agg(recordings=("trip_id", "size"),
              cold_candidates=("cold_candidate", "sum"),
              confirmed_starts=("confirmed_start_candidate", "sum"),
              unchanged_legacy=("legacy_unchanged", "sum")).to_string())
        if a.output:
            df.to_csv(a.output, index=False)
            print(f"\nSaved per-recording audit: {a.output}")
        return
    rows = []
    for unit in sorted({t["unit"] for t in trips}):
        mine = [t for t in trips if t["unit"] == unit]
        X = np.concatenate([t["X"] for t in mine])
        rows.append(dict(car=unit, drives=len(mine), hours=len(X) * DT / 3600,
                         code_minutes=sum(t["code"].sum() for t in mine) * DT / 60, used_as_healthy=unit not in FAULTY_CARS,
                         **{c: f"{(~np.isnan(X[:, j])).mean():.0%}" for j, c in enumerate(CH)}))
    print(pd.DataFrame(rows).round(2).to_string(index=False))
    print("\nsensor columns = share of 5-s rows where the car reports that sensor")


if __name__ == "__main__":
    main()

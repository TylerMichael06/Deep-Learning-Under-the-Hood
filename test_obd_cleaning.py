from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from obd_cleaning import (
    CANONICAL_SENSORS,
    clean_obd_file,
    parse_dtc_count_value,
    parse_numeric_value,
    parse_runtime_value,
)


class ParserTests(unittest.TestCase):
    def test_decimal_comma_and_units(self) -> None:
        examples = {
            "48,60%": 48.6,
            "2124RPM": 2124.0,
            "48km/h": 48.0,
            "100kPa": 100.0,
            "36C": 36.0,
            "24,77g/s": 24.77,
            "1,51336E+12": 1.51336e12,
        }
        for raw, expected in examples.items():
            with self.subTest(raw=raw):
                self.assertAlmostEqual(parse_numeric_value(raw), expected)

    def test_missing_and_invalid_numeric_values_stay_missing(self) -> None:
        for raw in ("", " ", None, "not a reading", "12 parsecs"):
            with self.subTest(raw=raw):
                self.assertTrue(np.isnan(parse_numeric_value(raw)))

    def test_runtime_and_dtc_parsers(self) -> None:
        self.assertEqual(parse_runtime_value("01:02:03"), 3723.0)
        self.assertTrue(np.isnan(parse_runtime_value("00:61:00")))
        self.assertEqual(parse_dtc_count_value("MIL is OFF0 codes"), 0.0)
        self.assertEqual(parse_dtc_count_value("4101000761001:00410100076100"), 0.0)
        self.assertEqual(parse_dtc_count_value("41 01 82 07 61 00"), 2.0)


class CleaningTests(unittest.TestCase):
    def test_traceability_masks_impossible_flags_and_trip_boundaries(self) -> None:
        header = (
            "TIMESTAMP,MARK,MODEL,CAR_YEAR,ENGINE_POWER,AUTOMATIC,VEHICLE_ID,"
            "BAROMETRIC_PRESSURE(KPA),ENGINE_COOLANT_TEMP,FUEL_LEVEL,ENGINE_LOAD,"
            "AMBIENT_AIR_TEMP,ENGINE_RPM,INTAKE_MANIFOLD_PRESSURE,MAF,"
            "AIR_INTAKE_TEMP,SPEED,SHORT TERM FUEL TRIM BANK 1,ENGINE_RUNTIME,"
            "THROTTLE_POS,DTC_NUMBER,TIMING_ADVANCE,EQUIV_RATIO\n"
        )
        rows = (
            '1500000000000,make,model,2017,"1,4",n,car1,100,90,"50,0%",20%,20,900,40,"4,2",30,0,"1,0%",00:01:00,20%,MIL is OFF0 codes,10%,"1,0%"\n'
            + "," * 22
            + "\n"
            + '1500000010000,make,model,2017,"1,4",n,car1,100,90,,20%,20,-1,40,"4,2",30,0,"1,0%",00:00:02,20%,MIL is OFF0 codes,10%,"1,0%"\n'
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "exp1_sample.csv"
            path.write_text(header + rows)
            cleaned, summary, metrics = clean_obd_file(path)

        self.assertEqual(len(cleaned), 2)
        self.assertEqual(summary["completely_blank_rows_dropped"], 1)
        self.assertEqual(cleaned["source_row"].tolist(), [2, 4])
        self.assertEqual(cleaned["trip_id"].nunique(), 2)
        self.assertEqual(cleaned["fuel_level_pct_observed"].tolist(), [1, 0])
        self.assertEqual(cleaned["engine_rpm_impossible"].tolist(), [0, 1])
        self.assertTrue(pd.isna(cleaned.loc[1, "fuel_level_pct"]))
        self.assertEqual(set(CANONICAL_SENSORS), {item["feature"] for item in metrics})


if __name__ == "__main__":
    unittest.main()

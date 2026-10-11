#!/usr/bin/env python3
"""Normalize the three raw OBD experiments into one canonical CSV.

The source CSV files are only read. Generated data and its quality report are
written under ``artifacts/`` by default.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


CANONICAL_SENSORS = (
    "barometric_pressure_kpa",
    "engine_coolant_temp_c",
    "fuel_level_pct",
    "engine_load_pct",
    "ambient_air_temp_c",
    "engine_rpm",
    "intake_manifold_pressure_kpa",
    "maf_g_per_s",
    "air_intake_temp_c",
    "speed_kph",
    "short_term_fuel_trim_bank_1_pct",
    "engine_runtime_seconds",
    "throttle_position_pct",
    "timing_advance",
    "equivalence_ratio",
    "dtc_count",
)

CANONICAL_UNITS = {
    "barometric_pressure_kpa": "kPa",
    "engine_coolant_temp_c": "degC",
    "fuel_level_pct": "percent",
    "engine_load_pct": "percent",
    "ambient_air_temp_c": "degC",
    "engine_rpm": "rpm",
    "intake_manifold_pressure_kpa": "kPa",
    "maf_g_per_s": "g/s",
    "air_intake_temp_c": "degC",
    "speed_kph": "km/h",
    "short_term_fuel_trim_bank_1_pct": "percent",
    "engine_runtime_seconds": "seconds",
    "throttle_position_pct": "percent",
    "timing_advance": "degrees",
    "equivalence_ratio": "ratio",
    "dtc_count": "count",
}

SOURCE_COLUMNS = {
    "barometric_pressure_kpa": ("BAROMETRIC_PRESSURE_KPA", "BAROMETRIC_PRESSURE"),
    "engine_coolant_temp_c": ("ENGINE_COOLANT_TEMP",),
    "fuel_level_pct": ("FUEL_LEVEL",),
    "engine_load_pct": ("ENGINE_LOAD",),
    "ambient_air_temp_c": ("AMBIENT_AIR_TEMP",),
    "engine_rpm": ("ENGINE_RPM",),
    "intake_manifold_pressure_kpa": ("INTAKE_MANIFOLD_PRESSURE",),
    "maf_g_per_s": ("MAF",),
    "air_intake_temp_c": ("AIR_INTAKE_TEMP",),
    "speed_kph": ("SPEED",),
    "short_term_fuel_trim_bank_1_pct": (
        "SHORT_TERM_FUEL_TRIM_BANK_1",
        "TERM_FUEL_TRIM_BANK_1",
    ),
    "engine_runtime_seconds": ("ENGINE_RUNTIME",),
    "throttle_position_pct": ("THROTTLE_POS",),
    "timing_advance": ("TIMING_ADVANCE",),
    "equivalence_ratio": ("EQUIV_RATIO",),
    "dtc_count": ("DTC_NUMBER",),
}

# These are broad physical/representational checks, not manufacturer limits.
POSSIBLE_RANGES = {
    "barometric_pressure_kpa": (0.0, 200.0),
    "engine_coolant_temp_c": (-80.0, 250.0),
    "fuel_level_pct": (0.0, 100.0),
    "engine_load_pct": (0.0, 100.0),
    "ambient_air_temp_c": (-100.0, 100.0),
    "engine_rpm": (0.0, 20_000.0),
    "intake_manifold_pressure_kpa": (0.0, 500.0),
    "maf_g_per_s": (0.0, 2_000.0),
    "air_intake_temp_c": (-100.0, 250.0),
    "speed_kph": (0.0, 500.0),
    "short_term_fuel_trim_bank_1_pct": (-100.0, 100.0),
    "engine_runtime_seconds": (0.0, 31_536_000.0),
    "throttle_position_pct": (0.0, 100.0),
    "timing_advance": (-180.0, 180.0),
    "equivalence_ratio": (0.0, 10.0),
    "dtc_count": (0.0, 255.0),
}

IDENTITY_COLUMNS = (
    "source_file",
    "source_row",
    "experiment",
    "timestamp",
    "trip_id",
    "source_vehicle_id",
    "actual_vehicle_id",
    "driver_id",
    "make",
    "model",
    "model_year",
    "engine",
    "transmission",
)

_NUMBER_PATTERN = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?$")
_UNIT_SUFFIX = re.compile(r"\s*(?:km/h|kpa|rpm|g/s|°?c|%)\s*$", re.IGNORECASE)


def normalized_column_name(value: str) -> str:
    """Return a case-insensitive, punctuation-insensitive column key."""

    return re.sub(r"_+", "_", re.sub(r"[^A-Z0-9]+", "_", value.strip().upper())).strip("_")


def parse_numeric_value(value: Any) -> float:
    """Parse a decimal-comma number after removing a supported unit suffix."""

    if value is None or pd.isna(value):
        return np.nan
    text = str(value).strip()
    if not text or text.casefold() in {"na", "n/a", "nan", "null", "none"}:
        return np.nan
    text = text.replace("−", "-").replace(",", ".")
    text = _UNIT_SUFFIX.sub("", text).strip().replace(" ", "")
    if not _NUMBER_PATTERN.fullmatch(text):
        return np.nan
    try:
        return float(text)
    except ValueError:
        return np.nan


def parse_runtime_value(value: Any) -> float:
    """Convert HH:MM:SS engine runtime to seconds."""

    if value is None or pd.isna(value):
        return np.nan
    text = str(value).strip()
    if not text:
        return np.nan
    match = re.fullmatch(r"(\d+):(\d{1,2}):(\d{1,2})", text)
    if not match:
        return parse_numeric_value(text)
    hours, minutes, seconds = (int(part) for part in match.groups())
    if minutes >= 60 or seconds >= 60:
        return np.nan
    return float(hours * 3600 + minutes * 60 + seconds)


def parse_dtc_count_value(value: Any) -> float:
    """Extract a DTC count from display text or an OBD Mode 01/PID 01 response."""

    if value is None or pd.isna(value):
        return np.nan
    text = str(value).strip()
    if not text:
        return np.nan
    match = re.search(r"(-?\d+)\s*codes?\b", text, flags=re.IGNORECASE)
    if match:
        return float(match.group(1))
    # A 41 01 response's first data byte contains the MIL bit followed by the
    # seven-bit count of confirmed emissions-related DTCs.
    compact = re.sub(r"[^0-9A-F]", "", text.upper())
    response = re.search(r"4101([0-9A-F]{2})", compact)
    if response:
        return float(int(response.group(1), 16) & 0x7F)
    return parse_numeric_value(text)


def parse_sensor_series(values: pd.Series, feature: str) -> pd.Series:
    parser = parse_numeric_value
    if feature == "engine_runtime_seconds":
        parser = parse_runtime_value
    elif feature == "dtc_count":
        parser = parse_dtc_count_value
    return values.map(parser).astype("Float64")


def _text_series(frame: pd.DataFrame, column: str | None) -> pd.Series:
    if column is None:
        return pd.Series(pd.NA, index=frame.index, dtype="string")
    result = frame[column].astype("string").str.strip()
    return result.mask(result.eq(""))


def _normalized_text_series(frame: pd.DataFrame, column: str | None) -> pd.Series:
    result = _text_series(frame, column)
    return result.str.casefold().str.replace(r"\s+", " ", regex=True)


def _first_source_column(
    normalized_columns: dict[str, str], candidates: Iterable[str]
) -> str | None:
    for candidate in candidates:
        if candidate in normalized_columns:
            return normalized_columns[candidate]
    return None


def _experiment_name(path: Path) -> str:
    match = re.match(r"(exp\d+)", path.name.casefold())
    if not match:
        raise ValueError(f"Cannot determine experiment from file name: {path.name}")
    return match.group(1)


def _parse_timestamp(values: pd.Series) -> pd.Series:
    milliseconds = values.map(parse_numeric_value).astype("Float64")
    return pd.to_datetime(milliseconds, unit="ms", utc=True, errors="coerce")


def _engine_series(frame: pd.DataFrame, column: str | None) -> pd.Series:
    raw = _normalized_text_series(frame, column)
    return raw.str.replace(",", ".", regex=False)


def _transmission_series(frame: pd.DataFrame, column: str | None) -> pd.Series:
    raw = _normalized_text_series(frame, column)
    mapping = {
        "y": "automatic",
        "yes": "automatic",
        "s": "automatic",
        "sim": "automatic",
        "true": "automatic",
        "n": "manual",
        "no": "manual",
        "false": "manual",
    }
    return raw.map(lambda value: mapping.get(value, value) if pd.notna(value) else pd.NA).astype("string")


def _build_trip_ids(frame: pd.DataFrame, experiment: str) -> pd.Series:
    """Create trips from ID changes, blank-row gaps, resets, and time gaps."""

    vehicle = frame["source_vehicle_id"]
    source_gap = frame["source_row"].diff().fillna(1).ne(1)
    id_change = vehicle.ne(vehicle.shift())
    runtime = frame["engine_runtime_seconds"].astype("Float64")
    runtime_reset = runtime.lt(runtime.shift() - 5).fillna(False) & ~id_change
    timestamp_delta = frame["timestamp"].diff().dt.total_seconds()
    time_boundary = (timestamp_delta.gt(30 * 60) | timestamp_delta.lt(0)).fillna(False)
    boundary = id_change | source_gap | runtime_reset | time_boundary
    if len(boundary):
        boundary.iloc[0] = True
    sequence = boundary.astype(int).cumsum()
    return sequence.map(lambda value: f"{experiment}_trip_{value:05d}").astype("string")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def clean_obd_file(path: Path) -> tuple[pd.DataFrame, dict[str, Any], list[dict[str, Any]]]:
    """Clean one source CSV and return rows, source summary, feature metrics."""

    raw = pd.read_csv(
        path,
        dtype="string",
        keep_default_na=False,
        skip_blank_lines=False,
        low_memory=False,
    )
    raw.columns = [str(column).strip() for column in raw.columns]
    source_rows = pd.Series(np.arange(2, len(raw) + 2), index=raw.index, dtype="int64")
    stripped = raw.apply(lambda column: column.astype("string").str.strip())
    blank_mask = stripped.fillna("").eq("").all(axis=1)
    retained = raw.loc[~blank_mask].copy()
    retained_source_rows = source_rows.loc[~blank_mask]
    experiment = _experiment_name(path)

    columns = {normalized_column_name(column): column for column in retained.columns}
    source_vehicle_column = columns.get("VEHICLE_ID")
    if source_vehicle_column is None:
        raise ValueError(f"{path.name} has no VEHICLE_ID column")

    source_vehicle = _text_series(retained, source_vehicle_column).fillna("unknown")
    timestamp_column = columns.get("TIMESTAMP") or columns.get("TIME")
    if timestamp_column is None:
        raise ValueError(f"{path.name} has no timestamp column")

    cleaned = pd.DataFrame(index=retained.index)
    cleaned["source_file"] = path.name
    cleaned["source_row"] = retained_source_rows
    cleaned["experiment"] = experiment
    cleaned["timestamp"] = _parse_timestamp(retained[timestamp_column])
    cleaned["source_vehicle_id"] = source_vehicle

    if experiment == "exp1":
        cleaned["actual_vehicle_id"] = "exp1:" + source_vehicle
        # The file describes 14 drivers paired one-to-one with 14 cars, but does
        # not provide a separate driver field. Keep this explicit as an inferred ID.
        cleaned["driver_id"] = "exp1:driver_for_" + source_vehicle
    else:
        cleaned["actual_vehicle_id"] = f"{experiment}:shared_car"
        cleaned["driver_id"] = experiment + ":" + source_vehicle

    cleaned["make"] = _normalized_text_series(retained, columns.get("MARK"))
    cleaned["model"] = _normalized_text_series(retained, columns.get("MODEL"))
    model_year_raw = _text_series(retained, columns.get("CAR_YEAR"))
    cleaned["model_year"] = pd.to_numeric(model_year_raw, errors="coerce").astype("Int64")
    cleaned["engine"] = _engine_series(retained, columns.get("ENGINE_POWER"))
    cleaned["transmission"] = _transmission_series(retained, columns.get("AUTOMATIC"))

    raw_nonblank_masks: dict[str, pd.Series] = {}
    for feature in CANONICAL_SENSORS:
        source_column = _first_source_column(columns, SOURCE_COLUMNS[feature])
        if source_column is None:
            source_values = pd.Series("", index=retained.index, dtype="string")
        else:
            source_values = retained[source_column].astype("string")
        raw_nonblank = source_values.str.strip().fillna("").ne("")
        parsed = parse_sensor_series(source_values, feature)
        minimum, maximum = POSSIBLE_RANGES[feature]
        impossible = parsed.notna() & ((parsed < minimum) | (parsed > maximum))
        cleaned[feature] = parsed
        cleaned[f"{feature}_observed"] = parsed.notna().astype("int8")
        cleaned[f"{feature}_impossible"] = impossible.astype("int8")
        raw_nonblank_masks[feature] = raw_nonblank

    cleaned["trip_id"] = _build_trip_ids(cleaned, experiment)
    cleaned["any_impossible"] = cleaned[
        [f"{feature}_impossible" for feature in CANONICAL_SENSORS]
    ].any(axis=1).astype("int8")

    ordered_columns = [
        *IDENTITY_COLUMNS,
        *CANONICAL_SENSORS,
        *(f"{feature}_observed" for feature in CANONICAL_SENSORS),
        *(f"{feature}_impossible" for feature in CANONICAL_SENSORS),
        "any_impossible",
    ]
    cleaned = cleaned.loc[:, ordered_columns].reset_index(drop=True)

    feature_metrics: list[dict[str, Any]] = []
    metrics_frame = cleaned.copy()
    for feature in CANONICAL_SENSORS:
        # The masks retain the original index while cleaned has a new RangeIndex.
        metrics_frame[f"_{feature}_raw_nonblank"] = raw_nonblank_masks[feature].to_numpy()
    for actual_vehicle_id, group in metrics_frame.groupby("actual_vehicle_id", sort=True):
        for feature in CANONICAL_SENSORS:
            raw_nonblank = group[f"_{feature}_raw_nonblank"].astype(bool)
            observed = group[f"{feature}_observed"].astype(bool)
            raw_count = int(raw_nonblank.sum())
            parsed_count = int((raw_nonblank & observed).sum())
            feature_metrics.append(
                {
                    "experiment": experiment,
                    "actual_vehicle_id": str(actual_vehicle_id),
                    "feature": feature,
                    "unit": CANONICAL_UNITS[feature],
                    "rows": int(len(group)),
                    "raw_nonblank": raw_count,
                    "parsed_values": parsed_count,
                    "parse_failures": int((raw_nonblank & ~observed).sum()),
                    "missing_values": int((~observed).sum()),
                    "impossible_values": int(group[f"{feature}_impossible"].sum()),
                    "parse_success_rate": parsed_count / raw_count if raw_count else None,
                }
            )

    source_summary = {
        "experiment": experiment,
        "source_file": path.name,
        "source_sha256": _file_sha256(path),
        "raw_rows": int(len(raw)),
        "completely_blank_rows_dropped": int(blank_mask.sum()),
        "retained_rows": int(len(cleaned)),
        "row_count_reconciles": int(len(raw)) == int(blank_mask.sum()) + int(len(cleaned)),
        "vehicles": int(cleaned["actual_vehicle_id"].nunique()),
        "drivers": int(cleaned["driver_id"].nunique()),
        "trips": int(cleaned["trip_id"].nunique()),
        "missing_timestamps": int(cleaned["timestamp"].isna().sum()),
        "unique_timestamps": int(cleaned["timestamp"].nunique()),
    }
    return cleaned, source_summary, feature_metrics


def _aggregate_feature_metrics(metrics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    aggregate: dict[tuple[str, str], dict[str, Any]] = {}
    for item in metrics:
        key = (item["experiment"], item["feature"])
        target = aggregate.setdefault(
            key,
            {
                "experiment": item["experiment"],
                "feature": item["feature"],
                "unit": item["unit"],
                "rows": 0,
                "raw_nonblank": 0,
                "parsed_values": 0,
                "parse_failures": 0,
                "missing_values": 0,
                "impossible_values": 0,
            },
        )
        for field in (
            "rows",
            "raw_nonblank",
            "parsed_values",
            "parse_failures",
            "missing_values",
            "impossible_values",
        ):
            target[field] += item[field]
    result = []
    for item in aggregate.values():
        raw_count = item["raw_nonblank"]
        item["parse_success_rate"] = item["parsed_values"] / raw_count if raw_count else None
        result.append(item)
    return sorted(result, key=lambda item: (item["experiment"], item["feature"]))


def clean_all(input_dir: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    paths = sorted(input_dir.glob("exp*_*.csv"))
    if not paths:
        raise FileNotFoundError(f"No experiment CSV files found in {input_dir}")

    frames: list[pd.DataFrame] = []
    sources: list[dict[str, Any]] = []
    feature_metrics: list[dict[str, Any]] = []
    for path in paths:
        frame, source_summary, source_feature_metrics = clean_obd_file(path)
        frames.append(frame)
        sources.append(source_summary)
        feature_metrics.extend(source_feature_metrics)
    cleaned = pd.concat(frames, ignore_index=True)

    experiment_feature_metrics = _aggregate_feature_metrics(feature_metrics)
    nonempty_metrics = [item for item in experiment_feature_metrics if item["raw_nonblank"]]
    minimum_parse_rate = min(
        (item["parse_success_rate"] for item in nonempty_metrics),
        default=None,
    )
    trace_columns = ["source_file", "source_row", "actual_vehicle_id", "driver_id", "trip_id"]
    quality_report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "canonical_units": CANONICAL_UNITS,
        "possible_ranges": {
            feature: {"minimum": bounds[0], "maximum": bounds[1]}
            for feature, bounds in POSSIBLE_RANGES.items()
        },
        "trip_boundary_rules": [
            "source VEHICLE_ID change",
            "gap caused by removed blank source rows",
            "engine runtime decrease greater than 5 seconds",
            "timestamp gap greater than 30 minutes or timestamp moving backward",
        ],
        "identity_assumptions": [
            "Experiment 1 has one inferred driver per car because no separate driver field exists.",
            "Experiments 2 and 3 treat source VEHICLE_ID as driver/session ID and use one shared-car ID per experiment.",
            "The shared cars in experiments 2 and 3 are not assumed to be the same physical car.",
        ],
        "totals": {
            "raw_rows": sum(item["raw_rows"] for item in sources),
            "completely_blank_rows_dropped": sum(
                item["completely_blank_rows_dropped"] for item in sources
            ),
            "retained_rows": int(len(cleaned)),
            "experiments": int(cleaned["experiment"].nunique()),
            "actual_vehicles": int(cleaned["actual_vehicle_id"].nunique()),
            "drivers": int(cleaned["driver_id"].nunique()),
            "trips": int(cleaned["trip_id"].nunique()),
            "rows_with_any_impossible_value": int(cleaned["any_impossible"].sum()),
        },
        "acceptance_checks": {
            "raw_rows_reconcile": all(item["row_count_reconciles"] for item in sources),
            "traceability_and_identity_complete": not cleaned[trace_columns].isna().any().any(),
            "minimum_nonempty_feature_parse_success_rate": minimum_parse_rate,
            "at_least_99_percent_parse_success": (
                minimum_parse_rate is not None and minimum_parse_rate >= 0.99
            ),
            "canonical_units_documented": set(CANONICAL_UNITS) == set(CANONICAL_SENSORS),
            "missing_values_have_observed_masks": all(
                cleaned[feature].isna().equals(cleaned[f"{feature}_observed"].eq(0))
                for feature in CANONICAL_SENSORS
            ),
        },
        "by_experiment": sources,
        "by_experiment_and_feature": experiment_feature_metrics,
        "by_experiment_vehicle_and_feature": feature_metrics,
    }
    return cleaned, quality_report


def write_outputs(
    cleaned: pd.DataFrame,
    quality_report: dict[str, Any],
    output_path: Path,
    report_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(output_path, index=False, date_format="%Y-%m-%dT%H:%M:%S.%fZ")
    report_path.write_text(json.dumps(quality_report, indent=2, allow_nan=False) + "\n")


def build_parser() -> argparse.ArgumentParser:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=script_dir.parent / "archive (2)",
        help="Directory containing the three raw experiment CSV files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=script_dir / "artifacts" / "cleaned_obd.csv",
        help="Canonical output CSV.",
    )
    parser.add_argument(
        "--quality-report",
        type=Path,
        default=script_dir / "artifacts" / "data_quality.json",
        help="JSON data-quality report.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    cleaned, quality_report = clean_all(args.input_dir.expanduser().resolve())
    write_outputs(
        cleaned,
        quality_report,
        args.output.expanduser().resolve(),
        args.quality_report.expanduser().resolve(),
    )
    totals = quality_report["totals"]
    checks = quality_report["acceptance_checks"]
    print(f"Wrote {totals['retained_rows']:,} canonical rows to {args.output}")
    print(f"Wrote data-quality report to {args.quality_report}")
    print(f"Trips: {totals['trips']:,}")
    print(f"Rows with impossible values: {totals['rows_with_any_impossible_value']:,}")
    print(
        "Minimum parse success for populated experiment/features: "
        f"{checks['minimum_nonempty_feature_parse_success_rate']:.2%}"
    )


if __name__ == "__main__":
    main()

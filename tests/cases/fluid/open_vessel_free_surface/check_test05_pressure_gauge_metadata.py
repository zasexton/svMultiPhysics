#!/usr/bin/env python3
"""Check SPHERIC Test05 pressure-gauge benchmark metadata."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CASES = (
    ROOT / "unfitted_level_set" / "spheric_test05_wet_bed_d18",
    ROOT / "unfitted_level_set" / "spheric_test05_wet_bed_d38",
)
PREVIOUS_INVALID_D18_GAUGE = {
    "node_id": 279,
    "initial_phi": -0.001806,
    "full_volume_hydrostatic_pressure": 17.6869,
    "hydrostatic_error_range": [-17.6869, 0.0],
}
TOL = 1.0e-9


def _load_pressure_gauge(path: Path) -> tuple[int, float]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise ValueError(f"expected exactly one pressure gauge row in {path}")
    return int(rows[0]["node_id"]), float(rows[0]["pressure"])


def _expected_verification(gauge: dict[str, object]) -> dict[str, object]:
    current_pressure = float(gauge["expected_initial_hydrostatic_pressure"])
    previous_pressure = PREVIOUS_INVALID_D18_GAUGE["full_volume_hydrostatic_pressure"]
    previous_range = PREVIOUS_INVALID_D18_GAUGE["hydrostatic_error_range"]
    return {
        "current_prescribed_pressure_matches_initial_hydrostatic": True,
        "initial_pressure_error_after_constraint": 0.0,
        "previous_invalid_d18_node_id": PREVIOUS_INVALID_D18_GAUGE["node_id"],
        "previous_invalid_d18_initial_phi": PREVIOUS_INVALID_D18_GAUGE["initial_phi"],
        "previous_invalid_d18_full_volume_hydrostatic_pressure": previous_pressure,
        "previous_invalid_d18_hydrostatic_error_range": previous_range,
        "current_pressure_matches_previous_invalid_offset": False,
        "current_pressure_matches_previous_invalid_error_range": (
            previous_range[0] <= current_pressure <= previous_range[1]
        ),
        "current_pressure_minus_previous_invalid_pressure": current_pressure - previous_pressure,
    }


def _nearly_equal(left: object, right: object) -> bool:
    if isinstance(left, float) or isinstance(right, float):
        return abs(float(left) - float(right)) <= TOL
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(_nearly_equal(a, b) for a, b in zip(left, right))
    return left == right


def check_case(case_dir: Path) -> dict[str, object]:
    metadata_path = case_dir / "benchmark.json"
    gauge_path = case_dir / "pressure_gauge.csv"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    gauge = metadata["pressure_gauge"]
    verification = metadata.get("pressure_gauge_verification")
    if not isinstance(verification, dict):
        raise ValueError(f"{metadata_path} is missing pressure_gauge_verification")

    csv_node, csv_pressure = _load_pressure_gauge(gauge_path)
    expected_pressure = float(gauge["expected_initial_hydrostatic_pressure"])
    if csv_node != int(gauge["node_id"]):
        raise ValueError(f"{gauge_path} node_id does not match benchmark metadata")
    if abs(csv_pressure - expected_pressure) > TOL:
        raise ValueError(f"{gauge_path} pressure does not match expected hydrostatic value")

    expected = _expected_verification(gauge)
    for key, expected_value in expected.items():
        if key not in verification:
            raise ValueError(f"{metadata_path} is missing verification field {key}")
        if not _nearly_equal(verification[key], expected_value):
            raise ValueError(
                f"{metadata_path} field {key} has {verification[key]!r}, "
                f"expected {expected_value!r}"
            )

    if not verification["current_prescribed_pressure_matches_initial_hydrostatic"]:
        raise ValueError(f"{metadata_path} does not prescribe the initial hydrostatic gauge pressure")
    if verification["current_pressure_matches_previous_invalid_offset"]:
        raise ValueError(f"{metadata_path} still matches the previous invalid D18 pressure offset")
    if verification["current_pressure_matches_previous_invalid_error_range"]:
        raise ValueError(f"{metadata_path} still lies in the previous invalid D18 error range")

    return {
        "case": case_dir.name,
        "node_id": csv_node,
        "pressure": csv_pressure,
        "previous_invalid_pressure": PREVIOUS_INVALID_D18_GAUGE["full_volume_hydrostatic_pressure"],
        "pressure_difference": verification["current_pressure_minus_previous_invalid_pressure"],
    }


def main() -> None:
    report = [check_case(case_dir) for case_dir in CASES]
    print(json.dumps({"checked_cases": report}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

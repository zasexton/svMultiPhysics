import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = (
    ROOT
    / "tests"
    / "cases"
    / "fluid"
    / "run_free_surface_wp10_rising_bubble_reference.py"
)
REGISTRY_PATH = RUNNER_PATH.with_name(
    "free_surface_wp10_literature_registry_v3.json"
)


def load_runner():
    specification = importlib.util.spec_from_file_location(
        "free_surface_wp10_rising_bubble_reference_runner",
        RUNNER_PATH,
    )
    module = importlib.util.module_from_spec(specification)
    assert specification.loader is not None
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def canonical_registry():
    return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))


def test_registry_extends_v2_and_pins_the_open_experimental_table():
    runner = load_runner()
    registry = runner.validate_registry(REGISTRY_PATH)

    assert registry["schema_version"] == 3
    assert registry["registry_id"] == "free_surface_wp10_literature_v3"
    assert registry["extends"] == {
        "registry_id": "free_surface_wp10_literature_v2",
        "repository_path": (
            "tests/cases/fluid/"
            "free_surface_wp10_literature_registry_v2.json"
        ),
        "sha256": (
            "e59e3abd4da037cdf1ed2d939c20a4cf316b7a9b7ceb58af8e3741806a7b7589"
        ),
        "mutation_policy": "PRESERVE_V2_BYTE_FOR_BYTE",
    }

    source = registry["sources"][0]
    assert source["citation"]["doi"] == "10.1007/s00348-023-03746-0"
    assert source["asset"]["sha256"] == (
        "d44ad9f5e8dda1cfa9df7912994d6436c32fc5a7017bea760a956aa628245bda"
    )
    assert source["asset"]["bytes"] == 210813
    assert source["access"]["license"] == "CC-BY-4.0"
    assert source["reference_locations"]["reported_values"] == "Table 3"


def test_reported_table_values_reproduce_dimensionless_groups_with_rounding():
    runner = load_runner()
    registry = runner.validate_registry(REGISTRY_PATH)
    evaluated = runner.evaluate_reported_cases(registry)

    assert [case["diameter_mm"] for case in evaluated] == [
        2.4,
        4.0,
        6.0,
        9.6,
    ]
    assert [case["reported_average_velocity_m_per_s"] for case in evaluated] == [
        0.32,
        0.27,
        0.24,
        0.25,
    ]
    for case in evaluated:
        for group in ("reynolds", "weber", "eotvos", "galilei", "morton"):
            assert case[f"relative_error_{group}"] <= 0.035


def test_reference_keeps_point_values_separate_from_acceptance_bands():
    runner = load_runner()
    registry = runner.validate_registry(REGISTRY_PATH)
    contract = registry["experimental_reference"]

    assert contract["reported_measurement_uncertainty"] is None
    assert contract["solver_acceptance_band"] is None
    assert contract["gate_policy"] == (
        "REPORT_POINTS_AND_NUMERICAL_UNCERTAINTY_NO_RELEASE_BAND"
    )
    assert contract["raw_dataset_access"] == "AVAILABLE_FROM_AUTHORS_ON_REQUEST"


def test_registry_rejects_unreviewed_point_or_gate_change(tmp_path):
    runner = load_runner()
    registry = copy.deepcopy(canonical_registry())
    registry["reference_cases"][0]["average_velocity_m_per_s"] = 0.31
    path = tmp_path / "registry.json"
    path.write_text(json.dumps(registry), encoding="utf-8")

    with pytest.raises(ValueError, match="reference case contract changed"):
        runner.validate_registry(path)

    registry = copy.deepcopy(canonical_registry())
    registry["experimental_reference"]["solver_acceptance_band"] = {
        "relative": 0.1
    }
    path.write_text(json.dumps(registry), encoding="utf-8")
    with pytest.raises(ValueError, match="experimental reference contract changed"):
        runner.validate_registry(path)


def test_validate_only_reports_reference_points_without_a_release_gate():
    result = subprocess.run(
        [sys.executable, str(RUNNER_PATH), "--validate-only"],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True,
    )
    assert json.loads(result.stdout) == {
        "experimental_case_count": 4,
        "outcome": "PASS",
        "registry_id": "free_surface_wp10_literature_v3",
        "release_gate_count": 0,
        "source_count": 1,
        "v2_preserved": True,
    }

import copy
import importlib.util
import json
import math
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
    / "run_free_surface_wp10_capillary_wave_reference.py"
)
REGISTRY_PATH = RUNNER_PATH.with_name(
    "free_surface_wp10_literature_registry_v2.json"
)


def load_runner():
    specification = importlib.util.spec_from_file_location(
        "free_surface_wp10_capillary_wave_reference_runner",
        RUNNER_PATH,
    )
    module = importlib.util.module_from_spec(specification)
    assert specification.loader is not None
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def canonical_registry():
    return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))


def test_registry_extends_v1_without_rewriting_its_blocked_record():
    runner = load_runner()
    registry = runner.validate_registry(REGISTRY_PATH)

    assert registry["schema_version"] == 2
    assert registry["registry_id"] == "free_surface_wp10_literature_v2"
    assert registry["extends"] == {
        "registry_id": "free_surface_wp10_literature_v1",
        "repository_path": (
            "tests/cases/fluid/"
            "free_surface_wp10_literature_registry_v1.json"
        ),
        "sha256": (
            "33953b7eab91540628340043faf01bfa1765be0be8e7257064269af553660f5d"
        ),
        "mutation_policy": "PRESERVE_V1_BYTE_FOR_BYTE",
    }

    sources = {source["id"]: source for source in registry["sources"]}
    article = sources["denner_2016_capillary_dispersion"]
    assert article["citation"]["doi"] == "10.1103/PhysRevE.94.023110"
    assert article["asset"]["sha256"] == (
        "07d63d4fcd6a10c82d17414f685cca80857e11c6af1253ce4333a4df8ca24853"
    )
    assert article["asset"]["bytes"] == 520532
    assert article["access"]["license"] == "CC-BY-3.0"

    script = sources["denner_2016_prosperetti_reference_script"]
    assert script["citation"]["doi"] == "10.5281/zenodo.166716"
    assert script["asset"]["md5"] == "c49622cf98249f13d5a315a0e28b1a86"
    assert script["asset"]["sha256"] == (
        "c94a19d3870393cc2393c0d300a07c94b1dea8e08121171a40595792cb6a52f1"
    )
    assert script["asset"]["bytes"] == 3544

    contract = registry["analytical_reference"]
    assert contract["equation_location"] == "Equations 12-13"
    assert contract["complex_erfc_argument"] == "z_i_times_sqrt_t"
    assert contract["author_script_role"] == (
        "CHECKSUM_PINNED_CORROBORATION_NOT_NUMERICAL_ORACLE"
    )
    assert contract["author_script_difference"] == (
        "uses_real_part_of_z_i_in_erfc_argument"
    )


def test_published_case_d_reference_checkpoints_are_reproduced():
    runner = load_runner()
    registry = runner.validate_registry(REGISTRY_PATH)
    case = runner.reference_case(registry, "denner_case_d")
    result = runner.evaluate_dimensionless_times(
        case,
        [0.0, 0.25, 0.5, 1.0, 2.0, math.pi],
    )

    assert result["inviscid_angular_frequency"] == pytest.approx(
        133639.87192396095,
        rel=2.0e-14,
    )
    assert result["normalized_amplitude"] == pytest.approx(
        [
            1.0,
            0.9692031250269033,
            0.8798093272804911,
            0.5561768692261875,
            -0.33032144562011306,
            -0.8586740656216184,
        ],
        rel=2.0e-12,
        abs=2.0e-12,
    )


def test_published_two_fluid_case_a_uses_both_phases_and_is_linear_in_amplitude():
    runner = load_runner()
    registry = runner.validate_registry(REGISTRY_PATH)
    case = runner.reference_case(registry, "denner_case_a_selected_wave_number")
    result = runner.evaluate_dimensionless_times(case, [0.0, 1.0, math.pi])

    assert case["upper_fluid"]["density"] == 5.0
    assert case["lower_fluid"]["density"] == 5.0
    assert case["upper_fluid"]["dynamic_viscosity"] == 0.7
    assert case["lower_fluid"]["dynamic_viscosity"] == 0.7
    assert result["normalized_amplitude"] == pytest.approx(
        [1.0, 0.5700063394492536, -0.8853920825880313],
        rel=2.0e-12,
        abs=2.0e-12,
    )

    doubled = copy.deepcopy(case)
    doubled["initial_amplitude"] *= 2.0
    doubled_result = runner.evaluate_dimensionless_times(
        doubled, [0.0, 1.0, math.pi]
    )
    assert doubled_result["amplitude"] == pytest.approx(
        [2.0 * value for value in result["amplitude"]],
        rel=2.0e-12,
        abs=2.0e-12,
    )


def test_reference_rejects_unequal_kinematic_viscosities():
    runner = load_runner()
    registry = runner.validate_registry(REGISTRY_PATH)
    case = copy.deepcopy(runner.reference_case(registry, "denner_case_a_selected_wave_number"))
    case["upper_fluid"]["dynamic_viscosity"] *= 1.01

    with pytest.raises(ValueError, match="equal kinematic viscosity"):
        runner.evaluate_dimensionless_times(case, [0.0, 1.0])


def test_registry_rejects_unreviewed_equation_or_checkpoint_change(tmp_path):
    runner = load_runner()
    registry = canonical_registry()
    registry["analytical_reference"]["complex_erfc_argument"] = (
        "real_z_i_times_sqrt_t"
    )
    path = tmp_path / "registry.json"
    path.write_text(json.dumps(registry), encoding="utf-8")

    with pytest.raises(ValueError, match="analytical reference contract changed"):
        runner.validate_registry(path)


def test_validate_only_reports_executable_capillary_reference():
    result = subprocess.run(
        [sys.executable, str(RUNNER_PATH), "--validate-only"],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True,
    )
    summary = json.loads(result.stdout)
    assert summary == {
        "executable_case_count": 2,
        "outcome": "PASS",
        "registry_id": "free_surface_wp10_literature_v2",
        "source_count": 2,
        "v1_preserved": True,
    }

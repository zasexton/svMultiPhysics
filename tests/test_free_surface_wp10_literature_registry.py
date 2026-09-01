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
    / "run_free_surface_wp10_literature_registry.py"
)
REGISTRY_PATH = RUNNER_PATH.with_name(
    "free_surface_wp10_literature_registry_v1.json"
)


def load_runner():
    specification = importlib.util.spec_from_file_location(
        "free_surface_wp10_literature_registry_runner",
        RUNNER_PATH,
    )
    module = importlib.util.module_from_spec(specification)
    assert specification.loader is not None
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def canonical_registry():
    return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))


def validate_mutation(runner, tmp_path, mutation):
    document = copy.deepcopy(canonical_registry())
    mutation(document)
    path = tmp_path / "registry.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return runner.validate_registry(path)


def test_canonical_registry_pins_only_executable_reference_data():
    runner = load_runner()
    registry = runner.validate_registry(REGISTRY_PATH)

    assert registry["schema_version"] == 1
    assert registry["registry_id"] == "free_surface_wp10_literature_v1"
    assert registry["status"] == "FROZEN_REFERENCE_CONTRACT"
    assert registry["verified_date"] == "2026-08-31"

    sources = {source["id"]: source for source in registry["sources"]}
    assert set(sources) == {
        "hysing_2009_bubble_benchmark",
        "prosperetti_1981_two_viscous_fluids",
    }

    hysing = sources["hysing_2009_bubble_benchmark"]
    assert hysing["citation"]["doi"] == "10.1002/fld.1934"
    assert hysing["asset"]["status"] == "EXTERNALLY_PINNED"
    assert hysing["asset"]["sha256"] == (
        "2c849a2c135a152b268af25e9c5e91ee3c7aed4048d205e2b48a1222d1600794"
    )
    assert hysing["asset"]["bytes"] == 1336058
    assert hysing["access"]["redistribution"] == "NOT_INCLUDED"
    assert hysing["disposition"] == "EXECUTABLE_INTERCODE_REFERENCE"

    prosperetti = sources["prosperetti_1981_two_viscous_fluids"]
    assert prosperetti["citation"]["doi"] == "10.1063/1.863522"
    assert prosperetti["asset"]["status"] == "SOURCE_ASSET_UNAVAILABLE"
    assert prosperetti["asset"]["sha256"] is None
    assert prosperetti["disposition"] == "BLOCKED_QUANTITATIVE_GATE"

    benchmarks = {
        benchmark["id"]: benchmark for benchmark in registry["benchmarks"]
    }
    assert set(benchmarks) == {
        "two_fluid_capillary_wave",
        "hysing_case_1",
        "hysing_case_2",
    }
    case_1 = benchmarks["hysing_case_1"]
    assert case_1["reference_source"] == "hysing_2009_bubble_benchmark"
    assert case_1["reference_location"] == "Table XII and pages 17-18"
    assert case_1["published_reference_bands"] == {
        "minimum_circularity": {"lower": 0.9011, "upper": 0.9013},
        "final_center_of_mass_y": {"lower": 1.08, "upper": 1.082},
        "maximum_rise_velocity": {"lower": 0.2417, "upper": 0.2421},
        "maximum_rise_velocity_time": {"lower": 0.921, "upper": 0.932},
    }
    assert case_1["gate_policy"] == "FINEST_PAIR_INSIDE_PUBLISHED_BANDS"

    case_2 = benchmarks["hysing_case_2"]
    assert case_2["pre_breakup_gate"]["first_rise_velocity_maximum"] == {
        "lower": 0.24,
        "upper": 0.26,
    }
    assert case_2["pre_breakup_gate"]["first_rise_velocity_maximum_time"] == {
        "lower": 0.71,
        "upper": 0.75,
    }
    assert case_2["post_breakup_policy"] == "REPORT_INTERCODE_RANGE_ONLY"
    assert case_2["post_breakup_shape_gate"] is None

    capillary_wave = benchmarks["two_fluid_capillary_wave"]
    assert capillary_wave["reference_source"] == (
        "prosperetti_1981_two_viscous_fluids"
    )
    assert capillary_wave["gate_policy"] == "BLOCKED_UNTIL_SOURCE_PINNED"
    assert capillary_wave["quantitative_gate"] is None

    refinement = registry["common_refinement_contract"]
    assert len(refinement["spatial_levels"]) == 3
    assert len(refinement["temporal_levels"]) == 3
    assert len(refinement["cut_offset_fractions"]) >= 3
    assert refinement["material_side_reversal"] == [False, True]
    assert refinement["mpi_ranks"] == [1, 2, 4]


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda value: value["sources"][0]["asset"].__setitem__(
                "sha256", "0" * 64
            ),
            "source contract changed",
        ),
        (
            lambda value: value["sources"][1].__setitem__(
                "disposition", "EXECUTABLE_INTERCODE_REFERENCE"
            ),
            "source contract changed",
        ),
        (
            lambda value: value["benchmarks"][2].__setitem__(
                "post_breakup_shape_gate", {"maximum_error": 0.1}
            ),
            "benchmark contract changed",
        ),
        (
            lambda value: value["benchmarks"][0].__setitem__(
                "quantitative_gate", {"maximum_error": 0.01}
            ),
            "benchmark contract changed",
        ),
        (
            lambda value: value["common_refinement_contract"].__setitem__(
                "spatial_levels", [40, 80]
            ),
            "refinement contract changed",
        ),
    ],
)
def test_registry_rejects_unreviewed_promotion(
    tmp_path, mutation, message
):
    runner = load_runner()
    with pytest.raises(ValueError, match=message):
        validate_mutation(runner, tmp_path, mutation)


def test_duplicate_json_key_is_rejected(tmp_path):
    runner = load_runner()
    path = tmp_path / "duplicate.json"
    path.write_text(
        REGISTRY_PATH.read_text(encoding="utf-8").replace(
            '  "status": ',
            '  "status": "WEAKENED",\n  "status": ',
            1,
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate JSON key: status"):
        runner.validate_registry(path)


def test_validate_only_reports_blocked_capillary_wave_gate():
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
        "benchmark_count": 3,
        "blocked_quantitative_gate_count": 1,
        "executable_reference_count": 1,
        "outcome": "PASS",
        "registry_id": "free_surface_wp10_literature_v1",
        "source_count": 2,
    }

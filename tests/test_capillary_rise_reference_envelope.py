import copy
import importlib.util
import json
from pathlib import Path
import sys

import pytest


def _repository() -> Path:
    return Path(__file__).resolve().parents[1]


def _builder_path() -> Path:
    return (
        _repository()
        / "tests"
        / "cases"
        / "fluid"
        / "open_vessel_free_surface"
        / "build_capillary_rise_reference_envelope.py"
    )


def _comparison_path() -> Path:
    return (
        _repository()
        / "tests"
        / "cases"
        / "fluid"
        / "free_surface_wp5_capillary_rise_comparison_v1.json"
    )


def _load_builder():
    specification = importlib.util.spec_from_file_location(
        "free_surface_capillary_rise_reference_envelope",
        _builder_path(),
    )
    assert specification is not None
    assert specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def test_comparison_contract_binds_the_source_registry_and_remains_open():
    builder = _load_builder()
    comparison, reference = builder.load_comparison(_comparison_path())

    assert comparison["primary_methods"] == [
        "interTrackFoam",
        "BoSSS",
        "FS3D",
    ]
    assert comparison["sensitivity_method"]["method"] == "interFoam"
    assert comparison["common_grid"]["point_count"] == 691
    assert reference["reference_id"] == (
        "gruending_2020_capillary_rise_omega1_resolved_slip"
    )
    assert comparison["qualification_disposition"]["candidate_executed"] is False
    assert comparison["qualification_disposition"]["wp5_closed"] is False


def test_duplicate_published_times_are_collapsed_without_a_time_shift():
    builder = _load_builder()
    rows = [
        (-0.1, 8.0),
        (0.0, 9.0),
        (0.0, 11.0),
        (0.2, 13.0),
    ]

    collapsed = builder.collapse_duplicate_times(rows)

    assert collapsed == [(-0.1, 8.0), (0.0, 10.0), (0.2, 13.0)]


def test_piecewise_linear_interpolation_never_extrapolates():
    builder = _load_builder()
    rows = [(-0.1, 8.0), (0.0, 10.0), (0.2, 14.0)]

    assert builder.interpolate_curve(rows, -0.1) == 8.0
    assert builder.interpolate_curve(rows, 0.1) == 12.0
    assert builder.interpolate_curve(rows, 0.2) == 14.0
    with pytest.raises(ValueError, match="outside published support"):
        builder.interpolate_curve(rows, -0.2)
    with pytest.raises(ValueError, match="outside published support"):
        builder.interpolate_curve(rows, 0.3)


def test_common_grid_uses_the_exact_declared_point_count():
    builder = _load_builder()
    grid = builder.common_grid(
        {"start_s": 0.0, "end_s": 0.004, "step_s": 0.001, "point_count": 5}
    )

    assert grid == [0.0, 0.001, 0.002, 0.003, 0.004]


def test_envelope_writer_refuses_to_replace_evidence(tmp_path):
    builder = _load_builder()
    output = tmp_path / "envelope.csv"
    rows = [{"time_s": 0.0, "reference_center_mm": 10.0}]

    builder.write_envelope(output, rows)

    original = output.read_bytes()
    with pytest.raises(FileExistsError):
        builder.write_envelope(output, rows)
    assert output.read_bytes() == original


def test_frozen_statistics_reject_post_selection_drift():
    builder = _load_builder()
    comparison = json.loads(_comparison_path().read_text(encoding="utf-8"))
    expected = comparison["frozen_statistics"]
    observed = copy.deepcopy(expected)

    builder.validate_frozen_statistics(observed, expected)
    observed["reference_center_peak_mm"] += 1.0e-6
    with pytest.raises(ValueError, match="reference_center_peak_mm"):
        builder.validate_frozen_statistics(observed, expected)

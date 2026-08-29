import math
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
CASE_DIRECTORY = ROOT / "tests" / "cases" / "fluid"
if str(CASE_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(CASE_DIRECTORY))

import free_surface_convergence as convergence


def samples(values):
    return [
        {"label": f"n{resolution}", "h": 1.0 / resolution, "value": value}
        for resolution, value in zip((8, 16, 32, 64), values)
    ]


def analyze(values, **overrides):
    options = {
        "reference_value": 2.0,
        "normalization": 2.0,
        "minimum_observed_order": 0.8,
        "finest_relative_error_limit": 0.01,
        "finest_gci_limit": 0.01,
    }
    options.update(overrides)
    return convergence.analyze_gci_sequence(samples(values), **options)


def test_second_order_sequence_reports_richardson_and_gci():
    record = analyze([2.16, 2.04, 2.01, 2.0025])

    assert record["status"] == "PASS"
    assert record["passed"] is True
    assert record["observed_order"] == pytest.approx(2.0)
    assert record["richardson_extrapolated_value"] == pytest.approx(2.0)
    assert record["finest_relative_error"] == pytest.approx(0.00125)
    assert record["finest_gci"] == pytest.approx(0.0015625)
    assert record["asymptotic_tail_labels"] == ["n16", "n32", "n64"]
    assert record["sample_relative_errors"] == pytest.approx(
        [0.08, 0.02, 0.005, 0.00125])


def test_three_level_nonmonotone_sequence_requires_another_level():
    record = convergence.analyze_gci_sequence(
        samples([2.08, 2.02, 2.03])[:3],
        reference_value=2.0,
        normalization=2.0,
        minimum_observed_order=0.8,
        finest_relative_error_limit=0.02,
        finest_gci_limit=0.02,
    )

    assert record["status"] == "ADDITIONAL_LEVEL_REQUIRED"
    assert record["gate_failures"] == ["asymptotic_tail_not_established"]


def test_fourth_level_can_establish_tail_after_coarse_reversal():
    record = analyze([2.08, 2.10, 2.025, 2.00625])

    assert record["status"] == "PASS"
    assert record["observed_order"] == pytest.approx(2.0)


def test_four_level_nonmonotone_tail_fails():
    record = analyze([2.08, 2.02, 2.03, 2.01])

    assert record["status"] == "FAIL"
    assert record["gate_failures"] == [
        "four_level_asymptotic_tail_not_established"]


def test_direct_error_and_gci_gates_are_independent():
    error_failure = analyze(
        [2.64, 2.32, 2.16, 2.08],
        finest_relative_error_limit=0.02,
        finest_gci_limit=0.1,
    )
    gci_failure = analyze(
        [2.08, 2.04, 2.02, 2.01],
        finest_relative_error_limit=0.01,
        finest_gci_limit=0.005,
    )

    assert error_failure["gate_failures"] == ["finest_relative_error"]
    assert gci_failure["gate_failures"] == ["finest_gci"]


def test_minimum_order_gate_is_not_replaced_by_finest_error():
    record = analyze(
        [2.03, 2.02, 2.014142135623731, 2.01],
        minimum_observed_order=0.8,
        finest_relative_error_limit=0.01,
        finest_gci_limit=1.0,
    )

    assert record["finest_relative_error"] < 0.01
    assert record["gate_failures"] == ["minimum_observed_order"]


def test_roundoff_sequence_is_accepted_only_when_direct_gate_passes():
    exact = convergence.analyze_gci_sequence(
        samples([2.0, 2.0, 2.0, 2.0]),
        reference_value=2.0,
        normalization=2.0,
        minimum_observed_order=0.8,
        finest_relative_error_limit=1.0e-13,
        finest_gci_limit=1.0e-13,
    )

    assert exact["status"] == "PASS"
    assert exact["numerically_exact"] is True
    assert exact["finest_gci"] == 0.0
    assert exact["richardson_relative_error"] == 0.0

    outside_direct_gate = convergence.analyze_gci_sequence(
        samples([2.0 + 1.0e-15] * 4),
        reference_value=2.0,
        normalization=1.0,
        minimum_observed_order=0.8,
        finest_relative_error_limit=1.0e-16,
        finest_gci_limit=1.0e-13,
    )
    assert outside_direct_gate["status"] == "FAIL"
    assert outside_direct_gate["gate_failures"] == ["finest_relative_error"]


def test_extremely_high_order_does_not_overflow_extrapolation():
    record = convergence.analyze_gci_sequence(
        samples([2.0, 1.0, 1.0e-320, 0.0]),
        reference_value=0.0,
        normalization=1.0,
        minimum_observed_order=0.8,
        finest_relative_error_limit=0.01,
        finest_gci_limit=0.01,
        exactness_floor=0.0,
    )

    assert record["status"] == "PASS"
    assert record["richardson_extrapolated_value"] == 0.0
    assert record["finest_gci"] == 0.0


def test_offset_envelope_uses_the_worst_sequence():
    envelope = convergence.analyze_offset_envelope(
        {
            "center": samples([2.16, 2.04, 2.01, 2.0025]),
            "translated": samples([2.32, 2.08, 2.02, 2.005]),
        },
        reference_value=2.0,
        normalization=2.0,
        minimum_observed_order=0.8,
        finest_relative_error_limit=0.01,
        finest_gci_limit=0.01,
    )

    assert envelope["status"] == "PASS"
    assert envelope["sequence_count"] == 2
    assert envelope["maximum_finest_relative_error"] == pytest.approx(0.0025)
    assert envelope["minimum_observed_order"] == pytest.approx(2.0)


def test_offset_envelope_propagates_additional_level_before_failure():
    envelope = convergence.analyze_offset_envelope(
        {
            "center": samples([2.16, 2.04, 2.01])[:3],
            "translated": samples([2.08, 2.02, 2.03])[:3],
        },
        reference_value=2.0,
        normalization=2.0,
        minimum_observed_order=0.8,
        finest_relative_error_limit=0.02,
        finest_gci_limit=0.02,
    )

    assert envelope["status"] == "ADDITIONAL_LEVEL_REQUIRED"
    assert envelope["additional_level_required"] == ["translated"]


@pytest.mark.parametrize(
    "mutation, message",
    [
        ([{"h": 0.1, "value": 1.0}], "at least three"),
        (
            [
                {"h": 1.0 / resolution, "value": 1.0}
                for resolution in (8, 16, 32, 64, 128)
            ],
            "at most four",
        ),
        (
            [
                {"h": 0.1, "value": 1.0},
                {"h": 0.05, "value": 1.0},
                {"h": 0.025, "value": math.nan},
            ],
            "must be finite",
        ),
        (
            [
                {"h": 0.1, "value": 1.0},
                {"h": 0.06, "value": 1.0},
                {"h": 0.025, "value": 1.0},
            ],
            "uniform refinement ratio",
        ),
    ],
)
def test_invalid_sequences_fail_closed(mutation, message):
    with pytest.raises(convergence.ConvergenceError, match=message):
        convergence.analyze_gci_sequence(
            mutation,
            reference_value=0.0,
            normalization=1.0,
            minimum_observed_order=0.8,
            finest_relative_error_limit=0.01,
            finest_gci_limit=0.01,
        )

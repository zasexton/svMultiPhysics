#!/usr/bin/env python3
"""Deterministic refinement and Grid Convergence Index calculations."""

from __future__ import annotations

import math
from typing import Any, Sequence


DEFAULT_SAFETY_FACTOR = 1.25
DEFAULT_RATIO_RELATIVE_TOLERANCE = 1.0e-10
DEFAULT_EXACTNESS_FLOOR = 1.0e-14


class ConvergenceError(ValueError):
    """Raised when a refinement sequence violates its frozen contract."""


def _finite_number(value: Any, context: str) -> float:
    if (isinstance(value, bool) or not isinstance(value, (int, float)) or
            not math.isfinite(float(value))):
        raise ConvergenceError(f"{context} must be finite")
    return float(value)


def normalized_samples(
        samples: Sequence[dict[str, Any]],
        *,
        h_key: str = "h",
        value_key: str = "value",
) -> list[dict[str, Any]]:
    """Return a coarsest-to-finest copy for a three- or four-level study."""
    if len(samples) < 3:
        raise ConvergenceError("a refinement sequence requires at least three levels")
    if len(samples) > 4:
        raise ConvergenceError("a refinement sequence permits at most four levels")
    result: list[dict[str, Any]] = []
    labels: set[str] = set()
    for index, sample in enumerate(samples):
        if not isinstance(sample, dict):
            raise ConvergenceError(f"sample {index} must be an object")
        h = _finite_number(sample.get(h_key), f"sample {index} mesh spacing")
        value = _finite_number(sample.get(value_key), f"sample {index} value")
        if h <= 0.0:
            raise ConvergenceError(f"sample {index} mesh spacing must be positive")
        label = str(sample.get("label", f"level_{index}"))
        if not label or label in labels:
            raise ConvergenceError("refinement sample labels must be unique")
        labels.add(label)
        result.append({**sample, "label": label, h_key: h, value_key: value})
    result.sort(key=lambda sample: float(sample[h_key]), reverse=True)
    spacings = [float(sample[h_key]) for sample in result]
    if any(coarse <= fine for coarse, fine in zip(spacings[:-1], spacings[1:])):
        raise ConvergenceError("mesh spacings must be distinct and strictly refined")
    return result


def _uniform_refinement_ratio(
        samples: Sequence[dict[str, Any]],
        *,
        h_key: str,
        relative_tolerance: float,
) -> float:
    tolerance = _finite_number(
        relative_tolerance, "refinement-ratio relative tolerance")
    if tolerance < 0.0:
        raise ConvergenceError(
            "refinement-ratio relative tolerance must be nonnegative")
    ratios = [
        float(coarse[h_key]) / float(fine[h_key])
        for coarse, fine in zip(samples[:-1], samples[1:])
    ]
    if any(ratio <= 1.0 for ratio in ratios):
        raise ConvergenceError("every refinement ratio must exceed one")
    reference = ratios[-1]
    if any(
            abs(ratio - reference) >
            tolerance * max(1.0, abs(reference))
            for ratio in ratios):
        raise ConvergenceError(
            "Grid Convergence Index evaluation requires a uniform refinement ratio")
    return reference


def analyze_gci_sequence(
        samples: Sequence[dict[str, Any]],
        *,
        reference_value: float,
        normalization: float,
        minimum_observed_order: float,
        finest_relative_error_limit: float,
        finest_gci_limit: float,
        h_key: str = "h",
        value_key: str = "value",
        safety_factor: float = DEFAULT_SAFETY_FACTOR,
        ratio_relative_tolerance: float = DEFAULT_RATIO_RELATIVE_TOLERANCE,
        exactness_floor: float = DEFAULT_EXACTNESS_FLOOR,
) -> dict[str, Any]:
    """Evaluate the finest asymptotic three-level tail of one sequence.

    A nonmonotone three-level sequence is not accepted or rejected. It returns
    ``ADDITIONAL_LEVEL_REQUIRED``. With four or more levels, the same decision
    is based on the finest three levels, so a coarse pre-asymptotic reversal is
    retained in the report without controlling the asymptotic tail.
    """
    ordered = normalized_samples(samples, h_key=h_key, value_key=value_key)
    reference = _finite_number(reference_value, "reference value")
    scale = _finite_number(normalization, "normalization")
    minimum_order = _finite_number(
        minimum_observed_order, "minimum observed order")
    error_limit = _finite_number(
        finest_relative_error_limit, "finest relative-error limit")
    gci_limit = _finite_number(finest_gci_limit, "finest GCI limit")
    factor = _finite_number(safety_factor, "GCI safety factor")
    floor = _finite_number(exactness_floor, "exactness floor")
    if scale <= 0.0:
        raise ConvergenceError("normalization must be positive")
    if minimum_order <= 0.0:
        raise ConvergenceError("minimum observed order must be positive")
    if min(error_limit, gci_limit, floor) < 0.0:
        raise ConvergenceError("error, GCI, and exactness limits must be nonnegative")
    if factor < 1.0:
        raise ConvergenceError("GCI safety factor must be at least one")

    ratio = _uniform_refinement_ratio(
        ordered, h_key=h_key, relative_tolerance=ratio_relative_tolerance)
    tail = ordered[-3:]
    coarse, medium, fine = tail
    values = [float(sample[value_key]) for sample in tail]
    sample_relative_errors = [
        abs(float(sample[value_key]) - reference) / scale
        for sample in ordered
    ]
    relative_errors = sample_relative_errors[-3:]
    monotone_to_reference = (
        relative_errors[0] > relative_errors[1] > relative_errors[2]
    )
    differences = (
        abs(values[0] - values[1]),
        abs(values[1] - values[2]),
    )
    resolution_scale = max(
        1.0,
        abs(reference),
        *(abs(value) for value in values),
    )
    numerically_exact = max(relative_errors) <= floor

    base: dict[str, Any] = {
        "sample_count": len(ordered),
        "samples": ordered,
        "asymptotic_tail_labels": [sample["label"] for sample in tail],
        "refinement_ratio": ratio,
        "reference_value": reference,
        "normalization": scale,
        "sample_relative_errors": sample_relative_errors,
        "relative_errors": relative_errors,
        "finest_relative_error": relative_errors[-1],
        "monotone_to_reference": monotone_to_reference,
        "safety_factor": factor,
        "minimum_observed_order": minimum_order,
        "finest_relative_error_limit": error_limit,
        "finest_gci_limit": gci_limit,
        "numerically_exact": numerically_exact,
    }
    if numerically_exact:
        passed = relative_errors[-1] <= error_limit
        return {
            **base,
            "status": "PASS" if passed else "FAIL",
            "passed": passed,
            "observed_order": None,
            "richardson_extrapolated_value": values[-1],
            "richardson_relative_error": relative_errors[-1],
            "finest_gci": 0.0,
            "gate_failures": ([] if passed else ["finest_relative_error"]),
        }

    if (not monotone_to_reference or
            min(differences) <= floor * resolution_scale):
        if len(ordered) == 3:
            return {
                **base,
                "status": "ADDITIONAL_LEVEL_REQUIRED",
                "passed": False,
                "observed_order": None,
                "richardson_extrapolated_value": None,
                "richardson_relative_error": None,
                "finest_gci": None,
                "gate_failures": ["asymptotic_tail_not_established"],
            }
        return {
            **base,
            "status": "FAIL",
            "passed": False,
            "observed_order": None,
            "richardson_extrapolated_value": None,
            "richardson_relative_error": None,
            "finest_gci": None,
            "gate_failures": ["four_level_asymptotic_tail_not_established"],
        }

    observed_order = (
        math.log(differences[0]) - math.log(differences[1])
    ) / math.log(ratio)
    if not math.isfinite(observed_order) or observed_order <= 0.0:
        if len(ordered) == 3:
            return {
                **base,
                "status": "ADDITIONAL_LEVEL_REQUIRED",
                "passed": False,
                "observed_order": observed_order,
                "richardson_extrapolated_value": None,
                "richardson_relative_error": None,
                "finest_gci": None,
                "gate_failures": ["positive_observed_order_not_established"],
            }
        return {
            **base,
            "status": "FAIL",
            "passed": False,
            "observed_order": observed_order,
            "richardson_extrapolated_value": None,
            "richardson_relative_error": None,
            "finest_gci": None,
            "gate_failures": ["four_level_positive_order_not_established"],
        }

    exponent = observed_order * math.log(ratio)
    exp_negative = math.exp(-exponent)
    inverse_denominator = exp_negative / -math.expm1(-exponent)
    richardson = (
        values[-1] + (values[-1] - values[-2]) * inverse_denominator
    )
    finest_gci = factor * differences[1] * inverse_denominator / scale
    gate_failures: list[str] = []
    if observed_order < minimum_order:
        gate_failures.append("minimum_observed_order")
    if relative_errors[-1] > error_limit:
        gate_failures.append("finest_relative_error")
    if finest_gci > gci_limit:
        gate_failures.append("finest_gci")
    return {
        **base,
        "status": "PASS" if not gate_failures else "FAIL",
        "passed": not gate_failures,
        "observed_order": observed_order,
        "richardson_extrapolated_value": richardson,
        "richardson_relative_error": abs(richardson - reference) / scale,
        "finest_gci": finest_gci,
        "gate_failures": gate_failures,
    }


def analyze_offset_envelope(
        sequences: dict[str, Sequence[dict[str, Any]]],
        **gci_arguments: Any,
) -> dict[str, Any]:
    """Require every predeclared subcell-offset sequence to qualify."""
    if not isinstance(sequences, dict) or not sequences:
        raise ConvergenceError("an offset envelope requires at least one sequence")
    if any(not isinstance(name, str) or not name for name in sequences):
        raise ConvergenceError("offset identifiers must be nonempty strings")
    records = {
        name: analyze_gci_sequence(samples, **gci_arguments)
        for name, samples in sorted(sequences.items())
    }
    additional = [
        name for name, record in records.items()
        if record["status"] == "ADDITIONAL_LEVEL_REQUIRED"
    ]
    failed = [
        name for name, record in records.items()
        if record["status"] == "FAIL"
    ]
    status = (
        "FAIL" if failed else
        "ADDITIONAL_LEVEL_REQUIRED" if additional else
        "PASS"
    )
    finite_orders = [
        float(record["observed_order"])
        for record in records.values()
        if isinstance(record.get("observed_order"), (int, float)) and
        math.isfinite(float(record["observed_order"]))
    ]
    finite_gci = [
        float(record["finest_gci"])
        for record in records.values()
        if isinstance(record.get("finest_gci"), (int, float)) and
        math.isfinite(float(record["finest_gci"]))
    ]
    return {
        "status": status,
        "passed": status == "PASS",
        "sequence_count": len(records),
        "additional_level_required": additional,
        "failed_sequences": failed,
        "minimum_observed_order": min(finite_orders) if finite_orders else None,
        "maximum_finest_gci": max(finite_gci) if finite_gci else None,
        "maximum_finest_relative_error": max(
            float(record["finest_relative_error"])
            for record in records.values()
        ),
        "sequences": records,
    }

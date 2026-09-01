#!/usr/bin/env python3
"""Validate and evaluate the frozen WP-10 capillary-wave reference."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.special import erfc


SCRIPT_PATH = Path(__file__).resolve()
REPOSITORY_ROOT = SCRIPT_PATH.parents[3]
DEFAULT_REGISTRY = SCRIPT_PATH.with_name(
    "free_surface_wp10_literature_registry_v2.json"
)
EXPECTED_EXTENDS_SHA256 = (
    "56781d349bcf2a03a514d5ea4b5f5744bca76809820431e427aa1c736527b052"
)
EXPECTED_SOURCES_SHA256 = (
    "9d55b2c63662ff4e473fa5b71341f02b23754e74f56176a7251b0d4204b44275"
)
EXPECTED_ANALYTICAL_REFERENCE_SHA256 = (
    "ec985109940fa02a604ff1dd2c7665bc3ee49bd8c8f72af3b4d70a71d6105a32"
)
EXPECTED_REFERENCE_CASES_SHA256 = (
    "134c6bc4907251afab7082ee07d4f91146f8890b669ec2cea5863aa751f27612"
)
EXPECTED_TOP_LEVEL_KEYS = {
    "schema_version",
    "registry_id",
    "status",
    "verified_date",
    "work_package",
    "qualification_campaign",
    "scope",
    "extends",
    "sources",
    "analytical_reference",
    "reference_cases",
}


def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty string")
    return value


def require_finite_positive(value: Any, label: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
        or value <= 0.0
    ):
        raise ValueError(f"{label} must be finite and positive")
    return float(value)


def _fluid_kinematic_viscosity(fluid: dict[str, Any], label: str) -> float | None:
    density = fluid.get("density")
    viscosity = fluid.get("dynamic_viscosity")
    if (
        not isinstance(density, (int, float))
        or isinstance(density, bool)
        or not math.isfinite(density)
        or density < 0.0
        or not isinstance(viscosity, (int, float))
        or isinstance(viscosity, bool)
        or not math.isfinite(viscosity)
        or viscosity < 0.0
    ):
        raise ValueError(f"{label} properties must be finite and nonnegative")
    if density == 0.0:
        if viscosity != 0.0:
            raise ValueError(f"{label} zero density requires zero viscosity")
        return None
    return float(viscosity / density)


def _material_parameters(case: dict[str, Any]) -> tuple[float, float, float]:
    upper = case.get("upper_fluid")
    lower = case.get("lower_fluid")
    if not isinstance(upper, dict) or not isinstance(lower, dict):
        raise ValueError("reference case must define both fluid records")
    upper_nu = _fluid_kinematic_viscosity(upper, "upper fluid")
    lower_nu = _fluid_kinematic_viscosity(lower, "lower fluid")
    if upper_nu is None and lower_nu is None:
        raise ValueError("at least one fluid must have positive density")
    if upper_nu is None:
        kinematic_viscosity = lower_nu
    elif lower_nu is None:
        kinematic_viscosity = upper_nu
    elif not math.isclose(
        upper_nu,
        lower_nu,
        rel_tol=1.0e-13,
        abs_tol=0.0,
    ):
        raise ValueError("the analytical reference requires equal kinematic viscosity")
    else:
        kinematic_viscosity = upper_nu
    assert kinematic_viscosity is not None
    total_density = float(upper["density"] + lower["density"])
    density_parameter = float(
        upper["density"] * lower["density"] / total_density**2
    )
    return kinematic_viscosity, total_density, density_parameter


def _derive_wavelength(case: dict[str, Any]) -> float:
    if "wavelength" in case:
        return require_finite_positive(case["wavelength"], "wavelength")
    selection = case.get("wave_number_selection")
    if (
        not isinstance(selection, dict)
        or selection.get("model") != "denner_equation_23"
    ):
        raise ValueError("reference case has no supported wave-number selection")
    ratio = require_finite_positive(
        selection.get("wave_number_over_critical"),
        "wave-number ratio",
    )
    upper = case["upper_fluid"]
    lower = case["lower_fluid"]
    upper_nu = _fluid_kinematic_viscosity(upper, "upper fluid")
    lower_nu = _fluid_kinematic_viscosity(lower, "lower fluid")
    _, total_density, _ = _material_parameters(case)
    total_viscosity = float(
        upper["dynamic_viscosity"] + lower["dynamic_viscosity"]
    )
    surface_tension = require_finite_positive(
        case.get("surface_tension"), "surface tension"
    )
    viscocapillary_length = (
        total_viscosity**2 / (surface_tension * total_density)
    )
    if upper_nu is None or lower_nu is None:
        property_parameter = 0.0
    else:
        density_factor = (
            upper["density"] * lower["density"] / total_density**2
        )
        viscosity_factor = upper_nu * lower_nu / (upper_nu + lower_nu) ** 2
        property_parameter = density_factor * viscosity_factor
    critical_wave_number = (
        2.0 ** (2.0 / 3.0)
        / viscocapillary_length
        * (1.0625 - property_parameter)
    )
    wave_number = ratio * critical_wave_number
    return 2.0 * math.pi / wave_number


def reference_case(registry: dict[str, Any], case_id: str) -> dict[str, Any]:
    cases = registry.get("reference_cases")
    if not isinstance(cases, list):
        raise ValueError("reference cases are unavailable")
    matches = [case for case in cases if case.get("id") == case_id]
    if len(matches) != 1:
        raise ValueError(f"reference case {case_id!r} is not uniquely defined")
    case = copy.deepcopy(matches[0])
    wavelength = _derive_wavelength(case)
    amplitude_ratio = require_finite_positive(
        case.get("initial_amplitude_over_wavelength"),
        "initial amplitude ratio",
    )
    case["wavelength"] = wavelength
    case["wave_number"] = 2.0 * math.pi / wavelength
    case["initial_amplitude"] = amplitude_ratio * wavelength
    return case


def _evaluate_times(
    case: dict[str, Any],
    physical_times: list[float],
) -> tuple[list[float], float]:
    kinematic_viscosity, total_density, density_parameter = (
        _material_parameters(case)
    )
    surface_tension = require_finite_positive(
        case.get("surface_tension"), "surface tension"
    )
    wavelength = require_finite_positive(case.get("wavelength"), "wavelength")
    initial_amplitude = require_finite_positive(
        case.get("initial_amplitude"), "initial amplitude"
    )
    wave_number = 2.0 * math.pi / wavelength
    inviscid_frequency_squared = (
        surface_tension * wave_number**3 / total_density
    )
    inviscid_frequency = math.sqrt(inviscid_frequency_squared)
    nu_k_squared = kinematic_viscosity * wave_number**2
    roots = np.roots(
        [
            1.0,
            -4.0 * density_parameter * math.sqrt(nu_k_squared),
            2.0 * (1.0 - 6.0 * density_parameter) * nu_k_squared,
            4.0
            * (1.0 - 3.0 * density_parameter)
            * nu_k_squared**1.5,
            (1.0 - 4.0 * density_parameter) * nu_k_squared**2
            + inviscid_frequency_squared,
        ]
    )
    values: list[float] = []
    for physical_time in physical_times:
        if (
            not isinstance(physical_time, (int, float))
            or isinstance(physical_time, bool)
            or not math.isfinite(physical_time)
            or physical_time < 0.0
        ):
            raise ValueError("evaluation times must be finite and nonnegative")
        value = (
            4.0
            * (1.0 - 4.0 * density_parameter)
            * nu_k_squared**2
            / (
                8.0
                * (1.0 - 4.0 * density_parameter)
                * nu_k_squared**2
                + inviscid_frequency_squared
            )
            * initial_amplitude
            * erfc(math.sqrt(nu_k_squared * physical_time))
        )
        for index, root in enumerate(roots):
            denominator = np.prod(
                [
                    other - root
                    for other_index, other in enumerate(roots)
                    if other_index != index
                ]
            )
            value += (
                root
                / denominator
                * (
                    inviscid_frequency_squared
                    * initial_amplitude
                    / (root**2 - nu_k_squared)
                )
                * np.exp((root**2 - nu_k_squared) * physical_time)
                * erfc(root * math.sqrt(physical_time))
            )
        complex_value = complex(value)
        imaginary_limit = 2.0e-11 * max(
            initial_amplitude, abs(complex_value.real)
        )
        if abs(complex_value.imag) > imaginary_limit:
            raise ValueError("analytical reference produced a non-real amplitude")
        values.append(float(complex_value.real))
    return values, inviscid_frequency


def evaluate_dimensionless_times(
    case: dict[str, Any],
    dimensionless_times: list[float],
) -> dict[str, Any]:
    _, total_density, _ = _material_parameters(case)
    surface_tension = require_finite_positive(
        case.get("surface_tension"), "surface tension"
    )
    wavelength = require_finite_positive(case.get("wavelength"), "wavelength")
    wave_number = 2.0 * math.pi / wavelength
    inviscid_frequency = math.sqrt(
        surface_tension * wave_number**3 / total_density
    )
    physical_times = [value / inviscid_frequency for value in dimensionless_times]
    amplitudes, evaluated_frequency = _evaluate_times(case, physical_times)
    initial_amplitude = require_finite_positive(
        case.get("initial_amplitude"), "initial amplitude"
    )
    return {
        "amplitude": amplitudes,
        "dimensionless_time": list(dimensionless_times),
        "inviscid_angular_frequency": evaluated_frequency,
        "normalized_amplitude": [
            value / initial_amplitude for value in amplitudes
        ],
        "physical_time": physical_times,
    }


def _validate_sources(sources: Any) -> None:
    if not isinstance(sources, list) or len(sources) != 2:
        raise ValueError("sources must contain the two frozen Denner records")
    indexed: dict[str, dict[str, Any]] = {}
    for source in sources:
        if not isinstance(source, dict):
            raise ValueError("source entry must be an object")
        source_id = require_nonempty_string(source.get("id"), "source id")
        if source_id in indexed:
            raise ValueError(f"duplicate source id: {source_id}")
        citation = source.get("citation")
        asset = source.get("asset")
        access = source.get("access")
        if not isinstance(citation, dict) or not isinstance(asset, dict):
            raise ValueError(f"source {source_id} lacks citation or asset metadata")
        if not isinstance(access, dict):
            raise ValueError(f"source {source_id} lacks access metadata")
        doi = require_nonempty_string(citation.get("doi"), "source DOI")
        if citation.get("persistent_url") != f"https://doi.org/{doi}":
            raise ValueError(f"source {source_id} DOI URL is inconsistent")
        checksum = asset.get("sha256")
        if (
            asset.get("status") != "EXTERNALLY_PINNED"
            or not isinstance(checksum, str)
            or len(checksum) != 64
            or any(character not in "0123456789abcdef" for character in checksum)
            or not isinstance(asset.get("bytes"), int)
            or asset["bytes"] <= 0
            or asset.get("repository_path") is not None
            or access.get("redistribution") != "NOT_INCLUDED"
        ):
            raise ValueError(f"source {source_id} has an invalid external asset pin")
        indexed[source_id] = source
    if set(indexed) != {
        "denner_2016_capillary_dispersion",
        "denner_2016_prosperetti_reference_script",
    }:
        raise ValueError("source identities changed")
    if (
        indexed["denner_2016_capillary_dispersion"]["access"].get("license")
        != "CC-BY-3.0"
        or indexed["denner_2016_capillary_dispersion"].get("disposition")
        != "EXECUTABLE_ANALYTICAL_REFERENCE"
        or indexed["denner_2016_prosperetti_reference_script"].get(
            "disposition"
        )
        != "CORROBORATING_ASSET_NOT_NUMERICAL_ORACLE"
    ):
        raise ValueError("source roles or reuse boundary changed")


def _validate_checkpoints(registry: dict[str, Any]) -> None:
    cases = registry.get("reference_cases")
    if not isinstance(cases, list) or len(cases) != 2:
        raise ValueError("reference cases must contain the two frozen entries")
    case_ids = {case.get("id") for case in cases if isinstance(case, dict)}
    if case_ids != {"denner_case_d", "denner_case_a_selected_wave_number"}:
        raise ValueError("reference case identities changed")
    for stored in cases:
        checkpoints = stored.get("checkpoints")
        if not isinstance(checkpoints, dict):
            raise ValueError("reference case lacks checkpoints")
        times = checkpoints.get("dimensionless_time")
        expected = checkpoints.get("normalized_amplitude")
        if (
            not isinstance(times, list)
            or not isinstance(expected, list)
            or len(times) != 6
            or len(expected) != len(times)
        ):
            raise ValueError("reference checkpoint coverage is incomplete")
        relative_tolerance = require_finite_positive(
            checkpoints.get("relative_tolerance"),
            "checkpoint relative tolerance",
        )
        absolute_tolerance = require_finite_positive(
            checkpoints.get("absolute_tolerance"),
            "checkpoint absolute tolerance",
        )
        case = reference_case(registry, stored["id"])
        evaluated = evaluate_dimensionless_times(case, times)
        if not math.isclose(
            evaluated["inviscid_angular_frequency"],
            checkpoints.get("inviscid_angular_frequency"),
            rel_tol=relative_tolerance,
            abs_tol=absolute_tolerance,
        ):
            raise ValueError("reference angular-frequency checkpoint changed")
        for actual, target in zip(evaluated["normalized_amplitude"], expected):
            if not math.isclose(
                actual,
                target,
                rel_tol=relative_tolerance,
                abs_tol=absolute_tolerance,
            ):
                raise ValueError("reference amplitude checkpoint changed")


def validate_registry(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as source:
        registry = json.load(source, object_pairs_hook=reject_duplicate_keys)
    if not isinstance(registry, dict) or set(registry) != EXPECTED_TOP_LEVEL_KEYS:
        raise ValueError("top-level registry contract changed")
    expected_metadata = {
        "schema_version": 2,
        "registry_id": "free_surface_wp10_literature_v2",
        "status": "FROZEN_EXECUTABLE_CAPILLARY_REFERENCE_CONTRACT",
        "verified_date": "2026-08-31",
        "work_package": "WP-10",
        "qualification_campaign": "Q7",
    }
    for key, expected in expected_metadata.items():
        if registry.get(key) != expected:
            raise ValueError("registry metadata changed")
    require_nonempty_string(registry.get("scope"), "registry scope")
    if canonical_sha256(registry.get("extends")) != EXPECTED_EXTENDS_SHA256:
        raise ValueError("version-1 extension contract changed")
    if canonical_sha256(registry.get("sources")) != EXPECTED_SOURCES_SHA256:
        raise ValueError("source contract changed")
    if (
        canonical_sha256(registry.get("analytical_reference"))
        != EXPECTED_ANALYTICAL_REFERENCE_SHA256
    ):
        raise ValueError("analytical reference contract changed")
    if (
        canonical_sha256(registry.get("reference_cases"))
        != EXPECTED_REFERENCE_CASES_SHA256
    ):
        raise ValueError("reference case contract changed")

    extension = registry["extends"]
    version_1_path = REPOSITORY_ROOT / extension["repository_path"]
    if (
        extension.get("mutation_policy") != "PRESERVE_V1_BYTE_FOR_BYTE"
        or not version_1_path.is_file()
        or file_sha256(version_1_path) != extension.get("sha256")
    ):
        raise ValueError("version-1 registry is not preserved byte for byte")
    analytical = registry["analytical_reference"]
    if (
        analytical.get("equation_location") != "Equations 12-13"
        or analytical.get("complex_erfc_argument") != "z_i_times_sqrt_t"
        or analytical.get("author_script_role")
        != "CHECKSUM_PINNED_CORROBORATION_NOT_NUMERICAL_ORACLE"
        or analytical.get("author_script_difference")
        != "uses_real_part_of_z_i_in_erfc_argument"
    ):
        raise ValueError("analytical reference boundary is inconsistent")
    _validate_sources(registry["sources"])
    _validate_checkpoints(registry)
    return registry


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    arguments = parser.parse_args()
    if not arguments.validate_only:
        parser.error("this runner only supports --validate-only")
    registry = validate_registry(arguments.registry)
    print(
        json.dumps(
            {
                "executable_case_count": len(registry["reference_cases"]),
                "outcome": "PASS",
                "registry_id": registry["registry_id"],
                "source_count": len(registry["sources"]),
                "v1_preserved": True,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

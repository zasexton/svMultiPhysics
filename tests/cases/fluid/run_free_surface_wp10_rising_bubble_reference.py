#!/usr/bin/env python3
"""Validate and evaluate the frozen WP-10 rising-bubble reference."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


SCRIPT_PATH = Path(__file__).resolve()
REPOSITORY_ROOT = SCRIPT_PATH.parents[3]
DEFAULT_REGISTRY = SCRIPT_PATH.with_name(
    "free_surface_wp10_literature_registry_v3.json"
)
EXPECTED_EXTENDS_SHA256 = (
    "6c54a9d0ecbfa347fb86fc5c21554bc285da8b55659715c9b64a863100791505"
)
EXPECTED_SOURCES_SHA256 = (
    "29273f5b209010d5ac71a557c562494dc295dd105a6e8e456b83c93b2ef4f450"
)
EXPECTED_EXPERIMENTAL_REFERENCE_SHA256 = (
    "6ce3d938b62b9457ee92fe26cceedafe7f79f76dbc5787a8b81b999cd8822c29"
)
EXPECTED_REFERENCE_CASES_SHA256 = (
    "e8258d97b4e7fe2c417c1387f00d0075655a482faa8447b3b757b03fc729d6a4"
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
    "experimental_reference",
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


def _relative_error(actual: float, reported: float) -> float:
    return abs(actual - reported) / abs(reported)


def evaluate_reported_cases(registry: dict[str, Any]) -> list[dict[str, float]]:
    reference = registry.get("experimental_reference")
    cases = registry.get("reference_cases")
    if not isinstance(reference, dict) or not isinstance(cases, list):
        raise ValueError("experimental reference or cases are unavailable")
    liquid = reference.get("liquid")
    if not isinstance(liquid, dict):
        raise ValueError("liquid properties are unavailable")
    density = require_finite_positive(
        liquid.get("density_kg_per_m3"), "liquid density"
    )
    viscosity = require_finite_positive(
        liquid.get("dynamic_viscosity_pa_s"), "liquid viscosity"
    )
    surface_tension = require_finite_positive(
        liquid.get("surface_tension_n_per_m"), "surface tension"
    )
    gravity = require_finite_positive(
        reference.get("gravity_m_per_s2"), "gravity"
    )

    evaluated: list[dict[str, float]] = []
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError("reference case must be an object")
        diameter_mm = require_finite_positive(
            case.get("diameter_mm"), "bubble diameter"
        )
        velocity = require_finite_positive(
            case.get("average_velocity_m_per_s"), "bubble velocity"
        )
        diameter = diameter_mm * 1.0e-3
        actual = {
            "reynolds": density * velocity * diameter / viscosity,
            "weber": density * velocity**2 * diameter / surface_tension,
            "eotvos": density * gravity * diameter**2 / surface_tension,
            "galilei": (
                density * math.sqrt(gravity) * diameter**1.5 / viscosity
            ),
            "morton": (
                gravity * viscosity**4 / (density * surface_tension**3)
            ),
        }
        result = {
            "diameter_mm": diameter_mm,
            "reported_average_velocity_m_per_s": velocity,
        }
        for group, calculated in actual.items():
            reported = require_finite_positive(
                case.get(group), f"reported {group} number"
            )
            result[f"calculated_{group}"] = calculated
            result[f"reported_{group}"] = reported
            result[f"relative_error_{group}"] = _relative_error(
                calculated, reported
            )
        evaluated.append(result)
    return evaluated


def _validate_source(sources: Any) -> None:
    if not isinstance(sources, list) or len(sources) != 1:
        raise ValueError("sources must contain the frozen experimental article")
    source = sources[0]
    if not isinstance(source, dict) or source.get("id") != (
        "chang_2024_rising_air_bubbles"
    ):
        raise ValueError("source identity changed")
    citation = source.get("citation")
    asset = source.get("asset")
    access = source.get("access")
    if not all(isinstance(value, dict) for value in (citation, asset, access)):
        raise ValueError("source lacks citation, asset, or access metadata")
    doi = require_nonempty_string(citation.get("doi"), "source DOI")
    checksum = asset.get("sha256")
    if (
        citation.get("persistent_url") != f"https://doi.org/{doi}"
        or asset.get("status") != "EXTERNALLY_PINNED"
        or not isinstance(checksum, str)
        or len(checksum) != 64
        or any(character not in "0123456789abcdef" for character in checksum)
        or not isinstance(asset.get("bytes"), int)
        or asset["bytes"] <= 0
        or asset.get("repository_path") is not None
        or access.get("license") != "CC-BY-4.0"
        or access.get("redistribution") != "NOT_INCLUDED"
        or source.get("disposition")
        != "EXPERIMENTAL_EXPECTATION_POINTS_NO_RELEASE_BAND"
    ):
        raise ValueError("source pin or reuse boundary changed")


def _validate_reference(registry: dict[str, Any]) -> None:
    reference = registry["experimental_reference"]
    if (
        reference.get("primary_source") != "chang_2024_rising_air_bubbles"
        or reference.get("reported_measurement_uncertainty") is not None
        or reference.get("solver_acceptance_band") is not None
        or reference.get("gate_policy")
        != "REPORT_POINTS_AND_NUMERICAL_UNCERTAINTY_NO_RELEASE_BAND"
        or reference.get("raw_dataset_access")
        != "AVAILABLE_FROM_AUTHORS_ON_REQUEST"
    ):
        raise ValueError("experimental reference boundary is inconsistent")
    tolerance = require_finite_positive(
        reference.get("reported_rounding_reproduction_relative_tolerance"),
        "reported rounding tolerance",
    )
    if tolerance > 0.035:
        raise ValueError("reported rounding tolerance is too weak")
    evaluated = evaluate_reported_cases(registry)
    if len(evaluated) != 4:
        raise ValueError("reference case coverage is incomplete")
    for case in evaluated:
        for group in ("reynolds", "weber", "eotvos", "galilei", "morton"):
            if case[f"relative_error_{group}"] > tolerance:
                raise ValueError(
                    f"reported {group} number is inconsistent with rounded inputs"
                )


def validate_registry(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as source:
        registry = json.load(source, object_pairs_hook=reject_duplicate_keys)
    if not isinstance(registry, dict) or set(registry) != EXPECTED_TOP_LEVEL_KEYS:
        raise ValueError("top-level registry contract changed")
    expected_metadata = {
        "schema_version": 3,
        "registry_id": "free_surface_wp10_literature_v3",
        "status": "FROZEN_RISING_BUBBLE_EXPERIMENTAL_REFERENCE_CONTRACT",
        "verified_date": "2026-08-31",
        "work_package": "WP-10",
        "qualification_campaign": "Q7",
    }
    for key, expected in expected_metadata.items():
        if registry.get(key) != expected:
            raise ValueError("registry metadata changed")
    require_nonempty_string(registry.get("scope"), "registry scope")
    if canonical_sha256(registry.get("extends")) != EXPECTED_EXTENDS_SHA256:
        raise ValueError("version-2 extension contract changed")
    if canonical_sha256(registry.get("sources")) != EXPECTED_SOURCES_SHA256:
        raise ValueError("source contract changed")
    if (
        canonical_sha256(registry.get("experimental_reference"))
        != EXPECTED_EXPERIMENTAL_REFERENCE_SHA256
    ):
        raise ValueError("experimental reference contract changed")
    if (
        canonical_sha256(registry.get("reference_cases"))
        != EXPECTED_REFERENCE_CASES_SHA256
    ):
        raise ValueError("reference case contract changed")

    extension = registry["extends"]
    version_2_path = REPOSITORY_ROOT / extension["repository_path"]
    if (
        extension.get("mutation_policy") != "PRESERVE_V2_BYTE_FOR_BYTE"
        or not version_2_path.is_file()
        or file_sha256(version_2_path) != extension.get("sha256")
    ):
        raise ValueError("version-2 registry is not preserved byte for byte")
    _validate_source(registry["sources"])
    _validate_reference(registry)
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
                "experimental_case_count": len(registry["reference_cases"]),
                "outcome": "PASS",
                "registry_id": registry["registry_id"],
                "release_gate_count": 0,
                "source_count": len(registry["sources"]),
                "v2_preserved": True,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Validate the frozen WP-10 primary-reference contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


SCRIPT_PATH = Path(__file__).resolve()
DEFAULT_REGISTRY = SCRIPT_PATH.with_name(
    "free_surface_wp10_literature_registry_v1.json"
)
EXPECTED_SOURCE_CONTRACT_SHA256 = (
    "6342ac3bb5097f64223cf066550d4b555ce5f8ff1ca0435c1981818c7093a039"
)
EXPECTED_BENCHMARK_CONTRACT_SHA256 = (
    "a572f079b24b7276f3ed5c694d35a431c59a222518e41281c1eaa0a8aedf9c5d"
)
EXPECTED_REFINEMENT_CONTRACT_SHA256 = (
    "a54286bf7647bf8c30fab9aab3c990e739a4068a78488190247415cb4ac1196c"
)
EXPECTED_TOP_LEVEL_KEYS = {
    "schema_version",
    "registry_id",
    "status",
    "verified_date",
    "work_package",
    "qualification_campaign",
    "scope",
    "sources",
    "benchmarks",
    "common_refinement_contract",
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


def require_nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty string")
    return value


def validate_source_structure(sources: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(sources, list) or len(sources) != 2:
        raise ValueError("sources must contain the two frozen primary references")
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
        status = asset.get("status")
        if status == "EXTERNALLY_PINNED":
            checksum = asset.get("sha256")
            if (
                not isinstance(checksum, str)
                or len(checksum) != 64
                or any(character not in "0123456789abcdef" for character in checksum)
                or not isinstance(asset.get("bytes"), int)
                or asset["bytes"] <= 0
                or access.get("redistribution") != "NOT_INCLUDED"
            ):
                raise ValueError(f"source {source_id} has an invalid pinned asset")
            if source.get("disposition") != "EXECUTABLE_INTERCODE_REFERENCE":
                raise ValueError(f"source {source_id} has an invalid executable disposition")
        elif status == "SOURCE_ASSET_UNAVAILABLE":
            if (
                asset.get("sha256") is not None
                or asset.get("bytes") is not None
                or asset.get("repository_path") is not None
                or source.get("disposition") != "BLOCKED_QUANTITATIVE_GATE"
            ):
                raise ValueError(f"source {source_id} overstates unavailable reference data")
        else:
            raise ValueError(f"source {source_id} has an unsupported asset status")
        indexed[source_id] = source
    return indexed


def validate_benchmark_structure(
    benchmarks: Any,
    sources: dict[str, dict[str, Any]],
) -> None:
    if not isinstance(benchmarks, list) or len(benchmarks) != 3:
        raise ValueError("benchmarks must contain the three frozen entries")
    ids: set[str] = set()
    for benchmark in benchmarks:
        if not isinstance(benchmark, dict):
            raise ValueError("benchmark entry must be an object")
        benchmark_id = require_nonempty_string(
            benchmark.get("id"), "benchmark id"
        )
        if benchmark_id in ids:
            raise ValueError(f"duplicate benchmark id: {benchmark_id}")
        ids.add(benchmark_id)
        source_id = benchmark.get("reference_source")
        if source_id not in sources:
            raise ValueError(f"benchmark {benchmark_id} names an unknown source")
        if sources[source_id]["disposition"] == "BLOCKED_QUANTITATIVE_GATE":
            if (
                benchmark.get("gate_policy") != "BLOCKED_UNTIL_SOURCE_PINNED"
                or benchmark.get("quantitative_gate") is not None
            ):
                raise ValueError(
                    f"benchmark {benchmark_id} promotes an unpinned source"
                )
        if benchmark_id == "hysing_case_2" and (
            benchmark.get("post_breakup_policy")
            != "REPORT_INTERCODE_RANGE_ONLY"
            or benchmark.get("post_breakup_shape_gate") is not None
        ):
            raise ValueError("Hysing case 2 post-breakup scope is overstated")


def validate_refinement_structure(value: Any) -> None:
    if not isinstance(value, dict):
        raise ValueError("common refinement contract must be an object")
    spatial = value.get("spatial_levels")
    temporal = value.get("temporal_levels")
    offsets = value.get("cut_offset_fractions")
    if (
        not isinstance(spatial, list)
        or len(spatial) != 3
        or not all(isinstance(item, int) and item > 0 for item in spatial)
        or spatial != sorted(spatial)
        or not isinstance(temporal, list)
        or len(temporal) != 3
        or not all(isinstance(item, (int, float)) and item > 0 for item in temporal)
        or temporal != sorted(temporal, reverse=True)
        or not isinstance(offsets, list)
        or len(offsets) < 3
        or value.get("material_side_reversal") != [False, True]
        or value.get("mpi_ranks") != [1, 2, 4]
        or value.get("raw_and_post_maintenance_histories_required") is not True
    ):
        raise ValueError("common refinement contract is incomplete")


def validate_registry(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as source:
        registry = json.load(source, object_pairs_hook=reject_duplicate_keys)
    if not isinstance(registry, dict) or set(registry) != EXPECTED_TOP_LEVEL_KEYS:
        raise ValueError("top-level registry contract changed")
    expected_metadata = {
        "schema_version": 1,
        "registry_id": "free_surface_wp10_literature_v1",
        "status": "FROZEN_REFERENCE_CONTRACT",
        "verified_date": "2026-08-31",
        "work_package": "WP-10",
        "qualification_campaign": "Q7",
    }
    for key, expected in expected_metadata.items():
        if registry.get(key) != expected:
            raise ValueError("registry metadata changed")
    require_nonempty_string(registry.get("scope"), "registry scope")

    if canonical_sha256(registry.get("sources")) != EXPECTED_SOURCE_CONTRACT_SHA256:
        raise ValueError("source contract changed")
    if (
        canonical_sha256(registry.get("benchmarks"))
        != EXPECTED_BENCHMARK_CONTRACT_SHA256
    ):
        raise ValueError("benchmark contract changed")
    if (
        canonical_sha256(registry.get("common_refinement_contract"))
        != EXPECTED_REFINEMENT_CONTRACT_SHA256
    ):
        raise ValueError("refinement contract changed")
    sources = validate_source_structure(registry.get("sources"))
    validate_benchmark_structure(registry.get("benchmarks"), sources)
    validate_refinement_structure(registry.get("common_refinement_contract"))
    return registry


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    arguments = parser.parse_args()
    if not arguments.validate_only:
        parser.error("this runner only supports --validate-only")
    registry = validate_registry(arguments.registry)
    executable = sum(
        source["disposition"] == "EXECUTABLE_INTERCODE_REFERENCE"
        for source in registry["sources"]
    )
    blocked = sum(
        benchmark["gate_policy"] == "BLOCKED_UNTIL_SOURCE_PINNED"
        for benchmark in registry["benchmarks"]
    )
    print(
        json.dumps(
            {
                "benchmark_count": len(registry["benchmarks"]),
                "blocked_quantitative_gate_count": blocked,
                "executable_reference_count": executable,
                "outcome": "PASS",
                "registry_id": registry["registry_id"],
                "source_count": len(registry["sources"]),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Run the versioned WP-3 sharp exterior-boundary closure matrix."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

sys.dont_write_bytecode = True


SCRIPT_PATH = Path(__file__).resolve()
REPOSITORY_ROOT = SCRIPT_PATH.parents[3]
DEFAULT_REGISTRY = SCRIPT_PATH.with_name(
    "free_surface_wp3_sharp_boundary_qualification_matrix_v5.json"
)
PARENT_RUNNER_PATH = SCRIPT_PATH.with_name(
    "run_free_surface_wp3_sharp_boundary_qualification_v4.py"
)
PARENT_MATRIX_PATH = SCRIPT_PATH.with_name(
    "free_surface_wp3_sharp_boundary_qualification_matrix_v4.json"
)
PARENT_FOCUSED_TEST_PATH = (
    REPOSITORY_ROOT
    / "tests/test_free_surface_wp3_sharp_boundary_qualification_runner_v4.py"
)
EXPECTED_PARENT_RUNNER_SHA256 = (
    "743900c0ecb654e192be4bbd3b25b15939ca9fe7871e92544a2ecd5f38ed187c"
)
EXPECTED_PARENT_MATRIX_SHA256 = (
    "2152dff9f850453774be1825951c8ff9697409a06bbf413b877128af3cea79c2"
)
EXPECTED_PARENT_FOCUSED_TEST_SHA256 = (
    "3b8eba001d0aaacfa955b57a585b32502f57549d7adaf51c289dd4a1ef7da61b"
)
EXPECTED_SHARED_RUNNER_SHA256 = (
    "5387dd19618139aeee45bb6f3c77f27fd8b26ce28713d221a866e1eea4662037"
)
EXPECTED_DISCOVERY_HELPER_SHA256 = (
    "cdf0c84761d8b78989291859d1e073b15822ce98ee011325ca1dc49b2d1a0f3a"
)


def _load_parent_runner() -> Any:
    locked = {
        PARENT_RUNNER_PATH: EXPECTED_PARENT_RUNNER_SHA256,
        PARENT_MATRIX_PATH: EXPECTED_PARENT_MATRIX_SHA256,
        PARENT_FOCUSED_TEST_PATH: EXPECTED_PARENT_FOCUSED_TEST_SHA256,
    }
    for path, expected in locked.items():
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"qualification parent bytes changed: {path}")
    specification = importlib.util.spec_from_file_location(
        "_free_surface_wp3_sharp_boundary_v5_parent",
        PARENT_RUNNER_PATH,
    )
    if specification is None or specification.loader is None:
        raise RuntimeError("cannot load the qualification parent")
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


_parent = _load_parent_runner()
strict_runner = _parent.strict_runner
_v4_validate_matrix_contract = _parent.validate_matrix_contract
_v4_validate_frozen_qualification_bundle = (
    _parent.validate_frozen_qualification_bundle
)
_v4_write_json = _parent.write_json
_v4_binary_record = strict_runner.binary_record
_v4_validate_only_summary = _parent.validate_only_summary

_BASE_REGISTRY = _parent.parse_json_document(PARENT_MATRIX_PATH)
_v4_validate_matrix_contract(copy.deepcopy(_BASE_REGISTRY))

EXPECTED_MATRIX_ID = "free_surface_wp3_sharp_boundary_closure_v5"
EXPECTED_WORK_PACKAGE = "WP-3"
DRAFT_MATRIX_STATUS = "DRAFT_UNEXECUTED"
EXECUTABLE_MATRIX_STATUS = "FROZEN_BEFORE_EXECUTION"
EXPECTED_CHECKED_IN_MATRIX_STATUS = EXECUTABLE_MATRIX_STATUS
ALLOWED_MATRIX_STATUSES = {
    DRAFT_MATRIX_STATUS,
    EXECUTABLE_MATRIX_STATUS,
}
EXPECTED_STATUS_REASONS = {
    DRAFT_MATRIX_STATUS: (
        "V5_RESOURCE_POLICY_DRAFT_AWAITING_RECIPROCAL_HASH_FREEZE"
    ),
    EXECUTABLE_MATRIX_STATUS: (
        "FROZEN_BEFORE_EXECUTION_AFTER_V4_PROVENANCE_LIMIT_REJECTION"
    ),
}
EXPECTED_IMPLEMENTATION_SOURCE_COMMIT = (
    "64fd010061a9de7cdf4b23c722884fd6c8db940e"
)
RUNNER_SHA256_ZERO_SENTINEL = "0" * 64
EXPECTED_NORMALIZED_REGISTRY_SHA256 = (
    "009784c169e67fa32c4b505918b392710d7f78ab7e9594ca8358dffaea2da985"
)
EXPECTED_FOCUSED_TEST_SHA256 = (
    "3d6c66f740863af2c2bb8b9932d7930e6b29073f625e95b485d6d7ca9e49056a"
)
EXPECTED_MATRIX_PATH = (
    "tests/cases/fluid/"
    "free_surface_wp3_sharp_boundary_qualification_matrix_v5.json"
)
EXPECTED_RUNNER_PATH = (
    "tests/cases/fluid/"
    "run_free_surface_wp3_sharp_boundary_qualification_v5.py"
)
EXPECTED_FOCUSED_TEST_PATH = (
    "tests/test_free_surface_wp3_sharp_boundary_qualification_runner_v5.py"
)
EXPECTED_BUNDLE_PATHS = (
    EXPECTED_MATRIX_PATH,
    EXPECTED_RUNNER_PATH,
    EXPECTED_FOCUSED_TEST_PATH,
)
EXPECTED_BUNDLE_COMMIT_RESOLUTION = (
    "unique_direct_child_of_implementation_source_commit_on_validation_"
    "HEAD_ancestry_matching_exact_paths_and_frozen_blobs"
)
EXPECTED_FROZEN_BUNDLE_AUTHORITY = (
    "reciprocal_sha256_plus_canonical_bundle_commit_history"
)
EXPECTED_BINARY_LINK_PROVENANCE_POLICY = {
    "address_and_resident_memory_mib": 1024,
    "command": "ldd",
    "launch_mode": "direct_serial",
    "maximum_qualification_job_memory_fraction": 0.05,
    "memory_enforcement_method": (
        "RLIMIT_AS_plus_sampled_session_resident_memory"
    ),
    "output_mib": 4,
    "timeout_seconds": 60,
    "v4_rejected_address_and_resident_memory_mib": 256,
}
EXPECTED_PARENT_ARTIFACTS = [
    {
        "path": (
            "tests/cases/fluid/"
            "free_surface_wp3_sharp_boundary_qualification_matrix_v4.json"
        ),
        "role": "rejected_qualification_matrix",
        "sha256": EXPECTED_PARENT_MATRIX_SHA256,
    },
    {
        "path": (
            "tests/cases/fluid/"
            "run_free_surface_wp3_sharp_boundary_qualification_v4.py"
        ),
        "role": "qualification_parent",
        "sha256": EXPECTED_PARENT_RUNNER_SHA256,
    },
    {
        "path": (
            "tests/"
            "test_free_surface_wp3_sharp_boundary_qualification_runner_v4.py"
        ),
        "role": "rejected_qualification_contract_test",
        "sha256": EXPECTED_PARENT_FOCUSED_TEST_SHA256,
    },
    {
        "path": (
            "tests/cases/fluid/"
            "run_free_surface_wp2_geometry_qualification.py"
        ),
        "role": "shared_execution_base",
        "sha256": EXPECTED_SHARED_RUNNER_SHA256,
    },
    {
        "path": "tests/cases/fluid/mpi_aware_gtest_discovery.py",
        "role": "test_discovery_helper",
        "sha256": EXPECTED_DISCOVERY_HELPER_SHA256,
    },
]
EXPECTED_RESOURCE_SAFEGUARDS = copy.deepcopy(
    _BASE_REGISTRY["resource_safeguards"]
)
EXPECTED_RESOURCE_SAFEGUARDS["binary_link_provenance_policy"] = (
    copy.deepcopy(EXPECTED_BINARY_LINK_PROVENANCE_POLICY)
)
EXPECTED_DELTA_KEYS = {
    "schema_version",
    "matrix_id",
    "status",
    "status_reason",
    "work_package",
    "findings",
    "implementation_source_commit",
    "parent_matrix",
    "parent_artifacts",
    "proposed_runner",
    "focused_contract_test",
    "runner_sha256",
    "qualification_bundle_binding",
    "draft_promotion_contract",
    "resource_safeguards",
}

EXPECTED_BINARY_KEYS = _parent.EXPECTED_BINARY_KEYS
EXPECTED_GATES = _parent.EXPECTED_GATES
EXPECTED_TEST_DISCOVERY_CONTRACT = (
    _parent.EXPECTED_TEST_DISCOVERY_CONTRACT
)
EXPECTED_CLOSURE_REQUEST_POLICY = (
    _parent.EXPECTED_CLOSURE_REQUEST_POLICY
)
EXPECTED_OPEN_OUTCOMES = _parent.EXPECTED_OPEN_OUTCOMES
EXPECTED_QUALIFICATION_DISPOSITION = (
    _parent.EXPECTED_QUALIFICATION_DISPOSITION
)
ACCEPTED_CLAIM = _parent.ACCEPTED_CLAIM

parse_json_document = _parent.parse_json_document
normalized_registry_bytes = _parent.normalized_registry_bytes
normalized_registry_sha256 = _parent.normalized_registry_sha256
sha256_file = _parent.sha256_file
_canonical_sha256 = _parent._canonical_sha256
_RUNNER_SHA256_FIELD_PATTERN = _parent._RUNNER_SHA256_FIELD_PATTERN


def _configure_parent() -> None:
    _parent.SCRIPT_PATH = SCRIPT_PATH
    _parent.DEFAULT_REGISTRY = DEFAULT_REGISTRY
    _parent.EXPECTED_MATRIX_ID = EXPECTED_MATRIX_ID
    _parent.EXPECTED_WORK_PACKAGE = EXPECTED_WORK_PACKAGE
    _parent.EXPECTED_CHECKED_IN_MATRIX_STATUS = (
        EXPECTED_CHECKED_IN_MATRIX_STATUS
    )
    _parent.EXPECTED_IMPLEMENTATION_SOURCE_COMMIT = (
        EXPECTED_IMPLEMENTATION_SOURCE_COMMIT
    )
    _parent.EXPECTED_NORMALIZED_REGISTRY_SHA256 = (
        EXPECTED_NORMALIZED_REGISTRY_SHA256
    )
    _parent.EXPECTED_FOCUSED_TEST_SHA256 = EXPECTED_FOCUSED_TEST_SHA256
    _parent.EXPECTED_MATRIX_PATH = EXPECTED_MATRIX_PATH
    _parent.EXPECTED_RUNNER_PATH = EXPECTED_RUNNER_PATH
    _parent.EXPECTED_FOCUSED_TEST_PATH = EXPECTED_FOCUSED_TEST_PATH
    _parent.EXPECTED_BUNDLE_PATHS = EXPECTED_BUNDLE_PATHS
    _parent.EXPECTED_BUNDLE_COMMIT_RESOLUTION = (
        EXPECTED_BUNDLE_COMMIT_RESOLUTION
    )
    _parent.EXPECTED_FROZEN_BUNDLE_AUTHORITY = (
        EXPECTED_FROZEN_BUNDLE_AUTHORITY
    )
    strict_runner.SCRIPT_PATH = SCRIPT_PATH
    strict_runner.DEFAULT_REGISTRY = DEFAULT_REGISTRY
    strict_runner.EXPECTED_MATRIX_ID = EXPECTED_MATRIX_ID
    strict_runner.EXPECTED_MATRIX_STATUS = EXPECTED_CHECKED_IN_MATRIX_STATUS
    strict_runner.EXPECTED_WORK_PACKAGE = EXPECTED_WORK_PACKAGE
    strict_runner.BINARY_LINK_PROVENANCE_MEMORY_MIB = (
        EXPECTED_BINARY_LINK_PROVENANCE_POLICY[
            "address_and_resident_memory_mib"
        ]
    )


_configure_parent()
_draft_bundle_binding = _parent._draft_bundle_binding
_frozen_bundle_binding = _parent._frozen_bundle_binding
_expected_promotion_contract = _parent._expected_promotion_contract


def validate_matrix_contract(delta: dict[str, Any]) -> dict[str, Any]:
    if set(delta) != EXPECTED_DELTA_KEYS:
        raise ValueError("qualification matrix top-level fields changed")
    if delta.get("schema_version") != 5:
        raise ValueError("unsupported qualification matrix schema")
    if delta.get("matrix_id") != EXPECTED_MATRIX_ID:
        raise ValueError("qualification matrix id changed")
    status = delta.get("status")
    if status not in ALLOWED_MATRIX_STATUSES:
        raise ValueError("qualification matrix lifecycle state is invalid")
    if delta.get("status_reason") != EXPECTED_STATUS_REASONS[status]:
        raise ValueError("qualification matrix status reason changed")
    if delta.get("work_package") != EXPECTED_WORK_PACKAGE:
        raise ValueError("qualification work package changed")
    if delta.get("findings") != ["FSR-16"]:
        raise ValueError("qualification finding inventory changed")
    if delta.get("implementation_source_commit") != (
        EXPECTED_IMPLEMENTATION_SOURCE_COMMIT
    ):
        raise ValueError("implementation source commit changed")
    if delta.get("parent_matrix") != {
        "path": PARENT_MATRIX_PATH.relative_to(REPOSITORY_ROOT).as_posix(),
        "sha256": EXPECTED_PARENT_MATRIX_SHA256,
    }:
        raise ValueError("parent matrix binding changed")
    if delta.get("parent_artifacts") != EXPECTED_PARENT_ARTIFACTS:
        raise ValueError("parent artifact contract changed")
    if delta.get("proposed_runner") != EXPECTED_RUNNER_PATH:
        raise ValueError("qualification runner path changed")
    if delta.get("focused_contract_test") != EXPECTED_FOCUSED_TEST_PATH:
        raise ValueError("focused contract test path changed")
    runner_digest = delta.get("runner_sha256")
    if not _parent._valid_digest(runner_digest):
        raise ValueError("runner_sha256 is not a lowercase SHA-256 digest")
    if delta.get("resource_safeguards") != EXPECTED_RESOURCE_SAFEGUARDS:
        raise ValueError("binary provenance resource policy changed")
    expected_binding = (
        _draft_bundle_binding()
        if status == DRAFT_MATRIX_STATUS
        else _frozen_bundle_binding()
    )
    if delta.get("qualification_bundle_binding") != expected_binding:
        raise ValueError("qualification bundle binding changed")
    if delta.get("draft_promotion_contract") != (
        _expected_promotion_contract(status)
    ):
        raise ValueError("draft promotion contract changed")
    if status == DRAFT_MATRIX_STATUS:
        if runner_digest != RUNNER_SHA256_ZERO_SENTINEL:
            raise ValueError("draft matrix runner hash must remain zero")
    elif runner_digest == RUNNER_SHA256_ZERO_SENTINEL:
        raise ValueError("frozen matrix runner hash must be nonzero")

    effective = copy.deepcopy(_BASE_REGISTRY)
    effective.update(
        {
            "schema_version": 5,
            "matrix_id": EXPECTED_MATRIX_ID,
            "status": status,
            "status_reason": delta["status_reason"],
            "implementation_source_commit": (
                EXPECTED_IMPLEMENTATION_SOURCE_COMMIT
            ),
            "parent_artifacts": copy.deepcopy(EXPECTED_PARENT_ARTIFACTS),
            "proposed_runner": EXPECTED_RUNNER_PATH,
            "focused_contract_test": EXPECTED_FOCUSED_TEST_PATH,
            "runner_sha256": runner_digest,
            "qualification_bundle_binding": copy.deepcopy(expected_binding),
            "draft_promotion_contract": copy.deepcopy(
                delta["draft_promotion_contract"]
            ),
            "resource_safeguards": copy.deepcopy(
                EXPECTED_RESOURCE_SAFEGUARDS
            ),
        }
    )
    return effective


def validate_frozen_qualification_bundle(
    registry: dict[str, Any],
    matrix_path: Path = DEFAULT_REGISTRY,
    repository_root: Path = REPOSITORY_ROOT,
    runner_path: Path = SCRIPT_PATH,
) -> dict[str, Any]:
    result = _v4_validate_frozen_qualification_bundle(
        registry,
        matrix_path,
        repository_root,
        runner_path,
    )
    result["binding_schema_version"] = 5
    return result


_frozen_qualification_bundle_binding: dict[str, Any] | None = None


def load_registry(path: Path) -> dict[str, Any]:
    global _frozen_qualification_bundle_binding
    _frozen_qualification_bundle_binding = None
    _parent._frozen_qualification_bundle_binding = None
    if path.resolve() != DEFAULT_REGISTRY.resolve():
        raise ValueError("qualification requires the canonical V5 matrix")
    if path.is_symlink() or not path.is_file():
        raise ValueError("canonical V5 matrix must be a regular file")
    registry = validate_matrix_contract(parse_json_document(path))
    if registry["status"] != EXPECTED_CHECKED_IN_MATRIX_STATUS:
        raise ValueError("checked-in matrix lifecycle state changed")
    if EXPECTED_NORMALIZED_REGISTRY_SHA256 != RUNNER_SHA256_ZERO_SENTINEL:
        if normalized_registry_sha256(path) != (
            EXPECTED_NORMALIZED_REGISTRY_SHA256
        ):
            raise ValueError("normalized matrix bytes changed")
    elif registry["status"] != DRAFT_MATRIX_STATUS:
        raise RuntimeError("frozen matrix normalized digest is not finalized")
    _parent.validate_frozen_dependencies(registry, REPOSITORY_ROOT)
    if registry["status"] == EXECUTABLE_MATRIX_STATUS:
        if EXPECTED_FOCUSED_TEST_SHA256 == RUNNER_SHA256_ZERO_SENTINEL:
            raise RuntimeError("focused test digest is not finalized")
        _frozen_qualification_bundle_binding = (
            validate_frozen_qualification_bundle(registry, path)
        )
        _parent._frozen_qualification_bundle_binding = copy.deepcopy(
            _frozen_qualification_bundle_binding
        )
    return registry


def binary_record(
    binary: Path,
    source_root: Path,
    output_root: Path,
    binary_key: str,
) -> dict[str, Any]:
    result = _v4_binary_record(
        binary, source_root, output_root, binary_key
    )
    result["linked_library_provenance_policy"] = copy.deepcopy(
        EXPECTED_BINARY_LINK_PROVENANCE_POLICY
    )
    return result


def write_json(path: Path, value: Any) -> None:
    if isinstance(value, dict) and path.name in {
        "build_preflight.json",
        "manifest.json",
        "build.json",
        "gates.json",
        "final_provenance.json",
        "summary.json",
    }:
        value = copy.deepcopy(value)
        value["binary_link_provenance_policy"] = copy.deepcopy(
            EXPECTED_BINARY_LINK_PROVENANCE_POLICY
        )
    _v4_write_json(path, value)


def validate_only_summary(
    registry: dict[str, Any], claim: str
) -> dict[str, Any]:
    result = _v4_validate_only_summary(registry, claim)
    result["binary_link_provenance_policy"] = copy.deepcopy(
        EXPECTED_BINARY_LINK_PROVENANCE_POLICY
    )
    return result


observe_implementation_sources = _parent.observe_implementation_sources
validate_frozen_dependencies = _parent.validate_frozen_dependencies
create_test_discovery = _parent.create_test_discovery
requested_claim = _parent.requested_claim

_parent.validate_matrix_contract = validate_matrix_contract
_parent.validate_frozen_qualification_bundle = (
    validate_frozen_qualification_bundle
)
_parent.load_registry = load_registry
_parent.write_json = write_json
_parent.validate_only_summary = validate_only_summary
strict_runner.load_registry = load_registry
strict_runner.write_json = write_json
strict_runner.binary_record = binary_record


def main(arguments: list[str] | None = None) -> int:
    return _parent.main(arguments)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)

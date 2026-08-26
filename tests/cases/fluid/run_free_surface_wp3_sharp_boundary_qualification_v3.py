#!/usr/bin/env python3
"""Run the versioned WP-3 sharp exterior-boundary closure matrix."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import copy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

sys.dont_write_bytecode = True


SCRIPT_PATH = Path(__file__).resolve()
REPOSITORY_ROOT = SCRIPT_PATH.parents[3]
DEFAULT_REGISTRY = SCRIPT_PATH.with_name(
    "free_surface_wp3_sharp_boundary_qualification_matrix_v3.json"
)
PARENT_RUNNER_PATH = SCRIPT_PATH.with_name(
    "run_free_surface_wp3_sharp_boundary_qualification_v2.py"
)
SHARED_RUNNER_PATH = SCRIPT_PATH.with_name(
    "run_free_surface_wp2_geometry_qualification.py"
)
EXPECTED_PARENT_RUNNER_SHA256 = (
    "634a003c8cde429756787771bd4c870ee453a372b95d704fb98e2e10b99876e0"
)
EXPECTED_SHARED_RUNNER_SHA256 = (
    "5387dd19618139aeee45bb6f3c77f27fd8b26ce28713d221a866e1eea4662037"
)


def _load_parent_runner() -> Any:
    if hashlib.sha256(PARENT_RUNNER_PATH.read_bytes()).hexdigest() != (
        EXPECTED_PARENT_RUNNER_SHA256
    ):
        raise RuntimeError("qualification parent bytes changed")
    if hashlib.sha256(SHARED_RUNNER_PATH.read_bytes()).hexdigest() != (
        EXPECTED_SHARED_RUNNER_SHA256
    ):
        raise RuntimeError("shared qualification base bytes changed")
    specification = importlib.util.spec_from_file_location(
        "_free_surface_wp3_sharp_boundary_v3_parent",
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
_shared_write_json = _parent._shared_write_json
_shared_write_text = _parent._shared_write_text
_shared_untracked_source_record = strict_runner.untracked_source_record
_shared_run_build_phase = strict_runner.run_build_phase

EXPECTED_MATRIX_ID = "free_surface_wp3_sharp_boundary_closure_v3"
EXPECTED_WORK_PACKAGE = "WP-3"
DRAFT_MATRIX_STATUS = "DRAFT_UNEXECUTED"
EXECUTABLE_MATRIX_STATUS = "FROZEN_BEFORE_EXECUTION"
EXPECTED_CHECKED_IN_MATRIX_STATUS = EXECUTABLE_MATRIX_STATUS
ALLOWED_MATRIX_STATUSES = {
    DRAFT_MATRIX_STATUS,
    EXECUTABLE_MATRIX_STATUS,
}
EXPECTED_IMPLEMENTATION_SOURCE_COMMIT = (
    "6984b783d87fc56859ff55a06321b46663b68ab0"
)
EXPECTED_NORMALIZED_REGISTRY_SHA256 = (
    "5953c0a4fd5141d5a8d58e76f8a242a65c97497784ee5cf1ac421e31cb1aefa6"
)
EXPECTED_FOCUSED_TEST_SHA256 = (
    "bf196e8f5db6cb2e3e73dce88d5b04e9bdfcd821dc0276e989d6c90ffb4b7934"
)
RUNNER_SHA256_ZERO_SENTINEL = "0" * 64
EXPECTED_MATRIX_PATH = (
    "tests/cases/fluid/"
    "free_surface_wp3_sharp_boundary_qualification_matrix_v3.json"
)
EXPECTED_RUNNER_PATH = (
    "tests/cases/fluid/"
    "run_free_surface_wp3_sharp_boundary_qualification_v3.py"
)
EXPECTED_FOCUSED_TEST_PATH = (
    "tests/"
    "test_free_surface_wp3_sharp_boundary_qualification_runner_v3.py"
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
EXPECTED_AUTHOR_NAME = "Zachary Sexton"
EXPECTED_AUTHOR_EMAIL = "zsexton@stanford.edu"

EXPECTED_STATUS_REASONS = {
    DRAFT_MATRIX_STATUS: (
        "The implementation source inventory is frozen at the recorded clean "
        "commit, but the runner and reciprocal bundle hashes are not yet "
        "finalized. No V3 qualification evidence has been executed."
    ),
    EXECUTABLE_MATRIX_STATUS: (
        "The implementation source inventory and reciprocal matrix, runner, "
        "and focused-test hashes are frozen for execution from the canonical "
        "bundle commit. No V3 qualification evidence has been executed."
    ),
}
EXPECTED_PROMOTION_REQUIREMENTS = [
    "validate every named test against fresh binaries",
    "freeze the focused contract test hash in the runner",
    "freeze the normalized matrix hash in the runner",
    "replace the matrix runner hash sentinel with the final runner digest",
    (
        "commit exactly the matrix runner and focused contract test as the "
        "unique direct child of the implementation source commit"
    ),
]
EXPECTED_SCOPE = (
    "Closure evidence for FSR-16 and WP-3 sharply clipped exterior boundary "
    "operators within the declared one-phase affine C0 P1 and LinearCorner "
    "envelope. A passing archive does not close WP-7, Q1, higher-order "
    "support, uniform cut conditioning, capillary balance, transport, fitted "
    "ALE, or two-phase physics."
)
ACCEPTED_CLAIM = "wp3_fsr16_c0_p1_linearcorner_closure"
EXPECTED_CLOSURE_REQUEST_POLICY = {
    "accepted_claim": ACCEPTED_CLAIM,
    "rejected_claims": [
        "wp7_closure",
        "q1_closure",
        "higher_order_closure",
        "uniform_cut_conditioning",
        "wp4_through_wp10_closure",
    ],
    "diagnostic": (
        "This matrix closes only sharp exterior-boundary routing in its "
        "declared envelope; later conditioning and physics work packages "
        "retain their independent gates."
    ),
}
EXPECTED_QUALIFICATION_DISPOSITION = {
    "on_pass": "CLOSE_FSR16_AND_WP3_WITHIN_DECLARED_ENVELOPE",
    "on_failure": "KEEP_FSR16_AND_WP3_OPEN",
    "wp7": "OPEN",
    "q1": "OPEN",
}
EXPECTED_OPEN_OUTCOMES = {
    "wp7": "OPEN",
    "q1": "OPEN",
    "higher_order": "UNSUPPORTED_FAIL_CLOSED",
    "uniform_cut_conditioning": "OPEN",
    "wp4_through_wp10": "OPEN",
    "q0_and_q2_through_q7": "OPEN",
}
EXPECTED_MODEL_ENVELOPE = (
    "one_phase_unfitted_active_liquid_with_affine_c0_p1_velocity_pressure_"
    "and_linearcorner_implicit_geometry_for_sharp_exterior_boundary_forms"
)
EXPECTED_WET_FRACTIONS = [
    0,
    1.0e-8,
    1.0e-6,
    1.0e-4,
    1.0e-2,
    0.1,
    0.25,
    0.49,
    0.5,
    0.51,
    1,
]
EXPECTED_OPERATOR_NAMES = {
    "traction",
    "robin",
    "outflow",
    "pressure_flux",
    "symmetric_nitsche",
    "unsymmetric_nitsche",
    "wall_slip",
    "coupled_rcr_outflow",
    "coupled_rcrcr_outflow",
    "pspg_normal_pressure_gradient",
    "pspg_tangential_pressure_gradient",
    "pspg_tangential_momentum_residual",
}
EXPECTED_OPERATOR_KEYS = {
    "operator",
    "cut_active_disposition",
    "full_domain_disposition",
    "dry_face_disposition",
    "missing_sharp_domain_disposition",
    "active_side_reversal",
}
EXPECTED_NITSCHE_CERTIFICATE_CONTRACT = {
    "required_for_generated_symmetric_and_unsymmetric_routes": True,
    "constant_viscosity_only": True,
    "fixed_exact_dimension_cap": 32,
    "published_bound": "direct_exact_factorized_dyadic_bound",
    "floating_spectral_values": "diagnostic_only",
    "certificate_digest_version": 5,
    (
        "digest_binds_factorized_flags_counts_input_digest_and_"
        "localized_proof_metadata"
    ): True,
    "symmetric_quarter_energy_safe_ratio_cap": 0.5625,
    (
        "missing_or_stale_certificate"
    ): "hard_error_before_operator_acceptance",
}
EXPECTED_REGULARIZED_CONTRACT = {
    "allowed_role": "diagnostic_bulk_blending_only",
    "exterior_boundary_substitution": False,
    "qualification_credit": False,
}

EXPECTED_BINARY_KEYS = {
    "geometry",
    "level_set",
    "systems",
    "assembly",
    "math",
    "physics",
    "application",
    "assembly_mpi",
    "application_mpi",
}
EXPECTED_BUILD_TARGETS = {
    "geometry": "test_fe_geometry",
    "level_set": "test_fe_levelset",
    "systems": "test_fe_systems",
    "assembly": "test_fe_assembly",
    "math": "test_fe_math",
    "physics": "test_physics",
    "application": "test_application",
    "assembly_mpi": "test_fe_assembly_mpi",
    "application_mpi": "test_application_mpi",
}
EXPECTED_BUILD_CMAKE_HOMES = {
    "geometry": "Code/Source/solver/FE",
    "level_set": "Code/Source/solver/FE",
    "systems": "Code/Source/solver/FE",
    "assembly": "Code/Source/solver/FE",
    "math": "Code/Source/solver/FE",
    "physics": "Code/Source/solver/Physics",
    "application": "Code",
    "assembly_mpi": "Code/Source/solver/FE",
    "application_mpi": "Code",
}
EXPECTED_GTEST_OUTPUT_COPIES = {
    "geometry": 1,
    "level_set": 1,
    "systems": 1,
    "assembly": 1,
    "math": 1,
    "physics": 1,
    "application": 1,
    "assembly_mpi": 2,
    "application_mpi": 2,
}
EXPECTED_GATES = {
    "expected_group_count": 13,
    "expected_distinct_test_count": 80,
    "expected_quantitative_evidence_count": 85,
    "expected_failures": 0,
    "expected_errors": 0,
    "expected_disabled": 0,
    "expected_skipped": 0,
}
EXPECTED_RECORDED_PROPERTY_COUNT = 70
EXPECTED_GROUPS_SHA256 = (
    "dffeeca4d05ba0941ade707517c9dde828ab92aadd09a96543593cde2309eba0"
)
EXPECTED_QUANTITATIVE_EVIDENCE_SHA256 = (
    "4bf276bd8c723feead0ccf33a82c094d56f8f56b7764c707edd83e6d77a6b6bc"
)
EXPECTED_CLOSURE_CONTRACT_SHA256 = (
    "99905d36eff2f53aa28fecd33efc4c42a083005bf7268c9559b15c9af3e3bac7"
)
EXPECTED_OPERATOR_CONTRACT_SHA256 = (
    "3aea28bc6b3ed3a67464574ace34ad329e04ffb637d93ad43dcda6260c704892"
)
EXPECTED_IMPLEMENTATION_SOURCES_SHA256 = (
    "967a621d9bc8468dcba9abe9ab2ab07983b71f402dd0000f888c72c71d4510af"
)
EXPECTED_PARENT_ARTIFACTS_SHA256 = (
    "2b70c6f1129f3e41e4e3519a499504a901f73caa3ebdbf2ff12376713c9f18f5"
)
EXPECTED_FRESH_CONFIGURE_SHA256 = (
    "d2a577defb384b66fb2fc7a5b7a2558997c20cbaaa5f5431daa81fc1874b3ddc"
)
EXPECTED_RESOURCE_SAFEGUARDS_SHA256 = (
    "c4f711b864c9d3187a48866c93ae6127c4472580ddcebb7e95e7039b08a35f86"
)
EXPECTED_PARENT_ARTIFACT_HASHES = {
    (
        "tests/cases/fluid/"
        "free_surface_wp3_sharp_boundary_qualification_matrix_v2.json"
    ): "72cffdc330f07b386fdb89681bcd3da83b7f884c5bd1d09f49eef3f6ae79d883",
    (
        "tests/cases/fluid/"
        "run_free_surface_wp3_sharp_boundary_qualification_v2.py"
    ): EXPECTED_PARENT_RUNNER_SHA256,
    (
        "tests/cases/fluid/"
        "run_free_surface_wp2_geometry_qualification.py"
    ): EXPECTED_SHARED_RUNNER_SHA256,
}
EXPECTED_TOP_LEVEL_KEYS = {
    "schema_version",
    "matrix_id",
    "status",
    "status_reason",
    "work_package",
    "findings",
    "implementation_source_commit",
    "source_inventory_hash_status",
    "implementation_sources",
    "parent_artifacts",
    "proposed_runner",
    "focused_contract_test",
    "runner_sha256",
    "qualification_bundle_binding",
    "draft_promotion_contract",
    "qualification_scope",
    "closure_request_policy",
    "qualification_disposition",
    "open_outcomes",
    "model_envelope",
    "method_limitations",
    "wet_fraction_sweep",
    "operator_disposition_contract",
    "nitsche_trace_certificate_contract",
    "regularized_experimental_model_contract",
    "build_targets",
    "build_cmake_homes",
    "fresh_configure_definitions",
    "resource_safeguards",
    "groups",
    "gates",
    "quantitative_evidence",
    "closure_contract",
    "prospective_tests",
}
QUANTITATIVE_TYPES = {"integer", "real"}
QUANTITATIVE_RELATIONS = {
    "equal",
    "less_than",
    "less_than_or_equal",
    "greater_than",
    "greater_than_or_equal",
}
GTEST_RESULT_FIELDS = {
    "classname",
    "failures",
    "file",
    "line",
    "name",
    "result",
    "status",
    "time",
    "timestamp",
    "type_param",
    "value_param",
}
_RUNNER_SHA256_FIELD_PATTERN = re.compile(
    rb'("runner_sha256"[ \t\r\n]*:[ \t\r\n]*")'
    rb"([0-9a-f]{64})"
    rb'(")'
)
GIT_NO_REPLACE_OBJECTS = "--no-replace-objects"
_frozen_qualification_bundle_binding: dict[str, Any] | None = None
_active_fresh_configure_definitions: dict[str, tuple[str, ...]] = {}


def _draft_bundle_binding() -> dict[str, Any]:
    return {
        "authority": (
            "PREFREEZE_NO_RECIPROCAL_AUTHORITY_UNTIL_HASH_LOCKS_FINALIZED"
        ),
        "matrix_sha256_source": (
            "future_runner_embedded_normalized_matrix_sha256"
        ),
        "matrix_hash_normalization": (
            "replace_the_unique_runner_sha256_value_with_64_ASCII_zero_digits"
        ),
        "runner_sha256_source": "matrix_zero_sentinel_until_runner_freeze",
        "focused_test_sha256_source": (
            "future_runner_embedded_focused_test_sha256"
        ),
        "bundle_commit_resolution": EXPECTED_BUNDLE_COMMIT_RESOLUTION,
        "exact_bundle_commit_blobs_required": list(EXPECTED_BUNDLE_PATHS),
        "bundle_commit_must_have_exactly_one_parent": True,
        "bundle_commit_parent_must_equal_implementation_source_commit": True,
        (
            "bundle_commit_changed_paths_must_equal_exact_bundle_commit_"
            "blobs_required"
        ): True,
        "bundle_commit_blobs_must_match_checked_out_frozen_bytes": True,
        "validation_HEAD_must_descend_from_bundle_commit": True,
        "execution_HEAD_must_equal_bundle_commit": True,
    }


def _frozen_bundle_binding() -> dict[str, Any]:
    return {
        "authority": EXPECTED_FROZEN_BUNDLE_AUTHORITY,
        "matrix_sha256_source": "runner_embedded_normalized_matrix_sha256",
        "matrix_hash_normalization": (
            "replace_the_unique_runner_sha256_value_with_64_ASCII_zero_digits"
        ),
        "runner_sha256_source": "matrix_runner_sha256",
        "focused_test_sha256_source": (
            "runner_embedded_focused_test_sha256"
        ),
        "bundle_commit_resolution": EXPECTED_BUNDLE_COMMIT_RESOLUTION,
        "exact_bundle_commit_blobs_required": list(EXPECTED_BUNDLE_PATHS),
        "bundle_commit_must_have_exactly_one_parent": True,
        "bundle_commit_parent_must_equal_implementation_source_commit": True,
        (
            "bundle_commit_changed_paths_must_equal_exact_bundle_commit_"
            "blobs_required"
        ): True,
        "bundle_commit_blobs_must_match_checked_out_frozen_bytes": True,
        "validation_HEAD_must_descend_from_bundle_commit": True,
        "execution_HEAD_must_equal_bundle_commit": True,
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _reject_duplicate_keys(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def parse_json_document(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("qualification matrix is not valid UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise ValueError("qualification matrix root must be an object")
    return value


def normalized_registry_bytes(raw_bytes: bytes) -> bytes:
    try:
        document = json.loads(
            raw_bytes.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("qualification matrix is not valid UTF-8 JSON") from error
    if not isinstance(document, dict):
        raise ValueError("qualification matrix root must be an object")
    matches = list(_RUNNER_SHA256_FIELD_PATTERN.finditer(raw_bytes))
    if len(matches) != 1:
        raise ValueError(
            "qualification matrix must contain exactly one runner_sha256 "
            "field with a 64-character lowercase hexadecimal value"
        )
    value_start, value_end = matches[0].span(2)
    return (
        raw_bytes[:value_start]
        + RUNNER_SHA256_ZERO_SENTINEL.encode("ascii")
        + raw_bytes[value_end:]
    )


def normalized_registry_sha256(path: Path) -> str:
    return hashlib.sha256(
        normalized_registry_bytes(path.read_bytes())
    ).hexdigest()


def _valid_digest(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _valid_threshold(value_type: str, threshold: Any) -> bool:
    if value_type == "integer":
        return isinstance(threshold, int) and not isinstance(threshold, bool)
    return (
        isinstance(threshold, (int, float))
        and not isinstance(threshold, bool)
        and math.isfinite(threshold)
    )


def _validate_property_contract(
    contract: Any,
    *,
    allowed_tests: set[str],
    default_test: str | None,
) -> tuple[str, str]:
    if not isinstance(contract, dict):
        raise ValueError("recorded property contract must be an object")
    expected_keys = {"property", "type", "relation", "threshold"}
    if default_test is None:
        expected_keys.add("test")
    if set(contract) != expected_keys:
        raise ValueError("recorded property contract has unexpected keys")
    test = contract.get("test", default_test)
    if not isinstance(test, str) or test not in allowed_tests:
        raise ValueError("recorded property cites a test outside its group")
    name = contract["property"]
    value_type = contract["type"]
    relation = contract["relation"]
    if (
        not isinstance(name, str)
        or not name
        or name in GTEST_RESULT_FIELDS
    ):
        raise ValueError("recorded property name is invalid")
    if value_type not in QUANTITATIVE_TYPES:
        raise ValueError("recorded property type is invalid")
    if relation not in QUANTITATIVE_RELATIONS:
        raise ValueError("recorded property relation is invalid")
    if not _valid_threshold(value_type, contract["threshold"]):
        raise ValueError("recorded property threshold is invalid")
    return test, name


def _expected_promotion_contract(status: str) -> dict[str, Any]:
    frozen = status == EXECUTABLE_MATRIX_STATUS
    return {
        "current_state": status,
        "source_hashes_frozen": True,
        "qualification_bundle_hashes_frozen": frozen,
        "qualification_evidence_executed": False,
        "validate_only_allowed": True,
        "execution_allowed": frozen,
        "required_execution_state": EXECUTABLE_MATRIX_STATUS,
        "promotion_requirements": EXPECTED_PROMOTION_REQUIREMENTS,
    }


def validate_matrix_contract(registry: dict[str, Any]) -> dict[str, Any]:
    if set(registry) != EXPECTED_TOP_LEVEL_KEYS:
        raise ValueError("qualification matrix top-level fields changed")
    if registry.get("schema_version") != 3:
        raise ValueError("unsupported qualification matrix schema")
    if registry.get("matrix_id") != EXPECTED_MATRIX_ID:
        raise ValueError("qualification matrix id changed")
    status = registry.get("status")
    if status not in ALLOWED_MATRIX_STATUSES:
        raise ValueError("qualification matrix lifecycle state is invalid")
    if registry.get("status_reason") != EXPECTED_STATUS_REASONS[status]:
        raise ValueError("qualification matrix status reason changed")
    if registry.get("work_package") != EXPECTED_WORK_PACKAGE:
        raise ValueError("qualification work package changed")
    if registry.get("findings") != ["FSR-16"]:
        raise ValueError("qualification finding inventory changed")
    if registry.get("implementation_source_commit") != (
        EXPECTED_IMPLEMENTATION_SOURCE_COMMIT
    ):
        raise ValueError("implementation source commit changed")
    if registry.get("source_inventory_hash_status") != "FROZEN":
        raise ValueError("implementation source inventory is not frozen")
    if registry.get("proposed_runner") != EXPECTED_RUNNER_PATH:
        raise ValueError("qualification runner path changed")
    if registry.get("focused_contract_test") != EXPECTED_FOCUSED_TEST_PATH:
        raise ValueError("focused contract test path changed")

    if registry.get("qualification_scope") != EXPECTED_SCOPE:
        raise ValueError("qualification scope changed")
    if registry.get("closure_request_policy") != (
        EXPECTED_CLOSURE_REQUEST_POLICY
    ):
        raise ValueError("closure request policy changed")
    if registry.get("qualification_disposition") != (
        EXPECTED_QUALIFICATION_DISPOSITION
    ):
        raise ValueError("qualification disposition changed")
    if registry.get("open_outcomes") != EXPECTED_OPEN_OUTCOMES:
        raise ValueError("open outcome inventory changed")
    if registry.get("model_envelope") != EXPECTED_MODEL_ENVELOPE:
        raise ValueError("model envelope changed")
    if registry.get("wet_fraction_sweep") != EXPECTED_WET_FRACTIONS:
        raise ValueError("wet-fraction sweep changed")
    limitations = registry.get("method_limitations")
    if (
        not isinstance(limitations, list)
        or len(limitations) != 7
        or len(set(limitations)) != len(limitations)
        or any(not isinstance(value, str) or not value for value in limitations)
    ):
        raise ValueError("method limitations changed")
    if registry.get("nitsche_trace_certificate_contract") != (
        EXPECTED_NITSCHE_CERTIFICATE_CONTRACT
    ):
        raise ValueError("Nitsche certificate contract changed")
    if registry.get("regularized_experimental_model_contract") != (
        EXPECTED_REGULARIZED_CONTRACT
    ):
        raise ValueError("regularized model boundary changed")

    runner_digest = registry.get("runner_sha256")
    if not _valid_digest(runner_digest):
        raise ValueError("runner_sha256 is not a lowercase SHA-256 digest")
    expected_binding = (
        _draft_bundle_binding()
        if status == DRAFT_MATRIX_STATUS
        else _frozen_bundle_binding()
    )
    if registry.get("qualification_bundle_binding") != expected_binding:
        raise ValueError("qualification bundle binding changed")
    if registry.get("draft_promotion_contract") != (
        _expected_promotion_contract(status)
    ):
        raise ValueError("draft promotion contract changed")
    if status == DRAFT_MATRIX_STATUS:
        if runner_digest != RUNNER_SHA256_ZERO_SENTINEL:
            raise ValueError("draft matrix runner hash must remain zero")
    elif runner_digest == RUNNER_SHA256_ZERO_SENTINEL:
        raise ValueError("frozen matrix runner hash must be nonzero")

    sources = registry.get("implementation_sources")
    if (
        not isinstance(sources, list)
        or len(sources) != 43
        or _canonical_sha256(sources) != EXPECTED_IMPLEMENTATION_SOURCES_SHA256
    ):
        raise ValueError("implementation source manifest changed")
    source_paths: set[str] = set()
    for entry in sources:
        if not isinstance(entry, dict) or set(entry) != {
            "path",
            "role",
            "sha256",
        }:
            raise ValueError("implementation source entry is malformed")
        path = entry["path"]
        role = entry["role"]
        if (
            not isinstance(path, str)
            or not path
            or Path(path).is_absolute()
            or ".." in Path(path).parts
            or path in source_paths
        ):
            raise ValueError("implementation source path is invalid")
        if not isinstance(role, str) or not role or not _valid_digest(
            entry["sha256"]
        ):
            raise ValueError("implementation source entry is invalid")
        source_paths.add(path)

    parents = registry.get("parent_artifacts")
    if _canonical_sha256(parents) != EXPECTED_PARENT_ARTIFACTS_SHA256:
        raise ValueError("parent artifact manifest changed")
    observed_parent_hashes = {
        entry.get("path"): entry.get("sha256")
        for entry in parents
        if isinstance(entry, dict)
    }
    if observed_parent_hashes != EXPECTED_PARENT_ARTIFACT_HASHES:
        raise ValueError("parent artifact hashes changed")

    if registry.get("build_targets") != EXPECTED_BUILD_TARGETS:
        raise ValueError("build target map changed")
    if registry.get("build_cmake_homes") != EXPECTED_BUILD_CMAKE_HOMES:
        raise ValueError("CMake source-home map changed")
    fresh = registry.get("fresh_configure_definitions")
    if _canonical_sha256(fresh) != EXPECTED_FRESH_CONFIGURE_SHA256:
        raise ValueError("fresh configure definitions changed")
    if not isinstance(fresh, dict) or set(fresh) != {
        "Code/Source/solver/FE",
        "Code/Source/solver/Physics",
        "Code",
    }:
        raise ValueError("fresh configure source homes changed")
    for definitions in fresh.values():
        if (
            not isinstance(definitions, list)
            or not definitions
            or len(set(definitions)) != len(definitions)
            or any(
                not isinstance(value, str)
                or not value.startswith("-D")
                or "=" not in value
                for value in definitions
            )
        ):
            raise ValueError("fresh configure definition is malformed")
    if _canonical_sha256(registry.get("resource_safeguards")) != (
        EXPECTED_RESOURCE_SAFEGUARDS_SHA256
    ):
        raise ValueError("resource safeguard contract changed")

    operators = registry.get("operator_disposition_contract")
    if (
        not isinstance(operators, list)
        or _canonical_sha256(operators) != EXPECTED_OPERATOR_CONTRACT_SHA256
    ):
        raise ValueError("operator disposition contract changed")
    observed_operators: set[str] = set()
    expected_dispositions = {
        "cut_active_disposition": "generated_active_boundary",
        "full_domain_disposition": "physical_boundary",
        "dry_face_disposition": "exact_zero",
        "missing_sharp_domain_disposition": "hard_error",
        "active_side_reversal": "complementary_sharp_subset",
    }
    for entry in operators:
        if not isinstance(entry, dict) or set(entry) != EXPECTED_OPERATOR_KEYS:
            raise ValueError("operator disposition entry is malformed")
        operator = entry["operator"]
        if operator in observed_operators:
            raise ValueError("operator disposition is duplicated")
        observed_operators.add(operator)
        for key, expected in expected_dispositions.items():
            if entry.get(key) != expected:
                raise ValueError("operator disposition changed")
    if observed_operators != EXPECTED_OPERATOR_NAMES:
        raise ValueError("operator inventory is incomplete")

    groups = registry.get("groups")
    if (
        not isinstance(groups, list)
        or _canonical_sha256(groups) != EXPECTED_GROUPS_SHA256
    ):
        raise ValueError("qualification group contract changed")
    group_ids: set[str] = set()
    test_names: set[str] = set()
    test_group: dict[str, dict[str, Any]] = {}
    recorded_keys: set[tuple[str, str]] = set()
    recorded_count = 0
    for group in groups:
        base_group_keys = {
            "id",
            "binary",
            "mpi_ranks",
            "gtest_output_copies",
            "tests",
            "execution",
        }
        group_keys = set(group) if isinstance(group, dict) else set()
        if (
            not isinstance(group, dict)
            or group_keys not in (
                base_group_keys,
                base_group_keys | {"recorded_properties"},
            )
        ):
            raise ValueError("qualification group fields changed")
        group_id = group["id"]
        if (
            not isinstance(group_id, str)
            or not re.fullmatch(r"[A-Za-z0-9_.-]+", group_id)
            or group_id in group_ids
        ):
            raise ValueError("qualification group id is invalid")
        group_ids.add(group_id)
        if group["binary"] not in EXPECTED_BINARY_KEYS:
            raise ValueError("qualification group binary is invalid")
        ranks = group["mpi_ranks"]
        copies = group["gtest_output_copies"]
        if (
            ranks not in {1, 2}
            or copies != EXPECTED_GTEST_OUTPUT_COPIES[group["binary"]]
        ):
            raise ValueError("qualification group rank contract changed")
        tests = group["tests"]
        if not isinstance(tests, list) or not tests:
            raise ValueError("qualification group has no tests")
        for test in tests:
            if (
                not isinstance(test, str)
                or not re.fullmatch(r"[A-Za-z0-9_]+\.[A-Za-z0-9_]+", test)
                or test in test_names
            ):
                raise ValueError("qualification test inventory is invalid")
            test_names.add(test)
            test_group[test] = group
        execution = group["execution"]
        if (
            not isinstance(execution, dict)
            or set(execution) != {
                "wall_time_seconds",
                "memory_mib",
                "output_mib",
            }
            or any(
                not isinstance(value, int)
                or isinstance(value, bool)
                or value <= 0
                for value in execution.values()
            )
        ):
            raise ValueError("qualification execution envelope is invalid")
        properties = group.get("recorded_properties", [])
        if not isinstance(properties, list):
            raise ValueError("recorded property list is invalid")
        default_test = tests[0] if len(tests) == 1 else None
        allowed_tests = set(tests)
        for contract in properties:
            key = _validate_property_contract(
                contract,
                allowed_tests=allowed_tests,
                default_test=default_test,
            )
            if key in recorded_keys:
                raise ValueError("recorded property is duplicated")
            recorded_keys.add(key)
            recorded_count += 1
    if (
        len(groups) != EXPECTED_GATES["expected_group_count"]
        or len(test_names) != EXPECTED_GATES["expected_distinct_test_count"]
        or recorded_count != EXPECTED_RECORDED_PROPERTY_COUNT
    ):
        raise ValueError("qualification group counts changed")

    evidence = registry.get("quantitative_evidence")
    if (
        not isinstance(evidence, list)
        or _canonical_sha256(evidence) != (
            EXPECTED_QUANTITATIVE_EVIDENCE_SHA256
        )
    ):
        raise ValueError("quantitative evidence contract changed")
    evidence_keys: set[tuple[str, str]] = set()
    for contract in evidence:
        if not isinstance(contract, dict) or set(contract) != {
            "test",
            "property",
            "type",
            "relation",
            "threshold",
        }:
            raise ValueError("quantitative evidence entry is malformed")
        test = contract["test"]
        if test not in test_names or test_group[test]["mpi_ranks"] != 1:
            raise ValueError("quantitative evidence test is invalid")
        key = _validate_property_contract(
            {
                "property": contract["property"],
                "type": contract["type"],
                "relation": contract["relation"],
                "threshold": contract["threshold"],
            },
            allowed_tests={test},
            default_test=test,
        )
        if key in evidence_keys:
            raise ValueError("quantitative evidence property is duplicated")
        evidence_keys.add(key)
    if len(evidence) != EXPECTED_GATES[
        "expected_quantitative_evidence_count"
    ]:
        raise ValueError("quantitative evidence count changed")

    closure = registry.get("closure_contract")
    if (
        not isinstance(closure, list)
        or _canonical_sha256(closure) != EXPECTED_CLOSURE_CONTRACT_SHA256
    ):
        raise ValueError("closure contract changed")
    claim_names: set[str] = set()
    for claim in closure:
        if not isinstance(claim, dict) or set(claim) != {
            "claim",
            "evidence",
        }:
            raise ValueError("closure claim is malformed")
        name = claim["claim"]
        cited = claim["evidence"]
        if (
            not isinstance(name, str)
            or not name
            or name in claim_names
            or not isinstance(cited, list)
            or not cited
            or any(test not in test_names for test in cited)
        ):
            raise ValueError("closure claim evidence is invalid")
        claim_names.add(name)
    if registry.get("gates") != EXPECTED_GATES:
        raise ValueError("qualification gates changed")
    if registry.get("prospective_tests") != []:
        raise ValueError("frozen matrix cannot contain prospective tests")

    global _active_fresh_configure_definitions
    _active_fresh_configure_definitions = {
        home: tuple(definitions)
        for home, definitions in fresh.items()
    }
    return registry


def _validated_commit_digest(value: str, label: str) -> str:
    if (
        len(value) not in {40, 64}
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} has an invalid commit digest")
    return value


def _exact_commit_output(raw: bytes, label: str) -> str:
    if not raw.endswith(b"\n") or raw.count(b"\n") != 1 or b"\r" in raw:
        raise ValueError(f"{label} commit output is malformed")
    try:
        value = raw[:-1].decode("ascii")
    except UnicodeDecodeError as error:
        raise ValueError(f"{label} commit output is malformed") from error
    return _validated_commit_digest(value, label)


def _resolved_commit(
    repository_root: Path,
    revision: str,
    label: str,
) -> str:
    _validated_commit_digest(revision, label)
    try:
        raw = strict_runner.git_bytes(
            repository_root,
            GIT_NO_REPLACE_OBJECTS,
            "rev-parse",
            "--verify",
            f"{revision}^{{commit}}",
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError(f"{label} is not an available commit") from error
    resolved = _exact_commit_output(raw, label)
    if resolved != revision:
        raise ValueError(f"{label} did not resolve exactly")
    return revision


def _current_head_commit(repository_root: Path) -> str:
    try:
        raw = strict_runner.git_bytes(
            repository_root,
            GIT_NO_REPLACE_OBJECTS,
            "rev-parse",
            "--verify",
            "HEAD^{commit}",
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError("validation HEAD is not an available commit") from error
    return _exact_commit_output(raw, "validation HEAD")


def _require_ancestor(
    repository_root: Path,
    ancestor: str,
    descendant: str,
    diagnostic: str,
) -> None:
    try:
        strict_runner.git_bytes(
            repository_root,
            GIT_NO_REPLACE_OBJECTS,
            "merge-base",
            "--is-ancestor",
            ancestor,
            descendant,
        )
    except subprocess.CalledProcessError as error:
        if error.returncode == 1:
            raise ValueError(diagnostic) from error
        raise ValueError("qualification ancestry is unavailable") from error
    except OSError as error:
        raise ValueError("qualification ancestry is unavailable") from error


def _validate_commit_identity(
    repository_root: Path,
    commit: str,
    label: str,
) -> None:
    try:
        raw = strict_runner.git_bytes(
            repository_root,
            GIT_NO_REPLACE_OBJECTS,
            "show",
            "-s",
            "--format=%an%x00%ae%x00%cn%x00%ce",
            commit,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError(f"{label} identity is unavailable") from error
    expected = (
        EXPECTED_AUTHOR_NAME.encode("utf-8")
        + b"\0"
        + EXPECTED_AUTHOR_EMAIL.encode("ascii")
        + b"\0"
        + EXPECTED_AUTHOR_NAME.encode("utf-8")
        + b"\0"
        + EXPECTED_AUTHOR_EMAIL.encode("ascii")
        + b"\n"
    )
    if raw != expected:
        raise ValueError(f"{label} author or committer identity changed")


def _ancestry_records(
    repository_root: Path,
    source_commit: str,
    validation_head: str,
) -> list[tuple[str, tuple[str, ...]]]:
    try:
        raw = strict_runner.git_bytes(
            repository_root,
            GIT_NO_REPLACE_OBJECTS,
            "rev-list",
            "--parents",
            "--ancestry-path",
            f"{source_commit}..{validation_head}",
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError("bundle ancestry record is unavailable") from error
    if not raw or not raw.endswith(b"\n") or b"\r" in raw or b"\t" in raw:
        raise ValueError("bundle ancestry record is malformed")
    try:
        lines = raw[:-1].decode("ascii").split("\n")
    except UnicodeDecodeError as error:
        raise ValueError("bundle ancestry record is malformed") from error
    records: list[tuple[str, tuple[str, ...]]] = []
    observed: set[str] = set()
    for line in lines:
        fields = line.split(" ")
        if not fields or " ".join(fields) != line:
            raise ValueError("bundle ancestry record is malformed")
        for field in fields:
            _validated_commit_digest(field, "bundle ancestry entry")
        commit, *parents = fields
        if commit in observed:
            raise ValueError("bundle ancestry record repeats a commit")
        observed.add(commit)
        records.append((commit, tuple(parents)))
    if records[0][0] != validation_head:
        raise ValueError("bundle ancestry record is malformed")
    return records


def _commit_changed_paths(
    repository_root: Path,
    commit: str,
) -> list[tuple[str, str]]:
    try:
        raw = strict_runner.git_bytes(
            repository_root,
            GIT_NO_REPLACE_OBJECTS,
            "diff-tree",
            "--no-commit-id",
            "-r",
            "--no-renames",
            "--name-status",
            "-z",
            commit,
            "--",
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError("bundle changed-path record is unavailable") from error
    if not raw:
        return []
    if not raw.endswith(b"\0"):
        raise ValueError("bundle changed-path record is malformed")
    fields = raw[:-1].split(b"\0")
    if len(fields) % 2:
        raise ValueError("bundle changed-path record is malformed")
    records: list[tuple[str, str]] = []
    observed: set[str] = set()
    for index in range(0, len(fields), 2):
        try:
            status = fields[index].decode("ascii")
            path = fields[index + 1].decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError("bundle changed-path record is malformed") from error
        if not status or not path or path in observed:
            raise ValueError("bundle changed-path record is malformed")
        observed.add(path)
        records.append((status, path))
    return records


def _commit_regular_blob(
    repository_root: Path,
    commit: str,
    relative_path: str,
    label: str,
    *,
    allowed_modes: frozenset[str] = frozenset({"100644"}),
) -> tuple[bytes, dict[str, str]]:
    try:
        raw_entry = strict_runner.git_bytes(
            repository_root,
            GIT_NO_REPLACE_OBJECTS,
            "ls-tree",
            "-z",
            commit,
            "--",
            relative_path,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError(f"{label} tree entry is unavailable") from error
    if not raw_entry or not raw_entry.endswith(b"\0"):
        raise ValueError(f"{label} tree entry is missing or malformed")
    entries = raw_entry[:-1].split(b"\0")
    if len(entries) != 1 or entries[0].count(b"\t") != 1:
        raise ValueError(f"{label} tree entry is ambiguous or malformed")
    raw_header, raw_path = entries[0].split(b"\t", 1)
    try:
        header = raw_header.decode("ascii")
        observed_path = raw_path.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"{label} tree entry is malformed") from error
    fields = header.split(" ")
    if len(fields) != 3 or " ".join(fields) != header:
        raise ValueError(f"{label} tree entry is malformed")
    mode, object_type, object_id = fields
    _validated_commit_digest(object_id, f"{label} blob object")
    if (
        observed_path != relative_path
        or mode not in allowed_modes
        or object_type != "blob"
    ):
        raise ValueError(f"{label} must be a regular frozen blob")
    try:
        blob = strict_runner.git_bytes(
            repository_root,
            GIT_NO_REPLACE_OBJECTS,
            "cat-file",
            "blob",
            object_id,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError(f"{label} blob is unavailable") from error
    return blob, {
        "git_mode": mode,
        "git_object_type": object_type,
        "git_blob_id": object_id,
    }


def validate_frozen_dependencies(
    registry: dict[str, Any],
    repository_root: Path = REPOSITORY_ROOT,
) -> None:
    for entry in registry["parent_artifacts"]:
        path = repository_root / entry["path"]
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"parent artifact is missing: {entry['path']}")
        if sha256_file(path) != entry["sha256"]:
            raise ValueError(f"parent artifact bytes changed: {entry['path']}")
    source_commit = registry["implementation_source_commit"]
    _resolved_commit(repository_root, source_commit, "implementation source commit")
    _validate_commit_identity(
        repository_root,
        source_commit,
        "implementation source commit",
    )
    validation_head = _current_head_commit(repository_root)
    _require_ancestor(
        repository_root,
        source_commit,
        validation_head,
        "validation HEAD must descend from the implementation source commit",
    )
    for entry in registry["implementation_sources"]:
        committed_bytes, _metadata = _commit_regular_blob(
            repository_root,
            source_commit,
            entry["path"],
            f"implementation source {entry['path']}",
            allowed_modes=frozenset({"100644", "100755"}),
        )
        if hashlib.sha256(committed_bytes).hexdigest() != entry["sha256"]:
            raise ValueError(
                "implementation source differs from its recorded commit: "
                f"{entry['path']}"
            )


def observe_implementation_sources(
    registry: dict[str, Any],
    repository_root: Path = REPOSITORY_ROOT,
) -> dict[str, Any]:
    source_commit = registry["implementation_source_commit"]
    records: list[dict[str, Any]] = []
    for entry in registry["implementation_sources"]:
        observed: str | None = None
        try:
            committed_bytes, _metadata = _commit_regular_blob(
                repository_root,
                source_commit,
                entry["path"],
                f"implementation source {entry['path']}",
                allowed_modes=frozenset({"100644", "100755"}),
            )
        except ValueError:
            pass
        else:
            observed = hashlib.sha256(committed_bytes).hexdigest()
        matches = observed == entry["sha256"]
        records.append(
            {
                "path": entry["path"],
                "expected_sha256": entry["sha256"],
                "observed_sha256": observed,
                "matches_recorded_source": matches,
            }
        )
    matching = sum(record["matches_recorded_source"] for record in records)
    return {
        "observation_authority": "recorded_implementation_source_commit",
        "observation_commit": source_commit,
        "inventory_count": len(records),
        "matching_count": matching,
        "drift_count": len(records) - matching,
        "missing_count": sum(
            record["observed_sha256"] is None for record in records
        ),
        "all_match": matching == len(records),
        "records": records,
    }


def _canonical_bundle_working_bytes(
    registry: dict[str, Any],
    matrix_path: Path,
    repository_root: Path,
    runner_path: Path,
) -> tuple[list[tuple[str, str, bytes]], str]:
    actual_paths = {
        "matrix": matrix_path,
        "runner": runner_path,
        "focused_test": repository_root / EXPECTED_FOCUSED_TEST_PATH,
    }
    expected_paths = {
        "matrix": EXPECTED_MATRIX_PATH,
        "runner": EXPECTED_RUNNER_PATH,
        "focused_test": EXPECTED_FOCUSED_TEST_PATH,
    }
    working: list[tuple[str, str, bytes]] = []
    for role in ("matrix", "runner", "focused_test"):
        actual = actual_paths[role]
        expected = (repository_root / expected_paths[role]).resolve()
        if actual.resolve() != expected:
            raise ValueError(f"qualification bundle {role} path is not canonical")
        if actual.is_symlink() or not actual.is_file():
            raise ValueError(f"qualification bundle {role} is not a regular file")
        working.append((role, expected_paths[role], actual.read_bytes()))
    matrix_bytes = working[0][2]
    runner_bytes = working[1][2]
    focused_bytes = working[2][2]
    normalized_digest = hashlib.sha256(
        normalized_registry_bytes(matrix_bytes)
    ).hexdigest()
    if normalized_digest != EXPECTED_NORMALIZED_REGISTRY_SHA256:
        raise ValueError("matrix does not match the embedded normalized digest")
    try:
        matrix_document = json.loads(
            matrix_bytes.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("qualification bundle matrix is malformed") from error
    matrix_runner_digest = matrix_document.get("runner_sha256")
    if matrix_runner_digest != registry.get("runner_sha256"):
        raise ValueError("matrix runner hash differs from the loaded registry")
    if hashlib.sha256(runner_bytes).hexdigest() != matrix_runner_digest:
        raise ValueError("runner bytes differ from the matrix runner hash")
    if hashlib.sha256(focused_bytes).hexdigest() != EXPECTED_FOCUSED_TEST_SHA256:
        raise ValueError("focused test bytes differ from the embedded hash")
    return working, normalized_digest


def validate_frozen_qualification_bundle(
    registry: dict[str, Any],
    matrix_path: Path = DEFAULT_REGISTRY,
    repository_root: Path = REPOSITORY_ROOT,
    runner_path: Path = SCRIPT_PATH,
) -> dict[str, Any]:
    if registry.get("status") != EXECUTABLE_MATRIX_STATUS:
        raise ValueError("qualification bundle requires a frozen matrix")
    if registry.get("qualification_bundle_binding") != (
        _frozen_bundle_binding()
    ):
        raise ValueError("qualification bundle exact-path contract changed")
    source_commit = registry["implementation_source_commit"]
    validation_head = _current_head_commit(repository_root)
    _require_ancestor(
        repository_root,
        source_commit,
        validation_head,
        "validation HEAD must descend from the implementation source commit",
    )
    if validation_head == source_commit:
        raise ValueError("qualification history contains no bundle candidate")
    ancestry = _ancestry_records(
        repository_root,
        source_commit,
        validation_head,
    )
    direct_children = [
        commit
        for commit, parents in ancestry
        if parents == (source_commit,)
    ]
    if not direct_children:
        raise ValueError("qualification history contains no direct-child bundle")
    working, normalized_digest = _canonical_bundle_working_bytes(
        registry,
        matrix_path,
        repository_root,
        runner_path,
    )
    matching: list[tuple[str, dict[str, dict[str, str]]]] = []
    exact_path_candidates: list[str] = []
    blob_drift_candidates: list[str] = []
    for candidate in direct_children:
        changed = _commit_changed_paths(repository_root, candidate)
        if (
            len(changed) != len(EXPECTED_BUNDLE_PATHS)
            or any(status != "A" for status, _path in changed)
            or {path for _status, path in changed} != set(EXPECTED_BUNDLE_PATHS)
        ):
            continue
        exact_path_candidates.append(candidate)
        metadata: dict[str, dict[str, str]] = {}
        candidate_matches = True
        for role, path, working_bytes in working:
            allowed_modes = (
                frozenset({"100755"})
                if role == "runner"
                else frozenset({"100644"})
            )
            committed_bytes, blob_metadata = _commit_regular_blob(
                repository_root,
                candidate,
                path,
                f"qualification bundle {path}",
                allowed_modes=allowed_modes,
            )
            metadata[path] = blob_metadata
            if committed_bytes != working_bytes:
                candidate_matches = False
        if candidate_matches:
            matching.append((candidate, metadata))
        else:
            blob_drift_candidates.append(candidate)
    if len(matching) != 1:
        if not matching and blob_drift_candidates:
            raise ValueError("canonical bundle candidate has frozen blob drift")
        if not matching and not exact_path_candidates:
            raise ValueError("direct-child changed paths do not equal bundle paths")
        raise ValueError(
            "qualification history must contain exactly one canonical bundle "
            f"candidate; observed {len(matching)}"
        )
    bundle_commit, metadata_by_path = matching[0]
    _validate_commit_identity(repository_root, bundle_commit, "bundle commit")
    _require_ancestor(
        repository_root,
        bundle_commit,
        validation_head,
        "validation HEAD must descend from the canonical bundle commit",
    )
    artifacts: list[dict[str, Any]] = []
    for role, path, working_bytes in working:
        artifact: dict[str, Any] = {
            "role": role,
            "path": path,
            "sha256": hashlib.sha256(working_bytes).hexdigest(),
            **metadata_by_path[path],
        }
        if role == "matrix":
            artifact["normalized_sha256"] = normalized_digest
        artifacts.append(artifact)
    return {
        "binding_schema_version": 3,
        "authority": EXPECTED_FROZEN_BUNDLE_AUTHORITY,
        "bundle_commit_resolution": EXPECTED_BUNDLE_COMMIT_RESOLUTION,
        "qualification_bundle_commit": bundle_commit,
        "validation_head_commit": validation_head,
        "implementation_source_commit": source_commit,
        "bundle_parent_commit": source_commit,
        "bundle_changed_paths": sorted(EXPECTED_BUNDLE_PATHS),
        "normalized_matrix_sha256_embedded_in_runner": normalized_digest,
        "runner_sha256_from_matrix": registry["runner_sha256"],
        "focused_test_sha256_from_runner": EXPECTED_FOCUSED_TEST_SHA256,
        "artifacts": artifacts,
    }


def load_registry(path: Path) -> dict[str, Any]:
    global _frozen_qualification_bundle_binding
    _frozen_qualification_bundle_binding = None
    if path.resolve() != DEFAULT_REGISTRY.resolve():
        raise ValueError("qualification requires the canonical V3 matrix")
    if path.is_symlink() or not path.is_file():
        raise ValueError("canonical V3 matrix must be a regular file")
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
    validate_frozen_dependencies(registry)
    if registry["status"] == EXECUTABLE_MATRIX_STATUS:
        if EXPECTED_FOCUSED_TEST_SHA256 == RUNNER_SHA256_ZERO_SENTINEL:
            raise RuntimeError("focused test digest is not finalized")
        _frozen_qualification_bundle_binding = (
            validate_frozen_qualification_bundle(registry, path)
        )
    return registry


def _validate_execution_source_worktree(
    source_root: Path,
) -> dict[str, str]:
    resolved_source_root = source_root.resolve()
    if resolved_source_root != REPOSITORY_ROOT.resolve():
        raise ValueError("execution source root must equal runner repository root")
    try:
        strict_runner.git_bytes(
            resolved_source_root,
            "symbolic-ref",
            "-q",
            "HEAD",
        )
    except subprocess.CalledProcessError as error:
        if error.returncode != 1:
            raise ValueError("source worktree HEAD state is unavailable") from error
    except OSError as error:
        raise ValueError("source worktree HEAD state is unavailable") from error
    else:
        raise ValueError("qualification execution requires detached HEAD")
    binding = _frozen_qualification_bundle_binding
    if not isinstance(binding, dict):
        raise ValueError("frozen bundle binding is unavailable")
    bundle_commit = binding["qualification_bundle_commit"]
    execution_head = _current_head_commit(resolved_source_root)
    if execution_head != bundle_commit:
        raise ValueError("execution HEAD must equal the canonical bundle commit")
    try:
        common_text = (
            strict_runner.git_bytes(
                resolved_source_root,
                "rev-parse",
                "--git-common-dir",
            )
            .decode("utf-8")
            .strip()
        )
    except (
        OSError,
        subprocess.CalledProcessError,
        UnicodeDecodeError,
    ) as error:
        raise ValueError("Git common directory is unavailable") from error
    common = Path(common_text)
    if not common.is_absolute():
        common = resolved_source_root / common
    common = common.resolve()
    if strict_runner.path_is_within(common, resolved_source_root):
        raise ValueError("qualification requires an external Git common directory")
    return {
        "source_root": str(resolved_source_root),
        "git_common_directory": str(common),
        "qualification_bundle_commit": bundle_commit,
        "execution_head_commit": execution_head,
    }


def untracked_source_record(
    source_root: Path,
    allowed_output_root: Path | None = None,
    ignored_source_roots: tuple[Path, ...] = (),
) -> dict[str, Any]:
    resolved = source_root.resolve()
    scan_roots = tuple(
        root for root in ignored_source_roots if root.resolve() != resolved
    ) + (resolved,)
    return _shared_untracked_source_record(
        resolved,
        allowed_output_root,
        scan_roots,
    )


def _existing_filesystem_probe(path: Path) -> Path:
    probe = path.resolve()
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    if not probe.exists():
        raise ValueError(f"cannot locate a filesystem ancestor for {path}")
    return probe


def _positive_scheduler_integer(
    environment: Mapping[str, str],
    name: str,
) -> int:
    raw = environment.get(name)
    if not isinstance(raw, str) or not re.fullmatch(r"[1-9][0-9]*", raw):
        raise ValueError(f"scheduler field {name} is missing or malformed")
    return int(raw)


def _validate_scheduler_allocation(
    safeguards: dict[str, Any],
    environment: Mapping[str, str],
) -> dict[str, int | str]:
    if environment.get("SLURM_JOB_ACCOUNT") != safeguards["scheduler_account"]:
        raise ValueError("qualification requires the recorded scheduler account")
    if environment.get("SLURM_JOB_PARTITION") != safeguards["scheduler_partition"]:
        raise ValueError("qualification requires the recorded scheduler partition")
    node_name = (
        "SLURM_JOB_NUM_NODES"
        if environment.get("SLURM_JOB_NUM_NODES") is not None
        else "SLURM_NNODES"
    )
    nodes = _positive_scheduler_integer(environment, node_name)
    tasks = _positive_scheduler_integer(environment, "SLURM_NTASKS")
    cpus_per_task = _positive_scheduler_integer(
        environment, "SLURM_CPUS_PER_TASK"
    )
    cpus_on_node = _positive_scheduler_integer(
        environment, "SLURM_CPUS_ON_NODE"
    )
    memory_mib = _positive_scheduler_integer(
        environment, "SLURM_MEM_PER_NODE"
    )
    expected_nodes = safeguards["qualification_job_nodes"]
    expected_tasks = safeguards["qualification_job_tasks"]
    expected_cpus_per_task = safeguards["qualification_job_cpus_per_task"]
    expected_memory_mib = safeguards["qualification_job_memory_gib"] * 1024
    if nodes != expected_nodes:
        raise ValueError("qualification job node count changed")
    if tasks != expected_tasks:
        raise ValueError("qualification job task count changed")
    if cpus_per_task != expected_cpus_per_task:
        raise ValueError("qualification CPUs per task changed")
    if cpus_on_node != tasks * cpus_per_task:
        raise ValueError("qualification CPU allocation is inconsistent")
    if cpus_on_node < safeguards["build_parallel"]:
        raise ValueError("qualification CPU allocation is below build parallelism")
    if memory_mib != expected_memory_mib:
        raise ValueError("qualification job memory changed")
    if nodes > safeguards["maximum_concurrent_nodes"]:
        raise ValueError("qualification node allocation exceeds the ceiling")
    if nodes * memory_mib > safeguards["maximum_concurrent_memory_gib"] * 1024:
        raise ValueError("qualification memory allocation exceeds the ceiling")
    return {
        "scheduler_account": safeguards["scheduler_account"],
        "scheduler_partition": safeguards["scheduler_partition"],
        "nodes": nodes,
        "tasks": tasks,
        "cpus_per_task": cpus_per_task,
        "cpus_on_node": cpus_on_node,
        "memory_mib": memory_mib,
    }


def require_execution_resource_preflight(
    source_root: Path,
    output_directory: Path,
    build_directories: tuple[Path, ...],
    registry: dict[str, Any],
) -> None:
    _validate_execution_source_worktree(source_root)
    safeguards = registry["resource_safeguards"]
    scratch_root = Path(safeguards["scratch_root"]).resolve()
    observed_python = ".".join(
        str(value) for value in sys.version_info[:3]
    )
    if observed_python != safeguards["python_version"]:
        raise ValueError("qualification Python version changed")
    if Path(sys.executable).resolve() != Path(
        safeguards["python_executable"]
    ).resolve():
        raise ValueError("qualification Python executable changed")
    loaded_modules = set(
        value
        for value in os.environ.get("LOADEDMODULES", "").split(":")
        if value
    )
    if safeguards["python_runtime_module"] not in loaded_modules:
        raise ValueError("qualification Python module is not loaded")
    selected_paths = (output_directory.resolve(), *build_directories)
    if len(build_directories) != 3:
        raise ValueError("qualification requires exactly three fresh build homes")
    if (
        safeguards["source_worktree_must_be_within_scratch_root"]
        and not strict_runner.path_is_within(source_root.resolve(), scratch_root)
    ):
        raise ValueError("qualification source worktree must remain in scratch")
    if any(
        not strict_runner.path_is_within(path.resolve(), scratch_root)
        for path in selected_paths
    ):
        raise ValueError("qualification build and output paths must remain in scratch")
    _validate_scheduler_allocation(safeguards, os.environ)
    available = strict_runner.host_available_memory_mib()
    required_memory = safeguards["execution_preflight_mem_available_mib"]
    if available is None or available < required_memory:
        raise ValueError(
            f"qualification requires at least {required_memory} MiB available"
        )
    free_floor = safeguards["runtime_filesystem_free_floor_mib"]
    for path in (source_root, output_directory.parent, *build_directories):
        probe = _existing_filesystem_probe(path)
        free_mib = strict_runner.filesystem_free_mib(probe)
        if free_mib is None or free_mib < free_floor:
            raise ValueError(
                f"qualification filesystem free-space floor failed at {probe}"
            )


def _cmake_cache_definition_name(argument: str) -> str:
    if argument == "-D" or not argument.startswith("-D"):
        raise ValueError("CMake definitions must use joined -DNAME=VALUE arguments")
    name_with_type, separator, _value = argument[2:].partition("=")
    name = name_with_type.partition(":")[0]
    if not separator or not name:
        raise ValueError("CMake definition is malformed")
    return name


def _locked_fresh_configure_command(
    command: list[str],
    source_root: Path,
) -> list[str]:
    if not command or command.count("-S") != 1 or command.count("-B") != 1:
        raise ValueError("CMake configure route must contain one -S and one -B")
    if command.count("--fresh") > 1:
        raise ValueError("CMake configure route repeats --fresh")
    caller_indices: set[int] = set()
    caller_definitions: dict[str, str] = {}
    for index, argument in enumerate(command):
        if argument == "-D":
            raise ValueError("CMake definitions must use joined arguments")
        if not argument.startswith("-D"):
            continue
        name = _cmake_cache_definition_name(argument)
        if name in caller_definitions:
            raise ValueError(f"CMake definition is ambiguous: {name}")
        caller_definitions[name] = argument
        caller_indices.add(index)
    structural = [
        argument
        for index, argument in enumerate(command)
        if index not in caller_indices and argument != "--fresh"
    ]
    if (
        len(structural) != 5
        or structural[1] != "-S"
        or structural[3] != "-B"
    ):
        raise ValueError("CMake configure route contains unexpected arguments")
    source_home = Path(structural[2])
    if not source_home.is_absolute():
        source_home = source_root.resolve() / source_home
    source_home = source_home.resolve()
    selected_home: str | None = None
    locked: tuple[str, ...] | None = None
    for relative_home, definitions in _active_fresh_configure_definitions.items():
        if source_home == (source_root.resolve() / relative_home).resolve():
            selected_home = relative_home
            locked = definitions
            break
    if selected_home is None or locked is None:
        raise ValueError("CMake configure source home is not recognized")
    locked_by_name = {
        _cmake_cache_definition_name(definition): definition
        for definition in locked
    }
    for name, argument in caller_definitions.items():
        if locked_by_name.get(name) != argument:
            raise ValueError(
                f"CMake definition conflicts with locked value: {name}"
            )
    return [structural[0], "--fresh", *locked, *structural[1:]]


def run_build_phase(
    command: list[str],
    source_root: Path,
    output_root: Path,
    stdout_path: Path,
    stderr_path: Path,
    timeout_seconds: int,
) -> dict[str, Any]:
    exact = list(command)
    if "--build" in exact:
        if (
            "-S" in exact
            or "-B" in exact
            or "--fresh" in exact
            or any(value == "-D" or value.startswith("-D") for value in exact)
        ):
            raise ValueError("CMake build and configure routes are ambiguous")
    else:
        exact = _locked_fresh_configure_command(exact, source_root)
    return _shared_run_build_phase(
        exact,
        source_root,
        output_root,
        stdout_path,
        stderr_path,
        timeout_seconds,
    )


def _inject_provenance(value: dict[str, Any]) -> None:
    value["implementation_source_commit"] = (
        EXPECTED_IMPLEMENTATION_SOURCE_COMMIT
    )
    value["qualification_scope"] = EXPECTED_SCOPE
    value["requested_claim"] = ACCEPTED_CLAIM
    value["qualification_disposition"] = copy.deepcopy(
        EXPECTED_QUALIFICATION_DISPOSITION
    )
    value["open_outcomes"] = copy.deepcopy(EXPECTED_OPEN_OUTCOMES)
    if _frozen_qualification_bundle_binding is not None:
        value["qualification_bundle_binding"] = copy.deepcopy(
            _frozen_qualification_bundle_binding
        )


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
        _inject_provenance(value)
        if path.name == "summary.json":
            passed = value.get("overall_outcome") == "PASS"
            value["fsr16_closed_within_declared_envelope"] = passed
            value["wp3_closed_within_declared_envelope"] = passed
            value["wp7_closed"] = False
            value["q1_closed"] = False
    _shared_write_json(path, value)


def write_text(path: Path, value: str) -> None:
    if path.name == "record.md":
        summary_path = path.parent / "summary.json"
        outcome = "UNKNOWN"
        if summary_path.is_file():
            try:
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                summary = {}
            outcome = str(summary.get("overall_outcome", "UNKNOWN"))
        value += (
            "\n## WP-3 closure boundary\n\n"
            + EXPECTED_SCOPE
            + "\n\n"
            + (
                "FSR-16 and WP-3 close within the declared envelope."
                if outcome == "PASS"
                else "FSR-16 and WP-3 remain open."
            )
            + " WP-7 and Q1 remain open.\n"
        )
    _shared_write_text(path, value)


def requested_claim(arguments: list[str]) -> tuple[str, bool, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--requested-claim", default=ACCEPTED_CLAIM)
    parser.add_argument("--validate-only", action="store_true")
    parsed, remaining = parser.parse_known_args(arguments)
    claim = parsed.requested_claim
    rejected = set(EXPECTED_CLOSURE_REQUEST_POLICY["rejected_claims"])
    if claim in rejected:
        raise ValueError(
            f"requested claim {claim!r} is outside this matrix: "
            f"{EXPECTED_CLOSURE_REQUEST_POLICY['diagnostic']}"
        )
    if claim != ACCEPTED_CLAIM:
        raise ValueError(
            f"unsupported requested claim {claim!r}; expected {ACCEPTED_CLAIM!r}"
        )
    return claim, parsed.validate_only, remaining


def validate_only_summary(
    registry: dict[str, Any],
    claim: str,
) -> dict[str, Any]:
    source_observation = observe_implementation_sources(registry)
    binding = _frozen_qualification_bundle_binding
    bundle_commit = (
        binding.get("qualification_bundle_commit")
        if isinstance(binding, dict)
        else None
    )
    validation_head = (
        binding.get("validation_head_commit")
        if isinstance(binding, dict)
        else None
    )
    at_bundle = (
        registry["status"] == EXECUTABLE_MATRIX_STATUS
        and isinstance(bundle_commit, str)
        and validation_head == bundle_commit
    )
    return {
        "matrix_id": registry["matrix_id"],
        "status": registry["status"],
        "execution_ready": at_bundle,
        "validation_scope": (
            "draft_structure_and_source_validation"
            if registry["status"] == DRAFT_MATRIX_STATUS
            else (
                "frozen_execution_preflight"
                if at_bundle
                else "frozen_historical_validation"
            )
        ),
        "qualification_bundle_commit": bundle_commit,
        "validation_head_commit": validation_head,
        "implementation_source_observation": source_observation,
        "requested_claim": claim,
        "group_count": len(registry["groups"]),
        "test_count": sum(len(group["tests"]) for group in registry["groups"]),
        "quantitative_evidence_gate_count": len(
            registry["quantitative_evidence"]
        ),
        "recorded_property_gate_count": sum(
            len(group.get("recorded_properties", []))
            for group in registry["groups"]
        ),
        "qualification_disposition": copy.deepcopy(
            EXPECTED_QUALIFICATION_DISPOSITION
        ),
        "open_outcomes": copy.deepcopy(EXPECTED_OPEN_OUTCOMES),
        "closure_outcome": (
            "PENDING_EXECUTION_FOR_DECLARED_ENVELOPE"
        ),
        "outcome": (
            "PASS_DRAFT_STRUCTURE_ONLY"
            if registry["status"] == DRAFT_MATRIX_STATUS
            and source_observation["all_match"]
            else (
                "PASS_FROZEN_VALIDATION"
                if source_observation["all_match"]
                else "SOURCE_DRIFT"
            )
        ),
    }


def execution_argument_parser() -> argparse.ArgumentParser:
    safeguards = parse_json_document(DEFAULT_REGISTRY)["resource_safeguards"]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--geometry-binary", type=Path, required=True)
    parser.add_argument("--level-set-binary", type=Path, required=True)
    parser.add_argument("--systems-binary", type=Path, required=True)
    parser.add_argument("--assembly-binary", type=Path, required=True)
    parser.add_argument("--math-binary", type=Path, required=True)
    parser.add_argument("--physics-binary", type=Path, required=True)
    parser.add_argument("--application-binary", type=Path, required=True)
    parser.add_argument("--assembly-mpi-binary", type=Path, required=True)
    parser.add_argument("--application-mpi-binary", type=Path, required=True)
    parser.add_argument(
        "--mpiexec",
        type=Path,
        default=Path(
            "/share/software/user/open/openmpi/4.1.2/bin/mpiexec"
        ),
    )
    parser.add_argument(
        "--cmake",
        type=Path,
        default=Path("/share/software/user/open/cmake/3.31.4/bin/cmake"),
    )
    parser.add_argument(
        "--build-parallel",
        type=int,
        default=safeguards["build_parallel"],
    )
    parser.add_argument(
        "--build-timeout-seconds",
        type=int,
        default=7200,
    )
    parser.add_argument("--source-root", type=Path, default=REPOSITORY_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(arguments: list[str] | None = None) -> int:
    selected = list(sys.argv[1:]) if arguments is None else list(arguments)
    claim, validate_only, remaining = requested_claim(selected)
    registry = load_registry(DEFAULT_REGISTRY)
    if validate_only:
        if remaining:
            raise ValueError("--validate-only does not accept execution arguments")
        print(json.dumps(validate_only_summary(registry, claim), sort_keys=True))
        return 0
    if registry["status"] != EXECUTABLE_MATRIX_STATUS:
        raise ValueError("full execution requires a frozen matrix and bundle")
    parser = execution_argument_parser()
    execution = parser.parse_args(remaining)
    if execution.registry.resolve() != DEFAULT_REGISTRY.resolve():
        parser.error("execution requires the canonical matrix")
    safeguards = registry["resource_safeguards"]
    if execution.build_parallel != safeguards["build_parallel"]:
        parser.error("clean-build parallelism differs from the frozen value")
    if not 1 <= execution.build_timeout_seconds <= 7200:
        parser.error("build timeout must be in the closed interval [1, 7200]")
    binaries = {
        "geometry": execution.geometry_binary,
        "level_set": execution.level_set_binary,
        "systems": execution.systems_binary,
        "assembly": execution.assembly_binary,
        "math": execution.math_binary,
        "physics": execution.physics_binary,
        "application": execution.application_binary,
        "assembly_mpi": execution.assembly_mpi_binary,
        "application_mpi": execution.application_mpi_binary,
    }
    build_directories = tuple(
        sorted(
            {
                cache.resolve().parent
                for binary in binaries.values()
                if (
                    cache := strict_runner.find_cmake_cache(binary.resolve())
                ) is not None
            },
            key=str,
        )
    )
    require_execution_resource_preflight(
        execution.source_root.resolve(),
        execution.output.resolve(),
        build_directories,
        registry,
    )
    saved_status = strict_runner.EXPECTED_MATRIX_STATUS
    try:
        strict_runner.EXPECTED_MATRIX_STATUS = registry["status"]
        return strict_runner.run_qualification(
            execution,
            binaries,
            expected_binary_keys=EXPECTED_BINARY_KEYS,
            parser=parser,
            record_title=(
                "WP-3 sharp exterior-boundary closure qualification record"
            ),
        )
    finally:
        strict_runner.EXPECTED_MATRIX_STATUS = saved_status


strict_runner.SCRIPT_PATH = SCRIPT_PATH
strict_runner.DEFAULT_REGISTRY = DEFAULT_REGISTRY
strict_runner.EXPECTED_MATRIX_ID = EXPECTED_MATRIX_ID
strict_runner.EXPECTED_MATRIX_STATUS = EXPECTED_CHECKED_IN_MATRIX_STATUS
strict_runner.EXPECTED_WORK_PACKAGE = EXPECTED_WORK_PACKAGE
strict_runner.QUALIFICATION_BINARY_KEYS = set(EXPECTED_BINARY_KEYS)
strict_runner.load_registry = load_registry
strict_runner.write_json = write_json
strict_runner.write_text = write_text
strict_runner.untracked_source_record = untracked_source_record
strict_runner.run_build_phase = run_build_phase
strict_runner.__doc__ = __doc__


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)

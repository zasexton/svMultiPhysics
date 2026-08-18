#!/usr/bin/env python3
"""Run the additive WP-3/WP-7 exact-dyadic and aggregate-trace prerequisite."""

from __future__ import annotations

import argparse
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


SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIRECTORY = SCRIPT_PATH.parent
REPOSITORY_ROOT = SCRIPT_PATH.parents[3]
DEFAULT_REGISTRY = SCRIPT_PATH.with_name(
    "free_surface_wp3_wp7_nitsche_coercivity_qualification_matrix_v2.json"
)
SHARED_RUNNER_PATH = SCRIPT_PATH.with_name(
    "run_free_surface_wp2_geometry_qualification.py"
)
V1_MATRIX_PATH = SCRIPT_PATH.with_name(
    "free_surface_wp3_wp7_nitsche_coercivity_qualification_matrix.json"
)
V1_RUNNER_PATH = SCRIPT_PATH.with_name(
    "run_free_surface_wp3_wp7_nitsche_coercivity_qualification.py"
)
V1_METHOD_PATH = (
    REPOSITORY_ROOT
    / "Documentation"
    / "free_surface_wp3_wp7_symmetric_nitsche_coercivity_method.md"
)

EXPECTED_NORMALIZED_REGISTRY_SHA256 = (
    "59b3c9f0c496659b92cafd09af30a69c8999a5e7421de70f0d92def4f7dec261"
)
RUNNER_SHA256_ZERO_SENTINEL = "0" * 64
EXPECTED_SHARED_RUNNER_SHA256 = (
    "4b2e8a97ff080450c560e921e58a9dd92474ff561faabbcb10bacd6264f542d9"
)
EXPECTED_V1_PARENT_SHA256 = {
    (
        "tests/cases/fluid/"
        "free_surface_wp3_wp7_nitsche_coercivity_qualification_matrix.json"
    ): "a75bbec8efe800f049375f190c07a121b3e365098da783b43ec1ba9df9610589",
    (
        "tests/cases/fluid/"
        "run_free_surface_wp3_wp7_nitsche_coercivity_qualification.py"
    ): "353c49c10881fd13acececb80cdf000c70abf7937f10021a2816d04d90bb9181",
    "Documentation/free_surface_wp3_wp7_symmetric_nitsche_coercivity_method.md": (
        "abc782ef828b3fd3996257f5544f85221d9c6d047b1cf730848ac93b695c6ead"
    ),
}

EXPECTED_MATRIX_ID = (
    "free_surface_wp3_wp7_symmetric_nitsche_certified_trace_prerequisite_v2"
)
EXPECTED_MATRIX_STATUS = "DRAFT_UNEXECUTED"
EXECUTABLE_MATRIX_STATUS = "FROZEN_BEFORE_EXECUTION"
ALLOWED_MATRIX_STATUSES = {
    EXPECTED_MATRIX_STATUS,
    EXECUTABLE_MATRIX_STATUS,
}
EXPECTED_DRAFT_SOURCE_HASH_STATUS = "DRAFT_OBSERVED_NOT_FROZEN"
EXPECTED_FROZEN_SOURCE_HASH_STATUS = "FROZEN"
EXPECTED_WORK_PACKAGE = "WP-3/WP-7"
EXPECTED_MATCHING_DERIVATION = (
    "Documentation/"
    "free_surface_wp3_wp7_symmetric_nitsche_coercivity_method_v2.md"
)
EXPECTED_PROPOSED_RUNNER = (
    "tests/cases/fluid/"
    "run_free_surface_wp3_wp7_nitsche_coercivity_qualification_v2.py"
)
EXPECTED_MATRIX_PATH = (
    "tests/cases/fluid/"
    "free_surface_wp3_wp7_nitsche_coercivity_qualification_matrix_v2.json"
)
EXPECTED_QUALIFICATION_BUNDLE_BINDING = {
    "authority": (
        "reciprocal_SHA256_plus_clean_qualification_execution_HEAD"
    ),
    "matrix_sha256_source": "runner_embedded_normalized_matrix_SHA256",
    "matrix_hash_normalization": (
        "replace_the_unique_runner_sha256_64_lowercase_hex_JSON_value_with_"
        "64_ASCII_zero_digits"
    ),
    "runner_sha256_source": "matrix_runner_sha256",
    "exact_HEAD_blobs_required": [
        EXPECTED_MATRIX_PATH,
        EXPECTED_PROPOSED_RUNNER,
    ],
    "HEAD_must_descend_from_implementation_source_commit": True,
}
EXPECTED_SCOPE = (
    "Certified finite-dimensional aggregate trace and symmetric-Nitsche "
    "prerequisite evidence in the stated affine P1 constant-viscosity "
    "envelope; this matrix does not close FSR-16, FSR-07, WP-3, WP-7, Q1, "
    "or establish a uniform method coercivity bound."
)
EXPECTED_CLOSURE_REQUEST_POLICY = {
    "accepted_claim": "joint_low_level_prerequisite",
    "rejected_claims": [
        "fsr16_closure",
        "fsr07_closure",
        "wp3_closure",
        "wp7_closure",
        "wp3_wp7_joint_closure",
        "q1_closure",
    ],
    "diagnostic": (
        "The certificate is a revision-bound finite-dimensional trace bound "
        "for a deliberately narrow production route. It does not supply the "
        "full operator, element-family, constitutive, topology, convergence, "
        "pressure-stability, solver, and MPI-rank envelope required for "
        "closure."
    ),
}
EXPECTED_DISPOSITION = {
    "fsr16_closed": False,
    "fsr07_closed": False,
    "wp3_closed": False,
    "wp7_closed": False,
    "q1_closed": False,
    "uniform_coercivity_bound_established": False,
}
EXPECTED_OPEN_OUTCOMES = {
    "fsr16": "OPEN",
    "fsr07": "OPEN",
    "wp3": "OPEN",
    "wp7": "OPEN",
    "q1": "OPEN",
}
EXPECTED_CASE_AXES = {
    "wall_fractions": [
        0.0,
        1.0e-8,
        1.0e-6,
        1.0e-4,
        1.0e-2,
        0.1,
        0.25,
        0.49,
        1.0,
    ],
    "orientations": ["axis", "oblique"],
    "affine_mesh_scales": [0.5, 1.0 / 3.0, 0.25],
    "active_sides": ["negative", "positive"],
    "case_count": 108,
    "wet_case_count": 96,
    "dry_case_count": 12,
    "configured_penalty_multiplier": 12.0,
    "sample_comparison_tolerance": 1.0e-11,
}
EXPECTED_TRACE_CONTRACT = {
    "supported_space": "affine_p1_product_velocity",
    "supported_cells": ["Triangle3", "Tetra4"],
    "frame": "reference",
    "viscosity": "constant_positive_finite",
    "aggregation_state": (
        "current_finalized_trace_eligible_closed_tangent_rows"
    ),
    "one_active_feature_per_patch": True,
    "maximum_terminal_tangent_dimension": 128,
    "maximum_exact_retained_quotient_dimension": 32,
    "quotient_authority": (
        "exact_binary64_dyadic_D_spd_N_psd_and_qD_minus_N_psd"
    ),
    "floating_spectral_role": "optional_diagnostics_only",
    "patch_inequality": (
        "integral_Gamma_(h_normal/mu)*abs(2*mu*epsilon(v)*n)^2 <= "
        "C_patch*integral_support_2*mu*epsilon(v):epsilon(v)"
    ),
    "global_bound_rule": (
        "outward_rounded_maximum_over_active_cells_of_summed_overlapping_"
        "patch_bounds"
    ),
    "effective_penalty_multiplier": (
        "alpha_i=gamma_i*p^2_when_scaled_else_gamma_i"
    ),
    "symmetric_group_ratio": "R_op=outward_rounded_sum_i(C_i/alpha_i)",
    "symmetric_acceptance": "R_op<1",
    "finite_space_energy_ratio_lower_bound": (
        "downward_rounded_(1-sqrt(R_op))"
    ),
    "unsymmetric_contract": (
        "revision_bound_continuity_diagnostic_without_symmetric_threshold"
    ),
    "certificate_selects_or_mutates_penalty": False,
    "navier_slip_robin_coefficient_rescaled": False,
    "cache_binding": (
        "cut_context_snapshot_source_value_affine_constraint_and_"
        "aggregation_revisions"
    ),
}

TRACE_TEST = (
    "FreeSurfaceCutStability."
    "DISABLED_SymmetricNitscheAggregateTraceCertificateMatrixV2"
)
TRACE_GROUP_ID = "symmetric_nitsche_certified_trace_108_case_diagnostic"
TRACE_CASE_PREFIX = "WP3_WP7_NITSCHE_TRACE_V2_CASE "
TRACE_SUMMARY_PREFIX = "WP3_WP7_NITSCHE_TRACE_V2_SUMMARY "
TRACE_EVIDENCE_ARTIFACT = "aggregate_trace_certificate_evidence.json"
TRACE_PENALTY_GAMMA = 12.0
EXPECTED_TRACE_CASE_COUNT = 108
EXPECTED_TRACE_WET_CASE_COUNT = 96
EXPECTED_TRACE_DRY_CASE_COUNT = 12
EXACT_DYADIC_RETAINED_QUOTIENT_DIMENSION_CAP = 32
EXACT_DYADIC_GROUP_ID = "exact_dyadic_spd_quotient_serial"
EXACT_DYADIC_TESTS = (
    "DenseLinearAlgebra."
    "ExactDyadicSpdGeneralizedBoundProvesDiagonalEquality",
    "DenseLinearAlgebra."
    "ExactDyadicSpdGeneralizedBoundRetainsTinyPositiveMode",
    "DenseLinearAlgebra."
    "ExactDyadicSpdGeneralizedBoundExercisesThreeByThreeBareissDivision",
    "DenseLinearAlgebra."
    "ExactDyadicSpdGeneralizedBoundExercisesSymmetricPsdPivotSwap",
    "DenseLinearAlgebra."
    "ExactDyadicSpdGeneralizedBoundRejectsLateIndefinitePivot",
    "DenseLinearAlgebra."
    "ExactDyadicSpdGeneralizedBoundRejectsSemidefiniteDenominator",
    "DenseLinearAlgebra."
    "ExactDyadicSpdGeneralizedBoundRejectsIndefiniteNumerator",
    "DenseLinearAlgebra."
    "ExactDyadicSpdGeneralizedBoundRejectsUnrepresentableUpperBound",
    "DenseLinearAlgebra."
    "ExactDyadicSpdGeneralizedBoundRejectsDimensionAboveCap",
    "DenseLinearAlgebra."
    "ExactDyadicSpdGeneralizedBoundRejectsMalformedInputs",
)
EXPECTED_EXACT_DYADIC_SOURCE_ROLES = {
    "Code/Source/solver/FE/Math/DenseLinearAlgebra.h": (
        "floating diagnostics and exact dyadic generalized-bound contract"
    ),
    "Code/Source/solver/FE/Math/DenseExactDyadic.cpp": (
        "authoritative exact binary64-dyadic SPD quotient certification"
    ),
    "Code/Source/solver/FE/Math/DenseLinearAlgebra.cpp": (
        "dense generalized eigenvalue certification implementation"
    ),
    "Code/Source/solver/FE/Tests/Unit/Math/test_DenseLinearAlgebra.cpp": (
        "floating and exact-dyadic generalized-bound evidence"
    ),
}
EXPECTED_BUILD_TARGETS = {
    "math": "test_fe_math",
    "assembly": "test_fe_assembly",
    "assembly_mpi": "test_fe_assembly_mpi",
    "physics": "test_physics",
}
EXPECTED_BUILD_CMAKE_HOMES = {
    "math": "Code/Source/solver/FE",
    "assembly": "Code/Source/solver/FE",
    "assembly_mpi": "Code/Source/solver/FE",
    "physics": "Code/Source/solver/Physics",
}

EXPECTED_GROUP_TESTS = {
    EXACT_DYADIC_GROUP_ID: (
        "math",
        1,
        1,
        EXACT_DYADIC_TESTS,
    ),
    "aggregate_trace_certificate_serial": (
        "assembly",
        1,
        1,
        (
            "GeneratedBoundaryAggregateTraceCertificate."
            "FormBindingRequiresExactlyOneRouteAnchorBeforeMutation",
            "GeneratedBoundaryAggregateTraceCertificate."
            "FullActiveUnitTriangleHasAnalyticBoundFour",
            "GeneratedBoundaryAggregateTraceCertificate."
            "RootedCutSquareCertifiesActualAggregateProlongation",
            "GeneratedBoundaryAggregateTraceCertificate."
            "RootlessAggregateSupportIsRejected",
            "GeneratedBoundaryAggregateTraceCertificate."
            "ImportedGeneratedDomainsWithoutAuthoritativeSnapshotFailClosed",
            "GeneratedBoundaryAggregateTraceCertificate."
            "ScalarFieldIsRejectedAsAnUnsupportedTraceSpace",
            "GeneratedBoundaryAggregateTraceCertificate."
            "SymmetricPolicyRejectsAnInsufficientConfiguredPenalty",
            "GeneratedBoundaryAggregateTraceCertificate."
            "UnsymmetricPolicyRetainsTheBoundWithoutACoercivityThreshold",
        ),
    ),
    "aggregate_trace_certificate_exact_two_rank_mpi": (
        "assembly_mpi",
        2,
        2,
        (
            "GeneratedBoundaryAggregateTraceCertificateMPI."
            "RootedCrossRankAggregateHasAnalyticBoundThirtyTwoOverSeventyNine",
        ),
    ),
    TRACE_GROUP_ID: (
        "physics",
        1,
        1,
        (TRACE_TEST,),
    ),
}
EXPECTED_GROUP_EXECUTION = {
    EXACT_DYADIC_GROUP_ID: (300, 1024, 64),
    "aggregate_trace_certificate_serial": (600, 1024, 64),
    "aggregate_trace_certificate_exact_two_rank_mpi": (600, 1024, 64),
    TRACE_GROUP_ID: (3600, 1024, 64),
}
EXPECTED_RESOURCE_SAFEGUARDS = {
    "execution_preflight_mem_available_mib": 10240,
    "runtime_mem_available_floor_mib": 4096,
    "runtime_filesystem_free_floor_mib": 4096,
    "build_parallel": 1,
    "build_process_session_memory_mib": 1024,
    "build_process_file_output_mib": 1024,
    "build_target_inventory_parse_limit_mib": 8,
    "binary_link_provenance_uses_monitored_process_sessions": True,
    "binary_link_provenance_timeout_seconds": 60,
    "binary_link_provenance_memory_mib": 256,
    "binary_link_provenance_output_mib": 4,
    "binary_link_provenance_fails_closed": True,
    "test_discovery_uses_monitored_process_sessions": True,
    "test_process_file_size_limit_matches_group_output_mib": True,
    "thread_environment": {
        "BLIS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "OMP_DYNAMIC": "FALSE",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
    },
}
EXPECTED_GATES = {
    "expected_group_count": 4,
    "expected_distinct_test_count": 20,
    "expected_quantitative_evidence_count": 7,
    "expected_failures": 0,
    "expected_errors": 0,
    "expected_disabled": 0,
    "expected_skipped": 0,
}
EXPECTED_QUANTITATIVE_EVIDENCE = {
    (TRACE_TEST, "wp3_wp7_nitsche_trace_v2_case_count"): (
        "integer",
        "equal",
        108,
    ),
    (TRACE_TEST, "wp3_wp7_nitsche_trace_v2_maximum_upper_bound"): (
        "real",
        "less_than",
        12.0,
    ),
    (
        TRACE_TEST,
        "wp3_wp7_nitsche_trace_v2_minimum_finite_sample_lower_bound",
    ): ("real", "greater_than", 0.0),
    (
        TRACE_TEST,
        "wp3_wp7_nitsche_trace_v2_minimum_sampled_eigenvalue_gap",
    ): ("real", "greater_than_or_equal", -1.0e-11),
    (
        TRACE_TEST,
        "wp3_wp7_nitsche_trace_v2_method_coercivity_lower_bound",
    ): ("string", "equal", "null"),
    (
        TRACE_TEST,
        "wp3_wp7_nitsche_trace_v2_uniform_bound_status",
    ): ("string", "equal", "UNFROZEN_NO_BOUND_INVENTED"),
    (
        TRACE_TEST,
        "wp3_wp7_nitsche_trace_v2_accepted_claim",
    ): ("string", "equal", "joint_low_level_prerequisite"),
}


def _load_shared_runner() -> Any:
    specification = importlib.util.spec_from_file_location(
        "_free_surface_wp3_wp7_trace_v2_base",
        SHARED_RUNNER_PATH,
    )
    if specification is None or specification.loader is None:
        raise RuntimeError("cannot load the shared qualification base")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


strict_runner = _load_shared_runner()
_shared_write_json = strict_runner.write_json
_shared_write_text = strict_runner.write_text
_shared_run_monitored = strict_runner.run_monitored
_shared_coerce_quantitative_value = strict_runner.coerce_quantitative_value

strict_runner.SCRIPT_PATH = SCRIPT_PATH
strict_runner.DEFAULT_REGISTRY = DEFAULT_REGISTRY
strict_runner.EXPECTED_MATRIX_ID = EXPECTED_MATRIX_ID
strict_runner.EXPECTED_MATRIX_STATUS = EXPECTED_MATRIX_STATUS
strict_runner.EXPECTED_WORK_PACKAGE = EXPECTED_WORK_PACKAGE
strict_runner.__doc__ = __doc__

_frozen_qualification_bundle_binding: dict[str, Any] | None = None
_RUNNER_SHA256_FIELD_PATTERN = re.compile(
    rb'("runner_sha256"[ \t\r\n]*:[ \t\r\n]*")'
    rb"([0-9a-f]{64})"
    rb'(")'
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalized_registry_bytes(raw_bytes: bytes) -> bytes:
    try:
        document = json.loads(
            raw_bytes.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(
            "qualification matrix is not valid UTF-8 JSON"
        ) from error
    if not isinstance(document, dict):
        raise ValueError("qualification matrix root must be an object")
    canonical_bytes = (
        json.dumps(document, indent=2, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    if raw_bytes != canonical_bytes:
        raise ValueError(
            "qualification matrix must use canonical two-space-indented "
            "JSON bytes before runner_sha256 normalization"
        )
    matches = list(_RUNNER_SHA256_FIELD_PATTERN.finditer(raw_bytes))
    if len(matches) != 1:
        raise ValueError(
            "qualification matrix must contain exactly one raw "
            "runner_sha256 field with a 64-character lowercase "
            "hexadecimal value"
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
    value = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_reject_duplicate_keys,
    )
    if not isinstance(value, dict):
        raise ValueError("qualification matrix root must be an object")
    return value


def _artifact_map(
    entries: Any,
    label: str,
    *,
    allow_role: bool = False,
) -> dict[str, str]:
    if not isinstance(entries, list):
        raise ValueError(f"{label} must be a list")
    result: dict[str, str] = {}
    for entry in entries:
        expected_keys = {"path", "sha256"}
        if allow_role:
            expected_keys.add("role")
        if not isinstance(entry, dict) or set(entry) != expected_keys:
            raise ValueError(f"{label} entry has unexpected keys")
        if allow_role and (
            not isinstance(entry["role"], str) or not entry["role"].strip()
        ):
            raise ValueError(f"{label} entry has an invalid role")
        path = entry["path"]
        digest = entry["sha256"]
        if (
            not isinstance(path, str)
            or not path
            or Path(path).is_absolute()
            or ".." in Path(path).parts
            or path in result
        ):
            raise ValueError(f"{label} entry has an unsafe or duplicate path")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(f"{label} entry has an invalid digest")
        result[path] = digest
    return result


def _property_contracts(
    entries: Any,
) -> dict[tuple[str, str], tuple[str, str, Any]]:
    if not isinstance(entries, list):
        raise ValueError("quantitative evidence must be a list")
    result: dict[tuple[str, str], tuple[str, str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {
            "test",
            "property",
            "type",
            "relation",
            "threshold",
        }:
            raise ValueError("quantitative evidence entry has unexpected keys")
        key = (entry["test"], entry["property"])
        if key in result:
            raise ValueError(f"duplicate quantitative evidence property: {key}")
        result[key] = (
            entry["type"],
            entry["relation"],
            entry["threshold"],
        )
    return result


def _validate_unique_field_set(
    fields: Any,
    expected: set[str],
    label: str,
) -> set[str]:
    if not isinstance(fields, list):
        raise ValueError(f"{label} must be a list")
    observed: set[str] = set()
    for field in fields:
        if not isinstance(field, str) or not field:
            raise ValueError(f"{label} contains an invalid field")
        if field in observed:
            raise ValueError(f"{label} contains duplicate field: {field}")
        observed.add(field)
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise ValueError(
            f"{label} changed: missing={missing}, extra={extra}"
        )
    return observed


def _validate_structured_output_contract(registry: dict[str, Any]) -> None:
    contract = registry.get("structured_output_contract")
    if not isinstance(contract, dict) or set(contract) != {
        "case_prefix",
        "summary_prefix",
        "expected_case_records",
        "case_required_fields",
        "case_identity_fields",
        "summary_required_fields",
    }:
        raise ValueError(
            "structured output contract must use the exact required keys"
        )
    if contract["case_prefix"] != TRACE_CASE_PREFIX:
        raise ValueError("structured output case prefix changed")
    if contract["summary_prefix"] != TRACE_SUMMARY_PREFIX:
        raise ValueError("structured output summary prefix changed")
    if contract["expected_case_records"] != EXPECTED_TRACE_CASE_COUNT:
        raise ValueError("structured output expected case count changed")
    case_fields = _validate_unique_field_set(
        contract["case_required_fields"],
        EXPECTED_TRACE_CASE_FIELDS,
        "structured output case_required_fields",
    )
    identity_fields = _validate_unique_field_set(
        contract["case_identity_fields"],
        EXPECTED_TRACE_IDENTITY_FIELDS,
        "structured output case_identity_fields",
    )
    if not identity_fields.issubset(case_fields):
        raise ValueError(
            "structured output identity fields must be required case fields"
        )
    _validate_unique_field_set(
        contract["summary_required_fields"],
        EXPECTED_TRACE_SUMMARY_FIELDS,
        "structured output summary_required_fields",
    )


def _validate_runtime_gates(registry: dict[str, Any]) -> None:
    gates = registry.get("runtime_gates")
    expected_tests = {
        test
        for _, _, _, tests in EXPECTED_GROUP_TESTS.values()
        for test in tests
    }
    if not isinstance(gates, list) or len(gates) != len(expected_tests):
        raise ValueError(
            "runtime_gates must contain exactly one entry for each "
            "qualification test"
        )
    observed_ids: set[str] = set()
    observed_tests: set[str] = set()
    for gate in gates:
        if not isinstance(gate, dict) or set(gate) != {
            "id",
            "test",
            "requirements",
        }:
            raise ValueError("runtime gate must use the exact required keys")
        gate_id = gate["id"]
        test = gate["test"]
        requirements = gate["requirements"]
        if not isinstance(gate_id, str) or not gate_id:
            raise ValueError("runtime gate has an invalid id")
        if gate_id in observed_ids:
            raise ValueError(f"duplicate runtime gate id: {gate_id}")
        observed_ids.add(gate_id)
        if not isinstance(test, str) or not test:
            raise ValueError(f"runtime gate {gate_id} has an invalid test")
        if test in observed_tests:
            raise ValueError(f"duplicate runtime gate test: {test}")
        observed_tests.add(test)
        if (
            not isinstance(requirements, list)
            or not requirements
            or any(
                not isinstance(requirement, str) or not requirement.strip()
                for requirement in requirements
            )
            or len(set(requirements)) != len(requirements)
        ):
            raise ValueError(
                f"runtime gate {gate_id} has invalid requirements"
            )
    if observed_tests != expected_tests:
        missing = sorted(expected_tests - observed_tests)
        extra = sorted(observed_tests - expected_tests)
        raise ValueError(
            f"runtime gate test map changed: missing={missing}, extra={extra}"
        )


def _validate_status_contract(registry: dict[str, Any]) -> None:
    status = registry.get("status")
    if status not in ALLOWED_MATRIX_STATUSES:
        raise ValueError(
            "certified-trace matrix status must be DRAFT_UNEXECUTED or "
            "FROZEN_BEFORE_EXECUTION"
        )
    promotion = registry.get("draft_promotion_contract")
    if not isinstance(promotion, dict):
        raise ValueError("draft promotion contract is missing")
    expected_promotion_values = {
        "current_state": status,
        "source_hashes_frozen": status == EXECUTABLE_MATRIX_STATUS,
        "qualification_evidence_executed": False,
        "validate_only_allowed": True,
        "execution_allowed": status == EXECUTABLE_MATRIX_STATUS,
        "required_execution_state": EXECUTABLE_MATRIX_STATUS,
    }
    for field, expected in expected_promotion_values.items():
        if promotion.get(field) != expected:
            raise ValueError(
                f"draft promotion contract is inconsistent with {status}: "
                f"{field}"
            )
    requirements = promotion.get("promotion_requirements")
    if (
        not isinstance(requirements, list)
        or not requirements
        or any(
            not isinstance(requirement, str) or not requirement.strip()
            for requirement in requirements
        )
    ):
        raise ValueError("draft promotion requirements are invalid")

    source_hash_status = registry.get("source_inventory_hash_status")
    implementation_commit = registry.get("implementation_source_commit")
    runner_digest = registry.get("runner_sha256")
    if (
        not isinstance(runner_digest, str)
        or len(runner_digest) != 64
        or any(
            character not in "0123456789abcdef"
            for character in runner_digest
        )
    ):
        raise ValueError(
            "runner_sha256 must be a 64-character lowercase "
            "hexadecimal digest"
        )
    if status == EXPECTED_MATRIX_STATUS:
        if implementation_commit is not None:
            raise ValueError(
                "draft implementation_source_commit must remain null"
            )
        if source_hash_status != EXPECTED_DRAFT_SOURCE_HASH_STATUS:
            raise ValueError(
                "draft source inventory must remain observed and unfrozen"
            )
        if runner_digest != RUNNER_SHA256_ZERO_SENTINEL:
            raise ValueError(
                "draft runner_sha256 must remain the zero sentinel"
            )
        return

    if (
        not isinstance(implementation_commit, str)
        or len(implementation_commit) not in {40, 64}
        or any(
            character not in "0123456789abcdef"
            for character in implementation_commit
        )
    ):
        raise ValueError(
            "frozen implementation_source_commit must be a 40- or "
            "64-character lowercase hexadecimal digest"
        )
    if source_hash_status != EXPECTED_FROZEN_SOURCE_HASH_STATUS:
        raise ValueError(
            "frozen source inventory must use intentionally frozen hashes"
        )
    if runner_digest == RUNNER_SHA256_ZERO_SENTINEL:
        raise ValueError(
            "frozen runner_sha256 must lock the exact runner bytes"
        )


def _validate_group_contracts(registry: dict[str, Any]) -> None:
    groups = registry.get("groups")
    if not isinstance(groups, list):
        raise ValueError("v2 qualification groups are missing")
    if [group.get("id") for group in groups] != list(EXPECTED_GROUP_TESTS):
        raise ValueError("v2 qualification group order changed")
    observed_tests: set[str] = set()
    for group in groups:
        group_id = group["id"]
        expected_binary, expected_ranks, expected_copies, expected_tests = (
            EXPECTED_GROUP_TESTS[group_id]
        )
        if (
            group.get("binary") != expected_binary
            or group.get("mpi_ranks") != expected_ranks
            or group.get("gtest_output_copies") != expected_copies
            or tuple(group.get("tests", [])) != expected_tests
        ):
            raise ValueError(f"v2 qualification group changed: {group_id}")
        execution = group.get("execution")
        expected_execution = EXPECTED_GROUP_EXECUTION[group_id]
        if (
            not isinstance(execution, dict)
            or set(execution)
            != {"wall_time_seconds", "memory_mib", "output_mib"}
            or any(
                not isinstance(execution[key], int)
                or isinstance(execution[key], bool)
                for key in execution
            )
            or (
                execution["wall_time_seconds"],
                execution["memory_mib"],
                execution["output_mib"],
            )
            != expected_execution
        ):
            raise ValueError(f"v2 group execution envelope changed: {group_id}")
        for test in expected_tests:
            if test in observed_tests:
                raise ValueError(f"duplicate v2 qualification test: {test}")
            observed_tests.add(test)
    by_id = {group["id"]: group for group in groups}
    if by_id[
        "aggregate_trace_certificate_exact_two_rank_mpi"
    ].get("exact_mpi_ranks") is not True:
        raise ValueError("aggregate trace MPI group must require exactly two ranks")
    if by_id[TRACE_GROUP_ID].get("gtest_also_run_disabled_tests") is not True:
        raise ValueError("trace diagnostic must explicitly run its disabled test")
    if len(observed_tests) != EXPECTED_GATES["expected_distinct_test_count"]:
        raise ValueError("v2 distinct test count changed")


def validate_v2_contract(registry: dict[str, Any]) -> dict[str, Any]:
    if registry.get("schema_version") != 2:
        raise ValueError("unsupported certified-trace qualification schema")
    if registry.get("matrix_id") != EXPECTED_MATRIX_ID:
        raise ValueError("unexpected certified-trace matrix id")
    _validate_status_contract(registry)
    if registry.get("work_package") != EXPECTED_WORK_PACKAGE:
        raise ValueError("certified-trace work package changed")
    if registry.get("qualification_scope") != EXPECTED_SCOPE:
        raise ValueError("certified-trace qualification scope changed")
    if registry.get("closure_request_policy") != (
        EXPECTED_CLOSURE_REQUEST_POLICY
    ):
        raise ValueError("certified-trace closure-request policy changed")
    if registry.get("qualification_disposition") != EXPECTED_DISPOSITION:
        raise ValueError("certified-trace qualification disposition changed")
    if registry.get("open_outcomes") != EXPECTED_OPEN_OUTCOMES:
        raise ValueError("certified-trace open outcomes changed")
    if registry.get("method_coercivity_lower_bound") is not None:
        raise ValueError("a method coercivity lower bound was invented")
    if registry.get("uniform_bound_status") != (
        "UNFROZEN_NO_BOUND_INVENTED"
    ):
        raise ValueError("uniform-bound status changed")
    if registry.get("case_axes") != EXPECTED_CASE_AXES:
        raise ValueError("certified-trace finite case axes changed")
    if registry.get("matching_derivation") != EXPECTED_MATCHING_DERIVATION:
        raise ValueError("certified-trace matching derivation changed")
    if registry.get("proposed_runner") != EXPECTED_PROPOSED_RUNNER:
        raise ValueError("certified-trace proposed runner changed")
    if registry.get("qualification_bundle_binding") != (
        EXPECTED_QUALIFICATION_BUNDLE_BINDING
    ):
        raise ValueError("qualification bundle binding contract changed")
    if registry.get("resource_safeguards") != (
        EXPECTED_RESOURCE_SAFEGUARDS
    ):
        raise ValueError("qualification resource safeguards changed")
    if (
        strict_runner.TARGET_INVENTORY_PARSE_LIMIT_BYTES
        != EXPECTED_RESOURCE_SAFEGUARDS[
            "build_target_inventory_parse_limit_mib"
        ]
        * 1024
        * 1024
        or strict_runner.BINARY_LINK_PROVENANCE_TIMEOUT_SECONDS
        != EXPECTED_RESOURCE_SAFEGUARDS[
            "binary_link_provenance_timeout_seconds"
        ]
        or strict_runner.BINARY_LINK_PROVENANCE_MEMORY_MIB
        != EXPECTED_RESOURCE_SAFEGUARDS[
            "binary_link_provenance_memory_mib"
        ]
        or strict_runner.BINARY_LINK_PROVENANCE_OUTPUT_MIB
        != EXPECTED_RESOURCE_SAFEGUARDS[
            "binary_link_provenance_output_mib"
        ]
    ):
        raise ValueError(
            "shared qualification post-processing resource limits changed"
        )
    if registry.get("certified_aggregate_trace_contract") != (
        EXPECTED_TRACE_CONTRACT
    ):
        raise ValueError("certified aggregate trace contract changed")
    _validate_structured_output_contract(registry)

    parents = _artifact_map(
        registry.get("parent_artifacts"), "parent artifacts"
    )
    for path, digest in EXPECTED_V1_PARENT_SHA256.items():
        if parents.get(path) != digest:
            raise ValueError(f"v1 parent artifact is not hash-locked: {path}")
    sources = _artifact_map(
        registry.get("implementation_sources"),
        "implementation sources",
        allow_role=True,
    )
    if not sources:
        raise ValueError("certified-trace implementation sources are empty")
    circular_bundle_sources = {
        EXPECTED_MATRIX_PATH,
        EXPECTED_PROPOSED_RUNNER,
    } & set(sources)
    if circular_bundle_sources:
        raise ValueError(
            "qualification bundle artifacts must use the reciprocal "
            "qualification-bundle "
            "binding, not the implementation-source inventory: "
            + ", ".join(sorted(circular_bundle_sources))
        )
    source_roles = {
        entry["path"]: entry["role"]
        for entry in registry["implementation_sources"]
    }
    for path, expected_role in EXPECTED_EXACT_DYADIC_SOURCE_ROLES.items():
        if source_roles.get(path) != expected_role:
            raise ValueError(
                "exact-dyadic implementation source inventory changed: "
                f"{path}"
            )
    if (
        registry["status"] == EXECUTABLE_MATRIX_STATUS
        and EXPECTED_MATCHING_DERIVATION not in parents
        and EXPECTED_MATCHING_DERIVATION not in sources
    ):
        raise ValueError("frozen v2 matching derivation is not hash-locked")

    build_targets = registry.get("build_targets")
    build_homes = registry.get("build_cmake_homes")
    if (
        build_targets != EXPECTED_BUILD_TARGETS
        or build_homes != EXPECTED_BUILD_CMAKE_HOMES
    ):
        raise ValueError("v2 build target/CMake-home inventory changed")
    certificate_envelope = registry.get("certificate_envelope")
    if (
        not isinstance(certificate_envelope, dict)
        or certificate_envelope.get(
            "hard_exact_retained_quotient_dimension_cap"
        )
        != EXACT_DYADIC_RETAINED_QUOTIENT_DIMENSION_CAP
    ):
        raise ValueError("exact-dyadic retained quotient cap changed")
    _validate_group_contracts(registry)
    _validate_runtime_gates(registry)
    if registry.get("gates") != EXPECTED_GATES:
        raise ValueError("v2 result/count gates changed")
    if _property_contracts(
        registry.get("quantitative_evidence")
    ) != EXPECTED_QUANTITATIVE_EVIDENCE:
        raise ValueError("v2 quantitative evidence contract changed")

    closure_contract = registry.get("closure_contract")
    if not isinstance(closure_contract, list) or not closure_contract:
        raise ValueError("v2 closure contract is missing")
    frozen_tests = {
        test
        for _, _, _, tests in EXPECTED_GROUP_TESTS.values()
        for test in tests
    }
    for claim in closure_contract:
        if (
            not isinstance(claim, dict)
            or not isinstance(claim.get("claim"), str)
            or not claim["claim"]
            or not isinstance(claim.get("evidence"), list)
            or not claim["evidence"]
            or any(test not in frozen_tests for test in claim["evidence"])
        ):
            raise ValueError("v2 closure-contract evidence is invalid")
    if TRACE_TEST not in {
        test
        for claim in closure_contract
        for test in claim["evidence"]
    }:
        raise ValueError("v2 closure contract omits the finite trace diagnostic")
    if registry.get("prospective_tests") != []:
        raise ValueError("v2 matrix cannot execute prospective tests")
    return registry


def validate_frozen_dependencies(
    registry: dict[str, Any],
    repository_root: Path = REPOSITORY_ROOT,
) -> None:
    inventories = [("parent artifact", registry["parent_artifacts"])]
    if registry["status"] == EXECUTABLE_MATRIX_STATUS:
        inventories.append(
            ("implementation source", registry["implementation_sources"])
        )
    for label, entries in inventories:
        for entry in entries:
            path = repository_root / entry["path"]
            if not path.is_file():
                raise ValueError(f"{label} is missing: {entry['path']}")
            if sha256_file(path) != entry["sha256"]:
                raise ValueError(f"{label} bytes changed: {entry['path']}")
    if registry["status"] == EXECUTABLE_MATRIX_STATUS:
        source_commit = registry["implementation_source_commit"]
        try:
            resolved_commit = (
                strict_runner.git_bytes(
                    repository_root,
                    "rev-parse",
                    "--verify",
                    f"{source_commit}^{{commit}}",
                )
                .decode()
                .strip()
            )
        except (
            OSError,
            subprocess.CalledProcessError,
            UnicodeDecodeError,
        ) as error:
            raise ValueError(
                "frozen implementation_source_commit is not a commit"
            ) from error
        if resolved_commit != source_commit:
            raise ValueError(
                "frozen implementation_source_commit did not resolve exactly"
            )
        try:
            strict_runner.git_bytes(
                repository_root,
                "merge-base",
                "--is-ancestor",
                source_commit,
                "HEAD",
            )
        except (OSError, subprocess.CalledProcessError) as error:
            raise ValueError(
                "frozen implementation_source_commit is not an ancestor "
                "of HEAD"
            ) from error
        for entry in registry["implementation_sources"]:
            try:
                committed_bytes = strict_runner.git_bytes(
                    repository_root,
                    "show",
                    f"{source_commit}:{entry['path']}",
                )
            except (OSError, subprocess.CalledProcessError) as error:
                raise ValueError(
                    "implementation source is absent from its frozen "
                    f"commit: {entry['path']}"
                ) from error
            if hashlib.sha256(committed_bytes).hexdigest() != entry["sha256"]:
                raise ValueError(
                    "implementation source differs from its frozen commit: "
                    f"{entry['path']}"
                )
    if sha256_file(SHARED_RUNNER_PATH) != EXPECTED_SHARED_RUNNER_SHA256:
        raise RuntimeError("shared qualification base changed during execution")


def validate_frozen_qualification_bundle(
    registry: dict[str, Any],
    matrix_path: Path = DEFAULT_REGISTRY,
    repository_root: Path = REPOSITORY_ROOT,
    runner_path: Path = SCRIPT_PATH,
) -> dict[str, Any]:
    if registry.get("status") != EXECUTABLE_MATRIX_STATUS:
        raise ValueError(
            "qualification bundle binding requires a frozen matrix"
        )
    expected_paths = {
        "matrix": EXPECTED_MATRIX_PATH,
        "runner": EXPECTED_PROPOSED_RUNNER,
    }
    actual_paths = {
        "matrix": matrix_path,
        "runner": runner_path,
    }
    for role, relative_path in expected_paths.items():
        expected_path = (repository_root / relative_path).resolve()
        if actual_paths[role].resolve() != expected_path:
            raise ValueError(
                f"qualification bundle {role} path is not canonical"
            )
        if not actual_paths[role].is_file():
            raise ValueError(
                f"qualification bundle {role} is missing: {relative_path}"
            )

    try:
        bundle_commit = (
            strict_runner.git_bytes(
                repository_root,
                "rev-parse",
                "--verify",
                "HEAD^{commit}",
            )
            .decode()
            .strip()
        )
    except (
        OSError,
        subprocess.CalledProcessError,
        UnicodeDecodeError,
    ) as error:
        raise ValueError(
            "qualification bundle HEAD is not a commit"
        ) from error
    if (
        len(bundle_commit) not in {40, 64}
        or any(
            character not in "0123456789abcdef"
            for character in bundle_commit
        )
    ):
        raise ValueError(
            "qualification bundle HEAD has an invalid commit digest"
        )

    artifacts: list[dict[str, Any]] = []
    matrix_runner_digest: str | None = None
    for role in ("matrix", "runner"):
        relative_path = expected_paths[role]
        try:
            working_bytes = actual_paths[role].read_bytes()
            committed_bytes = strict_runner.git_bytes(
                repository_root,
                "show",
                f"{bundle_commit}:{relative_path}",
            )
        except (OSError, subprocess.CalledProcessError) as error:
            raise ValueError(
                f"qualification bundle {role} is absent from HEAD"
            ) from error
        if working_bytes != committed_bytes:
            raise ValueError(
                f"qualification bundle {role} differs from its HEAD blob"
            )
        digest = hashlib.sha256(working_bytes).hexdigest()
        artifact: dict[str, Any] = {
            "role": role,
            "path": relative_path,
            "sha256": digest,
        }
        if role == "matrix":
            normalized_digest = hashlib.sha256(
                normalized_registry_bytes(working_bytes)
            ).hexdigest()
            if normalized_digest != EXPECTED_NORMALIZED_REGISTRY_SHA256:
                raise ValueError(
                    "qualification bundle matrix does not match the "
                    "runner's embedded normalized SHA-256"
                )
            matrix_document = json.loads(
                working_bytes.decode("utf-8"),
                object_pairs_hook=_reject_duplicate_keys,
            )
            matrix_runner_digest = matrix_document.get("runner_sha256")
            if matrix_runner_digest != registry.get("runner_sha256"):
                raise ValueError(
                    "qualification bundle matrix runner_sha256 does not "
                    "match the frozen registry"
                )
            artifact["normalized_sha256"] = normalized_digest
        elif digest != matrix_runner_digest:
            raise ValueError(
                "qualification bundle runner does not match the matrix "
                "runner_sha256"
            )
        artifacts.append(artifact)

    return {
        "binding_schema_version": 1,
        "authority": (
            "reciprocal_SHA256_plus_clean_qualification_execution_HEAD"
        ),
        "qualification_bundle_commit": bundle_commit,
        "implementation_source_commit": registry[
            "implementation_source_commit"
        ],
        "normalized_matrix_sha256_embedded_in_runner": (
            EXPECTED_NORMALIZED_REGISTRY_SHA256
        ),
        "runner_sha256_from_matrix": registry["runner_sha256"],
        "artifacts": artifacts,
    }


def _normalized_registry_digest_is_frozen() -> bool:
    return (
        isinstance(EXPECTED_NORMALIZED_REGISTRY_SHA256, str)
        and len(EXPECTED_NORMALIZED_REGISTRY_SHA256) == 64
        and all(
            character in "0123456789abcdef"
            for character in EXPECTED_NORMALIZED_REGISTRY_SHA256
        )
    )


def load_registry(path: Path) -> dict[str, Any]:
    global _frozen_qualification_bundle_binding
    _frozen_qualification_bundle_binding = None
    if not _normalized_registry_digest_is_frozen():
        raise RuntimeError(
            "certified-trace normalized matrix SHA-256 is not frozen"
        )
    if path.resolve() != DEFAULT_REGISTRY.resolve():
        raise ValueError("qualification requires the canonical v2 matrix")
    if normalized_registry_sha256(path) != (
        EXPECTED_NORMALIZED_REGISTRY_SHA256
    ):
        raise ValueError("certified-trace normalized matrix bytes changed")
    registry = validate_v2_contract(parse_json_document(path))
    validate_frozen_dependencies(registry)
    if registry["status"] == EXECUTABLE_MATRIX_STATUS:
        _frozen_qualification_bundle_binding = (
            validate_frozen_qualification_bundle(registry, path)
        )
    return registry


def _finite_real(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a non-null real")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _integer(value: Any, label: str, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    if value < (1 if positive else 0):
        qualifier = "positive" if positive else "nonnegative"
        raise ValueError(f"{label} must be {qualifier}")
    return value


def _close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=2.0e-13, abs_tol=2.0e-14)


def _json_records(stdout: str, prefix: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line in stdout.splitlines():
        if not line.startswith(prefix):
            continue
        value = json.loads(
            line[len(prefix) :],
            object_pairs_hook=_reject_duplicate_keys,
        )
        if not isinstance(value, dict):
            raise ValueError(f"{prefix.strip()} record must be an object")
        records.append(value)
    return records


def _expected_trace_case_keys() -> set[tuple[str, str, float, float]]:
    return {
        (orientation, side, scale, fraction)
        for orientation in EXPECTED_CASE_AXES["orientations"]
        for scale in EXPECTED_CASE_AXES["affine_mesh_scales"]
        for side in EXPECTED_CASE_AXES["active_sides"]
        for fraction in EXPECTED_CASE_AXES["wall_fractions"]
    }


def parse_trace_evidence(stdout: str) -> dict[str, Any]:
    cases = _json_records(stdout, TRACE_CASE_PREFIX)
    summaries = _json_records(stdout, TRACE_SUMMARY_PREFIX)
    if len(cases) != EXPECTED_TRACE_CASE_COUNT:
        raise ValueError(
            "trace evidence must contain exactly "
            f"{EXPECTED_TRACE_CASE_COUNT} case records"
        )
    if len(summaries) != 1:
        raise ValueError("trace evidence must contain exactly one summary record")

    case_ids: set[str] = set()
    case_keys: set[tuple[str, str, float, float]] = set()
    normalized_cases: list[dict[str, Any]] = []
    wet_count = 0
    dry_count = 0
    maximum_upper_bound = 0.0
    minimum_lower_bound = math.inf
    minimum_eigenvalue_gap = math.inf
    for ordinal, case in enumerate(cases):
        label = f"trace case {ordinal}"
        if set(case) != EXPECTED_TRACE_CASE_FIELDS:
            missing = sorted(EXPECTED_TRACE_CASE_FIELDS - set(case))
            extra = sorted(set(case) - EXPECTED_TRACE_CASE_FIELDS)
            raise ValueError(
                f"{label} fields changed: missing={missing}, extra={extra}"
            )
        case_id = case.get("case_id")
        orientation = case.get("orientation")
        active_side = case.get("active_side")
        if not isinstance(case_id, str) or not case_id:
            raise ValueError(f"{label} has an invalid case_id")
        if case_id in case_ids:
            raise ValueError(f"duplicate trace case_id: {case_id}")
        case_ids.add(case_id)
        if orientation not in EXPECTED_CASE_AXES["orientations"]:
            raise ValueError(f"{label} has an invalid orientation")
        if active_side not in EXPECTED_CASE_AXES["active_sides"]:
            raise ValueError(f"{label} has an invalid active side")
        mesh_scale = _finite_real(case.get("mesh_scale"), f"{label} mesh scale")
        fraction = _finite_real(
            case.get("target_wall_fraction"),
            f"{label} wall fraction",
        )
        key = (orientation, active_side, mesh_scale, fraction)
        if key in case_keys:
            raise ValueError(f"duplicate trace case axes: {key}")
        case_keys.add(key)

        _integer(
            case.get("certificate_digest"),
            f"{label} certificate digest",
            positive=True,
        )
        _integer(
            case.get("aggregation_digest"),
            f"{label} aggregation digest",
            positive=True,
        )
        _integer(
            case.get("cut_context_revision"),
            f"{label} cut-context revision",
            positive=True,
        )
        _integer(
            case.get("snapshot_revision"),
            f"{label} snapshot revision",
            positive=True,
        )
        _integer(
            case.get("source_value_revision"),
            f"{label} source-value revision",
            positive=True,
        )
        _integer(
            case.get("form_binding_digest"),
            f"{label} form-binding digest",
            positive=True,
        )
        _integer(
            case.get("source_formulation_record_index"),
            f"{label} source formulation record index",
        )
        if case.get("form_binding_source_match") is not True:
            raise ValueError(
                f"{label} form binding does not match its source formulation"
            )
        boundary_rules = _integer(
            case.get("boundary_rule_count"),
            f"{label} boundary-rule count",
        )
        patch_count = _integer(
            case.get("patch_count"),
            f"{label} patch count",
        )
        if case.get("deterministic") is not True:
            raise ValueError(f"{label} certificate is not deterministic")
        if case.get("revision_match") is not True:
            raise ValueError(f"{label} certificate revisions do not match")

        upper_bound = _finite_real(
            case.get("trace_upper_bound"),
            f"{label} trace upper bound",
        )
        penalty = _finite_real(
            case.get("effective_penalty_multiplier"),
            f"{label} effective penalty",
        )
        trace_ratio = _finite_real(
            case.get("trace_to_penalty_ratio"),
            f"{label} trace-to-penalty ratio",
        )
        grouped_ratio = _finite_real(
            case.get("grouped_symmetric_ratio"),
            f"{label} grouped symmetric ratio",
        )
        lower_bound = _finite_real(
            case.get("finite_sample_energy_lower_bound"),
            f"{label} finite-sample lower bound",
        )
        if upper_bound < 0.0 or upper_bound >= TRACE_PENALTY_GAMMA:
            raise ValueError(f"{label} trace upper bound is outside [0, 12)")
        if not _close(penalty, TRACE_PENALTY_GAMMA):
            raise ValueError(f"{label} effective penalty is not 12")
        if not _close(trace_ratio, upper_bound / penalty):
            raise ValueError(f"{label} trace-to-penalty ratio is inconsistent")
        if not _close(grouped_ratio, trace_ratio):
            raise ValueError(f"{label} symmetric group ratio is inconsistent")
        if grouped_ratio < 0.0 or grouped_ratio >= 1.0:
            raise ValueError(f"{label} symmetric group ratio is outside [0, 1)")
        mathematical_lower = 1.0 - math.sqrt(grouped_ratio)
        if lower_bound <= 0.0 or not _close(lower_bound, mathematical_lower):
            raise ValueError(f"{label} derived lower bound is invalid")

        dry = fraction == 0.0
        if dry:
            dry_count += 1
            if (
                boundary_rules != 0
                or patch_count != 0
                or upper_bound != 0.0
                or lower_bound != 1.0
            ):
                raise ValueError(f"{label} violates the exact dry certificate")
            for field in (
                "minimum_generalized_eigenvalue",
                "eigensolver_tolerance",
                "sampled_eigenvalue_gap",
            ):
                if case.get(field) is not None:
                    raise ValueError(f"{label} dry field {field} must be null")
        else:
            wet_count += 1
            if boundary_rules <= 0 or patch_count <= 0 or upper_bound <= 0.0:
                raise ValueError(f"{label} wet certificate is empty")
            minimum_eigenvalue = _finite_real(
                case.get("minimum_generalized_eigenvalue"),
                f"{label} minimum generalized eigenvalue",
            )
            eigensolver_tolerance = _finite_real(
                case.get("eigensolver_tolerance"),
                f"{label} eigensolver tolerance",
            )
            if eigensolver_tolerance < 0.0:
                raise ValueError(
                    f"{label} eigensolver tolerance must be nonnegative"
                )
            gap = _finite_real(
                case.get("sampled_eigenvalue_gap"),
                f"{label} sampled eigenvalue gap",
            )
            expected_gap = (
                minimum_eigenvalue
                - eigensolver_tolerance
                - lower_bound
            )
            if not _close(gap, expected_gap):
                raise ValueError(
                    f"{label} sampled eigenvalue gap is inconsistent"
                )
            if gap < -EXPECTED_CASE_AXES["sample_comparison_tolerance"]:
                raise ValueError(
                    f"{label} conservative sampled eigenvalue gap is below "
                    "the comparison tolerance"
                )
            minimum_eigenvalue_gap = min(minimum_eigenvalue_gap, gap)

        maximum_upper_bound = max(maximum_upper_bound, upper_bound)
        minimum_lower_bound = min(minimum_lower_bound, lower_bound)
        normalized_cases.append(copy.deepcopy(case))

    if case_keys != _expected_trace_case_keys():
        missing = sorted(_expected_trace_case_keys() - case_keys)
        extra = sorted(case_keys - _expected_trace_case_keys())
        raise ValueError(
            f"trace case axes are incomplete: missing={missing}, extra={extra}"
        )
    if len(case_ids) != EXPECTED_TRACE_CASE_COUNT:
        raise ValueError("trace evidence does not contain 108 unique case IDs")
    if (
        wet_count != EXPECTED_TRACE_WET_CASE_COUNT
        or dry_count != EXPECTED_TRACE_DRY_CASE_COUNT
    ):
        raise ValueError("trace wet/dry case counts changed")
    if not math.isfinite(minimum_eigenvalue_gap):
        raise ValueError("trace evidence has no wet eigenvalue gaps")

    summary = summaries[0]
    if set(summary) != EXPECTED_TRACE_SUMMARY_FIELDS:
        missing = sorted(EXPECTED_TRACE_SUMMARY_FIELDS - set(summary))
        extra = sorted(set(summary) - EXPECTED_TRACE_SUMMARY_FIELDS)
        raise ValueError(
            f"trace summary fields changed: missing={missing}, extra={extra}"
        )
    expected_summary_values = {
        "case_count": EXPECTED_TRACE_CASE_COUNT,
        "wet_case_count": EXPECTED_TRACE_WET_CASE_COUNT,
        "dry_case_count": EXPECTED_TRACE_DRY_CASE_COUNT,
        "deterministic_case_count": EXPECTED_TRACE_CASE_COUNT,
        "revision_match_case_count": EXPECTED_TRACE_CASE_COUNT,
    }
    for field, expected in expected_summary_values.items():
        if summary.get(field) != expected:
            raise ValueError(f"trace summary field changed: {field}")
    for field, expected in (
        ("maximum_trace_upper_bound", maximum_upper_bound),
        (
            "minimum_finite_sample_energy_lower_bound",
            minimum_lower_bound,
        ),
        ("minimum_sampled_eigenvalue_gap", minimum_eigenvalue_gap),
    ):
        observed = _finite_real(summary.get(field), f"trace summary {field}")
        if not _close(observed, expected):
            raise ValueError(f"trace summary aggregate changed: {field}")
    if summary.get("method_coercivity_lower_bound", object()) is not None:
        raise ValueError("trace summary invented a method coercivity bound")
    if summary.get("uniform_bound_status") != "UNFROZEN_NO_BOUND_INVENTED":
        raise ValueError("trace summary promoted the uniform-bound status")
    if summary.get("accepted_claim") != "joint_low_level_prerequisite":
        raise ValueError("trace summary promoted the accepted claim")

    normalized_cases.sort(
        key=lambda case: (
            EXPECTED_CASE_AXES["orientations"].index(case["orientation"]),
            EXPECTED_CASE_AXES["affine_mesh_scales"].index(
                float(case["mesh_scale"])
            ),
            EXPECTED_CASE_AXES["active_sides"].index(case["active_side"]),
            EXPECTED_CASE_AXES["wall_fractions"].index(
                float(case["target_wall_fraction"])
            ),
        )
    )
    return {
        "artifact_schema_version": 1,
        "matrix_id": EXPECTED_MATRIX_ID,
        "requested_claim": "joint_low_level_prerequisite",
        "method_coercivity_lower_bound": None,
        "uniform_bound_status": "UNFROZEN_NO_BOUND_INVENTED",
        "penalty_gamma": TRACE_PENALTY_GAMMA,
        "expected_case_count": EXPECTED_TRACE_CASE_COUNT,
        "observed_case_count": len(normalized_cases),
        "wet_case_count": wet_count,
        "dry_case_count": dry_count,
        "deterministic_case_count": len(normalized_cases),
        "revision_match_case_count": len(normalized_cases),
        "maximum_trace_upper_bound": maximum_upper_bound,
        "minimum_finite_sample_energy_lower_bound": minimum_lower_bound,
        "minimum_sampled_eigenvalue_gap": minimum_eigenvalue_gap,
        "cases": normalized_cases,
        "summary_record": copy.deepcopy(summary),
        "diagnostics": [],
        "outcome": "PASS",
    }


def evaluate_trace_evidence(stdout: str) -> dict[str, Any]:
    try:
        return parse_trace_evidence(stdout)
    except (
        json.JSONDecodeError,
        KeyError,
        OverflowError,
        TypeError,
        ValueError,
    ) as error:
        observed_case_count = sum(
            line.startswith(TRACE_CASE_PREFIX)
            for line in stdout.splitlines()
        )
        return {
            "artifact_schema_version": 1,
            "matrix_id": EXPECTED_MATRIX_ID,
            "requested_claim": "joint_low_level_prerequisite",
            "method_coercivity_lower_bound": None,
            "uniform_bound_status": "UNFROZEN_NO_BOUND_INVENTED",
            "penalty_gamma": TRACE_PENALTY_GAMMA,
            "expected_case_count": EXPECTED_TRACE_CASE_COUNT,
            "observed_case_count": observed_case_count,
            "diagnostics": [str(error)],
            "outcome": "FAIL_METHOD",
        }


def _load_trace_evidence(output_root: Path) -> dict[str, Any] | None:
    path = output_root / TRACE_EVIDENCE_ARTIFACT
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    return value if isinstance(value, dict) else None


def _trace_evidence_reference(output_root: Path) -> dict[str, Any]:
    path = output_root / TRACE_EVIDENCE_ARTIFACT
    evidence = _load_trace_evidence(output_root)
    if evidence is None:
        return {
            "path": TRACE_EVIDENCE_ARTIFACT,
            "present": False,
            "outcome": "NOT_PRODUCED",
        }
    return {
        "path": TRACE_EVIDENCE_ARTIFACT,
        "present": True,
        "sha256": sha256_file(path),
        "outcome": evidence.get("outcome"),
        "observed_case_count": evidence.get("observed_case_count"),
        "wet_case_count": evidence.get("wet_case_count"),
        "dry_case_count": evidence.get("dry_case_count"),
        "maximum_trace_upper_bound": evidence.get(
            "maximum_trace_upper_bound"
        ),
        "minimum_finite_sample_energy_lower_bound": evidence.get(
            "minimum_finite_sample_energy_lower_bound"
        ),
        "minimum_sampled_eigenvalue_gap": evidence.get(
            "minimum_sampled_eigenvalue_gap"
        ),
    }


EXPECTED_TRACE_CASE_FIELDS = {
    "case_id",
    "orientation",
    "active_side",
    "mesh_scale",
    "target_wall_fraction",
    "certificate_digest",
    "aggregation_digest",
    "cut_context_revision",
    "snapshot_revision",
    "source_value_revision",
    "form_binding_digest",
    "source_formulation_record_index",
    "form_binding_source_match",
    "boundary_rule_count",
    "patch_count",
    "trace_upper_bound",
    "effective_penalty_multiplier",
    "trace_to_penalty_ratio",
    "grouped_symmetric_ratio",
    "finite_sample_energy_lower_bound",
    "minimum_generalized_eigenvalue",
    "eigensolver_tolerance",
    "sampled_eigenvalue_gap",
    "deterministic",
    "revision_match",
}
EXPECTED_TRACE_IDENTITY_FIELDS = {
    "orientation",
    "active_side",
    "mesh_scale",
    "target_wall_fraction",
}
EXPECTED_TRACE_SUMMARY_FIELDS = {
    "case_count",
    "wet_case_count",
    "dry_case_count",
    "deterministic_case_count",
    "revision_match_case_count",
    "maximum_trace_upper_bound",
    "minimum_finite_sample_energy_lower_bound",
    "minimum_sampled_eigenvalue_gap",
    "method_coercivity_lower_bound",
    "uniform_bound_status",
    "accepted_claim",
}


def coerce_quantitative_value(
    raw_value: Any,
    value_type: str,
) -> tuple[Any, str | None]:
    if value_type == "string":
        if isinstance(raw_value, str):
            return raw_value, None
        return None, "property_type_mismatch"
    return _shared_coerce_quantitative_value(raw_value, value_type)


def run_monitored(
    command: list[str],
    environment: dict[str, str],
    working_directory: Path,
    stdout_path: Path,
    stderr_path: Path,
    output_directory: Path,
    wall_time_seconds: int,
    memory_mib: int,
    output_mib: int,
    launch_mode: str,
    required_simultaneous_process_samples: int = 1,
) -> dict[str, Any]:
    trace_filter = f"--gtest_filter={TRACE_TEST}"
    if trace_filter in command:
        if launch_mode != "direct_serial":
            raise ValueError("the trace evidence diagnostic must run serially")
        if "--gtest_also_run_disabled_tests" not in command:
            command.insert(
                command.index(trace_filter) + 1,
                "--gtest_also_run_disabled_tests",
            )
    pinned_environment = dict(environment)
    pinned_environment.update(
        EXPECTED_RESOURCE_SAFEGUARDS["thread_environment"]
    )
    return _shared_run_monitored(
        command,
        pinned_environment,
        working_directory,
        stdout_path,
        stderr_path,
        output_directory,
        wall_time_seconds,
        memory_mib,
        output_mib,
        launch_mode,
        required_simultaneous_process_samples,
        minimum_host_available_mib=(
            EXPECTED_RESOURCE_SAFEGUARDS[
                "runtime_mem_available_floor_mib"
            ]
        ),
        minimum_filesystem_free_mib=(
            EXPECTED_RESOURCE_SAFEGUARDS[
                "runtime_filesystem_free_floor_mib"
            ]
        ),
        filesystem_path=output_directory,
    )


def run_build_phase(
    command: list[str],
    source_root: Path,
    output_root: Path,
    stdout_path: Path,
    stderr_path: Path,
    timeout_seconds: int,
) -> dict[str, Any]:
    environment = os.environ.copy()
    environment.update(
        EXPECTED_RESOURCE_SAFEGUARDS["thread_environment"]
    )
    if "--build" in command:
        build_directory = Path(
            command[command.index("--build") + 1]
        ).resolve()
    elif "-B" in command:
        build_directory = Path(
            command[command.index("-B") + 1]
        ).resolve()
    else:
        build_directory = source_root
    resources = _shared_run_monitored(
        command,
        environment,
        source_root,
        stdout_path,
        stderr_path,
        output_root,
        timeout_seconds,
        EXPECTED_RESOURCE_SAFEGUARDS[
            "build_process_session_memory_mib"
        ],
        EXPECTED_RESOURCE_SAFEGUARDS[
            "build_process_file_output_mib"
        ],
        "direct_serial",
        1,
        minimum_host_available_mib=(
            EXPECTED_RESOURCE_SAFEGUARDS[
                "runtime_mem_available_floor_mib"
            ]
        ),
        minimum_filesystem_free_mib=(
            EXPECTED_RESOURCE_SAFEGUARDS[
                "runtime_filesystem_free_floor_mib"
            ]
        ),
        filesystem_path=build_directory,
        additional_filesystem_paths=(output_root,),
    )
    monitoring_passed = (
        resources["termination_reason"] is None
        and resources["resource_monitoring_outcome"] == "PASS"
    )
    process_return_code = resources["return_code"]
    return {
        "command": command,
        "return_code": (
            process_return_code if monitoring_passed else None
        ),
        "process_return_code": process_return_code,
        "timed_out": (
            resources["termination_reason"] ==
            "wall_time_envelope_exceeded"
        ),
        "termination": resources["termination"],
        "termination_reason": resources["termination_reason"],
        "elapsed_seconds": resources["wall_time_seconds"],
        "resource_monitoring": resources,
        "monitored_build_directory": str(build_directory),
        "stdout": str(stdout_path.relative_to(output_root)),
        "stderr": str(stderr_path.relative_to(output_root)),
        "stdout_sha256": sha256_file(stdout_path),
        "stderr_sha256": sha256_file(stderr_path),
    }


def monitored_test_discovery(
    source_root: Path,
    output_root: Path,
):
    def discover(binary: Path) -> set[str]:
        identity = hashlib.sha256(
            str(binary.resolve()).encode("utf-8")
        ).hexdigest()[:16]
        discovery_root = output_root / "test_discovery"
        discovery_root.mkdir(parents=True, exist_ok=True)
        discovery_directory = (
            discovery_root / f"{binary.name}_{identity}"
        )
        discovery_directory.mkdir(exist_ok=False)
        stdout_path = discovery_directory / "stdout.txt"
        stderr_path = discovery_directory / "stderr.txt"
        command = [str(binary), "--gtest_list_tests"]
        environment = os.environ.copy()
        environment.update(
            EXPECTED_RESOURCE_SAFEGUARDS["thread_environment"]
        )
        resources = _shared_run_monitored(
            command,
            environment,
            source_root,
            stdout_path,
            stderr_path,
            discovery_directory,
            strict_runner.GTEST_LIST_TIMEOUT_SECONDS,
            1024,
            64,
            "direct_serial",
            1,
            minimum_host_available_mib=(
                EXPECTED_RESOURCE_SAFEGUARDS[
                    "runtime_mem_available_floor_mib"
                ]
            ),
            minimum_filesystem_free_mib=(
                EXPECTED_RESOURCE_SAFEGUARDS[
                    "runtime_filesystem_free_floor_mib"
                ]
            ),
            filesystem_path=binary.parent,
            additional_filesystem_paths=(discovery_directory,),
        )
        passed = (
            resources["return_code"] == 0
            and resources["termination_reason"] is None
            and resources["resource_monitoring_outcome"] == "PASS"
        )
        _shared_write_json(
            discovery_directory / "result.json",
            {
                "binary": str(binary),
                "command": command,
                "resources": resources,
                "outcome": "PASS" if passed else "FAIL_METHOD",
            },
        )
        if not passed:
            raise subprocess.CalledProcessError(
                resources["return_code"]
                if resources["return_code"] is not None
                else 125,
                command,
            )
        suite = ""
        names: set[str] = set()
        for line in stdout_path.read_text(
            encoding="utf-8",
            errors="replace",
        ).splitlines():
            if line and not line[0].isspace():
                suite = (
                    line.split("#", 1)[0]
                    .strip()
                    .removesuffix(".")
                )
                continue
            test = line.split("#", 1)[0].strip()
            if suite and test:
                names.add(f"{suite}.{test}")
        return names

    return discover


def _current_frozen_qualification_bundle_binding() -> dict[str, Any] | None:
    if _frozen_qualification_bundle_binding is None:
        return None
    current = validate_frozen_qualification_bundle(
        {
            "status": EXECUTABLE_MATRIX_STATUS,
            "implementation_source_commit": (
                _frozen_qualification_bundle_binding[
                    "implementation_source_commit"
                ]
            ),
            "runner_sha256": _frozen_qualification_bundle_binding[
                "runner_sha256_from_matrix"
            ],
        }
    )
    if current != _frozen_qualification_bundle_binding:
        raise RuntimeError(
            "qualification bundle changed during execution"
        )
    return copy.deepcopy(current)


def _inject_claim_boundary(value: dict[str, Any]) -> None:
    value["qualification_scope"] = EXPECTED_SCOPE
    value["requested_claim"] = "joint_low_level_prerequisite"
    value["qualification_disposition"] = copy.deepcopy(EXPECTED_DISPOSITION)
    value["open_outcomes"] = copy.deepcopy(EXPECTED_OPEN_OUTCOMES)
    value["method_coercivity_lower_bound"] = None
    value["uniform_bound_status"] = "UNFROZEN_NO_BOUND_INVENTED"
    value["certified_aggregate_trace_contract"] = copy.deepcopy(
        EXPECTED_TRACE_CONTRACT
    )
    value["v1_parent_artifacts"] = [
        {"path": path, "sha256": digest}
        for path, digest in EXPECTED_V1_PARENT_SHA256.items()
    ]
    if _frozen_qualification_bundle_binding is not None:
        value["qualification_bundle_binding"] = copy.deepcopy(
            _frozen_qualification_bundle_binding
        )


def write_json(path: Path, value: Any) -> None:
    if sha256_file(SHARED_RUNNER_PATH) != EXPECTED_SHARED_RUNNER_SHA256:
        raise RuntimeError("shared qualification base changed during execution")
    if (
        isinstance(value, dict)
        and path.name == "final_provenance.json"
        and _frozen_qualification_bundle_binding is not None
    ):
        value["qualification_bundle_binding"] = (
            _current_frozen_qualification_bundle_binding()
        )
    if (
        isinstance(value, dict)
        and path.name == "result.json"
        and path.parent.name == TRACE_GROUP_ID
    ):
        stdout_path = path.parent / "stdout.txt"
        stdout = (
            stdout_path.read_text(encoding="utf-8", errors="replace")
            if stdout_path.is_file()
            else ""
        )
        evidence = evaluate_trace_evidence(stdout)
        output_root = path.parents[2]
        evidence_path = output_root / TRACE_EVIDENCE_ARTIFACT
        _shared_write_json(evidence_path, evidence)
        passed = evidence["outcome"] == "PASS"
        value.setdefault("checks", []).append(
            {
                "metric": "certified_aggregate_trace_evidence",
                "actual": evidence["outcome"],
                "expected": "PASS",
                "relation": "equal",
                "passed": passed,
            }
        )
        value["certified_aggregate_trace_evidence"] = (
            _trace_evidence_reference(output_root)
        )
        if not passed:
            value["outcome"] = "FAIL_METHOD"
            value["diagnostic"] = "certified_aggregate_trace_evidence_failed"

    if isinstance(value, dict) and path.name in {
        "build_preflight.json",
        "manifest.json",
        "final_provenance.json",
        "summary.json",
    }:
        _inject_claim_boundary(value)
        reference = _trace_evidence_reference(path.parent)
        value["certified_aggregate_trace_evidence"] = reference
        if path.name == "final_provenance.json" and reference["present"]:
            passed = reference["outcome"] == "PASS"
            value.setdefault("checks", []).append(
                {
                    "metric": "certified_aggregate_trace_evidence_outcome",
                    "actual": reference["outcome"],
                    "expected": "PASS",
                    "relation": "equal",
                    "passed": passed,
                }
            )
            if not passed:
                value["outcome"] = "FAIL_METHOD"
                value.setdefault("diagnostics", []).append(
                    "certified_aggregate_trace_evidence_outcome"
                )
        if (
            path.name == "summary.json"
            and reference["present"]
            and reference["outcome"] != "PASS"
        ):
            value["overall_outcome"] = "FAIL_METHOD"
            value["failure_stage"] = "certified_aggregate_trace_evidence"
    _shared_write_json(path, value)


def write_text(path: Path, value: str) -> None:
    if path.name == "record.md":
        evidence = _load_trace_evidence(path.parent)
        value += (
            "\n## Certified finite aggregate-trace prerequisite\n\n"
            + EXPECTED_SCOPE
            + "\n\n"
            "The method-wide coercivity lower bound remains `null`; "
            "FSR-16, FSR-07, WP-3, WP-7, and Q1 remain open.\n"
        )
        if evidence is not None:
            value += (
                "\n- Aggregate-trace evidence: "
                f"**{evidence.get('outcome', 'INVALID')}**\n"
                f"- Certified cases: {evidence.get('observed_case_count')}\n"
                "- Maximum finite-fixture trace upper bound: "
                f"{evidence.get('maximum_trace_upper_bound')}\n"
                "- Minimum finite-fixture energy lower bound: "
                f"{evidence.get('minimum_finite_sample_energy_lower_bound')}\n"
                "- Minimum conservative sampled eigenvalue gap: "
                f"{evidence.get('minimum_sampled_eigenvalue_gap')}\n"
            )
    _shared_write_text(path, value)


strict_runner.load_registry = load_registry
strict_runner.write_json = write_json
strict_runner.write_text = write_text
strict_runner.run_monitored = run_monitored
strict_runner.run_build_phase = run_build_phase
strict_runner.coerce_quantitative_value = coerce_quantitative_value


def requested_claim(arguments: list[str]) -> tuple[str, bool, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--requested-claim",
        default=EXPECTED_CLOSURE_REQUEST_POLICY["accepted_claim"],
    )
    parser.add_argument("--validate-only", action="store_true")
    parsed, remaining = parser.parse_known_args(arguments)
    claim = parsed.requested_claim
    accepted = EXPECTED_CLOSURE_REQUEST_POLICY["accepted_claim"]
    rejected = set(EXPECTED_CLOSURE_REQUEST_POLICY["rejected_claims"])
    if claim in rejected:
        raise ValueError(
            f"requested claim {claim!r} is outside this matrix: "
            f"{EXPECTED_CLOSURE_REQUEST_POLICY['diagnostic']}"
        )
    if claim != accepted:
        raise ValueError(
            f"unsupported v2 requested claim {claim!r}; expected {accepted!r}"
        )
    return claim, parsed.validate_only, remaining


def observe_implementation_sources(
    registry: dict[str, Any],
    repository_root: Path = REPOSITORY_ROOT,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for entry in registry["implementation_sources"]:
        path = repository_root / entry["path"]
        observed_sha256 = sha256_file(path) if path.is_file() else None
        records.append(
            {
                "path": entry["path"],
                "expected_sha256": entry["sha256"],
                "observed_sha256": observed_sha256,
                "matches_draft_observation": (
                    observed_sha256 == entry["sha256"]
                ),
            }
        )
    matching_count = sum(
        record["matches_draft_observation"] for record in records
    )
    missing_count = sum(
        record["observed_sha256"] is None for record in records
    )
    return {
        "inventory_count": len(records),
        "matching_count": matching_count,
        "drift_count": len(records) - matching_count,
        "missing_count": missing_count,
        "all_match": matching_count == len(records),
        "records": records,
    }


def validate_only_summary(
    registry: dict[str, Any],
    claim: str,
) -> dict[str, Any]:
    source_observation = observe_implementation_sources(registry)
    draft = registry["status"] == EXPECTED_MATRIX_STATUS
    return {
        "matrix_id": registry["matrix_id"],
        "status": registry["status"],
        "execution_ready": not draft,
        "validation_scope": (
            "draft_structure_and_dependency_validation"
            if draft
            else "frozen_execution_preflight"
        ),
        "implementation_source_observation": source_observation,
        "requested_claim": claim,
        "group_count": len(registry["groups"]),
        "test_count": sum(
            len(group["tests"]) for group in registry["groups"]
        ),
        "quantitative_evidence_gate_count": len(
            registry["quantitative_evidence"]
        ),
        "expected_trace_case_count": EXPECTED_TRACE_CASE_COUNT,
        "method_coercivity_lower_bound": None,
        "uniform_bound_status": "UNFROZEN_NO_BOUND_INVENTED",
        "qualification_disposition": copy.deepcopy(EXPECTED_DISPOSITION),
        "open_outcomes": copy.deepcopy(EXPECTED_OPEN_OUTCOMES),
        "closure_outcome": "OPEN_JOINT_LOW_LEVEL_PREREQUISITE",
        "outcome": (
            "PASS_DRAFT_STRUCTURE_ONLY"
            if draft and source_observation["all_match"]
            else (
                "DRAFT_SOURCE_DRIFT"
                if draft
                else "PASS_FROZEN_VALIDATION"
            )
        ),
    }


def _existing_filesystem_probe(path: Path) -> Path:
    probe = path.resolve()
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    if not probe.exists():
        raise ValueError(
            f"cannot locate a filesystem ancestor for {path}"
        )
    return probe


def require_execution_resource_preflight(
    source_root: Path,
    output_directory: Path,
    build_directories: tuple[Path, ...] = (),
) -> None:
    available_mib = strict_runner.host_available_memory_mib()
    required_available_mib = EXPECTED_RESOURCE_SAFEGUARDS[
        "execution_preflight_mem_available_mib"
    ]
    if (
        available_mib is None
        or available_mib < required_available_mib
    ):
        raise ValueError(
            "qualification execution requires at least "
            f"{required_available_mib} MiB MemAvailable; observed "
            f"{available_mib!r}"
        )
    required_free_mib = EXPECTED_RESOURCE_SAFEGUARDS[
        "runtime_filesystem_free_floor_mib"
    ]
    for label, path in (
        ("source/build", source_root),
        ("qualification output", output_directory.parent),
        *(
            (f"build directory {index}", path)
            for index, path in enumerate(
                build_directories,
                start=1,
            )
        ),
    ):
        probe = _existing_filesystem_probe(path)
        free_mib = strict_runner.filesystem_free_mib(probe)
        if free_mib is None or free_mib < required_free_mib:
            raise ValueError(
                f"{label} filesystem requires at least "
                f"{required_free_mib} MiB free; observed "
                f"{free_mib!r} MiB at {probe}"
            )


def execution_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--math-binary", type=Path, required=True)
    parser.add_argument("--assembly-binary", type=Path, required=True)
    parser.add_argument(
        "--assembly-mpi-binary",
        type=Path,
        required=True,
    )
    parser.add_argument("--physics-binary", type=Path, required=True)
    parser.add_argument(
        "--mpiexec",
        type=Path,
        default=Path("/usr/bin/mpiexec"),
    )
    parser.add_argument(
        "--cmake",
        type=Path,
        default=Path("/usr/bin/cmake"),
    )
    parser.add_argument(
        "--build-parallel",
        type=int,
        default=EXPECTED_RESOURCE_SAFEGUARDS["build_parallel"],
    )
    parser.add_argument(
        "--build-timeout-seconds",
        type=int,
        default=3600,
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=REPOSITORY_ROOT,
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(arguments: list[str] | None = None) -> int:
    selected_arguments = (
        list(sys.argv[1:]) if arguments is None else list(arguments)
    )
    claim, validate_only, remaining = requested_claim(selected_arguments)
    registry = load_registry(DEFAULT_REGISTRY)
    if validate_only:
        if remaining:
            raise ValueError(
                "--validate-only does not accept execution arguments"
            )
        print(
            json.dumps(
                validate_only_summary(registry, claim),
                sort_keys=True,
            )
        )
        return 0
    if registry["status"] != EXECUTABLE_MATRIX_STATUS:
        raise ValueError(
            "certified-trace matrix is DRAFT_UNEXECUTED; full execution "
            "requires promotion and a newly frozen matrix/runner hash"
        )
    parser = execution_argument_parser()
    execution_arguments = parser.parse_args(remaining)
    if execution_arguments.registry.resolve() != DEFAULT_REGISTRY.resolve():
        parser.error("qualification execution requires the canonical matrix")
    if (
        execution_arguments.build_parallel !=
        EXPECTED_RESOURCE_SAFEGUARDS["build_parallel"]
    ):
        parser.error("qualification clean builds require parallelism one")
    if (
        execution_arguments.build_timeout_seconds <= 0
        or execution_arguments.build_timeout_seconds > 3600
    ):
        parser.error(
            "qualification build timeout must be in [1, 3600] seconds"
        )
    source_root = execution_arguments.source_root.resolve()
    output_directory = execution_arguments.output.resolve()
    binaries = {
        "math": execution_arguments.math_binary,
        "assembly": execution_arguments.assembly_binary,
        "assembly_mpi": execution_arguments.assembly_mpi_binary,
        "physics": execution_arguments.physics_binary,
    }
    build_directories = tuple(
        sorted(
            {
                cache.resolve().parent
                for binary in binaries.values()
                if (
                    cache :=
                    strict_runner.find_cmake_cache(
                        binary.resolve()
                    )
                ) is not None
            },
            key=str,
        )
    )
    require_execution_resource_preflight(
        source_root,
        output_directory,
        build_directories,
    )
    original_shared_status = strict_runner.EXPECTED_MATRIX_STATUS
    try:
        strict_runner.EXPECTED_MATRIX_STATUS = registry["status"]
        return strict_runner.run_qualification(
            execution_arguments,
            binaries,
            expected_binary_keys=set(binaries),
            parser=parser,
            record_title=(
                "WP-3/WP-7 certified aggregate-trace "
                "prerequisite qualification record"
            ),
            test_discovery=monitored_test_discovery(
                source_root,
                output_directory,
            ),
        )
    finally:
        strict_runner.EXPECTED_MATRIX_STATUS = original_shared_status


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)

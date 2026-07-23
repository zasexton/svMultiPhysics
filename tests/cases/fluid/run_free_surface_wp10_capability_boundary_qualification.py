#!/usr/bin/env python3
"""Verify the frozen WP-10 one-phase capability boundary.

The only accepted claim is ``one_phase_capability_boundary``. Requests for
FSR-08, WP-10, Q7, two-fluid, or gas-sensitive closure are rejected before
any test binary is executed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


SCRIPT_PATH = Path(__file__).resolve()
REPOSITORY_ROOT = SCRIPT_PATH.parents[3]
DEFAULT_MATRIX = SCRIPT_PATH.with_name(
    "free_surface_wp10_capability_boundary_matrix.json"
)
EXPECTED_MATRIX_ID = "free_surface_wp10_capability_boundary_v1"
EXPECTED_STATUS = "FROZEN_CAPABILITY_BOUNDARY"
EXPECTED_ARCHITECTURE_RECORD = (
    "Documentation/free_surface_wp10_physical_capability_boundary.md"
)
EXPECTED_SCOPE = (
    "Explicit one-phase capability labeling and unsupported-scope containment "
    "only; this matrix does not close FSR-08, WP-10, or Q7 and does not "
    "qualify any two-fluid or gas-sensitive phenomenon."
)
EXPECTED_CURRENT_BOUNDARY = {
    "momentum_capability_label": "one_phase_liquid_sharp_interface",
    "transport_capability_labels": [
        "one_phase_interface_transport_nonlocal_conservation",
        "one_phase_locally_conservative_p1_indicator_transport",
    ],
    "liquid_velocity_field_count": 1,
    "liquid_pressure_field_count": 1,
    "material_density_state_count": 1,
    "material_viscosity_state_count": 1,
    "exterior_state": "prescribed_scalar_pressure_traction",
    "exterior_momentum_solved": False,
    "exterior_pressure_field_solved": False,
    "incompressible_two_fluid_implemented": False,
    "gas_dynamics_implemented": False,
    "wp10_closure_claimed": False,
    "q7_closure_claimed": False,
}
EXPECTED_SOURCE_CHECKS = {
    "single_liquid_option_state": {
        "path": (
            "Code/Source/solver/Physics/Formulations/NavierStokes/"
            "IncompressibleNavierStokesVMSModule.h"
        ),
        "required_fragments": [
            'std::string velocity_field_name{"u"};',
            'std::string pressure_field_name{"p"};',
            "FE::Real density{1.0};",
            "FE::Real viscosity{0.01};",
            "ScalarValue external_pressure{0.0};",
        ],
        "forbidden_fragments": [
            "gas_density",
            "gas_viscosity",
            "outside_density",
            "outside_viscosity",
            "two_fluid_enabled",
            "pressure_space_enrichment",
        ],
    },
    "one_phase_momentum_artifact": {
        "path": (
            "Code/Source/solver/Physics/Formulations/NavierStokes/"
            "IncompressibleNavierStokesVMSModule.cpp"
        ),
        "required_fragments": [
            '"one_phase_liquid_sharp_interface"',
            "const auto p_ext = bc::toScalarExpr(",
            "bc.external_pressure",
        ],
        "forbidden_fragments": [
            '"incompressible_two_fluid"',
            '"compressible_gas_free_surface"',
        ],
    },
    "one_phase_transport_artifacts": {
        "path": (
            "Code/Source/solver/Application/Translators/LevelSetEquationTranslator.cpp"
        ),
        "required_fragments": [
            '"one_phase_interface_transport_nonlocal_conservation"',
            '"one_phase_locally_conservative_p1_indicator_transport"',
        ],
        "forbidden_fragments": [
            '"two_fluid_momentum_transport"',
            '"compressible_gas_transport"',
        ],
    },
    "unsupported_scope_containment": {
        "path": "tests/cases/fluid/free_surface_one_phase_scope_guard.py",
        "required_fragments": [
            '"twophase"',
            '"pressureenrichment"',
            '"gasdensity"',
            '"gasviscosity"',
            '"unsupported_two_phase_or_jump_free_surface_scope"',
            "def validate_xml_config(",
            "def validate_json_config(",
            "def validate_config_mapping(",
            "def validate_mapping_wrapper_pairs(",
            "def validate_xml_scope_subtree(",
        ],
        "forbidden_fragments": [],
    },
}
EXPECTED_SCOPE_GUARD_CONTRACT_SHA256 = (
    "6caf114d8751e67191e36f4fbb7905aadb0414d63b65c624326a29f96c4d8b4f"
)
EXPECTED_GROUPS = {
    "momentum_capability_artifact_serial": {
        "binary_argument": "physics_binary",
        "tests": [
            (
                "MovingDomainPhysics."
                "NavierStokesEffectiveConfigurationSnapshotExpandsBoundaryDefaults"
            )
        ],
    },
    "transport_capability_artifacts_serial": {
        "binary_argument": "application_binary",
        "tests": [
            "LevelSetEquationTranslator.TranslatesFieldsAndBoundaries",
            "LevelSetEquationTranslator.TranslatesConservativePhaseControls",
        ],
    },
}
EXPECTED_UNIMPLEMENTED = {
    "phasewise_density_and_viscosity",
    "both_phase_fields_or_stable_one_field_jump_formulation",
    "interface_velocity_and_stress_conditions",
    "pressure_jump_space_treatment",
    "both_phase_stabilization",
    "phasewise_mass_conservation",
    "phase_flux_momentum_flux_consistency",
    "high_density_ratio_robust_solver",
    "gas_dynamics_and_thermodynamic_closure",
}
EXPECTED_WP10_EXITS = {
    "planar_pressure_jump",
    "planar_viscous_jump",
    "two_fluid_hydrostatics",
    "static_drop",
    "material_side_reversal",
    "both_phase_mass",
    "high_density_ratio_conditioning",
    "phase_and_momentum_flux_consistency",
}
EXPECTED_Q7_EXITS = {
    "hysing_case_1": "BLOCKED_BY_WP10",
    "two_fluid_capillary_waves": "BLOCKED_BY_WP10",
    "rising_bubble": "BLOCKED_BY_WP10",
    "hysing_case_2_intercode_range": "BLOCKED_BY_WP10",
    "air_cushioning": "BLOCKED_BY_GAS_MODEL",
    "trapped_gas": "BLOCKED_BY_GAS_MODEL",
    "ambient_pressure_sweep": "BLOCKED_BY_GAS_MODEL",
    "dry_wall_splash": "BLOCKED_BY_GAS_MODEL",
}
EXPECTED_EXCLUSIONS = {
    "two_fluid_hysing_flow",
    "gas_cushioning",
    "gas_inertia_or_viscosity",
    "gas_compressibility",
    "trapped_gas_pressure",
    "entrainment",
    "aerodynamic_sheet_breakup",
    "ambient_pressure_dependent_dry_splash",
    "late_atomization",
}
EXPECTED_POLICY = {
    "accepted_claim": "one_phase_capability_boundary",
    "rejected_claims": [
        "fsr08_closure",
        "wp10_closure",
        "q7_closure",
        "incompressible_two_fluid_qualification",
        "gas_sensitive_qualification",
    ],
    "diagnostic": (
        "This matrix verifies only the explicit one-phase model boundary. "
        "The two-fluid and gas formulations and every physical WP-10/Q7 exit "
        "are absent."
    ),
}


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_text_exclusive(path: Path, value: str) -> None:
    with path.open("x", encoding="utf-8") as output:
        output.write(value)
        output.flush()
        os.fsync(output.fileno())


def write_json_exclusive(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as output:
        json.dump(value, output, indent=2, sort_keys=True)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())


def validate_status_entries(
    value: Any,
    expected_ids: set[str],
    expected_status: str,
    label: str,
) -> None:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list")
    actual: set[str] = set()
    for entry in value:
        if (
            not isinstance(entry, dict)
            or set(entry) != {"id", "status"}
            or not isinstance(entry.get("id"), str)
            or entry.get("status") != expected_status
        ):
            raise ValueError(f"invalid {label} entry")
        actual.add(entry["id"])
    if len(actual) != len(value) or actual != expected_ids:
        raise ValueError(f"{label} changed after freeze")


def validate_matrix(path: Path) -> dict[str, Any]:
    if path.resolve() != DEFAULT_MATRIX.resolve():
        raise ValueError("WP-10 requires the canonical frozen matrix")
    matrix = json.loads(path.read_text(encoding="utf-8"))
    if matrix.get("schema_version") != 1:
        raise ValueError("unsupported WP-10 matrix schema")
    if matrix.get("matrix_id") != EXPECTED_MATRIX_ID:
        raise ValueError("unexpected WP-10 matrix id")
    if matrix.get("status") != EXPECTED_STATUS:
        raise ValueError("WP-10 capability matrix is not frozen")
    if matrix.get("work_package") != "WP-10":
        raise ValueError("unexpected work package")
    if matrix.get("finding") != "FSR-08":
        raise ValueError("unexpected finding")
    if matrix.get("qualification_campaign") != "Q7":
        raise ValueError("unexpected qualification campaign")
    if matrix.get("architecture_record") != EXPECTED_ARCHITECTURE_RECORD:
        raise ValueError("architecture-record path changed after freeze")
    if (
        matrix.get("model_envelope")
        != "one_phase_incompressible_liquid_with_prescribed_exterior_pressure"
    ):
        raise ValueError("one-phase model envelope changed after freeze")
    if matrix.get("current_capability_boundary") != EXPECTED_CURRENT_BOUNDARY:
        raise ValueError("current capability boundary changed after freeze")

    source_checks = matrix.get("source_checks")
    if not isinstance(source_checks, list):
        raise ValueError("source checks are missing")
    actual_source_checks: dict[str, dict[str, Any]] = {}
    for check in source_checks:
        if not isinstance(check, dict) or set(check) != {
            "id",
            "path",
            "required_fragments",
            "forbidden_fragments",
        }:
            raise ValueError("invalid source-check entry")
        identifier = check["id"]
        if not isinstance(identifier, str) or identifier in actual_source_checks:
            raise ValueError("duplicate or invalid source-check id")
        actual_source_checks[identifier] = {
            key: check[key]
            for key in ("path", "required_fragments", "forbidden_fragments")
        }
    if actual_source_checks != EXPECTED_SOURCE_CHECKS:
        raise ValueError("source checks changed after freeze")
    scope_guard_contract = matrix.get("scope_guard_contract")
    if (
        not isinstance(scope_guard_contract, dict)
        or set(scope_guard_contract)
        != {
            "path",
            "diagnostic",
            "accepted_cases",
            "rejected_cases",
            "invalid_cases",
        }
        or sha256_bytes(
            json.dumps(
                scope_guard_contract,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        != EXPECTED_SCOPE_GUARD_CONTRACT_SHA256
    ):
        raise ValueError("scope guard contract changed after freeze")

    groups = matrix.get("groups")
    if not isinstance(groups, list):
        raise ValueError("test groups are missing")
    actual_groups: dict[str, dict[str, Any]] = {}
    all_tests: set[str] = set()
    for group in groups:
        if not isinstance(group, dict) or set(group) != {
            "id",
            "binary_argument",
            "tests",
            "execution",
        }:
            raise ValueError("invalid test-group entry")
        identifier = group["id"]
        tests = group["tests"]
        execution = group["execution"]
        if (
            not isinstance(identifier, str)
            or identifier in actual_groups
            or not isinstance(tests, list)
            or not tests
            or not isinstance(execution, dict)
            or set(execution) != {"wall_time_seconds", "memory_mib", "output_mib"}
            or any(
                not isinstance(execution[key], int) or execution[key] <= 0
                for key in execution
            )
        ):
            raise ValueError("invalid test-group contract")
        for test in tests:
            if (
                not isinstance(test, str)
                or not re.fullmatch(r"[A-Za-z0-9_]+\.[A-Za-z0-9_]+", test)
                or test in all_tests
            ):
                raise ValueError("invalid or duplicate test name")
            all_tests.add(test)
        actual_groups[identifier] = {
            "binary_argument": group["binary_argument"],
            "tests": tests,
        }
    if actual_groups != EXPECTED_GROUPS:
        raise ValueError("test groups changed after freeze")

    validate_status_entries(
        matrix.get("unimplemented_wp10_requirements"),
        EXPECTED_UNIMPLEMENTED,
        "REQUIRED_NOT_IMPLEMENTED",
        "unimplemented WP-10 requirements",
    )
    validate_status_entries(
        matrix.get("blocked_wp10_qualification_exits"),
        EXPECTED_WP10_EXITS,
        "BLOCKED_BY_MISSING_IMPLEMENTATION",
        "blocked WP-10 exits",
    )

    q7_entries = matrix.get("blocked_q7_progression")
    if not isinstance(q7_entries, list):
        raise ValueError("blocked Q7 progression must be a list")
    actual_q7: dict[str, str] = {}
    for entry in q7_entries:
        if (
            not isinstance(entry, dict)
            or set(entry) != {"id", "status"}
            or not isinstance(entry.get("id"), str)
            or not isinstance(entry.get("status"), str)
            or entry["id"] in actual_q7
        ):
            raise ValueError("invalid blocked Q7 entry")
        actual_q7[entry["id"]] = entry["status"]
    if actual_q7 != EXPECTED_Q7_EXITS:
        raise ValueError("blocked Q7 progression changed after freeze")
    exclusions = matrix.get("excluded_current_claims")
    if (
        not isinstance(exclusions, list)
        or len(set(exclusions)) != len(exclusions)
        or set(exclusions) != EXPECTED_EXCLUSIONS
    ):
        raise ValueError("current claim exclusions changed after freeze")
    if matrix.get("closure_request_policy") != EXPECTED_POLICY:
        raise ValueError("closure request policy changed after freeze")
    if matrix.get("qualification_scope") != EXPECTED_SCOPE:
        raise ValueError("qualification scope changed after freeze")
    return matrix


def validate_requested_claim(matrix: dict[str, Any], claim: str) -> None:
    policy = matrix["closure_request_policy"]
    if claim in policy["rejected_claims"]:
        raise ValueError(
            f"requested claim {claim!r} is outside this matrix: {policy['diagnostic']}"
        )
    if claim != policy["accepted_claim"]:
        raise ValueError(
            f"unsupported requested claim {claim!r}; "
            f"expected {policy['accepted_claim']!r}"
        )


def validate_source_boundary(
    matrix: dict[str, Any], source_root: Path
) -> list[dict[str, Any]]:
    source_root = source_root.resolve()
    if not (source_root / ".git").exists():
        git_probe = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=source_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if git_probe.returncode != 0:
            raise ValueError("source root is not a Git work tree")
    records: list[dict[str, Any]] = []
    for contract in matrix["source_checks"]:
        relative = Path(contract["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("source-check path must be repository relative")
        path = (source_root / relative).resolve()
        if not path.is_relative_to(source_root) or not path.is_file():
            raise ValueError(f"source-check path is missing: {relative}")
        content = path.read_bytes()
        text = content.decode("utf-8")
        missing = [
            fragment
            for fragment in contract["required_fragments"]
            if fragment not in text
        ]
        present_forbidden = [
            fragment for fragment in contract["forbidden_fragments"] if fragment in text
        ]
        if missing:
            raise ValueError(
                f"source boundary check {contract['id']} is missing "
                f"required fragments: {missing}"
            )
        if present_forbidden:
            raise ValueError(
                f"source boundary check {contract['id']} found unsupported "
                f"production fragments: {present_forbidden}"
            )
        records.append(
            {
                "id": contract["id"],
                "path": relative.as_posix(),
                "sha256": sha256_bytes(content),
                "required_fragment_count": len(contract["required_fragments"]),
                "forbidden_fragment_count": len(contract["forbidden_fragments"]),
                "outcome": "PASS",
            }
        )
    return records


def validate_scope_guard_contract(
    matrix: dict[str, Any], source_root: Path
) -> dict[str, Any]:
    contract = matrix["scope_guard_contract"]
    source_root = source_root.resolve()
    relative = Path(contract["path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("scope-guard path must be repository relative")
    lexical_path = source_root
    for component in relative.parts:
        lexical_path = lexical_path / component
        if lexical_path.is_symlink():
            raise ValueError("scope-guard path contains a symbolic-link component")
    path = (source_root / relative).resolve()
    if not path.is_relative_to(source_root) or not path.is_file():
        raise ValueError(f"scope-guard path is missing: {relative}")

    guard_bytes = path.read_bytes()
    guard_sha256 = sha256_bytes(guard_bytes)
    namespace: dict[str, Any] = {
        "__file__": str(path),
        "__name__": "free_surface_one_phase_scope_guard_contract",
    }
    exec(compile(guard_bytes, str(path), "exec"), namespace)
    diagnostic = namespace.get("UNSUPPORTED_SCOPE_DIAGNOSTIC")
    validate_payload = namespace.get("validate_payload")
    unsupported_error = namespace.get("UnsupportedFreeSurfaceScope")
    if (
        diagnostic != contract["diagnostic"]
        or not callable(validate_payload)
        or not isinstance(unsupported_error, type)
        or not issubclass(unsupported_error, ValueError)
    ):
        raise ValueError("scope guard exports changed after freeze")

    accepted_ids: list[str] = []
    for case in contract["accepted_cases"]:
        try:
            validate_payload(case["format"], case["payload"])
        except Exception as error:
            raise ValueError(
                f"scope guard rejected accepted case: {case['id']}"
            ) from error
        accepted_ids.append(case["id"])

    rejected_ids: list[str] = []
    for case in contract["rejected_cases"]:
        try:
            validate_payload(case["format"], case["payload"])
        except unsupported_error as error:
            if str(error) != diagnostic:
                raise ValueError(
                    f"scope guard diagnostic changed for case: {case['id']}"
                ) from error
        except Exception as error:
            raise ValueError(
                f"scope guard used the wrong rejection for case: {case['id']}"
            ) from error
        else:
            raise ValueError(f"scope guard accepted unsupported case: {case['id']}")
        rejected_ids.append(case["id"])

    invalid_ids: list[str] = []
    for case in contract["invalid_cases"]:
        try:
            validate_payload(case["format"], case["payload"])
        except unsupported_error as error:
            raise ValueError(
                f"scope guard misclassified invalid case: {case['id']}"
            ) from error
        except ValueError as error:
            if str(error) != case["diagnostic"]:
                raise ValueError(
                    f"scope guard structural diagnostic changed for case: {case['id']}"
                ) from error
        except Exception as error:
            raise ValueError(
                f"scope guard used the wrong invalidity for case: {case['id']}"
            ) from error
        else:
            raise ValueError(
                f"scope guard accepted structurally invalid case: {case['id']}"
            )
        invalid_ids.append(case["id"])

    return {
        "id": "one_phase_scope_guard_contract",
        "path": relative.as_posix(),
        "sha256": guard_sha256,
        "diagnostic": diagnostic,
        "accepted_case_ids": accepted_ids,
        "rejected_case_ids": rejected_ids,
        "invalid_case_ids": invalid_ids,
        "accepted_case_count": len(accepted_ids),
        "rejected_case_count": len(rejected_ids),
        "invalid_case_count": len(invalid_ids),
        "outcome": "PASS",
    }


def bind_scope_guard_source_record(
    source_records: list[dict[str, Any]],
    scope_guard_record: dict[str, Any],
) -> None:
    matching = [
        record
        for record in source_records
        if record["id"] == "unsupported_scope_containment"
    ]
    if (
        len(matching) != 1
        or matching[0]["path"] != scope_guard_record["path"]
        or matching[0]["sha256"] != scope_guard_record["sha256"]
    ):
        raise ValueError("scope guard changed between source and contract validation")


def listed_gtests(binary: Path) -> set[str]:
    result = subprocess.run(
        [str(binary), "--gtest_list_tests"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=60,
    )
    suite = ""
    names: set[str] = set()
    for line in result.stdout.splitlines():
        if line and not line[0].isspace():
            suite = line.split("#", 1)[0].strip().removesuffix(".")
            continue
        test = line.split("#", 1)[0].strip()
        if suite and test:
            names.add(f"{suite}.{test}")
    return names


def flatten_gtest(document: dict[str, Any]) -> dict[str, dict[str, Any]]:
    flattened: dict[str, dict[str, Any]] = {}
    for suite in document.get("testsuites", []):
        suite_name = suite.get("name")
        if not isinstance(suite_name, str):
            continue
        for test in suite.get("testsuite", []):
            test_name = test.get("name")
            if not isinstance(test_name, str):
                continue
            flattened[f"{suite_name}.{test_name}"] = test
    return flattened


def run_group(
    group: dict[str, Any],
    binary: Path,
    output_root: Path,
) -> dict[str, Any]:
    binary = binary.resolve()
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise ValueError(f"test binary is not executable: {binary}")
    requested = set(group["tests"])
    missing = sorted(requested - listed_gtests(binary))
    if missing:
        raise ValueError(f"test binary is missing frozen tests: {missing}")

    group_root = output_root / group["id"]
    group_root.mkdir()
    result_path = group_root / "gtest.json"
    command = [
        str(binary),
        "--gtest_filter=" + ":".join(group["tests"]),
        "--gtest_output=json:" + str(result_path),
    ]
    started = time.monotonic()
    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=group["execution"]["wall_time_seconds"],
    )
    elapsed = time.monotonic() - started
    write_text_exclusive(group_root / "stdout.txt", result.stdout)
    write_text_exclusive(group_root / "stderr.txt", result.stderr)
    if result.returncode != 0:
        raise RuntimeError(f"group {group['id']} returned {result.returncode}")
    if not result_path.is_file():
        raise RuntimeError(f"group {group['id']} did not write gtest JSON")
    document = json.loads(result_path.read_text(encoding="utf-8"))
    flattened = flatten_gtest(document)
    if set(flattened) != requested:
        raise RuntimeError(
            f"group {group['id']} result inventory differs from the matrix"
        )
    for name, test in flattened.items():
        if (
            test.get("status") != "RUN"
            or test.get("result") != "COMPLETED"
            or test.get("failures")
        ):
            raise RuntimeError(f"frozen test did not pass: {name}")
    return {
        "id": group["id"],
        "binary": str(binary),
        "binary_sha256": sha256_file(binary),
        "test_count": len(requested),
        "elapsed_seconds": elapsed,
        "outcome": "PASS",
    }


def git_provenance(source_root: Path) -> dict[str, Any]:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=source_root,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain=v1", "-z"],
        cwd=source_root,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout
    return {
        "head": head,
        "worktree_clean": not status,
        "status_sha256": sha256_bytes(status),
    }


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--source-root", type=Path, default=REPOSITORY_ROOT)
    parser.add_argument("--requested-claim", default=EXPECTED_POLICY["accepted_claim"])
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--physics-binary", type=Path)
    parser.add_argument("--application-binary", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    matrix = validate_matrix(arguments.matrix)
    validate_requested_claim(matrix, arguments.requested_claim)
    source_root = arguments.source_root.resolve()
    source_records = validate_source_boundary(matrix, source_root)
    scope_guard_record = validate_scope_guard_contract(matrix, source_root)
    bind_scope_guard_source_record(source_records, scope_guard_record)
    validation_summary = {
        "matrix_id": matrix["matrix_id"],
        "status": matrix["status"],
        "requested_claim": arguments.requested_claim,
        "source_check_count": len(source_records),
        "scope_guard_accepted_case_count": scope_guard_record["accepted_case_count"],
        "scope_guard_rejected_case_count": scope_guard_record["rejected_case_count"],
        "scope_guard_invalid_case_count": scope_guard_record["invalid_case_count"],
        "group_count": len(matrix["groups"]),
        "test_count": sum(len(group["tests"]) for group in matrix["groups"]),
        "unimplemented_wp10_requirement_count": len(
            matrix["unimplemented_wp10_requirements"]
        ),
        "blocked_wp10_exit_count": len(matrix["blocked_wp10_qualification_exits"]),
        "blocked_q7_exit_count": len(matrix["blocked_q7_progression"]),
        "wp10_closed": False,
        "q7_closed": False,
        "outcome": "PASS",
    }
    if arguments.validate_only:
        if any(
            value is not None
            for value in (
                arguments.physics_binary,
                arguments.application_binary,
                arguments.output,
            )
        ):
            raise ValueError(
                "--validate-only does not accept binary or output arguments"
            )
        print(json.dumps(validation_summary, sort_keys=True))
        return 0

    if (
        arguments.physics_binary is None
        or arguments.application_binary is None
        or arguments.output is None
    ):
        raise ValueError(
            "execution requires --physics-binary, --application-binary, and --output"
        )
    output = arguments.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    matrix_sha256 = sha256_file(DEFAULT_MATRIX)
    script_sha256 = sha256_file(SCRIPT_PATH)
    source_hashes_before = {
        record["path"]: record["sha256"] for record in source_records
    }
    binary_arguments = {
        "physics_binary": arguments.physics_binary,
        "application_binary": arguments.application_binary,
    }
    group_results = [
        run_group(
            group,
            binary_arguments[group["binary_argument"]],
            output,
        )
        for group in matrix["groups"]
    ]
    source_records_after = validate_source_boundary(matrix, source_root)
    scope_guard_record_after = validate_scope_guard_contract(matrix, source_root)
    bind_scope_guard_source_record(source_records_after, scope_guard_record_after)
    source_hashes_after = {
        record["path"]: record["sha256"] for record in source_records_after
    }
    if source_hashes_after != source_hashes_before:
        raise RuntimeError("source boundary changed during execution")
    if scope_guard_record_after != scope_guard_record:
        raise RuntimeError("scope guard contract changed during execution")
    if (
        sha256_file(DEFAULT_MATRIX) != matrix_sha256
        or sha256_file(SCRIPT_PATH) != script_sha256
    ):
        raise RuntimeError("matrix or wrapper changed during execution")

    provenance = git_provenance(source_root)
    summary = {
        **validation_summary,
        "matrix_path": str(DEFAULT_MATRIX),
        "matrix_sha256": matrix_sha256,
        "wrapper_path": str(SCRIPT_PATH),
        "wrapper_sha256": script_sha256,
        "architecture_record": matrix["architecture_record"],
        "qualification_scope": matrix["qualification_scope"],
        "source_checks": source_records_after,
        "scope_guard_contract": scope_guard_record_after,
        "groups": group_results,
        "provenance": provenance,
        "excluded_current_claims": matrix["excluded_current_claims"],
    }
    write_json_exclusive(output / "summary.json", summary)
    print(json.dumps(validation_summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        json.JSONDecodeError,
        OSError,
        subprocess.SubprocessError,
        ValueError,
        RuntimeError,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)

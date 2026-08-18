#!/usr/bin/env python3
"""Run the frozen Q0 control-harness prerequisite matrix.

Only ``q0_control_prerequisite`` is accepted. Q0 closure and physical-gate
claims fail before binary validation or artifact creation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re
import resource
import signal
import subprocess
import sys
import time
from typing import Any
import xml.etree.ElementTree as ET


SCRIPT_PATH = Path(__file__).resolve()
REPOSITORY_ROOT = SCRIPT_PATH.parents[3]
DEFAULT_MATRIX = SCRIPT_PATH.with_name(
    "free_surface_q0_harness_qualification_matrix.json"
)
EXPECTED_MATRIX_SHA256 = (
    "f1e7c0446dc9a5af2678c53e15b018dd235e307a87aab1483c091a34c6635eaa"
)
EXPECTED_MATRIX_ID = "free_surface_q0_harness_prerequisite_v1"
EXPECTED_STATUS = "FROZEN_PREREQUISITE_NONCLOSURE"
EXPECTED_CLAIM = "q0_control_prerequisite"
REJECTED_CLAIMS = {
    "q0_closure",
    "q0_qualification",
    "q0_campaign_pass",
    "physical_gate_ready",
}
EXPECTED_DISPOSITION = {
    "prerequisite_controls_frozen": True,
    "wp0_invalid_configuration_evidence_available": True,
    "wp0_invalid_input_matrix_ci_registered": True,
    "q0_campaign_execution_registered": False,
    "q0_complete_artifact_archived": False,
    "q0_closed": False,
    "audit_q0_checkbox_may_be_checked": False,
}
EXPECTED_OPEN_EXITS = {
    "wp0_invalid_input_matrix_same_revision_hosted_ci_execution_not_archived",
    "accepted_step_stage_time_and_state_geometry_map_revision_history_not_archived",
    "accepted_step_raw_and_post_maintenance_region_phase_inventory_incomplete",
    "accepted_step_complete_energy_dissipation_and_work_account_blocked_by_wp8",
    "accepted_step_extension_geometry_solver_retry_rollback_and_resource_telemetry_incomplete",
    "compiler_library_option_machine_mesh_reference_parameter_group_and_threshold_provenance_not_archived_together",
    "same_revision_q0_campaign_execution_not_registered",
    "complete_q0_campaign_artifact_not_archived",
}
HEX64 = re.compile(r"^[0-9a-f]{64}$")
GTEST_NAME = re.compile(r"^[A-Za-z0-9_]+\.[A-Za-z0-9_]+$")
IDENTIFIER = re.compile(r"^[a-z][a-z0-9_]*$")
LDD_ADDRESS_SUFFIX = re.compile(r"^(?P<value>.+?)\s+\(0x[0-9A-Fa-f]+\)$")
WP0_MATRIX_RELATIVE_PATH = Path(
    "tests/cases/fluid/free_surface_configuration_qualification_matrix.json"
)
PHYSICS_CMAKE_RELATIVE_PATH = Path("Code/Source/solver/Physics/CMakeLists.txt")
TESTS_WORKFLOW_RELATIVE_PATH = Path(".github/workflows/tests.yml")
UBUNTU_ACTION_RELATIVE_PATH = Path(".github/actions/test-ubuntu/action.yml")
MACOS_ACTION_RELATIVE_PATH = Path(".github/actions/test-macos/action.yml")
WP0_CMAKE_TEST_LIST = "_SVMP_WP0_CONFIGURATION_TESTS"
WP0_CMAKE_TEST_FILTER = "_SVMP_WP0_CONFIGURATION_FILTER"
WP0_CTEST_NAME = "Physics_FreeSurfaceConfiguration_WP0"
CI_CHAIN_RELATIVE_PATHS = (
    PHYSICS_CMAKE_RELATIVE_PATH,
    TESTS_WORKFLOW_RELATIVE_PATH,
    UBUNTU_ACTION_RELATIVE_PATH,
    MACOS_ACTION_RELATIVE_PATH,
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def stable_stat_identity(stat_result: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        stat_result.st_dev,
        stat_result.st_ino,
        stat_result.st_size,
        stat_result.st_mtime_ns,
        stat_result.st_ctime_ns,
    )


def read_stable_bytes(path: Path) -> bytes:
    with path.open("rb") as source:
        before = stable_stat_identity(os.fstat(source.fileno()))
        value = source.read()
        after = stable_stat_identity(os.fstat(source.fileno()))
    if before != after or len(value) != before[2]:
        raise ValueError(f"file changed while it was read: {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        before = stable_stat_identity(os.fstat(source.fileno()))
        size = 0
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
            size += len(block)
        after = stable_stat_identity(os.fstat(source.fileno()))
    if before != after or size != before[2]:
        raise ValueError(f"file changed while it was hashed: {path}")
    return digest.hexdigest()


def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def load_matrix(path: Path = DEFAULT_MATRIX) -> dict[str, Any]:
    if path.is_symlink():
        raise ValueError("Q0 frozen matrix path must not be a symbolic link")
    canonical_path = DEFAULT_MATRIX.absolute()
    if path.absolute() != canonical_path:
        raise ValueError("Q0 requires the canonical frozen matrix path")
    resolved = path.resolve(strict=True)
    if resolved != canonical_path.resolve(strict=True) or not resolved.is_file():
        raise ValueError("Q0 requires the canonical frozen matrix file")
    matrix_bytes = read_stable_bytes(resolved)
    if sha256_bytes(matrix_bytes) != EXPECTED_MATRIX_SHA256:
        raise ValueError("Q0 frozen matrix bytes changed")
    matrix = json.loads(
        matrix_bytes.decode("utf-8"),
        object_pairs_hook=reject_duplicate_keys,
    )
    return validate_matrix_contract(matrix)


def positive_execution_envelope(value: Any, context: str) -> None:
    if not isinstance(value, dict) or set(value) != {
        "mpi_ranks",
        "threads",
        "wall_time_seconds",
        "memory_mib",
        "output_mib",
    }:
        raise ValueError(f"{context} execution envelope changed")
    if any(
        not isinstance(item, int) or isinstance(item, bool) or item <= 0
        for item in value.values()
    ):
        raise ValueError(f"{context} execution limits must be positive integers")


def validate_matrix_contract(matrix: dict[str, Any]) -> dict[str, Any]:
    expected_keys = {
        "schema_version",
        "matrix_id",
        "status",
        "campaign",
        "audit_contract",
        "architecture_record",
        "freeze_scope",
        "source_definitions",
        "gtest_group",
        "pytest_group",
        "implemented_prerequisites",
        "open_exits",
        "closure_request_policy",
        "qualification_disposition",
        "qualification_scope",
    }
    if not isinstance(matrix, dict) or set(matrix) != expected_keys:
        raise ValueError("Q0 matrix top-level contract changed")
    if matrix.get("schema_version") != 1:
        raise ValueError("unsupported Q0 matrix schema")
    if matrix.get("matrix_id") != EXPECTED_MATRIX_ID:
        raise ValueError("unexpected Q0 matrix identifier")
    if matrix.get("status") != EXPECTED_STATUS:
        raise ValueError("Q0 matrix is not frozen as prerequisite nonclosure")
    if matrix.get("campaign") != "Q0":
        raise ValueError("Q0 matrix campaign changed")

    definitions = matrix.get("source_definitions")
    if not isinstance(definitions, list) or len(definitions) != 20:
        raise ValueError("Q0 source-definition inventory changed")
    identifiers: set[str] = set()
    paths: set[str] = set()
    for definition in definitions:
        if not isinstance(definition, dict) or set(definition) != {
            "id",
            "role",
            "path",
            "sha256",
            "required_fragments",
        }:
            raise ValueError("invalid Q0 source definition")
        identifier = definition.get("id")
        path_text = definition.get("path")
        fragments = definition.get("required_fragments")
        if (
            not isinstance(identifier, str)
            or IDENTIFIER.fullmatch(identifier) is None
            or identifier in identifiers
        ):
            raise ValueError("invalid or duplicate Q0 source-definition id")
        path = PurePosixPath(path_text) if isinstance(path_text, str) else None
        if (
            path is None
            or path.is_absolute()
            or ".." in path.parts
            or "." in path.parts
            or path.as_posix() != path_text
            or path_text in paths
        ):
            raise ValueError("invalid or duplicate Q0 source-definition path")
        if (
            not isinstance(definition.get("sha256"), str)
            or HEX64.fullmatch(definition["sha256"]) is None
        ):
            raise ValueError("invalid Q0 source-definition hash")
        if (
            not isinstance(definition.get("role"), str)
            or not definition["role"]
            or not isinstance(fragments, list)
            or not fragments
            or any(not isinstance(item, str) or not item for item in fragments)
            or len(fragments) != len(set(fragments))
        ):
            raise ValueError("invalid Q0 source-definition semantics")
        identifiers.add(identifier)
        paths.add(path_text)

    gtest_group = matrix.get("gtest_group")
    if not isinstance(gtest_group, dict) or set(gtest_group) != {
        "id",
        "binary_argument",
        "tests",
        "execution",
    }:
        raise ValueError("Q0 GoogleTest group changed")
    if (
        gtest_group.get("id") != "wp0_invalid_configuration_serial"
        or gtest_group.get("binary_argument") != "physics_binary"
    ):
        raise ValueError("Q0 GoogleTest group identity changed")
    gtests = gtest_group.get("tests")
    if (
        not isinstance(gtests, list)
        or len(gtests) != 24
        or len(gtests) != len(set(gtests))
        or any(
            not isinstance(name, str) or GTEST_NAME.fullmatch(name) is None
            for name in gtests
        )
    ):
        raise ValueError("Q0 invalid-configuration test inventory changed")
    positive_execution_envelope(gtest_group.get("execution"), "Q0 GoogleTest")

    pytest_group = matrix.get("pytest_group")
    if not isinstance(pytest_group, dict) or set(pytest_group) != {
        "id",
        "paths",
        "tests",
        "expected_test_count",
        "execution",
    }:
        raise ValueError("Q0 pytest group changed")
    pytest_paths = pytest_group.get("paths")
    pytest_tests = pytest_group.get("tests")
    if (
        pytest_group.get("id") != "qualification_control_contracts"
        or not isinstance(pytest_paths, list)
        or len(pytest_paths) != 4
        or len(pytest_paths) != len(set(pytest_paths))
        or not isinstance(pytest_tests, list)
        or len(pytest_tests) != 44
        or len(pytest_tests) != len(set(pytest_tests))
        or pytest_group.get("expected_test_count") != len(pytest_tests)
        or any(
            not isinstance(name, str)
            or "::" not in name
            or not any(name.startswith(path + "::") for path in pytest_paths)
            for name in pytest_tests
        )
    ):
        raise ValueError("Q0 pytest inventory changed")
    positive_execution_envelope(pytest_group.get("execution"), "Q0 pytest")

    implemented = matrix.get("implemented_prerequisites")
    if (
        not isinstance(implemented, list)
        or not implemented
        or len(implemented) != len(set(implemented))
        or any(not isinstance(item, str) or not item for item in implemented)
    ):
        raise ValueError("Q0 implemented prerequisite list changed")
    exits = matrix.get("open_exits")
    if not isinstance(exits, list):
        raise ValueError("Q0 open-exit list is missing")
    exit_ids: set[str] = set()
    for entry in exits:
        if (
            not isinstance(entry, dict)
            or set(entry) != {"id", "status"}
            or entry.get("status") != "REQUIRED_NOT_CLAIMED"
            or not isinstance(entry.get("id"), str)
        ):
            raise ValueError("invalid Q0 open exit")
        exit_ids.add(entry["id"])
    if exit_ids != EXPECTED_OPEN_EXITS or len(exit_ids) != len(exits):
        raise ValueError("Q0 open-exit inventory changed")

    policy = matrix.get("closure_request_policy")
    if (
        not isinstance(policy, dict)
        or policy.get("accepted_claim") != EXPECTED_CLAIM
        or set(policy.get("rejected_claims", [])) != REJECTED_CLAIMS
        or policy.get("reject_any_claim_suffix") != "_closure"
        or not isinstance(policy.get("diagnostic"), str)
        or not policy["diagnostic"]
    ):
        raise ValueError("Q0 closure-request policy changed")
    if matrix.get("qualification_disposition") != EXPECTED_DISPOSITION:
        raise ValueError("Q0 nonclosure disposition changed")
    return matrix


def resolve_regular_repository_file(
    source_root: Path,
    relative: Path,
    context: str,
) -> Path:
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{context} path is outside the repository: {relative}")
    candidate = source_root
    for component in relative.parts:
        candidate /= component
        if candidate.is_symlink():
            raise ValueError(
                f"{context} contains a symbolic-link path component: {relative}"
            )
    try:
        path = candidate.resolve(strict=True)
    except OSError as error:
        raise ValueError(
            f"{context} is not a regular repository file: {relative}"
        ) from error
    if not path.is_relative_to(source_root) or not path.is_file():
        raise ValueError(f"{context} is not a regular repository file: {relative}")
    return path


def extract_cmake_list(cmake_text: str, variable: str) -> list[str]:
    pattern = re.compile(
        rf"(?ms)^[ \t]*set\({re.escape(variable)}[ \t]*\n"
        r"(?P<body>.*?)^[ \t]*\)[ \t]*$"
    )
    matches = list(pattern.finditer(cmake_text))
    if len(matches) != 1:
        raise ValueError(f"Q0 CMake list {variable} must be defined exactly once")
    values = [
        line.strip()
        for line in matches[0].group("body").splitlines()
        if line.strip()
    ]
    if (
        not values
        or len(values) != len(set(values))
        or any(GTEST_NAME.fullmatch(value) is None for value in values)
    ):
        raise ValueError(f"Q0 CMake list {variable} has an invalid test inventory")
    return values


def extract_indented_yaml_block(text: str, key: str, indent: int) -> str:
    lines = text.splitlines()
    prefix = " " * indent + key + ":"
    starts = [
        index
        for index, line in enumerate(lines)
        if line == prefix
    ]
    if len(starts) != 1:
        raise ValueError(
            f"Q0 CI YAML key {key!r} at indentation {indent} "
            "must appear exactly once"
        )
    start = starts[0]
    end = len(lines)
    for index in range(start + 1, len(lines)):
        stripped = lines[index].strip()
        if not stripped or stripped.startswith("#"):
            continue
        current_indent = len(lines[index]) - len(lines[index].lstrip(" "))
        if current_indent <= indent:
            end = index
            break
    return "\n".join(lines[start + 1 : end])


def require_unfiltered_ctest_action(action_text: str, action_name: str) -> None:
    lines = action_text.splitlines()
    step_starts = [
        index
        for index, line in enumerate(lines)
        if line == "    - name: Run unit tests"
    ]
    if len(step_starts) != 1:
        raise ValueError(
            f"Q0 {action_name} action must have one Run unit tests step"
        )
    start = step_starts[0]
    end = len(lines)
    for index in range(start + 1, len(lines)):
        if lines[index].startswith("    - "):
            end = index
            break
    step_lines = [line.strip() for line in lines[start:end] if line.strip()]
    ctest_commands = [
        line for line in step_lines if line == "ctest --verbose" or line.startswith("ctest ")
    ]
    if ctest_commands != ["ctest --verbose"]:
        raise ValueError(
            f"Q0 {action_name} action must invoke one unfiltered 'ctest --verbose'"
        )


def validate_wp0_ci_registration(
    matrix: dict[str, Any], source_root: Path
) -> dict[str, Any]:
    source_root = source_root.resolve(strict=True)
    paths = {
        relative: resolve_regular_repository_file(
            source_root,
            relative,
            "Q0 CI-chain source",
        )
        for relative in (WP0_MATRIX_RELATIVE_PATH, *CI_CHAIN_RELATIVE_PATHS)
    }
    tracked_output = git_bytes(
        source_root,
        "ls-files",
        "--",
        *(relative.as_posix() for relative in CI_CHAIN_RELATIVE_PATHS),
    ).decode("utf-8")
    tracked_paths = {line for line in tracked_output.splitlines() if line}
    expected_tracked_paths = {
        relative.as_posix() for relative in CI_CHAIN_RELATIVE_PATHS
    }
    if tracked_paths != expected_tracked_paths:
        raise ValueError(
            "Q0 CI chain must use the exact tracked workflow, actions, and "
            "Physics CMake files"
        )

    wp0_matrix = json.loads(
        read_stable_bytes(paths[WP0_MATRIX_RELATIVE_PATH]).decode("utf-8"),
        object_pairs_hook=reject_duplicate_keys,
    )
    wp0_tests = wp0_matrix.get("tests")
    if (
        wp0_matrix.get("work_package") != "WP-0"
        or wp0_matrix.get("gates", {}).get("expected_test_count") != 24
        or not isinstance(wp0_tests, list)
        or len(wp0_tests) != 24
    ):
        raise ValueError("Q0 WP-0 configuration matrix contract changed")
    q0_tests = matrix["gtest_group"]["tests"]
    if wp0_tests != q0_tests:
        raise ValueError("Q0 and WP-0 frozen test inventories differ")

    cmake_text = read_stable_bytes(
        paths[PHYSICS_CMAKE_RELATIVE_PATH]
    ).decode("utf-8")
    cmake_tests = extract_cmake_list(cmake_text, WP0_CMAKE_TEST_LIST)
    if cmake_tests != wp0_tests:
        raise ValueError(
            "Q0 dedicated CTest inventory differs from the frozen WP-0 matrix"
        )
    if (
        f'list(JOIN {WP0_CMAKE_TEST_LIST} ":" {WP0_CMAKE_TEST_FILTER})'
        not in cmake_text
    ):
        raise ValueError("Q0 dedicated CTest filter is not joined exactly")
    normalized_cmake = " ".join(cmake_text.split())
    expected_add_test = (
        f'add_test( NAME {WP0_CTEST_NAME} COMMAND test_physics '
        f'"--gtest_filter=${{{WP0_CMAKE_TEST_FILTER}}}" )'
    )
    if expected_add_test not in normalized_cmake:
        raise ValueError("Q0 dedicated CTest command changed")
    expected_properties = (
        f"set_tests_properties( {WP0_CTEST_NAME} PROPERTIES "
        'TIMEOUT 300 PROCESSORS 1 '
        'LABELS "physics;free-surface;qualification;wp0" )'
    )
    if expected_properties not in normalized_cmake:
        raise ValueError("Q0 dedicated CTest properties changed")

    workflow_text = read_stable_bytes(
        paths[TESTS_WORKFLOW_RELATIVE_PATH]
    ).decode("utf-8")
    trigger_block = extract_indented_yaml_block(workflow_text, "on", 0)
    trigger_keys = {
        line.strip().removesuffix(":")
        for line in trigger_block.splitlines()
        if line.startswith("  ") and not line.startswith("    ") and line.strip()
    }
    if not {"push", "pull_request"}.issubset(trigger_keys):
        raise ValueError("Q0 tests workflow must run for push and pull_request")
    jobs_block = extract_indented_yaml_block(workflow_text, "jobs", 0)
    for job_id, runner_name, action_path in (
        ("test-ubuntu", "ubuntu-latest", "./.github/actions/test-ubuntu"),
        ("test-macos", "macos-latest", "./.github/actions/test-macos"),
    ):
        job_block = extract_indented_yaml_block(jobs_block, job_id, 2)
        if f"runs-on: {runner_name}" not in job_block:
            raise ValueError(f"Q0 workflow job {job_id} runner changed")
        if job_block.count(f"uses: {action_path}") != 1:
            raise ValueError(
                f"Q0 workflow job {job_id} must invoke {action_path} exactly once"
            )

    require_unfiltered_ctest_action(
        read_stable_bytes(paths[UBUNTU_ACTION_RELATIVE_PATH]).decode("utf-8"),
        "Ubuntu",
    )
    require_unfiltered_ctest_action(
        read_stable_bytes(paths[MACOS_ACTION_RELATIVE_PATH]).decode("utf-8"),
        "macOS",
    )
    return {
        "ctest_name": WP0_CTEST_NAME,
        "test_count": len(cmake_tests),
        "workflow_triggers": ["pull_request", "push"],
        "workflow_jobs": ["test-ubuntu", "test-macos"],
        "hosted_execution_archived": False,
        "outcome": "REGISTERED_AWAITING_HOSTED_EXECUTION",
    }


def validate_source_definitions(
    matrix: dict[str, Any], source_root: Path
) -> list[dict[str, Any]]:
    if source_root.is_symlink():
        raise ValueError("Q0 source root must not be a symbolic link")
    source_root = source_root.resolve(strict=True)
    records: list[dict[str, Any]] = []
    for definition in matrix["source_definitions"]:
        relative = Path(definition["path"])
        path = resolve_regular_repository_file(
            source_root,
            relative,
            "Q0 source definition",
        )
        source_bytes = read_stable_bytes(path)
        digest = sha256_bytes(source_bytes)
        if digest != definition["sha256"]:
            raise ValueError(f"Q0 source-definition hash changed: {definition['id']}")
        text = source_bytes.decode("utf-8")
        missing = [
            fragment
            for fragment in definition["required_fragments"]
            if fragment not in text
        ]
        if missing:
            raise ValueError(
                f"Q0 source definition {definition['id']} is missing "
                f"required fragments: {missing}"
            )
        records.append(
            {
                "id": definition["id"],
                "role": definition["role"],
                "path": definition["path"],
                "sha256": digest,
                "required_fragment_count": len(definition["required_fragments"]),
                "outcome": "PASS",
            }
        )
    if matrix.get("matrix_id") == EXPECTED_MATRIX_ID:
        validate_wp0_ci_registration(matrix, source_root)
    return records


def canonical_source_root(path: Path) -> Path:
    if path.is_symlink():
        raise ValueError("Q0 source root must not be a symbolic link")
    canonical = REPOSITORY_ROOT.absolute()
    if path.absolute() != canonical:
        raise ValueError("Q0 requires the canonical repository source root")
    resolved = path.resolve(strict=True)
    if resolved != canonical.resolve(strict=True) or not resolved.is_dir():
        raise ValueError("Q0 requires the canonical repository source root")
    return resolved


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
    tests: set[str] = set()
    for line in result.stdout.splitlines():
        if line and not line[0].isspace():
            suite = line.split("#", 1)[0].strip().removesuffix(".")
            continue
        test = line.split("#", 1)[0].strip()
        if suite and test:
            name = f"{suite}.{test}"
            if name in tests:
                raise ValueError(f"duplicate listed GoogleTest identifier: {name}")
            tests.add(name)
    return tests


def collected_pytests(paths: list[str], source_root: Path) -> set[str]:
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", *paths],
        cwd=source_root,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=120,
    )
    return {
        line.strip()
        for line in result.stdout.splitlines()
        if "::" in line and any(line.startswith(path + "::") for path in paths)
    }


def discover_tests(
    matrix: dict[str, Any], physics_binary: Path, source_root: Path
) -> dict[str, Any]:
    physics_binary = physics_binary.resolve()
    if not physics_binary.is_file() or not os.access(physics_binary, os.X_OK):
        raise ValueError(f"test binary is not executable: {physics_binary}")
    expected_gtests = set(matrix["gtest_group"]["tests"])
    expected_pytests = set(matrix["pytest_group"]["tests"])
    actual_gtests = listed_gtests(physics_binary)
    actual_pytests = collected_pytests(matrix["pytest_group"]["paths"], source_root)
    missing_gtests = sorted(expected_gtests - actual_gtests)
    missing_pytests = sorted(expected_pytests - actual_pytests)
    unexpected_pytests = sorted(actual_pytests - expected_pytests)
    return {
        "physics_binary": str(physics_binary),
        "physics_binary_sha256": sha256_file(physics_binary),
        "expected_gtest_count": len(expected_gtests),
        "listed_gtest_count": len(expected_gtests & actual_gtests),
        "missing_gtests": missing_gtests,
        "expected_pytest_count": len(expected_pytests),
        "listed_pytest_count": len(expected_pytests & actual_pytests),
        "missing_pytests": missing_pytests,
        "unexpected_pytests": unexpected_pytests,
        "passed": not (missing_gtests or missing_pytests or unexpected_pytests),
    }


def git_bytes(source_root: Path, *arguments: str) -> bytes:
    return subprocess.run(
        ["git", *arguments],
        cwd=source_root,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout


def source_state(source_root: Path) -> dict[str, Any]:
    source_root = source_root.resolve()
    commit = git_bytes(source_root, "rev-parse", "HEAD").decode().strip()
    tree = git_bytes(source_root, "rev-parse", "HEAD^{tree}").decode().strip()
    tracked_diff = git_bytes(source_root, "diff", "--binary", "HEAD")
    status = git_bytes(source_root, "status", "--porcelain=v1", "-z")
    untracked_names = sorted(
        item.decode("utf-8", "surrogateescape")
        for item in git_bytes(
            source_root,
            "ls-files",
            "--others",
            "--exclude-standard",
            "-z",
        ).split(b"\0")
        if item
    )
    untracked: list[dict[str, Any]] = []
    for name in untracked_names:
        path = resolve_regular_repository_file(
            source_root,
            Path(name),
            "untracked source",
        )
        untracked.append(
            {
                "path": name,
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )
    dirty_record = {
        "tracked_diff_sha256": sha256_bytes(tracked_diff),
        "untracked_sources": untracked,
    }
    return {
        "source_commit": commit,
        "source_tree": tree,
        "worktree_clean": not status,
        "status_sha256": sha256_bytes(status),
        "tracked_diff_sha256": dirty_record["tracked_diff_sha256"],
        "untracked_sources": untracked,
        "dirty_tree_sha256": sha256_bytes(
            json.dumps(dirty_record, sort_keys=True, separators=(",", ":")).encode()
        ),
    }


def find_cmake_cache(binary: Path) -> Path | None:
    for directory in [binary.parent, *binary.parents]:
        candidate = directory / "CMakeCache.txt"
        if candidate.is_file():
            return candidate
    return None


def parse_cmake_cache(text: str) -> dict[str, tuple[str, str]]:
    entries: dict[str, tuple[str, str]] = {}
    for line in text.splitlines():
        if line.startswith(("#", "//")) or "=" not in line:
            continue
        left, value = line.split("=", 1)
        variable, separator, cache_type = left.partition(":")
        if not variable or variable in entries:
            raise ValueError(f"invalid or duplicate CMake cache variable: {variable}")
        entries[variable] = (cache_type if separator else "", value)
    return entries


def selected_cmake_cache_text(text: str) -> dict[str, str]:
    prefixes = (
        "CMAKE_BUILD_TYPE:",
        "CMAKE_CXX_COMPILER:",
        "CMAKE_CXX_COMPILER_ID:",
        "CMAKE_CXX_COMPILER_VERSION:",
        "CMAKE_CXX_FLAGS:",
        "CMAKE_CXX_FLAGS_",
        "FE_ENABLE_MPI:",
        "PHYSICS_BUILD_TESTS:",
    )
    selected: dict[str, str] = {}
    for line in text.splitlines():
        if not line.startswith(("#", "//")) and "=" in line:
            left, value = line.split("=", 1)
            if left.startswith(prefixes):
                selected[left] = value
    return selected


def selected_cmake_cache(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    return selected_cmake_cache_text(
        read_stable_bytes(path).decode("utf-8", errors="replace")
    )


def cmake_cache_snapshot(binary: Path) -> dict[str, Any]:
    cache = find_cmake_cache(binary)
    if cache is None:
        raise ValueError("test binary does not have a CMake cache")
    if cache.is_symlink():
        raise ValueError("test binary CMake cache must not be a symbolic link")
    cache = cache.resolve(strict=True)
    if not cache.is_file():
        raise ValueError("test binary CMake cache is not a regular file")
    build_directory = cache.parent.resolve(strict=True)
    binary = binary.resolve(strict=True)
    if not binary.is_relative_to(build_directory):
        raise ValueError("test binary is outside its CMake build directory")

    cache_bytes = read_stable_bytes(cache)
    cache_text = cache_bytes.decode("utf-8", errors="strict")
    entries = parse_cmake_cache(cache_text)
    required = {
        "CMAKE_CACHEFILE_DIR",
        "CMAKE_HOME_DIRECTORY",
        "CMAKE_PROJECT_NAME",
        "PHYSICS_BUILD_TESTS",
    }
    if not required.issubset(entries):
        raise ValueError("test binary CMake cache identity is incomplete")
    declared_build = Path(entries["CMAKE_CACHEFILE_DIR"][1])
    declared_source = Path(entries["CMAKE_HOME_DIRECTORY"][1])
    project_name = entries["CMAKE_PROJECT_NAME"][1]
    if (
        not declared_build.is_absolute()
        or declared_build.resolve(strict=True) != build_directory
    ):
        raise ValueError("test binary CMake cache build-directory identity changed")
    if not declared_source.is_absolute():
        raise ValueError("test binary CMake source directory is not absolute")
    source_directory = declared_source.resolve(strict=True)
    expected_source = (
        REPOSITORY_ROOT / "Code" / "Source" / "solver" / "Physics"
    ).resolve(strict=True)
    if (
        not source_directory.is_dir()
        or source_directory != expected_source
        or project_name != "svMultiPhysicsPhysics"
        or entries["PHYSICS_BUILD_TESTS"] != ("BOOL", "ON")
    ):
        raise ValueError("test binary CMake cache source identity changed")
    return {
        "path": str(cache),
        "sha256": sha256_bytes(cache_bytes),
        "size_bytes": len(cache_bytes),
        "build_directory": str(build_directory),
        "source_directory": str(source_directory),
        "project_name": project_name,
        "selected_entries": selected_cmake_cache_text(cache_text),
    }


def split_ldd_address(value: str) -> tuple[str, bool]:
    match = LDD_ADDRESS_SUFFIX.fullmatch(value.strip())
    if match is None:
        return value.strip(), False
    return match.group("value").strip(), True


def linked_library_manifest(output: str) -> dict[str, Any]:
    libraries: list[dict[str, Any]] = []
    virtual_dependencies: list[str] = []
    requested_names: set[str] = set()
    statically_linked = False

    for raw_line in output.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line == "statically linked":
            statically_linked = True
            continue
        if line == "not a dynamic executable":
            raise ValueError("test binary is not a dynamic executable")

        requested_name: str
        target_text: str | None
        if "=>" in line:
            requested_name, remainder = (
                item.strip() for item in line.split("=>", maxsplit=1)
            )
            if remainder == "not found" or remainder.startswith("not found "):
                raise ValueError(
                    f"linked library was not found: {requested_name or '<unknown>'}"
                )
            target_text, _had_address = split_ldd_address(remainder)
            if not requested_name or not target_text:
                raise ValueError(f"unrecognized linked-library record: {line}")
        else:
            subject, had_address = split_ldd_address(line)
            if not had_address:
                raise ValueError(f"unrecognized linked-library record: {line}")
            if subject.startswith("/"):
                target_text = subject
                requested_name = Path(subject).name
            else:
                target_text = None
                requested_name = subject

        if not requested_name or requested_name in requested_names:
            raise ValueError(f"duplicate linked-library record: {requested_name}")
        requested_names.add(requested_name)
        if target_text is None:
            virtual_dependencies.append(requested_name)
            continue

        reported_path = Path(target_text)
        if not reported_path.is_absolute():
            raise ValueError(
                f"linked library does not have an absolute path: {requested_name}"
            )
        try:
            resolved_path = reported_path.resolve(strict=True)
        except OSError as error:
            raise ValueError(
                f"linked library path cannot be resolved: {requested_name}"
            ) from error
        if not resolved_path.is_file():
            raise ValueError(f"linked library is not a regular file: {requested_name}")
        libraries.append(
            {
                "requested_name": requested_name,
                "resolved_path": str(resolved_path),
                "sha256": sha256_file(resolved_path),
                "size_bytes": resolved_path.stat().st_size,
            }
        )

    if statically_linked and (libraries or virtual_dependencies):
        raise ValueError("static-link marker conflicts with library records")
    if not statically_linked and not libraries:
        raise ValueError("dynamic test binary has no resolved library records")

    libraries.sort(key=lambda item: (item["requested_name"], item["resolved_path"]))
    virtual_dependencies.sort()
    canonical_record = {
        "linkage": "static" if statically_linked else "dynamic",
        "libraries": libraries,
        "virtual_dependencies": virtual_dependencies,
    }
    return {
        **canonical_record,
        "manifest_sha256": sha256_bytes(
            json.dumps(
                canonical_record,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ),
    }


def binary_build_provenance(binary: Path) -> dict[str, Any]:
    binary = binary.resolve(strict=True)
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise ValueError(f"test binary is not executable: {binary}")
    binary_sha256 = sha256_file(binary)
    cache = cmake_cache_snapshot(binary)
    linked_output = subprocess.run(
        ["ldd", str(binary)],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    ).stdout
    linked = linked_library_manifest(linked_output)
    if sha256_file(binary) != binary_sha256:
        raise ValueError("test binary changed while build provenance was captured")
    if cmake_cache_snapshot(binary) != cache:
        raise ValueError("CMake cache changed while build provenance was captured")
    return {
        "binary": str(binary),
        "binary_sha256": binary_sha256,
        "binary_size_bytes": binary.stat().st_size,
        "cmake_cache": cache,
        "linked_libraries": linked["libraries"],
        "virtual_dependencies": linked["virtual_dependencies"],
        "linkage": linked["linkage"],
        "linked_library_manifest_sha256": linked["manifest_sha256"],
    }


def require_unchanged_build_provenance(
    discovery: dict[str, Any],
    before: dict[str, Any],
    after: dict[str, Any],
) -> None:
    if discovery["physics_binary_sha256"] != before["binary_sha256"]:
        raise RuntimeError(
            "discovery and build provenance used different test-binary bytes"
        )
    if before != after:
        raise RuntimeError(
            "test binary, CMake cache, or linked-library provenance "
            "changed during execution"
        )


def write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as output:
        json.dump(value, output, indent=2, sort_keys=True)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())


def write_text(path: Path, value: str) -> None:
    with path.open("x", encoding="utf-8") as output:
        output.write(value)
        output.flush()
        os.fsync(output.fileno())


def directory_size(path: Path) -> int:
    return sum(
        candidate.stat().st_size
        for candidate in path.rglob("*")
        if candidate.is_file() and not candidate.is_symlink()
    )


def process_resident_kib(process_id: int) -> int | None:
    try:
        for line in (
            Path(f"/proc/{process_id}/status").read_text(encoding="utf-8").splitlines()
        ):
            if line.startswith("VmRSS:"):
                return int(line.split()[1])
    except (OSError, ValueError, IndexError):
        return None
    return None


def terminate_process_group(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=5)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def execution_control_record(execution: dict[str, int]) -> dict[str, Any]:
    return {
        "requested_parallelism": {
            "mpi_ranks": execution["mpi_ranks"],
            "threads": execution["threads"],
        },
        "enforced_resource_limits": {
            "address_space_mib": execution["memory_mib"],
            "wall_time_seconds": execution["wall_time_seconds"],
            "output_mib": execution["output_mib"],
        },
    }


def run_monitored(
    command: list[str],
    source_root: Path,
    output_root: Path,
    stdout_path: Path,
    stderr_path: Path,
    execution: dict[str, int],
) -> dict[str, Any]:
    memory_bytes = execution["memory_mib"] * 1024 * 1024

    def set_limits() -> None:
        resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))

    started = time.monotonic()
    peak_resident_kib = 0
    termination_reason: str | None = None
    with stdout_path.open("xb") as stdout_file, stderr_path.open("xb") as stderr_file:
        process = subprocess.Popen(
            command,
            cwd=source_root,
            stdout=stdout_file,
            stderr=stderr_file,
            start_new_session=True,
            preexec_fn=set_limits,
        )
        while process.poll() is None:
            resident = process_resident_kib(process.pid)
            if resident is not None:
                peak_resident_kib = max(peak_resident_kib, resident)
            if time.monotonic() - started > execution["wall_time_seconds"]:
                termination_reason = "wall_time_envelope_exceeded"
            elif directory_size(output_root) > execution["output_mib"] * 1024 * 1024:
                termination_reason = "output_envelope_exceeded"
            elif resident is not None and resident > execution["memory_mib"] * 1024:
                termination_reason = "memory_envelope_exceeded"
            if termination_reason is not None:
                terminate_process_group(process)
                break
            time.sleep(0.02)
        return_code = process.wait()
    return {
        "command": command,
        **execution_control_record(execution),
        "return_code": return_code,
        "termination_reason": termination_reason,
        "wall_time_seconds": time.monotonic() - started,
        "peak_resident_kib_sampled": peak_resident_kib,
        "final_output_bytes": directory_size(output_root),
    }


def flatten_gtests(document: dict[str, Any]) -> dict[str, dict[str, Any]]:
    flattened: dict[str, dict[str, Any]] = {}
    for suite in document.get("testsuites", []):
        suite_name = suite.get("name")
        for test in suite.get("testsuite", []):
            if isinstance(suite_name, str) and isinstance(test.get("name"), str):
                name = f"{suite_name}.{test['name']}"
                if name in flattened:
                    raise ValueError(f"duplicate GoogleTest result identifier: {name}")
                flattened[name] = test
    return flattened


def pytest_junit_identity(node_id: str) -> tuple[str, str]:
    parts = node_id.split("::")
    if len(parts) < 2 or not parts[0].endswith(".py"):
        raise ValueError(f"invalid pytest node identifier: {node_id}")
    module = parts[0][:-3].replace("/", ".")
    nested = parts[1:-1]
    classname = ".".join([module, *nested])
    return classname, parts[-1]


def local_xml_name(tag: str) -> str:
    return tag.rsplit("}", maxsplit=1)[-1]


def parse_pytest_junit(path: Path, expected_tests: list[str]) -> dict[str, Any]:
    try:
        root = ET.parse(path).getroot()
    except (OSError, ET.ParseError) as error:
        raise ValueError("Q0 pytest group did not produce valid JUnit XML") from error
    if local_xml_name(root.tag) not in {"testsuite", "testsuites"}:
        raise ValueError("Q0 pytest JUnit root element is invalid")

    expected_by_identity: dict[tuple[str, str], str] = {}
    for node_id in expected_tests:
        identity = pytest_junit_identity(node_id)
        if identity in expected_by_identity:
            raise ValueError("Q0 pytest JUnit identities are not unique")
        expected_by_identity[identity] = node_id

    observed_counts: dict[tuple[str, str], int] = {}
    observed_tests: list[str] = []
    unexpected_tests: list[str] = []
    skipped_tests: list[str] = []
    failed_tests: list[str] = []
    error_tests: list[str] = []
    passed_tests: list[str] = []
    testcases = [
        element for element in root.iter() if local_xml_name(element.tag) == "testcase"
    ]
    for testcase in testcases:
        classname = testcase.get("classname")
        name = testcase.get("name")
        if not isinstance(classname, str) or not classname:
            raise ValueError("Q0 pytest JUnit testcase is missing classname")
        if not isinstance(name, str) or not name:
            raise ValueError("Q0 pytest JUnit testcase is missing name")
        identity = (classname, name)
        observed_counts[identity] = observed_counts.get(identity, 0) + 1
        node_id = expected_by_identity.get(identity)
        display_name = node_id or f"{classname}::{name}"
        if node_id is None:
            unexpected_tests.append(display_name)
        else:
            observed_tests.append(node_id)

        child_kinds = {local_xml_name(child.tag) for child in testcase}
        if "skipped" in child_kinds:
            skipped_tests.append(display_name)
        if "failure" in child_kinds:
            failed_tests.append(display_name)
        if "error" in child_kinds:
            error_tests.append(display_name)
        if not child_kinds.intersection({"skipped", "failure", "error"}):
            passed_tests.append(display_name)

    duplicate_tests = sorted(
        expected_by_identity.get(identity, f"{identity[0]}::{identity[1]}")
        for identity, count in observed_counts.items()
        if count != 1
    )
    observed_identities = set(observed_counts)
    missing_tests = sorted(
        node_id
        for identity, node_id in expected_by_identity.items()
        if identity not in observed_identities
    )
    unexpected_tests = sorted(set(unexpected_tests))
    skipped_tests.sort()
    failed_tests.sort()
    error_tests.sort()
    passed_tests.sort()

    suites = []
    for element in root.iter():
        if local_xml_name(element.tag) != "testsuite":
            continue
        direct_testcases = [
            child for child in element if local_xml_name(child.tag) == "testcase"
        ]
        if any(local_xml_name(child.tag) == "testsuite" for child in element):
            raise ValueError("Q0 pytest JUnit suites must not be nested")
        suites.append((element, direct_testcases))
    if not suites or sum(len(cases) for _suite, cases in suites) != len(testcases):
        raise ValueError("Q0 pytest JUnit testcase ownership is invalid")

    declared_totals = {
        "tests": 0,
        "failures": 0,
        "errors": 0,
        "skipped": 0,
    }
    declared_total_mismatches: list[str] = []
    for suite_index, (suite, cases) in enumerate(suites):
        suite_actual = {
            "tests": len(cases),
            "failures": sum(
                any(local_xml_name(child.tag) == "failure" for child in case)
                for case in cases
            ),
            "errors": sum(
                any(local_xml_name(child.tag) == "error" for child in case)
                for case in cases
            ),
            "skipped": sum(
                any(local_xml_name(child.tag) == "skipped" for child in case)
                for case in cases
            ),
        }
        for key, actual in suite_actual.items():
            raw_value = suite.get(key)
            if raw_value is None:
                raise ValueError(f"Q0 pytest JUnit suite is missing {key!r} total")
            try:
                value = int(raw_value)
            except ValueError as error:
                raise ValueError(
                    f"Q0 pytest JUnit suite has invalid {key!r} total"
                ) from error
            if value < 0:
                raise ValueError(f"Q0 pytest JUnit suite has invalid {key!r} total")
            declared_totals[key] += value
            if value != actual:
                declared_total_mismatches.append(f"suite[{suite_index}].{key}")
    declared_total_mismatches.sort()

    expected_count = len(expected_tests)
    passed = (
        len(testcases) == expected_count
        and len(observed_counts) == expected_count
        and not missing_tests
        and not unexpected_tests
        and not duplicate_tests
        and not skipped_tests
        and not failed_tests
        and not error_tests
        and not declared_total_mismatches
        and len(passed_tests) == expected_count
    )
    return {
        "expected_test_count": expected_count,
        "observed_test_count": len(testcases),
        "unique_observed_test_count": len(observed_counts),
        "observed_tests": sorted(observed_tests),
        "missing_tests": missing_tests,
        "unexpected_tests": unexpected_tests,
        "duplicate_tests": duplicate_tests,
        "skipped_tests": skipped_tests,
        "failed_tests": failed_tests,
        "error_tests": error_tests,
        "passed_test_count": len(passed_tests),
        "declared_totals": declared_totals,
        "declared_total_mismatches": declared_total_mismatches,
        "passed": passed,
    }


def pytest_execution_command(matrix: dict[str, Any], junit_path: Path) -> list[str]:
    return [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "--junitxml=" + str(junit_path),
        *matrix["pytest_group"]["tests"],
    ]


def write_checksums(output: Path) -> None:
    entries = [
        f"{sha256_file(path)}  {path.relative_to(output).as_posix()}"
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "checksums.txt"
    ]
    write_text(output / "checksums.txt", "\n".join(entries) + "\n")


def validation_summary(
    matrix: dict[str, Any],
    source_records: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "matrix_id": matrix["matrix_id"],
        "matrix_sha256": EXPECTED_MATRIX_SHA256,
        "status": matrix["status"],
        "requested_claim": EXPECTED_CLAIM,
        "source_definition_count": len(source_records),
        "gtest_count": len(matrix["gtest_group"]["tests"]),
        "pytest_count": len(matrix["pytest_group"]["tests"]),
        "open_exit_count": len(matrix["open_exits"]),
        **matrix["qualification_disposition"],
        "outcome": "PASS_PREREQUISITE_NONCLOSURE",
    }


def parse_control(
    arguments: list[str],
) -> tuple[str, bool, bool, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--requested-claim", default=EXPECTED_CLAIM)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--list-only", action="store_true")
    parsed, remaining = parser.parse_known_args(arguments)
    if parsed.validate_only and parsed.list_only:
        raise ValueError("--validate-only and --list-only are mutually exclusive")
    claim = parsed.requested_claim
    if claim in REJECTED_CLAIMS or claim.endswith("_closure"):
        raise ValueError(
            f"requested claim {claim!r} is outside this matrix: "
            "the same-revision accepted-step Q0 campaign is missing"
        )
    if claim != EXPECTED_CLAIM:
        raise ValueError(
            f"unsupported Q0 requested claim {claim!r}; expected {EXPECTED_CLAIM!r}"
        )
    return claim, parsed.validate_only, parsed.list_only, remaining


def list_only(
    matrix: dict[str, Any],
    source_records: list[dict[str, Any]],
    arguments: list[str],
) -> int:
    parser = argparse.ArgumentParser(prog=f"{SCRIPT_PATH.name} --list-only")
    parser.add_argument("--physics-binary", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, default=REPOSITORY_ROOT)
    parsed = parser.parse_args(arguments)
    source_root = canonical_source_root(parsed.source_root)
    discovery = discover_tests(matrix, parsed.physics_binary, source_root)
    build = binary_build_provenance(parsed.physics_binary)
    require_unchanged_build_provenance(discovery, build, build)
    print(
        json.dumps(
            {
                **validation_summary(matrix, source_records),
                "discovery": discovery,
                "build_preflight": build,
                "source_definition_sha256": {
                    record["path"]: record["sha256"] for record in source_records
                },
                "tests_executed": 0,
                "artifacts_written": 0,
                "outcome": (
                    "PASS_PREREQUISITE_NONCLOSURE" if discovery["passed"] else "FAIL"
                ),
            },
            sort_keys=True,
        )
    )
    return 0 if discovery["passed"] else 2


def execute(
    matrix: dict[str, Any],
    source_records: list[dict[str, Any]],
    arguments: list[str],
) -> int:
    parser = argparse.ArgumentParser(prog=SCRIPT_PATH.name)
    parser.add_argument("--physics-binary", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, default=REPOSITORY_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parsed = parser.parse_args(arguments)
    source_root = canonical_source_root(parsed.source_root)
    git_root = Path(
        git_bytes(source_root, "rev-parse", "--show-toplevel").decode().strip()
    ).resolve()
    if git_root != source_root:
        raise ValueError("Q0 source root must equal the repository top level")
    output = parsed.output.resolve()
    if output.is_relative_to(source_root):
        raise ValueError("Q0 prerequisite artifacts must be outside the source tree")
    if output.exists():
        raise ValueError(f"refusing to replace output directory: {output}")
    physics_binary = parsed.physics_binary.resolve(strict=True)
    discovery = discover_tests(matrix, physics_binary, source_root)
    if not discovery["passed"]:
        raise ValueError("Q0 frozen test discovery failed")
    build_before = binary_build_provenance(physics_binary)
    require_unchanged_build_provenance(discovery, build_before, build_before)

    matrix_hash = sha256_file(DEFAULT_MATRIX)
    if matrix_hash != EXPECTED_MATRIX_SHA256:
        raise RuntimeError("Q0 frozen matrix changed after validation")
    runner_hash = sha256_file(SCRIPT_PATH)
    source_before = source_state(source_root)
    source_hashes_before = {
        record["path"]: record["sha256"] for record in source_records
    }
    output.mkdir(parents=True, exist_ok=False)

    gtest_root = output / "wp0_invalid_configuration_serial"
    gtest_root.mkdir()
    gtest_result_path = gtest_root / "gtest.json"
    gtest_command = [
        str(physics_binary),
        "--gtest_filter=" + ":".join(matrix["gtest_group"]["tests"]),
        "--gtest_output=json:" + str(gtest_result_path),
    ]
    gtest_run = run_monitored(
        gtest_command,
        source_root,
        gtest_root,
        gtest_root / "stdout.txt",
        gtest_root / "stderr.txt",
        matrix["gtest_group"]["execution"],
    )
    if not gtest_result_path.is_file():
        raise RuntimeError("Q0 GoogleTest group did not produce JSON")
    gtest_document = json.loads(gtest_result_path.read_text(encoding="utf-8"))
    gtest_results = flatten_gtests(gtest_document)
    expected_gtests = set(matrix["gtest_group"]["tests"])
    gtest_passed = (
        gtest_run["return_code"] == 0
        and gtest_run["termination_reason"] is None
        and set(gtest_results) == expected_gtests
        and all(
            test.get("status") == "RUN"
            and test.get("result") == "COMPLETED"
            and not test.get("failures")
            for test in gtest_results.values()
        )
    )

    pytest_root = output / "qualification_control_contracts"
    pytest_root.mkdir()
    pytest_result_path = pytest_root / "pytest.xml"
    pytest_command = pytest_execution_command(matrix, pytest_result_path)
    pytest_run = run_monitored(
        pytest_command,
        source_root,
        pytest_root,
        pytest_root / "stdout.txt",
        pytest_root / "stderr.txt",
        matrix["pytest_group"]["execution"],
    )
    pytest_inventory = parse_pytest_junit(
        pytest_result_path,
        matrix["pytest_group"]["tests"],
    )
    pytest_passed = (
        pytest_run["return_code"] == 0
        and pytest_run["termination_reason"] is None
        and pytest_inventory["passed"]
    )

    build_after = binary_build_provenance(physics_binary)
    require_unchanged_build_provenance(discovery, build_before, build_after)
    source_records_after = validate_source_definitions(matrix, source_root)
    source_hashes_after = {
        record["path"]: record["sha256"] for record in source_records_after
    }
    source_after = source_state(source_root)
    if (
        source_hashes_after != source_hashes_before
        or source_after != source_before
        or sha256_file(DEFAULT_MATRIX) != matrix_hash
        or sha256_file(SCRIPT_PATH) != runner_hash
    ):
        raise RuntimeError("Q0 source, matrix, or runner changed during execution")

    passed = gtest_passed and pytest_passed
    write_json(
        output / "manifest.json",
        {
            "artifact_schema_version": 1,
            "artifact_class": "q0_control_prerequisite_nonclosure",
            "matrix_id": matrix["matrix_id"],
            "matrix_sha256": matrix_hash,
            "runner_sha256": runner_hash,
            "requested_claim": EXPECTED_CLAIM,
            "qualification_disposition": matrix["qualification_disposition"],
            "open_exits": matrix["open_exits"],
            "outcome": ("PASS_PREREQUISITE_NONCLOSURE" if passed else "FAIL"),
        },
    )
    write_json(
        output / "provenance.json",
        {
            "artifact_schema_version": 1,
            **source_after,
            "matrix_path": str(DEFAULT_MATRIX),
            "matrix_sha256": matrix_hash,
            "runner_path": str(SCRIPT_PATH),
            "runner_sha256": runner_hash,
            "source_definitions": source_records_after,
            "build": build_after,
            "machine": {
                "platform": platform.platform(),
                "machine": platform.machine(),
                "processor": platform.processor(),
                "cpu_count": os.cpu_count(),
            },
            "execution_controls": {
                matrix["gtest_group"]["id"]: execution_control_record(
                    matrix["gtest_group"]["execution"]
                ),
                matrix["pytest_group"]["id"]: execution_control_record(
                    matrix["pytest_group"]["execution"]
                ),
            },
        },
    )
    write_json(
        output / "gates.json",
        {
            "artifact_schema_version": 1,
            "gtest_group": matrix["gtest_group"],
            "pytest_group": matrix["pytest_group"],
            "execution_semantics": {
                "mpi_ranks": "requested process topology",
                "threads": "requested concurrency",
                "wall_time_seconds": "enforced elapsed-time limit",
                "memory_mib": "enforced address-space limit",
                "output_mib": "enforced output-size limit",
            },
            "expected_failures": 0,
            "q0_closure_expected": False,
        },
    )
    write_json(
        output / "summary.json",
        {
            **validation_summary(matrix, source_records_after),
            "discovery": discovery,
            "gtest_run": gtest_run,
            "gtest_passed": gtest_passed,
            "pytest_run": pytest_run,
            "pytest_inventory": pytest_inventory,
            "pytest_passed": pytest_passed,
            "outcome": ("PASS_PREREQUISITE_NONCLOSURE" if passed else "FAIL"),
        },
    )
    write_checksums(output)
    print(output)
    print("PASS_PREREQUISITE_NONCLOSURE" if passed else "FAIL")
    return 0 if passed else 2


def main(arguments: list[str] | None = None) -> int:
    provided = sys.argv[1:] if arguments is None else arguments
    _claim, validate_only, list_mode, remaining = parse_control(provided)
    matrix = load_matrix(DEFAULT_MATRIX)
    source_root = REPOSITORY_ROOT.resolve()
    source_records = validate_source_definitions(matrix, source_root)
    if validate_only:
        if remaining:
            raise ValueError("--validate-only does not accept execution arguments")
        print(
            json.dumps(
                validation_summary(matrix, source_records),
                sort_keys=True,
            )
        )
        return 0
    if list_mode:
        return list_only(matrix, source_records, remaining)
    return execute(matrix, source_records, remaining)


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

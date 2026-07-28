#!/usr/bin/env python3
"""Run the frozen WP-7 active-cell topology telemetry prerequisite."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import re
import resource
import shlex
import shutil
import signal
import subprocess
import sys
import time
from typing import Any


SCRIPT_PATH = Path(__file__).resolve()
REPOSITORY_ROOT = SCRIPT_PATH.parents[3]
DEFAULT_MATRIX = SCRIPT_PATH.with_name(
    "free_surface_wp7_active_cell_topology_prerequisite_matrix.json"
)
EXPECTED_MATRIX_SHA256 = (
    "64f1478ea8fe0a8eda1e3bb76d89f48d1122a6675705e610c3000d905b7391af"
)
EXPECTED_MATRIX_ID = "free_surface_wp7_active_cell_topology_prerequisite_v1"
EXPECTED_SOURCE_COMMIT = "fa97fe108e82b633b44ffe871a89075f9f30f007"
EXPECTED_SOURCE_TREE = "addeffd2cbf01ef2d154cf9a99d98270d370f3ce"
EXPECTED_DEFINITION_PATHS = {
    "tests/cases/fluid/"
    "free_surface_wp7_active_cell_topology_prerequisite_matrix.json",
    "tests/cases/fluid/"
    "run_free_surface_wp7_active_cell_topology_prerequisite.py",
}
EXPECTED_SCOPE = (
    "Finite active-cell topology telemetry prerequisite evidence only: "
    "serial refresh-report lifecycle, two-rank canonical provider "
    "reconciliation, and one serial production P1 velocity/pressure fixture "
    "covering connected rooted, "
    "disconnected rooted, and rooted-plus-rootless background-cell graphs. "
    "This matrix does not establish physical-phase connectivity, "
    "liquid-volume deletion or conservation, topology-transition "
    "continuity, manufactured convergence, conditioning, Krylov stability, "
    "simulation exits, FSR-07, WP-7, or Q1 closure."
)
EXPECTED_CLAIM_POLICY = {
    "accepted_claim": "active_cell_topology_telemetry_prerequisite",
    "rejected_claims": [
        "fsr07_closure",
        "wp7_closure",
        "q1_closure",
        "uniform_cut_stability",
        "conservative_feature_deletion",
        "resolved_disconnected_stability",
    ],
    "diagnostic": (
        "The bounded fixtures omit physical-phase component reconstruction, "
        "conservative feature transfer, cut-position error and conditioning "
        "rates, production solver spread, continuous node crossing, and "
        "simulation exits."
    ),
}
EXPECTED_DISPOSITION = {
    "fsr07_closed": False,
    "wp7_closed": False,
    "q1_closed": False,
    "uniform_inf_sup_established": False,
    "conservation_established": False,
    "resolved_feature_survival_established": False,
}
EXPECTED_OPEN_OUTCOMES = {
    "fsr07": "OPEN",
    "wp7": "OPEN",
    "q1": "OPEN",
}
EXPECTED_PARENT_STATE = {
    "matrix_id": "free_surface_wp7_cut_stability_v2",
    "closure_state": "BLOCKED_BY_FROZEN_PROSPECTIVE_EVIDENCE",
    "test": (
        "FreeSurfaceCutStability."
        "ConnectedDisconnectedAndRootlessFeaturesReportTopologyPolicy"
    ),
    "classification": "PROSPECTIVE_IN_FROZEN_PARENT",
    "parent_matrix_modified": False,
}
EXPECTED_PREDECESSOR_STATUS = {
    "matrix_id": "free_surface_wp3_wp7_symmetric_nitsche_prerequisite_v1",
    "implementation_source_commit": (
        "f127ce715f5d9042af3fa409d197667bc289e03f"
    ),
    "record_scope": "historical_source_commit_only",
    "current_source_requalified": False,
    "diagnostic": (
        "The active-cell topology source commit changes byte-locked "
        "aggregation and Physics test sources, so the predecessor joint "
        "prerequisite is not inherited by the current source."
    ),
}
EXPECTED_PARENT_SHA256 = {
    "Documentation/free_surface_boundary_unfitted_audit_20260720.md": (
        "f3d39efdafaa16cdb9dabc5b06626906eb05f5a78b1c87583347dcc008473652"
    ),
    "Documentation/free_surface_wp7_combined_p1_method.md": (
        "85ba3d61f50b67a4d719efff6b760323fbe3fcd3124b06e28b67ee6230ca1ff9"
    ),
    "tests/cases/fluid/free_surface_wp7_cut_stability_qualification_matrix.json": (
        "a49cadbcbe1b56bf69e4520a5281fc942ac4dc9de82da4e3bdaa083d6334ab1f"
    ),
    "tests/cases/fluid/run_free_surface_wp7_cut_stability_qualification.py": (
        "30c59eb725ba88b87447b935a915c6561b9ae25f082d93052d5cb9be4e337f6f"
    ),
    (
        "tests/cases/fluid/"
        "free_surface_wp3_wp7_nitsche_coercivity_qualification_matrix.json"
    ): "a75bbec8efe800f049375f190c07a121b3e365098da783b43ec1ba9df9610589",
    (
        "tests/cases/fluid/"
        "run_free_surface_wp3_wp7_nitsche_coercivity_qualification.py"
    ): "353c49c10881fd13acececb80cdf000c70abf7937f10021a2816d04d90bb9181",
}
EXPECTED_SOURCE_SHA256 = {
    "Code/Source/solver/FE/Constraints/SmallCutAggregationConstraint.h": (
        "a6050cb586cbc9a036d044413843b90b4bc4bb233519ca96dc7b10028da3af92"
    ),
    "Code/Source/solver/FE/Constraints/SmallCutAggregationConstraint.cpp": (
        "52e64189329aaea76bc05d2ab01675c097fdb1fde5590bfe875e2ca292da00a7"
    ),
    (
        "Code/Source/solver/FE/Tests/Unit/Constraints/"
        "test_SmallCutAggregationConstraint.cpp"
    ): "fac8480df9ffa360b323623afa2c04829964546c1f9fa84b19fd5db295ff5187",
    (
        "Code/Source/solver/FE/Tests/Unit/Constraints/"
        "test_SmallCutAggregationConstraintMPI.cpp"
    ): "e7172cb5f4ec57e9dbc9720619bc9842cf1b8783fffc96f8d8a79b4d2797dc82",
    (
        "Code/Source/solver/Physics/Tests/Unit/"
        "test_FreeSurfaceCutStability.cpp"
    ): "905c6bc3aec31ea805863732464c2b23cd82c72c8e45d6c29563b9c22d7e9cca",
}
EXPECTED_GROUPS = {
    "active_cell_report_serial": (
        "constraints",
        1,
        (
            "SmallCutAggregationConstraint."
            "CompletedRefreshReportSeparatesRootedAndRootlessCandidates",
        ),
    ),
    "active_cell_canonicalization_mpi_2": (
        "constraints_mpi",
        2,
        (
            "SmallCutAggregationConstraintMPI."
            "FullOverlapOwnerNonCandidateImportsCanonicalGhostRoot",
            "SmallCutAggregationConstraintMPI."
            "RootlessFeatureTelemetryDoesNotSumReplicatedProviders",
        ),
    ),
    "active_cell_production_path_serial": (
        "physics",
        1,
        (
            "FreeSurfaceCutStability."
            "ConnectedDisconnectedAndRootlessFeaturesReportTopologyPolicy",
        ),
    ),
}
EXPECTED_QUANTITATIVE = (
    ("wp7_active_cell_topology_case_count", "integer", 3, 0.0),
    ("wp7_active_cell_topology_feature_count", "integer", 5, 0.0),
    ("wp7_active_cell_topology_rooted_feature_count", "integer", 4, 0.0),
    ("wp7_active_cell_topology_rootless_feature_count", "integer", 1, 0.0),
    (
        "wp7_active_cell_topology_rootless_retained_physical_volume",
        "real",
        1.0,
        1.0e-12,
    ),
    (
        "wp7_active_cell_topology_velocity_pressure_mismatch_count",
        "integer",
        0,
        0.0,
    ),
)


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=reject_duplicate_keys,
    )
    if not isinstance(value, dict):
        raise ValueError(f"JSON root is not an object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def safe_artifact_map(entries: Any, label: str) -> dict[str, str]:
    if not isinstance(entries, list):
        raise ValueError(f"{label} must be a list")
    result: dict[str, str] = {}
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {"path", "sha256"}:
            raise ValueError(f"{label} entry has unexpected keys")
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
            or re.fullmatch(r"[0-9a-f]{64}", digest) is None
        ):
            raise ValueError(f"{label} entry has an invalid digest")
        result[path] = digest
    return result


def validate_contract(matrix_path: Path) -> dict[str, Any]:
    if sha256_file(matrix_path) != EXPECTED_MATRIX_SHA256:
        raise ValueError("frozen active-cell topology matrix bytes changed")
    matrix = read_json(matrix_path)
    if (
        matrix.get("schema_version") != 1
        or matrix.get("matrix_id") != EXPECTED_MATRIX_ID
        or matrix.get("status") != "FROZEN_BEFORE_EXECUTION"
        or matrix.get("work_package") != "WP-7"
        or matrix.get("findings") != ["FSR-07"]
    ):
        raise ValueError("active-cell topology matrix identity changed")
    if matrix.get("implementation_source_commit") != EXPECTED_SOURCE_COMMIT:
        raise ValueError("implementation source commit changed")
    if matrix.get("implementation_source_tree") != EXPECTED_SOURCE_TREE:
        raise ValueError("implementation source tree changed")
    if matrix.get("qualification_scope") != EXPECTED_SCOPE:
        raise ValueError("qualification scope changed")
    if matrix.get("closure_request_policy") != EXPECTED_CLAIM_POLICY:
        raise ValueError("closure-request policy changed")
    if matrix.get("qualification_disposition") != EXPECTED_DISPOSITION:
        raise ValueError("qualification disposition changed")
    if matrix.get("open_outcomes") != EXPECTED_OPEN_OUTCOMES:
        raise ValueError("open outcomes changed")
    if matrix.get("parent_state") != EXPECTED_PARENT_STATE:
        raise ValueError("frozen parent state changed")
    if (
        matrix.get("predecessor_qualification_status")
        != EXPECTED_PREDECESSOR_STATUS
    ):
        raise ValueError("predecessor qualification status changed")
    if (
        safe_artifact_map(matrix.get("parent_artifacts"), "parent artifacts")
        != EXPECTED_PARENT_SHA256
    ):
        raise ValueError("parent artifact inventory changed")
    if (
        safe_artifact_map(
            matrix.get("implementation_sources"),
            "implementation sources",
        )
        != EXPECTED_SOURCE_SHA256
    ):
        raise ValueError("implementation source inventory changed")

    groups = matrix.get("groups")
    if not isinstance(groups, list) or [
        group.get("id") for group in groups
    ] != list(EXPECTED_GROUPS):
        raise ValueError("qualification group order changed")
    distinct_tests: set[str] = set()
    for group in groups:
        expected_binary, expected_ranks, expected_tests = EXPECTED_GROUPS[
            group["id"]
        ]
        tests = group.get("tests")
        if (
            group.get("binary") != expected_binary
            or group.get("mpi_ranks") != expected_ranks
            or group.get("gtest_output_copies") != expected_ranks
            or tuple(tests if isinstance(tests, list) else []) != expected_tests
        ):
            raise ValueError(f"qualification group changed: {group['id']}")
        execution = group.get("execution")
        if (
            not isinstance(execution, dict)
            or not isinstance(execution.get("wall_time_seconds"), int)
            or execution["wall_time_seconds"] <= 0
            or not isinstance(execution.get("memory_mib"), int)
            or execution["memory_mib"] <= 0
            or not isinstance(execution.get("output_mib"), int)
            or execution["output_mib"] <= 0
        ):
            raise ValueError(f"invalid execution budget: {group['id']}")
        for test in expected_tests:
            if test in distinct_tests:
                raise ValueError(f"duplicate test across groups: {test}")
            distinct_tests.add(test)

    evidence = matrix.get("quantitative_evidence")
    if not isinstance(evidence, list) or len(evidence) != len(
        EXPECTED_QUANTITATIVE
    ):
        raise ValueError("quantitative evidence inventory changed")
    observed_quantitative = []
    expected_test = EXPECTED_GROUPS[
        "active_cell_production_path_serial"
    ][2][0]
    for entry in evidence:
        if (
            not isinstance(entry, dict)
            or entry.get("test") != expected_test
            or entry.get("relation") != "equal"
        ):
            raise ValueError("quantitative evidence declaration changed")
        observed_quantitative.append(
            (
                entry.get("property"),
                entry.get("type"),
                entry.get("threshold"),
                entry.get("absolute_tolerance", 0.0),
            )
        )
    if tuple(observed_quantitative) != EXPECTED_QUANTITATIVE:
        raise ValueError("quantitative evidence thresholds changed")

    guards = matrix.get("resource_guard")
    if (
        not isinstance(guards, dict)
        or guards.get("minimum_available_memory_mib") != 2048
        or guards.get("minimum_available_storage_mib") != 4096
        or guards.get("poll_interval_seconds") != 0.1
    ):
        raise ValueError("resource guard changed")
    build_contract = matrix.get("build_contract")
    if (
        not isinstance(build_contract, dict)
        or build_contract.get("generator") != "Unix Makefiles"
        or build_contract.get("parallel") != 1
        or build_contract.get("maximum_build_storage_mib") != 6144
        or build_contract.get("cmake") != "/usr/bin/cmake"
        or build_contract.get("cxx_compiler") != "/usr/bin/c++"
        or build_contract.get("mpi_cxx_compiler") != "/usr/bin/mpicxx"
        or build_contract.get("mpi_launcher") != "/usr/bin/mpiexec"
    ):
        raise ValueError("controlled build contract changed")
    configurations = build_contract.get("configurations")
    if (
        not isinstance(configurations, list)
        or [entry.get("id") for entry in configurations]
        != ["constraints", "physics"]
    ):
        raise ValueError("controlled build configurations changed")
    built_binary_keys: set[str] = set()
    for configuration in configurations:
        source_home = configuration.get("source_home")
        build_directory = configuration.get("build_directory")
        options = configuration.get("cmake_options")
        targets = configuration.get("targets")
        binaries = configuration.get("binaries")
        if (
            not isinstance(source_home, str)
            or Path(source_home).is_absolute()
            or ".." in Path(source_home).parts
            or not isinstance(build_directory, str)
            or Path(build_directory).is_absolute()
            or len(Path(build_directory).parts) != 1
            or not isinstance(options, list)
            or not options
            or any(
                not isinstance(option, str) or not option.startswith("-D")
                for option in options
            )
            or not isinstance(targets, list)
            or not targets
            or any(
                not isinstance(target, str)
                or re.fullmatch(r"[A-Za-z0-9_]+", target) is None
                for target in targets
            )
            or not isinstance(binaries, dict)
            or not binaries
        ):
            raise ValueError("controlled build configuration is unsafe")
        for key, relative_binary in binaries.items():
            if (
                key in built_binary_keys
                or not isinstance(relative_binary, str)
                or Path(relative_binary).is_absolute()
                or len(Path(relative_binary).parts) != 1
            ):
                raise ValueError("controlled binary declaration is unsafe")
            built_binary_keys.add(key)
    if built_binary_keys != {"constraints", "constraints_mpi", "physics"}:
        raise ValueError("controlled build binary inventory changed")
    if matrix.get("binary_keys") != [
        "constraints",
        "constraints_mpi",
        "physics",
    ]:
        raise ValueError("binary inventory changed")
    if matrix.get("prospective_tests") != []:
        raise ValueError("child matrix cannot contain prospective tests")
    gates = matrix.get("gates")
    if gates != {
        "expected_group_count": 3,
        "expected_distinct_test_count": 4,
        "expected_quantitative_evidence_count": 6,
        "expected_failures": 0,
        "expected_errors": 0,
        "expected_disabled": 0,
        "expected_skipped": 0,
    }:
        raise ValueError("qualification gates changed")
    return matrix


def git_bytes(*arguments: str) -> bytes:
    return subprocess.check_output(
        ["git", "-C", str(REPOSITORY_ROOT), *arguments],
        stderr=subprocess.PIPE,
    )


def validate_locked_sources(
    matrix: dict[str, Any], *, require_clean_detached: bool = False
) -> dict[str, Any]:
    top_level = Path(
        git_bytes("rev-parse", "--show-toplevel").decode().strip()
    ).resolve()
    if top_level != REPOSITORY_ROOT:
        raise ValueError("runner repository root does not match Git")
    source_status = git_bytes(
        "status", "--porcelain=v1", "--untracked-files=all"
    )
    source_clean = not source_status
    if require_clean_detached and not source_clean:
        raise ValueError("qualification requires a clean source worktree")
    source_detached = (
        subprocess.run(
            [
                "git",
                "-C",
                str(REPOSITORY_ROOT),
                "symbolic-ref",
                "-q",
                "HEAD",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        ).returncode
        != 0
    )
    if require_clean_detached and not source_detached:
        raise ValueError("qualification requires a detached source worktree")
    definition_status = (
        git_bytes(
            "diff",
            "--name-status",
            EXPECTED_SOURCE_COMMIT,
            "HEAD",
        )
        .decode()
        .splitlines()
    )
    expected_definition_status = {
        f"A\t{path}" for path in EXPECTED_DEFINITION_PATHS
    }
    definition_only_descendant = (
        set(definition_status) == expected_definition_status
        and len(definition_status) == len(expected_definition_status)
        and git_bytes(
            "rev-list",
            "--count",
            f"{EXPECTED_SOURCE_COMMIT}..HEAD",
        )
        .decode()
        .strip()
        == "1"
        and git_bytes("rev-parse", "HEAD^").decode().strip()
        == EXPECTED_SOURCE_COMMIT
    )
    if require_clean_detached and not definition_only_descendant:
        raise ValueError(
            "execution HEAD must add only the frozen matrix and runner "
            "directly above the implementation source commit"
        )
    subprocess.run(
        [
            "git",
            "-C",
            str(REPOSITORY_ROOT),
            "cat-file",
            "-e",
            f"{EXPECTED_SOURCE_COMMIT}^{{commit}}",
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    if (
        subprocess.run(
            [
                "git",
                "-C",
                str(REPOSITORY_ROOT),
                "merge-base",
                "--is-ancestor",
                EXPECTED_SOURCE_COMMIT,
                "HEAD",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            check=False,
        ).returncode
        != 0
    ):
        raise ValueError("implementation source commit is not an ancestor")
    implementation_tree = (
        git_bytes("rev-parse", f"{EXPECTED_SOURCE_COMMIT}^{{tree}}")
        .decode()
        .strip()
    )
    if implementation_tree != EXPECTED_SOURCE_TREE:
        raise ValueError("implementation source tree changed")

    locked = {
        **safe_artifact_map(
            matrix["parent_artifacts"], "parent artifacts"
        ),
        **safe_artifact_map(
            matrix["implementation_sources"], "implementation sources"
        ),
    }
    records = []
    for relative_path, expected_digest in sorted(locked.items()):
        path = REPOSITORY_ROOT / relative_path
        if path.is_symlink() or not path.is_file():
            raise ValueError(
                f"locked artifact is missing or a symlink: {relative_path}"
            )
        working_digest = sha256_file(path)
        commit_bytes = git_bytes(
            "show", f"{EXPECTED_SOURCE_COMMIT}:{relative_path}"
        )
        commit_digest = sha256_bytes(commit_bytes)
        if (
            working_digest != expected_digest
            or commit_digest != expected_digest
        ):
            raise ValueError(f"locked artifact digest changed: {relative_path}")
        records.append(
            {
                "path": relative_path,
                "sha256": expected_digest,
                "working_tree_matches": True,
                "implementation_commit_matches": True,
            }
        )
    return {
        "repository_root": str(REPOSITORY_ROOT),
        "head_commit": git_bytes("rev-parse", "HEAD").decode().strip(),
        "head_tree": git_bytes("rev-parse", "HEAD^{tree}").decode().strip(),
        "implementation_source_commit": EXPECTED_SOURCE_COMMIT,
        "implementation_source_tree": implementation_tree,
        "implementation_source_is_ancestor": True,
        "source_worktree_clean": source_clean,
        "source_head_detached": source_detached,
        "definition_only_descendant": definition_only_descendant,
        "definition_diff": definition_status,
        "locked_artifact_count": len(records),
        "locked_artifacts": records,
    }


def memory_information() -> dict[str, int]:
    values: dict[str, int] = {}
    with Path("/proc/meminfo").open(encoding="utf-8") as source:
        for line in source:
            key, raw_value = line.split(":", 1)
            fields = raw_value.strip().split()
            if fields and fields[0].isdigit():
                values[key] = int(fields[0]) // 1024
    required = {"MemTotal", "MemAvailable", "SwapTotal", "SwapFree"}
    if not required.issubset(values):
        raise RuntimeError("required memory telemetry is unavailable")
    return {key: values[key] for key in sorted(required)}


def resource_snapshot(storage_path: Path) -> dict[str, Any]:
    memory = memory_information()
    storage = shutil.disk_usage(storage_path)
    return {
        "timestamp_utc": utc_now(),
        "memory_total_mib": memory["MemTotal"],
        "memory_available_mib": memory["MemAvailable"],
        "swap_total_mib": memory["SwapTotal"],
        "swap_free_mib": memory["SwapFree"],
        "storage_total_mib": storage.total // (1024 * 1024),
        "storage_free_mib": storage.free // (1024 * 1024),
    }


def resource_snapshot_passes(
    snapshot: dict[str, Any], guard: dict[str, Any]
) -> bool:
    return (
        snapshot["memory_available_mib"]
        >= guard["minimum_available_memory_mib"]
        and snapshot["storage_free_mib"]
        >= guard["minimum_available_storage_mib"]
    )


def listed_gtests(
    binary: Path, environment: dict[str, str]
) -> set[str]:
    result = subprocess.run(
        [str(binary), "--gtest_list_tests"],
        check=True,
        cwd=REPOSITORY_ROOT,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=30,
        preexec_fn=command_preexec(2048),
    )
    suite = ""
    tests: set[str] = set()
    for line in result.stdout.splitlines():
        if line and not line[0].isspace():
            suite = line.split("#", 1)[0].strip().removesuffix(".")
            continue
        test = line.split("#", 1)[0].strip()
        if suite and test:
            tests.add(f"{suite}.{test}")
    return tests


def validate_binaries(
    matrix: dict[str, Any],
    binaries: dict[str, Path],
    environment: dict[str, str],
) -> dict[str, Any]:
    if set(binaries) != set(matrix["binary_keys"]):
        raise ValueError("binary arguments do not match the matrix")
    discovered: dict[str, set[str]] = {}
    records = []
    for key in matrix["binary_keys"]:
        binary = binaries[key]
        if (
            binary.is_symlink()
            or not binary.is_file()
            or not os.access(binary, os.X_OK)
        ):
            raise ValueError(f"binary is not an executable regular file: {binary}")
        discovered[key] = listed_gtests(binary, environment)
        linked = subprocess.run(
            ["/usr/bin/ldd", str(binary)],
            check=True,
            cwd=REPOSITORY_ROOT,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=30,
            preexec_fn=command_preexec(2048),
        )
        linked_library_provenance = []
        for line in linked.stdout.splitlines():
            match = re.search(r"=>\s+(/[^\s]+)", line)
            if match is None:
                match = re.match(r"\s*(/[^\s]+)", line)
            if match is None:
                continue
            library = Path(match.group(1))
            resolved_library = library.resolve()
            if not resolved_library.is_file():
                raise ValueError(
                    f"linked library is not a regular file: {library}"
                )
            linked_library_provenance.append(
                {
                    "path": str(library),
                    "resolved_path": str(resolved_library),
                    "sha256": sha256_file(resolved_library),
                }
            )
        records.append(
            {
                "key": key,
                "path": str(binary),
                "sha256": sha256_file(binary),
                "listed_test_count": len(discovered[key]),
                "linked_libraries": linked.stdout.splitlines(),
                "linked_library_provenance": (
                    linked_library_provenance
                ),
            }
        )
    for group in matrix["groups"]:
        missing = sorted(
            set(group["tests"]) - discovered[group["binary"]]
        )
        if missing:
            raise ValueError(
                f"group {group['id']} has missing compiled tests: {missing}"
            )
    return {"binaries": records, "all_required_tests_discovered": True}


def mpi_stack_is_compatible(
    binary_record: dict[str, Any],
    launcher_record: dict[str, Any],
    build_record: dict[str, Any],
) -> bool:
    by_key = {
        entry["key"]: entry for entry in binary_record["binaries"]
    }
    mpi_links = "\n".join(
        by_key["constraints_mpi"]["linked_libraries"]
    ).lower()
    launcher_version = "\n".join(
        launcher_record["version_output"]
    ).lower()
    mpi_compiler_provenance = next(
        (
            configuration["cache"]["mpi_cxx_compiler_provenance"]
            for configuration in build_record["configurations"]
            if configuration["cache"] is not None
            and "mpi_cxx_compiler_provenance" in configuration["cache"]
        ),
        None,
    )
    if mpi_compiler_provenance is None:
        return False
    compiler_version = "\n".join(
        mpi_compiler_provenance["version_output"]
    ).lower()
    launcher_match = re.search(
        r"(?:open mpi|openrte)[^0-9]*([0-9]+\.[0-9]+\.[0-9]+)",
        launcher_version,
    )
    compiler_match = re.search(
        r"open mpi[^0-9]*([0-9]+\.[0-9]+\.[0-9]+)",
        compiler_version,
    )
    return (
        "libmpi" in mpi_links
        and launcher_record["path"] == "/usr/bin/mpiexec"
        and mpi_compiler_provenance["path"] == "/usr/bin/mpicxx"
        and launcher_match is not None
        and compiler_match is not None
        and launcher_match.group(1) == compiler_match.group(1)
        and (
            "open mpi" in launcher_version
            or "openrte" in launcher_version
        )
    )


def executable_provenance(
    path: Path,
    version_arguments: list[str],
    environment: dict[str, str],
) -> dict[str, Any]:
    requested_path = path.absolute()
    resolved_path = requested_path.resolve()
    if (
        not requested_path.is_file()
        or not resolved_path.is_file()
        or not os.access(requested_path, os.X_OK)
    ):
        raise ValueError(
            f"executable does not resolve to a regular file: {requested_path}"
        )
    version = subprocess.run(
        [str(requested_path), *version_arguments],
        check=True,
        cwd=REPOSITORY_ROOT,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=30,
        preexec_fn=command_preexec(2048),
    )
    return {
        "path": str(requested_path),
        "resolved_path": str(resolved_path),
        "symlink": requested_path.is_symlink(),
        "sha256": sha256_file(resolved_path),
        "version_output": version.stdout.splitlines(),
    }


def path_is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def cache_value(cache_path: Path, key: str) -> str | None:
    prefix = f"{key}:"
    for line in cache_path.read_text(encoding="utf-8").splitlines():
        if line.startswith(prefix):
            return line.split("=", 1)[1] if "=" in line else None
    return None


def run_controlled_builds(
    matrix: dict[str, Any],
    cmake: Path,
    build_root: Path,
    artifact_root: Path,
    environment: dict[str, str],
) -> tuple[dict[str, Path], dict[str, Any]]:
    contract = matrix["build_contract"]
    if build_root.exists():
        raise ValueError(f"refusing to reuse build root: {build_root}")
    if (
        not build_root.parent.is_dir()
        or build_root.parent.is_symlink()
        or path_is_within(build_root, REPOSITORY_ROOT)
        or path_is_within(build_root, artifact_root)
        or path_is_within(artifact_root, build_root)
    ):
        raise ValueError("controlled build root is unsafe")
    build_root.mkdir()
    initial_storage = resource_snapshot(build_root)["storage_free_mib"]
    records = []
    binaries: dict[str, Path] = {}
    for configuration in contract["configurations"]:
        configuration_id = configuration["id"]
        source_home = (
            REPOSITORY_ROOT / configuration["source_home"]
        ).resolve()
        build_directory = (
            build_root / configuration["build_directory"]
        ).resolve()
        configure_artifacts = (
            artifact_root / "builds" / configuration_id / "configure"
        )
        build_artifacts = (
            artifact_root / "builds" / configuration_id / "build"
        )
        configure_artifacts.mkdir(parents=True)
        build_artifacts.mkdir()
        configure_command = [
            str(cmake),
            "-S",
            str(source_home),
            "-B",
            str(build_directory),
            "-G",
            contract["generator"],
            *configuration["cmake_options"],
        ]
        configure_result = monitored_command(
            configure_command,
            configure_artifacts,
            contract["configure_execution"]["wall_time_seconds"],
            contract["configure_execution"]["memory_mib"],
            contract["configure_execution"]["output_mib"],
            matrix["resource_guard"],
            cwd=REPOSITORY_ROOT,
            environment=environment,
            maximum_storage_consumption_mib=contract[
                "maximum_build_storage_mib"
            ],
            storage_monitor_path=build_root,
            storage_baseline_free_mib=initial_storage,
        )
        build_result = None
        if configure_result["outcome"] == "PASS":
            build_command = [
                str(cmake),
                "--build",
                str(build_directory),
                "--target",
                *configuration["targets"],
                "--parallel",
                str(contract["parallel"]),
            ]
            build_result = monitored_command(
                build_command,
                build_artifacts,
                contract["build_execution"]["wall_time_seconds"],
                contract["build_execution"]["memory_mib"],
                contract["build_execution"]["output_mib"],
                matrix["resource_guard"],
                cwd=REPOSITORY_ROOT,
                environment=environment,
                maximum_storage_consumption_mib=contract[
                    "maximum_build_storage_mib"
                ],
                storage_monitor_path=build_root,
                storage_baseline_free_mib=initial_storage,
            )
        cache_path = build_directory / "CMakeCache.txt"
        cache_record = None
        configuration_passed = (
            configure_result["outcome"] == "PASS"
            and build_result is not None
            and build_result["outcome"] == "PASS"
            and cache_path.is_file()
            and not cache_path.is_symlink()
        )
        if configuration_passed:
            cache_home = cache_value(cache_path, "CMAKE_HOME_DIRECTORY")
            cache_generator = cache_value(cache_path, "CMAKE_GENERATOR")
            configuration_passed = (
                cache_home is not None
                and Path(cache_home).resolve() == source_home
                and cache_generator == contract["generator"]
                and cache_value(cache_path, "CMAKE_CXX_COMPILER")
                == contract["cxx_compiler"]
                and cache_value(cache_path, "MPI_CXX_COMPILER")
                == contract["mpi_cxx_compiler"]
            )
            cache_record = {
                "path": str(cache_path),
                "sha256": sha256_file(cache_path),
                "cmake_home_directory": cache_home,
                "cmake_generator": cache_generator,
                "cmake_build_type": cache_value(
                    cache_path, "CMAKE_BUILD_TYPE"
                ),
                "cxx_compiler": cache_value(
                    cache_path, "CMAKE_CXX_COMPILER"
                ),
                "mpi_cxx_compiler": cache_value(
                    cache_path, "MPI_CXX_COMPILER"
                ),
            }
            if cache_record["cxx_compiler"] is not None:
                cache_record["cxx_compiler_provenance"] = (
                    executable_provenance(
                        Path(cache_record["cxx_compiler"]),
                        ["--version"],
                        environment,
                    )
                )
            if cache_record["mpi_cxx_compiler"] is not None:
                cache_record["mpi_cxx_compiler_provenance"] = (
                    executable_provenance(
                        Path(cache_record["mpi_cxx_compiler"]),
                        ["--showme:version"],
                        environment,
                    )
                )
        if configuration_passed:
            for key, relative_binary in configuration["binaries"].items():
                binary = build_directory / relative_binary
                if (
                    binary.is_symlink()
                    or not binary.is_file()
                    or not os.access(binary, os.X_OK)
                ):
                    configuration_passed = False
                    break
                binaries[key] = binary.resolve()
        records.append(
            {
                "configuration_id": configuration_id,
                "source_home": str(source_home),
                "build_directory": str(build_directory),
                "configure": configure_result,
                "build": build_result,
                "cache": cache_record,
                "outcome": "PASS" if configuration_passed else "FAIL",
            }
        )
        if not configuration_passed:
            break
    final_storage = resource_snapshot(build_root)["storage_free_mib"]
    storage_consumption = max(0, initial_storage - final_storage)
    passed = (
        len(records) == len(contract["configurations"])
        and all(record["outcome"] == "PASS" for record in records)
        and set(binaries) == set(matrix["binary_keys"])
        and storage_consumption <= contract["maximum_build_storage_mib"]
    )
    return binaries, {
        "mode": "clean_controlled_build",
        "build_root": str(build_root),
        "parallel": contract["parallel"],
        "maximum_build_storage_mib": contract[
            "maximum_build_storage_mib"
        ],
        "observed_build_storage_consumption_mib": storage_consumption,
        "configurations": records,
        "outcome": "PASS" if passed else "FAIL",
    }


def directory_size(path: Path) -> int:
    return sum(
        entry.stat().st_size
        for entry in path.rglob("*")
        if entry.is_file() and not entry.is_symlink()
    )


def session_processes(session_id: int) -> tuple[list[dict[str, int]], bool]:
    page_size = os.sysconf("SC_PAGE_SIZE")
    processes = []
    complete = True
    for stat_path in Path("/proc").glob("[0-9]*/stat"):
        try:
            value = stat_path.read_text(encoding="utf-8")
            fields = value[value.rfind(")") + 2 :].split()
            if len(fields) < 4 or int(fields[3]) != session_id:
                continue
            pid = int(stat_path.parent.name)
            statm = (stat_path.parent / "statm").read_text(
                encoding="utf-8"
            )
            resident_pages = int(statm.split()[1])
            processes.append(
                {
                    "pid": pid,
                    "rss_mib": resident_pages * page_size // (1024 * 1024),
                }
            )
        except FileNotFoundError:
            continue
        except (IndexError, PermissionError, ValueError):
            complete = False
    return sorted(processes, key=lambda item: item["pid"]), complete


def terminate_session(session_id: int) -> dict[str, Any]:
    signalled: set[int] = set()
    complete = True
    for termination_signal in (signal.SIGTERM, signal.SIGKILL):
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            processes, sampled = session_processes(session_id)
            complete = complete and sampled
            remaining = [
                item["pid"]
                for item in processes
                if item["pid"] != os.getpid()
            ]
            if not remaining:
                return {
                    "signalled_pids": sorted(signalled),
                    "sampling_complete": complete,
                    "all_session_processes_terminated": True,
                }
            for pid in remaining:
                try:
                    os.kill(pid, termination_signal)
                    signalled.add(pid)
                except ProcessLookupError:
                    pass
                except PermissionError:
                    complete = False
            time.sleep(0.05)
    remaining, sampled = session_processes(session_id)
    complete = complete and sampled
    return {
        "signalled_pids": sorted(signalled),
        "sampling_complete": complete,
        "remaining_pids": [item["pid"] for item in remaining],
        "all_session_processes_terminated": not remaining,
    }


def controlled_environment() -> dict[str, str]:
    allowed = (
        "PATH",
        "LD_LIBRARY_PATH",
        "LIBRARY_PATH",
        "CPATH",
        "CMAKE_PREFIX_PATH",
        "PKG_CONFIG_PATH",
        "CC",
        "CXX",
        "TMPDIR",
        "USER",
        "LOGNAME",
    )
    environment = {
        key: os.environ[key] for key in allowed if key in os.environ
    }
    environment["LC_ALL"] = "C"
    environment["LANG"] = "C"
    environment.pop("BASH_ENV", None)
    environment.pop("ENV", None)
    return environment


def command_preexec(memory_limit_mib: int) -> Any:
    memory_limit_bytes = memory_limit_mib * 1024 * 1024

    def prepare() -> None:
        os.setsid()
        resource.setrlimit(
            resource.RLIMIT_AS,
            (memory_limit_bytes, memory_limit_bytes),
        )

    return prepare


def monitored_command(
    command: list[str],
    group_directory: Path,
    timeout_seconds: int,
    memory_limit_mib: int,
    output_limit_mib: int,
    guard: dict[str, Any],
    *,
    cwd: Path,
    environment: dict[str, str],
    maximum_storage_consumption_mib: int | None = None,
    storage_monitor_path: Path | None = None,
    storage_baseline_free_mib: int | None = None,
) -> dict[str, Any]:
    log_path = group_directory / "run.log"
    storage_path = storage_monitor_path or group_directory
    samples = [resource_snapshot(storage_path)]
    required_memory_mib = (
        guard["minimum_available_memory_mib"] + memory_limit_mib
    )
    if (
        not resource_snapshot_passes(samples[-1], guard)
        or samples[-1]["memory_available_mib"] < required_memory_mib
    ):
        return {
            "command": command,
            "return_code": None,
            "termination": "resource_guard_preflight",
            "required_preflight_memory_mib": required_memory_mib,
            "resource_samples": samples,
            "outcome": "FAIL_INFRASTRUCTURE",
        }
    initial_memory_available_mib = samples[-1]["memory_available_mib"]
    initial_storage_free_mib = (
        samples[-1]["storage_free_mib"]
        if storage_baseline_free_mib is None
        else storage_baseline_free_mib
    )
    started = time.monotonic()
    termination = None
    process: subprocess.Popen[bytes] | None = None
    session_samples: list[dict[str, Any]] = []
    cleanup = {
        "signalled_pids": [],
        "sampling_complete": True,
        "all_session_processes_terminated": True,
    }
    monitor_error = None
    return_code = None
    try:
        with log_path.open("wb") as log:
            process = subprocess.Popen(
                command,
                cwd=cwd,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                preexec_fn=command_preexec(memory_limit_mib),
            )
            while True:
                processes, complete = session_processes(process.pid)
                session_samples.append(
                    {
                        "timestamp_utc": utc_now(),
                        "sampling_complete": complete,
                        "processes": processes,
                        "aggregate_rss_mib": sum(
                            item["rss_mib"] for item in processes
                        ),
                    }
                )
                if not complete:
                    termination = "process_session_sampling_incomplete"
                    break
                sample = resource_snapshot(storage_path)
                samples.append(sample)
                elapsed = time.monotonic() - started
                output_size = directory_size(group_directory)
                storage_consumption = max(
                    0,
                    initial_storage_free_mib - sample["storage_free_mib"],
                )
                if not resource_snapshot_passes(sample, guard):
                    termination = "resource_guard_runtime"
                    break
                if session_samples[-1]["aggregate_rss_mib"] > memory_limit_mib:
                    termination = "memory_budget"
                    break
                if output_size > output_limit_mib * 1024 * 1024:
                    termination = "output_budget"
                    break
                if (
                    maximum_storage_consumption_mib is not None
                    and storage_consumption
                    > maximum_storage_consumption_mib
                ):
                    termination = "storage_budget"
                    break
                if elapsed > timeout_seconds:
                    termination = "wall_time_budget"
                    break
                if process.poll() is not None:
                    break
                time.sleep(guard["poll_interval_seconds"])
            if termination is not None and process.poll() is None:
                cleanup = terminate_session(process.pid)
            return_code = process.wait(timeout=10)
    except Exception as error:
        monitor_error = f"{type(error).__name__}: {error}"
        termination = termination or "monitor_exception"
    finally:
        if process is not None:
            if process.poll() is None:
                cleanup = terminate_session(process.pid)
                try:
                    return_code = process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    termination = "launcher_reap_failure"
            lingering, complete = session_processes(process.pid)
            if lingering:
                cleanup = terminate_session(process.pid)
            cleanup["sampling_complete"] = (
                cleanup.get("sampling_complete", True) and complete
            )
            cleanup["all_session_processes_terminated"] = (
                cleanup.get("all_session_processes_terminated", True)
                and not session_processes(process.pid)[0]
            )
    samples.append(resource_snapshot(storage_path))
    elapsed = time.monotonic() - started
    final_output_bytes = directory_size(group_directory)
    final_storage_consumption_mib = max(
        0,
        initial_storage_free_mib - samples[-1]["storage_free_mib"],
    )
    if termination is None and elapsed > timeout_seconds:
        termination = "wall_time_budget_final"
    if (
        termination is None
        and final_output_bytes > output_limit_mib * 1024 * 1024
    ):
        termination = "output_budget_final"
    if (
        termination is None
        and maximum_storage_consumption_mib is not None
        and final_storage_consumption_mib > maximum_storage_consumption_mib
    ):
        termination = "storage_budget_final"
    if termination is None and not cleanup["all_session_processes_terminated"]:
        termination = "lingering_session_processes"
    if termination is None and not cleanup["sampling_complete"]:
        termination = "process_session_sampling_incomplete"
    maximum_session_rss_mib = max(
        (
            sample["aggregate_rss_mib"]
            for sample in session_samples
        ),
        default=0,
    )
    outcome = (
        "PASS"
        if (
            return_code == 0
            and termination is None
            and monitor_error is None
            and maximum_session_rss_mib <= memory_limit_mib
        )
        else "FAIL_INFRASTRUCTURE"
    )
    return {
        "command": command,
        "cwd": str(cwd),
        "controlled_environment": environment,
        "storage_monitor_path": str(storage_path),
        "storage_baseline_free_mib": initial_storage_free_mib,
        "return_code": return_code,
        "termination": termination,
        "monitor_error": monitor_error,
        "elapsed_seconds": elapsed,
        "final_output_bytes": final_output_bytes,
        "final_storage_consumption_mib": final_storage_consumption_mib,
        "maximum_session_rss_mib": maximum_session_rss_mib,
        "maximum_session_process_count": max(
            (len(sample["processes"]) for sample in session_samples),
            default=0,
        ),
        "session_samples": session_samples,
        "session_cleanup": cleanup,
        "minimum_memory_available_mib": min(
            sample["memory_available_mib"] for sample in samples
        ),
        "minimum_swap_free_mib": min(
            sample["swap_free_mib"] for sample in samples
        ),
        "minimum_storage_free_mib": min(
            sample["storage_free_mib"] for sample in samples
        ),
        "maximum_observed_system_memory_consumption_mib": max(
            0,
            initial_memory_available_mib
            - min(sample["memory_available_mib"] for sample in samples),
        ),
        "resource_monitoring_outcome": (
            "PASS" if outcome == "PASS" else "FAIL"
        ),
        "resource_samples": samples,
        "outcome": outcome,
    }


def group_command(
    group: dict[str, Any],
    binary: Path,
    mpiexec: Path,
    group_directory: Path,
) -> list[str]:
    test_filter = "--gtest_filter=" + ":".join(group["tests"])
    common = [test_filter, "--gtest_color=no"]
    ranks = group["mpi_ranks"]
    if ranks == 1:
        return [
            str(binary),
            *common,
            (
                "--gtest_output=json:"
                f"{group_directory / 'gtest_rank0.json'}"
            ),
        ]
    output_prefix = shlex.quote(str(group_directory))
    shell_command = (
        "set -eu; umask 077; "
        'if [ -n "${OMPI_COMM_WORLD_RANK+x}" ]; then '
        'rank="${OMPI_COMM_WORLD_RANK}"; '
        'elif [ -n "${PMI_RANK+x}" ]; then rank="${PMI_RANK}"; '
        'elif [ -n "${PMIX_RANK+x}" ]; then rank="${PMIX_RANK}"; '
        "else exit 96; fi; "
        'case "${rank}" in *[!0-9]*|"") exit 97;; esac; '
        f'if [ "${{rank}}" -ge {ranks} ]; then exit 98; fi; '
        f"exec {shlex.quote(str(binary))} "
        f"{' '.join(shlex.quote(argument) for argument in common)} "
        f"--gtest_output=json:{output_prefix}/gtest_rank${{rank}}.json"
    )
    return [
        str(mpiexec),
        "-n",
        str(ranks),
        "/bin/bash",
        "-c",
        shell_command,
    ]


def flatten_gtest(document: dict[str, Any]) -> dict[str, dict[str, Any]]:
    flattened: dict[str, dict[str, Any]] = {}
    suites = document.get("testsuites")
    if not isinstance(suites, list):
        raise ValueError("gtest result has no test suites")
    for suite in suites:
        if not isinstance(suite, dict) or not isinstance(
            suite.get("name"), str
        ):
            raise ValueError("gtest suite is invalid")
        tests = suite.get("testsuite")
        if not isinstance(tests, list):
            raise ValueError("gtest suite has no test cases")
        for test in tests:
            if not isinstance(test, dict) or not isinstance(
                test.get("name"), str
            ):
                raise ValueError("gtest case is invalid")
            full_name = f"{suite['name']}.{test['name']}"
            if full_name in flattened:
                raise ValueError(f"duplicate gtest result: {full_name}")
            flattened[full_name] = test
    return flattened


def validate_gtest_result(
    result_path: Path, expected_tests: set[str]
) -> dict[str, Any]:
    document = read_json(result_path)
    flattened = flatten_gtest(document)
    counts = {
        key: document.get(key, 0)
        for key in ("tests", "failures", "disabled", "errors", "skipped")
    }
    observed_tests = set(flattened)
    clean_cases = all(
        test.get("status") == "RUN"
        and test.get("result") == "COMPLETED"
        and not test.get("failures")
        for test in flattened.values()
    )
    passed = (
        counts["tests"] == len(expected_tests)
        and all(counts[key] == 0 for key in counts if key != "tests")
        and observed_tests == expected_tests
        and clean_cases
    )
    return {
        "path": str(result_path),
        "counts": counts,
        "observed_tests": sorted(observed_tests),
        "expected_tests": sorted(expected_tests),
        "clean_cases": clean_cases,
        "outcome": "PASS" if passed else "FAIL_METHOD",
        "flattened": flattened,
    }


def run_group(
    group: dict[str, Any],
    binary: Path,
    mpiexec: Path,
    output_directory: Path,
    guard: dict[str, Any],
    environment: dict[str, str],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    group_directory = output_directory / "groups" / group["id"]
    group_directory.mkdir(parents=True)
    command = group_command(group, binary, mpiexec, group_directory)
    execution = monitored_command(
        command,
        group_directory,
        group["execution"]["wall_time_seconds"],
        group["execution"]["memory_mib"],
        group["execution"]["output_mib"],
        guard,
        cwd=REPOSITORY_ROOT,
        environment=environment,
    )
    parsed: list[dict[str, Any]] = []
    diagnostic = None
    expected_paths = [
        group_directory / f"gtest_rank{rank}.json"
        for rank in range(group["gtest_output_copies"])
    ]
    if execution["outcome"] == "PASS":
        try:
            parsed = [
                validate_gtest_result(path, set(group["tests"]))
                for path in expected_paths
            ]
        except (OSError, ValueError, json.JSONDecodeError) as error:
            diagnostic = f"gtest_result_invalid: {error}"
    if execution["outcome"] != "PASS":
        diagnostic = execution.get("termination") or "nonzero_exit"
    passed = (
        execution["outcome"] == "PASS"
        and len(parsed) == group["gtest_output_copies"]
        and all(result["outcome"] == "PASS" for result in parsed)
    )
    serializable_results = []
    flattened_results = []
    for rank, result in enumerate(parsed):
        flattened_results.append(result["flattened"])
        serializable_results.append(
            {key: value for key, value in result.items() if key != "flattened"}
        )
    record = {
        "group_id": group["id"],
        "binary": group["binary"],
        "mpi_ranks": group["mpi_ranks"],
        "expected_tests": group["tests"],
        "execution_budget": group["execution"],
        "execution": execution,
        "gtest_results": serializable_results,
        "diagnostic": diagnostic,
        "outcome": "PASS" if passed else "FAIL",
    }
    write_json(group_directory / "result.json", record)
    return record, flattened_results


def coerce_value(raw_value: Any, value_type: str) -> tuple[Any, str | None]:
    if value_type == "integer":
        if isinstance(raw_value, bool):
            return None, "property_type_mismatch"
        if isinstance(raw_value, int):
            return raw_value, None
        if isinstance(raw_value, str) and re.fullmatch(r"[+-]?[0-9]+", raw_value):
            return int(raw_value), None
        return None, "property_type_mismatch"
    if isinstance(raw_value, bool) or not isinstance(
        raw_value, (int, float, str)
    ):
        return None, "property_type_mismatch"
    try:
        value = float(raw_value)
    except (TypeError, ValueError, OverflowError):
        return None, "property_type_mismatch"
    if not math.isfinite(value):
        return None, "property_value_not_finite"
    return value, None


def evaluate_quantitative_evidence(
    matrix: dict[str, Any],
    flattened_by_test: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    checks = []
    for evidence in matrix["quantitative_evidence"]:
        test_result = flattened_by_test.get(evidence["test"])
        diagnostic = None
        if test_result is None:
            diagnostic = "test_result_missing"
            raw_value = None
        else:
            raw_value = test_result.get(evidence["property"])
            if evidence["property"] not in test_result:
                diagnostic = "property_missing"
        actual = None
        if diagnostic is None:
            actual, diagnostic = coerce_value(raw_value, evidence["type"])
        tolerance = evidence.get("absolute_tolerance", 0.0)
        passed = False
        if diagnostic is None:
            passed = abs(actual - evidence["threshold"]) <= tolerance
            if not passed:
                diagnostic = "relation_not_satisfied"
        checks.append(
            {
                **evidence,
                "raw_value": raw_value,
                "actual": actual,
                "diagnostic": diagnostic,
                "passed": passed,
            }
        )
    passed_count = sum(check["passed"] for check in checks)
    return {
        "declared_check_count": len(checks),
        "passed_check_count": passed_count,
        "checks": checks,
        "outcome": "PASS" if passed_count == len(checks) else "FAIL",
    }


def write_checksums(output_directory: Path) -> None:
    entries = []
    checksum_path = output_directory / "checksums.txt"
    for path in sorted(output_directory.rglob("*")):
        if path.is_symlink():
            raise RuntimeError(f"artifact path is a symlink: {path}")
        if path.is_file() and path != checksum_path:
            entries.append(
                f"{sha256_file(path)}  "
                f"{path.relative_to(output_directory).as_posix()}"
            )
    checksum_path.write_text(
        "\n".join(entries) + "\n",
        encoding="utf-8",
    )


def write_record(
    output_directory: Path,
    matrix: dict[str, Any],
    summary: dict[str, Any],
) -> None:
    group_rows = "\n".join(
        "| "
        + " | ".join(
            (
                group["group_id"],
                str(group["mpi_ranks"]),
                group["outcome"],
                str(
                    group["execution"].get(
                        "minimum_memory_available_mib", "not-run"
                    )
                ),
                str(
                    group["execution"].get(
                        "minimum_storage_free_mib", "not-run"
                    )
                ),
            )
        )
        + " |"
        for group in summary["groups"]
    )
    text = (
        "# WP-7 active-cell topology telemetry prerequisite record\n\n"
        f"- Matrix: `{matrix['matrix_id']}`\n"
        f"- Outcome: **{summary['outcome']}**\n"
        f"- Requested claim: `{summary['requested_claim']}`\n"
        f"- Execution source: `{summary['execution_source_commit']}`\n"
        f"- Implementation source: `{EXPECTED_SOURCE_COMMIT}`\n\n"
        "## Scope boundary\n\n"
        f"{EXPECTED_SCOPE}\n\n"
        "The record covers background-cell active topology, rooted/rootless "
        "classification, retained support volume, stable identities, and "
        "canonical MPI provider handling. It does not credit physical-phase "
        "component reconstruction, conservation, conditioning, convergence, "
        "solver-spread, node-crossing, or simulation-exit evidence.\n\n"
        "## Executed groups\n\n"
        "| Group | Ranks | Outcome | Minimum available memory MiB | "
        "Minimum free storage MiB |\n"
        "| --- | ---: | --- | ---: | ---: |\n"
        f"{group_rows}\n\n"
        "## Closure disposition\n\n"
        "- FSR-07: OPEN\n"
        "- WP-7: OPEN\n"
        "- Q1: OPEN\n"
        "- Joint WP-3/WP-7 Nitsche prerequisite on this source: "
        "REQUALIFICATION REQUIRED\n"
    )
    (output_directory / "record.md").write_text(text, encoding="utf-8")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument(
        "--requested-claim",
        default=EXPECTED_CLAIM_POLICY["accepted_claim"],
    )
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--cmake", type=Path, default=Path("/usr/bin/cmake"))
    parser.add_argument("--mpiexec", type=Path, default=Path("/usr/bin/mpiexec"))
    parser.add_argument("--build-root", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def mark_incomplete(
    staging_directory: Path,
    diagnostic: str,
    *,
    outcome: str = "FAIL",
) -> None:
    try:
        write_json(
            staging_directory / "state.json",
            {
                "state": "INCOMPLETE",
                "outcome": outcome,
                "diagnostic": diagnostic,
                "timestamp_utc": utc_now(),
            },
        )
        write_checksums(staging_directory)
    except OSError:
        pass


def provenance_equal(
    initial: dict[str, Any], final: dict[str, Any]
) -> bool:
    return initial == final


def main() -> int:
    arguments = parse_arguments()
    claim = arguments.requested_claim
    if claim in EXPECTED_CLAIM_POLICY["rejected_claims"]:
        raise ValueError(
            f"requested claim {claim!r} is outside this matrix: "
            f"{EXPECTED_CLAIM_POLICY['diagnostic']}"
        )
    if claim != EXPECTED_CLAIM_POLICY["accepted_claim"]:
        raise ValueError(f"unsupported requested claim: {claim!r}")

    matrix_path = arguments.matrix.resolve()
    matrix = validate_contract(matrix_path)
    locked_sources = validate_locked_sources(
        matrix, require_clean_detached=not arguments.validate_only
    )
    if arguments.validate_only:
        if any(
            value is not None
            for value in (
                arguments.build_root,
                arguments.output,
            )
        ):
            raise ValueError(
                "--validate-only does not accept binary or output arguments"
            )
        print(
            json.dumps(
                {
                    "matrix_id": matrix["matrix_id"],
                    "matrix_sha256": EXPECTED_MATRIX_SHA256,
                    "implementation_source_commit": EXPECTED_SOURCE_COMMIT,
                    "requested_claim": claim,
                    "group_count": len(matrix["groups"]),
                    "test_count": sum(
                        len(group["tests"]) for group in matrix["groups"]
                    ),
                    "locked_artifact_count": locked_sources[
                        "locked_artifact_count"
                    ],
                    **EXPECTED_DISPOSITION,
                    "outcome": "PASS",
                },
                sort_keys=True,
            )
        )
        return 0

    if arguments.build_root is None or arguments.output is None:
        raise ValueError("execution requires a build root and output path")
    environment = controlled_environment()
    cmake = arguments.cmake.absolute()
    mpiexec = arguments.mpiexec.absolute()
    if (
        str(cmake) != matrix["build_contract"]["cmake"]
        or str(mpiexec) != matrix["build_contract"]["mpi_launcher"]
    ):
        raise ValueError("CMake and MPI launcher paths are frozen")
    initial_cmake = executable_provenance(
        cmake, ["--version"], environment
    )
    initial_cxx_compiler = executable_provenance(
        Path(matrix["build_contract"]["cxx_compiler"]),
        ["--version"],
        environment,
    )
    initial_mpi_cxx_compiler = executable_provenance(
        Path(matrix["build_contract"]["mpi_cxx_compiler"]),
        ["--showme:version"],
        environment,
    )
    initial_mpiexec = executable_provenance(
        mpiexec, ["--version"], environment
    )
    build_root = arguments.build_root.resolve()
    final_output = arguments.output.resolve()
    expected_output_pattern = re.compile(
        r"free_surface_wp7_active_cell_topology_prerequisite_"
        r"[0-9]{8}_"
        + re.escape(locked_sources["head_commit"][:8])
    )
    if expected_output_pattern.fullmatch(final_output.name) is None:
        raise ValueError(
            "output basename must include the execution date and exact "
            "execution-commit prefix"
        )
    if final_output.exists():
        raise ValueError(f"refusing to replace output: {final_output}")
    if not final_output.parent.is_dir():
        raise ValueError("output parent directory does not exist")
    if final_output.parent.is_symlink():
        raise ValueError("output parent directory must not be a symlink")
    if (
        path_is_within(final_output, REPOSITORY_ROOT)
        or path_is_within(build_root, REPOSITORY_ROOT)
    ):
        raise ValueError(
            "build and output paths must be outside the clean source worktree"
        )
    maximum_memory_envelope_mib = max(
        matrix["build_contract"]["configure_execution"]["memory_mib"],
        matrix["build_contract"]["build_execution"]["memory_mib"],
        *(group["execution"]["memory_mib"] for group in matrix["groups"]),
    )
    preflight_resources = resource_snapshot(final_output.parent)
    if (
        preflight_resources["memory_available_mib"]
        < matrix["resource_guard"]["minimum_available_memory_mib"]
        + maximum_memory_envelope_mib
        or preflight_resources["storage_free_mib"]
        < matrix["resource_guard"]["minimum_available_storage_mib"]
        + matrix["build_contract"]["maximum_build_storage_mib"]
    ):
        raise RuntimeError(
            "RAM or storage headroom is below the largest frozen envelope"
        )
    staging_directory = final_output.parent / (
        f".{final_output.name}.inprogress-{os.getpid()}"
    )
    if staging_directory.exists():
        raise ValueError(f"staging directory already exists: {staging_directory}")
    staging_directory.mkdir()
    write_json(
        staging_directory / "state.json",
        {
            "state": "IN_PROGRESS",
            "outcome": None,
            "final_output": str(final_output),
            "timestamp_utc": utc_now(),
        },
    )
    started_utc = utc_now()
    initial_resources = resource_snapshot(staging_directory)
    if not resource_snapshot_passes(
        initial_resources, matrix["resource_guard"]
    ):
        mark_incomplete(
            staging_directory,
            "initial RAM or storage resource guard failed",
        )
        raise RuntimeError("initial RAM or storage resource guard failed")

    initial_runner_sha256 = sha256_file(SCRIPT_PATH)
    try:
        build_preflight = {
            "artifact_schema_version": 1,
            "mode": "clean_detached_controlled_build",
            "source_provenance": locked_sources,
            "matrix_path": str(matrix_path),
            "matrix_sha256": EXPECTED_MATRIX_SHA256,
            "runner_path": str(SCRIPT_PATH),
            "runner_sha256": initial_runner_sha256,
            "cmake": initial_cmake,
            "cxx_compiler": initial_cxx_compiler,
            "mpi_cxx_compiler": initial_mpi_cxx_compiler,
            "mpiexec": initial_mpiexec,
            "controlled_environment": environment,
            "build_contract": matrix["build_contract"],
            "initial_resources": initial_resources,
            "outcome": "PASS",
        }
        write_json(
            staging_directory / "build_preflight.json", build_preflight
        )
        binaries, build_record = run_controlled_builds(
            matrix,
            cmake,
            build_root,
            staging_directory,
            environment,
        )
        write_json(staging_directory / "build.json", build_record)
        if build_record["outcome"] != "PASS":
            mark_incomplete(
                staging_directory, "controlled build failed"
            )
            return 1
        if any(
            configuration["cache"] is None
            or configuration["cache"].get(
                "cxx_compiler_provenance"
            )
            != initial_cxx_compiler
            or configuration["cache"].get(
                "mpi_cxx_compiler_provenance"
            )
            != initial_mpi_cxx_compiler
            for configuration in build_record["configurations"]
        ):
            mark_incomplete(
                staging_directory,
                "configured compiler provenance changed during build",
            )
            return 1

        initial_binary_record = validate_binaries(
            matrix, binaries, environment
        )
        if not mpi_stack_is_compatible(
            initial_binary_record, initial_mpiexec, build_record
        ):
            mark_incomplete(
                staging_directory,
                "MPI launcher and linked test binary are incompatible",
            )
            return 1
        manifest = {
            "artifact_schema_version": 1,
            "matrix_id": matrix["matrix_id"],
            "matrix_path": str(matrix_path),
            "matrix_sha256": EXPECTED_MATRIX_SHA256,
            "runner_path": str(SCRIPT_PATH),
            "runner_sha256": initial_runner_sha256,
            "requested_claim": claim,
            "started_utc": started_utc,
            "source_provenance": locked_sources,
            "build_provenance": build_record,
            "binary_provenance": initial_binary_record,
            "cmake_provenance": initial_cmake,
            "cxx_compiler_provenance": initial_cxx_compiler,
            "mpi_cxx_compiler_provenance": initial_mpi_cxx_compiler,
            "mpi_launcher_provenance": initial_mpiexec,
            "mpi_stack_compatible": True,
            "resource_guard": matrix["resource_guard"],
            "initial_resources": initial_resources,
            "qualification_scope": EXPECTED_SCOPE,
            "qualification_disposition": EXPECTED_DISPOSITION,
            "predecessor_qualification_status": (
                EXPECTED_PREDECESSOR_STATUS
            ),
        }
        write_json(staging_directory / "manifest.json", manifest)

        groups = []
        flattened_by_test: dict[str, dict[str, Any]] = {}
        for group in matrix["groups"]:
            record, rank_results = run_group(
                group,
                binaries[group["binary"]],
                mpiexec,
                staging_directory,
                matrix["resource_guard"],
                environment,
            )
            groups.append(record)
            if group["mpi_ranks"] == 1 and rank_results:
                for test_name, test_result in rank_results[0].items():
                    if test_name in flattened_by_test:
                        raise RuntimeError(
                            f"duplicate serial test result: {test_name}"
                        )
                    flattened_by_test[test_name] = test_result
            if record["outcome"] != "PASS":
                break

        quantitative = evaluate_quantitative_evidence(
            matrix, flattened_by_test
        )
        write_json(
            staging_directory / "quantitative_evidence.json",
            quantitative,
        )
        write_json(
            staging_directory / "group_recorded_properties.json",
            {
                "declared_check_count": quantitative[
                    "declared_check_count"
                ],
                "checks": quantitative["checks"],
                "outcome": quantitative["outcome"],
            },
        )
        groups_passed = len(groups) == len(matrix["groups"]) and all(
            group["outcome"] == "PASS" for group in groups
        )
        write_json(
            staging_directory / "topology_evidence.json",
            {
                "telemetry_contract": matrix["telemetry_contract"],
                "method_limitations": matrix["method_limitations"],
                "groups": [
                    {
                        "group_id": group["group_id"],
                        "expected_tests": group["expected_tests"],
                        "outcome": group["outcome"],
                    }
                    for group in groups
                ],
                "quantitative_evidence": quantitative,
                "qualification_disposition": EXPECTED_DISPOSITION,
                "outcome": (
                    "PASS"
                    if groups_passed
                    and quantitative["outcome"] == "PASS"
                    else "FAIL"
                ),
            },
        )
        write_json(
            staging_directory / "histories.json",
            {
                "applicable": False,
                "diagnostic": "Static finite fixtures have no time history.",
                "outcome": "NOT_APPLICABLE",
            },
        )
        write_json(
            staging_directory / "checkpoints.json",
            {
                "applicable": False,
                "diagnostic": "Static finite fixtures have no restart checkpoint.",
                "outcome": "NOT_APPLICABLE",
            },
        )

        final_binary_record = validate_binaries(
            matrix, binaries, environment
        )
        final_cmake = executable_provenance(
            cmake, ["--version"], environment
        )
        final_cxx_compiler = executable_provenance(
            Path(matrix["build_contract"]["cxx_compiler"]),
            ["--version"],
            environment,
        )
        final_mpi_cxx_compiler = executable_provenance(
            Path(matrix["build_contract"]["mpi_cxx_compiler"]),
            ["--showme:version"],
            environment,
        )
        final_mpiexec = executable_provenance(
            mpiexec, ["--version"], environment
        )
        final_sources = validate_locked_sources(
            matrix, require_clean_detached=True
        )
        final_resources = resource_snapshot(staging_directory)
        provenance_checks = {
            "matrix_sha256_unchanged": (
                sha256_file(matrix_path) == EXPECTED_MATRIX_SHA256
            ),
            "runner_sha256_unchanged": (
                sha256_file(SCRIPT_PATH) == initial_runner_sha256
            ),
            "head_commit_unchanged": (
                final_sources["head_commit"]
                == locked_sources["head_commit"]
            ),
            "head_tree_unchanged": (
                final_sources["head_tree"] == locked_sources["head_tree"]
            ),
            "source_worktree_clean": final_sources[
                "source_worktree_clean"
            ],
            "source_head_detached": final_sources[
                "source_head_detached"
            ],
            "definition_only_descendant": final_sources[
                "definition_only_descendant"
            ],
            "locked_artifacts_unchanged": (
                final_sources["locked_artifacts"]
                == locked_sources["locked_artifacts"]
            ),
            "binary_provenance_unchanged": (
                final_binary_record == initial_binary_record
            ),
            "cmake_unchanged": provenance_equal(
                initial_cmake, final_cmake
            ),
            "cxx_compiler_unchanged": provenance_equal(
                initial_cxx_compiler, final_cxx_compiler
            ),
            "mpi_cxx_compiler_unchanged": provenance_equal(
                initial_mpi_cxx_compiler,
                final_mpi_cxx_compiler,
            ),
            "mpi_launcher_unchanged": provenance_equal(
                initial_mpiexec, final_mpiexec
            ),
            "mpi_stack_compatible": mpi_stack_is_compatible(
                final_binary_record, final_mpiexec, build_record
            ),
            "all_group_sessions_terminated": all(
                group["execution"]["session_cleanup"][
                    "all_session_processes_terminated"
                ]
                for group in groups
            ),
            "all_group_process_sampling_complete": all(
                group["execution"]["session_cleanup"][
                    "sampling_complete"
                ]
                for group in groups
            ),
            "final_resource_guard_passed": resource_snapshot_passes(
                final_resources, matrix["resource_guard"]
            ),
        }
        provenance_passed = all(provenance_checks.values())
        tests_passed = (
            groups_passed and quantitative["outcome"] == "PASS"
        )
        overall_passed = (
            build_record["outcome"] == "PASS"
            and tests_passed
            and provenance_passed
        )
        final_provenance = {
            "completed_utc": utc_now(),
            "checks": provenance_checks,
            "source_provenance": final_sources,
            "binary_provenance": final_binary_record,
            "cmake_provenance": final_cmake,
            "cxx_compiler_provenance": final_cxx_compiler,
            "mpi_cxx_compiler_provenance": (
                final_mpi_cxx_compiler
            ),
            "mpi_launcher_provenance": final_mpiexec,
            "final_resources": final_resources,
            "outcome": "PASS" if provenance_passed else "FAIL",
        }
        write_json(
            staging_directory / "final_provenance.json",
            final_provenance,
        )
        gates = {
            "expected": matrix["gates"],
            "observed_group_count": len(groups),
            "observed_distinct_test_count": len(
                {
                    test
                    for group in groups
                    for test in group["expected_tests"]
                }
            ),
            "observed_quantitative_evidence_count": quantitative[
                "declared_check_count"
            ],
            "groups_outcome": "PASS" if groups_passed else "FAIL",
            "quantitative_outcome": quantitative["outcome"],
            "provenance_outcome": final_provenance["outcome"],
            "outcome": "PASS" if overall_passed else "FAIL",
        }
        write_json(staging_directory / "gates.json", gates)
        summary = {
            "artifact_schema_version": 1,
            "matrix_id": matrix["matrix_id"],
            "requested_claim": claim,
            "execution_source_commit": locked_sources["head_commit"],
            "implementation_source_commit": EXPECTED_SOURCE_COMMIT,
            "group_count_executed": len(groups),
            "group_count_expected": len(matrix["groups"]),
            "groups": groups,
            "build_outcome": build_record["outcome"],
            "quantitative_evidence": quantitative,
            "provenance_outcome": final_provenance["outcome"],
            "final_resources": final_resources,
            "qualification_scope": EXPECTED_SCOPE,
            "qualification_disposition": EXPECTED_DISPOSITION,
            "open_outcomes": EXPECTED_OPEN_OUTCOMES,
            "predecessor_qualification_status": (
                EXPECTED_PREDECESSOR_STATUS
            ),
            "outcome": "PASS" if overall_passed else "FAIL",
        }
        write_json(staging_directory / "summary.json", summary)
        if not overall_passed:
            mark_incomplete(
                staging_directory,
                "one or more build, test, or provenance gates failed",
            )
            return 1
        write_record(staging_directory, matrix, summary)
        write_json(
            staging_directory / "state.json",
            {
                "state": "FINAL",
                "outcome": "PASS",
                "final_output": str(final_output),
                "timestamp_utc": utc_now(),
            },
        )
        write_checksums(staging_directory)
        staging_directory.rename(final_output)
        return 0
    except Exception as error:
        mark_incomplete(
            staging_directory,
            f"{type(error).__name__}: {error}",
        )
        raise


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        json.JSONDecodeError,
        KeyError,
        OSError,
        RuntimeError,
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
        ValueError,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)

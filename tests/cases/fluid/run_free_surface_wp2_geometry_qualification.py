#!/usr/bin/env python3
"""Run the frozen WP-2 authoritative-geometry qualification matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import resource
import signal
import subprocess
import sys
import time
from typing import Any


SCRIPT_PATH = Path(__file__).resolve()
DEFAULT_REGISTRY = SCRIPT_PATH.with_name(
    "free_surface_wp2_geometry_qualification_matrix.json"
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    if path.exists():
        raise RuntimeError(f"refusing to replace artifact path: {path}")
    with path.open("x", encoding="utf-8") as output:
        json.dump(value, output, indent=2, sort_keys=True)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())


def write_text(path: Path, value: str) -> None:
    if path.exists():
        raise RuntimeError(f"refusing to replace artifact path: {path}")
    with path.open("x", encoding="utf-8") as output:
        output.write(value)
        output.flush()
        os.fsync(output.fileno())


def git_bytes(source_root: Path, *arguments: str) -> bytes:
    result = subprocess.run(
        ["git", *arguments],
        cwd=source_root,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout


def equal_check(metric: str, actual: Any, expected: Any) -> dict[str, Any]:
    return {
        "metric": metric,
        "actual": actual,
        "expected": expected,
        "relation": "equal",
        "passed": actual == expected,
    }


def load_registry(path: Path) -> dict[str, Any]:
    registry = json.loads(path.read_text(encoding="utf-8"))
    if registry.get("schema_version") != 1:
        raise ValueError("unsupported WP-2 qualification schema")
    groups = registry.get("groups")
    gates = registry.get("gates", {})
    claims = registry.get("closure_contract")
    if not isinstance(groups, list) or not groups:
        raise ValueError("WP-2 qualification group list is empty")
    if not isinstance(claims, list) or not claims:
        raise ValueError("WP-2 closure contract is empty")
    if gates.get("expected_group_count") != len(groups):
        raise ValueError("expected group count does not match the frozen list")

    allowed_binaries = {
        "geometry",
        "level_set",
        "systems",
        "application",
        "assembly_mpi",
        "application_mpi",
    }
    group_ids: set[str] = set()
    test_names: set[str] = set()
    for group in groups:
        group_id = group.get("id")
        if not isinstance(group_id, str) or not group_id:
            raise ValueError("every qualification group needs an id")
        if group_id in group_ids:
            raise ValueError(f"duplicate qualification group: {group_id}")
        group_ids.add(group_id)
        binary = group.get("binary")
        if binary not in allowed_binaries:
            raise ValueError(f"unsupported binary key in group {group_id}")
        ranks = group.get("mpi_ranks")
        copies = group.get("gtest_output_copies")
        if not isinstance(ranks, int) or ranks <= 0:
            raise ValueError(f"group {group_id} needs positive mpi_ranks")
        if not isinstance(copies, int) or copies <= 0 or copies > ranks:
            raise ValueError(
                f"group {group_id} needs gtest_output_copies in [1, mpi_ranks]"
            )
        if ranks == 1 and copies != 1:
            raise ValueError(f"serial group {group_id} needs one output copy")
        tests = group.get("tests")
        if not isinstance(tests, list) or not tests:
            raise ValueError(f"group {group_id} has no tests")
        for name in tests:
            if not isinstance(name, str) or name.count(".") != 1:
                raise ValueError(f"invalid suite.name in group {group_id}: {name}")
            if name in test_names:
                raise ValueError(f"duplicate frozen test: {name}")
            test_names.add(name)
        execution = group.get("execution", {})
        for key in ("wall_time_seconds", "memory_mib", "output_mib"):
            if not isinstance(execution.get(key), int) or execution[key] <= 0:
                raise ValueError(
                    f"group {group_id} execution envelope {key} must be positive"
                )
    if gates.get("expected_distinct_test_count") != len(test_names):
        raise ValueError("expected distinct test count does not match the frozen list")

    claim_names: set[str] = set()
    for claim in claims:
        name = claim.get("claim")
        evidence = claim.get("evidence")
        if not isinstance(name, str) or not name or name in claim_names:
            raise ValueError("closure claims must have unique nonempty names")
        claim_names.add(name)
        if not isinstance(evidence, list) or not evidence:
            raise ValueError(f"closure claim {name} has no evidence")
        missing = sorted(set(evidence) - test_names)
        if missing:
            raise ValueError(f"closure claim {name} cites unfrozen tests: {missing}")
    return registry


def listed_gtests(binary: Path) -> set[str]:
    result = subprocess.run(
        [str(binary), "--gtest_list_tests"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
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
            full_name = f"{suite_name}.{test_name}"
            if full_name in flattened:
                raise ValueError(f"duplicate test result: {full_name}")
            flattened[full_name] = test
    return flattened


def evaluate_serial_result(
    expected_tests: list[str],
    document: dict[str, Any],
    return_code: int,
    termination_reason: str | None,
) -> list[dict[str, Any]]:
    expected = set(expected_tests)
    actual = flatten_gtest(document)
    actual_names = set(actual)
    incomplete = sorted(
        name
        for name, result in actual.items()
        if result.get("result") != "COMPLETED" or result.get("status") != "RUN"
    )
    failed_records = sorted(
        name for name, result in actual.items() if result.get("failures")
    )
    return [
        equal_check("process_return_code", return_code, 0),
        equal_check("termination_reason", termination_reason, None),
        equal_check("test_count", document.get("tests"), len(expected_tests)),
        equal_check("failure_count", document.get("failures"), 0),
        equal_check("error_count", document.get("errors"), 0),
        equal_check("disabled_count", document.get("disabled"), 0),
        equal_check("missing_tests", sorted(expected - actual_names), []),
        equal_check("unexpected_tests", sorted(actual_names - expected), []),
        equal_check("incomplete_or_skipped_tests", incomplete, []),
        equal_check("tests_with_failure_records", failed_records, []),
    ]


def evaluate_mpi_result(
    expected_tests: list[str],
    expected_output_copies: int,
    stdout: str,
    stderr: str,
    return_code: int,
    termination_reason: str | None,
) -> list[dict[str, Any]]:
    run_pattern = re.compile(r"\[ RUN\s+\]\s+([A-Za-z0-9_]+\.[A-Za-z0-9_]+)")
    ok_pattern = re.compile(r"\[\s+OK\s+\]\s+([A-Za-z0-9_]+\.[A-Za-z0-9_]+)")
    observed_runs = run_pattern.findall(stdout)
    observed_ok = ok_pattern.findall(stdout)
    expected = set(expected_tests)
    checks = [
        equal_check("process_return_code", return_code, 0),
        equal_check("termination_reason", termination_reason, None),
        equal_check("failure_marker_count", stdout.count("[  FAILED  ]"), 0),
        equal_check("stderr_failure_marker_count", stderr.count("[  FAILED  ]"), 0),
        equal_check("unexpected_run_tests", sorted(set(observed_runs) - expected), []),
        equal_check("unexpected_completed_tests", sorted(set(observed_ok) - expected), []),
    ]
    for name in expected_tests:
        checks.append(
            equal_check(
                f"run_multiplicity:{name}",
                observed_runs.count(name),
                expected_output_copies,
            )
        )
        checks.append(
            equal_check(
                f"pass_multiplicity:{name}",
                observed_ok.count(name),
                expected_output_copies,
            )
        )
    return checks


def directory_size(path: Path) -> int:
    total = 0
    for candidate in path.rglob("*"):
        try:
            if candidate.is_file() and not candidate.is_symlink():
                total += candidate.stat().st_size
        except FileNotFoundError:
            continue
    return total


def process_resident_kib(process_id: int) -> int | None:
    try:
        for line in Path(f"/proc/{process_id}/status").read_text(
            encoding="utf-8"
        ).splitlines():
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
) -> dict[str, Any]:
    memory_bytes = memory_mib * 1024 * 1024

    def set_limits() -> None:
        resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))

    started = time.monotonic()
    peak_resident_kib = 0
    termination_reason: str | None = None
    with stdout_path.open("xb") as stdout_file, stderr_path.open("xb") as stderr_file:
        process = subprocess.Popen(
            command,
            cwd=working_directory,
            env=environment,
            stdout=stdout_file,
            stderr=stderr_file,
            start_new_session=True,
            preexec_fn=set_limits,
        )
        while process.poll() is None:
            elapsed = time.monotonic() - started
            resident = process_resident_kib(process.pid)
            if resident is not None:
                peak_resident_kib = max(peak_resident_kib, resident)
            if elapsed > wall_time_seconds:
                termination_reason = "wall_time_envelope_exceeded"
            elif directory_size(output_directory) > output_mib * 1024 * 1024:
                termination_reason = "output_envelope_exceeded"
            elif resident is not None and resident > memory_mib * 1024:
                termination_reason = "memory_envelope_exceeded"
            if termination_reason is not None:
                terminate_process_group(process)
                break
            time.sleep(0.05)
        return_code = process.wait()
    return {
        "return_code": return_code,
        "termination_reason": termination_reason,
        "wall_time_seconds": time.monotonic() - started,
        "peak_resident_kib_sampled": peak_resident_kib,
        "final_output_bytes": directory_size(output_directory),
    }


def find_cmake_cache(binary: Path) -> Path | None:
    for directory in [binary.parent, *binary.parents]:
        candidate = directory / "CMakeCache.txt"
        if candidate.is_file():
            return candidate
    return None


def selected_cmake_cache(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    prefixes = (
        "CMAKE_BUILD_TYPE:",
        "CMAKE_CXX_COMPILER:",
        "CMAKE_CXX_COMPILER_ID:",
        "CMAKE_CXX_COMPILER_VERSION:",
        "CMAKE_CXX_FLAGS:",
        "CMAKE_CXX_FLAGS_",
        "FE_ENABLE_MPI:",
        "FE_ENABLE_LLVM_JIT:",
    )
    selected: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith(("#", "//")) or "=" not in line:
            continue
        left, value = line.split("=", 1)
        if left.startswith(prefixes):
            selected[left] = value
    return selected


def binary_record(binary: Path, source_root: Path) -> dict[str, Any]:
    cache = find_cmake_cache(binary)
    try:
        recorded_path = binary.relative_to(source_root).as_posix()
    except ValueError:
        recorded_path = str(binary)
    linked = subprocess.run(
        ["ldd", str(binary)],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    ).stdout
    return {
        "path": recorded_path,
        "sha256": sha256_file(binary),
        "cmake_cache_path": str(cache) if cache else None,
        "cmake_cache_sha256": sha256_file(cache) if cache else None,
        "selected_cmake_cache": selected_cmake_cache(cache),
        "linked_libraries": linked.splitlines(),
    }


def machine_memory_mib() -> int | None:
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) // 1024
    except (OSError, ValueError, IndexError):
        return None
    return None


def run_gtest_group(
    group: dict[str, Any],
    binaries: dict[str, Path],
    mpiexec: Path,
    source_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    group_directory = output_root / "groups" / group["id"]
    group_directory.mkdir(parents=True, exist_ok=False)
    stdout_path = group_directory / "stdout.txt"
    stderr_path = group_directory / "stderr.txt"
    binary = binaries[group["binary"]]
    test_filter = ":".join(group["tests"])
    ranks = group["mpi_ranks"]
    if ranks == 1:
        gtest_path = group_directory / "gtest.json"
        command = [
            str(binary),
            f"--gtest_filter={test_filter}",
            "--gtest_color=no",
            f"--gtest_output=json:{gtest_path}",
        ]
    else:
        gtest_path = None
        command = [
            str(mpiexec),
            "--oversubscribe",
            "-n",
            str(ranks),
            str(binary),
            f"--gtest_filter={test_filter}",
            "--gtest_color=no",
        ]
    environment = os.environ.copy()
    environment.update(
        {
            "OMP_NUM_THREADS": "1",
            "OMPI_ALLOW_RUN_AS_ROOT": "1",
            "OMPI_ALLOW_RUN_AS_ROOT_CONFIRM": "1",
        }
    )
    execution = group["execution"]
    resources = run_monitored(
        command,
        environment,
        source_root,
        stdout_path,
        stderr_path,
        group_directory,
        execution["wall_time_seconds"],
        execution["memory_mib"],
        execution["output_mib"],
    )
    stdout = stdout_path.read_text(encoding="utf-8", errors="replace")
    stderr = stderr_path.read_text(encoding="utf-8", errors="replace")
    if ranks == 1 and gtest_path is not None and gtest_path.is_file():
        document = json.loads(gtest_path.read_text(encoding="utf-8"))
        checks = evaluate_serial_result(
            group["tests"],
            document,
            resources["return_code"],
            resources["termination_reason"],
        )
    elif ranks == 1:
        checks = [equal_check("gtest_result_present", False, True)]
    else:
        checks = evaluate_mpi_result(
            group["tests"],
            group["gtest_output_copies"],
            stdout,
            stderr,
            resources["return_code"],
            resources["termination_reason"],
        )
    passed = bool(checks) and all(check["passed"] for check in checks)
    result = {
        "group_id": group["id"],
        "command": command,
        "mpi_ranks": ranks,
        "gtest_output_copies": group["gtest_output_copies"],
        "expected_tests": group["tests"],
        "execution": execution,
        "resources": resources,
        "checks": checks,
        "outcome": "PASS" if passed else "FAIL_METHOD",
    }
    write_json(group_directory / "result.json", result)
    return result


def write_checksums(output_directory: Path) -> None:
    entries = []
    for path in sorted(output_directory.rglob("*")):
        if path.is_file() and path.name != "checksums.txt":
            entries.append(
                f"{sha256_file(path)}  {path.relative_to(output_directory).as_posix()}"
            )
    write_text(output_directory / "checksums.txt", "\n".join(entries) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--geometry-binary", type=Path, required=True)
    parser.add_argument("--level-set-binary", type=Path, required=True)
    parser.add_argument("--systems-binary", type=Path, required=True)
    parser.add_argument("--application-binary", type=Path, required=True)
    parser.add_argument("--assembly-mpi-binary", type=Path, required=True)
    parser.add_argument("--application-mpi-binary", type=Path, required=True)
    parser.add_argument("--mpiexec", type=Path, default=Path("/usr/bin/mpiexec"))
    parser.add_argument("--source-root", type=Path, default=SCRIPT_PATH.parents[3])
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    source_root = arguments.source_root.resolve()
    registry_path = arguments.registry.resolve()
    output_directory = arguments.output.resolve()
    mpiexec = arguments.mpiexec.resolve()
    binaries = {
        "geometry": arguments.geometry_binary.resolve(),
        "level_set": arguments.level_set_binary.resolve(),
        "systems": arguments.systems_binary.resolve(),
        "application": arguments.application_binary.resolve(),
        "assembly_mpi": arguments.assembly_mpi_binary.resolve(),
        "application_mpi": arguments.application_mpi_binary.resolve(),
    }
    registry = load_registry(registry_path)
    if output_directory.exists():
        raise SystemExit(f"refusing to replace output directory: {output_directory}")
    for label, binary in binaries.items():
        if not binary.is_file() or not os.access(binary, os.X_OK):
            raise SystemExit(f"{label} test binary is not executable: {binary}")
    if not mpiexec.is_file():
        raise SystemExit(f"MPI launcher is not a file: {mpiexec}")

    configured_by_binary: dict[str, set[str]] = {}
    for group in registry["groups"]:
        configured_by_binary.setdefault(group["binary"], set()).update(group["tests"])
    for binary_key, expected in configured_by_binary.items():
        missing = sorted(expected - listed_gtests(binaries[binary_key]))
        if missing:
            raise SystemExit(
                f"frozen tests are missing from {binary_key} binary: {missing}"
            )

    tracked_status = git_bytes(
        source_root, "status", "--porcelain=v1", "--untracked-files=no"
    )
    if tracked_status:
        raise SystemExit("qualification requires clean tracked sources")
    untracked = git_bytes(
        source_root, "ls-files", "--others", "--exclude-standard", "-z"
    )
    untracked_count = sum(1 for value in untracked.split(b"\0") if value)
    source_commit = git_bytes(source_root, "rev-parse", "HEAD").decode().strip()
    source_tree = git_bytes(source_root, "rev-parse", "HEAD^{tree}").decode().strip()
    output_directory.mkdir(parents=True, exist_ok=False)

    write_json(
        output_directory / "manifest.json",
        {
            "artifact_schema_version": 1,
            "matrix_id": registry["matrix_id"],
            "matrix_status_at_execution": registry["status"],
            "registry_sha256": sha256_file(registry_path),
            "runner_sha256": sha256_file(SCRIPT_PATH),
            "work_package": registry["work_package"],
            "findings": registry["findings"],
            "model_envelope": registry["model_envelope"],
            "source_commit": source_commit,
            "source_tree": source_tree,
            "groups": registry["groups"],
            "closure_contract": registry["closure_contract"],
            "qualification_scope": registry["qualification_scope"],
        },
    )
    write_json(
        output_directory / "build.json",
        {
            "source_commit": source_commit,
            "source_tree": source_tree,
            "tracked_sources_clean": True,
            "untracked_path_count": untracked_count,
            "untracked_path_list_sha256": sha256_bytes(untracked),
            "binaries": {
                key: binary_record(binary, source_root)
                for key, binary in binaries.items()
            },
            "machine": {
                "platform": platform.platform(),
                "machine": platform.machine(),
                "processor": platform.processor(),
                "cpu_count": os.cpu_count(),
                "memory_mib": machine_memory_mib(),
            },
        },
    )
    write_json(
        output_directory / "gates.json",
        {
            "matrix_status_at_execution": registry["status"],
            "gates": registry["gates"],
            "closure_contract": registry["closure_contract"],
            "qualification_scope": registry["qualification_scope"],
        },
    )

    group_results = [
        run_gtest_group(group, binaries, mpiexec, source_root, output_directory)
        for group in registry["groups"]
    ]
    passed = all(result["outcome"] == "PASS" for result in group_results)
    summary = {
        "matrix_id": registry["matrix_id"],
        "source_commit": source_commit,
        "distinct_test_count": registry["gates"]["expected_distinct_test_count"],
        "group_outcomes": {
            result["group_id"]: result["outcome"] for result in group_results
        },
        "overall_outcome": "PASS" if passed else "FAIL_METHOD",
        "qualification_scope": registry["qualification_scope"],
    }
    write_json(output_directory / "summary.json", summary)
    record_lines = [
        "# WP-2 authoritative-geometry qualification record",
        "",
        f"- Source commit: `{source_commit}`",
        f"- Frozen matrix: `{registry['matrix_id']}`",
        f"- Outcome: **{summary['overall_outcome']}**",
        f"- Distinct tests: {registry['gates']['expected_distinct_test_count']}",
        f"- Serial and distributed groups: {len(registry['groups'])}",
        "",
        registry["qualification_scope"] + ".",
        "",
    ]
    write_text(output_directory / "record.md", "\n".join(record_lines))
    write_checksums(output_directory)
    print(output_directory)
    print(summary["overall_outcome"])
    return 0 if passed else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)

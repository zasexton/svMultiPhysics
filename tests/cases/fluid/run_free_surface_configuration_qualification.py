#!/usr/bin/env python3
"""Run the frozen free-surface configuration containment matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import signal
import subprocess
import sys
import time
from typing import Any


SCRIPT_PATH = Path(__file__).resolve()
DEFAULT_REGISTRY = SCRIPT_PATH.with_name(
    "free_surface_configuration_qualification_matrix.json"
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


def git_bytes(source_root: Path, *arguments: str) -> bytes:
    result = subprocess.run(
        ["git", *arguments],
        cwd=source_root,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout


def load_registry(path: Path) -> dict[str, Any]:
    registry = json.loads(path.read_text(encoding="utf-8"))
    if registry.get("schema_version") != 1:
        raise ValueError("unsupported configuration qualification schema")
    tests = registry.get("tests")
    if not isinstance(tests, list) or not tests:
        raise ValueError("configuration qualification test list is empty")
    if any(not isinstance(name, str) or name.count(".") != 1 for name in tests):
        raise ValueError("every configuration qualification test needs suite.name form")
    if len(set(tests)) != len(tests):
        raise ValueError("configuration qualification tests must be unique")
    gates = registry.get("gates", {})
    if gates.get("expected_test_count") != len(tests):
        raise ValueError("expected test count does not match the frozen list")
    execution = registry.get("execution", {})
    for key in ("wall_time_seconds", "memory_mib", "output_mib"):
        if not isinstance(execution.get(key), int) or execution[key] <= 0:
            raise ValueError(f"execution envelope {key} must be positive")
    return registry


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


def evaluate_results(
    registry: dict[str, Any],
    document: dict[str, Any],
    return_code: int,
    termination_reason: str | None,
) -> list[dict[str, Any]]:
    gates = registry["gates"]
    expected = set(registry["tests"])
    actual = flatten_gtest(document)
    actual_names = set(actual)
    checks: list[dict[str, Any]] = []

    def equal(metric: str, value: Any, target: Any) -> None:
        checks.append(
            {
                "metric": metric,
                "actual": value,
                "expected": target,
                "relation": "equal",
                "passed": value == target,
            }
        )

    equal("process_return_code", return_code, 0)
    equal("termination_reason", termination_reason, None)
    equal("test_count", document.get("tests"), gates["expected_test_count"])
    equal("failure_count", document.get("failures"), gates["expected_failures"])
    equal("error_count", document.get("errors"), gates["expected_errors"])
    equal("disabled_count", document.get("disabled"), gates["expected_disabled"])
    equal("missing_tests", sorted(expected - actual_names), [])
    equal("unexpected_tests", sorted(actual_names - expected), [])
    incomplete = sorted(
        name
        for name, result in actual.items()
        if result.get("result") != "COMPLETED" or result.get("status") != "RUN"
    )
    equal("incomplete_or_skipped_tests", incomplete, [])
    failed_results = sorted(
        name for name, result in actual.items() if result.get("failures")
    )
    equal("tests_with_failure_records", failed_results, [])
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
            time.sleep(0.02)
        return_code = process.wait()
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    return {
        "return_code": return_code,
        "termination_reason": termination_reason,
        "wall_time_seconds": time.monotonic() - started,
        "peak_resident_kib_sampled": peak_resident_kib,
        "child_max_resident_kib": usage.ru_maxrss,
        "user_cpu_seconds": usage.ru_utime,
        "system_cpu_seconds": usage.ru_stime,
        "final_output_bytes_before_manifest_completion": directory_size(
            output_directory
        ),
    }


def find_cmake_cache(test_binary: Path) -> Path | None:
    for directory in [test_binary.parent, *test_binary.parents]:
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
        "PHYSICS_BUILD_TESTS:",
    )
    selected: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith(("#", "//")) or "=" not in line:
            continue
        left, value = line.split("=", 1)
        if left.startswith(prefixes):
            selected[left] = value
    return selected


def machine_memory_mib() -> int | None:
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) // 1024
    except (OSError, ValueError, IndexError):
        return None
    return None


def write_checksums(output_directory: Path) -> None:
    entries = []
    for path in sorted(output_directory.rglob("*")):
        if path.is_file() and path.name != "checksums.txt":
            entries.append(
                f"{sha256_file(path)}  {path.relative_to(output_directory).as_posix()}"
            )
    checksum_path = output_directory / "checksums.txt"
    if checksum_path.exists():
        raise RuntimeError(f"refusing to replace artifact path: {checksum_path}")
    with checksum_path.open("x", encoding="utf-8") as output:
        output.write("\n".join(entries) + "\n")
        output.flush()
        os.fsync(output.fileno())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--test-binary", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, default=SCRIPT_PATH.parents[3])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    registry_path = args.registry.resolve()
    test_binary = args.test_binary.resolve()
    source_root = args.source_root.resolve()
    output_directory = args.output.resolve()
    registry = load_registry(registry_path)
    if not test_binary.is_file() or not os.access(test_binary, os.X_OK):
        raise SystemExit(f"test binary is not executable: {test_binary}")
    if output_directory.exists():
        raise SystemExit(f"refusing to replace output directory: {output_directory}")

    status = git_bytes(
        source_root, "status", "--porcelain=v1", "--untracked-files=all"
    )
    if status:
        raise SystemExit(
            "qualification requires a worktree with no tracked or untracked changes"
        )

    output_directory.mkdir(parents=True, exist_ok=False)
    source_commit = git_bytes(source_root, "rev-parse", "HEAD").decode().strip()
    source_tree = git_bytes(source_root, "rev-parse", "HEAD^{tree}").decode().strip()
    tracked_diff = git_bytes(source_root, "diff", "--binary", "HEAD")
    cmake_cache = find_cmake_cache(test_binary)
    execution = registry["execution"]
    gtest_path = output_directory / "gtest.json"
    stdout_path = output_directory / "stdout.txt"
    stderr_path = output_directory / "stderr.txt"
    test_filter = ":".join(registry["tests"])
    command = [
        str(test_binary),
        f"--gtest_filter={test_filter}",
        f"--gtest_output=json:{gtest_path}",
    ]

    write_json(
        output_directory / "manifest.json",
        {
            "artifact_schema_version": 1,
            "matrix_id": registry["matrix_id"],
            "registry_sha256": sha256_file(registry_path),
            "work_package": registry["work_package"],
            "findings": registry["findings"],
            "model_envelope": registry["model_envelope"],
            "source_commit": source_commit,
            "source_tree": source_tree,
            "tests": registry["tests"],
            "exit_contract": registry["exit_contract"],
        },
    )
    write_json(
        output_directory / "build.json",
        {
            "source_commit": source_commit,
            "source_tree": source_tree,
            "tracked_diff_sha256": sha256_bytes(tracked_diff),
            "worktree_clean_including_untracked": True,
            "test_binary": str(test_binary.relative_to(source_root)),
            "test_binary_sha256": sha256_file(test_binary),
            "cmake_cache": selected_cmake_cache(cmake_cache),
            "machine": {
                "platform": platform.platform(),
                "machine": platform.machine(),
                "processor": platform.processor(),
                "cpu_count": os.cpu_count(),
                "memory_mib": machine_memory_mib(),
            },
            "mpi_ranks": execution["mpi_ranks"],
            "threads": execution["threads"],
        },
    )
    write_json(
        output_directory / "gates.json",
        {
            "matrix_status_at_execution": registry["status"],
            "gates": registry["gates"],
            "resource_envelope": execution,
        },
    )

    environment = os.environ.copy()
    environment["OMP_NUM_THREADS"] = str(execution["threads"])
    run = run_monitored(
        command=command,
        environment=environment,
        working_directory=source_root,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        output_directory=output_directory,
        wall_time_seconds=execution["wall_time_seconds"],
        memory_mib=execution["memory_mib"],
        output_mib=execution["output_mib"],
    )

    if gtest_path.is_file():
        document = json.loads(gtest_path.read_text(encoding="utf-8"))
        checks = evaluate_results(
            registry,
            document,
            run["return_code"],
            run["termination_reason"],
        )
    else:
        checks = [
            {
                "metric": "gtest_result_present",
                "actual": False,
                "expected": True,
                "relation": "equal",
                "passed": False,
            }
        ]
    passed = all(check["passed"] for check in checks)
    run["outcome"] = "PASS" if passed else "FAIL"
    run["command"] = command
    write_json(output_directory / "run.json", run)
    write_json(
        output_directory / "comparison.json",
        {
            "matrix_id": registry["matrix_id"],
            "checks": checks,
            "disposition": (
                "PASS"
                if passed
                else (
                    "INFRASTRUCTURE_FAILURE"
                    if run["termination_reason"] is not None
                    else "FAIL_METHOD"
                )
            ),
            "reason": (
                "complete frozen configuration matrix passed"
                if passed
                else "one or more frozen configuration gates failed"
            ),
        },
    )
    write_checksums(output_directory)
    print(output_directory)
    print("PASS" if passed else "FAIL")
    return 0 if passed else 2


if __name__ == "__main__":
    sys.exit(main())

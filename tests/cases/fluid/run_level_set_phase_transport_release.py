#!/usr/bin/env python3
"""Run one frozen conservative phase-transport release-matrix point."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import resource
import shutil
import signal
import subprocess
import sys
import time
from typing import Any


SCRIPT_PATH = Path(__file__).resolve()
DEFAULT_REGISTRY = SCRIPT_PATH.with_name(
    "level_set_phase_transport_release_matrix.json"
)
TEST_FILTER = (
    "LevelSetConservativePhaseQualification."
    "RunsOneExplicitReleaseMatrixPoint"
)
ENVIRONMENT_PREFIX = "SVMP_PHASE_TRANSPORT_RELEASE_"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    if path.exists() or temporary.exists():
        raise RuntimeError(f"refusing to replace artifact path: {path}")
    with temporary.open("x", encoding="utf-8") as output:
        json.dump(value, output, indent=2, sort_keys=True)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
    os.link(temporary, path)
    temporary.unlink()


def git_bytes(source_root: Path, *arguments: str) -> bytes:
    result = subprocess.run(
        ["git", *arguments],
        cwd=source_root,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout


def find_cmake_cache(test_binary: Path) -> Path | None:
    for directory in [test_binary.parent, *test_binary.parents]:
        candidate = directory / "CMakeCache.txt"
        if candidate.is_file():
            return candidate
    return None


def selected_cmake_cache(cache_path: Path | None) -> dict[str, str]:
    if cache_path is None:
        return {}
    selected_prefixes = (
        "CMAKE_BUILD_TYPE:",
        "CMAKE_CXX_COMPILER:",
        "CMAKE_CXX_COMPILER_ID:",
        "CMAKE_CXX_COMPILER_VERSION:",
        "CMAKE_CXX_FLAGS:",
        "CMAKE_CXX_FLAGS_",
        "FE_ENABLE_MPI:",
        "FE_ENABLE_TESTS:",
        "SV_USE_MPI:",
    )
    values: dict[str, str] = {}
    for line in cache_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("//") or line.startswith("#") or "=" not in line:
            continue
        left, value = line.split("=", 1)
        if left.startswith(selected_prefixes):
            values[left] = value
    return values


def machine_memory_mib() -> int | None:
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) // 1024
    except (OSError, ValueError, IndexError):
        return None
    return None


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


def directory_size(path: Path) -> int:
    total = 0
    for candidate in path.rglob("*"):
        try:
            if candidate.is_file() and not candidate.is_symlink():
                total += candidate.stat().st_size
        except FileNotFoundError:
            continue
    return total


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
    wall_time_seconds: int,
    memory_mib: int,
    output_mib: int,
    output_directory: Path,
) -> dict[str, Any]:
    memory_bytes = memory_mib * 1024 * 1024

    def set_limits() -> None:
        resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))

    started = time.monotonic()
    peak_resident_kib = 0
    reason: str | None = None
    last_output_check = started
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
            now = time.monotonic()
            resident = process_resident_kib(process.pid)
            if resident is not None:
                peak_resident_kib = max(peak_resident_kib, resident)
                if resident > memory_mib * 1024:
                    reason = "memory_envelope_exceeded"
                    terminate_process_group(process)
                    break
            if now - started > wall_time_seconds:
                reason = "wall_time_envelope_exceeded"
                terminate_process_group(process)
                break
            if now - last_output_check >= 1.0:
                last_output_check = now
                if directory_size(output_directory) > output_mib * 1024 * 1024:
                    reason = "output_envelope_exceeded"
                    terminate_process_group(process)
                    break
            time.sleep(0.1)
        return_code = process.wait()
    finished = time.monotonic()
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    return {
        "return_code": return_code,
        "termination_reason": reason,
        "wall_time_seconds": finished - started,
        "peak_resident_kib_sampled": peak_resident_kib,
        "child_max_resident_kib": usage.ru_maxrss,
        "user_cpu_seconds": usage.ru_utime,
        "system_cpu_seconds": usage.ru_stime,
        "final_output_bytes": directory_size(output_directory),
    }


def find_test_properties(result_path: Path) -> dict[str, str]:
    document = json.loads(result_path.read_text(encoding="utf-8"))
    for suite in document.get("testsuites", []):
        for test in suite.get("testsuite", []):
            if test.get("classname") == "LevelSetConservativePhaseQualification":
                excluded = {
                    "name",
                    "status",
                    "result",
                    "timestamp",
                    "time",
                    "classname",
                }
                return {
                    key: str(value)
                    for key, value in test.items()
                    if key not in excluded
                }
    raise RuntimeError("qualification test properties are absent")


def evaluate_point(
    properties: dict[str, str],
    case_id: str,
    resolution: int,
    requested_cfl: float,
    common_gates: dict[str, float],
    case_gates: dict[str, float],
    history_path: Path,
    details_path: Path,
) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []

    def exact(name: str, actual: str, expected: str) -> None:
        checks.append(
            {
                "metric": name,
                "actual": actual,
                "expected": expected,
                "relation": "equal",
                "passed": actual == expected,
            }
        )

    def upper(name: str, value: float, limit: float) -> None:
        checks.append(
            {
                "metric": name,
                "actual": value,
                "limit": limit,
                "relation": "less_than_or_equal",
                "passed": value <= limit,
            }
        )

    def lower(name: str, value: float, limit: float) -> None:
        checks.append(
            {
                "metric": name,
                "actual": value,
                "limit": limit,
                "relation": "greater_than_or_equal",
                "passed": value >= limit,
            }
        )

    def number(name: str) -> float:
        if name not in properties:
            raise RuntimeError(f"missing numeric property: {name}")
        return float(properties[name])

    exact("matrix_case", properties.get("matrix_case", ""), case_id)
    exact("resolution", properties.get("resolution", ""), str(resolution))
    exact("requested_cfl", properties.get("requested_cfl", ""), str(requested_cfl))
    upper(
        "achieved_graph_cfl",
        number("achieved_graph_cfl"),
        requested_cfl + common_gates["achieved_cfl_excess"],
    )
    upper(
        "maximum_accounted_balance_error",
        number("maximum_accounted_balance_error"),
        common_gates["maximum_accounted_balance_error"],
    )
    lower(
        "minimum_indicator",
        number("minimum_indicator"),
        common_gates["minimum_indicator"],
    )
    upper(
        "maximum_indicator",
        number("maximum_indicator"),
        common_gates["maximum_indicator"],
    )
    upper(
        "maximum_local_balance_residual",
        number("maximum_local_balance_residual"),
        common_gates["maximum_local_balance_residual"],
    )
    upper(
        "maximum_raw_measure_error",
        number("maximum_raw_measure_error"),
        case_gates["maximum_raw_measure_error"],
    )
    upper(
        "interface_l1",
        number("interface_l1"),
        case_gates["maximum_interface_l1"],
    )
    required_files = [
        history_path,
        details_path / "control_volumes.csv",
        details_path / "edges.csv",
        details_path / "components.csv",
    ]
    for path in required_files:
        size = path.stat().st_size if path.is_file() else 0
        checks.append(
            {
                "metric": f"artifact:{path.name}",
                "actual_bytes": size,
                "relation": "greater_than_zero",
                "passed": size > 0,
            }
        )
    return checks


def write_checksums(output_directory: Path) -> None:
    checksum_path = output_directory / "checksums.txt"
    if checksum_path.exists():
        raise RuntimeError("checksums artifact already exists")
    paths = sorted(
        path
        for path in output_directory.rglob("*")
        if path.is_file() and path != checksum_path and not path.name.endswith(".tmp")
    )
    temporary = checksum_path.with_name(checksum_path.name + ".tmp")
    with temporary.open("x", encoding="utf-8") as output:
        for path in paths:
            output.write(f"{sha256_file(path)}  {path.relative_to(output_directory)}\n")
        output.flush()
        os.fsync(output.fileno())
    os.link(temporary, checksum_path)
    temporary.unlink()


def list_points(registry: dict[str, Any]) -> int:
    cfl_values = registry["common"]["cfl_values"]
    for case_id, case in registry["cases"].items():
        for resolution in case["resolutions"]:
            for cfl in cfl_values:
                print(f"{case_id} resolution={resolution} cfl={cfl}")
    return 0


def verify_checksums(point_directory: Path) -> list[str]:
    checksum_path = point_directory / "checksums.txt"
    if not checksum_path.is_file():
        return ["checksums.txt is missing"]
    failures: list[str] = []
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        if "  " not in line:
            failures.append("checksums.txt has a malformed line")
            continue
        expected, relative = line.split("  ", 1)
        artifact = point_directory / relative
        if not artifact.is_file():
            failures.append(f"missing artifact: {relative}")
        elif sha256_file(artifact) != expected:
            failures.append(f"checksum mismatch: {relative}")
    return failures


def weighted_control_volume_difference(first: Path, second: Path) -> float:
    total_weight = 0.0
    difference = 0.0
    with first.open("r", encoding="utf-8", newline="") as first_file, second.open(
        "r", encoding="utf-8", newline=""
    ) as second_file:
        first_rows = csv.DictReader(first_file)
        second_rows = csv.DictReader(second_file)
        for first_row, second_row in zip(first_rows, second_rows, strict=True):
            if first_row["node"] != second_row["node"]:
                raise RuntimeError("temporal comparison node order differs")
            first_weight = float(first_row["lumped_control_volume"])
            second_weight = float(second_row["lumped_control_volume"])
            if first_weight != second_weight:
                raise RuntimeError("temporal comparison control volumes differ")
            total_weight += first_weight
            difference += first_weight * abs(
                float(first_row["limited_indicator"])
                - float(second_row["limited_indicator"])
            )
    if not total_weight > 0.0:
        raise RuntimeError("temporal comparison has no positive control volume")
    return difference / total_weight


def observed_order(coarse: float, fine: float, ratio: float = 2.0) -> float:
    if not coarse > 0.0 or not fine > 0.0 or not ratio > 1.0:
        return math.nan
    return math.log(coarse / fine) / math.log(ratio)


def convergence_uncertainty(
    medium: float, fine: float, order: float, ratio: float = 2.0
) -> float:
    denominator = ratio**order - 1.0
    if not math.isfinite(order) or denominator <= 0.0:
        return math.inf
    return 1.25 * abs(medium - fine) / denominator


def write_text(path: Path, contents: str) -> None:
    temporary = path.with_name(path.name + ".tmp")
    if path.exists() or temporary.exists():
        raise RuntimeError(f"refusing to replace artifact path: {path}")
    with temporary.open("x", encoding="utf-8") as output:
        output.write(contents)
        output.flush()
        os.fsync(output.fileno())
    os.link(temporary, path)
    temporary.unlink()


def summarize_matrix(arguments: argparse.Namespace, registry: dict[str, Any]) -> int:
    points_root = arguments.points_root.resolve()
    output_directory = arguments.output_dir.resolve()
    registry_path = arguments.registry.resolve()
    if output_directory.exists():
        raise RuntimeError(f"summary output directory must be new: {output_directory}")
    if not points_root.is_dir():
        raise RuntimeError(f"points root is not a directory: {points_root}")
    output_directory.mkdir(parents=True)
    registry_digest = sha256_file(registry_path)
    expected_keys = {
        (case_id, int(resolution), float(cfl))
        for case_id, case in registry["cases"].items()
        for resolution in case["resolutions"]
        for cfl in registry["common"]["cfl_values"]
    }
    points: dict[tuple[str, int, float], dict[str, Any]] = {}
    infrastructure_failures: list[str] = []
    for manifest_path in sorted(points_root.rglob("manifest.json")):
        point_directory = manifest_path.parent
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("matrix_id") != registry["matrix_id"]:
            continue
        key = (
            str(manifest.get("case_id")),
            int(manifest.get("resolution")),
            float(manifest.get("requested_cfl")),
        )
        if key in points:
            infrastructure_failures.append(f"duplicate point: {key}")
            continue
        if manifest.get("registry_sha256") != registry_digest:
            infrastructure_failures.append(f"registry mismatch: {key}")
        comparison_path = point_directory / "comparison.json"
        build_path = point_directory / "build.json"
        result_path = point_directory / "test_result.json"
        if not comparison_path.is_file() or not build_path.is_file() or not result_path.is_file():
            infrastructure_failures.append(f"incomplete point directory: {key}")
            continue
        checksum_failures = verify_checksums(point_directory)
        infrastructure_failures.extend(f"{key}: {item}" for item in checksum_failures)
        comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
        build = json.loads(build_path.read_text(encoding="utf-8"))
        properties = find_test_properties(result_path)
        points[key] = {
            "directory": point_directory,
            "manifest": manifest,
            "comparison": comparison,
            "build": build,
            "properties": properties,
        }
    missing = sorted(expected_keys - set(points))
    unexpected = sorted(set(points) - expected_keys)
    infrastructure_failures.extend(f"missing point: {key}" for key in missing)
    infrastructure_failures.extend(f"unexpected point: {key}" for key in unexpected)

    commits = {point["manifest"]["source_commit"] for point in points.values()}
    binaries = {point["build"]["test_binary_sha256"] for point in points.values()}
    if len(commits) > 1:
        infrastructure_failures.append("point source commits differ")
    if len(binaries) > 1:
        infrastructure_failures.append("point test binary hashes differ")

    point_failures = [
        str(key)
        for key, point in points.items()
        if point["comparison"].get("point_outcome") != "PASS"
    ]
    spatial_studies: list[dict[str, Any]] = []
    temporal_studies: list[dict[str, Any]] = []
    convergence_failures: list[str] = []
    if not missing:
        for case_id, case in registry["cases"].items():
            resolutions = [int(value) for value in case["resolutions"]]
            fixed_cfl = float(case["fixed_cfl_for_space_study"])
            errors = [
                float(points[(case_id, resolution, fixed_cfl)]["properties"]["interface_l1"])
                for resolution in resolutions
            ]
            orders = [
                observed_order(errors[index - 1], errors[index])
                for index in range(1, len(errors))
            ]
            monotone = all(
                errors[index] < errors[index - 1]
                for index in range(1, len(errors))
            )
            minimum_order = float(case["gates"]["minimum_finest_spatial_order"])
            passed = monotone and math.isfinite(orders[-1]) and orders[-1] >= minimum_order
            if not passed:
                convergence_failures.append(f"spatial study failed: {case_id}")
            spatial_studies.append(
                {
                    "case_id": case_id,
                    "fixed_cfl": fixed_cfl,
                    "resolutions": resolutions,
                    "interface_l1": errors,
                    "observed_orders": orders,
                    "finest_error_sequence_uncertainty": convergence_uncertainty(
                        errors[-2], errors[-1], orders[-1]
                    ),
                    "minimum_finest_order": minimum_order,
                    "monotone": monotone,
                    "passed": passed,
                }
            )

            fixed_resolution = int(case["fixed_resolution_for_time_study"])
            cfl_values = [float(value) for value in registry["common"]["cfl_values"]]
            detail_paths = [
                points[(case_id, fixed_resolution, cfl)]["directory"]
                / "checkpoints"
                / "final"
                / "control_volumes.csv"
                for cfl in cfl_values
            ]
            coarse_medium = weighted_control_volume_difference(
                detail_paths[0], detail_paths[1]
            )
            medium_fine = weighted_control_volume_difference(
                detail_paths[1], detail_paths[2]
            )
            order = observed_order(coarse_medium, medium_fine)
            minimum_time_order = float(case["gates"]["minimum_finest_temporal_order"])
            passed = (
                medium_fine < coarse_medium
                and math.isfinite(order)
                and order >= minimum_time_order
            )
            if not passed:
                convergence_failures.append(f"temporal study failed: {case_id}")
            temporal_studies.append(
                {
                    "case_id": case_id,
                    "fixed_resolution": fixed_resolution,
                    "cfl_values": cfl_values,
                    "coarse_to_medium_l1": coarse_medium,
                    "medium_to_fine_l1": medium_fine,
                    "observed_order": order,
                    "finest_solution_uncertainty": convergence_uncertainty(
                        coarse_medium, medium_fine, order
                    ),
                    "minimum_order": minimum_time_order,
                    "passed": passed,
                }
            )

    matrix_passed = (
        not infrastructure_failures
        and not point_failures
        and not convergence_failures
        and set(points) == expected_keys
    )
    if matrix_passed:
        disposition = "PASS"
    elif infrastructure_failures:
        disposition = "INFRASTRUCTURE_FAILURE"
    else:
        disposition = "FAIL_METHOD"
    summary = {
        "schema_version": 1,
        "matrix_id": registry["matrix_id"],
        "registry_sha256": registry_digest,
        "expected_points": len(expected_keys),
        "found_points": len(points),
        "source_commits": sorted(commits),
        "test_binary_hashes": sorted(binaries),
        "infrastructure_failures": infrastructure_failures,
        "point_failures": point_failures,
        "convergence_failures": convergence_failures,
        "spatial_studies": spatial_studies,
        "temporal_studies": temporal_studies,
        "disposition": disposition,
        "matrix_passed": matrix_passed,
    }
    write_json(output_directory / "summary.json", summary)
    lines = [
        "# Conservative phase-transport release matrix",
        "",
        f"Disposition: `{disposition}`",
        "",
        f"Points: {len(points)}/{len(expected_keys)}",
        "",
        "A release pass requires every frozen point and both independent convergence studies.",
        "",
    ]
    for study in spatial_studies:
        lines.append(
            f"- Spatial `{study['case_id']}`: passed={study['passed']}, "
            f"orders={study['observed_orders']}"
        )
    for study in temporal_studies:
        lines.append(
            f"- Temporal `{study['case_id']}`: passed={study['passed']}, "
            f"order={study['observed_order']}"
        )
    if infrastructure_failures or point_failures or convergence_failures:
        lines.extend(["", "Failures:"])
        for failure in infrastructure_failures + point_failures + convergence_failures:
            lines.append(f"- {failure}")
    write_text(output_directory / "summary.md", "\n".join(lines) + "\n")
    write_checksums(output_directory)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if matrix_passed else 2


def run_point(arguments: argparse.Namespace, registry: dict[str, Any]) -> int:
    source_root = arguments.source_root.resolve()
    test_binary = arguments.test_binary.resolve()
    output_directory = arguments.output_dir.resolve()
    registry_path = arguments.registry.resolve()
    if output_directory.exists():
        raise RuntimeError(f"output directory must be new: {output_directory}")
    if not test_binary.is_file() or not os.access(test_binary, os.X_OK):
        raise RuntimeError(f"test binary is not executable: {test_binary}")
    if arguments.case_id not in registry["cases"]:
        raise RuntimeError(f"unknown case: {arguments.case_id}")
    case = registry["cases"][arguments.case_id]
    if arguments.resolution not in case["resolutions"]:
        raise RuntimeError("resolution is outside the frozen matrix")
    if arguments.cfl not in registry["common"]["cfl_values"]:
        raise RuntimeError("CFL is outside the frozen matrix")
    if registry["common"]["threads"] != 1 or registry["common"]["mpi_ranks"] != 1:
        raise RuntimeError("this runner currently requires one thread and one rank")

    tracked_status = git_bytes(
        source_root, "status", "--short", "--untracked-files=no"
    )
    if tracked_status:
        raise RuntimeError("tracked source must be clean before a release point")
    source_commit = git_bytes(source_root, "rev-parse", "HEAD").decode().strip()
    untracked = git_bytes(
        source_root,
        "ls-files",
        "--others",
        "--exclude-standard",
        "-z",
    )
    untracked_count = sum(1 for token in untracked.split(b"\0") if token)
    tracked_diff = git_bytes(source_root, "diff", "--binary", "HEAD", "--")
    index_diff = git_bytes(source_root, "diff", "--cached", "--binary", "HEAD", "--")

    envelope = case["resource_envelopes"][str(arguments.resolution)]
    output_directory.mkdir(parents=True)
    history_path = output_directory / "history.csv"
    details_path = output_directory / "checkpoints" / "final"
    test_result_path = output_directory / "test_result.json"
    stdout_path = output_directory / "stdout.log"
    stderr_path = output_directory / "stderr.log"
    registry_digest = sha256_file(registry_path)

    manifest = {
        "schema_version": 1,
        "matrix_id": registry["matrix_id"],
        "registry_sha256": registry_digest,
        "case_id": arguments.case_id,
        "resolution": arguments.resolution,
        "requested_cfl": arguments.cfl,
        "dimension": case["dimension"],
        "model_envelope": "one_phase_conservative_indicator_transport_only",
        "maintenance_mode": "disabled",
        "source_commit": source_commit,
        "threads": 1,
        "mpi_ranks": 1,
    }
    write_json(output_directory / "manifest.json", manifest)

    cmake_cache = arguments.cmake_cache
    if cmake_cache is None:
        cmake_cache = find_cmake_cache(test_binary)
    elif cmake_cache is not None:
        cmake_cache = cmake_cache.resolve()
    linked_libraries = subprocess.run(
        ["ldd", str(test_binary)],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    ).stdout
    build = {
        "schema_version": 1,
        "source_commit": source_commit,
        "tracked_diff_sha256": sha256_bytes(tracked_diff),
        "index_diff_sha256": sha256_bytes(index_diff),
        "tracked_tree_clean": True,
        "untracked_path_count": untracked_count,
        "untracked_path_list_sha256": sha256_bytes(untracked),
        "test_binary": str(test_binary),
        "test_binary_sha256": sha256_file(test_binary),
        "cmake_cache": str(cmake_cache) if cmake_cache else None,
        "cmake_cache_sha256": sha256_file(cmake_cache) if cmake_cache else None,
        "selected_cmake_cache": selected_cmake_cache(cmake_cache),
        "linked_libraries": linked_libraries.splitlines(),
        "machine": {
            "platform": platform.platform(),
            "node": platform.node(),
            "processor": platform.processor(),
            "logical_cpu_count": os.cpu_count(),
            "memory_mib": machine_memory_mib(),
        },
    }
    write_json(output_directory / "build.json", build)
    gates = {
        "schema_version": 1,
        "registry_sha256": registry_digest,
        "common": registry["common"]["gates"],
        "case": case["gates"],
        "resource_envelope": envelope,
        "qualification_rule": registry["qualification_rule"],
    }
    write_json(output_directory / "gates.json", gates)

    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(ENVIRONMENT_PREFIX)
    }
    environment.update(
        {
            "OMP_NUM_THREADS": "1",
            "SVMP_PHASE_TRANSPORT_RELEASE_CASE": arguments.case_id,
            "SVMP_PHASE_TRANSPORT_RELEASE_RESOLUTION": str(arguments.resolution),
            "SVMP_PHASE_TRANSPORT_RELEASE_CFL": str(arguments.cfl),
            "SVMP_PHASE_TRANSPORT_RELEASE_HISTORY": str(history_path),
            "SVMP_PHASE_TRANSPORT_RELEASE_DETAILS": str(details_path),
        }
    )
    command = [
        str(test_binary),
        f"--gtest_filter={TEST_FILTER}",
        f"--gtest_output=json:{test_result_path}",
    ]
    resources = run_monitored(
        command,
        environment,
        source_root,
        stdout_path,
        stderr_path,
        int(envelope["wall_time_seconds"]),
        int(envelope["memory_mib"]),
        int(envelope["output_mib"]),
        output_directory,
    )
    run_record = {
        "schema_version": 1,
        "command": command,
        "resource_envelope": envelope,
        "resources": resources,
        "outcome": "PASS" if resources["return_code"] == 0 else "FAIL",
    }
    write_json(output_directory / "run.json", run_record)

    checks: list[dict[str, Any]] = []
    failure_reason: str | None = resources["termination_reason"]
    if resources["return_code"] == 0 and test_result_path.is_file():
        try:
            properties = find_test_properties(test_result_path)
            checks = evaluate_point(
                properties,
                arguments.case_id,
                arguments.resolution,
                arguments.cfl,
                registry["common"]["gates"],
                case["gates"],
                history_path,
                details_path,
            )
        except (OSError, ValueError, KeyError, RuntimeError) as error:
            failure_reason = str(error)
    elif failure_reason is None:
        failure_reason = "qualification test returned nonzero"
    point_passed = (
        resources["return_code"] == 0
        and checks
        and all(check["passed"] for check in checks)
        and failure_reason is None
    )
    if point_passed:
        release_disposition = registry["qualification_rule"][
            "single_point_release_disposition"
        ]
    elif resources["return_code"] != 0 or resources["termination_reason"]:
        release_disposition = "INFRASTRUCTURE_FAILURE"
    else:
        release_disposition = "FAIL_METHOD"
    comparison = {
        "schema_version": 1,
        "case_id": arguments.case_id,
        "resolution": arguments.resolution,
        "requested_cfl": arguments.cfl,
        "point_outcome": "PASS" if point_passed else "FAIL_METHOD",
        "release_disposition": release_disposition,
        "failure_reason": failure_reason,
        "checks": checks,
        "note": "A passing point cannot qualify the release matrix by itself.",
    }
    write_json(output_directory / "comparison.json", comparison)
    write_checksums(output_directory)
    print(json.dumps(comparison, indent=2, sort_keys=True))
    return 0 if point_passed else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.set_defaults(registry=DEFAULT_REGISTRY)
    subparsers = parser.add_subparsers(dest="action", required=True)
    list_parser = subparsers.add_parser("list", help="list all frozen points")
    list_parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    run_parser = subparsers.add_parser("run", help="run one frozen point")
    run_parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    run_parser.add_argument("--test-binary", type=Path, required=True)
    run_parser.add_argument("--output-dir", type=Path, required=True)
    run_parser.add_argument("--case", dest="case_id", required=True)
    run_parser.add_argument("--resolution", type=int, required=True)
    run_parser.add_argument("--cfl", type=float, required=True)
    run_parser.add_argument(
        "--source-root", type=Path, default=SCRIPT_PATH.parents[3]
    )
    run_parser.add_argument("--cmake-cache", type=Path)
    summary_parser = subparsers.add_parser(
        "summarize", help="verify and summarize all frozen points"
    )
    summary_parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    summary_parser.add_argument("--points-root", type=Path, required=True)
    summary_parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> int:
    parser = build_parser()
    arguments = parser.parse_args()
    registry_path = arguments.registry.resolve()
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    if registry.get("schema_version") != 1:
        raise RuntimeError("unsupported release-matrix schema")
    if arguments.action == "list":
        return list_points(registry)
    if arguments.action == "summarize":
        return summarize_matrix(arguments, registry)
    return run_point(arguments, registry)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)

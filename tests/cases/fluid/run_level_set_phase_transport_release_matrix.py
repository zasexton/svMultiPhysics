#!/usr/bin/env python3
"""Run and summarize every point in the frozen phase-transport release matrix.

The campaign always visits all 18 points in deterministic order. A failed or
inconclusive point is retained and does not prevent later points from running.
The existing point runner remains the authority for resource limits, point
gates, artifacts, and independent spatial and temporal convergence summaries.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any


SCRIPT_PATH = Path(__file__).resolve()
DEFAULT_REGISTRY = SCRIPT_PATH.with_name(
    "level_set_phase_transport_release_matrix.json"
)
DEFAULT_POINT_RUNNER = SCRIPT_PATH.with_name(
    "run_level_set_phase_transport_release.py"
)
EXPECTED_REGISTRY_SHA256 = (
    "69892249dc9ead90ee90ebaf113427506dafe6fb52b8103f1c9497653e35585a"
)
EXPECTED_POINT_RUNNER_SHA256 = (
    "6244927b6accf7ca0f3acfb8e6fd9b8cc2ac4608d0fc7b4b167d5fd022d2c2d4"
)
EXPECTED_MATRIX_ID = "level_set_phase_transport_release_v1"
EXPECTED_MATRIX_STATUS = "FROZEN_BEFORE_COMPLETE_MATRIX"
EXPECTED_CFL_VALUES = [0.5, 0.25, 0.125]
EXPECTED_CASES = {
    "translating_drop_2d": {
        "dimension": 2,
        "resolutions": [64, 128, 256],
        "fixed_cfl_for_space_study": 0.125,
        "fixed_resolution_for_time_study": 256,
    },
    "enright_3d": {
        "dimension": 3,
        "resolutions": [32, 64, 128],
        "fixed_cfl_for_space_study": 0.125,
        "fixed_resolution_for_time_study": 128,
    },
}
EXPECTED_QUALIFICATION_RULE = {
    "required_points": (
        "cartesian_product_of_each_case_resolutions_and_common_cfl_values"
    ),
    "release_pass_requires": [
        "all_18_points_complete",
        "all_point_gates_pass",
        "independent_fixed_cfl_space_studies_pass",
        "independent_fixed_resolution_time_studies_pass",
        "every_point_has_history_and_final_flux_ledgers",
    ],
    "single_point_release_disposition": "INCONCLUSIVE_RESOLUTION",
}


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


def write_bytes(path: Path, value: bytes) -> None:
    temporary = path.with_name(path.name + ".tmp")
    if path.exists() or temporary.exists():
        raise RuntimeError(f"refusing to replace artifact path: {path}")
    with temporary.open("xb") as output:
        output.write(value)
        output.flush()
        os.fsync(output.fileno())
    os.link(temporary, path)
    temporary.unlink()


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def machine_memory_mib() -> int | None:
    try:
        for line in Path("/proc/meminfo").read_text(
            encoding="utf-8"
        ).splitlines():
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) // 1024
    except (IndexError, OSError, ValueError):
        return None
    return None


def existing_parent(path: Path) -> Path:
    candidate = path
    while not candidate.exists():
        if candidate.parent == candidate:
            raise RuntimeError(
                f"no existing parent for campaign output: {path}"
            )
        candidate = candidate.parent
    return candidate


def campaign_resource_requirements(
    registry: dict[str, Any]
) -> dict[str, int]:
    maximum_memory_mib = 0
    maximum_point_output_mib = 0
    retained_output_envelope_mib = 0
    sequential_wall_time_envelope_seconds = 0
    cfl_count = len(registry["common"]["cfl_values"])
    for case in registry["cases"].values():
        for resolution in case["resolutions"]:
            envelope = case["resource_envelopes"][str(resolution)]
            maximum_memory_mib = max(
                maximum_memory_mib, int(envelope["memory_mib"])
            )
            maximum_point_output_mib = max(
                maximum_point_output_mib, int(envelope["output_mib"])
            )
            retained_output_envelope_mib += (
                cfl_count * int(envelope["output_mib"])
            )
            sequential_wall_time_envelope_seconds += (
                cfl_count * int(envelope["wall_time_seconds"])
            )
    return {
        "maximum_point_memory_mib": maximum_memory_mib,
        "maximum_point_output_mib": maximum_point_output_mib,
        "retained_output_envelope_mib": retained_output_envelope_mib,
        "sequential_wall_time_envelope_seconds": (
            sequential_wall_time_envelope_seconds
        ),
    }


def git_text(source_root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=source_root,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return result.stdout.strip()


def load_frozen_inputs(
    registry_path: Path, point_runner_path: Path
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not registry_path.is_file():
        raise ValueError(f"release registry is not a file: {registry_path}")
    if not point_runner_path.is_file():
        raise ValueError(f"point runner is not a file: {point_runner_path}")
    if sha256_file(registry_path) != EXPECTED_REGISTRY_SHA256:
        raise ValueError("frozen release registry bytes changed")
    if sha256_file(point_runner_path) != EXPECTED_POINT_RUNNER_SHA256:
        raise ValueError("frozen release point runner bytes changed")

    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    if registry.get("schema_version") != 1:
        raise ValueError("unsupported release-matrix schema")
    if registry.get("matrix_id") != EXPECTED_MATRIX_ID:
        raise ValueError("release matrix id changed")
    if registry.get("status") != EXPECTED_MATRIX_STATUS:
        raise ValueError("release matrix status changed")
    common = registry.get("common")
    if not isinstance(common, dict):
        raise ValueError("release matrix common contract is missing")
    if common.get("cfl_values") != EXPECTED_CFL_VALUES:
        raise ValueError("release matrix CFL values changed")
    if common.get("threads") != 1 or common.get("mpi_ranks") != 1:
        raise ValueError("release points must use one thread and one rank")
    common_gates = common.get("gates")
    if not isinstance(common_gates, dict) or not common_gates:
        raise ValueError("release common gates are missing")

    cases = registry.get("cases")
    if not isinstance(cases, dict) or set(cases) != set(EXPECTED_CASES):
        raise ValueError("release case set changed")
    points: list[dict[str, Any]] = []
    point_keys: set[tuple[str, int, float]] = set()
    for case_id, expected in EXPECTED_CASES.items():
        case = cases[case_id]
        if not isinstance(case, dict):
            raise ValueError(f"release case is invalid: {case_id}")
        for key, value in expected.items():
            if case.get(key) != value:
                raise ValueError(f"release case contract changed: {case_id}.{key}")
        gates = case.get("gates")
        envelopes = case.get("resource_envelopes")
        if not isinstance(gates, dict) or not gates:
            raise ValueError(f"release case gates are missing: {case_id}")
        if not isinstance(envelopes, dict) or set(envelopes) != {
            str(value) for value in expected["resolutions"]
        }:
            raise ValueError(f"release resource envelopes changed: {case_id}")
        for resolution in expected["resolutions"]:
            envelope = envelopes[str(resolution)]
            if (
                not isinstance(envelope, dict)
                or set(envelope)
                != {"wall_time_seconds", "memory_mib", "output_mib"}
                or any(
                    not isinstance(value, int) or value <= 0
                    for value in envelope.values()
                )
            ):
                raise ValueError(
                    f"invalid release resource envelope: {case_id}.{resolution}"
                )
            for cfl in EXPECTED_CFL_VALUES:
                key = (case_id, resolution, cfl)
                if key in point_keys:
                    raise ValueError(f"duplicate release point: {key}")
                point_keys.add(key)
                points.append(
                    {
                        "case_id": case_id,
                        "resolution": resolution,
                        "cfl": cfl,
                        "point_id": point_id(case_id, resolution, cfl),
                    }
                )
    if len(points) != 18:
        raise ValueError("release matrix must contain exactly 18 points")
    if registry.get("qualification_rule") != EXPECTED_QUALIFICATION_RULE:
        raise ValueError("release qualification rule changed")
    return registry, points


def point_id(case_id: str, resolution: int, cfl: float) -> str:
    cfl_token = format(cfl, "g").replace(".", "p")
    return f"{case_id}_resolution_{resolution:04d}_cfl_{cfl_token}"


def read_json_if_present(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, UnicodeError):
        return None
    return value if isinstance(value, dict) else None


def run_logged(
    command: list[str],
    working_directory: Path,
    stdout_path: Path,
    stderr_path: Path,
) -> tuple[int, str, str]:
    if stdout_path.exists() or stderr_path.exists():
        raise RuntimeError("refusing to replace launcher log")
    started = utc_timestamp()
    with stdout_path.open("xb") as stdout_file, stderr_path.open(
        "xb"
    ) as stderr_file:
        result = subprocess.run(
            command,
            cwd=working_directory,
            stdout=stdout_file,
            stderr=stderr_file,
            check=False,
        )
    return result.returncode, started, utc_timestamp()


def write_campaign_checksums(output_directory: Path) -> None:
    checksum_path = output_directory / "campaign_checksums.txt"
    if checksum_path.exists():
        raise RuntimeError("campaign checksum artifact already exists")
    paths: set[Path] = {
        output_directory / "campaign_manifest.json",
        output_directory / "campaign_summary.json",
    }
    paths.update((output_directory / "launch_records").glob("*.json"))
    paths.update((output_directory / "launch_logs").glob("*"))
    paths.update((output_directory / "frozen_inputs").glob("*"))
    for point_directory in (output_directory / "points").iterdir():
        if not point_directory.is_dir():
            continue
        point_checksums = point_directory / "checksums.txt"
        if point_checksums.is_file():
            paths.add(point_checksums)
        else:
            paths.update(
                path for path in point_directory.rglob("*") if path.is_file()
            )
    matrix_checksums = output_directory / "matrix_summary" / "checksums.txt"
    if matrix_checksums.is_file():
        paths.add(matrix_checksums)
    elif matrix_checksums.parent.is_dir():
        paths.update(
            path for path in matrix_checksums.parent.rglob("*") if path.is_file()
        )
    existing = sorted(
        path for path in paths if path.is_file() and path != checksum_path
    )
    temporary = checksum_path.with_name(checksum_path.name + ".tmp")
    with temporary.open("x", encoding="utf-8") as output:
        for path in existing:
            output.write(
                f"{sha256_file(path)}  "
                f"{path.relative_to(output_directory)}\n"
            )
        output.flush()
        os.fsync(output.fileno())
    os.link(temporary, checksum_path)
    temporary.unlink()


def validate_action(arguments: argparse.Namespace) -> int:
    registry_path = arguments.registry.resolve()
    point_runner_path = arguments.point_runner.resolve()
    registry, points = load_frozen_inputs(
        registry_path, point_runner_path
    )
    resources = campaign_resource_requirements(registry)
    print(
        json.dumps(
            {
                "matrix_id": registry["matrix_id"],
                "status": registry["status"],
                "expected_point_count": len(points),
                "case_count": len(registry["cases"]),
                "cfl_count": len(registry["common"]["cfl_values"]),
                "registry_sha256": sha256_file(registry_path),
                "point_runner_sha256": sha256_file(point_runner_path),
                "failure_policy": "retain_and_continue",
                "resource_requirements": resources,
                "outcome": "PASS",
            },
            sort_keys=True,
        )
    )
    return 0


def run_action(arguments: argparse.Namespace) -> int:
    registry_path = arguments.registry.resolve()
    point_runner_path = arguments.point_runner.resolve()
    source_root = arguments.source_root.resolve()
    test_binary = arguments.test_binary.resolve()
    output_directory = arguments.output_dir.resolve()
    python_executable = arguments.python_executable.resolve()
    registry, points = load_frozen_inputs(
        registry_path, point_runner_path
    )
    resource_requirements = campaign_resource_requirements(registry)
    if output_directory.exists():
        raise RuntimeError(
            f"campaign output directory must be new: {output_directory}"
        )
    if not source_root.is_dir():
        raise RuntimeError(f"source root is not a directory: {source_root}")
    if not test_binary.is_file() or not os.access(test_binary, os.X_OK):
        raise RuntimeError(f"test binary is not executable: {test_binary}")
    if not python_executable.is_file() or not os.access(
        python_executable, os.X_OK
    ):
        raise RuntimeError(
            f"Python executable is not executable: {python_executable}"
        )
    memory_mib = machine_memory_mib()
    output_filesystem = existing_parent(output_directory)
    available_disk_mib = (
        shutil.disk_usage(output_filesystem).free // (1024 * 1024)
    )
    resource_preflight = {
        **resource_requirements,
        "machine_memory_mib": memory_mib,
        "output_filesystem": str(output_filesystem),
        "available_disk_mib": available_disk_mib,
        "memory_sufficient": (
            memory_mib is not None
            and memory_mib
            >= resource_requirements["maximum_point_memory_mib"]
        ),
        "retained_output_capacity_sufficient": (
            available_disk_mib
            >= resource_requirements["retained_output_envelope_mib"]
        ),
    }
    if not resource_preflight["memory_sufficient"]:
        raise RuntimeError(
            "campaign host memory is below the largest frozen point envelope"
        )
    if not resource_preflight["retained_output_capacity_sufficient"]:
        raise RuntimeError(
            "campaign output filesystem cannot retain all frozen point "
            "output envelopes"
        )
    tracked_status = git_text(
        source_root, "status", "--short", "--untracked-files=no"
    )
    if tracked_status:
        raise RuntimeError(
            "tracked source must be clean before a release campaign"
        )

    source_commit = git_text(source_root, "rev-parse", "HEAD")
    source_tree = git_text(source_root, "rev-parse", "HEAD^{tree}")
    initial_binary_sha256 = sha256_file(test_binary)
    output_directory.mkdir(parents=True)
    points_root = output_directory / "points"
    launch_records = output_directory / "launch_records"
    launch_logs = output_directory / "launch_logs"
    frozen_inputs = output_directory / "frozen_inputs"
    points_root.mkdir()
    launch_records.mkdir()
    launch_logs.mkdir()
    frozen_inputs.mkdir()
    frozen_registry = frozen_inputs / registry_path.name
    frozen_point_runner = frozen_inputs / point_runner_path.name
    frozen_orchestrator = frozen_inputs / SCRIPT_PATH.name
    write_bytes(frozen_registry, registry_path.read_bytes())
    write_bytes(frozen_point_runner, point_runner_path.read_bytes())
    write_bytes(frozen_orchestrator, SCRIPT_PATH.read_bytes())
    manifest = {
        "schema_version": 1,
        "campaign_id": "level_set_phase_transport_release_complete_v1",
        "matrix_id": registry["matrix_id"],
        "matrix_status": registry["status"],
        "registry": str(registry_path),
        "registry_sha256": sha256_file(registry_path),
        "point_runner": str(point_runner_path),
        "point_runner_sha256": sha256_file(point_runner_path),
        "orchestrator": str(SCRIPT_PATH),
        "orchestrator_sha256": sha256_file(SCRIPT_PATH),
        "frozen_input_snapshots": {
            "registry": str(frozen_registry.relative_to(output_directory)),
            "point_runner": str(
                frozen_point_runner.relative_to(output_directory)
            ),
            "orchestrator": str(
                frozen_orchestrator.relative_to(output_directory)
            ),
        },
        "source_root": str(source_root),
        "source_commit": source_commit,
        "source_tree": source_tree,
        "test_binary": str(test_binary),
        "test_binary_sha256": initial_binary_sha256,
        "python_executable": str(python_executable),
        "expected_point_count": len(points),
        "execution_order": points,
        "failure_policy": (
            "retain every failed or inconclusive point and continue through "
            "the complete frozen matrix"
        ),
        "single_point_release_disposition": (
            registry["qualification_rule"][
                "single_point_release_disposition"
            ]
        ),
        "resource_preflight": resource_preflight,
        "started_utc": utc_timestamp(),
    }
    write_json(output_directory / "campaign_manifest.json", manifest)

    point_results: list[dict[str, Any]] = []
    interrupted = False
    try:
        for point in points:
            point_directory = points_root / point["point_id"]
            command = [
                str(python_executable),
                str(point_runner_path),
                "run",
                "--registry",
                str(registry_path),
                "--test-binary",
                str(test_binary),
                "--output-dir",
                str(point_directory),
                "--case",
                point["case_id"],
                "--resolution",
                str(point["resolution"]),
                "--cfl",
                format(point["cfl"], "g"),
                "--source-root",
                str(source_root),
            ]
            stdout_path = (
                launch_logs / f"{point['point_id']}_stdout.txt"
            )
            stderr_path = (
                launch_logs / f"{point['point_id']}_stderr.txt"
            )
            return_code, started, finished = run_logged(
                command, source_root, stdout_path, stderr_path
            )
            comparison = read_json_if_present(
                point_directory / "comparison.json"
            )
            point_result = {
                **point,
                "command": command,
                "started_utc": started,
                "finished_utc": finished,
                "return_code": return_code,
                "comparison_present": comparison is not None,
                "point_outcome": (
                    comparison.get("point_outcome")
                    if comparison is not None
                    else None
                ),
                "release_disposition": (
                    comparison.get("release_disposition")
                    if comparison is not None
                    else "INFRASTRUCTURE_FAILURE"
                ),
            }
            write_json(
                launch_records / f"{point['point_id']}.json",
                point_result,
            )
            point_results.append(point_result)
    except KeyboardInterrupt:
        interrupted = True

    matrix_summary_return_code: int | None = None
    matrix_summary_started: str | None = None
    matrix_summary_finished: str | None = None
    matrix_summary: dict[str, Any] | None = None
    if not interrupted:
        matrix_summary_directory = output_directory / "matrix_summary"
        summary_command = [
            str(python_executable),
            str(point_runner_path),
            "summarize",
            "--registry",
            str(registry_path),
            "--points-root",
            str(points_root),
            "--output-dir",
            str(matrix_summary_directory),
        ]
        (
            matrix_summary_return_code,
            matrix_summary_started,
            matrix_summary_finished,
        ) = run_logged(
            summary_command,
            source_root,
            launch_logs / "matrix_summary_stdout.txt",
            launch_logs / "matrix_summary_stderr.txt",
        )
        matrix_summary = read_json_if_present(
            matrix_summary_directory / "summary.json"
        )
        write_json(
            launch_records / "matrix_summary.json",
            {
                "command": summary_command,
                "started_utc": matrix_summary_started,
                "finished_utc": matrix_summary_finished,
                "return_code": matrix_summary_return_code,
                "summary_present": matrix_summary is not None,
                "disposition": (
                    matrix_summary.get("disposition")
                    if matrix_summary is not None
                    else "INFRASTRUCTURE_FAILURE"
                ),
            },
        )

    final_commit = git_text(source_root, "rev-parse", "HEAD")
    final_tree = git_text(source_root, "rev-parse", "HEAD^{tree}")
    final_tracked_status = git_text(
        source_root, "status", "--short", "--untracked-files=no"
    )
    final_binary_sha256 = (
        sha256_file(test_binary) if test_binary.is_file() else None
    )
    final_registry_sha256 = (
        sha256_file(registry_path) if registry_path.is_file() else None
    )
    final_point_runner_sha256 = (
        sha256_file(point_runner_path)
        if point_runner_path.is_file()
        else None
    )
    final_orchestrator_sha256 = sha256_file(SCRIPT_PATH)
    final_checks = {
        "source_commit_unchanged": final_commit == source_commit,
        "source_tree_unchanged": final_tree == source_tree,
        "tracked_source_clean": final_tracked_status == "",
        "test_binary_unchanged": (
            final_binary_sha256 == initial_binary_sha256
        ),
        "registry_unchanged": (
            final_registry_sha256 == EXPECTED_REGISTRY_SHA256
        ),
        "point_runner_unchanged": (
            final_point_runner_sha256 == EXPECTED_POINT_RUNNER_SHA256
        ),
        "orchestrator_unchanged": (
            final_orchestrator_sha256 == manifest["orchestrator_sha256"]
        ),
    }
    completed_ids = {result["point_id"] for result in point_results}
    not_run = [
        point["point_id"]
        for point in points
        if point["point_id"] not in completed_ids
    ]
    disposition = (
        matrix_summary.get("disposition")
        if matrix_summary is not None
        else "INCOMPLETE"
    )
    if interrupted or not_run:
        disposition = "INCOMPLETE"
    elif not all(final_checks.values()):
        disposition = "INFRASTRUCTURE_FAILURE"
    elif matrix_summary_return_code is None or matrix_summary is None:
        disposition = "INFRASTRUCTURE_FAILURE"
    elif matrix_summary_return_code != 0 and disposition == "PASS":
        disposition = "INFRASTRUCTURE_FAILURE"
    campaign_passed = (
        disposition == "PASS"
        and len(point_results) == len(points)
        and matrix_summary_return_code == 0
        and all(final_checks.values())
    )
    dispositions: dict[str, int] = {}
    for result in point_results:
        key = str(result["release_disposition"])
        dispositions[key] = dispositions.get(key, 0) + 1
    campaign_summary = {
        "schema_version": 1,
        "campaign_id": manifest["campaign_id"],
        "matrix_id": registry["matrix_id"],
        "expected_point_count": len(points),
        "attempted_point_count": len(point_results),
        "not_run_point_ids": not_run,
        "interrupted": interrupted,
        "point_release_disposition_counts": dispositions,
        "point_results": point_results,
        "matrix_summary_return_code": matrix_summary_return_code,
        "matrix_summary_disposition": (
            matrix_summary.get("disposition")
            if matrix_summary is not None
            else None
        ),
        "matrix_summary_passed": (
            matrix_summary.get("matrix_passed")
            if matrix_summary is not None
            else False
        ),
        "source_commit_before": source_commit,
        "source_commit_after": final_commit,
        "source_tree_before": source_tree,
        "source_tree_after": final_tree,
        "tracked_status_after": final_tracked_status,
        "test_binary_sha256_before": initial_binary_sha256,
        "test_binary_sha256_after": final_binary_sha256,
        "registry_sha256_after": final_registry_sha256,
        "point_runner_sha256_after": final_point_runner_sha256,
        "orchestrator_sha256_after": final_orchestrator_sha256,
        "final_invariant_checks": final_checks,
        "disposition": disposition,
        "campaign_passed": campaign_passed,
        "finished_utc": utc_timestamp(),
        "scope_note": (
            "A passing complete release matrix is required but is not by "
            "itself WP-6 or Q3 closure."
        ),
    }
    write_json(
        output_directory / "campaign_summary.json", campaign_summary
    )
    write_campaign_checksums(output_directory)
    print(json.dumps(campaign_summary, indent=2, sort_keys=True))
    if interrupted:
        return 130
    return 0 if campaign_passed else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    validate_parser = subparsers.add_parser(
        "validate", help="validate the frozen campaign without running points"
    )
    validate_parser.add_argument(
        "--registry", type=Path, default=DEFAULT_REGISTRY
    )
    validate_parser.add_argument(
        "--point-runner", type=Path, default=DEFAULT_POINT_RUNNER
    )
    run_parser = subparsers.add_parser(
        "run", help="run and summarize all 18 frozen points"
    )
    run_parser.add_argument(
        "--registry", type=Path, default=DEFAULT_REGISTRY
    )
    run_parser.add_argument(
        "--point-runner", type=Path, default=DEFAULT_POINT_RUNNER
    )
    run_parser.add_argument("--test-binary", type=Path, required=True)
    run_parser.add_argument("--output-dir", type=Path, required=True)
    run_parser.add_argument(
        "--source-root", type=Path, default=SCRIPT_PATH.parents[3]
    )
    run_parser.add_argument(
        "--python-executable", type=Path, default=Path(sys.executable)
    )
    return parser


def main() -> int:
    arguments = build_parser().parse_args()
    if arguments.action == "validate":
        return validate_action(arguments)
    return run_action(arguments)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)

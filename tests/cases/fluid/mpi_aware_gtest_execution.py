#!/usr/bin/env python3
"""Route MPI-initializing serial GoogleTest groups through one rank."""

from __future__ import annotations

import copy
from pathlib import Path
from types import ModuleType
from typing import Any, Callable


MPI_SINGLE_RANK_ARGUMENTS = ("--oversubscribe", "-n", "1")


def mpi_initializing_binary_keys(
    groups: list[dict[str, Any]],
    binary_keys: set[str],
) -> set[str]:
    mpi_keys: set[str] = set()
    for group in groups:
        binary = group.get("binary")
        ranks = group.get("mpi_ranks")
        if not isinstance(binary, str) or binary not in binary_keys:
            raise ValueError(
                "GoogleTest execution group cites an unknown binary key"
            )
        if (
            not isinstance(ranks, int)
            or isinstance(ranks, bool)
            or ranks <= 0
        ):
            raise ValueError(
                "GoogleTest execution group has an invalid MPI rank count"
            )
        if ranks > 1:
            mpi_keys.add(binary)
    return mpi_keys


class MPIAwareGTestExecution:
    """Apply the discovery classification to serial group execution."""

    def __init__(
        self,
        runner_module: ModuleType,
        binaries: dict[str, Path],
        groups: list[dict[str, Any]],
        mpi_launcher: Path,
    ) -> None:
        if not binaries:
            raise ValueError("GoogleTest execution requires binary paths")
        resolved = {key: path.resolve() for key, path in binaries.items()}
        if len(set(resolved.values())) != len(resolved):
            raise ValueError(
                "GoogleTest execution binary paths must be unique"
            )
        required_attributes = (
            "run_gtest_group",
            "run_monitored",
            "write_json",
        )
        if any(not hasattr(runner_module, name) for name in required_attributes):
            raise ValueError(
                "GoogleTest execution runner interface is incomplete"
            )
        self._runner = runner_module
        self._parent_run_group: Callable[..., dict[str, Any]] = (
            runner_module.run_gtest_group
        )
        self._run_monitored: Callable[..., dict[str, Any]] = (
            runner_module.run_monitored
        )
        self._write_json: Callable[[Path, Any], None] = (
            runner_module.write_json
        )
        self._binary_paths = resolved
        self._mpi_binary_keys = mpi_initializing_binary_keys(
            groups,
            set(resolved),
        )
        self._mpi_launcher = mpi_launcher.resolve()
        self._active = False

    def contract(self) -> dict[str, Any]:
        all_keys = set(self._binary_paths)
        return {
            "schema_version": 1,
            "mpi_single_rank_arguments": list(MPI_SINGLE_RANK_ARGUMENTS),
            "mpi_single_rank_binary_keys": sorted(self._mpi_binary_keys),
            "direct_binary_keys": sorted(all_keys - self._mpi_binary_keys),
            "mpi_single_rank_monitoring": {
                "launch_mode": "mpi",
                "required_simultaneous_process_samples": 2,
            },
            "inherited_scheduler_process_count_policy": (
                "isolate_with_explicit_single_rank_mpi_world"
            ),
        }

    def _validate_invocation(
        self,
        group: dict[str, Any],
        binaries: dict[str, Path],
        mpi_launcher: Path,
    ) -> tuple[str, int]:
        binary_key = group.get("binary")
        ranks = group.get("mpi_ranks")
        if (
            not isinstance(binary_key, str)
            or binary_key not in self._binary_paths
        ):
            raise ValueError(
                "GoogleTest execution received an undeclared binary key"
            )
        if (
            not isinstance(ranks, int)
            or isinstance(ranks, bool)
            or ranks <= 0
        ):
            raise ValueError(
                "GoogleTest execution received an invalid MPI rank count"
            )
        observed_paths = {
            key: path.resolve() for key, path in binaries.items()
        }
        if observed_paths != self._binary_paths:
            raise ValueError(
                "GoogleTest execution binary inventory changed"
            )
        if mpi_launcher.resolve() != self._mpi_launcher:
            raise ValueError("GoogleTest execution MPI launcher changed")
        return binary_key, ranks

    def __call__(
        self,
        group: dict[str, Any],
        gates: dict[str, Any],
        binaries: dict[str, Path],
        mpi_launcher: Path,
        source_root: Path,
        output_root: Path,
    ) -> dict[str, Any]:
        binary_key, ranks = self._validate_invocation(
            group,
            binaries,
            mpi_launcher,
        )
        if ranks != 1 or binary_key not in self._mpi_binary_keys:
            return self._parent_run_group(
                group,
                gates,
                binaries,
                mpi_launcher,
                source_root,
                output_root,
            )
        if self._active:
            raise RuntimeError(
                "nested MPI-single-rank GoogleTest execution is forbidden"
            )
        if self._runner.run_monitored is not self._run_monitored:
            raise RuntimeError(
                "GoogleTest execution monitor changed after configuration"
            )
        if self._runner.write_json is not self._write_json:
            raise RuntimeError(
                "GoogleTest execution writer changed after configuration"
            )

        result_path = output_root / "groups" / group["id"] / "result.json"
        observed: dict[str, Any] = {
            "call_count": 0,
            "result_write_count": 0,
        }

        def run_mpi_single_rank(
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
            **options: Any,
        ) -> dict[str, Any]:
            observed["call_count"] += 1
            if observed["call_count"] != 1:
                raise RuntimeError(
                    "serial GoogleTest group launched more than once"
                )
            if launch_mode != "direct_serial":
                raise RuntimeError(
                    "MPI-single-rank route expected a direct serial parent"
                )
            if required_simultaneous_process_samples != 1:
                raise RuntimeError(
                    "direct serial parent monitoring contract changed"
                )
            routed_command = [
                str(self._mpi_launcher),
                *MPI_SINGLE_RANK_ARGUMENTS,
                *command,
            ]
            observed["command"] = routed_command
            observed["slurm_nprocs"] = environment.get("SLURM_NPROCS")
            return self._run_monitored(
                routed_command,
                environment,
                working_directory,
                stdout_path,
                stderr_path,
                output_directory,
                wall_time_seconds,
                memory_mib,
                output_mib,
                "mpi",
                required_simultaneous_process_samples=2,
                **options,
            )

        def capture_parent_result(path: Path, value: Any) -> None:
            if path != result_path:
                self._write_json(path, value)
                return
            observed["result_write_count"] += 1
            if observed["result_write_count"] != 1:
                raise RuntimeError(
                    "serial GoogleTest result was written more than once"
                )
            observed["parent_result"] = copy.deepcopy(value)

        self._active = True
        self._runner.run_monitored = run_mpi_single_rank
        self._runner.write_json = capture_parent_result
        try:
            result = self._parent_run_group(
                group,
                gates,
                binaries,
                mpi_launcher,
                source_root,
                output_root,
            )
        finally:
            monitor_changed = (
                self._runner.run_monitored is not run_mpi_single_rank
            )
            writer_changed = (
                self._runner.write_json is not capture_parent_result
            )
            self._runner.run_monitored = self._run_monitored
            self._runner.write_json = self._write_json
            self._active = False
            if monitor_changed or writer_changed:
                raise RuntimeError(
                    "GoogleTest execution hooks changed during launch"
                )
        if observed["call_count"] != 1 or "command" not in observed:
            raise RuntimeError(
                "MPI-single-rank GoogleTest route was not exercised"
            )
        if (
            observed["result_write_count"] != 1
            or observed.get("parent_result") != result
        ):
            raise RuntimeError(
                "MPI-single-rank parent result was not captured exactly"
            )
        result = copy.deepcopy(result)
        result["command"] = observed["command"]
        result["execution_route"] = {
            "binary_key": binary_key,
            "logical_mpi_ranks": 1,
            "launcher_mpi_ranks": 1,
            "mode": "mpi_single_rank",
            "inherited_slurm_nprocs": observed["slurm_nprocs"],
        }
        resources = result.get("resources")
        if not isinstance(resources, dict) or resources.get(
            "launch_mode"
        ) != "mpi":
            raise RuntimeError(
                "MPI-single-rank resource record is incomplete"
            )
        if resources.get("required_simultaneous_process_samples") != 2:
            raise RuntimeError(
                "MPI-single-rank process coverage contract is incomplete"
            )
        self._write_json(result_path, result)
        return result

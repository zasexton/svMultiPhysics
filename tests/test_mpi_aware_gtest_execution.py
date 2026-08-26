#!/usr/bin/env python3
"""Focused tests for scheduler-safe GoogleTest execution."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
HELPER_PATH = (
    REPOSITORY_ROOT
    / "tests/cases/fluid/mpi_aware_gtest_execution.py"
)
SHARED_RUNNER_PATH = (
    REPOSITORY_ROOT
    / "tests/cases/fluid/"
    "run_free_surface_wp2_geometry_qualification.py"
)
ZERO_GATES = {
    "expected_failures": 0,
    "expected_errors": 0,
    "expected_disabled": 0,
    "expected_skipped": 0,
}


def _load(path: Path, name: str):
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"cannot load focused module: {path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


helper = _load(HELPER_PATH, "_focused_mpi_gtest_execution")


def gtest_document(test_name: str) -> dict:
    suite, name = test_name.split(".", 1)
    return {
        "tests": 1,
        "failures": 0,
        "errors": 0,
        "disabled": 0,
        "skipped": 0,
        "testsuites": [
            {
                "name": suite,
                "testsuite": [
                    {
                        "name": name,
                        "status": "RUN",
                        "result": "COMPLETED",
                        "failures": [],
                    }
                ],
            }
        ],
    }


class MPIAwareGTestExecutionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.runner = _load(
            SHARED_RUNNER_PATH,
            f"_focused_shared_execution_{id(self)}",
        )
        self.binaries = {
            "geometry": Path("/scratch/tests/test_geometry"),
            "physics": Path("/scratch/tests/test_physics"),
            "application_mpi": Path(
                "/scratch/tests/test_application_mpi"
            ),
        }
        self.groups = [
            {"binary": "geometry", "mpi_ranks": 1},
            {"binary": "physics", "mpi_ranks": 1},
            {"binary": "physics", "mpi_ranks": 2},
            {"binary": "application_mpi", "mpi_ranks": 2},
        ]
        self.launcher = Path("/opt/mpi/bin/mpiexec")

    def test_contract_reuses_distributed_binary_classification(self) -> None:
        execution = helper.MPIAwareGTestExecution(
            self.runner,
            self.binaries,
            self.groups,
            self.launcher,
        )
        self.assertEqual(
            execution.contract(),
            {
                "schema_version": 1,
                "mpi_single_rank_arguments": [
                    "--oversubscribe",
                    "-n",
                    "1",
                ],
                "mpi_single_rank_binary_keys": [
                    "application_mpi",
                    "physics",
                ],
                "direct_binary_keys": ["geometry"],
                "mpi_single_rank_monitoring": {
                    "launch_mode": "mpi",
                    "required_simultaneous_process_samples": 2,
                },
                "inherited_scheduler_process_count_policy": (
                    "isolate_with_explicit_single_rank_mpi_world"
                ),
            },
        )

    def test_serial_mpi_binary_uses_explicit_single_rank_world(self) -> None:
        test_name = "Suite.Case"
        group = {
            "id": "physics_serial",
            "binary": "physics",
            "mpi_ranks": 1,
            "gtest_output_copies": 1,
            "tests": [test_name],
            "execution": {
                "wall_time_seconds": 60,
                "memory_mib": 256,
                "output_mib": 16,
            },
        }
        captured: dict = {}

        def monitored(
            command,
            environment,
            source_root,
            stdout_path,
            stderr_path,
            group_directory,
            wall_time_seconds,
            memory_mib,
            output_mib,
            launch_mode,
            required_simultaneous_process_samples=1,
            **options,
        ):
            captured.update(
                {
                    "command": command,
                    "environment": environment,
                    "source_root": source_root,
                    "wall_time_seconds": wall_time_seconds,
                    "memory_mib": memory_mib,
                    "output_mib": output_mib,
                    "launch_mode": launch_mode,
                    "samples": required_simultaneous_process_samples,
                    "options": options,
                }
            )
            stdout_path.write_bytes(b"")
            stderr_path.write_bytes(b"")
            (group_directory / "gtest.json").write_text(
                json.dumps(gtest_document(test_name)),
                encoding="utf-8",
            )
            return {
                "return_code": 0,
                "termination_reason": None,
                "resource_monitoring_outcome": "PASS",
                "launch_mode": launch_mode,
                "required_simultaneous_process_samples": (
                    required_simultaneous_process_samples
                ),
                "termination": None,
            }

        self.runner.run_monitored = monitored
        execution = helper.MPIAwareGTestExecution(
            self.runner,
            self.binaries,
            self.groups,
            self.launcher,
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with mock.patch.dict(
                os.environ,
                {"SLURM_NPROCS": "10"},
            ):
                result = execution(
                    group,
                    ZERO_GATES,
                    self.binaries,
                    self.launcher,
                    REPOSITORY_ROOT,
                    output,
                )
            written = json.loads(
                (
                    output
                    / "groups/physics_serial/result.json"
                ).read_text(encoding="utf-8")
            )

        self.assertEqual(
            captured["command"][:4],
            [
                str(self.launcher.resolve()),
                "--oversubscribe",
                "-n",
                "1",
            ],
        )
        self.assertEqual(
            captured["command"][4],
            str(self.binaries["physics"]),
        )
        self.assertEqual(captured["launch_mode"], "mpi")
        self.assertEqual(captured["samples"], 2)
        self.assertEqual(captured["environment"]["SLURM_NPROCS"], "10")
        self.assertEqual(captured["options"], {})
        self.assertEqual(result["outcome"], "PASS")
        self.assertEqual(result, written)
        self.assertEqual(
            result["execution_route"],
            {
                "binary_key": "physics",
                "logical_mpi_ranks": 1,
                "launcher_mpi_ranks": 1,
                "mode": "mpi_single_rank",
                "inherited_slurm_nprocs": "10",
            },
        )

    def test_direct_serial_binary_retains_direct_route(self) -> None:
        calls = []

        def parent(*arguments):
            calls.append(arguments)
            return {"outcome": "PASS", "command": ["direct"]}

        fake_runner = types.SimpleNamespace(
            run_gtest_group=parent,
            run_monitored=lambda *args, **kwargs: {},
            write_json=lambda *args, **kwargs: None,
        )
        execution = helper.MPIAwareGTestExecution(
            fake_runner,
            self.binaries,
            self.groups,
            self.launcher,
        )
        group = {"binary": "geometry", "mpi_ranks": 1}
        result = execution(
            group,
            ZERO_GATES,
            self.binaries,
            self.launcher,
            REPOSITORY_ROOT,
            Path("/scratch/tests/output"),
        )
        self.assertEqual(result["command"], ["direct"])
        self.assertEqual(len(calls), 1)

    def test_incomplete_single_rank_monitor_record_fails_closed(self) -> None:
        test_name = "Suite.Case"
        group = {
            "id": "physics_serial_incomplete",
            "binary": "physics",
            "mpi_ranks": 1,
            "gtest_output_copies": 1,
            "tests": [test_name],
            "execution": {
                "wall_time_seconds": 60,
                "memory_mib": 256,
                "output_mib": 16,
            },
        }

        def monitored(
            _command,
            _environment,
            _source_root,
            stdout_path,
            stderr_path,
            group_directory,
            *_arguments,
            **_options,
        ):
            stdout_path.write_bytes(b"")
            stderr_path.write_bytes(b"")
            (group_directory / "gtest.json").write_text(
                json.dumps(gtest_document(test_name)),
                encoding="utf-8",
            )
            return {
                "return_code": 0,
                "termination_reason": None,
                "resource_monitoring_outcome": "PASS",
                "launch_mode": "direct_serial",
                "required_simultaneous_process_samples": 1,
                "termination": None,
            }

        self.runner.run_monitored = monitored
        execution = helper.MPIAwareGTestExecution(
            self.runner,
            self.binaries,
            self.groups,
            self.launcher,
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with self.assertRaisesRegex(
                RuntimeError,
                "resource record is incomplete",
            ):
                execution(
                    group,
                    ZERO_GATES,
                    self.binaries,
                    self.launcher,
                    REPOSITORY_ROOT,
                    output,
                )
            self.assertFalse(
                (
                    output
                    / "groups/physics_serial_incomplete/result.json"
                ).exists()
            )

    def test_multi_rank_group_retains_parent_route(self) -> None:
        calls = []

        def parent(*arguments):
            calls.append(arguments)
            return {"outcome": "PASS", "command": ["mpi", "2"]}

        fake_runner = types.SimpleNamespace(
            run_gtest_group=parent,
            run_monitored=lambda *args, **kwargs: {},
            write_json=lambda *args, **kwargs: None,
        )
        execution = helper.MPIAwareGTestExecution(
            fake_runner,
            self.binaries,
            self.groups,
            self.launcher,
        )
        group = {"binary": "physics", "mpi_ranks": 2}
        result = execution(
            group,
            ZERO_GATES,
            self.binaries,
            self.launcher,
            REPOSITORY_ROOT,
            Path("/scratch/tests/output"),
        )
        self.assertEqual(result["command"], ["mpi", "2"])
        self.assertEqual(len(calls), 1)

    def test_invalid_groups_and_binary_paths_fail_closed(self) -> None:
        for invalid_rank in (None, True, 0, -1, 1.5, "2"):
            with self.subTest(invalid_rank=invalid_rank):
                with self.assertRaisesRegex(ValueError, "invalid MPI rank"):
                    helper.MPIAwareGTestExecution(
                        self.runner,
                        self.binaries,
                        [
                            {
                                "binary": "geometry",
                                "mpi_ranks": invalid_rank,
                            }
                        ],
                        self.launcher,
                    )
        with self.assertRaisesRegex(ValueError, "unknown binary key"):
            helper.MPIAwareGTestExecution(
                self.runner,
                self.binaries,
                [{"binary": "unknown", "mpi_ranks": 2}],
                self.launcher,
            )
        with self.assertRaisesRegex(ValueError, "paths must be unique"):
            helper.MPIAwareGTestExecution(
                self.runner,
                {
                    "geometry": Path("/scratch/tests/shared"),
                    "physics": Path("/scratch/tests/shared"),
                },
                [],
                self.launcher,
            )

    def test_invocation_inventory_and_launcher_drift_fail_closed(self) -> None:
        execution = helper.MPIAwareGTestExecution(
            self.runner,
            self.binaries,
            self.groups,
            self.launcher,
        )
        group = {"binary": "geometry", "mpi_ranks": 1}
        changed = dict(self.binaries)
        changed["geometry"] = Path("/scratch/tests/changed")
        with self.assertRaisesRegex(ValueError, "inventory changed"):
            execution(
                group,
                ZERO_GATES,
                changed,
                self.launcher,
                REPOSITORY_ROOT,
                Path("/scratch/tests/output"),
            )
        with self.assertRaisesRegex(ValueError, "launcher changed"):
            execution(
                group,
                ZERO_GATES,
                self.binaries,
                Path("/different/mpiexec"),
                REPOSITORY_ROOT,
                Path("/scratch/tests/output"),
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)

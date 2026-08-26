#!/usr/bin/env python3
"""Focused tests for scheduler-safe GoogleTest discovery."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import sys
import unittest
from unittest import mock


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
HELPER_PATH = (
    REPOSITORY_ROOT
    / "tests/cases/fluid/mpi_aware_gtest_discovery.py"
)


def _load_helper():
    specification = importlib.util.spec_from_file_location(
        "_focused_mpi_gtest_discovery",
        HELPER_PATH,
    )
    if specification is None or specification.loader is None:
        raise RuntimeError("cannot load the GoogleTest discovery helper")
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


helper = _load_helper()


class MPIAwareGTestDiscoveryTests(unittest.TestCase):
    def test_commands_distinguish_direct_and_single_rank_mpi_routes(
        self,
    ) -> None:
        binary = Path("/scratch/tests/test_binary")
        launcher = Path("/opt/mpi/bin/mpiexec")
        self.assertEqual(
            helper.gtest_list_command(binary),
            [str(binary), "--gtest_list_tests"],
        )
        self.assertEqual(
            helper.gtest_list_command(
                binary,
                mpi_launcher=launcher,
            ),
            [
                str(launcher),
                "--oversubscribe",
                "-n",
                "1",
                str(binary),
                "--gtest_list_tests",
            ],
        )

    def test_listing_parser_rejects_duplicate_identifiers(self) -> None:
        listing = "Suite.\n  First\n  Second\nOther.\n  Case\n"
        self.assertEqual(
            helper.parse_listed_gtests(listing),
            {"Suite.First", "Suite.Second", "Other.Case"},
        )
        with self.assertRaisesRegex(ValueError, "duplicate listed"):
            helper.parse_listed_gtests(
                "Suite.\n  Case\n  Case\n"
            )

    def test_routing_uses_mpi_for_any_binary_with_a_distributed_group(
        self,
    ) -> None:
        binaries = {
            "geometry": Path("/scratch/tests/test_geometry"),
            "physics": Path("/scratch/tests/test_physics"),
            "application_mpi": Path(
                "/scratch/tests/test_application_mpi"
            ),
        }
        groups = [
            {"binary": "geometry", "mpi_ranks": 1},
            {"binary": "physics", "mpi_ranks": 1},
            {"binary": "physics", "mpi_ranks": 2},
            {"binary": "application_mpi", "mpi_ranks": 2},
        ]
        discovery = helper.MPIAwareGTestDiscovery(
            binaries,
            groups,
            Path("/opt/mpi/bin/mpiexec"),
        )
        completed = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout="Suite.\n  Case\n",
            stderr="",
        )
        with mock.patch.object(
            helper.subprocess,
            "run",
            return_value=completed,
        ) as execute:
            self.assertEqual(
                discovery(binaries["geometry"]),
                {"Suite.Case"},
            )
            self.assertEqual(
                discovery(binaries["physics"]),
                {"Suite.Case"},
            )
            self.assertEqual(
                discovery(binaries["application_mpi"]),
                {"Suite.Case"},
            )
        commands = [call.args[0] for call in execute.call_args_list]
        self.assertEqual(
            commands[0],
            [
                str(binaries["geometry"].resolve()),
                "--gtest_list_tests",
            ],
        )
        for command in commands[1:]:
            self.assertEqual(
                command[:4],
                [
                    "/opt/mpi/bin/mpiexec",
                    "--oversubscribe",
                    "-n",
                    "1",
                ],
            )
        for call in execute.call_args_list:
            self.assertEqual(
                call.kwargs,
                {
                    "check": True,
                    "stdout": subprocess.PIPE,
                    "stderr": subprocess.PIPE,
                    "text": True,
                    "timeout": 60,
                },
            )
        self.assertEqual(
            discovery.contract(),
            {
                "schema_version": 1,
                "timeout_seconds": 60,
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
            },
        )

    def test_invalid_or_undeclared_routes_fail_closed(self) -> None:
        binaries = {"geometry": Path("/scratch/tests/test_geometry")}
        with self.assertRaisesRegex(ValueError, "unknown binary key"):
            helper.MPIAwareGTestDiscovery(
                binaries,
                [{"binary": "physics", "mpi_ranks": 2}],
                Path("/opt/mpi/bin/mpiexec"),
            )
        discovery = helper.MPIAwareGTestDiscovery(
            binaries,
            [{"binary": "geometry", "mpi_ranks": 1}],
            Path("/opt/mpi/bin/mpiexec"),
        )
        with self.assertRaisesRegex(ValueError, "undeclared binary path"):
            discovery(Path("/scratch/tests/test_other"))

    def test_invalid_rank_and_duplicate_path_contracts_fail_closed(
        self,
    ) -> None:
        binaries = {"geometry": Path("/scratch/tests/test_geometry")}
        for invalid_rank in (None, True, 0, -1, 1.5, "2"):
            with self.subTest(invalid_rank=invalid_rank):
                with self.assertRaisesRegex(ValueError, "invalid MPI rank"):
                    helper.MPIAwareGTestDiscovery(
                        binaries,
                        [
                            {
                                "binary": "geometry",
                                "mpi_ranks": invalid_rank,
                            }
                        ],
                        Path("/opt/mpi/bin/mpiexec"),
                    )
        with self.assertRaisesRegex(ValueError, "paths must be unique"):
            helper.MPIAwareGTestDiscovery(
                {
                    "geometry": Path("/scratch/tests/test_binary"),
                    "physics": Path("/scratch/tests/test_binary"),
                },
                [],
                Path("/opt/mpi/bin/mpiexec"),
            )

    def test_nonpositive_discovery_timeouts_fail_closed(self) -> None:
        binary = Path("/scratch/tests/test_geometry")
        for invalid_timeout in (0, -1):
            with self.subTest(invalid_timeout=invalid_timeout):
                with self.assertRaisesRegex(ValueError, "must be positive"):
                    helper.listed_gtests(
                        binary,
                        timeout_seconds=invalid_timeout,
                    )
                with self.assertRaisesRegex(ValueError, "must be positive"):
                    helper.MPIAwareGTestDiscovery(
                        {"geometry": binary},
                        [{"binary": "geometry", "mpi_ranks": 1}],
                        Path("/opt/mpi/bin/mpiexec"),
                        timeout_seconds=invalid_timeout,
                    )


if __name__ == "__main__":
    unittest.main(verbosity=2)

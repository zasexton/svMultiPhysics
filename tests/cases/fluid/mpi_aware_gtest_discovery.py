#!/usr/bin/env python3
"""Discover GoogleTest selectors without inheriting a multi-task MPI world."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any


DEFAULT_TIMEOUT_SECONDS = 60
MPI_SINGLE_RANK_ARGUMENTS = ("--oversubscribe", "-n", "1")


def gtest_list_command(
    binary: Path,
    *,
    mpi_launcher: Path | None = None,
) -> list[str]:
    command = [str(binary), "--gtest_list_tests"]
    if mpi_launcher is None:
        return command
    return [
        str(mpi_launcher),
        *MPI_SINGLE_RANK_ARGUMENTS,
        *command,
    ]


def parse_listed_gtests(output: str) -> set[str]:
    suite = ""
    names: set[str] = set()
    for line in output.splitlines():
        if line and not line[0].isspace():
            suite = line.split("#", 1)[0].strip().removesuffix(".")
            continue
        test = line.split("#", 1)[0].strip()
        if not suite or not test:
            continue
        identifier = f"{suite}.{test}"
        if identifier in names:
            raise ValueError(
                f"duplicate listed GoogleTest identifier: {identifier}"
            )
        names.add(identifier)
    return names


def listed_gtests(
    binary: Path,
    *,
    mpi_launcher: Path | None = None,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
) -> set[str]:
    if timeout_seconds <= 0:
        raise ValueError("GoogleTest discovery timeout must be positive")
    result = subprocess.run(
        gtest_list_command(binary, mpi_launcher=mpi_launcher),
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout_seconds,
    )
    return parse_listed_gtests(result.stdout)


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
                "GoogleTest discovery group cites an unknown binary key"
            )
        if (
            not isinstance(ranks, int)
            or isinstance(ranks, bool)
            or ranks <= 0
        ):
            raise ValueError(
                "GoogleTest discovery group has an invalid MPI rank count"
            )
        if ranks > 1:
            mpi_keys.add(binary)
    return mpi_keys


class MPIAwareGTestDiscovery:
    """Route MPI-initializing binaries through an explicit one-rank world."""

    def __init__(
        self,
        binaries: dict[str, Path],
        groups: list[dict[str, Any]],
        mpi_launcher: Path,
        *,
        timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("GoogleTest discovery timeout must be positive")
        if not binaries:
            raise ValueError("GoogleTest discovery requires binary paths")
        resolved = {key: path.resolve() for key, path in binaries.items()}
        if len(set(resolved.values())) != len(resolved):
            raise ValueError(
                "GoogleTest discovery binary paths must be unique"
            )
        self._binary_key_by_path = {
            path: key for key, path in resolved.items()
        }
        self._mpi_binary_keys = mpi_initializing_binary_keys(
            groups,
            set(resolved),
        )
        self._mpi_launcher = mpi_launcher.resolve()
        self._timeout_seconds = timeout_seconds

    def __call__(self, binary: Path) -> set[str]:
        resolved = binary.resolve()
        binary_key = self._binary_key_by_path.get(resolved)
        if binary_key is None:
            raise ValueError(
                "GoogleTest discovery received an undeclared binary path"
            )
        launcher = (
            self._mpi_launcher
            if binary_key in self._mpi_binary_keys
            else None
        )
        return listed_gtests(
            resolved,
            mpi_launcher=launcher,
            timeout_seconds=self._timeout_seconds,
        )

    def contract(self) -> dict[str, Any]:
        all_keys = set(self._binary_key_by_path.values())
        return {
            "schema_version": 1,
            "timeout_seconds": self._timeout_seconds,
            "mpi_single_rank_arguments": list(MPI_SINGLE_RANK_ARGUMENTS),
            "mpi_single_rank_binary_keys": sorted(self._mpi_binary_keys),
            "direct_binary_keys": sorted(
                all_keys - self._mpi_binary_keys
            ),
        }

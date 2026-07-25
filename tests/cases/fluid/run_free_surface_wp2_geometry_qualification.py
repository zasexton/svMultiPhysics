#!/usr/bin/env python3
"""Run the frozen WP-2 authoritative-geometry qualification matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
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
QUANTITATIVE_TYPES = {"integer", "real"}
QUANTITATIVE_RELATIONS = {
    "equal",
    "less_than",
    "less_than_or_equal",
    "greater_than",
    "greater_than_or_equal",
}
GTEST_RESULT_FIELDS = {
    "classname",
    "failures",
    "file",
    "line",
    "name",
    "result",
    "status",
    "time",
    "timestamp",
    "type_param",
    "value_param",
}
QUALIFICATION_BINARY_KEYS = {
    "geometry",
    "level_set",
    "systems",
    "assembly",
    "physics",
    "application",
    "assembly_mpi",
    "application_mpi",
}
EXPECTED_MATRIX_ID = "free_surface_wp2_geometry_v4"
EXPECTED_MATRIX_STATUS = "FROZEN_BEFORE_EXECUTION"
EXPECTED_WORK_PACKAGE = "WP-2"
EXPECTED_GATE_KEYS = {
    "expected_group_count",
    "expected_distinct_test_count",
    "expected_quantitative_evidence_count",
    "expected_failures",
    "expected_errors",
    "expected_disabled",
    "expected_skipped",
}
EXPECTED_ZERO_RESULT_GATES = {
    "expected_failures",
    "expected_errors",
    "expected_disabled",
    "expected_skipped",
}
GTEST_LIST_TIMEOUT_SECONDS = 60


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def path_present(path: Path) -> bool:
    return path.exists() or path.is_symlink()


def path_is_within(path: Path, root: Path) -> bool:
    return path == root or path.is_relative_to(root)


def paths_overlap(first: Path, second: Path) -> bool:
    return path_is_within(first, second) or path_is_within(second, first)


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


def untracked_source_record(
    source_root: Path,
    allowed_output_root: Path | None = None,
    ignored_source_roots: tuple[Path, ...] = (),
) -> dict[str, Any]:
    ordinary_raw = git_bytes(
        source_root, "ls-files", "--others", "--exclude-standard", "-z"
    )
    ignored_raw = b""
    scan_roots = set(ignored_source_roots)
    if (
        allowed_output_root is not None
        and path_is_within(allowed_output_root, source_root)
    ):
        scan_roots.add(allowed_output_root)
    relative_roots = sorted(
        {root.relative_to(source_root).as_posix() for root in scan_roots}
    )
    if relative_roots:
        ignored_raw = git_bytes(
            source_root,
            "ls-files",
            "--others",
            "--ignored",
            "--exclude-standard",
            "-z",
            "--",
            *relative_roots,
        )
    ordinary_entries = [
        value for value in ordinary_raw.split(b"\0") if value
    ]
    ignored_entries = [value for value in ignored_raw.split(b"\0") if value]
    entries = sorted(set(ordinary_entries + ignored_entries))
    allowed_entries: list[bytes] = []
    unexpected_entries: list[bytes] = []
    for entry in entries:
        candidate = source_root / Path(os.fsdecode(entry))
        if (
            allowed_output_root is not None
            and candidate.is_relative_to(allowed_output_root)
        ):
            allowed_entries.append(entry)
        else:
            unexpected_entries.append(entry)

    def encode(values: list[bytes]) -> bytes:
        return b"\0".join(values) + (b"\0" if values else b"")

    raw = encode(entries)
    allowed_raw = encode(allowed_entries)
    unexpected_raw = encode(unexpected_entries)
    return {
        "ignored_scan_roots": relative_roots,
        "allowed_output_root": (
            str(allowed_output_root) if allowed_output_root is not None else None
        ),
        "path_count": len(entries),
        "path_list_sha256": sha256_bytes(raw),
        "ordinary_path_count": len(ordinary_entries),
        "ignored_source_path_count": len(ignored_entries),
        "allowed_output_path_count": len(allowed_entries),
        "allowed_output_path_list_sha256": sha256_bytes(allowed_raw),
        "allowed_output_paths": [os.fsdecode(value) for value in allowed_entries],
        "unexpected_path_count": len(unexpected_entries),
        "unexpected_path_list_sha256": sha256_bytes(unexpected_raw),
        "unexpected_paths": [os.fsdecode(value) for value in unexpected_entries],
    }


def equal_check(metric: str, actual: Any, expected: Any) -> dict[str, Any]:
    return {
        "metric": metric,
        "actual": actual,
        "expected": expected,
        "relation": "equal",
        "passed": actual == expected,
    }


def load_registry(path: Path) -> dict[str, Any]:
    if path.resolve() != DEFAULT_REGISTRY.resolve():
        raise ValueError("WP-2 qualification requires the canonical frozen registry")
    registry = json.loads(path.read_text(encoding="utf-8"))
    if registry.get("schema_version") != 1:
        raise ValueError("unsupported WP-2 qualification schema")
    if registry.get("matrix_id") != EXPECTED_MATRIX_ID:
        raise ValueError("unexpected WP-2 qualification matrix id")
    if registry.get("status") != EXPECTED_MATRIX_STATUS:
        raise ValueError("WP-2 qualification matrix is not frozen before execution")
    if registry.get("work_package") != EXPECTED_WORK_PACKAGE:
        raise ValueError("unexpected WP-2 qualification work package")
    groups = registry.get("groups")
    gates = registry.get("gates", {})
    claims = registry.get("closure_contract")
    if not isinstance(groups, list) or not groups:
        raise ValueError("WP-2 qualification group list is empty")
    if not isinstance(claims, list) or not claims:
        raise ValueError("WP-2 closure contract is empty")
    if not isinstance(gates, dict) or set(gates) != EXPECTED_GATE_KEYS:
        raise ValueError("WP-2 qualification gates must use the exact frozen keys")
    if any(
        not isinstance(gates.get(key), int)
        or isinstance(gates.get(key), bool)
        or gates[key] != 0
        for key in EXPECTED_ZERO_RESULT_GATES
    ):
        raise ValueError(
            "WP-2 failure, error, disabled, and skipped gates must be exactly zero"
        )
    for key in EXPECTED_GATE_KEYS - EXPECTED_ZERO_RESULT_GATES:
        if (
            not isinstance(gates.get(key), int)
            or isinstance(gates.get(key), bool)
            or gates[key] <= 0
        ):
            raise ValueError(f"WP-2 qualification count gate must be positive: {key}")
    if gates.get("expected_group_count") != len(groups):
        raise ValueError("expected group count does not match the frozen list")

    build_targets = registry.get("build_targets")
    if not isinstance(build_targets, dict) or set(build_targets) != (
        QUALIFICATION_BINARY_KEYS
    ):
        raise ValueError(
            "WP-2 build target map must exactly cover every qualification binary"
        )
    if any(
        not isinstance(target, str)
        or not target
        or not re.fullmatch(r"[A-Za-z0-9_.+:-]+", target)
        for target in build_targets.values()
    ):
        raise ValueError("WP-2 build targets must be nonempty CMake target names")
    build_cmake_homes = registry.get("build_cmake_homes")
    if not isinstance(build_cmake_homes, dict) or set(build_cmake_homes) != (
        QUALIFICATION_BINARY_KEYS
    ):
        raise ValueError(
            "WP-2 CMake home map must exactly cover every qualification binary"
        )
    for relative_home in build_cmake_homes.values():
        if (
            not isinstance(relative_home, str)
            or not relative_home
            or Path(relative_home).is_absolute()
            or ".." in Path(relative_home).parts
        ):
            raise ValueError(
                "WP-2 CMake homes must be nonempty repository-relative paths"
            )

    group_ids: set[str] = set()
    test_names: set[str] = set()
    test_groups: dict[str, dict[str, Any]] = {}
    recorded_property_keys: set[tuple[str, str]] = set()
    for group in groups:
        group_id = group.get("id")
        if (
            not isinstance(group_id, str)
            or not re.fullmatch(r"[A-Za-z0-9_.-]+", group_id)
            or group_id in {".", ".."}
        ):
            raise ValueError(
                "every qualification group needs a safe single-path-component id"
            )
        if group_id in group_ids:
            raise ValueError(f"duplicate qualification group: {group_id}")
        group_ids.add(group_id)
        binary = group.get("binary")
        if binary not in QUALIFICATION_BINARY_KEYS:
            raise ValueError(f"unsupported binary key in group {group_id}")
        ranks = group.get("mpi_ranks")
        copies = group.get("gtest_output_copies")
        if (
            not isinstance(ranks, int)
            or isinstance(ranks, bool)
            or ranks <= 0
        ):
            raise ValueError(f"group {group_id} needs positive mpi_ranks")
        if (
            not isinstance(copies, int)
            or isinstance(copies, bool)
            or copies <= 0
            or copies > ranks
        ):
            raise ValueError(
                f"group {group_id} needs gtest_output_copies in [1, mpi_ranks]"
            )
        if ranks == 1 and copies != 1:
            raise ValueError(f"serial group {group_id} needs one output copy")
        tests = group.get("tests")
        if not isinstance(tests, list) or not tests:
            raise ValueError(f"group {group_id} has no tests")
        for name in tests:
            if not isinstance(name, str) or not re.fullmatch(
                r"[A-Za-z0-9_]+\.[A-Za-z0-9_]+", name
            ):
                raise ValueError(f"invalid suite.name in group {group_id}: {name}")
            if name in test_names:
                raise ValueError(f"duplicate frozen test: {name}")
            test_names.add(name)
            test_groups[name] = group
        execution = group.get("execution", {})
        for key in ("wall_time_seconds", "memory_mib", "output_mib"):
            if (
                not isinstance(execution.get(key), int)
                or isinstance(execution.get(key), bool)
                or execution[key] <= 0
            ):
                raise ValueError(
                    f"group {group_id} execution envelope {key} must be positive"
                )
        recorded_properties = group.get("recorded_properties", [])
        if not isinstance(recorded_properties, list):
            raise ValueError(f"group {group_id} recorded properties must be a list")
        for contract in recorded_properties:
            if not isinstance(contract, dict):
                raise ValueError(
                    f"group {group_id} recorded property must be an object"
                )
            contract_keys = set(contract)
            common_keys = {"property", "type", "relation", "threshold"}
            if contract_keys == common_keys:
                if len(tests) != 1:
                    raise ValueError(
                        f"group {group_id} must name the test for each"
                        " recorded property"
                    )
                test = tests[0]
            elif contract_keys == common_keys | {"test"}:
                test = contract["test"]
                if not isinstance(test, str) or test not in tests:
                    raise ValueError(
                        f"group {group_id} recorded property cites a test outside"
                        " the group"
                    )
            else:
                raise ValueError(
                    f"group {group_id} recorded property has unexpected keys"
                )
            property_name = contract["property"]
            value_type = contract["type"]
            relation = contract["relation"]
            threshold = contract["threshold"]
            if (
                not isinstance(property_name, str)
                or not property_name
                or property_name in GTEST_RESULT_FIELDS
            ):
                raise ValueError(
                    f"group {group_id} has an invalid recorded property name"
                )
            property_key = (test, property_name)
            if property_key in recorded_property_keys:
                raise ValueError(f"duplicate recorded property: {test}.{property_name}")
            recorded_property_keys.add(property_key)
            if not isinstance(value_type, str) or value_type not in QUANTITATIVE_TYPES:
                raise ValueError(f"unsupported recorded property type: {value_type}")
            if not isinstance(relation, str) or relation not in QUANTITATIVE_RELATIONS:
                raise ValueError(f"unsupported recorded property relation: {relation}")
            if value_type == "integer":
                valid_threshold = isinstance(threshold, int) and not isinstance(
                    threshold, bool
                )
            else:
                valid_threshold = (
                    isinstance(threshold, (int, float))
                    and not isinstance(threshold, bool)
                    and math.isfinite(threshold)
                )
            if not valid_threshold:
                raise ValueError(
                    f"invalid {value_type} threshold for {test}.{property_name}"
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

    quantitative_evidence = registry.get("quantitative_evidence", [])
    if not isinstance(quantitative_evidence, list):
        raise ValueError("quantitative evidence must be a list")
    evidence_properties: set[tuple[str, str]] = set()
    for evidence in quantitative_evidence:
        if not isinstance(evidence, dict):
            raise ValueError("every quantitative evidence entry must be an object")
        test = evidence.get("test")
        property_name = evidence.get("property")
        value_type = evidence.get("type")
        relation = evidence.get("relation")
        threshold = evidence.get("threshold")
        if not isinstance(test, str) or test not in test_names:
            raise ValueError(f"quantitative evidence cites an unfrozen test: {test}")
        if test_groups[test]["mpi_ranks"] != 1:
            raise ValueError(
                f"quantitative evidence requires a serial test result: {test}"
            )
        if (
            not isinstance(property_name, str)
            or not property_name
            or property_name in GTEST_RESULT_FIELDS
        ):
            raise ValueError(
                f"invalid recorded property for quantitative evidence: {property_name}"
            )
        evidence_key = (test, property_name)
        if evidence_key in evidence_properties:
            raise ValueError(
                f"duplicate quantitative evidence property: {test}.{property_name}"
            )
        evidence_properties.add(evidence_key)
        if not isinstance(value_type, str) or value_type not in QUANTITATIVE_TYPES:
            raise ValueError(f"unsupported quantitative evidence type: {value_type}")
        if not isinstance(relation, str) or relation not in QUANTITATIVE_RELATIONS:
            raise ValueError(f"unsupported quantitative evidence relation: {relation}")
        if value_type == "integer":
            valid_threshold = isinstance(threshold, int) and not isinstance(
                threshold, bool
            )
        else:
            valid_threshold = (
                isinstance(threshold, (int, float))
                and not isinstance(threshold, bool)
                and math.isfinite(threshold)
            )
        if not valid_threshold:
            raise ValueError(
                f"invalid {value_type} threshold for {test}.{property_name}"
            )
    if gates.get("expected_quantitative_evidence_count") != len(quantitative_evidence):
        raise ValueError(
            "expected quantitative evidence count does not match the frozen list"
        )
    registry["quantitative_evidence"] = quantitative_evidence
    return registry


def listed_gtests(
    binary: Path, timeout_seconds: int = GTEST_LIST_TIMEOUT_SECONDS
) -> set[str]:
    result = subprocess.run(
        [str(binary), "--gtest_list_tests"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout_seconds,
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


def coerce_quantitative_value(
    raw_value: Any, value_type: str
) -> tuple[Any, str | None]:
    if value_type == "integer":
        if isinstance(raw_value, bool):
            return None, "property_type_mismatch"
        if isinstance(raw_value, int):
            return raw_value, None
        if isinstance(raw_value, str) and re.fullmatch(r"[+-]?[0-9]+", raw_value):
            return int(raw_value), None
        return None, "property_type_mismatch"
    if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float, str)):
        return None, "property_type_mismatch"
    try:
        value = float(raw_value)
    except (OverflowError, TypeError, ValueError):
        return None, "property_type_mismatch"
    if not math.isfinite(value):
        return None, "property_value_not_finite"
    return value, None


def quantitative_relation_passes(actual: Any, relation: str, threshold: Any) -> bool:
    if relation == "equal":
        return actual == threshold
    if relation == "less_than":
        return actual < threshold
    if relation == "less_than_or_equal":
        return actual <= threshold
    if relation == "greater_than":
        return actual > threshold
    if relation == "greater_than_or_equal":
        return actual >= threshold
    raise ValueError(f"unsupported quantitative evidence relation: {relation}")


def evaluate_quantitative_evidence(
    registry: dict[str, Any], output_root: Path
) -> dict[str, Any]:
    group_by_test = {
        test: group for group in registry["groups"] for test in group["tests"]
    }
    flattened_by_group: dict[str, dict[str, dict[str, Any]] | None] = {}
    diagnostics_by_group: dict[str, str | None] = {}
    checks: list[dict[str, Any]] = []
    declarations = sorted(
        registry["quantitative_evidence"],
        key=lambda evidence: (evidence["test"], evidence["property"]),
    )
    for evidence in declarations:
        group = group_by_test[evidence["test"]]
        group_id = group["id"]
        relative_result_path = f"groups/{group_id}/gtest.json"
        if group_id not in flattened_by_group:
            result_path = output_root / relative_result_path
            if not result_path.is_file():
                flattened_by_group[group_id] = None
                diagnostics_by_group[group_id] = "gtest_result_missing"
            else:
                try:
                    document = json.loads(result_path.read_text(encoding="utf-8"))
                    flattened_by_group[group_id] = flatten_gtest(document)
                    diagnostics_by_group[group_id] = None
                except (
                    AttributeError,
                    json.JSONDecodeError,
                    OSError,
                    RecursionError,
                    TypeError,
                    ValueError,
                ):
                    flattened_by_group[group_id] = None
                    diagnostics_by_group[group_id] = "gtest_result_invalid"

        result = flattened_by_group[group_id]
        diagnostic = diagnostics_by_group[group_id]
        test_result = result.get(evidence["test"]) if result is not None else None
        if diagnostic is None and test_result is None:
            diagnostic = "test_result_missing"
        property_name = evidence["property"]
        raw_value = (
            test_result.get(property_name)
            if test_result is not None and property_name in test_result
            else None
        )
        if diagnostic is None and property_name not in test_result:
            diagnostic = "property_missing"
        actual = None
        if diagnostic is None:
            actual, diagnostic = coerce_quantitative_value(raw_value, evidence["type"])
        passed = False
        if diagnostic is None:
            passed = quantitative_relation_passes(
                actual, evidence["relation"], evidence["threshold"]
            )
            if not passed:
                diagnostic = "relation_not_satisfied"
        checks.append(
            {
                "test": evidence["test"],
                "property": property_name,
                "type": evidence["type"],
                "relation": evidence["relation"],
                "threshold": evidence["threshold"],
                "group_id": group_id,
                "gtest_result": relative_result_path,
                "raw_value": raw_value,
                "actual": actual,
                "diagnostic": diagnostic,
                "passed": passed,
            }
        )
    passed_count = sum(1 for check in checks if check["passed"])
    return {
        "artifact_schema_version": 1,
        "declared_check_count": len(checks),
        "passed_check_count": passed_count,
        "checks": checks,
        "outcome": "PASS" if passed_count == len(checks) else "FAIL_METHOD",
    }


def evaluate_group_recorded_properties(
    registry: dict[str, Any], output_root: Path
) -> dict[str, Any]:
    declarations: list[dict[str, Any]] = []
    for group in registry["groups"]:
        for contract in group.get("recorded_properties", []):
            declarations.append(
                {
                    **contract,
                    "test": contract.get("test", group["tests"][0]),
                    "group_id": group["id"],
                    "mpi_ranks": group["mpi_ranks"],
                }
            )
    declarations.sort(
        key=lambda contract: (
            contract["group_id"],
            contract["test"],
            contract["property"],
        )
    )

    flattened_by_group: dict[str, dict[str, dict[str, Any]] | None] = {}
    diagnostics_by_group: dict[str, str | None] = {}
    result_path_by_group: dict[str, str] = {}
    checks: list[dict[str, Any]] = []
    for contract in declarations:
        group_id = contract["group_id"]
        result_name = (
            "gtest.json" if contract["mpi_ranks"] == 1 else "gtest_rank_0.json"
        )
        relative_result_path = f"groups/{group_id}/{result_name}"
        if group_id not in flattened_by_group:
            result_path_by_group[group_id] = relative_result_path
            result_path = output_root / relative_result_path
            if not result_path.is_file():
                flattened_by_group[group_id] = None
                diagnostics_by_group[group_id] = "gtest_result_missing"
            else:
                try:
                    document = json.loads(result_path.read_text(encoding="utf-8"))
                    if not isinstance(document, dict):
                        raise ValueError("gtest result is not an object")
                    flattened_by_group[group_id] = flatten_gtest(document)
                    diagnostics_by_group[group_id] = None
                except (
                    AttributeError,
                    json.JSONDecodeError,
                    OSError,
                    RecursionError,
                    TypeError,
                    ValueError,
                ):
                    flattened_by_group[group_id] = None
                    diagnostics_by_group[group_id] = "gtest_result_invalid"

        result = flattened_by_group[group_id]
        diagnostic = diagnostics_by_group[group_id]
        test_result = result.get(contract["test"]) if result is not None else None
        if diagnostic is None and test_result is None:
            diagnostic = "test_result_missing"
        property_name = contract["property"]
        raw_value = (
            test_result.get(property_name)
            if test_result is not None and property_name in test_result
            else None
        )
        if (
            diagnostic is None
            and test_result is not None
            and property_name not in test_result
        ):
            diagnostic = "property_missing"
        actual = None
        if diagnostic is None:
            actual, diagnostic = coerce_quantitative_value(raw_value, contract["type"])
        passed = False
        if diagnostic is None:
            passed = quantitative_relation_passes(
                actual, contract["relation"], contract["threshold"]
            )
            if not passed:
                diagnostic = "relation_not_satisfied"
        checks.append(
            {
                "test": contract["test"],
                "property": property_name,
                "type": contract["type"],
                "relation": contract["relation"],
                "threshold": contract["threshold"],
                "group_id": group_id,
                "gtest_result": result_path_by_group[group_id],
                "result_rank": 0 if contract["mpi_ranks"] > 1 else None,
                "raw_value": raw_value,
                "actual": actual,
                "diagnostic": diagnostic,
                "passed": passed,
            }
        )
    passed_count = sum(1 for check in checks if check["passed"])
    return {
        "artifact_schema_version": 1,
        "declared_check_count": len(checks),
        "passed_check_count": passed_count,
        "checks": checks,
        "outcome": "PASS" if passed_count == len(checks) else "FAIL_METHOD",
    }


def evaluate_serial_result(
    expected_tests: list[str],
    document: dict[str, Any],
    return_code: int,
    termination_reason: str | None,
    gates: dict[str, Any],
) -> list[dict[str, Any]]:
    expected = set(expected_tests)
    actual = flatten_gtest(document)
    actual_names = set(actual)
    skipped_records = sorted(
        name for name, result in actual.items() if result.get("result") == "SKIPPED"
    )
    reported_skipped_count = document.get("skipped", len(skipped_records))
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
        equal_check(
            "failure_count",
            document.get("failures"),
            gates["expected_failures"],
        ),
        equal_check("error_count", document.get("errors"), gates["expected_errors"]),
        equal_check(
            "disabled_count",
            document.get("disabled"),
            gates["expected_disabled"],
        ),
        equal_check("skipped_count", reported_skipped_count, gates["expected_skipped"]),
        equal_check(
            "skipped_result_count",
            len(skipped_records),
            gates["expected_skipped"],
        ),
        equal_check("skipped_tests", skipped_records, []),
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
    gates: dict[str, Any],
) -> list[dict[str, Any]]:
    run_pattern = re.compile(r"\[ RUN\s+\]\s+([A-Za-z0-9_]+\.[A-Za-z0-9_]+)")
    ok_pattern = re.compile(r"\[\s+OK\s+\]\s+([A-Za-z0-9_]+\.[A-Za-z0-9_]+)")
    observed_runs = run_pattern.findall(stdout)
    observed_ok = ok_pattern.findall(stdout)
    expected = set(expected_tests)
    combined_output = stdout + "\n" + stderr
    checks = [
        equal_check("process_return_code", return_code, 0),
        equal_check("termination_reason", termination_reason, None),
        equal_check(
            "failure_count",
            combined_output.count("[  FAILED  ]"),
            gates["expected_failures"],
        ),
        equal_check(
            "error_count",
            len(re.findall(r"\[\s+ERROR\s+\]", combined_output)),
            gates["expected_errors"],
        ),
        equal_check(
            "disabled_count",
            len(re.findall(r"\[\s+DISABLED\s+\]", combined_output)),
            gates["expected_disabled"],
        ),
        equal_check(
            "skipped_count",
            len(re.findall(r"\[\s+SKIPPED\s+\]", combined_output)),
            gates["expected_skipped"],
        ),
        equal_check("failure_marker_count", stdout.count("[  FAILED  ]"), 0),
        equal_check("stderr_failure_marker_count", stderr.count("[  FAILED  ]"), 0),
        equal_check("unexpected_run_tests", sorted(set(observed_runs) - expected), []),
        equal_check(
            "unexpected_completed_tests", sorted(set(observed_ok) - expected), []
        ),
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


def evaluate_mpi_gtest_results(
    group_directory: Path,
    expected_tests: list[str],
    ranks: int,
    gates: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    expected_names = sorted(f"gtest_rank_{rank}.json" for rank in range(ranks))
    observed_names = sorted(
        path.name for path in group_directory.glob("gtest_rank_*.json")
    )
    checks = [equal_check("mpi_gtest_result_files", observed_names, expected_names)]
    records: list[dict[str, Any]] = []
    for rank in range(ranks):
        name = f"gtest_rank_{rank}.json"
        path = group_directory / name
        relative_path = f"groups/{group_directory.name}/{name}"
        present = path.is_file()
        checks.append(equal_check(f"mpi_gtest_result_present:{rank}", present, True))
        valid = False
        error_name: str | None = None
        document: dict[str, Any] | None = None
        if present:
            try:
                loaded = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(loaded, dict):
                    raise ValueError("MPI gtest result is not an object")
                document = loaded
                flatten_gtest(document)
                valid = True
            except (
                AttributeError,
                json.JSONDecodeError,
                OSError,
                RecursionError,
                TypeError,
                ValueError,
            ) as error:
                error_name = type(error).__name__
        checks.append(equal_check(f"mpi_gtest_result_valid:{rank}", valid, True))
        if document is not None and valid:
            for document_check in evaluate_serial_result(
                expected_tests,
                document,
                0,
                None,
                gates,
            ):
                document_check["metric"] = (
                    f"mpi_gtest_rank_{rank}:{document_check['metric']}"
                )
                checks.append(document_check)
        records.append(
            {
                "rank": rank,
                "gtest_result": relative_path,
                "present": present,
                "valid": valid,
                "error": error_name,
                "sha256": sha256_file(path) if present else None,
            }
        )
    return checks, records


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
        for line in (
            Path(f"/proc/{process_id}/status").read_text(encoding="utf-8").splitlines()
        ):
            if line.startswith("VmRSS:"):
                return int(line.split()[1])
    except (OSError, ValueError, IndexError):
        return None
    return None


def process_session_members(process_session_id: int) -> list[tuple[int, int]]:
    members: list[tuple[int, int]] = []
    try:
        process_directories = list(Path("/proc").iterdir())
    except OSError:
        return members
    for process_directory in process_directories:
        if not process_directory.name.isdigit():
            continue
        try:
            stat = (process_directory / "stat").read_text(
                encoding="utf-8", errors="replace"
            )
            closing_parenthesis = stat.rfind(")")
            if closing_parenthesis < 0:
                continue
            stat_fields = stat[closing_parenthesis + 2 :].split()
            if len(stat_fields) < 4 or int(stat_fields[3]) != process_session_id:
                continue
            process_id = int(process_directory.name)
            process_group_id = int(stat_fields[2])
        except (FileNotFoundError, OSError, ValueError):
            continue
        members.append((process_id, process_group_id))
    return members


def process_session_resources(process_session_id: int) -> dict[str, Any]:
    aggregate_resident_kib = 0
    stat_read_failure_count = 0
    resident_read_failure_count = 0
    try:
        process_directories = list(Path("/proc").iterdir())
    except OSError:
        return {
            "enumeration_available": False,
            "aggregate_resident_kib": 0,
            "process_count": 0,
            "process_group_count": 0,
            "resident_sample_count": 0,
            "stat_read_failure_count": 0,
            "resident_read_failure_count": 0,
        }
    members: list[tuple[int, int]] = []
    for process_directory in process_directories:
        if not process_directory.name.isdigit():
            continue
        try:
            stat = (process_directory / "stat").read_text(
                encoding="utf-8", errors="replace"
            )
            closing_parenthesis = stat.rfind(")")
            if closing_parenthesis < 0:
                stat_read_failure_count += 1
                continue
            stat_fields = stat[closing_parenthesis + 2 :].split()
            if len(stat_fields) < 4:
                stat_read_failure_count += 1
                continue
            if int(stat_fields[3]) != process_session_id:
                continue
            members.append((int(process_directory.name), int(stat_fields[2])))
        except (FileNotFoundError, OSError, ValueError):
            stat_read_failure_count += 1
    process_groups = {process_group_id for _process_id, process_group_id in members}
    resident_sample_count = 0
    for process_id, _process_group_id in members:
        resident_kib = process_resident_kib(process_id)
        if resident_kib is None:
            resident_read_failure_count += 1
            continue
        resident_sample_count += 1
        aggregate_resident_kib += resident_kib
    return {
        "enumeration_available": True,
        "aggregate_resident_kib": aggregate_resident_kib,
        "process_count": len(members),
        "process_group_count": len(process_groups),
        "resident_sample_count": resident_sample_count,
        "stat_read_failure_count": stat_read_failure_count,
        "resident_read_failure_count": resident_read_failure_count,
    }


def signal_process_session(process_session_id: int, signal_number: int) -> list[int]:
    runner_process_group = os.getpgrp()
    process_groups = sorted(
        {
            process_group_id
            for _process_id, process_group_id in process_session_members(
                process_session_id
            )
            if process_group_id != runner_process_group
        }
    )
    signaled: list[int] = []
    for process_group_id in process_groups:
        try:
            os.killpg(process_group_id, signal_number)
        except ProcessLookupError:
            continue
        signaled.append(process_group_id)
    return signaled


def terminate_process_session(process: subprocess.Popen[bytes]) -> dict[str, Any]:
    process_session_id = process.pid
    term_process_groups: set[int] = set()
    kill_process_groups: set[int] = set()
    term_deadline = time.monotonic() + 5.0
    while True:
        process.poll()
        if not process_session_members(process_session_id):
            break
        term_process_groups.update(
            signal_process_session(process_session_id, signal.SIGTERM)
        )
        if time.monotonic() >= term_deadline:
            break
        time.sleep(0.05)

    kill_deadline = time.monotonic() + 5.0
    while True:
        process.poll()
        if not process_session_members(process_session_id):
            break
        kill_process_groups.update(
            signal_process_session(process_session_id, signal.SIGKILL)
        )
        if time.monotonic() >= kill_deadline:
            break
        time.sleep(0.05)

    launcher_fallback_used = False
    if process.poll() is None:
        launcher_fallback_used = True
        try:
            os.killpg(process_session_id, signal.SIGTERM)
            term_process_groups.add(process_session_id)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process_session_id, signal.SIGKILL)
                kill_process_groups.add(process_session_id)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                pass
    lingering_members = process_session_members(process_session_id)
    final_session_sample = process_session_resources(process_session_id)
    return {
        "process_session_id": process_session_id,
        "launcher_fallback_used": launcher_fallback_used,
        "term_process_group_ids": sorted(term_process_groups),
        "kill_process_group_ids": sorted(kill_process_groups),
        "lingering_process_ids": sorted(
            process_id for process_id, _process_group_id in lingering_members
        ),
        "lingering_process_group_ids": sorted(
            {process_group_id for _process_id, process_group_id in lingering_members}
        ),
        "final_session_enumeration_available": final_session_sample[
            "enumeration_available"
        ],
        "all_session_processes_terminated": (
            final_session_sample["enumeration_available"]
            and process.poll() is not None
            and not lingering_members
        ),
    }


def poll_process_with_rusage(
    process: subprocess.Popen[bytes],
) -> tuple[int | None, int | None, bool]:
    if process.returncode is not None:
        return process.returncode, None, False
    try:
        waited_process_id, wait_status, usage = os.wait4(process.pid, os.WNOHANG)
    except (ChildProcessError, OSError):
        return process.poll(), None, False
    if waited_process_id == 0:
        return None, None, False
    process.returncode = os.waitstatus_to_exitcode(wait_status)
    return process.returncode, max(0, int(usage.ru_maxrss)), True


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
    launch_mode: str,
    required_simultaneous_process_samples: int = 1,
) -> dict[str, Any]:
    if launch_mode not in {"direct_serial", "mpi"}:
        raise ValueError(f"unsupported monitored launch mode: {launch_mode}")
    if required_simultaneous_process_samples <= 0:
        raise ValueError("required simultaneous process samples must be positive")
    if (
        launch_mode == "direct_serial"
        and required_simultaneous_process_samples != 1
    ):
        raise ValueError("direct serial monitoring requires one process sample")
    allow_reaped_serial_launcher_rusage = launch_mode == "direct_serial"
    memory_bytes = memory_mib * 1024 * 1024
    output_bytes = output_mib * 1024 * 1024

    def set_limits() -> None:
        resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))

    started = time.monotonic()
    peak_resident_kib = 0
    peak_aggregate_resident_kib = 0
    peak_process_count = 0
    peak_resident_sample_count = 0
    peak_process_group_count = 0
    monitor_sample_count = 0
    successful_session_sample_count = 0
    session_enumeration_failure_count = 0
    stat_read_failure_count = 0
    resident_read_failure_count = 0
    reaped_launcher_max_resident_kib: int | None = None
    reaped_launcher_rusage_available = False
    process_session_id: int | None = None
    termination_reason: str | None = None
    termination: dict[str, Any] | None = None
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
        process_session_id = process.pid
        while True:
            elapsed = time.monotonic() - started
            resident = process_resident_kib(process.pid)
            if resident is not None:
                peak_resident_kib = max(peak_resident_kib, resident)
            session_sample = process_session_resources(process.pid)
            monitor_sample_count += 1
            if not session_sample["enumeration_available"]:
                session_enumeration_failure_count += 1
            stat_read_failure_count += session_sample["stat_read_failure_count"]
            resident_read_failure_count += session_sample["resident_read_failure_count"]
            aggregate_resident_kib = session_sample["aggregate_resident_kib"]
            process_count = session_sample["process_count"]
            resident_sample_count = session_sample["resident_sample_count"]
            process_group_count = session_sample["process_group_count"]
            if (
                session_sample["enumeration_available"]
                and session_sample["resident_sample_count"] > 0
                and process_count > 0
            ):
                successful_session_sample_count += 1
            peak_aggregate_resident_kib = max(
                peak_aggregate_resident_kib, aggregate_resident_kib
            )
            peak_process_count = max(peak_process_count, process_count)
            peak_resident_sample_count = max(
                peak_resident_sample_count, resident_sample_count
            )
            peak_process_group_count = max(
                peak_process_group_count, process_group_count
            )
            (
                polled_return_code,
                polled_max_resident_kib,
                polled_rusage_available,
            ) = poll_process_with_rusage(process)
            if polled_rusage_available:
                reaped_launcher_rusage_available = True
                reaped_launcher_max_resident_kib = polled_max_resident_kib
            if polled_return_code is not None:
                break
            if not session_sample["enumeration_available"]:
                termination_reason = "session_resource_monitoring_unavailable"
            elif elapsed > wall_time_seconds:
                termination_reason = "wall_time_envelope_exceeded"
            elif directory_size(output_directory) > output_bytes:
                termination_reason = "output_envelope_exceeded"
            elif aggregate_resident_kib > memory_mib * 1024:
                termination_reason = "memory_envelope_exceeded"
            if termination_reason is not None:
                termination = terminate_process_session(process)
                break
            time.sleep(0.05)
        if termination_reason is None and process_session_members(process.pid):
            termination_reason = "launcher_exited_with_lingering_session_processes"
            termination = terminate_process_session(process)
        return_code = process.wait()
    final_wall_time_seconds = time.monotonic() - started
    final_output_bytes = directory_size(output_directory)
    reaped_launcher_fallback_used = (
        allow_reaped_serial_launcher_rusage
        and successful_session_sample_count == 0
        and session_enumeration_failure_count == 0
        and reaped_launcher_rusage_available
        and reaped_launcher_max_resident_kib is not None
        and peak_process_count <= 1
        and termination_reason
        != "launcher_exited_with_lingering_session_processes"
    )
    if termination_reason is None:
        if final_wall_time_seconds > wall_time_seconds:
            termination_reason = "wall_time_envelope_exceeded"
        elif final_output_bytes > output_bytes:
            termination_reason = "output_envelope_exceeded"
        elif (
            reaped_launcher_fallback_used
            and reaped_launcher_max_resident_kib > memory_mib * 1024
        ):
            termination_reason = "memory_envelope_exceeded"
    complete_session_process_coverage = (
        successful_session_sample_count > 0
        and peak_process_count >= required_simultaneous_process_samples
        and peak_resident_sample_count >= required_simultaneous_process_samples
    )
    resource_monitoring_succeeded = (
        complete_session_process_coverage or reaped_launcher_fallback_used
    )
    if not resource_monitoring_succeeded and termination_reason is None:
        termination_reason = "session_process_coverage_incomplete"
    return {
        "return_code": return_code,
        "termination_reason": termination_reason,
        "termination": termination,
        "launch_mode": launch_mode,
        "wall_time_seconds": final_wall_time_seconds,
        "memory_enforcement_scope": "spawned_process_session",
        "memory_enforcement_method": (
            "per_process_address_space_limit_and_sampled_session_resident_memory"
        ),
        "aggregate_memory_measurement": "sampled_peak",
        "resource_monitoring_outcome": (
            "PASS" if resource_monitoring_succeeded else "FAIL_METHOD"
        ),
        "resource_monitoring_method": (
            "sampled_process_session"
            if complete_session_process_coverage
            else (
                "reaped_serial_launcher_rusage"
                if reaped_launcher_fallback_used
                else "unavailable"
            )
        ),
        "monitor_sample_count": monitor_sample_count,
        "successful_session_sample_count": successful_session_sample_count,
        "required_simultaneous_process_samples": (
            required_simultaneous_process_samples
        ),
        "complete_session_process_coverage": complete_session_process_coverage,
        "session_enumeration_failure_count": session_enumeration_failure_count,
        "stat_read_failure_count": stat_read_failure_count,
        "resident_read_failure_count": resident_read_failure_count,
        "process_session_id": process_session_id,
        "peak_resident_kib_sampled": peak_resident_kib,
        "peak_aggregate_resident_kib_sampled": peak_aggregate_resident_kib,
        "peak_process_count_sampled": peak_process_count,
        "peak_resident_sample_count": peak_resident_sample_count,
        "peak_process_group_count_sampled": peak_process_group_count,
        "reaped_launcher_rusage_available": reaped_launcher_rusage_available,
        "reaped_launcher_max_resident_kib": reaped_launcher_max_resident_kib,
        "reaped_launcher_fallback_allowed": allow_reaped_serial_launcher_rusage,
        "reaped_launcher_fallback_used": reaped_launcher_fallback_used,
        "reaped_launcher_fallback_scope": "direct_serial_launcher_process",
        "reaped_launcher_fallback_source": "wait4_ru_maxrss",
        "reaped_launcher_fallback_requires_session_enumeration": True,
        "final_output_bytes": final_output_bytes,
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


def cmake_cache_values(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith(("#", "//")) or "=" not in line or ":" not in line:
            continue
        left, value = line.split("=", 1)
        name, _value_type = left.split(":", 1)
        values[name] = value
    return values


def run_build_phase(
    command: list[str],
    source_root: Path,
    output_root: Path,
    stdout_path: Path,
    stderr_path: Path,
    timeout_seconds: int,
) -> dict[str, Any]:
    started = time.monotonic()
    timed_out = False
    return_code: int | None = None
    termination: dict[str, Any] | None = None
    with stdout_path.open("xb") as stdout_file, stderr_path.open("xb") as stderr_file:
        process = subprocess.Popen(
            command,
            cwd=source_root,
            stdout=stdout_file,
            stderr=stderr_file,
            start_new_session=True,
        )
        try:
            return_code = process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            termination = terminate_process_session(process)
            return_code = process.returncode
    return {
        "command": command,
        "return_code": return_code,
        "timed_out": timed_out,
        "termination": termination,
        "elapsed_seconds": time.monotonic() - started,
        "stdout": str(stdout_path.relative_to(output_root)),
        "stderr": str(stderr_path.relative_to(output_root)),
        "stdout_sha256": sha256_file(stdout_path),
        "stderr_sha256": sha256_file(stderr_path),
    }


def run_clean_builds(
    registry: dict[str, Any],
    binaries: dict[str, Path],
    source_root: Path,
    output_root: Path,
    cmake: Path,
    parallel: int,
    timeout_seconds: int,
) -> dict[str, Any]:
    source_commit = git_bytes(source_root, "rev-parse", "HEAD").decode().strip()
    source_tree = git_bytes(source_root, "rev-parse", "HEAD^{tree}").decode().strip()
    result: dict[str, Any] = {
        "artifact_schema_version": 1,
        "source_root": str(source_root),
        "source_commit_before_build": source_commit,
        "source_tree_before_build": source_tree,
        "clean_first": True,
        "configure_before_clean": True,
        "separate_clean_phase": True,
        "parallel": parallel,
        "timeout_seconds_per_phase": timeout_seconds,
        "validation": [],
        "target_inventory": [],
        "builds": [],
        "diagnostic": None,
        "outcome": "PASS",
    }
    build_root = output_root / "builds"
    build_root.mkdir(parents=True, exist_ok=False)
    grouped: dict[Path, list[tuple[str, str, Path, Path]]] = {}
    for binary_key in sorted(binaries):
        binary = binaries[binary_key]
        cache = find_cmake_cache(binary)
        expected_home = (
            source_root / registry["build_cmake_homes"][binary_key]
        ).resolve()
        validation = {
            "binary_key": binary_key,
            "binary_path": str(binary),
            "target": registry["build_targets"][binary_key],
            "expected_cmake_home": str(expected_home),
            "cache_path": str(cache.resolve()) if cache is not None else None,
            "cache_present": cache is not None,
            "cache_is_symlink": cache.is_symlink() if cache is not None else None,
            "cmake_home_directory": None,
            "single_configuration": False,
            "passed": False,
            "diagnostic": None,
        }
        if cache is None:
            validation["diagnostic"] = "cmake_cache_missing"
            result["validation"].append(validation)
            continue
        if cache.is_symlink():
            validation["diagnostic"] = "cmake_cache_symlink_unsupported"
            result["validation"].append(validation)
            continue
        cache = cache.resolve()
        cache_values = cmake_cache_values(cache)
        home = cache_values.get("CMAKE_HOME_DIRECTORY")
        configurations = cache_values.get("CMAKE_CONFIGURATION_TYPES", "")
        validation["cmake_home_directory"] = home
        validation["single_configuration"] = not configurations.strip()
        if home is None or Path(home).resolve() != expected_home:
            validation["diagnostic"] = "cmake_home_mismatch"
        elif configurations.strip():
            validation["diagnostic"] = "multi_configuration_generator_unsupported"
        else:
            validation["passed"] = True
        result["validation"].append(validation)
        if not validation["passed"]:
            continue
        grouped.setdefault(cache.parent, []).append(
            (
                binary_key,
                registry["build_targets"][binary_key],
                binary,
                cache,
            )
        )
    failed_validation = [
        validation for validation in result["validation"] if not validation["passed"]
    ]
    if failed_validation:
        result["outcome"] = "FAIL_METHOD"
        result["diagnostic"] = "build_provenance_validation_failed"
        return result

    for ordinal, (build_directory, entries) in enumerate(
        sorted(grouped.items(), key=lambda item: str(item[0])), start=1
    ):
        targets = sorted({entry[1] for entry in entries})
        inventory_stdout = (
            build_root / f"target_inventory_{ordinal:02d}_stdout.txt"
        )
        inventory_stderr = (
            build_root / f"target_inventory_{ordinal:02d}_stderr.txt"
        )
        inventory = run_build_phase(
            [
                str(cmake),
                "--build",
                str(build_directory),
                "--target",
                "help",
            ],
            source_root,
            output_root,
            inventory_stdout,
            inventory_stderr,
            min(timeout_seconds, GTEST_LIST_TIMEOUT_SECONDS),
        )
        help_text = inventory_stdout.read_text(
            encoding="utf-8", errors="replace"
        )
        listed_targets = [
            target
            for target in targets
            if re.search(
                rf"(?<![A-Za-z0-9_.-]){re.escape(target)}"
                rf"(?![A-Za-z0-9_.-])",
                help_text,
            )
        ]
        missing_targets = sorted(set(targets) - set(listed_targets))
        inventory_passed = (
            not inventory["timed_out"]
            and inventory["return_code"] == 0
            and not missing_targets
        )
        inventory.update(
            {
                "build_directory": str(build_directory),
                "expected_targets": targets,
                "listed_expected_targets": listed_targets,
                "missing_targets": missing_targets,
                "outcome": "PASS" if inventory_passed else "FAIL_METHOD",
            }
        )
        result["target_inventory"].append(inventory)
        if not inventory_passed:
            result["outcome"] = "FAIL_METHOD"
            result["diagnostic"] = "build_target_inventory_failed"
            return result

    for ordinal, (build_directory, entries) in enumerate(
        sorted(grouped.items(), key=lambda item: str(item[0])), start=1
    ):
        group_directory = build_root / f"build_{ordinal:02d}"
        group_directory.mkdir(parents=True, exist_ok=False)
        group_caches = {entry[3] for entry in entries}
        if len(group_caches) != 1:
            raise RuntimeError(
                "one build directory cannot have multiple CMake caches"
            )
        cache = next(iter(group_caches))
        if cache.parent != build_directory:
            raise RuntimeError("build directory does not own its CMake cache")
        targets = sorted({entry[1] for entry in entries})
        binary_keys = sorted(binary_key for binary_key, *_ in entries)
        binaries_before_clean = {
            binary_key: {
                "path": str(binary),
                "exists": path_present(binary),
                "is_file": binary.is_file(),
                "sha256": sha256_file(binary) if binary.is_file() else None,
            }
            for binary_key, _target, binary, _cache in entries
        }
        expected_homes = {
            (
                source_root / registry["build_cmake_homes"][binary_key]
            ).resolve()
            for binary_key, _target, _binary, _cache in entries
        }
        if len(expected_homes) != 1:
            raise RuntimeError(
                "one build directory cannot have multiple CMake source homes"
            )
        expected_home = next(iter(expected_homes))
        configure_command = [
            str(cmake),
            "-S",
            str(expected_home),
            "-B",
            str(build_directory),
        ]
        configure_phase = run_build_phase(
            configure_command,
            source_root,
            output_root,
            group_directory / "configure_stdout.txt",
            group_directory / "configure_stderr.txt",
            timeout_seconds,
        )
        cache_present_after_configure = cache.is_file() and not cache.is_symlink()
        cache_values_after_configure = (
            cmake_cache_values(cache) if cache_present_after_configure else {}
        )
        home_after_configure = cache_values_after_configure.get(
            "CMAKE_HOME_DIRECTORY"
        )
        configurations_after_configure = cache_values_after_configure.get(
            "CMAKE_CONFIGURATION_TYPES", ""
        )
        home_matches_after_configure = (
            home_after_configure is not None
            and Path(home_after_configure).resolve() == expected_home
        )
        single_configuration_after_configure = (
            not configurations_after_configure.strip()
        )
        configure_passed = (
            not configure_phase["timed_out"]
            and configure_phase["return_code"] == 0
            and cache_present_after_configure
            and home_matches_after_configure
            and single_configuration_after_configure
        )
        configure_phase.update(
            {
                "cache_path": str(cache),
                "cache_present_after_configure": cache_present_after_configure,
                "cache_is_symlink_after_configure": cache.is_symlink(),
                "expected_cmake_home": str(expected_home),
                "cmake_home_directory_after_configure": home_after_configure,
                "cmake_home_matches_after_configure": home_matches_after_configure,
                "single_configuration_after_configure": (
                    single_configuration_after_configure
                ),
                "outcome": "PASS" if configure_passed else "FAIL_METHOD",
            }
        )
        if not configure_passed:
            record = {
                "build_directory": str(build_directory),
                "targets": targets,
                "binary_keys": binary_keys,
                "binaries_before_clean": binaries_before_clean,
                "configure": configure_phase,
                "clean": None,
                "build": None,
                "outcome": "FAIL_METHOD",
                "diagnostic": "configure_phase_failed",
            }
            write_json(group_directory / "result.json", record)
            result["builds"].append(record)
            result["outcome"] = "FAIL_METHOD"
            result["diagnostic"] = "clean_build_failed"
            return result
        clean_command = [
            str(cmake),
            "--build",
            str(build_directory),
            "--target",
            "clean",
        ]
        clean_phase = run_build_phase(
            clean_command,
            source_root,
            output_root,
            group_directory / "clean_stdout.txt",
            group_directory / "clean_stderr.txt",
            timeout_seconds,
        )
        cache_present_after_clean = cache.is_file() and not cache.is_symlink()
        cache_values_after_clean = (
            cmake_cache_values(cache) if cache_present_after_clean else {}
        )
        home_after_clean = cache_values_after_clean.get("CMAKE_HOME_DIRECTORY")
        configurations_after_clean = cache_values_after_clean.get(
            "CMAKE_CONFIGURATION_TYPES", ""
        )
        home_matches_after_clean = (
            home_after_clean is not None
            and Path(home_after_clean).resolve() == expected_home
        )
        single_configuration_after_clean = not configurations_after_clean.strip()
        binaries_after_clean = {
            binary_key: {
                "path": str(binary),
                "exists": path_present(binary),
                "is_file": binary.is_file(),
            }
            for binary_key, _target, binary, _cache in entries
        }
        clean_passed = (
            not clean_phase["timed_out"]
            and clean_phase["return_code"] == 0
            and cache_present_after_clean
            and home_matches_after_clean
            and single_configuration_after_clean
            and all(
                not binary_record["exists"]
                for binary_record in binaries_after_clean.values()
            )
        )
        clean_phase["binaries_after_clean"] = binaries_after_clean
        clean_phase["all_supplied_binaries_absent"] = all(
            not binary_record["exists"]
            for binary_record in binaries_after_clean.values()
        )
        clean_phase["cache_path"] = str(cache)
        clean_phase["cache_present_after_clean"] = cache_present_after_clean
        clean_phase["cache_is_symlink_after_clean"] = cache.is_symlink()
        clean_phase["cmake_home_directory_after_clean"] = home_after_clean
        clean_phase["cmake_home_matches_after_clean"] = home_matches_after_clean
        clean_phase["single_configuration_after_clean"] = (
            single_configuration_after_clean
        )
        clean_phase["outcome"] = "PASS" if clean_passed else "FAIL_METHOD"
        record: dict[str, Any] = {
            "build_directory": str(build_directory),
            "targets": targets,
            "binary_keys": binary_keys,
            "binaries_before_clean": binaries_before_clean,
            "configure": configure_phase,
            "clean": clean_phase,
            "build": None,
            "outcome": "FAIL_METHOD",
        }
        if not clean_passed:
            record["diagnostic"] = "clean_phase_failed_or_binary_survived"
            write_json(group_directory / "result.json", record)
            result["builds"].append(record)
            result["outcome"] = "FAIL_METHOD"
            result["diagnostic"] = "clean_build_failed"
            return result

        build_command = [
            str(cmake),
            "--build",
            str(build_directory),
            "--parallel",
            str(parallel),
            "--target",
            *targets,
        ]
        build_phase = run_build_phase(
            build_command,
            source_root,
            output_root,
            group_directory / "build_stdout.txt",
            group_directory / "build_stderr.txt",
            timeout_seconds,
        )

        cache = entries[0][3]
        cache_present = cache.is_file() and not cache.is_symlink()
        cache_values = cmake_cache_values(cache) if cache_present else {}
        home = cache_values.get("CMAKE_HOME_DIRECTORY")
        configurations = cache_values.get("CMAKE_CONFIGURATION_TYPES", "")
        single_configuration = not configurations.strip()
        expected_homes = {
            (source_root / registry["build_cmake_homes"][binary_key]).resolve()
            for binary_key, _target, _binary, _cache in entries
        }
        home_matches = (
            len(expected_homes) == 1
            and home is not None
            and Path(home).resolve() in expected_homes
        )
        binaries_after = {
            binary_key: {
                "path": str(binary),
                "exists": binary.is_file(),
                "executable": binary.is_file() and os.access(binary, os.X_OK),
                "sha256": sha256_file(binary) if binary.is_file() else None,
            }
            for binary_key, _target, binary, _cache in entries
        }
        passed = (
            not build_phase["timed_out"]
            and build_phase["return_code"] == 0
            and home_matches
            and single_configuration
            and all(
                record["exists"] and record["executable"]
                for record in binaries_after.values()
            )
        )
        build_phase.update(
            {
                "binaries_after_build": binaries_after,
                "all_supplied_binaries_recreated": all(
                    binary_record["exists"] and binary_record["executable"]
                    for binary_record in binaries_after.values()
                ),
                "outcome": "PASS" if passed else "FAIL_METHOD",
            }
        )
        record.update(
            {
                "cache_path": str(cache),
                "cache_present_after_build": cache_present,
                "cache_is_symlink_after_build": cache.is_symlink(),
                "cache_sha256": sha256_file(cache) if cache_present else None,
                "cmake_home_directory": home,
                "expected_cmake_homes": sorted(str(path) for path in expected_homes),
                "cmake_home_matches": home_matches,
                "single_configuration_after_build": single_configuration,
                "build": build_phase,
                "outcome": "PASS" if passed else "FAIL_METHOD",
                "diagnostic": None if passed else "build_phase_failed",
            }
        )
        write_json(group_directory / "result.json", record)
        result["builds"].append(record)
        if not passed:
            result["outcome"] = "FAIL_METHOD"
            result["diagnostic"] = "clean_build_failed"
            return result

    return result


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


def final_provenance_record(
    source_root: Path,
    output_root: Path,
    build_source_roots: tuple[Path, ...],
    registry_path: Path,
    source_commit: str,
    source_tree: str,
    registry_sha256: str,
    runner_sha256: str,
    binaries: dict[str, Path],
    binary_sha256s: dict[str, str],
) -> dict[str, Any]:
    final_commit = git_bytes(source_root, "rev-parse", "HEAD").decode().strip()
    final_tree = git_bytes(source_root, "rev-parse", "HEAD^{tree}").decode().strip()
    tracked_status = git_bytes(
        source_root, "status", "--porcelain=v1", "--untracked-files=no"
    ).decode(encoding="utf-8", errors="replace")
    final_untracked = untracked_source_record(
        source_root, output_root, build_source_roots
    )
    final_registry_sha256 = sha256_file(registry_path)
    final_runner_sha256 = sha256_file(SCRIPT_PATH)
    final_binaries: dict[str, dict[str, Any]] = {}
    checks = [
        equal_check("source_commit", final_commit, source_commit),
        equal_check("source_tree", final_tree, source_tree),
        equal_check("tracked_status", tracked_status, ""),
        equal_check(
            "unexpected_untracked_path_count",
            final_untracked["unexpected_path_count"],
            0,
        ),
        equal_check("registry_sha256", final_registry_sha256, registry_sha256),
        equal_check("runner_sha256", final_runner_sha256, runner_sha256),
    ]
    for binary_key in sorted(binaries):
        binary = binaries[binary_key]
        exists = path_present(binary)
        is_file = binary.is_file()
        executable = is_file and os.access(binary, os.X_OK)
        final_sha256 = sha256_file(binary) if is_file else None
        final_binaries[binary_key] = {
            "path": str(binary),
            "exists": exists,
            "is_file": is_file,
            "executable": executable,
            "sha256": final_sha256,
            "expected_sha256": binary_sha256s[binary_key],
        }
        checks.extend(
            [
                equal_check(f"binary_exists:{binary_key}", exists, True),
                equal_check(f"binary_executable:{binary_key}", executable, True),
                equal_check(
                    f"binary_sha256:{binary_key}",
                    final_sha256,
                    binary_sha256s[binary_key],
                ),
            ]
        )
    passed = all(check["passed"] for check in checks)
    return {
        "artifact_schema_version": 1,
        "canonical_registry": str(DEFAULT_REGISTRY.resolve()),
        "source_commit_before_execution": source_commit,
        "source_tree_before_execution": source_tree,
        "source_commit_after_execution": final_commit,
        "source_tree_after_execution": final_tree,
        "tracked_status_after_execution": tracked_status,
        "untracked_sources_after_execution": final_untracked,
        "registry_sha256_before_execution": registry_sha256,
        "registry_sha256_after_execution": final_registry_sha256,
        "runner_sha256_before_execution": runner_sha256,
        "runner_sha256_after_execution": final_runner_sha256,
        "binaries_after_execution": final_binaries,
        "checks": checks,
        "diagnostics": [check["metric"] for check in checks if not check["passed"]],
        "outcome": "PASS" if passed else "FAIL_METHOD",
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
    gates: dict[str, Any],
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
        rank_wrapper = (
            'rank_value="${OMPI_COMM_WORLD_RANK:-'
            "${PMI_RANK:-${PMIX_RANK:-${MV2_COMM_WORLD_RANK:-"
            '${SLURM_PROCID:-}}}}}"; '
            'case "$rank_value" in ""|*[!0-9]*) '
            'echo "invalid or missing MPI rank" >&2; exit 97;; esac; '
            'exec "$1" "$2" "$3" '
            '"--gtest_output=json:${4}/gtest_rank_${rank_value}.json"'
        )
        command = [
            str(mpiexec),
            "--oversubscribe",
            "-n",
            str(ranks),
            "/bin/sh",
            "-c",
            rank_wrapper,
            "qualification-rank",
            str(binary),
            f"--gtest_filter={test_filter}",
            "--gtest_color=no",
            str(group_directory),
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
        launch_mode=("direct_serial" if ranks == 1 else "mpi"),
        required_simultaneous_process_samples=(ranks + 1 if ranks > 1 else 1),
    )
    stdout = stdout_path.read_text(encoding="utf-8", errors="replace")
    stderr = stderr_path.read_text(encoding="utf-8", errors="replace")
    diagnostic: str | None = None
    gtest_result_error: str | None = None
    mpi_gtest_results: list[dict[str, Any]] = []
    if ranks == 1 and gtest_path is not None and gtest_path.is_file():
        try:
            document = json.loads(gtest_path.read_text(encoding="utf-8"))
            if not isinstance(document, dict):
                raise ValueError("serial gtest result is not an object")
            checks = evaluate_serial_result(
                group["tests"],
                document,
                resources["return_code"],
                resources["termination_reason"],
                gates,
            )
        except (
            AttributeError,
            json.JSONDecodeError,
            OSError,
            RecursionError,
            TypeError,
            ValueError,
        ) as error:
            diagnostic = "gtest_result_invalid"
            gtest_result_error = type(error).__name__
            checks = [
                equal_check("process_return_code", resources["return_code"], 0),
                equal_check(
                    "termination_reason", resources["termination_reason"], None
                ),
                equal_check("gtest_result_present", True, True),
                equal_check("gtest_result_valid", False, True),
            ]
    elif ranks == 1:
        diagnostic = "gtest_result_missing"
        checks = [
            equal_check("process_return_code", resources["return_code"], 0),
            equal_check("termination_reason", resources["termination_reason"], None),
            equal_check("gtest_result_present", False, True),
        ]
    else:
        checks = evaluate_mpi_result(
            group["tests"],
            group["gtest_output_copies"],
            stdout,
            stderr,
            resources["return_code"],
            resources["termination_reason"],
            gates,
        )
        mpi_gtest_checks, mpi_gtest_results = evaluate_mpi_gtest_results(
            group_directory,
            group["tests"],
            ranks,
            gates,
        )
        checks.extend(mpi_gtest_checks)
        if not all(check["passed"] for check in mpi_gtest_checks):
            diagnostic = "mpi_gtest_result_contract_failed"
    checks.append(
        equal_check(
            "resource_monitoring_outcome",
            resources.get("resource_monitoring_outcome"),
            "PASS",
        )
    )
    if resources.get("termination") is not None:
        checks.append(
            equal_check(
                "all_session_processes_terminated",
                resources["termination"].get("all_session_processes_terminated"),
                True,
            )
        )
    passed = bool(checks) and all(check["passed"] for check in checks)
    if ranks == 1 and not passed and diagnostic is None:
        diagnostic = "gtest_result_contract_failed"
    result = {
        "group_id": group["id"],
        "command": command,
        "mpi_ranks": ranks,
        "gtest_output_copies": group["gtest_output_copies"],
        "expected_tests": group["tests"],
        "execution": execution,
        "resources": resources,
        "diagnostic": diagnostic,
        "gtest_result_error": gtest_result_error,
        "mpi_gtest_results": mpi_gtest_results,
        "checks": checks,
        "outcome": "PASS" if passed else "FAIL_METHOD",
    }
    write_json(group_directory / "result.json", result)
    return result


def write_checksums(output_directory: Path) -> None:
    if output_directory.is_symlink() or not output_directory.is_dir():
        raise RuntimeError(
            f"refusing non-directory or symlink artifact root: {output_directory}"
        )
    entries = []
    for path in sorted(output_directory.rglob("*")):
        if path.is_symlink():
            raise RuntimeError(f"refusing symlink artifact path: {path}")
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
    parser.add_argument("--assembly-binary", type=Path, required=True)
    parser.add_argument("--physics-binary", type=Path, required=True)
    parser.add_argument("--application-binary", type=Path, required=True)
    parser.add_argument("--assembly-mpi-binary", type=Path, required=True)
    parser.add_argument("--application-mpi-binary", type=Path, required=True)
    parser.add_argument("--timestepping-binary", type=Path)
    parser.add_argument("--mpiexec", type=Path, default=Path("/usr/bin/mpiexec"))
    parser.add_argument("--cmake", type=Path, default=Path("/usr/bin/cmake"))
    parser.add_argument("--build-parallel", type=int, default=2)
    parser.add_argument("--build-timeout-seconds", type=int, default=3600)
    parser.add_argument("--source-root", type=Path, default=SCRIPT_PATH.parents[3])
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    source_root = arguments.source_root.resolve()
    registry_path = arguments.registry.resolve()
    output_directory = arguments.output.resolve()
    mpiexec = arguments.mpiexec.resolve()
    cmake = arguments.cmake.resolve()
    binaries = {
        "geometry": arguments.geometry_binary.resolve(),
        "level_set": arguments.level_set_binary.resolve(),
        "systems": arguments.systems_binary.resolve(),
        "assembly": arguments.assembly_binary.resolve(),
        "physics": arguments.physics_binary.resolve(),
        "application": arguments.application_binary.resolve(),
        "assembly_mpi": arguments.assembly_mpi_binary.resolve(),
        "application_mpi": arguments.application_mpi_binary.resolve(),
    }
    if arguments.timestepping_binary is not None:
        binaries["timestepping"] = (
            arguments.timestepping_binary.resolve()
        )
    if set(binaries) != QUALIFICATION_BINARY_KEYS:
        missing = sorted(QUALIFICATION_BINARY_KEYS - set(binaries))
        unexpected = sorted(set(binaries) - QUALIFICATION_BINARY_KEYS)
        parser.error(
            "qualification binary arguments do not match the declared "
            f"binary keys; missing={missing}, unexpected={unexpected}"
        )
    registry_sha256 = sha256_file(registry_path)
    runner_sha256 = sha256_file(SCRIPT_PATH)
    registry = load_registry(registry_path)
    if sha256_file(registry_path) != registry_sha256:
        raise SystemExit("canonical registry changed while it was being loaded")
    if output_directory.exists():
        raise SystemExit(f"refusing to replace output directory: {output_directory}")
    if not mpiexec.is_file():
        raise SystemExit(f"MPI launcher is not a file: {mpiexec}")
    if not cmake.is_file() or not os.access(cmake, os.X_OK):
        raise SystemExit(f"CMake executable is not usable: {cmake}")
    if arguments.build_parallel <= 0 or arguments.build_timeout_seconds <= 0:
        raise SystemExit("clean build parallelism and timeout must be positive")
    build_source_roots = tuple(
        sorted(
            {
                (source_root / relative_home).resolve()
                for relative_home in registry["build_cmake_homes"].values()
            },
            key=str,
        )
    )
    git_top_level = Path(
        git_bytes(source_root, "rev-parse", "--show-toplevel")
        .decode()
        .strip()
    ).resolve()
    if git_top_level != source_root:
        raise SystemExit(
            "qualification source root must equal the repository top level"
        )
    if any(
        not root.is_dir() or not path_is_within(root, source_root)
        for root in build_source_roots
    ):
        raise SystemExit(
            "qualification CMake source homes must be directories inside the repository"
        )
    git_directory_text = (
        git_bytes(source_root, "rev-parse", "--git-dir").decode().strip()
    )
    git_directory_path = Path(git_directory_text)
    git_directory = (
        git_directory_path
        if git_directory_path.is_absolute()
        else source_root / git_directory_path
    ).resolve()
    git_common_directory_text = (
        git_bytes(source_root, "rev-parse", "--git-common-dir").decode().strip()
    )
    git_common_directory_path = Path(git_common_directory_text)
    git_common_directory = (
        git_common_directory_path
        if git_common_directory_path.is_absolute()
        else source_root / git_common_directory_path
    ).resolve()
    repository_metadata_directories = tuple(
        sorted({git_directory, git_common_directory}, key=str)
    )
    selected_build_directories = tuple(
        sorted(
            {
                cache.resolve().parent
                for binary in binaries.values()
                if (cache := find_cmake_cache(binary)) is not None
            },
            key=str,
        )
    )
    if any(
        paths_overlap(build_directory, source_home)
        for build_directory in selected_build_directories
        for source_home in build_source_roots
    ):
        raise SystemExit(
            "qualification requires CMake build directories outside source homes"
        )
    if any(
        paths_overlap(build_directory, metadata_directory)
        for build_directory in selected_build_directories
        for metadata_directory in repository_metadata_directories
    ):
        raise SystemExit(
            "qualification requires CMake build directories outside repository metadata"
        )
    protected_output_roots = (
        *build_source_roots,
        *selected_build_directories,
        *repository_metadata_directories,
    )
    if any(
        path_is_within(output_directory, protected_root)
        for protected_root in protected_output_roots
    ):
        raise SystemExit(
            "qualification output must be outside source, build, and repository metadata directories"
        )
    path_contract = {
        "source_root": str(source_root),
        "git_top_level": str(git_top_level),
        "git_directory": str(git_directory),
        "git_common_directory": str(git_common_directory),
        "repository_metadata_directories": [
            str(path) for path in repository_metadata_directories
        ],
        "output_directory": str(output_directory),
        "build_source_roots": [str(path) for path in build_source_roots],
        "selected_build_directories": [
            str(path) for path in selected_build_directories
        ],
        "source_root_matches_git_top_level": True,
        "build_directories_disjoint_from_source_homes": True,
        "build_directories_disjoint_from_repository_metadata": True,
        "output_disjoint_from_protected_roots": True,
    }

    tracked_status = git_bytes(
        source_root, "status", "--porcelain=v1", "--untracked-files=no"
    )
    if tracked_status:
        raise SystemExit("qualification requires clean tracked sources")
    initial_untracked = untracked_source_record(
        source_root, ignored_source_roots=build_source_roots
    )
    if initial_untracked["unexpected_path_count"] != 0:
        raise SystemExit(
            "qualification requires a source worktree with zero untracked paths"
        )
    source_commit = git_bytes(source_root, "rev-parse", "HEAD").decode().strip()
    source_tree = git_bytes(source_root, "rev-parse", "HEAD^{tree}").decode().strip()
    output_directory.mkdir(parents=True, exist_ok=False)

    build_preflight = run_clean_builds(
        registry,
        binaries,
        source_root,
        output_directory,
        cmake,
        arguments.build_parallel,
        arguments.build_timeout_seconds,
    )
    build_preflight["path_contract"] = path_contract
    build_preflight["qualification_inputs"] = {
        "canonical_registry": str(DEFAULT_REGISTRY.resolve()),
        "registry_sha256": registry_sha256,
        "runner": str(SCRIPT_PATH),
        "runner_sha256": runner_sha256,
    }
    build_preflight["pre_build_source_provenance"] = {
        "tracked_sources_clean": True,
        "untracked_sources": initial_untracked,
        "source_commit": source_commit,
        "source_tree": source_tree,
    }
    if (
        build_preflight["source_commit_before_build"] != source_commit
        or build_preflight["source_tree_before_build"] != source_tree
    ):
        build_preflight["outcome"] = "FAIL_METHOD"
        build_preflight["diagnostic"] = "source_revision_changed_before_build"
    post_build_status = git_bytes(
        source_root, "status", "--porcelain=v1", "--untracked-files=no"
    )
    post_build_untracked = untracked_source_record(
        source_root, output_directory, build_source_roots
    )
    post_build_commit = git_bytes(source_root, "rev-parse", "HEAD").decode().strip()
    post_build_tree = (
        git_bytes(source_root, "rev-parse", "HEAD^{tree}").decode().strip()
    )
    if post_build_status:
        build_preflight["outcome"] = "FAIL_METHOD"
        build_preflight["diagnostic"] = "clean_build_changed_tracked_sources"
    if post_build_untracked["unexpected_path_count"] != 0:
        build_preflight["outcome"] = "FAIL_METHOD"
        build_preflight["diagnostic"] = "clean_build_created_untracked_sources"
    if post_build_commit != source_commit or post_build_tree != source_tree:
        build_preflight["outcome"] = "FAIL_METHOD"
        build_preflight["diagnostic"] = "source_revision_changed_during_build"
    build_preflight["post_build_tracked_sources_clean"] = not bool(post_build_status)
    build_preflight["post_build_untracked_sources"] = post_build_untracked
    build_preflight["post_build_source_commit"] = post_build_commit
    build_preflight["post_build_source_tree"] = post_build_tree

    post_build_validation: list[dict[str, Any]] = []
    for label, binary in binaries.items():
        executable = binary.is_file() and os.access(binary, os.X_OK)
        post_build_validation.append(
            {
                "binary_key": label,
                "binary_path": str(binary),
                "executable": executable,
                "listed_tests_present": None,
                "list_timeout_seconds": GTEST_LIST_TIMEOUT_SECONDS,
                "missing_tests": [],
                "passed": executable,
            }
        )
    configured_by_binary: dict[str, set[str]] = {}
    for group in registry["groups"]:
        configured_by_binary.setdefault(group["binary"], set()).update(group["tests"])
    for binary_key, expected in configured_by_binary.items():
        validation = next(
            item for item in post_build_validation if item["binary_key"] == binary_key
        )
        if not validation["executable"]:
            continue
        try:
            missing = sorted(expected - listed_gtests(binaries[binary_key]))
        except (
            OSError,
            subprocess.CalledProcessError,
            subprocess.TimeoutExpired,
        ) as error:
            missing = sorted(expected)
            validation["list_command_failed"] = True
            validation["list_command_diagnostic"] = type(error).__name__
        validation["missing_tests"] = missing
        validation["listed_tests_present"] = not missing
        validation["passed"] = validation["passed"] and not missing
    build_preflight["post_build_validation"] = post_build_validation
    if not all(item["passed"] for item in post_build_validation):
        build_preflight["outcome"] = "FAIL_METHOD"
        build_preflight["diagnostic"] = "post_build_binary_validation_failed"
    write_json(output_directory / "build_preflight.json", build_preflight)
    if build_preflight["outcome"] != "PASS":
        failure_summary = {
            "matrix_id": registry["matrix_id"],
            "source_commit": source_commit,
            "overall_outcome": "FAIL_METHOD",
            "failure_stage": "clean_build_preflight",
            "diagnostic": build_preflight["diagnostic"],
        }
        write_json(output_directory / "summary.json", failure_summary)
        write_checksums(output_directory)
        print(output_directory)
        print("FAIL_METHOD")
        return 2

    binary_sha256s = {
        binary_key: sha256_file(binary) for binary_key, binary in binaries.items()
    }

    write_json(
        output_directory / "manifest.json",
        {
            "artifact_schema_version": 1,
            "matrix_id": registry["matrix_id"],
            "matrix_status_at_execution": registry["status"],
            "registry_sha256": registry_sha256,
            "runner_sha256": runner_sha256,
            "work_package": registry["work_package"],
            "findings": registry["findings"],
            "model_envelope": registry["model_envelope"],
            "build_targets": registry["build_targets"],
            "build_cmake_homes": registry["build_cmake_homes"],
            "path_contract": path_contract,
            "source_commit": source_commit,
            "source_tree": source_tree,
            "groups": registry["groups"],
            "closure_contract": registry["closure_contract"],
            "quantitative_evidence": registry["quantitative_evidence"],
            "qualification_scope": registry["qualification_scope"],
        },
    )
    write_json(
        output_directory / "build.json",
        {
            "source_commit": source_commit,
            "source_tree": source_tree,
            "tracked_sources_clean": True,
            "clean_build_preflight": "build_preflight.json",
            "clean_build_preflight_outcome": build_preflight["outcome"],
            "untracked_sources_before_build": initial_untracked,
            "path_contract": path_contract,
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
            "quantitative_evidence": registry["quantitative_evidence"],
            "qualification_scope": registry["qualification_scope"],
        },
    )

    group_results = [
        run_gtest_group(
            group,
            registry["gates"],
            binaries,
            mpiexec,
            source_root,
            output_directory,
        )
        for group in registry["groups"]
    ]
    quantitative_result = evaluate_quantitative_evidence(registry, output_directory)
    write_json(output_directory / "quantitative_evidence.json", quantitative_result)
    recorded_properties_result = evaluate_group_recorded_properties(
        registry, output_directory
    )
    write_json(
        output_directory / "group_recorded_properties.json",
        recorded_properties_result,
    )
    groups_passed = all(result["outcome"] == "PASS" for result in group_results)
    final_provenance = final_provenance_record(
        source_root,
        output_directory,
        build_source_roots,
        registry_path,
        source_commit,
        source_tree,
        registry_sha256,
        runner_sha256,
        binaries,
        binary_sha256s,
    )
    write_json(output_directory / "final_provenance.json", final_provenance)
    passed = (
        groups_passed
        and quantitative_result["outcome"] == "PASS"
        and recorded_properties_result["outcome"] == "PASS"
        and final_provenance["outcome"] == "PASS"
    )
    summary = {
        "matrix_id": registry["matrix_id"],
        "source_commit": source_commit,
        "distinct_test_count": registry["gates"]["expected_distinct_test_count"],
        "group_outcomes": {
            result["group_id"]: result["outcome"] for result in group_results
        },
        "quantitative_evidence_outcome": quantitative_result["outcome"],
        "quantitative_evidence_check_count": quantitative_result[
            "declared_check_count"
        ],
        "group_recorded_properties_outcome": recorded_properties_result["outcome"],
        "group_recorded_properties_check_count": recorded_properties_result[
            "declared_check_count"
        ],
        "final_provenance_outcome": final_provenance["outcome"],
        "final_provenance_diagnostics": final_provenance["diagnostics"],
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
        "- Quantitative evidence: "
        f"**{quantitative_result['outcome']}** "
        f"({quantitative_result['declared_check_count']} checks)",
        "- Group recorded properties: "
        f"**{recorded_properties_result['outcome']}** "
        f"({recorded_properties_result['declared_check_count']} checks)",
        f"- Final provenance: **{final_provenance['outcome']}**",
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

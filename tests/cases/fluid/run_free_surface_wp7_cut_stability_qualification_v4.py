#!/usr/bin/env python3
"""Run the versioned WP-7 topology and node-crossing prerequisite matrix."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

sys.dont_write_bytecode = True


SCRIPT_PATH = Path(__file__).resolve()
REPOSITORY_ROOT = SCRIPT_PATH.parents[3]
DEFAULT_REGISTRY = SCRIPT_PATH.with_name(
    "free_surface_wp7_cut_stability_qualification_revision_v4.json"
)
PARENT_RUNNER_PATH = SCRIPT_PATH.with_name(
    "run_free_surface_wp7_cut_stability_qualification.py"
)
BASE_MATRIX_PATH = SCRIPT_PATH.with_name(
    "free_surface_wp7_cut_stability_qualification_matrix.json"
)
EXPECTED_PARENT_RUNNER_SHA256 = (
    "30c59eb725ba88b87447b935a915c6561b9ae25f082d93052d5cb9be4e337f6f"
)
EXPECTED_BASE_MATRIX_SHA256 = (
    "a49cadbcbe1b56bf69e4520a5281fc942ac4dc9de82da4e3bdaa083d6334ab1f"
)
EXPECTED_REGISTRY_SHA256 = (
    "16e6bbe50bb6b6597ac43e4c7c81e9df7469a833b48aef631c19acff8119afe0"
)
EXPECTED_FOCUSED_TEST_SHA256 = (
    "62c5c9531fb0a13bab8f360b4c4976a02fb6dd86e5bb126f5b7f79fec71fdc90"
)
EXPECTED_IMPLEMENTATION_SOURCE_SHA256 = (
    "5e93afc9f6678d1328c94d46ec5595ebd6dc649856cc612297f45843756c40d2"
)
EXPECTED_IMPLEMENTATION_SOURCE_COMMIT = (
    "79f05c22bf57712c16a9132ac9115a0b5efe0dce"
)
EXPECTED_IMPLEMENTATION_SOURCE_PATH = (
    "Code/Source/solver/Physics/Tests/Unit/test_FreeSurfaceCutStability.cpp"
)
EXPECTED_FOCUSED_TEST_PATH = (
    "tests/test_free_surface_wp7_cut_stability_qualification_runner_v4.py"
)
EXPECTED_MATRIX_ID = "free_surface_wp7_cut_stability_v4"
EXPECTED_SCOPE = (
    "Executable prerequisite evidence for the production connected, "
    "disconnected, and rootless active-feature policy and the physically "
    "refined affine-probe node-crossing response. Five prospective "
    "manufactured-error and simulation-exit rows remain absent, and "
    "production-preconditioner spread remains unresolved, so this revision "
    "does not close FSR-07, WP-7, or Q1."
)
EXPECTED_CLOSURE_STATE = "BLOCKED_BY_FIVE_PROSPECTIVE_EVIDENCE_ROWS"
EXPECTED_TOPOLOGY_TEST = (
    "FreeSurfaceCutStability."
    "ConnectedDisconnectedAndRootlessFeaturesReportTopologyPolicy"
)
EXPECTED_NODE_CROSSING_TEST = (
    "FreeSurfaceCutStability."
    "ContinuousNodeCrossingHasNoUnreportedOperatorOrSolutionJump"
)
EXPECTED_PROMOTED_TESTS = {
    EXPECTED_TOPOLOGY_TEST,
    EXPECTED_NODE_CROSSING_TEST,
}
EXPECTED_CLOSURE_REQUEST_POLICY = {
    "accepted_claim": "topology_and_node_crossing_prerequisite",
    "rejected_claims": ["fsr07_closure", "wp7_closure", "q1_closure"],
    "diagnostic": (
        "The two promoted rows are prerequisite evidence only; manufactured "
        "errors, production-preconditioner spread, distributed completion, "
        "and simulation exits remain open."
    ),
}
EXPECTED_DISPOSITION = {
    "fsr07_closed": False,
    "wp7_closed": False,
    "q1_closed": False,
}
EXPECTED_RUNTIME_GATES = {
    "expected_group_count": 4,
    "expected_distinct_test_count": 16,
    "expected_quantitative_evidence_count": 67,
    "expected_failures": 0,
    "expected_errors": 0,
    "expected_disabled": 0,
    "expected_skipped": 0,
}
EXPECTED_GROUP_IDS = {
    "wp7_finite_foundation_serial",
    "wp7_required_regimes_serial",
    "wp7_partition_mpi_2",
    "wp7_partition_mpi_4",
}
EXPECTED_TOPOLOGY_PROPERTIES = {
    "wp7_active_cell_topology_case_count": ("integer", "equal", 3),
    "wp7_active_cell_topology_feature_count": ("integer", "equal", 5),
    "wp7_active_cell_topology_rooted_feature_count": ("integer", "equal", 4),
    "wp7_active_cell_topology_rootless_feature_count": ("integer", "equal", 1),
    "wp7_active_cell_topology_rootless_retained_physical_volume": (
        "real",
        "less_than_or_equal",
        1.000000000001,
    ),
    "wp7_active_cell_topology_velocity_pressure_mismatch_count": (
        "integer",
        "equal",
        0,
    ),
}
EXPECTED_NODE_CROSSING_PROPERTIES = {
    "wp7_node_crossing_refined_level_count": ("integer", "equal", 3),
    "wp7_node_crossing_refined_reported_changed_transition_count": (
        "integer",
        "equal",
        3,
    ),
    "wp7_node_crossing_refined_maximum_linear_solve_relative_residual": (
        "real",
        "less_than_or_equal",
        1.0e-10,
    ),
    "wp7_node_crossing_refined_operator_global_order": (
        "real",
        "greater_than_or_equal",
        0.2,
    ),
    "wp7_node_crossing_refined_residual_global_order": (
        "real",
        "greater_than_or_equal",
        0.2,
    ),
    "wp7_node_crossing_refined_solved_state_global_order": (
        "real",
        "greater_than_or_equal",
        0.2,
    ),
    "wp7_node_crossing_refined_operator_maximum_adjacent_growth": (
        "real",
        "less_than_or_equal",
        1.1,
    ),
    "wp7_node_crossing_refined_residual_maximum_adjacent_growth": (
        "real",
        "less_than_or_equal",
        1.1,
    ),
    "wp7_node_crossing_refined_solved_state_maximum_adjacent_growth": (
        "real",
        "less_than_or_equal",
        1.1,
    ),
    "wp7_node_crossing_refined_operator_difference_2": (
        "real",
        "less_than_or_equal",
        0.6,
    ),
    "wp7_node_crossing_refined_residual_difference_2": (
        "real",
        "less_than_or_equal",
        0.65,
    ),
    "wp7_node_crossing_refined_solved_state_difference_2": (
        "real",
        "less_than_or_equal",
        0.2,
    ),
}
EXPECTED_PROMOTION_PROPERTIES = {
    EXPECTED_TOPOLOGY_TEST: EXPECTED_TOPOLOGY_PROPERTIES,
    EXPECTED_NODE_CROSSING_TEST: EXPECTED_NODE_CROSSING_PROPERTIES,
}
EXPECTED_TOP_LEVEL_KEYS = {
    "schema_version",
    "matrix_id",
    "status",
    "work_package",
    "findings",
    "implementation_source_commit",
    "base_artifacts",
    "implementation_source",
    "qualification_scope",
    "closure_request_policy",
    "qualification_disposition",
    "closure_state",
    "promotions",
    "remaining_prospective_test_count",
    "closure_contract",
    "runtime_gates",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_parent_runner() -> Any:
    if sha256_file(PARENT_RUNNER_PATH) != EXPECTED_PARENT_RUNNER_SHA256:
        raise RuntimeError("WP-7 parent runner bytes changed")
    if sha256_file(BASE_MATRIX_PATH) != EXPECTED_BASE_MATRIX_SHA256:
        raise RuntimeError("WP-7 base matrix bytes changed")
    # The parent imports the shared strict runner by its conventional module
    # name and then customizes that module. Load a private shared-runner copy
    # while importing the parent so an earlier historical revision in the
    # same test process cannot leak its monkeypatches into V4.
    shared_name = "run_free_surface_wp2_geometry_qualification"
    shared_path = PARENT_RUNNER_PATH.with_name(f"{shared_name}.py")
    shared_specification = importlib.util.spec_from_file_location(
        shared_name,
        shared_path,
    )
    specification = importlib.util.spec_from_file_location(
        "_free_surface_wp7_cut_stability_v4_parent",
        PARENT_RUNNER_PATH,
    )
    if (
        shared_specification is None
        or shared_specification.loader is None
        or specification is None
        or specification.loader is None
    ):
        raise RuntimeError("cannot load the WP-7 parent runner")
    previous_shared = sys.modules.get(shared_name)
    shared_module = importlib.util.module_from_spec(shared_specification)
    module = importlib.util.module_from_spec(specification)
    try:
        sys.modules[shared_name] = shared_module
        shared_specification.loader.exec_module(shared_module)
        sys.modules[specification.name] = module
        specification.loader.exec_module(module)
    finally:
        if previous_shared is None:
            sys.modules.pop(shared_name, None)
        else:
            sys.modules[shared_name] = previous_shared
    return module


_parent = _load_parent_runner()
strict_runner = _parent.strict_runner
_CACHE_ATTRIBUTE = "_wp7_v4_pristine_parent_contract"
if hasattr(strict_runner, _CACHE_ATTRIBUTE):
    _cached_parent_contract = getattr(strict_runner, _CACHE_ATTRIBUTE)
    _BASE_REGISTRY = copy.deepcopy(_cached_parent_contract["base_registry"])
    _shared_write_json = _cached_parent_contract["write_json"]
    _shared_write_text = _cached_parent_contract["write_text"]
else:
    _BASE_REGISTRY = _parent.load_registry(BASE_MATRIX_PATH)
    _shared_write_json = _parent._shared_write_json
    _shared_write_text = _parent._shared_write_text
    setattr(
        strict_runner,
        _CACHE_ATTRIBUTE,
        {
            "base_registry": copy.deepcopy(_BASE_REGISTRY),
            "write_json": _shared_write_json,
            "write_text": _shared_write_text,
        },
    )


def _git_result(*arguments: str) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        ["/share/software/user/open/git/2.45.1/bin/git", *arguments],
        cwd=REPOSITORY_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def validate_implementation_binding() -> dict[str, Any]:
    source_path = REPOSITORY_ROOT / EXPECTED_IMPLEMENTATION_SOURCE_PATH
    source_sha256 = sha256_file(source_path)
    if source_sha256 != EXPECTED_IMPLEMENTATION_SOURCE_SHA256:
        raise ValueError("WP-7 promoted implementation source bytes changed")
    focused_path = REPOSITORY_ROOT / EXPECTED_FOCUSED_TEST_PATH
    if sha256_file(focused_path) != EXPECTED_FOCUSED_TEST_SHA256:
        raise ValueError("WP-7 V4 focused contract-test bytes changed")
    ancestor = _git_result(
        "merge-base",
        "--is-ancestor",
        EXPECTED_IMPLEMENTATION_SOURCE_COMMIT,
        "HEAD",
    )
    if ancestor.returncode != 0:
        raise ValueError("WP-7 implementation source commit is not an ancestor")
    source_diff = _git_result(
        "diff",
        "--quiet",
        EXPECTED_IMPLEMENTATION_SOURCE_COMMIT,
        "HEAD",
        "--",
        EXPECTED_IMPLEMENTATION_SOURCE_PATH,
    )
    if source_diff.returncode != 0:
        raise ValueError("WP-7 promoted source changed after its bound commit")
    return {
        "implementation_source_commit": EXPECTED_IMPLEMENTATION_SOURCE_COMMIT,
        "implementation_source_path": EXPECTED_IMPLEMENTATION_SOURCE_PATH,
        "implementation_source_sha256": source_sha256,
        "focused_contract_test_path": EXPECTED_FOCUSED_TEST_PATH,
        "focused_contract_test_sha256": EXPECTED_FOCUSED_TEST_SHA256,
    }


def _promotion_property_contract(
    promotion: Any,
) -> dict[str, tuple[str, str, int | float]]:
    if not isinstance(promotion, dict) or set(promotion) != {
        "test",
        "source_state_before",
        "source_state_after",
        "quantitative_evidence",
    }:
        raise ValueError("WP-7 V4 promotion is malformed")
    test = promotion["test"]
    expected = EXPECTED_PROMOTION_PROPERTIES.get(test)
    if expected is None:
        raise ValueError("WP-7 V4 promotes an unexpected test")
    if (
        promotion["source_state_before"] != "prospective"
        or promotion["source_state_after"] != "executable"
    ):
        raise ValueError("WP-7 V4 promotion states changed")
    entries = promotion["quantitative_evidence"]
    if not isinstance(entries, list) or len(entries) != len(expected):
        raise ValueError("WP-7 V4 promotion gate count changed")
    result: dict[str, tuple[str, str, int | float]] = {}
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {
            "test",
            "property",
            "type",
            "relation",
            "threshold",
        }:
            raise ValueError("WP-7 V4 promotion gate is malformed")
        if entry["test"] != test:
            raise ValueError("WP-7 V4 promotion gate cites the wrong test")
        property_name = entry["property"]
        if property_name in result:
            raise ValueError("WP-7 V4 promotion gate is duplicated")
        result[property_name] = (
            entry["type"],
            entry["relation"],
            entry["threshold"],
        )
    return result


def validate_revision_contract(registry: dict[str, Any]) -> dict[str, Any]:
    if set(registry) != EXPECTED_TOP_LEVEL_KEYS:
        raise ValueError("WP-7 V4 revision keys changed")
    if registry.get("schema_version") != 1:
        raise ValueError("unsupported WP-7 V4 revision schema")
    if registry.get("matrix_id") != EXPECTED_MATRIX_ID:
        raise ValueError("unexpected WP-7 V4 matrix id")
    if registry.get("status") != "FROZEN_BEFORE_EXECUTION":
        raise ValueError("WP-7 V4 matrix is not frozen before execution")
    if registry.get("work_package") != "WP-7" or registry.get("findings") != [
        "FSR-07"
    ]:
        raise ValueError("WP-7 V4 ownership changed")
    if registry.get("implementation_source_commit") != (
        EXPECTED_IMPLEMENTATION_SOURCE_COMMIT
    ):
        raise ValueError("WP-7 V4 implementation commit changed")
    if registry.get("base_artifacts") != {
        "matrix_path": str(BASE_MATRIX_PATH.relative_to(REPOSITORY_ROOT)),
        "matrix_sha256": EXPECTED_BASE_MATRIX_SHA256,
        "runner_path": str(PARENT_RUNNER_PATH.relative_to(REPOSITORY_ROOT)),
        "runner_sha256": EXPECTED_PARENT_RUNNER_SHA256,
    }:
        raise ValueError("WP-7 V4 base-artifact binding changed")
    if registry.get("implementation_source") != {
        "path": EXPECTED_IMPLEMENTATION_SOURCE_PATH,
        "sha256": EXPECTED_IMPLEMENTATION_SOURCE_SHA256,
    }:
        raise ValueError("WP-7 V4 implementation-source binding changed")
    if registry.get("qualification_scope") != EXPECTED_SCOPE:
        raise ValueError("WP-7 V4 qualification scope changed")
    if registry.get("closure_request_policy") != (
        EXPECTED_CLOSURE_REQUEST_POLICY
    ):
        raise ValueError("WP-7 V4 closure-request policy changed")
    if registry.get("qualification_disposition") != EXPECTED_DISPOSITION:
        raise ValueError("WP-7 V4 qualification disposition changed")
    if registry.get("closure_state") != EXPECTED_CLOSURE_STATE:
        raise ValueError("WP-7 V4 closure state changed")
    if registry.get("remaining_prospective_test_count") != 5:
        raise ValueError("WP-7 V4 remaining prospective count changed")
    if registry.get("runtime_gates") != EXPECTED_RUNTIME_GATES:
        raise ValueError("WP-7 V4 runtime gates changed")

    promotions = registry.get("promotions")
    if not isinstance(promotions, list) or len(promotions) != 2:
        raise ValueError("WP-7 V4 needs exactly two promotions")
    observed_promotions: dict[
        str, dict[str, tuple[str, str, int | float]]
    ] = {}
    for promotion in promotions:
        test = promotion.get("test") if isinstance(promotion, dict) else None
        if not isinstance(test, str) or test in observed_promotions:
            raise ValueError("WP-7 V4 promotion identity is invalid")
        observed_promotions[test] = _promotion_property_contract(promotion)
    if observed_promotions != EXPECTED_PROMOTION_PROPERTIES:
        raise ValueError("WP-7 V4 quantitative promotion gates changed")

    for test in EXPECTED_PROMOTED_TESTS:
        if test not in _BASE_REGISTRY["prospective_tests"]:
            raise ValueError("WP-7 V4 test was not prospective in the base matrix")
        if test in _BASE_REGISTRY["executable_tests"]:
            raise ValueError("WP-7 V4 test was already executable in the base matrix")

    source_text = (
        REPOSITORY_ROOT / EXPECTED_IMPLEMENTATION_SOURCE_PATH
    ).read_text(encoding="utf-8")
    for promoted_test, properties in EXPECTED_PROMOTION_PROPERTIES.items():
        suite, test = promoted_test.split(".", 1)
        if suite != "FreeSurfaceCutStability" or not re.search(
            r"TEST\(\s*FreeSurfaceCutStability\s*,\s*"
            + re.escape(test)
            + r"\s*\)",
            source_text,
            re.MULTILINE,
        ):
            raise ValueError("WP-7 V4 promoted test implementation is absent")
        for property_name in properties:
            source_token = (
                property_name[:-1] if property_name.endswith("_2") else property_name
            )
            if f'"{source_token}"' not in source_text:
                raise ValueError(
                    f"WP-7 V4 promoted property is absent: {property_name}"
                )

    claims = registry.get("closure_contract")
    if not isinstance(claims, list) or len(claims) != 3:
        raise ValueError("WP-7 V4 closure contract is incomplete")
    claim_names = {claim.get("claim") for claim in claims if isinstance(claim, dict)}
    if claim_names != {
        "selected_combined_method_and_finite_cut_foundation_remain_gated",
        "connected_disconnected_and_rootless_feature_policy_is_executable",
        "physically_refined_node_crossing_response_is_executable",
    }:
        raise ValueError("WP-7 V4 closure claims changed")
    return registry


def build_runtime_registry(revision: dict[str, Any]) -> dict[str, Any]:
    runtime = copy.deepcopy(_BASE_REGISTRY)
    executable = set(runtime["executable_tests"]) | EXPECTED_PROMOTED_TESTS
    prospective = set(runtime["prospective_tests"]) - EXPECTED_PROMOTED_TESTS

    groups: list[dict[str, Any]] = []
    for base_group in runtime["groups"]:
        group = copy.deepcopy(base_group)
        group["tests"] = [test for test in group["tests"] if test in executable]
        if group["tests"]:
            groups.append(group)
    if {group["id"] for group in groups} != EXPECTED_GROUP_IDS:
        raise ValueError("WP-7 V4 runtime group selection changed")

    promoted_evidence = [
        copy.deepcopy(entry)
        for promotion in revision["promotions"]
        for entry in promotion["quantitative_evidence"]
    ]
    runtime.update(
        {
            "matrix_id": EXPECTED_MATRIX_ID,
            "status": "FROZEN_BEFORE_EXECUTION",
            "qualification_scope": EXPECTED_SCOPE,
            "closure_request_policy": EXPECTED_CLOSURE_REQUEST_POLICY,
            "qualification_disposition": EXPECTED_DISPOSITION,
            "closure_state": EXPECTED_CLOSURE_STATE,
            "groups": groups,
            "executable_tests": sorted(executable),
            "prospective_tests": sorted(prospective),
            "closure_contract": copy.deepcopy(revision["closure_contract"]),
            "quantitative_evidence": [
                *runtime["quantitative_evidence"],
                *promoted_evidence,
            ],
            "gates": copy.deepcopy(EXPECTED_RUNTIME_GATES),
            "build_targets": {"physics": runtime["build_targets"]["physics"]},
            "build_cmake_homes": {
                "physics": runtime["build_cmake_homes"]["physics"]
            },
        }
    )
    runtime_tests = {test for group in groups for test in group["tests"]}
    if runtime_tests != executable or runtime_tests & prospective:
        raise ValueError("WP-7 V4 runtime test partition is inconsistent")
    if len(runtime_tests) != EXPECTED_RUNTIME_GATES[
        "expected_distinct_test_count"
    ]:
        raise ValueError("WP-7 V4 runtime test count is inconsistent")
    if len(runtime["quantitative_evidence"]) != EXPECTED_RUNTIME_GATES[
        "expected_quantitative_evidence_count"
    ]:
        raise ValueError("WP-7 V4 runtime quantitative count is inconsistent")
    for claim in runtime["closure_contract"]:
        if set(claim["evidence"]) - runtime_tests:
            raise ValueError("WP-7 V4 closure claim cites a nonruntime test")
    return runtime


def load_registry(path: Path) -> dict[str, Any]:
    if path.resolve() != DEFAULT_REGISTRY.resolve():
        raise ValueError("WP-7 V4 requires the canonical frozen revision")
    if sha256_file(path) != EXPECTED_REGISTRY_SHA256:
        raise ValueError("WP-7 V4 frozen revision bytes changed")
    registry = json.loads(path.read_text(encoding="utf-8"))
    validate_revision_contract(registry)
    validate_implementation_binding()
    return build_runtime_registry(registry)


def write_json(path: Path, value: Any) -> None:
    if isinstance(value, dict) and path.name in {
        "build_preflight.json",
        "manifest.json",
        "final_provenance.json",
        "summary.json",
    }:
        value = copy.deepcopy(value)
        value["wp7_v4_scope_state"] = EXPECTED_CLOSURE_STATE
        value["wp7_v4_full_closure_claimed"] = False
        value["wp7_v4_qualification_scope"] = EXPECTED_SCOPE
        value["wp7_v4_requested_claim"] = (
            EXPECTED_CLOSURE_REQUEST_POLICY["accepted_claim"]
        )
        value["wp7_v4_implementation_binding"] = validate_implementation_binding()
    _shared_write_json(path, value)


def write_text(path: Path, value: str) -> None:
    if path.name == "record.md":
        value = value.replace(
            "# WP-2 authoritative-geometry qualification record",
            "# WP-7 topology and node-crossing prerequisite qualification record",
            1,
        )
        value = value.replace(
            "\n\n",
            (
                "\n\n> Scope state: five prospective evidence rows and "
                "production-preconditioner spread remain release blocking; "
                "this V4 record cannot establish WP-7 closure.\n\n"
            ),
            1,
        )
    _shared_write_text(path, value)


strict_runner.SCRIPT_PATH = SCRIPT_PATH
strict_runner.DEFAULT_REGISTRY = DEFAULT_REGISTRY
strict_runner.EXPECTED_MATRIX_ID = EXPECTED_MATRIX_ID
strict_runner.EXPECTED_MATRIX_STATUS = "FROZEN_BEFORE_EXECUTION"
strict_runner.EXPECTED_WORK_PACKAGE = "WP-7"
strict_runner.load_registry = load_registry
strict_runner.write_json = write_json
strict_runner.write_text = write_text


def requested_claim(arguments: list[str]) -> tuple[str, bool, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--requested-claim",
        default=EXPECTED_CLOSURE_REQUEST_POLICY["accepted_claim"],
    )
    parser.add_argument("--validate-only", action="store_true")
    parsed, remaining = parser.parse_known_args(arguments)
    claim = parsed.requested_claim
    if claim in EXPECTED_CLOSURE_REQUEST_POLICY["rejected_claims"]:
        raise ValueError(
            f"requested claim {claim!r} is outside this matrix: "
            f"{EXPECTED_CLOSURE_REQUEST_POLICY['diagnostic']}"
        )
    if claim != EXPECTED_CLOSURE_REQUEST_POLICY["accepted_claim"]:
        raise ValueError(
            f"unsupported WP-7 V4 requested claim {claim!r}; expected "
            f"{EXPECTED_CLOSURE_REQUEST_POLICY['accepted_claim']!r}"
        )
    return claim, parsed.validate_only, remaining


def run_execution(arguments: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--physics-binary", type=Path, required=True)
    parser.add_argument("--mpiexec", type=Path, default=Path("/usr/bin/mpiexec"))
    parser.add_argument("--cmake", type=Path, default=Path("/usr/bin/cmake"))
    parser.add_argument("--build-parallel", type=int, default=2)
    parser.add_argument("--build-timeout-seconds", type=int, default=3600)
    parser.add_argument("--source-root", type=Path, default=REPOSITORY_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parsed = parser.parse_args(arguments)
    return strict_runner.run_qualification(
        parsed,
        {"physics": parsed.physics_binary},
        expected_binary_keys={"physics"},
        parser=parser,
        record_title=(
            "WP-7 topology and node-crossing prerequisite qualification record"
        ),
    )


def main(arguments: list[str] | None = None) -> int:
    active_arguments = sys.argv[1:] if arguments is None else arguments
    claim, validate_only, remaining = requested_claim(active_arguments)
    if validate_only:
        if remaining:
            raise ValueError("--validate-only does not accept execution arguments")
        registry = load_registry(DEFAULT_REGISTRY)
        print(
            json.dumps(
                {
                    "matrix_id": registry["matrix_id"],
                    "status": registry["status"],
                    "requested_claim": claim,
                    "closure_state": registry["closure_state"],
                    "group_count": len(registry["groups"]),
                    "test_count": sum(
                        len(group["tests"]) for group in registry["groups"]
                    ),
                    "executable_test_count": len(registry["executable_tests"]),
                    "prospective_test_count": len(registry["prospective_tests"]),
                    "serial_quantitative_gate_count": len(
                        registry["quantitative_evidence"]
                    ),
                    **registry["qualification_disposition"],
                    "outcome": "PASS",
                },
                sort_keys=True,
            )
        )
        return 0
    return run_execution(remaining)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)

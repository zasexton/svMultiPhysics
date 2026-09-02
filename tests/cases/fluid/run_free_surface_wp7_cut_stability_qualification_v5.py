#!/usr/bin/env python3
"""Run the WP-7 prerequisite matrix after trace-route contract renewal."""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

sys.dont_write_bytecode = True


SCRIPT_PATH = Path(__file__).resolve()
REPOSITORY_ROOT = SCRIPT_PATH.parents[3]
DEFAULT_REGISTRY = SCRIPT_PATH.with_name(
    "free_surface_wp7_cut_stability_qualification_revision_v5.json"
)
PARENT_REVISION_RUNNER_PATH = SCRIPT_PATH.with_name(
    "run_free_surface_wp7_cut_stability_qualification_v4.py"
)
EXPECTED_PARENT_REVISION_RUNNER_SHA256 = (
    "4f3ebb3db7e28870c8dbdbe31d560967188732513e9e7985c6d736dfb2f0df64"
)
EXPECTED_REGISTRY_SHA256 = (
    "4821477a9105a572e21065d09dfcea37fcf74670b5825f7105d3791988f4ef57"
)
EXPECTED_FOCUSED_TEST_SHA256 = (
    "8ccf1c6f57be5f7d716ec078b84a47e278efbb9504108f177821c7afe5bb7602"
)
EXPECTED_IMPLEMENTATION_SOURCE_SHA256 = (
    "f68615a764db4f71ba1917a0f68fe10be7e7936070b56ce1a10b039014f29190"
)
EXPECTED_IMPLEMENTATION_SOURCE_COMMIT = (
    "01d5bbb6ac9ce069f4727096084af0bb6d8d39c3"
)
EXPECTED_IMPLEMENTATION_SOURCE_PATH = (
    "Code/Source/solver/Physics/Tests/Unit/test_FreeSurfaceCutStability.cpp"
)
EXPECTED_FOCUSED_TEST_PATH = (
    "tests/test_free_surface_wp7_cut_stability_qualification_runner_v5.py"
)
EXPECTED_MATRIX_ID = "free_surface_wp7_cut_stability_v5"
EXPECTED_SCOPE = (
    "Executable prerequisite evidence for the production connected, "
    "disconnected, and rootless active-feature policy, the physically "
    "refined affine-probe node-crossing response, and agreement between "
    "the two canonical generated-boundary trace routes. Five prospective "
    "manufactured-error and simulation-exit rows remain absent, and "
    "production-preconditioner spread remains unresolved, so this revision "
    "does not close FSR-07, WP-7, or Q1."
)


def _sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_parent_revision() -> Any:
    if _sha256_file(PARENT_REVISION_RUNNER_PATH) != (
        EXPECTED_PARENT_REVISION_RUNNER_SHA256
    ):
        raise RuntimeError("WP-7 parent revision runner bytes changed")
    specification = importlib.util.spec_from_file_location(
        "_free_surface_wp7_cut_stability_v5_parent",
        PARENT_REVISION_RUNNER_PATH,
    )
    if specification is None or specification.loader is None:
        raise RuntimeError("cannot load the WP-7 parent revision runner")
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


_parent = _load_parent_revision()
_parent.DEFAULT_REGISTRY = DEFAULT_REGISTRY
_parent.EXPECTED_REGISTRY_SHA256 = EXPECTED_REGISTRY_SHA256
_parent.EXPECTED_FOCUSED_TEST_SHA256 = EXPECTED_FOCUSED_TEST_SHA256
_parent.EXPECTED_IMPLEMENTATION_SOURCE_SHA256 = (
    EXPECTED_IMPLEMENTATION_SOURCE_SHA256
)
_parent.EXPECTED_IMPLEMENTATION_SOURCE_COMMIT = (
    EXPECTED_IMPLEMENTATION_SOURCE_COMMIT
)
_parent.EXPECTED_IMPLEMENTATION_SOURCE_PATH = (
    EXPECTED_IMPLEMENTATION_SOURCE_PATH
)
_parent.EXPECTED_FOCUSED_TEST_PATH = EXPECTED_FOCUSED_TEST_PATH
_parent.EXPECTED_MATRIX_ID = EXPECTED_MATRIX_ID
_parent.EXPECTED_SCOPE = EXPECTED_SCOPE

sha256_file = _parent.sha256_file
validate_revision_contract = _parent.validate_revision_contract
validate_implementation_binding = _parent.validate_implementation_binding
build_runtime_registry = _parent.build_runtime_registry
_promotion_property_contract = _parent._promotion_property_contract
EXPECTED_PROMOTED_TESTS = _parent.EXPECTED_PROMOTED_TESTS
EXPECTED_GROUP_IDS = _parent.EXPECTED_GROUP_IDS
EXPECTED_RUNTIME_GATES = _parent.EXPECTED_RUNTIME_GATES
EXPECTED_CLOSURE_STATE = _parent.EXPECTED_CLOSURE_STATE
EXPECTED_CLOSURE_REQUEST_POLICY = _parent.EXPECTED_CLOSURE_REQUEST_POLICY
EXPECTED_DISPOSITION = _parent.EXPECTED_DISPOSITION
PARENT_RUNNER_PATH = _parent.PARENT_RUNNER_PATH
BASE_MATRIX_PATH = _parent.BASE_MATRIX_PATH
EXPECTED_PARENT_RUNNER_SHA256 = _parent.EXPECTED_PARENT_RUNNER_SHA256
EXPECTED_BASE_MATRIX_SHA256 = _parent.EXPECTED_BASE_MATRIX_SHA256
strict_runner = _parent.strict_runner


def load_registry(path: Path) -> dict[str, Any]:
    if path.resolve() != DEFAULT_REGISTRY.resolve():
        raise ValueError("WP-7 V5 requires the canonical frozen revision")
    if sha256_file(path) != EXPECTED_REGISTRY_SHA256:
        raise ValueError("WP-7 V5 frozen revision bytes changed")
    revision = json.loads(path.read_text(encoding="utf-8"))
    validate_revision_contract(revision)
    validate_implementation_binding()
    return build_runtime_registry(revision)


def write_json(path: Path, value: Any) -> None:
    if isinstance(value, dict) and path.name in {
        "build_preflight.json",
        "manifest.json",
        "final_provenance.json",
        "summary.json",
    }:
        value = copy.deepcopy(value)
        value["wp7_v5_scope_state"] = EXPECTED_CLOSURE_STATE
        value["wp7_v5_full_closure_claimed"] = False
        value["wp7_v5_qualification_scope"] = EXPECTED_SCOPE
        value["wp7_v5_requested_claim"] = (
            EXPECTED_CLOSURE_REQUEST_POLICY["accepted_claim"]
        )
        value["wp7_v5_implementation_binding"] = (
            validate_implementation_binding()
        )
    _parent._shared_write_json(path, value)


def write_text(path: Path, value: str) -> None:
    if path.name == "record.md":
        value = value.replace(
            "# WP-2 authoritative-geometry qualification record",
            "# WP-7 trace-route prerequisite qualification record",
            1,
        )
        value = value.replace(
            "\n\n",
            (
                "\n\n> Scope state: five prospective evidence rows and "
                "production-preconditioner spread remain release blocking; "
                "this V5 record cannot establish WP-7 closure.\n\n"
            ),
            1,
        )
    _parent._shared_write_text(path, value)


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
            f"unsupported WP-7 V5 requested claim {claim!r}; expected "
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
        record_title="WP-7 trace-route prerequisite qualification record",
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

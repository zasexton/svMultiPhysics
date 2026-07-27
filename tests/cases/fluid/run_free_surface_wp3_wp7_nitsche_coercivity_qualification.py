#!/usr/bin/env python3
"""Run the frozen joint WP-3/WP-7 symmetric-Nitsche prerequisite."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any


SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIRECTORY = SCRIPT_PATH.parent
REPOSITORY_ROOT = SCRIPT_PATH.parents[3]
DEFAULT_REGISTRY = SCRIPT_PATH.with_name(
    "free_surface_wp3_wp7_nitsche_coercivity_qualification_matrix.json"
)
SHARED_RUNNER_PATH = SCRIPT_PATH.with_name(
    "run_free_surface_wp2_geometry_qualification.py"
)
EXPECTED_REGISTRY_SHA256 = (
    "a75bbec8efe800f049375f190c07a121b3e365098da783b43ec1ba9df9610589"
)
EXPECTED_SHARED_RUNNER_SHA256 = (
    "5aea557b1bd3123116bbf034f39c0dd5f914f66286df65dd2250ec9bd906561f"
)
EXPECTED_MATRIX_ID = (
    "free_surface_wp3_wp7_symmetric_nitsche_prerequisite_v1"
)
EXPECTED_MATRIX_STATUS = "FROZEN_BEFORE_EXECUTION"
EXPECTED_WORK_PACKAGE = "WP-3/WP-7"
EXPECTED_SOURCE_COMMIT = "f127ce715f5d9042af3fa409d197667bc289e03f"
EXPECTED_MATCHING_DERIVATION = (
    "Documentation/free_surface_wp3_wp7_symmetric_nitsche_coercivity_method.md"
)
EXPECTED_SCOPE = (
    "Finite serial P1 constant-viscosity symmetric-Nitsche and "
    "cut-volume-side-cache prerequisite evidence only; this matrix does not "
    "close FSR-16, FSR-07, WP-3, WP-7, Q1, or establish a uniform coercivity "
    "bound."
)
EXPECTED_CLOSURE_REQUEST_POLICY = {
    "accepted_claim": "joint_low_level_prerequisite",
    "rejected_claims": [
        "fsr16_closure",
        "fsr07_closure",
        "wp3_closure",
        "wp7_closure",
        "wp3_wp7_joint_closure",
        "q1_closure",
    ],
    "diagnostic": (
        "The finite serial symmetric-Nitsche spectrum omits the full "
        "boundary-operator, regime, pressure-stability, topology, "
        "convergence, and MPI exits required for closure."
    ),
}
EXPECTED_DISPOSITION = {
    "fsr16_closed": False,
    "fsr07_closed": False,
    "wp3_closed": False,
    "wp7_closed": False,
    "q1_closed": False,
    "uniform_coercivity_bound_established": False,
}
EXPECTED_OPEN_OUTCOMES = {
    "fsr16": "OPEN",
    "fsr07": "OPEN",
    "wp3": "OPEN",
    "wp7": "OPEN",
    "q1": "OPEN",
}
EXPECTED_CASE_AXES = {
    "wall_fractions": [
        0.0,
        1e-08,
        1e-06,
        0.0001,
        0.01,
        0.1,
        0.25,
        0.49,
        1.0,
    ],
    "orientations": ["axis", "oblique"],
    "affine_mesh_scales": [0.5, 0.3333333333333333, 0.25],
    "active_sides": ["negative", "positive"],
    "case_count": 108,
}
EXPECTED_PARENT_SHA256 = {
    "Documentation/free_surface_wp3_wp7_symmetric_nitsche_coercivity_method.md": (
        "abc782ef828b3fd3996257f5544f85221d9c6d047b1cf730848ac93b695c6ead"
    ),
    "tests/cases/fluid/free_surface_wp3_sharp_boundary_qualification_matrix_v2.json": (
        "72cffdc330f07b386fdb89681bcd3da83b7f884c5bd1d09f49eef3f6ae79d883"
    ),
    "tests/cases/fluid/run_free_surface_wp3_sharp_boundary_qualification_v2.py": (
        "634a003c8cde429756787771bd4c870ee453a372b95d704fb98e2e10b99876e0"
    ),
    "tests/cases/fluid/free_surface_wp7_cut_stability_qualification_matrix.json": (
        "a49cadbcbe1b56bf69e4520a5281fc942ac4dc9de82da4e3bdaa083d6334ab1f"
    ),
    "tests/cases/fluid/run_free_surface_wp7_cut_stability_qualification.py": (
        "30c59eb725ba88b87447b935a915c6561b9ae25f082d93052d5cb9be4e337f6f"
    ),
    "Documentation/free_surface_wp7_combined_p1_method.md": (
        "85ba3d61f50b67a4d719efff6b760323fbe3fcd3124b06e28b67ee6230ca1ff9"
    ),
    "tests/cases/fluid/run_free_surface_wp2_geometry_qualification.py": (
        EXPECTED_SHARED_RUNNER_SHA256
    ),
}
EXPECTED_SOURCE_SHA256 = {
    "Code/Source/solver/FE/Forms/FormKernels.cpp": (
        "5d725f97b91ddddda6dbcc191b620e90e0f4af816be32ad89a43cf2cb4b7ff42"
    ),
    "Code/Source/solver/FE/Tests/Unit/Forms/test_CutCellForms.cpp": (
        "8e4c050ea0a2d53a2e2b8e17fa9aa07abf8e0d81875a7d6c39272dfb62aefc78"
    ),
    "Code/Source/solver/FE/Constraints/SmallCutAggregationConstraint.cpp": (
        "cac351bfa4f9c19f7270ed186513858506b82dd7503f1280ea52b52e6423f3f6"
    ),
    "Code/Source/solver/FE/Constraints/SmallCutAggregationConstraint.h": (
        "3b728e30324815bda8a7f36ece8ed211dd1b8c1bb2321cb85a22fb35bb69389c"
    ),
    "Code/Source/solver/FE/Systems/FESystem.cpp": (
        "12d3ae9854cd7d538fbda28873220d525260a1649c7a2b686dbec93a0b3d8eea"
    ),
    "Code/Source/solver/FE/Systems/FESystem.h": (
        "f9773065b84fe1dafaae7908bde453c4df43246eb64853ce01b5cf1dc5d0d9a2"
    ),
    "Code/Source/solver/Physics/Formulations/NavierStokes/IncompressibleNavierStokesVMSModule.cpp": (
        "81ae095b5e80075a08febcd4513a44025d83f01601b435fda816659bda136684"
    ),
    "Code/Source/solver/Physics/Formulations/NavierStokes/IncompressibleNavierStokesVMSModule.h": (
        "64a239b57dbd9d9de1213baa9673d80a38e347d17f2f69edb5a27c5579d2792e"
    ),
    "Code/Source/solver/Physics/Formulations/NavierStokes/NavierStokesBCFactories.h": (
        "934ecc235f0e1bc86ea1e8e0faea22dbe6153983131057b46c340b2b398e1919"
    ),
    "Code/Source/solver/Physics/Tests/Unit/test_FreeSurfaceCutStability.cpp": (
        "20b1658df29367910ccaf956ab3f7b3897a729c57ece61c3308f30bc66293a93"
    ),
    "Code/Source/solver/Physics/Tests/Unit/test_NavierStokesPressureGauge.cpp": (
        "f0b59029512c186bf56ada6f125aecc8ac2d7eb71bca0de78ba21ef995e822df"
    ),
}
EXPECTED_GROUP_TESTS = {
    "cut_volume_side_cache_serial": (
        "geometry",
        1,
        (
            "CutCellForms.SymbolicTangentPreservesCutVolumeMeasure",
            "CutCellForms.SymbolicTangentCacheSeparatesCutVolumeSides",
        ),
    ),
    "symmetric_nitsche_prerequisite_serial": (
        "physics",
        1,
        (
            "NavierStokesPressureGauge."
            "PublishesVariableDynamicViscosityFromResidualExpression",
            "NavierStokesPressureGauge."
            "RejectsEnergyDiagnosticWithoutExplicitPrerequisiteScopeBeforeSystemMutation",
            "NavierStokesPressureGauge."
            "RejectsVariableViscosityEnergyDiagnosticBeforeSystemMutation",
            "FreeSurfaceCutStability."
            "SymmetricNitscheFiniteSampleEnergySpectrumUsesSharpBoundaryAndAggregation",
        ),
    ),
}


def _load_shared_runner() -> Any:
    specification = importlib.util.spec_from_file_location(
        "_free_surface_wp3_wp7_prerequisite_base",
        SHARED_RUNNER_PATH,
    )
    if specification is None or specification.loader is None:
        raise RuntimeError("cannot load the shared qualification base")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


strict_runner = _load_shared_runner()
_shared_load_registry = strict_runner.load_registry
_shared_write_json = strict_runner.write_json
_shared_write_text = strict_runner.write_text

strict_runner.SCRIPT_PATH = SCRIPT_PATH
strict_runner.DEFAULT_REGISTRY = DEFAULT_REGISTRY
strict_runner.EXPECTED_MATRIX_ID = EXPECTED_MATRIX_ID
strict_runner.EXPECTED_MATRIX_STATUS = EXPECTED_MATRIX_STATUS
strict_runner.EXPECTED_WORK_PACKAGE = EXPECTED_WORK_PACKAGE
strict_runner.__doc__ = __doc__


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _reject_duplicate_keys(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def parse_json_document(path: Path) -> dict[str, Any]:
    parsed = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_reject_duplicate_keys,
    )
    if not isinstance(parsed, dict):
        raise ValueError("qualification matrix root must be an object")
    return parsed


def _artifact_map(entries: Any, label: str) -> dict[str, str]:
    if not isinstance(entries, list):
        raise ValueError(f"{label} must be a list")
    result: dict[str, str] = {}
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {"path", "sha256"}:
            raise ValueError(f"{label} entry has unexpected keys")
        path = entry["path"]
        digest = entry["sha256"]
        if (
            not isinstance(path, str)
            or not path
            or Path(path).is_absolute()
            or ".." in Path(path).parts
            or path in result
        ):
            raise ValueError(f"{label} entry has an unsafe or duplicate path")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(f"{label} entry has an invalid digest")
        result[path] = digest
    return result


def validate_joint_contract(registry: dict[str, Any]) -> dict[str, Any]:
    if registry.get("qualification_scope") != EXPECTED_SCOPE:
        raise ValueError("qualification scope changed after freeze")
    if registry.get("closure_request_policy") != (
        EXPECTED_CLOSURE_REQUEST_POLICY
    ):
        raise ValueError("closure-request policy changed after freeze")
    if registry.get("qualification_disposition") != EXPECTED_DISPOSITION:
        raise ValueError("qualification disposition changed after freeze")
    if registry.get("open_outcomes") != EXPECTED_OPEN_OUTCOMES:
        raise ValueError("open outcomes changed after freeze")
    if registry.get("case_axes") != EXPECTED_CASE_AXES:
        raise ValueError("finite-sample case axes changed after freeze")
    if registry.get("implementation_source_commit") != EXPECTED_SOURCE_COMMIT:
        raise ValueError("implementation source commit changed after freeze")
    if registry.get("matching_derivation") != EXPECTED_MATCHING_DERIVATION:
        raise ValueError("matching derivation changed after freeze")
    if registry.get("method_coercivity_lower_bound") is not None:
        raise ValueError("a method coercivity lower bound was invented")
    if registry.get("uniform_bound_status") != (
        "UNFROZEN_NO_BOUND_INVENTED"
    ):
        raise ValueError("uniform-bound status changed after freeze")
    if _artifact_map(
        registry.get("parent_artifacts"), "parent artifacts"
    ) != EXPECTED_PARENT_SHA256:
        raise ValueError("parent artifact inventory changed after freeze")
    if EXPECTED_MATCHING_DERIVATION not in EXPECTED_PARENT_SHA256:
        raise RuntimeError("matching derivation is not byte-locked")
    if _artifact_map(
        registry.get("implementation_sources"), "implementation sources"
    ) != EXPECTED_SOURCE_SHA256:
        raise ValueError("implementation source inventory changed after freeze")
    groups = registry.get("groups")
    if not isinstance(groups, list):
        raise ValueError("qualification groups are missing")
    if [group.get("id") for group in groups] != list(EXPECTED_GROUP_TESTS):
        raise ValueError("qualification group order changed after freeze")
    for group in groups:
        expected_binary, expected_ranks, expected_tests = (
            EXPECTED_GROUP_TESTS[group["id"]]
        )
        if (
            group.get("binary") != expected_binary
            or group.get("mpi_ranks") != expected_ranks
            or tuple(group.get("tests", [])) != expected_tests
        ):
            raise ValueError(
                f"qualification group changed after freeze: {group['id']}"
            )
    fixture = registry.get("native_fixture_contract")
    if (
        not isinstance(fixture, dict)
        or fixture.get("sharp_wall_parent_face_count") != 1
        or fixture.get("strong_velocity_anchor_present") is not True
        or fixture.get("constant_viscosity_required") is not True
        or fixture.get("diagnostic_scope") != "JointLowLevelPrerequisite"
        or fixture.get("diagnostic_operator_count") != 4
        or fixture.get("dry_boundary_contract")
        != "bit_exact_matrix_equality_with_nonzero_spd_anchored_bulk"
    ):
        raise ValueError("native fixture contract changed after freeze")
    guards = registry.get("aggregation_guard_contract")
    if (
        not isinstance(guards, dict)
        or guards.get("production_default_maximum_root_path_length") != 8
        or guards.get("qualified_maximum_root_path_length") != 12
        or guards.get("expected_maximum_observed_root_path") != 9
        or guards.get("default_limit_rejection_required") is not True
        or guards.get("guard_rejections_in_accepted_sweep") != 0
    ):
        raise ValueError("aggregation guard contract changed after freeze")
    if registry.get("prospective_tests") != []:
        raise ValueError("this child matrix cannot execute prospective tests")
    return registry


def validate_frozen_dependencies(
    registry: dict[str, Any],
    repository_root: Path = REPOSITORY_ROOT,
) -> None:
    inventories = (
        ("parent artifact", registry["parent_artifacts"]),
        ("implementation source", registry["implementation_sources"]),
    )
    for label, entries in inventories:
        for entry in entries:
            path = repository_root / entry["path"]
            if not path.is_file():
                raise ValueError(f"{label} is missing: {entry['path']}")
            if sha256_file(path) != entry["sha256"]:
                raise ValueError(f"{label} bytes changed: {entry['path']}")
    if sha256_file(SHARED_RUNNER_PATH) != EXPECTED_SHARED_RUNNER_SHA256:
        raise RuntimeError("shared qualification base changed during execution")


def load_registry(path: Path) -> dict[str, Any]:
    if sha256_file(path) != EXPECTED_REGISTRY_SHA256:
        raise ValueError("frozen prerequisite matrix bytes changed")
    if path.resolve() != DEFAULT_REGISTRY.resolve():
        raise ValueError("qualification requires the canonical frozen matrix")
    parsed = parse_json_document(path)
    validate_joint_contract(parsed)
    registry = validate_joint_contract(_shared_load_registry(path))
    validate_frozen_dependencies(registry)
    return registry


def write_json(path: Path, value: Any) -> None:
    if sha256_file(SHARED_RUNNER_PATH) != EXPECTED_SHARED_RUNNER_SHA256:
        raise RuntimeError("shared qualification base changed during execution")
    if isinstance(value, dict) and path.name in {
        "build_preflight.json",
        "manifest.json",
        "final_provenance.json",
        "summary.json",
    }:
        value = copy.deepcopy(value)
        value["qualification_scope"] = EXPECTED_SCOPE
        value["requested_claim"] = "joint_low_level_prerequisite"
        value["qualification_disposition"] = copy.deepcopy(
            EXPECTED_DISPOSITION
        )
        value["open_outcomes"] = copy.deepcopy(EXPECTED_OPEN_OUTCOMES)
        value["method_coercivity_lower_bound"] = None
        value["uniform_bound_status"] = "UNFROZEN_NO_BOUND_INVENTED"
        value["implementation_source_commit"] = EXPECTED_SOURCE_COMMIT
        value["parent_artifacts"] = copy.deepcopy(
            value.get("parent_artifacts", [])
        )
        value["prerequisite_parent_artifacts"] = [
            {"path": path_name, "sha256": digest}
            for path_name, digest in EXPECTED_PARENT_SHA256.items()
        ]
        value["prerequisite_implementation_sources"] = [
            {"path": path_name, "sha256": digest}
            for path_name, digest in EXPECTED_SOURCE_SHA256.items()
        ]
    _shared_write_json(path, value)


def write_text(path: Path, value: str) -> None:
    if path.name == "record.md":
        value += (
            "\n## Scope boundary\n\n"
            + EXPECTED_SCOPE
            + "\n\n"
            "FSR-16, FSR-07, WP-3, WP-7, and Q1 remain open. "
            "No uniform coercivity bound is frozen.\n"
        )
    _shared_write_text(path, value)


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
    accepted = EXPECTED_CLOSURE_REQUEST_POLICY["accepted_claim"]
    rejected = set(EXPECTED_CLOSURE_REQUEST_POLICY["rejected_claims"])
    if claim in rejected:
        raise ValueError(
            f"requested claim {claim!r} is outside this matrix: "
            f"{EXPECTED_CLOSURE_REQUEST_POLICY['diagnostic']}"
        )
    if claim != accepted:
        raise ValueError(
            f"unsupported requested claim {claim!r}; expected {accepted!r}"
        )
    return claim, parsed.validate_only, remaining


def validate_only_summary(
    registry: dict[str, Any],
    claim: str,
) -> dict[str, Any]:
    return {
        "matrix_id": registry["matrix_id"],
        "status": registry["status"],
        "requested_claim": claim,
        "group_count": len(registry["groups"]),
        "test_count": sum(
            len(group["tests"]) for group in registry["groups"]
        ),
        "quantitative_evidence_gate_count": len(
            registry["quantitative_evidence"]
        ),
        "method_coercivity_lower_bound": None,
        "uniform_bound_status": "UNFROZEN_NO_BOUND_INVENTED",
        "qualification_disposition": copy.deepcopy(EXPECTED_DISPOSITION),
        "closure_outcome": "OPEN_JOINT_LOW_LEVEL_PREREQUISITE",
        "outcome": "PASS",
    }


def main(arguments: list[str] | None = None) -> int:
    selected_arguments = (
        list(sys.argv[1:]) if arguments is None else list(arguments)
    )
    claim, validate_only, remaining = requested_claim(selected_arguments)
    if validate_only:
        if remaining:
            raise ValueError(
                "--validate-only does not accept execution arguments"
            )
        print(
            json.dumps(
                validate_only_summary(load_registry(DEFAULT_REGISTRY), claim),
                sort_keys=True,
            )
        )
        return 0
    original_argv = sys.argv
    try:
        sys.argv = [str(SCRIPT_PATH), *remaining]
        return strict_runner.main()
    finally:
        sys.argv = original_argv


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)

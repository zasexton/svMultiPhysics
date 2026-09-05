#!/usr/bin/env python3
"""Validate, expand, run, and analyze the frozen WP-4 V3 matrix."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence


SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIRECTORY = SCRIPT_PATH.parent
REPOSITORY_ROOT = SCRIPT_PATH.parents[3]
EXPECTED_LFS_TRACKED_OBJECT_COUNT = 955
DEFAULT_REGISTRY = SCRIPT_PATH.with_name(
    "free_surface_wp4_balanced_capillary_matrix_v3.json"
)
PARENT_RUNNER_PATH = SCRIPT_PATH.with_name(
    "run_free_surface_wp4_balanced_capillary_matrix_v2.py"
)
PARENT_REGISTRY_PATH = SCRIPT_PATH.with_name(
    "free_surface_wp4_balanced_capillary_matrix_v2.json"
)
PHYSICAL_RUNNER = (
    SCRIPT_DIRECTORY
    / "open_vessel_free_surface/run_test05_velocity_growth_smoke.py"
)
EXPECTED_REGISTRY_SHA256 = (
    "9b07b2b5dbf98e3b3c115f6ad11499454e7f13718eaf85ae28142bc338eb2fbe"
)
EXPECTED_PARENT_RUNNER_SHA256 = (
    "480c0441a4da62dd7d5f16133c9dde7b16df90772c06f851bed6ff233f69d4c3"
)
EXPECTED_PARENT_REGISTRY_SHA256 = (
    "7605f4458191112bf0f03c38299b9b46838a11e9dcbf61c7196fecb0f89d7918"
)
EXPECTED_PHYSICAL_RUNNER_SHA256 = (
    "201d2b7f5451cc4d53b460578effb338bb7e31ae447fe0c60e5eb6cfedd3d1cc"
)
EXPECTED_MATRIX_ID = "free_surface_wp4_balanced_capillary_v3"
EXPECTED_STATUS = "FROZEN_BEFORE_EXECUTION"
EXACT_INVOCATION_WATCHDOG_SECONDS = 900
EXACT_TERMINATION_GRACE_SECONDS = 2.0
EXACT_DIAGNOSTIC_CAPTURE_SECONDS = 2.0
EXACT_DIAGNOSTIC_OUTPUT_BYTES = 65536
EXACT_DIAGNOSTIC_PROCESS_LIMIT = 64
EXACT_DIAGNOSTIC_TERMINATION_GRACE_SECONDS = 0.1

_PROCESS_SUPERVISOR = r"""
import ctypes
import os
import resource
import signal
import subprocess
import sys

PR_SET_CHILD_SUBREAPER = 36
libc = ctypes.CDLL(None, use_errno=True)
if libc.prctl(PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0) != 0:
    raise OSError(ctypes.get_errno(), "unable to establish process containment")
output_limit = int(sys.argv[1])
if output_limit > 0:
    resource.setrlimit(resource.RLIMIT_FSIZE, (output_limit, output_limit))
signal.signal(signal.SIGTERM, lambda *_: None)
primary = subprocess.Popen(sys.argv[2:])
primary_returncode = None
while True:
    try:
        child_pid, status = os.wait()
    except ChildProcessError:
        break
    if child_pid == primary.pid:
        primary_returncode = os.waitstatus_to_exitcode(status)
        primary.returncode = primary_returncode
if primary_returncode is None:
    primary_returncode = 125
if primary_returncode < 0:
    primary_returncode = 128 - primary_returncode
raise SystemExit(primary_returncode)
"""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_parent() -> Any:
    if _sha256_file(PARENT_RUNNER_PATH) != EXPECTED_PARENT_RUNNER_SHA256:
        raise RuntimeError("frozen V2 runner bytes changed")
    if _sha256_file(PARENT_REGISTRY_PATH) != EXPECTED_PARENT_REGISTRY_SHA256:
        raise RuntimeError("frozen V2 registry bytes changed")
    spec = importlib.util.spec_from_file_location(
        "free_surface_wp4_balanced_capillary_matrix_v2_parent",
        PARENT_RUNNER_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load frozen V2 runner")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_v2 = _load_parent()
MatrixError = _v2.MatrixError
sha256_file = _v2.sha256_file
read_json = _v2.read_json
write_json = _v2.write_json
finite_number = _v2.finite_number
nonempty_string = _v2.nonempty_string
select_cases = _v2.select_cases
extract_metric = _v2.extract_metric
evaluate_exact_property_gate = _v2.evaluate_exact_property_gate
exact_rank_properties_identical = _v2.exact_rank_properties_identical
_canonical_sha256 = _v2._canonical_digest

_V2_EXPAND_CASES = _v2.expand_cases
_V2_PHYSICAL_CASE_ARGUMENTS = _v2.physical_case_arguments
_V2_RUN_PHYSICAL_CASES = _v2.run_physical_cases
_V2_EVALUATE_EXACT_DOCUMENT = _v2.evaluate_exact_document
_V2_RUN_EXACT_GROUPS = _v2.run_exact_groups
_V2_ANALYZE_EVIDENCE = _v2.analyze_evidence

SUPPORTED_CASE_DIMENSIONS = dict(_v2.SUPPORTED_CASE_DIMENSIONS)
SUPPORTED_INITIALIZATIONS = set(_v2.SUPPORTED_INITIALIZATIONS)
REQUIRED_WALLS = {
    dimension: set(walls)
    for dimension, walls in _v2.REQUIRED_WALLS.items()
}
REQUIRED_ACTIVE_DOMAINS = set(_v2.REQUIRED_ACTIVE_DOMAINS)
REQUIRED_ANGLES = set(_v2.REQUIRED_ANGLES)
SUPPORTED_EXACT_PROPERTY_COMPARISONS = set(
    _v2.SUPPORTED_EXACT_PROPERTY_COMPARISONS
)

TOP_LEVEL_FIELDS = {
    "schema_version",
    "matrix_id",
    "status",
    "work_package",
    "findings",
    "qualification_scope",
    "model_envelope",
    "closure_policy",
    "maintenance_contract",
    "required_report_metrics",
    "resources",
    "refinement",
    "gates",
    "exact_groups",
    "common_runner_arguments",
    "literature_adaptations",
    "studies",
    "execution_contract",
    "artifact_contract",
    "provenance_contract",
}
GATE_FIELDS = {
    "exact_flat_scaled_residual_factor",
    "static_initializer",
    "finest_level",
    "convergence",
    "energy_variation",
    "invariance",
}
RESOURCE_FIELDS = {
    "partition",
    "maximum_concurrent_nodes",
    "maximum_total_memory_mib",
    "nodes_per_case",
    "memory_mib_per_node",
    "memory_model",
    "profiles",
    "exact_invocation_lifecycle",
}
EXACT_INVOCATION_LIFECYCLE_FIELDS = {"watchdog_seconds"}
MEMORY_MODEL_FIELDS = {
    "formula_version",
    "formula",
    "generated_vertex_formula",
    "simplex_count_by_dimension",
    "coupled_unknown_components_by_dimension",
    "stored_field_components_by_dimension",
    "adjacency_upper_bound_by_dimension",
    "fixed_mib",
    "vertex_bytes",
    "simplex_bytes",
    "sparse_entry_bytes",
    "sparse_operator_copies",
    "field_vector_copies",
    "scalar_bytes",
}
PROFILE_FIELDS = {
    "nodes",
    "tasks",
    "threads_per_task",
    "memory_mib",
    "wall_time_seconds",
    "output_mib",
}
REFINEMENT_FIELDS = {
    "spatial_levels_cells_per_radius",
    "conditional_spatial_level_cells_per_radius",
    "conditional_level_trigger",
    "conditional_trigger_record_policy",
    "conditional_trigger_record_schema_version",
    "conditional_level_by_dimension",
    "uniform_resolution_rule",
    "reported_mesh_coordinate",
    "uniform_ratio",
    "ratio_relative_tolerance",
    "safety_factor",
    "minimum_observed_order",
    "nonmonotone_three_level_disposition",
    "nonasymptotic_four_level_disposition",
}
COMMON_STUDY_FIELDS = {
    "id",
    "case",
    "dimension",
    "initialization",
    "refinement_axis",
    "steps",
    "axes",
    "metrics",
    "arguments",
    "resource_profile",
    "scope",
}
STUDY_FIELDS_BY_AXIS = {
    "resolution": COMMON_STUDY_FIELDS | {"time_step", "radius"},
    "phi_scale": COMMON_STUDY_FIELDS
    | {"time_step", "radius", "resolution", "refinement_levels"},
    "physical_scale": COMMON_STUDY_FIELDS
    | {"time_step", "cells_per_radius", "refinement_levels"},
    "time_step": COMMON_STUDY_FIELDS
    | {
        "radius",
        "resolution",
        "refinement_levels",
        "level_step_counts",
        "physical_horizon",
    },
    "bulk_redistance_cadence": COMMON_STUDY_FIELDS
    | {"time_step", "radius", "resolution", "refinement_levels"},
}
EXACT_CATEGORIES = {
    "focused_algebra",
    "sampled_convergence",
    "minimized_equilibrium",
    "restoring_motion",
    "mpi_parity",
}
REQUIRED_EXACT_TESTS = {
    "LevelSetInterfaceDomain.PlanarPolygonQuadraticRuleIntegratesTetrahedralCuts",
    "LevelSetInterfaceLifecycle.LinearBackendDriverReportsSupportAndOrders",
    "LevelSetInterfaceLifecycle.BackendCapabilityReportsMilestoneContract",
    "ApplicationDriverLevelSetWorkflows.KinematicAreaGradientMaintenanceBindsTotalEnergyDeclaration",
    "ApplicationDriverLevelSetWorkflows.TotalEnergyTractionRuleValidatorFailsClosedBeforeProjection",
    "ApplicationDriverLevelSetWorkflowsMPI.TotalEnergyTractionRuleValidatorIsCollective",
    "MovingDomainPhysics.KinematicAreaGradientTractionIsEnergyAdjointOnQuadraticTetraCut",
    "ApplicationDriverLevelSetWorkflows.MinimizedCircleSphereAndSessileCapsMeetProductionCertificates",
    "ApplicationDriverLevelSetWorkflows.SampledCircleSphereAndSessileControlsConvergeWithGci",
    "ApplicationDriverLevelSetWorkflows.SampledSessileFiveAngleTransformMatrixReportsPhysicalObservables",
    "ApplicationDriverLevelSetWorkflows.MinimizedCapillaryStateHasVolumeOrthogonalRestoringResponse",
    "ApplicationDriverLevelSetWorkflowsMPI.MinimizedCurvedCapillaryParityAcrossTwoOwnershipLayouts",
}
REQUIRED_MAIN_STUDIES = set(_v2.REQUIRED_MAIN_STUDIES)
REQUIRED_MAIN_METRICS = set(_v2.REQUIRED_METRICS)
REQUIRED_SESSILE_METRICS = set(_v2.REQUIRED_SESSILE_METRICS)


def _require_fields(value: Any, expected: set[str], context: str) -> None:
    if not isinstance(value, dict) or set(value) != expected:
        observed = sorted(value) if isinstance(value, dict) else type(value).__name__
        raise MatrixError(
            f"{context} changed: expected={sorted(expected)}, observed={observed}"
        )


def _unique_strings(value: Any, context: str) -> list[str]:
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(item, str) or not item for item in value)
    ):
        raise MatrixError(f"{context} must contain nonempty strings")
    if len(value) != len(set(value)):
        raise MatrixError(f"{context} contains duplicates")
    return list(value)


def _validate_number_list(
    value: Any, context: str, *, count: int | None = None, positive: bool = False
) -> list[float]:
    if not isinstance(value, list) or (count is not None and len(value) != count):
        raise MatrixError(f"{context} has invalid length")
    return [
        finite_number(item, f"{context} value", positive=positive) for item in value
    ]


def _option_values(arguments: Sequence[str], option: str) -> list[str]:
    return [
        arguments[index + 1]
        for index, value in enumerate(arguments[:-1])
        if value == option
    ]


def _exact_invocation_watchdog(resources: Any) -> float:
    if (
        not isinstance(resources, dict)
        or "exact_invocation_lifecycle" not in resources
    ):
        raise MatrixError("exact-invocation lifecycle is missing")
    lifecycle = resources["exact_invocation_lifecycle"]
    _require_fields(
        lifecycle,
        EXACT_INVOCATION_LIFECYCLE_FIELDS,
        "exact-invocation lifecycle fields",
    )
    watchdog = finite_number(
        lifecycle["watchdog_seconds"],
        "exact-invocation watchdog",
        positive=True,
    )
    if watchdog != EXACT_INVOCATION_WATCHDOG_SECONDS:
        raise MatrixError("exact-invocation watchdog must be 900 seconds")
    return watchdog


def _validate_resources(resources: Any) -> None:
    _require_fields(resources, RESOURCE_FIELDS, "resource fields")
    if resources["partition"] != "amarsden":
        raise MatrixError("resource partition must be amarsden")
    if resources["maximum_concurrent_nodes"] != 4:
        raise MatrixError("resource maximum concurrent nodes must be four")
    if resources["maximum_total_memory_mib"] != 40960:
        raise MatrixError("resource total memory must be 40960 MiB")
    if resources["nodes_per_case"] != 1:
        raise MatrixError("every physical case must use one node")
    if resources["memory_mib_per_node"] != 10240:
        raise MatrixError("one-node memory limit must be 10240 MiB")
    _exact_invocation_watchdog(resources)
    model = resources["memory_model"]
    _require_fields(model, MEMORY_MODEL_FIELDS, "memory-model fields")
    if model["formula_version"] != 1:
        raise MatrixError("memory formula version changed")
    expected_formula = (
        "ceil((fixed_bytes + generated_vertices*vertex_bytes + "
        "simplices*simplex_bytes + "
        "generated_vertices*coupled_components*adjacency_upper_bound*"
        "coupled_components*sparse_entry_bytes*sparse_operator_copies + "
        "generated_vertices*stored_field_components*scalar_bytes*"
        "field_vector_copies)/1048576)"
    )
    if model["formula"] != expected_formula:
        raise MatrixError("memory formula changed")
    if model["generated_vertex_formula"] != "(resolution + 1)^dimension":
        raise MatrixError("generated-vertex memory formula changed")
    expected_dimensions = {"2", "3"}
    for key in (
        "simplex_count_by_dimension",
        "coupled_unknown_components_by_dimension",
        "stored_field_components_by_dimension",
        "adjacency_upper_bound_by_dimension",
    ):
        table = model[key]
        if not isinstance(table, dict) or set(table) != expected_dimensions:
            raise MatrixError(f"memory-model table {key!r} changed")
        for dimension, amount in table.items():
            if not isinstance(amount, int) or isinstance(amount, bool) or amount < 1:
                raise MatrixError(f"memory-model table {key!r}[{dimension}] is invalid")
    expected_tables = {
        "simplex_count_by_dimension": {"2": 2, "3": 6},
        "coupled_unknown_components_by_dimension": {"2": 4, "3": 5},
        "stored_field_components_by_dimension": {"2": 5, "3": 6},
        "adjacency_upper_bound_by_dimension": {"2": 9, "3": 15},
    }
    if any(model[key] != value for key, value in expected_tables.items()):
        raise MatrixError("conservative memory-model tables changed")
    for key in (
        "fixed_mib",
        "vertex_bytes",
        "simplex_bytes",
        "sparse_entry_bytes",
        "sparse_operator_copies",
        "field_vector_copies",
        "scalar_bytes",
    ):
        amount = model[key]
        if not isinstance(amount, int) or isinstance(amount, bool) or amount < 1:
            raise MatrixError(f"memory-model value {key!r} is invalid")
    expected_storage = {
        "fixed_mib": 1024,
        "vertex_bytes": 128,
        "simplex_bytes": 256,
        "sparse_entry_bytes": 16,
        "sparse_operator_copies": 2,
        "field_vector_copies": 24,
        "scalar_bytes": 8,
    }
    if any(model[key] != value for key, value in expected_storage.items()):
        raise MatrixError("conservative memory-model storage factors changed")
    profiles = resources["profiles"]
    if not isinstance(profiles, dict) or not profiles:
        raise MatrixError("resource profiles are missing")
    for profile_id, profile in profiles.items():
        nonempty_string(profile_id, "resource profile id")
        _require_fields(profile, PROFILE_FIELDS, "resource-profile fields")
        for key, amount in profile.items():
            if not isinstance(amount, int) or isinstance(amount, bool) or amount < 1:
                raise MatrixError(f"resource profile {profile_id!r} {key!r} is invalid")
        if profile["nodes"] != 1:
            raise MatrixError("each resource profile must use one node")
        if profile["memory_mib"] > resources["memory_mib_per_node"]:
            raise MatrixError("resource profile exceeds one-node memory")


def _validate_refinement(refinement: Any) -> None:
    _require_fields(refinement, REFINEMENT_FIELDS, "refinement fields")
    if refinement["spatial_levels_cells_per_radius"] != [8, 16, 32]:
        raise MatrixError("required spatial levels must be R/h = 8, 16, 32")
    if refinement["conditional_spatial_level_cells_per_radius"] != 64:
        raise MatrixError("conditional spatial level must be R/h = 64")
    if refinement["conditional_level_trigger"] != (
        "nonmonotone_three_level_sequence_only"
    ):
        raise MatrixError("conditional spatial trigger changed")
    if (
        refinement["conditional_trigger_record_policy"]
        != "hash_bound_prior_three_level_analysis"
        or refinement["conditional_trigger_record_schema_version"] != 1
    ):
        raise MatrixError("conditional trigger record contract changed")
    conditional = refinement["conditional_level_by_dimension"]
    _require_fields(conditional, {"2", "3"}, "conditional dimension fields")
    _require_fields(
        conditional["2"],
        {"cells_per_radius", "availability", "disposition_when_required"},
        "conditional 2D fields",
    )
    _require_fields(
        conditional["3"],
        {"cells_per_radius", "availability", "disposition_when_required"},
        "conditional 3D fields",
    )
    if conditional["2"] != {
        "cells_per_radius": 64,
        "availability": "AVAILABLE",
        "disposition_when_required": "EXECUTE",
    }:
        raise MatrixError("2D conditional-level contract changed")
    if conditional["3"] != {
        "cells_per_radius": 64,
        "availability": "UNAVAILABLE_ONE_NODE_MEMORY_LIMIT",
        "disposition_when_required": "INCONCLUSIVE",
    }:
        raise MatrixError("3D conditional-level contract changed")
    if finite_number(refinement["uniform_ratio"], "uniform ratio", positive=True) != 2:
        raise MatrixError("uniform refinement ratio must be two")
    for key in (
        "ratio_relative_tolerance",
        "safety_factor",
        "minimum_observed_order",
    ):
        finite_number(refinement[key], key, positive=True)
    if refinement["nonmonotone_three_level_disposition"] != (
        "ADDITIONAL_LEVEL_REQUIRED"
    ):
        raise MatrixError("nonmonotone triplet disposition changed")
    if refinement["nonasymptotic_four_level_disposition"] != "FAIL":
        raise MatrixError("nonasymptotic quartet disposition changed")


def _validate_gates(gates: Any) -> None:
    _require_fields(gates, GATE_FIELDS, "gate fields")
    if gates["exact_flat_scaled_residual_factor"] != 256.0:
        raise MatrixError("exact scaled-roundoff factor changed")
    initializer_fields = {
        "volume_tolerance",
        "projected_gradient_tolerance",
        "pressure_representability_max_residual_norm",
        "pressure_representability_max_relative_distance",
        "physical_equilibrium_max_residual_norm",
        "constant_pressure_kkt_max_residual_norm",
        "constant_pressure_kkt_max_relative_distance",
        "maximum_iterations",
        "maximum_topology_epoch_transitions",
    }
    _require_fields(gates["static_initializer"], initializer_fields, "initializer gate fields")
    finest_fields = {
        "pressure_jump_relative_error",
        "contact_angle_absolute_error_degrees",
        "base_radius_relative_error",
        "apex_height_relative_error",
        "liquid_volume_relative_error",
        "parasitic_capillary_number",
        "kinetic_energy_proxy",
    }
    _require_fields(gates["finest_level"], finest_fields, "finest gate fields")
    finest = gates["finest_level"]
    if (
        finite_number(finest["pressure_jump_relative_error"], "pressure gate", positive=True) > 0.01
        or finite_number(finest["contact_angle_absolute_error_degrees"], "angle gate", positive=True) > 1.0
        or finite_number(finest["base_radius_relative_error"], "base gate", positive=True) > 0.01
        or finite_number(finest["apex_height_relative_error"], "apex gate", positive=True) > 0.01
        or finite_number(finest["parasitic_capillary_number"], "capillary-number gate", positive=True) > 1.0e-6
    ):
        raise MatrixError("predeclared finest-level gates were weakened")
    convergence = gates["convergence"]
    if not isinstance(convergence, dict) or not REQUIRED_MAIN_METRICS.issubset(convergence):
        raise MatrixError("convergence gates are incomplete")
    convergence_fields = {
        "reference",
        "normalization",
        "finest_error_limit",
        "finest_gci_limit",
    }
    for metric, limits in convergence.items():
        _require_fields(limits, convergence_fields, f"convergence gate {metric!r} fields")
        finite_number(limits["reference"], f"{metric} reference")
        for key in convergence_fields - {"reference"}:
            finite_number(limits[key], f"{metric} {key}", positive=True)
    energy = gates["energy_variation"]
    _require_fields(
        energy,
        {
            "analytic_derivatives_required",
            "finite_difference_components",
            "maximum_relative_directional_derivative_error",
        },
        "energy-variation gate fields",
    )
    components = energy["finite_difference_components"]
    if not isinstance(components, int) or isinstance(components, bool) or components < 1:
        raise MatrixError("finite-difference component count must be positive")
    if energy["analytic_derivatives_required"] is not True:
        raise MatrixError("analytic energy derivatives are required")
    finite_number(
        energy["maximum_relative_directional_derivative_error"],
        "energy derivative gate",
        positive=True,
    )
    invariance = gates["invariance"]
    _require_fields(invariance, {"phi_scale", "physical_scale"}, "invariance gate fields")
    for axis, metric_gates in invariance.items():
        if not isinstance(metric_gates, dict) or not metric_gates:
            raise MatrixError(f"{axis} invariance gates are missing")
        for metric, limits in metric_gates.items():
            _require_fields(limits, {"maximum_value", "maximum_spread"}, f"{axis} {metric} fields")
            finite_number(limits["maximum_value"], f"{axis} {metric} maximum", positive=True)
            finite_number(limits["maximum_spread"], f"{axis} {metric} spread", positive=True)


def _validate_property_gates(
    test: str, gates: Any, *, context: str
) -> list[dict[str, Any]]:
    if not isinstance(gates, list) or not gates:
        raise MatrixError(f"{context} exact test {test!r} has no property gates")
    seen: set[tuple[str, str]] = set()
    for gate_index, gate in enumerate(gates):
        gate_context = f"{context} test {test!r} gate {gate_index}"
        if not isinstance(gate, dict):
            raise MatrixError(f"{gate_context} must be an object")
        comparison = gate.get("comparison")
        if comparison not in SUPPORTED_EXACT_PROPERTY_COMPARISONS:
            raise MatrixError(f"{gate_context} comparison is invalid")
        expected_fields = {"property", "comparison"}
        if comparison in {"equal", "at_least", "at_most"}:
            expected_fields.add("expected")
        elif comparison == "scaled_roundoff":
            expected_fields.add("scale")
        _require_fields(gate, expected_fields, f"{gate_context} fields")
        property_name = nonempty_string(gate["property"], f"{gate_context} property")
        key = (property_name, comparison)
        if key in seen:
            raise MatrixError(f"{gate_context} is duplicated")
        seen.add(key)
        if "expected" in gate:
            finite_number(gate["expected"], f"{gate_context} expected")
        if "scale" in gate:
            finite_number(gate["scale"], f"{gate_context} scale", positive=True)
    return gates


def _validate_exact_groups(groups: Any, energy: dict[str, Any]) -> None:
    if not isinstance(groups, list) or not groups:
        raise MatrixError("exact groups are missing")
    category_ids: set[str] = set()
    invocation_ids: set[str] = set()
    all_tests: set[str] = set()
    energy_count_tests = {
        "FreeSurfaceGeometrySnapshot.DiscreteFunctionalFirstVariationMatchesCentralDifference",
        "FreeSurfaceGeometrySnapshot.DiscreteFunctionalFirstVariationMatchesThreeDimensionalCentralDifference",
    }
    observed_count_tests: set[str] = set()
    observed_error_tests: set[str] = set()
    for group_index, group in enumerate(groups):
        context = f"exact group {group_index}"
        _require_fields(group, {"id", "purpose", "invocations"}, f"{context} fields")
        category_id = nonempty_string(group["id"], f"{context} id")
        if category_id in category_ids:
            raise MatrixError(f"duplicate exact category {category_id!r}")
        category_ids.add(category_id)
        nonempty_string(group["purpose"], f"{context} purpose")
        invocations = group["invocations"]
        if not isinstance(invocations, list) or not invocations:
            raise MatrixError(f"exact category {category_id!r} has no invocations")
        for invocation_index, invocation in enumerate(invocations):
            invocation_context = (
                f"exact category {category_id!r} invocation {invocation_index}"
            )
            _require_fields(
                invocation,
                {
                    "id",
                    "binary",
                    "mpi_ranks",
                    "require_identical_rank_properties",
                    "resource_profile",
                    "tests",
                    "property_gates",
                },
                f"{invocation_context} fields",
            )
            invocation_id = nonempty_string(
                invocation["id"], f"{invocation_context} id"
            )
            qualified_id = f"{category_id}--{invocation_id}"
            if qualified_id in invocation_ids:
                raise MatrixError(f"duplicate exact invocation {qualified_id!r}")
            invocation_ids.add(qualified_id)
            nonempty_string(invocation["binary"], f"{invocation_context} binary")
            nonempty_string(
                invocation["resource_profile"], f"{invocation_context} resource profile"
            )
            ranks = invocation["mpi_ranks"]
            if not isinstance(ranks, int) or isinstance(ranks, bool) or ranks < 1:
                raise MatrixError(f"{invocation_context} rank count is invalid")
            expected_ranks = 2 if category_id == "mpi_parity" else 1
            if ranks != expected_ranks:
                raise MatrixError(f"{invocation_context} rank count changed")
            if invocation["require_identical_rank_properties"] is not (ranks > 1):
                raise MatrixError(f"{invocation_context} rank property policy changed")
            tests = _unique_strings(invocation["tests"], f"{invocation_context} tests")
            duplicates = all_tests.intersection(tests)
            if duplicates:
                raise MatrixError(f"exact tests appear in multiple invocations: {sorted(duplicates)}")
            all_tests.update(tests)
            property_gates = invocation["property_gates"]
            if not isinstance(property_gates, dict) or set(property_gates) != set(tests):
                raise MatrixError(f"{invocation_context} must gate every exact test")
            for test in tests:
                test_gates = _validate_property_gates(
                    test, property_gates[test], context=invocation_context
                )
                properties = {gate["property"] for gate in test_gates}
                if test in energy_count_tests:
                    count_gates = [
                        gate
                        for gate in test_gates
                        if gate["property"].endswith("fd_case_count")
                        and gate["comparison"] == "equal"
                        and gate.get("expected")
                        == energy["finite_difference_components"]
                    ]
                    if count_gates:
                        observed_count_tests.add(test)
                    error_gates = [
                        gate
                        for gate in test_gates
                        if gate["property"].endswith("max_relative_error")
                        and gate["comparison"] == "at_most"
                        and float(gate.get("expected", math.inf))
                        <= float(
                            energy[
                                "maximum_relative_directional_derivative_error"
                            ]
                        )
                    ]
                    if error_gates:
                        observed_error_tests.add(test)
                if not properties:
                    raise MatrixError(f"exact test {test!r} has no gated properties")
    if category_ids != EXACT_CATEGORIES:
        raise MatrixError("exact evidence categories changed")
    if observed_count_tests != energy_count_tests:
        raise MatrixError("energy finite-difference count gates are incomplete")
    if observed_error_tests != energy_count_tests:
        raise MatrixError("energy finite-difference error gates are incomplete")
    if not REQUIRED_EXACT_TESTS.issubset(all_tests):
        raise MatrixError("required Task 1-3 exact tests are missing")


def _validate_parent_exact_preservation(registry: dict[str, Any]) -> None:
    parent = read_json(PARENT_REGISTRY_PATH)
    parent_gates = {
        test: group["property_gates"][test]
        for group in parent["exact_groups"]
        for test in group["tests"]
    }
    current_gates = {
        test: invocation["property_gates"][test]
        for invocation in exact_invocations(registry)
        for test in invocation["tests"]
    }
    missing = set(parent_gates) - set(current_gates)
    if missing:
        raise MatrixError(f"V2 exact tests are missing: {sorted(missing)}")
    weakened = {
        test: [gate for gate in gates if gate not in current_gates[test]]
        for test, gates in parent_gates.items()
        if any(gate not in current_gates[test] for gate in gates)
    }
    if weakened:
        raise MatrixError(
            f"V2 exact property gates were weakened: {sorted(weakened)}"
        )


def _validate_studies(
    studies_value: Any,
    resources: dict[str, Any],
    report_metrics: set[str],
) -> None:
    if not isinstance(studies_value, list) or not studies_value:
        raise MatrixError("physical studies are missing")
    ids: set[str] = set()
    for index, study in enumerate(studies_value):
        if not isinstance(study, dict):
            raise MatrixError(f"study {index} must be an object")
        axis = study.get("refinement_axis")
        expected_fields = STUDY_FIELDS_BY_AXIS.get(axis)
        if expected_fields is None:
            raise MatrixError(f"study {index} refinement axis is invalid")
        _require_fields(study, expected_fields, "study fields")
        study_id = nonempty_string(study["id"], f"study {index} id")
        if study_id in ids:
            raise MatrixError(f"duplicate study id {study_id!r}")
        ids.add(study_id)
        case_name = study["case"]
        if case_name not in SUPPORTED_CASE_DIMENSIONS:
            raise MatrixError(f"study {study_id!r} case is unsupported")
        dimension = study["dimension"]
        if dimension != SUPPORTED_CASE_DIMENSIONS[case_name]:
            raise MatrixError(f"study {study_id!r} dimension is inconsistent")
        if study["initialization"] not in SUPPORTED_INITIALIZATIONS:
            raise MatrixError(f"study {study_id!r} initialization is invalid")
        steps = study["steps"]
        if not isinstance(steps, int) or isinstance(steps, bool) or steps < 1:
            raise MatrixError(f"study {study_id!r} step count is invalid")
        if study["resource_profile"] not in resources["profiles"]:
            raise MatrixError(f"study {study_id!r} resource profile is unknown")
        scope = study["scope"]
        _require_fields(
            scope,
            {"balanced_force_closure_evidence", "fsr04_closure_evidence", "purpose"},
            f"study {study_id!r} scope fields",
        )
        if scope["balanced_force_closure_evidence"] is not True:
            raise MatrixError(f"study {study_id!r} is not balanced-force evidence")
        if scope["fsr04_closure_evidence"] is not False:
            raise MatrixError(f"study {study_id!r} cannot close FSR-04")
        nonempty_string(scope["purpose"], f"study {study_id!r} purpose")
        arguments = study["arguments"]
        if not isinstance(arguments, list) or any(not isinstance(value, str) for value in arguments):
            raise MatrixError(f"study {study_id!r} arguments are invalid")
        if "--interface-quadrature-order" in arguments:
            raise MatrixError("study interface quadrature override is forbidden")
        axes = study["axes"]
        expected_axes = (
            {"active_domain", "offset_h"}
            if case_name in {"droplet2d", "sphere3d"}
            else {"active_domain", "contact_angle", "wall", "offset_h"}
        )
        _require_fields(axes, expected_axes, "axis fields")
        for axis_name, values in axes.items():
            if not isinstance(values, list) or not values:
                raise MatrixError(f"study {study_id!r} axis {axis_name!r} is empty")
        if set(axes["active_domain"]) != REQUIRED_ACTIVE_DOMAINS:
            raise MatrixError(f"study {study_id!r} active-domain signs changed")
        if len(axes["offset_h"]) < 2:
            raise MatrixError(f"study {study_id!r} needs at least two offsets")
        if "wall" in axes:
            if set(axes["wall"]) - REQUIRED_WALLS[dimension]:
                raise MatrixError(f"study {study_id!r} wall axis is invalid")
            if set(axes["contact_angle"]) - REQUIRED_ANGLES:
                raise MatrixError(f"study {study_id!r} angle axis is invalid")
            if study_id in REQUIRED_MAIN_STUDIES:
                if set(axes["wall"]) != REQUIRED_WALLS[dimension]:
                    raise MatrixError(f"study {study_id!r} wall coverage is incomplete")
                if set(axes["contact_angle"]) != REQUIRED_ANGLES:
                    raise MatrixError(f"study {study_id!r} angle coverage is incomplete")
        metrics = set(_unique_strings(study["metrics"], f"study {study_id!r} metrics"))
        if study_id in REQUIRED_MAIN_STUDIES:
            required = REQUIRED_MAIN_METRICS | report_metrics
            if case_name.startswith("sessile"):
                required |= REQUIRED_SESSILE_METRICS
            if not required.issubset(metrics | report_metrics):
                raise MatrixError(f"study {study_id!r} metrics are incomplete")
        if axis == "resolution":
            finite_number(study["radius"], f"study {study_id!r} radius", positive=True)
            finite_number(study["time_step"], f"study {study_id!r} time step", positive=True)
        elif axis == "physical_scale":
            levels = study["refinement_levels"]
            if not isinstance(levels, list) or len(levels) != 3:
                raise MatrixError(f"study {study_id!r} needs three physical scales")
            for level in levels:
                _require_fields(level, {"label", "radius", "surface_tension"}, "physical-scale level fields")
                nonempty_string(level["label"], "physical-scale label")
                finite_number(level["radius"], "physical-scale radius", positive=True)
                finite_number(level["surface_tension"], "physical-scale tension", positive=True)
            cells = study["cells_per_radius"]
            if not isinstance(cells, int) or isinstance(cells, bool) or cells < 2:
                raise MatrixError(f"study {study_id!r} cells-per-radius is invalid")
        else:
            levels = _validate_number_list(
                study["refinement_levels"],
                f"study {study_id!r} refinement levels",
                count=3,
                positive=True,
            )
            resolution = study["resolution"]
            if not isinstance(resolution, int) or isinstance(resolution, bool) or resolution < 2:
                raise MatrixError(f"study {study_id!r} resolution is invalid")
            finite_number(study["radius"], f"study {study_id!r} radius", positive=True)
            if axis == "time_step":
                counts = study["level_step_counts"]
                if (
                    not isinstance(counts, list)
                    or len(counts) != len(levels)
                    or any(
                        not isinstance(value, int)
                        or isinstance(value, bool)
                        or value < 1
                        for value in counts
                    )
                ):
                    raise MatrixError(f"study {study_id!r} level step counts are invalid")
                if any(
                    coarse <= fine for coarse, fine in zip(levels[:-1], levels[1:])
                ):
                    raise MatrixError(
                        f"study {study_id!r} time steps must be strictly decreasing"
                    )
                if any(
                    coarse >= fine for coarse, fine in zip(counts[:-1], counts[1:])
                ):
                    raise MatrixError(
                        f"study {study_id!r} step counts must be strictly increasing"
                    )
                horizon = finite_number(
                    study["physical_horizon"],
                    f"study {study_id!r} physical horizon",
                    positive=True,
                )
                products = [level * count for level, count in zip(levels, counts)]
                if any(product != horizon for product in products):
                    raise MatrixError(f"study {study_id!r} does not preserve its physical horizon")
            else:
                finite_number(study["time_step"], f"study {study_id!r} time step", positive=True)
            if axis == "phi_scale":
                if any(level <= 0 for level in levels):
                    raise MatrixError("positive level-set scaling requires positive levels")
                if arguments.count("--enable-level-set-reinitialization") != 1:
                    raise MatrixError("positive scaling must enable wall maintenance")
                if _option_values(arguments, "--reinitialization-cadence-steps") != ["1"]:
                    raise MatrixError("positive scaling wall maintenance cadence changed")
            if axis == "bulk_redistance_cadence":
                if levels != [4.0, 2.0, 1.0]:
                    raise MatrixError("bulk-redistance cadence levels changed")
                if arguments.count("--enable-level-set-reinitialization") != 1:
                    raise MatrixError("bulk-redistance study must enable reinitialization")
    if not REQUIRED_MAIN_STUDIES.issubset(ids):
        raise MatrixError("required curved physical studies are missing")


def _validate_literature(adaptations: Any, registry: dict[str, Any]) -> None:
    if not isinstance(adaptations, list) or not adaptations:
        raise MatrixError("literature adaptations are missing")
    category_ids = {group["id"] for group in registry["exact_groups"]}
    category_tests = {
        group["id"]: {
            test
            for invocation in group["invocations"]
            for test in invocation["tests"]
        }
        for group in registry["exact_groups"]
    }
    study_ids = {study["id"] for study in registry["studies"]}
    seen: set[str] = set()
    for index, adaptation in enumerate(adaptations):
        if not isinstance(adaptation, dict):
            raise MatrixError(f"literature adaptation {index} is invalid")
        expected = {"id", "source", "limitations"}
        if "adapted_study" in adaptation:
            expected.add("adapted_study")
        else:
            expected |= {"adapted_evidence_group", "adapted_test"}
        _require_fields(adaptation, expected, "literature-adaptation fields")
        adaptation_id = nonempty_string(adaptation["id"], "literature adaptation id")
        if adaptation_id in seen:
            raise MatrixError(f"duplicate literature adaptation {adaptation_id!r}")
        seen.add(adaptation_id)
        _require_fields(adaptation["source"], {"authors", "year", "doi"}, "literature-source fields")
        nonempty_string(adaptation["source"]["doi"], "literature DOI")
        _unique_strings(adaptation["limitations"], "literature limitations")
        if "adapted_study" in adaptation:
            if adaptation["adapted_study"] not in study_ids:
                raise MatrixError("literature adaptation names unknown study")
        else:
            category_id = adaptation["adapted_evidence_group"]
            if category_id not in category_ids:
                raise MatrixError("literature adaptation names unknown exact category")
            if adaptation["adapted_test"] not in category_tests[category_id]:
                raise MatrixError("literature adaptation names unknown exact test")


def validate_contract(registry: Any) -> dict[str, Any]:
    _require_fields(registry, TOP_LEVEL_FIELDS, "top-level fields")
    if registry["schema_version"] != 3:
        raise MatrixError("V3 schema version changed")
    if registry["matrix_id"] != EXPECTED_MATRIX_ID:
        raise MatrixError("V3 matrix id changed")
    if registry["status"] != EXPECTED_STATUS:
        raise MatrixError("V3 freeze status changed")
    if registry["work_package"] != "WP-4":
        raise MatrixError("V3 work-package id changed")
    if registry["findings"] != ["FSR-03", "FSR-04"]:
        raise MatrixError("V3 finding list changed")
    nonempty_string(registry["qualification_scope"], "qualification scope")
    _require_fields(
        registry["model_envelope"],
        {
            "phase_model",
            "exterior_model",
            "mesh_and_geometry",
            "capillary_force",
            "pressure_stabilization",
            "interface_quadrature_order",
            "force_projection_applied",
            "higher_order_claimed",
            "two_phase_claimed",
            "gas_sensitive_claimed",
        },
        "model-envelope fields",
    )
    model = registry["model_envelope"]
    if (
        model["capillary_force"] != "kinematic_area_gradient_energy_traction"
        or model["interface_quadrature_order"] < 2
        or model["force_projection_applied"] is not False
        or model["higher_order_claimed"] is not False
        or model["two_phase_claimed"] is not False
        or model["gas_sensitive_claimed"] is not False
    ):
        raise MatrixError("V3 model envelope changed")
    closure_fields = {
        "requested_claim",
        "requires_every_exact_group",
        "requires_every_required_case",
        "requires_every_convergence_sequence",
        "requires_conditional_level_when_available_and_triggered",
        "unavailable_conditional_level_disposition",
        "worst_offset_controls_disposition",
        "failed_and_inconclusive_runs_are_retained",
        "fsr03_closed_on_pass",
        "fsr04_closed_on_pass",
        "wp4_closed_on_pass",
        "q2_closed_on_pass",
    }
    _require_fields(registry["closure_policy"], closure_fields, "closure-policy fields")
    closure = registry["closure_policy"]
    if (
        closure["fsr03_closed_on_pass"] is not True
        or closure["fsr04_closed_on_pass"] is not False
        or closure["wp4_closed_on_pass"] is not False
        or closure["q2_closed_on_pass"] is not False
        or closure["failed_and_inconclusive_runs_are_retained"] is not True
    ):
        raise MatrixError("V3 closure policy overclaims qualification")
    maintenance = registry["maintenance_contract"]
    _require_fields(
        maintenance,
        {
            "prescribed_wall_maintenance",
            "bulk_redistance",
            "schedules_independent",
            "schedule_relationship",
            "qualification_note",
        },
        "maintenance-contract fields",
    )
    _require_fields(
        maintenance["prescribed_wall_maintenance"],
        {
            "enabled",
            "contact_model",
            "execution_stage",
            "enable_argument",
            "cadence_argument",
            "fsr04_closure_evidence",
        },
        "prescribed-maintenance fields",
    )
    _require_fields(
        maintenance["bulk_redistance"],
        {"enabled", "refinement_axis", "execution_stage", "cadence_argument"},
        "bulk-redistance fields",
    )
    if (
        maintenance["prescribed_wall_maintenance"]["enabled"] is not True
        or maintenance["prescribed_wall_maintenance"]["fsr04_closure_evidence"] is not False
        or maintenance["bulk_redistance"]["enabled"] is not True
        or maintenance["bulk_redistance"]["refinement_axis"]
        != "bulk_redistance_cadence"
        or maintenance["schedules_independent"] is not False
        or maintenance["schedule_relationship"]
        != "shared_projection_reinitialization_event_and_cadence"
    ):
        raise MatrixError("maintenance schedule contract changed")
    report_metrics = set(
        _unique_strings(registry["required_report_metrics"], "required report metrics")
    )
    if report_metrics != {"kinetic_energy_proxy", "liquid_volume_relative_error"}:
        raise MatrixError("required report metrics changed")
    _validate_resources(registry["resources"])
    _validate_refinement(registry["refinement"])
    _validate_gates(registry["gates"])
    _validate_exact_groups(registry["exact_groups"], registry["gates"]["energy_variation"])
    _validate_parent_exact_preservation(registry)
    _validate_studies(registry["studies"], registry["resources"], report_metrics)
    for invocation in exact_invocations(registry):
        if invocation["resource_profile"] not in registry["resources"]["profiles"]:
            raise MatrixError(
                f"exact invocation {invocation['id']!r} resource profile is unknown"
            )
    for study in registry["studies"]:
        axis = study["refinement_axis"]
        if axis in registry["gates"]["invariance"]:
            observed_metrics = set(study["metrics"]) | report_metrics
            expected_metrics = set(registry["gates"]["invariance"][axis])
            if observed_metrics != expected_metrics:
                raise MatrixError(f"study {study['id']!r} invariance metrics changed")
    common = registry["common_runner_arguments"]
    if not isinstance(common, list) or any(not isinstance(value, str) for value in common):
        raise MatrixError("common runner arguments are invalid")
    required_common = {
        "--capillary-force-form",
        "kinematic_area_gradient_traction",
        "--projected-curvature-field",
        "kappa_area_gradient",
        "--cut-cell-pressure-stabilization-policy",
        "incremental",
        "--defer-static-physical-gates-to-matrix",
        "--require-free-surface-energy-history",
    }
    if not required_common.issubset(common):
        raise MatrixError("common balanced-capillary arguments changed")
    orders = _option_values(common, "--interface-quadrature-order")
    if len(orders) != 1 or int(orders[0]) < 2:
        raise MatrixError("V3 requires exactly one quadratic interface order")
    _validate_literature(registry["literature_adaptations"], registry)
    _validate_artifact_and_provenance(registry)
    _validate_execution_contract(registry)
    cases = _expand_cases_unchecked(registry)
    validate_case_resources(registry, cases)
    for dimension in (2, 3):
        availability = registry["refinement"]["conditional_level_by_dimension"][
            str(dimension)
        ]["availability"]
        for study in registry["studies"]:
            if study["dimension"] != dimension or study["refinement_axis"] != "resolution":
                continue
            radius = float(study["radius"])
            base = math.ceil(8.0 / radius - 1.0e-14)
            conditional_case = {
                "case_id": f"{study['id']}--conditional-memory-check",
                "dimension": dimension,
                "resolution": base * 8,
            }
            estimate = estimate_case_memory_mib(registry, conditional_case)
            fits = estimate <= registry["resources"]["memory_mib_per_node"]
            if (availability == "AVAILABLE") is not fits:
                raise MatrixError(
                    f"study {study['id']!r} conditional-level availability "
                    "contradicts the frozen memory model"
                )
    return registry


def load_registry(path: Path = DEFAULT_REGISTRY) -> dict[str, Any]:
    resolved = path.resolve()
    if sha256_file(resolved) != EXPECTED_REGISTRY_SHA256:
        raise MatrixError("V3 frozen registry bytes changed")
    if sha256_file(PARENT_RUNNER_PATH) != EXPECTED_PARENT_RUNNER_SHA256:
        raise MatrixError("frozen V2 runner bytes changed")
    if sha256_file(PARENT_REGISTRY_PATH) != EXPECTED_PARENT_REGISTRY_SHA256:
        raise MatrixError("frozen V2 registry bytes changed")
    if sha256_file(PHYSICAL_RUNNER) != EXPECTED_PHYSICAL_RUNNER_SHA256:
        raise MatrixError("physical runner changed after V3 freeze")
    return validate_contract(read_json(resolved))


def _validate_artifact_and_provenance(registry: dict[str, Any]) -> None:
    artifact = registry["artifact_contract"]
    _require_fields(
        artifact,
        {
            "physical_case_files",
            "serial_exact_files",
            "mpi_exact_rank_file_template",
            "mpi_exact_common_files",
            "pre_execution_manifest_file",
            "conditional_trigger_record_file",
            "summary_files",
        },
        "artifact-contract fields",
    )
    for key in (
        "physical_case_files",
        "serial_exact_files",
        "mpi_exact_common_files",
        "summary_files",
    ):
        _unique_strings(artifact[key], f"artifact contract {key}")
    if artifact["mpi_exact_rank_file_template"] != "gtest_rank_{rank}.json":
        raise MatrixError("MPI exact rank artifact template changed")
    if (
        artifact["pre_execution_manifest_file"] != "pre_execution_manifest.json"
        or artifact["conditional_trigger_record_file"]
        != "conditional_trigger_record.json"
    ):
        raise MatrixError("named pre-execution artifacts changed")
    provenance = registry["provenance_contract"]
    _require_fields(
        provenance,
        {
            "manifest_schema_version",
            "required_dependency_keys",
            "required_binary_keys",
            "required_hash_bindings",
            "source_commit_format",
            "dry_manifest_required_before_execution",
            "source_worktree_requires_clean",
            "source_head_requires_detached",
            "source_head_must_equal_declared_commit",
            "tracked_source_digest_required",
            "required_missing_lfs_object_count",
            "required_lfs_tracked_object_count",
            "solver_hash_required",
            "mpi_launcher_must_match_bound_executable",
            "pre_execution_manifest_inside_output_root",
        },
        "provenance-contract fields",
    )
    if provenance["manifest_schema_version"] != 1:
        raise MatrixError("dry manifest schema version changed")
    dependencies = set(
        _unique_strings(
            provenance["required_dependency_keys"],
            "required dependency keys",
        )
    )
    if dependencies != {"openblas", "vtk", "ncurses", "libxml2"}:
        raise MatrixError("required dependency hash union changed")
    binaries = set(
        _unique_strings(provenance["required_binary_keys"], "required binary keys")
    )
    invocation_binaries = {
        invocation["binary"] for invocation in exact_invocations(registry)
    }
    if binaries != invocation_binaries:
        raise MatrixError("required binary hash union changed")
    bindings = set(
        _unique_strings(provenance["required_hash_bindings"], "required hash bindings")
    )
    if bindings != {
        "matrix",
        "runner",
        "physical_runner",
        "source_commit",
        "source_tree",
        "tracked_source",
        "compiler",
        "mpi",
        "dependencies",
        "binaries",
        "solver",
        "conditional_trigger",
        "exact_invocation_lifecycle",
    }:
        raise MatrixError("required manifest hash bindings changed")
    if (
        provenance["source_commit_format"] != "lowercase_40_hex"
        or provenance["dry_manifest_required_before_execution"] is not True
        or provenance["source_worktree_requires_clean"] is not True
        or provenance["source_head_requires_detached"] is not True
        or provenance["source_head_must_equal_declared_commit"] is not True
        or provenance["tracked_source_digest_required"] is not True
        or provenance["required_missing_lfs_object_count"] != 0
        or provenance["required_lfs_tracked_object_count"]
        != EXPECTED_LFS_TRACKED_OBJECT_COUNT
        or provenance["solver_hash_required"] is not True
        or provenance["mpi_launcher_must_match_bound_executable"] is not True
        or provenance["pre_execution_manifest_inside_output_root"] is not True
    ):
        raise MatrixError("dry manifest provenance policy changed")


def _validate_execution_contract(registry: dict[str, Any]) -> None:
    expected = {
        "output_root_creation": "exclusive",
        "physical_retry_policy": "reject_nonempty_target",
        "rerun_allowed": False,
        "revalidate_before_each_numerical_action": True,
        "operational_setup_phase": "Task4B",
    }
    if registry["execution_contract"] != expected:
        raise MatrixError("immutable execution contract changed")


def exact_invocations(registry: dict[str, Any]) -> list[dict[str, Any]]:
    invocations: list[dict[str, Any]] = []
    for category in registry["exact_groups"]:
        for invocation in category["invocations"]:
            record = copy.deepcopy(invocation)
            record["category_id"] = category["id"]
            record["category_purpose"] = category["purpose"]
            record["id"] = f"{category['id']}--{invocation['id']}"
            record["required_matrix_case_count"] = 0
            invocations.append(record)
    return invocations


def _parent_registry(registry: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(registry)
    result["exact_groups"] = exact_invocations(registry)
    result["resources"]["wall_time_seconds_per_case"] = max(
        profile["wall_time_seconds"]
        for profile in registry["resources"]["profiles"].values()
    )
    return result


def _base_case_set_sha256(cases: Sequence[dict[str, Any]]) -> str:
    return _canonical_sha256(
        [
            {"case_id": case["case_id"], "case_digest": case["case_digest"]}
            for case in cases
        ]
    )


def _conditional_record_header(
    registry: dict[str, Any], base_cases: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "matrix_id": registry["matrix_id"],
        "matrix_sha256": sha256_file(DEFAULT_REGISTRY),
        "runner_sha256": sha256_file(SCRIPT_PATH),
        "physical_runner_sha256": sha256_file(PHYSICAL_RUNNER),
        "trigger_policy": "nonmonotone_three_level_sequence_only",
        "prior_analysis_file": "summary.json",
        "prior_pre_execution_manifest_file": registry["artifact_contract"][
            "pre_execution_manifest_file"
        ],
        "base_case_count": len(base_cases),
        "base_case_set_sha256": _base_case_set_sha256(base_cases),
    }


def _conditional_identity(
    base_cases: Sequence[dict[str, Any]],
    study: dict[str, Any],
    metric: str,
    axes: Any,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if (
        metric not in study["metrics"]
        or not isinstance(axes, dict)
        or set(axes) != set(study["axes"])
        or any(value not in study["axes"][name] for name, value in axes.items())
    ):
        raise MatrixError("conditional sequence axes or metric are undeclared")
    matching = sorted(
        (
            case
            for case in base_cases
            if case["study_id"] == study["id"] and case["axes"] == axes
        ),
        key=lambda case: float(case["level"]["value"]),
    )
    if [case["level"]["value"] for case in matching] != [8.0, 16.0, 32.0]:
        raise MatrixError("conditional base sequence identity is incomplete")
    identity = {
        "study_id": study["id"],
        "dimension": study["dimension"],
        "metric": metric,
        "refinement_axis": "resolution",
        "axes": axes,
        "base_case_ids": [case["case_id"] for case in matching],
        "base_case_digests": [case["case_digest"] for case in matching],
    }
    return identity, matching


def _validate_prior_analysis_header(
    registry: dict[str, Any], analysis: dict[str, Any]
) -> None:
    base_cases = _expand_cases_unchecked(registry)
    frozen = {
        "matrix_id": registry["matrix_id"],
        "registry_sha256": sha256_file(DEFAULT_REGISTRY),
        "runner_sha256": sha256_file(SCRIPT_PATH),
        "physical_runner_sha256": sha256_file(PHYSICAL_RUNNER),
        "expected_case_count": len(base_cases),
    }
    if any(analysis.get(key) != expected for key, expected in frozen.items()):
        raise MatrixError("conditional prior analysis frozen identity changed")
    if analysis.get("conditional_trigger_record_sha256") is not None:
        raise MatrixError("conditional trigger must come from a three-level base analysis")
    if not re.fullmatch(
        r"[0-9a-f]{64}", str(analysis.get("pre_execution_manifest_sha256"))
    ):
        raise MatrixError("conditional prior provenance manifest hash is missing")
    if (
        analysis.get("qualification_outcome")
        not in {"PASS", "ADDITIONAL_LEVEL_REQUIRED", "INCONCLUSIVE"}
        or analysis.get("exact_groups_passed") is not True
        or not isinstance(analysis.get("invariance"), dict)
        or analysis["invariance"].get("status") != "PASS"
        or not isinstance(analysis.get("finest_level"), dict)
        or analysis["finest_level"].get("status") != "PASS"
    ):
        raise MatrixError("conditional prior analysis contains an actual failure")


def _dict_member(value: Any, key: str, context: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not isinstance(value.get(key), dict):
        raise MatrixError(f"{context} is malformed")
    return value[key]


def _json_object(value: Any, context: str) -> dict[str, Any]:
    try:
        result = json.loads(value)
    except (TypeError, json.JSONDecodeError) as error:
        raise MatrixError(f"{context} is malformed") from error
    if not isinstance(result, dict):
        raise MatrixError(f"{context} must be an object")
    return result


def _conditional_sequence_records(
    registry: dict[str, Any], analysis: dict[str, Any]
) -> list[dict[str, Any]]:
    base_cases = _expand_cases_unchecked(registry)
    analyzed_studies = _dict_member(
        _dict_member(analysis, "convergence", "conditional prior convergence"),
        "studies",
        "conditional prior convergence studies",
    )
    studies = {study["id"]: study for study in registry["studies"]}
    records: list[dict[str, Any]] = []
    for study_id, study_analysis in analyzed_studies.items():
        study = studies.get(study_id)
        if study is None or study["refinement_axis"] != "resolution":
            continue
        groups = _dict_member(study_analysis, "groups", "conditional study analysis")
        for group_key, group_analysis in groups.items():
            group_axes = _json_object(group_key, "conditional group identity")
            metrics = _dict_member(
                group_analysis, "metrics", "conditional group analysis"
            )
            for metric, metric_analysis in metrics.items():
                sequences = _dict_member(
                    metric_analysis, "sequences", "conditional metric analysis"
                )
                for offset_key, sequence in sequences.items():
                    if not isinstance(sequence, dict):
                        raise MatrixError("conditional sequence analysis is malformed")
                    if sequence.get("status") != "ADDITIONAL_LEVEL_REQUIRED":
                        continue
                    if (
                        sequence.get("sample_count") != 3
                        or sequence.get("monotone_to_reference") is not False
                        or sequence.get("gate_failures")
                        != ["asymptotic_tail_not_established"]
                    ):
                        raise MatrixError(
                            "conditional trigger is not a nonmonotone three-level sequence"
                        )
                    samples = sequence.get("samples")
                    if not isinstance(samples, list) or len(samples) != 3:
                        raise MatrixError("conditional sequence samples are malformed")
                    try:
                        offset = json.loads(offset_key)
                    except (TypeError, json.JSONDecodeError) as error:
                        raise MatrixError(
                            "conditional offset identity is malformed"
                        ) from error
                    axes = {**group_axes, "offset_h": offset}
                    identity, matching = _conditional_identity(
                        base_cases, study, metric, axes
                    )
                    expected_labels = [case["level"]["label"] for case in matching]
                    observed_labels = [sample.get("label") for sample in samples]
                    if observed_labels != expected_labels:
                        raise MatrixError("conditional base sequence labels changed")
                    conditional = registry["refinement"][
                        "conditional_level_by_dimension"
                    ][str(study["dimension"])]
                    records.append(
                        {
                            "sequence_id": _canonical_sha256(identity),
                            **identity,
                            "prior_status": "ADDITIONAL_LEVEL_REQUIRED",
                            "trigger_reason": "nonmonotone_three_level_sequence",
                            "availability": conditional["availability"],
                            "disposition": conditional["disposition_when_required"],
                        }
                    )
    sequence_ids = [record["sequence_id"] for record in records]
    if len(sequence_ids) != len(set(sequence_ids)):
        raise MatrixError("conditional sequence identities are duplicated")
    return sorted(records, key=lambda record: record["sequence_id"])


def build_conditional_trigger_record(
    registry: dict[str, Any], prior_analysis_path: Path
) -> dict[str, Any]:
    prior_analysis_path = prior_analysis_path.resolve()
    if prior_analysis_path.name != "summary.json":
        raise MatrixError("conditional prior analysis must be the named summary")
    analysis = read_json(prior_analysis_path)
    if not isinstance(analysis, dict):
        raise MatrixError("conditional prior analysis must be an object")
    _validate_prior_analysis_header(registry, analysis)
    base_cases = _expand_cases_unchecked(registry)
    prior_manifest_path = prior_analysis_path.with_name(
        registry["artifact_contract"]["pre_execution_manifest_file"]
    )
    if (
        not prior_manifest_path.is_file()
        or sha256_file(prior_manifest_path)
        != analysis["pre_execution_manifest_sha256"]
    ):
        raise MatrixError("conditional prior provenance manifest changed")
    return {
        **_conditional_record_header(registry, base_cases),
        "prior_analysis_sha256": sha256_file(prior_analysis_path),
        "prior_pre_execution_manifest_sha256": sha256_file(prior_manifest_path),
        "sequences": _conditional_sequence_records(registry, analysis),
    }


def load_conditional_trigger_record(
    registry: dict[str, Any], path: Path
) -> dict[str, Any]:
    resolved = path.resolve()
    record = read_json(resolved)
    if not isinstance(record, dict) or record.get("prior_analysis_file") != "summary.json":
        raise MatrixError("conditional trigger record is malformed")
    expected = build_conditional_trigger_record(
        registry, resolved.parent / record["prior_analysis_file"]
    )
    if record != expected:
        raise MatrixError("conditional trigger record is stale or malformed")
    return record


def _conditional_expansion_keys(
    registry: dict[str, Any], record: dict[str, Any] | None
) -> dict[str, set[str]]:
    if record is None:
        return {}
    base_cases = _expand_cases_unchecked(registry)
    expected_header = _conditional_record_header(registry, base_cases)
    digest_fields = {"prior_analysis_sha256", "prior_pre_execution_manifest_sha256"}
    _require_fields(
        record,
        set(expected_header) | digest_fields | {"sequences"},
        "conditional-trigger fields",
    )
    if (
        any(record[key] != value for key, value in expected_header.items())
        or any(not re.fullmatch(r"[0-9a-f]{64}", str(record[key]))
               for key in digest_fields)
    ):
        raise MatrixError("conditional trigger frozen provenance changed")
    if not isinstance(record["sequences"], list):
        raise MatrixError("conditional trigger sequences are malformed")
    if not record["sequences"]:
        raise MatrixError("conditional trigger declares no sequences")
    studies = {study["id"]: study for study in registry["studies"]}
    sequence_fields = {
        "sequence_id",
        "study_id",
        "dimension",
        "metric",
        "refinement_axis",
        "axes",
        "base_case_ids",
        "base_case_digests",
        "prior_status",
        "trigger_reason",
        "availability",
        "disposition",
    }
    keys: dict[str, set[str]] = {}
    sequence_ids: set[str] = set()
    for sequence in record["sequences"]:
        if not isinstance(sequence, dict):
            raise MatrixError("conditional trigger sequence is malformed")
        if set(sequence) != sequence_fields:
            raise MatrixError("conditional trigger sequence fields changed")
        if sequence["sequence_id"] in sequence_ids:
            raise MatrixError("conditional trigger sequence is duplicated")
        sequence_ids.add(sequence["sequence_id"])
        study = studies.get(sequence["study_id"])
        if (
            study is None
            or sequence["dimension"] != study["dimension"]
            or sequence["refinement_axis"] != "resolution"
            or study["refinement_axis"] != "resolution"
            or sequence["prior_status"] != "ADDITIONAL_LEVEL_REQUIRED"
            or sequence["trigger_reason"] != "nonmonotone_three_level_sequence"
        ):
            raise MatrixError("conditional trigger sequence is undeclared")
        identity, _ = _conditional_identity(
            base_cases, study, sequence["metric"], sequence["axes"]
        )
        conditional = registry["refinement"]["conditional_level_by_dimension"][
            str(study["dimension"])
        ]
        if (
            sequence["sequence_id"] != _canonical_sha256(identity)
            or sequence["base_case_ids"] != identity["base_case_ids"]
            or sequence["base_case_digests"] != identity["base_case_digests"]
            or sequence["availability"] != conditional["availability"]
            or sequence["disposition"] != conditional["disposition_when_required"]
        ):
            raise MatrixError("conditional trigger sequence identity changed")
        if sequence.get("disposition") != "EXECUTE":
            continue
        if sequence.get("dimension") != 2 or sequence.get("availability") != "AVAILABLE":
            raise MatrixError("unavailable conditional sequence cannot be expanded")
        keys.setdefault(study["id"], set()).add(_canonical_sha256(sequence["axes"]))
    if not keys:
        raise MatrixError("conditional trigger contains no executable sequences")
    return keys


def _expand_cases_unchecked(
    registry: dict[str, Any], *, conditional_trigger_record: dict[str, Any] | None = None
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    conditional_keys = _conditional_expansion_keys(
        registry, conditional_trigger_record
    )
    for study_index, study in enumerate(registry["studies"]):
        local = copy.deepcopy(registry)
        local["studies"] = [copy.deepcopy(study)]
        include_study_conditional = (
            study["refinement_axis"] == "resolution"
            and study["dimension"] == 2
            and study["id"] in conditional_keys
        )
        expanded = _V2_EXPAND_CASES(
            local,
            include_conditional_level=include_study_conditional,
        )
        for case in expanded:
            is_conditional = (
                study["refinement_axis"] == "resolution"
                and float(case["level"]["value"])
                == float(
                    registry["refinement"][
                        "conditional_spatial_level_cells_per_radius"
                    ]
                )
            )
            if is_conditional and _canonical_sha256(case["axes"]) not in (
                conditional_keys.get(study["id"], set())
            ):
                continue
            case["study_index"] = study_index
            if study["refinement_axis"] == "time_step":
                level_index = next(
                    index
                    for index, value in enumerate(study["refinement_levels"])
                    if float(value) == float(case["level"]["value"])
                )
                case["step_count"] = int(study["level_step_counts"][level_index])
            else:
                case["step_count"] = int(study["steps"])
            case["time_step"] = (
                float(case["level"]["value"])
                if study["refinement_axis"] == "time_step"
                else float(study["time_step"])
            )
            case["physical_horizon"] = case["time_step"] * case["step_count"]
            case["resource_profile"] = study["resource_profile"]
            case["estimated_memory_mib"] = estimate_case_memory_mib(registry, case)
            result.append(case)
    ids = [case["case_id"] for case in result]
    digests = [case["case_digest"] for case in result]
    if len(ids) != len(set(ids)):
        raise MatrixError("expanded physical case ids are not unique")
    if len(digests) != len(set(digests)):
        raise MatrixError("expanded physical case digests are not unique")
    for index, case in enumerate(result):
        case["index"] = index
    return result


def expand_cases(
    registry: dict[str, Any], *, conditional_trigger_record: dict[str, Any] | None = None
) -> list[dict[str, Any]]:
    cases = _expand_cases_unchecked(
        registry, conditional_trigger_record=conditional_trigger_record
    )
    validate_case_resources(registry, cases)
    return cases


def estimate_case_memory_mib(
    registry: dict[str, Any], case: dict[str, Any]
) -> int:
    model = registry["resources"]["memory_model"]
    dimension = int(case["dimension"])
    resolution = int(case["resolution"])
    if dimension not in {2, 3} or resolution < 1:
        raise MatrixError("case dimension or resolution is invalid for memory estimate")
    dimension_key = str(dimension)
    vertices = (resolution + 1) ** dimension
    simplices = (
        int(model["simplex_count_by_dimension"][dimension_key])
        * resolution**dimension
    )
    coupled = int(model["coupled_unknown_components_by_dimension"][dimension_key])
    stored = int(model["stored_field_components_by_dimension"][dimension_key])
    adjacency = int(model["adjacency_upper_bound_by_dimension"][dimension_key])
    fixed_bytes = int(model["fixed_mib"]) * 1024 * 1024
    generated_bytes = vertices * int(model["vertex_bytes"])
    simplex_bytes = simplices * int(model["simplex_bytes"])
    sparse_bytes = (
        vertices
        * coupled
        * adjacency
        * coupled
        * int(model["sparse_entry_bytes"])
        * int(model["sparse_operator_copies"])
    )
    vector_bytes = (
        vertices
        * stored
        * int(model["scalar_bytes"])
        * int(model["field_vector_copies"])
    )
    return math.ceil(
        (fixed_bytes + generated_bytes + simplex_bytes + sparse_bytes + vector_bytes)
        / (1024 * 1024)
    )


def validate_case_resources(
    registry: dict[str, Any], cases: Sequence[dict[str, Any]]
) -> None:
    one_node_limit = int(registry["resources"]["memory_mib_per_node"])
    profiles = registry["resources"]["profiles"]
    for case in cases:
        estimate = estimate_case_memory_mib(registry, case)
        recorded = case.get("estimated_memory_mib")
        if recorded is not None and recorded != estimate:
            raise MatrixError(
                f"case {case.get('case_id')!r} memory estimate does not match formula"
            )
        if estimate > one_node_limit:
            raise MatrixError(
                f"case {case.get('case_id')!r} exceeds one-node memory: "
                f"{estimate} MiB > {one_node_limit} MiB"
            )
        profile_id = case.get("resource_profile")
        if profile_id is not None:
            if profile_id not in profiles:
                raise MatrixError(f"case {case.get('case_id')!r} has unknown resource profile")
            if estimate > int(profiles[profile_id]["memory_mib"]):
                raise MatrixError(
                    f"case {case.get('case_id')!r} exceeds its resource profile memory"
                )


def physical_case_arguments(
    registry: dict[str, Any],
    case: dict[str, Any],
    *,
    solver: Path,
    qualification_log: Path,
) -> list[str]:
    adapted = _parent_registry(registry)
    study = adapted["studies"][case["study_index"]]
    study["steps"] = int(case["step_count"])
    arguments = _V2_PHYSICAL_CASE_ARGUMENTS(
        adapted,
        case,
        solver=solver,
        qualification_log=qualification_log,
    )
    if case["case"] == "droplet2d":
        arguments.extend(
            ["--capillary-droplet-radius", str(case["radius"])]
        )
    if case["refinement_axis"] == "bulk_redistance_cadence":
        arguments.extend(
            ["--reinitialization-cadence-steps", str(int(case["level"]["value"]))]
        )
    orders = _option_values(arguments, "--interface-quadrature-order")
    if len(orders) != 1 or int(orders[0]) < 2:
        raise MatrixError(
            f"case {case['case_id']!r} does not select one quadratic interface rule"
        )
    if float(case["surface_tension"]) > 0.0:
        if _option_values(arguments, "--capillary-force-form") != [
            "kinematic_area_gradient_traction"
        ]:
            raise MatrixError(
                f"case {case['case_id']!r} does not select energy traction"
            )
    return arguments


def _completion_property(payload: Any, test: str) -> int:
    if not isinstance(payload, dict):
        return 0
    for suite in payload.get("testsuites", []):
        if not isinstance(suite, dict):
            continue
        suite_name = suite.get("name")
        for case in suite.get("testsuite", []):
            if not isinstance(case, dict):
                continue
            classname = case.get("classname", suite_name)
            if f"{classname}.{case.get('name')}" != test:
                continue
            failures = case.get("failures")
            return int(
                case.get("status") == "RUN"
                and case.get("result") == "COMPLETED"
                and failures in (None, [])
            )
    return 0


def evaluate_exact_document(
    payload: Any,
    group: dict[str, Any],
    *,
    roundoff_factor: float,
    context: str,
) -> dict[str, Any]:
    enriched = copy.deepcopy(payload)
    if isinstance(enriched, dict):
        for suite in enriched.get("testsuites", []):
            if not isinstance(suite, dict):
                continue
            suite_name = suite.get("name")
            for case in suite.get("testsuite", []):
                if not isinstance(case, dict):
                    continue
                classname = case.get("classname", suite_name)
                full_name = f"{classname}.{case.get('name')}"
                gates = group.get("property_gates", {}).get(full_name, [])
                if any(
                    gate.get("property") == "wp4_exact_test_completed"
                    for gate in gates
                    if isinstance(gate, dict)
                ):
                    case["wp4_exact_test_completed"] = _completion_property(
                        enriched, full_name
                    )
    return _V2_EVALUATE_EXACT_DOCUMENT(
        enriched,
        group,
        roundoff_factor=roundoff_factor,
        context=context,
    )


@contextmanager
def _parent_overrides(**values: Any) -> Iterator[None]:
    previous = {name: getattr(_v2, name) for name in values}
    try:
        for name, value in values.items():
            setattr(_v2, name, value)
        yield
    finally:
        for name, value in previous.items():
            setattr(_v2, name, value)


def _command_with_guard(guard: Callable[[], None] | None) -> Callable[..., Any]:
    parent_command = _v2._run_command
    if guard is None:
        return parent_command

    def guarded(*args: Any, **options: Any) -> Any:
        guard()
        return parent_command(*args, **options)

    return guarded


def _positive_finite_seconds(value: Any, context: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise MatrixError(f"{context} must be a positive finite number")
    return float(value)


def _process_table_records() -> dict[int, dict[str, Any]]:
    records: dict[int, dict[str, Any]] = {}
    try:
        process_directories = list(Path("/proc").iterdir())
    except OSError:
        return records
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
            fields = stat[closing_parenthesis + 2 :].split()
            if len(fields) < 20:
                continue
            process_id = int(process_directory.name)
            records[process_id] = {
                "pid": process_id,
                "parent_pid": int(fields[1]),
                "process_group_id": int(fields[2]),
                "session_id": int(fields[3]),
                "state": fields[0],
                "start_time_ticks": int(fields[19]),
            }
        except (FileNotFoundError, OSError, ValueError):
            continue
    return records


def _owned_process_records(
    root_pid: int, identities: dict[int, int]
) -> list[dict[str, Any]]:
    table = _process_table_records()
    root = table.get(root_pid)
    if root is not None and root_pid not in identities:
        identities[root_pid] = root["start_time_ticks"]
    current = {
        process_id
        for process_id, start_time in identities.items()
        if process_id in table
        and table[process_id]["start_time_ticks"] == start_time
    }
    changed = True
    while changed:
        changed = False
        for process_id, record in table.items():
            if process_id in current or record["parent_pid"] not in current:
                continue
            if process_id in identities:
                continue
            identities[process_id] = record["start_time_ticks"]
            current.add(process_id)
            changed = True
    records = []
    for process_id in sorted(current):
        record = dict(table[process_id])
        try:
            with Path(f"/proc/{process_id}/cmdline").open("rb") as stream:
                command_bytes = stream.read(EXACT_DIAGNOSTIC_OUTPUT_BYTES)
            record["command_status"] = "captured"
        except PermissionError:
            command_bytes = b""
            record["command_status"] = "permission_denied"
        except OSError:
            command_bytes = b""
            record["command_status"] = "unavailable"
        record["command"] = command_bytes.replace(b"\0", b" ").decode(
            encoding="utf-8", errors="replace"
        )
        records.append(record)
    return records


def _managed_process(
    command: Sequence[str],
    *,
    cwd: Path | None = None,
    environment: dict[str, str] | None = None,
    stdout: Any = None,
    stderr: Any = None,
    output_limit: int = 0,
) -> subprocess.Popen[Any]:
    return subprocess.Popen(
        [sys.executable, "-c", _PROCESS_SUPERVISOR, str(output_limit), *command],
        cwd=cwd,
        env=environment,
        stdout=stdout,
        stderr=stderr,
        start_new_session=True,
    )


def _bounded_proc_read(path: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as stream:
            payload = stream.read(EXACT_DIAGNOSTIC_OUTPUT_BYTES)
    except PermissionError as error:
        return {"status": "permission_denied", "error": type(error).__name__}
    except FileNotFoundError as error:
        return {"status": "unavailable", "error": type(error).__name__}
    except OSError as error:
        return {"status": "read_failed", "error": type(error).__name__}
    return {
        "status": "captured",
        "content": payload.decode(encoding="utf-8", errors="replace"),
        "truncated": len(payload) == EXACT_DIAGNOSTIC_OUTPUT_BYTES,
    }


def _capture_stack_tool(
    processes: Sequence[dict[str, Any]], *, deadline: float
) -> dict[str, Any]:
    selected = None
    for name in ("gstack", "pstack"):
        selected = shutil.which(name)
        if selected is not None:
            break
    if selected is None:
        return {
            "status": "unavailable",
            "diagnostic": "gstack and pstack are unavailable on PATH",
            "captures": [],
        }
    captures: list[dict[str, Any]] = []
    for record in processes:
        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
            captures.append(
                {
                    "pid": record["pid"],
                    "status": "capture_budget_exhausted",
                }
            )
            break
        identities: dict[int, int] = {}
        process: subprocess.Popen[Any] | None = None
        timed_out = False
        termination: dict[str, Any] | None = None
        try:
            with tempfile.TemporaryFile(mode="w+b") as stdout_stream:
                with tempfile.TemporaryFile(mode="w+b") as stderr_stream:
                    process = _managed_process(
                        [selected, str(record["pid"])],
                        stdout=stdout_stream,
                        stderr=stderr_stream,
                        output_limit=EXACT_DIAGNOSTIC_OUTPUT_BYTES,
                    )
                    capture_deadline = time.monotonic() + min(0.25, remaining)
                    while process.poll() is None:
                        _owned_process_records(process.pid, identities)
                        capture_remaining = capture_deadline - time.monotonic()
                        if capture_remaining <= 0.0:
                            timed_out = True
                            break
                        time.sleep(min(0.01, capture_remaining))
                    if timed_out:
                        termination = _terminate_owned_processes(
                            process,
                            identities,
                            grace_seconds=(
                                EXACT_DIAGNOSTIC_TERMINATION_GRACE_SECONDS
                            ),
                        )
                    else:
                        process.wait()
                    stdout_stream.flush()
                    stderr_stream.flush()
                    stdout_stream.seek(0, os.SEEK_END)
                    stdout_size = stdout_stream.tell()
                    stderr_stream.seek(0, os.SEEK_END)
                    stderr_size = stderr_stream.tell()
                    stdout_stream.seek(0)
                    stderr_stream.seek(0)
                    stdout_payload = stdout_stream.read(
                        EXACT_DIAGNOSTIC_OUTPUT_BYTES
                    )
                    stderr_payload = stderr_stream.read(
                        EXACT_DIAGNOSTIC_OUTPUT_BYTES
                    )
            output_limited = (
                stdout_size >= EXACT_DIAGNOSTIC_OUTPUT_BYTES
                or stderr_size >= EXACT_DIAGNOSTIC_OUTPUT_BYTES
            )
            if output_limited:
                status = "output_limit_exceeded"
            elif timed_out:
                status = "capture_timed_out"
            elif process.returncode == 0:
                status = "captured"
            else:
                status = "attach_failed"
            captures.append(
                {
                    "pid": record["pid"],
                    "status": status,
                    "returncode": process.returncode,
                    "stdout": stdout_payload.decode(
                        encoding="utf-8", errors="replace"
                    ),
                    "stderr": stderr_payload.decode(
                        encoding="utf-8", errors="replace"
                    ),
                    "stdout_bytes": stdout_size,
                    "stderr_bytes": stderr_size,
                    "termination": termination,
                }
            )
        except (OSError, ValueError) as error:
            if process is not None and process.poll() is None:
                _terminate_owned_processes(
                    process,
                    identities,
                    grace_seconds=EXACT_DIAGNOSTIC_TERMINATION_GRACE_SECONDS,
                )
            captures.append(
                {
                    "pid": record["pid"],
                    "status": "capture_failed",
                    "error": type(error).__name__,
                }
            )
    return {"status": "attempted", "tool": selected, "captures": captures}


def _capture_timeout_diagnostics(
    command: Sequence[str],
    *,
    process: subprocess.Popen[Any],
    identities: dict[int, int],
    launcher_mode: str,
    ranks: int,
    elapsed_seconds: float,
) -> dict[str, Any]:
    all_processes = _owned_process_records(process.pid, identities)
    processes = all_processes[:EXACT_DIAGNOSTIC_PROCESS_LIMIT]
    capture_deadline = time.monotonic() + EXACT_DIAGNOSTIC_CAPTURE_SECONDS
    proc_captures = [
        {
            "pid": record["pid"],
            "stack": _bounded_proc_read(Path(f"/proc/{record['pid']}/stack")),
            "system_call": _bounded_proc_read(
                Path(f"/proc/{record['pid']}/syscall")
            ),
        }
        for record in processes
    ]
    return {
        "schema_version": 1,
        "command": list(command),
        "launcher_mode": launcher_mode,
        "ranks": ranks,
        "elapsed_seconds": elapsed_seconds,
        "launcher_pid": process.pid,
        "containment_pid": process.pid,
        "process_group_id": process.pid,
        "session_id": process.pid,
        "processes_before_termination": processes,
        "process_count_before_termination": len(all_processes),
        "process_capture_truncated": len(processes) != len(all_processes),
        "proc_captures": proc_captures,
        "stack_tool": _capture_stack_tool(processes, deadline=capture_deadline),
        "termination": None,
    }


def _signal_owned_processes(
    root_pid: int, identities: dict[int, int], signal_number: int
) -> list[int]:
    runner_process_group = os.getpgrp()
    process_groups = sorted(
        {
            record["process_group_id"]
            for record in _owned_process_records(root_pid, identities)
            if record["process_group_id"] != runner_process_group
            and record["state"] != "Z"
        }
    )
    signaled: list[int] = []
    for process_group in process_groups:
        try:
            os.killpg(process_group, signal_number)
        except ProcessLookupError:
            continue
        signaled.append(process_group)
    return signaled


def _terminate_owned_processes(
    process: subprocess.Popen[Any],
    identities: dict[int, int],
    *,
    grace_seconds: float,
) -> dict[str, Any]:
    grace = _positive_finite_seconds(grace_seconds, "exact termination grace")
    terminated_groups = set(
        _signal_owned_processes(process.pid, identities, signal.SIGTERM)
    )
    deadline = time.monotonic() + grace
    while time.monotonic() < deadline:
        process.poll()
        living = [
            record
            for record in _owned_process_records(process.pid, identities)
            if record["state"] != "Z"
        ]
        if not living:
            break
        time.sleep(min(0.02, max(0.0, deadline - time.monotonic())))
    killed_groups = set(
        _signal_owned_processes(process.pid, identities, signal.SIGKILL)
    )
    try:
        process.wait(timeout=max(0.05, grace))
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
            killed_groups.add(process.pid)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=max(0.05, grace))
        except subprocess.TimeoutExpired:
            pass
    remaining = _owned_process_records(process.pid, identities)
    living = [record for record in remaining if record["state"] != "Z"]
    all_terminated = process.poll() is not None and not living
    return {
        "terminate_process_group_ids": sorted(terminated_groups),
        "kill_process_group_ids": sorted(killed_groups),
        "remaining_processes": remaining,
        "all_owned_descendants_terminated": all_terminated,
        "all_session_processes_terminated": all_terminated,
    }


def _run_bounded_exact_command(
    command: Sequence[str],
    *,
    cwd: Path,
    stdout_path: Path,
    stderr_path: Path,
    environment: dict[str, str] | None = None,
    timeout_seconds: Any,
    termination_grace_seconds: Any,
    launcher_mode: str,
    ranks: int,
) -> dict[str, Any]:
    watchdog = _positive_finite_seconds(
        timeout_seconds, "exact-invocation watchdog"
    )
    grace = _positive_finite_seconds(
        termination_grace_seconds, "exact termination grace"
    )
    if launcher_mode not in {"mpiexec", "srun"}:
        raise MatrixError("exact MPI launcher mode is invalid")
    if not isinstance(ranks, int) or isinstance(ranks, bool) or ranks < 1:
        raise MatrixError("exact invocation rank count is invalid")
    started = time.monotonic()
    timed_out = False
    diagnostics_path: Path | None = None
    termination: dict[str, Any] | None = None
    identities: dict[int, int] = {}
    with stdout_path.open("w", encoding="utf-8") as stdout_stream:
        with stderr_path.open("w", encoding="utf-8") as stderr_stream:
            process = _managed_process(
                command,
                cwd=cwd,
                environment=environment,
                stdout=stdout_stream,
                stderr=stderr_stream,
            )
            try:
                deadline = started + watchdog
                while True:
                    process.poll()
                    living = [
                        record
                        for record in _owned_process_records(
                            process.pid, identities
                        )
                        if record["state"] != "Z"
                    ]
                    if process.returncode is not None and not living:
                        break
                    remaining = deadline - time.monotonic()
                    if remaining <= 0.0:
                        timed_out = True
                        break
                    time.sleep(min(0.02, remaining))
                if timed_out:
                    diagnostics_path = stdout_path.parent / "timeout_diagnostics.json"
                    try:
                        diagnostics = _capture_timeout_diagnostics(
                            command,
                            process=process,
                            identities=identities,
                            launcher_mode=launcher_mode,
                            ranks=ranks,
                            elapsed_seconds=time.monotonic() - started,
                        )
                    except BaseException as error:
                        diagnostics = {
                            "schema_version": 1,
                            "command": list(command),
                            "launcher_mode": launcher_mode,
                            "ranks": ranks,
                            "elapsed_seconds": time.monotonic() - started,
                            "launcher_pid": process.pid,
                            "capture_status": "failed",
                            "capture_error": type(error).__name__,
                            "termination": None,
                        }
                    write_json(diagnostics_path, diagnostics)
                    termination = _terminate_owned_processes(
                        process, identities, grace_seconds=grace
                    )
                    diagnostics["termination"] = termination
                    write_json(diagnostics_path, diagnostics)
            except BaseException:
                _terminate_owned_processes(
                    process, identities, grace_seconds=grace
                )
                raise
    return {
        "command": list(command),
        "returncode": process.returncode,
        "timed_out": timed_out,
        "terminal": process.poll() is not None,
        "elapsed_seconds": time.monotonic() - started,
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
        "process_id": process.pid,
        "containment_process_id": process.pid,
        "process_group_id": process.pid,
        "session_id": process.pid,
        "launcher_mode": launcher_mode,
        "mpi_ranks": ranks,
        "watchdog_seconds": watchdog,
        "timeout_diagnostics": (
            str(diagnostics_path) if diagnostics_path is not None else None
        ),
        "termination": termination,
    }


def _explicit_exact_command(
    parent_command: Sequence[str],
    *,
    launcher: Path,
    launcher_mode: str,
    ranks: int,
) -> list[str]:
    if launcher_mode not in {"mpiexec", "srun"}:
        raise MatrixError("exact MPI launcher mode is invalid")
    if not isinstance(ranks, int) or isinstance(ranks, bool) or ranks < 1:
        raise MatrixError("exact invocation rank count is invalid")
    launcher_path = launcher.resolve()
    launcher_arguments = _v2.exact_mpi_launcher_arguments(launcher_mode, ranks)
    command = list(parent_command)
    if ranks == 1:
        return [str(launcher_path), *launcher_arguments, *command]
    prefix_length = 1 + len(launcher_arguments)
    if (
        len(command) <= prefix_length
        or Path(command[0]).resolve() != launcher_path
        or command[1:prefix_length] != launcher_arguments
    ):
        raise MatrixError("parent exact command changed its declared rank route")
    return [str(launcher_path), *command[1:]]


def _exact_command_adapter(
    *,
    launcher: Path,
    launcher_mode: str,
    invocation_ranks: Sequence[int],
    watchdog_seconds: Any,
    termination_grace_seconds: Any,
    pre_execution_guard: Callable[[], None] | None,
) -> Callable[..., Any]:
    ranks_by_call = list(invocation_ranks)
    invocation_index = 0

    def run(command: Sequence[str], **options: Any) -> dict[str, Any]:
        nonlocal invocation_index
        if invocation_index >= len(ranks_by_call):
            raise MatrixError("parent exact runner issued an unexpected command")
        if options.get("timeout") is not None:
            raise MatrixError("exact invocation cannot override the frozen watchdog")
        options.pop("timeout", None)
        if pre_execution_guard is not None:
            pre_execution_guard()
        resolved_launcher = launcher.resolve()
        if not resolved_launcher.is_file() or not os.access(
            resolved_launcher, os.X_OK
        ):
            raise MatrixError("exact MPI launcher is not executable after provenance guard")
        ranks = ranks_by_call[invocation_index]
        invocation_index += 1
        explicit_command = _explicit_exact_command(
            command,
            launcher=resolved_launcher,
            launcher_mode=launcher_mode,
            ranks=ranks,
        )
        return _run_bounded_exact_command(
            explicit_command,
            timeout_seconds=watchdog_seconds,
            termination_grace_seconds=termination_grace_seconds,
            launcher_mode=launcher_mode,
            ranks=ranks,
            **options,
        )

    return run


def run_physical_cases(
    registry: dict[str, Any],
    cases: Sequence[dict[str, Any]],
    *,
    solver: Path,
    output_root: Path,
    rerun: bool,
    pre_execution_guard: Callable[[], None] | None = None,
) -> dict[str, Any]:
    validate_case_resources(registry, cases)
    if rerun:
        raise MatrixError(
            "V3 immutable physical evidence requires a fresh output root; "
            "rerun is not permitted"
        )
    immutable_targets = [
        output_root / "physical_execution_manifest.json",
        *(output_root / "cases" / case["case_id"] for case in cases),
    ]
    existing_targets = [path for path in immutable_targets if path.exists()]
    if existing_targets:
        raise MatrixError(
            "immutable physical evidence target already exists: "
            f"{existing_targets[0]}"
        )
    adapted = _parent_registry(registry)

    def mapped_arguments(
        unused_registry: dict[str, Any],
        case: dict[str, Any],
        *,
        solver: Path,
        qualification_log: Path,
    ) -> list[str]:
        del unused_registry
        return physical_case_arguments(
            registry,
            case,
            solver=solver,
            qualification_log=qualification_log,
        )

    with _parent_overrides(
        DEFAULT_REGISTRY=DEFAULT_REGISTRY,
        PHYSICAL_RUNNER=PHYSICAL_RUNNER,
        physical_case_arguments=mapped_arguments,
        _run_command=_command_with_guard(pre_execution_guard),
    ):
        return _V2_RUN_PHYSICAL_CASES(
            adapted,
            cases,
            solver=solver,
            output_root=output_root,
            rerun=False,
        )


def run_exact_groups(
    registry: dict[str, Any],
    *,
    binaries: dict[str, Path],
    mpi: Path,
    mpi_launcher_mode: str,
    output_root: Path,
    pre_execution_guard: Callable[[], None] | None = None,
) -> dict[str, Any]:
    adapted = _parent_registry(registry)
    watchdog_seconds = _exact_invocation_watchdog(registry.get("resources"))
    command_adapter = _exact_command_adapter(
        launcher=mpi,
        launcher_mode=mpi_launcher_mode,
        invocation_ranks=[
            invocation["mpi_ranks"] for invocation in exact_invocations(registry)
        ],
        watchdog_seconds=watchdog_seconds,
        termination_grace_seconds=EXACT_TERMINATION_GRACE_SECONDS,
        pre_execution_guard=pre_execution_guard,
    )
    with _parent_overrides(
        DEFAULT_REGISTRY=DEFAULT_REGISTRY,
        PHYSICAL_RUNNER=PHYSICAL_RUNNER,
        evaluate_exact_document=evaluate_exact_document,
        _run_command=command_adapter,
    ):
        return _V2_RUN_EXACT_GROUPS(
            adapted,
            binaries=binaries,
            mpi_launcher=mpi,
            mpi_launcher_mode=mpi_launcher_mode,
            output_root=output_root,
        )


def _analysis_cases(
    registry: dict[str, Any], *, conditional_trigger_record: dict[str, Any] | None
) -> list[dict[str, Any]]:
    cases = expand_cases(
        registry, conditional_trigger_record=conditional_trigger_record
    )
    for case in cases:
        if case["refinement_axis"] == "bulk_redistance_cadence":
            case["refinement_axis"] = "reinitialization_cadence"
    return cases


def analyze_evidence(
    registry: dict[str, Any],
    *,
    roots: Sequence[Path],
    output_root: Path,
    conditional_trigger_record_path: Path | None,
    exact_summary_path: Path | None,
) -> dict[str, Any]:
    conditional_trigger_record = (
        load_conditional_trigger_record(registry, conditional_trigger_record_path)
        if conditional_trigger_record_path is not None
        else None
    )
    adapted = _parent_registry(registry)
    for study in adapted["studies"]:
        if study["refinement_axis"] == "bulk_redistance_cadence":
            study["refinement_axis"] = "reinitialization_cadence"

    def adapted_expansion(
        unused_registry: dict[str, Any], *, include_conditional_level: bool = False
    ) -> list[dict[str, Any]]:
        del unused_registry, include_conditional_level
        return _analysis_cases(
            registry, conditional_trigger_record=conditional_trigger_record
        )

    with _parent_overrides(
        DEFAULT_REGISTRY=DEFAULT_REGISTRY,
        PHYSICAL_RUNNER=PHYSICAL_RUNNER,
        expand_cases=adapted_expansion,
    ):
        summary = _V2_ANALYZE_EVIDENCE(
            adapted,
            roots=roots,
            output_root=output_root,
            include_conditional_level=False,
            exact_summary_path=exact_summary_path,
        )
    summary["runner_sha256"] = sha256_file(SCRIPT_PATH)
    summary["conditional_trigger_record_sha256"] = (
        sha256_file(conditional_trigger_record_path)
        if conditional_trigger_record_path is not None
        else None
    )
    pre_execution_manifest_path = output_root / registry["artifact_contract"][
        "pre_execution_manifest_file"
    ]
    summary["pre_execution_manifest_sha256"] = (
        sha256_file(pre_execution_manifest_path)
        if pre_execution_manifest_path.is_file()
        else None
    )
    failure_status = any(
        isinstance(summary.get(section), dict)
        and summary[section].get("status") == "FAIL"
        for section in ("convergence", "invariance", "finest_level")
    )
    if summary.get("exact_groups_passed") is False:
        failure_status = True
    nonconditional_errors = [
        error
        for error in summary.get("errors", [])
        if error != "convergence disposition is ADDITIONAL_LEVEL_REQUIRED"
    ]
    if nonconditional_errors:
        failure_status = True
    conditional_dispositions = _conditional_sequence_records(registry, summary)
    summary["conditional_level_dispositions"] = conditional_dispositions
    passed = (
        summary.get("passed") is True
        and not conditional_dispositions
        and not failure_status
    )
    summary["passed"] = passed
    available_conditional_required = any(
        record["disposition"] == "EXECUTE"
        for record in conditional_dispositions
    )
    unavailable_conditional_required = any(
        record["disposition"] == "INCONCLUSIVE"
        for record in conditional_dispositions
    )
    summary["qualification_outcome"] = (
        "PASS"
        if passed
        else "FAIL"
        if failure_status
        else "ADDITIONAL_LEVEL_REQUIRED"
        if available_conditional_required
        else "INCONCLUSIVE"
        if unavailable_conditional_required
        else "FAIL"
    )
    if unavailable_conditional_required:
        errors = summary.setdefault("errors", [])
        errors.append(
            "a triggered three-dimensional conditional level is unavailable "
            "within the frozen one-node memory bound"
        )
    summary["disposition"] = {
        "fsr03_closed": passed,
        "fsr04_closed": False,
        "wp4_closed": False,
        "q2_closed": False,
    }
    summary["scope_exclusions"] = [
        "prescribed-angle maintenance does not close FSR-04",
        "two-phase and gas-sensitive behavior are outside this matrix",
        "higher-order and projected-force behavior are outside this matrix",
    ]
    write_json(output_root / "summary.json", summary)
    if conditional_trigger_record_path is None and not failure_status:
        trigger = build_conditional_trigger_record(
            registry, output_root / "summary.json"
        )
        _write_json_exclusive(
            output_root / "conditional_trigger_record.json", trigger
        )
    manifest_lines = []
    for path in sorted(output_root.rglob("*")):
        if path.is_file() and path.name != "manifest.sha256":
            manifest_lines.append(
                f"{sha256_file(path)}  {path.relative_to(output_root)}"
            )
    (output_root / "manifest.sha256").write_text(
        "\n".join(manifest_lines) + "\n", encoding="utf-8"
    )
    return summary


def expected_artifact_paths(
    registry: dict[str, Any], cases: Sequence[dict[str, Any]]
) -> list[str]:
    contract = registry["artifact_contract"]
    paths: set[str] = set()
    for case in cases:
        for name in contract["physical_case_files"]:
            paths.add(f"cases/{case['case_id']}/{name}")
    for invocation in exact_invocations(registry):
        directory = f"exact/{invocation['id']}"
        if invocation["mpi_ranks"] == 1:
            for name in contract["serial_exact_files"]:
                paths.add(f"{directory}/{name}")
        else:
            for rank in range(invocation["mpi_ranks"]):
                name = contract["mpi_exact_rank_file_template"].format(rank=rank)
                paths.add(f"{directory}/{name}")
            for name in contract["mpi_exact_common_files"]:
                paths.add(f"{directory}/{name}")
    paths.add(contract["pre_execution_manifest_file"])
    paths.add(contract["conditional_trigger_record_file"])
    paths.update(contract["summary_files"])
    return sorted(paths)


def _git_command(
    source_root: Path, arguments: Sequence[str], *, check: bool = True
) -> subprocess.CompletedProcess[bytes]:
    result = subprocess.run(
        ["git", "-C", str(source_root), *arguments],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if check and result.returncode != 0:
        raise MatrixError(
            f"source Git command failed: {' '.join(arguments)}"
        )
    return result


def validate_source_provenance(
    record: Any, *, declared_commit: str
) -> dict[str, Any]:
    _require_fields(record, {
        "source_root", "git_top_level", "head_commit", "head_tree",
        "head_detached", "worktree_clean", "status_sha256",
        "tracked_path_count", "tracked_source_digest_semantics",
        "tracked_source_sha256", "lfs",
    }, "source-provenance fields")
    _require_fields(
        record["lfs"],
        {"fsck_passed", "tracked_object_count", "missing_object_count",
         "pointer_checkout_count"},
        "source LFS fields",
    )
    source_root = Path(record["source_root"]).resolve()
    checks = (
        (source_root != Path(record["git_top_level"]).resolve(),
         "source root does not match its Git top level"),
        (source_root != REPOSITORY_ROOT.resolve(),
         "source root is not the running V3 repository"),
        (not re.fullmatch(r"[0-9a-f]{40}", declared_commit),
         "declared source commit must be lowercase 40-hex"),
        (record["head_commit"] != declared_commit,
         "source HEAD does not equal the declared commit"),
        (not re.fullmatch(r"[0-9a-f]{40}", str(record["head_tree"])),
         "source tree id is invalid"),
        (record["head_detached"] is not True,
         "qualification requires a detached source HEAD"),
        (record["worktree_clean"] is not True,
         "qualification requires a clean source worktree"),
        (not re.fullmatch(r"[0-9a-f]{64}", str(record["status_sha256"])),
         "source status digest is invalid"),
        (record["tracked_source_digest_semantics"] != "git_ls_files_stage_z_sha256"
         or not isinstance(record["tracked_path_count"], int)
         or isinstance(record["tracked_path_count"], bool)
         or record["tracked_path_count"] < 1
         or not re.fullmatch(r"[0-9a-f]{64}", str(record["tracked_source_sha256"])),
         "tracked source digest is invalid"),
    )
    for failed, message in checks:
        if failed:
            raise MatrixError(message)
    lfs = record["lfs"]
    counts = ("tracked_object_count", "missing_object_count", "pointer_checkout_count")
    if any(not isinstance(lfs[key], int) or isinstance(lfs[key], bool)
           or lfs[key] < 0 for key in counts):
        raise MatrixError("source LFS counts are invalid")
    if (
        lfs["fsck_passed"] is not True
        or lfs["tracked_object_count"] != EXPECTED_LFS_TRACKED_OBJECT_COUNT
        or lfs["missing_object_count"] != 0
        or lfs["pointer_checkout_count"] != 0
    ):
        raise MatrixError("qualification source LFS objects are not fully available")
    return record


def _lfs_inventory(
    source_root: Path,
    lfs_check: subprocess.CompletedProcess[bytes],
    lfs_list: subprocess.CompletedProcess[bytes],
) -> dict[str, Any]:
    if (
        lfs_check.returncode != 0
        or lfs_check.stderr.strip()
        or b"Git LFS fsck OK" not in lfs_check.stdout
        or lfs_list.returncode != 0
        or lfs_list.stderr.strip()
    ):
        raise MatrixError("qualification source LFS verification failed")
    lfs_entries: list[tuple[str, bytes]] = []
    for line in lfs_list.stdout.splitlines():
        if not line:
            continue
        fields = line.split(maxsplit=2)
        if len(fields) != 3 or not re.fullmatch(rb"[0-9a-f]{64}", fields[0]):
            raise MatrixError("qualification source LFS listing is malformed")
        lfs_entries.append((os.fsdecode(fields[2]), fields[1]))
    pointer_prefix = b"version https://git-lfs.github.com/spec/v1"
    pointer_checkouts = 0
    missing_objects = 0
    for relative, status_marker in lfs_entries:
        path = source_root / relative
        if status_marker != b"*":
            missing_objects += 1
        prefix = b""
        if path.is_file() and not path.is_symlink():
            with path.open("rb") as stream:
                prefix = stream.read(len(pointer_prefix))
        if (
            not path.is_file()
            or path.is_symlink()
            or prefix == pointer_prefix
            or status_marker != b"*"
        ):
            pointer_checkouts += 1
    return {
        "fsck_passed": True,
        "tracked_object_count": len(lfs_entries),
        "missing_object_count": missing_objects,
        "pointer_checkout_count": pointer_checkouts,
    }


def collect_source_provenance(source_root: Path) -> dict[str, Any]:
    source_root = source_root.resolve()

    def output(*arguments: str) -> bytes:
        return _git_command(source_root, list(arguments)).stdout

    top_level = Path(output("rev-parse", "--show-toplevel").decode().strip()).resolve()
    head_commit = output("rev-parse", "HEAD").decode("ascii").strip()
    head_tree = output("rev-parse", "HEAD^{tree}").decode("ascii").strip()
    status = output(
        "status", "--porcelain=v1", "-z", "--untracked-files=all"
    )
    detached = _git_command(
        source_root, ["symbolic-ref", "--quiet", "HEAD"], check=False
    )
    if detached.returncode not in {0, 1}:
        raise MatrixError("unable to determine whether source HEAD is detached")
    tracked = output("ls-files", "--stage", "-z")
    record = {
        "source_root": str(source_root),
        "git_top_level": str(top_level),
        "head_commit": head_commit,
        "head_tree": head_tree,
        "head_detached": detached.returncode == 1,
        "worktree_clean": not status,
        "status_sha256": hashlib.sha256(status).hexdigest(),
        "tracked_path_count": len([item for item in tracked.split(b"\0") if item]),
        "tracked_source_digest_semantics": "git_ls_files_stage_z_sha256",
        "tracked_source_sha256": hashlib.sha256(tracked).hexdigest(),
        "lfs": _lfs_inventory(
            source_root,
            _git_command(source_root, ["lfs", "fsck"], check=False),
            _git_command(source_root, ["lfs", "ls-files", "-l"], check=False),
        ),
    }
    return validate_source_provenance(record, declared_commit=head_commit)


def _hash_binding(path: Path, context: str) -> dict[str, str]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise MatrixError(f"{context} hash input is not a file: {resolved}")
    return {"path": str(resolved), "sha256": sha256_file(resolved)}


def _validate_hash_map(
    values: dict[str, Path], expected: set[str], context: str
) -> dict[str, dict[str, str]]:
    if set(values) != expected:
        raise MatrixError(
            f"{context} hash keys changed: expected={sorted(expected)}, "
            f"observed={sorted(values)}"
        )
    return {
        key: _hash_binding(values[key], f"{context} {key!r}")
        for key in sorted(values)
    }


def _conditional_trigger_binding(
    registry: dict[str, Any], path: Path | None
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    if path is None:
        return None, None
    resolved = path.resolve()
    record = load_conditional_trigger_record(registry, resolved)
    prior_path = resolved.parent / record["prior_analysis_file"]
    return record, {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "prior_analysis_path": str(prior_path.resolve()),
        "prior_analysis_sha256": sha256_file(prior_path),
        "sequence_ids": [item["sequence_id"] for item in record["sequences"]],
    }


def build_pre_execution_manifest(
    registry: dict[str, Any],
    cases: Sequence[dict[str, Any]],
    *,
    source_commit: str,
    source_root: Path,
    compiler: Path,
    mpi: Path,
    dependencies: dict[str, Path],
    binaries: dict[str, Path],
    solver: Path,
    conditional_trigger_record_path: Path | None,
) -> dict[str, Any]:
    if not re.fullmatch(r"[0-9a-f]{40}", source_commit):
        raise MatrixError("source commit must be lowercase 40-hex")
    _exact_invocation_watchdog(registry.get("resources"))
    source = collect_source_provenance(source_root)
    validate_source_provenance(source, declared_commit=source_commit)
    conditional_record, conditional_binding = _conditional_trigger_binding(
        registry, conditional_trigger_record_path
    )
    expected_cases = expand_cases(
        registry, conditional_trigger_record=conditional_record
    )
    if list(cases) != expected_cases:
        raise MatrixError("pre-execution physical case set changed")
    validate_case_resources(registry, cases)
    provenance = registry["provenance_contract"]
    dependency_keys = set(provenance["required_dependency_keys"])
    binary_keys = set(provenance["required_binary_keys"])
    artifacts = expected_artifact_paths(registry, cases)
    artifact_payload = json.dumps(
        artifacts, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    tests = sorted(
        {
            test
            for invocation in exact_invocations(registry)
            for test in invocation["tests"]
        }
    )
    resources = sorted(
        {study["resource_profile"] for study in registry["studies"]}
        | {
            invocation["resource_profile"]
            for invocation in exact_invocations(registry)
        }
    )
    return {
        "schema_version": provenance["manifest_schema_version"],
        "matrix_id": registry["matrix_id"],
        "matrix_path": str(DEFAULT_REGISTRY),
        "matrix_sha256": sha256_file(DEFAULT_REGISTRY),
        "runner_path": str(SCRIPT_PATH),
        "runner_sha256": sha256_file(SCRIPT_PATH),
        "physical_runner_path": str(PHYSICAL_RUNNER),
        "physical_runner_sha256": sha256_file(PHYSICAL_RUNNER),
        "parent_matrix_sha256": sha256_file(PARENT_REGISTRY_PATH),
        "parent_runner_sha256": sha256_file(PARENT_RUNNER_PATH),
        "source_commit": source_commit,
        "source": source,
        "compiler": _hash_binding(compiler, "compiler"),
        "mpi": _hash_binding(mpi, "MPI"),
        "dependencies": _validate_hash_map(
            dependencies, dependency_keys, "dependency"
        ),
        "binaries": _validate_hash_map(binaries, binary_keys, "binary"),
        "solver": _hash_binding(solver, "solver"),
        "conditional_trigger": conditional_binding,
        "exact_invocation_lifecycle": copy.deepcopy(
            registry["resources"]["exact_invocation_lifecycle"]
        ),
        "physical_case_count": len(cases),
        "physical_case_ids": [case["case_id"] for case in cases],
        "physical_case_digests": [case["case_digest"] for case in cases],
        "maximum_estimated_memory_mib": max(
            (case["estimated_memory_mib"] for case in cases), default=0
        ),
        "required_test_union": tests,
        "required_resource_union": resources,
        "expected_artifact_union": artifacts,
        "expected_artifact_count": len(artifacts),
        "expected_artifact_union_sha256": hashlib.sha256(
            artifact_payload
        ).hexdigest(),
        "conditional_level_by_dimension": copy.deepcopy(
            registry["refinement"]["conditional_level_by_dimension"]
        ),
        "numerical_execution_performed": False,
    }


def revalidate_pre_execution_manifest(
    manifest_path: Path,
    registry: dict[str, Any],
    cases: Sequence[dict[str, Any]],
    *,
    output_root: Path,
    source_commit: str,
    source_root: Path,
    compiler: Path,
    mpi: Path,
    dependencies: dict[str, Path],
    binaries: dict[str, Path],
    solver: Path,
    conditional_trigger_record_path: Path | None,
) -> dict[str, Any]:
    resolved = manifest_path.resolve()
    expected_name = registry["artifact_contract"]["pre_execution_manifest_file"]
    declared_path = output_root.resolve() / expected_name
    if resolved != declared_path or not resolved.is_file():
        raise MatrixError(
            "pre-execution manifest is not the declared output-root artifact"
        )
    observed = read_json(resolved)
    expected = build_pre_execution_manifest(
        registry,
        cases,
        source_commit=source_commit,
        source_root=source_root,
        compiler=compiler,
        mpi=mpi,
        dependencies=dependencies,
        binaries=binaries,
        solver=solver,
        conditional_trigger_record_path=conditional_trigger_record_path,
    )
    if observed != expected:
        raise MatrixError("pre-execution manifest drift detected")
    return expected


def _parse_binding(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("hash binding must be KEY=PATH")
    key, path = value.split("=", 1)
    if not key or not path:
        raise argparse.ArgumentTypeError("hash binding must be KEY=PATH")
    return key, Path(path)


def _binding_map(
    values: Sequence[tuple[str, Path]], context: str
) -> dict[str, Path]:
    result = dict(values)
    if len(result) != len(values):
        raise MatrixError(f"duplicate {context} keys")
    return result


def _write_bytes_exclusive(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError as error:
        raise MatrixError(f"refusing to replace immutable artifact: {path}") from error


def _write_json_exclusive(path: Path, value: Any) -> None:
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    _write_bytes_exclusive(path, payload)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--list-cases", action="store_true")
    parser.add_argument("--conditional-trigger-record", type=Path)
    parser.add_argument("--study", action="append", default=[])
    parser.add_argument("--case-index", type=int)
    parser.add_argument("--shard-index", type=int)
    parser.add_argument("--shard-count", type=int)
    parser.add_argument("--dry-manifest", action="store_true")
    parser.add_argument("--source-commit")
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--compiler", type=Path)
    parser.add_argument("--mpi", type=Path)
    parser.add_argument(
        "--dependency", type=_parse_binding, action="append", default=[]
    )
    parser.add_argument("--binary", type=_parse_binding, action="append", default=[])
    parser.add_argument("--run-physical", action="store_true")
    parser.add_argument("--run-exact", action="store_true")
    parser.add_argument("--analyze", action="store_true")
    parser.add_argument("--solver", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--results-root", type=Path, action="append", default=[])
    parser.add_argument("--exact-summary", type=Path)
    parser.add_argument(
        "--mpi-launcher-mode", choices=("mpiexec", "srun"), default="mpiexec"
    )
    parser.add_argument("--rerun", action="store_true")
    return parser


def _require_pre_execution_inputs(args: argparse.Namespace) -> None:
    missing = [
        name
        for name in ("source_commit", "source_root", "compiler", "mpi", "solver")
        if getattr(args, name) is None
    ]
    if missing:
        raise MatrixError(f"pre-execution manifest is missing inputs: {missing}")


def main(arguments: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(arguments)
    registry = load_registry(args.registry)
    conditional_trigger = (
        load_conditional_trigger_record(registry, args.conditional_trigger_record)
        if args.conditional_trigger_record is not None
        else None
    )
    cases = expand_cases(
        registry, conditional_trigger_record=conditional_trigger
    )
    selected = select_cases(
        cases,
        studies=args.study,
        case_index=args.case_index,
        shard_index=args.shard_index,
        shard_count=args.shard_count,
    )
    numerical_actions = args.run_physical or args.run_exact or args.analyze
    if args.validate_only:
        if args.list_cases or args.dry_manifest or numerical_actions:
            raise MatrixError("--validate-only cannot be combined with other actions")
        print(
            json.dumps(
                {
                    "matrix_id": registry["matrix_id"],
                    "status": registry["status"],
                    "exact_category_count": len(registry["exact_groups"]),
                    "exact_invocation_count": len(exact_invocations(registry)),
                    "study_count": len(registry["studies"]),
                    "physical_case_count": len(cases),
                    "conditional_sequence_count": (
                        len(conditional_trigger["sequences"])
                        if conditional_trigger is not None
                        else 0
                    ),
                    "maximum_estimated_memory_mib": max(
                        case["estimated_memory_mib"] for case in cases
                    ),
                    "outcome": "PASS",
                },
                sort_keys=True,
            )
        )
        return 0
    if args.list_cases:
        if args.dry_manifest or numerical_actions:
            raise MatrixError("--list-cases cannot be combined with other actions")
        for case in selected:
            print(json.dumps(case, sort_keys=True))
        return 0

    if not numerical_actions and not args.dry_manifest:
        raise MatrixError(
            "select --validate-only, --list-cases, --dry-manifest, or an execution action"
        )
    if args.output_root is None:
        raise MatrixError("manifest preparation and execution require --output-root")
    if args.rerun:
        raise MatrixError("immutable V3 evidence does not permit --rerun")
    _require_pre_execution_inputs(args)
    dependencies = _binding_map(args.dependency, "dependency")
    binaries = _binding_map(args.binary, "binary")
    output_root = args.output_root.resolve()
    manifest_options = {
        "source_commit": args.source_commit,
        "source_root": args.source_root,
        "compiler": args.compiler,
        "mpi": args.mpi,
        "dependencies": dependencies,
        "binaries": binaries,
        "solver": args.solver,
        "conditional_trigger_record_path": args.conditional_trigger_record,
    }
    manifest = build_pre_execution_manifest(
        registry,
        cases,
        **manifest_options,
    )
    try:
        output_root.mkdir(parents=True, exist_ok=False)
    except FileExistsError as error:
        raise MatrixError(
            f"immutable output root already exists: {output_root}"
        ) from error
    if args.conditional_trigger_record is not None:
        trigger_artifact = output_root / registry["artifact_contract"][
            "conditional_trigger_record_file"
        ]
        _write_bytes_exclusive(
            trigger_artifact, args.conditional_trigger_record.resolve().read_bytes()
        )
    manifest_path = output_root / registry["artifact_contract"][
        "pre_execution_manifest_file"
    ]
    _write_json_exclusive(manifest_path, manifest)
    if not numerical_actions:
        print(
            json.dumps(
                {
                    "matrix_id": registry["matrix_id"],
                    "physical_case_count": len(cases),
                    "expected_artifact_count": manifest["expected_artifact_count"],
                    "maximum_estimated_memory_mib": manifest[
                        "maximum_estimated_memory_mib"
                    ],
                    "manifest": str(manifest_path),
                    "outcome": "PASS",
                },
                sort_keys=True,
            )
        )
        return 0

    def revalidate() -> None:
        revalidate_pre_execution_manifest(
            manifest_path,
            registry,
            cases,
            output_root=output_root,
            **manifest_options,
        )

    if args.run_physical:
        run_physical_cases(
            registry,
            selected,
            solver=args.solver,
            output_root=output_root,
            rerun=args.rerun,
            pre_execution_guard=revalidate,
        )
    exact_summary = None
    if args.run_exact:
        exact_summary = run_exact_groups(
            registry,
            binaries=binaries,
            mpi=args.mpi.resolve(),
            mpi_launcher_mode=args.mpi_launcher_mode,
            output_root=output_root,
            pre_execution_guard=revalidate,
        )
        if exact_summary["passed"] is not True and not args.analyze:
            return 1
    if args.analyze:
        revalidate()
        roots = args.results_root or [output_root]
        summary = analyze_evidence(
            registry,
            roots=[path.resolve() for path in roots],
            output_root=output_root,
            conditional_trigger_record_path=args.conditional_trigger_record,
            exact_summary_path=(
                args.exact_summary.resolve()
                if args.exact_summary is not None
                else output_root / "exact_summary.json"
                if exact_summary is not None
                else None
            ),
        )
        return 0 if summary["passed"] else 1
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (MatrixError, OSError, RuntimeError, KeyError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)

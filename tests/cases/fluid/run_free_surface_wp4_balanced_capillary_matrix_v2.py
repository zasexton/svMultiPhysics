#!/usr/bin/env python3
"""Run and analyze the frozen WP-4 balanced-capillary V2 matrix."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Sequence


SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIRECTORY = SCRIPT_PATH.parent
DEFAULT_REGISTRY = SCRIPT_PATH.with_name(
    "free_surface_wp4_balanced_capillary_matrix_v2.json"
)
PHYSICAL_RUNNER = (
    SCRIPT_DIRECTORY
    / "open_vessel_free_surface/run_test05_velocity_growth_smoke.py"
)
EXPECTED_REGISTRY_SHA256 = (
    "7605f4458191112bf0f03c38299b9b46838a11e9dcbf61c7196fecb0f89d7918"
)
EXPECTED_PHYSICAL_RUNNER_SHA256 = (
    "6891f0d1c2bc1c6c049111b2e16cc1584fe32e75489fbc62e3db3749c8474177"
)
EXPECTED_MATRIX_ID = "free_surface_wp4_balanced_capillary_v2"
EXPECTED_STATUS = "FROZEN_BEFORE_EXECUTION"
EXPECTED_WORK_PACKAGE = "WP-4"
SUPPORTED_CASE_DIMENSIONS = {
    "droplet2d": 2,
    "sphere3d": 3,
    "sessile2d": 2,
    "sessile3d": 3,
}
SUPPORTED_INITIALIZATIONS = {
    "sampled_analytic",
    "discrete_energy_minimized",
}
SUPPORTED_REFINEMENT_AXES = {
    "resolution",
    "phi_scale",
    "physical_scale",
    "time_step",
    "reinitialization_cadence",
}
SUPPORTED_AXES = {
    "active_domain",
    "contact_angle",
    "offset_h",
    "wall",
}
REQUIRED_ANGLES = {30, 60, 90, 120, 150}
REQUIRED_ACTIVE_DOMAINS = {"LevelSetNegative", "LevelSetPositive"}
REQUIRED_WALLS = {
    2: {"wall_bottom", "wall_left", "wall_right", "wall_top"},
    3: {
        "wall_left",
        "wall_right",
        "wall_bottom",
        "wall_top",
        "wall_front",
        "wall_back",
    },
}
REQUIRED_MAIN_STUDIES = {
    "closed_circle_sampled_analytic",
    "closed_circle_discrete_minimizer",
    "closed_sphere_sampled_analytic",
    "closed_sphere_discrete_minimizer",
    "sessile_caps_2d_sampled_analytic",
    "sessile_caps_2d_discrete_minimizer",
    "sessile_caps_3d_sampled_analytic",
    "sessile_caps_3d_discrete_minimizer",
}
REQUIRED_METRICS = {
    "pressure_jump_relative_error",
    "pressure_space_relative_distance",
    "conservative_balance_normalized_imbalance",
    "parasitic_capillary_number",
}
REQUIRED_SESSILE_METRICS = {
    "contact_angle_absolute_error_degrees",
    "base_radius_relative_error",
    "apex_height_relative_error",
}
REQUIRED_INVARIANCE_METRICS = (
    REQUIRED_METRICS | REQUIRED_SESSILE_METRICS |
    {"kinetic_energy_proxy", "liquid_volume_relative_error"}
)
SUPPORTED_EXACT_PROPERTY_COMPARISONS = {
    "at_least",
    "at_most",
    "equal",
    "finite",
    "scaled_roundoff",
}
GTEST_CASE_METADATA_KEYS = {
    "classname",
    "failures",
    "file",
    "line",
    "name",
    "result",
    "status",
    "time",
    "timestamp",
}


class MatrixError(ValueError):
    """Raised when matrix input or evidence violates the frozen contract."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise MatrixError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def read_json(path: Path) -> Any:
    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_strict_object,
        )
    except json.JSONDecodeError as error:
        raise MatrixError(f"invalid JSON in {path}: {error}") from error


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def finite_number(value: Any, context: str, *, positive: bool = False) -> float:
    if (isinstance(value, bool) or not isinstance(value, (int, float)) or
            not math.isfinite(float(value))):
        raise MatrixError(f"{context} must be finite")
    result = float(value)
    if positive and result <= 0.0:
        raise MatrixError(f"{context} must be positive")
    return result


def nonempty_string(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise MatrixError(f"{context} must be a nonempty string")
    return value


def _unique_strings(values: Any, context: str) -> list[str]:
    if (not isinstance(values, list) or not values or
            any(not isinstance(value, str) or not value for value in values)):
        raise MatrixError(f"{context} must contain nonempty strings")
    if len(set(values)) != len(values):
        raise MatrixError(f"{context} contains duplicates")
    return list(values)


def _validate_resources(resources: Any) -> None:
    if not isinstance(resources, dict):
        raise MatrixError("matrix resources are missing")
    if resources.get("partition") != "amarsden":
        raise MatrixError("WP-4 jobs must use the amarsden partition")
    if resources.get("maximum_concurrent_nodes") != 3:
        raise MatrixError("WP-4 maximum concurrent node count changed")
    if resources.get("maximum_total_memory_mib") != 40960:
        raise MatrixError("WP-4 maximum total memory changed")
    if resources.get("nodes_per_case") != 1:
        raise MatrixError("each WP-4 case must use one node")
    memory = resources.get("memory_mib_per_case")
    if (not isinstance(memory, int) or isinstance(memory, bool) or
            memory <= 0 or memory > 40960):
        raise MatrixError("per-case memory must be in (0, 40960] MiB")


def _validate_refinement(refinement: Any) -> None:
    if not isinstance(refinement, dict):
        raise MatrixError("refinement contract is missing")
    if refinement.get("spatial_levels_cells_per_radius") != [8, 16, 32]:
        raise MatrixError("spatial levels must be R/dx = 8, 16, 32")
    if refinement.get("conditional_spatial_level_cells_per_radius") != 64:
        raise MatrixError("conditional spatial level must be R/dx = 64")
    if finite_number(
            refinement.get("uniform_ratio"),
            "uniform refinement ratio", positive=True) != 2.0:
        raise MatrixError("uniform refinement ratio must be two")
    if refinement.get("nonmonotone_three_level_disposition") != (
            "ADDITIONAL_LEVEL_REQUIRED"):
        raise MatrixError("three-level nonmonotone disposition changed")
    if refinement.get("nonasymptotic_four_level_disposition") != "FAIL":
        raise MatrixError("four-level nonasymptotic disposition changed")


def _validate_gates(gates: Any) -> None:
    if not isinstance(gates, dict):
        raise MatrixError("matrix gates are missing")
    if gates.get("exact_flat_scaled_residual_factor") != 256:
        raise MatrixError("exact scaled-roundoff factor changed")
    finest = gates.get("finest_level")
    if not isinstance(finest, dict):
        raise MatrixError("finest-level gates are missing")
    required_finest = {
        "pressure_jump_relative_error",
        "contact_angle_absolute_error_degrees",
        "base_radius_relative_error",
        "apex_height_relative_error",
        "liquid_volume_relative_error",
        "parasitic_capillary_number",
        "kinetic_energy_proxy",
    }
    if set(finest) != required_finest:
        raise MatrixError("finest-level metric gates changed")
    for metric, limit in finest.items():
        finite_number(limit, f"finest-level gate {metric!r}", positive=True)
    convergence = gates.get("convergence")
    if (not isinstance(convergence, dict) or
            not REQUIRED_METRICS.issubset(convergence)):
        raise MatrixError("convergence gates are incomplete")
    invariance = gates.get("invariance")
    if (not isinstance(invariance, dict) or
            set(invariance) != {"phi_scale", "physical_scale"}):
        raise MatrixError("invariance gates are incomplete")
    for axis, metric_gates in invariance.items():
        if (not isinstance(metric_gates, dict) or
                set(metric_gates) != REQUIRED_INVARIANCE_METRICS):
            raise MatrixError(f"{axis} invariance metric gates changed")
        for metric, limits in metric_gates.items():
            if not isinstance(limits, dict):
                raise MatrixError(f"{axis} gate {metric!r} is invalid")
            finite_number(
                limits.get("maximum_value"),
                f"{axis} gate {metric!r} maximum value", positive=True)
            finite_number(
                limits.get("maximum_spread"),
                f"{axis} gate {metric!r} maximum spread", positive=True)


def _validate_exact_groups(groups: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(groups, list) or not groups:
        raise MatrixError("exact groups are missing")
    result: dict[str, dict[str, Any]] = {}
    supported_binaries = {
        "application",
        "application_mpi",
        "geometry",
        "level_set",
        "level_set_mpi",
    }
    for index, group in enumerate(groups):
        if not isinstance(group, dict):
            raise MatrixError(f"exact group {index} must be an object")
        group_id = nonempty_string(group.get("id"), f"exact group {index} id")
        if group_id in result:
            raise MatrixError(f"duplicate exact group id {group_id!r}")
        if group.get("binary") not in supported_binaries:
            raise MatrixError(f"exact group {group_id!r} has unknown binary")
        ranks = group.get("mpi_ranks")
        if not isinstance(ranks, int) or isinstance(ranks, bool) or ranks < 1:
            raise MatrixError(f"exact group {group_id!r} has invalid rank count")
        tests = _unique_strings(
            group.get("tests"), f"exact group {group_id!r} tests")
        if any(test.startswith("DISABLED_") or ".DISABLED_" in test
               for test in tests):
            raise MatrixError("disabled tests cannot qualify WP-4")
        count = group.get("required_matrix_case_count")
        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            raise MatrixError(
                f"exact group {group_id!r} has invalid matrix case count")
        require_identity = group.get("require_identical_rank_properties")
        if require_identity is not (ranks > 1):
            raise MatrixError(
                f"exact group {group_id!r} rank-property policy changed")
        gates = group.get("property_gates")
        if not isinstance(gates, dict) or set(gates) != set(tests):
            raise MatrixError(
                f"exact group {group_id!r} must gate every exact test")
        has_matrix_count_gate = count == 0
        for test in tests:
            test_gates = gates[test]
            if not isinstance(test_gates, list) or not test_gates:
                raise MatrixError(
                    f"exact test {test!r} has no quantitative gates")
            seen: set[tuple[str, str]] = set()
            for gate_index, gate in enumerate(test_gates):
                context = f"exact test {test!r} gate {gate_index}"
                if not isinstance(gate, dict):
                    raise MatrixError(f"{context} must be an object")
                property_name = nonempty_string(
                    gate.get("property"), f"{context} property")
                comparison = gate.get("comparison")
                if comparison not in SUPPORTED_EXACT_PROPERTY_COMPARISONS:
                    raise MatrixError(f"{context} comparison is invalid")
                key = (property_name, comparison)
                if key in seen:
                    raise MatrixError(f"{context} is duplicated")
                seen.add(key)
                if comparison in {"equal", "at_least", "at_most"}:
                    finite_number(gate.get("expected"), f"{context} expected")
                elif comparison == "scaled_roundoff":
                    finite_number(
                        gate.get("scale"), f"{context} scale", positive=True)
                elif "expected" in gate or "scale" in gate:
                    raise MatrixError(f"{context} has an unused limit")
                if (property_name.endswith("_matrix_case_count") and
                        comparison == "equal" and
                        gate.get("expected") == count):
                    has_matrix_count_gate = True
        if not has_matrix_count_gate:
            raise MatrixError(
                f"exact group {group_id!r} does not enforce its matrix count")
        result[group_id] = group
    flat_test = (
        "LevelSetCurvatureProjection."
        "KinematicAreaGradientIsRoundoffBalancedForAffineFlatInterface"
    )
    all_tests = {
        test for group in result.values() for test in group["tests"]
    }
    if flat_test not in all_tests:
        raise MatrixError("the exact affine-flat area-gradient test is missing")
    return result


def _validate_literature(
        adaptations: Any,
        exact_groups: dict[str, dict[str, Any]],
        studies: dict[str, dict[str, Any]],
) -> None:
    if not isinstance(adaptations, list) or not adaptations:
        raise MatrixError("literature adaptations are missing")
    records: dict[str, dict[str, Any]] = {}
    for index, adaptation in enumerate(adaptations):
        if not isinstance(adaptation, dict):
            raise MatrixError(f"literature adaptation {index} is invalid")
        adaptation_id = nonempty_string(
            adaptation.get("id"), f"literature adaptation {index} id")
        if adaptation_id in records:
            raise MatrixError(f"duplicate literature adaptation {adaptation_id!r}")
        source = adaptation.get("source")
        if (not isinstance(source, dict) or
                not isinstance(source.get("doi"), str) or
                not source["doi"]):
            raise MatrixError(
                f"literature adaptation {adaptation_id!r} lacks a DOI")
        limitations = _unique_strings(
            adaptation.get("limitations"),
            f"literature adaptation {adaptation_id!r} limitations",
        )
        if not limitations:
            raise MatrixError("literature limitations cannot be empty")
        group_id = adaptation.get("adapted_evidence_group")
        study_id = adaptation.get("adapted_study")
        if (group_id is None) == (study_id is None):
            raise MatrixError(
                f"literature adaptation {adaptation_id!r} must name exactly "
                "one exact group or physical study")
        if group_id is not None:
            if group_id not in exact_groups:
                raise MatrixError(
                    f"literature adaptation {adaptation_id!r} names an "
                    "unknown exact group")
            test = adaptation.get("adapted_test")
            if test not in exact_groups[group_id]["tests"]:
                raise MatrixError(
                    f"literature adaptation {adaptation_id!r} names an "
                    "unknown exact test")
        if study_id is not None and study_id not in studies:
            raise MatrixError(
                f"literature adaptation {adaptation_id!r} names an "
                "unknown study")
        records[adaptation_id] = adaptation
    referenced = {
        study.get("literature_adaptation")
        for study in studies.values()
        if "literature_adaptation" in study
    }
    physical_ids = {
        record["id"] for record in adaptations if "adapted_study" in record
    }
    if referenced != physical_ids:
        raise MatrixError("physical literature-adaptation links are incomplete")


def _validate_studies(studies_value: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(studies_value, list) or not studies_value:
        raise MatrixError("physical studies are missing")
    studies: dict[str, dict[str, Any]] = {}
    for index, study in enumerate(studies_value):
        if not isinstance(study, dict):
            raise MatrixError(f"study {index} must be an object")
        study_id = nonempty_string(study.get("id"), f"study {index} id")
        if study_id in studies:
            raise MatrixError(f"duplicate study id {study_id!r}")
        case_name = study.get("case")
        if case_name not in SUPPORTED_CASE_DIMENSIONS:
            raise MatrixError(f"study {study_id!r} has unsupported case")
        dimension = study.get("dimension")
        if dimension != SUPPORTED_CASE_DIMENSIONS[case_name]:
            raise MatrixError(f"study {study_id!r} dimension is inconsistent")
        if study.get("initialization") not in SUPPORTED_INITIALIZATIONS:
            raise MatrixError(f"study {study_id!r} initialization is invalid")
        refinement_axis = study.get("refinement_axis")
        if refinement_axis not in SUPPORTED_REFINEMENT_AXES:
            raise MatrixError(f"study {study_id!r} refinement axis is invalid")
        if refinement_axis == "resolution":
            finite_number(study.get("radius"), "study radius", positive=True)
            if "refinement_levels" in study:
                raise MatrixError(
                    f"resolution study {study_id!r} cannot replace frozen levels")
        elif refinement_axis == "physical_scale":
            levels = study.get("refinement_levels")
            if not isinstance(levels, list) or len(levels) != 3:
                raise MatrixError(
                    f"physical-scale study {study_id!r} needs three levels")
            labels: set[str] = set()
            for level in levels:
                if not isinstance(level, dict):
                    raise MatrixError("physical-scale levels must be objects")
                label = nonempty_string(level.get("label"), "scale label")
                if label in labels:
                    raise MatrixError("physical-scale labels must be unique")
                labels.add(label)
                finite_number(level.get("radius"), "scale radius", positive=True)
                finite_number(
                    level.get("surface_tension"),
                    "scale surface tension", positive=True)
            cells = study.get("cells_per_radius")
            if not isinstance(cells, int) or isinstance(cells, bool) or cells < 2:
                raise MatrixError("physical-scale cells per radius are invalid")
        else:
            levels = study.get("refinement_levels")
            if not isinstance(levels, list) or len(levels) != 3:
                raise MatrixError(
                    f"study {study_id!r} needs exactly three refinement levels")
            for level in levels:
                finite_number(level, f"study {study_id!r} refinement level")
            resolution = study.get("resolution")
            if (not isinstance(resolution, int) or isinstance(resolution, bool) or
                    resolution < 2):
                raise MatrixError(f"study {study_id!r} resolution is invalid")
        axes = study.get("axes")
        if not isinstance(axes, dict) or not axes:
            raise MatrixError(f"study {study_id!r} axes are missing")
        unknown_axes = set(axes) - SUPPORTED_AXES
        if unknown_axes:
            raise MatrixError(
                f"study {study_id!r} has unsupported axes {sorted(unknown_axes)}")
        for axis, values in axes.items():
            if not isinstance(values, list) or not values:
                raise MatrixError(
                    f"study {study_id!r} axis {axis!r} is empty")
        metrics = set(_unique_strings(
            study.get("metrics"), f"study {study_id!r} metrics"))
        if study_id in REQUIRED_MAIN_STUDIES:
            missing = REQUIRED_METRICS - metrics
            if case_name.startswith("sessile"):
                missing |= REQUIRED_SESSILE_METRICS - metrics
            if missing:
                raise MatrixError(
                    f"main study {study_id!r} lacks metrics {sorted(missing)}")
        studies[study_id] = study

    if not REQUIRED_MAIN_STUDIES.issubset(studies):
        raise MatrixError("the sampled and minimized circle/sphere/cap studies are incomplete")
    for dimension in (2, 3):
        for initialization in SUPPORTED_INITIALIZATIONS:
            study_id = (
                f"sessile_caps_{dimension}d_"
                + ("sampled_analytic" if initialization == "sampled_analytic"
                   else "discrete_minimizer")
            )
            study = studies[study_id]
            axes = study["axes"]
            if set(axes.get("contact_angle", [])) != REQUIRED_ANGLES:
                raise MatrixError(f"study {study_id!r} lacks the five angles")
            if set(axes.get("wall", [])) != REQUIRED_WALLS[dimension]:
                raise MatrixError(f"study {study_id!r} lacks wall rotations")
            if set(axes.get("active_domain", [])) != REQUIRED_ACTIVE_DOMAINS:
                raise MatrixError(f"study {study_id!r} lacks phase signs")
    if any(study["case"] == "capillaryarc2d" for study in studies.values()):
        raise MatrixError("a curved arc cannot stand in for the planar force test")
    return studies


def validate_registry(registry: Any, path: Path) -> dict[str, Any]:
    if not isinstance(registry, dict):
        raise MatrixError("registry root must be an object")
    if sha256_file(path) != EXPECTED_REGISTRY_SHA256:
        raise MatrixError("WP-4 V2 frozen registry bytes changed")
    if registry.get("schema_version") != 2:
        raise MatrixError("WP-4 V2 schema version changed")
    if registry.get("matrix_id") != EXPECTED_MATRIX_ID:
        raise MatrixError("WP-4 V2 matrix id changed")
    if registry.get("status") != EXPECTED_STATUS:
        raise MatrixError("WP-4 V2 matrix is not frozen before execution")
    if registry.get("work_package") != EXPECTED_WORK_PACKAGE:
        raise MatrixError("WP-4 V2 work-package id changed")
    model = registry.get("model_envelope")
    if (not isinstance(model, dict) or
            model.get("capillary_force") !=
            "kinematic_area_gradient_energy_traction" or
            model.get("force_projection_applied") is not False or
            model.get("two_phase_claimed") is not False or
            model.get("higher_order_claimed") is not False):
        raise MatrixError("WP-4 V2 model envelope changed")
    report_metrics = set(_unique_strings(
        registry.get("required_report_metrics"),
        "required report metrics",
    ))
    if report_metrics != {
            "kinetic_energy_proxy", "liquid_volume_relative_error"}:
        raise MatrixError("required physical report metrics changed")
    _validate_resources(registry.get("resources"))
    _validate_refinement(registry.get("refinement"))
    _validate_gates(registry.get("gates"))
    exact_groups = _validate_exact_groups(registry.get("exact_groups"))
    studies = _validate_studies(registry.get("studies"))
    invariance_gates = registry["gates"]["invariance"]
    for study in studies.values():
        axis = study["refinement_axis"]
        if axis not in invariance_gates:
            continue
        metrics = set(study["metrics"]) | report_metrics
        if metrics != set(invariance_gates[axis]):
            raise MatrixError(
                f"study {study['id']!r} invariance metrics changed")
    _validate_literature(
        registry.get("literature_adaptations"), exact_groups, studies)
    common = registry.get("common_runner_arguments")
    if not isinstance(common, list) or any(not isinstance(value, str)
                                           for value in common):
        raise MatrixError("common runner arguments are invalid")
    required_arguments = {
        "--capillary-force-form",
        "kinematic_area_gradient_traction",
        "--cut-cell-pressure-stabilization-policy",
        "incremental",
        "--defer-static-physical-gates-to-matrix",
        "--require-free-surface-energy-history",
    }
    if not required_arguments.issubset(common):
        raise MatrixError("common balanced-capillary runner arguments changed")
    if sha256_file(PHYSICAL_RUNNER) != EXPECTED_PHYSICAL_RUNNER_SHA256:
        raise MatrixError("physical runner changed after WP-4 V2 freeze")
    return registry


def load_registry(path: Path = DEFAULT_REGISTRY) -> dict[str, Any]:
    resolved = path.resolve()
    return validate_registry(read_json(resolved), resolved)


def _argument_value(arguments: Sequence[str], option: str) -> str | None:
    result: str | None = None
    for index, value in enumerate(arguments[:-1]):
        if value == option:
            result = arguments[index + 1]
    return result


def _level_records(
        registry: dict[str, Any],
        study: dict[str, Any],
        include_conditional_level: bool,
) -> list[dict[str, Any]]:
    axis = study["refinement_axis"]
    if axis == "resolution":
        levels = list(
            registry["refinement"]["spatial_levels_cells_per_radius"])
        if include_conditional_level:
            levels.append(
                registry["refinement"]
                ["conditional_spatial_level_cells_per_radius"])
        radius = float(study["radius"])
        first = float(levels[0])
        base_resolution = int(math.ceil(first / radius - 1.0e-14))
        return [
            {
                "label": f"rdx_{int(level)}",
                "value": float(level),
                "resolution": int(
                    base_resolution * round(float(level) / first)),
                "radius": radius,
            }
            for level in levels
        ]
    if axis == "physical_scale":
        cells = int(study["cells_per_radius"])
        return [
            {
                **level,
                "value": index,
                "resolution": int(math.ceil(
                    cells / float(level["radius"]) - 1.0e-14)),
            }
            for index, level in enumerate(study["refinement_levels"])
        ]
    resolution = int(study["resolution"])
    records = []
    for value in study["refinement_levels"]:
        label_value = str(value).replace("-", "m").replace(".", "p")
        records.append({
            "label": f"{axis}_{label_value}",
            "value": value,
            "resolution": resolution,
            "radius": float(study["radius"]),
        })
    return records


def _axis_combinations(study: dict[str, Any]) -> Iterable[dict[str, Any]]:
    names = list(study["axes"])
    values = [study["axes"][name] for name in names]
    for combination in itertools.product(*values):
        yield dict(zip(names, combination))


def _canonical_digest(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def expand_cases(
        registry: dict[str, Any],
        *,
        include_conditional_level: bool = False,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for study_index, study in enumerate(registry["studies"]):
        levels = _level_records(registry, study, include_conditional_level)
        combinations = list(_axis_combinations(study))
        for axes in combinations:
            for level in levels:
                resolution = int(level["resolution"])
                radius = float(level.get("radius", study.get("radius", 0.0)))
                canonical = {
                    "study": study["id"],
                    "level": level,
                    "axes": axes,
                    "case": study["case"],
                    "initialization": study["initialization"],
                }
                digest = _canonical_digest(canonical)
                result.append({
                    "case_id": f"{study['id']}--{digest[:16]}",
                    "case_digest": digest,
                    "study_index": study_index,
                    "study_id": study["id"],
                    "case": study["case"],
                    "dimension": study["dimension"],
                    "initialization": study["initialization"],
                    "refinement_axis": study["refinement_axis"],
                    "level": level,
                    "axes": axes,
                    "resolution": resolution,
                    "radius": radius,
                    "surface_tension": float(level.get(
                        "surface_tension",
                        _argument_value(
                            study.get("arguments", []), "--surface-tension")
                        or _argument_value(
                            registry["common_runner_arguments"],
                            "--surface-tension")
                        or 1.0,
                    )),
                    "h": 1.0 / resolution,
                    "radius_over_h": radius * resolution,
                    "metrics": sorted(set(study["metrics"]) | set(
                        registry["required_report_metrics"])),
                })
    ids = [case["case_id"] for case in result]
    if len(ids) != len(set(ids)):
        raise MatrixError("expanded physical case ids are not unique")
    for index, case in enumerate(result):
        case["index"] = index
    return result


def select_cases(
        cases: Sequence[dict[str, Any]],
        *,
        studies: Sequence[str] = (),
        case_index: int | None = None,
        shard_index: int | None = None,
        shard_count: int | None = None,
) -> list[dict[str, Any]]:
    selected = list(cases)
    if studies:
        wanted = set(studies)
        known = {case["study_id"] for case in cases}
        unknown = wanted - known
        if unknown:
            raise MatrixError(f"unknown studies requested: {sorted(unknown)}")
        selected = [case for case in selected if case["study_id"] in wanted]
    if case_index is not None:
        if case_index < 0 or case_index >= len(cases):
            raise MatrixError("case index is outside the expanded matrix")
        indexed = cases[case_index]
        selected = [case for case in selected if case is indexed]
    if (shard_index is None) != (shard_count is None):
        raise MatrixError("shard index and shard count must be provided together")
    if shard_count is not None:
        if shard_count < 1 or shard_index is None or not 0 <= shard_index < shard_count:
            raise MatrixError("invalid shard selection")
        selected = [
            case for case in selected
            if int(case["case_digest"], 16) % shard_count == shard_index
        ]
    return selected


def _append_option(arguments: list[str], name: str, *values: Any) -> None:
    arguments.append(name)
    arguments.extend(str(value) for value in values)


def physical_case_arguments(
        registry: dict[str, Any],
        case: dict[str, Any],
        *,
        solver: Path,
        qualification_log: Path,
) -> list[str]:
    study = registry["studies"][case["study_index"]]
    arguments = list(registry["common_runner_arguments"])
    arguments.extend(study.get("arguments", []))
    _append_option(arguments, "--solver", solver)
    _append_option(arguments, "--case", case["case"])
    _append_option(arguments, "--steps", study["steps"])
    _append_option(arguments, "--synthetic-nx", case["resolution"])
    _append_option(arguments, "--synthetic-ny", case["resolution"])
    if case["dimension"] == 3:
        _append_option(arguments, "--synthetic-nz", case["resolution"])

    level = case["level"]
    time_step = (
        float(level["value"])
        if case["refinement_axis"] == "time_step" else
        float(study["time_step"])
    )
    _append_option(arguments, "--time-step-size", f"{time_step:.17g}")
    _append_option(
        arguments, "--timeout-seconds",
        registry["resources"]["wall_time_seconds_per_case"])

    if case["refinement_axis"] == "physical_scale":
        _append_option(arguments, "--sessile-radius", case["radius"])
        _append_option(
            arguments, "--surface-tension", case["surface_tension"])
    if case["refinement_axis"] == "phi_scale":
        _append_option(
            arguments, "--level-set-positive-scale", level["value"])
    if case["refinement_axis"] == "reinitialization_cadence":
        arguments.append("--enable-level-set-reinitialization")
        _append_option(
            arguments, "--reinitialization-cadence-steps",
            int(level["value"]))

    axes = case["axes"]
    active_domain = axes.get("active_domain")
    if active_domain is not None:
        _append_option(arguments, "--level-set-active-domain", active_domain)
    contact_angle = axes.get("contact_angle")
    if contact_angle is not None:
        _append_option(
            arguments, "--contact-angle-degrees", contact_angle)
    wall = axes.get("wall")
    if wall is not None:
        option = (
            "--sessile-contact-wall-3d"
            if case["dimension"] == 3 else "--sessile-contact-wall")
        _append_option(arguments, option, wall)

    offset = axes.get("offset_h")
    if offset is not None:
        values = offset if isinstance(offset, list) else [offset]
        physical = [float(value) * case["h"] for value in values]
        if case["case"] == "droplet2d":
            option = "--capillary-droplet-center-offset"
        elif case["case"] == "sphere3d":
            option = "--capillary-sphere-center-offset"
        elif case["case"] == "sessile2d":
            option = "--sessile-tangent-center-offset"
        else:
            option = "--sessile-tangent-center-offset-3d"
        _append_option(arguments, option, *physical)

    if case["initialization"] == "discrete_energy_minimized":
        arguments.append("--initialize-discrete-static-capillary-equilibrium")
    else:
        arguments.append("--initialize-static-compatible-pressure")
    _append_option(arguments, "--qualification-log", qualification_log)
    return arguments


def _run_command(
        command: Sequence[str],
        *,
        cwd: Path,
        stdout_path: Path,
        stderr_path: Path,
        environment: dict[str, str] | None = None,
        timeout: float | None = None,
) -> dict[str, Any]:
    start = time.monotonic()
    timed_out = False
    return_code: int | None = None
    with stdout_path.open("w", encoding="utf-8") as stdout_stream:
        with stderr_path.open("w", encoding="utf-8") as stderr_stream:
            try:
                completed = subprocess.run(
                    list(command),
                    cwd=cwd,
                    env=environment,
                    stdout=stdout_stream,
                    stderr=stderr_stream,
                    text=True,
                    check=False,
                    timeout=timeout,
                )
                return_code = completed.returncode
            except subprocess.TimeoutExpired:
                timed_out = True
    return {
        "command": list(command),
        "returncode": return_code,
        "timed_out": timed_out,
        "elapsed_seconds": time.monotonic() - start,
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
    }


def run_physical_cases(
        registry: dict[str, Any],
        cases: Sequence[dict[str, Any]],
        *,
        solver: Path,
        output_root: Path,
        rerun: bool,
) -> dict[str, Any]:
    solver = solver.resolve()
    if not solver.is_file() or not os.access(solver, os.X_OK):
        raise MatrixError(f"solver is not executable: {solver}")
    physical_sha = sha256_file(PHYSICAL_RUNNER)
    solver_sha = sha256_file(solver)
    records: list[dict[str, Any]] = []
    for case in cases:
        case_directory = output_root / "cases" / case["case_id"]
        case_directory.mkdir(parents=True, exist_ok=True)
        qualification_log = case_directory / "qualification.json"
        result_path = case_directory / "execution.json"
        if result_path.exists() and not rerun:
            existing = read_json(result_path)
            if (isinstance(existing, dict) and
                    existing.get("returncode") == 0 and
                    existing.get("timed_out") is False and
                    qualification_log.exists()):
                records.append(existing)
                continue
        write_json(case_directory / "case.json", case)
        arguments = physical_case_arguments(
            registry, case, solver=solver,
            qualification_log=qualification_log)
        command = [sys.executable, str(PHYSICAL_RUNNER), *arguments]
        write_json(case_directory / "command.json", command)
        environment = os.environ.copy()
        environment["TMPDIR"] = str(case_directory / "tmp")
        (case_directory / "tmp").mkdir(parents=True, exist_ok=True)
        execution = _run_command(
            command,
            cwd=SCRIPT_DIRECTORY.parents[2],
            stdout_path=case_directory / "stdout.txt",
            stderr_path=case_directory / "stderr.txt",
            environment=environment,
            timeout=float(
                registry["resources"]["wall_time_seconds_per_case"]),
        )
        record = {
            **execution,
            "case_id": case["case_id"],
            "case_digest": case["case_digest"],
            "qualification_log": str(qualification_log),
        }
        write_json(result_path, record)
        records.append(record)
    manifest = {
        "schema_version": 1,
        "matrix_id": registry["matrix_id"],
        "registry_sha256": sha256_file(DEFAULT_REGISTRY),
        "physical_runner_sha256": physical_sha,
        "solver": str(solver),
        "solver_sha256": solver_sha,
        "selected_case_count": len(cases),
        "selected_case_ids": [case["case_id"] for case in cases],
        "records": records,
    }
    write_json(output_root / "physical_execution_manifest.json", manifest)
    return manifest


def _gtest_cases(payload: Any, context: str) -> dict[str, dict[str, Any]]:
    if not isinstance(payload, dict):
        raise MatrixError(f"{context} root must be an object")
    suites = payload.get("testsuites")
    if not isinstance(suites, list):
        raise MatrixError(f"{context} has no test suites")
    result: dict[str, dict[str, Any]] = {}
    for suite_index, suite in enumerate(suites):
        if not isinstance(suite, dict):
            raise MatrixError(f"{context} suite {suite_index} is invalid")
        suite_name = nonempty_string(
            suite.get("name"), f"{context} suite {suite_index} name")
        cases = suite.get("testsuite")
        if not isinstance(cases, list):
            raise MatrixError(f"{context} suite {suite_name!r} has no cases")
        for case_index, case in enumerate(cases):
            if not isinstance(case, dict):
                raise MatrixError(
                    f"{context} suite {suite_name!r} case {case_index} "
                    "is invalid")
            name = nonempty_string(
                case.get("name"),
                f"{context} suite {suite_name!r} case {case_index} name")
            classname = case.get("classname", suite_name)
            if not isinstance(classname, str) or not classname:
                raise MatrixError(f"{context} case {name!r} has no class")
            full_name = f"{classname}.{name}"
            if full_name in result:
                raise MatrixError(
                    f"{context} repeats exact test {full_name!r}")
            result[full_name] = case
    return result


def _gtest_custom_properties(case: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value for key, value in sorted(case.items())
        if key not in GTEST_CASE_METADATA_KEYS
    }


def _finite_property(value: Any, context: str) -> float:
    if isinstance(value, bool):
        raise MatrixError(f"{context} is not numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise MatrixError(f"{context} is not numeric") from error
    if not math.isfinite(result):
        raise MatrixError(f"{context} is not finite")
    return result


def evaluate_exact_property_gate(
        properties: dict[str, Any],
        gate: dict[str, Any],
        *,
        roundoff_factor: float,
) -> dict[str, Any]:
    property_name = gate["property"]
    comparison = gate["comparison"]
    record: dict[str, Any] = {
        "property": property_name,
        "comparison": comparison,
        "passed": False,
    }
    if property_name not in properties:
        record["error"] = "property is missing"
        return record
    record["observed"] = properties[property_name]
    try:
        observed = _finite_property(
            properties[property_name], f"exact property {property_name!r}")
    except MatrixError as error:
        record["error"] = str(error)
        return record
    record["observed_numeric"] = observed
    if comparison == "finite":
        record["passed"] = True
        return record
    if comparison == "scaled_roundoff":
        scale = float(gate["scale"])
        limit = roundoff_factor * sys.float_info.epsilon * max(1.0, abs(scale))
        record["scale"] = scale
        record["limit"] = limit
        record["passed"] = abs(observed) <= limit
        return record
    expected = float(gate["expected"])
    record["expected"] = expected
    if comparison == "equal":
        record["passed"] = observed == expected
    elif comparison == "at_most":
        record["passed"] = observed <= expected
    else:
        record["passed"] = observed >= expected
    return record


def evaluate_exact_document(
        payload: Any,
        group: dict[str, Any],
        *,
        roundoff_factor: float,
        context: str,
) -> dict[str, Any]:
    errors: list[str] = []
    try:
        cases = _gtest_cases(payload, context)
    except MatrixError as error:
        return {
            "passed": False,
            "errors": [str(error)],
            "properties": {},
            "property_gates": {},
        }
    expected_tests = set(group["tests"])
    observed_tests = set(cases)
    if observed_tests != expected_tests:
        missing = sorted(expected_tests - observed_tests)
        unexpected = sorted(observed_tests - expected_tests)
        errors.append(
            f"exact test-name mismatch: missing={missing}, "
            f"unexpected={unexpected}")
    top_level_checks = {
        "tests": len(group["tests"]),
        "failures": 0,
        "disabled": 0,
    }
    if not isinstance(payload, dict):
        errors.append("GoogleTest payload is not an object")
    else:
        for key, expected in top_level_checks.items():
            if payload.get(key) != expected:
                errors.append(
                    f"GoogleTest {key} is {payload.get(key)!r}, "
                    f"expected {expected}")
    properties: dict[str, dict[str, Any]] = {}
    property_gates: dict[str, list[dict[str, Any]]] = {}
    for test in group["tests"]:
        case = cases.get(test)
        if case is None:
            continue
        if case.get("status") != "RUN" or case.get("result") != "COMPLETED":
            errors.append(
                f"exact test {test!r} did not complete: "
                f"status={case.get('status')!r}, result={case.get('result')!r}")
        failures = case.get("failures")
        if failures not in (None, []):
            errors.append(f"exact test {test!r} contains failures")
        custom = _gtest_custom_properties(case)
        properties[test] = custom
        gate_records = [
            evaluate_exact_property_gate(
                custom, gate, roundoff_factor=roundoff_factor)
            for gate in group["property_gates"][test]
        ]
        property_gates[test] = gate_records
        for gate_record in gate_records:
            if not gate_record["passed"]:
                errors.append(
                    f"exact test {test!r} failed property gate "
                    f"{gate_record['property']!r}")
    return {
        "passed": not errors,
        "errors": errors,
        "properties": properties,
        "property_gates": property_gates,
    }


def exact_rank_properties_identical(
        documents: Sequence[dict[str, Any]], expected_ranks: int) -> bool:
    property_maps = [document.get("properties") for document in documents]
    return (
        len(property_maps) == expected_ranks and
        all(isinstance(value, dict) for value in property_maps) and
        all(value == property_maps[0] for value in property_maps[1:])
    )


def _rank_wrapper() -> str:
    return (
        'rank_value="${OMPI_COMM_WORLD_RANK:-'
        "${PMI_RANK:-${PMIX_RANK:-${MV2_COMM_WORLD_RANK:-"
        '${SLURM_PROCID:-}}}}}"; '
        'case "$rank_value" in ""|*[!0-9]*) '
        'echo "invalid or missing MPI rank" >&2; exit 97;; esac; '
        'exec "$1" "$2" "$3" '
        '"--gtest_output=json:${4}/gtest_rank_${rank_value}.json"'
    )


def exact_mpi_launcher_arguments(mode: str, ranks: int) -> list[str]:
    if mode == "mpiexec":
        return ["--oversubscribe", "-n", str(ranks)]
    if mode == "srun":
        return [
            "--overlap",
            "--nodes=1",
            "--ntasks",
            str(ranks),
            "--cpus-per-task=1",
            "--cpu-bind=none",
        ]
    raise MatrixError("exact MPI launcher mode is invalid")


def run_exact_groups(
        registry: dict[str, Any],
        *,
        binaries: dict[str, Path],
        mpi_launcher: Path,
        mpi_launcher_mode: str,
        output_root: Path,
) -> dict[str, Any]:
    if mpi_launcher_mode not in {"mpiexec", "srun"}:
        raise MatrixError("exact MPI launcher mode is invalid")
    if not mpi_launcher.is_file() or not os.access(mpi_launcher, os.X_OK):
        raise MatrixError("exact MPI launcher is not executable")
    if mpi_launcher_mode == "srun" and not os.environ.get("SLURM_JOB_ID"):
        raise MatrixError("srun exact launch requires an active Slurm allocation")
    required = {group["binary"] for group in registry["exact_groups"]}
    missing = required - set(binaries)
    if missing:
        raise MatrixError(f"missing exact-group binaries: {sorted(missing)}")
    records = []
    roundoff_factor = finite_number(
        registry["gates"].get("exact_flat_scaled_residual_factor"),
        "exact scaled-roundoff factor", positive=True)
    for group in registry["exact_groups"]:
        binary = binaries[group["binary"]].resolve()
        if not binary.is_file() or not os.access(binary, os.X_OK):
            raise MatrixError(f"exact-group binary is not executable: {binary}")
        directory = output_root / "exact" / group["id"]
        directory.mkdir(parents=True, exist_ok=False)
        test_filter = ":".join(group["tests"])
        ranks = int(group["mpi_ranks"])
        if ranks == 1:
            command = [
                str(binary),
                f"--gtest_filter={test_filter}",
                "--gtest_color=no",
                f"--gtest_output=json:{directory / 'gtest.json'}",
            ]
        else:
            launcher_arguments = exact_mpi_launcher_arguments(
                mpi_launcher_mode, ranks)
            command = [str(mpi_launcher), *launcher_arguments,
                       "/bin/sh", "-c", _rank_wrapper(), "wp4-v2-rank",
                       str(binary), f"--gtest_filter={test_filter}",
                       "--gtest_color=no", str(directory)]
        environment = os.environ.copy()
        environment.update({
            "OMP_NUM_THREADS": "1",
            "OMPI_ALLOW_RUN_AS_ROOT": "1",
            "OMPI_ALLOW_RUN_AS_ROOT_CONFIRM": "1",
        })
        execution = _run_command(
            command,
            cwd=SCRIPT_DIRECTORY.parents[2],
            stdout_path=directory / "stdout.txt",
            stderr_path=directory / "stderr.txt",
            environment=environment,
        )
        documents = sorted(directory.glob("gtest*.json"))
        expected_documents = ranks
        document_records = []
        group_errors: list[str] = []
        passed = (
            execution["returncode"] == 0 and not execution["timed_out"] and
            len(documents) == expected_documents)
        if execution["returncode"] != 0 or execution["timed_out"]:
            group_errors.append("exact command did not complete successfully")
        if len(documents) != expected_documents:
            group_errors.append(
                f"found {len(documents)} exact documents, "
                f"expected {expected_documents}")
        for document in documents:
            try:
                payload = read_json(document)
                evaluation = evaluate_exact_document(
                    payload,
                    group,
                    roundoff_factor=roundoff_factor,
                    context=str(document),
                )
            except MatrixError as error:
                payload = None
                evaluation = {
                    "passed": False,
                    "errors": [str(error)],
                    "properties": {},
                    "property_gates": {},
                }
            document_record = {
                "path": str(document),
                "sha256": sha256_file(document),
                "tests": (
                    payload.get("tests") if isinstance(payload, dict) else None),
                "failures": (
                    payload.get("failures")
                    if isinstance(payload, dict) else None),
                "disabled": (
                    payload.get("disabled")
                    if isinstance(payload, dict) else None),
                **evaluation,
            }
            document_records.append(document_record)
            group_errors.extend(evaluation["errors"])
            passed = passed and evaluation["passed"]
        rank_properties_identical = True
        if group["require_identical_rank_properties"]:
            rank_properties_identical = exact_rank_properties_identical(
                document_records, ranks)
            if not rank_properties_identical:
                group_errors.append(
                    "custom exact-test properties differ across MPI ranks")
            passed = passed and rank_properties_identical
        record = {
            "group_id": group["id"],
            "binary": str(binary),
            "binary_sha256": sha256_file(binary),
            "mpi_ranks": ranks,
            "mpi_launcher": str(mpi_launcher),
            "mpi_launcher_mode": mpi_launcher_mode,
            "expected_tests": group["tests"],
            "passed": passed,
            "errors": group_errors,
            "rank_properties_identical": rank_properties_identical,
            "execution": execution,
            "documents": document_records,
        }
        write_json(directory / "group_summary.json", record)
        records.append(record)
    summary = {
        "matrix_id": registry["matrix_id"],
        "registry_sha256": sha256_file(DEFAULT_REGISTRY),
        "physical_runner_sha256": sha256_file(PHYSICAL_RUNNER),
        "group_count": len(records),
        "passed": all(record["passed"] for record in records),
        "groups": records,
    }
    write_json(output_root / "exact_summary.json", summary)
    return summary


def _nested_number(value: Any, *paths: Sequence[str]) -> float | None:
    for path in paths:
        current = value
        for key in path:
            if not isinstance(current, dict) or key not in current:
                break
            current = current[key]
        else:
            if (isinstance(current, (int, float)) and
                    not isinstance(current, bool) and
                    math.isfinite(float(current))):
                return float(current)
    return None


def extract_metric(
        name: str, probe: dict[str, Any], case: dict[str, Any]) -> float:
    if name == "pressure_jump_relative_error":
        direct = _nested_number(
            probe,
            ("capillary_pressure_jump_relative_error",),
            ("sessile_final_pressure_jump_relative_error",),
        )
        if direct is not None:
            return abs(direct)
        observed = _nested_number(
            probe,
            ("capillary_final_pressure_jump",),
            ("spatial_capillary_final_pressure_jump",),
            ("final_sessile_state", "pressure_jump"),
        )
        if observed is not None:
            factor = 2.0 if case["dimension"] == 3 else 1.0
            expected = factor * case["surface_tension"] / case["radius"]
            return abs(observed - expected) / abs(expected)
    elif name == "pressure_space_relative_distance":
        value = _nested_number(
            probe,
            ("static_capillary_pressure_representability_relative_distance",),
            ("diagnostic_free_surface_pressure_representability_relative_residual",),
            ("latest_free_surface_pressure_representability_distance_gate",
             "pressure_representability_relative_residual"),
        )
        if value is not None:
            return abs(value)
    elif name == "conservative_balance_normalized_imbalance":
        value = _nested_number(
            probe,
            ("diagnostic_free_surface_conservative_balance_normalized_imbalance",),
            ("latest_available_free_surface_conservative_balance",
             "normalized_imbalance"),
        )
        if value is not None:
            return abs(value)
    elif name == "contact_angle_absolute_error_degrees":
        value = _nested_number(
            probe, ("sessile_final_contact_angle_absolute_error_degrees",))
        if value is not None:
            return abs(value)
    elif name == "base_radius_relative_error":
        value = _nested_number(
            probe, ("sessile_final_base_radius_relative_error",))
        if value is not None:
            return abs(value)
    elif name == "apex_height_relative_error":
        value = _nested_number(
            probe, ("sessile_final_apex_height_relative_error",))
        if value is not None:
            return abs(value)
    elif name == "parasitic_capillary_number":
        value = _nested_number(
            probe,
            ("capillary_final_parasitic_capillary_number",),
            ("sessile_final_parasitic_capillary_number",),
        )
        if value is not None:
            return abs(value)
        speed = _nested_number(
            probe,
            ("spatial_capillary_final_max_liquid_speed",),
            ("max_speed",),
        )
        viscosity = _nested_number(probe, ("benchmark", "viscosity"))
        if speed is not None and viscosity is not None:
            return abs(viscosity * speed / case["surface_tension"])
    elif name == "kinetic_energy_proxy":
        history = probe.get("free_surface_energy_history")
        if isinstance(history, list) and history:
            value = _nested_number(history[-1], ("kinetic_energy_proxy",))
            if value is not None:
                return abs(value)
    elif name == "liquid_volume_relative_error":
        value = _nested_number(
            probe,
            ("sessile_final_liquid_volume_relative_error",),
            ("sessile_final_liquid_area_relative_error",),
        )
        if value is not None:
            return abs(value)
        observed = _nested_number(
            probe,
            ("spatial_capillary_final_liquid_volume",),
            ("production_physical_liquid_volume_final",),
            ("wet_fraction_volume",),
        )
        if observed is not None:
            expected = (
                4.0 * math.pi * case["radius"] ** 3 / 3.0
                if case["dimension"] == 3 else
                math.pi * case["radius"] ** 2
            )
            return abs(observed - expected) / expected
    raise MatrixError(
        f"case {case['case_id']!r} lacks finite metric {name!r}")


def _load_physical_evidence(
        roots: Sequence[Path],
) -> dict[str, tuple[dict[str, Any], dict[str, Any], Path]]:
    result: dict[str, tuple[dict[str, Any], dict[str, Any], Path]] = {}
    for root in roots:
        for case_path in sorted((root / "cases").glob("*/case.json")):
            case = read_json(case_path)
            if not isinstance(case, dict):
                raise MatrixError(f"invalid case record {case_path}")
            case_id = nonempty_string(case.get("case_id"), "evidence case id")
            qualification_path = case_path.parent / "qualification.json"
            if not qualification_path.exists():
                raise MatrixError(f"missing qualification log for {case_id}")
            qualification = read_json(qualification_path)
            existing = result.get(case_id)
            if existing is not None:
                if existing[0] != case or existing[1] != qualification:
                    raise MatrixError(f"conflicting evidence for case {case_id}")
                continue
            result[case_id] = (case, qualification, qualification_path)
    return result


def _case_metric_records(
        expected_cases: Sequence[dict[str, Any]],
        evidence: dict[str, tuple[dict[str, Any], dict[str, Any], Path]],
) -> tuple[list[dict[str, Any]], list[str]]:
    records = []
    errors = []
    for expected in expected_cases:
        available = evidence.get(expected["case_id"])
        if available is None:
            errors.append(f"missing physical case {expected['case_id']}")
            continue
        case, qualification, path = available
        if case.get("case_digest") != expected["case_digest"]:
            errors.append(f"case digest mismatch for {expected['case_id']}")
            continue
        probes = qualification.get("probes")
        if (qualification.get("complete") is not True or
                not isinstance(probes, list) or len(probes) != 1 or
                not isinstance(probes[0], dict)):
            errors.append(f"incomplete qualification log {path}")
            continue
        probe = probes[0]
        if probe.get("passed") is not True or probe.get("errors") not in ([], None):
            errors.append(f"physical case did not pass: {expected['case_id']}")
        metrics: dict[str, float] = {}
        for name in expected["metrics"]:
            try:
                metrics[name] = extract_metric(name, probe, expected)
            except MatrixError as error:
                errors.append(str(error))
        records.append({
            "case": expected,
            "qualification_path": str(path),
            "qualification_sha256": sha256_file(path),
            "metrics": metrics,
        })
    unexpected = set(evidence) - {case["case_id"] for case in expected_cases}
    if unexpected:
        errors.append(f"unexpected physical cases: {sorted(unexpected)}")
    return records, errors


def _load_convergence_module() -> Any:
    if str(SCRIPT_DIRECTORY) not in sys.path:
        sys.path.insert(0, str(SCRIPT_DIRECTORY))
    import free_surface_convergence as convergence
    return convergence


def _group_key(case: dict[str, Any], *, omit_offset: bool) -> str:
    axes = {
        key: value for key, value in case["axes"].items()
        if not (omit_offset and key == "offset_h")
    }
    return json.dumps(axes, sort_keys=True, separators=(",", ":"))


def _aggregate_status(statuses: Sequence[str]) -> str:
    if not statuses or "FAIL" in statuses:
        return "FAIL"
    if "ADDITIONAL_LEVEL_REQUIRED" in statuses:
        return "ADDITIONAL_LEVEL_REQUIRED"
    return "PASS"


def analyze_convergence(
        registry: dict[str, Any],
        records: Sequence[dict[str, Any]],
        expected_cases: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    convergence = _load_convergence_module()
    gates = registry["gates"]["convergence"]
    refinement = registry["refinement"]
    by_study: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_study.setdefault(record["case"]["study_id"], []).append(record)
    expected_by_study: dict[str, list[dict[str, Any]]] = {}
    for case in expected_cases:
        expected_by_study.setdefault(case["study_id"], []).append(case)
    analyses: dict[str, Any] = {}
    for study in registry["studies"]:
        axis = study["refinement_axis"]
        if axis not in {"resolution", "time_step", "reinitialization_cadence"}:
            continue
        study_records = by_study.get(study["id"], [])
        expected_study_cases = expected_by_study.get(study["id"], [])
        groups: dict[str, list[dict[str, Any]]] = {}
        expected_groups: dict[str, list[dict[str, Any]]] = {}
        for record in study_records:
            groups.setdefault(
                _group_key(record["case"], omit_offset=True), []).append(record)
        for case in expected_study_cases:
            expected_groups.setdefault(
                _group_key(case, omit_offset=True), []).append(case)
        study_analysis: dict[str, Any] = {}
        for group_key, group_expected in sorted(expected_groups.items()):
            group_records = groups.get(group_key, [])
            records_by_id = {
                record["case"]["case_id"]: record for record in group_records
            }
            missing_cases = sorted(
                case["case_id"] for case in group_expected
                if case["case_id"] not in records_by_id)
            group_errors = (
                [f"missing convergence cases: {missing_cases}"]
                if missing_cases else [])
            metric_records: dict[str, Any] = {}
            for metric in study["metrics"]:
                if metric not in gates:
                    continue
                sequences: dict[str, list[dict[str, Any]]] = {}
                missing_metrics = []
                for case in group_expected:
                    offset = json.dumps(
                        case["axes"].get("offset_h", "none"),
                        sort_keys=True, separators=(",", ":"))
                    record = records_by_id.get(case["case_id"])
                    if record is None or metric not in record["metrics"]:
                        missing_metrics.append(case["case_id"])
                        continue
                    if axis == "resolution":
                        spacing = float(case["h"])
                    else:
                        spacing = float(case["level"]["value"])
                    sequences.setdefault(offset, []).append({
                        "label": case["level"]["label"],
                        "h": spacing,
                        "value": record["metrics"][metric],
                    })
                gate = gates[metric]
                if missing_metrics:
                    metric_records[metric] = {
                        "status": "FAIL",
                        "error": (
                            "missing finite metric values for cases: "
                            f"{sorted(missing_metrics)}"),
                        "sequences": sequences,
                    }
                    continue
                try:
                    metric_records[metric] = (
                        convergence.analyze_offset_envelope(
                            sequences,
                            reference_value=float(gate["reference"]),
                            normalization=float(gate["normalization"]),
                            minimum_observed_order=float(
                                refinement["minimum_observed_order"]),
                            finest_relative_error_limit=float(
                                gate["finest_error_limit"]),
                            finest_gci_limit=float(gate["finest_gci_limit"]),
                            safety_factor=float(refinement["safety_factor"]),
                            ratio_relative_tolerance=float(
                                refinement["ratio_relative_tolerance"]),
                        )
                    )
                except (KeyError, MatrixError, TypeError, ValueError,
                        ZeroDivisionError) as error:
                    metric_records[metric] = {
                        "status": "FAIL",
                        "error": str(error),
                        "sequences": sequences,
                    }
            if not metric_records:
                group_errors.append("no convergence metrics were analyzed")
            statuses = [
                value["status"] for value in metric_records.values()
            ]
            if group_errors:
                statuses.append("FAIL")
            study_analysis[group_key] = {
                "status": _aggregate_status(statuses),
                "expected_case_count": len(group_expected),
                "accepted_case_count": len(group_records),
                "errors": group_errors,
                "metrics": metric_records,
            }
        statuses = [value["status"] for value in study_analysis.values()]
        analyses[study["id"]] = {
            "status": _aggregate_status(statuses),
            "groups": study_analysis,
        }
    statuses = [value["status"] for value in analyses.values()]
    return {
        "status": _aggregate_status(statuses),
        "studies": analyses,
    }


def analyze_invariance(
        registry: dict[str, Any],
        records: Sequence[dict[str, Any]],
        expected_cases: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    gates = registry["gates"]["invariance"]
    records_by_id = {
        record["case"]["case_id"]: record for record in records
    }
    analyses: dict[str, Any] = {}
    for study in registry["studies"]:
        axis = study["refinement_axis"]
        if axis not in gates:
            continue
        study_cases = [
            case for case in expected_cases
            if case["study_id"] == study["id"]
        ]
        expected_groups: dict[str, list[dict[str, Any]]] = {}
        for case in study_cases:
            expected_groups.setdefault(
                _group_key(case, omit_offset=False), []).append(case)
        group_analyses: dict[str, Any] = {}
        for group_key, group_cases in sorted(expected_groups.items()):
            metric_analyses: dict[str, Any] = {}
            for metric, limits in gates[axis].items():
                values = []
                missing = []
                for case in group_cases:
                    record = records_by_id.get(case["case_id"])
                    if record is None or metric not in record["metrics"]:
                        missing.append(case["case_id"])
                        continue
                    values.append({
                        "case_id": case["case_id"],
                        "level": case["level"]["label"],
                        "value": record["metrics"][metric],
                    })
                observed = [value["value"] for value in values]
                maximum = max(observed) if observed else None
                spread = (
                    max(observed) - min(observed) if observed else None)
                passed = (
                    not missing and len(values) == len(group_cases) and
                    maximum is not None and spread is not None and
                    maximum <= float(limits["maximum_value"]) and
                    spread <= float(limits["maximum_spread"])
                )
                metric_analyses[metric] = {
                    "status": "PASS" if passed else "FAIL",
                    "expected_case_count": len(group_cases),
                    "values": values,
                    "missing_cases": sorted(missing),
                    "maximum_value": maximum,
                    "maximum_value_limit": limits["maximum_value"],
                    "spread": spread,
                    "maximum_spread": limits["maximum_spread"],
                }
            group_analyses[group_key] = {
                "status": _aggregate_status([
                    value["status"] for value in metric_analyses.values()
                ]),
                "metrics": metric_analyses,
            }
        analyses[study["id"]] = {
            "refinement_axis": axis,
            "status": _aggregate_status([
                value["status"] for value in group_analyses.values()
            ]),
            "groups": group_analyses,
        }
    return {
        "status": _aggregate_status([
            value["status"] for value in analyses.values()
        ]),
        "studies": analyses,
    }


def analyze_finest_level(
        registry: dict[str, Any],
        records: Sequence[dict[str, Any]],
        expected_cases: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    limits = registry["gates"]["finest_level"]
    records_by_id = {
        record["case"]["case_id"]: record for record in records
    }
    selected: list[dict[str, Any]] = []
    for study in registry["studies"]:
        cases = [
            case for case in expected_cases
            if case["study_id"] == study["id"]
        ]
        axis = study["refinement_axis"]
        if axis in {"phi_scale", "physical_scale"}:
            selected.extend(cases)
            continue
        level_values = [float(case["level"]["value"]) for case in cases]
        if not level_values:
            continue
        finest = (
            max(level_values) if axis == "resolution" else min(level_values)
        )
        selected.extend(
            case for case in cases
            if float(case["level"]["value"]) == finest)
    case_analyses = []
    for case in selected:
        record = records_by_id.get(case["case_id"])
        metric_analyses = []
        for metric in case["metrics"]:
            if metric not in limits:
                continue
            observed = (
                record["metrics"].get(metric) if record is not None else None
            )
            passed = (
                observed is not None and
                observed <= float(limits[metric])
            )
            metric_analyses.append({
                "metric": metric,
                "observed": observed,
                "limit": limits[metric],
                "passed": passed,
            })
        case_analyses.append({
            "case_id": case["case_id"],
            "study_id": case["study_id"],
            "status": _aggregate_status([
                "PASS" if metric["passed"] else "FAIL"
                for metric in metric_analyses
            ]),
            "metrics": metric_analyses,
        })
    return {
        "status": _aggregate_status([
            case["status"] for case in case_analyses
        ]),
        "case_count": len(case_analyses),
        "cases": case_analyses,
    }


def analyze_evidence(
        registry: dict[str, Any],
        *,
        roots: Sequence[Path],
        output_root: Path,
        include_conditional_level: bool,
        exact_summary_path: Path | None,
) -> dict[str, Any]:
    expected_cases = expand_cases(
        registry, include_conditional_level=include_conditional_level)
    errors: list[str] = []
    try:
        evidence = _load_physical_evidence(roots)
    except MatrixError as error:
        evidence = {}
        errors.append(str(error))
    records, record_errors = _case_metric_records(expected_cases, evidence)
    errors.extend(record_errors)
    convergence = analyze_convergence(registry, records, expected_cases)
    invariance = analyze_invariance(registry, records, expected_cases)
    finest_level = analyze_finest_level(registry, records, expected_cases)
    exact_passed = False
    exact_summary = None
    if exact_summary_path is None:
        errors.append("exact-group summary is missing")
    else:
        try:
            exact_summary = read_json(exact_summary_path)
            expected_group_ids = {
                group["id"] for group in registry["exact_groups"]
            }
            observed_group_ids = {
                group.get("group_id")
                for group in exact_summary.get("groups", [])
                if isinstance(group, dict)
            } if isinstance(exact_summary, dict) else set()
            exact_passed = (
                isinstance(exact_summary, dict) and
                exact_summary.get("matrix_id") == registry["matrix_id"] and
                exact_summary.get("registry_sha256") ==
                sha256_file(DEFAULT_REGISTRY) and
                exact_summary.get("physical_runner_sha256") ==
                sha256_file(PHYSICAL_RUNNER) and
                exact_summary.get("passed") is True and
                exact_summary.get("group_count") ==
                len(registry["exact_groups"]) and
                observed_group_ids == expected_group_ids)
        except MatrixError as error:
            errors.append(str(error))
        if not exact_passed:
            errors.append("exact-group summary did not pass")
    if convergence["status"] != "PASS":
        errors.append(
            f"convergence disposition is {convergence['status']}")
    if invariance["status"] != "PASS":
        errors.append(f"invariance disposition is {invariance['status']}")
    if finest_level["status"] != "PASS":
        errors.append(
            f"finest-level disposition is {finest_level['status']}")
    passed = not errors
    summary = {
        "schema_version": 1,
        "matrix_id": registry["matrix_id"],
        "registry_sha256": sha256_file(DEFAULT_REGISTRY),
        "physical_runner_sha256": sha256_file(PHYSICAL_RUNNER),
        "expected_case_count": len(expected_cases),
        "accepted_case_count": len(records),
        "include_conditional_level": include_conditional_level,
        "exact_groups_passed": exact_passed,
        "convergence": convergence,
        "invariance": invariance,
        "finest_level": finest_level,
        "errors": errors,
        "passed": passed,
        "disposition": {
            "fsr03_closed": passed,
            "fsr04_closed": passed,
            "wp4_closed": passed,
            "q2_closed": passed,
        },
        "physical_records": records,
        "exact_summary": exact_summary,
    }
    write_json(output_root / "summary.json", summary)
    manifest_lines = []
    for path in sorted(output_root.rglob("*")):
        if path.is_file() and path.name != "manifest.sha256":
            manifest_lines.append(
                f"{sha256_file(path)}  {path.relative_to(output_root)}")
    (output_root / "manifest.sha256").write_text(
        "\n".join(manifest_lines) + "\n", encoding="utf-8")
    return summary


def _parse_binary(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("binary must be KEY=PATH")
    key, path = value.split("=", 1)
    if not key or not path:
        raise argparse.ArgumentTypeError("binary must be KEY=PATH")
    return key, Path(path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--list-cases", action="store_true")
    parser.add_argument("--include-conditional-level", action="store_true")
    parser.add_argument("--study", action="append", default=[])
    parser.add_argument("--case-index", type=int)
    parser.add_argument("--shard-index", type=int)
    parser.add_argument("--shard-count", type=int)
    parser.add_argument("--run-physical", action="store_true")
    parser.add_argument("--run-exact", action="store_true")
    parser.add_argument("--analyze", action="store_true")
    parser.add_argument("--solver", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--results-root", type=Path, action="append", default=[])
    parser.add_argument("--exact-summary", type=Path)
    parser.add_argument("--binary", type=_parse_binary, action="append", default=[])
    parser.add_argument(
        "--mpi-launcher", type=Path, default=Path("/usr/bin/mpiexec"))
    parser.add_argument(
        "--mpi-launcher-mode", choices=("mpiexec", "srun"),
        default="mpiexec")
    parser.add_argument("--rerun", action="store_true")
    return parser


def main(arguments: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(arguments)
    registry = load_registry(args.registry)
    cases = expand_cases(
        registry, include_conditional_level=args.include_conditional_level)
    selected = select_cases(
        cases,
        studies=args.study,
        case_index=args.case_index,
        shard_index=args.shard_index,
        shard_count=args.shard_count,
    )
    if args.validate_only:
        if any((args.list_cases, args.run_physical, args.run_exact, args.analyze)):
            raise MatrixError("--validate-only cannot be combined with other actions")
        print(json.dumps({
            "matrix_id": registry["matrix_id"],
            "status": registry["status"],
            "exact_group_count": len(registry["exact_groups"]),
            "study_count": len(registry["studies"]),
            "physical_case_count": len(cases),
            "conditional_level_included": args.include_conditional_level,
            "outcome": "PASS",
        }, sort_keys=True))
        return 0
    if args.list_cases:
        for case in selected:
            print(json.dumps(case, sort_keys=True))
        return 0
    if not any((args.run_physical, args.run_exact, args.analyze)):
        raise MatrixError("select --validate-only, --list-cases, or an execution action")
    if args.output_root is None:
        raise MatrixError("execution and analysis require --output-root")
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    if args.run_physical:
        if args.solver is None:
            raise MatrixError("--run-physical requires --solver")
        run_physical_cases(
            registry, selected, solver=args.solver,
            output_root=output_root, rerun=args.rerun)
    exact_summary = None
    if args.run_exact:
        binaries = dict(args.binary)
        if len(binaries) != len(args.binary):
            raise MatrixError("duplicate exact-group binary keys")
        exact_summary = run_exact_groups(
            registry, binaries=binaries,
            mpi_launcher=args.mpi_launcher.resolve(),
            mpi_launcher_mode=args.mpi_launcher_mode,
            output_root=output_root)
        if exact_summary["passed"] is not True and not args.analyze:
            return 1
    if args.analyze:
        roots = args.results_root or [output_root]
        summary = analyze_evidence(
            registry,
            roots=[path.resolve() for path in roots],
            output_root=output_root,
            include_conditional_level=args.include_conditional_level,
            exact_summary_path=(
                args.exact_summary.resolve()
                if args.exact_summary is not None else
                output_root / "exact_summary.json"
                if exact_summary is not None else None),
        )
        return 0 if summary["passed"] else 1
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (MatrixError, OSError, RuntimeError, KeyError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)

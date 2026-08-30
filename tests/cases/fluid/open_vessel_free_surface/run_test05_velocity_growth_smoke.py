#!/usr/bin/env python3
"""Run a short unfitted dam-break velocity-growth solver probe."""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import re
import signal
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pyvista as pv


ROOT = Path(__file__).resolve().parents[4]
CASE_ROOT = ROOT / "tests/cases/fluid/open_vessel_free_surface/unfitted_level_set"
TOOLS_DIR = ROOT / "tools"
SCRIPT_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from collate_vtk_time_series import collate_time_series
from free_surface_energy import (
    energy_history_gate_errors,
    free_surface_energy_state_2d,
    free_surface_energy_state_3d,
    summarize_energy_history,
)
from static_capillary_3d import (
    active_signed_level_set,
    normalized_active_domain,
    oriented_level_set,
    spatial_capillary_state_metrics,
    write_sessile_sphere_case,
    write_sphere_case,
)

CASES = {
    "mini2d": None,
    "static2d": None,
    "capillaryarc2d": None,
    "droplet2d": None,
    "sphere3d": None,
    "capillarywave2d": None,
    "sessile2d": None,
    "sessile3d": None,
    "dynamiccontact2d": None,
    "curvedtet3d": None,
    "open2d": CASE_ROOT,
    "d18": CASE_ROOT / "spheric_test05_wet_bed_d18",
    "d38": CASE_ROOT / "spheric_test05_wet_bed_d38",
    "mms2d": CASE_ROOT / "mms_traveling_interface_2d",
    "sloshing2d": CASE_ROOT / "linear_sloshing_2d",
    "tilt2d": CASE_ROOT / "square_tank_tilt_settling",
}
CASE_COPY_ENTRIES = {
    CASE_ROOT: ("solver.xml", "pressure_gauge.csv", "mesh"),
}
CASE_GATE_X = {
    "capillaryarc2d": 0.5,
    "droplet2d": 0.5,
    "sphere3d": 0.5,
    "capillarywave2d": 0.5,
    "sessile2d": 0.5,
    "sessile3d": 0.5,
    "dynamiccontact2d": 0.5,
    "mini2d": 0.4,
    "static2d": 0.5,
    "open2d": 0.5,
    "mms2d": 0.5,
    "sloshing2d": 0.5,
    "tilt2d": 0.5,
    "curvedtet3d": 0.5,
}
HIGH_ORDER_PRODUCTION_CASES = ("sloshing2d", "tilt2d")
HIGH_ORDER_MPI_PRODUCTION_CASES = ("sloshing2d", "tilt2d")
HIGH_ORDER_VISIBLE_MOTION_CASES = ("tilt2d",)
HIGH_ORDER_3D_BENCHMARK_CASES = ("d18",)
HIGH_ORDER_3D_BENCHMARK_QUALIFICATION_CASES = ("d18", "d38")
HIGH_ORDER_3D_BENCHMARK_PROFILE_CASES = ("d18", "d38")
HIGH_ORDER_CURVED_3D_SIMPLEX_CASES = ("curvedtet3d",)
HIGH_ORDER_MPI_MOTION_CASES = ("sloshing2d",)
HIGH_ORDER_CAPILLARY_PROJECTION_CASES = ("sloshing2d",)
HIGH_ORDER_CAPILLARY_RESPONSE_CASES = ("capillaryarc2d",)
HIGH_ORDER_CAPILLARY_BALANCE_CASES = ("capillaryarc2d",)
HIGH_ORDER_CAPILLARY_DROPLET_EQUILIBRIUM_CASES = ("droplet2d",)
HIGH_ORDER_CAPILLARY_WAVE_CASES = ("capillarywave2d",)
HIGH_ORDER_VOLUME_CORRECTED_MOTION_CASES = ("sloshing2d",)
QUALIFICATION_REPORT_SCHEMA_VERSION = 2
HIGH_ORDER_SYNTHETIC_CASES = {
    "capillaryarc2d",
    "capillarywave2d",
    "curvedtet3d",
    "droplet2d",
}
GENERALIZED_ALPHA_STAGE_STATE_SOURCE = (
    "reconstructed_generalized_alpha_first_order_stage_from_adjacent_endpoint_VTK"
)
GENERALIZED_ALPHA_REN_E_PREDICTION_SOURCE = (
    "generalized_alpha_stage_LinearCorner_generated_fragment_normal_at_phi_zero_wall_roots"
)
GENERALIZED_ALPHA_REN_E_VELOCITY_SOURCE = (
    "generalized_alpha_stage_Q1_velocity_and_generated_fragment_normal_at_phi_zero_wall_roots"
)
CUT_CONTEXT_VOLUME_RE = re.compile(r"active_side_volume=([-+0-9.eE]+)")
CUT_ASSEMBLY_VOLUME_RE = re.compile(r"(?<!_)active_wet_volume=([-+0-9.eE]+)")
KEY_VALUE_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)=('[^']*'|\"[^\"]*\"|[^\s\]]+)")

FREE_SURFACE_PRESSURE_REPRESENTABILITY_OPERATOR_TAG = (
    "equations_diagnostic_ns_free_surface_pressure_representability_pair"
)
FREE_SURFACE_CONSERVATIVE_BALANCE_OPERATOR_TAGS = frozenset({
    "equations_diagnostic_ns_free_surface_pressure_virtual_work",
    "equations_diagnostic_ns_free_surface_surface_energy_virtual_work",
    "equations_diagnostic_ns_free_surface_conservative_balance",
    FREE_SURFACE_PRESSURE_REPRESENTABILITY_OPERATOR_TAG,
})
JIT_MINUS_SHAPE_RE = re.compile(
    r"minus\[qpts=([^,\]]+),test=([^,\]]+),trial=([^\]]+)\]"
)
JIT_PLUS_SHAPE_RE = re.compile(
    r"plus\[qpts=([^,\]]+),test=([^,\]]+),trial=([^\]]+)\]"
)
RANK_RE = re.compile(r"\[R([0-9]+)\]")
COMPONENT_NORM_RE = re.compile(
    r"\[(.*?) norm=([-+0-9.eE]+) mean=([-+0-9.eE]+)"
    r"(?: min=([-+0-9.eE]+) max=([-+0-9.eE]+))?\]"
)
JACOBIAN_COMPONENT_NORM_RE = re.compile(
    r"\[(.*?) fd=([-+0-9.eE]+) total_err=([-+0-9.eE]+)"
    r" matrix_err=([-+0-9.eE]+)\]"
)
JACOBIAN_COMPONENT_DETAIL_RE = re.compile(
    r"\[(.*?) base=([-+0-9.eE]+) perturbed=([-+0-9.eE]+)"
    r" fd=([-+0-9.eE]+) matrix=([-+0-9.eE]+) full=([-+0-9.eE]+)"
    r" matrix_err=([-+0-9.eE]+) total_err=([-+0-9.eE]+)"
    r" sign_flip_err=([-+0-9.eE]+)\]"
)
JACOBIAN_TOP_MISMATCH_RE = re.compile(
    r"\[(.*?) fd=([-+0-9.eE]+) jv=([-+0-9.eE]+) err=([-+0-9.eE]+)\]"
)
DOUBLE_BAR_VALUE_RE = re.compile(r"\|\|([^|]+)\|\|=([-+0-9.eE]+)")
VECTOR_COMPONENT_LABEL_RE = re.compile(r"label=('[^']*'|\"[^\"]*\"|[^\s\]]+)")
BLOCK_SUMMARY_RE = re.compile(r"(?P<name>[^;{}]+)\{(?P<body>[^}]*)\}")
JACOBIAN_COMPONENT_BLOCK_MIN_DENOMINATOR = 1.0e-12
LINEAR_SOLVER_RE = re.compile(
    r"SimulationBuilder: linear solver method=(?P<method>\S+)"
    r" preconditioner=(?P<preconditioner>\S+)"
    r" rel_tol=(?P<rel_tol>[-+0-9.eE]+)"
    r" abs_tol=(?P<abs_tol>[-+0-9.eE]+)"
    r" max_iter=(?P<max_iter>[0-9]+)"
    r"(?: block_layout=(?P<block_layout>\[[^\]]+\]))?"
    r"(?: saddle_point=\((?P<saddle_momentum>[0-9]+),(?P<saddle_constraint>[0-9]+)\))?"
)
TIME_STEPPING_RE = re.compile(
    r"Time stepping: Number_of_time_steps=(?P<number_of_time_steps>[0-9]+)"
    r" Time_step_size=(?P<time_step_size>[-+0-9.eE]+)"
)
TRANSIENT_SOLVE_RE = re.compile(
    r"Transient solve: t0=(?P<t0>[-+0-9.eE]+)"
    r" dt=(?P<dt>[-+0-9.eE]+)"
    r" t_end=(?P<t_end>[-+0-9.eE]+)"
    r" max_steps=(?P<max_steps>[0-9]+)"
    r" scheme=(?P<scheme>\S+)"
    r" rho_inf=(?P<rho_inf>[-+0-9.eE]+)"
    r"(?: pde_udot_init=(?P<pde_udot_init>[01]))?"
    r"(?: last_step_absorb_fraction="
    r"(?P<last_step_absorb_fraction>[-+0-9.eE]+))?"
    r" newton\(max_it=(?P<newton_max_it>[0-9]+),"
    r" min_it=(?P<newton_min_it>[0-9]+),"
    r" abs_tol=(?P<newton_abs_tol>[-+0-9.eE]+),"
    r" rel_tol=(?P<newton_rel_tol>[-+0-9.eE]+)\)"
)
TIMELOOP_ADAPTIVE_RE = re.compile(
    r"TimeLoop adaptive controller enabled:"
    r" min_dt=(?P<min_dt>[-+0-9.eE]+)"
    r" max_dt=(?P<max_dt>[-+0-9.eE]+)"
    r" max_retries=(?P<max_retries>[0-9]+)"
    r" decrease_factor=(?P<decrease_factor>[-+0-9.eE]+)"
    r" increase_factor=(?P<increase_factor>[-+0-9.eE]+)"
    r" target_newton_iterations=(?P<target_newton_iterations>[0-9]+)"
    r" max_steps=(?P<max_steps>[0-9]+)"
)
TIMELOOP_NONLINEAR_RE = re.compile(
    r"TimeLoop: nonlinear_done step=(?P<step>[0-9]+)"
    r" time=(?P<time>[-+0-9.eE]+)"
    r" converged=(?P<converged>[01])"
    r" iters=(?P<nonlinear_iterations>[0-9]+)"
    r" \|\|r\|\|=(?P<residual>[-+0-9.eE]+)"
    r"(?: outer_iters=(?P<outer_iterations>[0-9]+)"
    r" inner_iters_total=(?P<inner_iterations_total>[0-9]+)"
    r" outer_state_change_norm="
    r"(?P<outer_state_change_norm>[-+0-9.eE]+))?"
    r" \|\|r_field\|\|=(?P<field_residual>[-+0-9.eE]+)"
    r" \|\|r_aux\|\|=(?P<aux_residual>[-+0-9.eE]+)"
    r" \(linear: converged=(?P<linear_converged>[01])"
    r" iters=(?P<linear_iterations>[0-9]+)"
    r" rel=(?P<linear_relative_residual>[-+0-9.eE]+)\)"
)
TIMELOOP_ACCEPTED_RE = re.compile(
    r"TimeLoop: step_accepted step=(?P<step>[0-9]+)"
    r" time=(?P<time>[-+0-9.eE]+)"
    r" dt=(?P<dt>[-+0-9.eE]+)"
)
TIMELOOP_REJECTED_RE = re.compile(
    r"TimeLoop: step_rejected step=(?P<step>[0-9]+)"
    r" time=(?P<time>[-+0-9.eE]+)"
    r" dt=(?P<dt>[-+0-9.eE]+)"
    r" reason=(?P<reason>\S+)"
    r" \(newton: converged=(?P<converged>[01])"
    r" iters=(?P<nonlinear_iterations>[0-9]+)"
    r"(?: outer_iters=(?P<outer_iterations>[0-9]+)"
    r" inner_iters_total=(?P<inner_iterations_total>[0-9]+))?"
    r" \|\|r\|\|=(?P<residual>[-+0-9.eE]+)"
    r" \|\|r_field\|\|=(?P<field_residual>[-+0-9.eE]+)"
    r" \|\|r_aux\|\|=(?P<aux_residual>[-+0-9.eE]+)\)"
)
TIMELOOP_DT_UPDATED_RE = re.compile(
    r"TimeLoop: dt_updated step=(?P<step>[0-9]+)"
    r" attempt=(?P<attempt>[0-9]+)"
    r" old_dt=(?P<old_dt>[-+0-9.eE]+)"
    r" new_dt=(?P<new_dt>[-+0-9.eE]+)"
)
VTK_WRITE_RE = re.compile(r"Wrote VTK: (?P<path>.+)$")
ASSEMBLY_TIMING_HEADER_RE = re.compile(
    r"assembleOperator TIMING \(rank (?P<rank>[0-9]+), op='(?P<op>[^']+)'\)"
)
ASSEMBLY_TIMING_VALUE_RE = re.compile(
    r"^\s*(?P<label>[A-Za-z0-9 ()+]+):\s+"
    r"(?P<seconds>[-+0-9.eE]+)\s+s"
)
INTERIOR_FACE_TIMING_VALUE_RE = re.compile(
    r"([A-Za-z_][A-Za-z0-9_]*)=\s*([-+0-9.eE]+)"
)
STATIC_INTERFACE_HEIGHT = 0.53
CAPILLARY_ARC_CENTER_X = 0.5
CAPILLARY_ARC_CENTER_Y = -0.3
CAPILLARY_ARC_RADIUS = 0.8
CAPILLARY_DROPLET_CENTER_X = 0.5
CAPILLARY_DROPLET_CENTER_Y = 0.5
CAPILLARY_DROPLET_RADIUS = 0.3
CAPILLARY_WAVE_BASE_HEIGHT = 0.5
CAPILLARY_WAVE_AMPLITUDE = 0.004
CAPILLARY_WAVE_WAVELENGTH = 1.0
CAPILLARY_WAVE_DENSITY = 998.2
CAPILLARY_WAVE_DEPTH = CAPILLARY_WAVE_BASE_HEIGHT
CAPILLARY_WAVE_MINIMUM_FREQUENCY_PHASE_SPAN = 0.75
# A closed, impermeable capillary-wave tank has no physical liquid-volume
# flux.  Keep the accepted-state temporal drift gate distinct from the
# same-state VTK-versus-cut-context consistency check.  The fixed one-part in
# 100,000 relative budget is intentionally tighter than the wave-amplitude
# gate and is not inferred from an observed run.
CAPILLARY_WAVE_MAX_TEMPORAL_LIQUID_VOLUME_RELATIVE_DRIFT = 1.0e-5
# The saved-state energy diagnostic is intentionally an output-space proxy,
# not the quadrature-exact discrete energy.  Its interface is VTK-linearized
# on Q1 cells and its kinetic density is formed at vertices before clipping.
# Permit one part in 10,000 of the initial proxy energy for either a positive
# accepted-step increment or a transient value above the initialized state.
# The bound is fixed for the final static/dynamic/wave qualification matrix and
# remains separate from any claim of a discrete energy theorem.
FREE_SURFACE_ENERGY_MAX_POSITIVE_STEP_INCREMENT_RELATIVE = 1.0e-4
FREE_SURFACE_ENERGY_MAX_ABOVE_INITIAL_RELATIVE = 1.0e-4
# The accepted static sessile state must leave at most five percent of the
# surface-area plus Young-wall load outside the constrained pressure-gradient
# range.  This is intentionally opt-in: moving-interface qualifications report
# the same diagnostic without rejecting physically necessary capillary loads.
STATIC_PRESSURE_REPRESENTABILITY_MAX_RELATIVE_DISTANCE = 5.0e-2
# The generated FS16 mini-decks initialize phi in physical length units on a
# one-metre tank.  Their P1 transport solve is allowed one part per million of
# length as coefficient-representability slack while sign preservation remains
# an independent, nearly exact invariant.  This is a qualification-deck
# contract, not a relaxation of the solver's global defaults.
LEVEL_SET_CHARACTERISTIC_LENGTH = 1.0
MAX_BOUND_REPRESENTABILITY_SLACK_OVER_LENGTH = 1.0e-6
DEFAULT_BOUND_REPRESENTABILITY_SLACK = (
    MAX_BOUND_REPRESENTABILITY_SLACK_OVER_LENGTH *
    LEVEL_SET_CHARACTERISTIC_LENGTH
)
BOUND_SIGN_TOLERANCE = 1.0e-12
STATE_SYNC_CUT_CONTEXT_PROVENANCES = {
    "accepted",
    "residual",
    "jacobian",
    "jacobian_and_residual",
    "line_search_trial",
    "restored",
    "final_residual",
}
VECTOR_CUT_CONTEXT_PROVENANCES = {
    "before_physics_solve",
    "accepted_step",
    "steady_initial",
    "steady_accepted",
}


def point_array(mesh: pv.DataSet, name: str) -> np.ndarray:
    if name not in mesh.point_data:
        names = ", ".join(sorted(mesh.point_data.keys()))
        raise ValueError(f"missing point array {name!r}; found: {names}")
    return np.asarray(mesh.point_data[name])


def global_node_ids(mesh: pv.DataSet) -> np.ndarray:
    for name in ("GlobalNodeID", "GlobalVertexID"):
        if name in mesh.point_data:
            return np.asarray(mesh.point_data[name], dtype=np.int64)
    raise ValueError("missing GlobalNodeID or GlobalVertexID point array")


def result_indices_by_initial_gid(initial: pv.DataSet,
                                  result: pv.DataSet) -> np.ndarray:
    result_first_index: dict[int, int] = {}
    for index, gid in enumerate(global_node_ids(result)):
        result_first_index.setdefault(int(gid), index)

    missing = []
    indices = []
    for gid in global_node_ids(initial):
        key = int(gid)
        if key not in result_first_index:
            missing.append(key)
        else:
            indices.append(result_first_index[key])
    if missing:
        raise ValueError(f"result omits {len(missing)} initial node ids")
    return np.asarray(indices, dtype=np.int64)


def point_scalar_in_initial_gid_order(initial: pv.DataSet,
                                      result: pv.DataSet,
                                      name: str) -> np.ndarray:
    """Map an output scalar onto the initial mesh by reconciled global ID.

    Parallel VTK files may repeat a point in multiple pieces.  Repeated global
    IDs are valid only when every copy has the same coordinates and scalar up
    to a tight, scale-conditioned floating-point tolerance.  When VTK point
    ghost metadata is available, select a non-ghost copy and require every
    requested global ID to have at least one such copy.  This prevents a stale
    or inconsistent halo value from silently determining a physical-history
    gate while preserving the unique-ID serial path.
    """
    initial_gids = global_node_ids(initial).reshape(-1)
    result_gids = global_node_ids(result).reshape(-1)
    if (initial_gids.size != initial.n_points or
            len(set(map(int, initial_gids))) != initial_gids.size):
        raise ValueError("initial GlobalNodeID values are missing or duplicated")
    if result_gids.size != result.n_points:
        raise ValueError("output GlobalNodeID values are missing")

    coordinates = np.asarray(result.points, dtype=float)
    if (coordinates.shape != (result.n_points, 3) or
            not np.isfinite(coordinates).all()):
        raise ValueError("output point coordinates are missing or non-finite")
    values = np.asarray(point_array(result, name), dtype=float).reshape(-1)
    if values.size != result.n_points:
        raise ValueError(
            f"output scalar {name!r} does not contain one value per point")
    if not np.isfinite(values).all():
        raise ValueError(f"output scalar {name!r} contains non-finite values")

    ghost_flags: np.ndarray | None = None
    if "vtkGhostType" in result.point_data:
        raw_ghost_flags = np.asarray(result.point_data["vtkGhostType"])
        if (raw_ghost_flags.size != result.n_points or
                not np.issubdtype(raw_ghost_flags.dtype, np.integer)):
            raise ValueError("output vtkGhostType point metadata is invalid")
        ghost_flags = raw_ghost_flags.reshape(-1).astype(np.int64)
        if np.any(ghost_flags < 0) or np.any(ghost_flags > 255):
            raise ValueError("output vtkGhostType point metadata is invalid")

    indices_by_gid: dict[int, list[int]] = {}
    for index, gid in enumerate(result_gids):
        indices_by_gid.setdefault(int(gid), []).append(index)

    duplicate_ulps = 64.0

    def tight_tolerance(samples: np.ndarray) -> float:
        scale = max(1.0, float(np.max(np.abs(samples))))
        return duplicate_ulps * np.finfo(float).eps * scale

    by_gid: dict[int, float] = {}
    for gid, indices in indices_by_gid.items():
        preferred_indices = indices
        if ghost_flags is not None:
            preferred_indices = [
                index for index in indices if ghost_flags[index] == 0
            ]
            if not preferred_indices:
                raise ValueError(
                    f"output GlobalNodeID {gid} has only ghost copies; "
                    "owned coverage is ambiguous")
        selected_index = preferred_indices[0]

        group_coordinates = coordinates[indices]
        coordinate_tolerance = tight_tolerance(group_coordinates)
        with np.errstate(over="ignore", invalid="ignore"):
            coordinate_difference = float(np.max(np.abs(
                group_coordinates - coordinates[selected_index]
            )))
        if (not math.isfinite(coordinate_difference) or
                coordinate_difference > coordinate_tolerance):
            raise ValueError(
                f"output GlobalNodeID {gid} has inconsistent coordinates "
                f"across pieces (difference={coordinate_difference:.16g}, "
                f"tolerance={coordinate_tolerance:.16g})")

        group_values = values[indices]
        value_tolerance = tight_tolerance(group_values)
        with np.errstate(over="ignore", invalid="ignore"):
            value_difference = float(np.max(np.abs(
                group_values - values[selected_index]
            )))
        if (not math.isfinite(value_difference) or
                value_difference > value_tolerance):
            raise ValueError(
                f"output GlobalNodeID {gid} has inconsistent scalar {name!r} "
                f"across pieces (difference={value_difference:.16g}, "
                f"tolerance={value_tolerance:.16g})")

        by_gid[gid] = float(values[selected_index])

    missing = [int(gid) for gid in initial_gids if int(gid) not in by_gid]
    if missing:
        raise ValueError(
            f"output omits {len(missing)} initial GlobalNodeID value(s); "
            f"first={missing[0]}")
    return np.asarray([by_gid[int(gid)] for gid in initial_gids], dtype=float)


def point_field_in_initial_gid_order(initial: pv.DataSet,
                                     result: pv.DataSet,
                                     name: str) -> np.ndarray:
    """Map a scalar or multi-component point field through GlobalNodeID.

    The scalar mapper above owns the duplicate-piece and ghost-copy contract.
    Reusing it component by component keeps generalized-alpha reconstruction
    subject to exactly the same strict ownership and consistency checks.
    """
    values = np.asarray(point_array(result, name), dtype=float)
    if values.ndim == 1:
        return point_scalar_in_initial_gid_order(initial, result, name)
    if values.ndim != 2 or values.shape[0] != result.n_points or not values.shape[1]:
        raise ValueError(
            f"output point field {name!r} does not contain one value per point")
    if not np.isfinite(values).all():
        raise ValueError(f"output point field {name!r} contains non-finite values")

    component_dataset = result.copy(deep=True)
    component_name = "__svmp_stage_reconstruction_component"
    components = []
    for component in range(values.shape[1]):
        component_dataset.point_data[component_name] = values[:, component]
        components.append(point_scalar_in_initial_gid_order(
            initial, component_dataset, component_name))
    return np.column_stack(components)


def generalized_alpha_first_order_stage_parameters(
        case_dir: Path,
        transient_solve: dict[str, Any] | None = None,
        ) -> dict[str, Any]:
    """Return the parsed first-order generalized-alpha stage parameters."""
    source = "parsed_solver_transient_diagnostics"
    if isinstance(transient_solve, dict) and transient_solve:
        scheme = transient_solve.get("scheme")
        rho_inf = transient_solve.get("rho_inf")
        normalized_scheme = re.sub(r"[^a-z0-9]", "", str(scheme).lower())
        if normalized_scheme != "generalizedalpha":
            raise ValueError(
                f"dynamic contact stage reconstruction requires GeneralizedAlpha; "
                f"parsed scheme is {scheme!r}")
    else:
        # The OOP transient path uses first-order generalized-alpha; generated
        # standalone instrumentation tests do not have a solver log, so parse
        # the same spectral-radius input directly from the deck.
        try:
            root = ET.parse(case_dir / "solver.xml").getroot()
        except (OSError, ET.ParseError) as exc:
            raise ValueError(f"cannot parse generalized-alpha controls: {exc}") from exc
        rho_text = root.findtext(
            "GeneralSimulationParameters/Spectral_radius_of_infinite_time_step")
        if rho_text is None:
            raise ValueError("missing Spectral_radius_of_infinite_time_step")
        rho_inf = rho_text
        scheme = "GeneralizedAlpha"
        source = "parsed_solver_xml_generalized_alpha_spectral_radius"

    try:
        rho = float(rho_inf)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid generalized-alpha rho_inf {rho_inf!r}") from exc
    if not math.isfinite(rho) or not 0.0 <= rho <= 1.0:
        raise ValueError("generalized-alpha rho_inf must be finite and in [0,1]")
    return {
        "scheme": "GeneralizedAlpha",
        "rho_inf": rho,
        "alpha_f": 1.0 / (1.0 + rho),
        "parameter_source": source,
    }


def reconstruct_generalized_alpha_first_order_stage(
        initial: pv.DataSet,
        previous_endpoint: pv.DataSet,
        current_endpoint: pv.DataSet,
        alpha_f: float,
        differential_fields: tuple[str, ...] = ("phi", "Velocity"),
        ) -> pv.DataSet:
    """Reconstruct u_(n+alpha_f) from adjacent saved endpoint states."""
    if not math.isfinite(alpha_f) or not 0.0 < alpha_f <= 1.0:
        raise ValueError("generalized-alpha alpha_f must be finite and in (0,1]")
    stage = initial.copy(deep=True)
    for name in differential_fields:
        previous = point_field_in_initial_gid_order(
            initial, previous_endpoint, name)
        current = point_field_in_initial_gid_order(
            initial, current_endpoint, name)
        # The synthetic 2-D input mesh follows VTK's conventional three-slot
        # vector storage, while the solved FE Velocity field is written with
        # its two active components.  Reconcile that first-interval storage
        # mismatch only when the omitted trailing input components are
        # identically zero.  Any nonzero discarded state, row mismatch, or
        # opposite component-count transition remains fail-closed.
        if (name == "Velocity" and previous.ndim == 2 and
                current.ndim == 2 and
                previous.shape[0] == current.shape[0] and
                previous.shape[1] > current.shape[1]):
            omitted = previous[:, current.shape[1]:]
            if np.all(omitted == 0.0):
                previous = previous[:, :current.shape[1]]
        if previous.shape != current.shape:
            raise ValueError(
                f"adjacent endpoint field {name!r} shapes differ: "
                f"{previous.shape} != {current.shape}")
        stage.point_data[name] = (
            (1.0 - alpha_f) * previous + alpha_f * current)
    return stage


def cell_measure(mesh: pv.DataSet) -> np.ndarray:
    sized = mesh.compute_cell_sizes(length=False, area=True, volume=True)
    for name in ("Volume", "Area"):
        if name in sized.cell_data:
            values = np.asarray(sized.cell_data[name], dtype=float)
            if np.any(np.abs(values) > 0.0):
                return values
    raise ValueError("mesh cell sizes do not include nonzero area or volume")


def text(root: ET.Element, path: str) -> str:
    element = root.find(path)
    if element is None or element.text is None:
        return ""
    return element.text.strip()


def require_text(root: ET.Element, path: str, expected: str) -> None:
    value = text(root, path)
    if value != expected:
        raise ValueError(f"{path} is {value!r}, expected {expected!r}")


def set_text(parent: ET.Element, name: str, value: str) -> None:
    element = parent.find(name)
    if element is None:
        element = ET.SubElement(parent, name)
    element.text = value


def set_linear_algebra_backend(solver: ET.Element,
                               backend: str,
                               preconditioner: str = "none") -> None:
    element = solver.find("Linear_algebra")
    if element is None:
        element = ET.SubElement(solver, "Linear_algebra")
    element.set("type", backend)
    set_text(element, "Preconditioner", preconditioner)


def default_preconditioner_for_backend(backend: str) -> str:
    if backend.strip().lower() == "fsils":
        return "fsils"
    return "none"


def free_surface_bc(root: ET.Element) -> ET.Element:
    for equation in root.findall("Add_equation"):
        if equation.attrib.get("type") != "fluid":
            continue
        for bc in equation.findall("Add_BC"):
            if bc.attrib.get("name") == "free_surface":
                return bc
    raise ValueError("missing fluid free-surface boundary condition")


def level_set_equation(root: ET.Element) -> ET.Element:
    for equation in root.findall("Add_equation"):
        if equation.attrib.get("type") == "level_set":
            return equation
    raise ValueError("missing level-set equation")


def fluid_equation(root: ET.Element) -> ET.Element:
    for equation in root.findall("Add_equation"):
        if equation.attrib.get("type") == "fluid":
            return equation
    raise ValueError("missing fluid equation")


def navier_stokes_linear_solver(root: ET.Element) -> ET.Element:
    solvers = fluid_equation(root).findall("LS")
    for solver in solvers:
        if solver.attrib.get("type") == "NS":
            return solver
    for solver in solvers:
        if solver.find("NS_GM_max_iterations") is not None:
            return solver
    if len(solvers) == 1:
        return solvers[0]
    raise ValueError("missing fluid NS linear solver block")


def configure_solver(solver_xml: Path,
                     steps: int,
                     time_step_size: float | None = None,
                     disable_cut_stabilization: bool = False,
                     max_nonlinear_iterations: int | None = None,
                     linear_relative_tolerance: float | None = None,
                     linear_absolute_tolerance: float | None = None,
                     linear_max_iterations: int | None = None,
                     linear_krylov_space_dimension: int | None = None,
                     ns_gm_max_iterations: int | None = None,
                     ns_cg_max_iterations: int | None = None,
                     ns_gm_tolerance: float | None = None,
                     ns_cg_tolerance: float | None = None,
                     linear_solver_type: str | None = None,
                     linear_algebra_backend: str | None = None,
                     linear_preconditioner: str | None = None,
                     disable_coupled_outer_fgmres: bool = False,
                     disable_cut_metadata_scale: bool = False,
                     disable_velocity_extension: bool = False,
                     disable_vtk_output: bool = False,
                     final_output_only: bool = False,
                     vtk_save_increment: int | None = None,
                     start_saving_after_step: int | None = None,
                     generated_interface_geometry: str | None = None,
                     implicit_cut_quadrature_backend: str | None = None,
                     implicit_cut_fallback_policy: str | None = None,
                     required_implicit_cut_backend_qualification: str | None = None,
                     implicit_cut_root_tolerance: float | None = None,
                     implicit_cut_max_subdivision_depth: int | None = None,
                     generated_interface_quadrature_order: int | None = None,
                     interface_quadrature_order: int | None = None,
                     volume_quadrature_order: int | None = None,
                     cut_cell_velocity_gradient_penalty: float | None = None,
                     cut_cell_pressure_gradient_penalty: float | None = None,
                     cut_cell_pressure_stabilization_policy: str | None = None,
                     active_domain: str = "LevelSetNegative",
                     surface_tension: float | None = None,
                     capillary_force_form: str = "surface_stress",
                     prescribed_capillary_curvature: float | None = None,
                     wet_extension_advection_velocity_method: str | None = None,
                     projected_curvature_field: str | None = None,
                     curvature_projection_cadence_steps: int | None = None,
                     curvature_projection_max_normalized_fit_residual: float | None = None,
                     curvature_projection_max_neighbor_fallback_vertices: int | None = None,
                     curvature_projection_max_zero_fallback_vertices: int | None = None,
                     curvature_projection_supplemental_sample_weight: float | None = None,
                     curvature_projection_recovery_mode: str | None = None,
                     curvature_projection_kinematic_area_gradient_filter_coefficient:
                     float | None = None,
                     curvature_projection_narrow_band_width: float | None = None,
                     curvature_projection_smoothing_iterations: int | None = None,
                     curvature_projection_smoothing_relaxation: float | None = None,
                     curvature_projection_smoothing_mode: str | None = None,
                     enable_static_capillary_equilibrium_initialization:
                     bool | None = None,
                     static_capillary_volume_tolerance: float | None = None,
                     static_capillary_projected_gradient_tolerance:
                     float | None = None,
                     static_capillary_pressure_representability_max_residual_norm:
                     float | None = None,
                     static_capillary_pressure_representability_max_relative_distance:
                     float | None = None,
                     static_capillary_physical_equilibrium_max_residual_norm:
                     float | None = None,
                     static_capillary_constant_pressure_kkt_max_residual_norm:
                     float | None = None,
                     static_capillary_constant_pressure_kkt_max_relative_distance:
                     float | None = None,
                     static_capillary_finite_difference_relative_step:
                     float | None = None,
                     static_capillary_max_iterations: int | None = None,
                     static_capillary_max_topology_epoch_transitions:
                     int | None = None,
                     static_capillary_limited_memory_history_size:
                     int | None = None,
                     static_capillary_limited_memory_curvature_tolerance:
                     float | None = None,
                     enable_reinitialization: bool | None = None,
                     reinitialization_cadence_steps: int | None = None,
                     enable_volume_correction: bool | None = None,
                     volume_correction_cadence_steps: int | None = None,
                     volume_correction_use_initial_volume: bool | None = None,
                     volume_correction_tolerance: float | None = None,
                     volume_correction_max_iterations: int | None = None,
                     volume_correction_maximum_cumulative_interface_displacement_fraction:
                     float | None = None) -> None:
    tree = ET.parse(solver_xml)
    root = tree.getroot()
    general = root.find("GeneralSimulationParameters")
    if general is None:
        raise ValueError("missing GeneralSimulationParameters")
    for parent in root.iter():
        for linear_algebra in list(parent.findall("Linear_algebra")):
            if linear_algebra.attrib.get("type", "").strip().lower() == "eigen":
                parent.remove(linear_algebra)

    set_text(general, "Number_of_time_steps", str(steps))
    set_text(general, "Save_results_to_VTK_format", "false" if disable_vtk_output else "true")
    set_text(general, "Combine_time_series", "false" if disable_vtk_output else "true")
    set_text(general, "Name_prefix_of_saved_VTK_files", "result")
    save_increment = vtk_save_increment if vtk_save_increment is not None else 1
    start_step = start_saving_after_step if start_saving_after_step is not None else 1
    if final_output_only:
        save_increment = steps
        start_step = steps
    set_text(general, "Increment_in_saving_VTK_files", str(save_increment))
    set_text(general, "Start_saving_after_time_step", str(start_step))
    set_text(general, "Increment_in_saving_restart_files", str(steps))
    if time_step_size is not None:
        set_text(general, "Time_step_size", f"{time_step_size:.16g}")

    if max_nonlinear_iterations is not None:
        for equation in root.findall("Add_equation"):
            set_text(equation, "Max_iterations", str(max_nonlinear_iterations))

    needs_ns_solver = (
        linear_relative_tolerance is not None or
        linear_absolute_tolerance is not None or
        linear_max_iterations is not None or
        linear_krylov_space_dimension is not None or
        ns_gm_max_iterations is not None or
        ns_cg_max_iterations is not None or
        ns_gm_tolerance is not None or
        ns_cg_tolerance is not None or
        linear_solver_type is not None or
        linear_algebra_backend is not None or
        linear_preconditioner is not None or
        disable_coupled_outer_fgmres
    )
    ns_solver = navier_stokes_linear_solver(root) if needs_ns_solver else None
    if linear_solver_type is not None:
        assert ns_solver is not None
        ns_solver.set("type", linear_solver_type)
        if (linear_solver_type.strip().lower() != "direct" and
                linear_algebra_backend is None):
            set_linear_algebra_backend(
                ns_solver,
                "fsils",
                linear_preconditioner or "fsils",
            )
    if linear_algebra_backend is not None:
        assert ns_solver is not None
        set_linear_algebra_backend(
            ns_solver,
            linear_algebra_backend,
            linear_preconditioner or
            default_preconditioner_for_backend(linear_algebra_backend),
        )
    elif linear_preconditioner is not None:
        assert ns_solver is not None
        set_linear_algebra_backend(ns_solver, "fsils", linear_preconditioner)
    if ns_solver is not None:
        linear_algebra = ns_solver.find("Linear_algebra")
        configured_backend = (
            linear_algebra.attrib.get("type", "").strip().lower()
            if linear_algebra is not None else ""
        )
        configured_method = ns_solver.attrib.get("type", "").strip().lower()
        if configured_backend == "fsils" and configured_method == "direct":
            raise ValueError(
                "FSILS does not support the Direct fluid linear-solver method; "
                "select NS or GMRES"
            )
    if linear_relative_tolerance is not None:
        assert ns_solver is not None
        set_text(ns_solver, "Tolerance", f"{linear_relative_tolerance:.16g}")
    if linear_absolute_tolerance is not None:
        assert ns_solver is not None
        set_text(ns_solver, "Absolute_tolerance", f"{linear_absolute_tolerance:.16g}")
    if linear_max_iterations is not None:
        assert ns_solver is not None
        set_text(ns_solver, "Max_iterations", str(linear_max_iterations))
    if linear_krylov_space_dimension is not None:
        if linear_krylov_space_dimension <= 0:
            raise ValueError("linear Krylov space dimension must be positive")
        assert ns_solver is not None
        set_text(ns_solver, "Krylov_space_dimension",
                 str(linear_krylov_space_dimension))
    if ns_gm_max_iterations is not None:
        assert ns_solver is not None
        set_text(ns_solver, "NS_GM_max_iterations", str(ns_gm_max_iterations))
    if ns_cg_max_iterations is not None:
        assert ns_solver is not None
        set_text(ns_solver, "NS_CG_max_iterations", str(ns_cg_max_iterations))
    if ns_gm_tolerance is not None:
        assert ns_solver is not None
        set_text(ns_solver, "NS_GM_tolerance", f"{ns_gm_tolerance:.16g}")
    if ns_cg_tolerance is not None:
        assert ns_solver is not None
        set_text(ns_solver, "NS_CG_tolerance", f"{ns_cg_tolerance:.16g}")

    free_surface = free_surface_bc(root)
    require_text(free_surface, "Implementation", "UnfittedLevelSet")
    active_domain = normalized_active_domain(active_domain)
    require_text(free_surface, "Active_domain", active_domain)
    require_text(free_surface, "Active_domain_method", "CutVolume")
    if disable_cut_metadata_scale:
        set_text(free_surface, "Use_cut_metadata_scale", "false")
    else:
        require_text(free_surface, "Use_cut_metadata_scale", "true")
    if disable_cut_stabilization:
        set_text(free_surface, "Enable_cut_cell_stabilization", "false")
    else:
        require_text(free_surface, "Enable_cut_cell_stabilization", "true")
    if disable_velocity_extension:
        set_text(free_surface, "Enable_velocity_extension", "false")
    else:
        extension_value = text(
            free_surface, "Enable_velocity_extension"
        ).strip().lower()
        if extension_value in {"true", "1", "yes", "on"}:
            raise ValueError(
                "the same-field dry-domain velocity extension is retired; "
                "remove it from the case or pass --disable-velocity-extension "
                "when replaying an archived input"
            )
    if generated_interface_geometry is not None:
        set_text(free_surface, "Generated_interface_geometry", generated_interface_geometry)
    if implicit_cut_quadrature_backend is not None:
        set_text(free_surface, "Implicit_cut_quadrature_backend", implicit_cut_quadrature_backend)
    if implicit_cut_fallback_policy is not None:
        set_text(free_surface, "Implicit_cut_fallback_policy", implicit_cut_fallback_policy)
    if required_implicit_cut_backend_qualification is not None:
        set_text(
            free_surface,
            "Required_implicit_cut_backend_qualification",
            required_implicit_cut_backend_qualification,
        )
    if implicit_cut_root_tolerance is not None:
        set_text(free_surface, "Implicit_cut_root_tolerance", f"{implicit_cut_root_tolerance:.16g}")
    if implicit_cut_max_subdivision_depth is not None:
        set_text(free_surface, "Implicit_cut_max_subdivision_depth", str(implicit_cut_max_subdivision_depth))
    if generated_interface_quadrature_order is not None:
        set_text(free_surface, "Generated_interface_quadrature_order", str(generated_interface_quadrature_order))
    if interface_quadrature_order is not None:
        set_text(free_surface, "Interface_quadrature_order", str(interface_quadrature_order))
    if volume_quadrature_order is not None:
        set_text(free_surface, "Volume_quadrature_order", str(volume_quadrature_order))
    if cut_cell_velocity_gradient_penalty is not None:
        set_text(
            free_surface,
            "Cut_cell_velocity_gradient_penalty",
            f"{cut_cell_velocity_gradient_penalty:.16g}",
        )
    if cut_cell_pressure_gradient_penalty is not None:
        set_text(
            free_surface,
            "Cut_cell_pressure_gradient_penalty",
            f"{cut_cell_pressure_gradient_penalty:.16g}",
        )
    pressure_stabilization_policy_names = {
        "enabled": "Enabled",
        "incremental": "Incremental",
        "disabled": "Disabled",
        "disabled_for_refreshed_frozen_high_order":
            "DisabledForRefreshedFrozenHighOrder",
    }
    if cut_cell_pressure_stabilization_policy is not None:
        if (cut_cell_pressure_stabilization_policy not in
                pressure_stabilization_policy_names):
            raise ValueError(
                "cut-cell pressure stabilization policy must be enabled, "
                "incremental, disabled, or "
                "disabled_for_refreshed_frozen_high_order")
        set_text(
            free_surface,
            "Cut_cell_pressure_stabilization_policy",
            pressure_stabilization_policy_names[
                cut_cell_pressure_stabilization_policy],
        )
    capillary_force_form_names = {
        "surface_stress": "SurfaceStress",
        "generated_curvature_traction": "GeneratedCurvatureTraction",
        "kinematic_area_gradient_traction": "KinematicAreaGradientTraction",
    }
    if capillary_force_form not in capillary_force_form_names:
        raise ValueError(
            "capillary force form must be surface_stress, "
            "generated_curvature_traction, or "
            "kinematic_area_gradient_traction")
    if prescribed_capillary_curvature is not None:
        if (not math.isfinite(prescribed_capillary_curvature) or
                prescribed_capillary_curvature <= 0.0):
            raise ValueError(
                "prescribed capillary curvature must be positive and finite")
        if capillary_force_form != "generated_curvature_traction":
            raise ValueError(
                "prescribed capillary curvature requires "
                "generated_curvature_traction")
    if surface_tension is not None:
        set_text(free_surface, "Surface_tension", f"{surface_tension:.16g}")
        if surface_tension > 0.0:
            set_text(
                free_surface,
                "Surface_tension_form",
                capillary_force_form_names[capillary_force_form],
            )
    if wet_extension_advection_velocity_method is not None:
        level_set = level_set_equation(root)
        constant_velocity = level_set.find("Constant_velocity")
        if constant_velocity is not None:
            level_set.remove(constant_velocity)
        set_text(level_set, "Velocity_source", "prescribed_data")
        set_text(level_set, "Velocity_field_name", "LevelSetAdvectionVelocity")
        set_text(level_set, "Auto_register_velocity_field", "true")
        set_text(level_set, "Use_wet_extension_advection_velocity", "true")
        set_text(level_set, "Source_velocity_field_name", "Velocity")
        set_text(
            level_set,
            "Wet_extension_advection_velocity_method",
            wet_extension_advection_velocity_method,
        )
    if (capillary_force_form == "generated_curvature_traction" and
            surface_tension is not None and surface_tension > 0.0 and
            projected_curvature_field and
            prescribed_capillary_curvature is not None):
        raise ValueError(
            "generated curvature traction requires exactly one curvature "
            "source, not both a field and a prescribed scalar")
    if (capillary_force_form == "kinematic_area_gradient_traction" and
            surface_tension is not None and surface_tension > 0.0):
        if not projected_curvature_field:
            raise ValueError(
                "kinematic area-gradient traction requires an explicit "
                "projected curvature field")
        if curvature_projection_recovery_mode != "kinematic_area_gradient":
            raise ValueError(
                "kinematic area-gradient traction requires the "
                "kinematic_area_gradient recovery mode")
        if curvature_projection_kinematic_area_gradient_filter_coefficient != 0.0:
            raise ValueError(
                "kinematic area-gradient traction requires an explicit zero "
                "area-gradient filter coefficient")
        if curvature_projection_smoothing_iterations not in (None, 0):
            raise ValueError(
                "kinematic area-gradient traction does not admit separate "
                "post-projection smoothing")
    if projected_curvature_field:
        level_set = level_set_equation(root)
        set_text(level_set, "Enable_curvature_projection", "true")
        set_text(level_set, "Projected_curvature_field", projected_curvature_field)
        if (capillary_force_form in {
                "generated_curvature_traction",
                "kinematic_area_gradient_traction",
        } or
                not (surface_tension is not None and surface_tension > 0.0)):
            set_text(free_surface, "Curvature_field", projected_curvature_field)
            set_text(free_surface, "Use_level_set_curvature", "false")
        else:
            curvature_field = free_surface.find("Curvature_field")
            if curvature_field is not None:
                free_surface.remove(curvature_field)
        if curvature_projection_cadence_steps is not None:
            set_text(
                level_set,
                "Curvature_projection_cadence_steps",
                str(curvature_projection_cadence_steps),
            )
        if curvature_projection_max_normalized_fit_residual is not None:
            set_text(
                level_set,
                "Curvature_projection_max_normalized_fit_residual",
                f"{curvature_projection_max_normalized_fit_residual:.16g}",
            )
        if curvature_projection_max_neighbor_fallback_vertices is not None:
            set_text(
                level_set,
                "Curvature_projection_max_neighbor_fallback_vertices",
                str(curvature_projection_max_neighbor_fallback_vertices),
            )
        if curvature_projection_max_zero_fallback_vertices is not None:
            set_text(
                level_set,
                "Curvature_projection_max_zero_fallback_vertices",
                str(curvature_projection_max_zero_fallback_vertices),
            )
        if curvature_projection_supplemental_sample_weight is not None:
            set_text(
                level_set,
                "Curvature_projection_supplemental_sample_weight",
                f"{curvature_projection_supplemental_sample_weight:.16g}",
            )
        if curvature_projection_recovery_mode is not None:
            set_text(
                level_set,
                "Curvature_projection_recovery_mode",
                curvature_projection_recovery_mode,
            )
        if (curvature_projection_kinematic_area_gradient_filter_coefficient
                is not None):
            set_text(
                level_set,
                "Curvature_projection_kinematic_area_gradient_filter_coefficient",
                f"{curvature_projection_kinematic_area_gradient_filter_coefficient:.16g}",
            )
        if curvature_projection_narrow_band_width is not None:
            set_text(
                level_set,
                "Curvature_projection_narrow_band_width",
                f"{curvature_projection_narrow_band_width:.16g}",
            )
        if curvature_projection_smoothing_iterations is not None:
            set_text(
                level_set,
                "Curvature_projection_smoothing_iterations",
                str(curvature_projection_smoothing_iterations),
            )
        if curvature_projection_smoothing_relaxation is not None:
            set_text(
                level_set,
                "Curvature_projection_smoothing_relaxation",
                f"{curvature_projection_smoothing_relaxation:.16g}",
            )
        if curvature_projection_smoothing_mode is not None:
            set_text(
                level_set,
                "Curvature_projection_smoothing_mode",
                curvature_projection_smoothing_mode,
            )
    if (capillary_force_form == "generated_curvature_traction" and
            surface_tension is not None and surface_tension > 0.0):
        if prescribed_capillary_curvature is not None:
            curvature_field = free_surface.find("Curvature_field")
            if curvature_field is not None:
                free_surface.remove(curvature_field)
            set_text(
                free_surface,
                "Curvature",
                f"{prescribed_capillary_curvature:.16g}",
            )
            set_text(free_surface, "Use_level_set_curvature", "false")
        elif not projected_curvature_field:
            raise ValueError(
                "generated curvature traction requires either "
                "--prescribed-capillary-curvature or "
                "--projected-curvature-field")
    if enable_static_capillary_equilibrium_initialization is not None:
        level_set = level_set_equation(root)
        set_text(
            level_set,
            "Enable_static_capillary_equilibrium_initialization",
            ("true" if enable_static_capillary_equilibrium_initialization
             else "false"),
        )
    if (static_capillary_finite_difference_relative_step is not None and
            (not math.isfinite(
                static_capillary_finite_difference_relative_step) or
             static_capillary_finite_difference_relative_step <= 0.0)):
        raise ValueError(
            "static capillary finite-difference relative step must be "
            "positive and finite")
    if (static_capillary_limited_memory_history_size is not None and
            (not isinstance(
                static_capillary_limited_memory_history_size, int) or
             isinstance(static_capillary_limited_memory_history_size, bool) or
             static_capillary_limited_memory_history_size < 0)):
        raise ValueError(
            "static capillary limited-memory history size must be "
            "a nonnegative integer")
    if (static_capillary_max_topology_epoch_transitions is not None and
            (not isinstance(
                static_capillary_max_topology_epoch_transitions, int) or
             isinstance(
                 static_capillary_max_topology_epoch_transitions, bool) or
             static_capillary_max_topology_epoch_transitions < 0)):
        raise ValueError(
            "static capillary maximum topology epoch transitions must be "
            "a nonnegative integer")
    if (static_capillary_limited_memory_curvature_tolerance is not None and
            (not math.isfinite(
                static_capillary_limited_memory_curvature_tolerance) or
             static_capillary_limited_memory_curvature_tolerance <= 0.0)):
        raise ValueError(
            "static capillary limited-memory curvature tolerance must be "
            "positive and finite")
    for name, value in (
            ("Static_capillary_volume_tolerance",
             static_capillary_volume_tolerance),
            ("Static_capillary_projected_gradient_tolerance",
             static_capillary_projected_gradient_tolerance),
            ("Static_capillary_pressure_representability_max_residual_norm",
             static_capillary_pressure_representability_max_residual_norm),
            ("Static_capillary_pressure_representability_max_relative_distance",
             static_capillary_pressure_representability_max_relative_distance),
            ("Static_capillary_physical_equilibrium_max_residual_norm",
             static_capillary_physical_equilibrium_max_residual_norm),
            ("Static_capillary_constant_pressure_kkt_max_residual_norm",
             static_capillary_constant_pressure_kkt_max_residual_norm),
            ("Static_capillary_constant_pressure_kkt_max_relative_distance",
             static_capillary_constant_pressure_kkt_max_relative_distance),
            ("Static_capillary_finite_difference_relative_step",
             static_capillary_finite_difference_relative_step),
            ("Static_capillary_limited_memory_curvature_tolerance",
             static_capillary_limited_memory_curvature_tolerance),
    ):
        if value is not None:
            level_set = level_set_equation(root)
            set_text(level_set, name, f"{value:.16g}")
    if static_capillary_max_iterations is not None:
        level_set = level_set_equation(root)
        set_text(
            level_set,
            "Static_capillary_max_iterations",
            str(static_capillary_max_iterations),
        )
    if static_capillary_max_topology_epoch_transitions is not None:
        level_set = level_set_equation(root)
        set_text(
            level_set,
            "Static_capillary_max_topology_epoch_transitions",
            str(static_capillary_max_topology_epoch_transitions),
        )
    if static_capillary_limited_memory_history_size is not None:
        level_set = level_set_equation(root)
        set_text(
            level_set,
            "Static_capillary_limited_memory_history_size",
            str(static_capillary_limited_memory_history_size),
        )

    if (enable_reinitialization is not None and
            not isinstance(enable_reinitialization, bool)):
        raise ValueError(
            "level-set reinitialization enable control must be boolean")
    if (reinitialization_cadence_steps is not None and
            (isinstance(reinitialization_cadence_steps, bool) or
             not isinstance(reinitialization_cadence_steps, int) or
             reinitialization_cadence_steps <= 0)):
        raise ValueError(
            "level-set reinitialization cadence must be a positive integer")
    if enable_reinitialization is not None:
        level_set = level_set_equation(root)
        set_text(level_set, "Enable_reinitialization",
                 "true" if enable_reinitialization else "false")
    if reinitialization_cadence_steps is not None:
        level_set = level_set_equation(root)
        set_text(level_set, "Reinitialization_cadence_steps",
                 str(reinitialization_cadence_steps))

    if enable_volume_correction is not None:
        level_set = level_set_equation(root)
        set_text(level_set, "Enable_volume_correction",
                 "true" if enable_volume_correction else "false")
    if volume_correction_cadence_steps is not None:
        level_set = level_set_equation(root)
        set_text(level_set, "Volume_correction_cadence_steps",
                 str(volume_correction_cadence_steps))
    if volume_correction_use_initial_volume is not None:
        level_set = level_set_equation(root)
        set_text(level_set, "Volume_correction_use_initial_volume",
                 "true" if volume_correction_use_initial_volume else "false")
    if volume_correction_tolerance is not None:
        level_set = level_set_equation(root)
        set_text(level_set, "Volume_correction_tolerance",
                 f"{volume_correction_tolerance:.16g}")
    if volume_correction_max_iterations is not None:
        level_set = level_set_equation(root)
        set_text(level_set, "Volume_correction_max_iterations",
                 str(volume_correction_max_iterations))
    if (volume_correction_maximum_cumulative_interface_displacement_fraction
            is not None):
        level_set = level_set_equation(root)
        set_text(
            level_set,
            "Volume_correction_maximum_cumulative_interface_displacement_fraction",
            f"{volume_correction_maximum_cumulative_interface_displacement_fraction:.16g}",
        )

    if disable_coupled_outer_fgmres:
        assert ns_solver is not None
        set_text(ns_solver, "NS_Use_coupled_outer_FGMRES", "false")

    tree.write(solver_xml, encoding="UTF-8", xml_declaration=True)


def regenerate_mms_case_if_requested(case_name: str,
                                     run_dir: Path,
                                     args: argparse.Namespace) -> None:
    if case_name != "mms2d" or (args.mms_nx is None and args.mms_ny is None):
        return
    generator = run_dir / "generate_case.py"
    if not generator.exists():
        raise FileNotFoundError(generator)
    nx = args.mms_nx if args.mms_nx is not None else args.mms_ny
    ny = args.mms_ny if args.mms_ny is not None else args.mms_nx
    if nx is None or ny is None:
        raise ValueError("MMS grid regeneration requires nx and ny")
    if nx < 2 or ny < 2:
        raise ValueError("MMS grid regeneration requires nx and ny to be at least 2")
    command = [
        sys.executable,
        str(generator),
        "--nx",
        str(nx),
        "--ny",
        str(ny),
        "--element-order",
        "2",
    ]
    completed = subprocess.run(
        command,
        cwd=run_dir,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "failed to regenerate compact MMS case:\n" + completed.stdout
        )


def solver_candidates() -> list[Path]:
    env_value = os.environ.get("SVMULTIPHYSICS_EXECUTABLE")
    paths = []
    if env_value:
        paths.append(Path(env_value))
    paths.extend([
        ROOT / "build/svMultiPhysics-build/bin/svmultiphysics",
        ROOT / "build-oop-clean-20260430/svMultiPhysics-build/bin/svmultiphysics",
    ])
    return paths


def resolve_solver(explicit: Path | None) -> Path:
    if explicit is not None:
        if explicit.exists():
            return explicit.resolve()
        raise FileNotFoundError(f"solver executable not found: {explicit}")
    for path in solver_candidates():
        if path.exists() and os.access(path, os.X_OK):
            return path.resolve()
    raise FileNotFoundError(
        "solver executable not found; set SVMULTIPHYSICS_EXECUTABLE or pass --solver"
    )


def solver_command(solver: Path, args: argparse.Namespace) -> list[str]:
    if args.mpi_ranks is None:
        return [str(solver), "solver.xml"]
    if args.mpi_ranks < 1:
        raise ValueError("--mpi-ranks must be at least 1")
    return [str(args.mpiexec), "-np", str(args.mpi_ranks), str(solver), "solver.xml"]


def solver_environment(args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    pressure_distance_maximum = getattr(
        args,
        "max_free_surface_pressure_representability_relative_distance",
        None,
    )
    initialize_static_compatible_pressure = bool(getattr(
        args, "initialize_static_compatible_pressure", False))
    initialize_discrete_static_capillary_equilibrium = bool(getattr(
        args, "initialize_discrete_static_capillary_equilibrium", False))
    if (getattr(args,
                "enable_free_surface_conservative_balance_diagnostic",
                False) or
            getattr(args,
                    "require_free_surface_conservative_balance",
                    False) or
            getattr(args,
                    "require_free_surface_pressure_representability_diagnostic",
                    False) or
            getattr(
                args,
                "max_free_surface_conservative_balance_normalized_imbalance",
                None) is not None or
            pressure_distance_maximum is not None or
            initialize_static_compatible_pressure or
            initialize_discrete_static_capillary_equilibrium):
        env[
            "SVMP_NS_FREE_SURFACE_CONSERVATIVE_BALANCE_DIAGNOSTIC"
        ] = "1"
    if pressure_distance_maximum is not None:
        if (isinstance(pressure_distance_maximum, bool) or
                not isinstance(pressure_distance_maximum, (int, float)) or
                not math.isfinite(float(pressure_distance_maximum)) or
                float(pressure_distance_maximum) < 0.0):
            raise ValueError(
                "--max-free-surface-pressure-representability-relative-distance "
                "must be finite and nonnegative"
            )
        env[
            "SVMP_NS_FREE_SURFACE_PRESSURE_REPRESENTABILITY_MAX_RELATIVE_DISTANCE"
        ] = f"{float(pressure_distance_maximum):.17g}"
    if initialize_static_compatible_pressure:
        if pressure_distance_maximum is None:
            raise ValueError(
                "--initialize-static-compatible-pressure requires "
                "--max-free-surface-pressure-representability-relative-distance"
            )
        env[
            "SVMP_NS_FREE_SURFACE_STATIC_COMPATIBLE_PRESSURE_INITIALIZER"
        ] = "1"
    if getattr(args, "require_compiled_cut_volume_jit", False):
        # Qualification must not inherit an ambient opt-out and silently run
        # the interpreted tangent path.  A build without LLVM still cannot
        # create JIT kernels; the evidence gate below will fail closed.
        env["SVMP_OOP_JIT_ENABLE"] = "1"
        env["SVMP_OOP_JIT_SPECIALIZATION_ENABLE"] = "1"
        env["SVMP_JIT_TRACE_SPECIALIZATION"] = "1"
        env["SVMP_JIT_CACHE_DIAGNOSTICS"] = "1"
    if args.enable_blockschur_true_residual_retry:
        env["SVMP_FSILS_ENABLE_BLOCKSCHUR_TRUE_RESIDUAL_RETRY"] = "1"
    if args.enable_jacobian_check:
        env["SVMP_FE_JACOBIAN_CHECK"] = "1"
        if args.jacobian_check_iteration is not None:
            env["SVMP_FE_JACOBIAN_CHECK_IT"] = str(args.jacobian_check_iteration)
        if args.jacobian_check_step is not None:
            env["SVMP_FE_JACOBIAN_CHECK_STEP"] = f"{args.jacobian_check_step:.16g}"
        if args.jacobian_check_scheme:
            env["SVMP_FE_JACOBIAN_CHECK_SCHEME"] = args.jacobian_check_scheme
        if args.jacobian_check_components:
            env["SVMP_FE_JACOBIAN_CHECK_COMPONENTS"] = args.jacobian_check_components
        if args.jacobian_check_component_sweeps:
            env["SVMP_FE_JACOBIAN_CHECK_COMPONENT_SWEEPS"] = (
                args.jacobian_check_component_sweeps
            )
    if args.enable_newton_direction_check:
        env["SVMP_NEWTON_DIRECTION_CHECK"] = "1"
    if (args.enable_newton_assembly_diagnostics or
            args.require_newton_assembly_diagnostics):
        env["SVMP_NEWTON_ASSEMBLY_DIAGNOSTICS"] = "1"
    if args.newton_line_search_fail_on_no_reduction:
        env["SVMP_NEWTON_LINE_SEARCH_FAIL_ON_NO_REDUCTION"] = "1"
    if args.newton_line_search_max_iterations is not None:
        env["SVMP_NEWTON_LINE_SEARCH_MAX_ITERATIONS"] = str(
            args.newton_line_search_max_iterations
        )
    if args.enable_linear_solve_history:
        env["SVMP_DEBUG_LINEAR_SOLVE_HISTORY"] = "1"
        if args.linear_solve_history_max_calls is not None:
            env["SVMP_DEBUG_LINEAR_SOLVE_HISTORY_MAX_CALLS"] = str(
                args.linear_solve_history_max_calls
            )
    if args.enable_linear_solve_component_norms:
        env["SVMP_DEBUG_LINEAR_SOLVE_COMPONENT_NORMS"] = "1"
        if args.linear_solve_component_norms_max_newton_it is not None:
            env["SVMP_DEBUG_LINEAR_SOLVE_COMPONENT_NORMS_MAX_NEWTON_IT"] = str(
                args.linear_solve_component_norms_max_newton_it
            )
    if args.enable_linear_solve_memory_diagnostics:
        env["SVMP_LINEAR_SOLVE_MEMORY_DIAGNOSTICS"] = "1"
    if args.enable_fsils_matrix_diagnostics:
        env["SVMP_FSILS_MATRIX_DIAGNOSTICS"] = "1"
        if args.fsils_matrix_diagnostics_every_n is not None:
            env["SVMP_FSILS_MATRIX_DIAGNOSTICS_EVERY_N"] = str(
                args.fsils_matrix_diagnostics_every_n
            )
        if args.fsils_matrix_diagnostics_max_records is not None:
            env["SVMP_FSILS_MATRIX_DIAGNOSTICS_MAX_RECORDS"] = str(
                args.fsils_matrix_diagnostics_max_records
            )
    if args.require_eigen_factorization_diagnostics:
        env["SVMP_FE_EIGEN_FACTOR_DIAGNOSTICS"] = "1"
    if (args.enable_timeloop_initialization_diagnostics or
            args.require_timeloop_initialization_diagnostics):
        env["SVMP_TIMELOOP_INITIALIZATION_DIAGNOSTICS"] = "1"
    if args.enable_form_block_diagnostics:
        env["SVMP_FE_FORM_BLOCK_DIAGNOSTICS"] = "1"
    if args.enable_interior_face_timing:
        env["SVMP_INTERIOR_FACE_TIMING"] = "1"
    if args.enable_cut_volume_timing:
        env["SVMP_CUT_VOLUME_TIMING"] = "1"
    if args.enable_jit_specialization_trace:
        env["SVMP_JIT_TRACE_SPECIALIZATION"] = "1"
    if args.enable_jit_cache_diagnostics:
        env["SVMP_JIT_CACHE_DIAGNOSTICS"] = "1"
    if args.trace_level_set_advection_velocity:
        env["SVMP_TRACE_LEVEL_SET_ADVECTION"] = "1"
    if args.enable_adaptive_time_loop:
        env["SVMP_TIMELOOP_ADAPTIVE"] = "1"
        env["SVMP_VTK_OUTPUT_FINAL_TIME"] = "1"
        for arg_name, env_name in (
            ("adaptive_time_loop_min_dt", "SVMP_TIMELOOP_MIN_DT"),
            ("adaptive_time_loop_max_dt", "SVMP_TIMELOOP_MAX_DT"),
            ("adaptive_time_loop_max_retries", "SVMP_TIMELOOP_MAX_RETRIES"),
            ("adaptive_time_loop_decrease_factor", "SVMP_TIMELOOP_DECREASE_FACTOR"),
            ("adaptive_time_loop_increase_factor", "SVMP_TIMELOOP_INCREASE_FACTOR"),
            ("adaptive_time_loop_target_newton_iterations",
             "SVMP_TIMELOOP_TARGET_NEWTON_ITERATIONS"),
            ("adaptive_time_loop_max_steps_multiplier",
             "SVMP_TIMELOOP_MAX_STEPS_MULTIPLIER"),
        ):
            value = getattr(args, arg_name)
            if value is not None:
                env[env_name] = f"{value:.16g}" if isinstance(value, float) else str(value)
    return env


def case_artifact_ignore(_path: str, names: list[str]) -> set[str]:
    ignored = set()
    for name in names:
        if re.match(r"result_.*\.p?vtu$", name):
            ignored.add(name)
        elif name in {"result.pvd", "1-procs", "2-procs", "3-procs", "4-procs"}:
            ignored.add(name)
        elif name.startswith("restart") or name.endswith(".log"):
            ignored.add(name)
    return ignored


def collate_solver_time_series(run_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    if args.disable_vtk_output:
        return {"generated": False, "reason": "vtk_output_disabled"}
    try:
        return collate_time_series(run_dir)
    except Exception as exc:  # pragma: no cover - diagnostic path only
        return {
            "generated": False,
            "reason": "collation_error",
            "error": str(exc),
        }


def copy_selected_entries(source_root: Path,
                          destination: Path,
                          entries: tuple[str, ...]) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for entry in entries:
        source = source_root / entry
        target = destination / entry
        if source.is_dir():
            shutil.copytree(source, target, ignore=case_artifact_ignore)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)


def copy_case_from_ref(case_dir: Path,
                       destination: Path,
                       source_ref: str,
                       entries: tuple[str, ...] | None = None) -> None:
    relative = case_dir.relative_to(ROOT)
    archive_paths = [str(relative)]
    if entries is not None:
        archive_paths = [str(relative / entry) for entry in entries]
    completed = subprocess.run(
        ["git", "archive", "--format=tar", source_ref, *archive_paths],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.decode("utf-8", errors="replace"))

    archive_root = destination.parent / "_archive"
    archive_root.mkdir()
    with tarfile.open(fileobj=io.BytesIO(completed.stdout)) as archive:
        try:
            archive.extractall(archive_root, filter="data")
        except TypeError:
            archive.extractall(archive_root)
    if entries is None:
        shutil.move(str(archive_root / relative), destination)
    else:
        copy_selected_entries(archive_root / relative, destination, entries)


def copy_case(case_dir: Path, destination: Path, source_ref: str | None) -> None:
    entries = CASE_COPY_ENTRIES.get(case_dir)
    if source_ref is not None:
        copy_case_from_ref(case_dir, destination, source_ref, entries)
        return

    if entries is not None:
        copy_selected_entries(case_dir, destination, entries)
    else:
        shutil.copytree(case_dir, destination, ignore=case_artifact_ignore)


def write_boundary(path: Path,
                   points: np.ndarray,
                   node_ids: list[int],
                   first_cell_id: int) -> None:
    lines = []
    for index in range(len(node_ids) - 1):
        lines.extend([2, index, index + 1])
    poly = pv.PolyData()
    poly.points = points[node_ids]
    poly.lines = np.asarray(lines, dtype=np.int64)
    poly.point_data["GlobalNodeID"] = np.asarray(node_ids, dtype=np.int64)
    poly.cell_data["GlobalElementID"] = np.arange(
        first_cell_id, first_cell_id + len(node_ids) - 1, dtype=np.int64)
    poly.save(path)


def write_mini_mesh(case_dir: Path,
                    static: bool = False,
                    nx: int = 8,
                    ny: int = 8,
                    simplex_mesh: bool = False) -> tuple[int, float]:
    if nx < 2 or ny < 2:
        raise ValueError("synthetic mesh resolution must be at least 2 by 2")
    tank_height = 1.0
    tank_length = 1.0
    bed_depth = 0.2
    column_height = 0.75
    column_width = 0.4
    rho = 998.2
    gravity = 9.81

    xs = np.linspace(0.0, tank_length, nx + 1)
    ys = np.linspace(0.0, tank_height, ny + 1)
    points = np.array([[x, y, 0.0] for y in ys for x in xs], dtype=float)

    cells: list[int] = []
    for j in range(ny):
        for i in range(nx):
            lower_left = j * (nx + 1) + i
            lower_right = lower_left + 1
            upper_right = lower_left + nx + 2
            upper_left = lower_left + nx + 1
            if simplex_mesh:
                cells.extend([
                    3, lower_left, lower_right, upper_right,
                    3, lower_left, upper_right, upper_left,
                ])
            else:
                cells.extend([
                    4,
                    lower_left,
                    lower_right,
                    upper_right,
                    upper_left,
                ])
    cell_count = (2 if simplex_mesh else 1) * nx * ny
    cell_type = pv.CellType.TRIANGLE if simplex_mesh else pv.CellType.QUAD
    cell_types = np.full(cell_count, cell_type, dtype=np.uint8)
    grid = pv.UnstructuredGrid(np.asarray(cells, dtype=np.int64), cell_types, points)

    x = points[:, 0]
    y = points[:, 1]
    if static:
        phi = y - STATIC_INTERFACE_HEIGHT
        pressure = np.zeros(points.shape[0], dtype=float)
    else:
        phi = np.minimum(y - bed_depth, np.maximum(x - column_width, y - column_height))
        free_surface_height = np.where(x <= column_width, column_height, bed_depth)
        pressure = rho * gravity * np.maximum(free_surface_height - y, 0.0)
        pressure[phi > 0.0] = 0.0

    grid.point_data["GlobalNodeID"] = np.arange(points.shape[0], dtype=np.int64)
    grid.point_data["phi"] = phi
    grid.point_data["Pressure"] = pressure
    grid.point_data["Velocity"] = np.zeros((points.shape[0], 3), dtype=float)
    grid.cell_data["GlobalElementID"] = np.arange(cell_count, dtype=np.int64)

    mesh_dir = case_dir / "mesh/background"
    surface_dir = mesh_dir / "mesh-surfaces"
    surface_dir.mkdir(parents=True)
    grid.save(mesh_dir / "mesh-complete.mesh.vtu")

    left = [j * (nx + 1) for j in range(ny + 1)]
    right = [j * (nx + 1) + nx for j in range(ny + 1)]
    bottom = list(range(nx + 1))
    top = [ny * (nx + 1) + i for i in range(nx + 1)]
    write_boundary(surface_dir / "wall_left.vtp", points, left, 0)
    write_boundary(surface_dir / "wall_right.vtp", points, right, ny)
    write_boundary(surface_dir / "wall_bottom.vtp", points, bottom, 2 * ny)
    write_boundary(surface_dir / "wall_top.vtp", points, top, 2 * ny + nx)

    gauge_node = 0
    gauge_pressure = 0.0 if static else float(rho * gravity * column_height)
    return gauge_node, gauge_pressure


def write_mini_solver_xml(case_dir: Path,
                          steps: int,
                          gauge_node: int,
                          gauge_pressure: float,
                          static: bool = False) -> None:
    force_y = "0.0" if static else "-9.81"
    hydrostatic_initialization = "false" if static else "true"
    hydrostatic_reference_y = STATIC_INTERFACE_HEIGHT if static else 0.75
    (case_dir / "pressure_gauge.csv").write_text(
        f"node_id,pressure\n{gauge_node},{gauge_pressure:.16g}\n", encoding="utf-8")
    (case_dir / "solver.xml").write_text(f"""<?xml version="1.0" encoding="UTF-8" ?>
<svMultiPhysicsFile version="0.1">

<GeneralSimulationParameters>
  <Use_new_OOP_solver>true</Use_new_OOP_solver>
  <Continue_previous_simulation>false</Continue_previous_simulation>
  <Number_of_spatial_dimensions>2</Number_of_spatial_dimensions>
  <Number_of_time_steps>{steps}</Number_of_time_steps>
  <Time_step_size>0.001</Time_step_size>
  <Spectral_radius_of_infinite_time_step>0.50</Spectral_radius_of_infinite_time_step>
  <Searched_file_name_to_trigger_stop>STOP_SIM</Searched_file_name_to_trigger_stop>
  <Save_results_to_VTK_format>true</Save_results_to_VTK_format>
  <Name_prefix_of_saved_VTK_files>result</Name_prefix_of_saved_VTK_files>
  <Increment_in_saving_VTK_files>1</Increment_in_saving_VTK_files>
  <Start_saving_after_time_step>1</Start_saving_after_time_step>
  <Increment_in_saving_restart_files>{steps}</Increment_in_saving_restart_files>
  <Convert_BIN_to_VTK_format>0</Convert_BIN_to_VTK_format>
  <Verbose>1</Verbose>
  <Warning>0</Warning>
  <Debug>0</Debug>
</GeneralSimulationParameters>

<Add_mesh name="tank">
  <Mesh_file_path>mesh/background/mesh-complete.mesh.vtu</Mesh_file_path>
  <Add_face name="wall_left">
    <Face_file_path>mesh/background/mesh-surfaces/wall_left.vtp</Face_file_path>
  </Add_face>
  <Add_face name="wall_right">
    <Face_file_path>mesh/background/mesh-surfaces/wall_right.vtp</Face_file_path>
  </Add_face>
  <Add_face name="wall_bottom">
    <Face_file_path>mesh/background/mesh-surfaces/wall_bottom.vtp</Face_file_path>
  </Add_face>
  <Add_face name="wall_top">
    <Face_file_path>mesh/background/mesh-surfaces/wall_top.vtp</Face_file_path>
  </Add_face>
</Add_mesh>

<Add_equation type="level_set">
  <Coupled>true</Coupled>
  <Min_iterations>1</Min_iterations>
  <Max_iterations>2</Max_iterations>
  <Tolerance>1.0e-4</Tolerance>
  <Level_set_field_name>phi</Level_set_field_name>
  <Operator_tag>equations</Operator_tag>
  <Level_set_source>prescribed_data</Level_set_source>
  <Velocity_source>constant</Velocity_source>
  <Constant_velocity>0.0 0.0 0.0</Constant_velocity>
  <Enable_SUPG>true</Enable_SUPG>
  <SUPG_tau_scale>0.5</SUPG_tau_scale>
  <SUPG_transient_scale>2.0</SUPG_transient_scale>
  <Enable_discontinuity_capturing>true</Enable_discontinuity_capturing>
  <Discontinuity_capturing_scale>0.1</Discontinuity_capturing_scale>
  <Discontinuity_capturing_gradient_epsilon>1.0e-12</Discontinuity_capturing_gradient_epsilon>
  <Discontinuity_capturing_max_courant>0.5</Discontinuity_capturing_max_courant>
  <Enable_bound_preserving_limiter>true</Enable_bound_preserving_limiter>
  <Bound_preserving_bound_tolerance>{DEFAULT_BOUND_REPRESENTABILITY_SLACK:.16g}</Bound_preserving_bound_tolerance>
  <Bound_preserving_sign_tolerance>{BOUND_SIGN_TOLERANCE:.16g}</Bound_preserving_sign_tolerance>
  <Bound_preserving_maximum_courant>1.0</Bound_preserving_maximum_courant>
  <Bound_preserving_enforce_courant_limit>true</Bound_preserving_enforce_courant_limit>
  <Bound_preserving_enforce_impermeable_boundaries>true</Bound_preserving_enforce_impermeable_boundaries>
  <Bound_preserving_impermeable_normal_velocity_tolerance>1.0e-10</Bound_preserving_impermeable_normal_velocity_tolerance>
  <Enable_reinitialization>false</Enable_reinitialization>
  <Enable_volume_correction>false</Enable_volume_correction>
  <Output type="Spatial">
    <Level_set>true</Level_set>
    <Generated_interface>true</Generated_interface>
    <Surface_position>true</Surface_position>
  </Output>
  <Output type="Volume_integral">
    <Volume>true</Volume>
  </Output>
  <LS type="Direct">
    <Linear_algebra type="eigen">
      <Preconditioner>none</Preconditioner>
    </Linear_algebra>
    <Max_iterations>1</Max_iterations>
    <Krylov_space_dimension>1</Krylov_space_dimension>
    <Tolerance>1.0e-4</Tolerance>
    <Absolute_tolerance>1.0e-4</Absolute_tolerance>
  </LS>
</Add_equation>

<Add_equation type="fluid">
  <Coupled>true</Coupled>
  <Min_iterations>1</Min_iterations>
  <Max_iterations>8</Max_iterations>
  <Tolerance>1.0e-4</Tolerance>
  <Backflow_stabilization_coefficient>0.0</Backflow_stabilization_coefficient>
  <Density>998.2</Density>
  <Force_x>0.0</Force_x>
  <Force_y>{force_y}</Force_y>
  <Force_z>0.0</Force_z>
  <Hydrostatic_pressure_initialization>{hydrostatic_initialization}</Hydrostatic_pressure_initialization>
  <Hydrostatic_pressure_reference>0.0</Hydrostatic_pressure_reference>
  <Hydrostatic_pressure_reference_point>0.0 {hydrostatic_reference_y:.16g} 0.0</Hydrostatic_pressure_reference_point>
  <Node_pressure_constraints>
    <Id_type>Global_vertex_gid</Id_type>
    <Values_file_path>pressure_gauge.csv</Values_file_path>
  </Node_pressure_constraints>
  <Viscosity model="Constant">
    <Value>1.003e-3</Value>
  </Viscosity>
  <Output type="Spatial">
    <Velocity>true</Velocity>
    <Pressure>true</Pressure>
    <Divergence>true</Divergence>
  </Output>
  <Output type="Volume_integral">
    <Volume>true</Volume>
  </Output>
  <LS type="Direct">
    <Linear_algebra type="eigen">
      <Preconditioner>none</Preconditioner>
    </Linear_algebra>
    <Max_iterations>1</Max_iterations>
    <Krylov_space_dimension>1</Krylov_space_dimension>
    <Tolerance>1.0e-4</Tolerance>
    <Absolute_tolerance>1.0e-4</Absolute_tolerance>
  </LS>
  <Add_BC name="wall_left">
    <Type>Dir</Type>
    <Value>0.0</Value>
  </Add_BC>
  <Add_BC name="wall_right">
    <Type>Dir</Type>
    <Value>0.0</Value>
  </Add_BC>
  <Add_BC name="wall_bottom">
    <Type>Dir</Type>
    <Value>0.0</Value>
  </Add_BC>
  <Add_BC name="free_surface">
    <Type>Free_surface</Type>
    <Implementation>UnfittedLevelSet</Implementation>
    <Level_set_field_name>phi</Level_set_field_name>
    <Generated_interface_domain_id>open_vessel_surface</Generated_interface_domain_id>
    <Level_set_isovalue>0.0</Level_set_isovalue>
    <Active_domain>LevelSetNegative</Active_domain>
    <Active_domain_method>CutVolume</Active_domain_method>
    <External_pressure>0.0</External_pressure>
    <Surface_tension>0.0</Surface_tension>
    <Enable_velocity_extension>false</Enable_velocity_extension>
    <Enable_cut_cell_stabilization>true</Enable_cut_cell_stabilization>
    <Use_cut_metadata_scale>true</Use_cut_metadata_scale>
    <Cut_cell_pressure_gradient_penalty>1.0</Cut_cell_pressure_gradient_penalty>
  </Add_BC>
</Add_equation>

</svMultiPhysicsFile>
""", encoding="utf-8")


def write_mini_case(case_dir: Path,
                    steps: int,
                    static: bool = False,
                    nx: int = 8,
                    ny: int = 8,
                    simplex_mesh: bool = False) -> None:
    case_dir.mkdir(parents=True)
    gauge_node, gauge_pressure = write_mini_mesh(
        case_dir, static, nx, ny, simplex_mesh)
    write_mini_solver_xml(case_dir, steps, gauge_node, gauge_pressure, static)


def remove_synthetic_pressure_pin(case_dir: Path) -> None:
    """Leave absolute pressure to the physical free-surface traction anchor."""
    solver_xml = case_dir / "solver.xml"
    tree = ET.parse(solver_xml)
    fluid = fluid_equation(tree.getroot())
    constraints = fluid.find("Node_pressure_constraints")
    if constraints is not None:
        fluid.remove(constraints)
    ET.indent(tree, space="  ")
    tree.write(solver_xml, encoding="utf-8", xml_declaration=True)


def set_case_active_domain(case_dir: Path, active_domain: str) -> None:
    """Set the liquid-side declaration in one generated synthetic case."""
    solver_xml = case_dir / "solver.xml"
    tree = ET.parse(solver_xml)
    set_text(
        free_surface_bc(tree.getroot()),
        "Active_domain",
        normalized_active_domain(active_domain),
    )
    ET.indent(tree, space="  ")
    tree.write(solver_xml, encoding="utf-8", xml_declaration=True)


def configure_capillary_wave_wall_boundary_contract(case_dir: Path) -> None:
    """Install the impermeable-slip walls assumed by linear wave theory."""
    solver_xml = case_dir / "solver.xml"
    tree = ET.parse(solver_xml)
    root = tree.getroot()
    fluid = fluid_equation(root)
    by_name = {
        bc.attrib.get("name"): bc
        for bc in fluid.findall("Add_BC")
    }
    for name in ("wall_left", "wall_right"):
        wall = by_name.get(name)
        if wall is None:
            raise ValueError(f"capillary-wave case is missing {name}")
        # Constrain only the horizontal wall-normal component.  Full no-slip
        # would pin the cosine-wave contact points and invalidate the mode.
        set_text(wall, "Effective_direction", "1 0")
    bottom = by_name.get("wall_bottom")
    if bottom is None:
        raise ValueError("capillary-wave case is missing wall_bottom")
    # The finite-depth dispersion relation assumes an impermeable free-slip
    # bottom.  Constraining tangential velocity here would add a no-slip
    # boundary layer that is absent from the reference solution.
    set_text(bottom, "Effective_direction", "0 1")

    level_set = level_set_equation(root)
    top = next(
        (bc for bc in level_set.findall("Add_BC")
         if bc.attrib.get("name") == "wall_top"),
        None,
    )
    if top is None:
        top = ET.SubElement(level_set, "Add_BC", {"name": "wall_top"})
    # The upper background boundary lies wholly in the unmodelled gas.  It is
    # an open numerical truncation, not an impermeable liquid wall.  Declaring
    # it as level-set outflow prevents the dry velocity extension from being
    # misclassified by the impermeable-wall safety audit.
    set_text(top, "Type", "LevelSetOutflow")
    ET.indent(tree, space="  ")
    tree.write(solver_xml, encoding="utf-8", xml_declaration=True)


def sessile_circle_geometry(contact_angle_degrees: float,
                            radius: float) -> tuple[float, float, float]:
    angle = math.radians(contact_angle_degrees)
    if not (0.0 < angle < math.pi):
        raise ValueError("sessile contact angle must be strictly between 0 and 180 degrees")
    center_y = -radius * math.cos(angle)
    half_footprint = radius * math.sin(angle)
    area = radius * radius * (angle - math.sin(angle) * math.cos(angle))
    return center_y, half_footprint, area


def sessile_circle_radius_for_area(contact_angle_degrees: float,
                                   area: float) -> float:
    """Return the circular-cap radius at a prescribed liquid area.

    Dynamic advancing/receding comparisons must not silently change the
    conserved liquid amount when the initial angle changes.  The caller can
    therefore define one reference equilibrium cap and construct each
    perturbation at its area rather than at its radius.
    """
    if not math.isfinite(area) or area <= 0.0:
        raise ValueError("sessile liquid area must be positive and finite")
    angle = math.radians(contact_angle_degrees)
    if not (0.0 < angle < math.pi):
        raise ValueError("sessile contact angle must be strictly between 0 and 180 degrees")
    area_factor = angle - math.sin(angle) * math.cos(angle)
    return math.sqrt(area / area_factor)


def sessile_contact_wall_spec(wall_face: str) -> dict[str, Any]:
    """Return the axis-aligned wall frame used by the synthetic 2-D cap.

    ``wall_tangent`` is deliberately oriented in the positive coordinate
    direction.  Consequently, the two contact roots are ordered by the same
    scalar wall coordinate and their outward-footprint directions are
    ``-wall_tangent`` and ``+wall_tangent``.  The configured wall normal always
    points out of the square fluid domain and into the solid.
    """
    specs: dict[str, dict[str, Any]] = {
        "wall_bottom": {
            "wall_face": "wall_bottom",
            "wall_axis": 1,
            "wall_coordinate": 0.0,
            "wall_normal": (0.0, -1.0, 0.0),
            "wall_tangent_axis": 0,
            "wall_tangent": (1.0, 0.0, 0.0),
            "effective_direction": "0 1",
        },
        "wall_left": {
            "wall_face": "wall_left",
            "wall_axis": 0,
            "wall_coordinate": 0.0,
            "wall_normal": (-1.0, 0.0, 0.0),
            "wall_tangent_axis": 1,
            "wall_tangent": (0.0, 1.0, 0.0),
            "effective_direction": "1 0",
        },
        "wall_right": {
            "wall_face": "wall_right",
            "wall_axis": 0,
            "wall_coordinate": 1.0,
            "wall_normal": (1.0, 0.0, 0.0),
            "wall_tangent_axis": 1,
            "wall_tangent": (0.0, 1.0, 0.0),
            "effective_direction": "1 0",
        },
        "wall_top": {
            "wall_face": "wall_top",
            "wall_axis": 1,
            "wall_coordinate": 1.0,
            "wall_normal": (0.0, 1.0, 0.0),
            "wall_tangent_axis": 0,
            "wall_tangent": (1.0, 0.0, 0.0),
            "effective_direction": "0 1",
        },
    }
    try:
        return dict(specs[wall_face])
    except KeyError as exc:
        raise ValueError(
            "synthetic sessile contact wall must be one of wall_bottom, "
            "wall_left, wall_right, or wall_top"
        ) from exc


def ren_e_speed_sign_agrees(measured_speed: float,
                            predicted_speed: float,
                            tolerance: float = 1.0e-14) -> bool:
    """Compare advancing/receding direction without treating rest as either."""
    measured_is_zero = abs(measured_speed) <= tolerance
    predicted_is_zero = abs(predicted_speed) <= tolerance
    if measured_is_zero or predicted_is_zero:
        return measured_is_zero and predicted_is_zero
    return math.copysign(1.0, measured_speed) == math.copysign(
        1.0, predicted_speed)


def configure_sessile_solver_xml(case_dir: Path,
                                  steps: int,
                                  time_step_size: float,
                                  equilibrium_angle_degrees: float,
                                  surface_tension: float,
                                  mobility: float,
                                  slip_length: float,
                                  dynamic: bool,
                                  smoothing_width: float,
                                  wall_face: str = "wall_bottom",
                                  contact_line_model: str = "dynamic",
                                  active_domain: str = "LevelSetNegative",
                                  ) -> None:
    if contact_line_model not in {"dynamic", "prescribed"}:
        raise ValueError(
            "sessile contact-line model must be dynamic or prescribed")
    solver_xml = case_dir / "solver.xml"
    tree = ET.parse(solver_xml)
    root = tree.getroot()
    general = root.find("GeneralSimulationParameters")
    if general is None:
        raise ValueError("synthetic sessile case is missing GeneralSimulationParameters")
    set_text(general, "Number_of_time_steps", str(steps))
    set_text(general, "Time_step_size", f"{time_step_size:.16g}")

    level_set = level_set_equation(root)
    set_text(level_set, "Operator_tag", "equations")
    # PrescribedData denotes mesh-field initialization only; the transport
    # installer still registers phi as an unknown owned by the time integrator.
    set_text(level_set, "Level_set_source", "prescribed_data")
    # Disabled maintenance operators still validate their cadence values.
    set_text(level_set, "Reinitialization_cadence_steps", "1")
    set_text(level_set, "Volume_correction_cadence_steps", "1")
    # SurfaceStress is the first variation of the generated-interface measure
    # and does not consume a projected scalar curvature.  Keeping an otherwise
    # unused kappa field in this physical wetting deck is more than overhead:
    # it adds a non-residual generated field to the nonlinear outer-state
    # transaction and makes projection freshness look like force freshness.
    # Dedicated curvature-projection smokes retain that independent recovery
    # path.  Remove inherited controls defensively so this deck proves the
    # actual SurfaceStress contract.
    for name in (
            "Enable_curvature_projection",
            "Projected_curvature_field",
            "Curvature_projection_cadence_steps",
            "Curvature_projection_supplemental_sample_weight",
            "Curvature_projection_recovery_mode",
            "Curvature_projection_narrow_band_width",
            "Curvature_projection_smoothing_iterations",
            "Curvature_projection_smoothing_relaxation",
            "Curvature_projection_smoothing_mode"):
        element = level_set.find(name)
        if element is not None:
            level_set.remove(element)
    # Keep phi coupled to the solved velocity for both contact models. The
    # moving probe exercises Ren--E kinematics; the stationary prescribed
    # probe exercises accepted-state wall repair and Young wall energy.
    constant_velocity = level_set.find("Constant_velocity")
    if constant_velocity is not None:
        level_set.remove(constant_velocity)
    set_text(level_set, "Velocity_source", "coupled_field")
    set_text(level_set, "Velocity_field_name", "Velocity")
    set_text(level_set, "Auto_register_velocity_field", "true")
    if contact_line_model == "prescribed":
        # Accepted-state wall repair owns the contact geometry for this model.
        # The transport limiter would compare that repaired field against the
        # unrepaired nodal range and reject an otherwise stationary update.
        set_text(level_set, "Enable_bound_preserving_limiter", "false")

    fluid = fluid_equation(root)
    set_text(fluid, "Density", "1.0")
    pressure_constraints = fluid.find("Node_pressure_constraints")
    if pressure_constraints is not None:
        fluid.remove(pressure_constraints)
    viscosity = fluid.find("Viscosity")
    if viscosity is None:
        viscosity = ET.SubElement(fluid, "Viscosity", {"model": "Constant"})
    viscosity.set("model", "Constant")
    set_text(viscosity, "Value", "0.1")

    wall_spec = sessile_contact_wall_spec(wall_face)
    contact_wall = next(
        (bc for bc in fluid.findall("Add_BC")
         if bc.attrib.get("name") == wall_spec["wall_face"]),
        None,
    )
    if contact_wall is None and wall_spec["wall_face"] == "wall_top":
        contact_wall = ET.SubElement(
            fluid, "Add_BC", {"name": wall_spec["wall_face"]})
        set_text(contact_wall, "Type", "Dir")
        set_text(contact_wall, "Value", "0.0")
    if contact_wall is None:
        raise ValueError(
            f"synthetic sessile case is missing {wall_spec['wall_face']}")
    # Strongly impose only impermeability. The dynamic model supplies its
    # Navier wall law; the stationary prescribed-angle matrix retains the same
    # free tangential trace so Young wall-energy virtual work remains visible.
    set_text(
        contact_wall, "Effective_direction", wall_spec["effective_direction"])

    free_surface = free_surface_bc(root)
    set_text(
        free_surface,
        "Active_domain",
        normalized_active_domain(active_domain),
    )
    set_text(free_surface, "Generated_interface_geometry", "LinearCorner")
    set_text(free_surface, "Enable_velocity_extension", "false")
    set_text(free_surface, "Surface_tension", f"{surface_tension:.16g}")
    set_text(free_surface, "Surface_tension_form", "SurfaceStress")
    curvature_field = free_surface.find("Curvature_field")
    if curvature_field is not None:
        free_surface.remove(curvature_field)
    set_text(
        free_surface,
        "Contact_line_model",
        ("DynamicContactAngle" if contact_line_model == "dynamic"
         else "PrescribedContactAngle"),
    )
    set_text(
        free_surface, "Contact_line_wall_face", wall_spec["wall_face"])
    set_text(
        free_surface,
        "Contact_line_wall_normal",
        " ".join(f"{value:.1f}" for value in wall_spec["wall_normal"]),
    )
    set_text(free_surface, "Contact_angle_degrees",
             f"{equilibrium_angle_degrees:.16g}")
    set_text(free_surface, "Active_domain_smoothing_width",
             f"{smoothing_width:.16g}")
    if contact_line_model == "dynamic":
        set_text(free_surface, "Contact_line_mobility", f"{mobility:.16g}")
        set_text(free_surface, "Wall_slip_model", "Navier")
        set_text(free_surface, "Wall_slip_length", f"{slip_length:.16g}")
    else:
        for name in (
                "Contact_line_mobility", "Wall_slip_model",
                "Wall_slip_length"):
            element = free_surface.find(name)
            if element is not None:
                free_surface.remove(element)

    ET.indent(tree, space="  ")
    tree.write(solver_xml, encoding="utf-8", xml_declaration=True)


def write_sessile2d_case(case_dir: Path,
                         steps: int,
                         nx: int,
                         ny: int,
                         initial_angle_degrees: float,
                         equilibrium_angle_degrees: float,
                         radius: float,
                         surface_tension: float,
                         time_step_size: float,
                         mobility: float,
                         slip_length: float,
                         dynamic: bool,
                         wall_face: str = "wall_bottom",
                         contact_line_model: str = "dynamic",
                         level_set_positive_scale: float = 1.0,
                         initialize_discrete_static_contact_geometry: bool = False,
                         simplex_mesh: bool = False,
                         active_domain: str = "LevelSetNegative",
                         tangent_center_offset: float = 0.0,
                         ) -> None:
    if radius <= 0.0 or surface_tension <= 0.0:
        raise ValueError("sessile radius and surface tension must be positive")
    if mobility <= 0.0 or slip_length <= 0.0:
        raise ValueError("sessile mobility and slip length must be positive")
    if contact_line_model not in {"dynamic", "prescribed"}:
        raise ValueError(
            "sessile contact-line model must be dynamic or prescribed")
    if dynamic and contact_line_model != "dynamic":
        raise ValueError(
            "moving sessile contact requires the dynamic contact-line model")
    if dynamic and initialize_discrete_static_contact_geometry:
        raise ValueError(
            "discrete static contact initialization is only available for "
            "stationary sessile cases")
    if (not math.isfinite(level_set_positive_scale) or
            level_set_positive_scale <= 0.0):
        raise ValueError("level-set positive scale must be positive and finite")
    active_domain = normalized_active_domain(active_domain)
    if not math.isfinite(tangent_center_offset):
        raise ValueError("sessile tangent-center offset must be finite")

    wall_spec = sessile_contact_wall_spec(wall_face)
    wall_axis = int(wall_spec["wall_axis"])
    tangent_axis = int(wall_spec["wall_tangent_axis"])
    wall_coordinate = float(wall_spec["wall_coordinate"])
    wall_normal = np.asarray(wall_spec["wall_normal"][:2], dtype=float)
    wall_tangent = np.asarray(wall_spec["wall_tangent"][:2], dtype=float)
    wall_inward = -wall_normal

    write_mini_case(
        case_dir,
        steps,
        static=True,
        nx=nx,
        ny=ny,
        simplex_mesh=simplex_mesh,
    )
    mesh_path = case_dir / "mesh/background/mesh-complete.mesh.vtu"
    grid = pv.read(mesh_path)
    points = np.asarray(grid.points, dtype=float)
    # ``radius`` defines the common equilibrium reference drop.  Construct a
    # dynamic perturbation at that drop's liquid area; holding radius fixed
    # while changing angle would change liquid mass and would make the
    # advancing/receding pair physically asymmetric before the solve starts.
    _equilibrium_center_y, _equilibrium_half_footprint, expected_area = (
        sessile_circle_geometry(equilibrium_angle_degrees, radius)
    )
    initial_radius = sessile_circle_radius_for_area(
        initial_angle_degrees, expected_area)
    center_inward_distance, half_footprint, initial_area = sessile_circle_geometry(
        initial_angle_degrees, initial_radius)
    if not math.isclose(initial_area, expected_area, rel_tol=1.0e-13,
                        abs_tol=1.0e-15):
        raise RuntimeError("fixed-area sessile construction is inconsistent")
    center = np.zeros(2, dtype=float)
    center[tangent_axis] = 0.5 + tangent_center_offset
    center[wall_axis] = (
        wall_coordinate + center_inward_distance * wall_inward[wall_axis])
    if (center[tangent_axis] - half_footprint <= 0.0 or
            center[tangent_axis] + half_footprint >= 1.0):
        raise ValueError(
            "sessile contact points must remain strictly inside the wall")
    phi = np.linalg.norm(points[:, :2] - center, axis=1) - initial_radius
    # A circle sampled at Q1 vertices does not, in general, give the
    # LinearCorner fragment in the wall-adjacent cut cell the analytic circle
    # tangent.  That chord error is part of the static P1 physical-refinement
    # problem and must be measured by the existing generated-normal angle
    # gate.  Overwriting the two contact cells in a nominal equilibrium makes
    # the contact chord exact by introducing two O(h)-localized kinks into an
    # otherwise globally sampled circle; those kinks dominate the
    # SurfaceStress pressure-balance error and manufacture parasitic current.
    #
    # The separate dynamic constitutive probe has a narrower purpose: it must
    # compare equal-and-opposite Ren--E perturbations at a prescribed *discrete*
    # angle.  Retain the tangent replacement there so its force-law input is
    # controlled.  An explicit static switch provides the corresponding Tier-0
    # manufactured contact-equilibrium state without changing the default
    # continuum-cap refinement cases.
    contact_cell_ids: list[int] = []
    if dynamic or initialize_discrete_static_contact_geometry:
        tangent_angle = math.radians(initial_angle_degrees)
        for side, contact_coordinate in (
                (-1.0, center[tangent_axis] - half_footprint),
                (1.0, center[tangent_axis] + half_footprint)):
            candidates: list[tuple[float, int, np.ndarray]] = []
            for cell_id in range(grid.n_cells):
                cell = grid.get_cell(cell_id)
                point_ids = np.asarray(cell.point_ids, dtype=int)
                if point_ids.size not in (3, 4):
                    continue
                cell_points = points[point_ids, :2]
                wall_vertices = np.isclose(
                    cell_points[:, wall_axis], wall_coordinate,
                    rtol=0.0, atol=1.0e-12)
                if np.count_nonzero(wall_vertices) != 2:
                    continue
                tangent_min = float(np.min(cell_points[:, tangent_axis]))
                tangent_max = float(np.max(cell_points[:, tangent_axis]))
                if (contact_coordinate < tangent_min - 1.0e-12 or
                        contact_coordinate > tangent_max + 1.0e-12):
                    continue
                cell_tangent_center = 0.5 * (tangent_min + tangent_max)
                # If the contact lies exactly on a cell edge, select the cell
                # on the liquid side along the wall (above/right of the lower
                # contact, below/left of the upper contact).
                inward_mismatch = max(
                    0.0,
                    -side * (contact_coordinate - cell_tangent_center),
                )
                candidates.append((inward_mismatch, cell_id, point_ids))
            if not candidates:
                raise RuntimeError(
                    "unable to locate a wall-adjacent sessile contact cell")
            _mismatch, cell_id, point_ids = min(
                candidates, key=lambda item: (item[0], item[1]))
            contact_point = np.zeros(2, dtype=float)
            contact_point[wall_axis] = wall_coordinate
            contact_point[tangent_axis] = contact_coordinate
            # n.n_w=-cos(theta) with n outward from the liquid.  Its wall
            # projection is -t at the lower root and +t at the upper root.
            outward_normal = (
                side * math.sin(tangent_angle) * wall_tangent -
                math.cos(tangent_angle) * wall_normal)
            local_points = points[point_ids, :2]
            phi[point_ids] = (local_points - contact_point) @ outward_normal
            contact_cell_ids.append(cell_id)
    phi = oriented_level_set(
        phi, active_domain, level_set_positive_scale)
    pressure_jump = surface_tension / initial_radius
    grid.point_data["phi"] = phi
    # Extend the constant liquid pressure across the background cut-element
    # support.  Active-side constraints subsequently zero truly inactive
    # pressure DOFs; setting dry vertices to zero here would instead create a
    # spurious pressure gradient inside every retained cut cell.
    grid.point_data["Pressure"] = np.full(points.shape[0], pressure_jump)
    grid.point_data["Velocity"] = np.zeros((points.shape[0], 3), dtype=float)
    grid.save(mesh_path)

    apex_coordinate = center[wall_axis] + (
        initial_radius * wall_inward[wall_axis])
    gauge_point = np.zeros(3, dtype=float)
    gauge_point[tangent_axis] = center[tangent_axis]
    gauge_point[wall_axis] = (
        wall_coordinate +
        max(0.25 * (apex_coordinate - wall_coordinate), 1.0e-8) *
        wall_inward[wall_axis])
    gauge_node = nearest_negative_level_set_node(
        points,
        active_signed_level_set(phi, active_domain),
        gauge_point,
    )
    (case_dir / "pressure_gauge.csv").write_text(
        f"node_id,pressure\n{gauge_node},{pressure_jump:.16g}\n",
        encoding="utf-8",
    )

    configure_sessile_solver_xml(
        case_dir,
        steps,
        time_step_size,
        equilibrium_angle_degrees,
        surface_tension,
        mobility,
        slip_length,
        dynamic,
        # DynamicContactAngle is assembled on the sharp CutVolume wall and
        # contact domains.  It has no diffuse active-domain width, so a
        # nonzero value is an unsupported model rather than a mesh-scale
        # regularization choice.
        0.0,
        wall_face,
        contact_line_model,
        active_domain,
    )
    predicted_initial_speed = (
        mobility * surface_tension *
        (math.cos(math.radians(equilibrium_angle_degrees)) -
         math.cos(math.radians(initial_angle_degrees)))
    )
    benchmark = {
        "benchmark": (
            "synthetic Ren--E dynamic sessile contact-line relaxation"
            if dynamic else
            ("synthetic stationary Ren--E sessile-drop equilibrium"
             if contact_line_model == "dynamic" else
             "synthetic stationary prescribed-angle sessile-drop equilibrium")
        ),
        "representation": "unfitted_level_set",
        "capillary_geometry": {
            "wall_bottom": "sessile_circle_2d",
            "wall_left": "vertical_wall_attached_circle_2d",
            "wall_right": "vertical_wall_attached_circle_2d",
            "wall_top": "ceiling_attached_circle_2d",
        }[wall_face],
        "capillary_radius": initial_radius,
        "active_domain": active_domain,
        "tangent_center_offset": tangent_center_offset,
        "initial_active_pressure": pressure_jump,
        # Keep the post-processed kinetic energy on the same physical scale
        # as the generated solver deck above.  Reusing the dimensional
        # capillary-wave density here inflated the sessile kinetic term by
        # 998.2 even though this benchmark solves with rho=1.
        "density": 1.0,
        "surface_tension": surface_tension,
        "viscosity": 0.1,
        "initial_pressure_extension": (
            "constant gamma/R on background support; inactive pressure DOFs "
            "are removed by active-side constraints"
        ),
        "mesh_resolution": {
            "nx": nx,
            "ny": ny,
            "h": 1.0 / max(nx, ny),
            "cell_type": "Triangle3" if simplex_mesh else "Quad4",
            "cell_count": int(grid.n_cells),
        },
        "sessile_contact": {
            "wall": wall_face,
            "wall_axis": wall_axis,
            "wall_coordinate": wall_coordinate,
            "wall_tangent_axis": tangent_axis,
            "wall_normal": list(wall_spec["wall_normal"]),
            "wall_tangent": list(wall_spec["wall_tangent"]),
            "wall_geometry_contract": (
                "unit_square axis-aligned boundary; wall normal points out of "
                "the fluid domain into the solid; positive coordinate tangent "
                "orders the two contact roots"
            ),
            **({"wall_y": wall_coordinate}
               if wall_axis == 1 else {"wall_x": wall_coordinate}),
            "active_domain": active_domain,
            "level_set_positive_scale": level_set_positive_scale,
            "initial_contact_angle_degrees": initial_angle_degrees,
            "equilibrium_contact_angle_degrees": equilibrium_angle_degrees,
            "contact_angle_perturbation_degrees": (
                initial_angle_degrees - equilibrium_angle_degrees),
            "circle_center": [float(center[0]), float(center[1]), 0.0],
            "circle_radius": initial_radius,
            "equilibrium_reference_radius": radius,
            "liquid_area_contract": "fixed_at_equilibrium_reference_cap",
            "expected_initial_half_footprint": half_footprint,
            "expected_initial_footprint": 2.0 * half_footprint,
            "expected_initial_liquid_area": expected_area,
            "dynamic": dynamic,
            "contact_line_model": (
                "DynamicContactAngle" if contact_line_model == "dynamic"
                else "PrescribedContactAngle"),
            **({
                "wall_slip_model": "Navier",
                "mobility": mobility,
                "line_friction": 1.0 / mobility,
                "slip_length": slip_length,
            } if contact_line_model == "dynamic" else {
                "level_set_geometry_owner": (
                    "accepted_state_wall_aware_repair"),
                "momentum_owner": "young_wall_energy",
            }),
            "curvature_projection_narrow_band_width": 1.0 / max(nx, ny),
            "discrete_contact_initialization": (
                "wall-adjacent LinearCorner fragments replaced by exact "
                "initial-angle tangent planes"
                if (dynamic or
                    initialize_discrete_static_contact_geometry) else
                "unmodified analytic circular-cap signed distance sampled at "
                "affine mesh vertices; generated-chord angle error is measured"),
            "discrete_contact_initialization_local_overwrite": (
                dynamic or initialize_discrete_static_contact_geometry),
            "discrete_static_contact_initialization": (
                initialize_discrete_static_contact_geometry),
            "discrete_contact_initialization_cell_ids": contact_cell_ids,
            **({
                "predicted_initial_contact_line_speed":
                    predicted_initial_speed,
                "ren_e_relation": (
                    "V = mobility*gamma*(cos(theta_e)-cos(theta_d))"),
                "ren_e_direct_observable": (
                    "wall-interpolated solved contact-fluid velocity dot "
                    "outward footprint direction at the phi=0 wall "
                    "intersections"),
            } if contact_line_model == "dynamic" else {}),
            "geometric_speed_observable": (
                "accepted-time finite difference of the phi=0 wall footprint; "
                "reported separately as a transport/kinematic diagnostic"
            ),
        },
        "dimensions_m": {
            "tank_length": 1.0,
            "tank_height": 1.0,
            "profile_window_x_min": 0.5,
        },
        "pressure_gauge": {
            "node_id": gauge_node,
            "expected_initial_hydrostatic_pressure": pressure_jump,
            "constraint_applied": False,
            "role": "read-only interior pressure probe; free-surface traction anchors pressure",
        },
    }
    (case_dir / "benchmark.json").write_text(
        json.dumps(benchmark, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def capillary_arc2d_phi(points: np.ndarray) -> np.ndarray:
    return np.sqrt((points[:, 0] - CAPILLARY_ARC_CENTER_X) ** 2 +
                   (points[:, 1] - CAPILLARY_ARC_CENTER_Y) ** 2) - (
                       CAPILLARY_ARC_RADIUS)


def write_capillary_arc2d_case(case_dir: Path,
                               steps: int,
                               pressure_jump: float = 0.0,
                               nx: int = 8,
                               ny: int = 8,
                               simplex_mesh: bool = False,
                               active_domain: str = "LevelSetNegative",
                               ) -> None:
    active_domain = normalized_active_domain(active_domain)
    write_mini_case(
        case_dir, steps, static=True, nx=nx, ny=ny,
        simplex_mesh=simplex_mesh)
    remove_synthetic_pressure_pin(case_dir)
    set_case_active_domain(case_dir, active_domain)

    mesh_path = case_dir / "mesh/background/mesh-complete.mesh.vtu"
    grid = pv.read(mesh_path)
    points = np.asarray(grid.points, dtype=float)
    phi = oriented_level_set(
        capillary_arc2d_phi(points), active_domain, 1.0)
    grid.point_data["phi"] = phi
    # Pressure DOFs whose vertices have phi>0 can still have active liquid
    # support in a cut cell and are intentionally retained by the production
    # active-side constraint.  Extend the constant Young--Laplace preload over
    # the complete background P1 support; truly inactive DOFs are constrained
    # later.  A sign-based zero here creates an artificial O(jump/h) cut-cell
    # gradient and is not a discrete equilibrium initialization.
    grid.point_data["Pressure"] = np.full(points.shape[0], pressure_jump,
                                           dtype=float)
    grid.point_data["Velocity"] = np.zeros((points.shape[0], 3), dtype=float)
    grid.save(mesh_path)

    gauge_point = np.array([0.5, 0.0, 0.0], dtype=float)
    gauge_node = nearest_negative_level_set_node(
        points,
        active_signed_level_set(phi, active_domain),
        gauge_point,
    )
    (case_dir / "pressure_gauge.csv").write_text(
        f"node_id,pressure\n{gauge_node},{pressure_jump:.16g}\n",
        encoding="utf-8")
    benchmark = {
        "benchmark": "synthetic zero-gravity capillary arc smoke",
        "representation": "unfitted_level_set",
        "active_domain": active_domain,
        "capillary_arc_radius": CAPILLARY_ARC_RADIUS,
        "initial_active_pressure": pressure_jump,
        "initial_pressure_extension": (
            "constant gamma/R on background support; inactive pressure DOFs "
            "are removed by active-side constraints"
        ),
        "mesh_resolution": {
            "nx": nx,
            "ny": ny,
            "h": 1.0 / max(nx, ny),
            "cell_type": "Triangle3" if simplex_mesh else "Quad4",
            "cell_count": int(grid.n_cells),
        },
        "dimensions_m": {
            "tank_length": 1.0,
            "tank_height": 1.0,
            "profile_window_x_min": 0.5,
        },
        "pressure_gauge": {
            "node_id": gauge_node,
            "expected_initial_hydrostatic_pressure": pressure_jump,
            "constraint_applied": False,
            "role": "read-only pressure probe; free-surface traction anchors pressure",
        },
        "notes": [
            "The wall-supported circular arc starts from zero velocity and zero gravity.",
            "A zero active pressure preload exercises capillary response.",
            "A positive gamma/R liquid pressure jump exercises a Young--Laplace capillary balance.",
        ],
    }
    (case_dir / "benchmark.json").write_text(
        json.dumps(benchmark, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")


def capillary_droplet2d_phi(points: np.ndarray) -> np.ndarray:
    return np.sqrt((points[:, 0] - CAPILLARY_DROPLET_CENTER_X) ** 2 +
                   (points[:, 1] - CAPILLARY_DROPLET_CENTER_Y) ** 2) - (
                       CAPILLARY_DROPLET_RADIUS)


def write_capillary_droplet2d_case(case_dir: Path,
                                   steps: int,
                                   pressure_jump: float = 0.0,
                                   nx: int = 8,
                                   ny: int = 8,
                                   simplex_mesh: bool = False,
                                   active_domain: str = "LevelSetNegative",
                                   center_offset: Sequence[float] = (0.0, 0.0),
                                   surface_tension: float | None = None,
                                   ) -> None:
    active_domain = normalized_active_domain(active_domain)
    center_offset = np.asarray(center_offset, dtype=float).reshape(-1)
    if center_offset.shape != (2,) or not np.isfinite(center_offset).all():
        raise ValueError(
            "capillary droplet center offset must contain two finite values")
    if (surface_tension is not None and
            (not math.isfinite(surface_tension) or surface_tension <= 0.0)):
        raise ValueError("capillary droplet surface tension must be positive")
    center = np.asarray([
        CAPILLARY_DROPLET_CENTER_X,
        CAPILLARY_DROPLET_CENTER_Y,
    ], dtype=float) + center_offset
    if (np.any(center - CAPILLARY_DROPLET_RADIUS <= 0.0) or
            np.any(center + CAPILLARY_DROPLET_RADIUS >= 1.0)):
        raise ValueError(
            "closed capillary droplet must remain strictly inside the tank")
    write_mini_case(
        case_dir, steps, static=True, nx=nx, ny=ny,
        simplex_mesh=simplex_mesh)
    remove_synthetic_pressure_pin(case_dir)
    set_case_active_domain(case_dir, active_domain)

    mesh_path = case_dir / "mesh/background/mesh-complete.mesh.vtu"
    grid = pv.read(mesh_path)
    points = np.asarray(grid.points, dtype=float)
    phi = oriented_level_set(
        np.linalg.norm(points[:, :2] - center, axis=1) -
        CAPILLARY_DROPLET_RADIUS,
        active_domain,
        1.0,
    )
    grid.point_data["phi"] = phi
    # CutVolume retains pressure basis functions whose vertices may lie on the
    # inactive side of a cut cell.  Preload the constant liquid jump on the
    # whole background support; active-side constraints remove truly inactive
    # coefficients.  A phi-sign mask here creates an artificial O(dp/h)
    # gradient inside every retained cut cell.
    grid.point_data["Pressure"] = np.full(points.shape[0], pressure_jump)
    grid.point_data["Velocity"] = np.zeros((points.shape[0], 3), dtype=float)
    grid.save(mesh_path)

    gauge_point = np.array([center[0], center[1], 0.0], dtype=float)
    gauge_node = nearest_negative_level_set_node(
        points,
        active_signed_level_set(phi, active_domain),
        gauge_point,
    )
    (case_dir / "pressure_gauge.csv").write_text(
        f"node_id,pressure\n{gauge_node},{pressure_jump:.16g}\n",
        encoding="utf-8")
    benchmark = {
        "benchmark": "synthetic zero-gravity capillary droplet equilibrium smoke",
        "representation": "unfitted_level_set",
        "active_domain": active_domain,
        "spatial_dimension": 2,
        "density": 1.0,
        "capillary_geometry": "droplet2d",
        "capillary_radius": CAPILLARY_DROPLET_RADIUS,
        "circle_center": center.tolist(),
        "circle_center_offset": center_offset.tolist(),
        "initial_active_pressure": pressure_jump,
        "initial_pressure_extension": (
            "constant gamma/R on background support; inactive pressure DOFs "
            "are removed by active-side constraints"),
        "mesh_resolution": {
            "nx": nx,
            "ny": ny,
            "h": 1.0 / max(nx, ny),
            "cell_type": "Triangle3" if simplex_mesh else "Quad4",
            "cell_count": int(grid.n_cells),
        },
        "dimensions_m": {
            "tank_length": 1.0,
            "tank_height": 1.0,
            "profile_window_x_min": 0.5,
        },
        "pressure_gauge": {
            "node_id": gauge_node,
            "expected_initial_hydrostatic_pressure": pressure_jump,
            "constraint_applied": False,
            "role": "read-only pressure probe; free-surface traction anchors pressure",
        },
        "notes": [
            "The closed circular droplet starts from zero velocity and zero gravity.",
            "A positive gamma/R liquid pressure jump exercises a static Young--Laplace equilibrium.",
        ],
    }
    if surface_tension is not None:
        benchmark["surface_tension"] = float(surface_tension)
    (case_dir / "benchmark.json").write_text(
        json.dumps(benchmark, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")


def capillary_wave_wavenumber() -> float:
    return 2.0 * math.pi / CAPILLARY_WAVE_WAVELENGTH


def capillary_wave_omega(surface_tension: float) -> float:
    k = capillary_wave_wavenumber()
    return math.sqrt(
        max(float(surface_tension), 0.0) * k ** 3 *
        math.tanh(k * CAPILLARY_WAVE_DEPTH) /
        CAPILLARY_WAVE_DENSITY
    )


def capillary_wave_minimum_steps_for_frequency_fit(
        surface_tension: float,
        time_step_size: float,
        minimum_phase_span: float = CAPILLARY_WAVE_MINIMUM_FREQUENCY_PHASE_SPAN,
) -> int:
    """Return the shortest nominal history that spans the frequency-fit gate."""
    dt = float(time_step_size)
    phase_span = float(minimum_phase_span)
    omega = capillary_wave_omega(surface_tension)
    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError("capillary-wave time step size must be positive and finite")
    if not math.isfinite(phase_span) or phase_span <= 0.0:
        raise ValueError("capillary-wave minimum phase span must be positive and finite")
    if not math.isfinite(omega) or omega <= 0.0:
        raise ValueError(
            "capillary-wave surface tension must give a positive finite frequency"
        )

    phase_per_step = omega * dt
    steps = max(1, int(math.ceil(phase_span / phase_per_step)))
    # Guard the mathematical lower bound against a downward floating-point
    # rounding in the quotient used by ceil.
    if float(steps) * phase_per_step < phase_span:
        steps += 1
    return steps


def capillary_wave_height(x: np.ndarray,
                          time_s: float,
                          surface_tension: float) -> np.ndarray:
    k = capillary_wave_wavenumber()
    omega = capillary_wave_omega(surface_tension)
    return (
        CAPILLARY_WAVE_BASE_HEIGHT +
        CAPILLARY_WAVE_AMPLITUDE * np.cos(k * x) * math.cos(omega * time_s)
    )


def capillary_wave2d_phi(points: np.ndarray) -> np.ndarray:
    return points[:, 1] - capillary_wave_height(points[:, 0], 0.0, 1.0)


def capillary_wave_initial_pressure(points: np.ndarray,
                                    surface_tension: float) -> np.ndarray:
    """Linear finite-depth pressure mode at maximum wave displacement.

    At ``t=0`` the standing wave has zero velocity but nonzero pressure.  The
    mode below follows from ``p=-rho*d(Phi)/dt`` and satisfies the linearized
    Young--Laplace condition ``p=gamma*kappa`` at the mean free surface.
    It is continued smoothly through dry background vertices because cut-cell
    interpolation needs coefficients on both sides of the interface.
    """
    k = capillary_wave_wavenumber()
    return (
        float(surface_tension) * CAPILLARY_WAVE_AMPLITUDE * k ** 2 *
        np.cos(k * points[:, 0]) *
        np.cosh(k * points[:, 1]) /
        math.cosh(k * CAPILLARY_WAVE_DEPTH)
    )


def nearest_negative_level_set_node(points: np.ndarray,
                                    phi: np.ndarray,
                                    target: np.ndarray) -> int:
    active = np.flatnonzero(phi < 0.0)
    if active.size == 0:
        raise ValueError("synthetic case has no negative level-set vertices")
    distances = np.linalg.norm(points[active] - target, axis=1)
    return int(active[int(np.argmin(distances))])


def write_capillary_wave_reference_profile(profile_path: Path,
                                           time_s: float,
                                           surface_tension: float) -> None:
    xs = np.linspace(0.0, 1.0, 129)
    ys = capillary_wave_height(xs, time_s, surface_tension)
    # compare_test05_profiles treats reference-profile files as centimeters.
    np.savetxt(profile_path, np.column_stack((100.0 * xs, 100.0 * ys)),
               fmt="%.16g")


def write_capillary_wave2d_case(case_dir: Path,
                                steps: int,
                                surface_tension: float,
                                time_step_size: float | None,
                                nx: int = 8,
                                ny: int = 8,
                                simplex_mesh: bool = False) -> None:
    write_mini_case(
        case_dir, steps, static=True, nx=nx, ny=ny,
        simplex_mesh=simplex_mesh)
    remove_synthetic_pressure_pin(case_dir)
    configure_capillary_wave_wall_boundary_contract(case_dir)

    mesh_path = case_dir / "mesh/background/mesh-complete.mesh.vtu"
    grid = pv.read(mesh_path)
    points = np.asarray(grid.points, dtype=float)
    phi = capillary_wave2d_phi(points)
    grid.point_data["phi"] = phi
    initial_pressure = capillary_wave_initial_pressure(points, surface_tension)
    grid.point_data["Pressure"] = initial_pressure
    grid.point_data["Velocity"] = np.zeros((points.shape[0], 3), dtype=float)
    grid.save(mesh_path)

    dt = 0.001 if time_step_size is None else float(time_step_size)
    final_time = float(steps) * dt
    profile_path = case_dir / "capillary_wave_reference_profile.csv"
    write_capillary_wave_reference_profile(profile_path, final_time, surface_tension)

    gauge_point = np.array([0.5, CAPILLARY_WAVE_BASE_HEIGHT, 0.0], dtype=float)
    gauge_node = nearest_negative_level_set_node(points, phi, gauge_point)
    gauge_pressure = float(initial_pressure[gauge_node])
    (case_dir / "pressure_gauge.csv").write_text(
        f"node_id,pressure\n{gauge_node},{gauge_pressure:.16g}\n",
        encoding="utf-8")

    k = capillary_wave_wavenumber()
    omega = capillary_wave_omega(surface_tension)
    benchmark = {
        "benchmark": "synthetic small-amplitude capillary wave smoke",
        "representation": "unfitted_level_set",
        "capillary_geometry": "standing_wave_2d",
        "capillary_wave": {
            "mode": "standing_cosine",
            "base_height": CAPILLARY_WAVE_BASE_HEIGHT,
            "amplitude": CAPILLARY_WAVE_AMPLITUDE,
            "wavelength": CAPILLARY_WAVE_WAVELENGTH,
            "wavenumber": k,
            "depth": CAPILLARY_WAVE_DEPTH,
            "finite_depth_factor": math.tanh(k * CAPILLARY_WAVE_DEPTH),
            "density": CAPILLARY_WAVE_DENSITY,
            "surface_tension": float(surface_tension),
            "omega": omega,
            "final_time_s": final_time,
        },
        "mesh_resolution": {
            "nx": nx,
            "ny": ny,
            "h": 1.0 / max(nx, ny),
            "cell_type": "Triangle3" if simplex_mesh else "Quad4",
            "cell_count": int(grid.n_cells),
        },
        "boundary_contract": {
            "wall_left": "impermeable normal-only; vertical tangential motion free",
            "wall_right": "impermeable normal-only; vertical tangential motion free",
            "wall_bottom": "impermeable normal-only; horizontal tangential motion free",
            "wall_top": "dry open numerical truncation (LevelSetOutflow)",
            "vertical_wall_effective_direction": [1.0, 0.0],
            "required_transport_extension": "wall_compatible_normal",
        },
        "dimensions_m": {
            "tank_length": 1.0,
            "tank_height": 1.0,
            "profile_window_x_min": 0.5,
            "profile_window_x_max": 1.0,
        },
        "pressure_gauge": {
            "node_id": gauge_node,
            "expected_initial_pressure": gauge_pressure,
            "expected_initial_capillary_pressure": gauge_pressure,
            "expected_initial_hydrostatic_pressure_component": 0.0,
            "constraint_applied": False,
            "role": "read-only pressure probe; free-surface traction anchors pressure",
        },
        "reference_profiles": [
            {
                "time_s": final_time,
                "path": str(profile_path),
            },
        ],
        "notes": [
            "The initial interface is a small standing cosine perturbation.",
            "The reference profile uses omega^2 = (gamma/rho) k^3 tanh(k h) for a finite-depth pure capillary wave.",
            "Vertical walls constrain only normal velocity so their contact points may move tangentially.",
            "All solid walls are impermeable-slip, and level-set transport uses wall-compatible normal extension.",
        ],
    }
    (case_dir / "benchmark.json").write_text(
        json.dumps(benchmark, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")


TETRA10_EDGE_PAIRS = (
    (0, 1),
    (1, 2),
    (2, 0),
    (0, 3),
    (1, 3),
    (2, 3),
)
TETRA10_FACE_CORNERS = (
    (1, 2, 3),
    (0, 3, 2),
    (0, 1, 3),
    (0, 2, 1),
)


def curved_tet3d_surface_height(points: np.ndarray) -> np.ndarray:
    x = points[:, 0]
    z = points[:, 2]
    return 0.55 + 0.08 * np.sin(np.pi * x) * np.cos(2.0 * np.pi * z / 0.25)


def curved_tet3d_phi(points: np.ndarray) -> np.ndarray:
    return points[:, 1] - curved_tet3d_surface_height(points)


def curved_tet3d_pressure(points: np.ndarray) -> np.ndarray:
    rho = 998.2
    gravity = 9.81
    return rho * gravity * np.maximum(curved_tet3d_surface_height(points) - points[:, 1], 0.0)


def curved_tet3d_midpoint(base_points: np.ndarray, a: int, b: int) -> np.ndarray:
    point = 0.5 * (base_points[a] + base_points[b])
    x, y, z = point
    displacement = np.array([
        0.0,
        0.012 * np.sin(np.pi * x) * np.sin(np.pi * y) * np.sin(np.pi * z / 0.25),
        0.006 * np.sin(np.pi * x) * np.sin(np.pi * y) * np.sin(2.0 * np.pi * z / 0.25),
    ])
    return point + displacement


def orient_tetra_positive(points: np.ndarray, tet: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    a, b, c, d = tet
    volume = float(np.dot(np.cross(points[b] - points[a], points[c] - points[a]),
                          points[d] - points[a]))
    if volume < 0.0:
        return (a, b, d, c)
    return tet


def write_curved_tet3d_grid(case_dir: Path) -> tuple[int, float]:
    nx, ny, nz = 2, 3, 2
    length, height, width = 1.0, 1.0, 0.25
    xs = np.linspace(0.0, length, nx + 1)
    ys = np.linspace(0.0, height, ny + 1)
    zs = np.linspace(0.0, width, nz + 1)
    base_points = np.array([[x, y, z] for z in zs for y in ys for x in xs], dtype=float)

    def node(i: int, j: int, k: int) -> int:
        return k * (ny + 1) * (nx + 1) + j * (nx + 1) + i

    linear_tets: list[tuple[int, int, int, int]] = []
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                n000 = node(i, j, k)
                n100 = node(i + 1, j, k)
                n010 = node(i, j + 1, k)
                n110 = node(i + 1, j + 1, k)
                n001 = node(i, j, k + 1)
                n101 = node(i + 1, j, k + 1)
                n011 = node(i, j + 1, k + 1)
                n111 = node(i + 1, j + 1, k + 1)
                linear_tets.extend([
                    (n000, n001, n011, n111),
                    (n000, n011, n010, n111),
                    (n000, n010, n110, n111),
                    (n000, n110, n100, n111),
                    (n000, n100, n101, n111),
                    (n000, n101, n001, n111),
                ])
    linear_tets = [orient_tetra_positive(base_points, tet) for tet in linear_tets]

    points = [point.copy() for point in base_points]
    edge_midpoints: dict[tuple[int, int], int] = {}

    def midpoint_id(a: int, b: int) -> int:
        key = tuple(sorted((int(a), int(b))))
        if key not in edge_midpoints:
            edge_midpoints[key] = len(points)
            points.append(curved_tet3d_midpoint(base_points, key[0], key[1]))
        return edge_midpoints[key]

    tet10_cells: list[list[int]] = []
    for tet in linear_tets:
        cell = list(tet)
        cell.extend(midpoint_id(tet[a], tet[b]) for a, b in TETRA10_EDGE_PAIRS)
        tet10_cells.append(cell)

    point_array_values = np.asarray(points, dtype=float)
    cells = np.asarray([[10, *cell] for cell in tet10_cells], dtype=np.int64).ravel()
    cell_types = np.full(len(tet10_cells), int(pv.CellType.QUADRATIC_TETRA), dtype=np.uint8)
    grid = pv.UnstructuredGrid(cells, cell_types, point_array_values)
    grid.point_data["GlobalNodeID"] = np.arange(grid.n_points, dtype=np.int64)
    grid.point_data["phi"] = curved_tet3d_phi(point_array_values)
    grid.point_data["Pressure"] = curved_tet3d_pressure(point_array_values)
    grid.point_data["Velocity"] = np.zeros((grid.n_points, 3), dtype=float)
    grid.cell_data["GlobalElementID"] = np.arange(grid.n_cells, dtype=np.int64)

    mesh_dir = case_dir / "mesh/background"
    surface_dir = mesh_dir / "mesh-surfaces"
    surface_dir.mkdir(parents=True)
    grid.save(mesh_dir / "mesh-complete.mesh.vtu", binary=False)

    face_counts: dict[tuple[int, int, int], tuple[int, list[int]]] = {}
    for cell in tet10_cells:
        corners = cell[:4]
        for face in TETRA10_FACE_CORNERS:
            face_corners = [corners[index] for index in face]
            key = tuple(sorted(face_corners))
            if key not in face_counts:
                mids = [
                    midpoint_id(face_corners[0], face_corners[1]),
                    midpoint_id(face_corners[1], face_corners[2]),
                    midpoint_id(face_corners[2], face_corners[0]),
                ]
                face_counts[key] = (0, [*face_corners, *mids])
            count, stored = face_counts[key]
            face_counts[key] = (count + 1, stored)

    surfaces: dict[str, list[list[int]]] = {
        "wall_left": [],
        "wall_right": [],
        "wall_bottom": [],
        "wall_front": [],
        "wall_back": [],
        "wall_top": [],
    }
    tol = 1.0e-12
    for key, (count, face_nodes) in face_counts.items():
        if count != 1:
            continue
        center = np.mean(base_points[np.asarray(key, dtype=np.int64)], axis=0)
        if abs(center[0]) <= tol:
            surfaces["wall_left"].append(face_nodes)
        elif abs(center[0] - length) <= tol:
            surfaces["wall_right"].append(face_nodes)
        elif abs(center[1]) <= tol:
            surfaces["wall_bottom"].append(face_nodes)
        elif abs(center[2]) <= tol:
            surfaces["wall_front"].append(face_nodes)
        elif abs(center[2] - width) <= tol:
            surfaces["wall_back"].append(face_nodes)
        elif abs(center[1] - height) <= tol:
            surfaces["wall_top"].append(face_nodes)

    for name, faces in surfaces.items():
        if not faces:
            raise RuntimeError(f"curvedtet3d surface {name!r} has no faces")
        used = sorted({node_id for face in faces for node_id in face})
        local = {node_id: index for index, node_id in enumerate(used)}
        surface_cells = np.asarray(
            [[6, *(local[node_id] for node_id in face)] for face in faces],
            dtype=np.int64,
        ).ravel()
        surface_types = np.full(len(faces), int(pv.CellType.QUADRATIC_TRIANGLE), dtype=np.uint8)
        surface = pv.UnstructuredGrid(surface_cells, surface_types, point_array_values[used])
        surface.point_data["GlobalNodeID"] = np.asarray(used, dtype=np.int64)
        surface.cell_data["GlobalElementID"] = np.arange(len(faces), dtype=np.int64)
        surface.save(surface_dir / f"{name}.vtu", binary=False)

    gauge_point = np.array([0.5, 0.0, 0.125], dtype=float)
    gauge_node = int(np.argmin(np.linalg.norm(point_array_values - gauge_point, axis=1)))
    gauge_pressure = float(curved_tet3d_pressure(point_array_values[[gauge_node]])[0])
    return gauge_node, gauge_pressure


def write_curved_tet3d_solver_xml(case_dir: Path,
                                  steps: int,
                                  gauge_node: int,
                                  gauge_pressure: float) -> None:
    (case_dir / "pressure_gauge.csv").write_text(
        f"node_id,pressure\n{gauge_node},{gauge_pressure:.16g}\n", encoding="utf-8")
    benchmark = {
        "benchmark": "synthetic curved Tetra10 open-vessel free-surface smoke",
        "representation": "unfitted_level_set",
        "dimensions_m": {
            "tank_length": 1.0,
            "tank_height": 1.0,
            "tank_width": 0.25,
            "profile_window_x_min": 0.5,
        },
        "pressure_gauge": {
            "node_id": gauge_node,
            "expected_initial_hydrostatic_pressure": gauge_pressure,
        },
        "notes": [
            "Generated at run time to exercise solver-level curved 3D Tetra10 geometry.",
            "Quadratic tetrahedra use curved midside coordinates and quadratic triangle wall files.",
        ],
    }
    (case_dir / "benchmark.json").write_text(
        json.dumps(benchmark, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    face_blocks = "\n".join(
        f"""  <Add_face name="{name}">
    <Face_file_path>mesh/background/mesh-surfaces/{name}.vtu</Face_file_path>
  </Add_face>"""
        for name in ("wall_left", "wall_right", "wall_bottom", "wall_front", "wall_back", "wall_top")
    )
    wall_bc_blocks = "\n".join(
        f"""  <Add_BC name="{name}">
    <Type>Dir</Type>
    <Value>0.0</Value>
  </Add_BC>"""
        for name in ("wall_left", "wall_right", "wall_bottom", "wall_front", "wall_back")
    )
    (case_dir / "solver.xml").write_text(f"""<?xml version="1.0" encoding="UTF-8" ?>
<svMultiPhysicsFile version="0.1">

<GeneralSimulationParameters>
  <Use_new_OOP_solver>true</Use_new_OOP_solver>
  <Continue_previous_simulation>false</Continue_previous_simulation>
  <Number_of_spatial_dimensions>3</Number_of_spatial_dimensions>
  <Number_of_time_steps>{steps}</Number_of_time_steps>
  <Time_step_size>0.001</Time_step_size>
  <Spectral_radius_of_infinite_time_step>0.50</Spectral_radius_of_infinite_time_step>
  <Searched_file_name_to_trigger_stop>STOP_SIM</Searched_file_name_to_trigger_stop>
  <Save_results_to_VTK_format>true</Save_results_to_VTK_format>
  <Name_prefix_of_saved_VTK_files>result</Name_prefix_of_saved_VTK_files>
  <Increment_in_saving_VTK_files>1</Increment_in_saving_VTK_files>
  <Start_saving_after_time_step>1</Start_saving_after_time_step>
  <Increment_in_saving_restart_files>{steps}</Increment_in_saving_restart_files>
  <Convert_BIN_to_VTK_format>0</Convert_BIN_to_VTK_format>
  <Verbose>1</Verbose>
  <Warning>0</Warning>
  <Debug>0</Debug>
</GeneralSimulationParameters>

<Add_mesh name="tank">
  <Mesh_file_path>mesh/background/mesh-complete.mesh.vtu</Mesh_file_path>
{face_blocks}
</Add_mesh>

<Add_equation type="level_set">
  <Coupled>true</Coupled>
  <Min_iterations>1</Min_iterations>
  <Max_iterations>4</Max_iterations>
  <Tolerance>1.0e-4</Tolerance>
  <Level_set_field_name>phi</Level_set_field_name>
  <Operator_tag>equations</Operator_tag>
  <Level_set_source>prescribed_data</Level_set_source>
  <Velocity_source>coupled_field</Velocity_source>
  <Velocity_field_name>Velocity</Velocity_field_name>
  <Auto_register_velocity_field>true</Auto_register_velocity_field>
  <Enable_SUPG>true</Enable_SUPG>
  <SUPG_tau_scale>0.5</SUPG_tau_scale>
  <Enable_reinitialization>false</Enable_reinitialization>
  <Enable_volume_correction>false</Enable_volume_correction>
  <Output type="Spatial">
    <Level_set>true</Level_set>
    <Generated_interface>true</Generated_interface>
    <Surface_position>true</Surface_position>
  </Output>
  <Output type="Volume_integral">
    <Volume>true</Volume>
  </Output>
  <LS type="Direct">
    <Linear_algebra type="eigen">
      <Preconditioner>none</Preconditioner>
    </Linear_algebra>
    <Max_iterations>1</Max_iterations>
    <Krylov_space_dimension>1</Krylov_space_dimension>
    <Tolerance>1.0e-4</Tolerance>
    <Absolute_tolerance>1.0e-4</Absolute_tolerance>
  </LS>
</Add_equation>

<Add_equation type="fluid">
  <Coupled>true</Coupled>
  <Min_iterations>1</Min_iterations>
  <Max_iterations>8</Max_iterations>
  <Tolerance>1.0e-4</Tolerance>
  <Backflow_stabilization_coefficient>0.0</Backflow_stabilization_coefficient>
  <Density>998.2</Density>
  <Force_x>0.0</Force_x>
  <Force_y>-9.81</Force_y>
  <Force_z>0.0</Force_z>
  <Hydrostatic_pressure_initialization>true</Hydrostatic_pressure_initialization>
  <Hydrostatic_pressure_reference>0.0</Hydrostatic_pressure_reference>
  <Hydrostatic_pressure_reference_point>0.0 0.55 0.0</Hydrostatic_pressure_reference_point>
  <Node_pressure_constraints>
    <Id_type>Global_vertex_gid</Id_type>
    <Values_file_path>pressure_gauge.csv</Values_file_path>
  </Node_pressure_constraints>
  <Viscosity model="Constant">
    <Value>1.003e-3</Value>
  </Viscosity>
  <Output type="Spatial">
    <Velocity>true</Velocity>
    <Pressure>true</Pressure>
    <Divergence>true</Divergence>
  </Output>
  <Output type="Volume_integral">
    <Volume>true</Volume>
  </Output>
  <LS type="Direct">
    <Linear_algebra type="eigen">
      <Preconditioner>none</Preconditioner>
    </Linear_algebra>
    <Max_iterations>1</Max_iterations>
    <Krylov_space_dimension>1</Krylov_space_dimension>
    <Tolerance>1.0e-4</Tolerance>
    <Absolute_tolerance>1.0e-4</Absolute_tolerance>
  </LS>
{wall_bc_blocks}
  <Add_BC name="free_surface">
    <Type>Free_surface</Type>
    <Implementation>UnfittedLevelSet</Implementation>
    <Level_set_field_name>phi</Level_set_field_name>
    <Generated_interface_domain_id>open_vessel_surface</Generated_interface_domain_id>
    <Level_set_isovalue>0.0</Level_set_isovalue>
    <Active_domain>LevelSetNegative</Active_domain>
    <Active_domain_method>CutVolume</Active_domain_method>
    <Enable_velocity_extension>false</Enable_velocity_extension>
    <Velocity_extension_diffusivity>1.0</Velocity_extension_diffusivity>
    <External_pressure>0.0</External_pressure>
    <Surface_tension>0.0</Surface_tension>
    <Enable_cut_cell_stabilization>true</Enable_cut_cell_stabilization>
    <Use_cut_metadata_scale>true</Use_cut_metadata_scale>
    <Cut_cell_pressure_gradient_penalty>1.0</Cut_cell_pressure_gradient_penalty>
  </Add_BC>
</Add_equation>

</svMultiPhysicsFile>
""", encoding="utf-8")


def write_curved_tet3d_case(case_dir: Path, steps: int) -> None:
    case_dir.mkdir(parents=True)
    gauge_node, gauge_pressure = write_curved_tet3d_grid(case_dir)
    write_curved_tet3d_solver_xml(case_dir, steps, gauge_node, gauge_pressure)


def result_path(case_dir: Path, step: int) -> Path:
    names = [
        f"result_{step:03d}.vtu",
        f"result_{step:03d}.pvtu",
        f"1-procs/result_{step:03d}.vtu",
        f"1-procs/result_{step:03d}.pvtu",
    ]
    for name in names:
        candidate = case_dir / name
        if candidate.exists():
            return candidate
    candidates = sorted([*case_dir.rglob("result_*.vtu"), *case_dir.rglob("result_*.pvtu")])
    if candidates:
        return candidates[-1]
    raise FileNotFoundError(f"no result file found under {case_dir}")


def final_result_step(default_step: int,
                      diagnostics: dict[str, Any]) -> int:
    time_loop = diagnostics.get("time_loop", {})
    if isinstance(time_loop, dict):
        accepted_steps = time_loop.get("accepted_steps", [])
        if accepted_steps:
            final_step = accepted_steps[-1].get("step")
            if isinstance(final_step, int) and final_step > 0:
                return final_step
    return default_step


def value_span(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return float(max(values) - min(values))


def parse_active_volume_history(solver_output: str) -> dict[str, Any]:
    context_volumes = [
        float(match.group(1))
        for match in CUT_CONTEXT_VOLUME_RE.finditer(solver_output)
    ]
    assembly_volumes = [
        float(match.group(1))
        for match in CUT_ASSEMBLY_VOLUME_RE.finditer(solver_output)
    ]

    return {
        "cut_context_active_side_volumes": context_volumes,
        "assembly_active_wet_volumes": assembly_volumes,
        "cut_context_active_side_volume_change": value_span(context_volumes),
        "assembly_active_wet_volume_change": value_span(assembly_volumes),
    }


def parse_scalar(value: str) -> Any:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    if value in {"true", "false"}:
        return value == "true"
    try:
        if re.fullmatch(r"[-+]?[0-9]+", value):
            return int(value)
        return float(value)
    except ValueError:
        return value


def parse_key_values(line: str) -> dict[str, Any]:
    values = {
        match.group(1): parse_scalar(match.group(2))
        for match in KEY_VALUE_RE.finditer(line)
    }
    rank_match = RANK_RE.search(line)
    if rank_match is not None:
        values["rank"] = int(rank_match.group(1))
    return values


def parse_jit_specialization_trace(line: str) -> dict[str, Any]:
    values = parse_key_values(line)
    for prefix, pattern in (
        ("minus", JIT_MINUS_SHAPE_RE),
        ("plus", JIT_PLUS_SHAPE_RE),
    ):
        match = pattern.search(line)
        if match is None:
            continue
        values[f"{prefix}_qpts"] = parse_scalar(match.group(1))
        values[f"{prefix}_test_dofs"] = parse_scalar(match.group(2))
        values[f"{prefix}_trial_dofs"] = parse_scalar(match.group(3))
    return values


def parse_interior_face_timing(line: str) -> dict[str, Any]:
    values = {
        match.group(1): parse_scalar(match.group(2))
        for match in INTERIOR_FACE_TIMING_VALUE_RE.finditer(line)
    }
    values["diagnostic"] = "interior_face_timing"
    return values


def parse_cut_volume_timing(line: str) -> dict[str, Any]:
    values = parse_key_values(line)
    values.update({
        match.group(1): parse_scalar(match.group(2))
        for match in INTERIOR_FACE_TIMING_VALUE_RE.finditer(line)
    })
    values["diagnostic"] = "cut_volume_timing"
    return values


def count_key(record: dict[str, Any], fields: tuple[str, ...]) -> str:
    parts = []
    for field in fields:
        value = record.get(field)
        if value is not None:
            parts.append(f"{field}={value}")
    return ",".join(parts) if parts else "unclassified"


def top_counts(counts: dict[str, int], limit: int = 24) -> dict[str, int]:
    return {
        key: counts[key]
        for key in sorted(counts, key=lambda item: (-counts[item], item))[:limit]
    }


def increment_count(counts: dict[str, int], key: str) -> None:
    counts[key] = counts.get(key, 0) + 1


def diagnostic_flag(record: dict[str, Any], name: str) -> bool:
    value = record.get(name, 0)
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes")
    return bool(value)


def parse_count_summary(value: Any) -> dict[str, int]:
    if not isinstance(value, str) or not value or value == "none":
        return {}
    counts: dict[str, int] = {}
    for part in value.split(","):
        if ":" not in part:
            continue
        name, raw_count = part.split(":", 1)
        name = name.strip()
        try:
            count = int(raw_count.strip())
        except ValueError:
            continue
        if name and count > 0:
            counts[name] = counts.get(name, 0) + count
    return counts


def jit_shape_key(record: dict[str, Any]) -> str:
    shape = count_key(record, ("trigger", "domain", "role"))
    if "minus_qpts" in record:
        shape += (
            f",minus_qpts={record.get('minus_qpts')}"
            f",minus_test={record.get('minus_test_dofs')}"
            f",minus_trial={record.get('minus_trial_dofs')}"
        )
    elif "n_qpts" in record:
        shape += (
            f",qpts={record.get('n_qpts')}"
            f",test={record.get('n_test_dofs')}"
            f",trial={record.get('n_trial_dofs')}"
        )
    if "plus_qpts" in record:
        shape += (
            f",plus_qpts={record.get('plus_qpts')}"
            f",plus_test={record.get('plus_test_dofs')}"
            f",plus_trial={record.get('plus_trial_dofs')}"
        )
    if "affine" in record:
        shape += f",affine={record.get('affine')}"
    return shape


def timing_mode_key(record: dict[str, Any]) -> str:
    return f"matrix={record.get('matrix', '?')},vector={record.get('vector', '?')}"


def summarize_timing_modes(records: list[dict[str, Any]],
                           count_fields: tuple[str, ...],
                           time_fields: tuple[str, ...]) -> dict[str, Any]:
    summaries: dict[str, dict[str, Any]] = {}
    for record in records:
        key = timing_mode_key(record)
        summary = summaries.setdefault(key, {"records": 0})
        summary["records"] += 1
        for field in count_fields:
            value = record.get(field)
            if isinstance(value, int):
                target = f"max_{field}"
                summary[target] = max(int(summary.get(target, value)), value)
        for field in time_fields:
            value = record.get(field)
            if isinstance(value, (int, float)):
                target = f"max_{field}_seconds"
                summary[target] = max(float(summary.get(target, value)), float(value))
    return summaries


def convert_match(match: re.Match[str]) -> dict[str, Any]:
    return {
        name: parse_scalar(value)
        for name, value in match.groupdict().items()
        if value is not None
    }


def distribution(values: list[int]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    return counts


def numeric_range(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    return {
        "min": float(min(values)),
        "max": float(max(values)),
        "mean": float(sum(values) / len(values)),
    }


def sum_numeric(records: list[dict[str, Any]], key: str) -> float:
    return float(sum(
        float(record[key])
        for record in records
        if isinstance(record.get(key), (int, float))
    ))


def sum_integer(records: list[dict[str, Any]], key: str) -> int:
    return int(sum(
        int(record[key])
        for record in records
        if isinstance(record.get(key), int)
    ))


def finite_min(values: list[float], default: float = 0.0) -> float:
    finite = [value for value in values if np.isfinite(value)]
    return float(min(finite)) if finite else default


def finite_max(values: list[float], default: float = 0.0) -> float:
    finite = [value for value in values if np.isfinite(value)]
    return float(max(finite)) if finite else default


def group_rank_records(records: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    groups: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    ranks_seen: set[int] = set()
    for record in records:
        rank = record.get("rank")
        if isinstance(rank, int):
            if rank in ranks_seen and current:
                groups.append(current)
                current = []
                ranks_seen = set()
            ranks_seen.add(rank)
        elif current:
            groups.append(current)
            current = []
            ranks_seen = set()
        current.append(record)
        if not isinstance(rank, int):
            groups.append(current)
            current = []
    if current:
        groups.append(current)
    return groups


def aggregate_cut_volume_assemblies(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups = []
    for group_index, group in enumerate(group_rank_records(records)):
        active_records = [
            record for record in group
            if float(record.get("rules", 0) or 0) > 0.0
        ]
        records_for_extrema = active_records or group
        aggregate: dict[str, Any] = {
            "diagnostic": "cut_volume_assembly_global",
            "group_index": group_index,
            "rank_records": len(group),
        }
        first = group[0]
        for key in ("marker", "side"):
            if key in first:
                aggregate[key] = first[key]
        for key in (
            "active_wet_volume",
            "cut_cell_active_wet_volume",
            "full_cell_active_wet_volume",
        ):
            aggregate[key] = sum_numeric(group, key)
        for key in (
            "rules",
            "cut_cell_rules",
            "full_cell_rules",
            "quadrature_points",
            "null_rules",
            "zero_quadrature_rules",
            "nonfinite_measure_rules",
            "negative_measure_rules",
            "nonfinite_volume_fraction_rules",
        ):
            aggregate[key] = sum_integer(group, key)
        for key in ("min_rule_measure", "min_volume_fraction", "min_exact_order"):
            values = [
                float(record[key])
                for record in records_for_extrema
                if isinstance(record.get(key), (int, float))
            ]
            aggregate[key] = finite_min(values)
        for key in ("max_rule_measure", "max_volume_fraction", "max_exact_order"):
            values = [
                float(record[key])
                for record in records_for_extrema
                if isinstance(record.get(key), (int, float))
            ]
            aggregate[key] = finite_max(values)
        if "min_exact_order" in aggregate:
            aggregate["min_exact_order"] = int(aggregate["min_exact_order"])
        if "max_exact_order" in aggregate:
            aggregate["max_exact_order"] = int(aggregate["max_exact_order"])
        groups.append(aggregate)
    return groups


def normalized_level_set_side(value: Any) -> str | None:
    text = str(value).strip().lower()
    if not text:
        return None
    if "negative" in text:
        return "negative"
    if "positive" in text:
        return "positive"
    return text


def active_cut_volume_side(diagnostics: dict[str, Any]) -> str | None:
    for record in reversed(diagnostics.get("cut_context_rebuilds", [])):
        side = normalized_level_set_side(record.get("active_side"))
        if side is not None:
            return side
    return None


def active_cut_volume_records(diagnostics: dict[str, Any]) -> list[dict[str, Any]]:
    records = (
        diagnostics.get("cut_volume_assembly_groups")
        or diagnostics.get("cut_volume_assemblies", [])
    )
    active_side = active_cut_volume_side(diagnostics)
    if active_side is None:
        return list(records)
    return [
        record for record in records
        if normalized_level_set_side(record.get("side")) == active_side
    ]


def summarize_time_loop(time_loop: dict[str, Any]) -> dict[str, Any]:
    nonlinear_records = time_loop.get("nonlinear_records", [])
    accepted_steps = time_loop.get("accepted_steps", [])
    rejected_steps = time_loop.get("rejected_steps", [])
    dt_updates = time_loop.get("dt_updates", [])
    summary: dict[str, Any] = {
        "nonlinear_records": len(nonlinear_records),
        "accepted_steps": len(accepted_steps),
        "rejected_steps": len(rejected_steps),
        "dt_updates": len(dt_updates),
        "vtk_outputs": len(time_loop.get("vtk_outputs", [])),
    }
    if accepted_steps:
        final_step = accepted_steps[-1]
        summary["final_accepted_step"] = final_step.get("step")
        summary["final_accepted_time"] = final_step.get("time")
        accepted_dt = [
            float(record["dt"])
            for record in accepted_steps
            if isinstance(record.get("dt"), (int, float))
        ]
        accepted_dt_range = numeric_range(accepted_dt)
        if accepted_dt_range is not None:
            summary["accepted_dt"] = accepted_dt_range
    if rejected_steps:
        rejected_dt = [
            float(record["dt"])
            for record in rejected_steps
            if isinstance(record.get("dt"), (int, float))
        ]
        rejected_dt_range = numeric_range(rejected_dt)
        if rejected_dt_range is not None:
            summary["rejected_dt"] = rejected_dt_range
        reasons: dict[str, int] = {}
        for record in rejected_steps:
            reason = str(record.get("reason", "unknown"))
            reasons[reason] = reasons.get(reason, 0) + 1
        summary["rejection_reasons"] = reasons
    if dt_updates:
        next_dt = [
            float(record["new_dt"])
            for record in dt_updates
            if isinstance(record.get("new_dt"), (int, float))
        ]
        next_dt_range = numeric_range(next_dt)
        if next_dt_range is not None:
            summary["updated_dt"] = next_dt_range
    if nonlinear_records:
        nonlinear_iterations = [
            int(record["nonlinear_iterations"])
            for record in nonlinear_records
            if isinstance(record.get("nonlinear_iterations"), int)
        ]
        linear_iterations = [
            int(record["linear_iterations"])
            for record in nonlinear_records
            if isinstance(record.get("linear_iterations"), int)
        ]
        outer_iterations = [
            int(record["outer_iterations"])
            for record in nonlinear_records
            if isinstance(record.get("outer_iterations"), int)
        ]
        inner_iteration_totals = [
            int(record["inner_iterations_total"])
            for record in nonlinear_records
            if isinstance(record.get("inner_iterations_total"), int)
        ]
        outer_state_changes = [
            float(record["outer_state_change_norm"])
            for record in nonlinear_records
            if isinstance(record.get("outer_state_change_norm"), (int, float))
        ]
        nonlinear_residuals = [
            float(record["residual"])
            for record in nonlinear_records
            if isinstance(record.get("residual"), (int, float))
        ]
        linear_residuals = [
            float(record["linear_relative_residual"])
            for record in nonlinear_records
            if isinstance(record.get("linear_relative_residual"), (int, float))
        ]
        summary["all_nonlinear_converged"] = all(
            bool(record.get("converged")) for record in nonlinear_records
        )
        summary["all_linear_converged"] = all(
            bool(record.get("linear_converged")) for record in nonlinear_records
        )
        if nonlinear_iterations:
            summary["nonlinear_iterations_total"] = int(sum(nonlinear_iterations))
            summary["nonlinear_iterations_max"] = int(max(nonlinear_iterations))
            summary["nonlinear_iteration_distribution"] = distribution(nonlinear_iterations)
        if linear_iterations:
            summary["linear_iterations_total"] = int(sum(linear_iterations))
            summary["linear_iterations_max"] = int(max(linear_iterations))
            summary["linear_iteration_distribution"] = distribution(linear_iterations)
        if outer_iterations:
            summary["external_state_fixed_point_records"] = len(outer_iterations)
            summary["outer_iterations_total"] = int(sum(outer_iterations))
            summary["outer_iterations_max"] = int(max(outer_iterations))
            summary["outer_iteration_distribution"] = distribution(outer_iterations)
        if inner_iteration_totals:
            summary["inner_iterations_total_sum"] = int(sum(inner_iteration_totals))
            summary["inner_iterations_total_max"] = int(max(inner_iteration_totals))
            summary["inner_iterations_total_distribution"] = distribution(
                inner_iteration_totals)
        outer_state_change_range = numeric_range(outer_state_changes)
        if outer_state_change_range is not None:
            summary["outer_state_change_norm"] = outer_state_change_range
        nonlinear_range = numeric_range(nonlinear_residuals)
        if nonlinear_range is not None:
            summary["nonlinear_residual"] = nonlinear_range
        linear_range = numeric_range(linear_residuals)
        if linear_range is not None:
            summary["linear_relative_residual"] = linear_range
    return summary


def parse_component_norms(line: str) -> list[dict[str, Any]]:
    label_match = VECTOR_COMPONENT_LABEL_RE.search(line)
    if label_match is not None:
        line = line[label_match.end():]
    components = []
    for match in COMPONENT_NORM_RE.finditer(line):
        record = {
            "component": match.group(1),
            "norm": float(match.group(2)),
            "mean": float(match.group(3)),
        }
        if match.group(4) is not None and match.group(5) is not None:
            record["min"] = float(match.group(4))
            record["max"] = float(match.group(5))
        components.append(record)
    return components


def parse_jacobian_component_norms(line: str) -> list[dict[str, Any]]:
    if "component norms " in line:
        line = line.split("component norms ", 1)[1]
    components = []
    for match in JACOBIAN_COMPONENT_NORM_RE.finditer(line):
        components.append({
            "component": match.group(1),
            "fd": float(match.group(2)),
            "total_err": float(match.group(3)),
            "matrix_err": float(match.group(4)),
        })
    return components


def parse_jacobian_component_details(line: str) -> list[dict[str, Any]]:
    component_start = line.find(" [", line.find("diagnostic=jacobian_check_component_details"))
    if component_start >= 0:
        line = line[component_start + 1:]
    components = []
    for match in JACOBIAN_COMPONENT_DETAIL_RE.finditer(line):
        components.append({
            "component": match.group(1),
            "base": float(match.group(2)),
            "perturbed": float(match.group(3)),
            "fd": float(match.group(4)),
            "matrix": float(match.group(5)),
            "full": float(match.group(6)),
            "matrix_err": float(match.group(7)),
            "total_err": float(match.group(8)),
            "sign_flip_err": float(match.group(9)),
        })
    return components


def parse_jacobian_top_mismatch(line: str) -> list[dict[str, Any]]:
    entry_start = line.find(" [", line.find("diagnostic=jacobian_check_top_mismatch"))
    if entry_start >= 0:
        line = line[entry_start + 1:]
    entries = []
    for match in JACOBIAN_TOP_MISMATCH_RE.finditer(line):
        entries.append({
            "component": match.group(1),
            "fd": float(match.group(2)),
            "jv": float(match.group(3)),
            "err": float(match.group(4)),
        })
    return entries


def diagnostic_header(line: str, marker: str) -> str:
    marker_index = line.find(marker)
    if marker_index < 0:
        return line
    payload_index = line.find(" [", marker_index)
    if payload_index < 0:
        return line
    return line[:payload_index]


def normalized_component_sweeps(value: str | None) -> list[str]:
    if not value:
        return []
    separator = ";" if ";" in value else ","
    sweeps = []
    for group in value.split(separator):
        tokens = [
            token.strip().lower()
            for token in group.split(",")
            if token.strip()
        ]
        if not tokens or tokens == ["all"]:
            label = "all"
        else:
            label = ",".join(tokens)
        if label not in sweeps:
            sweeps.append(label)
    return sweeps


def jacobian_component_block_metrics(
        records: list[dict[str, Any]]) -> dict[str, Any]:
    relative_errors: dict[str, float] = {}
    matrix_relative_errors: dict[str, float] = {}
    filters: list[str] = []
    skipped = 0
    for record in records:
        raw_filter = record.get("component_filter", record.get("components", "all"))
        column_filter = str(raw_filter or "all")
        if column_filter not in filters:
            filters.append(column_filter)
        components = record.get("components")
        if not isinstance(components, list):
            continue
        for component in components:
            if not isinstance(component, dict):
                continue
            row = str(component.get("component", "unknown"))
            fd_norm = component.get("fd")
            full_norm = component.get("full")
            matrix_norm = component.get("matrix")
            total_err = component.get("total_err")
            matrix_err = component.get("matrix_err")
            if not all(isinstance(value, (int, float)) for value in (
                    fd_norm, full_norm, matrix_norm, total_err, matrix_err)):
                continue
            full_denominator = max(abs(float(fd_norm)), abs(float(full_norm)))
            matrix_denominator = max(abs(float(fd_norm)), abs(float(matrix_norm)))
            if full_denominator < JACOBIAN_COMPONENT_BLOCK_MIN_DENOMINATOR:
                skipped += 1
            else:
                key = f"column={column_filter},row={row}"
                relative_errors[key] = abs(float(total_err)) / full_denominator
            if matrix_denominator >= JACOBIAN_COMPONENT_BLOCK_MIN_DENOMINATOR:
                key = f"column={column_filter},row={row}"
                matrix_relative_errors[key] = abs(float(matrix_err)) / matrix_denominator
    result: dict[str, Any] = {
        "filters": filters,
        "skipped_near_zero_blocks": skipped,
    }
    if relative_errors:
        result["relative_errors"] = dict(sorted(relative_errors.items()))
        result["max_relative_error"] = max(relative_errors.values())
    if matrix_relative_errors:
        result["matrix_relative_errors"] = dict(sorted(matrix_relative_errors.items()))
        result["max_matrix_relative_error"] = max(matrix_relative_errors.values())
    return result


def norm_key(label: str) -> str:
    key = label.strip().lower()
    key = re.sub(r"used_op=([^)]*)", r"used_op_\1", key)
    key = key.replace("*", "_")
    key = key.replace("-", "_minus_")
    key = re.sub(r"[^a-z0-9]+", "_", key).strip("_")
    return f"{key}_norm" if key else "norm"


def timing_key(label: str) -> str:
    key = label.strip().lower()
    key = key.replace("dg+global", "dg_global")
    return re.sub(r"[^a-z0-9]+", "_", key).strip("_")


def parse_norm_key_values(line: str) -> dict[str, Any]:
    values = parse_key_values(DOUBLE_BAR_VALUE_RE.sub("", line))
    for match in DOUBLE_BAR_VALUE_RE.finditer(line):
        values[norm_key(match.group(1))] = parse_scalar(match.group(2))
    return values


def vector_component_header(line: str) -> str:
    label_match = VECTOR_COMPONENT_LABEL_RE.search(line)
    if label_match is None:
        return line
    return line[:label_match.end()]


def parse_eigen_factorization_diagnostic(line: str) -> dict[str, Any]:
    record = parse_key_values(line)
    block_match = re.search(r"block_summaries=(.*)$", line)
    blocks = []
    if block_match is not None:
        for match in BLOCK_SUMMARY_RE.finditer(block_match.group(1)):
            block = parse_key_values(match.group("body").replace(",", " "))
            block["name"] = match.group("name").strip()
            blocks.append(block)
            if block["name"] == "Pressure":
                pressure_zero_rows = block.get("zero_rows")
                pressure_zero_cols = block.get("zero_cols")
                if isinstance(pressure_zero_rows, int):
                    record["pressure_zero_rows"] = pressure_zero_rows
                if isinstance(pressure_zero_cols, int):
                    record["pressure_zero_cols"] = pressure_zero_cols
                for key in ("zero_row_runs_local", "zero_col_runs_local"):
                    value = block.get(key)
                    if isinstance(value, str):
                        record[f"pressure_{key}"] = value
    if blocks:
        record["blocks"] = blocks
    return record


def parse_solver_diagnostics(solver_output: str) -> dict[str, Any]:
    diagnostics: dict[str, Any] = {
        "solver_controls": {},
        "cut_context_rebuilds": [],
        "cut_context_refresh_skips": [],
        "cut_volume_assemblies": [],
        "hydrostatic_initializations": [],
        "pressure_gauge_checks": [],
        "residual_block_norms": [],
        "fsils_true_residuals": [],
        "fsils_prepared_matrices": [],
        "fsils_solve_summaries": [],
        "fsils_blockschur_retries": [],
        "timeloop_initialization_solves": [],
        "vector_component_norms": [],
        "newton_assemblies": [],
        "newton_direction_checks": [],
        "jacobian_checks": [],
        "jacobian_check_component_norms": [],
        "jacobian_check_component_details": [],
        "jacobian_check_component_filters": [],
        "jacobian_check_sweep_plans": [],
        "jacobian_check_top_mismatches": [],
        "form_block_dependencies": [],
        "form_block_installs": [],
        "form_mixed_plans": [],
        "linear_solve_histories": [],
        "jit_specialization_traces": [],
        "jit_cache_diagnostics": [],
        "jit_failure_messages": [],
        "assembly_timings": [],
        "process_memory": [],
        "interior_face_timings": [],
        "cut_volume_timings": [],
        "eigen_factorization_diagnostics": [],
        "active_pressure_support_constraints": [],
        "curvature_projections": [],
        "dynamic_contact_operator_angles": [],
        "free_surface_conservative_balances": [],
        "free_surface_pressure_representability_distance_gates": [],
        "free_surface_static_compatible_pressure_initializers": [],
        "static_capillary_equilibrium_initializations": [],
        "level_set_advection_velocity_updates": [],
        "level_set_volume_corrections": [],
        "level_set_maintenance": [],
        "level_set_nonconservative_warnings": [],
        "wet_volume_diagnostics": [],
        "time_loop": {
            "nonlinear_records": [],
            "accepted_steps": [],
            "rejected_steps": [],
            "dt_updates": [],
            "vtk_outputs": [],
        },
        "true_residual_failure_count": solver_output.count("true residual check failed"),
    }
    active_assembly_timing: dict[str, Any] | None = None
    for line in solver_output.splitlines():
        timing_header = ASSEMBLY_TIMING_HEADER_RE.search(line)
        if timing_header is not None:
            active_assembly_timing = {
                "rank": int(timing_header.group("rank")),
                "op": timing_header.group("op"),
            }
            continue
        if active_assembly_timing is not None:
            timing_value = ASSEMBLY_TIMING_VALUE_RE.search(line)
            if timing_value is not None:
                active_assembly_timing[timing_key(timing_value.group("label"))] = (
                    float(timing_value.group("seconds"))
                )
                continue
            if line.strip().startswith("==="):
                diagnostics["assembly_timings"].append(active_assembly_timing)
                active_assembly_timing = None
                continue

        linear_match = LINEAR_SOLVER_RE.search(line)
        time_stepping_match = TIME_STEPPING_RE.search(line)
        transient_match = TRANSIENT_SOLVE_RE.search(line)
        adaptive_match = TIMELOOP_ADAPTIVE_RE.search(line)
        nonlinear_match = TIMELOOP_NONLINEAR_RE.search(line)
        accepted_match = TIMELOOP_ACCEPTED_RE.search(line)
        rejected_match = TIMELOOP_REJECTED_RE.search(line)
        dt_updated_match = TIMELOOP_DT_UPDATED_RE.search(line)
        vtk_match = VTK_WRITE_RE.search(line)
        if linear_match is not None:
            diagnostics["solver_controls"]["linear_solver"] = convert_match(linear_match)
        elif time_stepping_match is not None:
            diagnostics["solver_controls"]["time_stepping"] = convert_match(time_stepping_match)
        elif transient_match is not None:
            diagnostics["solver_controls"]["transient_solve"] = convert_match(transient_match)
        elif adaptive_match is not None:
            diagnostics["solver_controls"]["adaptive_time_loop"] = convert_match(adaptive_match)
        elif nonlinear_match is not None:
            diagnostics["time_loop"]["nonlinear_records"].append(convert_match(nonlinear_match))
        elif accepted_match is not None:
            diagnostics["time_loop"]["accepted_steps"].append(convert_match(accepted_match))
        elif rejected_match is not None:
            diagnostics["time_loop"]["rejected_steps"].append(convert_match(rejected_match))
        elif dt_updated_match is not None:
            diagnostics["time_loop"]["dt_updates"].append(convert_match(dt_updated_match))
        elif vtk_match is not None:
            diagnostics["time_loop"]["vtk_outputs"].append(vtk_match.group("path").strip())
        elif "diagnostic=cut_context_refresh_skip" in line:
            diagnostics["cut_context_refresh_skips"].append(parse_key_values(line))
        elif "Active-domain cut context" in line:
            record = parse_key_values(line)
            diagnostics["cut_context_rebuilds"].append(record)
            if isinstance(record.get("process_rss_kb"), (int, float)):
                memory_record = dict(record)
                memory_record["phase"] = "cut_context_rebuild"
                diagnostics["process_memory"].append(memory_record)
        elif "diagnostic=process_memory" in line:
            diagnostics["process_memory"].append(parse_key_values(line))
        elif "Updated level-set advection velocity" in line:
            diagnostics["level_set_advection_velocity_updates"].append(
                parse_key_values(line)
            )
        elif "cut-volume active-domain diagnostics" in line:
            diagnostics["cut_volume_assemblies"].append(parse_key_values(line))
        elif "hydrostatic pressure initialization" in line:
            diagnostics["hydrostatic_initializations"].append(parse_key_values(line))
        elif "pressure gauge diagnostic" in line:
            diagnostics["pressure_gauge_checks"].append(parse_key_values(line))
        elif "diagnostic=dynamic_contact_operator_angle" in line:
            diagnostics["dynamic_contact_operator_angles"].append(
                parse_key_values(line)
            )
        elif "diagnostic=free_surface_conservative_balance" in line:
            diagnostics["free_surface_conservative_balances"].append(
                parse_key_values(line)
            )
        elif (
                "diagnostic=free_surface_pressure_representability_distance_gate"
                in line):
            diagnostics[
                "free_surface_pressure_representability_distance_gates"
            ].append(parse_key_values(line))
        elif (
                "diagnostic=free_surface_static_compatible_pressure_initializer"
                in line):
            diagnostics[
                "free_surface_static_compatible_pressure_initializers"
            ].append(parse_key_values(line))
        elif "diagnostic=static_capillary_equilibrium_initialization" in line:
            diagnostics["static_capillary_equilibrium_initializations"].append(
                parse_key_values(line)
            )
        elif "residual block norms" in line:
            diagnostics["residual_block_norms"].append(parse_key_values(line))
        elif "diagnostic=newton_assembly" in line:
            diagnostics["newton_assemblies"].append(parse_key_values(line))
        elif "true residual diagnostics" in line:
            diagnostics["fsils_true_residuals"].append(parse_key_values(line))
        elif "diagnostic=fsils_prepared_matrix" in line:
            diagnostics["fsils_prepared_matrices"].append(parse_key_values(line))
        elif "diagnostic=fsils_solve_summary" in line:
            diagnostics["fsils_solve_summaries"].append(parse_key_values(line))
        elif "diagnostic=fsils_blockschur_true_residual_retry" in line:
            diagnostics["fsils_blockschur_retries"].append(parse_key_values(line))
        elif "diagnostic=timeloop_initialization_linear_solve" in line:
            diagnostics["timeloop_initialization_solves"].append(parse_key_values(line))
        elif "NewtonSolver: direction check" in line:
            diagnostics["newton_direction_checks"].append(parse_norm_key_values(line))
        elif "NewtonSolver: Jacobian check jacobian_op=" in line:
            diagnostics["jacobian_checks"].append(parse_norm_key_values(line))
        elif "NewtonSolver: Jacobian check component norms" in line:
            record = parse_key_values(
                diagnostic_header(line, "diagnostic=jacobian_check_component_norms")
            )
            record["components"] = parse_jacobian_component_norms(line)
            diagnostics["jacobian_check_component_norms"].append(record)
        elif "diagnostic=jacobian_check_component_details" in line:
            record = parse_key_values(
                diagnostic_header(line, "diagnostic=jacobian_check_component_details")
            )
            record["components"] = parse_jacobian_component_details(line)
            diagnostics["jacobian_check_component_details"].append(record)
        elif "diagnostic=jacobian_check_component_filter" in line:
            diagnostics["jacobian_check_component_filters"].append(parse_key_values(line))
        elif "diagnostic=jacobian_check_sweep_plan" in line:
            diagnostics["jacobian_check_sweep_plans"].append(parse_key_values(line))
        elif "diagnostic=jacobian_check_top_mismatch" in line:
            record = parse_key_values(
                diagnostic_header(line, "diagnostic=jacobian_check_top_mismatch")
            )
            record["entries"] = parse_jacobian_top_mismatch(line)
            diagnostics["jacobian_check_top_mismatches"].append(record)
        elif "diagnostic=form_block_dependencies" in line:
            diagnostics["form_block_dependencies"].append(parse_key_values(line))
        elif "diagnostic=form_block_install" in line:
            diagnostics["form_block_installs"].append(parse_key_values(line))
        elif "diagnostic=form_mixed_plan" in line:
            diagnostics["form_mixed_plans"].append(parse_key_values(line))
        elif "NewtonSolver: linear solve history" in line:
            diagnostics["linear_solve_histories"].append(parse_key_values(line))
        elif "Eigen direct factorization diagnostic" in line:
            diagnostics["eigen_factorization_diagnostics"].append(
                parse_eigen_factorization_diagnostic(line)
            )
        elif "diagnostic=level_set_active_side_vertex_constraint" in line:
            diagnostics["active_pressure_support_constraints"].append(
                parse_key_values(line)
            )
        elif "Level-set curvature projected" in line:
            diagnostics["curvature_projections"].append(parse_key_values(line))
        elif "Level-set volume corrected" in line:
            diagnostics["level_set_volume_corrections"].append(parse_key_values(line))
        elif "Level-set maintenance diagnostic" in line:
            diagnostics["level_set_maintenance"].append(parse_key_values(line))
        elif "Wet volume diagnostic" in line:
            diagnostics["wet_volume_diagnostics"].append(parse_key_values(line))
        elif ("WARNING unfitted free-surface level-set has no enabled "
              "reinitialization or volume-correction request") in line:
            diagnostics["level_set_nonconservative_warnings"].append(
                parse_key_values(line)
            )
        elif ("JIT: failed to compile" in line or
              "JIT: runtime failure" in line or
              ("JIT requested for kernel" in line and
               "using interpreter" in line)):
            diagnostics["jit_failure_messages"].append(line.strip())
        elif "JIT specialization trace:" in line:
            diagnostics["jit_specialization_traces"].append(parse_jit_specialization_trace(line))
        elif "diagnostic=jit_cache" in line:
            diagnostics["jit_cache_diagnostics"].append(parse_key_values(line))
        elif "[INTERIOR_FACE_TIMING]" in line:
            diagnostics["interior_face_timings"].append(parse_interior_face_timing(line))
        elif "[CUT_VOLUME_TIMING]" in line:
            diagnostics["cut_volume_timings"].append(parse_cut_volume_timing(line))
        elif "vector component norms" in line:
            record = parse_key_values(vector_component_header(line))
            record["components"] = parse_component_norms(line)
            diagnostics["vector_component_norms"].append(record)

    if active_assembly_timing is not None:
        diagnostics["assembly_timings"].append(active_assembly_timing)
    diagnostics["counts"] = {
        name: len(records)
        for name, records in diagnostics.items()
        if isinstance(records, list)
    }
    diagnostics["time_loop"]["summary"] = summarize_time_loop(diagnostics["time_loop"])
    diagnostics.update(parse_active_volume_history(solver_output))
    diagnostics["cut_volume_assembly_groups"] = aggregate_cut_volume_assemblies(
        diagnostics["cut_volume_assemblies"]
    )
    diagnostics["counts"]["cut_volume_assembly_groups"] = len(
        diagnostics["cut_volume_assembly_groups"]
    )
    if diagnostics["cut_volume_assembly_groups"]:
        assembly_volumes = [
            float(record["active_wet_volume"])
            for record in active_cut_volume_records(diagnostics)
            if isinstance(record.get("active_wet_volume"), (int, float))
        ]
        diagnostics["assembly_active_wet_volumes"] = assembly_volumes
        diagnostics["assembly_active_wet_volume_change"] = value_span(assembly_volumes)
    return diagnostics


def load_benchmark(case_dir: Path) -> dict[str, Any]:
    benchmark_path = case_dir / "benchmark.json"
    if not benchmark_path.exists():
        return {}
    return json.loads(benchmark_path.read_text(encoding="utf-8"))


def benchmark_active_domain(benchmark: dict[str, Any]) -> str:
    """Return the declared liquid side, retaining the historical default."""
    active_domain = benchmark.get("active_domain")
    contact = benchmark.get("sessile_contact")
    if active_domain is None and isinstance(contact, dict):
        active_domain = contact.get("active_domain")
    if active_domain is None:
        active_domain = "LevelSetNegative"
    return normalized_active_domain(str(active_domain))


def latest_component_record(diagnostics: dict[str, Any],
                            label: str) -> list[dict[str, Any]]:
    for record in reversed(diagnostics.get("vector_component_norms", [])):
        if record.get("label") == label:
            components = record.get("components")
            if isinstance(components, list):
                return components
    return []


def component_by_name(components: list[dict[str, Any]],
                      name: str) -> dict[str, Any] | None:
    for component in components:
        if component.get("component") == name:
            return component
    return None


def component_range(component: dict[str, Any] | None) -> float | None:
    if component is None:
        return None
    min_value = component.get("min")
    max_value = component.get("max")
    if isinstance(min_value, (int, float)) and isinstance(max_value, (int, float)):
        return float(max_value) - float(min_value)
    norm = component.get("norm")
    if isinstance(norm, (int, float)):
        return float(norm)
    return None


def diagnostic_solution_velocity_range(diagnostics: dict[str, Any]) -> float | None:
    components = latest_component_record(diagnostics, "solution_state")
    ranges = []
    for component in components:
        if str(component.get("component", "")).startswith("Velocity"):
            value = component_range(component)
            if value is not None:
                ranges.append(abs(value))
    if not ranges:
        return None
    return max(ranges)


def diagnostic_solution_pressure_range(diagnostics: dict[str, Any]) -> float | None:
    components = latest_component_record(diagnostics, "solution_state")
    return component_range(component_by_name(components, "Pressure"))


def diagnostic_active_volume_error(diagnostics: dict[str, Any]) -> float | None:
    context_volumes = diagnostic_context_active_side_volumes(
        diagnostics,
        prefer_physical=False,
    )
    assembly_volumes = [
        float(record["active_wet_volume"])
        for record in active_cut_volume_records(diagnostics)
        if isinstance(record.get("active_wet_volume"), (int, float))
    ]
    if not context_volumes or not assembly_volumes:
        return None
    return max(
        min(abs(assembly_volume - context_volume) for context_volume in context_volumes)
        for assembly_volume in assembly_volumes
    )


def diagnostic_context_active_side_physical_volumes(
        diagnostics: dict[str, Any]) -> list[float]:
    return [
        float(record["active_side_physical_volume"])
        for record in diagnostics.get("cut_context_rebuilds", [])
        if isinstance(record.get("active_side_physical_volume"), (int, float))
    ]


def diagnostic_context_active_side_reference_volumes(
        diagnostics: dict[str, Any]) -> list[float]:
    return [
        float(record["active_side_volume"])
        for record in diagnostics.get("cut_context_rebuilds", [])
        if isinstance(record.get("active_side_volume"), (int, float))
    ]


def diagnostic_context_active_side_volumes(
        diagnostics: dict[str, Any],
        *,
        prefer_physical: bool = True) -> list[float]:
    physical_volumes = diagnostic_context_active_side_physical_volumes(diagnostics)
    reference_volumes = diagnostic_context_active_side_reference_volumes(diagnostics)
    if prefer_physical and physical_volumes:
        return physical_volumes
    return reference_volumes


def diagnostic_cut_volume_min_exact_order(diagnostics: dict[str, Any]) -> int | None:
    orders = [
        int(record["min_exact_order"])
        for record in active_cut_volume_records(diagnostics)
        if isinstance(record.get("min_exact_order"), int)
    ]
    if not orders:
        return None
    return min(orders)


def diagnostic_cut_volume_max_exact_order(diagnostics: dict[str, Any]) -> int | None:
    orders = [
        int(record["max_exact_order"])
        for record in active_cut_volume_records(diagnostics)
        if isinstance(record.get("max_exact_order"), int)
    ]
    if not orders:
        return None
    return max(orders)


def diagnostic_cut_adjacent_max_scale(diagnostics: dict[str, Any]) -> float | None:
    scales = [
        float(record["cut_adjacent_max_scale"])
        for record in diagnostics.get("cut_context_rebuilds", [])
        if isinstance(record.get("cut_adjacent_max_scale"), (int, float))
    ]
    if not scales:
        return None
    return max(scales)


def diagnostic_cut_adjacent_capped_scale_count(diagnostics: dict[str, Any]) -> int | None:
    counts = [
        int(record["cut_adjacent_capped_scale"])
        for record in diagnostics.get("cut_context_rebuilds", [])
        if isinstance(record.get("cut_adjacent_capped_scale"), int)
    ]
    if not counts:
        return None
    return max(counts)


def diagnostic_active_pruned_volume_regions(diagnostics: dict[str, Any]) -> int | None:
    counts = [
        int(record["active_pruned_volume_regions"])
        for record in diagnostics.get("cut_context_rebuilds", [])
        if isinstance(record.get("active_pruned_volume_regions"), int)
    ]
    if not counts:
        return None
    return max(counts)


def diagnostic_active_pruned_volume(diagnostics: dict[str, Any]) -> float | None:
    volumes = [
        float(record["active_pruned_volume"])
        for record in diagnostics.get("cut_context_rebuilds", [])
        if isinstance(record.get("active_pruned_volume"), (int, float))
    ]
    if not volumes:
        return None
    return max(volumes)


def diagnostic_active_min_volume_fraction(diagnostics: dict[str, Any]) -> float | None:
    fractions = [
        float(record["active_min_volume_fraction"])
        for record in diagnostics.get("cut_context_rebuilds", [])
        if isinstance(record.get("active_min_volume_fraction"), (int, float))
    ]
    if not fractions:
        return None
    return min(fractions)


def diagnostic_generated_pruned_volume_rules(diagnostics: dict[str, Any]) -> int | None:
    counts = [
        int(record["generated_pruned_volume_rules"])
        for record in diagnostics.get("cut_context_rebuilds", [])
        if isinstance(record.get("generated_pruned_volume_rules"), int)
    ]
    if not counts:
        return None
    return max(counts)


def diagnostic_generated_pruned_volume(diagnostics: dict[str, Any]) -> float | None:
    volumes = [
        float(record["generated_pruned_volume"])
        for record in diagnostics.get("cut_context_rebuilds", [])
        if isinstance(record.get("generated_pruned_volume"), (int, float))
    ]
    if not volumes:
        return None
    return max(volumes)


def diagnostic_implicit_cut_fallback_cells(diagnostics: dict[str, Any]) -> int | None:
    counts = [
        int(record["implicit_cut_fallback_cells"])
        for record in diagnostics.get("cut_context_rebuilds", [])
        if isinstance(record.get("implicit_cut_fallback_cells"), int)
    ]
    if not counts:
        return None
    return max(counts)


def diagnostic_cut_context_min_int(diagnostics: dict[str, Any], key: str) -> int | None:
    values = [
        int(record[key])
        for record in diagnostics.get("cut_context_rebuilds", [])
        if isinstance(record.get(key), int)
    ]
    if not values:
        return None
    return min(values)


def diagnostic_cut_context_value_counts(diagnostics: dict[str, Any], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in diagnostics.get("cut_context_rebuilds", []):
        value = record.get(key)
        if value is not None:
            increment_count(counts, str(value))
    return counts


def diagnostic_cut_context_summary_counts(diagnostics: dict[str, Any], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in diagnostics.get("cut_context_rebuilds", []):
        for name, count in parse_count_summary(record.get(key)).items():
            counts[name] = counts.get(name, 0) + count
    return counts


def diagnostic_pressure_gauge_value(diagnostics: dict[str, Any]) -> float | None:
    for record in reversed(diagnostics.get("hydrostatic_initializations", [])):
        checked = record.get("checked_gauge_constraints")
        pressure_min = record.get("gauge_pressure_min")
        pressure_max = record.get("gauge_pressure_max")
        if checked and isinstance(pressure_min, (int, float)) and isinstance(pressure_max, (int, float)):
            return 0.5 * (float(pressure_min) + float(pressure_max))
    for record in reversed(diagnostics.get("pressure_gauge_checks", [])):
        pressure_min = record.get("constraint_pressure_min")
        pressure_max = record.get("constraint_pressure_max")
        if isinstance(pressure_min, (int, float)) and isinstance(pressure_max, (int, float)):
            return 0.5 * (float(pressure_min) + float(pressure_max))
    return None


def cut_context_solution_source_summary(diagnostics: dict[str, Any]) -> dict[str, Any]:
    records = diagnostics.get("cut_context_rebuilds", [])
    if not isinstance(records, list):
        return {}
    source_counts: dict[str, int] = {}
    state_refresh_count = 0
    vector_refresh_count = 0
    missing_source_count = 0
    for record in records:
        if not isinstance(record, dict):
            continue
        source = record.get("solution_source")
        if isinstance(source, str) and source:
            source_counts[source] = source_counts.get(source, 0) + 1
        else:
            missing_source_count += 1
        provenance = record.get("provenance")
        if provenance in STATE_SYNC_CUT_CONTEXT_PROVENANCES:
            state_refresh_count += 1
        elif provenance in VECTOR_CUT_CONTEXT_PROVENANCES:
            vector_refresh_count += 1
    return {
        "source_counts": source_counts,
        "state_refresh_count": state_refresh_count,
        "vector_refresh_count": vector_refresh_count,
        "missing_source_count": missing_source_count,
    }


def cut_context_rebuild_provenance_counts(diagnostics: dict[str, Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in diagnostics.get("cut_context_rebuilds", []):
        if not isinstance(record, dict):
            continue
        provenance = record.get("provenance")
        key = str(provenance) if provenance else "missing"
        increment_count(counts, key)
    return counts


def cut_context_refresh_skip_provenance_counts(diagnostics: dict[str, Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in diagnostics.get("cut_context_refresh_skips", []):
        if not isinstance(record, dict):
            continue
        provenance = record.get("provenance")
        key = str(provenance) if provenance else "missing"
        increment_count(counts, key)
    return counts


def generated_cell_cache_summary(diagnostics: dict[str, Any]) -> dict[str, int]:
    records = diagnostics.get("cut_context_rebuilds", [])
    if not isinstance(records, list):
        return {}
    summary = {
        "rebuilds_with_cell_cache": 0,
        "total_hits": 0,
        "total_misses": 0,
        "unchanged_dof_hits": 0,
        "refresh_candidates": 0,
        "domain_hits": 0,
        "full_miss_rebuilds": 0,
    }
    for record in records:
        if not isinstance(record, dict):
            continue
        hits = record.get("generated_cell_cache_hits")
        misses = record.get("generated_cell_cache_misses")
        cell_count = record.get("cell_count")
        if not isinstance(hits, int) or not isinstance(misses, int):
            continue
        summary["rebuilds_with_cell_cache"] += 1
        summary["total_hits"] += hits
        summary["total_misses"] += misses
        unchanged_dof_hits = record.get("generated_cell_cache_unchanged_dof_hits")
        if isinstance(unchanged_dof_hits, int):
            summary["unchanged_dof_hits"] += unchanged_dof_hits
        refresh_candidates = record.get("generated_cell_refresh_candidates")
        if isinstance(refresh_candidates, int):
            summary["refresh_candidates"] += refresh_candidates
        domain_hits = record.get("generated_domain_cache_hits")
        if isinstance(domain_hits, int):
            summary["domain_hits"] += domain_hits
        if (isinstance(cell_count, int) and cell_count > 0 and
                hits == 0 and misses == cell_count):
            summary["full_miss_rebuilds"] += 1
    return summary


def cut_context_solution_source_errors(diagnostics: dict[str, Any]) -> list[str]:
    records = diagnostics.get("cut_context_rebuilds", [])
    if not records:
        return ["cut-context rebuild diagnostics were not reported"]

    errors = []
    missing = [
        record for record in records
        if isinstance(record, dict) and "solution_source" not in record
    ]
    if missing:
        errors.append(
            f"{len(missing)} cut-context rebuild diagnostic(s) do not report solution_source"
        )

    state_records = [
        record for record in records
        if isinstance(record, dict) and
        record.get("provenance") in STATE_SYNC_CUT_CONTEXT_PROVENANCES
    ]
    if state_records:
        bad_state = [
            record for record in state_records
            if record.get("solution_source") != "state_vector_fe_ordered"
        ]
        if bad_state:
            examples = ", ".join(
                f"{record.get('provenance', 'unknown')}:{record.get('solution_source', 'missing')}"
                for record in bad_state[:3]
            )
            errors.append(
                "Newton cut-context refreshes did not all use state_vector_fe_ordered "
                f"({examples})"
            )

    vector_records = [
        record for record in records
        if isinstance(record, dict) and
        record.get("provenance") in VECTOR_CUT_CONTEXT_PROVENANCES
    ]
    bad_vector = [
        record for record in vector_records
        if record.get("solution_source") != "fe_vector"
    ]
    if bad_vector:
        examples = ", ".join(
            f"{record.get('provenance', 'unknown')}:{record.get('solution_source', 'missing')}"
            for record in bad_vector[:3]
        )
        errors.append(
            "vector cut-context refreshes did not all use fe_vector "
            f"({examples})"
        )
    return errors


def assembly_efficiency_errors(metrics: dict[str, Any],
                               args: argparse.Namespace) -> list[str]:
    checks = (
        ("max_diagnostic_assembly_timings_per_step",
         "diagnostic_assembly_timings_per_accepted_step",
         "assembly timing records per accepted step"),
        ("max_diagnostic_extra_assembly_timings_per_step",
         "diagnostic_extra_assembly_timings_per_accepted_step",
         "extra assembly timing records per accepted step"),
        ("max_diagnostic_cut_context_rebuilds_per_step",
         "diagnostic_cut_context_rebuilds_per_accepted_step",
         "cut-context rebuilds per accepted step"),
        ("max_diagnostic_newton_matrix_assemblies_per_step",
         "diagnostic_newton_matrix_assemblies_per_accepted_step",
         "Newton matrix assemblies per accepted step"),
        ("max_diagnostic_generated_cell_cache_full_miss_rebuilds",
         "diagnostic_generated_cell_cache_full_miss_rebuilds",
         "generated-interface full cell-cache miss rebuilds"),
    )
    errors = []
    for arg_name, metric_name, label in checks:
        threshold = getattr(args, arg_name)
        if threshold is None:
            continue
        value = metrics.get(metric_name)
        if not isinstance(value, (int, float)):
            errors.append(f"{label} diagnostic is unavailable")
        elif float(value) > float(threshold):
            errors.append(
                f"{label} {float(value):.6g} exceeds {float(threshold):.6g}"
            )
    if args.min_diagnostic_cut_context_refresh_skips is not None:
        value = metrics.get("diagnostic_cut_context_refresh_skip_count")
        if not isinstance(value, int):
            errors.append("cut-context refresh skip diagnostic is unavailable")
        elif value < args.min_diagnostic_cut_context_refresh_skips:
            errors.append(
                f"cut-context refresh skips {value} are below "
                f"{args.min_diagnostic_cut_context_refresh_skips}"
            )
    return errors


def level_set_advection_velocity_errors(metrics: dict[str, Any],
                                        args: argparse.Namespace) -> list[str]:
    errors = []
    diagnostics = metrics.get("diagnostics", {})
    records = diagnostics.get("level_set_advection_velocity_updates", [])
    if args.require_level_set_advection_velocity_diagnostics and not records:
        errors.append("level-set advection velocity diagnostics were not reported")
    for arg_name, metric_name, label in (
            ("expect_level_set_advection_velocity_extension_method",
             "diagnostic_level_set_advection_velocity_extension_method_counts",
             "level-set advection velocity extension method"),
            ("expect_level_set_advection_velocity_interface_sample_source",
             "diagnostic_level_set_advection_velocity_interface_sample_source_counts",
             "level-set advection velocity interface sample source")):
        expected = getattr(args, arg_name)
        if expected is None:
            continue
        counts = metrics.get(metric_name)
        if not isinstance(counts, dict) or not counts:
            errors.append(f"diagnostic {label} counts are unavailable")
        elif expected not in counts:
            observed = ", ".join(str(key) for key in sorted(counts))
            errors.append(
                f"diagnostic {label} {observed or 'unavailable'} does not include {expected}"
            )
    if args.min_diagnostic_level_set_advection_interface_samples is not None:
        value = metrics.get(
            "diagnostic_level_set_advection_velocity_max_interface_samples")
        if not isinstance(value, int):
            errors.append("level-set advection interface-sample diagnostic is unavailable")
        elif value < args.min_diagnostic_level_set_advection_interface_samples:
            errors.append(
                f"level-set advection interface samples {value} are below "
                f"{args.min_diagnostic_level_set_advection_interface_samples}"
            )
    return errors


def resource_ceiling_errors(metrics: dict[str, Any],
                            args: argparse.Namespace) -> list[str]:
    checks = (
        ("max_diagnostic_process_rss_kb",
         "diagnostic_process_max_rss_kb",
         "process RSS"),
        ("max_diagnostic_process_rss_growth_kb",
         "diagnostic_process_rss_growth_kb",
         "process RSS growth"),
        ("max_diagnostic_process_basis_cache_entry_growth",
         "diagnostic_process_basis_cache_entry_growth",
         "basis-cache entry growth"),
    )
    errors = []
    for arg_name, metric_name, label in checks:
        threshold = getattr(args, arg_name)
        if threshold is None:
            continue
        value = metrics.get(metric_name)
        if not isinstance(value, (int, float)):
            errors.append(f"{label} diagnostic is unavailable")
        elif float(value) > float(threshold):
            errors.append(
                f"{label} {float(value):.6g} exceeds {float(threshold):.6g}"
            )
    return errors


def cut_context_policy_errors(metrics: dict[str, Any],
                              args: argparse.Namespace) -> list[str]:
    errors = []
    diagnostics = metrics["diagnostics"]
    records = diagnostics.get("cut_context_rebuilds", [])
    if args.require_high_order_cut_context_diagnostics:
        if not records:
            errors.append("cut-context rebuild diagnostics were not reported")
        else:
            required = (
                "generated_interface_geometry",
                "implicit_cut_quadrature_backend",
                "selected_implicit_cut_quadrature_backend_counts",
                "implicit_cut_backend_seconds",
                "implicit_cut_backend_seconds_max",
                "implicit_cut_fallback_policy",
                "implicit_cut_fallback_cells",
                "implicit_cut_backend_qualification_counts",
                "required_implicit_cut_backend_qualification",
                "achieved_interface_quadrature_order",
                "achieved_volume_quadrature_order",
                "interface_rule_count",
                "interface_quadrature_point_count",
                "active_volume_rule_count",
                "active_volume_quadrature_point_count",
            )
            missing = [
                key for key in required
                if not any(key in record for record in records)
            ]
            if missing:
                errors.append(
                    "cut-context diagnostics are missing high-order policy field(s): "
                    + ", ".join(missing)
                )
    for arg_name, metric_name, label in (
            ("expect_generated_interface_geometry",
             "diagnostic_generated_interface_geometry_counts",
             "generated interface geometry"),
            ("expect_implicit_cut_quadrature_backend",
             "diagnostic_implicit_cut_quadrature_backend_counts",
             "implicit cut quadrature backend"),
            ("expect_selected_implicit_cut_quadrature_backend",
             "diagnostic_selected_implicit_cut_quadrature_backend_counts",
             "selected implicit cut quadrature backend"),
            ("expect_implicit_cut_backend_qualification",
             "diagnostic_implicit_cut_backend_qualification_counts",
             "implicit cut backend qualification"),
            ("expect_implicit_cut_fallback_policy",
             "diagnostic_implicit_cut_fallback_policy_counts",
             "implicit cut fallback policy")):
        expected = getattr(args, arg_name)
        if expected is None:
            continue
        counts = metrics.get(metric_name)
        if not isinstance(counts, dict) or not counts:
            errors.append(f"diagnostic {label} counts are unavailable")
        elif expected not in counts:
            observed = ", ".join(str(key) for key in sorted(counts))
            errors.append(
                f"diagnostic {label} {observed or 'unavailable'} does not include {expected}"
            )
    if args.max_diagnostic_implicit_cut_fallback_cells is not None:
        fallback_cells = metrics.get("diagnostic_implicit_cut_fallback_cells")
        if not isinstance(fallback_cells, int):
            errors.append("diagnostic implicit-cut fallback cell count is unavailable")
        elif fallback_cells > args.max_diagnostic_implicit_cut_fallback_cells:
            errors.append(
                f"diagnostic implicit-cut fallback cells {fallback_cells} exceed "
                f"{args.max_diagnostic_implicit_cut_fallback_cells}"
            )
    for arg_name, metric_name, label in (
            ("min_diagnostic_achieved_interface_quadrature_order",
             "diagnostic_achieved_interface_quadrature_order_min",
             "achieved interface quadrature order"),
            ("min_diagnostic_achieved_volume_quadrature_order",
             "diagnostic_achieved_volume_quadrature_order_min",
             "achieved volume quadrature order")):
        minimum = getattr(args, arg_name)
        if minimum is None:
            continue
        value = metrics.get(metric_name)
        if not isinstance(value, int):
            errors.append(f"diagnostic {label} is unavailable")
        elif value < minimum:
            errors.append(
                f"diagnostic {label} {value} is below {minimum}"
            )
    return errors


def curvature_projection_errors(metrics: dict[str, Any],
                                args: argparse.Namespace) -> list[str]:
    errors = []
    if args.require_curvature_projection_diagnostics and not metrics["diagnostics"].get(
            "curvature_projections"):
        errors.append("curvature projection diagnostics were not reported")
    if args.min_diagnostic_curvature_projection_count is not None:
        count = metrics.get("diagnostic_curvature_projection_count")
        if not isinstance(count, int):
            errors.append("curvature projection diagnostic count is unavailable")
        elif count < args.min_diagnostic_curvature_projection_count:
            errors.append(
                f"curvature projection diagnostic count {count} is below "
                f"{args.min_diagnostic_curvature_projection_count}"
            )
    if args.min_diagnostic_curvature_projection_max_abs_curvature is not None:
        value = metrics.get("diagnostic_curvature_projection_max_abs_curvature")
        if not isinstance(value, (int, float)):
            errors.append("curvature projection max-abs-curvature diagnostic is unavailable")
        elif value < args.min_diagnostic_curvature_projection_max_abs_curvature:
            errors.append(
                f"curvature projection max abs curvature {value:.6g} is below "
                f"{args.min_diagnostic_curvature_projection_max_abs_curvature:.6g}"
            )
    if args.max_diagnostic_curvature_projection_fallback_vertices is not None:
        value = metrics.get("diagnostic_curvature_projection_max_fallback_vertices")
        if not isinstance(value, (int, float)):
            errors.append("curvature projection fallback diagnostic is unavailable")
        elif value > args.max_diagnostic_curvature_projection_fallback_vertices:
            errors.append(
                f"curvature projection fallback vertices {value} exceed "
                f"{args.max_diagnostic_curvature_projection_fallback_vertices}"
            )
    if args.max_diagnostic_curvature_projection_zero_fallback_vertices is not None:
        value = metrics.get(
            "diagnostic_curvature_projection_max_zero_fallback_vertices")
        if not isinstance(value, (int, float)):
            errors.append("curvature projection zero-fallback diagnostic is unavailable")
        elif value > args.max_diagnostic_curvature_projection_zero_fallback_vertices:
            errors.append(
                f"curvature projection zero fallback vertices {value} exceed "
                f"{args.max_diagnostic_curvature_projection_zero_fallback_vertices}"
            )
    if args.max_diagnostic_curvature_projection_normalized_fit_residual is not None:
        value = metrics.get(
            "diagnostic_curvature_projection_max_normalized_fit_residual")
        if not isinstance(value, (int, float)):
            errors.append("curvature projection normalized fit residual diagnostic is unavailable")
        elif value > args.max_diagnostic_curvature_projection_normalized_fit_residual:
            errors.append(
                f"curvature projection normalized fit residual {value:.6g} exceeds "
                f"{args.max_diagnostic_curvature_projection_normalized_fit_residual:.6g}"
            )
    if args.min_diagnostic_curvature_projection_smoothing_iterations is not None:
        value = metrics.get(
            "diagnostic_curvature_projection_max_smoothing_iterations")
        if not isinstance(value, (int, float)):
            errors.append("curvature projection smoothing diagnostic is unavailable")
        elif value < args.min_diagnostic_curvature_projection_smoothing_iterations:
            errors.append(
                f"curvature projection smoothing iterations {value} are below "
                f"{args.min_diagnostic_curvature_projection_smoothing_iterations}"
            )
    if args.expect_curvature_projection_smoothing_mode is not None:
        counts = metrics.get(
            "diagnostic_curvature_projection_smoothing_mode_counts")
        expected = args.expect_curvature_projection_smoothing_mode
        if not isinstance(counts, dict) or not counts:
            errors.append("curvature projection smoothing-mode diagnostics are unavailable")
        elif expected not in counts:
            observed = ", ".join(str(key) for key in sorted(counts))
            errors.append(
                "curvature projection smoothing mode "
                f"{observed or 'unavailable'} does not include {expected}"
            )
    expected_recovery_mode = getattr(
        args, "expect_curvature_projection_recovery_mode", None)
    if expected_recovery_mode is not None:
        counts = metrics.get(
            "diagnostic_curvature_projection_recovery_mode_counts")
        expected = expected_recovery_mode
        if not isinstance(counts, dict) or not counts:
            errors.append("curvature projection recovery-mode diagnostics are unavailable")
        elif expected not in counts:
            observed = ", ".join(str(key) for key in sorted(counts))
            errors.append(
                "curvature projection recovery mode "
                f"{observed or 'unavailable'} does not include {expected}"
            )
    for argument, metric, label in (
        (
            "min_diagnostic_curvature_projection_interface_geometry_samples",
            "diagnostic_curvature_projection_max_interface_geometry_samples",
            "generated interface geometry samples",
        ),
        (
            "min_diagnostic_curvature_projection_interface_patch_fitted_vertices",
            "diagnostic_curvature_projection_max_interface_patch_fitted_vertices",
            "generated interface patch fitted vertices",
        ),
    ):
        minimum = getattr(args, argument, None)
        if minimum is None:
            continue
        value = metrics.get(metric)
        if not isinstance(value, (int, float)):
            errors.append(f"curvature projection {label} diagnostic is unavailable")
        elif value < minimum:
            errors.append(
                f"curvature projection {label} {value} is below {minimum}"
            )
    if args.min_diagnostic_curvature_projection_operator_edges is not None:
        value = metrics.get(
            "diagnostic_curvature_projection_max_smoothing_operator_edges")
        if not isinstance(value, (int, float)):
            errors.append("curvature projection smoothing-operator edge diagnostic is unavailable")
        elif value < args.min_diagnostic_curvature_projection_operator_edges:
            errors.append(
                f"curvature projection smoothing operator edges {value} are below "
                f"{args.min_diagnostic_curvature_projection_operator_edges}"
            )
    if args.min_diagnostic_curvature_projection_skipped_count is not None:
        value = metrics.get("diagnostic_curvature_projection_skipped_count")
        if not isinstance(value, int):
            errors.append("curvature projection skipped-count diagnostic is unavailable")
        elif value < args.min_diagnostic_curvature_projection_skipped_count:
            errors.append(
                f"curvature projection skipped count {value} is below "
                f"{args.min_diagnostic_curvature_projection_skipped_count}"
            )
    if args.min_diagnostic_curvature_projection_cache_hit_count is not None:
        value = metrics.get("diagnostic_curvature_projection_cache_hit_count")
        if not isinstance(value, int):
            errors.append("curvature projection cache-hit diagnostic is unavailable")
        elif value < args.min_diagnostic_curvature_projection_cache_hit_count:
            errors.append(
                f"curvature projection cache-hit count {value} is below "
                f"{args.min_diagnostic_curvature_projection_cache_hit_count}"
            )
    if args.max_diagnostic_curvature_projection_cache_miss_count is not None:
        value = metrics.get("diagnostic_curvature_projection_cache_miss_count")
        if not isinstance(value, int):
            errors.append("curvature projection cache-miss diagnostic is unavailable")
        elif value > args.max_diagnostic_curvature_projection_cache_miss_count:
            errors.append(
                f"curvature projection cache-miss count {value} exceeds "
                f"{args.max_diagnostic_curvature_projection_cache_miss_count}"
            )
    if args.min_diagnostic_curvature_projection_cut_signature_cache_hit_count is not None:
        value = metrics.get(
            "diagnostic_curvature_projection_cut_signature_cache_hit_count")
        if not isinstance(value, int):
            errors.append(
                "curvature projection cut-signature cache-hit diagnostic is unavailable")
        elif value < args.min_diagnostic_curvature_projection_cut_signature_cache_hit_count:
            errors.append(
                f"curvature projection cut-signature cache-hit count {value} is below "
                f"{args.min_diagnostic_curvature_projection_cut_signature_cache_hit_count}"
            )
    if args.min_diagnostic_curvature_projection_reused_vertex_adjacency_count is not None:
        value = metrics.get(
            "diagnostic_curvature_projection_reused_vertex_adjacency_count")
        if not isinstance(value, int):
            errors.append("curvature projection vertex-adjacency reuse diagnostic is unavailable")
        elif value < args.min_diagnostic_curvature_projection_reused_vertex_adjacency_count:
            errors.append(
                f"curvature projection vertex-adjacency reuse count {value} is below "
                f"{args.min_diagnostic_curvature_projection_reused_vertex_adjacency_count}"
            )
    if args.min_diagnostic_curvature_projection_reused_sample_adjacency_count is not None:
        value = metrics.get(
            "diagnostic_curvature_projection_reused_sample_adjacency_count")
        if not isinstance(value, int):
            errors.append("curvature projection sample-adjacency reuse diagnostic is unavailable")
        elif value < args.min_diagnostic_curvature_projection_reused_sample_adjacency_count:
            errors.append(
                f"curvature projection sample-adjacency reuse count {value} is below "
                f"{args.min_diagnostic_curvature_projection_reused_sample_adjacency_count}"
            )
    if args.max_diagnostic_curvature_projection_vertex_adjacency_builds is not None:
        value = metrics.get(
            "diagnostic_curvature_projection_max_vertex_adjacency_builds")
        if not isinstance(value, (int, float)):
            errors.append("curvature projection vertex-adjacency build diagnostic is unavailable")
        elif value > args.max_diagnostic_curvature_projection_vertex_adjacency_builds:
            errors.append(
                f"curvature projection vertex-adjacency builds {value} exceed "
                f"{args.max_diagnostic_curvature_projection_vertex_adjacency_builds}"
            )
    if args.max_diagnostic_curvature_projection_sample_adjacency_builds is not None:
        value = metrics.get(
            "diagnostic_curvature_projection_max_sample_adjacency_builds")
        if not isinstance(value, (int, float)):
            errors.append("curvature projection sample-adjacency build diagnostic is unavailable")
        elif value > args.max_diagnostic_curvature_projection_sample_adjacency_builds:
            errors.append(
                f"curvature projection sample-adjacency builds {value} exceed "
                f"{args.max_diagnostic_curvature_projection_sample_adjacency_builds}"
            )
    if args.require_curvature_projection_newton_freshness:
        reason_counts = metrics.get("diagnostic_curvature_projection_reason_counts")
        if not isinstance(reason_counts, dict):
            errors.append("curvature projection reason-count diagnostics are unavailable")
            return errors
        summary = metrics.get("time_loop", {}).get("summary", {})
        accepted_steps = summary.get("accepted_steps") if isinstance(summary, dict) else None
        if not isinstance(accepted_steps, int) or accepted_steps <= 0:
            errors.append("curvature projection freshness requires accepted-step count")
            return errors
        required_reasons = {
            "initial": 1,
            "before_physics_solve": accepted_steps,
            "accepted_step": accepted_steps,
        }
        # ApplicationDriver refreshes cut geometry and phi-derived state at
        # line-search trials whenever that state defines the nonlinear
        # residual (the default for coupled free-surface workflows).  The
        # callback is still conditional on both the residual contract and
        # whether backtracking actually evaluates a trial, and it retains an
        # explicit SVMP_SYNC_LINE_SEARCH_TRIALS override.  Requiring one trial
        # refresh per accepted step would therefore be semantically wrong:
        # some accepted steps do not backtrack or do not have residual-defining
        # generated state.
        for reason, minimum in required_reasons.items():
            count = reason_counts.get(reason, 0)
            if not isinstance(count, int) or count < minimum:
                errors.append(
                    f"curvature projection reason '{reason}' count {count} is below {minimum}"
                )
        residual_refreshes = reason_counts.get("jacobian_and_residual", 0)
        residual_path_valid = (
            isinstance(residual_refreshes, int) and
            residual_refreshes >= accepted_steps
        )
        outer_iterations = summary.get("outer_iterations_total")
        projected_outer_refreshes = reason_counts.get(
            "projected_outer_fixed_point", 0)
        outer_path_valid = (
            isinstance(outer_iterations, int) and outer_iterations > 0 and
            isinstance(projected_outer_refreshes, int) and
            projected_outer_refreshes >= outer_iterations
        )
        # Production generated geometry has two admissible nonlinear
        # contracts.  Refreshed residual/Jacobian solves synchronize generated
        # state at assembly.  Frozen-generated-state solves instead establish
        # one projected constraint/state fixed point before every inner solve;
        # the inner residual and Jacobian then share that exact state.  The
        # latter deliberately has no jacobian_and_residual callback.
        if not residual_path_valid and not outer_path_valid:
            errors.append(
                "curvature projection freshness has neither "
                "jacobian_and_residual refreshes for every accepted step "
                "nor projected_outer_fixed_point refreshes for every "
                "reported outer iteration "
                f"(jacobian_and_residual={residual_refreshes!r}, "
                f"accepted_steps={accepted_steps}, "
                f"projected_outer_fixed_point={projected_outer_refreshes!r}, "
                f"outer_iterations_total={outer_iterations!r})"
            )
    return errors


def projected_curvature_interface_band_observation(
        metrics: dict[str, Any]) -> dict[str, Any] | None:
    """Return final-state projected curvature measured near the interface.

    Prefer the mean of an explicitly serialized projected-curvature field.
    Production decks currently expose the projection through the solver
    diagnostic instead, in which case ``max_abs_curvature`` is admissible only
    when the latest projection used a positive narrow band and actually
    excluded far vertices.  That qualification prevents the all-domain
    maximum (including the signed-distance medial-axis singularity) from being
    mistaken for interface curvature.
    """
    observed = metrics.get("projected_curvature_near_interface_mean_abs")
    band_width = metrics.get("projected_curvature_near_interface_band_width")
    sample_count = metrics.get("projected_curvature_near_interface_point_count")
    if (isinstance(observed, (int, float)) and not isinstance(observed, bool) and
            math.isfinite(float(observed)) and
            isinstance(band_width, (int, float)) and
            not isinstance(band_width, bool) and
            math.isfinite(float(band_width)) and float(band_width) > 0.0 and
            isinstance(sample_count, int) and not isinstance(sample_count, bool) and
            sample_count > 0):
        return {
            "value": float(observed),
            "metric": "projected_curvature_near_interface_mean_abs",
            "source": "vtk_point_data_projected_curvature_field",
            "source_field": metrics.get("projected_curvature_field_name"),
            "statistic": "mean_abs",
            "band_width": float(band_width),
            "sample_count": sample_count,
            "sampling_domain": "abs(phi)<=postprocessed_interface_band_width",
        }

    latest = metrics.get("latest_curvature_projection")
    if not isinstance(latest, dict):
        return None
    observed = latest.get("max_abs_curvature")
    band_width = latest.get("narrow_band_width")
    sample_count = latest.get("narrow_band_vertices")
    skipped_far = latest.get("skipped_far_vertices")
    if (not isinstance(observed, (int, float)) or isinstance(observed, bool) or
            not math.isfinite(float(observed)) or
            not isinstance(band_width, (int, float)) or
            isinstance(band_width, bool) or
            not math.isfinite(float(band_width)) or float(band_width) <= 0.0 or
            not isinstance(sample_count, int) or isinstance(sample_count, bool) or
            sample_count <= 0 or
            not isinstance(skipped_far, int) or isinstance(skipped_far, bool) or
            skipped_far <= 0):
        return None
    return {
        "value": float(observed),
        "metric": "latest_curvature_projection.max_abs_curvature",
        "source": "solver_latest_curvature_projection_diagnostic",
        "source_field": latest.get("curvature_field"),
        "statistic": "max_abs",
        "band_width": float(band_width),
        "sample_count": sample_count,
        "skipped_far_vertex_count": skipped_far,
        "sampling_domain": (
            "configured_narrow_band_and_interface_sample_adjacency"
        ),
    }


def capillary_benchmark_errors(metrics: dict[str, Any],
                               args: argparse.Namespace) -> list[str]:
    errors = []
    benchmark = metrics.get("benchmark")
    if not isinstance(benchmark, dict):
        benchmark = {}

    radius = benchmark.get("capillary_radius", benchmark.get("capillary_arc_radius"))
    curvature_factor = benchmark.get("capillary_curvature_factor", 1.0)
    if args.max_capillary_curvature_relative_error is not None:
        observation = projected_curvature_interface_band_observation(metrics)
        if not isinstance(radius, (int, float)) or float(radius) <= 0.0:
            errors.append("capillary curvature gate requires benchmark capillary radius")
        elif observation is None:
            errors.append(
                "capillary curvature gate requires a projected-curvature "
                "interface-band statistic"
            )
        else:
            observed = observation["value"]
            if (not isinstance(curvature_factor, (int, float)) or
                    isinstance(curvature_factor, bool) or
                    not math.isfinite(float(curvature_factor)) or
                    float(curvature_factor) <= 0.0):
                errors.append(
                    "capillary curvature gate requires a positive curvature "
                    "factor")
                return errors
            expected = float(curvature_factor) / float(radius)
            relative_error = abs(float(observed) - expected) / max(abs(expected), 1.0e-300)
            metrics["capillary_benchmark_radius"] = float(radius)
            metrics["capillary_expected_curvature"] = expected
            metrics["capillary_observed_curvature"] = float(observed)
            statistic = str(observation["statistic"])
            metrics[
                f"capillary_projected_curvature_interface_band_{statistic}"
            ] = float(observed)
            metrics["capillary_projected_curvature_observed_metric"] = (
                observation["metric"]
            )
            metrics["capillary_projected_curvature_observed_source"] = (
                observation["source"]
            )
            metrics["capillary_projected_curvature_observed_statistic"] = statistic
            metrics["capillary_projected_curvature_interface_band_width"] = (
                observation["band_width"]
            )
            metrics["capillary_projected_curvature_interface_band_sample_count"] = (
                observation["sample_count"]
            )
            metrics["capillary_projected_curvature_sampling_domain"] = (
                observation["sampling_domain"]
            )
            source_field = observation.get("source_field")
            if isinstance(source_field, str) and source_field:
                metrics["capillary_projected_curvature_source_field"] = source_field
            skipped_far = observation.get("skipped_far_vertex_count")
            if isinstance(skipped_far, int):
                metrics[
                    "capillary_projected_curvature_interface_band_skipped_far_vertex_count"
                ] = skipped_far
            metrics["capillary_curvature_relative_error"] = relative_error
            if relative_error > args.max_capillary_curvature_relative_error:
                errors.append(
                    f"capillary curvature relative error {relative_error:.6g} exceeds "
                    f"{args.max_capillary_curvature_relative_error:.6g}"
                )

    if args.max_capillary_pressure_jump_relative_error is not None:
        surface_tension = metrics.get("surface_tension")
        observed_jump = metrics.get("capillary_final_pressure_jump")
        if not isinstance(radius, (int, float)) or float(radius) <= 0.0:
            errors.append("capillary pressure-jump gate requires benchmark capillary radius")
        elif not isinstance(surface_tension, (int, float)):
            errors.append("capillary pressure-jump gate requires surface-tension control")
        elif not isinstance(observed_jump, (int, float)):
            errors.append(
                "capillary pressure-jump gate requires final liquid/gas pressure samples")
        else:
            # The level-set convention is phi < 0 in the liquid with the
            # interface normal pointing out of the liquid.  Positive convex
            # curvature therefore requires p_liquid - p_external = gamma*kappa.
            if (not isinstance(curvature_factor, (int, float)) or
                    isinstance(curvature_factor, bool) or
                    not math.isfinite(float(curvature_factor)) or
                    float(curvature_factor) <= 0.0):
                errors.append(
                    "capillary pressure-jump gate requires a positive "
                    "curvature factor")
                return errors
            expected_jump = (
                float(curvature_factor) * float(surface_tension) /
                float(radius)
            )
            relative_error = (
                abs(float(observed_jump) - expected_jump) /
                max(abs(expected_jump), 1.0e-300)
            )
            metrics["capillary_benchmark_radius"] = float(radius)
            metrics["capillary_expected_pressure_jump"] = expected_jump
            metrics["capillary_observed_final_pressure_jump"] = float(observed_jump)
            metrics["capillary_pressure_jump_relative_error"] = relative_error
            if relative_error > args.max_capillary_pressure_jump_relative_error:
                errors.append(
                    f"capillary pressure-jump relative error {relative_error:.6g} exceeds "
                    f"{args.max_capillary_pressure_jump_relative_error:.6g}"
                )

    maximum_capillary_number = getattr(
        args, "max_capillary_parasitic_capillary_number", None)
    if maximum_capillary_number is not None:
        if (not isinstance(maximum_capillary_number, (int, float)) or
                isinstance(maximum_capillary_number, bool) or
                not math.isfinite(float(maximum_capillary_number)) or
                float(maximum_capillary_number) < 0.0):
            errors.append(
                "maximum capillary number must be finite and nonnegative")
            return errors
        viscosity = benchmark.get("viscosity")
        surface_tension = metrics.get(
            "surface_tension", benchmark.get("surface_tension"))
        max_liquid_speed = metrics.get(
            "spatial_capillary_final_max_liquid_speed")
        if (not isinstance(viscosity, (int, float)) or
                isinstance(viscosity, bool) or
                not math.isfinite(float(viscosity)) or
                float(viscosity) < 0.0):
            errors.append(
                "capillary-number gate requires finite nonnegative viscosity")
        elif (not isinstance(surface_tension, (int, float)) or
              isinstance(surface_tension, bool) or
              not math.isfinite(float(surface_tension)) or
              float(surface_tension) <= 0.0):
            errors.append(
                "capillary-number gate requires positive surface tension")
        elif (not isinstance(max_liquid_speed, (int, float)) or
              isinstance(max_liquid_speed, bool) or
              not math.isfinite(float(max_liquid_speed)) or
              float(max_liquid_speed) < 0.0):
            errors.append(
                "capillary-number gate requires final spatial liquid speed")
        else:
            capillary_number = (
                float(viscosity) * float(max_liquid_speed) /
                float(surface_tension)
            )
            metrics["capillary_final_parasitic_capillary_number"] = (
                capillary_number)
            if capillary_number > float(maximum_capillary_number):
                errors.append(
                    "capillary parasitic capillary number "
                    f"{capillary_number:.6g} exceeds "
                    f"{float(maximum_capillary_number):.6g}"
                )
    return errors


def capillary_wave_expected_omega(wave: dict[str, Any]) -> float | None:
    surface_tension = wave.get("surface_tension")
    wavenumber = wave.get("wavenumber")
    density = wave.get("density")
    depth = wave.get("depth")
    if not all(isinstance(value, (int, float))
               for value in (surface_tension, wavenumber, density, depth)):
        return None
    if (float(density) <= 0.0 or float(surface_tension) < 0.0 or
            float(wavenumber) <= 0.0 or float(depth) <= 0.0):
        return None
    return math.sqrt(
        float(surface_tension) * float(wavenumber) ** 3 *
        math.tanh(float(wavenumber) * float(depth)) /
        float(density)
    )


def capillary_wave_final_time(metrics: dict[str, Any],
                              args: argparse.Namespace) -> float | None:
    time_loop = metrics.get("time_loop")
    if not isinstance(time_loop, dict):
        diagnostics = metrics.get("diagnostics", {})
        if isinstance(diagnostics, dict):
            time_loop = diagnostics.get("time_loop", {})
    summary = time_loop.get("summary") if isinstance(time_loop, dict) else None
    if isinstance(summary, dict):
        final_time = summary.get("final_accepted_time")
        if isinstance(final_time, (int, float)):
            return float(final_time)

    dt = getattr(args, "time_step_size", None)
    if not isinstance(dt, (int, float)):
        dt = metrics.get("time_step_size")
    if isinstance(dt, (int, float)):
        steps = metrics.get("steps", getattr(args, "steps", None))
        if isinstance(steps, (int, float)):
            return float(steps) * float(dt)
    return None


def capillary_wave_benchmark_errors(metrics: dict[str, Any],
                                    args: argparse.Namespace) -> list[str]:
    max_frequency_error = getattr(
        args, "max_capillary_wave_frequency_relative_error", None)
    max_profile_error = getattr(
        args, "max_capillary_wave_profile_relative_error", None)
    max_mean_offset = getattr(args, "max_capillary_wave_mean_offset", None)
    max_temporal_volume_drift = getattr(
        args,
        "max_capillary_wave_temporal_liquid_volume_relative_drift",
        None,
    )
    if (max_frequency_error is None and max_profile_error is None and
            max_mean_offset is None and max_temporal_volume_drift is None):
        return []

    errors = []
    benchmark = metrics.get("benchmark")
    wave = benchmark.get("capillary_wave") if isinstance(benchmark, dict) else None
    if not isinstance(wave, dict):
        return ["capillary-wave gates require benchmark capillary_wave metadata"]

    expected_omega = capillary_wave_expected_omega(wave)
    observed_omega = metrics.get("capillary_wave_observed_omega")
    if max_frequency_error is not None:
        if expected_omega is None or not isinstance(observed_omega, (int, float)):
            errors.append(
                "capillary-wave frequency gate requires an omega fitted from solved time history")
        else:
            relative_error = (
                abs(float(observed_omega) - expected_omega) /
                max(abs(expected_omega), 1.0e-300)
            )
            metrics["capillary_wave_expected_omega"] = expected_omega
            metrics["capillary_wave_omega_relative_error"] = relative_error
            phase_span = metrics.get("capillary_wave_frequency_observed_phase_span")
            if (not isinstance(phase_span, (int, float)) or
                    float(phase_span) <
                    CAPILLARY_WAVE_MINIMUM_FREQUENCY_PHASE_SPAN):
                errors.append(
                    "capillary-wave frequency observation spans less than "
                    f"{CAPILLARY_WAVE_MINIMUM_FREQUENCY_PHASE_SPAN:g} radians"
                )
            if relative_error > max_frequency_error:
                errors.append(
                    f"capillary-wave omega relative error {relative_error:.6g} "
                    f"exceeds {max_frequency_error:.6g}"
                )

    if max_profile_error is not None:
        amplitude = wave.get("amplitude")
        if not isinstance(amplitude, (int, float)) or float(amplitude) <= 0.0:
            errors.append("capillary-wave profile gate requires positive amplitude")
        elif expected_omega is None:
            errors.append("capillary-wave profile gate requires omega metadata")
        elif not metrics.get("final_capillary_wave_profile_available", False):
            errors.append("capillary-wave final interface profile fit is unavailable")
        else:
            final_time = capillary_wave_final_time(metrics, args)
            if final_time is None:
                errors.append("capillary-wave final time is unavailable")
            else:
                expected_cosine = (
                    float(amplitude) * math.cos(expected_omega * final_time)
                )
                expected_sine = 0.0
                observed_cosine = metrics.get("final_capillary_wave_cosine_amplitude")
                observed_sine = metrics.get("final_capillary_wave_sine_amplitude")
                if not all(isinstance(value, (int, float))
                           for value in (observed_cosine, observed_sine)):
                    errors.append("capillary-wave final profile fit is unavailable")
                else:
                    profile_error = math.hypot(
                        float(observed_cosine) - expected_cosine,
                        float(observed_sine) - expected_sine,
                    ) / max(abs(float(amplitude)), 1.0e-300)
                    metrics["capillary_wave_final_time_s"] = final_time
                    metrics["capillary_wave_expected_final_cosine_amplitude"] = (
                        expected_cosine
                    )
                    metrics["capillary_wave_expected_final_sine_amplitude"] = (
                        expected_sine
                    )
                    metrics["capillary_wave_profile_relative_error"] = profile_error
                    metrics["capillary_wave_signed_amplitude_error"] = (
                        float(observed_cosine) - expected_cosine
                    )
                    metrics["capillary_wave_normalized_amplitude_error"] = (
                        (float(observed_cosine) - expected_cosine) /
                        max(abs(float(amplitude)), 1.0e-300)
                    )
                    metrics["capillary_wave_apparent_damping_vs_inviscid"] = (
                        (abs(expected_cosine) - abs(float(observed_cosine))) /
                        max(abs(float(amplitude)), 1.0e-300)
                    )
                    if profile_error > max_profile_error:
                        errors.append(
                            "capillary-wave profile relative error "
                            f"{profile_error:.6g} exceeds {max_profile_error:.6g}"
                        )

    if max_mean_offset is not None:
        offset = metrics.get("final_capillary_wave_mean_offset")
        if not isinstance(offset, (int, float)):
            errors.append("capillary-wave mean-offset diagnostic is unavailable")
        elif abs(float(offset)) > max_mean_offset:
            errors.append(
                f"capillary-wave mean offset {abs(float(offset)):.6g} exceeds "
                f"{max_mean_offset:.6g}"
            )

    if max_temporal_volume_drift is not None:
        drift = metrics.get(
            "capillary_wave_temporal_liquid_volume_max_relative_drift")
        if (metrics.get("capillary_wave_temporal_liquid_volume_available") is not True or
                not isinstance(drift, (int, float)) or
                not math.isfinite(float(drift))):
            detail = metrics.get(
                "capillary_wave_temporal_liquid_volume_error")
            suffix = f": {detail}" if isinstance(detail, str) and detail else ""
            errors.append(
                "capillary-wave temporal liquid-volume drift diagnostic is "
                f"unavailable{suffix}")
        elif float(drift) > max_temporal_volume_drift:
            errors.append(
                "capillary-wave temporal liquid-volume relative drift "
                f"{float(drift):.6g} exceeds {max_temporal_volume_drift:.6g}"
            )
    return errors


def capillary_wave_boundary_contract_errors(
        metrics: dict[str, Any], args: argparse.Namespace) -> list[str]:
    if not getattr(args, "high_order_capillary_wave_smoke", False):
        return []
    if metrics.get("capillary_wave_boundary_contract_valid") is True:
        return []
    details = metrics.get("capillary_wave_boundary_contract_errors")
    if isinstance(details, list) and details:
        return [
            "capillary-wave wall/transport boundary contract failed: "
            + "; ".join(str(detail) for detail in details)
        ]
    return ["capillary-wave wall/transport boundary contract was not audited"]


def sessile_physical_errors(metrics: dict[str, Any],
                            args: argparse.Namespace) -> list[str]:
    """Apply nontrivial, solution-derived sessile/contact-line gates.

    The thresholds are opt-in so legacy dam-break probes are unaffected.  The
    FS-16 sessile cases enable them before running the solver; unavailable
    solved-field metrics fail instead of falling back to initial metadata.
    """
    errors: list[str] = []
    maximum_gates = (
        (
            "max_sessile_contact_angle_error_degrees",
            "sessile_final_contact_angle_absolute_error_degrees",
            "sessile contact-angle absolute error",
        ),
        (
            "max_sessile_pressure_jump_relative_error",
            "sessile_final_pressure_jump_relative_error",
            "sessile pressure-jump relative error",
        ),
        (
            "max_sessile_liquid_area_relative_error",
            "sessile_final_liquid_area_relative_error",
            "sessile liquid-area relative error",
        ),
        (
            "max_sessile_liquid_volume_relative_error",
            "sessile_final_liquid_volume_relative_error",
            "sessile liquid-volume relative error",
        ),
        (
            "max_sessile_base_radius_relative_error",
            "sessile_final_base_radius_relative_error",
            "sessile base-radius relative error",
        ),
        (
            "max_sessile_apex_height_relative_error",
            "sessile_final_apex_height_relative_error",
            "sessile apex-height relative error",
        ),
        (
            "max_sessile_parasitic_capillary_number",
            "sessile_final_parasitic_capillary_number",
            "sessile parasitic capillary number",
        ),
        (
            "max_ren_e_speed_relative_error",
            "ren_e_contact_fluid_speed_relative_error",
            "Ren--E contact-fluid constitutive speed relative error",
        ),
    )
    for argument, metric, label in maximum_gates:
        maximum = getattr(args, argument, None)
        if maximum is None:
            continue
        value = metrics.get(metric)
        if (not isinstance(value, (int, float)) or isinstance(value, bool) or
                not math.isfinite(float(value))):
            errors.append(f"{label} is unavailable from the solved time history")
        elif float(value) > float(maximum):
            errors.append(
                f"{label} {float(value):.6g} exceeds {float(maximum):.6g}"
            )

    accepted_angle_sources = {
        "same_state_LinearCorner_generated_fragment_normal_at_phi_zero_wall_roots",
        "same_state_LinearCorner_generated_triangle_normal_at_phi_zero_wall_edges",
    }
    if (getattr(args, "max_sessile_contact_angle_error_degrees", None)
            is not None and
            metrics.get("sessile_final_contact_angle_source") not in
            accepted_angle_sources):
        errors.append(
            "sessile contact angle was not evaluated from the same-state "
            "LinearCorner generated normal on the phi=0 wall contact"
        )

    ren_e_gate_enabled = (
        getattr(args, "max_ren_e_speed_relative_error", None) is not None or
        bool(getattr(args, "require_ren_e_speed_sign", False))
    )
    if (ren_e_gate_enabled and
            metrics.get("ren_e_contact_fluid_evaluation_source") !=
            GENERALIZED_ALPHA_REN_E_VELOCITY_SOURCE):
        errors.append(
            "Ren--E contact-fluid speed and footprint direction were not "
            "evaluated from reconstructed generalized-alpha stage Q1 velocity "
            "and the generated-fragment normal at both phi=0 wall roots"
        )

    if getattr(args, "require_ren_e_speed_sign", False):
        if metrics.get("ren_e_contact_fluid_speed_sign_agrees") is not True:
            errors.append(
                "Ren--E contact-line speed does not have the predicted "
                "advancing/receding sign"
            )
    return errors


def fsils_accepted_true_residual_errors(
        metrics: dict[str, Any], args: argparse.Namespace) -> list[str]:
    """Gate Newton's explicitly accepted inexact FSILS solves.

    Newton may accept a finite FSILS correction that misses the backend's
    requested target when its assembled true residual is already small versus
    the nonlinear tolerance.  Physical qualification must make that secondary
    acceptance threshold explicit instead of treating the rewritten
    `linear.converged` flag as proof that the original FSILS target was met.
    """
    maximum = getattr(args, "max_fsils_accepted_true_residual_norm", None)
    if maximum is None:
        return []
    if not isinstance(maximum, (int, float)) or not math.isfinite(float(maximum)) \
            or float(maximum) < 0.0:
        return ["maximum accepted FSILS true residual must be finite and nonnegative"]

    diagnostics = metrics.get("diagnostics", {})
    records = diagnostics.get("fsils_solve_summaries", []) \
        if isinstance(diagnostics, dict) else []
    errors: list[str] = []
    for index, record in enumerate(records):
        if not isinstance(record, dict) or record.get("converged") not in (0, False):
            continue
        value = record.get("final_residual_norm")
        if (not isinstance(value, (int, float)) or isinstance(value, bool) or
                not math.isfinite(float(value))):
            errors.append(
                f"FSILS nonconverged solve {index} has no finite assembled true residual"
            )
        elif float(value) > float(maximum):
            errors.append(
                f"FSILS accepted true residual {float(value):.6g} exceeds "
                f"{float(maximum):.6g} for nonconverged solve {index}"
            )
    return errors


def fsils_matrix_diag_col_mismatch_errors(
        metrics: dict[str, Any], args: argparse.Namespace) -> list[str]:
    """Gate disagreement between FSILS diagonal pointers and column indices."""
    maximum = getattr(args, "max_fsils_matrix_diag_col_mismatch", None)
    if maximum is None:
        return []
    mismatch = metrics.get(
        "diagnostic_fsils_prepared_matrix_max_diag_col_mismatch"
    )
    if not isinstance(mismatch, (int, float)) or isinstance(mismatch, bool):
        return [
            "FSILS prepared-matrix diagonal-column mismatch diagnostics are unavailable"
        ]
    if mismatch > maximum:
        return [
            f"FSILS prepared-matrix diagonal-column mismatches {mismatch} exceed "
            f"{maximum}"
        ]
    return []


def fsils_matrix_duplicate_diag_errors(
        metrics: dict[str, Any], args: argparse.Namespace) -> list[str]:
    """Gate duplicate structural diagonal blocks in prepared FSILS rows."""
    errors = []
    for argument, metric, label in (
            ("max_fsils_matrix_duplicate_diag_entries",
             "diagnostic_fsils_prepared_matrix_max_duplicate_diag_entries",
             "duplicate diagonal entries"),
            ("max_fsils_matrix_duplicate_diag_rows",
             "diagnostic_fsils_prepared_matrix_max_duplicate_diag_rows",
             "rows with duplicate diagonals")):
        maximum = getattr(args, argument, None)
        if maximum is None:
            continue
        value = metrics.get(metric)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            errors.append(
                f"FSILS prepared-matrix {label} diagnostics are unavailable"
            )
        elif value > maximum:
            errors.append(
                f"FSILS prepared-matrix {label} {value} exceed {maximum}"
            )
    return errors


def capillary_stability_errors(metrics: dict[str, Any],
                               args: argparse.Namespace) -> list[str]:
    gate_names = (
        "max_capillary_rejected_steps",
        "max_capillary_dt_updates",
        "max_capillary_speed_per_surface_tension",
        "max_capillary_nonlinear_residual",
        "max_capillary_linear_relative_residual",
    )
    if all(getattr(args, name, None) is None for name in gate_names):
        return []

    time_loop = metrics.get("time_loop")
    if not isinstance(time_loop, dict):
        diagnostics = metrics.get("diagnostics", {})
        if isinstance(diagnostics, dict):
            time_loop = diagnostics.get("time_loop", {})
    summary = time_loop.get("summary") if isinstance(time_loop, dict) else None
    if not isinstance(summary, dict):
        return ["capillary stability gates require time-loop convergence summary"]

    errors = []
    max_rejected = getattr(args, "max_capillary_rejected_steps", None)
    if max_rejected is not None:
        rejected_steps = summary.get("rejected_steps")
        if not isinstance(rejected_steps, int):
            errors.append("capillary rejected-step count was not reported")
        elif rejected_steps > max_rejected:
            errors.append(
                f"capillary rejected steps {rejected_steps} exceed {max_rejected}"
            )

    max_dt_updates = getattr(args, "max_capillary_dt_updates", None)
    if max_dt_updates is not None:
        dt_updates = summary.get("dt_updates")
        if not isinstance(dt_updates, int):
            errors.append("capillary dt-update count was not reported")
        elif dt_updates > max_dt_updates:
            errors.append(
                f"capillary dt updates {dt_updates} exceed {max_dt_updates}"
            )

    max_speed_per_surface_tension = getattr(
        args, "max_capillary_speed_per_surface_tension", None)
    if max_speed_per_surface_tension is not None:
        surface_tension = metrics.get("surface_tension")
        max_speed = metrics.get("max_speed")
        if not isinstance(surface_tension, (int, float)):
            errors.append("surface-tension control is unavailable")
        elif abs(float(surface_tension)) <= 0.0:
            errors.append("surface tension is zero; capillary stability cannot be normalized")
        elif not isinstance(max_speed, (int, float)):
            errors.append("capillary stability speed diagnostic is unavailable")
        else:
            normalized_speed = float(max_speed) / abs(float(surface_tension))
            metrics["capillary_stability_speed_per_surface_tension"] = (
                normalized_speed
            )
            if normalized_speed > max_speed_per_surface_tension:
                errors.append(
                    "capillary speed per surface tension "
                    f"{normalized_speed:.6g} exceeds "
                    f"{max_speed_per_surface_tension:.6g}"
                )

    max_nonlinear_residual = getattr(
        args, "max_capillary_nonlinear_residual", None)
    if max_nonlinear_residual is not None:
        nonlinear_residual = summary.get("nonlinear_residual")
        residual_max = (
            nonlinear_residual.get("max")
            if isinstance(nonlinear_residual, dict)
            else None
        )
        if not isinstance(residual_max, (int, float)):
            errors.append("capillary nonlinear residual summary was not reported")
        elif float(residual_max) > max_nonlinear_residual:
            errors.append(
                f"capillary nonlinear residual {float(residual_max):.6g} exceeds "
                f"{max_nonlinear_residual:.6g}"
            )

    max_linear_residual = getattr(
        args, "max_capillary_linear_relative_residual", None)
    if max_linear_residual is not None:
        linear_residual = summary.get("linear_relative_residual")
        residual_max = (
            linear_residual.get("max")
            if isinstance(linear_residual, dict)
            else None
        )
        if not isinstance(residual_max, (int, float)):
            errors.append("capillary linear relative residual summary was not reported")
        elif float(residual_max) > max_linear_residual:
            errors.append(
                "capillary linear relative residual "
                f"{float(residual_max):.6g} exceeds {max_linear_residual:.6g}"
            )

    return errors


def free_surface_conservative_balance_errors(
        metrics: dict[str, Any],
        args: argparse.Namespace) -> list[str]:
    """Validate the instantaneous conservative virtual-work diagnostic.

    This is deliberately not an energy-history or time-discrete energy-law
    gate.  It compares, at one synchronized nonlinear state, the constrained
    velocity-test coefficient vectors for pressure work, the declared physical
    potential first variation, and their sum.
    """
    required = bool(getattr(
        args, "require_free_surface_conservative_balance", False))
    maximum = getattr(
        args,
        "max_free_surface_conservative_balance_normalized_imbalance",
        None)
    if not required and maximum is None:
        return []

    diagnostics = metrics.get("diagnostics", {})
    records = diagnostics.get(
        "free_surface_conservative_balances", [])
    if not isinstance(records, list) or not records:
        return [
            "free-surface conservative virtual-work balance diagnostic was "
            "not reported"
        ]

    record = records[-1]
    if not isinstance(record, dict) or record.get("available") not in (1, True):
        reason = record.get("reason", "unavailable") if isinstance(
            record, dict) else "malformed_record"
        return [
            "free-surface conservative virtual-work balance diagnostic is "
            f"unavailable ({reason})"
        ]

    errors = []
    expected_contract = "instantaneous_constrained_velocity_test_virtual_work"
    if record.get("scope") != (
            "pressure_and_physical_potential_first_variations_only"):
        errors.append(
            "free-surface conservative balance has an unexpected physical scope"
        )
    if record.get("contract") != expected_contract:
        errors.append(
            "free-surface conservative balance has an unexpected contract "
            f"{record.get('contract')!r}"
        )
    if record.get("normalization") != "pressure_plus_physical_potential_norms":
        errors.append(
            "free-surface conservative balance has an unexpected normalization"
        )
    if record.get("discrete_energy_theorem_claimed") not in (0, False):
        errors.append(
            "free-surface conservative balance must not claim a discrete "
            "energy theorem"
        )
    if record.get("total_momentum_equilibrium_claimed") not in (0, False):
        errors.append(
            "free-surface conservative balance must not claim the complete "
            "momentum equilibrium"
        )

    numeric_keys = (
        "pressure_virtual_work_norm",
        "physical_potential_virtual_work_norm",
        "conservative_balance_norm",
        "normalized_imbalance",
    )
    numeric: dict[str, float] = {}
    for key in numeric_keys:
        value = record.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            errors.append(
                f"free-surface conservative balance {key} is unavailable or nonfinite"
            )
        else:
            numeric[key] = float(value)
            if float(value) < 0.0:
                errors.append(
                    f"free-surface conservative balance {key} is negative"
                )

    if all(key in numeric for key in numeric_keys):
        denominator = (
            numeric["pressure_virtual_work_norm"] +
            numeric["physical_potential_virtual_work_norm"])
        if not denominator > 0.0:
            errors.append(
                "free-surface conservative balance normalization denominator "
                "is zero"
            )
        else:
            recomputed = numeric["conservative_balance_norm"] / denominator
            if not math.isclose(
                    recomputed,
                    numeric["normalized_imbalance"],
                    rel_tol=1.0e-12,
                    abs_tol=1.0e-14):
                errors.append(
                    "free-surface conservative balance normalized imbalance "
                    "is inconsistent with its component norms"
                )
            # This triangle-inequality check is independent of any physical
            # threshold and catches a broken operator-sum contract.
            if numeric["normalized_imbalance"] > 1.0 + 1.0e-10:
                errors.append(
                    "free-surface conservative balance normalized imbalance "
                    "violates the operator-sum triangle bound"
                )
            if (maximum is not None and
                    numeric["normalized_imbalance"] > float(maximum)):
                errors.append(
                    "free-surface conservative balance normalized imbalance "
                    f"{numeric['normalized_imbalance']:.6g} exceeds "
                    f"{float(maximum):.6g}"
                )
    return errors


def free_surface_pressure_representability_errors(
        metrics: dict[str, Any],
        args: argparse.Namespace) -> list[str]:
    """Validate the pressure-space representability diagnostic contract.

    The diagnostic solves the constrained reduced pressure-to-velocity pairing
    problem for the prescribed-pressure, surface-area, Young-wall-energy, and
    gravitational-potential load. Telemetry-only callers require finite
    normal-equation stationarity. A configured maximum additionally requires
    the solver's accepted-static-state distance gate.
    """
    maximum = getattr(
        args,
        "max_free_surface_pressure_representability_relative_distance",
        None,
    )
    required = bool(getattr(
        args,
        "require_free_surface_pressure_representability_diagnostic",
        False,
    ))
    initializer_required = bool(getattr(
        args, "initialize_static_compatible_pressure", False))
    if not required and maximum is None and not initializer_required:
        return []

    diagnostics = metrics.get("diagnostics", {})
    records = diagnostics.get("free_surface_conservative_balances", [])
    if not isinstance(records, list) or not records:
        return [
            "free-surface pressure-representability diagnostic was not reported"
        ]

    record = records[-1]
    if not isinstance(record, dict):
        return [
            "free-surface pressure-representability diagnostic record is malformed"
        ]
    if record.get("pressure_representability_available") not in (1, True):
        reason = record.get(
            "pressure_representability_reason",
            record.get("reason", "unavailable"),
        )
        return [
            "free-surface pressure-representability diagnostic is unavailable "
            f"({reason})"
        ]

    errors = []
    expected_strings = {
        "pressure_representability_method": "lsqr",
        "pressure_representability_convergence": (
            "normal_equation_stationarity"
        ),
        "pressure_representability_norm": (
            "constrained_reduced_coefficient_l2"
        ),
        "pressure_representability_load": (
            "prescribed_external_pressure_plus_surface_area_variation_plus_"
            "young_wall_energy_plus_gravitational_potential"
        ),
    }
    for key, expected in expected_strings.items():
        value = record.get(key)
        if value != expected:
            errors.append(
                "free-surface pressure-representability diagnostic has "
                f"unexpected {key} {value!r}; expected {expected!r}"
            )

    expected_flags = {
        "pressure_representability_distance_gate_applied": 0,
        "pressure_representability_claimed": 0,
    }
    for key, expected in expected_flags.items():
        value = record.get(key)
        if value not in (expected, bool(expected)):
            errors.append(
                "free-surface pressure-representability diagnostic has "
                f"unexpected {key} {value!r}; expected {expected}"
            )

    if record.get("pressure_representability_converged") not in (1, True):
        errors.append(
            "free-surface pressure-representability diagnostic did not converge"
        )
    if record.get("pressure_representability_breakdown") not in (0, False):
        errors.append(
            "free-surface pressure-representability diagnostic reported breakdown"
        )

    for key in (
            "pressure_representability_residual_norm",
            "pressure_representability_relative_residual",
            "pressure_representability_normal_residual_norm",
            "pressure_representability_relative_normal_residual",
            "pressure_representability_pressure_norm"):
        value = record.get(key)
        if (isinstance(value, bool) or
                not isinstance(value, (int, float)) or
                not math.isfinite(float(value)) or
                float(value) < 0.0):
            errors.append(
                "free-surface pressure-representability diagnostic "
                f"{key} is unavailable, nonfinite, or negative"
            )

    iterations = record.get("pressure_representability_iterations")
    if (isinstance(iterations, bool) or
            not isinstance(iterations, int) or iterations < 0):
        errors.append(
            "free-surface pressure-representability diagnostic "
            "pressure_representability_iterations is unavailable or negative"
        )

    if maximum is None:
        if initializer_required:
            errors.append(
                "static compatible-pressure initializer requires a finite "
                "pressure-representability maximum relative distance"
            )
        return errors
    if (isinstance(maximum, bool) or
            not isinstance(maximum, (int, float)) or
            not math.isfinite(float(maximum)) or float(maximum) < 0.0):
        errors.append(
            "free-surface pressure-representability maximum relative distance "
            "is unavailable, nonfinite, or negative"
        )
        return errors

    if initializer_required:
        initializer_records = diagnostics.get(
            "free_surface_static_compatible_pressure_initializers", [])
        if not isinstance(initializer_records, list) or not initializer_records:
            errors.append(
                "static compatible-pressure initializer was not reported"
            )
        else:
            malformed = any(
                not isinstance(item, dict) for item in initializer_records)
            if malformed:
                errors.append(
                    "static compatible-pressure initializer record is malformed"
                )
            else:
                failed = [
                    item for item in initializer_records
                    if item.get("passed") not in (1, True)
                ]
                if failed:
                    errors.append(
                        "static compatible-pressure initializer reported a "
                        f"failed attempt ({failed[-1].get('reason', 'unknown')})"
                    )
                applied = [
                    item for item in initializer_records
                    if item.get("applied") in (1, True)
                ]
                if not applied:
                    errors.append(
                        "static compatible-pressure initializer never applied "
                        "the pressure preload"
                    )
                else:
                    initialized = applied[0]
                    expected_initializer_flags = {
                        "requested": 1,
                        "applied": 1,
                        "passed": 1,
                        "pressure_representability_available": 1,
                        "pressure_representability_converged": 1,
                        "pressure_representability_breakdown": 0,
                        "force_projection_applied": 0,
                        "production_capillary_operator_changed": 0,
                    }
                    for key, expected in expected_initializer_flags.items():
                        value = initialized.get(key)
                        if value not in (expected, bool(expected)):
                            errors.append(
                                "static compatible-pressure initializer has "
                                f"unexpected {key} {value!r}; expected {expected}"
                            )
                    if initialized.get("reason") != (
                            "initialized_within_threshold"):
                        errors.append(
                            "static compatible-pressure initializer has "
                            f"unexpected reason {initialized.get('reason')!r}"
                        )
                    initializer_distance = initialized.get(
                        "pressure_representability_relative_residual")
                    initializer_maximum = initialized.get(
                        "pressure_representability_max_relative_distance")
                    for key, value in (
                            ("pressure_representability_relative_residual",
                             initializer_distance),
                            ("pressure_representability_max_relative_distance",
                             initializer_maximum)):
                        if (isinstance(value, bool) or
                                not isinstance(value, (int, float)) or
                                not math.isfinite(float(value)) or
                                float(value) < 0.0):
                            errors.append(
                                "static compatible-pressure initializer "
                                f"{key} is unavailable, nonfinite, or negative"
                            )
                    if (isinstance(initializer_maximum, (int, float)) and
                            not isinstance(initializer_maximum, bool) and
                            math.isfinite(float(initializer_maximum)) and
                            not math.isclose(
                                float(initializer_maximum), float(maximum),
                                rel_tol=1.0e-13, abs_tol=1.0e-15)):
                        errors.append(
                            "static compatible-pressure initializer maximum "
                            f"{float(initializer_maximum):.6g} does not match "
                            f"{float(maximum):.6g}"
                        )
                    if (isinstance(initializer_distance, (int, float)) and
                            not isinstance(initializer_distance, bool) and
                            math.isfinite(float(initializer_distance)) and
                            float(initializer_distance) > float(maximum)):
                        errors.append(
                            "static compatible-pressure initializer distance "
                            f"{float(initializer_distance):.6g} exceeds "
                            f"{float(maximum):.6g}"
                        )

    gate_records = diagnostics.get(
        "free_surface_pressure_representability_distance_gates", [])
    if not isinstance(gate_records, list) or not gate_records:
        errors.append(
            "accepted-static pressure-representability distance gate was not "
            "reported"
        )
        return errors
    gate_record = gate_records[-1]
    if not isinstance(gate_record, dict):
        errors.append(
            "accepted-static pressure-representability distance gate record "
            "is malformed"
        )
        return errors

    expected_gate_flags = {
        "accepted_static_state": 1,
        "pressure_representability_distance_gate_applied": 1,
        "pressure_representability_available": 1,
        "pressure_representability_converged": 1,
        "pressure_representability_breakdown": 0,
        "pressure_representability_distance_gate_passed": 1,
        "pressure_representability_claimed": 1,
    }
    for key, expected in expected_gate_flags.items():
        value = gate_record.get(key)
        if value not in (expected, bool(expected)):
            errors.append(
                "accepted-static pressure-representability distance gate has "
                f"unexpected {key} {value!r}; expected {expected}"
            )

    relative_distance = gate_record.get(
        "pressure_representability_relative_residual")
    reported_maximum = gate_record.get(
        "pressure_representability_max_relative_distance")
    for key, value in (
            ("pressure_representability_relative_residual", relative_distance),
            ("pressure_representability_max_relative_distance",
             reported_maximum)):
        if (isinstance(value, bool) or
                not isinstance(value, (int, float)) or
                not math.isfinite(float(value)) or float(value) < 0.0):
            errors.append(
                "accepted-static pressure-representability distance gate "
                f"{key} is unavailable, nonfinite, or negative"
            )

    if (isinstance(reported_maximum, (int, float)) and
            not isinstance(reported_maximum, bool) and
            math.isfinite(float(reported_maximum)) and
            not math.isclose(
                float(reported_maximum),
                float(maximum),
                rel_tol=1.0e-13,
                abs_tol=1.0e-15)):
        errors.append(
            "accepted-static pressure-representability distance gate reported "
            f"maximum {float(reported_maximum):.6g}; expected "
            f"{float(maximum):.6g}"
        )
    if (isinstance(relative_distance, (int, float)) and
            not isinstance(relative_distance, bool) and
            math.isfinite(float(relative_distance)) and
            float(relative_distance) > float(maximum)):
        errors.append(
            "accepted-static pressure-representability relative distance "
            f"{float(relative_distance):.6g} exceeds {float(maximum):.6g}"
        )
    if gate_record.get("reason") != "within_threshold":
        errors.append(
            "accepted-static pressure-representability distance gate has "
            f"unexpected reason {gate_record.get('reason')!r}; expected "
            "'within_threshold'"
        )
    return errors


def static_capillary_equilibrium_initialization_errors(
        metrics: dict[str, Any], args: argparse.Namespace) -> list[str]:
    if not bool(getattr(
            args, "initialize_discrete_static_capillary_equilibrium", False)):
        return []

    diagnostics = metrics.get("diagnostics", {})
    records = diagnostics.get(
        "static_capillary_equilibrium_initializations", [])
    if not isinstance(records, list) or len(records) != 1:
        count = len(records) if isinstance(records, list) else "malformed"
        return [
            "discrete static-capillary equilibrium initialization requires "
            f"exactly one diagnostic record (observed {count})"
        ]
    record = records[0]
    if not isinstance(record, dict):
        return [
            "discrete static-capillary equilibrium initialization diagnostic "
            "is malformed"
        ]

    errors: list[str] = []
    expected_flags = {
        "pressure_representability_available": 1,
        "pressure_representability_converged": 1,
        "pressure_representability_breakdown": 0,
        "production_force_projection_applied": 0,
    }
    for key, expected in expected_flags.items():
        value = record.get(key)
        if value not in (expected, bool(expected)):
            errors.append(
                "discrete static-capillary initialization has unexpected "
                f"{key} {value!r}; expected {expected}"
            )

    constant_pressure_required = record.get("constant_pressure_kkt_required")
    constant_pressure_available = record.get(
        "constant_pressure_kkt_available")
    if constant_pressure_required not in (0, 1, False, True):
        errors.append(
            "discrete static-capillary initialization has invalid "
            "constant_pressure_kkt_required flag"
        )
    if constant_pressure_available not in (0, 1, False, True):
        errors.append(
            "discrete static-capillary initialization has invalid "
            "constant_pressure_kkt_available flag"
        )
    if bool(constant_pressure_required) != bool(constant_pressure_available):
        errors.append(
            "discrete static-capillary initialization constant-pressure "
            "certificate availability disagrees with whether it was required"
        )

    for key in ("active_coefficients", "functional_evaluations",
                "acceptance_certificate_evaluations"):
        value = record.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            errors.append(
                "discrete static-capillary initialization has invalid "
                f"{key} {value!r}"
            )
    for key in (
            "finite_difference_fourth_order_components",
            "analytic_derivative_evaluations",
            "derivative_resolution_step_acceptances"):
        value = record.get(key)
        if (isinstance(value, bool) or not isinstance(value, int) or
                value < 0):
            errors.append(
                "discrete static-capillary initialization has invalid "
                f"{key} {value!r}"
            )
    if getattr(args, "capillary_force_form", None) == (
            "kinematic_area_gradient_traction"):
        analytic_evaluations = record.get("analytic_derivative_evaluations")
        difference_components = record.get(
            "finite_difference_fourth_order_components")
        if (isinstance(analytic_evaluations, bool) or
                not isinstance(analytic_evaluations, int) or
                analytic_evaluations <= 0):
            errors.append(
                "kinematic area-gradient static initialization did not use "
                "exact functional derivatives"
            )
        if difference_components != 0:
            errors.append(
                "kinematic area-gradient static initialization unexpectedly "
                "used finite-difference derivative components"
            )
    iterations = record.get("iterations")
    if (isinstance(iterations, bool) or not isinstance(iterations, int) or
            iterations < 0):
        errors.append(
            "discrete static-capillary initialization has invalid iteration "
            f"count {iterations!r}"
        )
    maximum_iterations = getattr(
        args, "static_capillary_max_iterations", None)
    if (isinstance(iterations, int) and
            isinstance(maximum_iterations, int) and
            iterations > maximum_iterations):
        errors.append(
            "discrete static-capillary initialization iteration count "
            f"{iterations} exceeds {maximum_iterations}"
        )
    topology_transitions = record.get("topology_epoch_transitions")
    reported_topology_limit = record.get(
        "max_topology_epoch_transitions")
    requested_topology_limit = getattr(
        args, "static_capillary_max_topology_epoch_transitions", None)
    if (isinstance(topology_transitions, bool) or
            not isinstance(topology_transitions, int) or
            topology_transitions < 0):
        errors.append(
            "discrete static-capillary initialization has invalid topology "
            f"epoch transition count {topology_transitions!r}"
        )
    if (isinstance(reported_topology_limit, bool) or
            not isinstance(reported_topology_limit, int) or
            reported_topology_limit < 0):
        errors.append(
            "discrete static-capillary initialization has invalid maximum "
            f"topology epoch transitions {reported_topology_limit!r}"
        )
    elif (isinstance(topology_transitions, int) and
          topology_transitions > reported_topology_limit):
        errors.append(
            "discrete static-capillary initialization topology epoch "
            f"transition count {topology_transitions} exceeds reported "
            f"limit {reported_topology_limit}"
        )
    if (requested_topology_limit is not None and
            reported_topology_limit != requested_topology_limit):
        errors.append(
            "discrete static-capillary initialization reported maximum "
            f"topology epoch transitions {reported_topology_limit!r}; "
            f"expected {requested_topology_limit}"
        )

    target_volume = record.get("target_liquid_volume")
    if (isinstance(target_volume, bool) or
            not isinstance(target_volume, (int, float)) or
            not math.isfinite(float(target_volume)) or
            float(target_volume) <= 0.0):
        errors.append(
            "discrete static-capillary initialization target volume is "
            "unavailable, nonfinite, or nonpositive"
        )

    for key in (
            "initial_physical_potential_energy",
            "final_physical_potential_energy",
            "final_volume_error",
            "final_projected_gradient_norm",
            "pressure_representability_residual_norm",
            "pressure_representability_relative_distance",
            "production_residual_norm"):
        value = record.get(key)
        if (isinstance(value, bool) or
                not isinstance(value, (int, float)) or
                not math.isfinite(float(value))):
            errors.append(
                "discrete static-capillary initialization has unavailable "
                f"or nonfinite {key}"
            )

    initial_energy = record.get("initial_physical_potential_energy")
    final_energy = record.get("final_physical_potential_energy")
    if all(
            isinstance(value, (int, float)) and not isinstance(value, bool) and
            math.isfinite(float(value))
            for value in (initial_energy, final_energy)):
        energy_scale = max(1.0, abs(float(initial_energy)))
        if float(final_energy) > (
                float(initial_energy) + 1.0e-12 * energy_scale):
            errors.append(
                "discrete static-capillary initialization increased physical "
                "potential energy"
            )

    threshold_checks = (
        ("final_volume_error", "static_capillary_volume_tolerance", True),
        ("final_projected_gradient_norm",
         "static_capillary_projected_gradient_tolerance", False),
        ("pressure_representability_residual_norm",
         "static_capillary_pressure_representability_max_residual_norm",
         False),
        ("pressure_representability_relative_distance",
         "static_capillary_pressure_representability_max_relative_distance",
         False),
        ("production_residual_norm",
         "static_capillary_physical_equilibrium_max_residual_norm", False),
    )
    for metric_name, argument_name, use_absolute_value in threshold_checks:
        threshold = getattr(args, argument_name, None)
        value = record.get(metric_name)
        if threshold is None:
            continue
        if (isinstance(threshold, bool) or
                not isinstance(threshold, (int, float)) or
                not math.isfinite(float(threshold)) or
                float(threshold) < 0.0):
            errors.append(f"{argument_name} is invalid")
            continue
        if (isinstance(value, (int, float)) and
                not isinstance(value, bool) and math.isfinite(float(value))):
            compared = abs(float(value)) if use_absolute_value else float(value)
            if compared > float(threshold):
                errors.append(
                    "discrete static-capillary initialization "
                    f"{metric_name} {compared:.6g} exceeds "
                    f"{float(threshold):.6g}"
                )

    if bool(constant_pressure_required):
        for metric_name, argument_name in (
                ("constant_pressure_kkt_residual_norm",
                 "static_capillary_constant_pressure_kkt_max_residual_norm"),
                ("constant_pressure_kkt_relative_distance",
                 "static_capillary_constant_pressure_kkt_max_relative_distance")):
            value = record.get(metric_name)
            if (isinstance(value, bool) or
                    not isinstance(value, (int, float)) or
                    not math.isfinite(float(value)) or float(value) < 0.0):
                errors.append(
                    "discrete static-capillary initialization has unavailable, "
                    f"nonfinite, or negative {metric_name}"
                )
                continue
            threshold = getattr(args, argument_name, None)
            if threshold is None:
                continue
            if (isinstance(threshold, bool) or
                    not isinstance(threshold, (int, float)) or
                    not math.isfinite(float(threshold)) or
                    float(threshold) < 0.0):
                errors.append(f"{argument_name} is invalid")
            elif float(value) > float(threshold):
                errors.append(
                    "discrete static-capillary initialization "
                    f"{metric_name} {float(value):.6g} exceeds "
                    f"{float(threshold):.6g}"
                )

    required_qualification = getattr(
        args, "require_static_capillary_balance_qualification", None)
    if (required_qualification is not None and
            record.get("qualification") != required_qualification):
        errors.append(
            "discrete static-capillary initialization reported qualification "
            f"{record.get('qualification')!r}; expected "
            f"{required_qualification!r}"
        )
    return errors


def normalized_capillary_convergence_metrics(value: Any) -> list[str]:
    if value is None:
        return ["capillary_curvature_relative_error"]
    raw_values: list[Any]
    if isinstance(value, (list, tuple)):
        raw_values = list(value)
    else:
        raw_values = [value]
    metrics: list[str] = []
    for raw in raw_values:
        for item in str(raw).split(","):
            name = item.strip()
            if name:
                metrics.append(name)
    return metrics or ["capillary_curvature_relative_error"]


def capillary_convergence_rate_errors(
        probes: list[dict[str, Any]],
        args: argparse.Namespace) -> list[str]:
    min_rate = getattr(args, "min_capillary_convergence_rate", None)
    if min_rate is None:
        return []
    min_points = getattr(args, "min_capillary_convergence_points", None)
    if min_points is None:
        min_points = 2
    resolution_key = (
        getattr(args, "capillary_convergence_resolution_key", None) or
        "capillary_convergence_resolution"
    )
    metric_names = normalized_capillary_convergence_metrics(
        getattr(args, "capillary_convergence_metric", None))

    errors = []
    if min_points < 2:
        errors.append("capillary convergence requires at least two points")
        min_points = 2

    for metric_name in metric_names:
        samples: list[tuple[float, float]] = []
        for probe in probes:
            if probe.get("passed") is False:
                continue
            resolution = probe.get(resolution_key)
            metric = probe.get(metric_name)
            if isinstance(resolution, (int, float)) and isinstance(metric, (int, float)):
                if float(resolution) > 0.0 and float(metric) > 0.0:
                    samples.append((float(resolution), float(metric)))

        if len(samples) < min_points:
            errors.append(
                f"capillary convergence metric {metric_name} has {len(samples)} "
                f"usable sample(s), below {min_points}; expected positive numeric "
                f"{resolution_key} and {metric_name} values"
            )
            continue

        samples.sort(key=lambda item: item[0])
        duplicate_resolutions = [
            samples[i][0]
            for i in range(1, len(samples))
            if samples[i][0] == samples[i - 1][0]
        ]
        if duplicate_resolutions:
            errors.append(
                f"capillary convergence metric {metric_name} has duplicate "
                f"{resolution_key} value {duplicate_resolutions[0]:.6g}"
            )
            continue

        observed_rates: list[float] = []
        for (coarse_resolution, coarse_error), (fine_resolution, fine_error) in zip(
                samples, samples[1:]):
            if fine_resolution <= coarse_resolution:
                continue
            rate = (
                math.log(coarse_error / fine_error) /
                math.log(fine_resolution / coarse_resolution)
            )
            observed_rates.append(rate)

        if not observed_rates:
            errors.append(
                f"capillary convergence metric {metric_name} has no strictly "
                f"increasing {resolution_key} samples"
            )
            continue

        min_observed = min(observed_rates)
        if min_observed < float(min_rate):
            errors.append(
                f"capillary convergence rate for {metric_name} "
                f"{min_observed:.6g} is below {float(min_rate):.6g}"
            )

    return errors


def timeout_before_solution_state(diagnostics: dict[str, Any]) -> bool:
    summary = diagnostics.get("time_loop", {}).get("summary", {})
    if not isinstance(summary, dict):
        return False
    nonlinear_records = summary.get("nonlinear_records", 0)
    accepted_steps = summary.get("accepted_steps", 0)
    return int(nonlinear_records or 0) == 0 and int(accepted_steps or 0) == 0


def assembly_topology_consistency_errors(diagnostics: dict[str, Any]) -> list[str]:
    errors = []
    facet_counts = [
        int(record["cut_adjacent_facets"])
        for record in diagnostics.get("cut_context_rebuilds", [])
        if isinstance(record.get("cut_adjacent_facets"), int)
    ]
    if diagnostics.get("interior_face_timings"):
        if not facet_counts:
            errors.append("cut-adjacent facet count is unavailable for interior-face timing checks")
        else:
            expected_facets = set(facet_counts)
            mismatched = [
                int(record["faces_assembled"])
                for record in diagnostics["interior_face_timings"]
                if isinstance(record.get("faces_assembled"), int) and
                int(record["faces_assembled"]) not in expected_facets
            ]
            if mismatched:
                errors.append(
                    "interior-face timing assembled counts do not match cut-adjacent facets "
                    f"(expected one of {sorted(expected_facets)}, examples {mismatched[:3]})"
                )

    assembly_records: dict[tuple[Any, Any], list[dict[str, Any]]] = {}
    for record in diagnostics.get("cut_volume_assemblies", []):
        key = (record.get("marker"), record.get("side"))
        assembly_records.setdefault(key, []).append(record)

    for timing in diagnostics.get("cut_volume_timings", []):
        key = (timing.get("marker"), timing.get("side"))
        if key[0] is None or key[1] is None:
            errors.append("cut-volume timing record is missing marker or side")
            continue
        matches = assembly_records.get(key, [])
        if not matches:
            errors.append(f"cut-volume timing record has no assembly diagnostic for marker/side {key}")
            continue
        if timing.get("indexed") != 1:
            errors.append(f"cut-volume timing for marker/side {key} did not use indexed rule traversal")
        considered = timing.get("rules_considered")
        assembled = timing.get("rules_assembled")
        if isinstance(considered, int) and isinstance(assembled, int) and considered != assembled:
            errors.append(
                f"cut-volume timing for marker/side {key} considered {considered} rules but assembled {assembled}"
            )

        matched_counts = False
        for assembly in matches:
            if (assembled == assembly.get("rules") and
                    timing.get("full_rules") == assembly.get("full_cell_rules") and
                    timing.get("partial_rules") == assembly.get("cut_cell_rules")):
                matched_counts = True
                break
        if not matched_counts:
            errors.append(
                "cut-volume timing rule counts do not match cut-volume assembly diagnostics "
                f"for marker/side {key}"
            )
    return errors


def has_marked_interior_face_fallback_trace(diagnostics: dict[str, Any]) -> bool:
    return any(
        record.get("event") == "runtime_skip" and
        record.get("reason") == "marked_interior_face_fallback" and
        record.get("domain") == "InteriorFace"
        for record in diagnostics.get("jit_specialization_traces", [])
    )


def compiled_cut_volume_jit_errors(diagnostics: dict[str, Any]) -> list[str]:
    errors = []
    traces = diagnostics.get("jit_specialization_traces", [])

    if not any(record.get("event") == "generic_compile" for record in traces):
        errors.append(
            "compiled CutVolume JIT gate requires generic JIT compile evidence"
        )
    if not diagnostics.get("jit_cache_diagnostics"):
        errors.append(
            "compiled CutVolume JIT gate requires JIT cache diagnostics"
        )

    runtime_cut_volume_roles = {
        record.get("role")
        for record in traces
        if record.get("event") == "compile" and
        record.get("trigger") == "runtime" and
        record.get("domain") == "CutVolume"
    }
    missing_roles = [
        role for role in ("Tangent", "Residual")
        if role not in runtime_cut_volume_roles
    ]
    if missing_roles:
        errors.append(
            "compiled CutVolume JIT gate is missing runtime compile evidence "
            "for role(s): " + ", ".join(missing_roles)
        )

    failed_trace_count = sum(
        record.get("event") == "compile_failed" for record in traces
    )
    failure_messages = diagnostics.get("jit_failure_messages", [])
    if failed_trace_count or failure_messages:
        errors.append(
            "compiled CutVolume JIT gate observed JIT compile/runtime failure "
            f"diagnostics (trace_failures={failed_trace_count}, "
            f"messages={len(failure_messages)})"
        )
    return errors


def has_linear_solve_memory_diagnostics(diagnostics: dict[str, Any]) -> bool:
    phases = {
        record.get("phase")
        for record in diagnostics.get("process_memory", [])
        if record.get("phase") in {"before_linear_solve", "after_linear_solve"}
    }
    return {"before_linear_solve", "after_linear_solve"}.issubset(phases)


def add_diagnostic_metrics(metrics: dict[str, Any],
                           diagnostics: dict[str, Any]) -> None:
    metrics["diagnostics"] = diagnostics
    metrics["solver_controls"] = diagnostics.get("solver_controls", {})
    metrics["time_loop"] = diagnostics.get("time_loop", {})
    metrics.update(parse_active_volume_history_from_diagnostics(diagnostics))
    metrics["diagnostic_jit_failure_message_count"] = len(
        diagnostics.get("jit_failure_messages", [])
    )

    operator_angles = diagnostics.get("dynamic_contact_operator_angles", [])
    if isinstance(operator_angles, list):
        metrics["diagnostic_dynamic_contact_operator_angle_count"] = len(
            operator_angles)
        available_operator_angles = [
            record for record in operator_angles
            if isinstance(record, dict) and record.get("status") == "available"
        ]
        metrics["diagnostic_dynamic_contact_operator_angle_available_count"] = (
            len(available_operator_angles)
        )
        if available_operator_angles:
            latest_operator_angle = available_operator_angles[-1]
            metrics["latest_dynamic_contact_operator_angle"] = (
                latest_operator_angle)
            tangent_norms = [
                float(record["min_wall_tangential_normal_norm"])
                for record in available_operator_angles
                if isinstance(record.get("min_wall_tangential_normal_norm"),
                              (int, float))
            ]
            if tangent_norms:
                metrics[
                    "diagnostic_dynamic_contact_operator_angle_min_wall_tangential_normal_norm"
                ] = min(tangent_norms)

    conservative_balances = diagnostics.get(
        "free_surface_conservative_balances", [])
    if isinstance(conservative_balances, list):
        metrics["diagnostic_free_surface_conservative_balance_count"] = len(
            conservative_balances)
        available_balances = [
            record for record in conservative_balances
            if isinstance(record, dict) and record.get("available") in (1, True)
        ]
        metrics[
            "diagnostic_free_surface_conservative_balance_available_count"
        ] = len(available_balances)
        if conservative_balances:
            metrics["latest_free_surface_conservative_balance"] = (
                conservative_balances[-1])
        representability_records = [
            record for record in conservative_balances
            if isinstance(record, dict) and
            "pressure_representability_available" in record
        ]
        metrics[
            "diagnostic_free_surface_pressure_representability_count"
        ] = len(representability_records)
        available_representability_records = [
            record for record in representability_records
            if record.get("pressure_representability_available") in (1, True)
        ]
        metrics[
            "diagnostic_free_surface_pressure_representability_available_count"
        ] = len(available_representability_records)
        if representability_records:
            latest_representability = representability_records[-1]
            metrics["latest_free_surface_pressure_representability"] = (
                latest_representability)
            for source in (
                    "pressure_representability_available",
                    "pressure_representability_method",
                    "pressure_representability_convergence",
                    "pressure_representability_distance_gate_applied",
                    "pressure_representability_claimed",
                    "pressure_representability_residual_norm",
                    "pressure_representability_relative_residual",
                    "pressure_representability_normal_residual_norm",
                    "pressure_representability_relative_normal_residual",
                    "pressure_representability_pressure_norm",
                    "pressure_representability_iterations",
                    "pressure_representability_converged",
                    "pressure_representability_breakdown",
                    "pressure_representability_norm",
                    "pressure_representability_load"):
                value = latest_representability.get(source)
                if isinstance(value, (bool, int, float, str)):
                    metrics[f"diagnostic_free_surface_{source}"] = value
        if available_balances:
            latest_available = available_balances[-1]
            metrics["latest_available_free_surface_conservative_balance"] = (
                latest_available)
            for source, target in (
                    ("pressure_virtual_work_norm",
                     "diagnostic_free_surface_pressure_virtual_work_norm"),
                    ("surface_energy_virtual_work_norm",
                     "diagnostic_free_surface_surface_energy_virtual_work_norm"),
                    ("physical_potential_virtual_work_norm",
                     "diagnostic_free_surface_physical_potential_virtual_work_norm"),
                    ("conservative_balance_norm",
                     "diagnostic_free_surface_conservative_balance_norm"),
                    ("normalized_imbalance",
                     "diagnostic_free_surface_conservative_balance_normalized_imbalance"),
                    ("magnitude_mismatch",
                     "diagnostic_free_surface_conservative_balance_magnitude_mismatch"),
                    ("alignment_cosine",
                     "diagnostic_free_surface_conservative_balance_alignment_cosine")):
                value = latest_available.get(source)
                if isinstance(value, (int, float)):
                    metrics[target] = float(value)

    pressure_distance_gates = diagnostics.get(
        "free_surface_pressure_representability_distance_gates", [])
    if isinstance(pressure_distance_gates, list):
        metrics[
            "diagnostic_free_surface_pressure_representability_distance_gate_count"
        ] = len(pressure_distance_gates)
        if pressure_distance_gates:
            metrics[
                "latest_free_surface_pressure_representability_distance_gate"
            ] = pressure_distance_gates[-1]

    compatible_pressure_initializers = diagnostics.get(
        "free_surface_static_compatible_pressure_initializers", [])
    if isinstance(compatible_pressure_initializers, list):
        metrics[
            "diagnostic_free_surface_static_compatible_pressure_initializer_count"
        ] = len(compatible_pressure_initializers)
        if compatible_pressure_initializers:
            metrics[
                "latest_free_surface_static_compatible_pressure_initializer"
            ] = compatible_pressure_initializers[-1]

    static_capillary_initializations = diagnostics.get(
        "static_capillary_equilibrium_initializations", [])
    if isinstance(static_capillary_initializations, list):
        metrics[
            "diagnostic_static_capillary_equilibrium_initialization_count"
        ] = len(static_capillary_initializations)
        if static_capillary_initializations:
            latest_static_capillary = static_capillary_initializations[-1]
            metrics["latest_static_capillary_equilibrium_initialization"] = (
                latest_static_capillary)
            for source in (
                    "active_coefficients",
                    "target_liquid_volume",
                    "initial_physical_potential_energy",
                    "final_physical_potential_energy",
                    "final_volume_error",
                    "final_projected_gradient_norm",
                    "pressure_representability_residual_norm",
                    "pressure_representability_relative_distance",
                    "production_residual_norm",
                    "constant_pressure_kkt_required",
                    "constant_pressure_kkt_available",
                    "constant_pressure_kkt_residual_norm",
                    "constant_pressure_kkt_relative_distance",
                    "iterations",
                    "functional_evaluations",
                    "finite_difference_fourth_order_components",
                    "analytic_derivative_evaluations",
                    "derivative_resolution_step_acceptances",
                    "topology_change_rejections",
                    "topology_epoch_transitions",
                    "max_topology_epoch_transitions",
                    "constraint_change_rejections",
                    "qualification"):
                value = latest_static_capillary.get(source)
                if isinstance(value, (bool, int, float, str)):
                    metrics[f"static_capillary_{source}"] = value

    velocity_range = diagnostic_solution_velocity_range(diagnostics)
    if velocity_range is not None:
        metrics["diagnostic_solution_velocity_range"] = velocity_range
    pressure_range = diagnostic_solution_pressure_range(diagnostics)
    if pressure_range is not None:
        metrics["diagnostic_solution_pressure_range"] = pressure_range
    active_volume_error = diagnostic_active_volume_error(diagnostics)
    if active_volume_error is not None:
        metrics["diagnostic_active_volume_error"] = active_volume_error
    min_exact_order = diagnostic_cut_volume_min_exact_order(diagnostics)
    if min_exact_order is not None:
        metrics["diagnostic_cut_volume_min_exact_order"] = min_exact_order
    max_exact_order = diagnostic_cut_volume_max_exact_order(diagnostics)
    if max_exact_order is not None:
        metrics["diagnostic_cut_volume_max_exact_order"] = max_exact_order
    cut_adjacent_max_scale = diagnostic_cut_adjacent_max_scale(diagnostics)
    if cut_adjacent_max_scale is not None:
        metrics["diagnostic_cut_adjacent_max_scale"] = cut_adjacent_max_scale
    capped_scale_count = diagnostic_cut_adjacent_capped_scale_count(diagnostics)
    if capped_scale_count is not None:
        metrics["diagnostic_cut_adjacent_capped_scale_count"] = capped_scale_count
    pruned_volume_regions = diagnostic_active_pruned_volume_regions(diagnostics)
    if pruned_volume_regions is not None:
        metrics["diagnostic_active_pruned_volume_regions"] = pruned_volume_regions
    pruned_volume = diagnostic_active_pruned_volume(diagnostics)
    if pruned_volume is not None:
        metrics["diagnostic_active_pruned_volume"] = pruned_volume
    active_min_fraction = diagnostic_active_min_volume_fraction(diagnostics)
    if active_min_fraction is not None:
        metrics["diagnostic_active_min_volume_fraction"] = active_min_fraction
    generated_pruned_rules = diagnostic_generated_pruned_volume_rules(diagnostics)
    if generated_pruned_rules is not None:
        metrics["diagnostic_generated_pruned_volume_rules"] = generated_pruned_rules
    generated_pruned_volume = diagnostic_generated_pruned_volume(diagnostics)
    if generated_pruned_volume is not None:
        metrics["diagnostic_generated_pruned_volume"] = generated_pruned_volume
    implicit_fallback_cells = diagnostic_implicit_cut_fallback_cells(diagnostics)
    if implicit_fallback_cells is not None:
        metrics["diagnostic_implicit_cut_fallback_cells"] = implicit_fallback_cells
    for source, target in (
            ("achieved_interface_quadrature_order",
             "diagnostic_achieved_interface_quadrature_order_min"),
            ("achieved_volume_quadrature_order",
             "diagnostic_achieved_volume_quadrature_order_min")):
        value = diagnostic_cut_context_min_int(diagnostics, source)
        if value is not None:
            metrics[target] = value
    for source, target in (
            ("generated_interface_geometry",
             "diagnostic_generated_interface_geometry_counts"),
            ("implicit_cut_quadrature_backend",
             "diagnostic_implicit_cut_quadrature_backend_counts"),
            ("implicit_cut_fallback_policy",
             "diagnostic_implicit_cut_fallback_policy_counts")):
        counts = diagnostic_cut_context_value_counts(diagnostics, source)
        if counts:
            metrics[target] = top_counts(counts)
    selected_backend_counts = diagnostic_cut_context_summary_counts(
        diagnostics, "selected_implicit_cut_quadrature_backend_counts")
    if selected_backend_counts:
        metrics["diagnostic_selected_implicit_cut_quadrature_backend_counts"] = (
            top_counts(selected_backend_counts)
        )
    backend_qualification_counts = diagnostic_cut_context_summary_counts(
        diagnostics, "implicit_cut_backend_qualification_counts")
    if backend_qualification_counts:
        metrics["diagnostic_implicit_cut_backend_qualification_counts"] = (
            top_counts(backend_qualification_counts)
        )
    gauge_value = diagnostic_pressure_gauge_value(diagnostics)
    if gauge_value is not None:
        metrics["diagnostic_pressure_gauge_value"] = gauge_value
    if diagnostics.get("hydrostatic_initializations"):
        latest_hydrostatic = diagnostics["hydrostatic_initializations"][-1]
        metrics["latest_hydrostatic_initialization"] = latest_hydrostatic
        for name in (
            "wet_pressure_vertices",
            "dry_pressure_vertices",
            "gauge_constraints",
            "checked_gauge_constraints",
            "skipped_gauge_constraints",
            "initialized_pressure_min",
            "initialized_pressure_max",
            "wet_pressure_min",
            "wet_pressure_max",
            "gauge_pressure_min",
            "gauge_pressure_max",
            "gauge_initialized_pressure_min",
            "gauge_initialized_pressure_max",
            "gauge_pressure_max_abs_error",
        ):
            value = latest_hydrostatic.get(name)
            if isinstance(value, (int, float)):
                metrics[f"diagnostic_hydrostatic_{name}"] = value
    solution_source_summary = cut_context_solution_source_summary(diagnostics)
    if solution_source_summary:
        metrics["diagnostic_cut_context_solution_sources"] = solution_source_summary
    rebuild_provenance_counts = cut_context_rebuild_provenance_counts(diagnostics)
    if rebuild_provenance_counts:
        rebuild_count = sum(rebuild_provenance_counts.values())
        metrics["diagnostic_cut_context_rebuild_count"] = rebuild_count
        metrics["diagnostic_cut_context_rebuild_provenance_counts"] = (
            top_counts(rebuild_provenance_counts)
        )
        nonlinear_refresh_count = sum(
            count for provenance, count in rebuild_provenance_counts.items()
            if provenance in STATE_SYNC_CUT_CONTEXT_PROVENANCES
        )
        vector_refresh_count = sum(
            count for provenance, count in rebuild_provenance_counts.items()
            if provenance in VECTOR_CUT_CONTEXT_PROVENANCES
        )
        metrics["diagnostic_cut_context_nonlinear_refresh_count"] = (
            nonlinear_refresh_count
        )
        metrics["diagnostic_cut_context_vector_refresh_count"] = vector_refresh_count
    skip_provenance_counts = cut_context_refresh_skip_provenance_counts(diagnostics)
    if skip_provenance_counts:
        skip_count = sum(skip_provenance_counts.values())
        metrics["diagnostic_cut_context_refresh_skip_count"] = skip_count
        metrics["diagnostic_cut_context_refresh_skip_provenance_counts"] = (
            top_counts(skip_provenance_counts)
        )
    cell_cache_summary = generated_cell_cache_summary(diagnostics)
    if cell_cache_summary:
        metrics["diagnostic_generated_cell_cache_summary"] = cell_cache_summary
        metrics["diagnostic_generated_cell_cache_total_hits"] = (
            cell_cache_summary["total_hits"]
        )
        metrics["diagnostic_generated_cell_cache_total_misses"] = (
            cell_cache_summary["total_misses"]
        )
        metrics["diagnostic_generated_cell_cache_unchanged_dof_hits"] = (
            cell_cache_summary["unchanged_dof_hits"]
        )
        metrics["diagnostic_generated_cell_refresh_candidates"] = (
            cell_cache_summary["refresh_candidates"]
        )
        metrics["diagnostic_generated_domain_cache_hits"] = (
            cell_cache_summary["domain_hits"]
        )
        metrics["diagnostic_generated_cell_cache_full_miss_rebuilds"] = (
            cell_cache_summary["full_miss_rebuilds"]
        )
    if diagnostics.get("level_set_maintenance"):
        maintenance = diagnostics["level_set_maintenance"]
        metrics["diagnostic_level_set_maintenance_count"] = len(maintenance)
        metrics["latest_level_set_maintenance"] = maintenance[-1]
    if diagnostics.get("wet_volume_diagnostics"):
        # This is the authoritative accepted-state history emitted directly
        # from the production cut context.  Do not conflate it with the VTK
        # WetVolumeMeasure time series below, which is retained to check
        # same-state serialization/output consistency.
        metrics["production_wet_volume_diagnostic_history"] = list(
            diagnostics["wet_volume_diagnostics"])
        metrics["diagnostic_wet_volume_count"] = len(
            diagnostics["wet_volume_diagnostics"])
    if diagnostics.get("level_set_advection_velocity_updates"):
        records = diagnostics["level_set_advection_velocity_updates"]
        metrics["diagnostic_level_set_advection_velocity_update_count"] = len(records)
        metrics["latest_level_set_advection_velocity_update"] = records[-1]
        for source, target in (
                ("extension_method",
                 "diagnostic_level_set_advection_velocity_extension_method_counts"),
                ("interface_sample_source",
                 "diagnostic_level_set_advection_velocity_interface_sample_source_counts")):
            counts: dict[str, int] = {}
            for record in records:
                value = record.get(source)
                if value is not None:
                    increment_count(counts, str(value))
            if counts:
                metrics[target] = top_counts(counts)
        for source, target in (
                ("interface_sample_candidates",
                 "diagnostic_level_set_advection_velocity_max_interface_sample_candidates"),
                ("interface_samples",
                 "diagnostic_level_set_advection_velocity_max_interface_samples")):
            values = [
                int(record[source])
                for record in records
                if isinstance(record.get(source), int)
            ]
            if values:
                metrics[target] = max(values)
    if diagnostics.get("level_set_nonconservative_warnings"):
        warnings = diagnostics["level_set_nonconservative_warnings"]
        metrics["diagnostic_level_set_nonconservative_warning_count"] = len(warnings)
        metrics["latest_level_set_nonconservative_warning"] = warnings[-1]
    if diagnostics.get("fsils_true_residuals"):
        latest_true_residual = diagnostics["fsils_true_residuals"][-1]
        metrics["latest_fsils_true_residual"] = latest_true_residual
        for name in (
            "constraint_solution_mean",
            "constraint_solution_rms",
            "constraint_solution_fluctuation_rms",
            "constraint_solution_mean_dominance",
            "constraint_residual_mean",
            "constraint_residual_rms",
        ):
            value = latest_true_residual.get(name)
            if isinstance(value, (int, float)):
                metrics[f"latest_fsils_{name}"] = value
    if diagnostics.get("fsils_solve_summaries"):
        latest_solve_summary = diagnostics["fsils_solve_summaries"][-1]
        metrics["latest_fsils_solve_summary"] = latest_solve_summary
        for name in (
            "blockschur_schur_iterations",
            "blockschur_schur_mitr",
            "blockschur_schur_rel_tol",
            "blockschur_schur_abs_tol",
            "blockschur_momentum_iterations",
            "blockschur_momentum_mitr",
            "internal_final_norm",
            "internal_success",
            "true_residual_retries",
        ):
            value = latest_solve_summary.get(name)
            if isinstance(value, (int, float)):
                metrics[f"latest_fsils_{name}"] = value
    if diagnostics.get("newton_direction_checks"):
        latest_direction_check = diagnostics["newton_direction_checks"][-1]
        metrics["latest_newton_direction_check"] = latest_direction_check
        value = latest_direction_check.get("rel")
        if isinstance(value, (int, float)):
            metrics["diagnostic_newton_direction_relative_error"] = float(value)
    if diagnostics.get("newton_assemblies"):
        all_records = diagnostics["newton_assemblies"]
        balance_records = [
            record for record in all_records
            if record.get("op") in
            FREE_SURFACE_CONSERVATIVE_BALANCE_OPERATOR_TAGS
        ]
        records = [
            record for record in all_records
            if record.get("op") not in
            FREE_SURFACE_CONSERVATIVE_BALANCE_OPERATOR_TAGS
        ]
        metrics[
            "diagnostic_free_surface_conservative_balance_newton_assembly_count"
        ] = len(balance_records)
        metrics[
            "diagnostic_free_surface_conservative_balance_newton_matrix_assembly_count"
        ] = sum(bool(record.get("want_matrix")) for record in balance_records)
        metrics[
            "diagnostic_free_surface_conservative_balance_newton_vector_assembly_count"
        ] = sum(bool(record.get("want_vector")) for record in balance_records)
        representability_assembly_records = [
            record for record in balance_records
            if record.get("op") ==
            FREE_SURFACE_PRESSURE_REPRESENTABILITY_OPERATOR_TAG
        ]
        metrics[
            "diagnostic_free_surface_pressure_representability_newton_assembly_count"
        ] = len(representability_assembly_records)
        metrics[
            "diagnostic_free_surface_pressure_representability_newton_matrix_assembly_count"
        ] = sum(
            bool(record.get("want_matrix"))
            for record in representability_assembly_records
        )
        if balance_records:
            metrics[
                "latest_free_surface_conservative_balance_newton_assembly"
            ] = balance_records[-1]
        if records:
            metrics["latest_newton_assembly"] = records[-1]
        # Preserve the legacy production assembly-efficiency contract.  The
        # three vector virtual-work samplers and the matrix-only pressure-
        # representability pairing are reported separately above and must not
        # look like extra nonlinear production assemblies.
        metrics["diagnostic_newton_assembly_count"] = len(records)
        phase_counts: dict[str, int] = {}
        sync_point_counts: dict[str, int] = {}
        matrix_count = 0
        vector_count = 0
        post_first_iteration_matrix_count = 0
        for record in records:
            increment_count(
                phase_counts, str(record.get("phase", "unknown"))
            )
            increment_count(
                sync_point_counts, str(record.get("sync_point", "unknown"))
            )
            want_matrix = bool(record.get("want_matrix"))
            want_vector = bool(record.get("want_vector"))
            if want_matrix:
                matrix_count += 1
                iteration = record.get("iteration")
                if isinstance(iteration, (int, float)) and int(iteration) > 0:
                    post_first_iteration_matrix_count += 1
            if want_vector:
                vector_count += 1
        metrics["diagnostic_newton_assembly_phase_counts"] = (
            top_counts(phase_counts)
        )
        metrics["diagnostic_newton_assembly_sync_point_counts"] = (
            top_counts(sync_point_counts)
        )
        metrics["diagnostic_newton_matrix_assembly_count"] = matrix_count
        metrics["diagnostic_newton_vector_assembly_count"] = vector_count
        metrics[
            "diagnostic_newton_post_first_iteration_matrix_assembly_count"
        ] = post_first_iteration_matrix_count
    if diagnostics.get("jacobian_checks"):
        latest_jacobian_check = diagnostics["jacobian_checks"][-1]
        metrics["latest_jacobian_check"] = latest_jacobian_check
        value = latest_jacobian_check.get("rel")
        if isinstance(value, (int, float)):
            metrics["diagnostic_jacobian_check_relative_error"] = float(value)
    if diagnostics.get("jacobian_check_component_details"):
        details = diagnostics["jacobian_check_component_details"]
        metrics["latest_jacobian_check_component_details"] = details[-1]
        block_metrics = jacobian_component_block_metrics(details)
        filters = block_metrics.get("filters")
        if isinstance(filters, list):
            metrics["diagnostic_jacobian_component_sweep_filters"] = filters
            metrics["diagnostic_jacobian_component_sweep_count"] = len(filters)
        if "relative_errors" in block_metrics:
            metrics["diagnostic_jacobian_component_block_relative_errors"] = (
                block_metrics["relative_errors"]
            )
        if "matrix_relative_errors" in block_metrics:
            metrics["diagnostic_jacobian_component_block_matrix_relative_errors"] = (
                block_metrics["matrix_relative_errors"]
            )
        for source, target in (
                ("max_relative_error",
                 "diagnostic_jacobian_component_block_max_relative_error"),
                ("max_matrix_relative_error",
                 "diagnostic_jacobian_component_block_max_matrix_relative_error"),
                ("skipped_near_zero_blocks",
                 "diagnostic_jacobian_component_block_skipped_near_zero_count")):
            value = block_metrics.get(source)
            if isinstance(value, (int, float)):
                metrics[target] = value
    if diagnostics.get("jacobian_check_sweep_plans"):
        metrics["latest_jacobian_check_sweep_plan"] = (
            diagnostics["jacobian_check_sweep_plans"][-1]
        )
    if diagnostics.get("jacobian_check_top_mismatches"):
        metrics["latest_jacobian_check_top_mismatch"] = (
            diagnostics["jacobian_check_top_mismatches"][-1]
        )
    if diagnostics.get("form_mixed_plans"):
        metrics["latest_form_mixed_plan"] = diagnostics["form_mixed_plans"][-1]
    if diagnostics.get("form_block_installs"):
        metrics["form_block_install_count"] = len(diagnostics["form_block_installs"])
    if diagnostics.get("linear_solve_histories"):
        metrics["latest_linear_solve_history"] = diagnostics["linear_solve_histories"][-1]
    if diagnostics.get("timeloop_initialization_solves"):
        records = diagnostics["timeloop_initialization_solves"]
        metrics["latest_timeloop_initialization_solve"] = records[-1]
        metrics["diagnostic_timeloop_initialization_solve_count"] = len(records)
        for source, target in (
                ("dirichlet_dofs", "diagnostic_timeloop_initialization_max_dirichlet_dofs"),
                ("constraints", "diagnostic_timeloop_initialization_max_constraints"),
                ("rhs_norm", "diagnostic_timeloop_initialization_max_rhs_norm")):
            values = [
                record.get(source)
                for record in records
                if isinstance(record.get(source), (int, float))
            ]
            if values:
                metrics[target] = max(values)
    if diagnostics.get("fsils_prepared_matrices"):
        records = diagnostics["fsils_prepared_matrices"]
        latest_matrix = records[-1]
        metrics["latest_fsils_prepared_matrix"] = latest_matrix
        metrics["diagnostic_fsils_prepared_matrix_count"] = len(records)
        for source, target in (
                ("zero_rows", "diagnostic_fsils_prepared_matrix_max_zero_rows"),
                ("missing_diag", "diagnostic_fsils_prepared_matrix_max_missing_diag"),
                ("diag_col_mismatch",
                 "diagnostic_fsils_prepared_matrix_max_diag_col_mismatch"),
                ("duplicate_diag_entries",
                 "diagnostic_fsils_prepared_matrix_max_duplicate_diag_entries"),
                ("duplicate_diag_rows",
                 "diagnostic_fsils_prepared_matrix_max_duplicate_diag_rows"),
                ("zero_diag", "diagnostic_fsils_prepared_matrix_max_zero_diag"),
                ("nonfinite_entries",
                 "diagnostic_fsils_prepared_matrix_max_nonfinite_entries"),
                ("max_row_sum_to_abs_diag",
                 "diagnostic_fsils_prepared_matrix_max_row_sum_to_abs_diag")):
            values = [
                record.get(source)
                for record in records
                if isinstance(record.get(source), (int, float))
            ]
            if values:
                metrics[target] = max(values)
        values = [
            record.get("min_abs_diag_to_row_sum")
            for record in records
            if isinstance(record.get("min_abs_diag_to_row_sum"), (int, float))
        ]
        if values:
            metrics["diagnostic_fsils_prepared_matrix_min_abs_diag_to_row_sum"] = min(values)
    if diagnostics.get("eigen_factorization_diagnostics"):
        records = diagnostics["eigen_factorization_diagnostics"]
        latest_eigen = records[-1]
        metrics["latest_eigen_factorization_diagnostic"] = latest_eigen
        metrics["diagnostic_eigen_factorization_count"] = len(records)
        for source, target in (
                ("zero_rows", "diagnostic_eigen_factorization_max_zero_rows"),
                ("zero_cols", "diagnostic_eigen_factorization_max_zero_cols"),
                ("nonfinite_entries", "diagnostic_eigen_factorization_max_nonfinite_entries"),
                ("pressure_zero_rows", "diagnostic_eigen_factorization_max_pressure_zero_rows"),
                ("pressure_zero_cols", "diagnostic_eigen_factorization_max_pressure_zero_cols")):
            values = [
                record.get(source)
                for record in records
                if isinstance(record.get(source), (int, float))
            ]
            if values:
                metrics[target] = max(values)
        for key in ("zero_row_runs", "zero_col_runs",
                    "pressure_zero_row_runs_local",
                    "pressure_zero_col_runs_local"):
            value = latest_eigen.get(key)
            if isinstance(value, str):
                metrics[f"diagnostic_eigen_factorization_latest_{key}"] = value
    if diagnostics.get("active_pressure_support_constraints"):
        records = diagnostics["active_pressure_support_constraints"]
        latest_support = records[-1]
        metrics["latest_active_pressure_support_constraint"] = latest_support
        metrics["diagnostic_active_pressure_support_constraint_count"] = len(records)
        for source, target in (
                ("active_support_cells",
                 "diagnostic_active_pressure_support_max_active_support_cells"),
                ("active_support_vertices",
                 "diagnostic_active_pressure_support_max_active_support_vertices"),
                ("inactive_vertices",
                 "diagnostic_active_pressure_support_max_inactive_vertices"),
                ("constrained_owned_dofs",
                 "diagnostic_active_pressure_support_max_constrained_owned_dofs"),
                ("inactive_sign_vertices_with_support",
                 "diagnostic_active_pressure_support_max_inactive_sign_vertices_with_support"),
                ("active_sign_vertices_without_support",
                 "diagnostic_active_pressure_support_max_active_sign_vertices_without_support")):
            values = [
                record.get(source)
                for record in records
                if isinstance(record.get(source), (int, float))
            ]
            if values:
                metrics[target] = max(values)
        value = latest_support.get("inactive_vertex_runs")
        if isinstance(value, str):
            metrics["diagnostic_active_pressure_support_latest_inactive_vertex_runs"] = value
    if diagnostics.get("curvature_projections"):
        records = diagnostics["curvature_projections"]
        metrics["latest_curvature_projection"] = records[-1]
        metrics["diagnostic_curvature_projection_count"] = len(records)
        field_counts: dict[str, int] = {}
        for record in records:
            increment_count(
                field_counts,
                str(record.get("curvature_field", "unknown")),
            )
        metrics["diagnostic_curvature_projection_field_counts"] = (
            top_counts(field_counts)
        )
        reason_counts: dict[str, int] = {}
        for record in records:
            increment_count(
                reason_counts,
                str(record.get("reason", "unknown")),
            )
        metrics["diagnostic_curvature_projection_reason_counts"] = (
            top_counts(reason_counts)
        )
        smoothing_mode_counts: dict[str, int] = {}
        for record in records:
            increment_count(
                smoothing_mode_counts,
                str(record.get("smoothing_mode", "unknown")),
            )
        metrics["diagnostic_curvature_projection_smoothing_mode_counts"] = (
            top_counts(smoothing_mode_counts)
        )
        recovery_mode_counts: dict[str, int] = {}
        for record in records:
            increment_count(
                recovery_mode_counts,
                str(record.get("recovery_mode", "unknown")),
            )
        metrics["diagnostic_curvature_projection_recovery_mode_counts"] = (
            top_counts(recovery_mode_counts)
        )
        cache_counts: dict[str, int] = {}
        cut_signature_cache_counts: dict[str, int] = {}
        skipped_count = 0
        cache_hit_count = 0
        cache_miss_count = 0
        cut_signature_cache_hit_count = 0
        cut_signature_cache_miss_count = 0
        reused_vertex_adjacency_count = 0
        reused_sample_adjacency_count = 0
        for record in records:
            cache_state = str(record.get("cache", "unknown"))
            increment_count(cache_counts, cache_state)
            cut_signature_cache_state = str(
                record.get("cut_signature_cache", "unknown"))
            increment_count(cut_signature_cache_counts,
                            cut_signature_cache_state)
            if diagnostic_flag(record, "projection_skipped"):
                skipped_count += 1
            if cache_state in ("hit", "fast_hit"):
                cache_hit_count += 1
            if cache_state == "miss":
                cache_miss_count += 1
            if cut_signature_cache_state == "hit":
                cut_signature_cache_hit_count += 1
            if cut_signature_cache_state == "miss":
                cut_signature_cache_miss_count += 1
            if diagnostic_flag(record, "reused_vertex_adjacency"):
                reused_vertex_adjacency_count += 1
            if diagnostic_flag(record, "reused_sample_adjacency"):
                reused_sample_adjacency_count += 1
        metrics["diagnostic_curvature_projection_cache_counts"] = (
            top_counts(cache_counts)
        )
        metrics["diagnostic_curvature_projection_skipped_count"] = skipped_count
        metrics["diagnostic_curvature_projection_cache_hit_count"] = cache_hit_count
        metrics["diagnostic_curvature_projection_cache_miss_count"] = cache_miss_count
        metrics["diagnostic_curvature_projection_cut_signature_cache_counts"] = (
            top_counts(cut_signature_cache_counts)
        )
        metrics["diagnostic_curvature_projection_cut_signature_cache_hit_count"] = (
            cut_signature_cache_hit_count
        )
        metrics["diagnostic_curvature_projection_cut_signature_cache_miss_count"] = (
            cut_signature_cache_miss_count
        )
        metrics["diagnostic_curvature_projection_reused_vertex_adjacency_count"] = (
            reused_vertex_adjacency_count
        )
        metrics["diagnostic_curvature_projection_reused_sample_adjacency_count"] = (
            reused_sample_adjacency_count
        )
        for source, target in (
                ("fitted_vertices",
                 "diagnostic_curvature_projection_max_fitted_vertices"),
                ("generated_interface_geometry_samples",
                 "diagnostic_curvature_projection_max_interface_geometry_samples"),
                ("generated_interface_patch_fitted_vertices",
                 "diagnostic_curvature_projection_max_interface_patch_fitted_vertices"),
                ("generated_interface_patch_expanded_vertices",
                 "diagnostic_curvature_projection_max_interface_patch_expanded_vertices"),
                ("fallback_vertices",
                 "diagnostic_curvature_projection_max_fallback_vertices"),
                ("zero_fallback_vertices",
                 "diagnostic_curvature_projection_max_zero_fallback_vertices"),
                ("insufficient_stencil_vertices",
                 "diagnostic_curvature_projection_max_insufficient_stencil_vertices"),
                ("singular_stencil_vertices",
                 "diagnostic_curvature_projection_max_singular_stencil_vertices"),
                ("small_gradient_vertices",
                 "diagnostic_curvature_projection_max_small_gradient_vertices"),
                ("fit_residual_failure_vertices",
                 "diagnostic_curvature_projection_max_fit_residual_failure_vertices"),
                ("narrow_band_width",
                 "diagnostic_curvature_projection_max_narrow_band_width"),
                ("narrow_band_vertices",
                 "diagnostic_curvature_projection_max_narrow_band_vertices"),
                ("skipped_far_vertices",
                 "diagnostic_curvature_projection_max_skipped_far_vertices"),
                ("smoothing_iterations",
                 "diagnostic_curvature_projection_max_smoothing_iterations"),
                ("smoothing_operator_edges",
                 "diagnostic_curvature_projection_max_smoothing_operator_edges"),
                ("smoothing_mean_abs_update",
                 "diagnostic_curvature_projection_max_smoothing_mean_abs_update"),
                ("smoothing_max_abs_update",
                 "diagnostic_curvature_projection_max_smoothing_max_abs_update"),
                ("mean_normalized_fit_residual",
                 "diagnostic_curvature_projection_max_mean_normalized_fit_residual"),
                ("max_normalized_fit_residual",
                 "diagnostic_curvature_projection_max_normalized_fit_residual"),
                ("vertex_adjacency_builds",
                 "diagnostic_curvature_projection_max_vertex_adjacency_builds"),
                ("sample_adjacency_builds",
                 "diagnostic_curvature_projection_max_sample_adjacency_builds"),
                ("max_abs_curvature",
                 "diagnostic_curvature_projection_max_abs_curvature")):
            values = [
                record.get(source)
                for record in records
                if isinstance(record.get(source), (int, float))
            ]
            if values:
                metrics[target] = max(values)
    if diagnostics.get("level_set_volume_corrections"):
        records = diagnostics["level_set_volume_corrections"]
        metrics["latest_level_set_volume_correction"] = records[-1]
        metrics["diagnostic_level_set_volume_correction_count"] = len(records)
        metrics["level_set_mass_correction_history"] = [
            {
                "step": record.get("step"),
                "uncorrected_volume": record.get("initial_negative_volume"),
                "corrected_volume": record.get("corrected_negative_volume"),
                "target_volume": record.get("target_negative_volume"),
                "uncorrected_error": record.get("initial_volume_error"),
                "corrected_error": record.get("achieved_volume_error"),
                "applied_level_set_shift": record.get("applied_shift"),
                "cumulative_interface_displacement": record.get(
                    "cumulative_interface_displacement"),
                "cumulative_contact_line_displacement": record.get(
                    "cumulative_contact_line_displacement"),
                "maximum_cumulative_interface_displacement": record.get(
                    "maximum_cumulative_interface_displacement"),
                "volume_measure_source": record.get("volume_measure_source"),
            }
            for record in records
        ]
        for source, target in (
                ("achieved_volume_error",
                 "diagnostic_level_set_volume_correction_max_abs_achieved_error"),
                ("applied_shift_magnitude",
                 "diagnostic_level_set_volume_correction_max_shift_magnitude"),
                ("cumulative_interface_displacement",
                 "diagnostic_level_set_volume_correction_max_cumulative_interface_displacement"),
                ("cumulative_contact_line_displacement",
                 "diagnostic_level_set_volume_correction_max_cumulative_contact_line_displacement"),
                ("iterations",
                 "diagnostic_level_set_volume_correction_max_iterations")):
            values = [
                abs(record.get(source)) if source == "achieved_volume_error"
                else record.get(source)
                for record in records
                if isinstance(record.get(source), (int, float))
            ]
            if values:
                metrics[target] = max(values)
    if diagnostics.get("jit_specialization_traces"):
        traces = diagnostics["jit_specialization_traces"]
        metrics["latest_jit_specialization_trace"] = traces[-1]
        event_counts: dict[str, int] = {}
        trigger_counts: dict[str, int] = {}
        event_trigger_domain_role_counts: dict[str, int] = {}
        event_reason_domain_counts: dict[str, int] = {}
        compile_domain_role_counts: dict[str, int] = {}
        runtime_compile_domain_role_counts: dict[str, int] = {}
        runtime_skip_reason_domain_counts: dict[str, int] = {}
        compile_shape_counts: dict[str, int] = {}
        generic_compile_kind_counts: dict[str, int] = {}
        for record in traces:
            event = record.get("event")
            if isinstance(event, str):
                event_counts[event] = event_counts.get(event, 0) + 1
            trigger = record.get("trigger")
            if isinstance(trigger, str):
                trigger_counts[trigger] = trigger_counts.get(trigger, 0) + 1
            increment_count(
                event_trigger_domain_role_counts,
                count_key(record, ("event", "trigger", "domain", "role")),
            )
            increment_count(
                event_reason_domain_counts,
                count_key(record, ("event", "reason", "domain")),
            )
            if event == "compile":
                increment_count(
                    compile_domain_role_counts,
                    count_key(record, ("domain", "role")),
                )
                increment_count(compile_shape_counts, jit_shape_key(record))
                if trigger == "runtime":
                    increment_count(
                        runtime_compile_domain_role_counts,
                        count_key(record, ("domain", "role")),
                    )
            elif event == "generic_compile":
                increment_count(generic_compile_kind_counts, count_key(record, ("kind",)))
            elif event == "runtime_skip":
                increment_count(
                    runtime_skip_reason_domain_counts,
                    count_key(record, ("reason", "domain")),
                )
        metrics["diagnostic_jit_specialization_trace_count"] = len(traces)
        metrics["diagnostic_jit_specialization_event_counts"] = event_counts
        metrics["diagnostic_jit_specialization_trigger_counts"] = trigger_counts
        metrics["diagnostic_jit_specialization_event_trigger_domain_role_counts"] = (
            top_counts(event_trigger_domain_role_counts)
        )
        metrics["diagnostic_jit_specialization_event_reason_domain_counts"] = (
            top_counts(event_reason_domain_counts)
        )
        metrics["diagnostic_jit_specialization_compile_domain_role_counts"] = (
            top_counts(compile_domain_role_counts)
        )
        metrics["diagnostic_jit_specialization_runtime_compile_domain_role_counts"] = (
            top_counts(runtime_compile_domain_role_counts)
        )
        metrics["diagnostic_jit_specialization_runtime_skip_reason_domain_counts"] = (
            top_counts(runtime_skip_reason_domain_counts)
        )
        metrics["diagnostic_jit_specialization_compile_shape_counts"] = (
            top_counts(compile_shape_counts)
        )
        metrics["diagnostic_jit_specialization_generic_compile_kind_counts"] = (
            top_counts(generic_compile_kind_counts)
        )
        metrics["diagnostic_jit_specialization_compile_count"] = sum(
            count for event, count in event_counts.items()
            if event in {"compile", "generic_compile"}
        )
    if diagnostics.get("assembly_timings"):
        all_timings = diagnostics["assembly_timings"]
        balance_timings = [
            record for record in all_timings
            if record.get("op") in
            FREE_SURFACE_CONSERVATIVE_BALANCE_OPERATOR_TAGS
        ]
        timings = [
            record for record in all_timings
            if record.get("op") not in
            FREE_SURFACE_CONSERVATIVE_BALANCE_OPERATOR_TAGS
        ]
        metrics[
            "diagnostic_free_surface_conservative_balance_assembly_timing_count"
        ] = len(balance_timings)
        representability_timings = [
            record for record in balance_timings
            if record.get("op") ==
            FREE_SURFACE_PRESSURE_REPRESENTABILITY_OPERATOR_TAG
        ]
        metrics[
            "diagnostic_free_surface_pressure_representability_assembly_timing_count"
        ] = len(representability_timings)
        if balance_timings:
            metrics[
                "latest_free_surface_conservative_balance_assembly_timing"
            ] = balance_timings[-1]
        metrics["diagnostic_assembly_timing_count"] = len(timings)
        if timings:
            metrics["latest_assembly_timing"] = timings[-1]
        for name in (
            "total",
            "cell_terms",
            "boundary_terms",
            "other_dg_global",
            "interior_faces",
            "interface_faces",
            "cut_volumes",
            "global_terms",
        ):
            values = [
                float(record[name])
                for record in timings
                if isinstance(record.get(name), (int, float))
            ]
            if values:
                metrics[f"diagnostic_assembly_timing_max_{name}_seconds"] = max(values)
    summary = diagnostics.get("time_loop", {}).get("summary", {})
    if isinstance(summary, dict):
        accepted_steps = summary.get("accepted_steps")
        nonlinear_iterations = summary.get("nonlinear_iterations_total")
        assembly_count = metrics.get("diagnostic_assembly_timing_count")
        cut_rebuild_count = metrics.get("diagnostic_cut_context_rebuild_count")
        newton_assembly_count = metrics.get("diagnostic_newton_assembly_count")
        newton_matrix_count = metrics.get("diagnostic_newton_matrix_assembly_count")
        if isinstance(accepted_steps, (int, float)) and accepted_steps > 0:
            if isinstance(assembly_count, (int, float)):
                metrics["diagnostic_assembly_timings_per_accepted_step"] = (
                    float(assembly_count) / float(accepted_steps)
                )
            if isinstance(newton_assembly_count, (int, float)):
                metrics["diagnostic_newton_assemblies_per_accepted_step"] = (
                    float(newton_assembly_count) / float(accepted_steps)
                )
            if isinstance(newton_matrix_count, (int, float)):
                metrics["diagnostic_newton_matrix_assemblies_per_accepted_step"] = (
                    float(newton_matrix_count) / float(accepted_steps)
                )
            if isinstance(cut_rebuild_count, (int, float)):
                metrics["diagnostic_cut_context_rebuilds_per_accepted_step"] = (
                    float(cut_rebuild_count) / float(accepted_steps)
                )
            if (isinstance(assembly_count, (int, float)) and
                    isinstance(nonlinear_iterations, (int, float))):
                extra_assemblies = int(assembly_count) - int(nonlinear_iterations)
                metrics["diagnostic_extra_assembly_timing_count_vs_nonlinear_iterations"] = (
                    extra_assemblies
                )
                metrics["diagnostic_extra_assembly_timings_per_accepted_step"] = (
                    float(extra_assemblies) / float(accepted_steps)
                )
    process_memory_records = list(diagnostics.get("process_memory", []))
    if process_memory_records:
        metrics["latest_process_memory"] = process_memory_records[-1]
        rss_values = [
            float(record["process_rss_kb"])
            for record in process_memory_records
            if isinstance(record.get("process_rss_kb"), (int, float))
        ]
        vm_values = [
            float(record["process_vm_kb"])
            for record in process_memory_records
            if isinstance(record.get("process_vm_kb"), (int, float))
        ]
        if rss_values:
            metrics["diagnostic_process_rss_kb"] = numeric_range(rss_values)
            metrics["diagnostic_process_max_rss_kb"] = max(rss_values)
            metrics["diagnostic_process_rss_growth_kb"] = rss_values[-1] - rss_values[0]
        if vm_values:
            metrics["diagnostic_process_vm_kb"] = numeric_range(vm_values)
            metrics["diagnostic_process_max_vm_kb"] = max(vm_values)
        basis_cache_values = [
            int(record["basis_cache_entries"])
            for record in process_memory_records
            if isinstance(record.get("basis_cache_entries"), (int, float))
        ]
        if basis_cache_values:
            metrics["diagnostic_process_max_basis_cache_entries"] = max(basis_cache_values)
            metrics["diagnostic_process_basis_cache_entry_growth"] = (
                basis_cache_values[-1] - basis_cache_values[0]
            )
    if diagnostics.get("jit_cache_diagnostics"):
        jit_records = diagnostics["jit_cache_diagnostics"]
        metrics["latest_jit_cache_diagnostics"] = jit_records[-1]
        for name in (
            "kernel_cache_size",
            "kernel_cache_hits",
            "kernel_cache_misses",
            "kernel_cache_symbol_hits",
            "kernel_cache_stores",
            "kernel_cache_evictions",
            "object_cache_entries",
            "object_cache_notify_compiled",
            "object_cache_gets",
            "object_cache_mem_hits",
            "object_cache_disk_hits",
            "object_cache_misses",
            "object_cache_bytes_written",
            "object_cache_bytes_read",
        ):
            values = [
                float(record[name])
                for record in jit_records
                if isinstance(record.get(name), (int, float))
            ]
            if values:
                metrics[f"diagnostic_jit_cache_max_{name}"] = max(values)
    if diagnostics.get("interior_face_timings"):
        timings = diagnostics["interior_face_timings"]
        metrics["latest_interior_face_timing"] = timings[-1]
        metrics["diagnostic_interior_face_timing_by_mode"] = summarize_timing_modes(
            timings,
            ("faces_considered", "faces_assembled"),
            (
                "total",
                "kernel",
                "insert",
                "prepare_minus",
                "prepare_plus",
                "solution",
                "field",
            ),
        )
        for name in (
            "faces_considered",
            "faces_assembled",
        ):
            values = [
                int(record[name])
                for record in timings
                if isinstance(record.get(name), int)
            ]
            if values:
                metrics[f"diagnostic_interior_face_timing_max_{name}"] = max(values)
        for name in (
            "total",
            "setup",
            "filter",
            "dofs",
            "local_face",
            "align",
            "prepare_minus",
            "prepare_plus",
            "ctx",
            "cut_scale",
            "solution",
            "field",
            "material",
            "kernel",
            "orient",
            "insert",
        ):
            values = [
                float(record[name])
                for record in timings
                if isinstance(record.get(name), (int, float))
            ]
            if values:
                metrics[f"diagnostic_interior_face_timing_max_{name}_seconds"] = max(values)
    if diagnostics.get("cut_volume_timings"):
        timings = diagnostics["cut_volume_timings"]
        metrics["latest_cut_volume_timing"] = timings[-1]
        metrics["diagnostic_cut_volume_timing_by_mode"] = summarize_timing_modes(
            timings,
            ("rules_considered", "rules_assembled", "full_rules", "partial_rules", "qpts"),
            (
                "total",
                "kernel",
                "insert",
                "rule",
                "geometry",
                "basis",
                "solution",
                "field",
            ),
        )
        for name in (
            "indexed",
            "rules_considered",
            "rules_assembled",
            "full_rules",
            "partial_rules",
            "qpts",
        ):
            values = [
                int(record[name])
                for record in timings
                if isinstance(record.get(name), int)
            ]
            if values:
                metrics[f"diagnostic_cut_volume_timing_max_{name}"] = max(values)
        for name in (
            "total",
            "setup",
            "filter",
            "dofs",
            "rule",
            "geometry",
            "basis",
            "frame",
            "context",
            "jit",
            "solution",
            "field",
            "material",
            "kernel",
            "orient",
            "insert",
        ):
            values = [
                float(record[name])
                for record in timings
                if isinstance(record.get(name), (int, float))
            ]
            if values:
                metrics[f"diagnostic_cut_volume_timing_max_{name}_seconds"] = max(values)
    retry_counts = [
        int(record["true_residual_retries"])
        for record in diagnostics.get("fsils_solve_summaries", [])
        if isinstance(record.get("true_residual_retries"), int)
    ]
    retry_counts.extend(
        1 for record in diagnostics.get("fsils_blockschur_retries", [])
        if record.get("diagnostic") == "fsils_blockschur_true_residual_retry"
    )
    if retry_counts:
        metrics["diagnostic_blockschur_true_residual_retries"] = max(retry_counts)

    wet_fraction_volume = metrics.get("wet_fraction_volume")
    context_volumes = metrics.get("cut_context_active_side_physical_volumes", [])
    if isinstance(wet_fraction_volume, (int, float)) and context_volumes:
        metrics["wet_fraction_volume_drift_vs_initial_physical_cut_context"] = (
            float(wet_fraction_volume) - float(context_volumes[0])
        )


def solver_fluid_density(case_dir: Path) -> float | None:
    """Read the constant fluid density used to convert volume to true mass."""
    try:
        root = ET.parse(case_dir / "solver.xml").getroot()
        raw = fluid_equation(root).findtext("Density")
        value = float(raw) if raw is not None else math.nan
    except (OSError, ET.ParseError, ValueError):
        return None
    return value if math.isfinite(value) and value > 0.0 else None


def add_level_set_mass_correction_history_metrics(
        metrics: dict[str, Any], density: float | None) -> None:
    """Build separate pre/post-correction histories on accepted-step clocks."""
    prefix = "level_set_mass_correction"

    def unavailable(reason: str) -> None:
        metrics[f"{prefix}_history_available"] = False
        metrics[f"{prefix}_history_error"] = reason

    if (not isinstance(density, (int, float)) or isinstance(density, bool) or
            not math.isfinite(float(density)) or float(density) <= 0.0):
        unavailable("positive finite fluid density is unavailable")
        return
    density_value = float(density)

    diagnostics = metrics.get("diagnostics")
    records = (
        diagnostics.get("level_set_volume_corrections")
        if isinstance(diagnostics, dict) else None
    )
    if not isinstance(records, list) or not records or not all(
            isinstance(record, dict) for record in records):
        unavailable("level-set volume-correction diagnostics were not reported")
        return

    time_loop = metrics.get("time_loop")
    accepted = time_loop.get("accepted_steps") if isinstance(time_loop, dict) else None
    clock, clock_errors = accepted_step_clock(accepted)
    if clock_errors:
        unavailable("accepted-step clock is invalid: " + "; ".join(clock_errors))
        return
    if not clock:
        unavailable("accepted-step clock is unavailable")
        return

    physical_key = metrics.get("production_physical_liquid_volume_history_key")
    field = physical_key.get("field") if isinstance(physical_key, dict) else None
    if not isinstance(field, str) or not field:
        unavailable("production physical liquid-volume field key is unavailable")
        return
    selected = [record for record in records if record.get("field") == field]
    expected_steps = list(clock)
    selected_steps = [record.get("step") for record in selected]
    if selected_steps != expected_steps:
        unavailable(
            "expected exactly one volume-correction diagnostic for every "
            f"accepted step of field {field!r}; expected={expected_steps!r} "
            f"reported={selected_steps!r}"
        )
        return

    production_history = metrics.get(
        "production_physical_liquid_volume_history")
    if not isinstance(production_history, list):
        unavailable("validated production physical liquid-volume history is unavailable")
        return
    corrected_physical_by_step = {
        record.get("step"): record.get("physical_liquid_volume")
        for record in production_history
        if isinstance(record, dict) and record.get("step") in clock
    }
    if set(corrected_physical_by_step) != set(expected_steps):
        unavailable(
            "production physical liquid-volume history does not cover every "
            "accepted correction step"
        )
        return

    uncorrected_history: list[dict[str, Any]] = []
    corrected_history: list[dict[str, Any]] = []
    for record in selected:
        step = int(record["step"])
        time, step_dt = clock[step]
        uncorrected = record.get("initial_negative_volume")
        corrected = record.get("corrected_negative_volume")
        target = record.get("target_negative_volume")
        uncorrected_error = record.get("initial_volume_error")
        corrected_error = record.get("achieved_volume_error")
        numeric = (
            uncorrected, corrected, target,
            uncorrected_error, corrected_error,
        )
        if not all(isinstance(value, (int, float)) and
                   not isinstance(value, bool) and math.isfinite(float(value))
                   for value in numeric):
            unavailable(
                f"volume-correction record for step {step} is incomplete or non-finite")
            return
        uncorrected_volume = float(uncorrected)
        corrected_volume = float(corrected)
        target_volume = float(target)
        if min(uncorrected_volume, corrected_volume, target_volume) <= 0.0:
            unavailable(
                f"volume-correction record for step {step} has nonpositive volume")
            return
        production_volume = corrected_physical_by_step[step]
        if (not isinstance(production_volume, (int, float)) or
                isinstance(production_volume, bool) or
                not math.isfinite(float(production_volume))):
            unavailable(
                f"production corrected physical volume for step {step} is invalid")
            return
        consistency_tolerance = 1.0e-10 * max(
            1.0, abs(corrected_volume), abs(float(production_volume)))
        if abs(corrected_volume - float(production_volume)) > consistency_tolerance:
            unavailable(
                f"corrected volume at step {step} disagrees with the same-state "
                "production physical cut volume"
            )
            return
        common = {
            "step": step,
            "time": time,
            "dt": step_dt,
            "field": field,
            "density": density_value,
            "target_liquid_volume": target_volume,
            "target_liquid_mass": density_value * target_volume,
            "volume_measure_source": record.get("volume_measure_source"),
            "correction_triggered": record.get("correction_triggered"),
            "correction_applied": record.get("correction_applied"),
            "applied_level_set_shift": record.get("applied_shift"),
            "cumulative_interface_displacement": record.get(
                "cumulative_interface_displacement"),
            "cumulative_contact_line_displacement": record.get(
                "cumulative_contact_line_displacement"),
        }
        uncorrected_history.append({
            **common,
            "state_stage": "accepted_pre_volume_correction",
            "liquid_volume": uncorrected_volume,
            "liquid_mass": density_value * uncorrected_volume,
            "volume_error": float(uncorrected_error),
            "mass_error": density_value * float(uncorrected_error),
        })
        corrected_history.append({
            **common,
            "state_stage": "accepted_post_volume_correction",
            "liquid_volume": corrected_volume,
            "liquid_mass": density_value * corrected_volume,
            "volume_error": float(corrected_error),
            "mass_error": density_value * float(corrected_error),
        })

    metrics[f"{prefix}_history_available"] = True
    metrics[f"{prefix}_field"] = field
    metrics[f"{prefix}_density"] = density_value
    metrics["level_set_uncorrected_mass_history"] = uncorrected_history
    metrics["level_set_corrected_mass_history"] = corrected_history
    metrics[f"{prefix}_accepted_state_count"] = len(corrected_history)


def add_solver_control_overrides(metrics: dict[str, Any],
                                 args: argparse.Namespace) -> None:
    for name in (
        "linear_relative_tolerance",
        "linear_absolute_tolerance",
        "linear_max_iterations",
        "linear_krylov_space_dimension",
        "ns_gm_max_iterations",
        "ns_cg_max_iterations",
        "ns_gm_tolerance",
        "ns_cg_tolerance",
        "linear_solver_type",
        "linear_algebra_backend",
        "linear_preconditioner",
        "generated_interface_geometry",
        "implicit_cut_quadrature_backend",
        "implicit_cut_fallback_policy",
        "required_implicit_cut_backend_qualification",
        "implicit_cut_root_tolerance",
        "implicit_cut_max_subdivision_depth",
        "generated_interface_quadrature_order",
        "interface_quadrature_order",
        "volume_quadrature_order",
        "surface_tension",
        "level_set_active_domain",
        "capillary_force_form",
        "cut_cell_pressure_stabilization_policy",
        "prescribed_capillary_curvature",
        "wet_extension_advection_velocity_method",
        "projected_curvature_field",
        "curvature_projection_cadence_steps",
        "curvature_projection_max_normalized_fit_residual",
        "curvature_projection_max_neighbor_fallback_vertices",
        "curvature_projection_max_zero_fallback_vertices",
        "curvature_projection_supplemental_sample_weight",
        "curvature_projection_recovery_mode",
        "curvature_projection_kinematic_area_gradient_filter_coefficient",
        "curvature_projection_narrow_band_width",
        "curvature_projection_smoothing_iterations",
        "curvature_projection_smoothing_relaxation",
        "curvature_projection_smoothing_mode",
        "sessile_contact_line_model",
        "static_capillary_volume_tolerance",
        "static_capillary_projected_gradient_tolerance",
        "static_capillary_pressure_representability_max_residual_norm",
        "static_capillary_pressure_representability_max_relative_distance",
        "static_capillary_physical_equilibrium_max_residual_norm",
        "static_capillary_constant_pressure_kkt_max_residual_norm",
        "static_capillary_constant_pressure_kkt_max_relative_distance",
        "static_capillary_finite_difference_relative_step",
        "static_capillary_max_iterations",
        "static_capillary_max_topology_epoch_transitions",
        "static_capillary_limited_memory_history_size",
        "static_capillary_limited_memory_curvature_tolerance",
        "enable_level_set_reinitialization",
        "reinitialization_cadence_steps",
        "require_static_capillary_balance_qualification",
        "expect_curvature_projection_smoothing_mode",
        "expect_curvature_projection_recovery_mode",
        "min_diagnostic_curvature_projection_operator_edges",
        "min_diagnostic_curvature_projection_interface_geometry_samples",
        "min_diagnostic_curvature_projection_interface_patch_fitted_vertices",
        "max_capillary_curvature_relative_error",
        "max_capillary_pressure_jump_relative_error",
        "max_capillary_parasitic_capillary_number",
        "max_sessile_contact_angle_error_degrees",
        "max_sessile_pressure_jump_relative_error",
        "max_sessile_liquid_area_relative_error",
        "max_sessile_liquid_volume_relative_error",
        "max_sessile_base_radius_relative_error",
        "max_sessile_apex_height_relative_error",
        "max_sessile_parasitic_capillary_number",
        "max_capillary_rejected_steps",
        "max_capillary_dt_updates",
        "max_capillary_speed_per_surface_tension",
        "max_capillary_nonlinear_residual",
        "max_capillary_linear_relative_residual",
        "max_capillary_wave_frequency_relative_error",
        "max_capillary_wave_profile_relative_error",
        "max_capillary_wave_mean_offset",
        "max_capillary_wave_temporal_liquid_volume_relative_drift",
        "max_free_surface_energy_positive_step_increment_relative",
        "max_free_surface_energy_above_initial_relative",
        "max_free_surface_conservative_balance_normalized_imbalance",
        "max_free_surface_pressure_representability_relative_distance",
        "capillary_convergence_resolution_key",
        "capillary_convergence_metric",
        "min_capillary_convergence_rate",
        "min_capillary_convergence_points",
        "enable_level_set_volume_correction",
        "volume_correction_cadence_steps",
        "volume_correction_use_initial_volume",
        "volume_correction_tolerance",
        "volume_correction_max_iterations",
        "volume_correction_maximum_cumulative_interface_displacement_fraction",
        "mpi_ranks",
        "synthetic_nx",
        "synthetic_ny",
        "synthetic_nz",
        "contact_angle_degrees",
        "sessile_radius",
        "sessile_contact_wall",
        "sessile_contact_wall_3d",
        "level_set_positive_scale",
        "capillary_droplet_center_offset",
        "capillary_sphere_center_offset",
        "sessile_tangent_center_offset",
        "sessile_tangent_center_offset_3d",
        "mms_nx",
        "mms_ny",
        "max_diagnostic_implicit_cut_fallback_cells",
        "min_diagnostic_achieved_interface_quadrature_order",
        "min_diagnostic_achieved_volume_quadrature_order",
        "expect_generated_interface_geometry",
        "expect_implicit_cut_quadrature_backend",
        "expect_selected_implicit_cut_quadrature_backend",
        "expect_implicit_cut_backend_qualification",
        "expect_implicit_cut_fallback_policy",
        "expect_level_set_advection_velocity_extension_method",
        "expect_level_set_advection_velocity_interface_sample_source",
    ):
        value = getattr(args, name, None)
        if value is not None:
            metrics[name] = value
    for name in (
        "enable_jacobian_check",
        "enable_newton_direction_check",
        "enable_newton_assembly_diagnostics",
        "enable_free_surface_conservative_balance_diagnostic",
        "initialize_static_compatible_pressure",
        "initialize_discrete_static_capillary_equilibrium",
        "initialize_discrete_static_contact_geometry",
        "newton_line_search_fail_on_no_reduction",
        "defer_static_physical_gates_to_matrix",
        "disable_cut_stabilization",
        "enable_linear_solve_history",
        "enable_linear_solve_component_norms",
        "enable_fsils_matrix_diagnostics",
        "enable_form_block_diagnostics",
        "enable_interior_face_timing",
        "enable_cut_volume_timing",
        "enable_jit_specialization_trace",
        "enable_jit_cache_diagnostics",
        "require_compiled_cut_volume_jit",
        "require_cut_context_solution_source_diagnostics",
        "require_newton_assembly_diagnostics",
        "require_assembly_timing_diagnostics",
        "require_interior_face_timing_diagnostics",
        "require_cut_volume_timing_diagnostics",
        "require_jit_specialization_trace_diagnostics",
        "require_process_memory_diagnostics",
        "require_basis_cache_diagnostics",
        "require_marked_interior_face_fallback_diagnostics",
        "require_jacobian_component_block_diagnostics",
        "require_eigen_factorization_diagnostics",
        "require_active_pressure_support_diagnostics",
        "require_level_set_advection_velocity_diagnostics",
        "require_curvature_projection_diagnostics",
        "require_curvature_projection_newton_freshness",
        "require_fsils_matrix_diagnostics",
        "require_assembly_topology_consistency",
        "require_high_order_cut_context_diagnostics",
        "high_order_production_qualification",
        "high_order_mpi_production_qualification",
        "high_order_3d_benchmark_smoke",
        "high_order_3d_benchmark_qualification",
        "high_order_3d_benchmark_profile_qualification",
        "high_order_curved_3d_simplex_smoke",
        "high_order_mpi_motion_smoke",
        "high_order_capillary_droplet_equilibrium_smoke",
        "high_order_capillary_wave_smoke",
        "use_high_order_implicit_cuts",
        "require_reference_profile_comparison",
        "require_free_surface_energy_history",
        "require_free_surface_conservative_balance",
        "require_free_surface_pressure_representability_diagnostic",
        "enable_adaptive_time_loop",
        "allow_experimental_profile_linear_solver",
        "allow_failure_diagnostics",
        "trace_level_set_advection_velocity",
    ):
        if getattr(args, name, False):
            metrics[name] = True
    for name in (
        "jacobian_check_iteration",
        "jacobian_check_step",
        "jacobian_check_scheme",
        "jacobian_check_components",
        "jacobian_check_component_sweeps",
        "linear_solve_history_max_calls",
        "linear_solve_component_norms_max_newton_it",
        "newton_line_search_max_iterations",
        "max_diagnostic_assembly_timings_per_step",
        "max_diagnostic_extra_assembly_timings_per_step",
        "max_diagnostic_cut_context_rebuilds_per_step",
        "min_diagnostic_cut_context_refresh_skips",
        "max_diagnostic_newton_matrix_assemblies_per_step",
        "max_diagnostic_generated_cell_cache_full_miss_rebuilds",
        "max_diagnostic_process_rss_kb",
        "max_diagnostic_process_rss_growth_kb",
        "max_diagnostic_process_basis_cache_entries",
        "max_diagnostic_process_basis_cache_entry_growth",
        "max_wet_fraction_volume_error",
        "max_reference_profile_rmse",
        "max_reference_profile_mae",
        "max_reference_profile_max_abs_error",
        "max_reference_profile_elevated_front_lag",
        "max_solver_elapsed_wall_seconds",
        "curvature_projection_max_normalized_fit_residual",
        "curvature_projection_max_neighbor_fallback_vertices",
        "curvature_projection_max_zero_fallback_vertices",
        "curvature_projection_supplemental_sample_weight",
        "curvature_projection_recovery_mode",
        "curvature_projection_smoothing_iterations",
        "curvature_projection_smoothing_relaxation",
        "expect_curvature_projection_smoothing_mode",
        "expect_curvature_projection_recovery_mode",
        "min_diagnostic_curvature_projection_operator_edges",
        "min_diagnostic_curvature_projection_interface_geometry_samples",
        "min_diagnostic_curvature_projection_interface_patch_fitted_vertices",
        "max_capillary_curvature_relative_error",
        "max_capillary_pressure_jump_relative_error",
        "max_capillary_rejected_steps",
        "max_capillary_dt_updates",
        "max_capillary_speed_per_surface_tension",
        "max_capillary_nonlinear_residual",
        "max_capillary_linear_relative_residual",
        "max_capillary_wave_frequency_relative_error",
        "max_capillary_wave_profile_relative_error",
        "max_capillary_wave_mean_offset",
        "max_capillary_wave_temporal_liquid_volume_relative_drift",
        "min_capillary_convergence_rate",
        "min_capillary_convergence_points",
        "min_diagnostic_curvature_projection_count",
        "min_diagnostic_curvature_projection_max_abs_curvature",
        "max_diagnostic_curvature_projection_fallback_vertices",
        "max_diagnostic_curvature_projection_zero_fallback_vertices",
        "max_diagnostic_curvature_projection_normalized_fit_residual",
        "min_diagnostic_curvature_projection_smoothing_iterations",
        "min_diagnostic_curvature_projection_skipped_count",
        "min_diagnostic_curvature_projection_cache_hit_count",
        "max_diagnostic_curvature_projection_cache_miss_count",
        "min_diagnostic_curvature_projection_cut_signature_cache_hit_count",
        "min_diagnostic_curvature_projection_reused_vertex_adjacency_count",
        "min_diagnostic_curvature_projection_reused_sample_adjacency_count",
        "max_diagnostic_curvature_projection_vertex_adjacency_builds",
        "max_diagnostic_curvature_projection_sample_adjacency_builds",
        "min_diagnostic_level_set_advection_interface_samples",
        "max_solver_elapsed_seconds_per_accepted_step",
        "min_reference_profile_coverage",
        "min_reference_profile_direct_coverage",
        "max_fsils_matrix_zero_rows",
        "max_fsils_matrix_missing_diag",
        "max_fsils_matrix_diag_col_mismatch",
        "max_fsils_matrix_duplicate_diag_entries",
        "max_fsils_matrix_duplicate_diag_rows",
        "max_fsils_matrix_zero_diag",
        "max_fsils_matrix_nonfinite_entries",
        "max_eigen_factorization_pressure_zero_cols",
        "max_time_loop_nonlinear_iterations_per_step",
        "max_time_loop_linear_iterations_per_step",
        "max_time_loop_outer_iterations_per_step",
        "max_time_loop_inner_iterations_total_per_step",
        "min_interface_height_change",
        "min_interface_mean_abs_height_change",
        "min_interface_slope_change",
        "min_interface_final_height_span",
        "cut_cell_velocity_gradient_penalty",
        "cut_cell_pressure_gradient_penalty",
        "reference_profile_sample_radius",
        "reference_profile_elevated_front_clearance",
        "adaptive_time_loop_min_dt",
        "adaptive_time_loop_max_dt",
        "adaptive_time_loop_max_retries",
        "adaptive_time_loop_decrease_factor",
        "adaptive_time_loop_increase_factor",
        "adaptive_time_loop_target_newton_iterations",
        "adaptive_time_loop_max_steps_multiplier",
    ):
        value = getattr(args, name)
        if value is not None:
            metrics[name] = value


def parse_active_volume_history_from_diagnostics(diagnostics: dict[str, Any]) -> dict[str, Any]:
    metrics = {
        key: diagnostics[key]
        for key in (
            "cut_context_active_side_volumes",
            "assembly_active_wet_volumes",
            "cut_context_active_side_volume_change",
            "assembly_active_wet_volume_change",
        )
        if key in diagnostics
    }
    physical_volumes = diagnostic_context_active_side_physical_volumes(diagnostics)
    if physical_volumes:
        metrics["cut_context_active_side_physical_volumes"] = physical_volumes
        metrics["cut_context_active_side_physical_volume_change"] = value_span(
            physical_volumes
        )
    return metrics


def previous_invalid_pressure(benchmark: dict[str, Any]) -> float | None:
    verification = benchmark.get("pressure_gauge_verification")
    if not isinstance(verification, dict):
        return None
    value = verification.get("previous_invalid_d18_full_volume_hydrostatic_pressure")
    if isinstance(value, (int, float)):
        return float(value)
    return None


def diagnostic_timeout_metrics(case_name: str,
                               run_dir: Path,
                               diagnostics: dict[str, Any]) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "case": case_name,
        "run_dir": str(run_dir),
        "timed_out": True,
    }
    add_diagnostic_metrics(metrics, diagnostics)

    previous = previous_invalid_pressure(load_benchmark(run_dir))
    if previous is not None:
        metrics["pressure_gauge_previous_invalid"] = previous
        gauge_value = metrics.get("diagnostic_pressure_gauge_value")
        if gauge_value is not None:
            metrics["diagnostic_pressure_gauge_previous_invalid_difference"] = (
                gauge_value - previous
            )
    return metrics


def accepted_step_count_for_elapsed_budget(metrics: dict[str, Any]) -> int | None:
    time_loop = metrics.get("time_loop")
    if not isinstance(time_loop, dict):
        diagnostics = metrics.get("diagnostics", {})
        if isinstance(diagnostics, dict):
            time_loop = diagnostics.get("time_loop", {})
    summary = time_loop.get("summary") if isinstance(time_loop, dict) else None
    if isinstance(summary, dict):
        accepted_steps = summary.get("accepted_steps")
        if isinstance(accepted_steps, int) and accepted_steps > 0:
            return accepted_steps
    result_step = metrics.get("result_step")
    if isinstance(result_step, int) and result_step > 0:
        return result_step
    return None


def solver_elapsed_time_errors(metrics: dict[str, Any],
                               args: argparse.Namespace) -> list[str]:
    errors = []
    max_wall_seconds = getattr(args, "max_solver_elapsed_wall_seconds", None)
    max_seconds_per_step = getattr(
        args, "max_solver_elapsed_seconds_per_accepted_step", None)
    if max_wall_seconds is None and max_seconds_per_step is None:
        return errors
    elapsed = metrics.get("solver_elapsed_wall_seconds")
    if not isinstance(elapsed, (int, float)):
        return ["solver elapsed wall time was not reported"]
    if max_wall_seconds is not None and float(elapsed) > max_wall_seconds:
        errors.append(
            f"solver elapsed wall time {float(elapsed):.3f}s exceeds "
            f"{max_wall_seconds:.3f}s"
        )
    if max_seconds_per_step is not None:
        accepted_steps = accepted_step_count_for_elapsed_budget(metrics)
        if accepted_steps is None:
            errors.append("accepted-step count is unavailable for elapsed-time budget")
        else:
            seconds_per_step = float(elapsed) / float(accepted_steps)
            metrics["solver_elapsed_seconds_per_accepted_step"] = seconds_per_step
            if seconds_per_step > max_seconds_per_step:
                errors.append(
                    "solver elapsed time per accepted step "
                    f"{seconds_per_step:.3f}s exceeds "
                    f"{max_seconds_per_step:.3f}s"
                )
    return errors


def evaluate_timeout_diagnostics(metrics: dict[str, Any],
                                 args: argparse.Namespace) -> list[str]:
    errors = []
    diagnostics = metrics["diagnostics"]
    errors.extend(solver_elapsed_time_errors(metrics, args))
    errors.extend(time_loop_convergence_errors(metrics, args))
    gauge_required = metrics.get("case") in {"d18", "d38", "mini2d", "static2d"}
    pre_solution_timeout = timeout_before_solution_state(diagnostics)
    if not diagnostics.get("cut_context_rebuilds"):
        errors.append("cut-context rebuild diagnostics were not reported")
    if not diagnostics.get("cut_volume_assemblies"):
        errors.append("cut-volume assembly diagnostics were not reported")
    if gauge_required and not (
            diagnostics.get("pressure_gauge_checks") or
            diagnostics.get("hydrostatic_initializations")):
        errors.append("pressure-gauge or hydrostatic initialization diagnostics were not reported")
    if gauge_required and not diagnostics.get("hydrostatic_initializations"):
        errors.append("hydrostatic initialization diagnostics were not reported")
    if not pre_solution_timeout and not latest_component_record(diagnostics, "solution_state"):
        errors.append("solution-state component diagnostics were not reported")
    if (diagnostics.get("true_residual_failure_count", 0) > 0 and
            not diagnostics.get("fsils_true_residuals")):
        errors.append("FSILS true-residual diagnostics were not reported")
    if args.require_newton_direction_check_diagnostics and not diagnostics.get("newton_direction_checks"):
        errors.append("Newton direction-check diagnostics were not reported")
    if args.require_jacobian_check_diagnostics and not diagnostics.get("jacobian_checks"):
        errors.append("Jacobian finite-difference diagnostics were not reported")
    if args.require_jacobian_top_mismatch_diagnostics and not diagnostics.get("jacobian_check_top_mismatches"):
        errors.append("Jacobian top-mismatch diagnostics were not reported")
    if args.require_jacobian_component_block_diagnostics:
        if not diagnostics.get("jacobian_check_component_details"):
            errors.append("Jacobian component-block diagnostics were not reported")
        expected_filters = normalized_component_sweeps(args.jacobian_check_component_sweeps)
        actual_filters = metrics.get("diagnostic_jacobian_component_sweep_filters", [])
        if expected_filters:
            missing_filters = [
                label for label in expected_filters
                if label not in actual_filters
            ]
            if missing_filters:
                errors.append(
                    "Jacobian component-block diagnostics are missing sweep filter(s): "
                    + ", ".join(missing_filters)
                )
    if args.require_linear_solve_history_diagnostics and not diagnostics.get("linear_solve_histories"):
        errors.append("linear solve history diagnostics were not reported")
    if args.require_form_block_diagnostics and (
            not diagnostics.get("form_block_installs") or not diagnostics.get("form_mixed_plans")):
        errors.append("form block installation diagnostics were not reported")
    if args.require_cut_context_solution_source_diagnostics:
        errors.extend(cut_context_solution_source_errors(diagnostics))
    errors.extend(cut_context_policy_errors(metrics, args))
    errors.extend(curvature_projection_errors(metrics, args))
    errors.extend(capillary_benchmark_errors(metrics, args))
    errors.extend(capillary_wave_benchmark_errors(metrics, args))
    errors.extend(capillary_wave_boundary_contract_errors(metrics, args))
    errors.extend(capillary_stability_errors(metrics, args))
    errors.extend(free_surface_conservative_balance_errors(metrics, args))
    errors.extend(free_surface_pressure_representability_errors(metrics, args))
    errors.extend(
        static_capillary_equilibrium_initialization_errors(metrics, args))
    # Output-free runs may be useful for solver diagnostics, but they cannot
    # satisfy an enabled sessile/contact-line qualification gate: every one of
    # those metrics is derived from the saved solved-field history.  Treat the
    # missing metrics exactly as the normal evaluation path does instead of
    # returning a false-positive `passed=true` result.
    errors.extend(sessile_physical_errors(metrics, args))
    errors.extend(fsils_accepted_true_residual_errors(metrics, args))
    errors.extend(fsils_matrix_diag_col_mismatch_errors(metrics, args))
    errors.extend(fsils_matrix_duplicate_diag_errors(metrics, args))
    if (args.require_newton_assembly_diagnostics and
            not diagnostics.get("newton_assemblies")):
        errors.append("Newton assembly diagnostics were not reported")
    if args.require_assembly_timing_diagnostics and not diagnostics.get("assembly_timings"):
        errors.append("assembly timing diagnostics were not reported")
    errors.extend(assembly_efficiency_errors(metrics, args))
    if args.require_process_memory_diagnostics:
        has_process_memory = (
            diagnostics.get("process_memory") or
            any(
                isinstance(record.get("process_rss_kb"), (int, float))
                for record in diagnostics.get("cut_context_rebuilds", [])
            )
        )
        if not has_process_memory:
            errors.append("process memory diagnostics were not reported")
    if (args.require_linear_solve_memory_diagnostics and
            not has_linear_solve_memory_diagnostics(diagnostics)):
        errors.append("linear-solve memory diagnostics were not reported")
    if (args.require_timeloop_initialization_diagnostics and
            not diagnostics.get("timeloop_initialization_solves")):
        errors.append("TimeLoop initialization linear-solve diagnostics were not reported")
    if (args.require_fsils_matrix_diagnostics and
            not diagnostics.get("fsils_prepared_matrices")):
        errors.append("FSILS prepared-matrix diagnostics were not reported")
    if args.max_fsils_matrix_zero_rows is not None:
        zero_rows = metrics.get("diagnostic_fsils_prepared_matrix_max_zero_rows")
        if not isinstance(zero_rows, (int, float)):
            errors.append("FSILS prepared-matrix zero-row diagnostics are unavailable")
        elif zero_rows > args.max_fsils_matrix_zero_rows:
            errors.append(
                f"FSILS prepared-matrix zero rows {zero_rows} exceed "
                f"{args.max_fsils_matrix_zero_rows}"
            )
    if args.max_fsils_matrix_missing_diag is not None:
        missing_diag = metrics.get(
            "diagnostic_fsils_prepared_matrix_max_missing_diag"
        )
        if not isinstance(missing_diag, (int, float)):
            errors.append("FSILS prepared-matrix missing-diagonal diagnostics are unavailable")
        elif missing_diag > args.max_fsils_matrix_missing_diag:
            errors.append(
                f"FSILS prepared-matrix missing diagonals {missing_diag} exceed "
                f"{args.max_fsils_matrix_missing_diag}"
            )
    if args.max_fsils_matrix_zero_diag is not None:
        zero_diag = metrics.get("diagnostic_fsils_prepared_matrix_max_zero_diag")
        if not isinstance(zero_diag, (int, float)):
            errors.append("FSILS prepared-matrix zero-diagonal diagnostics are unavailable")
        elif zero_diag > args.max_fsils_matrix_zero_diag:
            errors.append(
                f"FSILS prepared-matrix zero diagonals {zero_diag} exceed "
                f"{args.max_fsils_matrix_zero_diag}"
            )
    if args.max_fsils_matrix_nonfinite_entries is not None:
        nonfinite = metrics.get(
            "diagnostic_fsils_prepared_matrix_max_nonfinite_entries"
        )
        if not isinstance(nonfinite, (int, float)):
            errors.append("FSILS prepared-matrix nonfinite-entry diagnostics are unavailable")
        elif nonfinite > args.max_fsils_matrix_nonfinite_entries:
            errors.append(
                f"FSILS prepared-matrix nonfinite entries {nonfinite} exceed "
                f"{args.max_fsils_matrix_nonfinite_entries}"
            )
    if args.require_basis_cache_diagnostics:
        has_basis_cache = any(
            isinstance(record.get("basis_cache_entries"), (int, float))
            for record in diagnostics.get("process_memory", [])
        )
        if not has_basis_cache:
            errors.append("basis-cache diagnostics were not reported")
    if args.max_diagnostic_process_basis_cache_entries is not None:
        basis_cache_entries = metrics.get("diagnostic_process_max_basis_cache_entries")
        if not isinstance(basis_cache_entries, (int, float)):
            errors.append("basis-cache entry diagnostics are unavailable")
        elif basis_cache_entries > args.max_diagnostic_process_basis_cache_entries:
            errors.append(
                f"basis-cache entries {basis_cache_entries} exceed "
                f"{args.max_diagnostic_process_basis_cache_entries}"
            )
    errors.extend(resource_ceiling_errors(metrics, args))
    if args.require_interior_face_timing_diagnostics and not diagnostics.get("interior_face_timings"):
        errors.append("interior-face timing diagnostics were not reported")
    if args.require_cut_volume_timing_diagnostics and not diagnostics.get("cut_volume_timings"):
        errors.append("cut-volume timing diagnostics were not reported")
    if args.require_jit_specialization_trace_diagnostics and not diagnostics.get("jit_specialization_traces"):
        errors.append("JIT specialization trace diagnostics were not reported")
    if args.require_jit_cache_diagnostics and not diagnostics.get("jit_cache_diagnostics"):
        errors.append("JIT cache diagnostics were not reported")
    if getattr(args, "require_compiled_cut_volume_jit", False):
        errors.extend(compiled_cut_volume_jit_errors(diagnostics))
    if (args.require_marked_interior_face_fallback_diagnostics and
            not has_marked_interior_face_fallback_trace(diagnostics)):
        errors.append("marked interior-face fallback diagnostics were not reported")
    if args.require_assembly_topology_consistency:
        errors.extend(assembly_topology_consistency_errors(diagnostics))
    if (args.require_eigen_factorization_diagnostics and
            not diagnostics.get("eigen_factorization_diagnostics")):
        errors.append("Eigen factorization diagnostics were not reported")
    if (args.require_active_pressure_support_diagnostics and
            not diagnostics.get("active_pressure_support_constraints")):
        errors.append("active pressure support diagnostics were not reported")
    if args.max_eigen_factorization_zero_rows is not None:
        zero_rows = metrics.get("diagnostic_eigen_factorization_max_zero_rows")
        if not isinstance(zero_rows, (int, float)):
            errors.append("Eigen factorization zero-row diagnostics are unavailable")
        elif zero_rows > args.max_eigen_factorization_zero_rows:
            errors.append(
                f"Eigen factorization zero rows {zero_rows} exceed "
                f"{args.max_eigen_factorization_zero_rows}"
            )
    if args.max_eigen_factorization_pressure_zero_rows is not None:
        pressure_zero_rows = metrics.get(
            "diagnostic_eigen_factorization_max_pressure_zero_rows"
        )
        if not isinstance(pressure_zero_rows, (int, float)):
            errors.append("Eigen factorization pressure zero-row diagnostics are unavailable")
        elif pressure_zero_rows > args.max_eigen_factorization_pressure_zero_rows:
            errors.append(
                f"Eigen factorization pressure zero rows {pressure_zero_rows} exceed "
                f"{args.max_eigen_factorization_pressure_zero_rows}"
            )
    if args.max_eigen_factorization_pressure_zero_cols is not None:
        pressure_zero_cols = metrics.get(
            "diagnostic_eigen_factorization_max_pressure_zero_cols"
        )
        if not isinstance(pressure_zero_cols, (int, float)):
            errors.append("Eigen factorization pressure zero-column diagnostics are unavailable")
        elif pressure_zero_cols > args.max_eigen_factorization_pressure_zero_cols:
            errors.append(
                f"Eigen factorization pressure zero columns {pressure_zero_cols} exceed "
                f"{args.max_eigen_factorization_pressure_zero_cols}"
            )
    if args.max_eigen_factorization_nonfinite_entries is not None:
        nonfinite = metrics.get(
            "diagnostic_eigen_factorization_max_nonfinite_entries"
        )
        if not isinstance(nonfinite, (int, float)):
            errors.append("Eigen factorization nonfinite-entry diagnostics are unavailable")
        elif nonfinite > args.max_eigen_factorization_nonfinite_entries:
            errors.append(
                f"Eigen factorization nonfinite entries {nonfinite} exceed "
                f"{args.max_eigen_factorization_nonfinite_entries}"
            )

    if args.min_diagnostic_solution_velocity_range is not None:
        velocity_range = metrics.get("diagnostic_solution_velocity_range")
        if not isinstance(velocity_range, (int, float)):
            errors.append("diagnostic solution velocity range is unavailable")
        elif velocity_range < args.min_diagnostic_solution_velocity_range:
            errors.append(
                f"diagnostic solution velocity range {velocity_range:.6g} is below "
                f"{args.min_diagnostic_solution_velocity_range:.6g}"
            )
    if args.min_diagnostic_pressure_range is not None:
        pressure_range = metrics.get("diagnostic_solution_pressure_range")
        if not isinstance(pressure_range, (int, float)):
            errors.append("diagnostic solution pressure range is unavailable")
        elif pressure_range < args.min_diagnostic_pressure_range:
            errors.append(
                f"diagnostic solution pressure range {pressure_range:.6g} is below "
                f"{args.min_diagnostic_pressure_range:.6g}"
            )
    if args.max_diagnostic_active_volume_error is not None:
        volume_error = metrics.get("diagnostic_active_volume_error")
        if not isinstance(volume_error, (int, float)):
            errors.append("diagnostic active-volume consistency error is unavailable")
        elif volume_error > args.max_diagnostic_active_volume_error:
            errors.append(
                f"diagnostic active-volume error {volume_error:.6g} exceeds "
                f"{args.max_diagnostic_active_volume_error:.6g}"
            )
    if args.min_diagnostic_cut_volume_exact_order is not None:
        exact_order = metrics.get("diagnostic_cut_volume_min_exact_order")
        if not isinstance(exact_order, int):
            errors.append("diagnostic cut-volume exact order is unavailable")
        elif exact_order < args.min_diagnostic_cut_volume_exact_order:
            errors.append(
                f"diagnostic cut-volume exact order {exact_order} is below "
                f"{args.min_diagnostic_cut_volume_exact_order}"
            )
    if args.min_diagnostic_cut_volume_max_exact_order is not None:
        exact_order = metrics.get("diagnostic_cut_volume_max_exact_order")
        if not isinstance(exact_order, int):
            errors.append("diagnostic cut-volume max exact order is unavailable")
        elif exact_order < args.min_diagnostic_cut_volume_max_exact_order:
            errors.append(
                f"diagnostic cut-volume max exact order {exact_order} is below "
                f"{args.min_diagnostic_cut_volume_max_exact_order}"
            )
    if args.max_diagnostic_cut_adjacent_scale is not None:
        max_scale = metrics.get("diagnostic_cut_adjacent_max_scale")
        if not isinstance(max_scale, (int, float)):
            errors.append("diagnostic cut-adjacent max scale is unavailable")
        elif max_scale > args.max_diagnostic_cut_adjacent_scale:
            errors.append(
                f"diagnostic cut-adjacent max scale {max_scale:.6g} exceeds "
                f"{args.max_diagnostic_cut_adjacent_scale:.6g}"
            )
    if args.min_diagnostic_cut_adjacent_capped_scale_count is not None:
        capped_count = metrics.get("diagnostic_cut_adjacent_capped_scale_count")
        if not isinstance(capped_count, int):
            errors.append("diagnostic cut-adjacent capped scale count is unavailable")
        elif capped_count < args.min_diagnostic_cut_adjacent_capped_scale_count:
            errors.append(
                f"diagnostic cut-adjacent capped scale count {capped_count} is below "
                f"{args.min_diagnostic_cut_adjacent_capped_scale_count}"
            )
    if args.min_diagnostic_active_pruned_volume_regions is not None:
        pruned_count = metrics.get("diagnostic_active_pruned_volume_regions")
        if not isinstance(pruned_count, int):
            errors.append("diagnostic active pruned volume-region count is unavailable")
        elif pruned_count < args.min_diagnostic_active_pruned_volume_regions:
            errors.append(
                f"diagnostic active pruned volume-region count {pruned_count} is below "
                f"{args.min_diagnostic_active_pruned_volume_regions}"
            )
    if args.min_diagnostic_active_min_volume_fraction is not None:
        min_fraction = metrics.get("diagnostic_active_min_volume_fraction")
        if not isinstance(min_fraction, (int, float)):
            errors.append("diagnostic active min volume fraction is unavailable")
        elif min_fraction < args.min_diagnostic_active_min_volume_fraction:
            errors.append(
                f"diagnostic active min volume fraction {min_fraction:.6g} is below "
                f"{args.min_diagnostic_active_min_volume_fraction:.6g}"
            )
    if args.min_diagnostic_generated_pruned_volume_rules is not None:
        pruned_rules = metrics.get("diagnostic_generated_pruned_volume_rules")
        if not isinstance(pruned_rules, int):
            errors.append("diagnostic generated pruned volume-rule count is unavailable")
        elif pruned_rules < args.min_diagnostic_generated_pruned_volume_rules:
            errors.append(
                f"diagnostic generated pruned volume-rule count {pruned_rules} is below "
                f"{args.min_diagnostic_generated_pruned_volume_rules}"
            )
    if args.min_diagnostic_blockschur_true_residual_retries is not None:
        retries = metrics.get("diagnostic_blockschur_true_residual_retries")
        if not isinstance(retries, int):
            errors.append("diagnostic BlockSchur true-residual retry count is unavailable")
        elif retries < args.min_diagnostic_blockschur_true_residual_retries:
            errors.append(
                f"diagnostic BlockSchur true-residual retry count {retries} is below "
                f"{args.min_diagnostic_blockschur_true_residual_retries}"
            )
    if args.max_newton_direction_relative_error is not None:
        value = metrics.get("diagnostic_newton_direction_relative_error")
        if not isinstance(value, (int, float)):
            errors.append("Newton direction-check relative error is unavailable")
        elif value > args.max_newton_direction_relative_error:
            errors.append(
                f"Newton direction-check relative error {value:.6g} exceeds "
                f"{args.max_newton_direction_relative_error:.6g}"
            )
    if args.max_jacobian_check_relative_error is not None:
        value = metrics.get("diagnostic_jacobian_check_relative_error")
        if not isinstance(value, (int, float)):
            errors.append("Jacobian finite-difference relative error is unavailable")
        elif value > args.max_jacobian_check_relative_error:
            errors.append(
                f"Jacobian finite-difference relative error {value:.6g} exceeds "
                f"{args.max_jacobian_check_relative_error:.6g}"
            )
    if args.max_jacobian_component_block_relative_error is not None:
        value = metrics.get("diagnostic_jacobian_component_block_max_relative_error")
        if not isinstance(value, (int, float)):
            errors.append("Jacobian component-block relative error is unavailable")
        elif value > args.max_jacobian_component_block_relative_error:
            errors.append(
                f"Jacobian component-block relative error {value:.6g} exceeds "
                f"{args.max_jacobian_component_block_relative_error:.6g}"
            )
    if args.stale_pressure_gauge_tolerance is not None:
        stale_difference = metrics.get("diagnostic_pressure_gauge_previous_invalid_difference")
        if not isinstance(stale_difference, (int, float)):
            errors.append("diagnostic pressure gauge stale-value difference is unavailable")
        elif abs(float(stale_difference)) <= args.stale_pressure_gauge_tolerance:
            errors.append(
                "diagnostic pressure gauge remains close to the previous "
                "full-volume hydrostatic value"
            )
    return errors


def pressure_gauge_metrics(output: pv.DataSet, benchmark: dict[str, Any]) -> dict[str, Any]:
    gauge = benchmark.get("pressure_gauge")
    if not isinstance(gauge, dict) or "Pressure" not in output.point_data:
        return {}
    node_id = gauge.get("node_id")
    if node_id is None:
        return {}

    gids = None
    for name in ("GlobalNodeID", "GlobalVertexID"):
        if name in output.point_data:
            gids = np.asarray(output.point_data[name], dtype=np.int64).reshape(-1)
            break
    if gids is None:
        return {"pressure_gauge_found": False}

    indices = np.flatnonzero(gids == int(node_id))
    if indices.size == 0:
        return {"pressure_gauge_found": False}

    pressure = np.asarray(output.point_data["Pressure"], dtype=float).reshape(-1)
    value = float(pressure[indices[0]])
    metrics: dict[str, Any] = {
        "pressure_gauge_found": True,
        "pressure_gauge_node_id": int(node_id),
        "pressure_gauge_value": value,
        "pressure_gauge_matches": int(indices.size),
    }
    expected = gauge.get("expected_initial_hydrostatic_pressure")
    if isinstance(expected, (int, float)):
        metrics["pressure_gauge_expected_initial"] = float(expected)
        metrics["pressure_gauge_initial_error"] = value - float(expected)

    verification = benchmark.get("pressure_gauge_verification")
    if isinstance(verification, dict):
        stale = verification.get("previous_invalid_d18_full_volume_hydrostatic_pressure")
        if isinstance(stale, (int, float)):
            metrics["pressure_gauge_previous_invalid"] = float(stale)
            metrics["pressure_gauge_previous_invalid_difference"] = value - float(stale)
    return metrics


def interface_profile_xy(dataset: pv.DataSet) -> tuple[np.ndarray, np.ndarray] | None:
    if "phi" not in dataset.point_data:
        return None
    try:
        interface = dataset.contour(isosurfaces=[0.0], scalars="phi")
    except Exception:
        return None
    points = np.asarray(interface.points, dtype=float)
    if points.ndim != 2 or points.shape[0] == 0 or points.shape[1] < 2:
        return None
    x = points[:, 0]
    y = points[:, 1]
    finite = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(finite) < 2:
        return None
    x = x[finite]
    y = y[finite]
    order = np.argsort(x, kind="mergesort")
    x = x[order]
    y = y[order]

    unique_x: list[float] = []
    averaged_y: list[float] = []
    start = 0
    tolerance = 1.0e-12
    while start < x.size:
        end = start + 1
        while end < x.size and abs(float(x[end] - x[start])) <= tolerance:
            end += 1
        unique_x.append(float(np.mean(x[start:end])))
        averaged_y.append(float(np.mean(y[start:end])))
        start = end
    if len(unique_x) < 2:
        return None
    return np.asarray(unique_x, dtype=float), np.asarray(averaged_y, dtype=float)


def add_interface_profile_summary(metrics: dict[str, Any],
                                  prefix: str,
                                  profile: tuple[np.ndarray, np.ndarray] | None) -> None:
    if profile is None:
        metrics[f"{prefix}_interface_available"] = False
        return
    x, y = profile
    metrics[f"{prefix}_interface_available"] = True
    metrics[f"{prefix}_interface_points"] = int(x.size)
    metrics[f"{prefix}_interface_x_min"] = float(np.min(x))
    metrics[f"{prefix}_interface_x_max"] = float(np.max(x))
    metrics[f"{prefix}_interface_height_min"] = float(np.min(y))
    metrics[f"{prefix}_interface_height_max"] = float(np.max(y))
    metrics[f"{prefix}_interface_height_mean"] = float(np.mean(y))
    metrics[f"{prefix}_interface_height_span"] = float(np.max(y) - np.min(y))
    if np.max(x) > np.min(x):
        slope, intercept = np.polyfit(x, y, 1)
        metrics[f"{prefix}_interface_slope"] = float(slope)
        metrics[f"{prefix}_interface_intercept"] = float(intercept)


def add_interface_motion_metrics(metrics: dict[str, Any],
                                 initial: pv.DataSet,
                                 output: pv.DataSet) -> None:
    initial_profile = interface_profile_xy(initial)
    final_profile = interface_profile_xy(output)
    add_interface_profile_summary(metrics, "initial", initial_profile)
    add_interface_profile_summary(metrics, "final", final_profile)
    if initial_profile is None or final_profile is None:
        metrics["interface_motion_available"] = False
        return

    initial_x, initial_y = initial_profile
    final_x, final_y = final_profile
    x_min = max(float(np.min(initial_x)), float(np.min(final_x)))
    x_max = min(float(np.max(initial_x)), float(np.max(final_x)))
    if not np.isfinite(x_min) or not np.isfinite(x_max) or x_max <= x_min:
        metrics["interface_motion_available"] = False
        metrics["interface_motion_unavailable_reason"] = "profiles_do_not_overlap_in_x"
        return

    sample_count = min(201, max(25, int(min(initial_x.size, final_x.size) * 4)))
    sample_x = np.linspace(x_min, x_max, sample_count)
    initial_sample_y = np.interp(sample_x, initial_x, initial_y)
    final_sample_y = np.interp(sample_x, final_x, final_y)
    delta = final_sample_y - initial_sample_y

    metrics["interface_motion_available"] = True
    metrics["interface_motion_sample_count"] = int(sample_count)
    metrics["interface_motion_x_min"] = float(x_min)
    metrics["interface_motion_x_max"] = float(x_max)
    metrics["interface_height_max_abs_change"] = float(np.max(np.abs(delta)))
    metrics["interface_height_mean_abs_change"] = float(np.mean(np.abs(delta)))
    metrics["interface_height_rms_change"] = float(np.sqrt(np.mean(delta * delta)))
    metrics["interface_height_signed_mean_change"] = float(np.mean(delta))
    metrics["interface_height_change_min"] = float(np.min(delta))
    metrics["interface_height_change_max"] = float(np.max(delta))

    initial_slope = metrics.get("initial_interface_slope")
    final_slope = metrics.get("final_interface_slope")
    if isinstance(initial_slope, (int, float)) and isinstance(final_slope, (int, float)):
        metrics["interface_slope_change"] = float(final_slope) - float(initial_slope)
        metrics["interface_slope_abs_change"] = abs(float(final_slope) - float(initial_slope))


def add_capillary_wave_profile_fit(metrics: dict[str, Any],
                                   benchmark: dict[str, Any],
                                   prefix: str,
                                   profile: tuple[np.ndarray, np.ndarray] | None
                                   ) -> None:
    wave = benchmark.get("capillary_wave")
    if not isinstance(wave, dict):
        return
    if profile is None:
        metrics[f"{prefix}_capillary_wave_profile_available"] = False
        return
    k = wave.get("wavenumber")
    base_height = wave.get("base_height")
    if not isinstance(k, (int, float)) or not isinstance(base_height, (int, float)):
        metrics[f"{prefix}_capillary_wave_profile_available"] = False
        metrics[f"{prefix}_capillary_wave_profile_error"] = "missing benchmark wave data"
        return

    x, y = profile
    if x.size < 3:
        metrics[f"{prefix}_capillary_wave_profile_available"] = False
        metrics[f"{prefix}_capillary_wave_profile_error"] = "not enough interface samples"
        return

    design = np.column_stack((
        np.cos(float(k) * x),
        np.sin(float(k) * x),
        np.ones_like(x),
    ))
    coeffs, *_ = np.linalg.lstsq(design, y - float(base_height), rcond=None)
    cosine, sine, mean_offset = (float(value) for value in coeffs)
    residual = (design @ coeffs) - (y - float(base_height))
    metrics[f"{prefix}_capillary_wave_profile_available"] = True
    metrics[f"{prefix}_capillary_wave_cosine_amplitude"] = cosine
    metrics[f"{prefix}_capillary_wave_sine_amplitude"] = sine
    metrics[f"{prefix}_capillary_wave_mean_offset"] = mean_offset
    metrics[f"{prefix}_capillary_wave_amplitude"] = float(math.hypot(cosine, sine))
    metrics[f"{prefix}_capillary_wave_fit_rmse"] = float(
        math.sqrt(float(np.mean(residual * residual)))
    )


def add_capillary_wave_profile_metrics(metrics: dict[str, Any],
                                       benchmark: dict[str, Any],
                                       initial: pv.DataSet,
                                       output: pv.DataSet) -> None:
    if not isinstance(benchmark.get("capillary_wave"), dict):
        return
    add_capillary_wave_profile_fit(
        metrics, benchmark, "initial", interface_profile_xy(initial))
    add_capillary_wave_profile_fit(
        metrics, benchmark, "final", interface_profile_xy(output))


def coordinate_min_spacing(points: np.ndarray) -> float | None:
    spacings = []
    for axis in range(min(points.shape[1], 3)):
        unique = np.unique(np.round(points[:, axis], decimals=12))
        if unique.size < 2:
            continue
        diffs = np.diff(np.sort(unique))
        positive = diffs[diffs > 1.0e-12]
        if positive.size:
            spacings.append(float(np.min(positive)))
    return min(spacings) if spacings else None


def add_projected_curvature_field_metrics(metrics: dict[str, Any],
                                          output: pv.DataSet) -> None:
    if "phi" not in output.point_data:
        return
    curvature_name = None
    for candidate in ("kappa_projected", "curvature_projected"):
        if candidate in output.point_data:
            curvature_name = candidate
            break
    if curvature_name is None:
        return

    phi = np.asarray(output.point_data["phi"], dtype=float).reshape(-1)
    curvature = np.asarray(output.point_data[curvature_name], dtype=float).reshape(-1)
    if phi.shape[0] != curvature.shape[0]:
        return

    points = np.asarray(output.points, dtype=float)
    spacing = coordinate_min_spacing(points)
    band_width = 0.15 if spacing is None else max(1.0e-12, 1.5 * spacing)
    finite = np.isfinite(phi) & np.isfinite(curvature)
    near_interface = finite & (np.abs(phi) <= band_width)
    if not np.any(near_interface):
        return

    near_values = curvature[near_interface]
    metrics["projected_curvature_field_name"] = curvature_name
    metrics["projected_curvature_near_interface_source"] = (
        "vtk_point_data_projected_curvature_field"
    )
    metrics["projected_curvature_near_interface_sampling_domain"] = (
        "abs(phi)<=postprocessed_interface_band_width"
    )
    metrics["projected_curvature_near_interface_band_width"] = float(band_width)
    metrics["projected_curvature_near_interface_point_count"] = int(
        np.count_nonzero(near_interface))
    metrics["projected_curvature_near_interface_mean_abs"] = float(
        np.mean(np.abs(near_values)))
    metrics["projected_curvature_near_interface_median_abs"] = float(
        np.median(np.abs(near_values)))
    metrics["projected_curvature_near_interface_max_abs"] = float(
        np.max(np.abs(near_values)))


def result_time_series_paths(case_dir: Path) -> list[tuple[int, Path]]:
    by_step: dict[int, list[Path]] = {}
    for path in [*case_dir.rglob("result_*.vtu"), *case_dir.rglob("result_*.pvtu")]:
        match = re.fullmatch(r"result_([0-9]+)\.p?vtu", path.name)
        if match is None:
            continue
        by_step.setdefault(int(match.group(1)), []).append(path)

    selected = []
    for step, paths in sorted(by_step.items()):
        # Prefer a parallel collection over one of its pieces, then prefer the
        # least deeply nested serial output.  This leaves one physical state
        # per accepted output step.
        paths.sort(key=lambda path: (path.suffix != ".pvtu", len(path.parts), str(path)))
        selected.append((step, paths[0]))
    return selected


def sessile_contact_wall_frame(contact: dict[str, Any]) -> dict[str, Any]:
    """Validate benchmark wall metadata and return a normalized 2-D frame."""
    wall_face = str(contact.get("wall", "wall_bottom"))
    expected = sessile_contact_wall_spec(wall_face)
    wall_axis = contact.get("wall_axis", expected["wall_axis"])
    tangent_axis = contact.get(
        "wall_tangent_axis", expected["wall_tangent_axis"])
    if (not isinstance(wall_axis, int) or isinstance(wall_axis, bool) or
            not isinstance(tangent_axis, int) or isinstance(tangent_axis, bool) or
            {wall_axis, tangent_axis} != {0, 1}):
        raise ValueError("sessile wall axes must be the distinct 2-D coordinate axes")

    legacy_coordinate_name = "wall_y" if wall_axis == 1 else "wall_x"
    wall_coordinate = contact.get(
        "wall_coordinate",
        contact.get(legacy_coordinate_name, expected["wall_coordinate"]),
    )
    if (not isinstance(wall_coordinate, (int, float)) or
            isinstance(wall_coordinate, bool) or
            not math.isfinite(float(wall_coordinate))):
        raise ValueError("sessile wall coordinate must be finite")

    raw_normal = contact.get("wall_normal", expected["wall_normal"])
    raw_tangent = contact.get("wall_tangent", expected["wall_tangent"])
    try:
        wall_normal = np.asarray(raw_normal[:2], dtype=float)
        wall_tangent = np.asarray(raw_tangent[:2], dtype=float)
    except (TypeError, ValueError, IndexError) as exc:
        raise ValueError("sessile wall normal/tangent metadata is invalid") from exc
    normal_norm = float(np.linalg.norm(wall_normal))
    tangent_norm = float(np.linalg.norm(wall_tangent))
    if (not np.isfinite(wall_normal).all() or
            not np.isfinite(wall_tangent).all() or
            normal_norm <= 0.0 or tangent_norm <= 0.0):
        raise ValueError("sessile wall normal/tangent metadata is degenerate")
    wall_normal /= normal_norm
    wall_tangent /= tangent_norm
    expected_normal = np.asarray(expected["wall_normal"][:2], dtype=float)
    expected_tangent = np.asarray(expected["wall_tangent"][:2], dtype=float)
    if (not np.allclose(wall_normal, expected_normal, rtol=0.0, atol=1.0e-12) or
            not np.allclose(wall_tangent, expected_tangent,
                            rtol=0.0, atol=1.0e-12) or
            wall_axis != expected["wall_axis"] or
            tangent_axis != expected["wall_tangent_axis"] or
            not math.isclose(float(wall_coordinate),
                             float(expected["wall_coordinate"]),
                             rel_tol=0.0, abs_tol=1.0e-12)):
        raise ValueError(
            f"sessile wall metadata is inconsistent with {wall_face}")
    if abs(float(np.dot(wall_normal, wall_tangent))) > 1.0e-12:
        raise ValueError("sessile wall normal and tangent are not orthogonal")
    return {
        "wall_face": wall_face,
        "wall_axis": wall_axis,
        "wall_coordinate": float(wall_coordinate),
        "wall_normal": wall_normal,
        "wall_tangent_axis": tangent_axis,
        "wall_tangent": wall_tangent,
    }


def fit_sessile_interface(dataset: pv.DataSet,
                          wall_y: float = 0.0,
                          *,
                          wall_axis: int = 1,
                          wall_normal: np.ndarray | None = None) -> dict[str, Any]:
    if "phi" not in dataset.point_data:
        return {"available": False, "error": "missing phi field"}
    try:
        interface = dataset.contour(isosurfaces=[0.0], scalars="phi")
    except Exception as exc:
        return {"available": False, "error": str(exc)}
    points = np.asarray(interface.points, dtype=float)
    finite = np.isfinite(points).all(axis=1) if points.ndim == 2 else np.zeros(0, dtype=bool)
    points = points[finite]
    if points.shape[0] < 5:
        return {
            "available": False,
            "error": "fewer than five finite interface points",
            "interface_points": int(points.shape[0]),
        }

    xy = points[:, :2]
    design = np.column_stack((xy[:, 0], xy[:, 1], np.ones(xy.shape[0])))
    rhs = -(xy[:, 0] * xy[:, 0] + xy[:, 1] * xy[:, 1])
    coefficients, _residuals, rank, _singular = np.linalg.lstsq(design, rhs, rcond=None)
    if rank < 3:
        return {"available": False, "error": "rank-deficient circle fit"}
    center_x = -0.5 * float(coefficients[0])
    center_y = -0.5 * float(coefficients[1])
    radius_squared = center_x * center_x + center_y * center_y - float(coefficients[2])
    if not math.isfinite(radius_squared) or radius_squared <= 0.0:
        return {"available": False, "error": "nonpositive fitted radius"}
    radius = math.sqrt(radius_squared)
    radial = np.sqrt((xy[:, 0] - center_x) ** 2 + (xy[:, 1] - center_y) ** 2)
    fit_rmse = math.sqrt(float(np.mean((radial - radius) ** 2)))
    if wall_axis not in {0, 1}:
        return {"available": False, "error": "wall axis must be 0 or 1"}
    tangent_axis = 1 - wall_axis
    if wall_normal is None:
        wall_normal = np.zeros(2, dtype=float)
        wall_normal[wall_axis] = -1.0
    wall_normal = np.asarray(wall_normal, dtype=float).reshape(-1)
    if (wall_normal.size < 2 or not np.isfinite(wall_normal[:2]).all() or
            not float(np.linalg.norm(wall_normal[:2])) > 0.0):
        return {"available": False, "error": "invalid wall normal"}
    wall_normal = wall_normal[:2] / float(np.linalg.norm(wall_normal[:2]))
    center = np.asarray([center_x, center_y], dtype=float)
    wall_point = center.copy()
    wall_point[wall_axis] = wall_y
    cosine = max(
        -1.0,
        min(1.0, float(np.dot(center - wall_point, wall_normal)) / radius),
    )
    angle_degrees = math.degrees(math.acos(cosine))
    center_wall_distance = center[wall_axis] - wall_y
    contact_half_width = math.sqrt(max(
        radius * radius - center_wall_distance * center_wall_distance, 0.0))
    near_wall_tolerance = max(1.0e-10, 0.02 * radius)
    near_wall = np.abs(xy[:, wall_axis] - wall_y) <= near_wall_tolerance
    observed_contact_coordinate = sorted(
        float(value) for value in xy[near_wall, tangent_axis])
    fitted_contact_coordinate = [
        float(center[tangent_axis] - contact_half_width),
        float(center[tangent_axis] + contact_half_width),
    ]
    result = {
        "available": True,
        "interface_points": int(points.shape[0]),
        "circle_center": [center_x, center_y],
        "circle_radius": radius,
        "circle_fit_rmse": fit_rmse,
        "contact_angle_degrees": angle_degrees,
        "half_footprint": contact_half_width,
        "footprint": 2.0 * contact_half_width,
        "wall_axis": wall_axis,
        "wall_coordinate": wall_y,
        "wall_tangent_axis": tangent_axis,
        "fitted_contact_coordinate": fitted_contact_coordinate,
        "near_wall_contact_coordinate_samples": observed_contact_coordinate,
    }
    tangent_name = "x" if tangent_axis == 0 else "y"
    result[f"fitted_contact_{tangent_name}"] = fitted_contact_coordinate
    result[f"near_wall_contact_{tangent_name}_samples"] = (
        observed_contact_coordinate)
    if len(observed_contact_coordinate) >= 2:
        lower_contact = min(observed_contact_coordinate)
        upper_contact = max(observed_contact_coordinate)
        wall_contacts = [lower_contact, upper_contact]
        result["wall_contact_coordinate"] = wall_contacts
        result[f"wall_contact_{tangent_name}"] = wall_contacts
        result["wall_footprint"] = upper_contact - lower_contact
        result["wall_half_footprint"] = 0.5 * (
            upper_contact - lower_contact)
    return result


def dataset_wet_volume(dataset: pv.DataSet) -> tuple[float | None, str | None]:
    if "WetVolumeMeasure" in dataset.cell_data:
        values = np.asarray(dataset.cell_data["WetVolumeMeasure"], dtype=float).reshape(-1)
        if values.shape[0] == dataset.n_cells:
            return float(np.sum(values)), "WetVolumeMeasure"
    if "WetVolumeFraction" in dataset.cell_data:
        fractions = np.asarray(dataset.cell_data["WetVolumeFraction"], dtype=float).reshape(-1)
        measures = cell_measure(dataset)
        if fractions.shape[0] == measures.shape[0]:
            return float(np.sum(fractions * measures)), "WetVolumeFraction"
    return None, None


def add_sessile_contact_fluid_speed(state: dict[str, Any],
                                    dataset: pv.DataSet,
                                    wall_coordinate: float,
                                    wall_axis: int = 1) -> None:
    """Interpolate the wall-tangential fluid speed at both contact points.

    The Ren--E residual uses ``V_CL = dot(u, m)`` on the generated contact
    set.  A finite difference of the globally fitted footprint is instead a
    transport/kinematic observable and includes startup and time-integration
    lag.  This output-space interpolation is the closest available solved
    field observable to the velocity used by the constitutive residual.
    """
    if wall_axis not in {0, 1}:
        return
    tangent_axis = 1 - wall_axis
    contact_source = (
        "phi_zero_wall_intersections" if "wall_contact_coordinate" in state
        else "fitted_circle_wall_intersections"
    )
    contact_coordinate = state.get(
        "wall_contact_coordinate", state.get("fitted_contact_coordinate"))
    if (not isinstance(contact_coordinate, list) or
            len(contact_coordinate) != 2 or
            "Velocity" not in dataset.point_data):
        return
    points = np.asarray(dataset.points, dtype=float)
    velocity = np.asarray(dataset.point_data["Velocity"], dtype=float)
    if velocity.ndim == 1:
        velocity = velocity.reshape((-1, 1))
    if (points.ndim != 2 or points.shape[0] != velocity.shape[0] or
            points.shape[1] < 2 or velocity.shape[1] <= tangent_axis):
        return
    spacing = coordinate_min_spacing(points)
    wall_tolerance = max(1.0e-12, 1.0e-8 * (spacing or 1.0))
    wall = (
        np.isfinite(points[:, 0]) & np.isfinite(points[:, 1]) &
        np.isfinite(velocity[:, tangent_axis]) &
        np.isclose(
            points[:, wall_axis], wall_coordinate,
            rtol=0.0, atol=wall_tolerance)
    )
    if np.count_nonzero(wall) < 2:
        return
    wall_position = points[wall, tangent_axis]
    wall_tangent_velocity = velocity[wall, tangent_axis]
    order = np.argsort(wall_position)
    wall_position = wall_position[order]
    wall_tangent_velocity = wall_tangent_velocity[order]
    unique_position, first = np.unique(wall_position, return_index=True)
    wall_tangent_velocity = wall_tangent_velocity[first]
    if unique_position.size < 2:
        return
    lower, upper = (
        float(contact_coordinate[0]), float(contact_coordinate[1]))
    if (not math.isfinite(lower) or not math.isfinite(upper) or
            lower < unique_position[0] - wall_tolerance or
            upper > unique_position[-1] + wall_tolerance):
        return
    lower_velocity = float(np.interp(
        lower, unique_position, wall_tangent_velocity))
    upper_velocity = float(np.interp(
        upper, unique_position, wall_tangent_velocity))
    tangent_name = "x" if tangent_axis == 0 else "y"
    state["contact_fluid_velocity_tangent"] = [
        lower_velocity, upper_velocity]
    state[f"contact_fluid_velocity_{tangent_name}"] = [
        lower_velocity, upper_velocity]
    state["contact_fluid_evaluation_contact_coordinate"] = [lower, upper]
    state[f"contact_fluid_evaluation_contact_{tangent_name}"] = [lower, upper]
    state["contact_fluid_wall_axis"] = wall_axis
    state["contact_fluid_wall_tangent_axis"] = tangent_axis
    state["contact_fluid_evaluation_source"] = contact_source
    # The footprint direction is -t at the lower coordinate and +t at the
    # upper coordinate for either the horizontal or vertical wall.
    state["contact_fluid_outward_speed"] = 0.5 * (
        upper_velocity - lower_velocity)
    state["contact_fluid_symmetry_defect"] = 0.5 * (
        upper_velocity + lower_velocity)


def add_sessile_operator_contact_geometry(state: dict[str, Any],
                                          dataset: pv.DataSet,
                                          benchmark: dict[str, Any]) -> None:
    """Evaluate the contact state with the generated LinearCorner normal.

    SurfaceStress and its contact term use the normal carried by the same
    generated interface fragment that supplies ``dI`` and
    ``dInterfaceBoundary``.  For a Quad4 field that chord normal generally
    differs from ``unitNormalFromLevelSet(phi)`` at the contact root.  Rebuild
    the production LinearCorner segment (edge roots, farthest pair, and
    least-squares gradient orientation) from the saved state.  The former Q1
    normal is retained in each sample as a representation diagnostic only.
    """
    contact = benchmark.get("sessile_contact")
    if not isinstance(contact, dict):
        return

    def unavailable(reason: str) -> None:
        state["operator_contact_geometry_available"] = False
        state["operator_contact_geometry_error"] = reason

    if "phi" not in dataset.point_data:
        unavailable("missing phi field")
        return
    points = np.asarray(dataset.points, dtype=float)
    phi = np.asarray(dataset.point_data["phi"], dtype=float).reshape(-1)
    if (points.ndim != 2 or points.shape[0] != phi.shape[0] or
            points.shape[1] < 2):
        unavailable("incompatible point geometry or phi field")
        return
    velocity: np.ndarray | None = None
    if "Velocity" in dataset.point_data:
        velocity = np.asarray(dataset.point_data["Velocity"], dtype=float)
        if velocity.ndim == 1:
            velocity = velocity.reshape((-1, 1))
        if velocity.shape[0] != points.shape[0]:
            velocity = None

    try:
        wall_frame = sessile_contact_wall_frame(contact)
    except ValueError as exc:
        unavailable(str(exc))
        return
    wall_axis = int(wall_frame["wall_axis"])
    tangent_axis = int(wall_frame["wall_tangent_axis"])
    wall_coordinate = float(wall_frame["wall_coordinate"])
    wall_normal = np.asarray(wall_frame["wall_normal"], dtype=float)
    active_domain = contact.get("active_domain", "LevelSetNegative")
    if active_domain not in {"LevelSetNegative", "LevelSetPositive"}:
        unavailable("unsupported active-domain metadata")
        return
    target_angle = contact.get("equilibrium_contact_angle_degrees")
    contact_line_model = contact.get("contact_line_model")
    mobility = contact.get("mobility")
    surface_tension = benchmark.get("surface_tension")
    if contact_line_model not in {
            "DynamicContactAngle", "PrescribedContactAngle"}:
        unavailable("unsupported contact-line model metadata")
        return
    if (not isinstance(target_angle, (int, float)) or
            isinstance(target_angle, bool) or
            not math.isfinite(float(target_angle))):
        unavailable("incomplete contact-angle metadata")
        return
    dynamic_model = contact_line_model == "DynamicContactAngle"
    if (dynamic_model and
            not all(isinstance(value, (int, float)) and
                    not isinstance(value, bool) and math.isfinite(float(value))
                    for value in (mobility, surface_tension))):
        unavailable("incomplete Ren--E coefficient metadata")
        return
    target_cos = math.cos(math.radians(float(target_angle)))

    quad_reference_nodes = np.asarray([
        [-1.0, -1.0],
        [1.0, -1.0],
        [1.0, 1.0],
        [-1.0, 1.0],
    ])
    quad_reference_edges = ((0, 1), (1, 2), (2, 3), (3, 0))
    quad_reference_edge_set = {
        tuple(sorted(edge)) for edge in quad_reference_edges}
    triangle_reference_edges = ((0, 1), (1, 2), (2, 0))
    spacing = coordinate_min_spacing(points)
    wall_tolerance = max(1.0e-12, 1.0e-8 * (spacing or 1.0))
    samples: list[dict[str, Any]] = []
    for cell_id in range(dataset.n_cells):
        cell = dataset.get_cell(cell_id)
        point_ids = np.asarray(cell.point_ids, dtype=int)
        cell_type = int(dataset.celltypes[cell_id])
        is_triangle = (
            cell_type == int(pv.CellType.TRIANGLE) and point_ids.size == 3)
        is_quad = (
            cell_type == int(pv.CellType.QUAD) and point_ids.size == 4)
        if not (is_triangle or is_quad):
            continue
        reference_edges = (
            triangle_reference_edges if is_triangle else quad_reference_edges)
        cell_points = points[point_ids, :2]
        cell_phi = phi[point_ids]
        if not (np.isfinite(cell_points).all() and np.isfinite(cell_phi).all()):
            continue
        wall_local = np.flatnonzero(np.isclose(
            cell_points[:, wall_axis], wall_coordinate,
            rtol=0.0, atol=wall_tolerance))
        if wall_local.size != 2:
            continue
        a, b = (int(wall_local[0]), int(wall_local[1]))
        if (is_quad and
                tuple(sorted((a, b))) not in quad_reference_edge_set):
            continue
        phi_a = float(cell_phi[a])
        phi_b = float(cell_phi[b])
        if phi_a * phi_b > 0.0 or math.isclose(
                phi_a, phi_b, rel_tol=0.0, abs_tol=1.0e-300):
            continue
        if phi_a == 0.0:
            edge_t = 0.0
        elif phi_b == 0.0:
            edge_t = 1.0
        else:
            edge_t = -phi_a / (phi_b - phi_a)
        if edge_t < -1.0e-12 or edge_t > 1.0 + 1.0e-12:
            continue
        edge_t = min(1.0, max(0.0, edge_t))
        if is_triangle:
            shape = np.zeros(3, dtype=float)
            shape[a] = 1.0 - edge_t
            shape[b] = edge_t
            parent_coordinate = [float(shape[1]), float(shape[2])]
            affine_design = np.column_stack((
                cell_points[:, 0], cell_points[:, 1], np.ones(3)))
            try:
                affine_coefficients = np.linalg.solve(
                    affine_design, cell_phi)
            except np.linalg.LinAlgError:
                continue
            physical_gradient = np.asarray(
                affine_coefficients[:2], dtype=float)
        else:
            parent = ((1.0 - edge_t) * quad_reference_nodes[a] +
                      edge_t * quad_reference_nodes[b])
            xi, eta = float(parent[0]), float(parent[1])
            parent_coordinate = [xi, eta]
            shape = 0.25 * np.asarray([
                (1.0 - xi) * (1.0 - eta),
                (1.0 + xi) * (1.0 - eta),
                (1.0 + xi) * (1.0 + eta),
                (1.0 - xi) * (1.0 + eta),
            ])
            shape_gradient = 0.25 * np.asarray([
                [-(1.0 - eta), -(1.0 - xi)],
                [1.0 - eta, -(1.0 + xi)],
                [1.0 + eta, 1.0 + xi],
                [-(1.0 + eta), 1.0 - xi],
            ])
            jacobian = cell_points.T @ shape_gradient
            try:
                physical_gradient = np.linalg.solve(
                    jacobian.T, shape_gradient.T @ cell_phi)
            except np.linalg.LinAlgError:
                continue
        gradient_norm = float(np.linalg.norm(physical_gradient))
        safe_gradient_norm = math.sqrt(gradient_norm * gradient_norm + 1.0e-24)
        if not safe_gradient_norm > 0.0:
            continue
        element_active_normal = physical_gradient / safe_gradient_norm
        if active_domain == "LevelSetPositive":
            element_active_normal = -element_active_normal

        # Reproduce cutLinearLevelSetCell2D: collect unique edge roots, take
        # the farthest pair as the fragment, form its chord normal, and orient
        # it from phi<0 to phi>0 with the affine least-squares gradient.
        cut_points: list[np.ndarray] = []
        point_tolerance = max(1.0e-14, 1.0e-10 * (spacing or 1.0))
        for edge_a, edge_b in reference_edges:
            value_a = float(cell_phi[edge_a])
            value_b = float(cell_phi[edge_b])
            edge_points: list[np.ndarray] = []
            if value_a == 0.0:
                edge_points.append(cell_points[edge_a])
            if value_b == 0.0:
                edge_points.append(cell_points[edge_b])
            if value_a * value_b < 0.0:
                fraction = value_a / (value_a - value_b)
                edge_points.append(
                    (1.0 - fraction) * cell_points[edge_a] +
                    fraction * cell_points[edge_b])
            for candidate in edge_points:
                if not any(float(np.linalg.norm(candidate - existing)) <=
                           point_tolerance for existing in cut_points):
                    cut_points.append(np.asarray(candidate, dtype=float))
        if len(cut_points) < 2:
            continue
        farthest_a = 0
        farthest_b = 1
        farthest_distance = -1.0
        for index_a in range(len(cut_points)):
            for index_b in range(index_a + 1, len(cut_points)):
                candidate_distance = float(np.linalg.norm(
                    cut_points[index_b] - cut_points[index_a]))
                if candidate_distance > farthest_distance:
                    farthest_a = index_a
                    farthest_b = index_b
                    farthest_distance = candidate_distance
        if farthest_distance <= point_tolerance:
            continue
        tangent = cut_points[farthest_b] - cut_points[farthest_a]
        fragment_normal = np.asarray([tangent[1], -tangent[0]], dtype=float)
        fragment_normal /= float(np.linalg.norm(fragment_normal))
        affine_design = np.column_stack((
            cell_points[:, 0],
            cell_points[:, 1],
            np.ones(point_ids.size),
        ))
        affine_coefficients, *_ = np.linalg.lstsq(
            affine_design, cell_phi, rcond=None)
        affine_gradient = np.asarray(affine_coefficients[:2], dtype=float)
        affine_gradient_norm = float(np.linalg.norm(affine_gradient))
        if affine_gradient_norm <= 1.0e-30:
            continue
        affine_gradient /= affine_gradient_norm
        if float(np.dot(fragment_normal, affine_gradient)) < 0.0:
            fragment_normal = -fragment_normal
        active_normal = fragment_normal
        if active_domain == "LevelSetPositive":
            active_normal = -active_normal
        normal_dot_wall = float(np.dot(active_normal, wall_normal))
        dynamic_cos = -normal_dot_wall
        young_gap = target_cos - dynamic_cos
        wall_tangent = active_normal - normal_dot_wall * wall_normal
        wall_tangent_norm = float(np.linalg.norm(wall_tangent))
        q1_dynamic_cos = -float(np.dot(element_active_normal, wall_normal))
        root = shape @ cell_points
        sample: dict[str, Any] = {
            "cell_id": cell_id,
            "cell_type": "Triangle3" if is_triangle else "Quad4",
            "point": [float(root[0]), float(root[1])],
            "parent_coordinate": parent_coordinate,
            "dynamic_cos": dynamic_cos,
            "dynamic_angle_degrees": math.degrees(math.acos(
                min(1.0, max(-1.0, dynamic_cos)))),
            "young_gap": young_gap,
            "level_set_gradient_norm": gradient_norm,
            "q1_dynamic_cos": q1_dynamic_cos,
            "q1_dynamic_angle_degrees": math.degrees(math.acos(
                min(1.0, max(-1.0, q1_dynamic_cos)))),
            "generated_fragment_normal": [
                float(active_normal[0]), float(active_normal[1])],
            "wall_tangential_normal_norm": wall_tangent_norm,
        }
        if dynamic_model:
            sample["predicted_contact_line_speed"] = (
                float(mobility) * float(surface_tension) * young_gap)
        if velocity is not None and velocity.shape[1] >= 2 and wall_tangent_norm > 0.0:
            contact_velocity = shape @ velocity[point_ids, :2]
            footprint_direction = wall_tangent / wall_tangent_norm
            sample["contact_fluid_speed"] = float(np.dot(
                contact_velocity, footprint_direction))
        samples.append(sample)

    samples.sort(key=lambda sample: sample["point"][tangent_axis])
    unique_samples: list[dict[str, Any]] = []
    for sample in samples:
        if (unique_samples and abs(
                sample["point"][tangent_axis] -
                unique_samples[-1]["point"][tangent_axis]) <=
                wall_tolerance):
            continue
        unique_samples.append(sample)
    state["operator_contact_geometry_samples"] = unique_samples
    state["operator_contact_geometry_sample_count"] = len(unique_samples)
    if len(unique_samples) != 2:
        unavailable(
            f"expected two generated phi=0 wall roots, found {len(unique_samples)}")
        return

    def sample_mean(name: str) -> float:
        return float(np.mean([float(sample[name]) for sample in unique_samples]))

    state["operator_contact_geometry_available"] = True
    state["operator_contact_geometry_source"] = (
        "LinearCorner_generated_fragment_normal_at_phi_zero_wall_roots")
    state["operator_dynamic_cos_mean"] = sample_mean("dynamic_cos")
    state["operator_dynamic_angle_degrees_mean"] = sample_mean(
        "dynamic_angle_degrees")
    state["diagnostic_q1_dynamic_cos_mean"] = sample_mean("q1_dynamic_cos")
    state["diagnostic_q1_dynamic_angle_degrees_mean"] = sample_mean(
        "q1_dynamic_angle_degrees")
    state["operator_young_gap_mean"] = sample_mean("young_gap")
    state["operator_wall_tangential_normal_norm_min"] = min(
        float(sample["wall_tangential_normal_norm"])
        for sample in unique_samples)
    if all("predicted_contact_line_speed" in sample
           for sample in unique_samples):
        state["operator_predicted_contact_line_speed"] = sample_mean(
            "predicted_contact_line_speed")
    if all("contact_fluid_speed" in sample for sample in unique_samples):
        state["operator_contact_fluid_speed"] = sample_mean(
            "contact_fluid_speed")
        state["operator_contact_fluid_evaluation_source"] = (
            "Q1_velocity_and_generated_fragment_normal_at_phi_zero_wall_roots")


def add_production_physical_liquid_volume_metrics(
        metrics: dict[str, Any]) -> None:
    """Validate initialized-plus-accepted production physical cut volumes.

    VTK ``WetVolumeMeasure`` is deliberately not accepted here: it is an
    output serialization of one accepted state.  This history is reconstructed
    from the production cut context, requires the true initialized state plus
    every solver-accepted state, and is shared by the D18/D38 and capillary-wave
    qualifications.
    """

    prefix = "production_physical_liquid_volume"

    def unavailable(reason: str) -> None:
        metrics[f"{prefix}_available"] = False
        metrics[f"{prefix}_error"] = reason

    history = metrics.get("production_wet_volume_diagnostic_history")
    if not isinstance(history, list) or not history:
        unavailable("production wet-volume diagnostics were not reported")
        return
    if not all(isinstance(record, dict) for record in history):
        unavailable("invalid production wet-volume diagnostic record")
        return

    time_loop = metrics.get("time_loop")
    accepted = time_loop.get("accepted_steps") if isinstance(time_loop, dict) else None
    if not isinstance(accepted, list) or not accepted:
        unavailable("accepted-step clock is unavailable")
        return
    expected_steps: list[int] = []
    expected_times: list[float] = []
    transient = metrics.get("solver_controls", {}).get("transient_solve", {})
    t0 = transient.get("t0", 0.0) if isinstance(transient, dict) else 0.0
    if not isinstance(t0, (int, float)) or not math.isfinite(float(t0)):
        unavailable("transient initial time is unavailable")
        return
    accepted_steps: list[int] = []
    accepted_times: list[float] = []
    accepted_dts: list[float] = []
    for record in accepted:
        step = record.get("step") if isinstance(record, dict) else None
        time = record.get("time") if isinstance(record, dict) else None
        step_dt = record.get("dt") if isinstance(record, dict) else None
        if (not isinstance(step, int) or isinstance(step, bool) or step <= 0 or
                not isinstance(time, (int, float)) or
                not math.isfinite(float(time)) or
                not isinstance(step_dt, (int, float)) or
                isinstance(step_dt, bool) or
                not math.isfinite(float(step_dt)) or float(step_dt) <= 0.0):
            unavailable("accepted-step clock contains an invalid record")
            return
        accepted_steps.append(step)
        accepted_times.append(float(time))
        accepted_dts.append(float(step_dt))
    if accepted_steps != sorted(set(accepted_steps)):
        unavailable("accepted-step clock is duplicated or nonmonotone")
        return
    previous_time = float(t0)
    for step, time, step_dt in zip(
            accepted_steps, accepted_times, accepted_dts):
        elapsed = time - previous_time
        tolerance = 1.0e-12 * max(
            1.0, abs(time), abs(previous_time), abs(step_dt))
        if elapsed <= 0.0 or abs(elapsed - step_dt) > tolerance:
            unavailable(
                f"accepted-step time increment for step {step} does not match dt")
            return
        previous_time = time
    # TimeHistory advances the integer step exactly once for an accepted
    # state.  Deriving the pre-loop index from the first accepted record keeps
    # this gate valid for restart/resume runs instead of silently assuming 0.
    expected_steps = [accepted_steps[0] - 1, *accepted_steps]
    expected_times = [float(t0), *accepted_times]
    expected_dts = [0.0, *accepted_dts]

    key_names = ("field", "domain_id", "marker", "active_side", "isovalue")
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for record in history:
        key = tuple(record.get(name) for name in key_names)
        grouped.setdefault(key, []).append(record)
    candidates = [
        (key, records)
        for key, records in grouped.items()
        if [record.get("step") for record in records] == expected_steps
    ]
    if len(candidates) != 1:
        available_steps = {
            repr(key): [record.get("step") for record in records]
            for key, records in grouped.items()
        }
        unavailable(
            "expected exactly one production wet-volume history containing "
            f"initial and every accepted state; expected_steps={expected_steps!r} "
            f"available={available_steps!r}")
        return

    selected_key, selected = candidates[0]
    volumes: list[float] = []
    validated_history: list[dict[str, Any]] = []
    for index, (record, expected_step, expected_time, expected_dt) in enumerate(
            zip(selected, expected_steps, expected_times, expected_dts)):
        step = record.get("step")
        time = record.get("time")
        wet_volume = record.get("wet_volume")
        physical_volume = record.get("physical_wet_volume")
        initial_volume = record.get("initial_wet_volume")
        frame = record.get("wet_volume_frame")
        rule_count = record.get("volume_rule_count")
        physical_rule_count = record.get("physical_volume_rule_count")
        skipped_rule_count = record.get("skipped_physical_volume_rule_count")
        numeric = (time, wet_volume, physical_volume, initial_volume)
        if (step != expected_step or
                not all(isinstance(value, (int, float)) and
                        not isinstance(value, bool) and
                        math.isfinite(float(value)) for value in numeric)):
            unavailable(
                f"production wet-volume record {index} is incomplete or non-finite")
            return
        time_scale = max(1.0, abs(expected_time))
        if abs(float(time) - expected_time) > 1.0e-12 * time_scale:
            unavailable(
                f"production wet-volume time for step {expected_step} does not "
                "match the accepted-step clock")
            return
        if frame != "physical":
            unavailable(
                f"production wet-volume record at step {expected_step} uses "
                f"nonphysical frame {frame!r}")
            return
        if (not isinstance(rule_count, int) or isinstance(rule_count, bool) or
                rule_count <= 0 or
                not isinstance(physical_rule_count, int) or
                isinstance(physical_rule_count, bool) or
                physical_rule_count != rule_count or
                not isinstance(skipped_rule_count, int) or
                isinstance(skipped_rule_count, bool) or
                skipped_rule_count != 0):
            unavailable(
                f"production wet-volume record at step {expected_step} lacks "
                "a complete physical rule set")
            return
        physical = float(physical_volume)
        if abs(float(wet_volume) - physical) > (
                1.0e-12 * max(1.0, abs(physical))):
            unavailable(
                f"selected wet volume at step {expected_step} disagrees with "
                "the physical cut volume")
            return
        volumes.append(physical)
        validated_history.append({
            "step": expected_step,
            "time": expected_time,
            "dt": expected_dt,
            "physical_liquid_volume": physical,
            "initial_physical_liquid_volume": float(initial_volume),
            "wet_volume_frame": frame,
            "volume_rule_count": rule_count,
            "physical_volume_rule_count": physical_rule_count,
            "skipped_physical_volume_rule_count": skipped_rule_count,
            "state_stage": (
                "initialized" if index == 0 else
                "accepted_post_level_set_maintenance"
            ),
        })

    reference_step = expected_steps[0]
    reference_time = expected_times[0]
    reference_volume = volumes[0]
    if not reference_volume > 0.0:
        unavailable("initial production physical liquid volume is not positive")
        return
    for record in selected:
        if abs(float(record["initial_wet_volume"]) - reference_volume) > (
                1.0e-12 * max(1.0, abs(reference_volume))):
            unavailable("production wet-volume baseline changed during the run")
            return

    final_step = expected_steps[-1]
    final_time = expected_times[-1]
    final_volume = volumes[-1]
    scale = max(abs(reference_volume), 1.0e-300)
    drifts = [volume - reference_volume for volume in volumes]
    maximum_absolute_drift = max(abs(value) for value in drifts)
    for record, drift in zip(validated_history, drifts):
        record["signed_volume_drift"] = drift
        record["relative_volume_drift"] = drift / scale

    metrics[f"{prefix}_available"] = True
    metrics[f"{prefix}_source"] = (
        "production_physical_cut_context_diagnostic")
    metrics[f"{prefix}_history_key"] = dict(
        zip(key_names, selected_key))
    metrics[f"{prefix}_history"] = validated_history
    metrics[f"{prefix}_state_count"] = len(selected)
    metrics[f"{prefix}_reference_step"] = reference_step
    metrics[f"{prefix}_reference_time"] = reference_time
    metrics[f"{prefix}_reference"] = reference_volume
    metrics[f"{prefix}_final_step"] = final_step
    metrics[f"{prefix}_final_time"] = final_time
    metrics[f"{prefix}_final"] = final_volume
    metrics[f"{prefix}_signed_drift"] = (
        final_volume - reference_volume)
    metrics[f"{prefix}_relative_drift"] = (
        abs(final_volume - reference_volume) / scale)
    metrics[f"{prefix}_max_absolute_drift"] = (
        maximum_absolute_drift)
    metrics[f"{prefix}_max_relative_drift"] = (
        maximum_absolute_drift / scale)


def add_capillary_wave_temporal_liquid_volume_metrics(
        metrics: dict[str, Any]) -> None:
    """Expose the shared production-volume validation under wave gate names."""
    add_production_physical_liquid_volume_metrics(metrics)
    source_prefix = "production_physical_liquid_volume"
    target_prefix = "capillary_wave_temporal_liquid_volume"
    if metrics.get(f"{source_prefix}_available") is not True:
        metrics[f"{target_prefix}_available"] = False
        metrics[f"{target_prefix}_error"] = metrics.get(
            f"{source_prefix}_error",
            "production physical liquid-volume history is unavailable",
        )
        return

    metrics[f"{target_prefix}_available"] = True
    for suffix in (
            "source", "history_key", "state_count", "reference_step",
            "reference_time", "reference", "final_step", "final_time",
            "final", "signed_drift", "relative_drift",
            "max_absolute_drift", "max_relative_drift"):
        metrics[f"{target_prefix}_{suffix}"] = metrics[
            f"{source_prefix}_{suffix}"]


def sessile_state_metrics(dataset: pv.DataSet,
                           benchmark: dict[str, Any]) -> dict[str, Any]:
    contact = benchmark.get("sessile_contact", {})
    if not isinstance(contact, dict):
        return {"available": False, "error": "missing sessile contact metadata"}
    try:
        wall_frame = sessile_contact_wall_frame(contact)
    except ValueError as exc:
        return {"available": False, "error": str(exc)}
    wall_coordinate = float(wall_frame["wall_coordinate"])
    wall_axis = int(wall_frame["wall_axis"])
    state = fit_sessile_interface(
        dataset,
        wall_coordinate,
        wall_axis=wall_axis,
        wall_normal=np.asarray(wall_frame["wall_normal"], dtype=float),
    )
    if not state.get("available"):
        return state
    state["wall_face"] = wall_frame["wall_face"]
    add_sessile_contact_fluid_speed(
        state, dataset, wall_coordinate, wall_axis)
    add_sessile_operator_contact_geometry(state, dataset, benchmark)

    if "phi" in dataset.point_data:
        active_domain = benchmark_active_domain(benchmark)
        phi = active_signed_level_set(
            np.asarray(dataset.point_data["phi"], dtype=float).reshape(-1),
            active_domain,
        )
        state["active_domain"] = active_domain
        finite_phi = np.isfinite(phi)
        h = benchmark.get("mesh_resolution", {}).get("h", 0.0)
        band = 0.5 * float(h) if isinstance(h, (int, float)) else 0.0
        liquid = finite_phi & (phi < -band)
        gas = finite_phi & (phi > band)
        if "Velocity" in dataset.point_data and np.any(liquid):
            velocity = np.asarray(dataset.point_data["Velocity"], dtype=float)
            if velocity.ndim == 1:
                velocity = velocity.reshape((-1, 1))
            speed = np.linalg.norm(velocity, axis=1)
            strict_interior_max = float(np.nanmax(speed[liquid]))
            state["max_strict_interior_liquid_nodal_speed"] = (
                strict_interior_max)
            state["mean_liquid_speed"] = float(np.nanmean(speed[liquid]))
            # A static wetting equilibrium must be at rest at the contact line
            # as well as at vertices at least h/2 inside the liquid.  The old
            # deep-interior-only maximum could hide the largest physical
            # velocity exactly where the Ren--E law acts.  Include every
            # active-sign liquid vertex and the production-geometry contact
            # interpolation in the capillary-number observable.  Keep each
            # component explicit so the strengthened maximum cannot be
            # mistaken for a dry-extension coefficient norm.
            speed_candidates = [
                ("strict_interior_liquid_vertex", strict_interior_max),
            ]
            active_vertices = finite_phi & (phi <= 0.0)
            if np.any(active_vertices):
                active_vertex_max = float(np.nanmax(speed[active_vertices]))
                state["max_active_side_liquid_nodal_speed"] = (
                    active_vertex_max)
                speed_candidates.append(
                    ("active_side_liquid_vertex", active_vertex_max))
            contact_samples = state.get(
                "operator_contact_geometry_samples", [])
            if isinstance(contact_samples, list):
                contact_speeds = [
                    abs(float(sample["contact_fluid_speed"]))
                    for sample in contact_samples
                    if isinstance(sample, dict) and
                    isinstance(sample.get("contact_fluid_speed"),
                               (int, float)) and
                    math.isfinite(float(sample["contact_fluid_speed"]))
                ]
                if contact_speeds:
                    contact_max = max(contact_speeds)
                    state["max_generated_contact_fluid_speed"] = contact_max
                    speed_candidates.append(
                        ("generated_contact_fluid_interpolation", contact_max))
            maximum_source, maximum_speed = max(
                speed_candidates, key=lambda item: item[1])
            state["max_liquid_speed"] = maximum_speed
            state["max_liquid_speed_source"] = maximum_source
            state["max_liquid_speed_contract"] = (
                "maximum_of_strict_interior_and_active_side_liquid_vertex_"
                "speeds_and_generated_contact_fluid_interpolation")
        if "Pressure" in dataset.point_data:
            pressure = np.asarray(dataset.point_data["Pressure"], dtype=float).reshape(-1)
            if pressure.shape[0] == phi.shape[0]:
                if np.any(liquid):
                    state["liquid_pressure_median"] = float(np.nanmedian(pressure[liquid]))
                if np.any(gas):
                    state["gas_pressure_median"] = float(np.nanmedian(pressure[gas]))
                if ("liquid_pressure_median" in state and
                        "gas_pressure_median" in state):
                    state["pressure_jump"] = (
                        state["liquid_pressure_median"] - state["gas_pressure_median"]
                    )

    wet_volume, source = dataset_wet_volume(dataset)
    if wet_volume is not None:
        state["liquid_area"] = wet_volume
        state["liquid_area_source"] = source
    return state


def add_capillary_output_pressure_metrics(metrics: dict[str, Any],
                                          benchmark: dict[str, Any],
                                          output: pv.DataSet) -> None:
    radius = benchmark.get("capillary_radius", benchmark.get("capillary_arc_radius"))
    if not isinstance(radius, (int, float)) or float(radius) <= 0.0:
        return
    if "phi" not in output.point_data or "Pressure" not in output.point_data:
        return
    active_domain = benchmark_active_domain(benchmark)
    phi = active_signed_level_set(
        np.asarray(output.point_data["phi"], dtype=float).reshape(-1),
        active_domain,
    )
    pressure = np.asarray(output.point_data["Pressure"], dtype=float).reshape(-1)
    if phi.shape[0] != pressure.shape[0]:
        return
    spacing = coordinate_min_spacing(np.asarray(output.points, dtype=float))
    band = 0.5 * spacing if spacing is not None else 0.0
    finite = np.isfinite(phi) & np.isfinite(pressure)
    liquid = finite & (phi < -band)
    gas = finite & (phi > band)
    if not np.any(liquid) or not np.any(gas):
        return
    liquid_pressure = float(np.median(pressure[liquid]))
    gas_pressure = float(np.median(pressure[gas]))
    metrics["capillary_final_liquid_pressure_median"] = liquid_pressure
    metrics["capillary_final_gas_pressure_median"] = gas_pressure
    metrics["capillary_final_pressure_jump"] = liquid_pressure - gas_pressure
    metrics["capillary_final_pressure_sample_band"] = float(band)
    metrics["capillary_final_liquid_pressure_samples"] = int(np.count_nonzero(liquid))
    metrics["capillary_final_gas_pressure_samples"] = int(np.count_nonzero(gas))


def boundary_face_point_indices(case_dir: Path,
                                initial: pv.DataSet) -> dict[str, np.ndarray]:
    if "GlobalNodeID" not in initial.point_data:
        return {}
    initial_gids = np.asarray(initial.point_data["GlobalNodeID"], dtype=np.int64).reshape(-1)
    gid_to_index = {int(gid): index for index, gid in enumerate(initial_gids)}
    result: dict[str, np.ndarray] = {}
    surface_dir = case_dir / "mesh/background/mesh-surfaces"
    surface_paths = sorted({
        *surface_dir.glob("wall_*.vtp"),
        *surface_dir.glob("wall_*.vtu"),
    })
    for path in surface_paths:
        surface = pv.read(path)
        if "GlobalNodeID" not in surface.point_data:
            continue
        indices = [
            gid_to_index[int(gid)]
            for gid in np.asarray(surface.point_data["GlobalNodeID"]).reshape(-1)
            if int(gid) in gid_to_index
        ]
        if indices:
            prior = result.get(path.stem, np.asarray([], dtype=np.int64))
            result[path.stem] = np.unique(np.concatenate((
                prior,
                np.asarray(indices, dtype=np.int64),
            )))
    return result


def wall_false_wet_applicability(
        case_dir: Path,
        initial: pv.DataSet,
        wall_indices: dict[str, np.ndarray],
) -> dict[str, Any]:
    """Classify whether initially dry wall vertices must be monitored.

    A continuous P1 interface cannot meet a boundary face when all boundary
    vertices are strictly on one side of zero.  The closed-interface
    exemption additionally requires both signs in the volume mesh, so a
    missing or one-phase level-set field can never be mistaken for evidence
    that the wall symptom is inapplicable.
    """
    evidence: dict[str, Any] = {
        "wall_only_false_wet_applicability": "indeterminate",
        "wall_only_false_wet_closed_interface_certified": False,
        "wall_only_false_wet_boundary_surface_count": len(wall_indices),
    }
    try:
        solver_root = ET.parse(case_dir / "solver.xml").getroot()
        declared_walls = sorted({
            element.attrib["name"]
            for element in solver_root.findall(".//Add_face")
            if element.attrib.get("name", "").startswith("wall_")
        })
    except (OSError, ET.ParseError):
        declared_walls = []
    observed_walls = sorted(wall_indices)
    missing_walls = sorted(set(declared_walls) - set(observed_walls))
    unexpected_walls = sorted(set(observed_walls) - set(declared_walls))
    wall_coverage_complete = bool(
        declared_walls and not missing_walls and not unexpected_walls)
    evidence.update({
        "wall_only_false_wet_declared_boundary_names": declared_walls,
        "wall_only_false_wet_observed_boundary_names": observed_walls,
        "wall_only_false_wet_missing_boundary_names": missing_walls,
        "wall_only_false_wet_unexpected_boundary_names": unexpected_walls,
        "wall_only_false_wet_boundary_coverage_complete": (
            wall_coverage_complete),
    })
    if not wall_indices:
        evidence["wall_only_false_wet_applicability_reason"] = (
            "no_readable_wall_surface_global_node_ids")
        return evidence
    if "phi" not in initial.point_data:
        evidence["wall_only_false_wet_applicability_reason"] = (
            "initial_level_set_field_unavailable")
        return evidence

    phi = np.asarray(initial.point_data["phi"], dtype=float).reshape(-1)
    if phi.size != initial.n_points or not np.isfinite(phi).all():
        evidence["wall_only_false_wet_applicability_reason"] = (
            "initial_level_set_field_invalid")
        return evidence
    wall_points = np.unique(np.concatenate([
        np.asarray(indices, dtype=np.int64).reshape(-1)
        for indices in wall_indices.values()
    ]))
    if (wall_points.size == 0 or int(wall_points[0]) < 0 or
            int(wall_points[-1]) >= initial.n_points):
        evidence["wall_only_false_wet_applicability_reason"] = (
            "wall_point_indices_invalid")
        return evidence

    spacing = coordinate_min_spacing(np.asarray(initial.points, dtype=float))
    tolerance = (
        max(1.0e-10, 1.0e-3 * spacing)
        if spacing is not None else 1.0e-10
    )
    wall_phi = phi[wall_points]
    domain_min = float(np.min(phi))
    domain_max = float(np.max(phi))
    boundary_min = float(np.min(wall_phi))
    boundary_max = float(np.max(wall_phi))
    evidence.update({
        "wall_only_false_wet_boundary_point_count": int(wall_points.size),
        "wall_only_false_wet_initial_domain_phi_min": domain_min,
        "wall_only_false_wet_initial_domain_phi_max": domain_max,
        "wall_only_false_wet_initial_boundary_phi_min": boundary_min,
        "wall_only_false_wet_initial_boundary_phi_max": boundary_max,
        "wall_only_false_wet_initial_boundary_phi_min_abs": float(
            np.min(np.abs(wall_phi))),
        "wall_only_false_wet_level_set_tolerance": tolerance,
    })
    two_phase_domain = domain_min < -tolerance and domain_max > tolerance
    boundary_strictly_positive = boundary_min > tolerance
    boundary_strictly_negative = boundary_max < -tolerance
    if (two_phase_domain and wall_coverage_complete and
            (boundary_strictly_positive or boundary_strictly_negative)):
        evidence.update({
            "wall_only_false_wet_applicability": (
                "not_applicable_closed_interface"),
            "wall_only_false_wet_applicability_reason": (
                "two_phase_P1_field_has_uniform_nonzero_boundary_sign"),
            "wall_only_false_wet_closed_interface_certified": True,
            "wall_only_false_wet_initial_boundary_sign": (
                "positive" if boundary_strictly_positive else "negative"),
        })
        return evidence

    if (two_phase_domain and
            (boundary_strictly_positive or boundary_strictly_negative)):
        evidence["wall_only_false_wet_applicability_reason"] = (
            "uniform_boundary_sign_but_declared_wall_coverage_is_incomplete")
        return evidence

    evidence.update({
        "wall_only_false_wet_applicability": (
            "applicable_interface_may_contact_boundary"),
        "wall_only_false_wet_applicability_reason": (
            "boundary_sign_is_not_uniformly_separated_from_zero"),
    })
    return evidence


def inward_cell_centroid_stencils_by_wall(
        initial: pv.DataSet,
        wall_indices: dict[str, np.ndarray],
) -> tuple[
        dict[tuple[str, int], list[dict[str, Any]]],
        list[dict[str, Any]],
]:
    """Build production-independent first-order cell-interior wall stencils.

    Every incident Tetra4 centroid is strictly inside its nondegenerate volume
    cell and evaluates a continuous-P1 field with four equal nodal weights.
    This remains valid when a one-element-thick extrusion has no interior mesh
    vertex.  The explicitly 2D qualification decks may use Triangle3/P1 or
    Quad4/Q1 cells; their reference-cell centers use equal nodal weights.  A
    mesh containing any volume cell is strictly Tetra4-only at every monitored
    wall vertex, so unsupported Hex8 and mixed volume cells still fail closed.
    """
    monitored = {
        int(index)
        for indices in wall_indices.values()
        for index in indices
    }
    points = np.asarray(initial.points, dtype=float)
    errors: list[dict[str, Any]] = []
    try:
        gids = global_node_ids(initial).reshape(-1)
    except ValueError as exc:
        return {}, [{"reason": "missing_initial_global_node_ids",
                     "detail": str(exc)}]
    if (gids.size != initial.n_points or len(set(map(int, gids))) != gids.size):
        return {}, [{"reason": "invalid_initial_global_node_ids"}]
    if points.shape != (initial.n_points, 3) or not np.isfinite(points).all():
        return {}, [{"reason": "invalid_initial_point_coordinates"}]

    connectivity = np.asarray(initial.cells, dtype=np.int64).reshape(-1)
    cell_types = np.asarray(initial.celltypes, dtype=np.int64).reshape(-1)
    tetra_type = int(pv.CellType.TETRA)
    triangle_type = int(pv.CellType.TRIANGLE)
    quad_type = int(pv.CellType.QUAD)
    surface_types = np.asarray([triangle_type, quad_type], dtype=np.int64)
    has_volume_cells = bool(np.any(~np.isin(cell_types, surface_types)))
    supported_name = (
        "Tetra4" if has_volume_cells else "Triangle3 or Quad4")

    incident: dict[int, list[dict[str, Any]]] = {
        point_index: [] for point_index in monitored
    }
    cursor = 0
    cell_id = 0
    while cursor < connectivity.size:
        if cell_id >= cell_types.size:
            errors.append({
                "reason": "cell_connectivity_exceeds_cell_type_count",
                "cell_id": cell_id,
            })
            break
        count = int(connectivity[cursor])
        end = cursor + 1 + count
        if count <= 0 or end > connectivity.size:
            errors.append({
                "reason": "invalid_cell_connectivity",
                "cell_id": cell_id,
                "vertex_count": count,
            })
            break
        cell_points = tuple(int(value) for value in connectivity[cursor + 1:end])
        touched = monitored.intersection(cell_points)
        if touched:
            cell_type = int(cell_types[cell_id])
            supported_cell = (
                (has_volume_cells and cell_type == tetra_type and count == 4) or
                (not has_volume_cells and
                 ((cell_type == triangle_type and count == 3) or
                  (cell_type == quad_type and count == 4)))
            )
            if not supported_cell:
                for point_index in sorted(touched):
                    errors.append({
                        "point_index": point_index,
                        "global_node_id": int(gids[point_index]),
                        "cell_id": cell_id,
                        "reason": "unsupported_incident_cell",
                        "expected_cell": supported_name,
                        "cell_type": cell_type,
                        "vertex_count": count,
                    })
            elif (len(set(cell_points)) != count or
                  min(cell_points) < 0 or max(cell_points) >= initial.n_points):
                for point_index in sorted(touched):
                    errors.append({
                        "point_index": point_index,
                        "global_node_id": int(gids[point_index]),
                        "cell_id": cell_id,
                        "reason": "invalid_incident_cell_point_ids",
                    })
            else:
                cell_coordinates = points[list(cell_points)]
                edge_scale = max(
                    float(np.linalg.norm(cell_coordinates[i] - cell_coordinates[j]))
                    for i in range(count)
                    for j in range(i)
                )
                mapping_valid = True
                if cell_type == tetra_type:
                    jacobian = abs(float(np.dot(
                        cell_coordinates[1] - cell_coordinates[0],
                        np.cross(
                            cell_coordinates[2] - cell_coordinates[0],
                            cell_coordinates[3] - cell_coordinates[0],
                        ),
                    )))
                    degeneracy_scale = edge_scale ** 3
                elif cell_type == triangle_type:
                    jacobian = float(np.linalg.norm(np.cross(
                        cell_coordinates[1] - cell_coordinates[0],
                        cell_coordinates[2] - cell_coordinates[0],
                    )))
                    degeneracy_scale = edge_scale ** 2
                else:
                    # Quad4/Q1 shape derivatives at the reference center.
                    # Equal nodal weights therefore evaluate the actual Q1
                    # field at a point strictly inside a regular mapped cell.
                    dxi = 0.25 * (
                        -cell_coordinates[0] + cell_coordinates[1] +
                        cell_coordinates[2] - cell_coordinates[3])
                    deta = 0.25 * (
                        -cell_coordinates[0] - cell_coordinates[1] +
                        cell_coordinates[2] + cell_coordinates[3])
                    center_cross = np.cross(dxi, deta)
                    center_measure = float(np.linalg.norm(center_cross))
                    degeneracy_scale = edge_scale ** 2
                    orientation_tolerance = (
                        128.0 * np.finfo(float).eps * degeneracy_scale)
                    if (not math.isfinite(center_measure) or
                            center_measure <= orientation_tolerance):
                        mapping_valid = False
                        jacobian = center_measure
                    else:
                        center_normal = center_cross / center_measure
                        corner_projections: list[float] = []
                        for xi, eta in (
                                (-1.0, -1.0), (1.0, -1.0),
                                (1.0, 1.0), (-1.0, 1.0)):
                            dshape_dxi = 0.25 * np.asarray([
                                -(1.0 - eta), 1.0 - eta,
                                1.0 + eta, -(1.0 + eta),
                            ])
                            dshape_deta = 0.25 * np.asarray([
                                -(1.0 - xi), -(1.0 + xi),
                                1.0 + xi, 1.0 - xi,
                            ])
                            tangent_xi = dshape_dxi @ cell_coordinates
                            tangent_eta = dshape_deta @ cell_coordinates
                            corner_projections.append(float(np.dot(
                                np.cross(tangent_xi, tangent_eta),
                                center_normal,
                            )))
                        jacobian = min(corner_projections)
                        mapping_valid = bool(
                            all(math.isfinite(value) and
                                value > orientation_tolerance
                                for value in corner_projections))
                degeneracy_tolerance = (
                    128.0 * np.finfo(float).eps * degeneracy_scale)
                centroid = np.mean(cell_coordinates, axis=0)
                if (not math.isfinite(edge_scale) or edge_scale <= 0.0 or
                        not mapping_valid or not math.isfinite(jacobian) or
                        jacobian <= degeneracy_tolerance or
                        not np.isfinite(centroid).all()):
                    for point_index in sorted(touched):
                        errors.append({
                            "point_index": point_index,
                            "global_node_id": int(gids[point_index]),
                            "cell_id": cell_id,
                            "reason": "nondegenerate_cell_interior_unavailable",
                            "jacobian_measure": jacobian,
                            "degeneracy_tolerance": degeneracy_tolerance,
                        })
                else:
                    stencil = {
                        "cell_id": cell_id,
                        "point_indices": list(cell_points),
                        "global_node_ids": [int(gids[index]) for index in cell_points],
                        "weights": [1.0 / count] * count,
                        "centroid": [float(value) for value in centroid],
                        "cell_type": (
                            "Tetra4" if cell_type == tetra_type else
                            "Triangle3" if cell_type == triangle_type else
                            "Quad4"),
                    }
                    for point_index in touched:
                        incident[point_index].append(stencil)
        cursor = end
        cell_id += 1
    if cursor == connectivity.size and cell_id != cell_types.size:
        errors.append({
            "reason": "cell_type_count_exceeds_connectivity_count",
            "cell_type_count": int(cell_types.size),
            "connectivity_cell_count": cell_id,
        })

    result: dict[tuple[str, int], list[dict[str, Any]]] = {}
    error_points = {
        int(error["point_index"])
        for error in errors
        if isinstance(error.get("point_index"), int)
    }
    for wall_name, indices in wall_indices.items():
        for raw_index in indices:
            point_index = int(raw_index)
            candidates = incident.get(point_index, [])
            if point_index in error_points:
                continue
            if not candidates:
                errors.append({
                    "wall": wall_name,
                    "point_index": point_index,
                    "global_node_id": int(gids[point_index]),
                    "reason": "no_incident_cell_interior_stencil",
                })
                continue
            result[(wall_name, point_index)] = sorted(
                candidates, key=lambda stencil: int(stencil["cell_id"]))
    return result, errors


def accepted_step_clock(
        accepted_steps: list[dict[str, Any]] | None,
) -> tuple[dict[int, tuple[float, float]], list[str]]:
    """Index solver-reported accepted times and step sizes by output step."""
    clock: dict[int, tuple[float, float]] = {}
    errors: list[str] = []
    if accepted_steps is None:
        return clock, errors

    previous_step: int | None = None
    previous_time: float | None = None
    for record in accepted_steps:
        step = record.get("step") if isinstance(record, dict) else None
        time = record.get("time") if isinstance(record, dict) else None
        step_dt = record.get("dt") if isinstance(record, dict) else None
        if (not isinstance(step, int) or isinstance(step, bool) or step <= 0 or
                not isinstance(time, (int, float)) or
                not isinstance(step_dt, (int, float)) or
                not math.isfinite(float(time)) or
                not math.isfinite(float(step_dt)) or float(step_dt) <= 0.0):
            errors.append(f"invalid accepted-step clock record: {record!r}")
            continue
        value = (float(time), float(step_dt))
        if previous_step is not None and step <= previous_step:
            errors.append(
                "accepted-step clock is duplicated or nonmonotone at "
                f"step {step} after {previous_step}"
            )
        if previous_time is not None:
            if value[0] <= previous_time:
                errors.append(
                    "accepted-step time is nonmonotone at "
                    f"step {step}: {value[0]!r} after {previous_time!r}"
                )
            elapsed = value[0] - previous_time
            tolerance = 1.0e-12 * max(
                1.0, abs(value[0]), abs(previous_time), abs(value[1]))
            if abs(elapsed - value[1]) > tolerance:
                errors.append(
                    f"accepted-step dt for step {step} ({value[1]!r}) does not "
                    f"match the accepted time increment ({elapsed!r})"
                )
        previous = clock.get(step)
        if previous is not None and previous != value:
            errors.append(
                f"conflicting accepted-step clock records for step {step}: "
                f"{previous!r} and {value!r}"
            )
            continue
        clock[step] = value
        previous_step = step
        previous_time = value[0]
    return clock, errors


def physical_history_stamp(
        step: int,
        clock: dict[int, tuple[float, float]],
        nominal_dt: float,
) -> tuple[dict[str, int | float], bool]:
    """Return a history stamp and whether it came from the accepted-step log."""
    if step == 0:
        return {"step": 0, "time": 0.0, "dt": 0.0}, True
    accepted = clock.get(step)
    if accepted is not None:
        return {
            "step": step,
            "time": accepted[0],
            "dt": accepted[1],
        }, True
    return {
        "step": step,
        "time": step * nominal_dt,
        "dt": nominal_dt,
    }, False


def add_free_surface_energy_history_metrics(
        metrics: dict[str, Any],
        benchmark: dict[str, Any],
        initial: pv.DataSet,
        paths: list[tuple[int, Path]],
        clock: dict[int, tuple[float, float]],
        clock_errors: list[str],
) -> None:
    """Record an initial-plus-every-accepted output-space energy history.

    This is deliberately fail-closed and remains an output diagnostic: the
    kinetic density and Q1 contour are not the assembled quadrature-exact
    energy, so the result cannot be presented as a discrete energy theorem.
    """

    def unavailable(reason: str) -> None:
        metrics["free_surface_energy_history_available"] = False
        metrics["free_surface_energy_history_error"] = reason

    sessile = benchmark.get("sessile_contact")
    wave = benchmark.get("capillary_wave")
    equilibrium_angle: float | None = None
    wall_coordinate = 0.0
    wall_axis = 1
    spatial_dimension = benchmark.get("spatial_dimension", 2)
    if (not isinstance(spatial_dimension, int) or
            isinstance(spatial_dimension, bool) or
            spatial_dimension not in {2, 3}):
        unavailable("free-surface energy spatial dimension must be 2 or 3")
        return
    if isinstance(sessile, dict):
        density = benchmark.get("density")
        surface_tension = benchmark.get("surface_tension")
        equilibrium_angle = sessile.get("equilibrium_contact_angle_degrees")
        if spatial_dimension == 3:
            wall_coordinate = sessile.get("wall_coordinate")
            wall_axis = sessile.get("wall_axis")
        else:
            try:
                wall_frame = sessile_contact_wall_frame(sessile)
            except ValueError as exc:
                unavailable(str(exc))
                return
            wall_coordinate = wall_frame["wall_coordinate"]
            wall_axis = wall_frame["wall_axis"]
        energy_case = "sessile_contact"
    elif isinstance(wave, dict):
        density = wave.get("density")
        surface_tension = wave.get("surface_tension")
        energy_case = "capillary_wave"
    elif (spatial_dimension == 2 and
          benchmark.get("capillary_geometry") == "droplet2d"):
        density = benchmark.get("density")
        surface_tension = benchmark.get("surface_tension")
        energy_case = "closed_circle"
    elif (spatial_dimension == 3 and
          benchmark.get("capillary_geometry") == "sphere_3d"):
        density = benchmark.get("density")
        surface_tension = benchmark.get("surface_tension")
        energy_case = "closed_sphere"
    else:
        return

    try:
        active_domain = benchmark_active_domain(benchmark)
    except ValueError as exc:
        unavailable(str(exc))
        return

    numeric_parameters = (density, surface_tension, wall_coordinate)
    if (not all(isinstance(value, (int, float)) and
                not isinstance(value, bool) and math.isfinite(float(value))
                for value in numeric_parameters) or
            float(density) <= 0.0 or float(surface_tension) <= 0.0):
        unavailable("free-surface energy parameters are unavailable or invalid")
        return
    if (not isinstance(wall_axis, int) or isinstance(wall_axis, bool) or
            wall_axis < 0 or wall_axis >= spatial_dimension):
        unavailable("free-surface energy wall axis is unavailable or invalid")
        return
    if isinstance(sessile, dict) and (
            not isinstance(equilibrium_angle, (int, float)) or
            isinstance(equilibrium_angle, bool) or
            not math.isfinite(float(equilibrium_angle))):
        unavailable("sessile equilibrium angle is unavailable or invalid")
        return
    if clock_errors:
        unavailable(
            "accepted-step clock is invalid: " + "; ".join(clock_errors))
        return
    if not clock:
        unavailable("accepted-step clock is unavailable")
        return

    accepted_step_ids = sorted(clock)
    output_step_ids = [step for step, _path in paths]
    if output_step_ids != accepted_step_ids:
        unavailable(
            "energy history requires one VTK state for every accepted step; "
            f"accepted={accepted_step_ids!r} output={output_step_ids!r}")
        return

    history: list[dict[str, Any]] = []
    try:
        energy_function = (
            free_surface_energy_state_3d
            if spatial_dimension == 3 else free_surface_energy_state_2d
        )
        initial_energy = energy_function(
            initial,
            density=float(density),
            surface_tension=float(surface_tension),
            equilibrium_contact_angle_degrees=(
                None if equilibrium_angle is None else
                float(equilibrium_angle)),
            wall_axis=int(wall_axis),
            wall_coordinate=float(wall_coordinate),
            active_domain=active_domain,
        )
        history.append({
            **initial_energy,
            "step": 0,
            "time": 0.0,
            "dt": 0.0,
            "state_source": "initialized_mesh",
        })
        for step, path in paths:
            time_value, dt_value = clock[step]
            state_energy = energy_function(
                pv.read(path),
                density=float(density),
                surface_tension=float(surface_tension),
                equilibrium_contact_angle_degrees=(
                    None if equilibrium_angle is None else
                    float(equilibrium_angle)),
                wall_axis=int(wall_axis),
                wall_coordinate=float(wall_coordinate),
                active_domain=active_domain,
            )
            history.append({
                **state_energy,
                "step": step,
                "time": time_value,
                "dt": dt_value,
                "state_source": str(path),
            })
        summary = summarize_energy_history(history)
    except Exception as exc:
        unavailable(f"free-surface energy history evaluation failed: {exc}")
        return

    metrics["free_surface_energy_history_available"] = True
    metrics["free_surface_energy_history_case"] = energy_case
    metrics["free_surface_energy_history"] = history
    metrics["free_surface_energy_summary"] = summary
    metrics["free_surface_energy_discrete_theorem_claimed"] = False
    for name, value in summary.items():
        metrics[f"free_surface_energy_{name}"] = value


def add_physical_time_history_metrics(metrics: dict[str, Any],
                                      case_dir: Path,
                                      benchmark: dict[str, Any],
                                      initial: pv.DataSet,
                                      accepted_steps: list[dict[str, Any]] | None = None,
                                      transient_solve: dict[str, Any] | None = None,
                                      ) -> None:
    paths = result_time_series_paths(case_dir)
    metrics["physical_time_history_output_count"] = len(paths)
    clock, clock_errors = accepted_step_clock(accepted_steps)
    accepted_step_ids = list(clock)
    output_step_ids = [step for step, _path in paths]
    missing_output_steps = sorted(set(accepted_step_ids) - set(output_step_ids))
    unexpected_output_steps = sorted(set(output_step_ids) - set(accepted_step_ids))
    exact_output_identity = output_step_ids == accepted_step_ids
    metrics["physical_time_history_accepted_step_ids"] = accepted_step_ids
    metrics["physical_time_history_output_step_ids"] = output_step_ids
    metrics["physical_time_history_missing_output_step_ids"] = (
        missing_output_steps)
    metrics["physical_time_history_unexpected_output_step_ids"] = (
        unexpected_output_steps)
    metrics["physical_time_history_output_step_identity_complete"] = (
        exact_output_identity)
    metrics["physical_time_history_clock_source"] = (
        "solver_accepted_steps" if accepted_steps is not None else
        "nominal_solver_dt"
    )
    metrics["physical_time_history_clock_errors"] = clock_errors
    if not paths:
        metrics["physical_time_history_missing_accepted_step_ids"] = []
        metrics["physical_time_history_clock_complete"] = False
        return
    try:
        dt_text = ET.parse(case_dir / "solver.xml").getroot().findtext(
            "GeneralSimulationParameters/Time_step_size")
        dt = float(dt_text) if dt_text is not None else 0.0
    except (OSError, ET.ParseError, ValueError):
        dt = 0.0

    missing_clock_steps: set[int] = set()

    def stamp(step: int) -> dict[str, int | float]:
        record, from_accepted_log = physical_history_stamp(step, clock, dt)
        if accepted_steps is not None and not from_accepted_log:
            missing_clock_steps.add(step)
        return record

    liquid_measure_history = []
    for step, path in paths:
        volume, source = dataset_wet_volume(pv.read(path))
        liquid_measure_history.append({
            **stamp(step),
            "corrected_state_liquid_measure": volume,
            "measure_source": source,
        })
    metrics["vtk_liquid_measure_history"] = liquid_measure_history
    # Compatibility alias for existing reports.  This history contains saved
    # accepted VTK states only; it is not the initial-to-accepted conservation
    # evidence used by the capillary-wave temporal-volume gate.
    metrics["physical_liquid_measure_history"] = liquid_measure_history

    add_free_surface_energy_history_metrics(
        metrics,
        benchmark,
        initial,
        paths,
        clock,
        clock_errors,
    )

    sessile = benchmark.get("sessile_contact")
    if isinstance(sessile, dict):
        spatial_sessile = benchmark.get("spatial_dimension", 2) == 3
        state_function = (
            spatial_capillary_state_metrics
            if spatial_sessile else sessile_state_metrics
        )
        initial_state = state_function(initial, benchmark)
        initial_state.update(stamp(0))
        history = [initial_state]
        for step, path in paths:
            state = state_function(pv.read(path), benchmark)
            state.update(stamp(step))
            state["path"] = str(path)
            history.append(state)
        metrics["sessile_contact_history"] = history
        metrics["initial_sessile_state"] = history[0]
        metrics["final_sessile_state"] = history[-1]
        final_state = history[-1]
        stage_history: list[dict[str, Any]] = []
        if sessile.get("dynamic"):
            try:
                parameters = generalized_alpha_first_order_stage_parameters(
                    case_dir, transient_solve)
                if clock_errors:
                    raise ValueError(
                        "accepted-step clock is invalid: " + "; ".join(clock_errors))
                if not exact_output_identity or not clock:
                    raise ValueError(
                        "stage reconstruction requires one adjacent endpoint VTK "
                        "state for every solver-accepted step")
                expected_steps = list(range(1, len(paths) + 1))
                if output_step_ids != expected_steps:
                    raise ValueError(
                        "stage reconstruction requires consecutive endpoint steps "
                        f"{expected_steps!r}; found {output_step_ids!r}")

                previous_endpoint = initial
                previous_source = "initialized_mesh"
                alpha_f = float(parameters["alpha_f"])
                for step, path in paths:
                    current_endpoint = pv.read(path)
                    stage_dataset = reconstruct_generalized_alpha_first_order_stage(
                        initial,
                        previous_endpoint,
                        current_endpoint,
                        alpha_f,
                    )
                    endpoint_time, endpoint_dt = clock[step]
                    stage_state: dict[str, Any] = {}
                    add_sessile_operator_contact_geometry(
                        stage_state, stage_dataset, benchmark)
                    stage_state.update({
                        "step": step,
                        "time": endpoint_time - (1.0 - alpha_f) * endpoint_dt,
                        "dt": endpoint_dt,
                        "endpoint_time": endpoint_time,
                        "stage_fraction_alpha_f": alpha_f,
                        "rho_inf": float(parameters["rho_inf"]),
                        "state_source": GENERALIZED_ALPHA_STAGE_STATE_SOURCE,
                        "parameter_source": parameters["parameter_source"],
                        "previous_endpoint_source": previous_source,
                        "current_endpoint_source": str(path),
                        "reconstructed_differential_fields": ["phi", "Velocity"],
                    })
                    stage_history.append(stage_state)
                    previous_endpoint = current_endpoint
                    previous_source = str(path)

                metrics["sessile_contact_stage_history"] = stage_history
                metrics["ren_e_stage_reconstruction_available"] = True
                metrics["ren_e_stage_state_source"] = (
                    GENERALIZED_ALPHA_STAGE_STATE_SOURCE)
                metrics["ren_e_generalized_alpha_parameter_source"] = (
                    parameters["parameter_source"])
                metrics["ren_e_generalized_alpha_rho_inf"] = float(
                    parameters["rho_inf"])
                metrics["ren_e_generalized_alpha_alpha_f"] = alpha_f
                metrics["ren_e_stage_reconstructed_differential_fields"] = [
                    "phi", "Velocity"]
            except Exception as exc:
                metrics["sessile_contact_stage_history"] = stage_history
                metrics["ren_e_stage_reconstruction_available"] = False
                metrics["ren_e_stage_reconstruction_error"] = str(exc)
        target_angle = sessile.get("equilibrium_contact_angle_degrees")
        fitted_observed_angle = final_state.get("contact_angle_degrees")
        if isinstance(fitted_observed_angle, (int, float)):
            fitted_shape = "sphere" if spatial_sessile else "circle"
            metrics[
                f"sessile_final_fitted_{fitted_shape}_contact_angle_degrees"
            ] = float(
                fitted_observed_angle)
            if isinstance(target_angle, (int, float)):
                metrics[
                    f"sessile_final_fitted_{fitted_shape}_contact_angle_error_degrees"
                ] = float(fitted_observed_angle) - float(target_angle)
        operator_observed_angle = final_state.get(
            "operator_dynamic_angle_degrees_mean")
        if (isinstance(target_angle, (int, float)) and
                isinstance(operator_observed_angle, (int, float))):
            metrics["sessile_final_contact_angle_source"] = (
                "same_state_LinearCorner_generated_triangle_normal_at_phi_zero_wall_edges"
                if spatial_sessile else
                "same_state_LinearCorner_generated_fragment_normal_at_phi_zero_wall_roots"
            )
            metrics["sessile_final_contact_angle_degrees"] = float(
                operator_observed_angle)
            metrics["sessile_final_contact_angle_error_degrees"] = (
                float(operator_observed_angle) - float(target_angle)
            )
            metrics["sessile_final_contact_angle_absolute_error_degrees"] = abs(
                float(operator_observed_angle) - float(target_angle)
            )
        expected_area = sessile.get("expected_initial_liquid_area")
        observed_area = final_state.get("liquid_area")
        if isinstance(expected_area, (int, float)) and isinstance(observed_area, (int, float)):
            metrics["sessile_expected_liquid_area"] = float(expected_area)
            metrics["sessile_final_liquid_area_error"] = (
                float(observed_area) - float(expected_area)
            )
            metrics["sessile_final_liquid_area_relative_error"] = (
                abs(float(observed_area) - float(expected_area)) /
                max(abs(float(expected_area)), 1.0e-300)
            )
        expected_volume = sessile.get("expected_initial_liquid_volume")
        observed_volume = final_state.get("liquid_volume")
        if (isinstance(expected_volume, (int, float)) and
                isinstance(observed_volume, (int, float))):
            metrics["sessile_expected_liquid_volume"] = float(expected_volume)
            metrics["sessile_final_liquid_volume_error"] = (
                float(observed_volume) - float(expected_volume)
            )
            metrics["sessile_final_liquid_volume_relative_error"] = (
                abs(float(observed_volume) - float(expected_volume)) /
                max(abs(float(expected_volume)), 1.0e-300)
            )
        for expected_name, observed_name, metric_prefix in (
                ("expected_initial_base_radius", "base_radius",
                 "sessile_final_base_radius"),
                ("expected_initial_apex_height", "apex_height",
                 "sessile_final_apex_height"),
                ("expected_initial_liquid_gas_area", "liquid_gas_area",
                 "sessile_final_liquid_gas_area"),
                ("expected_initial_contact_line_measure",
                 "contact_line_measure",
                 "sessile_final_contact_line_measure")):
            expected_value = sessile.get(expected_name)
            observed_value = final_state.get(observed_name)
            if (isinstance(expected_value, (int, float)) and
                    isinstance(observed_value, (int, float))):
                metrics[f"{metric_prefix}_expected"] = float(expected_value)
                metrics[f"{metric_prefix}_observed"] = float(observed_value)
                metrics[f"{metric_prefix}_relative_error"] = (
                    abs(float(observed_value) - float(expected_value)) /
                    max(abs(float(expected_value)), 1.0e-300)
                )
        expected_pressure_jump = benchmark.get("initial_active_pressure")
        observed_pressure_jump = final_state.get("pressure_jump")
        if (isinstance(expected_pressure_jump, (int, float)) and
                isinstance(observed_pressure_jump, (int, float))):
            metrics["sessile_expected_pressure_jump"] = float(expected_pressure_jump)
            metrics["sessile_final_pressure_jump_error"] = (
                float(observed_pressure_jump) - float(expected_pressure_jump)
            )
            metrics["sessile_final_pressure_jump_relative_error"] = (
                abs(float(observed_pressure_jump) - float(expected_pressure_jump)) /
                max(abs(float(expected_pressure_jump)), 1.0e-300)
            )
        if isinstance(final_state.get("max_liquid_speed"), (int, float)):
            metrics["sessile_final_max_parasitic_speed"] = float(
                final_state["max_liquid_speed"])
            viscosity = benchmark.get("viscosity")
            surface_tension = benchmark.get("surface_tension")
            if (isinstance(viscosity, (int, float)) and
                    isinstance(surface_tension, (int, float)) and
                    float(surface_tension) > 0.0):
                metrics["sessile_final_parasitic_capillary_number"] = (
                    float(viscosity) * float(final_state["max_liquid_speed"]) /
                    float(surface_tension)
                )
        if isinstance(final_state.get("contact_fluid_outward_speed"), (int, float)):
            metrics["sessile_final_contact_fluid_outward_speed"] = float(
                final_state["contact_fluid_outward_speed"])
            source = final_state.get("contact_fluid_evaluation_source")
            if isinstance(source, str):
                metrics["sessile_final_contact_fluid_evaluation_source"] = source
        for state_name, metric_name in (
                ("operator_dynamic_cos_mean",
                 "sessile_final_operator_dynamic_cos"),
                ("operator_dynamic_angle_degrees_mean",
                 "sessile_final_operator_dynamic_angle_degrees"),
                ("operator_young_gap_mean",
                 "sessile_final_operator_young_gap"),
                ("operator_wall_tangential_normal_norm_min",
                 "sessile_final_operator_wall_tangential_normal_norm_min")):
            value = final_state.get(state_name)
            if isinstance(value, (int, float)):
                metrics[metric_name] = float(value)
        if sessile.get("dynamic") and len(history) > 1:
            initial_half = history[0].get(
                "wall_half_footprint", history[0].get("half_footprint"))
            final_half = history[-1].get(
                "wall_half_footprint", history[-1].get("half_footprint"))
            elapsed = history[-1].get("time")
            predicted_initial = sessile.get("predicted_initial_contact_line_speed")
            if isinstance(predicted_initial, (int, float)):
                metrics["ren_e_predicted_initial_contact_line_speed"] = float(
                    predicted_initial)
            if all(isinstance(value, (int, float))
                   for value in (initial_half, final_half, elapsed)) and float(elapsed) > 0.0:
                geometric_speed = (
                    (float(final_half) - float(initial_half)) / float(elapsed)
                )
                metrics["ren_e_measured_mean_geometric_contact_line_speed"] = (
                    geometric_speed)
                # Retain the historical name as a clearly identified geometric
                # observable for downstream readers of existing qualification
                # logs.  It is no longer used as the direct constitutive gate.
                metrics["ren_e_measured_mean_contact_line_speed"] = geometric_speed
                if isinstance(predicted_initial, (int, float)):
                    metrics["ren_e_geometric_speed_sign_agrees"] = (
                        ren_e_speed_sign_agrees(
                            geometric_speed, float(predicted_initial))
                    )
                    if abs(float(predicted_initial)) > 1.0e-14:
                        metrics["ren_e_geometric_speed_ratio_to_initial_prediction"] = (
                            geometric_speed / float(predicted_initial)
                        )
                        metrics["ren_e_geometric_speed_relative_error"] = (
                            abs(geometric_speed - float(predicted_initial)) /
                            abs(float(predicted_initial))
                        )

            # Preserve the fitted-circle prediction as a geometric diagnostic.
            # It is not the state used by the weak contact-line operator.
            if isinstance(fitted_observed_angle, (int, float)):
                mobility = sessile.get("mobility")
                surface_tension = benchmark.get("surface_tension")
                if (isinstance(mobility, (int, float)) and
                        isinstance(surface_tension, (int, float)) and
                        isinstance(target_angle, (int, float))):
                    fitted_circle_prediction = (
                        float(mobility) * float(surface_tension) *
                        (math.cos(math.radians(float(target_angle))) -
                         math.cos(math.radians(float(fitted_observed_angle))))
                    )
                    metrics[
                        "ren_e_fitted_circle_predicted_final_contact_line_speed"
                    ] = fitted_circle_prediction

            constitutive_state = (
                stage_history[-1]
                if metrics.get("ren_e_stage_reconstruction_available") is True and
                stage_history else {})
            if constitutive_state:
                metrics["ren_e_constitutive_stage_time"] = float(
                    constitutive_state["time"])
                metrics["ren_e_constitutive_endpoint_time"] = float(
                    constitutive_state["endpoint_time"])
                metrics["ren_e_constitutive_endpoint_step"] = int(
                    constitutive_state["step"])

            predicted_final = constitutive_state.get(
                "operator_predicted_contact_line_speed")
            if isinstance(predicted_final, (int, float)):
                predicted_final = float(predicted_final)
                metrics["ren_e_predicted_final_contact_line_speed"] = (
                    predicted_final)
                metrics["ren_e_prediction_source"] = (
                    GENERALIZED_ALPHA_REN_E_PREDICTION_SOURCE)
            else:
                predicted_final = None

            contact_fluid_speed = constitutive_state.get(
                "operator_contact_fluid_speed")
            contact_fluid_source = constitutive_state.get(
                "operator_contact_fluid_evaluation_source")
            if contact_fluid_source == (
                    "Q1_velocity_and_generated_fragment_normal_at_phi_zero_wall_roots"):
                metrics["ren_e_contact_fluid_evaluation_source"] = (
                    GENERALIZED_ALPHA_REN_E_VELOCITY_SOURCE)
            if (isinstance(contact_fluid_speed, (int, float)) and
                    predicted_final is not None):
                contact_fluid_speed = float(contact_fluid_speed)
                metrics["ren_e_measured_final_contact_fluid_speed"] = (
                    contact_fluid_speed)
                metrics["ren_e_contact_fluid_speed_sign_agrees"] = (
                    ren_e_speed_sign_agrees(contact_fluid_speed, predicted_final)
                )
                if abs(predicted_final) > 1.0e-14:
                    ratio = contact_fluid_speed / predicted_final
                    relative_error = abs(contact_fluid_speed - predicted_final) / abs(
                        predicted_final)
                    metrics["ren_e_contact_fluid_speed_ratio_to_prediction"] = ratio
                    metrics["ren_e_contact_fluid_speed_relative_error"] = (
                        relative_error)
                    # Backward-compatible gate aliases now refer to the
                    # constitutive velocity u.m used in the assembled line
                    # residual, not to a startup-averaged footprint fit.
                    metrics["ren_e_speed_sign_agrees"] = (
                        metrics["ren_e_contact_fluid_speed_sign_agrees"])
                    metrics["ren_e_speed_ratio_measured_to_predicted"] = ratio
                    metrics["ren_e_speed_relative_error"] = relative_error

            previous_state = history[-2]
            previous_half = previous_state.get(
                "wall_half_footprint", previous_state.get("half_footprint"))
            previous_time = previous_state.get("time")
            final_time = final_state.get("time")
            if all(isinstance(value, (int, float)) for value in (
                    previous_half, final_half, previous_time, final_time)):
                interval_dt = float(final_time) - float(previous_time)
                if interval_dt > 0.0:
                    metrics["ren_e_final_interval_geometric_contact_line_speed"] = (
                        (float(final_half) - float(previous_half)) / interval_dt
                    )

    wave = benchmark.get("capillary_wave")
    if isinstance(wave, dict):
        wave_history = []
        for step, path in [(0, case_dir / "mesh/background/mesh-complete.mesh.vtu"),
                           *paths]:
            dataset = initial if step == 0 else pv.read(path)
            state_metrics: dict[str, Any] = {}
            add_capillary_wave_profile_fit(
                state_metrics,
                benchmark,
                "history",
                interface_profile_xy(dataset),
            )
            wave_history.append({
                **stamp(step),
                "cosine_amplitude": state_metrics.get(
                    "history_capillary_wave_cosine_amplitude"),
                "sine_amplitude": state_metrics.get(
                    "history_capillary_wave_sine_amplitude"),
                "mean_offset": state_metrics.get(
                    "history_capillary_wave_mean_offset"),
                "fit_rmse": state_metrics.get(
                    "history_capillary_wave_fit_rmse"),
                "available": state_metrics.get(
                    "history_capillary_wave_profile_available", False),
            })
        metrics["capillary_wave_amplitude_history"] = wave_history
        expected_omega = capillary_wave_expected_omega(wave)
        samples = [
            record for record in wave_history
            if isinstance(record.get("time"), (int, float)) and
            isinstance(record.get("cosine_amplitude"), (int, float))
        ]
        if expected_omega is not None and len(samples) >= 3:
            times = np.asarray([float(record["time"]) for record in samples])
            amplitudes = np.asarray([
                float(record["cosine_amplitude"]) for record in samples
            ])
            initial_amplitude = float(amplitudes[0])
            if abs(initial_amplitude) > 1.0e-14 and times[-1] > 0.0:
                candidates = np.linspace(
                    0.25 * expected_omega,
                    2.0 * expected_omega,
                    7001,
                )
                predicted = initial_amplitude * np.cos(
                    candidates[:, None] * times[None, :])
                squared_error = np.mean(
                    (predicted - amplitudes[None, :]) ** 2, axis=1)
                best = int(np.argmin(squared_error))
                observed_omega = float(candidates[best])
                metrics["capillary_wave_observed_omega"] = observed_omega
                metrics["capillary_wave_frequency_fit_rmse"] = float(
                    math.sqrt(float(squared_error[best])))
                metrics["capillary_wave_frequency_history_samples"] = len(samples)
                metrics["capillary_wave_frequency_observation_time"] = float(times[-1])
                metrics["capillary_wave_frequency_observed_phase_span"] = (
                    expected_omega * float(times[-1])
                )

    metrics["physical_time_history_missing_accepted_step_ids"] = sorted(
        missing_clock_steps)
    metrics["physical_time_history_clock_complete"] = (
        not clock_errors and not missing_clock_steps and exact_output_identity
    )
    wall_indices = boundary_face_point_indices(case_dir, initial)
    applicability = wall_false_wet_applicability(
        case_dir, initial, wall_indices)
    metrics.update(applicability)
    if (applicability["wall_only_false_wet_applicability"] ==
            "not_applicable_closed_interface"):
        metrics["wall_only_false_wet_history"] = []
        metrics["first_wall_only_false_wet"] = None
        return
    if not wall_indices or "phi" not in initial.point_data:
        return
    phi0 = np.asarray(initial.point_data["phi"], dtype=float).reshape(-1)
    points = np.asarray(initial.points, dtype=float)
    spacing = coordinate_min_spacing(points)
    tolerance = max(1.0e-10, 1.0e-3 * spacing) if spacing is not None else 1.0e-10
    centroid_stencils, stencil_errors = inward_cell_centroid_stencils_by_wall(
        initial, wall_indices)
    required_stencil_errors = [
        error for error in stencil_errors
        if (not isinstance(error.get("point_index"), int) or
            phi0[int(error["point_index"])] > tolerance)
    ]
    metrics["wall_inward_cell_centroid_stencil_error_count"] = len(
        required_stencil_errors)
    metrics["wall_inward_cell_centroid_stencil_errors"] = (
        required_stencil_errors)
    output_stencil_errors: list[dict[str, Any]] = []
    first_event: dict[str, Any] | None = None
    event_counts = []
    for step, path in paths:
        output = pv.read(path)
        try:
            phi = point_scalar_in_initial_gid_order(initial, output, "phi")
        except ValueError as exc:
            output_stencil_errors.append({
                "step": step,
                "path": str(path),
                "reason": "invalid_output_global_id_or_phi_mapping",
                "detail": str(exc),
            })
            event_counts.append({**stamp(step), "count": None})
            continue
        count = 0
        for wall_name, indices in wall_indices.items():
            for index in indices:
                point_index = int(index)
                candidates = centroid_stencils.get((wall_name, point_index))
                if not candidates or phi0[point_index] <= tolerance:
                    continue
                centroid_phi: list[float] = []
                stencil_valid = True
                for stencil in candidates:
                    point_ids = stencil.get("point_indices")
                    weights = stencil.get("weights")
                    if (not isinstance(point_ids, list) or
                            not isinstance(weights, list) or
                            len(point_ids) not in (3, 4) or
                            len(point_ids) != len(weights) or
                            not all(isinstance(value, int) for value in point_ids) or
                            not all(isinstance(value, (int, float)) and
                                    math.isfinite(float(value)) and float(value) > 0.0
                                    for value in weights) or
                            not math.isclose(sum(map(float, weights)), 1.0,
                                             rel_tol=0.0, abs_tol=1.0e-14)):
                        output_stencil_errors.append({
                            "step": step,
                            "wall": wall_name,
                            "point_index": point_index,
                            "global_node_id": int(
                                np.asarray(initial.point_data["GlobalNodeID"])[
                                    point_index]),
                            "cell_id": stencil.get("cell_id"),
                            "reason": "invalid_cell_interior_stencil",
                        })
                        stencil_valid = False
                        break
                    value = float(np.dot(
                        phi[np.asarray(point_ids, dtype=np.int64)],
                        np.asarray(weights, dtype=float),
                    ))
                    if not math.isfinite(value):
                        output_stencil_errors.append({
                            "step": step,
                            "wall": wall_name,
                            "point_index": point_index,
                            "cell_id": stencil.get("cell_id"),
                            "reason": "nonfinite_cell_centroid_phi",
                        })
                        stencil_valid = False
                        break
                    centroid_phi.append(value)
                if not stencil_valid or not centroid_phi:
                    continue
                if (phi[point_index] < -tolerance and
                        all(value > tolerance for value in centroid_phi)):
                    count += 1
                    if first_event is None:
                        first_event = {
                            **stamp(step),
                            "wall": wall_name,
                            "global_node_id": int(
                                np.asarray(initial.point_data["GlobalNodeID"])[
                                    point_index]),
                            "point": [float(value) for value in points[point_index]],
                            "initial_wall_phi": float(phi0[point_index]),
                            "wall_phi": float(phi[point_index]),
                            "inward_cell_centroid_candidate_count": len(
                                centroid_phi),
                            "inward_cell_centroid_phi_min": min(centroid_phi),
                            "inward_cell_centroid_phi_max": max(centroid_phi),
                            "inward_cell_ids": [
                                int(stencil["cell_id"]) for stencil in candidates],
                            "criterion": (
                                "initially dry wall vertex became liquid while all "
                                "incident P1 cell-centroid samples remained dry"
                            ),
                        }
        event_counts.append({**stamp(step), "count": count})
    metrics["wall_inward_cell_centroid_output_error_count"] = len(
        output_stencil_errors)
    metrics["wall_inward_cell_centroid_output_errors"] = output_stencil_errors
    metrics["wall_inward_cell_centroid_stencil_complete"] = (
        not required_stencil_errors and not output_stencil_errors)
    metrics["wall_only_false_wet_history"] = event_counts
    metrics["first_wall_only_false_wet"] = first_event
    metrics["physical_time_history_missing_accepted_step_ids"] = sorted(
        missing_clock_steps)
    metrics["physical_time_history_clock_complete"] = (
        not clock_errors and not missing_clock_steps and exact_output_identity
    )


def free_surface_energy_history_errors(
        metrics: dict[str, Any], args: argparse.Namespace) -> list[str]:
    """Apply fixed, fail-closed growth bounds to the saved-state proxy."""
    required = bool(getattr(args, "require_free_surface_energy_history", False))
    max_step = getattr(
        args,
        "max_free_surface_energy_positive_step_increment_relative",
        None,
    )
    max_above = getattr(
        args,
        "max_free_surface_energy_above_initial_relative",
        None,
    )
    if not required and max_step is None and max_above is None:
        return []
    if metrics.get("free_surface_energy_history_available") is not True:
        reason = metrics.get(
            "free_surface_energy_history_error",
            "free-surface energy history is unavailable",
        )
        return [str(reason)]
    if max_step is None or max_above is None:
        return [
            "free-surface energy gate requires both positive-step and "
            "above-initial relative bounds"
        ]
    summary = metrics.get("free_surface_energy_summary")
    if not isinstance(summary, dict):
        return ["free-surface energy summary is unavailable"]
    return energy_history_gate_errors(
        summary,
        max_positive_step_increment_relative=float(max_step),
        max_above_initial_relative=float(max_above),
        require_final_not_above_initial=True,
    )


def physical_history_clock_errors(
        metrics: dict[str, Any], args: argparse.Namespace) -> list[str]:
    """Require one exactly clocked VTK state for every accepted solver step."""
    if not getattr(args, "enable_physical_history_instrumentation", False):
        return []
    errors: list[str] = []
    if metrics.get("physical_time_history_clock_source") != "solver_accepted_steps":
        errors.append(
            "physical history was not timed from solver accepted-step diagnostics"
        )
    clock_errors = metrics.get("physical_time_history_clock_errors")
    if isinstance(clock_errors, list):
        errors.extend(str(error) for error in clock_errors)
    missing = metrics.get("physical_time_history_missing_accepted_step_ids")
    if isinstance(missing, list) and missing:
        errors.append(
            "physical history has no solver accepted-step time/dt for output step(s) "
            + ", ".join(str(step) for step in missing)
        )
    missing_outputs = metrics.get("physical_time_history_missing_output_step_ids")
    if isinstance(missing_outputs, list) and missing_outputs:
        errors.append(
            "physical history is missing VTK output for accepted step(s) "
            + ", ".join(str(step) for step in missing_outputs)
        )
    unexpected_outputs = metrics.get(
        "physical_time_history_unexpected_output_step_ids")
    if isinstance(unexpected_outputs, list) and unexpected_outputs:
        errors.append(
            "physical history contains VTK output with no accepted-step record "
            "for step(s) "
            + ", ".join(str(step) for step in unexpected_outputs)
        )
    if (metrics.get("physical_time_history_output_step_identity_complete") is
            not True and not missing_outputs and not unexpected_outputs):
        errors.append(
            "physical-history output step order does not exactly match the "
            "accepted-step order"
        )
    if metrics.get("physical_time_history_clock_complete") is not True:
        if not errors:
            errors.append("physical accepted-step history clock is incomplete")
    return errors


def production_physical_liquid_volume_history_errors(
        metrics: dict[str, Any], args: argparse.Namespace) -> list[str]:
    """Require true physical volume at initialization and every accepted step."""
    if not getattr(args, "enable_physical_history_instrumentation", False):
        return []
    if metrics.get("production_physical_liquid_volume_available") is True:
        return []
    return [str(metrics.get(
        "production_physical_liquid_volume_error",
        "production physical liquid-volume history is unavailable",
    ))]


def level_set_mass_correction_history_errors(
        metrics: dict[str, Any], args: argparse.Namespace) -> list[str]:
    """Require separate, accepted-clock pre/post correction mass histories."""
    if not getattr(args, "require_level_set_mass_correction_histories", False):
        return []
    if metrics.get("level_set_mass_correction_history_available") is True:
        return []
    return [str(metrics.get(
        "level_set_mass_correction_history_error",
        "separate level-set mass-correction histories are unavailable",
    ))]


def false_wall_wet_history_errors(
        metrics: dict[str, Any], args: argparse.Namespace) -> list[str]:
    """Fail an instrumented run on missing history or the first false-wet event."""
    if not getattr(args, "enable_physical_history_instrumentation", False):
        return []
    applicability = metrics.get("wall_only_false_wet_applicability")
    if applicability == "not_applicable_closed_interface":
        if metrics.get(
                "wall_only_false_wet_closed_interface_certified") is True:
            return []
        return [
            "wall false-wet instrumentation claimed a closed-interface "
            "exemption without a valid initial P1 boundary-sign certificate"
        ]
    errors: list[str] = []
    if metrics.get("wall_inward_cell_centroid_stencil_complete") is not True:
        structural_count = metrics.get(
            "wall_inward_cell_centroid_stencil_error_count")
        output_count = metrics.get(
            "wall_inward_cell_centroid_output_error_count")
        structural = metrics.get("wall_inward_cell_centroid_stencil_errors")
        output = metrics.get("wall_inward_cell_centroid_output_errors")
        first = (
            structural[0] if isinstance(structural, list) and structural else
            output[0] if isinstance(output, list) and output else None
        )
        errors.append(
            "wall false-wet instrumentation has invalid P1 cell-interior "
            f"stencils (structural={structural_count!r}, "
            f"output={output_count!r})"
            + (f"; first={first!r}" if first is not None else "")
        )
    history = metrics.get("wall_only_false_wet_history")
    if not isinstance(history, list) or not history:
        errors.append(
            "physical wall-wetting history is unavailable or empty; "
            "the instrumented run cannot certify the transient wall state"
        )
        return errors
    first_event = metrics.get("first_wall_only_false_wet")
    if first_event is None:
        return errors
    if isinstance(first_event, dict):
        errors.append(
            "false wall wetting detected at step "
            f"{first_event.get('step')!r}, time {first_event.get('time')!r}, "
            f"wall {first_event.get('wall')!r}, vertex "
            f"{first_event.get('global_node_id')!r}"
        )
        return errors
    errors.append(f"false wall wetting detected: {first_event!r}")
    return errors


def capillary_wave_boundary_contract_metrics(case_dir: Path) -> dict[str, Any]:
    """Audit the actual wave deck, including transport and wall kinematics."""
    errors: list[str] = []
    try:
        root = ET.parse(case_dir / "solver.xml").getroot()
        fluid = fluid_equation(root)
        level_set = level_set_equation(root)
        free_surface = free_surface_bc(root)
    except (OSError, ET.ParseError, ValueError) as exc:
        return {
            "capillary_wave_boundary_contract_valid": False,
            "capillary_wave_boundary_contract_errors": [str(exc)],
        }

    walls = {
        bc.attrib.get("name"): bc
        for bc in fluid.findall("Add_BC")
    }
    directions: dict[str, list[float] | None] = {}
    for name in ("wall_left", "wall_right"):
        wall = walls.get(name)
        if wall is None:
            directions[name] = None
            errors.append(f"missing {name}")
            continue
        raw = (wall.findtext("Effective_direction") or "").split()
        try:
            direction = [float(value) for value in raw]
        except ValueError:
            direction = []
        directions[name] = direction
        if (len(direction) != 2 or
                not math.isclose(direction[0], 1.0, abs_tol=1.0e-14) or
                not math.isclose(direction[1], 0.0, abs_tol=1.0e-14)):
            errors.append(
                f"{name} must constrain only horizontal normal velocity "
                "with Effective_direction='1 0'"
            )

    bottom = walls.get("wall_bottom")
    if bottom is None:
        errors.append("missing wall_bottom")
    else:
        raw = (bottom.findtext("Effective_direction") or "").split()
        try:
            bottom_direction = [float(value) for value in raw]
        except ValueError:
            bottom_direction = []
        if (len(bottom_direction) != 2 or
                not math.isclose(bottom_direction[0], 0.0, abs_tol=1.0e-14) or
                not math.isclose(bottom_direction[1], 1.0, abs_tol=1.0e-14)):
            errors.append(
                "wall_bottom must constrain only vertical normal velocity "
                "with Effective_direction='0 1'"
            )
        if (bottom.findtext("Type") or "").strip().lower() != "dir":
            errors.append("wall_bottom must use a normal-only Dirichlet condition")
        try:
            bottom_value = float((bottom.findtext("Value") or "nan").strip())
        except ValueError:
            bottom_value = math.nan
        if not math.isfinite(bottom_value) or abs(bottom_value) > 1.0e-14:
            errors.append("wall_bottom prescribed normal velocity must be zero")

    velocity_source = (level_set.findtext("Velocity_source") or "").strip()
    wet_extension = (
        level_set.findtext("Use_wet_extension_advection_velocity") or ""
    ).strip().lower()
    extension_method = (
        level_set.findtext("Wet_extension_advection_velocity_method") or ""
    ).strip().lower()
    velocity_extension = (
        free_surface.findtext("Enable_velocity_extension") or ""
    ).strip().lower()
    top_boundaries = [
        bc for bc in level_set.findall("Add_BC")
        if bc.attrib.get("name") == "wall_top"
    ]
    if len(top_boundaries) != 1 or (
            top_boundaries[0].findtext("Type") or "").strip().lower() != (
                "levelsetoutflow"):
        errors.append(
            "capillary-wave dry top must be declared as LevelSetOutflow"
        )
    if velocity_source.lower() != "prescribed_data":
        errors.append("capillary-wave level-set velocity source must be prescribed_data")
    if wet_extension != "true":
        errors.append("capillary-wave wet velocity extension must be enabled")
    if extension_method != "wall_compatible_normal":
        errors.append(
            "capillary-wave transport must use wall_compatible_normal extension"
        )
    if velocity_extension in {"true", "1", "yes", "on"}:
        errors.append(
            "capillary-wave physical momentum must not use the retired "
            "same-field dry-domain velocity extension"
        )

    return {
        "capillary_wave_boundary_contract_valid": not errors,
        "capillary_wave_boundary_contract_errors": errors,
        "capillary_wave_vertical_wall_effective_directions": directions,
        "capillary_wave_bottom_effective_direction": (
            bottom_direction if bottom is not None else None
        ),
        "capillary_wave_level_set_velocity_source": velocity_source,
        "capillary_wave_dry_top_boundary_type": (
            (top_boundaries[0].findtext("Type") or "").strip()
            if len(top_boundaries) == 1 else None
        ),
        "capillary_wave_wet_extension_enabled": wet_extension == "true",
        "capillary_wave_wet_extension_method": extension_method,
        "capillary_wave_fluid_velocity_extension_enabled": (
            velocity_extension in {"true", "1", "yes", "on"}
        ),
    }


def compute_metrics(case_name: str,
                    case_dir: Path,
                    result: Path,
                    enable_physical_history: bool = False,
                    accepted_steps: list[dict[str, Any]] | None = None,
                    transient_solve: dict[str, Any] | None = None,
                    ) -> dict[str, Any]:
    benchmark = load_benchmark(case_dir)
    if benchmark:
        dimensions = benchmark.get("dimensions_m", {})
        gate_x = float(dimensions.get("profile_window_x_min", CASE_GATE_X.get(case_name, 0.4)))
    else:
        gate_x = CASE_GATE_X[case_name]

    initial = pv.read(case_dir / "mesh/background/mesh-complete.mesh.vtu")
    output = pv.read(result)
    output_index = result_indices_by_initial_gid(initial, output)

    points = np.asarray(initial.points, dtype=float)
    active_domain = benchmark_active_domain(benchmark)
    phi0 = active_signed_level_set(
        point_array(initial, "phi").astype(float), active_domain)
    velocity = point_array(output, "Velocity").astype(float)[output_index]
    speed = np.linalg.norm(velocity, axis=1)

    wet0 = phi0 < 0.0
    gate_half_width = 0.025 if case_name != "mini2d" else 0.15
    gate_region = (np.abs(points[:, 0] - gate_x) <= gate_half_width) & wet0
    front_region = (points[:, 0] >= gate_x - 0.03) & (points[:, 0] <= gate_x + 0.07) & wet0
    if case_name == "mini2d":
        front_region = (points[:, 0] >= gate_x) & (points[:, 0] <= gate_x + 0.3) & wet0

    wet_speed = speed[wet0]
    wet_velocity = velocity[wet0]

    def mean_velocity(region: np.ndarray) -> list[float]:
        if not np.any(region):
            return [0.0 for _ in range(velocity.shape[1])]
        return [float(value) for value in np.nanmean(velocity[region], axis=0)]

    metrics: dict[str, Any] = {
        "result": str(result),
        "max_speed": float(np.nanmax(wet_speed)),
        "wet_mean_speed": float(np.nanmean(wet_speed)),
        "wet_mean_velocity": [float(value) for value in np.nanmean(wet_velocity, axis=0)],
        "gate_mean_velocity": mean_velocity(gate_region),
        "front_mean_velocity": mean_velocity(front_region),
        "finite_velocity": bool(np.isfinite(wet_velocity).all()),
        "wet_nodes": int(np.count_nonzero(wet0)),
        "gate_nodes": int(np.count_nonzero(gate_region)),
        "front_nodes": int(np.count_nonzero(front_region)),
        "active_domain": active_domain,
    }
    if case_name == "capillarywave2d":
        metrics.update(capillary_wave_boundary_contract_metrics(case_dir))
    if "Pressure" in output.point_data:
        pressure = np.asarray(output.point_data["Pressure"], dtype=float).reshape(-1)
        pressure_min = float(np.nanmin(pressure))
        pressure_max = float(np.nanmax(pressure))
        metrics["pressure_min"] = pressure_min
        metrics["pressure_max"] = pressure_max
        metrics["pressure_range"] = pressure_max - pressure_min
        metrics["pressure_mean"] = float(np.nanmean(pressure))
    if "Velocity" in output.point_data:
        output_velocity = np.asarray(output.point_data["Velocity"], dtype=float)
        if output_velocity.ndim == 1:
            output_velocity = output_velocity.reshape((-1, 1))
        output_speed = np.linalg.norm(output_velocity, axis=1)
        metrics["velocity_max"] = float(np.nanmax(output_speed))
        metrics["velocity_mean"] = float(np.nanmean(output_speed))
        component_ranges = []
        for component in range(output_velocity.shape[1]):
            values = output_velocity[:, component]
            component_ranges.append(float(np.nanmax(values) - np.nanmin(values)))
        metrics["velocity_component_ranges"] = component_ranges
        metrics["velocity_range"] = float(max(component_ranges)) if component_ranges else 0.0
    if "phi" in output.point_data:
        try:
            interface = output.contour(isosurfaces=[0.0], scalars="phi")
            interface_points = np.asarray(interface.points, dtype=float)
            metrics["interface_points"] = int(interface.n_points)
            if interface_points.size:
                metrics["interface_peak_height"] = float(np.nanmax(interface_points[:, 1]))
                metrics["interface_front_x"] = float(np.nanmax(interface_points[:, 0]))
        except Exception as exc:
            metrics["interface_extraction_error"] = str(exc)
    add_interface_motion_metrics(metrics, initial, output)
    add_capillary_wave_profile_metrics(metrics, benchmark, initial, output)
    add_projected_curvature_field_metrics(metrics, output)
    add_capillary_output_pressure_metrics(metrics, benchmark, output)
    if (benchmark.get("spatial_dimension") == 3 and
            benchmark.get("capillary_geometry") in {
                "sphere_3d", "sessile_spherical_cap_3d"}):
        spatial_state = spatial_capillary_state_metrics(output, benchmark)
        metrics["spatial_capillary_final_state"] = spatial_state
        for name, value in spatial_state.items():
            if isinstance(value, (int, float, str, bool)):
                metrics[f"spatial_capillary_final_{name}"] = value
    if "WetVolumeMeasure" in output.cell_data:
        wet_measures = np.asarray(output.cell_data["WetVolumeMeasure"], dtype=float).reshape(-1)
        if wet_measures.shape[0] == output.n_cells:
            metrics["wet_fraction_volume"] = float(np.sum(wet_measures))
            metrics["wet_fraction_volume_source"] = "WetVolumeMeasure"
            metrics["wet_volume_measure_cell_count"] = int(wet_measures.shape[0])
            metrics["wet_volume_measure_min"] = float(np.min(wet_measures))
            metrics["wet_volume_measure_max"] = float(np.max(wet_measures))
    if "WetVolumeFraction" in output.cell_data:
        fractions = np.asarray(output.cell_data["WetVolumeFraction"], dtype=float).reshape(-1)
        measures = cell_measure(output)
        if fractions.shape[0] == measures.shape[0]:
            metrics["wet_fraction_cell_count"] = int(fractions.shape[0])
            if "wet_fraction_volume" not in metrics:
                metrics["wet_fraction_volume"] = float(np.sum(fractions * measures))
                metrics["wet_fraction_volume_source"] = "WetVolumeFraction"
            metrics["wet_fraction_min"] = float(np.min(fractions))
            metrics["wet_fraction_max"] = float(np.max(fractions))
    metrics.update(pressure_gauge_metrics(output, benchmark))
    metrics.update(mms_verification_metrics(case_name, case_dir, result))
    if enable_physical_history:
        add_physical_time_history_metrics(
            metrics,
            case_dir,
            benchmark,
            initial,
            accepted_steps=accepted_steps,
            transient_solve=transient_solve,
        )
    return metrics


def add_incomplete_solve_output_metrics(
        metrics: dict[str, Any],
        case_name: str,
        run_dir: Path,
        diagnostics: dict[str, Any],
        args: argparse.Namespace,
        ) -> None:
    """Postprocess every safely accepted state left by an incomplete solve.

    A timeout or nonzero solver exit must never become qualification evidence,
    but it also must not erase direct physical observables from already
    accepted, fully written VTK states.  In particular, dynamic-contact
    diagnosis needs adjacent accepted endpoints to reconstruct the
    generalized-alpha stage used by the Ren--E residual.
    """
    metrics["incomplete_solve_output_metrics_available"] = False
    metrics["incomplete_solve_output_metrics_scope"] = (
        "accepted_states_before_incomplete_solver_exit")
    if args.disable_vtk_output:
        metrics["incomplete_solve_output_metrics_error"] = "VTK output disabled"
        return

    time_loop = diagnostics.get("time_loop", {})
    accepted_steps = (
        time_loop.get("accepted_steps", [])
        if isinstance(time_loop, dict) else [])
    if not isinstance(accepted_steps, list) or not accepted_steps:
        metrics["incomplete_solve_output_metrics_error"] = (
            "no solver-accepted endpoint is available")
        return
    final_step = accepted_steps[-1].get("step")
    if not isinstance(final_step, int) or isinstance(final_step, bool) or final_step <= 0:
        metrics["incomplete_solve_output_metrics_error"] = (
            "final solver-accepted step is invalid")
        return

    try:
        result = result_path(run_dir, final_step)
        output_metrics = compute_metrics(
            case_name,
            run_dir,
            result,
            enable_physical_history=args.enable_physical_history_instrumentation,
            accepted_steps=accepted_steps,
            transient_solve=diagnostics.get("solver_controls", {}).get(
                "transient_solve"),
        )
    except Exception as exc:
        metrics["incomplete_solve_output_metrics_error"] = str(exc)
        return

    metrics.update(output_metrics)
    metrics["result_step"] = final_step
    metrics["result_path"] = str(result)
    benchmark = load_benchmark(run_dir)
    if benchmark:
        metrics["benchmark"] = benchmark
    if args.enable_physical_history_instrumentation:
        add_production_physical_liquid_volume_metrics(metrics)
    if getattr(args, "require_level_set_mass_correction_histories", False):
        add_level_set_mass_correction_history_metrics(
            metrics, solver_fluid_density(run_dir))
    if isinstance(benchmark.get("capillary_wave"), dict):
        add_capillary_wave_temporal_liquid_volume_metrics(metrics)
    metrics["incomplete_solve_output_metrics_available"] = True


def mms_verification_metrics(case_name: str,
                             case_dir: Path,
                             result: Path) -> dict[str, Any]:
    if case_name != "mms2d":
        return {}
    verifier = case_dir / "verify_expected_results.py"
    if not verifier.exists():
        return {
            "mms_verification_available": False,
            "mms_verification_error": f"missing verifier {verifier}",
        }
    completed = subprocess.run(
        [sys.executable, str(verifier), str(result)],
        cwd=case_dir,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    metrics: dict[str, Any] = {
        "mms_verification_available": True,
        "mms_verification_returncode": completed.returncode,
    }
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError:
        metrics["mms_verification_passed"] = False
        metrics["mms_verification_stdout_tail"] = "\n".join(
            completed.stdout.splitlines()[-40:])
        return metrics

    metrics["mms_verification"] = payload
    metrics["mms_verification_passed"] = (
        completed.returncode == 0 and bool(payload.get("passed", False)))
    failed_checks = payload.get("failed_checks", [])
    if isinstance(failed_checks, list):
        metrics["mms_verification_failed_checks"] = failed_checks
    for key in (
            "phi_rms_error",
            "phi_max_abs_error",
            "interface_shift_error",
            "interface_l2_height_error",
            "area_relative_error",
            "centroid_y_error",
            "velocity_relative_l2_error",
            "pressure_relative_rms_error",
            "pressure_relative_rms_error_after_constant_offset_removal",
            "interface_pressure_max_abs",
            "manufactured_residual_x_max",
            "manufactured_residual_y_max",
            "level_set_residual_max"):
        value = payload.get(key)
        if isinstance(value, (int, float)):
            metrics[f"mms_{key}"] = float(value)
    return metrics


def solver_time_step_size(case_dir: Path) -> float | None:
    try:
        root = ET.parse(case_dir / "solver.xml").getroot()
    except Exception:
        return None
    raw = text(root, "./GeneralSimulationParameters/Time_step_size")
    try:
        return float(raw)
    except ValueError:
        return None


def reference_profile_for_time(
        benchmark: dict[str, Any],
        final_time: float,
        tolerance: float) -> dict[str, Any] | None:
    profiles = benchmark.get("reference_profiles")
    if not isinstance(profiles, list):
        return None
    candidates = [
        profile for profile in profiles
        if isinstance(profile, dict) and
        isinstance(profile.get("time_s"), (int, float)) and
        isinstance(profile.get("path"), str)
    ]
    if not candidates:
        return None
    best = min(candidates, key=lambda profile: abs(float(profile["time_s"]) - final_time))
    if abs(float(best["time_s"]) - final_time) > tolerance:
        return None
    return best


def add_reference_profile_metrics(metrics: dict[str, Any],
                                  case_dir: Path,
                                  result: Path,
                                  args: argparse.Namespace) -> None:
    if not args.require_reference_profile_comparison:
        return
    benchmark = load_benchmark(case_dir)
    dt = args.time_step_size
    if dt is None:
        dt = solver_time_step_size(case_dir)
    if dt is None:
        metrics["reference_profile_error"] = "solver time step size is unavailable"
        return

    final_time = float(args.steps) * float(dt)
    tolerance = (
        args.reference_profile_time_tolerance
        if args.reference_profile_time_tolerance is not None
        else max(1.0e-12, 0.5 * float(dt))
    )
    profile = reference_profile_for_time(benchmark, final_time, tolerance)
    if profile is None:
        metrics["reference_profile_error"] = (
            f"no reference profile within {tolerance:.6g}s of final time "
            f"{final_time:.6g}s"
        )
        metrics["reference_profile_time_s"] = final_time
        return

    reference_path = ROOT / str(profile["path"])
    metrics["reference_profile_time_s"] = final_time
    metrics["reference_profile_target_time_s"] = float(profile["time_s"])
    metrics["reference_profile_path"] = str(reference_path)
    try:
        import compare_test05_profiles as test05_profiles

        report = test05_profiles.compare(argparse.Namespace(
            result=result,
            reference_profile=reference_path,
            scalar="phi",
            benchmark_json=case_dir / "benchmark.json",
            density=1000.0,
            initial_wet_volume=None,
            initial_kinetic_energy=0.0,
            front_diagnostic_only=False,
            stale_pressure_gauge_tolerance=None,
            min_velocity_max=None,
            x_min=None,
            x_max=None,
            sample_radius=args.reference_profile_sample_radius,
            elevated_front_clearance=(
                args.reference_profile_elevated_front_clearance
                if args.reference_profile_elevated_front_clearance is not None
                else 0.005
            ),
            max_elevated_front_lag=args.max_reference_profile_elevated_front_lag,
            plot_output=None,
            output=None,
        ))
    except Exception as exc:
        metrics["reference_profile_error"] = str(exc)
        return

    validation = report.get("validation", {})
    if isinstance(validation, dict):
        metrics["reference_profile_validation_passed"] = bool(
            validation.get("passed", False))
        failures = validation.get("failures", [])
        if isinstance(failures, list):
            metrics["reference_profile_validation_failures"] = failures

    comparison = report.get("profile_comparison", {})
    if not isinstance(comparison, dict):
        return
    profile_metrics = comparison.get("metrics", {})
    if not isinstance(profile_metrics, dict):
        return
    for key, value in profile_metrics.items():
        if isinstance(value, (int, float, str)) or value is None:
            metrics[f"reference_profile_{key}"] = value


def evaluate(metrics: dict[str, Any], args: argparse.Namespace) -> list[str]:
    errors = []
    if metrics.get("output_metrics_skipped"):
        return evaluate_timeout_diagnostics(metrics, args)
    errors.extend(solver_elapsed_time_errors(metrics, args))
    errors.extend(time_loop_convergence_errors(metrics, args))
    if args.require_mms_verification:
        if not metrics.get("mms_verification_available"):
            errors.append("MMS verification was not available")
        elif not metrics.get("mms_verification_passed"):
            failed_checks = metrics.get("mms_verification_failed_checks", [])
            if isinstance(failed_checks, list) and failed_checks:
                errors.append(
                    "MMS verification failed check(s): " +
                    ", ".join(str(item) for item in failed_checks))
            else:
                errors.append("MMS verification did not pass")
    if args.require_cut_context_solution_source_diagnostics:
        errors.extend(cut_context_solution_source_errors(metrics["diagnostics"]))
    errors.extend(cut_context_policy_errors(metrics, args))
    errors.extend(curvature_projection_errors(metrics, args))
    errors.extend(capillary_benchmark_errors(metrics, args))
    errors.extend(capillary_wave_benchmark_errors(metrics, args))
    errors.extend(capillary_stability_errors(metrics, args))
    errors.extend(free_surface_conservative_balance_errors(metrics, args))
    errors.extend(free_surface_pressure_representability_errors(metrics, args))
    errors.extend(
        static_capillary_equilibrium_initialization_errors(metrics, args))
    errors.extend(sessile_physical_errors(metrics, args))
    errors.extend(fsils_accepted_true_residual_errors(metrics, args))
    errors.extend(fsils_matrix_diag_col_mismatch_errors(metrics, args))
    errors.extend(fsils_matrix_duplicate_diag_errors(metrics, args))
    errors.extend(level_set_advection_velocity_errors(metrics, args))
    errors.extend(free_surface_energy_history_errors(metrics, args))
    errors.extend(physical_history_clock_errors(metrics, args))
    errors.extend(production_physical_liquid_volume_history_errors(metrics, args))
    errors.extend(level_set_mass_correction_history_errors(metrics, args))
    errors.extend(false_wall_wet_history_errors(metrics, args))
    if (args.require_newton_assembly_diagnostics and
            not metrics["diagnostics"].get("newton_assemblies")):
        errors.append("Newton assembly diagnostics were not reported")
    if args.require_assembly_timing_diagnostics and not metrics["diagnostics"].get("assembly_timings"):
        errors.append("assembly timing diagnostics were not reported")
    errors.extend(assembly_efficiency_errors(metrics, args))
    if args.require_process_memory_diagnostics:
        diagnostics = metrics["diagnostics"]
        has_process_memory = (
            diagnostics.get("process_memory") or
            any(
                isinstance(record.get("process_rss_kb"), (int, float))
                for record in diagnostics.get("cut_context_rebuilds", [])
            )
        )
        if not has_process_memory:
            errors.append("process memory diagnostics were not reported")
    if (args.require_linear_solve_memory_diagnostics and
            not has_linear_solve_memory_diagnostics(metrics["diagnostics"])):
        errors.append("linear-solve memory diagnostics were not reported")
    if (args.require_fsils_matrix_diagnostics and
            not metrics["diagnostics"].get("fsils_prepared_matrices")):
        errors.append("FSILS prepared-matrix diagnostics were not reported")
    if args.max_fsils_matrix_zero_rows is not None:
        zero_rows = metrics.get("diagnostic_fsils_prepared_matrix_max_zero_rows")
        if not isinstance(zero_rows, (int, float)):
            errors.append("FSILS prepared-matrix zero-row diagnostics are unavailable")
        elif zero_rows > args.max_fsils_matrix_zero_rows:
            errors.append(
                f"FSILS prepared-matrix zero rows {zero_rows} exceed "
                f"{args.max_fsils_matrix_zero_rows}"
            )
    if args.max_fsils_matrix_missing_diag is not None:
        missing_diag = metrics.get(
            "diagnostic_fsils_prepared_matrix_max_missing_diag"
        )
        if not isinstance(missing_diag, (int, float)):
            errors.append("FSILS prepared-matrix missing-diagonal diagnostics are unavailable")
        elif missing_diag > args.max_fsils_matrix_missing_diag:
            errors.append(
                f"FSILS prepared-matrix missing diagonals {missing_diag} exceed "
                f"{args.max_fsils_matrix_missing_diag}"
            )
    if args.max_fsils_matrix_zero_diag is not None:
        zero_diag = metrics.get("diagnostic_fsils_prepared_matrix_max_zero_diag")
        if not isinstance(zero_diag, (int, float)):
            errors.append("FSILS prepared-matrix zero-diagonal diagnostics are unavailable")
        elif zero_diag > args.max_fsils_matrix_zero_diag:
            errors.append(
                f"FSILS prepared-matrix zero diagonals {zero_diag} exceed "
                f"{args.max_fsils_matrix_zero_diag}"
            )
    if args.max_fsils_matrix_nonfinite_entries is not None:
        nonfinite = metrics.get(
            "diagnostic_fsils_prepared_matrix_max_nonfinite_entries"
        )
        if not isinstance(nonfinite, (int, float)):
            errors.append("FSILS prepared-matrix nonfinite-entry diagnostics are unavailable")
        elif nonfinite > args.max_fsils_matrix_nonfinite_entries:
            errors.append(
                f"FSILS prepared-matrix nonfinite entries {nonfinite} exceed "
                f"{args.max_fsils_matrix_nonfinite_entries}"
            )
    if args.require_basis_cache_diagnostics:
        has_basis_cache = any(
            isinstance(record.get("basis_cache_entries"), (int, float))
            for record in metrics["diagnostics"].get("process_memory", [])
        )
        if not has_basis_cache:
            errors.append("basis-cache diagnostics were not reported")
    if args.max_diagnostic_process_basis_cache_entries is not None:
        basis_cache_entries = metrics.get("diagnostic_process_max_basis_cache_entries")
        if not isinstance(basis_cache_entries, (int, float)):
            errors.append("basis-cache entry diagnostics are unavailable")
        elif basis_cache_entries > args.max_diagnostic_process_basis_cache_entries:
            errors.append(
                f"basis-cache entries {basis_cache_entries} exceed "
                f"{args.max_diagnostic_process_basis_cache_entries}"
            )
    errors.extend(resource_ceiling_errors(metrics, args))
    if (args.require_interior_face_timing_diagnostics and
            not metrics["diagnostics"].get("interior_face_timings")):
        errors.append("interior-face timing diagnostics were not reported")
    if (args.require_cut_volume_timing_diagnostics and
            not metrics["diagnostics"].get("cut_volume_timings")):
        errors.append("cut-volume timing diagnostics were not reported")
    if (args.require_jit_specialization_trace_diagnostics and
            not metrics["diagnostics"].get("jit_specialization_traces")):
        errors.append("JIT specialization trace diagnostics were not reported")
    if (args.require_jit_cache_diagnostics and
            not metrics["diagnostics"].get("jit_cache_diagnostics")):
        errors.append("JIT cache diagnostics were not reported")
    if getattr(args, "require_compiled_cut_volume_jit", False):
        errors.extend(compiled_cut_volume_jit_errors(metrics["diagnostics"]))
    if (args.require_marked_interior_face_fallback_diagnostics and
            not has_marked_interior_face_fallback_trace(metrics["diagnostics"])):
        errors.append("marked interior-face fallback diagnostics were not reported")
    if args.require_assembly_topology_consistency:
        errors.extend(assembly_topology_consistency_errors(metrics["diagnostics"]))
    if (args.require_eigen_factorization_diagnostics and
            not metrics["diagnostics"].get("eigen_factorization_diagnostics")):
        errors.append("Eigen factorization diagnostics were not reported")
    if (args.require_active_pressure_support_diagnostics and
            not metrics["diagnostics"].get("active_pressure_support_constraints")):
        errors.append("active pressure support diagnostics were not reported")
    if args.max_eigen_factorization_zero_rows is not None:
        zero_rows = metrics.get("diagnostic_eigen_factorization_max_zero_rows")
        if not isinstance(zero_rows, (int, float)):
            errors.append("Eigen factorization zero-row diagnostics are unavailable")
        elif zero_rows > args.max_eigen_factorization_zero_rows:
            errors.append(
                f"Eigen factorization zero rows {zero_rows} exceed "
                f"{args.max_eigen_factorization_zero_rows}"
            )
    if args.max_eigen_factorization_pressure_zero_rows is not None:
        pressure_zero_rows = metrics.get(
            "diagnostic_eigen_factorization_max_pressure_zero_rows"
        )
        if not isinstance(pressure_zero_rows, (int, float)):
            errors.append("Eigen factorization pressure zero-row diagnostics are unavailable")
        elif pressure_zero_rows > args.max_eigen_factorization_pressure_zero_rows:
            errors.append(
                f"Eigen factorization pressure zero rows {pressure_zero_rows} exceed "
                f"{args.max_eigen_factorization_pressure_zero_rows}"
            )
    if args.max_eigen_factorization_pressure_zero_cols is not None:
        pressure_zero_cols = metrics.get(
            "diagnostic_eigen_factorization_max_pressure_zero_cols"
        )
        if not isinstance(pressure_zero_cols, (int, float)):
            errors.append("Eigen factorization pressure zero-column diagnostics are unavailable")
        elif pressure_zero_cols > args.max_eigen_factorization_pressure_zero_cols:
            errors.append(
                f"Eigen factorization pressure zero columns {pressure_zero_cols} exceed "
                f"{args.max_eigen_factorization_pressure_zero_cols}"
            )
    if args.max_eigen_factorization_nonfinite_entries is not None:
        nonfinite = metrics.get(
            "diagnostic_eigen_factorization_max_nonfinite_entries"
        )
        if not isinstance(nonfinite, (int, float)):
            errors.append("Eigen factorization nonfinite-entry diagnostics are unavailable")
        elif nonfinite > args.max_eigen_factorization_nonfinite_entries:
            errors.append(
                f"Eigen factorization nonfinite entries {nonfinite} exceed "
                f"{args.max_eigen_factorization_nonfinite_entries}"
            )
    if not metrics["finite_velocity"]:
        errors.append("Velocity contains non-finite values")
    if args.min_capillary_response_speed_per_surface_tension is not None:
        surface_tension = metrics.get("surface_tension")
        max_speed = metrics.get("max_speed")
        if not isinstance(surface_tension, (int, float)):
            errors.append("surface-tension control is unavailable")
        elif abs(float(surface_tension)) <= 0.0:
            errors.append("surface tension is zero; capillary response cannot be normalized")
        elif not isinstance(max_speed, (int, float)):
            errors.append("capillary response speed diagnostic is unavailable")
        else:
            normalized_speed = float(max_speed) / abs(float(surface_tension))
            metrics["capillary_response_speed_per_surface_tension"] = normalized_speed
            if normalized_speed < args.min_capillary_response_speed_per_surface_tension:
                errors.append(
                    "capillary response speed per surface tension "
                    f"{normalized_speed:.6g} is below "
                    f"{args.min_capillary_response_speed_per_surface_tension:.6g}"
                )
    if args.max_capillary_balance_speed_per_surface_tension is not None:
        surface_tension = metrics.get("surface_tension")
        max_speed = metrics.get("max_speed")
        if not isinstance(surface_tension, (int, float)):
            errors.append("surface-tension control is unavailable")
        elif abs(float(surface_tension)) <= 0.0:
            errors.append("surface tension is zero; capillary balance cannot be normalized")
        elif not isinstance(max_speed, (int, float)):
            errors.append("capillary balance speed diagnostic is unavailable")
        else:
            normalized_speed = float(max_speed) / abs(float(surface_tension))
            metrics["capillary_balance_speed_per_surface_tension"] = normalized_speed
            if normalized_speed > args.max_capillary_balance_speed_per_surface_tension:
                errors.append(
                    "capillary balance speed per surface tension "
                    f"{normalized_speed:.6g} exceeds "
                    f"{args.max_capillary_balance_speed_per_surface_tension:.6g}"
                )
    if args.min_diagnostic_level_set_volume_correction_count is not None:
        count = metrics.get("diagnostic_level_set_volume_correction_count")
        if not isinstance(count, int):
            errors.append("level-set volume-correction diagnostics are unavailable")
        elif count < args.min_diagnostic_level_set_volume_correction_count:
            errors.append(
                f"level-set volume-correction count {count} is below "
                f"{args.min_diagnostic_level_set_volume_correction_count}"
            )
    if args.max_diagnostic_level_set_volume_correction_achieved_error is not None:
        error = metrics.get(
            "diagnostic_level_set_volume_correction_max_abs_achieved_error"
        )
        if not isinstance(error, (int, float)):
            errors.append("level-set volume-correction achieved-error diagnostic is unavailable")
        elif float(error) > args.max_diagnostic_level_set_volume_correction_achieved_error:
            errors.append(
                "level-set volume-correction achieved error "
                f"{float(error):.6g} exceeds "
                f"{args.max_diagnostic_level_set_volume_correction_achieved_error:.6g}"
            )
    if metrics.get("case") == "static2d":
        if metrics["max_speed"] > args.max_static_speed:
            errors.append(
                f"static max speed {metrics['max_speed']:.6g} exceeds "
                f"{args.max_static_speed:.6g}"
            )
        return errors
    if metrics["max_speed"] < args.min_max_speed:
        errors.append(
            f"max speed {metrics['max_speed']:.6g} is below {args.min_max_speed:.6g}"
        )
    if metrics["wet_mean_speed"] < args.min_wet_mean_speed:
        errors.append(
            f"wet mean speed {metrics['wet_mean_speed']:.6g} is below "
            f"{args.min_wet_mean_speed:.6g}"
        )
    if metrics.get("gate_nodes", 0) <= 0:
        if args.min_gate_mean_ux > -1.0:
            errors.append("gate region contains no wet nodes")
    elif metrics["gate_mean_velocity"][0] < args.min_gate_mean_ux:
        errors.append(
            f"gate mean ux {metrics['gate_mean_velocity'][0]:.6g} is below "
            f"{args.min_gate_mean_ux:.6g}"
        )
    if metrics.get("front_nodes", 0) <= 0:
        if args.min_front_mean_ux > -1.0:
            errors.append("front region contains no wet nodes")
    elif metrics["front_mean_velocity"][0] < args.min_front_mean_ux:
        errors.append(
            f"front mean ux {metrics['front_mean_velocity'][0]:.6g} is below "
            f"{args.min_front_mean_ux:.6g}"
        )
    if args.min_active_volume_change > 0.0:
        volume_change = metrics.get("assembly_active_wet_volume_change", 0.0)
        volume_count = len(metrics.get("assembly_active_wet_volumes", []))
        if volume_count < 2:
            errors.append("assembly active wet volume was not reported at least twice")
        elif volume_change < args.min_active_volume_change:
            errors.append(
                f"assembly active wet-volume change {volume_change:.6g} is below "
                f"{args.min_active_volume_change:.6g}"
            )
    for arg_name, metric_name, label in (
            ("min_interface_height_change",
             "interface_height_max_abs_change",
             "interface height max absolute change"),
            ("min_interface_mean_abs_height_change",
             "interface_height_mean_abs_change",
             "interface height mean absolute change"),
            ("min_interface_slope_change",
             "interface_slope_abs_change",
             "interface slope absolute change"),
            ("min_interface_final_height_span",
             "final_interface_height_span",
             "final interface height span")):
        minimum = getattr(args, arg_name)
        if minimum is None:
            continue
        if not metrics.get("interface_motion_available", False):
            reason = metrics.get("interface_motion_unavailable_reason", "unavailable")
            errors.append(f"interface motion diagnostics are unavailable ({reason})")
            continue
        value = metrics.get(metric_name)
        if not isinstance(value, (int, float)):
            errors.append(f"{label} diagnostic is unavailable")
        elif float(value) < float(minimum):
            errors.append(
                f"{label} {float(value):.6g} is below {float(minimum):.6g}"
            )
    if args.stale_pressure_gauge_tolerance is not None:
        if not metrics.get("pressure_gauge_found", False):
            errors.append("pressure gauge was not found in the solver output")
        else:
            stale_difference = metrics.get("pressure_gauge_previous_invalid_difference")
            if not isinstance(stale_difference, (int, float)):
                errors.append("previous invalid pressure gauge value is unavailable")
            elif abs(float(stale_difference)) <= args.stale_pressure_gauge_tolerance:
                errors.append(
                    "pressure gauge remains close to the previous full-volume "
                    "hydrostatic value"
                )
    if args.max_wet_fraction_volume_error is not None:
        wet_fraction_volume = metrics.get("wet_fraction_volume")
        context_volumes = metrics.get("cut_context_active_side_physical_volumes", [])
        wet_volume_source = str(metrics.get("wet_fraction_volume_source", "WetVolumeFraction"))
        if not isinstance(wet_fraction_volume, (int, float)):
            errors.append("WetVolumeFraction/WetVolumeMeasure output volume is unavailable")
        elif not context_volumes:
            errors.append("physical cut-context active-side volume was not reported")
        else:
            error = abs(float(wet_fraction_volume) - float(context_volumes[-1]))
            metrics["wet_fraction_volume_comparison_frame"] = "physical"
            metrics["wet_fraction_volume_comparison_kind"] = (
                "same_state_vtk_output_vs_last_cut_context")
            metrics["wet_fraction_volume_error_vs_last_cut_context"] = error
            if error > args.max_wet_fraction_volume_error:
                errors.append(
                    f"{wet_volume_source} volume error {error:.6g} exceeds "
                    f"{args.max_wet_fraction_volume_error:.6g}"
                )
    if args.require_reference_profile_comparison:
        if metrics.get("reference_profile_error"):
            errors.append(
                "reference profile comparison failed: "
                f"{metrics['reference_profile_error']}"
            )
        if metrics.get("reference_profile_validation_passed") is False:
            failures = metrics.get("reference_profile_validation_failures", [])
            errors.append(
                "reference profile validation failed"
                + (f": {failures}" if failures else "")
            )
        for arg_name, metric_name, label in (
                ("min_reference_profile_coverage",
                 "reference_profile_coverage_fraction",
                 "reference profile coverage"),
                ("min_reference_profile_direct_coverage",
                 "reference_profile_direct_coverage_fraction",
                 "reference profile direct coverage")):
            minimum = getattr(args, arg_name)
            if minimum is None:
                continue
            value = metrics.get(metric_name)
            if not isinstance(value, (int, float)):
                errors.append(f"{label} diagnostic is unavailable")
            elif float(value) < float(minimum):
                errors.append(
                    f"{label} {float(value):.6g} is below {float(minimum):.6g}"
                )
        for arg_name, metric_name, label in (
                ("max_reference_profile_rmse",
                 "reference_profile_rmse_m",
                 "reference profile RMSE"),
                ("max_reference_profile_mae",
                 "reference_profile_mae_m",
                 "reference profile MAE"),
                ("max_reference_profile_max_abs_error",
                 "reference_profile_max_abs_error_m",
                 "reference profile max absolute error")):
            maximum = getattr(args, arg_name)
            if maximum is None:
                continue
            value = metrics.get(metric_name)
            if not isinstance(value, (int, float)):
                errors.append(f"{label} diagnostic is unavailable")
            elif float(value) > float(maximum):
                errors.append(
                    f"{label} {float(value):.6g} exceeds {float(maximum):.6g}"
                )
    return errors


def time_loop_convergence_errors(metrics: dict[str, Any],
                                 args: argparse.Namespace) -> list[str]:
    if not args.require_time_loop_convergence:
        return []
    time_loop = metrics.get("time_loop")
    if not isinstance(time_loop, dict):
        diagnostics = metrics.get("diagnostics", {})
        if isinstance(diagnostics, dict):
            time_loop = diagnostics.get("time_loop", {})
    summary = time_loop.get("summary") if isinstance(time_loop, dict) else None
    if not isinstance(summary, dict):
        return ["time-loop convergence summary was not reported"]

    errors = []
    expected_steps_value = metrics.get("steps")
    if not isinstance(expected_steps_value, (int, float)):
        controls = metrics.get("solver_controls", {})
        if isinstance(controls, dict):
            time_stepping = controls.get("time_stepping", {})
            if isinstance(time_stepping, dict):
                expected_steps_value = time_stepping.get("number_of_time_steps")
    expected_steps = (
        int(expected_steps_value)
        if isinstance(expected_steps_value, (int, float)) else 0
    )
    accepted_steps = summary.get("accepted_steps")
    if not isinstance(accepted_steps, int):
        errors.append("accepted-step count was not reported")
    elif expected_steps > 0 and accepted_steps < expected_steps:
        errors.append(
            f"accepted steps {accepted_steps} below requested steps {expected_steps}")
    adaptive_enabled = args.enable_adaptive_time_loop
    if not adaptive_enabled:
        controls = metrics.get("solver_controls", {})
        if isinstance(controls, dict):
            adaptive_enabled = isinstance(controls.get("adaptive_time_loop"), dict)

    if adaptive_enabled:
        final_time = summary.get("final_accepted_time")
        time_step = metrics.get("time_step_size")
        if not isinstance(time_step, (int, float)):
            controls = metrics.get("solver_controls", {})
            if isinstance(controls, dict):
                time_stepping = controls.get("time_stepping", {})
                if isinstance(time_stepping, dict):
                    time_step = time_stepping.get("time_step_size")
        if expected_steps > 0 and isinstance(time_step, (int, float)):
            expected_time = expected_steps * float(time_step)
            tolerance = max(1.0e-12, 1.0e-9 * max(1.0, abs(expected_time)))
            if not isinstance(final_time, (int, float)):
                errors.append("final accepted time was not reported")
            elif float(final_time) + tolerance < expected_time:
                errors.append(
                    f"final accepted time {float(final_time):.6g} below requested "
                    f"time {expected_time:.6g}"
                )
    elif summary.get("all_nonlinear_converged") is not True:
        errors.append("not all nonlinear solves converged")
    if not adaptive_enabled and summary.get("all_linear_converged") is not True:
        errors.append("not all linear solves converged")
    if args.max_time_loop_nonlinear_iterations_per_step is not None:
        max_nonlinear = summary.get("nonlinear_iterations_max")
        if not isinstance(max_nonlinear, int):
            errors.append("maximum nonlinear iteration count was not reported")
        elif max_nonlinear > args.max_time_loop_nonlinear_iterations_per_step:
            errors.append(
                f"maximum nonlinear iterations per step {max_nonlinear} exceed "
                f"{args.max_time_loop_nonlinear_iterations_per_step}"
            )
    max_outer = getattr(
        args, "max_time_loop_outer_iterations_per_step", None)
    if max_outer is not None:
        observed = summary.get("outer_iterations_max")
        if not isinstance(observed, int):
            errors.append("maximum external-state outer iteration count was not reported")
        elif observed > max_outer:
            errors.append(
                f"maximum external-state outer iterations per step {observed} "
                f"exceed {max_outer}"
            )
    max_inner_total = getattr(
        args, "max_time_loop_inner_iterations_total_per_step", None)
    if max_inner_total is not None:
        observed = summary.get("inner_iterations_total_max")
        if not isinstance(observed, int):
            errors.append(
                "maximum external-state total inner iteration count was not reported"
            )
        elif observed > max_inner_total:
            errors.append(
                "maximum external-state total inner iterations per step "
                f"{observed} exceed {max_inner_total}"
            )
    if args.max_time_loop_linear_iterations_per_step is not None:
        max_linear = summary.get("linear_iterations_max")
        if not isinstance(max_linear, int):
            errors.append("maximum linear iteration count was not reported")
        elif max_linear > args.max_time_loop_linear_iterations_per_step:
            errors.append(
                f"maximum linear iterations per step {max_linear} exceed "
                f"{args.max_time_loop_linear_iterations_per_step}"
            )
    return errors


def compact_qualification_probe(probe: dict[str, Any]) -> dict[str, Any]:
    compact = {
        key: value
        for key, value in probe.items()
        if key != "diagnostics"
    }
    diagnostics = probe.get("diagnostics")
    if not isinstance(diagnostics, dict):
        return compact

    evidence: dict[str, Any] = {
        "full_records_embedded": False,
        "retention_requires_preserve_run_dir": True,
    }
    counts = diagnostics.get("counts")
    if isinstance(counts, dict):
        evidence["record_counts"] = dict(counts)
    run_dir = probe.get("run_dir")
    if isinstance(run_dir, str) and run_dir:
        evidence["solver_log_path"] = str(Path(run_dir) / "solver_run.log")
    compact["diagnostic_evidence"] = evidence
    return compact


def qualification_payload(solver: Path,
                          probes: list[dict[str, Any]],
                          complete: bool) -> dict[str, Any]:
    return {
        "schema_version": QUALIFICATION_REPORT_SCHEMA_VERSION,
        "solver": str(solver),
        "complete": complete,
        "probes": [compact_qualification_probe(probe) for probe in probes],
    }


def write_qualification_log(path: Path | None,
                            solver: Path,
                            probes: list[dict[str, Any]],
                            complete: bool) -> None:
    if path is None:
        return
    payload = qualification_payload(solver, probes, complete)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")


def format_failure_exception(failure: dict[str, Any],
                             qualification_log: Path | None) -> str:
    summary_keys = [
        "case",
        "run_dir",
        "result_path",
        "returncode",
        "timeout_seconds",
        "solver_elapsed_wall_seconds",
        "solver_elapsed_seconds_per_accepted_step",
        "result_step",
        "diagnostic_assembly_timing_count",
        "diagnostic_assembly_timings_per_accepted_step",
        "diagnostic_cut_context_rebuild_count",
        "diagnostic_cut_context_rebuilds_per_accepted_step",
        "diagnostic_assembly_timing_max_cut_volumes_seconds",
        "reference_profile_time_s",
        "reference_profile_validation_passed",
        "passed",
        "errors",
        "diagnostic_errors",
    ]
    summary = {
        key: failure[key]
        for key in summary_keys
        if key in failure
    }
    if qualification_log is not None:
        summary["qualification_log"] = str(qualification_log)
    return json.dumps(summary, indent=2, sort_keys=True)


def case_args_for_run(case_name: str,
                      args: argparse.Namespace) -> argparse.Namespace:
    case_args = argparse.Namespace(**vars(args))
    if (getattr(case_args, "defer_static_physical_gates_to_matrix", False) and
            case_name not in {
                "capillaryarc2d", "droplet2d", "sphere3d",
                "sessile2d", "sessile3d",
            }):
        raise ValueError(
            "matrix-owned static physical gates require a static capillary case")
    kinematic_area_gradient_traction = (
        getattr(case_args, "capillary_force_form", "surface_stress") ==
        "kinematic_area_gradient_traction"
    )
    if kinematic_area_gradient_traction:
        if getattr(case_args, "use_high_order_implicit_cuts", False):
            raise ValueError(
                "kinematic area-gradient traction requires an affine P1 "
                "simplex mesh")
        if not getattr(case_args, "projected_curvature_field", None):
            raise ValueError(
                "kinematic area-gradient traction requires an explicit "
                "projected curvature field")
        set_default(
            case_args,
            "curvature_projection_recovery_mode",
            "kinematic_area_gradient",
        )
        set_default(
            case_args,
            "curvature_projection_kinematic_area_gradient_filter_coefficient",
            0.0,
        )
        set_default(
            case_args,
            "curvature_projection_smoothing_iterations",
            0,
        )
        set_default(
            case_args,
            "cut_cell_pressure_stabilization_policy",
            "incremental",
        )
        if (case_args.curvature_projection_recovery_mode !=
                "kinematic_area_gradient"):
            raise ValueError(
                "kinematic area-gradient traction requires the "
                "kinematic_area_gradient recovery mode")
        if (case_args
                .curvature_projection_kinematic_area_gradient_filter_coefficient
                != 0.0):
            raise ValueError(
                "kinematic area-gradient traction requires a zero "
                "area-gradient filter coefficient")
        if case_args.curvature_projection_smoothing_iterations != 0:
            raise ValueError(
                "kinematic area-gradient traction does not admit "
                "separate post-projection smoothing")
        case_args.require_curvature_projection_diagnostics = True
        case_args.require_curvature_projection_newton_freshness = True
        set_default(
            case_args,
            "expect_curvature_projection_recovery_mode",
            "kinematic_area_gradient",
        )
        set_default(
            case_args,
            "min_diagnostic_curvature_projection_count",
            1,
        )
        set_default(
            case_args,
            "max_diagnostic_curvature_projection_zero_fallback_vertices",
            0,
        )
    if (getattr(
            case_args,
            "initialize_discrete_static_contact_geometry",
            False) and
            case_name != "sessile2d"):
        raise ValueError(
            "discrete static contact initialization is only available for "
            "the stationary two-dimensional sessile case")
    if (case_args.high_order_mpi_production_qualification and
            case_name == "tilt2d"):
        if not getattr(args, "_explicit_linear_solver_type", False):
            case_args.linear_solver_type = "ns"
        if not getattr(args, "_explicit_linear_max_iterations", False):
            case_args.linear_max_iterations = 100
    if case_name in {
            "sessile2d", "sessile3d", "sphere3d", "dynamiccontact2d"}:
        if case_args.steps is None:
            case_args.steps = 10 if case_name == "dynamiccontact2d" else 3
        if case_args.time_step_size is None:
            case_args.time_step_size = 1.0e-3
        if case_args.timeout_seconds is None:
            case_args.timeout_seconds = 300.0
        if case_args.surface_tension is None:
            case_args.surface_tension = 1.0
        # These cases report quantitative physical metrics below.  The legacy
        # dam-break direction/speed gates are unrelated and would otherwise
        # reject a valid static equilibrium or a symmetric contact-line flow.
        case_args.min_max_speed = 0.0
        case_args.min_wet_mean_speed = 0.0
        case_args.min_gate_mean_ux = -1.0
        case_args.min_front_mean_ux = -1.0
        case_args.enable_physical_history_instrumentation = True
        case_args.require_free_surface_energy_history = True
        set_default(
            case_args,
            "max_free_surface_energy_positive_step_increment_relative",
            FREE_SURFACE_ENERGY_MAX_POSITIVE_STEP_INCREMENT_RELATIVE,
        )
        set_default(
            case_args,
            "max_free_surface_energy_above_initial_relative",
            FREE_SURFACE_ENERGY_MAX_ABOVE_INITIAL_RELATIVE,
        )
        case_args.require_time_loop_convergence = True
        case_args.disable_velocity_extension = True
        # The coupled unknown ordering is [phi, velocity, pressure].  FSILS'
        # Navier--Stokes BlockSchur method treats phi as part of its momentum
        # block and can leave the level-set correction too inaccurate for the
        # physical capillary/contact qualification.  Use the monolithic Krylov
        # route by default; every setting remains an ordinary set_default so a
        # caller can still select an alternative explicitly.
        set_default(case_args, "linear_solver_type", "gmres")
        set_default(case_args, "linear_algebra_backend", "fsils")
        set_default(case_args, "linear_preconditioner", "rcs")
        # The inherited P1 mini deck has Max_iterations=1 and
        # Krylov_space_dimension=1.  The latter degenerates the solve to
        # GMRES(1), which stagnates on the capillary saddle system even when
        # given the same total Krylov-step budget as a useful restart space.
        # These are fixed production linear controls; nonlinear tolerances and
        # iteration limits remain unchanged.
        set_default(case_args, "linear_max_iterations", 100)
        set_default(case_args, "linear_krylov_space_dimension", 50)
        # The inherited 1e-4 absolute tolerance lets FSILS accept an internal
        # preconditioned residual while the assembled true residual remains
        # above the nonlinear tolerance.  These fixed values resolve the
        # linearized system accurately without changing any nonlinear or
        # physical acceptance criterion.
        set_default(case_args, "linear_relative_tolerance", 1.0e-8)
        set_default(case_args, "linear_absolute_tolerance", 1.0e-10)
        # Newton is allowed to accept an inexact FSILS correction once its
        # assembled true residual is negligible on the nonlinear scale.  Keep
        # that secondary physical-qualification gate explicit and three orders
        # below the 1e-6 nonlinear absolute tolerance.
        set_default(case_args, "max_fsils_accepted_true_residual_norm", 1.0e-9)
        if case_name in {"sessile2d", "sessile3d", "sphere3d"}:
            generated_curvature_traction = (
                getattr(case_args, "capillary_force_form", "surface_stress") ==
                "generated_curvature_traction"
            )
            if generated_curvature_traction:
                if getattr(
                        case_args,
                        "initialize_discrete_static_capillary_equilibrium",
                        False):
                    raise ValueError(
                        "discrete surface-energy initialization is not "
                        "available for generated curvature traction")
                if (getattr(
                        case_args,
                        "enable_free_surface_conservative_balance_diagnostic",
                        False) or
                        getattr(
                            case_args,
                            "require_free_surface_conservative_balance",
                            False) or
                        getattr(
                            case_args,
                            "max_free_surface_conservative_balance_normalized_imbalance",
                            None) is not None):
                    raise ValueError(
                        "surface-energy conservative-balance controls are "
                        "not available for generated curvature traction")
                if (getattr(
                        case_args,
                        "require_free_surface_pressure_representability_diagnostic",
                        False) or
                        getattr(
                            case_args,
                            "max_free_surface_pressure_representability_relative_distance",
                            None) is not None or
                        getattr(
                            case_args,
                            "initialize_static_compatible_pressure",
                            False)):
                    raise ValueError(
                        "surface-energy pressure-representability controls "
                        "are not available for generated curvature traction")
                case_args.require_free_surface_conservative_balance = False
                case_args.require_free_surface_pressure_representability_diagnostic = False
                case_args.max_free_surface_pressure_representability_relative_distance = None
                case_args.initialize_static_compatible_pressure = False
            else:
                # Require production evidence for the instantaneous
                # conservative pressure/surface-energy split.  The
                # quantitative threshold is a separate opt-in until the
                # refinement matrix calibrates one a priori.
                case_args.require_free_surface_conservative_balance = True
                # Static qualification also requires the accepted Newton
                # state to lie within a fixed distance of the constrained
                # pressure-gradient range.
                case_args.require_free_surface_pressure_representability_diagnostic = True
                set_default(
                    case_args,
                    "max_free_surface_pressure_representability_relative_distance",
                    STATIC_PRESSURE_REPRESENTABILITY_MAX_RELATIVE_DISTANCE,
                )
                # Seed the first static solve from the same constrained
                # pressure/surface-energy pair used by the distance gate.
                set_default(
                    case_args,
                    "initialize_static_compatible_pressure",
                    not bool(getattr(
                        case_args,
                        "initialize_discrete_static_capillary_equilibrium",
                        False,
                    )),
                )
            defer_physical_gates = bool(getattr(
                case_args, "defer_static_physical_gates_to_matrix", False))
            physical_gate_names = (
                "max_capillary_pressure_jump_relative_error",
                "max_capillary_parasitic_capillary_number",
                "max_sessile_contact_angle_error_degrees",
                "max_sessile_pressure_jump_relative_error",
                "max_sessile_liquid_area_relative_error",
                "max_sessile_liquid_volume_relative_error",
                "max_sessile_base_radius_relative_error",
                "max_sessile_apex_height_relative_error",
                "max_sessile_parasitic_capillary_number",
            )
            if defer_physical_gates:
                explicit_gates = [
                    name for name in physical_gate_names
                    if getattr(case_args, name, None) is not None
                ]
                if explicit_gates:
                    raise ValueError(
                        "matrix-owned static physical gates conflict with "
                        "per-run thresholds: " + ", ".join(explicit_gates))
                for name in physical_gate_names:
                    setattr(case_args, name, None)
            # Fixed before execution for the n=16/32 FS-16 matrix.  The angle,
            # area, and pressure bounds are intentionally well below an
            # order-one error while allowing P1 interface sampling error.  A
            # capillary number below 1e-2 rejects dynamically meaningful
            # parasitic currents in a nominal equilibrium.
            if defer_physical_gates:
                pass
            elif case_name == "sphere3d":
                set_default(
                    case_args,
                    "max_capillary_pressure_jump_relative_error",
                    0.15,
                )
                set_default(
                    case_args,
                    "max_capillary_parasitic_capillary_number",
                    1.0e-2,
                )
            else:
                set_default(
                    case_args,
                    "max_sessile_contact_angle_error_degrees",
                    5.0,
                )
                set_default(
                    case_args,
                    "max_sessile_pressure_jump_relative_error",
                    0.15,
                )
                set_default(
                    case_args,
                    "max_sessile_liquid_area_relative_error",
                    0.05,
                )
                if case_name == "sessile3d":
                    case_args.max_sessile_liquid_area_relative_error = None
                    set_default(
                        case_args,
                        "max_sessile_liquid_volume_relative_error",
                        0.05,
                    )
                    set_default(
                        case_args,
                        "max_sessile_base_radius_relative_error",
                        0.05,
                    )
                    set_default(
                        case_args,
                        "max_sessile_apex_height_relative_error",
                        0.05,
                    )
                set_default(
                    case_args,
                    "max_sessile_parasitic_capillary_number",
                    1.0e-2,
                )
        else:
            set_default(
                case_args, "initialize_static_compatible_pressure", False)
            # The continuum Ren--E law supplies a deliberately strict target.
            # The direct gate compares the final wall-interpolated u.m used by
            # the contact residual with the force-law prediction at the same
            # fitted state.  Footprint finite differences remain separately
            # reported as a kinematic/transport observable.
            set_default(case_args, "require_ren_e_speed_sign", True)
            set_default(case_args, "max_ren_e_speed_relative_error", 0.50)
    return case_args


def read_solver_log(run_dir: Path) -> str:
    log_path = run_dir / "solver_run.log"
    if not log_path.exists():
        return ""
    return log_path.read_text(encoding="utf-8", errors="replace")


def terminate_solver_process(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=5.0)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    process.wait()


def run_solver_command(command: list[str],
                       run_dir: Path,
                       args: argparse.Namespace
                       ) -> tuple[subprocess.CompletedProcess[str], float]:
    log_path = run_dir / "solver_run.log"
    start = time.monotonic()
    with log_path.open("w", encoding="utf-8", buffering=1) as log_file:
        process = subprocess.Popen(
            command,
            cwd=run_dir,
            env=solver_environment(args),
            text=True,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            returncode = process.wait(timeout=args.timeout_seconds)
        except subprocess.TimeoutExpired as exc:
            elapsed = time.monotonic() - start
            terminate_solver_process(process)
            exc.output = read_solver_log(run_dir)
            exc.timeout = elapsed
            setattr(exc, "configured_timeout_seconds", args.timeout_seconds)
            raise
    elapsed = time.monotonic() - start
    output = read_solver_log(run_dir)
    completed = subprocess.CompletedProcess(
        args=command,
        returncode=returncode,
        stdout=output,
        stderr=None,
    )
    return completed, elapsed


def configure_case_solver_xml(run_dir: Path, args: argparse.Namespace) -> None:
    configure_solver(
        run_dir / "solver.xml",
        args.steps,
        time_step_size=args.time_step_size,
        disable_cut_stabilization=args.disable_cut_stabilization,
        max_nonlinear_iterations=args.max_nonlinear_iterations,
        linear_relative_tolerance=args.linear_relative_tolerance,
        linear_absolute_tolerance=args.linear_absolute_tolerance,
        linear_max_iterations=args.linear_max_iterations,
        linear_krylov_space_dimension=getattr(
            args, "linear_krylov_space_dimension", None),
        ns_gm_max_iterations=args.ns_gm_max_iterations,
        ns_cg_max_iterations=args.ns_cg_max_iterations,
        ns_gm_tolerance=args.ns_gm_tolerance,
        ns_cg_tolerance=args.ns_cg_tolerance,
        linear_solver_type=args.linear_solver_type,
        linear_algebra_backend=args.linear_algebra_backend,
        linear_preconditioner=args.linear_preconditioner,
        disable_coupled_outer_fgmres=args.disable_coupled_outer_fgmres,
        disable_cut_metadata_scale=args.disable_cut_metadata_scale,
        disable_velocity_extension=args.disable_velocity_extension,
        disable_vtk_output=args.disable_vtk_output,
        final_output_only=args.final_output_only,
        vtk_save_increment=args.vtk_save_increment,
        start_saving_after_step=args.start_saving_after_step,
        generated_interface_geometry=args.generated_interface_geometry,
        implicit_cut_quadrature_backend=args.implicit_cut_quadrature_backend,
        implicit_cut_fallback_policy=args.implicit_cut_fallback_policy,
        required_implicit_cut_backend_qualification=(
            args.required_implicit_cut_backend_qualification),
        implicit_cut_root_tolerance=args.implicit_cut_root_tolerance,
        implicit_cut_max_subdivision_depth=args.implicit_cut_max_subdivision_depth,
        generated_interface_quadrature_order=args.generated_interface_quadrature_order,
        interface_quadrature_order=args.interface_quadrature_order,
        volume_quadrature_order=args.volume_quadrature_order,
        cut_cell_velocity_gradient_penalty=args.cut_cell_velocity_gradient_penalty,
        cut_cell_pressure_gradient_penalty=args.cut_cell_pressure_gradient_penalty,
        cut_cell_pressure_stabilization_policy=getattr(
            args, "cut_cell_pressure_stabilization_policy", None),
        active_domain=getattr(
            args, "level_set_active_domain", "LevelSetNegative"),
        surface_tension=args.surface_tension,
        capillary_force_form=getattr(
            args, "capillary_force_form", "surface_stress"),
        prescribed_capillary_curvature=getattr(
            args, "prescribed_capillary_curvature", None),
        wet_extension_advection_velocity_method=(
            args.wet_extension_advection_velocity_method),
        projected_curvature_field=args.projected_curvature_field,
        curvature_projection_cadence_steps=args.curvature_projection_cadence_steps,
        curvature_projection_max_normalized_fit_residual=(
            args.curvature_projection_max_normalized_fit_residual),
        curvature_projection_max_neighbor_fallback_vertices=(
            args.curvature_projection_max_neighbor_fallback_vertices),
        curvature_projection_max_zero_fallback_vertices=(
            args.curvature_projection_max_zero_fallback_vertices),
        curvature_projection_supplemental_sample_weight=(
            args.curvature_projection_supplemental_sample_weight),
        curvature_projection_recovery_mode=(
            args.curvature_projection_recovery_mode),
        curvature_projection_kinematic_area_gradient_filter_coefficient=(
            getattr(
                args,
                "curvature_projection_kinematic_area_gradient_filter_coefficient",
                None,
            )
        ),
        curvature_projection_narrow_band_width=(
            args.curvature_projection_narrow_band_width),
        curvature_projection_smoothing_iterations=(
            args.curvature_projection_smoothing_iterations),
        curvature_projection_smoothing_relaxation=(
            args.curvature_projection_smoothing_relaxation),
        curvature_projection_smoothing_mode=(
            args.curvature_projection_smoothing_mode),
        enable_static_capillary_equilibrium_initialization=getattr(
            args,
            "initialize_discrete_static_capillary_equilibrium",
            None,
        ),
        static_capillary_volume_tolerance=getattr(
            args, "static_capillary_volume_tolerance", None),
        static_capillary_projected_gradient_tolerance=getattr(
            args,
            "static_capillary_projected_gradient_tolerance",
            None,
        ),
        static_capillary_pressure_representability_max_residual_norm=getattr(
            args,
            "static_capillary_pressure_representability_max_residual_norm",
            None,
        ),
        static_capillary_pressure_representability_max_relative_distance=getattr(
            args,
            "static_capillary_pressure_representability_max_relative_distance",
            None,
        ),
        static_capillary_physical_equilibrium_max_residual_norm=getattr(
            args,
            "static_capillary_physical_equilibrium_max_residual_norm",
            None,
        ),
        static_capillary_constant_pressure_kkt_max_residual_norm=getattr(
            args,
            "static_capillary_constant_pressure_kkt_max_residual_norm",
            None,
        ),
        static_capillary_constant_pressure_kkt_max_relative_distance=getattr(
            args,
            "static_capillary_constant_pressure_kkt_max_relative_distance",
            None,
        ),
        static_capillary_finite_difference_relative_step=getattr(
            args,
            "static_capillary_finite_difference_relative_step",
            None,
        ),
        static_capillary_max_iterations=getattr(
            args, "static_capillary_max_iterations", None),
        static_capillary_max_topology_epoch_transitions=getattr(
            args,
            "static_capillary_max_topology_epoch_transitions",
            None,
        ),
        static_capillary_limited_memory_history_size=getattr(
            args,
            "static_capillary_limited_memory_history_size",
            None,
        ),
        static_capillary_limited_memory_curvature_tolerance=getattr(
            args,
            "static_capillary_limited_memory_curvature_tolerance",
            None,
        ),
        enable_reinitialization=getattr(
            args, "enable_level_set_reinitialization", None),
        reinitialization_cadence_steps=getattr(
            args, "reinitialization_cadence_steps", None),
        enable_volume_correction=args.enable_level_set_volume_correction,
        volume_correction_cadence_steps=args.volume_correction_cadence_steps,
        volume_correction_use_initial_volume=(
            args.volume_correction_use_initial_volume),
        volume_correction_tolerance=args.volume_correction_tolerance,
        volume_correction_max_iterations=args.volume_correction_max_iterations,
        volume_correction_maximum_cumulative_interface_displacement_fraction=(
            args.volume_correction_maximum_cumulative_interface_displacement_fraction
        ),
    )


def run_case(case_name: str, solver: Path, args: argparse.Namespace) -> dict[str, Any]:
    source = CASES[case_name]
    if source is not None and not source.exists():
        raise FileNotFoundError(source)
    active_domain = normalized_active_domain(getattr(
        args, "level_set_active_domain", "LevelSetNegative"))
    active_side_cases = {
        "capillaryarc2d",
        "droplet2d",
        "sphere3d",
        "sessile2d",
        "sessile3d",
        "dynamiccontact2d",
    }
    if (active_domain != "LevelSetNegative" and
            case_name not in active_side_cases):
        raise ValueError(
            "the positive liquid-side option is available only for supported "
            "synthetic static capillary cases")

    temp_context = None
    if args.preserve_run_dir:
        temp_name = tempfile.mkdtemp(prefix=f"dam_break_{case_name}_")
    else:
        temp_context = tempfile.TemporaryDirectory(prefix=f"dam_break_{case_name}_")
        temp_name = temp_context.name

    def write_solver_log(run_dir: Path, output: str) -> None:
        (run_dir / "solver_run.log").write_text(output, encoding="utf-8")

    def write_failure(failure: dict[str, Any]) -> None:
        failure.setdefault("passed", False)
        write_qualification_log(args.qualification_log, solver, [failure], complete=False)

    try:
        run_dir = Path(temp_name) / case_name
        uses_kinematic_area_gradient_traction = (
            getattr(args, "capillary_force_form", "surface_stress") ==
            "kinematic_area_gradient_traction"
        )
        if source is None:
            if case_name == "curvedtet3d":
                write_curved_tet3d_case(run_dir, args.steps)
            elif case_name == "capillaryarc2d":
                pressure_jump = 0.0
                if args.high_order_capillary_balance_smoke:
                    pressure_jump = float(args.surface_tension) / CAPILLARY_ARC_RADIUS
                write_capillary_arc2d_case(
                    run_dir, args.steps, pressure_jump,
                    args.synthetic_nx, args.synthetic_ny,
                    uses_kinematic_area_gradient_traction,
                    active_domain)
            elif case_name == "droplet2d":
                pressure_jump = 0.0
                if getattr(args, "high_order_capillary_droplet_equilibrium_smoke", False):
                    pressure_jump = (
                        float(args.surface_tension) / CAPILLARY_DROPLET_RADIUS
                    )
                write_capillary_droplet2d_case(
                    run_dir, args.steps, pressure_jump,
                    args.synthetic_nx, args.synthetic_ny,
                    uses_kinematic_area_gradient_traction,
                    active_domain,
                    getattr(
                        args,
                        "capillary_droplet_center_offset",
                        (0.0, 0.0),
                    ),
                    surface_tension=(
                        None if args.surface_tension is None else
                        float(args.surface_tension)))
            elif case_name == "sphere3d":
                if not (
                        args.synthetic_nx == args.synthetic_ny ==
                        args.synthetic_nz):
                    raise ValueError(
                        "sphere3d requires an isotropic synthetic resolution")
                write_sphere_case(
                    run_dir,
                    args.steps,
                    args.synthetic_nx,
                    args.sessile_radius,
                    float(args.surface_tension),
                    float(args.time_step_size),
                    args.level_set_positive_scale,
                    active_domain,
                    getattr(
                        args,
                        "capillary_sphere_center_offset",
                        (0.0, 0.0, 0.0),
                    ),
                )
            elif case_name == "capillarywave2d":
                surface_tension = (
                    0.5 if args.surface_tension is None else float(args.surface_tension)
                )
                write_capillary_wave2d_case(
                    run_dir, args.steps, surface_tension, args.time_step_size,
                    args.synthetic_nx, args.synthetic_ny,
                    uses_kinematic_area_gradient_traction)
            elif case_name in {"sessile2d", "dynamiccontact2d"}:
                dynamic = case_name == "dynamiccontact2d"
                initial_angle = (
                    args.dynamic_initial_contact_angle_degrees
                    if dynamic else args.contact_angle_degrees
                )
                write_sessile2d_case(
                    run_dir,
                    args.steps,
                    args.synthetic_nx,
                    args.synthetic_ny,
                    initial_angle,
                    args.contact_angle_degrees,
                    args.sessile_radius,
                    float(args.surface_tension),
                    float(args.time_step_size),
                    args.contact_line_mobility,
                    args.wall_slip_length,
                    dynamic,
                    (args.dynamic_contact_wall
                     if dynamic else getattr(
                         args, "sessile_contact_wall", "wall_bottom")),
                    ("dynamic" if dynamic else
                     getattr(args, "sessile_contact_line_model", "dynamic")),
                    getattr(args, "level_set_positive_scale", 1.0),
                    getattr(
                        args,
                        "initialize_discrete_static_contact_geometry",
                        False,
                    ),
                    uses_kinematic_area_gradient_traction,
                    active_domain,
                    getattr(args, "sessile_tangent_center_offset", 0.0),
                )
            elif case_name == "sessile3d":
                if not (
                        args.synthetic_nx == args.synthetic_ny ==
                        args.synthetic_nz):
                    raise ValueError(
                        "sessile3d requires an isotropic synthetic resolution")
                write_sessile_sphere_case(
                    run_dir,
                    args.steps,
                    args.synthetic_nx,
                    args.contact_angle_degrees,
                    args.sessile_radius,
                    float(args.surface_tension),
                    float(args.time_step_size),
                    args.sessile_contact_wall_3d,
                    args.level_set_positive_scale,
                    active_domain,
                    getattr(
                        args,
                        "sessile_tangent_center_offset_3d",
                        (0.0, 0.0),
                    ),
                )
            else:
                write_mini_case(
                    run_dir,
                    args.steps,
                    static=(case_name == "static2d"),
                    simplex_mesh=uses_kinematic_area_gradient_traction,
                )
            if (args.use_high_order_implicit_cuts or
                    uses_kinematic_area_gradient_traction or
                    case_name in {
                        "sessile2d", "sessile3d", "dynamiccontact2d",
                        "sphere3d",
                    }):
                configure_case_solver_xml(run_dir, args)
        else:
            run_dir = Path(temp_name) / source.name
            copy_case(source, run_dir, args.source_ref)
            regenerate_mms_case_if_requested(case_name, run_dir, args)
            configure_case_solver_xml(run_dir, args)

        try:
            command = solver_command(solver, args)
            completed, solver_elapsed_wall_seconds = run_solver_command(
                command, run_dir, args)
        except subprocess.TimeoutExpired as exc:
            output = exc.stdout or exc.output or ""
            if isinstance(output, bytes):
                output = output.decode("utf-8", errors="replace")
            write_solver_log(run_dir, output)
            tail = "\n".join(output.splitlines()[-80:])
            diagnostics = parse_solver_diagnostics(output)
            failure = diagnostic_timeout_metrics(case_name, run_dir, diagnostics)
            add_incomplete_solve_output_metrics(
                failure, case_name, run_dir, diagnostics, args)
            failure["time_series_collation"] = collate_solver_time_series(run_dir, args)
            failure["timeout_seconds"] = getattr(
                exc, "configured_timeout_seconds", args.timeout_seconds)
            failure["solver_elapsed_wall_seconds"] = exc.timeout
            failure["command"] = command
            failure["stdout_tail"] = tail
            diagnostic_errors = evaluate_timeout_diagnostics(failure, args)
            failure["diagnostic_errors"] = diagnostic_errors
            if args.disable_coupled_outer_fgmres:
                failure["disable_coupled_outer_fgmres"] = True
            if args.disable_cut_metadata_scale:
                failure["disable_cut_metadata_scale"] = True
            if args.disable_velocity_extension:
                failure["disable_velocity_extension"] = True
            if args.disable_vtk_output:
                failure["disable_vtk_output"] = True
            if args.enable_blockschur_true_residual_retry:
                failure["enable_blockschur_true_residual_retry"] = True
            add_solver_control_overrides(failure, args)
            if args.allow_timeout_diagnostics and not diagnostic_errors:
                failure["passed"] = True
                failure["errors"] = []
                return failure
            failure["errors"] = diagnostic_errors or ["solver timed out"]
            write_failure(failure)
            raise RuntimeError(format_failure_exception(
                failure, args.qualification_log)) from exc
        write_solver_log(run_dir, completed.stdout)
        if completed.returncode != 0:
            tail = "\n".join(completed.stdout.splitlines()[-80:])
            failure = {
                "case": case_name,
                "run_dir": str(run_dir),
                "command": solver_command(solver, args),
                "returncode": completed.returncode,
                "solver_elapsed_wall_seconds": solver_elapsed_wall_seconds,
                "diagnostics": parse_solver_diagnostics(completed.stdout),
                "stdout_tail": tail,
                "time_series_collation": collate_solver_time_series(run_dir, args),
            }
            add_diagnostic_metrics(failure, failure["diagnostics"])
            add_incomplete_solve_output_metrics(
                failure, case_name, run_dir, failure["diagnostics"], args)
            previous = previous_invalid_pressure(load_benchmark(run_dir))
            if previous is not None:
                failure["pressure_gauge_previous_invalid"] = previous
                gauge_value = failure.get("diagnostic_pressure_gauge_value")
                if gauge_value is not None:
                    failure["diagnostic_pressure_gauge_previous_invalid_difference"] = (
                        gauge_value - previous
                    )
            diagnostic_errors = evaluate_timeout_diagnostics(failure, args)
            failure["diagnostic_errors"] = diagnostic_errors
            if args.disable_coupled_outer_fgmres:
                failure["disable_coupled_outer_fgmres"] = True
            if args.disable_cut_metadata_scale:
                failure["disable_cut_metadata_scale"] = True
            if args.disable_velocity_extension:
                failure["disable_velocity_extension"] = True
            if args.disable_vtk_output:
                failure["disable_vtk_output"] = True
            if args.enable_blockschur_true_residual_retry:
                failure["enable_blockschur_true_residual_retry"] = True
            add_solver_control_overrides(failure, args)
            failure["errors"] = [
                f"solver exited with return code {completed.returncode}",
                *diagnostic_errors,
            ]
            if args.allow_failure_diagnostics and not diagnostic_errors:
                failure["passed"] = True
                failure["errors"] = []
                return failure
            write_failure(failure)
            raise RuntimeError(format_failure_exception(
                failure, args.qualification_log))

        diagnostics = parse_solver_diagnostics(completed.stdout)
        if args.disable_vtk_output:
            metrics = {
                "output_metrics_skipped": True,
                "output_metrics_skip_reason": "VTK output disabled",
            }
        else:
            result_step = final_result_step(args.steps, diagnostics)
            result = result_path(run_dir, result_step)
            metrics = compute_metrics(
                case_name,
                run_dir,
                result,
                enable_physical_history=args.enable_physical_history_instrumentation,
                accepted_steps=diagnostics.get("time_loop", {}).get(
                    "accepted_steps", []),
                transient_solve=diagnostics.get("solver_controls", {}).get(
                    "transient_solve"),
            )
            metrics["result_step"] = result_step
            metrics["result_path"] = str(result)
        benchmark = load_benchmark(run_dir)
        if benchmark:
            metrics["benchmark"] = benchmark
        add_diagnostic_metrics(metrics, diagnostics)
        if args.enable_physical_history_instrumentation:
            add_production_physical_liquid_volume_metrics(metrics)
        if getattr(args, "require_level_set_mass_correction_histories", False):
            add_level_set_mass_correction_history_metrics(
                metrics, solver_fluid_density(run_dir))
        if isinstance(benchmark.get("capillary_wave"), dict):
            add_capillary_wave_temporal_liquid_volume_metrics(metrics)
        metrics["time_series_collation"] = collate_solver_time_series(run_dir, args)
        if not args.disable_vtk_output:
            add_reference_profile_metrics(metrics, run_dir, result, args)
        metrics["case"] = case_name
        metrics["command"] = solver_command(solver, args)
        metrics["run_dir"] = str(run_dir)
        metrics["solver_elapsed_wall_seconds"] = solver_elapsed_wall_seconds
        metrics["steps"] = args.steps
        if args.time_step_size is not None:
            metrics["time_step_size"] = args.time_step_size
        if args.disable_coupled_outer_fgmres:
            metrics["disable_coupled_outer_fgmres"] = True
        if args.disable_cut_metadata_scale:
            metrics["disable_cut_metadata_scale"] = True
        if args.disable_velocity_extension:
            metrics["disable_velocity_extension"] = True
        if args.disable_vtk_output:
            metrics["disable_vtk_output"] = True
        if args.enable_blockschur_true_residual_retry:
            metrics["enable_blockschur_true_residual_retry"] = True
        if args.final_output_only:
            metrics["final_output_only"] = True
        add_solver_control_overrides(metrics, args)
        errors = evaluate(metrics, args)
        metrics["passed"] = not errors
        metrics["errors"] = errors
        if errors:
            write_failure(metrics)
            raise RuntimeError(format_failure_exception(
                metrics, args.qualification_log))
        return metrics
    finally:
        if temp_context is not None:
            temp_context.cleanup()


def set_default(args: argparse.Namespace, name: str, value: Any) -> None:
    if getattr(args, name, None) is None:
        setattr(args, name, value)


def synthetic_curvature_projection_band_width(args: argparse.Namespace) -> float:
    # Namespace-based unit/instrumentation callers may omit parser defaults;
    # mirror the CLI's 16-by-16 synthetic mesh in that case.
    nx = getattr(args, "synthetic_nx", 16)
    ny = getattr(args, "synthetic_ny", 16)
    if (not isinstance(nx, int) or isinstance(nx, bool) or nx <= 0 or
            not isinstance(ny, int) or isinstance(ny, bool) or ny <= 0):
        raise ValueError(
            "synthetic curvature-projection band requires positive nx and ny"
        )
    return 1.0 / float(max(nx, ny))


def curvature_sample_adjacency_build_budget(args: argparse.Namespace) -> int:
    steps = getattr(args, "steps", None)
    if isinstance(steps, int) and steps > 0:
        return steps
    return 1


def capillary_wave_curvature_projection_cache_miss_budget(
        args: argparse.Namespace) -> int:
    """Budget the state-changing projection opportunities in a wave run."""
    steps = getattr(args, "steps", None)
    if not isinstance(steps, int) or steps <= 0:
        return 1
    # One initial projection, then accepted_step plus the two accepted-state
    # synchronization points exercised by each transient step.  Cache hits at
    # residual/Jacobian synchronization points remain independently required.
    return 1 + 3 * steps


def remember_explicit_cli_overrides(args: argparse.Namespace) -> None:
    for name in (
        "linear_solver_type",
        "linear_max_iterations",
    ):
        setattr(args, f"_explicit_{name}", getattr(args, name) is not None)


def normalized_option(value: str | None) -> str:
    return (value or "").strip().lower()


def require_profile_production_linear_solver_policy(args: argparse.Namespace) -> None:
    if (not args.high_order_3d_benchmark_profile_qualification or
            args.allow_experimental_profile_linear_solver):
        return

    required = {
        "linear_algebra_backend": "fsils",
        "linear_preconditioner": "fsils",
        "linear_solver_type": "ns",
    }
    actual = {
        name: normalized_option(getattr(args, name))
        for name in required
    }
    mismatches = [
        f"{name}={actual[name] or '<unset>'} (required {expected})"
        for name, expected in required.items()
        if actual[name] != expected
    ]
    if mismatches:
        raise ValueError(
            "The D18/D38 high-order profile qualification is production-gated "
            "on FSILS BlockSchur because the GMRES/RCS route is known to stall "
            "on long profiles. Use --allow-experimental-profile-linear-solver "
            "only for diagnostic probes. Mismatches: " + "; ".join(mismatches)
        )


def apply_level_set_advection_velocity_diagnostic_gate_defaults(
        args: argparse.Namespace) -> None:
    high_order_motion_gate = any(
        getattr(args, name, False)
        for name in (
            "high_order_production_qualification",
            "high_order_mpi_production_qualification",
            "high_order_visible_motion_demo",
            "high_order_mpi_motion_smoke",
            "high_order_capillary_wave_smoke",
            "high_order_volume_corrected_motion_smoke",
        )
    )
    if not high_order_motion_gate:
        return

    args.trace_level_set_advection_velocity = True
    args.require_level_set_advection_velocity_diagnostics = True
    set_default(
        args,
        "wet_extension_advection_velocity_method",
        "wall_compatible_normal",
    )
    expected_method = args.wet_extension_advection_velocity_method
    set_default(
        args,
        "expect_level_set_advection_velocity_extension_method",
        expected_method,
    )
    if expected_method == "nearest_active_vertex":
        set_default(
            args,
            "expect_level_set_advection_velocity_interface_sample_source",
            "all_cells",
        )


def apply_high_order_production_qualification_defaults(
        args: argparse.Namespace) -> None:
    if not args.high_order_production_qualification:
        if (args.steps is None and
                not args.high_order_mpi_production_qualification and
                not args.high_order_visible_motion_demo and
                not args.high_order_3d_benchmark_smoke and
                not args.high_order_3d_benchmark_qualification and
                not args.high_order_3d_benchmark_profile_qualification and
                not args.high_order_curved_3d_simplex_smoke and
                not args.high_order_mpi_motion_smoke and
                not args.high_order_capillary_projection_smoke and
                not args.high_order_capillary_response_smoke and
                not args.high_order_capillary_balance_smoke and
                not getattr(
                    args,
                    "high_order_capillary_droplet_equilibrium_smoke",
                    False,
                ) and
                not getattr(args, "high_order_capillary_wave_smoke", False) and
                not args.high_order_volume_corrected_motion_smoke):
            args.steps = 1
        return
    if args.high_order_mpi_production_qualification:
        raise ValueError(
            "--high-order-production-qualification cannot be combined with "
            "--high-order-mpi-production-qualification"
        )
    if args.high_order_3d_benchmark_qualification:
        raise ValueError(
            "--high-order-production-qualification cannot be combined with "
            "--high-order-3d-benchmark-qualification"
        )
    if args.high_order_3d_benchmark_profile_qualification:
        raise ValueError(
            "--high-order-production-qualification cannot be combined with "
            "--high-order-3d-benchmark-profile-qualification"
        )
    if args.high_order_mpi_motion_smoke:
        raise ValueError(
            "--high-order-production-qualification cannot be combined with "
            "--high-order-mpi-motion-smoke"
        )
    if args.high_order_curved_3d_simplex_smoke:
        raise ValueError(
            "--high-order-production-qualification cannot be combined with "
            "--high-order-curved-3d-simplex-smoke"
        )

    if not args.case:
        args.case = list(HIGH_ORDER_PRODUCTION_CASES)
    if args.steps is None:
        args.steps = 20
    set_default(args, "timeout_seconds", 900.0)
    args.use_high_order_implicit_cuts = True
    args.required_implicit_cut_backend_qualification = "ProductionQualified"
    args.disable_cut_stabilization = False
    args.require_time_loop_convergence = True
    args.require_process_memory_diagnostics = True
    args.require_basis_cache_diagnostics = True
    args.require_high_order_cut_context_diagnostics = True
    args.require_eigen_factorization_diagnostics = True
    args.require_active_pressure_support_diagnostics = True

    set_default(args, "linear_algebra_backend", "eigen")
    if args.min_max_speed == 1.0e-2:
        args.min_max_speed = 1.0e-3
    if args.min_wet_mean_speed == 2.5e-4:
        args.min_wet_mean_speed = 1.0e-4
    if args.min_gate_mean_ux == 1.0e-4:
        args.min_gate_mean_ux = -1.0
    if args.min_front_mean_ux == 1.0e-4:
        args.min_front_mean_ux = -1.0
    set_default(args, "min_diagnostic_solution_velocity_range", 1.0e-3)
    set_default(args, "min_diagnostic_pressure_range", 100.0)
    set_default(args, "max_wet_fraction_volume_error", 1.0e-8)
    set_default(args, "max_diagnostic_cut_context_rebuilds_per_step", 4.0)
    set_default(args, "min_diagnostic_cut_context_refresh_skips", 1)
    set_default(args, "max_diagnostic_generated_cell_cache_full_miss_rebuilds", 1)
    set_default(args, "max_diagnostic_assembly_timings_per_step", 4.0)
    set_default(args, "max_diagnostic_extra_assembly_timings_per_step", 3.0)
    set_default(args, "max_diagnostic_process_basis_cache_entries", 8)
    set_default(args, "max_diagnostic_process_basis_cache_entry_growth", 8)
    set_default(args, "max_diagnostic_implicit_cut_fallback_cells", 0)
    set_default(args, "min_diagnostic_achieved_interface_quadrature_order", 2)
    set_default(args, "min_diagnostic_achieved_volume_quadrature_order", 2)
    set_default(args, "max_eigen_factorization_pressure_zero_rows", 0)
    set_default(args, "max_eigen_factorization_pressure_zero_cols", 0)
    set_default(args, "max_eigen_factorization_nonfinite_entries", 0)
    set_default(args, "max_time_loop_nonlinear_iterations_per_step", 3)
    set_default(args, "max_time_loop_linear_iterations_per_step", 10)
    set_default(args, "min_interface_height_change", 1.0e-4)
    set_default(args, "min_interface_mean_abs_height_change", 2.0e-5)
    set_default(args, "min_interface_slope_change", 5.0e-5)
    set_default(args, "min_interface_final_height_span", 1.0e-4)


def apply_high_order_mpi_production_qualification_defaults(
        args: argparse.Namespace) -> None:
    if not args.high_order_mpi_production_qualification:
        return
    if args.high_order_3d_benchmark_smoke:
        raise ValueError(
            "--high-order-mpi-production-qualification cannot be combined with "
            "--high-order-3d-benchmark-smoke"
        )
    if args.high_order_3d_benchmark_qualification:
        raise ValueError(
            "--high-order-mpi-production-qualification cannot be combined with "
            "--high-order-3d-benchmark-qualification"
        )
    if args.high_order_3d_benchmark_profile_qualification:
        raise ValueError(
            "--high-order-mpi-production-qualification cannot be combined with "
            "--high-order-3d-benchmark-profile-qualification"
        )
    if args.high_order_mpi_motion_smoke:
        raise ValueError(
            "--high-order-mpi-production-qualification cannot be combined with "
            "--high-order-mpi-motion-smoke"
        )
    if args.high_order_curved_3d_simplex_smoke:
        raise ValueError(
            "--high-order-mpi-production-qualification cannot be combined with "
            "--high-order-curved-3d-simplex-smoke"
        )

    if not args.case:
        args.case = list(HIGH_ORDER_MPI_PRODUCTION_CASES)
    if args.steps is None:
        args.steps = 20
    set_default(args, "timeout_seconds", 1200.0)
    set_default(args, "mpi_ranks", 2)
    args.use_high_order_implicit_cuts = True
    args.required_implicit_cut_backend_qualification = "ProductionQualified"
    args.disable_cut_stabilization = False
    args.require_time_loop_convergence = True
    args.require_process_memory_diagnostics = True
    args.require_basis_cache_diagnostics = True
    args.require_high_order_cut_context_diagnostics = True
    args.require_cut_context_solution_source_diagnostics = True
    args.require_active_pressure_support_diagnostics = True
    args.enable_fsils_matrix_diagnostics = True
    args.require_fsils_matrix_diagnostics = True

    set_default(args, "linear_algebra_backend", "fsils")
    set_default(args, "linear_preconditioner", "fsils")
    set_default(args, "linear_solver_type", "gmres")
    set_default(args, "linear_relative_tolerance", 1.0e-4)
    set_default(args, "linear_absolute_tolerance", 1.0e-4)
    # FSILS restarted GMRES interprets Max_iterations as restart cycles.
    # With the production cases' Krylov dimension of 80, 7 cycles cap the
    # reported total Krylov work at 567 iterations per nonlinear solve.
    set_default(args, "linear_max_iterations", 7)
    if args.min_max_speed == 1.0e-2:
        args.min_max_speed = 1.0e-3
    if args.min_wet_mean_speed == 2.5e-4:
        args.min_wet_mean_speed = 1.0e-4
    if args.min_gate_mean_ux == 1.0e-4:
        args.min_gate_mean_ux = -1.0
    if args.min_front_mean_ux == 1.0e-4:
        args.min_front_mean_ux = -1.0
    set_default(args, "min_diagnostic_solution_velocity_range", 1.0e-3)
    set_default(args, "min_diagnostic_pressure_range", 100.0)
    set_default(args, "max_wet_fraction_volume_error", 1.0e-8)
    set_default(args, "max_fsils_matrix_missing_diag", 0)
    set_default(args, "max_fsils_matrix_duplicate_diag_entries", 0)
    set_default(args, "max_fsils_matrix_duplicate_diag_rows", 0)
    set_default(args, "max_fsils_matrix_nonfinite_entries", 0)
    set_default(args, "max_diagnostic_cut_context_rebuilds_per_step", 5.0)
    set_default(args, "min_diagnostic_cut_context_refresh_skips", 1)
    set_default(args, "max_diagnostic_assembly_timings_per_step", 4.0)
    set_default(args, "max_diagnostic_extra_assembly_timings_per_step", 3.0)
    set_default(args, "max_diagnostic_process_rss_kb", 350000.0)
    set_default(args, "max_diagnostic_process_rss_growth_kb", 175000.0)
    set_default(args, "max_diagnostic_process_basis_cache_entries", 8)
    set_default(args, "max_diagnostic_process_basis_cache_entry_growth", 8)
    set_default(args, "max_diagnostic_implicit_cut_fallback_cells", 0)
    set_default(args, "min_diagnostic_achieved_interface_quadrature_order", 2)
    set_default(args, "min_diagnostic_achieved_volume_quadrature_order", 2)
    set_default(args, "expect_selected_implicit_cut_quadrature_backend",
                "SayeHyperrectangle")
    set_default(args, "expect_implicit_cut_backend_qualification",
                "ProductionQualified")
    set_default(args, "max_time_loop_nonlinear_iterations_per_step", 3)
    set_default(args, "max_time_loop_linear_iterations_per_step", 600)
    set_default(args, "min_interface_height_change", 1.0e-4)
    set_default(args, "min_interface_mean_abs_height_change", 2.0e-5)
    set_default(args, "min_interface_slope_change", 5.0e-5)
    set_default(args, "min_interface_final_height_span", 1.0e-4)


def apply_high_order_visible_motion_demo_defaults(
        args: argparse.Namespace) -> None:
    if not args.high_order_visible_motion_demo:
        return
    if args.high_order_production_qualification:
        raise ValueError(
            "--high-order-visible-motion-demo cannot be combined with "
            "--high-order-production-qualification"
        )
    if args.high_order_mpi_production_qualification:
        raise ValueError(
            "--high-order-visible-motion-demo cannot be combined with "
            "--high-order-mpi-production-qualification"
        )
    if args.high_order_3d_benchmark_smoke:
        raise ValueError(
            "--high-order-visible-motion-demo cannot be combined with "
            "--high-order-3d-benchmark-smoke"
        )
    if args.high_order_3d_benchmark_qualification:
        raise ValueError(
            "--high-order-visible-motion-demo cannot be combined with "
            "--high-order-3d-benchmark-qualification"
        )
    if args.high_order_3d_benchmark_profile_qualification:
        raise ValueError(
            "--high-order-visible-motion-demo cannot be combined with "
            "--high-order-3d-benchmark-profile-qualification"
        )
    if args.high_order_curved_3d_simplex_smoke:
        raise ValueError(
            "--high-order-visible-motion-demo cannot be combined with "
            "--high-order-curved-3d-simplex-smoke"
        )
    if args.high_order_mpi_motion_smoke:
        raise ValueError(
            "--high-order-visible-motion-demo cannot be combined with "
            "--high-order-mpi-motion-smoke"
        )

    if not args.case:
        args.case = list(HIGH_ORDER_VISIBLE_MOTION_CASES)
    if args.steps is None:
        args.steps = 20
    set_default(args, "timeout_seconds", 300.0)
    set_default(args, "mpi_ranks", 2)
    set_default(args, "max_solver_elapsed_seconds_per_accepted_step", 1.0)
    args.use_high_order_implicit_cuts = True
    args.required_implicit_cut_backend_qualification = "ProductionQualified"
    args.disable_cut_stabilization = False
    args.require_time_loop_convergence = True
    args.require_process_memory_diagnostics = True
    args.require_basis_cache_diagnostics = True
    args.require_high_order_cut_context_diagnostics = True
    args.require_cut_context_solution_source_diagnostics = True
    args.require_active_pressure_support_diagnostics = True
    args.enable_fsils_matrix_diagnostics = True
    args.require_fsils_matrix_diagnostics = True

    set_default(args, "linear_algebra_backend", "fsils")
    set_default(args, "linear_preconditioner", "fsils")
    set_default(args, "linear_solver_type", "ns")
    set_default(args, "linear_relative_tolerance", 1.0e-4)
    set_default(args, "linear_absolute_tolerance", 1.0e-4)
    set_default(args, "linear_max_iterations", 100)
    if args.min_max_speed == 1.0e-2:
        args.min_max_speed = 0.05
    if args.min_wet_mean_speed == 2.5e-4:
        args.min_wet_mean_speed = 0.01
    if args.min_gate_mean_ux == 1.0e-4:
        args.min_gate_mean_ux = -1.0
    if args.min_front_mean_ux == 1.0e-4:
        args.min_front_mean_ux = -1.0
    set_default(args, "min_diagnostic_solution_velocity_range", 0.02)
    set_default(args, "min_diagnostic_pressure_range", 100.0)
    set_default(args, "max_wet_fraction_volume_error", 1.0e-8)
    set_default(args, "max_fsils_matrix_missing_diag", 0)
    set_default(args, "max_fsils_matrix_duplicate_diag_entries", 0)
    set_default(args, "max_fsils_matrix_duplicate_diag_rows", 0)
    set_default(args, "max_fsils_matrix_nonfinite_entries", 0)
    set_default(args, "max_diagnostic_cut_context_rebuilds_per_step", 5.0)
    set_default(args, "min_diagnostic_cut_context_refresh_skips", 1)
    set_default(args, "max_diagnostic_generated_cell_cache_full_miss_rebuilds", 1)
    set_default(args, "max_diagnostic_assembly_timings_per_step", 4.0)
    set_default(args, "max_diagnostic_extra_assembly_timings_per_step", 3.0)
    set_default(args, "max_diagnostic_process_rss_kb", 350000.0)
    set_default(args, "max_diagnostic_process_rss_growth_kb", 175000.0)
    set_default(args, "max_diagnostic_process_basis_cache_entries", 8)
    set_default(args, "max_diagnostic_process_basis_cache_entry_growth", 8)
    set_default(args, "max_diagnostic_implicit_cut_fallback_cells", 0)
    set_default(args, "min_diagnostic_achieved_interface_quadrature_order", 2)
    set_default(args, "min_diagnostic_achieved_volume_quadrature_order", 2)
    set_default(args, "expect_selected_implicit_cut_quadrature_backend",
                "SayeHyperrectangle")
    set_default(args, "expect_implicit_cut_backend_qualification",
                "ProductionQualified")
    set_default(args, "max_time_loop_nonlinear_iterations_per_step", 3)
    set_default(args, "max_time_loop_linear_iterations_per_step", 100)
    set_default(args, "min_interface_height_change", 0.02)
    set_default(args, "min_interface_mean_abs_height_change", 0.005)
    set_default(args, "min_interface_slope_change", 0.02)
    set_default(args, "min_interface_final_height_span", 0.02)


def apply_high_order_3d_benchmark_smoke_defaults(
        args: argparse.Namespace) -> None:
    if not args.high_order_3d_benchmark_smoke:
        return
    if args.high_order_production_qualification:
        raise ValueError(
            "--high-order-3d-benchmark-smoke cannot be combined with "
            "--high-order-production-qualification"
        )
    if args.high_order_mpi_production_qualification:
        raise ValueError(
            "--high-order-3d-benchmark-smoke cannot be combined with "
            "--high-order-mpi-production-qualification"
        )
    if args.high_order_3d_benchmark_qualification:
        raise ValueError(
            "--high-order-3d-benchmark-smoke cannot be combined with "
            "--high-order-3d-benchmark-qualification"
        )
    if args.high_order_3d_benchmark_profile_qualification:
        raise ValueError(
            "--high-order-3d-benchmark-smoke cannot be combined with "
            "--high-order-3d-benchmark-profile-qualification"
        )
    if args.high_order_mpi_motion_smoke:
        raise ValueError(
            "--high-order-3d-benchmark-smoke cannot be combined with "
            "--high-order-mpi-motion-smoke"
        )
    if args.high_order_curved_3d_simplex_smoke:
        raise ValueError(
            "--high-order-3d-benchmark-smoke cannot be combined with "
            "--high-order-curved-3d-simplex-smoke"
        )

    if not args.case:
        args.case = list(HIGH_ORDER_3D_BENCHMARK_CASES)
    if args.steps is None:
        args.steps = 1
    set_default(args, "timeout_seconds", 600.0)
    args.use_high_order_implicit_cuts = True
    args.disable_vtk_output = True
    args.require_time_loop_convergence = True
    args.require_process_memory_diagnostics = True
    args.require_basis_cache_diagnostics = True
    args.require_high_order_cut_context_diagnostics = True
    args.require_cut_context_solution_source_diagnostics = True
    args.enable_fsils_matrix_diagnostics = True
    args.require_fsils_matrix_diagnostics = True

    set_default(args, "implicit_cut_quadrature_backend", "Auto")
    set_default(args, "expect_selected_implicit_cut_quadrature_backend",
                "HighOrderSubcell")
    set_default(args, "linear_algebra_backend", "fsils")
    set_default(args, "linear_preconditioner", "fsils")
    set_default(args, "linear_solver_type", "ns")
    set_default(args, "ns_gm_max_iterations", 200)
    set_default(args, "ns_cg_max_iterations", 200)
    set_default(args, "ns_gm_tolerance", 1.0e-4)
    set_default(args, "ns_cg_tolerance", 1.0e-4)
    set_default(args, "max_fsils_matrix_missing_diag", 0)
    set_default(args, "max_fsils_matrix_duplicate_diag_entries", 0)
    set_default(args, "max_fsils_matrix_duplicate_diag_rows", 0)
    set_default(args, "max_fsils_matrix_nonfinite_entries", 0)
    set_default(args, "max_diagnostic_implicit_cut_fallback_cells", 0)
    set_default(args, "min_diagnostic_achieved_interface_quadrature_order", 1)
    set_default(args, "min_diagnostic_achieved_volume_quadrature_order", 2)
    set_default(args, "max_diagnostic_cut_context_rebuilds_per_step", 4.0)
    set_default(args, "max_diagnostic_generated_cell_cache_full_miss_rebuilds", 1)
    set_default(args, "max_diagnostic_process_rss_kb", 700000.0)
    set_default(args, "max_diagnostic_process_rss_growth_kb", 300000.0)
    set_default(args, "max_diagnostic_process_basis_cache_entries", 32)
    set_default(args, "max_diagnostic_process_basis_cache_entry_growth", 32)


def apply_high_order_3d_benchmark_qualification_defaults(
        args: argparse.Namespace) -> None:
    if not args.high_order_3d_benchmark_qualification:
        return
    if args.high_order_mpi_production_qualification:
        raise ValueError(
            "--high-order-3d-benchmark-qualification cannot be combined with "
            "--high-order-mpi-production-qualification"
        )
    if args.high_order_mpi_motion_smoke:
        raise ValueError(
            "--high-order-3d-benchmark-qualification cannot be combined with "
            "--high-order-mpi-motion-smoke"
        )
    if args.high_order_curved_3d_simplex_smoke:
        raise ValueError(
            "--high-order-3d-benchmark-qualification cannot be combined with "
            "--high-order-curved-3d-simplex-smoke"
        )
    if args.high_order_3d_benchmark_profile_qualification:
        raise ValueError(
            "--high-order-3d-benchmark-qualification cannot be combined with "
            "--high-order-3d-benchmark-profile-qualification"
        )

    if not args.case:
        args.case = list(HIGH_ORDER_3D_BENCHMARK_QUALIFICATION_CASES)
    if args.steps is None:
        args.steps = 3
    set_default(args, "timeout_seconds", 1200.0)
    set_default(args, "max_solver_elapsed_seconds_per_accepted_step", 6.0)
    args.use_high_order_implicit_cuts = True
    args.disable_vtk_output = True
    args.require_time_loop_convergence = True
    args.require_process_memory_diagnostics = True
    args.require_basis_cache_diagnostics = True
    args.require_high_order_cut_context_diagnostics = True
    args.require_cut_context_solution_source_diagnostics = True
    args.enable_fsils_matrix_diagnostics = True
    args.require_fsils_matrix_diagnostics = True

    set_default(args, "implicit_cut_quadrature_backend", "Auto")
    set_default(args, "expect_selected_implicit_cut_quadrature_backend",
                "HighOrderSubcell")
    set_default(args, "linear_algebra_backend", "fsils")
    set_default(args, "linear_preconditioner", "fsils")
    set_default(args, "linear_solver_type", "ns")
    set_default(args, "ns_gm_max_iterations", 200)
    set_default(args, "ns_cg_max_iterations", 200)
    set_default(args, "ns_gm_tolerance", 1.0e-4)
    set_default(args, "ns_cg_tolerance", 1.0e-4)
    set_default(args, "max_fsils_matrix_missing_diag", 0)
    set_default(args, "max_fsils_matrix_duplicate_diag_entries", 0)
    set_default(args, "max_fsils_matrix_duplicate_diag_rows", 0)
    set_default(args, "max_fsils_matrix_nonfinite_entries", 0)
    set_default(args, "max_diagnostic_implicit_cut_fallback_cells", 0)
    set_default(args, "min_diagnostic_achieved_interface_quadrature_order", 1)
    set_default(args, "min_diagnostic_achieved_volume_quadrature_order", 2)
    set_default(args, "max_diagnostic_assembly_timings_per_step", 4.0)
    set_default(args, "max_diagnostic_extra_assembly_timings_per_step", 3.0)
    set_default(args, "max_diagnostic_cut_context_rebuilds_per_step", 4.0)
    set_default(args, "max_diagnostic_generated_cell_cache_full_miss_rebuilds", 1)
    set_default(args, "max_diagnostic_process_rss_kb", 800000.0)
    set_default(args, "max_diagnostic_process_rss_growth_kb", 350000.0)
    set_default(args, "max_diagnostic_process_basis_cache_entries", 32)
    set_default(args, "max_diagnostic_process_basis_cache_entry_growth", 32)
    set_default(args, "max_time_loop_nonlinear_iterations_per_step", 3)
    set_default(args, "max_time_loop_linear_iterations_per_step", 250)


def apply_high_order_3d_benchmark_profile_qualification_defaults(
        args: argparse.Namespace) -> None:
    if not args.high_order_3d_benchmark_profile_qualification:
        return
    if args.high_order_mpi_motion_smoke:
        raise ValueError(
            "--high-order-3d-benchmark-profile-qualification cannot be combined with "
            "--high-order-mpi-motion-smoke"
        )
    if args.high_order_curved_3d_simplex_smoke:
        raise ValueError(
            "--high-order-3d-benchmark-profile-qualification cannot be combined with "
            "--high-order-curved-3d-simplex-smoke"
        )

    if not args.case:
        args.case = list(HIGH_ORDER_3D_BENCHMARK_PROFILE_CASES)
    if args.steps is None:
        # The historical D18 false-wall-wetting event occurs near t=0.235 s;
        # 562 accepted steps at the benchmark dt reach the t=0.281 s profile.
        args.steps = 562
    set_default(args, "timeout_seconds", 7200.0)
    set_default(args, "max_solver_elapsed_seconds_per_accepted_step", 6.0)
    args.use_high_order_implicit_cuts = True
    # D18/D38 deliberately use the unscaled cut-stabilization policy.  Prior
    # production probes showed that inverse wet-volume metadata amplification
    # worsens the small-cut nonlinear/pressure behavior, and both checked-in
    # benchmark decks declare Use_cut_metadata_scale=false.  Keep the profile
    # configurator consistent with that audited deck contract instead of
    # failing preflight by applying the generic "scale enabled" expectation.
    args.disable_cut_metadata_scale = True
    args.disable_vtk_output = False
    # False-wall-wet qualification is a transient-history gate.  Retain every
    # accepted-step VTK state so an event before the final profile cannot be
    # hidden by final-output-only sampling.
    args.final_output_only = False
    args.require_time_loop_convergence = True
    args.require_process_memory_diagnostics = True
    args.require_basis_cache_diagnostics = True
    args.require_high_order_cut_context_diagnostics = True
    args.require_cut_context_solution_source_diagnostics = True
    args.enable_fsils_matrix_diagnostics = True
    args.require_fsils_matrix_diagnostics = True
    set_default(args, "fsils_matrix_diagnostics_every_n", 25)
    set_default(args, "fsils_matrix_diagnostics_max_records", 64)
    args.require_reference_profile_comparison = True
    args.enable_physical_history_instrumentation = True
    args.require_level_set_mass_correction_histories = True
    args.enable_adaptive_time_loop = True
    args.newton_line_search_fail_on_no_reduction = True

    set_default(args, "implicit_cut_quadrature_backend", "Auto")
    set_default(args, "expect_selected_implicit_cut_quadrature_backend",
                "HighOrderSubcell")
    set_default(args, "linear_algebra_backend", "fsils")
    set_default(args, "linear_preconditioner", "fsils")
    set_default(args, "linear_solver_type", "ns")
    # The checked-in D18/D38 fluid decks carry a permissive 1e-4 outer
    # relative/absolute solve tolerance.  That is suitable for the historical
    # benchmark run, but it cannot resolve the coupled level-set residual
    # floor used by this fail-closed profile qualification.  Set the outer
    # FSILS targets independently of the NS inner block controls below; an
    # explicit command-line value still takes precedence through set_default.
    set_default(args, "linear_relative_tolerance", 1.0e-10)
    set_default(args, "linear_absolute_tolerance", 1.0e-12)
    set_default(args, "ns_gm_max_iterations", 200)
    set_default(args, "ns_cg_max_iterations", 200)
    set_default(args, "ns_gm_tolerance", 1.0e-4)
    set_default(args, "ns_cg_tolerance", 1.0e-4)
    # Preserve the audited D18/D38 adaptive range.  The bound-preserving gate
    # has demonstrated valid recovery below 6.25e-5 without any sign or wall-
    # normal violation, so a higher preset floor can reject an otherwise valid
    # retry before the deck's own minimum is reached.
    set_default(args, "adaptive_time_loop_min_dt", 1.5625e-5)
    set_default(args, "adaptive_time_loop_max_dt", 5.0e-4)
    set_default(args, "adaptive_time_loop_max_retries", 8)
    set_default(args, "adaptive_time_loop_decrease_factor", 0.5)
    set_default(args, "adaptive_time_loop_increase_factor", 1.5)
    set_default(args, "adaptive_time_loop_target_newton_iterations", 6)
    set_default(args, "adaptive_time_loop_max_steps_multiplier", 64)
    set_default(args, "newton_line_search_max_iterations", 6)
    set_default(args, "max_fsils_matrix_missing_diag", 0)
    set_default(args, "max_fsils_matrix_duplicate_diag_entries", 0)
    set_default(args, "max_fsils_matrix_duplicate_diag_rows", 0)
    set_default(args, "max_fsils_matrix_nonfinite_entries", 0)
    set_default(args, "max_diagnostic_implicit_cut_fallback_cells", 0)
    set_default(args, "min_diagnostic_achieved_interface_quadrature_order", 1)
    set_default(args, "min_diagnostic_achieved_volume_quadrature_order", 2)
    set_default(args, "max_diagnostic_assembly_timings_per_step", 6.0)
    set_default(args, "max_diagnostic_extra_assembly_timings_per_step", 4.0)
    set_default(args, "max_diagnostic_cut_context_rebuilds_per_step", 6.0)
    set_default(args, "max_diagnostic_generated_cell_cache_full_miss_rebuilds", 1)
    set_default(args, "max_diagnostic_process_rss_kb", 1000000.0)
    set_default(args, "max_diagnostic_process_rss_growth_kb", 650000.0)
    set_default(args, "max_diagnostic_process_basis_cache_entries", 32)
    set_default(args, "max_diagnostic_process_basis_cache_entry_growth", 32)
    set_default(args, "max_time_loop_nonlinear_iterations_per_step", 9)
    set_default(args, "max_time_loop_linear_iterations_per_step", 250)
    set_default(args, "min_reference_profile_coverage", 0.95)
    set_default(args, "min_reference_profile_direct_coverage", 0.25)
    set_default(args, "max_reference_profile_rmse", 0.12)
    set_default(args, "max_reference_profile_mae", 0.10)
    set_default(args, "max_reference_profile_max_abs_error", 0.18)
    # D38 has a long shallow reference tail a few mm above the wet-bed depth;
    # use a material-height front threshold that tracks the moving wave.
    set_default(args, "reference_profile_elevated_front_clearance", 0.010)
    set_default(args, "max_reference_profile_elevated_front_lag", 0.30)


def apply_high_order_curved_3d_simplex_smoke_defaults(
        args: argparse.Namespace) -> None:
    if not args.high_order_curved_3d_simplex_smoke:
        return
    if args.high_order_mpi_motion_smoke:
        raise ValueError(
            "--high-order-curved-3d-simplex-smoke cannot be combined with "
            "--high-order-mpi-motion-smoke"
        )

    if not args.case:
        args.case = list(HIGH_ORDER_CURVED_3D_SIMPLEX_CASES)
    if args.steps is None:
        args.steps = 1
    set_default(args, "timeout_seconds", 600.0)
    args.use_high_order_implicit_cuts = True
    args.disable_cut_stabilization = False
    args.require_time_loop_convergence = True
    args.require_process_memory_diagnostics = True
    args.require_basis_cache_diagnostics = True
    args.require_high_order_cut_context_diagnostics = True
    args.require_cut_context_solution_source_diagnostics = True
    args.require_eigen_factorization_diagnostics = True
    args.require_active_pressure_support_diagnostics = True

    set_default(args, "implicit_cut_quadrature_backend", "Auto")
    set_default(args, "expect_selected_implicit_cut_quadrature_backend",
                "HighOrderSubcell")
    # The current production contract for curved 3D simplex support is
    # conservative positive-weight cut-volume quadrature with verified volume
    # order 2 and interface order 1.  Do not request interface order 2 here
    # until the Tetra10 path has root-polished curved leaf rules end-to-end.
    set_default(args, "generated_interface_quadrature_order", 1)
    set_default(args, "interface_quadrature_order", 1)
    set_default(args, "volume_quadrature_order", 2)
    set_default(args, "implicit_cut_max_subdivision_depth", 2)
    set_default(args, "time_step_size", 2.0e-4)
    set_default(args, "linear_algebra_backend", "eigen")
    set_default(args, "linear_solver_type", "direct")
    # The curved Tetra10 hydrostatic smoke keeps velocity cut stabilization and
    # active pressure support enabled, but disables the pressure-gradient ghost
    # penalty. That penalty is not pressure-gradient robust for this curved
    # hydrostatic state and otherwise dominates the refreshed-quadrature Newton
    # solve.
    set_default(args, "cut_cell_pressure_gradient_penalty", 0.0)
    if args.min_max_speed == 1.0e-2:
        args.min_max_speed = 1.0e-4
    if args.min_wet_mean_speed == 2.5e-4:
        args.min_wet_mean_speed = 1.0e-6
    if args.min_gate_mean_ux == 1.0e-4:
        args.min_gate_mean_ux = -1.0
    if args.min_front_mean_ux == 1.0e-4:
        args.min_front_mean_ux = -1.0
    set_default(args, "min_diagnostic_solution_velocity_range", 1.0e-4)
    set_default(args, "min_diagnostic_pressure_range", 10.0)
    set_default(args, "max_wet_fraction_volume_error", 1.0e-8)
    set_default(args, "max_diagnostic_implicit_cut_fallback_cells", 0)
    set_default(args, "min_diagnostic_achieved_interface_quadrature_order", 1)
    set_default(args, "min_diagnostic_achieved_volume_quadrature_order", 2)
    set_default(args, "max_diagnostic_cut_context_rebuilds_per_step", 15.0)
    set_default(args, "max_diagnostic_assembly_timings_per_step", 20.0)
    set_default(args, "max_diagnostic_extra_assembly_timings_per_step", 15.0)
    set_default(args, "max_diagnostic_process_rss_kb", 350000.0)
    set_default(args, "max_diagnostic_process_rss_growth_kb", 150000.0)
    set_default(args, "max_diagnostic_process_basis_cache_entries", 24)
    set_default(args, "max_diagnostic_process_basis_cache_entry_growth", 24)
    set_default(args, "max_eigen_factorization_pressure_zero_rows", 0)
    set_default(args, "max_eigen_factorization_pressure_zero_cols", 0)
    set_default(args, "max_eigen_factorization_nonfinite_entries", 0)
    set_default(args, "max_time_loop_nonlinear_iterations_per_step", 6)
    set_default(args, "max_time_loop_linear_iterations_per_step", 10)
    set_default(args, "min_interface_height_change", 1.0e-4)
    set_default(args, "min_interface_mean_abs_height_change", 1.0e-5)
    set_default(args, "min_interface_slope_change", 1.0e-4)
    set_default(args, "min_interface_final_height_span", 1.0e-3)


def apply_high_order_mpi_motion_smoke_defaults(
        args: argparse.Namespace) -> None:
    if not args.high_order_mpi_motion_smoke:
        return
    if args.high_order_mpi_production_qualification:
        raise ValueError(
            "--high-order-mpi-motion-smoke cannot be combined with "
            "--high-order-mpi-production-qualification"
        )
    if args.high_order_curved_3d_simplex_smoke:
        raise ValueError(
            "--high-order-mpi-motion-smoke cannot be combined with "
            "--high-order-curved-3d-simplex-smoke"
        )

    if not args.case:
        args.case = list(HIGH_ORDER_MPI_MOTION_CASES)
    if args.steps is None:
        args.steps = 5
    set_default(args, "timeout_seconds", 600.0)
    set_default(args, "mpi_ranks", 2)
    args.use_high_order_implicit_cuts = True
    args.disable_cut_stabilization = False
    args.require_time_loop_convergence = True
    args.require_process_memory_diagnostics = True
    args.require_basis_cache_diagnostics = True
    args.require_high_order_cut_context_diagnostics = True
    args.require_cut_context_solution_source_diagnostics = True
    args.enable_fsils_matrix_diagnostics = True
    args.require_fsils_matrix_diagnostics = True

    set_default(args, "linear_algebra_backend", "fsils")
    set_default(args, "linear_preconditioner", "fsils")
    set_default(args, "linear_solver_type", "gmres")
    set_default(args, "linear_relative_tolerance", 1.0e-4)
    set_default(args, "linear_absolute_tolerance", 1.0e-4)
    set_default(args, "linear_max_iterations", 7)
    if args.min_max_speed == 1.0e-2:
        args.min_max_speed = 1.0e-3
    if args.min_wet_mean_speed == 2.5e-4:
        args.min_wet_mean_speed = 1.0e-4
    set_default(args, "max_fsils_matrix_missing_diag", 0)
    set_default(args, "max_fsils_matrix_duplicate_diag_entries", 0)
    set_default(args, "max_fsils_matrix_duplicate_diag_rows", 0)
    set_default(args, "max_fsils_matrix_nonfinite_entries", 0)
    set_default(args, "max_diagnostic_implicit_cut_fallback_cells", 0)
    set_default(args, "min_diagnostic_achieved_interface_quadrature_order", 2)
    set_default(args, "min_diagnostic_achieved_volume_quadrature_order", 2)
    set_default(args, "expect_selected_implicit_cut_quadrature_backend",
                "SayeHyperrectangle")
    set_default(args, "max_diagnostic_cut_context_rebuilds_per_step", 5.0)
    set_default(args, "min_diagnostic_cut_context_refresh_skips", 1)
    set_default(args, "max_diagnostic_process_rss_kb", 300000.0)
    set_default(args, "max_diagnostic_process_rss_growth_kb", 150000.0)
    set_default(args, "max_diagnostic_process_basis_cache_entries", 8)
    set_default(args, "max_diagnostic_process_basis_cache_entry_growth", 8)
    set_default(args, "max_time_loop_nonlinear_iterations_per_step", 3)
    set_default(args, "max_time_loop_linear_iterations_per_step", 500)
    set_default(args, "min_interface_height_change", 1.0e-4)
    set_default(args, "min_interface_mean_abs_height_change", 2.0e-5)
    set_default(args, "min_interface_slope_change", 5.0e-5)
    set_default(args, "min_interface_final_height_span", 1.0e-4)


def apply_high_order_capillary_projection_smoke_defaults(
        args: argparse.Namespace) -> None:
    if not args.high_order_capillary_projection_smoke:
        return
    if args.high_order_3d_benchmark_qualification:
        raise ValueError(
            "--high-order-capillary-projection-smoke cannot be combined with "
            "--high-order-3d-benchmark-qualification"
        )
    if args.high_order_3d_benchmark_profile_qualification:
        raise ValueError(
            "--high-order-capillary-projection-smoke cannot be combined with "
            "--high-order-3d-benchmark-profile-qualification"
        )
    if args.high_order_curved_3d_simplex_smoke:
        raise ValueError(
            "--high-order-capillary-projection-smoke cannot be combined with "
            "--high-order-curved-3d-simplex-smoke"
        )
    if args.high_order_mpi_motion_smoke:
        raise ValueError(
            "--high-order-capillary-projection-smoke cannot be combined with "
            "--high-order-mpi-motion-smoke"
        )
    if getattr(args, "high_order_capillary_droplet_equilibrium_smoke", False):
        raise ValueError(
            "--high-order-capillary-projection-smoke cannot be combined with "
            "--high-order-capillary-droplet-equilibrium-smoke"
        )

    if not args.case:
        args.case = list(HIGH_ORDER_CAPILLARY_PROJECTION_CASES)
    if args.steps is None:
        args.steps = 10
    set_default(args, "timeout_seconds", 600.0)
    args.use_high_order_implicit_cuts = True
    args.disable_cut_stabilization = False
    args.require_time_loop_convergence = True
    args.require_process_memory_diagnostics = True
    args.require_basis_cache_diagnostics = True
    args.require_high_order_cut_context_diagnostics = True
    args.require_eigen_factorization_diagnostics = True
    args.require_active_pressure_support_diagnostics = True
    args.require_curvature_projection_diagnostics = True
    args.require_curvature_projection_newton_freshness = True

    set_default(args, "surface_tension", 1.0e-3)
    set_default(args, "projected_curvature_field", "kappa_projected")
    set_default(args, "curvature_projection_cadence_steps", 1)
    set_default(args, "curvature_projection_max_normalized_fit_residual", 5.0e-2)
    set_default(args, "curvature_projection_max_zero_fallback_vertices", 0)
    set_default(args, "curvature_projection_smoothing_iterations", 1)
    set_default(args, "curvature_projection_smoothing_relaxation", 0.25)
    set_default(args, "curvature_projection_smoothing_mode", "mass_stiffness_operator")
    set_default(args, "expect_curvature_projection_smoothing_mode", "mass_stiffness_operator")
    set_default(args, "min_diagnostic_curvature_projection_operator_edges", 1)
    set_default(args, "max_capillary_rejected_steps", 0)
    set_default(args, "max_capillary_dt_updates", 0)
    set_default(args, "max_capillary_speed_per_surface_tension", 10.0)
    args.required_implicit_cut_backend_qualification = "ProductionQualified"
    set_default(args, "linear_algebra_backend", "eigen")
    if args.min_max_speed == 1.0e-2:
        args.min_max_speed = 1.0e-3
    if args.min_wet_mean_speed == 2.5e-4:
        args.min_wet_mean_speed = 1.0e-4
    if args.min_gate_mean_ux == 1.0e-4:
        args.min_gate_mean_ux = -1.0
    if args.min_front_mean_ux == 1.0e-4:
        args.min_front_mean_ux = -1.0
    set_default(args, "min_diagnostic_solution_velocity_range", 1.0e-3)
    set_default(args, "min_diagnostic_pressure_range", 100.0)
    set_default(args, "max_wet_fraction_volume_error", 1.0e-8)
    set_default(args, "max_diagnostic_cut_context_rebuilds_per_step", 4.0)
    set_default(args, "min_diagnostic_cut_context_refresh_skips", 1)
    set_default(args, "max_diagnostic_assembly_timings_per_step", 4.0)
    set_default(args, "max_diagnostic_extra_assembly_timings_per_step", 3.0)
    set_default(args, "max_diagnostic_process_basis_cache_entries", 8)
    set_default(args, "max_diagnostic_process_basis_cache_entry_growth", 8)
    set_default(args, "max_diagnostic_implicit_cut_fallback_cells", 0)
    set_default(args, "min_diagnostic_achieved_interface_quadrature_order", 2)
    set_default(args, "min_diagnostic_achieved_volume_quadrature_order", 2)
    set_default(args, "max_eigen_factorization_pressure_zero_rows", 0)
    set_default(args, "max_eigen_factorization_pressure_zero_cols", 0)
    set_default(args, "max_eigen_factorization_nonfinite_entries", 0)
    set_default(args, "max_time_loop_nonlinear_iterations_per_step", 3)
    set_default(args, "max_time_loop_linear_iterations_per_step", 10)
    set_default(args, "min_interface_height_change", 1.0e-4)
    set_default(args, "min_interface_mean_abs_height_change", 2.0e-5)
    set_default(args, "min_interface_slope_change", 5.0e-5)
    set_default(args, "min_interface_final_height_span", 1.0e-4)
    set_default(args, "min_diagnostic_curvature_projection_count", 1)
    set_default(args, "min_diagnostic_curvature_projection_max_abs_curvature", 1.0e-6)
    set_default(args, "max_diagnostic_curvature_projection_zero_fallback_vertices", 0)
    set_default(args, "max_diagnostic_curvature_projection_normalized_fit_residual", 5.0e-2)
    set_default(args, "min_diagnostic_curvature_projection_smoothing_iterations", 1)
    set_default(args, "min_diagnostic_curvature_projection_skipped_count", 1)
    set_default(args, "min_diagnostic_curvature_projection_cache_hit_count", 1)
    set_default(args, "max_diagnostic_curvature_projection_cache_miss_count", 35)
    set_default(args,
                "min_diagnostic_curvature_projection_cut_signature_cache_hit_count",
                1)
    set_default(args, "min_diagnostic_curvature_projection_reused_vertex_adjacency_count", 1)
    set_default(args, "min_diagnostic_curvature_projection_reused_sample_adjacency_count", 1)
    set_default(args, "max_diagnostic_curvature_projection_vertex_adjacency_builds", 1)
    set_default(args, "max_diagnostic_curvature_projection_sample_adjacency_builds",
                curvature_sample_adjacency_build_budget(args))


def apply_high_order_capillary_response_smoke_defaults(
        args: argparse.Namespace) -> None:
    if not args.high_order_capillary_response_smoke:
        return
    if args.high_order_3d_benchmark_qualification:
        raise ValueError(
            "--high-order-capillary-response-smoke cannot be combined with "
            "--high-order-3d-benchmark-qualification"
        )
    if args.high_order_3d_benchmark_profile_qualification:
        raise ValueError(
            "--high-order-capillary-response-smoke cannot be combined with "
            "--high-order-3d-benchmark-profile-qualification"
        )
    if args.high_order_curved_3d_simplex_smoke:
        raise ValueError(
            "--high-order-capillary-response-smoke cannot be combined with "
            "--high-order-curved-3d-simplex-smoke"
        )
    if args.high_order_mpi_motion_smoke:
        raise ValueError(
            "--high-order-capillary-response-smoke cannot be combined with "
            "--high-order-mpi-motion-smoke"
        )
    if args.high_order_capillary_projection_smoke:
        raise ValueError(
            "--high-order-capillary-response-smoke cannot be combined with "
            "--high-order-capillary-projection-smoke"
        )
    if getattr(args, "high_order_capillary_droplet_equilibrium_smoke", False):
        raise ValueError(
            "--high-order-capillary-response-smoke cannot be combined with "
            "--high-order-capillary-droplet-equilibrium-smoke"
        )

    if not args.case:
        args.case = list(HIGH_ORDER_CAPILLARY_RESPONSE_CASES)
    if args.steps is None:
        args.steps = 3
    set_default(args, "timeout_seconds", 300.0)
    set_default(args, "max_solver_elapsed_seconds_per_accepted_step", 1.0)
    args.use_high_order_implicit_cuts = True
    args.disable_cut_stabilization = False
    args.require_time_loop_convergence = True
    args.require_process_memory_diagnostics = True
    args.require_basis_cache_diagnostics = True
    args.require_high_order_cut_context_diagnostics = True
    args.require_eigen_factorization_diagnostics = True
    args.require_active_pressure_support_diagnostics = True
    args.require_curvature_projection_diagnostics = True
    args.require_curvature_projection_newton_freshness = True

    set_default(args, "surface_tension", 0.5)
    set_default(args, "projected_curvature_field", "kappa_projected")
    set_default(args, "curvature_projection_cadence_steps", 1)
    set_default(
        args,
        "curvature_projection_narrow_band_width",
        synthetic_curvature_projection_band_width(args),
    )
    set_default(args, "curvature_projection_max_normalized_fit_residual", 5.0e-2)
    set_default(args, "curvature_projection_max_zero_fallback_vertices", 0)
    set_default(args, "curvature_projection_smoothing_iterations", 1)
    set_default(args, "curvature_projection_smoothing_relaxation", 0.25)
    set_default(args, "curvature_projection_smoothing_mode", "mass_stiffness_operator")
    set_default(args, "expect_curvature_projection_smoothing_mode", "mass_stiffness_operator")
    set_default(args, "min_diagnostic_curvature_projection_operator_edges", 1)
    set_default(args, "max_capillary_curvature_relative_error", 0.75)
    set_default(args, "max_capillary_rejected_steps", 0)
    set_default(args, "max_capillary_dt_updates", 0)
    set_default(args, "max_capillary_speed_per_surface_tension", 10.0)
    args.required_implicit_cut_backend_qualification = "ProductionQualified"
    set_default(args, "linear_algebra_backend", "eigen")
    if args.min_max_speed == 1.0e-2:
        args.min_max_speed = 1.0e-6
    if args.min_wet_mean_speed == 2.5e-4:
        args.min_wet_mean_speed = 1.0e-7
    if args.min_gate_mean_ux == 1.0e-4:
        args.min_gate_mean_ux = -1.0
    if args.min_front_mean_ux == 1.0e-4:
        args.min_front_mean_ux = -1.0
    set_default(args, "min_diagnostic_solution_velocity_range", 1.0e-6)
    set_default(args, "min_capillary_response_speed_per_surface_tension", 1.0e-6)
    set_default(args, "max_wet_fraction_volume_error", 1.0e-8)
    set_default(args, "max_diagnostic_cut_context_rebuilds_per_step", 4.0)
    set_default(args, "min_diagnostic_cut_context_refresh_skips", 1)
    set_default(args, "max_diagnostic_generated_cell_cache_full_miss_rebuilds", 1)
    set_default(args, "max_diagnostic_assembly_timings_per_step", 4.0)
    set_default(args, "max_diagnostic_extra_assembly_timings_per_step", 3.0)
    set_default(args, "max_diagnostic_process_basis_cache_entries", 8)
    set_default(args, "max_diagnostic_process_basis_cache_entry_growth", 8)
    set_default(args, "max_diagnostic_implicit_cut_fallback_cells", 0)
    set_default(args, "min_diagnostic_achieved_interface_quadrature_order", 2)
    set_default(args, "min_diagnostic_achieved_volume_quadrature_order", 2)
    set_default(args, "max_eigen_factorization_pressure_zero_rows", 0)
    set_default(args, "max_eigen_factorization_pressure_zero_cols", 0)
    set_default(args, "max_eigen_factorization_nonfinite_entries", 0)
    set_default(args, "max_time_loop_nonlinear_iterations_per_step", 3)
    set_default(args, "max_time_loop_linear_iterations_per_step", 10)
    set_default(args, "min_diagnostic_curvature_projection_count", 1)
    set_default(args, "min_diagnostic_curvature_projection_max_abs_curvature", 1.0)
    set_default(args, "max_diagnostic_curvature_projection_zero_fallback_vertices", 0)
    set_default(args, "max_diagnostic_curvature_projection_normalized_fit_residual", 5.0e-2)
    set_default(args, "min_diagnostic_curvature_projection_smoothing_iterations", 1)
    set_default(args, "min_diagnostic_curvature_projection_skipped_count", 1)
    set_default(args, "min_diagnostic_curvature_projection_cache_hit_count", 1)
    set_default(args, "max_diagnostic_curvature_projection_cache_miss_count", 35)
    set_default(args,
                "min_diagnostic_curvature_projection_cut_signature_cache_hit_count",
                1)
    set_default(args, "min_diagnostic_curvature_projection_reused_vertex_adjacency_count", 1)
    set_default(args, "min_diagnostic_curvature_projection_reused_sample_adjacency_count", 1)
    set_default(args, "max_diagnostic_curvature_projection_vertex_adjacency_builds", 1)
    set_default(args, "max_diagnostic_curvature_projection_sample_adjacency_builds",
                curvature_sample_adjacency_build_budget(args))


def apply_high_order_capillary_balance_smoke_defaults(
        args: argparse.Namespace) -> None:
    if not args.high_order_capillary_balance_smoke:
        return
    if args.high_order_3d_benchmark_qualification:
        raise ValueError(
            "--high-order-capillary-balance-smoke cannot be combined with "
            "--high-order-3d-benchmark-qualification"
        )
    if args.high_order_3d_benchmark_profile_qualification:
        raise ValueError(
            "--high-order-capillary-balance-smoke cannot be combined with "
            "--high-order-3d-benchmark-profile-qualification"
        )
    if args.high_order_curved_3d_simplex_smoke:
        raise ValueError(
            "--high-order-capillary-balance-smoke cannot be combined with "
            "--high-order-curved-3d-simplex-smoke"
        )
    if args.high_order_mpi_motion_smoke:
        raise ValueError(
            "--high-order-capillary-balance-smoke cannot be combined with "
            "--high-order-mpi-motion-smoke"
        )
    if args.high_order_capillary_projection_smoke:
        raise ValueError(
            "--high-order-capillary-balance-smoke cannot be combined with "
            "--high-order-capillary-projection-smoke"
        )
    if args.high_order_capillary_response_smoke:
        raise ValueError(
            "--high-order-capillary-balance-smoke cannot be combined with "
            "--high-order-capillary-response-smoke"
        )
    if getattr(args, "high_order_capillary_droplet_equilibrium_smoke", False):
        raise ValueError(
            "--high-order-capillary-balance-smoke cannot be combined with "
            "--high-order-capillary-droplet-equilibrium-smoke"
        )

    if not args.case:
        args.case = list(HIGH_ORDER_CAPILLARY_BALANCE_CASES)
    if args.steps is None:
        args.steps = 3
    set_default(args, "timeout_seconds", 300.0)
    set_default(args, "max_solver_elapsed_seconds_per_accepted_step", 1.0)
    args.use_high_order_implicit_cuts = True
    args.disable_cut_stabilization = False
    args.require_time_loop_convergence = True
    args.require_process_memory_diagnostics = True
    args.require_basis_cache_diagnostics = True
    args.require_high_order_cut_context_diagnostics = True
    args.require_eigen_factorization_diagnostics = True
    args.require_active_pressure_support_diagnostics = True
    args.require_curvature_projection_diagnostics = True
    args.require_curvature_projection_newton_freshness = True

    set_default(args, "surface_tension", 0.5)
    set_default(args, "projected_curvature_field", "kappa_projected")
    set_default(args, "curvature_projection_cadence_steps", 1)
    set_default(
        args,
        "curvature_projection_narrow_band_width",
        synthetic_curvature_projection_band_width(args),
    )
    set_default(args, "curvature_projection_max_normalized_fit_residual", 5.0e-2)
    set_default(args, "curvature_projection_max_zero_fallback_vertices", 0)
    set_default(args, "curvature_projection_smoothing_iterations", 1)
    set_default(args, "curvature_projection_smoothing_relaxation", 0.25)
    set_default(args, "curvature_projection_smoothing_mode", "mass_stiffness_operator")
    set_default(args, "expect_curvature_projection_smoothing_mode", "mass_stiffness_operator")
    set_default(args, "min_diagnostic_curvature_projection_operator_edges", 1)
    set_default(args, "max_capillary_curvature_relative_error", 0.75)
    # Solved P1 pressure samples across an unfitted circle cannot satisfy a
    # machine-precision Young--Laplace jump.  This fixed 15% bound is applied
    # to the final liquid/gas medians, never to the preloaded initial field.
    set_default(args, "max_capillary_pressure_jump_relative_error", 0.15)
    set_default(args, "max_capillary_rejected_steps", 0)
    set_default(args, "max_capillary_dt_updates", 0)
    set_default(args, "max_capillary_speed_per_surface_tension", 1.0e-6)
    args.required_implicit_cut_backend_qualification = "ProductionQualified"
    set_default(args, "linear_algebra_backend", "eigen")
    if args.min_max_speed == 1.0e-2:
        args.min_max_speed = 0.0
    if args.min_wet_mean_speed == 2.5e-4:
        args.min_wet_mean_speed = 0.0
    if args.min_gate_mean_ux == 1.0e-4:
        args.min_gate_mean_ux = -1.0
    if args.min_front_mean_ux == 1.0e-4:
        args.min_front_mean_ux = -1.0
    set_default(args, "max_capillary_balance_speed_per_surface_tension", 1.0e-6)
    set_default(args, "min_diagnostic_pressure_range", 0.5)
    set_default(args, "max_wet_fraction_volume_error", 1.0e-8)
    set_default(args, "max_diagnostic_cut_context_rebuilds_per_step", 4.0)
    set_default(args, "min_diagnostic_cut_context_refresh_skips", 1)
    set_default(args, "max_diagnostic_generated_cell_cache_full_miss_rebuilds", 1)
    set_default(args, "max_diagnostic_assembly_timings_per_step", 4.0)
    set_default(args, "max_diagnostic_extra_assembly_timings_per_step", 3.0)
    set_default(args, "max_diagnostic_process_basis_cache_entries", 8)
    set_default(args, "max_diagnostic_process_basis_cache_entry_growth", 8)
    set_default(args, "max_diagnostic_implicit_cut_fallback_cells", 0)
    set_default(args, "min_diagnostic_achieved_interface_quadrature_order", 2)
    set_default(args, "min_diagnostic_achieved_volume_quadrature_order", 2)
    set_default(args, "max_eigen_factorization_pressure_zero_rows", 0)
    set_default(args, "max_eigen_factorization_pressure_zero_cols", 0)
    set_default(args, "max_eigen_factorization_nonfinite_entries", 0)
    set_default(args, "max_time_loop_nonlinear_iterations_per_step", 3)
    set_default(args, "max_time_loop_linear_iterations_per_step", 10)
    set_default(args, "min_diagnostic_curvature_projection_count", 1)
    set_default(args, "min_diagnostic_curvature_projection_max_abs_curvature", 1.0)
    set_default(args, "max_diagnostic_curvature_projection_zero_fallback_vertices", 0)
    set_default(args, "max_diagnostic_curvature_projection_normalized_fit_residual", 5.0e-2)
    set_default(args, "min_diagnostic_curvature_projection_smoothing_iterations", 1)
    set_default(args, "min_diagnostic_curvature_projection_skipped_count", 1)
    set_default(args, "min_diagnostic_curvature_projection_cache_hit_count", 1)
    set_default(args, "max_diagnostic_curvature_projection_cache_miss_count", 35)
    set_default(args,
                "min_diagnostic_curvature_projection_cut_signature_cache_hit_count",
                1)
    set_default(args, "min_diagnostic_curvature_projection_reused_vertex_adjacency_count", 1)
    set_default(args, "min_diagnostic_curvature_projection_reused_sample_adjacency_count", 1)
    set_default(args, "max_diagnostic_curvature_projection_vertex_adjacency_builds", 1)
    set_default(args, "max_diagnostic_curvature_projection_sample_adjacency_builds",
                curvature_sample_adjacency_build_budget(args))


def apply_high_order_capillary_droplet_equilibrium_smoke_defaults(
        args: argparse.Namespace) -> None:
    if not getattr(args, "high_order_capillary_droplet_equilibrium_smoke", False):
        return
    if getattr(args, "high_order_3d_benchmark_qualification", False):
        raise ValueError(
            "--high-order-capillary-droplet-equilibrium-smoke cannot be "
            "combined with --high-order-3d-benchmark-qualification"
        )
    if getattr(args, "high_order_3d_benchmark_profile_qualification", False):
        raise ValueError(
            "--high-order-capillary-droplet-equilibrium-smoke cannot be "
            "combined with --high-order-3d-benchmark-profile-qualification"
        )
    if getattr(args, "high_order_curved_3d_simplex_smoke", False):
        raise ValueError(
            "--high-order-capillary-droplet-equilibrium-smoke cannot be "
            "combined with --high-order-curved-3d-simplex-smoke"
        )
    if getattr(args, "high_order_mpi_motion_smoke", False):
        raise ValueError(
            "--high-order-capillary-droplet-equilibrium-smoke cannot be "
            "combined with --high-order-mpi-motion-smoke"
        )
    if getattr(args, "high_order_capillary_projection_smoke", False):
        raise ValueError(
            "--high-order-capillary-droplet-equilibrium-smoke cannot be "
            "combined with --high-order-capillary-projection-smoke"
        )
    if getattr(args, "high_order_capillary_response_smoke", False):
        raise ValueError(
            "--high-order-capillary-droplet-equilibrium-smoke cannot be "
            "combined with --high-order-capillary-response-smoke"
        )
    if getattr(args, "high_order_capillary_balance_smoke", False):
        raise ValueError(
            "--high-order-capillary-droplet-equilibrium-smoke cannot be "
            "combined with --high-order-capillary-balance-smoke"
        )

    if not args.case:
        args.case = list(HIGH_ORDER_CAPILLARY_DROPLET_EQUILIBRIUM_CASES)
    if args.steps is None:
        args.steps = 3
    set_default(args, "timeout_seconds", 300.0)
    # The timeout is a liveness guard.  Machine-dependent elapsed time,
    # diagnostic-assembly counts, and legacy single-Newton iteration ceilings
    # are not physical equilibrium criteria, especially now that generated
    # geometry is closed by an explicit outer fixed point.
    args.use_high_order_implicit_cuts = True
    args.disable_cut_stabilization = False
    args.require_time_loop_convergence = True
    args.require_process_memory_diagnostics = True
    args.require_basis_cache_diagnostics = True
    args.require_high_order_cut_context_diagnostics = True
    args.require_eigen_factorization_diagnostics = False
    args.enable_fsils_matrix_diagnostics = True
    args.require_fsils_matrix_diagnostics = True
    args.require_active_pressure_support_diagnostics = True
    # SurfaceStress consumes generated normals and measure directly.  A scalar
    # curvature projection is output-only when a caller explicitly requests
    # one; it is not a residual-freshness or force-accuracy requirement.
    args.require_curvature_projection_diagnostics = bool(
        getattr(args, "projected_curvature_field", None))
    args.require_curvature_projection_newton_freshness = False

    set_default(args, "surface_tension", 0.5)
    if getattr(args, "projected_curvature_field", None):
        set_default(args, "curvature_projection_cadence_steps", 1)
        set_default(
            args,
            "curvature_projection_narrow_band_width",
            synthetic_curvature_projection_band_width(args),
        )
        set_default(args, "curvature_projection_max_normalized_fit_residual", 5.0e-2)
        set_default(args, "curvature_projection_max_zero_fallback_vertices", 0)
        set_default(args, "curvature_projection_smoothing_iterations", 1)
        set_default(args, "curvature_projection_smoothing_relaxation", 0.25)
        set_default(args, "curvature_projection_smoothing_mode", "mass_stiffness_operator")
        set_default(args, "expect_curvature_projection_smoothing_mode",
                    "mass_stiffness_operator")
        set_default(args, "min_diagnostic_curvature_projection_operator_edges", 1)
    # Fixed a priori for the n=8/16/32 P1 refinement matrix and evaluated from
    # final solved liquid/gas pressure samples (not initial preload metadata).
    set_default(args, "max_capillary_pressure_jump_relative_error", 0.15)
    set_default(args, "max_capillary_rejected_steps", 0)
    set_default(args, "max_capillary_dt_updates", 0)
    set_default(args, "max_capillary_speed_per_surface_tension", 1.0e-5)
    args.required_implicit_cut_backend_qualification = "ProductionQualified"
    # This is a monolithic level-set/velocity/pressure system, so use generic
    # FSILS GMRES instead of interpreting [phi, velocity] as an NS momentum
    # block in the approximate BlockSchur path.
    set_default(args, "linear_solver_type", "gmres")
    set_default(args, "linear_algebra_backend", "fsils")
    set_default(args, "linear_preconditioner", "rcs")
    # Do not inherit the synthetic mini deck's two-step GMRES(1) budget or its
    # 1e-4 residual floor.  These controls match the other FS-16 monolithic
    # capillary qualifications and leave all physical equilibrium gates intact.
    set_default(args, "linear_max_iterations", 100)
    set_default(args, "linear_krylov_space_dimension", 50)
    set_default(args, "linear_relative_tolerance", 1.0e-8)
    set_default(args, "linear_absolute_tolerance", 1.0e-10)
    if getattr(args, "min_max_speed", None) == 1.0e-2:
        args.min_max_speed = 0.0
    if getattr(args, "min_wet_mean_speed", None) == 2.5e-4:
        args.min_wet_mean_speed = 0.0
    if getattr(args, "min_gate_mean_ux", None) == 1.0e-4:
        args.min_gate_mean_ux = -1.0
    if getattr(args, "min_front_mean_ux", None) == 1.0e-4:
        args.min_front_mean_ux = -1.0
    set_default(args, "max_capillary_balance_speed_per_surface_tension", 1.0e-5)
    set_default(args, "min_diagnostic_pressure_range", 0.5)
    set_default(args, "max_wet_fraction_volume_error", 1.0e-8)
    set_default(args, "max_diagnostic_implicit_cut_fallback_cells", 0)
    set_default(args, "min_diagnostic_achieved_interface_quadrature_order", 2)
    set_default(args, "min_diagnostic_achieved_volume_quadrature_order", 2)
    set_default(args, "max_fsils_matrix_zero_rows", 0)
    set_default(args, "max_fsils_matrix_missing_diag", 0)
    set_default(args, "max_fsils_matrix_diag_col_mismatch", 0)
    set_default(args, "max_fsils_matrix_duplicate_diag_entries", 0)
    set_default(args, "max_fsils_matrix_duplicate_diag_rows", 0)
    set_default(args, "max_fsils_matrix_zero_diag", 0)
    set_default(args, "max_fsils_matrix_nonfinite_entries", 0)
    if getattr(args, "projected_curvature_field", None):
        set_default(args, "min_diagnostic_curvature_projection_count", 1)
        set_default(args, "min_diagnostic_curvature_projection_max_abs_curvature", 1.0)
        set_default(args, "max_diagnostic_curvature_projection_zero_fallback_vertices", 0)
        set_default(args, "max_diagnostic_curvature_projection_normalized_fit_residual",
                    5.0e-2)
        set_default(args, "min_diagnostic_curvature_projection_smoothing_iterations", 1)
        set_default(args, "min_diagnostic_curvature_projection_skipped_count", 1)
        set_default(args, "min_diagnostic_curvature_projection_cache_hit_count", 1)
        set_default(args, "max_diagnostic_curvature_projection_cache_miss_count", 35)
        set_default(args,
                    "min_diagnostic_curvature_projection_cut_signature_cache_hit_count",
                    1)
        set_default(args,
                    "min_diagnostic_curvature_projection_reused_vertex_adjacency_count",
                    1)
        set_default(args,
                    "min_diagnostic_curvature_projection_reused_sample_adjacency_count",
                    1)
        set_default(args, "max_diagnostic_curvature_projection_vertex_adjacency_builds", 1)
        set_default(args, "max_diagnostic_curvature_projection_sample_adjacency_builds",
                    curvature_sample_adjacency_build_budget(args))


def apply_high_order_capillary_wave_smoke_defaults(args: argparse.Namespace) -> None:
    if not getattr(args, "high_order_capillary_wave_smoke", False):
        return
    if getattr(args, "high_order_3d_benchmark_qualification", False):
        raise ValueError(
            "--high-order-capillary-wave-smoke cannot be combined with "
            "--high-order-3d-benchmark-qualification"
        )
    if getattr(args, "high_order_3d_benchmark_profile_qualification", False):
        raise ValueError(
            "--high-order-capillary-wave-smoke cannot be combined with "
            "--high-order-3d-benchmark-profile-qualification"
        )
    if getattr(args, "high_order_curved_3d_simplex_smoke", False):
        raise ValueError(
            "--high-order-capillary-wave-smoke cannot be combined with "
            "--high-order-curved-3d-simplex-smoke"
        )
    if getattr(args, "high_order_mpi_motion_smoke", False):
        raise ValueError(
            "--high-order-capillary-wave-smoke cannot be combined with "
            "--high-order-mpi-motion-smoke"
        )
    if getattr(args, "high_order_capillary_projection_smoke", False):
        raise ValueError(
            "--high-order-capillary-wave-smoke cannot be combined with "
            "--high-order-capillary-projection-smoke"
        )
    if getattr(args, "high_order_capillary_response_smoke", False):
        raise ValueError(
            "--high-order-capillary-wave-smoke cannot be combined with "
            "--high-order-capillary-response-smoke"
        )
    if getattr(args, "high_order_capillary_balance_smoke", False):
        raise ValueError(
            "--high-order-capillary-wave-smoke cannot be combined with "
            "--high-order-capillary-balance-smoke"
        )
    if getattr(args, "high_order_capillary_droplet_equilibrium_smoke", False):
        raise ValueError(
            "--high-order-capillary-wave-smoke cannot be combined with "
            "--high-order-capillary-droplet-equilibrium-smoke"
        )

    if not args.case:
        args.case = list(HIGH_ORDER_CAPILLARY_WAVE_CASES)
    set_default(args, "time_step_size", 2.0e-3)
    set_default(args, "surface_tension", 50.0)
    if args.steps is None:
        args.steps = capillary_wave_minimum_steps_for_frequency_fit(
            args.surface_tension, args.time_step_size)
    set_default(args, "timeout_seconds", 600.0)
    # Keep a liveness timeout, but do not mix machine performance or legacy
    # single-Newton iteration budgets into the wave-accuracy verdict.  Outer
    # and total-inner counts are reported separately by summarize_time_loop.
    args.use_high_order_implicit_cuts = True
    args.disable_cut_stabilization = False
    args.require_time_loop_convergence = True
    args.require_process_memory_diagnostics = True
    args.require_basis_cache_diagnostics = True
    args.require_high_order_cut_context_diagnostics = True
    args.require_eigen_factorization_diagnostics = False
    args.enable_fsils_matrix_diagnostics = True
    args.require_fsils_matrix_diagnostics = True
    args.require_active_pressure_support_diagnostics = True
    # SurfaceStress consumes generated normals and measure directly.  A scalar
    # curvature projection is output-only when a caller explicitly requests
    # one; it is not a residual-freshness or force-accuracy requirement.
    args.require_curvature_projection_diagnostics = bool(
        getattr(args, "projected_curvature_field", None))
    args.require_curvature_projection_newton_freshness = False
    args.require_reference_profile_comparison = True
    args.enable_physical_history_instrumentation = True
    args.require_free_surface_energy_history = True
    args.require_compiled_cut_volume_jit = True
    args.enable_jit_specialization_trace = True
    args.enable_jit_cache_diagnostics = True
    set_default(
        args,
        "max_free_surface_energy_positive_step_increment_relative",
        FREE_SURFACE_ENERGY_MAX_POSITIVE_STEP_INCREMENT_RELATIVE,
    )
    set_default(
        args,
        "max_free_surface_energy_above_initial_relative",
        FREE_SURFACE_ENERGY_MAX_ABOVE_INITIAL_RELATIVE,
    )

    set_default(args, "wet_extension_advection_velocity_method",
                "wall_compatible_normal")
    if getattr(args, "projected_curvature_field", None):
        set_default(args, "curvature_projection_cadence_steps", 1)
        set_default(args, "curvature_projection_max_normalized_fit_residual", 5.0e-2)
        set_default(args, "curvature_projection_max_zero_fallback_vertices", 0)
        set_default(args, "curvature_projection_smoothing_iterations", 1)
        set_default(args, "curvature_projection_smoothing_relaxation", 0.25)
        set_default(args, "curvature_projection_smoothing_mode", "mass_stiffness_operator")
        set_default(args, "expect_curvature_projection_smoothing_mode",
                    "mass_stiffness_operator")
        set_default(args, "min_diagnostic_curvature_projection_operator_edges", 1)
    set_default(args, "max_capillary_rejected_steps", 0)
    set_default(args, "max_capillary_dt_updates", 0)
    set_default(args, "max_capillary_speed_per_surface_tension", 10.0)
    # P1 interface geometry observed for the declared minimum phase span
    # cannot support a machine-precision physical-frequency gate.  Ten percent
    # frequency and twenty-five percent normalized profile error are fixed a
    # priori for the 8/16/32 refinement benchmark; convergence rates remain
    # separately required before production qualification.
    set_default(args, "max_capillary_wave_frequency_relative_error", 0.10)
    set_default(args, "max_capillary_wave_profile_relative_error", 0.25)
    set_default(args, "max_capillary_wave_mean_offset", 4.0e-3)
    set_default(
        args,
        "max_capillary_wave_temporal_liquid_volume_relative_drift",
        CAPILLARY_WAVE_MAX_TEMPORAL_LIQUID_VOLUME_RELATIVE_DRIFT,
    )
    args.required_implicit_cut_backend_qualification = "ProductionQualified"
    # Keep the physical wave solve monolithic: phi is an independent transport
    # unknown, not another velocity component for the NS BlockSchur split.
    set_default(args, "linear_solver_type", "gmres")
    set_default(args, "linear_algebra_backend", "fsils")
    set_default(args, "linear_preconditioner", "rcs")
    # The inherited mini deck permits only one loose FSILS restart.  Retain the
    # same fixed, production Krylov budget as the sessile qualification while
    # keeping the independent nonlinear and physical acceptance gates.
    set_default(args, "linear_max_iterations", 100)
    set_default(args, "linear_krylov_space_dimension", 50)
    set_default(args, "linear_relative_tolerance", 1.0e-8)
    set_default(args, "linear_absolute_tolerance", 1.0e-10)
    set_default(args, "max_fsils_accepted_true_residual_norm", 1.0e-9)
    if getattr(args, "min_max_speed", None) == 1.0e-2:
        args.min_max_speed = 0.0
    if getattr(args, "min_wet_mean_speed", None) == 2.5e-4:
        args.min_wet_mean_speed = 0.0
    if getattr(args, "min_gate_mean_ux", None) == 1.0e-4:
        args.min_gate_mean_ux = -1.0
    if getattr(args, "min_front_mean_ux", None) == 1.0e-4:
        args.min_front_mean_ux = -1.0
    set_default(args, "reference_profile_sample_radius", 0.02)
    set_default(args, "min_reference_profile_coverage", 0.75)
    set_default(args, "min_reference_profile_direct_coverage", 0.20)
    set_default(args, "max_reference_profile_rmse", 0.01)
    set_default(args, "max_reference_profile_mae", 0.01)
    set_default(args, "max_reference_profile_max_abs_error", 0.02)
    set_default(args, "min_interface_height_change", 1.0e-7)
    set_default(args, "min_interface_mean_abs_height_change", 1.0e-7)
    set_default(args, "max_wet_fraction_volume_error", 1.0e-8)
    set_default(args, "max_diagnostic_implicit_cut_fallback_cells", 0)
    set_default(args, "min_diagnostic_achieved_interface_quadrature_order", 2)
    set_default(args, "min_diagnostic_achieved_volume_quadrature_order", 2)
    set_default(args, "max_fsils_matrix_zero_rows", 0)
    set_default(args, "max_fsils_matrix_missing_diag", 0)
    set_default(args, "max_fsils_matrix_diag_col_mismatch", 0)
    set_default(args, "max_fsils_matrix_duplicate_diag_entries", 0)
    set_default(args, "max_fsils_matrix_duplicate_diag_rows", 0)
    set_default(args, "max_fsils_matrix_zero_diag", 0)
    set_default(args, "max_fsils_matrix_nonfinite_entries", 0)
    if getattr(args, "projected_curvature_field", None):
        set_default(args, "min_diagnostic_curvature_projection_count", 1)
        set_default(args, "min_diagnostic_curvature_projection_max_abs_curvature", 0.01)
        set_default(args, "max_diagnostic_curvature_projection_zero_fallback_vertices", 0)
        set_default(args, "max_diagnostic_curvature_projection_normalized_fit_residual",
                    5.0e-2)
        set_default(args, "min_diagnostic_curvature_projection_smoothing_iterations", 1)
        set_default(args, "min_diagnostic_curvature_projection_skipped_count", 1)
        set_default(args, "min_diagnostic_curvature_projection_cache_hit_count", 1)
        set_default(
            args,
            "max_diagnostic_curvature_projection_cache_miss_count",
            capillary_wave_curvature_projection_cache_miss_budget(args),
        )
        set_default(args,
                    "min_diagnostic_curvature_projection_cut_signature_cache_hit_count",
                    1)
        set_default(args,
                    "min_diagnostic_curvature_projection_reused_vertex_adjacency_count",
                    1)
        set_default(args,
                    "min_diagnostic_curvature_projection_reused_sample_adjacency_count",
                    1)
        set_default(args, "max_diagnostic_curvature_projection_vertex_adjacency_builds", 1)
        set_default(args, "max_diagnostic_curvature_projection_sample_adjacency_builds",
                    curvature_sample_adjacency_build_budget(args))


def apply_high_order_volume_corrected_motion_smoke_defaults(
        args: argparse.Namespace) -> None:
    if not args.high_order_volume_corrected_motion_smoke:
        return
    if args.high_order_3d_benchmark_qualification:
        raise ValueError(
            "--high-order-volume-corrected-motion-smoke cannot be combined with "
            "--high-order-3d-benchmark-qualification"
        )
    if args.high_order_3d_benchmark_profile_qualification:
        raise ValueError(
            "--high-order-volume-corrected-motion-smoke cannot be combined with "
            "--high-order-3d-benchmark-profile-qualification"
        )
    if args.high_order_curved_3d_simplex_smoke:
        raise ValueError(
            "--high-order-volume-corrected-motion-smoke cannot be combined with "
            "--high-order-curved-3d-simplex-smoke"
        )
    if args.high_order_mpi_motion_smoke:
        raise ValueError(
            "--high-order-volume-corrected-motion-smoke cannot be combined with "
            "--high-order-mpi-motion-smoke"
        )
    if args.high_order_capillary_projection_smoke:
        raise ValueError(
            "--high-order-volume-corrected-motion-smoke cannot be combined with "
            "--high-order-capillary-projection-smoke"
        )
    if args.high_order_capillary_response_smoke:
        raise ValueError(
            "--high-order-volume-corrected-motion-smoke cannot be combined with "
            "--high-order-capillary-response-smoke"
        )
    if args.high_order_capillary_balance_smoke:
        raise ValueError(
            "--high-order-volume-corrected-motion-smoke cannot be combined with "
            "--high-order-capillary-balance-smoke"
        )
    if getattr(args, "high_order_capillary_droplet_equilibrium_smoke", False):
        raise ValueError(
            "--high-order-volume-corrected-motion-smoke cannot be combined with "
            "--high-order-capillary-droplet-equilibrium-smoke"
        )
    if getattr(args, "high_order_capillary_wave_smoke", False):
        raise ValueError(
            "--high-order-volume-corrected-motion-smoke cannot be combined with "
            "--high-order-capillary-wave-smoke"
        )

    if not args.case:
        args.case = list(HIGH_ORDER_VOLUME_CORRECTED_MOTION_CASES)
    if args.steps is None:
        args.steps = 10
    set_default(args, "timeout_seconds", 600.0)
    args.use_high_order_implicit_cuts = True
    args.disable_cut_stabilization = False
    args.require_time_loop_convergence = True
    args.require_process_memory_diagnostics = True
    args.require_basis_cache_diagnostics = True
    args.require_high_order_cut_context_diagnostics = True
    args.require_eigen_factorization_diagnostics = True
    args.require_active_pressure_support_diagnostics = True

    set_default(args, "enable_level_set_volume_correction", True)
    set_default(args, "volume_correction_use_initial_volume", True)
    set_default(args, "volume_correction_cadence_steps", 1)
    set_default(args, "volume_correction_tolerance", 1.0e-10)
    set_default(args, "volume_correction_max_iterations", 50)
    set_default(
        args,
        "volume_correction_maximum_cumulative_interface_displacement_fraction",
        1.0,
    )
    set_default(args, "linear_algebra_backend", "eigen")
    if args.min_max_speed == 1.0e-2:
        args.min_max_speed = 1.0e-3
    if args.min_wet_mean_speed == 2.5e-4:
        args.min_wet_mean_speed = 1.0e-4
    if args.min_gate_mean_ux == 1.0e-4:
        args.min_gate_mean_ux = -1.0
    if args.min_front_mean_ux == 1.0e-4:
        args.min_front_mean_ux = -1.0
    set_default(args, "min_diagnostic_solution_velocity_range", 1.0e-3)
    set_default(args, "min_diagnostic_pressure_range", 100.0)
    set_default(args, "max_wet_fraction_volume_error", 1.0e-8)
    set_default(args, "min_diagnostic_level_set_volume_correction_count", 1)
    set_default(args, "max_diagnostic_level_set_volume_correction_achieved_error", 1.0e-8)
    set_default(args, "max_diagnostic_cut_context_rebuilds_per_step", 5.0)
    set_default(args, "max_diagnostic_generated_cell_cache_full_miss_rebuilds", 1)
    set_default(args, "max_diagnostic_assembly_timings_per_step", 4.0)
    set_default(args, "max_diagnostic_extra_assembly_timings_per_step", 3.0)
    set_default(args, "max_diagnostic_process_basis_cache_entries", 8)
    set_default(args, "max_diagnostic_process_basis_cache_entry_growth", 8)
    set_default(args, "max_diagnostic_implicit_cut_fallback_cells", 0)
    set_default(args, "min_diagnostic_achieved_interface_quadrature_order", 2)
    set_default(args, "min_diagnostic_achieved_volume_quadrature_order", 2)
    set_default(args, "max_eigen_factorization_pressure_zero_rows", 0)
    set_default(args, "max_eigen_factorization_pressure_zero_cols", 0)
    set_default(args, "max_eigen_factorization_nonfinite_entries", 0)
    set_default(args, "max_time_loop_nonlinear_iterations_per_step", 3)
    set_default(args, "max_time_loop_linear_iterations_per_step", 10)
    set_default(args, "min_interface_height_change", 1.0e-4)
    set_default(args, "min_interface_mean_abs_height_change", 2.0e-5)
    set_default(args, "min_interface_slope_change", 5.0e-5)
    set_default(args, "min_interface_final_height_span", 1.0e-4)


def apply_high_order_implicit_defaults(args: argparse.Namespace) -> None:
    if not args.use_high_order_implicit_cuts:
        return
    if args.generated_interface_geometry is None:
        args.generated_interface_geometry = "HighOrderImplicit"
    if args.implicit_cut_quadrature_backend is None:
        args.implicit_cut_quadrature_backend = "SayeHyperrectangle"
    if args.implicit_cut_fallback_policy is None:
        args.implicit_cut_fallback_policy = "Fail"
    if args.implicit_cut_root_tolerance is None:
        args.implicit_cut_root_tolerance = 1.0e-10
    if args.implicit_cut_max_subdivision_depth is None:
        args.implicit_cut_max_subdivision_depth = 8
    if args.generated_interface_quadrature_order is None:
        args.generated_interface_quadrature_order = 2
    if args.interface_quadrature_order is None:
        args.interface_quadrature_order = 2
    if args.volume_quadrature_order is None:
        args.volume_quadrature_order = 2
    if args.linear_algebra_backend is None:
        args.linear_algebra_backend = "eigen"
    if args.disable_cut_stabilization is None:
        args.disable_cut_stabilization = True
    if args.mms_nx is None:
        args.mms_nx = 2
    if args.mms_ny is None:
        args.mms_ny = args.mms_nx
    args.require_process_memory_diagnostics = True
    args.require_basis_cache_diagnostics = True
    surface_stress_physical_accuracy = bool(
        getattr(args, "high_order_capillary_droplet_equilibrium_smoke", False) or
        getattr(args, "high_order_capillary_wave_smoke", False)
    )
    # The generic high-order smoke uses tight resource/count ceilings to catch
    # accidental work amplification.  Those machine- and implementation-
    # dependent limits must not be silently reintroduced after the physical
    # SurfaceStress presets deliberately leave them unset.  Explicit CLI
    # ceilings remain intact because this branch only controls defaults.
    if not surface_stress_physical_accuracy:
        if args.max_diagnostic_assembly_timings_per_step is None:
            args.max_diagnostic_assembly_timings_per_step = 4.0
        if args.max_diagnostic_extra_assembly_timings_per_step is None:
            args.max_diagnostic_extra_assembly_timings_per_step = 3.0
        if args.max_diagnostic_cut_context_rebuilds_per_step is None:
            args.max_diagnostic_cut_context_rebuilds_per_step = 4.0
        if args.max_diagnostic_process_rss_kb is None:
            args.max_diagnostic_process_rss_kb = 300000.0
        if args.max_diagnostic_process_rss_growth_kb is None:
            args.max_diagnostic_process_rss_growth_kb = 100000.0
        if args.max_diagnostic_process_basis_cache_entries is None:
            args.max_diagnostic_process_basis_cache_entries = 4
        if args.max_diagnostic_process_basis_cache_entry_growth is None:
            args.max_diagnostic_process_basis_cache_entry_growth = 3
    if args.expect_generated_interface_geometry is None:
        args.expect_generated_interface_geometry = args.generated_interface_geometry
    if args.expect_implicit_cut_quadrature_backend is None:
        args.expect_implicit_cut_quadrature_backend = args.implicit_cut_quadrature_backend
    if (args.expect_selected_implicit_cut_quadrature_backend is None and
            args.implicit_cut_quadrature_backend != "Auto"):
        args.expect_selected_implicit_cut_quadrature_backend = (
            args.implicit_cut_quadrature_backend
        )
    if args.expect_implicit_cut_fallback_policy is None:
        args.expect_implicit_cut_fallback_policy = args.implicit_cut_fallback_policy
    if args.max_diagnostic_implicit_cut_fallback_cells is None:
        args.max_diagnostic_implicit_cut_fallback_cells = 0
    if args.min_diagnostic_achieved_volume_quadrature_order is None:
        args.min_diagnostic_achieved_volume_quadrature_order = 2
    args.require_high_order_cut_context_diagnostics = True


def validate_high_order_implicit_cases(cases: list[str],
                                       args: argparse.Namespace) -> None:
    if not args.use_high_order_implicit_cuts:
        return
    synthetic = [
        name for name in cases
        if CASES[name] is None and name not in HIGH_ORDER_SYNTHETIC_CASES
    ]
    if synthetic:
        names = ", ".join(synthetic)
        raise ValueError(
            "--use-high-order-implicit-cuts requires solver.xml-backed cases; "
            f"synthetic case(s) cannot be rewritten: {names}"
        )


def add_linear_solver_control_arguments(
        parser: argparse.ArgumentParser) -> None:
    """Register runner controls that rewrite the fluid equation's LS block."""
    parser.add_argument("--linear-relative-tolerance", type=float)
    parser.add_argument("--linear-absolute-tolerance", type=float)
    parser.add_argument("--linear-max-iterations", type=int)
    parser.add_argument("--linear-krylov-space-dimension", type=int)
    parser.add_argument("--ns-gm-max-iterations", type=int)
    parser.add_argument("--ns-cg-max-iterations", type=int)
    parser.add_argument("--ns-gm-tolerance", type=float)
    parser.add_argument("--ns-cg-tolerance", type=float)
    parser.add_argument("--linear-solver-type")
    parser.add_argument("--linear-algebra-backend")
    parser.add_argument("--linear-preconditioner")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solver", type=Path)
    parser.add_argument("--mpiexec", type=Path, default=Path("mpiexec"))
    parser.add_argument("--mpi-ranks", type=int)
    parser.add_argument("--case", choices=sorted(CASES), action="append")
    parser.add_argument("--high-order-production-qualification", action="store_true",
                        help=("enable strict high-order implicit free-surface "
                              "qualification defaults"))
    parser.add_argument("--high-order-mpi-production-qualification",
                        action="store_true",
                        help=("enable strict MPI high-order implicit "
                              "free-surface production qualification defaults"))
    parser.add_argument("--high-order-visible-motion-demo",
                        action="store_true",
                        help=("enable a strict high-order implicit "
                              "free-surface demonstration with visibly large "
                              "interface motion"))
    parser.add_argument("--high-order-3d-benchmark-smoke", action="store_true",
                        help=("enable the high-order implicit 3D D18 benchmark "
                              "diagnostics smoke defaults"))
    parser.add_argument("--high-order-3d-benchmark-qualification",
                        action="store_true",
                        help=("enable multi-step high-order implicit D18/D38 "
                              "benchmark qualification defaults"))
    parser.add_argument("--high-order-3d-benchmark-profile-qualification",
                        action="store_true",
                        help=("enable full first-profile-time high-order "
                              "implicit D18/D38 benchmark qualification "
                              "defaults"))
    parser.add_argument("--high-order-curved-3d-simplex-smoke",
                        action="store_true",
                        help=("enable the high-order implicit curved Tetra10 "
                              "solver-level smoke defaults"))
    parser.add_argument("--high-order-mpi-motion-smoke", action="store_true",
                        help=("enable the high-order implicit MPI free-surface "
                              "motion smoke defaults"))
    parser.add_argument("--high-order-capillary-projection-smoke",
                        action="store_true",
                        help=("enable a high-order implicit free-surface smoke "
                              "with nonzero surface tension and projected "
                              "level-set curvature"))
    parser.add_argument("--high-order-capillary-response-smoke",
                        action="store_true",
                        help=("enable a zero-gravity high-order implicit "
                              "capillary response smoke with projected "
                              "level-set curvature"))
    parser.add_argument("--high-order-capillary-balance-smoke",
                        action="store_true",
                        help=("enable a zero-gravity high-order implicit "
                              "Laplace-style capillary balance smoke with "
                              "projected level-set curvature"))
    parser.add_argument("--high-order-capillary-droplet-equilibrium-smoke",
                        action="store_true",
                        help=("enable a zero-gravity high-order implicit "
                              "closed-droplet SurfaceStress equilibrium smoke; "
                              "projected curvature is optional output only"))
    parser.add_argument("--high-order-capillary-wave-smoke",
                        action="store_true",
                        help=("enable a zero-gravity high-order implicit "
                              "small-amplitude SurfaceStress capillary-wave "
                              "smoke"))
    parser.add_argument("--high-order-volume-corrected-motion-smoke",
                        action="store_true",
                        help=("enable a high-order implicit free-surface "
                              "motion smoke with runtime global level-set "
                              "volume correction"))
    parser.add_argument("--source-ref")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--time-step-size", type=float)
    parser.add_argument("--synthetic-nx", type=int, default=16,
                        help="x resolution for synthetic sessile/contact-line cases")
    parser.add_argument("--synthetic-ny", type=int, default=16,
                        help="y resolution for synthetic sessile/contact-line cases")
    parser.add_argument("--synthetic-nz", type=int, default=16,
                        help="z resolution for synthetic spatial capillary cases")
    parser.add_argument("--contact-angle-degrees", type=float, default=90.0,
                        help="static or dynamic equilibrium contact angle through the liquid")
    parser.add_argument("--dynamic-initial-contact-angle-degrees", type=float,
                        default=95.0,
                        help=("initial dynamic-contact test angle through the liquid; "
                              "the generated cap preserves the equilibrium reference area"))
    parser.add_argument(
        "--sessile-radius", type=float, default=0.3,
        help=("equilibrium-reference circular-cap radius; dynamic initial "
              "radii are adjusted to preserve this cap's liquid area"),
    )
    parser.add_argument("--contact-line-mobility", type=float, default=1.0)
    parser.add_argument("--wall-slip-length", type=float, default=0.1)
    parser.add_argument(
        "--sessile-contact-line-model",
        choices=("dynamic", "prescribed"),
        default="dynamic",
        help=("contact model for the stationary sessile case; the moving "
              "contact case always uses the dynamic model"),
    )
    parser.add_argument(
        "--sessile-contact-wall",
        choices=("wall_bottom", "wall_left", "wall_right", "wall_top"),
        default="wall_bottom",
        help="wall carrying the stationary synthetic sessile cap",
    )
    parser.add_argument(
        "--sessile-contact-wall-3d",
        choices=(
            "wall_left", "wall_right", "wall_bottom", "wall_top",
            "wall_front", "wall_back",
        ),
        default="wall_bottom",
        help="wall carrying the stationary synthetic spherical cap",
    )
    parser.add_argument(
        "--level-set-positive-scale",
        type=float,
        default=1.0,
        help="positive multiplier applied to the synthetic level-set field",
    )
    parser.add_argument(
        "--level-set-active-domain",
        choices=("LevelSetNegative", "LevelSetPositive"),
        default="LevelSetNegative",
        help="declared liquid side for supported synthetic level-set cases",
    )
    parser.add_argument(
        "--capillary-droplet-center-offset",
        type=float,
        nargs=2,
        default=(0.0, 0.0),
        metavar=("DX", "DY"),
        help="translation of the synthetic two-dimensional closed droplet",
    )
    parser.add_argument(
        "--capillary-sphere-center-offset",
        type=float,
        nargs=3,
        default=(0.0, 0.0, 0.0),
        metavar=("DX", "DY", "DZ"),
        help="translation of the synthetic three-dimensional closed sphere",
    )
    parser.add_argument(
        "--sessile-tangent-center-offset",
        type=float,
        default=0.0,
        help="wall-tangent translation of the synthetic two-dimensional cap",
    )
    parser.add_argument(
        "--sessile-tangent-center-offset-3d",
        type=float,
        nargs=2,
        default=(0.0, 0.0),
        metavar=("DA", "DB"),
        help=("translations along the ordered tangent axes of the selected "
              "three-dimensional wall"),
    )
    parser.add_argument(
        "--initialize-discrete-static-contact-geometry",
        action="store_true",
        help=("replace only the wall-adjacent contact cells in a stationary "
              "two-dimensional sessile case by target-angle tangent planes; "
              "intended for manufactured discrete contact-equilibrium tests"),
    )
    parser.add_argument(
        "--dynamic-contact-wall",
        choices=("wall_bottom", "wall_left"),
        default="wall_bottom",
        help=("wall carrying the synthetic dynamic contact line; wall_left is "
              "the x=0 vertical side with outward normal (-1,0,0) and +y "
              "wall-tangent orientation"),
    )
    parser.add_argument("--max-sessile-contact-angle-error-degrees", type=float)
    parser.add_argument("--max-sessile-pressure-jump-relative-error", type=float)
    parser.add_argument("--max-sessile-liquid-area-relative-error", type=float)
    parser.add_argument("--max-sessile-liquid-volume-relative-error", type=float)
    parser.add_argument("--max-sessile-base-radius-relative-error", type=float)
    parser.add_argument("--max-sessile-apex-height-relative-error", type=float)
    parser.add_argument("--max-sessile-parasitic-capillary-number", type=float)
    parser.add_argument("--max-capillary-parasitic-capillary-number", type=float)
    parser.add_argument(
        "--defer-static-physical-gates-to-matrix",
        action="store_true",
        help=("leave static shape, pressure, angle, and parasitic-current "
              "acceptance to a predeclared matrix-level convergence gate"),
    )
    parser.add_argument(
        "--require-ren-e-speed-sign",
        dest="require_ren_e_speed_sign",
        action="store_true",
        default=None,
    )
    parser.add_argument(
        "--allow-ren-e-speed-sign-mismatch",
        dest="require_ren_e_speed_sign",
        action="store_false",
    )
    parser.add_argument("--max-ren-e-speed-relative-error", type=float)
    parser.add_argument("--timeout-seconds", type=float)
    parser.add_argument("--preserve-run-dir", action="store_true")
    parser.add_argument(
        "--enable-physical-history-instrumentation",
        action="store_true",
        help=("read every saved state to report liquid-measure history and the "
              "first initially-dry wall vertex that wets while all incident "
              "P1 cell-centroid samples remain dry (intended for D18/D38 audits)"),
    )
    parser.add_argument(
        "--require-level-set-mass-correction-histories",
        action="store_true",
        help=("require separate accepted-clock uncorrected and corrected liquid-"
              "mass histories, cross-checked against production physical cut "
              "volumes"),
    )
    parser.add_argument(
        "--require-free-surface-energy-history",
        action="store_true",
        help=("require an initialized-plus-every-accepted saved-state energy "
              "proxy history; this is an output diagnostic, not a discrete "
              "energy-theorem claim"),
    )
    parser.add_argument(
        "--max-free-surface-energy-positive-step-increment-relative",
        type=float,
    )
    parser.add_argument(
        "--max-free-surface-energy-above-initial-relative",
        type=float,
    )
    parser.add_argument(
        "--enable-free-surface-conservative-balance-diagnostic",
        action="store_true",
        help=("assemble and report the instantaneous constrained pressure/"
              "surface-energy virtual-work split; this is not a discrete "
              "energy-theorem diagnostic"),
    )
    parser.add_argument(
        "--require-free-surface-conservative-balance",
        "--require-free-surface-conservative-balance-diagnostic",
        dest="require_free_surface_conservative_balance",
        action="store_true",
    )
    parser.add_argument(
        "--require-free-surface-pressure-representability-diagnostic",
        dest="require_free_surface_pressure_representability_diagnostic",
        action="store_true",
        help=("require finite normal-equation-stationary constrained pressure-"
              "space diagnostic telemetry for the surface-area plus Young-"
              "wall-energy load; this applies no residual-distance gate and "
              "makes no physical representability claim"),
    )
    parser.add_argument(
        "--max-free-surface-pressure-representability-relative-distance",
        type=float,
        help=("opt into a fail-closed pressure-range distance gate at the "
              "final accepted static Newton state; moving qualifications "
              "leave this unset"),
    )
    parser.add_argument(
        "--initialize-static-compatible-pressure",
        action="store_true",
        default=None,
        help=("once preload the static pressure coefficients from the same "
              "constrained surface-energy/pressure pair used by the required "
              "representability-distance gate"),
    )
    parser.add_argument(
        "--initialize-discrete-static-capillary-equilibrium",
        action="store_true",
        default=None,
        help=("replace the sampled static level set with a fixed-topology, "
              "fixed-volume stationary point of the declared discrete "
              "surface, wall, and gravitational potential"),
    )
    parser.add_argument("--static-capillary-volume-tolerance", type=float)
    parser.add_argument(
        "--static-capillary-projected-gradient-tolerance", type=float)
    parser.add_argument(
        "--static-capillary-pressure-representability-max-residual-norm",
        type=float,
    )
    parser.add_argument(
        "--static-capillary-pressure-representability-max-relative-distance",
        type=float,
    )
    parser.add_argument(
        "--static-capillary-physical-equilibrium-max-residual-norm",
        type=float,
    )
    parser.add_argument(
        "--static-capillary-constant-pressure-kkt-max-residual-norm",
        type=float,
    )
    parser.add_argument(
        "--static-capillary-constant-pressure-kkt-max-relative-distance",
        type=float,
    )
    parser.add_argument(
        "--static-capillary-finite-difference-relative-step",
        type=float,
    )
    parser.add_argument("--static-capillary-max-iterations", type=int)
    parser.add_argument(
        "--static-capillary-max-topology-epoch-transitions", type=int)
    parser.add_argument(
        "--static-capillary-limited-memory-history-size",
        type=int,
    )
    parser.add_argument(
        "--static-capillary-limited-memory-curvature-tolerance",
        type=float,
    )
    parser.add_argument(
        "--enable-level-set-reinitialization",
        dest="enable_level_set_reinitialization",
        action="store_true",
        default=None,
    )
    parser.add_argument(
        "--disable-level-set-reinitialization",
        dest="enable_level_set_reinitialization",
        action="store_false",
    )
    parser.add_argument("--reinitialization-cadence-steps", type=int)
    parser.add_argument(
        "--require-static-capillary-balance-qualification",
        choices=("prerequisite_only", "qualified"),
    )
    parser.add_argument(
        "--max-free-surface-conservative-balance-normalized-imbalance",
        "--max-free-surface-conservative-balance-relative-residual",
        dest="max_free_surface_conservative_balance_normalized_imbalance",
        type=float,
    )
    parser.add_argument("--qualification-log", type=Path)
    parser.add_argument("--disable-vtk-output", action="store_true")
    parser.add_argument("--final-output-only", action="store_true")
    parser.add_argument("--vtk-save-increment", type=int)
    parser.add_argument("--start-saving-after-step", type=int)
    parser.add_argument("--min-max-speed", type=float, default=1.0e-2)
    parser.add_argument("--min-wet-mean-speed", type=float, default=2.5e-4)
    parser.add_argument("--min-gate-mean-ux", type=float, default=1.0e-4)
    parser.add_argument("--min-front-mean-ux", type=float, default=1.0e-4)
    parser.add_argument("--min-active-volume-change", type=float, default=0.0)
    parser.add_argument("--min-interface-height-change", type=float)
    parser.add_argument("--min-interface-mean-abs-height-change", type=float)
    parser.add_argument("--min-interface-slope-change", type=float)
    parser.add_argument("--min-interface-final-height-span", type=float)
    parser.add_argument("--max-static-speed", type=float, default=1.0e-9)
    parser.add_argument("--stale-pressure-gauge-tolerance", type=float)
    parser.add_argument("--max-wet-fraction-volume-error", type=float)
    parser.add_argument("--require-reference-profile-comparison", action="store_true")
    parser.add_argument("--reference-profile-time-tolerance", type=float)
    parser.add_argument("--reference-profile-sample-radius", type=float)
    parser.add_argument("--reference-profile-elevated-front-clearance",
                        type=float)
    parser.add_argument("--min-reference-profile-coverage", type=float)
    parser.add_argument("--min-reference-profile-direct-coverage", type=float)
    parser.add_argument("--max-reference-profile-rmse", type=float)
    parser.add_argument("--max-reference-profile-mae", type=float)
    parser.add_argument("--max-reference-profile-max-abs-error", type=float)
    parser.add_argument("--max-reference-profile-elevated-front-lag", type=float)
    parser.add_argument("--allow-experimental-profile-linear-solver",
                        action="store_true",
                        help=("permit non-BlockSchur linear-solver overrides "
                              "for D18/D38 profile diagnostics"))
    parser.add_argument("--max-solver-elapsed-wall-seconds", type=float,
                        help=("fail a completed solver run whose measured wall "
                              "time exceeds this budget"))
    parser.add_argument("--max-solver-elapsed-seconds-per-accepted-step",
                        type=float,
                        help=("fail a completed solver run whose measured wall "
                              "time per accepted time step exceeds this budget"))
    parser.add_argument("--allow-timeout-diagnostics", action="store_true")
    parser.add_argument("--allow-failure-diagnostics", action="store_true")
    parser.add_argument("--min-diagnostic-solution-velocity-range", type=float)
    parser.add_argument("--min-diagnostic-pressure-range", type=float)
    parser.add_argument("--min-capillary-response-speed-per-surface-tension",
                        type=float)
    parser.add_argument("--max-capillary-balance-speed-per-surface-tension",
                        type=float)
    parser.add_argument("--min-diagnostic-level-set-volume-correction-count",
                        type=int)
    parser.add_argument("--max-diagnostic-level-set-volume-correction-achieved-error",
                        type=float)
    parser.add_argument("--max-diagnostic-active-volume-error", type=float)
    parser.add_argument("--min-diagnostic-cut-volume-exact-order", type=int)
    parser.add_argument("--min-diagnostic-cut-volume-max-exact-order", type=int)
    parser.add_argument("--max-diagnostic-cut-adjacent-scale", type=float)
    parser.add_argument("--min-diagnostic-cut-adjacent-capped-scale-count", type=int)
    parser.add_argument("--min-diagnostic-active-pruned-volume-regions", type=int)
    parser.add_argument("--min-diagnostic-active-min-volume-fraction", type=float)
    parser.add_argument("--min-diagnostic-generated-pruned-volume-rules", type=int)
    parser.add_argument("--max-diagnostic-implicit-cut-fallback-cells", type=int)
    parser.add_argument("--min-diagnostic-achieved-interface-quadrature-order", type=int)
    parser.add_argument("--min-diagnostic-achieved-volume-quadrature-order", type=int)
    parser.add_argument("--expect-generated-interface-geometry")
    parser.add_argument("--expect-implicit-cut-quadrature-backend")
    parser.add_argument("--required-implicit-cut-backend-qualification")
    parser.add_argument("--expect-selected-implicit-cut-quadrature-backend")
    parser.add_argument("--expect-implicit-cut-backend-qualification")
    parser.add_argument("--expect-implicit-cut-fallback-policy")
    parser.add_argument("--require-high-order-cut-context-diagnostics", action="store_true")
    parser.add_argument("--require-mms-verification", action="store_true")
    parser.add_argument("--require-time-loop-convergence", action="store_true")
    parser.add_argument("--min-diagnostic-blockschur-true-residual-retries", type=int)
    parser.add_argument("--require-newton-direction-check-diagnostics", action="store_true")
    parser.add_argument("--require-jacobian-check-diagnostics", action="store_true")
    parser.add_argument("--require-jacobian-top-mismatch-diagnostics", action="store_true")
    parser.add_argument("--require-jacobian-component-block-diagnostics", action="store_true")
    parser.add_argument("--require-linear-solve-history-diagnostics", action="store_true")
    parser.add_argument("--require-form-block-diagnostics", action="store_true")
    parser.add_argument("--require-cut-context-solution-source-diagnostics", action="store_true")
    parser.add_argument("--enable-newton-assembly-diagnostics", action="store_true")
    parser.add_argument("--require-newton-assembly-diagnostics", action="store_true")
    parser.add_argument("--require-assembly-timing-diagnostics", action="store_true")
    parser.add_argument("--max-diagnostic-assembly-timings-per-step", type=float)
    parser.add_argument("--max-diagnostic-extra-assembly-timings-per-step", type=float)
    parser.add_argument("--max-diagnostic-cut-context-rebuilds-per-step", type=float)
    parser.add_argument("--max-diagnostic-newton-matrix-assemblies-per-step", type=float)
    parser.add_argument("--max-newton-direction-relative-error", type=float)
    parser.add_argument("--max-jacobian-check-relative-error", type=float)
    parser.add_argument("--max-jacobian-component-block-relative-error", type=float)
    parser.add_argument(
        "--jacobian-check-scheme",
        choices=("forward", "central"),
        help="Finite-difference scheme used by the solver Jacobian diagnostic.",
    )
    parser.add_argument("--disable-cut-stabilization",
                        dest="disable_cut_stabilization",
                        action="store_true",
                        default=None)
    parser.add_argument("--enable-cut-stabilization",
                        dest="disable_cut_stabilization",
                        action="store_false")
    parser.add_argument("--disable-cut-metadata-scale", action="store_true")
    parser.add_argument("--disable-velocity-extension", action="store_true")
    parser.add_argument(
        "--wet-extension-advection-velocity-method",
        choices=(
            "wall_compatible_normal",
            "nearest_active_vertex",
            "nearest_interface_point",
        ),
    )
    parser.add_argument("--trace-level-set-advection-velocity", action="store_true")
    parser.add_argument("--cut-cell-velocity-gradient-penalty", type=float)
    parser.add_argument("--cut-cell-pressure-gradient-penalty", type=float)
    parser.add_argument(
        "--cut-cell-pressure-stabilization-policy",
        choices=(
            "enabled",
            "incremental",
            "disabled",
            "disabled_for_refreshed_frozen_high_order",
        ),
        help=("cut-adjacent pressure ghost-penalty state; the area-gradient "
              "pair defaults to incremental pressure stabilization"),
    )
    parser.add_argument("--surface-tension", type=float)
    parser.add_argument(
        "--capillary-force-form",
        choices=(
            "surface_stress",
            "generated_curvature_traction",
            "kinematic_area_gradient_traction",
        ),
        default="surface_stress",
        help=("surface force discretization; generated curvature traction "
              "and kinematic area-gradient traction are explicit unfitted "
              "candidates"),
    )
    parser.add_argument(
        "--prescribed-capillary-curvature",
        type=float,
        help=("positive scalar curvature for generated curvature traction; "
              "mutually exclusive with a projected curvature field"),
    )
    parser.add_argument("--projected-curvature-field")
    parser.add_argument("--curvature-projection-cadence-steps", type=int)
    parser.add_argument("--curvature-projection-max-normalized-fit-residual", type=float)
    parser.add_argument("--curvature-projection-max-neighbor-fallback-vertices", type=int)
    parser.add_argument("--curvature-projection-max-zero-fallback-vertices", type=int)
    parser.add_argument("--curvature-projection-supplemental-sample-weight", type=float)
    parser.add_argument(
        "--curvature-projection-recovery-mode",
        choices=(
            "level_set_quadratic",
            "generated_interface_patch",
            "kinematic_area_gradient",
        ),
    )
    parser.add_argument(
        "--curvature-projection-kinematic-area-gradient-filter-coefficient",
        type=float,
    )
    parser.add_argument("--curvature-projection-narrow-band-width", type=float)
    parser.add_argument("--curvature-projection-smoothing-iterations", type=int)
    parser.add_argument("--curvature-projection-smoothing-relaxation", type=float)
    parser.add_argument(
        "--curvature-projection-smoothing-mode",
        choices=("local_graph", "mass_stiffness_operator", "mass_stiffness"),
    )
    parser.add_argument("--expect-curvature-projection-smoothing-mode")
    parser.add_argument(
        "--expect-curvature-projection-recovery-mode",
        choices=(
            "level_set_quadratic",
            "generated_interface_patch",
            "kinematic_area_gradient",
        ),
    )
    parser.add_argument("--min-diagnostic-curvature-projection-operator-edges", type=int)
    parser.add_argument(
        "--min-diagnostic-curvature-projection-interface-geometry-samples",
        type=int,
    )
    parser.add_argument(
        "--min-diagnostic-curvature-projection-interface-patch-fitted-vertices",
        type=int,
    )
    parser.add_argument("--max-capillary-curvature-relative-error", type=float)
    parser.add_argument("--max-capillary-pressure-jump-relative-error", type=float)
    parser.add_argument("--max-capillary-rejected-steps", type=int)
    parser.add_argument("--max-capillary-dt-updates", type=int)
    parser.add_argument("--max-capillary-speed-per-surface-tension", type=float)
    parser.add_argument("--max-capillary-nonlinear-residual", type=float)
    parser.add_argument("--max-capillary-linear-relative-residual", type=float)
    parser.add_argument("--max-capillary-wave-frequency-relative-error", type=float)
    parser.add_argument("--max-capillary-wave-profile-relative-error", type=float)
    parser.add_argument("--max-capillary-wave-mean-offset", type=float)
    parser.add_argument(
        "--max-capillary-wave-temporal-liquid-volume-relative-drift",
        type=float,
        help=("fail when any accepted production physical cut volume drifts "
              "from the pre-loop initialized state by more than this relative "
              "amount"),
    )
    parser.add_argument("--capillary-convergence-resolution-key")
    parser.add_argument("--capillary-convergence-metric", action="append")
    parser.add_argument("--min-capillary-convergence-rate", type=float)
    parser.add_argument("--min-capillary-convergence-points", type=int)
    parser.add_argument("--enable-level-set-volume-correction",
                        dest="enable_level_set_volume_correction",
                        action="store_true",
                        default=None)
    parser.add_argument("--disable-level-set-volume-correction",
                        dest="enable_level_set_volume_correction",
                        action="store_false")
    parser.add_argument("--volume-correction-cadence-steps", type=int)
    parser.add_argument("--volume-correction-use-initial-volume",
                        dest="volume_correction_use_initial_volume",
                        action="store_true",
                        default=None)
    parser.add_argument("--volume-correction-target-volume",
                        dest="volume_correction_use_initial_volume",
                        action="store_false")
    parser.add_argument("--volume-correction-tolerance", type=float)
    parser.add_argument("--volume-correction-max-iterations", type=int)
    parser.add_argument(
        "--volume-correction-maximum-cumulative-interface-displacement-fraction",
        type=float,
    )
    parser.add_argument("--max-nonlinear-iterations", type=int)
    add_linear_solver_control_arguments(parser)
    parser.add_argument("--disable-coupled-outer-fgmres", action="store_true")
    parser.add_argument("--use-high-order-implicit-cuts", action="store_true")
    parser.add_argument("--mms-nx", type=int)
    parser.add_argument("--mms-ny", type=int)
    parser.add_argument("--generated-interface-geometry")
    parser.add_argument("--implicit-cut-quadrature-backend")
    parser.add_argument("--implicit-cut-fallback-policy")
    parser.add_argument("--implicit-cut-root-tolerance", type=float)
    parser.add_argument("--implicit-cut-max-subdivision-depth", type=int)
    parser.add_argument("--generated-interface-quadrature-order", type=int)
    parser.add_argument("--interface-quadrature-order", type=int)
    parser.add_argument("--volume-quadrature-order", type=int)
    parser.add_argument("--enable-blockschur-true-residual-retry", action="store_true")
    parser.add_argument("--enable-jacobian-check", action="store_true")
    parser.add_argument("--jacobian-check-iteration", type=int)
    parser.add_argument("--jacobian-check-step", type=float)
    parser.add_argument("--jacobian-check-components")
    parser.add_argument("--jacobian-check-component-sweeps")
    parser.add_argument("--enable-newton-direction-check", action="store_true")
    parser.add_argument("--newton-line-search-fail-on-no-reduction",
                        action="store_true")
    parser.add_argument("--newton-line-search-max-iterations", type=int)
    parser.add_argument("--enable-linear-solve-history", action="store_true")
    parser.add_argument("--linear-solve-history-max-calls", type=int)
    parser.add_argument("--enable-linear-solve-component-norms", action="store_true")
    parser.add_argument("--linear-solve-component-norms-max-newton-it", type=int)
    parser.add_argument("--enable-linear-solve-memory-diagnostics", action="store_true")
    parser.add_argument("--require-linear-solve-memory-diagnostics", action="store_true")
    parser.add_argument("--enable-timeloop-initialization-diagnostics", action="store_true")
    parser.add_argument("--require-timeloop-initialization-diagnostics", action="store_true")
    parser.add_argument("--enable-fsils-matrix-diagnostics", action="store_true")
    parser.add_argument("--require-fsils-matrix-diagnostics", action="store_true")
    parser.add_argument("--fsils-matrix-diagnostics-every-n", type=int)
    parser.add_argument("--fsils-matrix-diagnostics-max-records", type=int)
    parser.add_argument("--max-fsils-matrix-zero-rows", type=int)
    parser.add_argument("--max-fsils-matrix-missing-diag", type=int)
    parser.add_argument("--max-fsils-matrix-diag-col-mismatch", type=int)
    parser.add_argument("--max-fsils-matrix-duplicate-diag-entries", type=int)
    parser.add_argument("--max-fsils-matrix-duplicate-diag-rows", type=int)
    parser.add_argument("--max-fsils-matrix-zero-diag", type=int)
    parser.add_argument("--max-fsils-matrix-nonfinite-entries", type=int)
    parser.add_argument("--max-fsils-accepted-true-residual-norm", type=float)
    parser.add_argument("--require-basis-cache-diagnostics", action="store_true")
    parser.add_argument("--max-diagnostic-process-basis-cache-entries", type=int)
    parser.add_argument("--max-diagnostic-process-rss-kb", type=float)
    parser.add_argument("--max-diagnostic-process-rss-growth-kb", type=float)
    parser.add_argument("--max-diagnostic-process-basis-cache-entry-growth", type=int)
    parser.add_argument("--enable-form-block-diagnostics", action="store_true")
    parser.add_argument("--enable-interior-face-timing", action="store_true")
    parser.add_argument("--require-interior-face-timing-diagnostics", action="store_true")
    parser.add_argument("--enable-cut-volume-timing", action="store_true")
    parser.add_argument("--require-cut-volume-timing-diagnostics", action="store_true")
    parser.add_argument("--enable-jit-specialization-trace", action="store_true")
    parser.add_argument("--require-jit-specialization-trace-diagnostics", action="store_true")
    parser.add_argument("--enable-jit-cache-diagnostics", action="store_true")
    parser.add_argument("--require-jit-cache-diagnostics", action="store_true")
    parser.add_argument("--require-compiled-cut-volume-jit", action="store_true")
    parser.add_argument("--require-process-memory-diagnostics", action="store_true")
    parser.add_argument("--require-marked-interior-face-fallback-diagnostics", action="store_true")
    parser.add_argument("--require-assembly-topology-consistency", action="store_true")
    parser.add_argument("--max-diagnostic-generated-cell-cache-full-miss-rebuilds", type=int)
    parser.add_argument("--min-diagnostic-cut-context-refresh-skips", type=int)
    parser.add_argument("--require-eigen-factorization-diagnostics", action="store_true")
    parser.add_argument("--require-active-pressure-support-diagnostics", action="store_true")
    parser.add_argument("--require-level-set-advection-velocity-diagnostics", action="store_true")
    parser.add_argument("--expect-level-set-advection-velocity-extension-method")
    parser.add_argument("--expect-level-set-advection-velocity-interface-sample-source")
    parser.add_argument("--min-diagnostic-level-set-advection-interface-samples", type=int)
    parser.add_argument("--require-curvature-projection-diagnostics", action="store_true")
    parser.add_argument("--require-curvature-projection-newton-freshness", action="store_true")
    parser.add_argument("--min-diagnostic-curvature-projection-count", type=int)
    parser.add_argument("--min-diagnostic-curvature-projection-max-abs-curvature", type=float)
    parser.add_argument("--max-diagnostic-curvature-projection-fallback-vertices", type=int)
    parser.add_argument("--max-diagnostic-curvature-projection-zero-fallback-vertices", type=int)
    parser.add_argument("--max-diagnostic-curvature-projection-normalized-fit-residual", type=float)
    parser.add_argument("--min-diagnostic-curvature-projection-smoothing-iterations", type=int)
    parser.add_argument("--min-diagnostic-curvature-projection-skipped-count", type=int)
    parser.add_argument("--min-diagnostic-curvature-projection-cache-hit-count", type=int)
    parser.add_argument("--max-diagnostic-curvature-projection-cache-miss-count", type=int)
    parser.add_argument("--min-diagnostic-curvature-projection-cut-signature-cache-hit-count", type=int)
    parser.add_argument("--min-diagnostic-curvature-projection-reused-vertex-adjacency-count", type=int)
    parser.add_argument("--min-diagnostic-curvature-projection-reused-sample-adjacency-count", type=int)
    parser.add_argument("--max-diagnostic-curvature-projection-vertex-adjacency-builds", type=int)
    parser.add_argument("--max-diagnostic-curvature-projection-sample-adjacency-builds", type=int)
    parser.add_argument("--max-eigen-factorization-zero-rows", type=int)
    parser.add_argument("--max-eigen-factorization-pressure-zero-rows", type=int)
    parser.add_argument("--max-eigen-factorization-pressure-zero-cols", type=int)
    parser.add_argument("--max-eigen-factorization-nonfinite-entries", type=int)
    parser.add_argument("--max-time-loop-nonlinear-iterations-per-step", type=int)
    parser.add_argument("--max-time-loop-linear-iterations-per-step", type=int)
    parser.add_argument("--max-time-loop-outer-iterations-per-step", type=int)
    parser.add_argument(
        "--max-time-loop-inner-iterations-total-per-step", type=int)
    parser.add_argument("--enable-adaptive-time-loop", action="store_true")
    parser.add_argument("--adaptive-time-loop-min-dt", type=float)
    parser.add_argument("--adaptive-time-loop-max-dt", type=float)
    parser.add_argument("--adaptive-time-loop-max-retries", type=int)
    parser.add_argument("--adaptive-time-loop-decrease-factor", type=float)
    parser.add_argument("--adaptive-time-loop-increase-factor", type=float)
    parser.add_argument("--adaptive-time-loop-target-newton-iterations", type=int)
    parser.add_argument("--adaptive-time-loop-max-steps-multiplier", type=int)
    args = parser.parse_args()
    remember_explicit_cli_overrides(args)
    apply_high_order_production_qualification_defaults(args)
    apply_high_order_mpi_production_qualification_defaults(args)
    apply_high_order_visible_motion_demo_defaults(args)
    apply_high_order_3d_benchmark_smoke_defaults(args)
    apply_high_order_3d_benchmark_qualification_defaults(args)
    apply_high_order_3d_benchmark_profile_qualification_defaults(args)
    apply_high_order_curved_3d_simplex_smoke_defaults(args)
    apply_high_order_mpi_motion_smoke_defaults(args)
    apply_high_order_capillary_projection_smoke_defaults(args)
    apply_high_order_capillary_response_smoke_defaults(args)
    apply_high_order_capillary_balance_smoke_defaults(args)
    apply_high_order_capillary_droplet_equilibrium_smoke_defaults(args)
    apply_high_order_capillary_wave_smoke_defaults(args)
    apply_high_order_volume_corrected_motion_smoke_defaults(args)
    apply_high_order_implicit_defaults(args)
    apply_level_set_advection_velocity_diagnostic_gate_defaults(args)
    require_profile_production_linear_solver_policy(args)

    solver = resolve_solver(args.solver)
    cases = args.case or ["mini2d"]
    validate_high_order_implicit_cases(cases, args)
    report = []
    for case_name in cases:
        case_args = case_args_for_run(case_name, args)
        report.append(run_case(case_name, solver, case_args))
        write_qualification_log(args.qualification_log, solver, report, complete=False)
    convergence_errors = capillary_convergence_rate_errors(report, args)
    if convergence_errors:
        failure = {
            "case": "capillary_convergence",
            "passed": False,
            "errors": convergence_errors,
        }
        add_solver_control_overrides(failure, args)
        report.append(failure)
        write_qualification_log(args.qualification_log, solver, report, complete=False)
        raise RuntimeError(format_failure_exception(failure, args.qualification_log))
    write_qualification_log(args.qualification_log, solver, report, complete=True)
    print(json.dumps(
        qualification_payload(solver, report, complete=True),
        indent=2,
        sort_keys=True,
    ))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)

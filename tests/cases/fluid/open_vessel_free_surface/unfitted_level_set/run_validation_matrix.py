#!/usr/bin/env python3
"""Run named unfitted level-set free-surface validation probes.

The probes in this file intentionally copy source cases into a temporary work
directory before generation or solver execution. That keeps checked-in case
directories free of result files while making the broader validation evidence
repeatable.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[5]
CASE_ROOT = ROOT / "tests/cases/fluid/open_vessel_free_surface/unfitted_level_set"
TOOLS_DIR = ROOT / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from collate_vtk_time_series import collate_time_series

TIMELOOP_STEP_START_RE = re.compile(
    r"TimeLoop: step_start step=(?P<step>[0-9]+)"
    r" time=(?P<time>[-+0-9.eE]+)"
    r" dt=(?P<dt>[-+0-9.eE]+)"
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
    r" reason=(?P<reason>[A-Za-z0-9_]+)"
    r" \(newton: converged=(?P<converged>[01])"
    r" iters=(?P<iters>[0-9]+)"
    r" \|\|r\|\|=(?P<residual>[-+0-9.eE]+)"
    r"(?: \|\|r_field\|\|=(?P<field_residual>[-+0-9.eE]+)"
    r" \|\|r_aux\|\|=(?P<aux_residual>[-+0-9.eE]+))?"
    r"\)"
)
TIMELOOP_DONE_RE = re.compile(
    r"TimeLoop: loop\.run\(\) returned success=(?P<success>[01])"
    r" steps_taken=(?P<steps>[0-9]+)"
    r" final_time=(?P<final_time>[-+0-9.eE]+)"
)
TIMELOOP_NONLINEAR_RE = re.compile(
    r"TimeLoop: nonlinear_done step=(?P<step>[0-9]+)"
    r" time=(?P<time>[-+0-9.eE]+)"
    r" converged=(?P<converged>[01])"
    r" iters=(?P<iters>[0-9]+)"
    r" \|\|r\|\|=(?P<residual>[-+0-9.eE]+)"
    r"(?: \|\|r_field\|\|=(?P<field_residual>[-+0-9.eE]+)"
    r" \|\|r_aux\|\|=(?P<aux_residual>[-+0-9.eE]+))?"
    r"(?: \(linear: converged=(?P<linear_converged>[01])"
    r" iters=(?P<linear_iters>[0-9]+)"
    r" rel=(?P<linear_relative_residual>[-+0-9.eE]+)\))?"
)
KEY_VALUE_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)=('[^']*'|\"[^\"]*\"|[^\s\]]+)")
COMPONENT_NORM_RE = re.compile(
    r"\[(?P<name>.*?) norm=(?P<norm>[-+0-9.eE]+)"
    r" mean=(?P<mean>[-+0-9.eE]+)"
    r" min=(?P<min>[-+0-9.eE]+)"
    r" max=(?P<max>[-+0-9.eE]+)\]"
)
TIMING_LINE_RE = re.compile(
    r"^\s*(?P<label>[A-Za-z][A-Za-z0-9 /()+-]*?):\s*"
    r"(?P<seconds>[-+0-9.eE]+)\s+s(?:\s+\((?P<detail>[^)]*)\))?"
)
NEWTON_TIMING_DETAIL_RE = re.compile(
    r"(?P<newton>[0-9]+)\s+Newton iters,\s+"
    r"(?P<assemblies>[0-9]+)\s+assemblies,\s+"
    r"(?P<linear>[0-9]+)\s+linear iters"
)
CMAKE_CACHE_PROVENANCE_KEYS = (
    "CMAKE_BUILD_TYPE",
    "CMAKE_CXX_COMPILER",
    "CMAKE_C_COMPILER",
    "FE_ENABLE_EIGEN",
    "FE_ENABLE_LLVM_JIT",
    "FE_ENABLE_LLVM_JIT_RESOLVED",
    "FE_ENABLE_MPI",
    "FE_USE_MPI_WRAPPERS",
    "SV_USE_INTERNAL_EIGEN",
    "USE_SYSTEM_EIGEN",
    "MESH_ENABLE_EIGEN",
    "MESH_ENABLE_MPI",
    "MPIEXEC_EXECUTABLE",
    "MPIEXEC_MAX_NUMPROCS",
    "MPIEXEC_NUMPROC_FLAG",
    "MPI_CXX_COMPILER",
)
SOLVER_XML_CONTROL_TAGS = {
    "Advection_velocity_from_field",
    "Auto_register_velocity_field",
    "Constant_velocity",
    "Cut_cell_pressure_gradient_penalty",
    "Cut_cell_pressure_stabilization_policy",
    "Cut_cell_metadata_scale_cap",
    "Element_order",
    "Enable_reinitialization",
    "Enable_SUPG",
    "Enable_velocity_extension",
    "Enable_volume_correction",
    "Generated_interface_geometry",
    "Generated_interface_quadrature_order",
    "Geometry_tangent_policy",
    "Interface_quadrature_order",
    "Implicit_cut_fallback_policy",
    "Implicit_cut_max_subdivision_depth",
    "Implicit_cut_quadrature_backend",
    "Implicit_cut_root_tolerance",
    "Level_set_source",
    "Linear_algebra",
    "Max_iterations",
    "Number_of_time_steps",
    "Operator_tag",
    "Output_frequency",
    "Preconditioner",
    "Reinitialization_cadence_steps",
    "Reinitialization_method",
    "SUPG_tau_scale",
    "Spectral_radius_of_infinite_time_step",
    "Temporal_and_spatial_values_file_path",
    "Time_step_size",
    "Tolerance",
    "Volume_correction_cadence_steps",
    "Volume_correction_max_iterations",
    "Volume_correction_tolerance",
    "Volume_correction_use_initial_volume",
    "Transport_form",
    "Use_cut_metadata_scale",
    "Use_wet_extension_advection_velocity",
    "Velocity_field_name",
    "Velocity_source",
    "Volume_quadrature_order",
}
ACTIVE_DOMAIN_ACCEPTED_TOKENS = {
    "levelsetnegative",
    "negative",
    "phinegative",
    "levelsetpositive",
    "positive",
    "phipositive",
    "none",
    "off",
    "inactive",
}
ACTIVE_DOMAIN_METHOD_ACCEPTED_TOKENS = {
    "cutvolume",
    "smoothedindicator",
}
ACTIVE_DOMAIN_XML_LITERAL_RE = re.compile(
    r"<Active_domain>\s*([^<{][^<]*?)\s*</Active_domain>"
)
ACTIVE_DOMAIN_JSON_LITERAL_RE = re.compile(
    r'"active_domain"\s*:\s*"([^"]+)"'
)
FREE_SURFACE_CURVATURE_FIELD_TAGS = (
    "Curvature_field_name",
    "CurvatureFieldName",
    "Curvature_field",
    "CurvatureField",
    "Projected_curvature_field",
    "ProjectedCurvatureField",
    "Free_surface_curvature_field",
    "FreeSurfaceCurvatureField",
)
FREE_SURFACE_UNSUPPORTED_SCOPE_TAG_TOKENS = {
    "airdensity",
    "airviscosity",
    "densityjump",
    "densityoutside",
    "enabletwophase",
    "enabletwosidedtraces",
    "enrichedpressure",
    "exteriorfluiddensity",
    "exteriorfluidviscosity",
    "gasdensity",
    "gasviscosity",
    "inactivefluiddensity",
    "inactivefluidviscosity",
    "interfacialjumpcondition",
    "interfacematerialjump",
    "jumpcondition",
    "materialjump",
    "outsidedensity",
    "outsideviscosity",
    "pressureenrichment",
    "pressurespaceenrichment",
    "twophase",
    "twophaseinterface",
    "twosidedinterface",
    "twosidedtraces",
    "usepressureenrichment",
    "viscosityjump",
    "viscosityoutside",
}
FREE_SURFACE_UNSUPPORTED_SCOPE_TAG_FRAGMENTS = (
    "densityjump",
    "viscosityjump",
    "materialjump",
    "pressureenrichment",
    "pressurespaceenrichment",
    "twophase",
    "twosidedtrace",
    "twosidedinterface",
    "jumpcondition",
    "exteriorfluid",
    "inactivefluid",
    "gasdensity",
    "gasviscosity",
    "airdensity",
    "airviscosity",
    "densityoutside",
    "viscosityoutside",
    "outsidedensity",
    "outsideviscosity",
)
FREE_SURFACE_SCOPE_MODE_TAG_TOKENS = {
    "fluidmodel",
    "formulation",
    "interfaceformulation",
    "interfacemodel",
    "interfacephysics",
    "materialmodel",
    "phasemodel",
}
FREE_SURFACE_UNSUPPORTED_SCOPE_VALUE_FRAGMENTS = (
    "cutfemjump",
    "densityjump",
    "jumpcondition",
    "materialjump",
    "pressureenrichment",
    "twophase",
    "twosided",
    "viscosityjump",
)
HIGH_ORDER_MMS_REQUIRED_SOLVER_FEATURES = (
    "FE_ENABLE_LLVM_JIT",
    "FE_ENABLE_LLVM_JIT_RESOLVED",
    "FE_ENABLE_EIGEN",
)
HIGH_ORDER_MMS_BASE_XML_CONTROLS = (
    ("Generated_interface_geometry", "HighOrderImplicit"),
    ("Implicit_cut_quadrature_backend", "SayeHyperrectangle"),
    ("Implicit_cut_fallback_policy", "Fail"),
    ("Generated_interface_quadrature_order", "2"),
    ("Interface_quadrature_order", "2"),
    ("Volume_quadrature_order", "2"),
    ("Use_cut_metadata_scale", "false"),
    ("Cut_cell_pressure_stabilization_policy", "DisabledForRefreshedFrozenHighOrder"),
)
HIGH_ORDER_MMS_BASE_PRESSURE_ENABLED_XML_CONTROLS = (
    ("Generated_interface_geometry", "HighOrderImplicit"),
    ("Implicit_cut_quadrature_backend", "SayeHyperrectangle"),
    ("Implicit_cut_fallback_policy", "Fail"),
    ("Generated_interface_quadrature_order", "2"),
    ("Interface_quadrature_order", "2"),
    ("Volume_quadrature_order", "2"),
    ("Use_cut_metadata_scale", "false"),
    ("Cut_cell_pressure_gradient_penalty", "1"),
    ("Cut_cell_pressure_stabilization_policy", "Enabled"),
)
HIGH_ORDER_MMS_PRESCRIBED_XML_CONTROLS = HIGH_ORDER_MMS_BASE_XML_CONTROLS + (
    ("Level_set_source", "prescribed_data"),
    ("Velocity_source", "prescribed_data"),
    ("Velocity_field_name", "LevelSetAdvectionVelocity"),
    ("Auto_register_velocity_field", "true"),
    ("Use_wet_extension_advection_velocity", "true"),
    ("Advection_velocity_from_field", "Velocity"),
)
HIGH_ORDER_MMS_PRESCRIBED_EXACT_INFLOW_XML_CONTROLS = (
    HIGH_ORDER_MMS_PRESCRIBED_XML_CONTROLS
    + (
        ("Temporal_and_spatial_values_file_path", "bc/wall_left_phi.dat"),
        ("Temporal_and_spatial_values_file_path", "bc/wall_right_phi.dat"),
    )
)
HIGH_ORDER_MMS_PRESCRIBED_EXACT_INFLOW_NO_REINIT_XML_CONTROLS = (
    HIGH_ORDER_MMS_PRESCRIBED_EXACT_INFLOW_XML_CONTROLS
    + (
        ("Enable_reinitialization", "false"),
    )
)
HIGH_ORDER_MMS_PRESCRIBED_EXACT_INFLOW_NO_REINIT_PRESSURE_ENABLED_XML_CONTROLS = (
    HIGH_ORDER_MMS_BASE_PRESSURE_ENABLED_XML_CONTROLS
    + (
        ("Level_set_source", "prescribed_data"),
        ("Velocity_source", "prescribed_data"),
        ("Velocity_field_name", "LevelSetAdvectionVelocity"),
        ("Auto_register_velocity_field", "true"),
        ("Use_wet_extension_advection_velocity", "true"),
        ("Advection_velocity_from_field", "Velocity"),
        ("Temporal_and_spatial_values_file_path", "bc/wall_left_phi.dat"),
        ("Temporal_and_spatial_values_file_path", "bc/wall_right_phi.dat"),
        ("Enable_reinitialization", "false"),
    )
)
HIGH_ORDER_MMS_CONSTANT_XML_CONTROLS = HIGH_ORDER_MMS_BASE_XML_CONTROLS + (
    ("Level_set_source", "prescribed_data"),
    ("Velocity_source", "constant"),
    ("Constant_velocity", "0.1 0 0"),
)
HIGH_ORDER_MMS_CONSTANT_NO_SUPG_XML_CONTROLS = HIGH_ORDER_MMS_CONSTANT_XML_CONTROLS + (
    ("Transport_form", "advective"),
    ("Enable_SUPG", "false"),
)
HIGH_ORDER_MMS_CONSTANT_CONSERVATIVE_XML_CONTROLS = HIGH_ORDER_MMS_CONSTANT_XML_CONTROLS + (
    ("Transport_form", "conservative_divergence"),
    ("Enable_SUPG", "true"),
)
HIGH_ORDER_MMS_LEVEL_SET_ONLY_CONSTANT_XML_CONTROLS = (
    ("Operator_tag", "equations"),
    ("Level_set_source", "prescribed_data"),
    ("Velocity_source", "constant"),
    ("Constant_velocity", "0.1 0 0"),
    ("Transport_form", "advective"),
    ("Enable_SUPG", "true"),
    ("SUPG_tau_scale", "0.5"),
)
HIGH_ORDER_MMS_LEVEL_SET_ONLY_CONSTANT_NO_SUPG_XML_CONTROLS = (
    ("Operator_tag", "equations"),
    ("Level_set_source", "prescribed_data"),
    ("Velocity_source", "constant"),
    ("Constant_velocity", "0.1 0 0"),
    ("Transport_form", "advective"),
    ("Enable_SUPG", "false"),
    ("SUPG_tau_scale", "0.5"),
)
HIGH_ORDER_MMS_LEVEL_SET_ONLY_CONSTANT_NO_SUPG_EXACT_INFLOW_XML_CONTROLS = (
    HIGH_ORDER_MMS_LEVEL_SET_ONLY_CONSTANT_NO_SUPG_XML_CONTROLS
    + (
        ("Temporal_and_spatial_values_file_path", "bc/wall_left_phi.dat"),
        ("Temporal_and_spatial_values_file_path", "bc/wall_right_phi.dat"),
    )
)
HIGH_ORDER_MMS_LEVEL_SET_ONLY_STATIC_XML_CONTROLS = (
    ("Operator_tag", "equations"),
    ("Level_set_source", "prescribed_data"),
    ("Velocity_source", "constant"),
    ("Constant_velocity", "0 0 0"),
    ("Transport_form", "advective"),
    ("Enable_SUPG", "true"),
    ("SUPG_tau_scale", "0.5"),
)
HIGH_ORDER_MMS_LEVEL_SET_ONLY_CONSTANT_RHO0_XML_CONTROLS = (
    HIGH_ORDER_MMS_LEVEL_SET_ONLY_CONSTANT_XML_CONTROLS
    + (("Spectral_radius_of_infinite_time_step", "0"),)
)
MMS_LEVEL_SET_ONLY_P1_CONSTANT_XML_CONTROLS = (
    HIGH_ORDER_MMS_LEVEL_SET_ONLY_CONSTANT_XML_CONTROLS
    + (("Element_order", "1"),)
)
HIGH_ORDER_MMS_CONSTANT_EXACT_INFLOW_XML_CONTROLS = HIGH_ORDER_MMS_CONSTANT_XML_CONTROLS + (
    ("Temporal_and_spatial_values_file_path", "bc/wall_left_phi.dat"),
    ("Temporal_and_spatial_values_file_path", "bc/wall_right_phi.dat"),
)
MMS_REFINEMENT_TREND_PROBES = {
    "linear_geometry": {
        2: "mms-linear-nx2-history-10step",
        3: "mms-linear-nx3-history-10step",
        4: "mms-linear-compact-history-10step",
    },
    "high_order_constant_level_set_velocity": {
        2: "mms-constant-lsvel-depth6-nx2-history-10step",
        3: "mms-constant-lsvel-depth6-nx3-history-10step",
        4: "mms-constant-lsvel-depth6-history-10step",
    },
    "high_order_constant_translation": {
        2: "mms-constant-translation-depth6-nx2-history-10step",
        3: "mms-constant-translation-depth6-nx3-history-10step",
        4: "mms-constant-translation-depth6-nx4-history-10step",
    },
    "high_order_constant_translation_dt001_t005": {
        2: "mms-constant-translation-depth6-nx2-dt001-t005",
        3: "mms-constant-translation-depth6-nx3-dt001-t005",
        4: "mms-constant-translation-depth6-nx4-dt001-t005",
    },
    "high_order_constant_translation_no_supg_dt001_t005": {
        2: "mms-constant-translation-nosupg-depth6-nx2-dt001-t005",
        3: "mms-constant-translation-nosupg-depth6-nx3-dt001-t005",
        4: "mms-constant-translation-nosupg-depth6-nx4-dt001-t005",
    },
    "high_order_constant_translation_conservative_dt001_t005": {
        2: "mms-constant-translation-conservative-depth6-nx2-dt001-t005",
        3: "mms-constant-translation-conservative-depth6-nx3-dt001-t005",
        4: "mms-constant-translation-conservative-depth6-nx4-dt001-t005",
    },
    "level_set_only_constant_translation_dt001_t005": {
        2: "mms-ls-only-constant-translation-nx2-dt001-t005",
        3: "mms-ls-only-constant-translation-nx3-dt001-t005",
        4: "mms-ls-only-constant-translation-nx4-dt001-t005",
    },
    "level_set_only_p2_constant_translation_no_supg_dt001_t005": {
        2: "mms-ls-only-p2-constant-translation-nosupg-nx2-dt001-t005",
        3: "mms-ls-only-p2-constant-translation-nosupg-nx3-dt001-t005",
        4: "mms-ls-only-p2-constant-translation-nosupg-nx4-dt001-t005",
    },
    "level_set_only_p2_constant_translation_no_supg_extended_dt001_t005": {
        2: "mms-ls-only-p2-constant-translation-nosupg-nx2-dt001-t005",
        3: "mms-ls-only-p2-constant-translation-nosupg-nx3-dt001-t005",
        4: "mms-ls-only-p2-constant-translation-nosupg-nx4-dt001-t005",
        5: "mms-ls-only-p2-constant-translation-nosupg-nx5-dt001-t005",
        6: "mms-ls-only-p2-constant-translation-nosupg-nx6-dt001-t005",
    },
    "level_set_only_p2_constant_translation_no_supg_asymptotic_dt001_t005": {
        6: "mms-ls-only-p2-constant-translation-nosupg-nx6-dt001-t005",
        8: "mms-ls-only-p2-constant-translation-nosupg-nx8-dt001-t005",
        10: "mms-ls-only-p2-constant-translation-nosupg-nx10-dt001-t005",
        12: "mms-ls-only-p2-constant-translation-nosupg-nx12-dt001-t005",
    },
    "level_set_only_p2_flat_horizontal_no_supg_dt001_t005": {
        2: "mms-ls-only-p2-flat-horizontal-nosupg-nx2-dt001-t005",
        3: "mms-ls-only-p2-flat-horizontal-nosupg-nx3-dt001-t005",
        4: "mms-ls-only-p2-flat-horizontal-nosupg-nx4-dt001-t005",
    },
    "level_set_only_p2_flat_horizontal_no_supg_asymptotic_dt001_t005": {
        6: "mms-ls-only-p2-flat-horizontal-nosupg-nx6-dt001-t005",
        8: "mms-ls-only-p2-flat-horizontal-nosupg-nx8-dt001-t005",
        10: "mms-ls-only-p2-flat-horizontal-nosupg-nx10-dt001-t005",
        12: "mms-ls-only-p2-flat-horizontal-nosupg-nx12-dt001-t005",
    },
    "level_set_only_p2_constant_translation_no_supg_exact_inflow_dt001_t005": {
        2: "mms-ls-only-p2-constant-translation-nosupg-inflow-nx2-dt001-t005",
        3: "mms-ls-only-p2-constant-translation-nosupg-inflow-nx3-dt001-t005",
        4: "mms-ls-only-p2-constant-translation-nosupg-inflow-nx4-dt001-t005",
    },
    "level_set_only_p2_constant_translation_no_supg_exact_inflow_asymptotic_dt001_t005": {
        6: "mms-ls-only-p2-constant-translation-nosupg-inflow-nx6-dt001-t005",
        8: "mms-ls-only-p2-constant-translation-nosupg-inflow-nx8-dt001-t005",
        10: "mms-ls-only-p2-constant-translation-nosupg-inflow-nx10-dt001-t005",
        12: "mms-ls-only-p2-constant-translation-nosupg-inflow-nx12-dt001-t005",
    },
    "level_set_only_p2_constant_translation_onestep_dt001_t001": {
        2: "mms-ls-only-p2-constant-translation-onestep-nx2-dt001-t001",
        3: "mms-ls-only-p2-constant-translation-onestep-nx3-dt001-t001",
        4: "mms-ls-only-p2-constant-translation-onestep-nx4-dt001-t001",
    },
    "level_set_only_p2_constant_translation_no_supg_onestep_dt001_t001": {
        2: "mms-ls-only-p2-constant-translation-nosupg-onestep-nx2-dt001-t001",
        3: "mms-ls-only-p2-constant-translation-nosupg-onestep-nx3-dt001-t001",
        4: "mms-ls-only-p2-constant-translation-nosupg-onestep-nx4-dt001-t001",
    },
    "level_set_only_static_p2_dt001_t001": {
        2: "mms-ls-only-static-p2-nx2-dt001-t001",
        3: "mms-ls-only-static-p2-nx3-dt001-t001",
        4: "mms-ls-only-static-p2-nx4-dt001-t001",
    },
    "level_set_only_flat_static_p2_dt001_t001": {
        2: "mms-ls-only-flat-static-p2-nx2-dt001-t001",
        3: "mms-ls-only-flat-static-p2-nx3-dt001-t001",
        4: "mms-ls-only-flat-static-p2-nx4-dt001-t001",
    },
    "level_set_only_p1_constant_translation_dt001_t005": {
        2: "mms-ls-only-p1-constant-translation-nx2-dt001-t005",
        3: "mms-ls-only-p1-constant-translation-nx3-dt001-t005",
        4: "mms-ls-only-p1-constant-translation-nx4-dt001-t005",
    },
    "level_set_only_constant_translation_rho0_dt001_t005": {
        2: "mms-ls-only-constant-translation-rho0-nx2-dt001-t005",
        3: "mms-ls-only-constant-translation-rho0-nx3-dt001-t005",
        4: "mms-ls-only-constant-translation-rho0-nx4-dt001-t005",
    },
    "high_order_constant_translation_exact_inflow": {
        2: "mms-constant-translation-inflow-depth6-nx2-history-10step",
        3: "mms-constant-translation-inflow-depth6-nx3-history-10step",
        4: "mms-constant-translation-inflow-depth6-nx4-history-10step",
    },
    "high_order_prescribed_exact_inflow_no_reinit_pressure_stab": {
        2: "mms-compact-exact-inflow-no-reinit-pressure-stab-nx2-10step",
        3: "mms-compact-exact-inflow-no-reinit-pressure-stab-nx3-10step",
        4: "mms-compact-exact-inflow-no-reinit-pressure-stab-10step",
    },
}
MMS_TEMPORAL_TREND_PROBES = {
    "level_set_only_p2_constant_translation_no_supg_nx4_t005": {
        5: "mms-ls-only-p2-constant-translation-nosupg-nx4-dt001-t005",
        10: "mms-ls-only-p2-constant-translation-nosupg-nx4-dt0005-t005",
        20: "mms-ls-only-p2-constant-translation-nosupg-nx4-dt00025-t005",
    },
}
MMS_REFINEMENT_TREND_METRICS = (
    "phi_l2_error",
    "quadrature_phi_l2_error",
    "quadrature_phi_grad_l2_error",
    "quadrature_phi_grad_x_l2_error",
    "quadrature_phi_grad_y_l2_error",
    "quadrature_level_set_spatial_residual_l2_error",
    "interior_phi_l2_error",
    "quadrature_interior_phi_l2_error",
    "interior_quadrature_phi_grad_l2_error",
    "interior_quadrature_phi_grad_x_l2_error",
    "interior_quadrature_phi_grad_y_l2_error",
    "interior_quadrature_level_set_spatial_residual_l2_error",
    "quadrature_implied_interface_height_l2_error",
    "quadrature_implied_interface_shift_error",
    "interior_quadrature_implied_interface_height_l2_error",
    "interior_quadrature_implied_interface_shift_error",
    "interface_shift_error",
    "area_relative_error",
    "velocity_relative_l2_error",
    "quadrature_velocity_relative_l2_error",
    "bulk_velocity_relative_l2_error",
    "bulk_quadrature_velocity_relative_l2_error",
    "pressure_relative_rms_error",
    "pressure_relative_rms_error_after_constant_offset_removal",
    "quadrature_pressure_relative_rms_error",
    "quadrature_pressure_relative_rms_error_after_constant_offset_removal",
    "bulk_pressure_relative_rms_error",
    "bulk_pressure_relative_rms_error_after_constant_offset_removal",
    "bulk_quadrature_pressure_relative_rms_error",
    "bulk_quadrature_pressure_relative_rms_error_after_constant_offset_removal",
)
MMS_REFINEMENT_PRIMARY_GATE_METRICS = (
    "quadrature_phi_l2_error",
    "quadrature_phi_grad_l2_error",
    "quadrature_phi_grad_x_l2_error",
    "quadrature_level_set_spatial_residual_l2_error",
    "quadrature_interior_phi_l2_error",
    "interior_quadrature_phi_grad_l2_error",
    "interior_quadrature_phi_grad_x_l2_error",
    "interior_quadrature_level_set_spatial_residual_l2_error",
    "quadrature_implied_interface_height_l2_error",
    "quadrature_implied_interface_shift_error",
    "interior_quadrature_implied_interface_height_l2_error",
    "interior_quadrature_implied_interface_shift_error",
    "interface_shift_error",
    "area_relative_error",
    "pressure_relative_rms_error",
    "pressure_relative_rms_error_after_constant_offset_removal",
    "bulk_pressure_relative_rms_error",
    "bulk_pressure_relative_rms_error_after_constant_offset_removal",
)
MMS_REFINEMENT_TREND_ZERO_TOL = 1.0e-12
DEFAULT_MPI_POLICY = "serial_direct_only_no_mpi_parity_claim"
DEFAULT_MPI_RANKS = 1
MPI2_POLICY = "mpi2_validation"
MPI4_POLICY = "mpi4_validation"
LATEST_RESULT_NAME = "latest"


@dataclass(frozen=True)
class Probe:
    name: str
    case_subdir: str
    generate_args: tuple[str, ...]
    result_name: str | None
    expect_solver_success: bool
    expect_verifier_pass: bool | None
    description: str
    solver_env: tuple[tuple[str, str], ...] = ()
    verify_history: bool = False
    required_solver_features: tuple[str, ...] = ()
    required_solver_xml_controls: tuple[tuple[str, str], ...] = ()
    validation_scope: str = "regression_gate"
    timeout_policy: str = "fail_validation"
    mpi_policy: str = DEFAULT_MPI_POLICY
    mpi_ranks: int = DEFAULT_MPI_RANKS


def mms_ls_only_p2_flat_horizontal_no_supg_probe(nx: int) -> Probe:
    name = f"mms-ls-only-p2-flat-horizontal-nosupg-nx{nx}-dt001-t005"
    return Probe(
        name=name,
        case_subdir="mms_traveling_interface_2d",
        generate_args=(
            "--nx", str(nx),
            "--ny", str(nx),
            "--time-steps", "5",
            "--time-step", "0.001",
            "--output-cadence", "5",
            "--amplitude", "0.0",
            "--omega", "0.0",
            "--level-set-only",
            "--level-set-velocity-source", "constant",
            "--level-set-constant-velocity", "0.1", "0.0", "0.0",
            "--disable-level-set-supg",
        ),
        result_name="result_005.vtu",
        expect_solver_success=True,
        expect_verifier_pass=None,
        description=(
            "P2 level-set-only flat-interface horizontal-advection invariant "
            f"diagnostic on an nx={nx} mesh with SUPG disabled. The exact "
            "solution has phi_x=0 and grad_y=1, so this isolates transport "
            "operator drift of the y-linear null mode."
        ),
        verify_history=True,
        required_solver_xml_controls=HIGH_ORDER_MMS_LEVEL_SET_ONLY_CONSTANT_NO_SUPG_XML_CONTROLS,
        validation_scope="trend_diagnostic",
    )


def mms_ls_only_flat_static_p2_probe(nx: int) -> Probe:
    name = f"mms-ls-only-flat-static-p2-nx{nx}-dt001-t001"
    return Probe(
        name=name,
        case_subdir="mms_traveling_interface_2d",
        generate_args=(
            "--nx", str(nx),
            "--ny", str(nx),
            "--time-steps", "1",
            "--time-step", "0.001",
            "--output-cadence", "1",
            "--amplitude", "0.0",
            "--omega", "0.0",
            "--u0", "0.0",
            "--level-set-only",
            "--level-set-velocity-source", "constant",
            "--level-set-constant-velocity", "0.0", "0.0", "0.0",
        ),
        result_name="result_001.vtu",
        expect_solver_success=True,
        expect_verifier_pass=None,
        description=(
            "P2 level-set-only flat-interface zero-advection control on an "
            f"nx={nx} mesh. This checks amplitude-zero initialization and "
            "state advancement before attributing flat-interface drift to "
            "horizontal advection."
        ),
        verify_history=True,
        required_solver_xml_controls=HIGH_ORDER_MMS_LEVEL_SET_ONLY_STATIC_XML_CONTROLS,
        validation_scope="trend_diagnostic",
    )


PROBES: dict[str, Probe] = {
    "mms-one-step-nx2": Probe(
        name="mms-one-step-nx2",
        case_subdir="mms_traveling_interface_2d",
        generate_args=("--nx", "2", "--ny", "2", "--time-steps", "1", "--time-step", "0.005", "--output-cadence", "1"),
        result_name="result_001.vtu",
        expect_solver_success=True,
        expect_verifier_pass=True,
        description="Compact Taylor-Hood MMS one-step smoke expected to pass the verifier.",
        required_solver_features=HIGH_ORDER_MMS_REQUIRED_SOLVER_FEATURES,
        required_solver_xml_controls=HIGH_ORDER_MMS_PRESCRIBED_XML_CONTROLS,
        validation_scope="regression_gate",
    ),
    "mms-compact-10step": Probe(
        name="mms-compact-10step",
        case_subdir="mms_traveling_interface_2d",
        generate_args=("--nx", "4", "--ny", "4", "--time-steps", "10", "--time-step", "0.005", "--output-cadence", "10"),
        result_name="result_010.vtu",
        expect_solver_success=True,
        expect_verifier_pass=False,
        description=(
            "Compact moving-interface transient diagnostic. Current evidence "
            "expects this case to reach final time with fallback-free high-order "
            "cut contexts but fail the full MMS accuracy verifier, exposing the "
            "open moving-cut transient accuracy blocker."
        ),
        required_solver_features=HIGH_ORDER_MMS_REQUIRED_SOLVER_FEATURES,
        required_solver_xml_controls=HIGH_ORDER_MMS_PRESCRIBED_XML_CONTROLS,
        validation_scope="expected_failure_diagnostic",
    ),
    "mms-compact-exact-inflow-10step": Probe(
        name="mms-compact-exact-inflow-10step",
        case_subdir="mms_traveling_interface_2d",
        generate_args=(
            "--nx",
            "4",
            "--ny",
            "4",
            "--time-steps",
            "10",
            "--time-step",
            "0.005",
            "--output-cadence",
            "10",
            "--level-set-exact-inflow",
        ),
        result_name="result_010.vtu",
        expect_solver_success=True,
        expect_verifier_pass=False,
        description=(
            "Compact moving-interface transient diagnostic with exact side-wall "
            "level-set inflow data. Current evidence expects this case to reach "
            "final time but fail the full MMS accuracy verifier, confirming that "
            "the 10-step prescribed moving-interface accuracy blocker is not "
            "caused by the no-inflow XML fallback alone."
        ),
        required_solver_features=HIGH_ORDER_MMS_REQUIRED_SOLVER_FEATURES,
        required_solver_xml_controls=HIGH_ORDER_MMS_PRESCRIBED_EXACT_INFLOW_XML_CONTROLS,
        validation_scope="expected_failure_diagnostic",
    ),
    "mms-compact-exact-inflow-history-10step": Probe(
        name="mms-compact-exact-inflow-history-10step",
        case_subdir="mms_traveling_interface_2d",
        generate_args=(
            "--nx",
            "4",
            "--ny",
            "4",
            "--time-steps",
            "10",
            "--time-step",
            "0.005",
            "--output-cadence",
            "1",
            "--level-set-exact-inflow",
        ),
        result_name="result_010.vtu",
        expect_solver_success=True,
        expect_verifier_pass=False,
        description=(
            "Every-step companion for the exact-inflow compact moving-interface "
            "diagnostic. This records the first saved output where the full MMS "
            "accuracy verifier fails after the bounded 4-step regression gate."
        ),
        verify_history=True,
        required_solver_features=HIGH_ORDER_MMS_REQUIRED_SOLVER_FEATURES,
        required_solver_xml_controls=HIGH_ORDER_MMS_PRESCRIBED_EXACT_INFLOW_XML_CONTROLS,
        validation_scope="expected_failure_diagnostic",
    ),
    "mms-compact-exact-inflow-no-reinit-history-10step": Probe(
        name="mms-compact-exact-inflow-no-reinit-history-10step",
        case_subdir="mms_traveling_interface_2d",
        generate_args=(
            "--nx",
            "4",
            "--ny",
            "4",
            "--time-steps",
            "10",
            "--time-step",
            "0.005",
            "--output-cadence",
            "1",
            "--level-set-exact-inflow",
            "--disable-reinitialization",
        ),
        result_name="result_010.vtu",
        expect_solver_success=True,
        expect_verifier_pass=False,
        description=(
            "Every-step exact-inflow compact moving-interface diagnostic with "
            "projection reinitialization disabled. This separates transported "
            "manufactured level-set accuracy from the level-set maintenance "
            "operation that fires on step 10 in the default compact probe."
        ),
        verify_history=True,
        required_solver_features=HIGH_ORDER_MMS_REQUIRED_SOLVER_FEATURES,
        required_solver_xml_controls=HIGH_ORDER_MMS_PRESCRIBED_EXACT_INFLOW_NO_REINIT_XML_CONTROLS,
        validation_scope="expected_failure_diagnostic",
    ),
    "mms-compact-exact-inflow-no-reinit-pressure-stab-nx2-10step": Probe(
        name="mms-compact-exact-inflow-no-reinit-pressure-stab-nx2-10step",
        case_subdir="mms_traveling_interface_2d",
        generate_args=(
            "--nx",
            "2",
            "--ny",
            "2",
            "--time-steps",
            "10",
            "--time-step",
            "0.005",
            "--output-cadence",
            "10",
            "--level-set-exact-inflow",
            "--disable-reinitialization",
            "--cut-cell-pressure-stabilization-policy",
            "Enabled",
        ),
        result_name="result_010.vtu",
        expect_solver_success=True,
        expect_verifier_pass=True,
        description=(
            "Coarse compact high-order MMS regression with exact side-wall "
            "level-set inflow, no projection reinitialization, and enabled "
            "pressure ghost stabilization."
        ),
        required_solver_features=HIGH_ORDER_MMS_REQUIRED_SOLVER_FEATURES,
        required_solver_xml_controls=HIGH_ORDER_MMS_PRESCRIBED_EXACT_INFLOW_NO_REINIT_PRESSURE_ENABLED_XML_CONTROLS,
        validation_scope="regression_gate",
    ),
    "mms-compact-exact-inflow-no-reinit-pressure-stab-nx3-10step": Probe(
        name="mms-compact-exact-inflow-no-reinit-pressure-stab-nx3-10step",
        case_subdir="mms_traveling_interface_2d",
        generate_args=(
            "--nx",
            "3",
            "--ny",
            "3",
            "--time-steps",
            "10",
            "--time-step",
            "0.005",
            "--output-cadence",
            "10",
            "--level-set-exact-inflow",
            "--disable-reinitialization",
            "--cut-cell-pressure-stabilization-policy",
            "Enabled",
        ),
        result_name="result_010.vtu",
        expect_solver_success=True,
        expect_verifier_pass=True,
        description=(
            "Intermediate compact high-order MMS regression with exact "
            "side-wall level-set inflow, no projection reinitialization, and "
            "enabled pressure ghost stabilization."
        ),
        required_solver_features=HIGH_ORDER_MMS_REQUIRED_SOLVER_FEATURES,
        required_solver_xml_controls=HIGH_ORDER_MMS_PRESCRIBED_EXACT_INFLOW_NO_REINIT_PRESSURE_ENABLED_XML_CONTROLS,
        validation_scope="regression_gate",
    ),
    "mms-compact-exact-inflow-no-reinit-pressure-stab-10step": Probe(
        name="mms-compact-exact-inflow-no-reinit-pressure-stab-10step",
        case_subdir="mms_traveling_interface_2d",
        generate_args=(
            "--nx",
            "4",
            "--ny",
            "4",
            "--time-steps",
            "10",
            "--time-step",
            "0.005",
            "--output-cadence",
            "10",
            "--level-set-exact-inflow",
            "--disable-reinitialization",
            "--cut-cell-pressure-stabilization-policy",
            "Enabled",
        ),
        result_name="result_010.vtu",
        expect_solver_success=True,
        expect_verifier_pass=True,
        description=(
            "Compact high-order MMS regression with exact side-wall level-set "
            "inflow, no projection reinitialization, and enabled pressure "
            "ghost stabilization."
        ),
        required_solver_features=HIGH_ORDER_MMS_REQUIRED_SOLVER_FEATURES,
        required_solver_xml_controls=HIGH_ORDER_MMS_PRESCRIBED_EXACT_INFLOW_NO_REINIT_PRESSURE_ENABLED_XML_CONTROLS,
        validation_scope="regression_gate",
    ),
    "mms-compact-exact-inflow-no-reinit-pressure-stab-history-10step": Probe(
        name="mms-compact-exact-inflow-no-reinit-pressure-stab-history-10step",
        case_subdir="mms_traveling_interface_2d",
        generate_args=(
            "--nx",
            "4",
            "--ny",
            "4",
            "--time-steps",
            "10",
            "--time-step",
            "0.005",
            "--output-cadence",
            "1",
            "--level-set-exact-inflow",
            "--disable-reinitialization",
            "--cut-cell-pressure-stabilization-policy",
            "Enabled",
        ),
        result_name="result_010.vtu",
        expect_solver_success=True,
        expect_verifier_pass=True,
        description=(
            "Every-step exact-inflow/no-reinitialization compact MMS pressure "
            "stabilization diagnostic. This tests whether the late pressure "
            "accuracy failure is reduced by enabling pressure ghost penalties "
            "on the moving high-order implicit interface."
        ),
        verify_history=True,
        required_solver_features=HIGH_ORDER_MMS_REQUIRED_SOLVER_FEATURES,
        required_solver_xml_controls=HIGH_ORDER_MMS_PRESCRIBED_EXACT_INFLOW_NO_REINIT_PRESSURE_ENABLED_XML_CONTROLS,
        validation_scope="regression_gate",
    ),
    "mms-prescribed-lsvel-history-4step": Probe(
        name="mms-prescribed-lsvel-history-4step",
        case_subdir="mms_traveling_interface_2d",
        generate_args=(
            "--nx", "4",
            "--ny", "4",
            "--time-steps", "4",
            "--time-step", "0.005",
            "--output-cadence", "1",
        ),
        result_name="result_004.vtu",
        expect_solver_success=True,
        expect_verifier_pass=True,
        description=(
            "Compact high-order MMS diagnostic using the default prescribed "
            "wet-extension level-set advection velocity. This bounded 4-step "
            "regression records every-step verifier history and gates the "
            "post-projection-fix moving-interface transient before the longer "
            "10-step accuracy diagnostic diverges from the MMS target."
        ),
        verify_history=True,
        required_solver_features=HIGH_ORDER_MMS_REQUIRED_SOLVER_FEATURES,
        required_solver_xml_controls=HIGH_ORDER_MMS_PRESCRIBED_XML_CONTROLS,
        validation_scope="regression_gate",
    ),
    "mms-linear-compact-10step": Probe(
        name="mms-linear-compact-10step",
        case_subdir="mms_traveling_interface_2d",
        generate_args=(
            "--nx", "4",
            "--ny", "4",
            "--element-order", "1",
            "--time-steps", "10",
            "--time-step", "0.005",
            "--output-cadence", "10",
        ),
        result_name="result_010.vtu",
        expect_solver_success=True,
        expect_verifier_pass=True,
        description=(
            "Compact linear-geometry MMS transient baseline. This is expected "
            "to pass the 10-step verifier and separates MMS source/verifier "
            "wiring from the high-order implicit moving-cut blocker."
        ),
    ),
    "mms-linear-compact-history-10step": Probe(
        name="mms-linear-compact-history-10step",
        case_subdir="mms_traveling_interface_2d",
        generate_args=(
            "--nx", "4",
            "--ny", "4",
            "--element-order", "1",
            "--time-steps", "10",
            "--time-step", "0.005",
            "--output-cadence", "1",
        ),
        result_name="result_010.vtu",
        expect_solver_success=True,
        expect_verifier_pass=True,
        description=(
            "Compact linear-geometry MMS transient baseline with every-step "
            "output verification. This records transient history metrics for "
            "comparison against high-order implicit moving-cut diagnostics."
        ),
        verify_history=True,
    ),
    "mms-linear-nx2-history-10step": Probe(
        name="mms-linear-nx2-history-10step",
        case_subdir="mms_traveling_interface_2d",
        generate_args=(
            "--nx", "2",
            "--ny", "2",
            "--element-order", "1",
            "--time-steps", "10",
            "--time-step", "0.005",
            "--output-cadence", "1",
        ),
        result_name="result_010.vtu",
        expect_solver_success=True,
        expect_verifier_pass=None,
        description=(
            "Coarser compact linear-geometry MMS history baseline for bounded "
            "refinement comparisons against high-order implicit moving-cut "
            "diagnostics. The solver is expected to reach final time, while "
            "the coarse pressure verifier outcome is recorded as trend data."
        ),
        verify_history=True,
        validation_scope="trend_diagnostic",
    ),
    "mms-linear-nx3-history-10step": Probe(
        name="mms-linear-nx3-history-10step",
        case_subdir="mms_traveling_interface_2d",
        generate_args=(
            "--nx", "3",
            "--ny", "3",
            "--element-order", "1",
            "--time-steps", "10",
            "--time-step", "0.005",
            "--output-cadence", "1",
        ),
        result_name="result_010.vtu",
        expect_solver_success=True,
        expect_verifier_pass=None,
        description=(
            "Intermediate compact linear-geometry MMS history baseline for "
            "bounded refinement comparisons against high-order implicit "
            "moving-cut diagnostics. The solver is expected to reach final "
            "time, while the coarse pressure verifier outcome is recorded as "
            "trend data."
        ),
        verify_history=True,
        validation_scope="trend_diagnostic",
    ),
    "mms-constant-lsvel-depth6-10step": Probe(
        name="mms-constant-lsvel-depth6-10step",
        case_subdir="mms_traveling_interface_2d",
        generate_args=(
            "--nx", "4",
            "--ny", "4",
            "--time-steps", "10",
            "--time-step", "0.005",
            "--output-cadence", "10",
            "--level-set-velocity-source", "constant",
            "--level-set-constant-velocity", "0.1", "0.0", "0.0",
            "--implicit-cut-max-subdivision-depth", "6",
        ),
        result_name="result_010.vtu",
        expect_solver_success=True,
        expect_verifier_pass=False,
        description=(
            "Compact high-order MMS diagnostic with prescribed constant level-set "
            "advection velocity. Depth 6 is expected to reach final time after "
            "terminal multi-crossing topology refinement, but still fail the "
            "full MMS accuracy verifier."
        ),
        required_solver_features=HIGH_ORDER_MMS_REQUIRED_SOLVER_FEATURES,
        required_solver_xml_controls=HIGH_ORDER_MMS_CONSTANT_XML_CONTROLS,
        validation_scope="expected_failure_diagnostic",
    ),
    "mms-constant-lsvel-depth6-history-10step": Probe(
        name="mms-constant-lsvel-depth6-history-10step",
        case_subdir="mms_traveling_interface_2d",
        generate_args=(
            "--nx", "4",
            "--ny", "4",
            "--time-steps", "10",
            "--time-step", "0.005",
            "--output-cadence", "1",
            "--level-set-velocity-source", "constant",
            "--level-set-constant-velocity", "0.1", "0.0", "0.0",
            "--implicit-cut-max-subdivision-depth", "6",
        ),
        result_name="result_010.vtu",
        expect_solver_success=True,
        expect_verifier_pass=False,
        description=(
            "Compact high-order MMS diagnostic with prescribed constant level-set "
            "advection velocity and every-step output verification. The solver "
            "is expected to reach final time fallback-free, while the verifier "
            "history documents the unresolved transient accuracy error model."
        ),
        verify_history=True,
        required_solver_features=HIGH_ORDER_MMS_REQUIRED_SOLVER_FEATURES,
        required_solver_xml_controls=HIGH_ORDER_MMS_CONSTANT_XML_CONTROLS,
        validation_scope="expected_failure_diagnostic",
    ),
    "mms-constant-lsvel-depth6-nx2-history-10step": Probe(
        name="mms-constant-lsvel-depth6-nx2-history-10step",
        case_subdir="mms_traveling_interface_2d",
        generate_args=(
            "--nx", "2",
            "--ny", "2",
            "--time-steps", "10",
            "--time-step", "0.005",
            "--output-cadence", "1",
            "--level-set-velocity-source", "constant",
            "--level-set-constant-velocity", "0.1", "0.0", "0.0",
            "--implicit-cut-max-subdivision-depth", "6",
        ),
        result_name="result_010.vtu",
        expect_solver_success=True,
        expect_verifier_pass=None,
        description=(
            "Coarser compact high-order MMS history diagnostic with prescribed "
            "constant level-set advection velocity. This records bounded "
            "refinement trend data without pre-classifying the verifier outcome."
        ),
        verify_history=True,
        required_solver_features=HIGH_ORDER_MMS_REQUIRED_SOLVER_FEATURES,
        required_solver_xml_controls=HIGH_ORDER_MMS_CONSTANT_XML_CONTROLS,
        validation_scope="expected_failure_diagnostic",
    ),
    "mms-constant-lsvel-depth6-nx3-history-10step": Probe(
        name="mms-constant-lsvel-depth6-nx3-history-10step",
        case_subdir="mms_traveling_interface_2d",
        generate_args=(
            "--nx", "3",
            "--ny", "3",
            "--time-steps", "10",
            "--time-step", "0.005",
            "--output-cadence", "1",
            "--level-set-velocity-source", "constant",
            "--level-set-constant-velocity", "0.1", "0.0", "0.0",
            "--implicit-cut-max-subdivision-depth", "6",
        ),
        result_name="result_010.vtu",
        expect_solver_success=True,
        expect_verifier_pass=None,
        description=(
            "Intermediate compact high-order MMS history diagnostic with prescribed "
            "constant level-set advection velocity. This records bounded "
            "refinement trend data without pre-classifying the verifier outcome."
        ),
        verify_history=True,
        required_solver_features=HIGH_ORDER_MMS_REQUIRED_SOLVER_FEATURES,
        required_solver_xml_controls=HIGH_ORDER_MMS_CONSTANT_XML_CONTROLS,
        validation_scope="expected_failure_diagnostic",
    ),
    "mms-constant-translation-depth6-nx2-history-10step": Probe(
        name="mms-constant-translation-depth6-nx2-history-10step",
        case_subdir="mms_traveling_interface_2d",
        generate_args=(
            "--nx", "2",
            "--ny", "2",
            "--time-steps", "10",
            "--time-step", "0.005",
            "--output-cadence", "1",
            "--omega", "0.0",
            "--level-set-velocity-source", "constant",
            "--level-set-constant-velocity", "0.1", "0.0", "0.0",
            "--implicit-cut-max-subdivision-depth", "6",
        ),
        result_name="result_010.vtu",
        expect_solver_success=True,
        expect_verifier_pass=None,
        description=(
            "Coarser compact high-order MMS history diagnostic with exact "
            "constant-translation level-set advection velocity. This records "
            "bounded refinement trend data for the Omega=0 manufactured solution."
        ),
        verify_history=True,
        required_solver_features=HIGH_ORDER_MMS_REQUIRED_SOLVER_FEATURES,
        required_solver_xml_controls=HIGH_ORDER_MMS_CONSTANT_XML_CONTROLS,
        validation_scope="trend_diagnostic",
    ),
    "mms-constant-translation-depth6-nx3-history-10step": Probe(
        name="mms-constant-translation-depth6-nx3-history-10step",
        case_subdir="mms_traveling_interface_2d",
        generate_args=(
            "--nx", "3",
            "--ny", "3",
            "--time-steps", "10",
            "--time-step", "0.005",
            "--output-cadence", "1",
            "--omega", "0.0",
            "--level-set-velocity-source", "constant",
            "--level-set-constant-velocity", "0.1", "0.0", "0.0",
            "--implicit-cut-max-subdivision-depth", "6",
        ),
        result_name="result_010.vtu",
        expect_solver_success=True,
        expect_verifier_pass=None,
        description=(
            "Intermediate compact high-order MMS history diagnostic with exact "
            "constant-translation level-set advection velocity. This records "
            "bounded refinement trend data for the Omega=0 manufactured solution."
        ),
        verify_history=True,
        required_solver_features=HIGH_ORDER_MMS_REQUIRED_SOLVER_FEATURES,
        required_solver_xml_controls=HIGH_ORDER_MMS_CONSTANT_XML_CONTROLS,
        validation_scope="trend_diagnostic",
    ),
    "mms-constant-translation-depth6-nx4-history-10step": Probe(
        name="mms-constant-translation-depth6-nx4-history-10step",
        case_subdir="mms_traveling_interface_2d",
        generate_args=(
            "--nx", "4",
            "--ny", "4",
            "--time-steps", "10",
            "--time-step", "0.005",
            "--output-cadence", "1",
            "--omega", "0.0",
            "--level-set-velocity-source", "constant",
            "--level-set-constant-velocity", "0.1", "0.0", "0.0",
            "--implicit-cut-max-subdivision-depth", "6",
        ),
        result_name="result_010.vtu",
        expect_solver_success=True,
        expect_verifier_pass=None,
        description=(
            "Compact high-order MMS history diagnostic with exact "
            "constant-translation level-set advection velocity. This records "
            "bounded refinement trend data for the Omega=0 manufactured solution."
        ),
        verify_history=True,
        required_solver_features=HIGH_ORDER_MMS_REQUIRED_SOLVER_FEATURES,
        required_solver_xml_controls=HIGH_ORDER_MMS_CONSTANT_XML_CONTROLS,
        validation_scope="trend_diagnostic",
    ),
    "mms-constant-translation-depth6-nx2-dt001-t005": Probe(
        name="mms-constant-translation-depth6-nx2-dt001-t005",
        case_subdir="mms_traveling_interface_2d",
        generate_args=(
            "--nx", "2",
            "--ny", "2",
            "--time-steps", "5",
            "--time-step", "0.001",
            "--output-cadence", "5",
            "--omega", "0.0",
            "--level-set-velocity-source", "constant",
            "--level-set-constant-velocity", "0.1", "0.0", "0.0",
            "--implicit-cut-max-subdivision-depth", "6",
        ),
        result_name="result_005.vtu",
        expect_solver_success=True,
        expect_verifier_pass=None,
        description=(
            "Coarser high-order constant-translation MMS time-step isolation "
            "probe. It reaches the same physical time as the first saved "
            "10-step history output using five smaller dt=0.001 steps, so the "
            "refinement trend can distinguish first-output temporal error from "
            "spatial high-order moving-cut behavior."
        ),
        verify_history=True,
        required_solver_features=HIGH_ORDER_MMS_REQUIRED_SOLVER_FEATURES,
        required_solver_xml_controls=HIGH_ORDER_MMS_CONSTANT_XML_CONTROLS,
        validation_scope="trend_diagnostic",
    ),
    "mms-constant-translation-depth6-nx3-dt001-t005": Probe(
        name="mms-constant-translation-depth6-nx3-dt001-t005",
        case_subdir="mms_traveling_interface_2d",
        generate_args=(
            "--nx", "3",
            "--ny", "3",
            "--time-steps", "5",
            "--time-step", "0.001",
            "--output-cadence", "5",
            "--omega", "0.0",
            "--level-set-velocity-source", "constant",
            "--level-set-constant-velocity", "0.1", "0.0", "0.0",
            "--implicit-cut-max-subdivision-depth", "6",
        ),
        result_name="result_005.vtu",
        expect_solver_success=True,
        expect_verifier_pass=None,
        description=(
            "Intermediate high-order constant-translation MMS time-step "
            "isolation probe using five dt=0.001 steps to t=0.005."
        ),
        verify_history=True,
        required_solver_features=HIGH_ORDER_MMS_REQUIRED_SOLVER_FEATURES,
        required_solver_xml_controls=HIGH_ORDER_MMS_CONSTANT_XML_CONTROLS,
        validation_scope="trend_diagnostic",
    ),
    "mms-constant-translation-depth6-nx4-dt001-t005": Probe(
        name="mms-constant-translation-depth6-nx4-dt001-t005",
        case_subdir="mms_traveling_interface_2d",
        generate_args=(
            "--nx", "4",
            "--ny", "4",
            "--time-steps", "5",
            "--time-step", "0.001",
            "--output-cadence", "5",
            "--omega", "0.0",
            "--level-set-velocity-source", "constant",
            "--level-set-constant-velocity", "0.1", "0.0", "0.0",
            "--implicit-cut-max-subdivision-depth", "6",
        ),
        result_name="result_005.vtu",
        expect_solver_success=True,
        expect_verifier_pass=None,
        description=(
            "Compact high-order constant-translation MMS time-step isolation "
            "probe using five dt=0.001 steps to t=0.005."
        ),
        verify_history=True,
        required_solver_features=HIGH_ORDER_MMS_REQUIRED_SOLVER_FEATURES,
        required_solver_xml_controls=HIGH_ORDER_MMS_CONSTANT_XML_CONTROLS,
        validation_scope="trend_diagnostic",
    ),
    "mms-ls-only-constant-translation-nx2-dt001-t005": Probe(
        name="mms-ls-only-constant-translation-nx2-dt001-t005",
        case_subdir="mms_traveling_interface_2d",
        generate_args=(
            "--nx", "2",
            "--ny", "2",
            "--time-steps", "5",
            "--time-step", "0.001",
            "--output-cadence", "5",
            "--omega", "0.0",
            "--level-set-only",
            "--level-set-velocity-source", "constant",
            "--level-set-constant-velocity", "0.1", "0.0", "0.0",
        ),
        result_name="result_005.vtu",
        expect_solver_success=True,
        expect_verifier_pass=None,
        description=(
            "Coarser level-set-only exact-translation transport diagnostic. "
            "This removes the fluid/free-surface block so the first-output "
            "phi/interface trend can be compared against the coupled moving-cut "
            "W2 evidence."
        ),
        verify_history=True,
        required_solver_xml_controls=HIGH_ORDER_MMS_LEVEL_SET_ONLY_CONSTANT_XML_CONTROLS,
        validation_scope="trend_diagnostic",
    ),
    "mms-ls-only-constant-translation-nx3-dt001-t005": Probe(
        name="mms-ls-only-constant-translation-nx3-dt001-t005",
        case_subdir="mms_traveling_interface_2d",
        generate_args=(
            "--nx", "3",
            "--ny", "3",
            "--time-steps", "5",
            "--time-step", "0.001",
            "--output-cadence", "5",
            "--omega", "0.0",
            "--level-set-only",
            "--level-set-velocity-source", "constant",
            "--level-set-constant-velocity", "0.1", "0.0", "0.0",
        ),
        result_name="result_005.vtu",
        expect_solver_success=True,
        expect_verifier_pass=None,
        description=(
            "Intermediate level-set-only exact-translation transport diagnostic "
            "using five dt=0.001 steps to t=0.005."
        ),
        verify_history=True,
        required_solver_xml_controls=HIGH_ORDER_MMS_LEVEL_SET_ONLY_CONSTANT_XML_CONTROLS,
        validation_scope="trend_diagnostic",
    ),
    "mms-ls-only-constant-translation-nx4-dt001-t005": Probe(
        name="mms-ls-only-constant-translation-nx4-dt001-t005",
        case_subdir="mms_traveling_interface_2d",
        generate_args=(
            "--nx", "4",
            "--ny", "4",
            "--time-steps", "5",
            "--time-step", "0.001",
            "--output-cadence", "5",
            "--omega", "0.0",
            "--level-set-only",
            "--level-set-velocity-source", "constant",
            "--level-set-constant-velocity", "0.1", "0.0", "0.0",
        ),
        result_name="result_005.vtu",
        expect_solver_success=True,
        expect_verifier_pass=None,
        description=(
            "Compact level-set-only exact-translation transport diagnostic "
            "using five dt=0.001 steps to t=0.005."
        ),
        verify_history=True,
        required_solver_xml_controls=HIGH_ORDER_MMS_LEVEL_SET_ONLY_CONSTANT_XML_CONTROLS,
        validation_scope="trend_diagnostic",
    ),
    "mms-ls-only-p2-constant-translation-onestep-nx2-dt001-t001": Probe(
        name="mms-ls-only-p2-constant-translation-onestep-nx2-dt001-t001",
        case_subdir="mms_traveling_interface_2d",
        generate_args=(
            "--nx", "2",
            "--ny", "2",
            "--time-steps", "1",
            "--time-step", "0.001",
            "--output-cadence", "1",
            "--omega", "0.0",
            "--level-set-only",
            "--level-set-velocity-source", "constant",
            "--level-set-constant-velocity", "0.1", "0.0", "0.0",
        ),
        result_name="result_001.vtu",
        expect_solver_success=True,
        expect_verifier_pass=None,
        description=(
            "Coarser P2 level-set-only exact-translation one-step diagnostic. "
            "This checks whether the quadratic standalone transport trend is "
            "already incoherent after a single dt=0.001 step."
        ),
        verify_history=True,
        required_solver_xml_controls=HIGH_ORDER_MMS_LEVEL_SET_ONLY_CONSTANT_XML_CONTROLS,
        validation_scope="trend_diagnostic",
    ),
    "mms-ls-only-p2-constant-translation-onestep-nx3-dt001-t001": Probe(
        name="mms-ls-only-p2-constant-translation-onestep-nx3-dt001-t001",
        case_subdir="mms_traveling_interface_2d",
        generate_args=(
            "--nx", "3",
            "--ny", "3",
            "--time-steps", "1",
            "--time-step", "0.001",
            "--output-cadence", "1",
            "--omega", "0.0",
            "--level-set-only",
            "--level-set-velocity-source", "constant",
            "--level-set-constant-velocity", "0.1", "0.0", "0.0",
        ),
        result_name="result_001.vtu",
        expect_solver_success=True,
        expect_verifier_pass=None,
        description=(
            "Intermediate P2 level-set-only exact-translation one-step "
            "diagnostic."
        ),
        verify_history=True,
        required_solver_xml_controls=HIGH_ORDER_MMS_LEVEL_SET_ONLY_CONSTANT_XML_CONTROLS,
        validation_scope="trend_diagnostic",
    ),
    "mms-ls-only-p2-constant-translation-onestep-nx4-dt001-t001": Probe(
        name="mms-ls-only-p2-constant-translation-onestep-nx4-dt001-t001",
        case_subdir="mms_traveling_interface_2d",
        generate_args=(
            "--nx", "4",
            "--ny", "4",
            "--time-steps", "1",
            "--time-step", "0.001",
            "--output-cadence", "1",
            "--omega", "0.0",
            "--level-set-only",
            "--level-set-velocity-source", "constant",
            "--level-set-constant-velocity", "0.1", "0.0", "0.0",
        ),
        result_name="result_001.vtu",
        expect_solver_success=True,
        expect_verifier_pass=None,
        description=(
            "Compact P2 level-set-only exact-translation one-step diagnostic."
        ),
        verify_history=True,
        required_solver_xml_controls=HIGH_ORDER_MMS_LEVEL_SET_ONLY_CONSTANT_XML_CONTROLS,
        validation_scope="trend_diagnostic",
    ),
    "mms-ls-only-p2-constant-translation-nosupg-onestep-nx2-dt001-t001": Probe(
        name="mms-ls-only-p2-constant-translation-nosupg-onestep-nx2-dt001-t001",
        case_subdir="mms_traveling_interface_2d",
        generate_args=(
            "--nx", "2",
            "--ny", "2",
            "--time-steps", "1",
            "--time-step", "0.001",
            "--output-cadence", "1",
            "--omega", "0.0",
            "--level-set-only",
            "--level-set-velocity-source", "constant",
            "--level-set-constant-velocity", "0.1", "0.0", "0.0",
            "--disable-level-set-supg",
        ),
        result_name="result_001.vtu",
        expect_solver_success=True,
        expect_verifier_pass=None,
        description=(
            "Coarser P2 level-set-only exact-translation one-step diagnostic "
            "with SUPG disabled to isolate the standalone stabilization path."
        ),
        verify_history=True,
        required_solver_xml_controls=HIGH_ORDER_MMS_LEVEL_SET_ONLY_CONSTANT_NO_SUPG_XML_CONTROLS,
        validation_scope="trend_diagnostic",
    ),
    "mms-ls-only-p2-constant-translation-nosupg-onestep-nx3-dt001-t001": Probe(
        name="mms-ls-only-p2-constant-translation-nosupg-onestep-nx3-dt001-t001",
        case_subdir="mms_traveling_interface_2d",
        generate_args=(
            "--nx", "3",
            "--ny", "3",
            "--time-steps", "1",
            "--time-step", "0.001",
            "--output-cadence", "1",
            "--omega", "0.0",
            "--level-set-only",
            "--level-set-velocity-source", "constant",
            "--level-set-constant-velocity", "0.1", "0.0", "0.0",
            "--disable-level-set-supg",
        ),
        result_name="result_001.vtu",
        expect_solver_success=True,
        expect_verifier_pass=None,
        description=(
            "Intermediate P2 level-set-only exact-translation one-step "
            "diagnostic with SUPG disabled."
        ),
        verify_history=True,
        required_solver_xml_controls=HIGH_ORDER_MMS_LEVEL_SET_ONLY_CONSTANT_NO_SUPG_XML_CONTROLS,
        validation_scope="trend_diagnostic",
    ),
    "mms-ls-only-p2-constant-translation-nosupg-onestep-nx4-dt001-t001": Probe(
        name="mms-ls-only-p2-constant-translation-nosupg-onestep-nx4-dt001-t001",
        case_subdir="mms_traveling_interface_2d",
        generate_args=(
            "--nx", "4",
            "--ny", "4",
            "--time-steps", "1",
            "--time-step", "0.001",
            "--output-cadence", "1",
            "--omega", "0.0",
            "--level-set-only",
            "--level-set-velocity-source", "constant",
            "--level-set-constant-velocity", "0.1", "0.0", "0.0",
            "--disable-level-set-supg",
        ),
        result_name="result_001.vtu",
        expect_solver_success=True,
        expect_verifier_pass=None,
        description=(
            "Compact P2 level-set-only exact-translation one-step diagnostic "
            "with SUPG disabled."
        ),
        verify_history=True,
        required_solver_xml_controls=HIGH_ORDER_MMS_LEVEL_SET_ONLY_CONSTANT_NO_SUPG_XML_CONTROLS,
        validation_scope="trend_diagnostic",
    ),
    "mms-ls-only-p2-constant-translation-nosupg-nx2-dt001-t005": Probe(
        name="mms-ls-only-p2-constant-translation-nosupg-nx2-dt001-t005",
        case_subdir="mms_traveling_interface_2d",
        generate_args=(
            "--nx", "2",
            "--ny", "2",
            "--time-steps", "5",
            "--time-step", "0.001",
            "--output-cadence", "5",
            "--omega", "0.0",
            "--level-set-only",
            "--level-set-velocity-source", "constant",
            "--level-set-constant-velocity", "0.1", "0.0", "0.0",
            "--disable-level-set-supg",
        ),
        result_name="result_005.vtu",
        expect_solver_success=True,
        expect_verifier_pass=None,
        description=(
            "Coarser P2 level-set-only exact-translation five-step diagnostic "
            "with SUPG disabled."
        ),
        verify_history=True,
        required_solver_xml_controls=HIGH_ORDER_MMS_LEVEL_SET_ONLY_CONSTANT_NO_SUPG_XML_CONTROLS,
        validation_scope="trend_diagnostic",
    ),
    "mms-ls-only-p2-constant-translation-nosupg-nx3-dt001-t005": Probe(
        name="mms-ls-only-p2-constant-translation-nosupg-nx3-dt001-t005",
        case_subdir="mms_traveling_interface_2d",
        generate_args=(
            "--nx", "3",
            "--ny", "3",
            "--time-steps", "5",
            "--time-step", "0.001",
            "--output-cadence", "5",
            "--omega", "0.0",
            "--level-set-only",
            "--level-set-velocity-source", "constant",
            "--level-set-constant-velocity", "0.1", "0.0", "0.0",
            "--disable-level-set-supg",
        ),
        result_name="result_005.vtu",
        expect_solver_success=True,
        expect_verifier_pass=None,
        description=(
            "Intermediate P2 level-set-only exact-translation five-step "
            "diagnostic with SUPG disabled."
        ),
        verify_history=True,
        required_solver_xml_controls=HIGH_ORDER_MMS_LEVEL_SET_ONLY_CONSTANT_NO_SUPG_XML_CONTROLS,
        validation_scope="trend_diagnostic",
    ),
    "mms-ls-only-p2-constant-translation-nosupg-nx4-dt001-t005": Probe(
        name="mms-ls-only-p2-constant-translation-nosupg-nx4-dt001-t005",
        case_subdir="mms_traveling_interface_2d",
        generate_args=(
            "--nx", "4",
            "--ny", "4",
            "--time-steps", "5",
            "--time-step", "0.001",
            "--output-cadence", "5",
            "--omega", "0.0",
            "--level-set-only",
            "--level-set-velocity-source", "constant",
            "--level-set-constant-velocity", "0.1", "0.0", "0.0",
            "--disable-level-set-supg",
        ),
        result_name="result_005.vtu",
        expect_solver_success=True,
        expect_verifier_pass=None,
        description=(
            "Compact P2 level-set-only exact-translation five-step diagnostic "
            "with SUPG disabled."
        ),
        verify_history=True,
        required_solver_xml_controls=HIGH_ORDER_MMS_LEVEL_SET_ONLY_CONSTANT_NO_SUPG_XML_CONTROLS,
        validation_scope="trend_diagnostic",
    ),
    "mms-ls-only-p2-constant-translation-nosupg-nx4-dt0005-t005": Probe(
        name="mms-ls-only-p2-constant-translation-nosupg-nx4-dt0005-t005",
        case_subdir="mms_traveling_interface_2d",
        generate_args=(
            "--nx", "4",
            "--ny", "4",
            "--time-steps", "10",
            "--time-step", "0.0005",
            "--output-cadence", "10",
            "--omega", "0.0",
            "--level-set-only",
            "--level-set-velocity-source", "constant",
            "--level-set-constant-velocity", "0.1", "0.0", "0.0",
            "--disable-level-set-supg",
        ),
        result_name="result_010.vtu",
        expect_solver_success=True,
        expect_verifier_pass=None,
        description=(
            "Compact P2 level-set-only exact-translation temporal-refinement "
            "diagnostic at dt=0.0005 with SUPG disabled."
        ),
        verify_history=True,
        required_solver_xml_controls=HIGH_ORDER_MMS_LEVEL_SET_ONLY_CONSTANT_NO_SUPG_XML_CONTROLS,
        validation_scope="trend_diagnostic",
    ),
    "mms-ls-only-p2-constant-translation-nosupg-nx4-dt00025-t005": Probe(
        name="mms-ls-only-p2-constant-translation-nosupg-nx4-dt00025-t005",
        case_subdir="mms_traveling_interface_2d",
        generate_args=(
            "--nx", "4",
            "--ny", "4",
            "--time-steps", "20",
            "--time-step", "0.00025",
            "--output-cadence", "20",
            "--omega", "0.0",
            "--level-set-only",
            "--level-set-velocity-source", "constant",
            "--level-set-constant-velocity", "0.1", "0.0", "0.0",
            "--disable-level-set-supg",
        ),
        result_name="result_020.vtu",
        expect_solver_success=True,
        expect_verifier_pass=None,
        description=(
            "Compact P2 level-set-only exact-translation temporal-refinement "
            "diagnostic at dt=0.00025 with SUPG disabled."
        ),
        verify_history=True,
        required_solver_xml_controls=HIGH_ORDER_MMS_LEVEL_SET_ONLY_CONSTANT_NO_SUPG_XML_CONTROLS,
        validation_scope="trend_diagnostic",
    ),
    "mms-ls-only-p2-constant-translation-nosupg-nx5-dt001-t005": Probe(
        name="mms-ls-only-p2-constant-translation-nosupg-nx5-dt001-t005",
        case_subdir="mms_traveling_interface_2d",
        generate_args=(
            "--nx", "5",
            "--ny", "5",
            "--time-steps", "5",
            "--time-step", "0.001",
            "--output-cadence", "5",
            "--omega", "0.0",
            "--level-set-only",
            "--level-set-velocity-source", "constant",
            "--level-set-constant-velocity", "0.1", "0.0", "0.0",
            "--disable-level-set-supg",
        ),
        result_name="result_005.vtu",
        expect_solver_success=True,
        expect_verifier_pass=None,
        description=(
            "Finer P2 level-set-only exact-translation five-step diagnostic "
            "with SUPG disabled for the extended refinement window."
        ),
        verify_history=True,
        required_solver_xml_controls=HIGH_ORDER_MMS_LEVEL_SET_ONLY_CONSTANT_NO_SUPG_XML_CONTROLS,
        validation_scope="trend_diagnostic",
    ),
    "mms-ls-only-p2-constant-translation-nosupg-nx6-dt001-t005": Probe(
        name="mms-ls-only-p2-constant-translation-nosupg-nx6-dt001-t005",
        case_subdir="mms_traveling_interface_2d",
        generate_args=(
            "--nx", "6",
            "--ny", "6",
            "--time-steps", "5",
            "--time-step", "0.001",
            "--output-cadence", "5",
            "--omega", "0.0",
            "--level-set-only",
            "--level-set-velocity-source", "constant",
            "--level-set-constant-velocity", "0.1", "0.0", "0.0",
            "--disable-level-set-supg",
        ),
        result_name="result_005.vtu",
        expect_solver_success=True,
        expect_verifier_pass=None,
        description=(
            "Finest P2 level-set-only exact-translation five-step diagnostic "
            "with SUPG disabled for the extended refinement window."
        ),
        verify_history=True,
        required_solver_xml_controls=HIGH_ORDER_MMS_LEVEL_SET_ONLY_CONSTANT_NO_SUPG_XML_CONTROLS,
        validation_scope="trend_diagnostic",
    ),
    "mms-ls-only-p2-constant-translation-nosupg-nx8-dt001-t005": Probe(
        name="mms-ls-only-p2-constant-translation-nosupg-nx8-dt001-t005",
        case_subdir="mms_traveling_interface_2d",
        generate_args=(
            "--nx", "8",
            "--ny", "8",
            "--time-steps", "5",
            "--time-step", "0.001",
            "--output-cadence", "5",
            "--omega", "0.0",
            "--level-set-only",
            "--level-set-velocity-source", "constant",
            "--level-set-constant-velocity", "0.1", "0.0", "0.0",
            "--disable-level-set-supg",
        ),
        result_name="result_005.vtu",
        expect_solver_success=True,
        expect_verifier_pass=None,
        description=(
            "Asymptotic-window P2 level-set-only exact-translation five-step "
            "diagnostic with SUPG disabled."
        ),
        verify_history=True,
        required_solver_xml_controls=HIGH_ORDER_MMS_LEVEL_SET_ONLY_CONSTANT_NO_SUPG_XML_CONTROLS,
        validation_scope="trend_diagnostic",
    ),
    "mms-ls-only-p2-constant-translation-nosupg-nx10-dt001-t005": Probe(
        name="mms-ls-only-p2-constant-translation-nosupg-nx10-dt001-t005",
        case_subdir="mms_traveling_interface_2d",
        generate_args=(
            "--nx", "10",
            "--ny", "10",
            "--time-steps", "5",
            "--time-step", "0.001",
            "--output-cadence", "5",
            "--omega", "0.0",
            "--level-set-only",
            "--level-set-velocity-source", "constant",
            "--level-set-constant-velocity", "0.1", "0.0", "0.0",
            "--disable-level-set-supg",
        ),
        result_name="result_005.vtu",
        expect_solver_success=True,
        expect_verifier_pass=None,
        description=(
            "Higher-resolution P2 level-set-only exact-translation five-step "
            "diagnostic with SUPG disabled for the asymptotic window."
        ),
        verify_history=True,
        required_solver_xml_controls=HIGH_ORDER_MMS_LEVEL_SET_ONLY_CONSTANT_NO_SUPG_XML_CONTROLS,
        validation_scope="trend_diagnostic",
    ),
    "mms-ls-only-p2-constant-translation-nosupg-nx12-dt001-t005": Probe(
        name="mms-ls-only-p2-constant-translation-nosupg-nx12-dt001-t005",
        case_subdir="mms_traveling_interface_2d",
        generate_args=(
            "--nx", "12",
            "--ny", "12",
            "--time-steps", "5",
            "--time-step", "0.001",
            "--output-cadence", "5",
            "--omega", "0.0",
            "--level-set-only",
            "--level-set-velocity-source", "constant",
            "--level-set-constant-velocity", "0.1", "0.0", "0.0",
            "--disable-level-set-supg",
        ),
        result_name="result_005.vtu",
        expect_solver_success=True,
        expect_verifier_pass=None,
        description=(
            "Finest P2 level-set-only exact-translation five-step diagnostic "
            "with SUPG disabled for the asymptotic window."
        ),
        verify_history=True,
        required_solver_xml_controls=HIGH_ORDER_MMS_LEVEL_SET_ONLY_CONSTANT_NO_SUPG_XML_CONTROLS,
        validation_scope="trend_diagnostic",
    ),
    "mms-ls-only-p2-constant-translation-nosupg-inflow-nx2-dt001-t005": Probe(
        name="mms-ls-only-p2-constant-translation-nosupg-inflow-nx2-dt001-t005",
        case_subdir="mms_traveling_interface_2d",
        generate_args=(
            "--nx", "2",
            "--ny", "2",
            "--time-steps", "5",
            "--time-step", "0.001",
            "--output-cadence", "5",
            "--omega", "0.0",
            "--level-set-only",
            "--level-set-velocity-source", "constant",
            "--level-set-constant-velocity", "0.1", "0.0", "0.0",
            "--disable-level-set-supg",
            "--level-set-exact-inflow",
        ),
        result_name="result_005.vtu",
        expect_solver_success=True,
        expect_verifier_pass=None,
        description=(
            "Coarser standalone P2 exact-translation diagnostic with SUPG "
            "disabled and exact side-wall level-set inflow data."
        ),
        verify_history=True,
        required_solver_xml_controls=HIGH_ORDER_MMS_LEVEL_SET_ONLY_CONSTANT_NO_SUPG_EXACT_INFLOW_XML_CONTROLS,
        validation_scope="trend_diagnostic",
    ),
    "mms-ls-only-p2-constant-translation-nosupg-inflow-nx3-dt001-t005": Probe(
        name="mms-ls-only-p2-constant-translation-nosupg-inflow-nx3-dt001-t005",
        case_subdir="mms_traveling_interface_2d",
        generate_args=(
            "--nx", "3",
            "--ny", "3",
            "--time-steps", "5",
            "--time-step", "0.001",
            "--output-cadence", "5",
            "--omega", "0.0",
            "--level-set-only",
            "--level-set-velocity-source", "constant",
            "--level-set-constant-velocity", "0.1", "0.0", "0.0",
            "--disable-level-set-supg",
            "--level-set-exact-inflow",
        ),
        result_name="result_005.vtu",
        expect_solver_success=True,
        expect_verifier_pass=None,
        description=(
            "Intermediate standalone P2 exact-translation diagnostic with "
            "SUPG disabled and exact side-wall level-set inflow data."
        ),
        verify_history=True,
        required_solver_xml_controls=HIGH_ORDER_MMS_LEVEL_SET_ONLY_CONSTANT_NO_SUPG_EXACT_INFLOW_XML_CONTROLS,
        validation_scope="trend_diagnostic",
    ),
    "mms-ls-only-p2-constant-translation-nosupg-inflow-nx4-dt001-t005": Probe(
        name="mms-ls-only-p2-constant-translation-nosupg-inflow-nx4-dt001-t005",
        case_subdir="mms_traveling_interface_2d",
        generate_args=(
            "--nx", "4",
            "--ny", "4",
            "--time-steps", "5",
            "--time-step", "0.001",
            "--output-cadence", "5",
            "--omega", "0.0",
            "--level-set-only",
            "--level-set-velocity-source", "constant",
            "--level-set-constant-velocity", "0.1", "0.0", "0.0",
            "--disable-level-set-supg",
            "--level-set-exact-inflow",
        ),
        result_name="result_005.vtu",
        expect_solver_success=True,
        expect_verifier_pass=None,
        description=(
            "Compact standalone P2 exact-translation diagnostic with SUPG "
            "disabled and exact side-wall level-set inflow data."
        ),
        verify_history=True,
        required_solver_xml_controls=HIGH_ORDER_MMS_LEVEL_SET_ONLY_CONSTANT_NO_SUPG_EXACT_INFLOW_XML_CONTROLS,
        validation_scope="trend_diagnostic",
    ),
    "mms-ls-only-p2-constant-translation-nosupg-inflow-nx6-dt001-t005": Probe(
        name="mms-ls-only-p2-constant-translation-nosupg-inflow-nx6-dt001-t005",
        case_subdir="mms_traveling_interface_2d",
        generate_args=(
            "--nx", "6",
            "--ny", "6",
            "--time-steps", "5",
            "--time-step", "0.001",
            "--output-cadence", "5",
            "--omega", "0.0",
            "--level-set-only",
            "--level-set-velocity-source", "constant",
            "--level-set-constant-velocity", "0.1", "0.0", "0.0",
            "--disable-level-set-supg",
            "--level-set-exact-inflow",
        ),
        result_name="result_005.vtu",
        expect_solver_success=True,
        expect_verifier_pass=None,
        description=(
            "Asymptotic-window standalone P2 exact-translation diagnostic "
            "with SUPG disabled and exact side-wall level-set inflow data."
        ),
        verify_history=True,
        required_solver_xml_controls=HIGH_ORDER_MMS_LEVEL_SET_ONLY_CONSTANT_NO_SUPG_EXACT_INFLOW_XML_CONTROLS,
        validation_scope="trend_diagnostic",
    ),
    "mms-ls-only-p2-constant-translation-nosupg-inflow-nx8-dt001-t005": Probe(
        name="mms-ls-only-p2-constant-translation-nosupg-inflow-nx8-dt001-t005",
        case_subdir="mms_traveling_interface_2d",
        generate_args=(
            "--nx", "8",
            "--ny", "8",
            "--time-steps", "5",
            "--time-step", "0.001",
            "--output-cadence", "5",
            "--omega", "0.0",
            "--level-set-only",
            "--level-set-velocity-source", "constant",
            "--level-set-constant-velocity", "0.1", "0.0", "0.0",
            "--disable-level-set-supg",
            "--level-set-exact-inflow",
        ),
        result_name="result_005.vtu",
        expect_solver_success=True,
        expect_verifier_pass=None,
        description=(
            "Higher-resolution standalone P2 exact-translation diagnostic "
            "with SUPG disabled and exact side-wall level-set inflow data."
        ),
        verify_history=True,
        required_solver_xml_controls=HIGH_ORDER_MMS_LEVEL_SET_ONLY_CONSTANT_NO_SUPG_EXACT_INFLOW_XML_CONTROLS,
        validation_scope="trend_diagnostic",
    ),
    "mms-ls-only-p2-constant-translation-nosupg-inflow-nx10-dt001-t005": Probe(
        name="mms-ls-only-p2-constant-translation-nosupg-inflow-nx10-dt001-t005",
        case_subdir="mms_traveling_interface_2d",
        generate_args=(
            "--nx", "10",
            "--ny", "10",
            "--time-steps", "5",
            "--time-step", "0.001",
            "--output-cadence", "5",
            "--omega", "0.0",
            "--level-set-only",
            "--level-set-velocity-source", "constant",
            "--level-set-constant-velocity", "0.1", "0.0", "0.0",
            "--disable-level-set-supg",
            "--level-set-exact-inflow",
        ),
        result_name="result_005.vtu",
        expect_solver_success=True,
        expect_verifier_pass=None,
        description=(
            "Higher-resolution standalone P2 exact-translation diagnostic "
            "with SUPG disabled and exact side-wall level-set inflow data."
        ),
        verify_history=True,
        required_solver_xml_controls=HIGH_ORDER_MMS_LEVEL_SET_ONLY_CONSTANT_NO_SUPG_EXACT_INFLOW_XML_CONTROLS,
        validation_scope="trend_diagnostic",
    ),
    "mms-ls-only-p2-constant-translation-nosupg-inflow-nx12-dt001-t005": Probe(
        name="mms-ls-only-p2-constant-translation-nosupg-inflow-nx12-dt001-t005",
        case_subdir="mms_traveling_interface_2d",
        generate_args=(
            "--nx", "12",
            "--ny", "12",
            "--time-steps", "5",
            "--time-step", "0.001",
            "--output-cadence", "5",
            "--omega", "0.0",
            "--level-set-only",
            "--level-set-velocity-source", "constant",
            "--level-set-constant-velocity", "0.1", "0.0", "0.0",
            "--disable-level-set-supg",
            "--level-set-exact-inflow",
        ),
        result_name="result_005.vtu",
        expect_solver_success=True,
        expect_verifier_pass=None,
        description=(
            "Finest standalone P2 exact-translation diagnostic with SUPG "
            "disabled and exact side-wall level-set inflow data."
        ),
        verify_history=True,
        required_solver_xml_controls=HIGH_ORDER_MMS_LEVEL_SET_ONLY_CONSTANT_NO_SUPG_EXACT_INFLOW_XML_CONTROLS,
        validation_scope="trend_diagnostic",
    ),
    **{
        f"mms-ls-only-p2-flat-horizontal-nosupg-nx{nx}-dt001-t005": (
            mms_ls_only_p2_flat_horizontal_no_supg_probe(nx)
        )
        for nx in (2, 3, 4, 6, 8, 10, 12)
    },
    **{
        f"mms-ls-only-flat-static-p2-nx{nx}-dt001-t001": mms_ls_only_flat_static_p2_probe(nx)
        for nx in (2, 3, 4)
    },
    "mms-ls-only-static-p2-nx2-dt001-t001": Probe(
        name="mms-ls-only-static-p2-nx2-dt001-t001",
        case_subdir="mms_traveling_interface_2d",
        generate_args=(
            "--nx", "2",
            "--ny", "2",
            "--time-steps", "1",
            "--time-step", "0.001",
            "--output-cadence", "1",
            "--omega", "0.0",
            "--u0", "0.0",
            "--level-set-only",
            "--level-set-velocity-source", "constant",
            "--level-set-constant-velocity", "0.0", "0.0", "0.0",
        ),
        result_name="result_001.vtu",
        expect_solver_success=True,
        expect_verifier_pass=True,
        description=(
            "Coarser P2 level-set-only zero-advection control. This checks "
            "that the standalone generalized-alpha state path preserves a "
            "stationary high-order level set before attributing W2 to advection."
        ),
        verify_history=True,
        required_solver_xml_controls=HIGH_ORDER_MMS_LEVEL_SET_ONLY_STATIC_XML_CONTROLS,
        validation_scope="trend_diagnostic",
    ),
    "mms-ls-only-static-p2-nx3-dt001-t001": Probe(
        name="mms-ls-only-static-p2-nx3-dt001-t001",
        case_subdir="mms_traveling_interface_2d",
        generate_args=(
            "--nx", "3",
            "--ny", "3",
            "--time-steps", "1",
            "--time-step", "0.001",
            "--output-cadence", "1",
            "--omega", "0.0",
            "--u0", "0.0",
            "--level-set-only",
            "--level-set-velocity-source", "constant",
            "--level-set-constant-velocity", "0.0", "0.0", "0.0",
        ),
        result_name="result_001.vtu",
        expect_solver_success=True,
        expect_verifier_pass=True,
        description=(
            "Intermediate P2 level-set-only zero-advection control using one "
            "dt=0.001 step."
        ),
        verify_history=True,
        required_solver_xml_controls=HIGH_ORDER_MMS_LEVEL_SET_ONLY_STATIC_XML_CONTROLS,
        validation_scope="trend_diagnostic",
    ),
    "mms-ls-only-static-p2-nx4-dt001-t001": Probe(
        name="mms-ls-only-static-p2-nx4-dt001-t001",
        case_subdir="mms_traveling_interface_2d",
        generate_args=(
            "--nx", "4",
            "--ny", "4",
            "--time-steps", "1",
            "--time-step", "0.001",
            "--output-cadence", "1",
            "--omega", "0.0",
            "--u0", "0.0",
            "--level-set-only",
            "--level-set-velocity-source", "constant",
            "--level-set-constant-velocity", "0.0", "0.0", "0.0",
        ),
        result_name="result_001.vtu",
        expect_solver_success=True,
        expect_verifier_pass=True,
        description=(
            "Compact P2 level-set-only zero-advection control using one "
            "dt=0.001 step."
        ),
        verify_history=True,
        required_solver_xml_controls=HIGH_ORDER_MMS_LEVEL_SET_ONLY_STATIC_XML_CONTROLS,
        validation_scope="trend_diagnostic",
    ),
    "mms-ls-only-p1-constant-translation-nx2-dt001-t005": Probe(
        name="mms-ls-only-p1-constant-translation-nx2-dt001-t005",
        case_subdir="mms_traveling_interface_2d",
        generate_args=(
            "--nx", "2",
            "--ny", "2",
            "--element-order", "1",
            "--time-steps", "5",
            "--time-step", "0.001",
            "--output-cadence", "5",
            "--omega", "0.0",
            "--level-set-only",
            "--level-set-velocity-source", "constant",
            "--level-set-constant-velocity", "0.1", "0.0", "0.0",
        ),
        result_name="result_005.vtu",
        expect_solver_success=True,
        expect_verifier_pass=None,
        description=(
            "Coarser P1 level-set-only exact-translation transport control. "
            "This separates the standalone transport wiring from the quadratic "
            "P2 high-order branch used by the W2 MMS diagnostics."
        ),
        verify_history=True,
        required_solver_xml_controls=MMS_LEVEL_SET_ONLY_P1_CONSTANT_XML_CONTROLS,
        validation_scope="trend_diagnostic",
    ),
    "mms-ls-only-p1-constant-translation-nx3-dt001-t005": Probe(
        name="mms-ls-only-p1-constant-translation-nx3-dt001-t005",
        case_subdir="mms_traveling_interface_2d",
        generate_args=(
            "--nx", "3",
            "--ny", "3",
            "--element-order", "1",
            "--time-steps", "5",
            "--time-step", "0.001",
            "--output-cadence", "5",
            "--omega", "0.0",
            "--level-set-only",
            "--level-set-velocity-source", "constant",
            "--level-set-constant-velocity", "0.1", "0.0", "0.0",
        ),
        result_name="result_005.vtu",
        expect_solver_success=True,
        expect_verifier_pass=None,
        description=(
            "Intermediate P1 level-set-only exact-translation transport control "
            "using five dt=0.001 steps to t=0.005."
        ),
        verify_history=True,
        required_solver_xml_controls=MMS_LEVEL_SET_ONLY_P1_CONSTANT_XML_CONTROLS,
        validation_scope="trend_diagnostic",
    ),
    "mms-ls-only-p1-constant-translation-nx4-dt001-t005": Probe(
        name="mms-ls-only-p1-constant-translation-nx4-dt001-t005",
        case_subdir="mms_traveling_interface_2d",
        generate_args=(
            "--nx", "4",
            "--ny", "4",
            "--element-order", "1",
            "--time-steps", "5",
            "--time-step", "0.001",
            "--output-cadence", "5",
            "--omega", "0.0",
            "--level-set-only",
            "--level-set-velocity-source", "constant",
            "--level-set-constant-velocity", "0.1", "0.0", "0.0",
        ),
        result_name="result_005.vtu",
        expect_solver_success=True,
        expect_verifier_pass=None,
        description=(
            "Compact P1 level-set-only exact-translation transport control "
            "using five dt=0.001 steps to t=0.005."
        ),
        verify_history=True,
        required_solver_xml_controls=MMS_LEVEL_SET_ONLY_P1_CONSTANT_XML_CONTROLS,
        validation_scope="trend_diagnostic",
    ),
    "mms-ls-only-constant-translation-rho0-nx2-dt001-t005": Probe(
        name="mms-ls-only-constant-translation-rho0-nx2-dt001-t005",
        case_subdir="mms_traveling_interface_2d",
        generate_args=(
            "--nx", "2",
            "--ny", "2",
            "--time-steps", "5",
            "--time-step", "0.001",
            "--output-cadence", "5",
            "--spectral-radius", "0.0",
            "--omega", "0.0",
            "--level-set-only",
            "--level-set-velocity-source", "constant",
            "--level-set-constant-velocity", "0.1", "0.0", "0.0",
        ),
        result_name="result_005.vtu",
        expect_solver_success=True,
        expect_verifier_pass=None,
        description=(
            "Coarser level-set-only exact-translation transport diagnostic "
            "with generalized-alpha rho_inf=0 to isolate time-integration "
            "damping sensitivity."
        ),
        verify_history=True,
        required_solver_xml_controls=HIGH_ORDER_MMS_LEVEL_SET_ONLY_CONSTANT_RHO0_XML_CONTROLS,
        validation_scope="trend_diagnostic",
    ),
    "mms-ls-only-constant-translation-rho0-nx3-dt001-t005": Probe(
        name="mms-ls-only-constant-translation-rho0-nx3-dt001-t005",
        case_subdir="mms_traveling_interface_2d",
        generate_args=(
            "--nx", "3",
            "--ny", "3",
            "--time-steps", "5",
            "--time-step", "0.001",
            "--output-cadence", "5",
            "--spectral-radius", "0.0",
            "--omega", "0.0",
            "--level-set-only",
            "--level-set-velocity-source", "constant",
            "--level-set-constant-velocity", "0.1", "0.0", "0.0",
        ),
        result_name="result_005.vtu",
        expect_solver_success=True,
        expect_verifier_pass=None,
        description=(
            "Intermediate level-set-only exact-translation transport diagnostic "
            "with generalized-alpha rho_inf=0 to isolate time-integration "
            "damping sensitivity."
        ),
        verify_history=True,
        required_solver_xml_controls=HIGH_ORDER_MMS_LEVEL_SET_ONLY_CONSTANT_RHO0_XML_CONTROLS,
        validation_scope="trend_diagnostic",
    ),
    "mms-ls-only-constant-translation-rho0-nx4-dt001-t005": Probe(
        name="mms-ls-only-constant-translation-rho0-nx4-dt001-t005",
        case_subdir="mms_traveling_interface_2d",
        generate_args=(
            "--nx", "4",
            "--ny", "4",
            "--time-steps", "5",
            "--time-step", "0.001",
            "--output-cadence", "5",
            "--spectral-radius", "0.0",
            "--omega", "0.0",
            "--level-set-only",
            "--level-set-velocity-source", "constant",
            "--level-set-constant-velocity", "0.1", "0.0", "0.0",
        ),
        result_name="result_005.vtu",
        expect_solver_success=True,
        expect_verifier_pass=None,
        description=(
            "Compact level-set-only exact-translation transport diagnostic "
            "with generalized-alpha rho_inf=0 to isolate time-integration "
            "damping sensitivity."
        ),
        verify_history=True,
        required_solver_xml_controls=HIGH_ORDER_MMS_LEVEL_SET_ONLY_CONSTANT_RHO0_XML_CONTROLS,
        validation_scope="trend_diagnostic",
    ),
    "mms-constant-translation-nosupg-depth6-nx2-dt001-t005": Probe(
        name="mms-constant-translation-nosupg-depth6-nx2-dt001-t005",
        case_subdir="mms_traveling_interface_2d",
        generate_args=(
            "--nx", "2",
            "--ny", "2",
            "--time-steps", "5",
            "--time-step", "0.001",
            "--output-cadence", "5",
            "--omega", "0.0",
            "--level-set-velocity-source", "constant",
            "--level-set-constant-velocity", "0.1", "0.0", "0.0",
            "--disable-level-set-supg",
            "--implicit-cut-max-subdivision-depth", "6",
        ),
        result_name="result_005.vtu",
        expect_solver_success=True,
        expect_verifier_pass=None,
        description=(
            "Coarser high-order exact-translation MMS first-output diagnostic "
            "with level-set SUPG disabled. This isolates whether the early "
            "phi/interface trend incoherence is caused by transport "
            "stabilization rather than the moving-cut/free-surface coupling."
        ),
        verify_history=True,
        required_solver_features=HIGH_ORDER_MMS_REQUIRED_SOLVER_FEATURES,
        required_solver_xml_controls=HIGH_ORDER_MMS_CONSTANT_NO_SUPG_XML_CONTROLS,
        validation_scope="trend_diagnostic",
    ),
    "mms-constant-translation-nosupg-depth6-nx3-dt001-t005": Probe(
        name="mms-constant-translation-nosupg-depth6-nx3-dt001-t005",
        case_subdir="mms_traveling_interface_2d",
        generate_args=(
            "--nx", "3",
            "--ny", "3",
            "--time-steps", "5",
            "--time-step", "0.001",
            "--output-cadence", "5",
            "--omega", "0.0",
            "--level-set-velocity-source", "constant",
            "--level-set-constant-velocity", "0.1", "0.0", "0.0",
            "--disable-level-set-supg",
            "--implicit-cut-max-subdivision-depth", "6",
        ),
        result_name="result_005.vtu",
        expect_solver_success=True,
        expect_verifier_pass=None,
        description=(
            "Intermediate high-order exact-translation MMS first-output "
            "diagnostic with level-set SUPG disabled."
        ),
        verify_history=True,
        required_solver_features=HIGH_ORDER_MMS_REQUIRED_SOLVER_FEATURES,
        required_solver_xml_controls=HIGH_ORDER_MMS_CONSTANT_NO_SUPG_XML_CONTROLS,
        validation_scope="trend_diagnostic",
    ),
    "mms-constant-translation-nosupg-depth6-nx4-dt001-t005": Probe(
        name="mms-constant-translation-nosupg-depth6-nx4-dt001-t005",
        case_subdir="mms_traveling_interface_2d",
        generate_args=(
            "--nx", "4",
            "--ny", "4",
            "--time-steps", "5",
            "--time-step", "0.001",
            "--output-cadence", "5",
            "--omega", "0.0",
            "--level-set-velocity-source", "constant",
            "--level-set-constant-velocity", "0.1", "0.0", "0.0",
            "--disable-level-set-supg",
            "--implicit-cut-max-subdivision-depth", "6",
        ),
        result_name="result_005.vtu",
        expect_solver_success=True,
        expect_verifier_pass=None,
        description=(
            "Compact high-order exact-translation MMS first-output diagnostic "
            "with level-set SUPG disabled."
        ),
        verify_history=True,
        required_solver_features=HIGH_ORDER_MMS_REQUIRED_SOLVER_FEATURES,
        required_solver_xml_controls=HIGH_ORDER_MMS_CONSTANT_NO_SUPG_XML_CONTROLS,
        validation_scope="trend_diagnostic",
    ),
    "mms-constant-translation-conservative-depth6-nx2-dt001-t005": Probe(
        name="mms-constant-translation-conservative-depth6-nx2-dt001-t005",
        case_subdir="mms_traveling_interface_2d",
        generate_args=(
            "--nx", "2",
            "--ny", "2",
            "--time-steps", "5",
            "--time-step", "0.001",
            "--output-cadence", "5",
            "--omega", "0.0",
            "--level-set-velocity-source", "constant",
            "--level-set-constant-velocity", "0.1", "0.0", "0.0",
            "--level-set-transport-form", "conservative_divergence",
            "--implicit-cut-max-subdivision-depth", "6",
        ),
        result_name="result_005.vtu",
        expect_solver_success=True,
        expect_verifier_pass=None,
        description=(
            "Coarser high-order exact-translation MMS first-output diagnostic "
            "using conservative-divergence level-set transport. This isolates "
            "whether the early phi/interface trend incoherence depends on the "
            "transport residual form."
        ),
        verify_history=True,
        required_solver_features=HIGH_ORDER_MMS_REQUIRED_SOLVER_FEATURES,
        required_solver_xml_controls=HIGH_ORDER_MMS_CONSTANT_CONSERVATIVE_XML_CONTROLS,
        validation_scope="trend_diagnostic",
    ),
    "mms-constant-translation-conservative-depth6-nx3-dt001-t005": Probe(
        name="mms-constant-translation-conservative-depth6-nx3-dt001-t005",
        case_subdir="mms_traveling_interface_2d",
        generate_args=(
            "--nx", "3",
            "--ny", "3",
            "--time-steps", "5",
            "--time-step", "0.001",
            "--output-cadence", "5",
            "--omega", "0.0",
            "--level-set-velocity-source", "constant",
            "--level-set-constant-velocity", "0.1", "0.0", "0.0",
            "--level-set-transport-form", "conservative_divergence",
            "--implicit-cut-max-subdivision-depth", "6",
        ),
        result_name="result_005.vtu",
        expect_solver_success=True,
        expect_verifier_pass=None,
        description=(
            "Intermediate high-order exact-translation MMS first-output "
            "diagnostic using conservative-divergence level-set transport."
        ),
        verify_history=True,
        required_solver_features=HIGH_ORDER_MMS_REQUIRED_SOLVER_FEATURES,
        required_solver_xml_controls=HIGH_ORDER_MMS_CONSTANT_CONSERVATIVE_XML_CONTROLS,
        validation_scope="trend_diagnostic",
    ),
    "mms-constant-translation-conservative-depth6-nx4-dt001-t005": Probe(
        name="mms-constant-translation-conservative-depth6-nx4-dt001-t005",
        case_subdir="mms_traveling_interface_2d",
        generate_args=(
            "--nx", "4",
            "--ny", "4",
            "--time-steps", "5",
            "--time-step", "0.001",
            "--output-cadence", "5",
            "--omega", "0.0",
            "--level-set-velocity-source", "constant",
            "--level-set-constant-velocity", "0.1", "0.0", "0.0",
            "--level-set-transport-form", "conservative_divergence",
            "--implicit-cut-max-subdivision-depth", "6",
        ),
        result_name="result_005.vtu",
        expect_solver_success=True,
        expect_verifier_pass=None,
        description=(
            "Compact high-order exact-translation MMS first-output diagnostic "
            "using conservative-divergence level-set transport."
        ),
        verify_history=True,
        required_solver_features=HIGH_ORDER_MMS_REQUIRED_SOLVER_FEATURES,
        required_solver_xml_controls=HIGH_ORDER_MMS_CONSTANT_CONSERVATIVE_XML_CONTROLS,
        validation_scope="trend_diagnostic",
    ),
    "mms-constant-translation-inflow-depth6-nx2-history-10step": Probe(
        name="mms-constant-translation-inflow-depth6-nx2-history-10step",
        case_subdir="mms_traveling_interface_2d",
        generate_args=(
            "--nx", "2",
            "--ny", "2",
            "--time-steps", "10",
            "--time-step", "0.005",
            "--output-cadence", "1",
            "--omega", "0.0",
            "--level-set-velocity-source", "constant",
            "--level-set-constant-velocity", "0.1", "0.0", "0.0",
            "--level-set-exact-inflow",
            "--implicit-cut-max-subdivision-depth", "6",
        ),
        result_name="result_010.vtu",
        expect_solver_success=True,
        expect_verifier_pass=None,
        description=(
            "Coarser compact high-order MMS history diagnostic with exact "
            "constant-translation level-set velocity and exact side-wall "
            "level-set inflow data. This isolates whether the first-output "
            "phase lag is caused by missing inflow boundary state."
        ),
        verify_history=True,
        required_solver_features=HIGH_ORDER_MMS_REQUIRED_SOLVER_FEATURES,
        required_solver_xml_controls=HIGH_ORDER_MMS_CONSTANT_EXACT_INFLOW_XML_CONTROLS,
        validation_scope="trend_diagnostic",
    ),
    "mms-constant-translation-inflow-depth6-nx3-history-10step": Probe(
        name="mms-constant-translation-inflow-depth6-nx3-history-10step",
        case_subdir="mms_traveling_interface_2d",
        generate_args=(
            "--nx", "3",
            "--ny", "3",
            "--time-steps", "10",
            "--time-step", "0.005",
            "--output-cadence", "1",
            "--omega", "0.0",
            "--level-set-velocity-source", "constant",
            "--level-set-constant-velocity", "0.1", "0.0", "0.0",
            "--level-set-exact-inflow",
            "--implicit-cut-max-subdivision-depth", "6",
        ),
        result_name="result_010.vtu",
        expect_solver_success=True,
        expect_verifier_pass=None,
        description=(
            "Intermediate compact high-order MMS history diagnostic with exact "
            "constant-translation level-set velocity and exact side-wall "
            "level-set inflow data."
        ),
        verify_history=True,
        required_solver_features=HIGH_ORDER_MMS_REQUIRED_SOLVER_FEATURES,
        required_solver_xml_controls=HIGH_ORDER_MMS_CONSTANT_EXACT_INFLOW_XML_CONTROLS,
        validation_scope="trend_diagnostic",
    ),
    "mms-constant-translation-inflow-depth6-nx4-history-10step": Probe(
        name="mms-constant-translation-inflow-depth6-nx4-history-10step",
        case_subdir="mms_traveling_interface_2d",
        generate_args=(
            "--nx", "4",
            "--ny", "4",
            "--time-steps", "10",
            "--time-step", "0.005",
            "--output-cadence", "1",
            "--omega", "0.0",
            "--level-set-velocity-source", "constant",
            "--level-set-constant-velocity", "0.1", "0.0", "0.0",
            "--level-set-exact-inflow",
            "--implicit-cut-max-subdivision-depth", "6",
        ),
        result_name="result_010.vtu",
        expect_solver_success=True,
        expect_verifier_pass=None,
        description=(
            "Compact high-order MMS history diagnostic with exact "
            "constant-translation level-set velocity and exact side-wall "
            "level-set inflow data."
        ),
        verify_history=True,
        required_solver_features=HIGH_ORDER_MMS_REQUIRED_SOLVER_FEATURES,
        required_solver_xml_controls=HIGH_ORDER_MMS_CONSTANT_EXACT_INFLOW_XML_CONTROLS,
        validation_scope="trend_diagnostic",
    ),
    "mms-constant-lsvel-depth8-10step": Probe(
        name="mms-constant-lsvel-depth8-10step",
        case_subdir="mms_traveling_interface_2d",
        generate_args=(
            "--nx", "4",
            "--ny", "4",
            "--time-steps", "10",
            "--time-step", "0.005",
            "--output-cadence", "10",
            "--level-set-velocity-source", "constant",
            "--level-set-constant-velocity", "0.1", "0.0", "0.0",
            "--implicit-cut-max-subdivision-depth", "8",
        ),
        result_name="result_010.vtu",
        expect_solver_success=True,
        expect_verifier_pass=False,
        description=(
            "Compact high-order MMS diagnostic with prescribed constant level-set "
            "advection velocity and deeper Saye recursion. Current evidence "
            "expects the solver to reach final time but fail the full MMS "
            "accuracy verifier at high cost."
        ),
        required_solver_features=HIGH_ORDER_MMS_REQUIRED_SOLVER_FEATURES,
        required_solver_xml_controls=HIGH_ORDER_MMS_CONSTANT_XML_CONTROLS,
        validation_scope="expected_failure_diagnostic",
    ),
    "mms-full-10step": Probe(
        name="mms-full-10step",
        case_subdir="mms_traveling_interface_2d",
        generate_args=("--time-steps", "10", "--time-step", "0.005", "--output-cadence", "10"),
        result_name="result_010.vtu",
        expect_solver_success=True,
        expect_verifier_pass=True,
        description=(
            "Full-size MMS transient target. Current evidence shows this is too expensive "
            "for a short bounded probe, so use an explicit timeout."
        ),
        required_solver_features=HIGH_ORDER_MMS_REQUIRED_SOLVER_FEATURES,
        required_solver_xml_controls=HIGH_ORDER_MMS_PRESCRIBED_XML_CONTROLS,
        validation_scope="open_performance_target",
        timeout_policy="diagnostic_timeout_not_validation_pass",
    ),
    "linear-sloshing-default": Probe(
        name="linear-sloshing-default",
        case_subdir="linear_sloshing_2d",
        generate_args=(),
        result_name="result_050.vtu",
        expect_solver_success=True,
        expect_verifier_pass=True,
        description=(
            "Checked-in 2D linear sloshing regression against the "
            "small-amplitude one-phase potential-flow reference. This is a "
            "free-surface motion and wet-domain pressure gate, not a strict "
            "viscous no-slip Navier-Stokes MMS."
        ),
    ),
    "square-default-smoke": Probe(
        name="square-default-smoke",
        case_subdir="square_tank_tilt_settling",
        generate_args=(),
        result_name="result_1000.vtu",
        expect_solver_success=True,
        expect_verifier_pass=True,
        description="Checked-in finite-time square-tank settling smoke.",
    ),
    "square-equilibrium-smoke": Probe(
        name="square-equilibrium-smoke",
        case_subdir="square_tank_tilt_settling",
        generate_args=("--initial-state", "equilibrium", "--time-steps", "5"),
        result_name="result_005.vtu",
        expect_solver_success=True,
        expect_verifier_pass=True,
        description="Strict hydrostatic tilted-square equilibrium companion.",
    ),
    "square-equilibrium-smoke-mpi2": Probe(
        name="square-equilibrium-smoke-mpi2",
        case_subdir="square_tank_tilt_settling",
        generate_args=("--initial-state", "equilibrium", "--time-steps", "5"),
        result_name=LATEST_RESULT_NAME,
        expect_solver_success=True,
        expect_verifier_pass=True,
        description=(
            "MPI-2 strict hydrostatic tilted-square equilibrium companion. "
            "This records distributed generated-interface, wet-volume, "
            "fallback, timing, and pressure/traction verifier evidence."
        ),
        required_solver_features=("FE_ENABLE_MPI",),
        required_solver_xml_controls=(("Preconditioner", "fsils"),),
        mpi_policy=MPI2_POLICY,
        mpi_ranks=2,
    ),
    "square-equilibrium-smoke-mpi4": Probe(
        name="square-equilibrium-smoke-mpi4",
        case_subdir="square_tank_tilt_settling",
        generate_args=("--initial-state", "equilibrium", "--time-steps", "5"),
        result_name=LATEST_RESULT_NAME,
        expect_solver_success=True,
        expect_verifier_pass=True,
        description=(
            "MPI-4 strict hydrostatic tilted-square equilibrium companion. "
            "This extends the distributed generated-interface gate to four ranks."
        ),
        required_solver_features=("FE_ENABLE_MPI",),
        required_solver_xml_controls=(("Preconditioner", "fsils"),),
        mpi_policy=MPI4_POLICY,
        mpi_ranks=4,
    ),
    "square-refined-nx17": Probe(
        name="square-refined-nx17",
        case_subdir="square_tank_tilt_settling",
        generate_args=("--nx", "17", "--ny", "17", "--time-step", "0.001", "--time-steps", "1000"),
        result_name="result_1000.vtu",
        expect_solver_success=False,
        expect_verifier_pass=None,
        description=(
            "Refined square-tank diagnostic. Current evidence expects nonlinear failure "
            "before final output, exposing the open refinement gate."
        ),
        validation_scope="expected_failure_diagnostic",
    ),
    "square-refined-fixed-t02": Probe(
        name="square-refined-fixed-t02",
        case_subdir="square_tank_tilt_settling",
        generate_args=(
            "--nx", "17",
            "--ny", "17",
            "--time-step", "0.001",
            "--time-steps", "200",
            "--verification-profile", "early_transient",
        ),
        result_name=LATEST_RESULT_NAME,
        expect_solver_success=False,
        expect_verifier_pass=None,
        description=(
            "Refined square-tank fixed-step comparison target through t=0.2. "
            "Current evidence expects the fixed dt=0.001 run to fail before "
            "the target, while preserving the latest accepted early-transient "
            "verifier context."
        ),
        validation_scope="expected_failure_diagnostic",
    ),
    "square-refined-short-smoke": Probe(
        name="square-refined-short-smoke",
        case_subdir="square_tank_tilt_settling",
        generate_args=(
            "--nx", "17",
            "--ny", "17",
            "--time-step", "0.001",
            "--time-steps", "100",
            "--verification-profile", "early_transient",
        ),
        result_name="result_100.vtu",
        expect_solver_success=True,
        expect_verifier_pass=True,
        description=(
            "Refined square-tank early-transient smoke. This gates finite fields, "
            "interface geometry, volume, probe pressure, and solver consistency "
            "without claiming final hydrostatic equilibrium."
        ),
    ),
    "square-refined-adaptive-t02": Probe(
        name="square-refined-adaptive-t02",
        case_subdir="square_tank_tilt_settling",
        generate_args=(
            "--nx", "17",
            "--ny", "17",
            "--time-step", "0.001",
            "--time-steps", "200",
            "--verification-profile", "early_transient",
        ),
        result_name=LATEST_RESULT_NAME,
        expect_solver_success=True,
        expect_verifier_pass=False,
        description=(
            "Refined square-tank adaptive-step comparison target through "
            "t=0.2 using the same early-transient verifier profile as the "
            "fixed-step t=0.2 diagnostic. Current evidence expects the solver "
            "to cross the fixed-step event, but the latest output still fails "
            "interface-pressure gates, so this remains diagnostic-only."
        ),
        solver_env=(
            ("SVMP_TIMELOOP_ADAPTIVE", "1"),
            ("SVMP_TIMELOOP_MIN_DT", "0.000125"),
            ("SVMP_TIMELOOP_MAX_DT", "0.001"),
            ("SVMP_TIMELOOP_MAX_RETRIES", "8"),
        ),
        validation_scope="expected_failure_diagnostic",
    ),
    "square-refined-halfstep-t02": Probe(
        name="square-refined-halfstep-t02",
        case_subdir="square_tank_tilt_settling",
        generate_args=(
            "--nx", "17",
            "--ny", "17",
            "--time-step", "0.0005",
            "--time-steps", "400",
            "--verification-profile", "early_transient",
        ),
        result_name="result_400.vtu",
        expect_solver_success=True,
        expect_verifier_pass=True,
        description=(
            "Refined square-tank half-step early transient through t=0.2. "
            "This crosses the fixed-dt=0.001 step-173 topology event and gates "
            "the same finite-field, interface, volume, probe-pressure, and "
            "solver-consistency checks as the short refined smoke."
        ),
    ),
    "square-refined-halfstep-t02-pressure": Probe(
        name="square-refined-halfstep-t02-pressure",
        case_subdir="square_tank_tilt_settling",
        generate_args=(
            "--nx", "17",
            "--ny", "17",
            "--time-step", "0.0005",
            "--time-steps", "400",
            "--verification-profile", "transient_pressure",
        ),
        result_name="result_400.vtu",
        expect_solver_success=True,
        expect_verifier_pass=True,
        description=(
            "Refined square-tank half-step t=0.2 staged transient target with "
            "hydrostatic pressure-gradient verification enabled while still "
            "deferring final-equilibrium intercept closure."
        ),
    ),
    "square-refined-halfstep-t03-pressure": Probe(
        name="square-refined-halfstep-t03-pressure",
        case_subdir="square_tank_tilt_settling",
        generate_args=(
            "--nx", "17",
            "--ny", "17",
            "--time-step", "0.0005",
            "--time-steps", "600",
            "--verification-profile", "transient_pressure",
        ),
        result_name="result_600.vtu",
        expect_solver_success=True,
        expect_verifier_pass=False,
        description=(
            "Refined square-tank half-step t=0.3 staged transient diagnostic. "
            "Current evidence expects solver success through the target but a "
            "strict transient-pressure verifier failure on free-surface "
            "interface-pressure gates."
        ),
        validation_scope="expected_failure_diagnostic",
    ),
    "square-refined-halfstep-t03-pressure-interior": Probe(
        name="square-refined-halfstep-t03-pressure-interior",
        case_subdir="square_tank_tilt_settling",
        generate_args=(
            "--nx", "17",
            "--ny", "17",
            "--time-step", "0.0005",
            "--time-steps", "600",
            "--verification-profile", "transient_pressure_interior",
        ),
        result_name="result_600.vtu",
        expect_solver_success=True,
        expect_verifier_pass=True,
        description=(
            "Refined square-tank half-step t=0.3 contact-line-aware staged "
            "target. It preserves the strict pressure-gradient, volume, and "
            "finite-interface checks while applying the pressure trace RMS/max "
            "tolerances to the interior free-surface subset and retaining a "
            "bounded wall-contact endpoint guard."
        ),
    ),
    "square-refined-halfstep-t05-pressure-interior": Probe(
        name="square-refined-halfstep-t05-pressure-interior",
        case_subdir="square_tank_tilt_settling",
        generate_args=(
            "--nx", "17",
            "--ny", "17",
            "--time-step", "0.0005",
            "--time-steps", "1000",
            "--verification-profile", "transient_pressure_interior",
        ),
        result_name=LATEST_RESULT_NAME,
        expect_solver_success=False,
        expect_verifier_pass=None,
        description=(
            "Refined square-tank half-step t=0.5 contact-line-aware staged "
            "diagnostic. Current manual evidence expects a fallback-free "
            "nonlinear failure shortly after the t=0.3 staged target; the "
            "latest accepted output is still verified to preserve pressure, "
            "volume, and interface context for the next W3 blocker."
        ),
        validation_scope="expected_failure_diagnostic",
    ),
    "square-refined-adaptive-strict-ls-t05-pressure-interior": Probe(
        name="square-refined-adaptive-strict-ls-t05-pressure-interior",
        case_subdir="square_tank_tilt_settling",
        generate_args=(
            "--nx", "17",
            "--ny", "17",
            "--time-step", "0.0005",
            "--time-steps", "1000",
            "--verification-profile", "transient_pressure_interior",
        ),
        result_name=LATEST_RESULT_NAME,
        expect_solver_success=True,
        expect_verifier_pass=False,
        description=(
            "Refined square-tank adaptive strict no-reduction diagnostic through "
            "t=0.5 with the contact-line-aware staged pressure profile. This "
            "confirms the existing production-style adaptive controller can retry "
            "through the fixed half-step post-t=0.305 nonlinear failure, while "
            "recording the remaining final-time wall-adjacent pressure-quality "
            "blocker as an expected diagnostic failure."
        ),
        solver_env=(
            ("SVMP_TIMELOOP_ADAPTIVE", "1"),
            ("SVMP_TIMELOOP_MIN_DT", "0.000125"),
            ("SVMP_TIMELOOP_MAX_DT", "0.0005"),
            ("SVMP_TIMELOOP_MAX_RETRIES", "10"),
            ("SVMP_NEWTON_LINE_SEARCH_FAIL_ON_NO_REDUCTION", "1"),
        ),
        validation_scope="expected_failure_diagnostic",
    ),
    "square-refined-adaptive-strict-ls-t05-pressure-core": Probe(
        name="square-refined-adaptive-strict-ls-t05-pressure-core",
        case_subdir="square_tank_tilt_settling",
        generate_args=(
            "--nx", "17",
            "--ny", "17",
            "--time-step", "0.0005",
            "--time-steps", "1000",
            "--verification-profile", "transient_pressure_core",
        ),
        result_name=LATEST_RESULT_NAME,
        expect_solver_success=True,
        expect_verifier_pass=True,
        description=(
            "Refined square-tank adaptive strict no-reduction target through "
            "t=0.5 with the core free-surface pressure profile. This gates the "
            "pressure trace away from wall-contact and one-cell near-wall "
            "subsets while keeping both regions under bounded-pressure guards."
        ),
        solver_env=(
            ("SVMP_TIMELOOP_ADAPTIVE", "1"),
            ("SVMP_TIMELOOP_MIN_DT", "0.000125"),
            ("SVMP_TIMELOOP_MAX_DT", "0.0005"),
            ("SVMP_TIMELOOP_MAX_RETRIES", "10"),
            ("SVMP_NEWTON_LINE_SEARCH_FAIL_ON_NO_REDUCTION", "1"),
        ),
    ),
    "square-refined-adaptive-strict-ls-t10-pressure-core": Probe(
        name="square-refined-adaptive-strict-ls-t10-pressure-core",
        case_subdir="square_tank_tilt_settling",
        generate_args=(
            "--nx", "17",
            "--ny", "17",
            "--time-step", "0.0005",
            "--time-steps", "2000",
            "--verification-profile", "transient_pressure_core",
        ),
        result_name=LATEST_RESULT_NAME,
        expect_solver_success=True,
        expect_verifier_pass=True,
        description=(
            "Refined square-tank adaptive strict no-reduction open target through "
            "t=1.0 with the core free-surface pressure profile. This extends the "
            "passing t=0.5 staged target to the long-transient production horizon "
            "without allowing timeout diagnostics to count as broad validation."
        ),
        solver_env=(
            ("SVMP_TIMELOOP_ADAPTIVE", "1"),
            ("SVMP_TIMELOOP_MIN_DT", "0.000125"),
            ("SVMP_TIMELOOP_MAX_DT", "0.0005"),
            ("SVMP_TIMELOOP_MAX_RETRIES", "10"),
            ("SVMP_NEWTON_LINE_SEARCH_FAIL_ON_NO_REDUCTION", "1"),
        ),
        validation_scope="open_performance_target",
        timeout_policy="diagnostic_timeout_not_validation_pass",
    ),
    "square-refined-adaptive-strict-ls-t10-pressure-core-mindt6e-5": Probe(
        name="square-refined-adaptive-strict-ls-t10-pressure-core-mindt6e-5",
        case_subdir="square_tank_tilt_settling",
        generate_args=(
            "--nx", "17",
            "--ny", "17",
            "--time-step", "0.0005",
            "--time-steps", "2000",
            "--verification-profile", "transient_pressure_core",
        ),
        result_name=LATEST_RESULT_NAME,
        expect_solver_success=True,
        expect_verifier_pass=True,
        description=(
            "Refined square-tank adaptive strict no-reduction target through "
            "t=1.0 with the core free-surface pressure profile and a halved "
            "minimum adaptive step. This is the promoted W3 long-transient "
            "gate after the coarser adaptive floor isolated the late t=0.94 "
            "reachability miss as a step-floor issue rather than a pressure "
            "or interface-state failure."
        ),
        solver_env=(
            ("SVMP_TIMELOOP_ADAPTIVE", "1"),
            ("SVMP_TIMELOOP_MIN_DT", "0.0000625"),
            ("SVMP_TIMELOOP_MAX_DT", "0.0005"),
            ("SVMP_TIMELOOP_MAX_RETRIES", "10"),
            ("SVMP_NEWTON_LINE_SEARCH_FAIL_ON_NO_REDUCTION", "1"),
        ),
    ),
    "square-refined-adaptive-strict-ls-t03-pressure": Probe(
        name="square-refined-adaptive-strict-ls-t03-pressure",
        case_subdir="square_tank_tilt_settling",
        generate_args=(
            "--nx", "17",
            "--ny", "17",
            "--time-step", "0.0005",
            "--time-steps", "600",
            "--verification-profile", "transient_pressure",
        ),
        result_name=LATEST_RESULT_NAME,
        expect_solver_success=True,
        expect_verifier_pass=False,
        description=(
            "Refined square-tank strict adaptive/globalization diagnostic "
            "through t=0.3. Current evidence expects it to reach the staged "
            "target without rejected steps, showing that strict no-reduction "
            "line-search failure handling does not improve pressure quality "
            "relative to the fixed half-step target."
        ),
        solver_env=(
            ("SVMP_TIMELOOP_ADAPTIVE", "1"),
            ("SVMP_TIMELOOP_MIN_DT", "0.000125"),
            ("SVMP_TIMELOOP_MAX_DT", "0.0005"),
            ("SVMP_TIMELOOP_MAX_RETRIES", "10"),
            ("SVMP_NEWTON_LINE_SEARCH_FAIL_ON_NO_REDUCTION", "1"),
        ),
        validation_scope="expected_failure_diagnostic",
    ),
    "square-refined-vpen10-t04-pressure": Probe(
        name="square-refined-vpen10-t04-pressure",
        case_subdir="square_tank_tilt_settling",
        generate_args=(
            "--nx", "17",
            "--ny", "17",
            "--time-step", "0.001",
            "--time-steps", "400",
            "--verification-profile", "transient_pressure",
            "--cut-cell-velocity-gradient-penalty", "10.0",
        ),
        result_name="result_400.vtu",
        expect_solver_success=True,
        expect_verifier_pass=False,
        description=(
            "Refined square-tank high velocity-penalty diagnostic through "
            "t=0.4. Current evidence expects the stronger velocity ghost "
            "penalty to reach final time while failing strict free-surface "
            "interface-pressure gates."
        ),
        required_solver_xml_controls=(
            ("Use_cut_metadata_scale", "false"),
            ("Cut_cell_pressure_gradient_penalty", "1"),
        ),
        validation_scope="expected_failure_diagnostic",
    ),
    "square-refined-metadata-scale-t02": Probe(
        name="square-refined-metadata-scale-t02",
        case_subdir="square_tank_tilt_settling",
        generate_args=(
            "--nx", "17",
            "--ny", "17",
            "--time-step", "0.001",
            "--time-steps", "200",
            "--verification-profile", "early_transient",
            "--use-cut-metadata-scale",
        ),
        result_name=LATEST_RESULT_NAME,
        expect_solver_success=False,
        expect_verifier_pass=None,
        description=(
            "Refined square-tank metadata-scale diagnostic through t=0.2. "
            "Current evidence expects the capped cut-metadata scale to "
            "reproduce the early nonlinear failure near the fixed-step "
            "topology event, making the scaling regression reproducible "
            "from generator controls."
        ),
        required_solver_xml_controls=(
            ("Use_cut_metadata_scale", "true"),
            ("Cut_cell_pressure_gradient_penalty", "1"),
        ),
        validation_scope="expected_failure_diagnostic",
    ),
    "square-refined-metadata-penalty1e-3-t03-pressure": Probe(
        name="square-refined-metadata-penalty1e-3-t03-pressure",
        case_subdir="square_tank_tilt_settling",
        generate_args=(
            "--nx", "17",
            "--ny", "17",
            "--time-step", "0.0005",
            "--time-steps", "600",
            "--verification-profile", "transient_pressure",
            "--use-cut-metadata-scale",
            "--cut-cell-velocity-gradient-penalty", "0.001",
            "--cut-cell-pressure-gradient-penalty", "0.001",
        ),
        result_name="result_600.vtu",
        expect_solver_success=True,
        expect_verifier_pass=None,
        description=(
            "Refined square-tank bounded metadata-scale diagnostic through "
            "t=0.3. This keeps metadata scaling enabled but scales both "
            "cut-cell gradient penalties by 1e-3, emulating a strongly "
            "bounded effective metadata-scale cap while recording strict "
            "transient-pressure verifier metrics as trend data."
        ),
        required_solver_xml_controls=(
            ("Use_cut_metadata_scale", "true"),
            ("Cut_cell_pressure_gradient_penalty", "0.001"),
        ),
        validation_scope="trend_diagnostic",
    ),
    "square-refined-metadata-cap1-t03-pressure": Probe(
        name="square-refined-metadata-cap1-t03-pressure",
        case_subdir="square_tank_tilt_settling",
        generate_args=(
            "--nx", "17",
            "--ny", "17",
            "--time-step", "0.0005",
            "--time-steps", "600",
            "--verification-profile", "transient_pressure",
            "--use-cut-metadata-scale",
            "--cut-cell-metadata-scale-cap", "1.0",
        ),
        result_name="result_600.vtu",
        expect_solver_success=True,
        expect_verifier_pass=None,
        description=(
            "Refined square-tank bounded metadata-scale cap diagnostic through "
            "t=0.3. This keeps metadata scaling enabled but applies an explicit "
            "unit cap in the cut-stabilization form, separating a true local "
            "scale cap from the earlier small-penalty emulation."
        ),
        required_solver_xml_controls=(
            ("Use_cut_metadata_scale", "true"),
            ("Cut_cell_metadata_scale_cap", "1"),
            ("Cut_cell_pressure_gradient_penalty", "1"),
        ),
        validation_scope="trend_diagnostic",
    ),
    "square-refined-metadata-cap10-t03-pressure": Probe(
        name="square-refined-metadata-cap10-t03-pressure",
        case_subdir="square_tank_tilt_settling",
        generate_args=(
            "--nx", "17",
            "--ny", "17",
            "--time-step", "0.0005",
            "--time-steps", "600",
            "--verification-profile", "transient_pressure",
            "--use-cut-metadata-scale",
            "--cut-cell-metadata-scale-cap", "10.0",
        ),
        result_name=LATEST_RESULT_NAME,
        expect_solver_success=False,
        expect_verifier_pass=None,
        description=(
            "Refined square-tank bounded metadata-scale cap diagnostic through "
            "t=0.3. Current evidence expects this moderate local scale cap to "
            "fail near t=0.2 while remaining fallback-free, bracketing the "
            "cap range between the successful unit-cap pressure diagnostic and "
            "the unbounded metadata-scale nonlinear failure."
        ),
        required_solver_xml_controls=(
            ("Use_cut_metadata_scale", "true"),
            ("Cut_cell_metadata_scale_cap", "10"),
            ("Cut_cell_pressure_gradient_penalty", "1"),
        ),
        validation_scope="expected_failure_diagnostic",
    ),
    "square-refined-metadata-cap100-t03-pressure": Probe(
        name="square-refined-metadata-cap100-t03-pressure",
        case_subdir="square_tank_tilt_settling",
        generate_args=(
            "--nx", "17",
            "--ny", "17",
            "--time-step", "0.0005",
            "--time-steps", "600",
            "--verification-profile", "transient_pressure",
            "--use-cut-metadata-scale",
            "--cut-cell-metadata-scale-cap", "100.0",
        ),
        result_name=LATEST_RESULT_NAME,
        expect_solver_success=False,
        expect_verifier_pass=None,
        description=(
            "Refined square-tank bounded metadata-scale cap diagnostic through "
            "t=0.3. Current evidence expects this stronger local scale cap to "
            "fail near t=0.2 while remaining fallback-free, confirming that "
            "raising the metadata-scale cap reintroduces the small-cut "
            "nonlinear blocker before it closes the pressure-quality gate."
        ),
        required_solver_xml_controls=(
            ("Use_cut_metadata_scale", "true"),
            ("Cut_cell_metadata_scale_cap", "100"),
            ("Cut_cell_pressure_gradient_penalty", "1"),
        ),
        validation_scope="expected_failure_diagnostic",
    ),
    "square-refined-pressure-penalty0-t03-pressure": Probe(
        name="square-refined-pressure-penalty0-t03-pressure",
        case_subdir="square_tank_tilt_settling",
        generate_args=(
            "--nx", "17",
            "--ny", "17",
            "--time-step", "0.0005",
            "--time-steps", "600",
            "--verification-profile", "transient_pressure",
            "--cut-cell-pressure-gradient-penalty", "0.0",
        ),
        result_name="result_600.vtu",
        expect_solver_success=True,
        expect_verifier_pass=None,
        description=(
            "Refined square-tank pressure-ghost-penalty isolation through "
            "t=0.3. This keeps the default unscaled velocity ghost penalty "
            "active while disabling the pressure gradient facet term to test "
            "whether pressure stabilization is the remaining interface-pressure "
            "quality blocker."
        ),
        required_solver_xml_controls=(
            ("Use_cut_metadata_scale", "false"),
            ("Cut_cell_pressure_gradient_penalty", "0"),
        ),
        validation_scope="trend_diagnostic",
    ),
    "square-refined-pressure-penalty0p1-t03-pressure": Probe(
        name="square-refined-pressure-penalty0p1-t03-pressure",
        case_subdir="square_tank_tilt_settling",
        generate_args=(
            "--nx", "17",
            "--ny", "17",
            "--time-step", "0.0005",
            "--time-steps", "600",
            "--verification-profile", "transient_pressure",
            "--cut-cell-pressure-gradient-penalty", "0.1",
        ),
        result_name="result_600.vtu",
        expect_solver_success=True,
        expect_verifier_pass=None,
        description=(
            "Refined square-tank intermediate pressure-ghost-penalty scaling "
            "through t=0.3. This keeps the default unscaled velocity ghost "
            "penalty active while reducing the pressure gradient facet "
            "coefficient to test whether pressure stabilization can be scaled "
            "without losing hydrostatic pressure recovery."
        ),
        required_solver_xml_controls=(
            ("Use_cut_metadata_scale", "false"),
            ("Cut_cell_pressure_gradient_penalty", "0.1"),
        ),
        validation_scope="trend_diagnostic",
    ),
    "square-refined-pressure-penalty0p01-t03-pressure": Probe(
        name="square-refined-pressure-penalty0p01-t03-pressure",
        case_subdir="square_tank_tilt_settling",
        generate_args=(
            "--nx", "17",
            "--ny", "17",
            "--time-step", "0.0005",
            "--time-steps", "600",
            "--verification-profile", "transient_pressure",
            "--cut-cell-pressure-gradient-penalty", "0.01",
        ),
        result_name="result_600.vtu",
        expect_solver_success=True,
        expect_verifier_pass=None,
        description=(
            "Refined square-tank low pressure-ghost-penalty scaling through "
            "t=0.3. This extends the pressure-stabilization sweep by one "
            "decade below 0.1 while preserving strict hydrostatic pressure and "
            "free-surface pressure diagnostics."
        ),
        required_solver_xml_controls=(
            ("Use_cut_metadata_scale", "false"),
            ("Cut_cell_pressure_gradient_penalty", "0.01"),
        ),
        validation_scope="trend_diagnostic",
    ),
    "square-refined-pressure-penalty1e-3-t03-pressure": Probe(
        name="square-refined-pressure-penalty1e-3-t03-pressure",
        case_subdir="square_tank_tilt_settling",
        generate_args=(
            "--nx", "17",
            "--ny", "17",
            "--time-step", "0.0005",
            "--time-steps", "600",
            "--verification-profile", "transient_pressure",
            "--cut-cell-pressure-gradient-penalty", "0.001",
        ),
        result_name=LATEST_RESULT_NAME,
        expect_solver_success=False,
        expect_verifier_pass=None,
        description=(
            "Refined square-tank very-low pressure-ghost-penalty scaling "
            "through t=0.3. Current evidence expects the solver to fail before "
            "the target after several hundred accepted half-steps, bracketing "
            "the point where pressure stabilization becomes too weak relative "
            "to the successful-but-still-high 0.01 run and the pressure-"
            "destroying zero-penalty run."
        ),
        required_solver_xml_controls=(
            ("Use_cut_metadata_scale", "false"),
            ("Cut_cell_pressure_gradient_penalty", "0.001"),
        ),
        validation_scope="expected_failure_diagnostic",
    ),
    "square-refined-ptc-g1": Probe(
        name="square-refined-ptc-g1",
        case_subdir="square_tank_tilt_settling",
        generate_args=(
            "--nx", "17",
            "--ny", "17",
            "--time-step", "0.001",
            "--time-steps", "400",
            "--verification-profile", "early_transient",
        ),
        result_name="result_400.vtu",
        expect_solver_success=False,
        expect_verifier_pass=None,
        description=(
            "Refined square-tank always-on Newton PTC diagnostic. Current evidence "
            "expects it to fail at the fixed-step topology event while reducing "
            "the nonlinear residual relative to the default path."
        ),
        solver_env=(
            ("SVMP_NEWTON_PTC", "1"),
            ("SVMP_NEWTON_PTC_ACTIVATE_ON_LINEAR_FAILURE", "0"),
            ("SVMP_NEWTON_PTC_GAMMA_INITIAL", "1.0"),
            ("SVMP_NEWTON_PTC_GAMMA_MAX", "10000"),
        ),
        validation_scope="expected_failure_diagnostic",
    ),
    "square-refined-p2th-linearcorner": Probe(
        name="square-refined-p2th-linearcorner",
        case_subdir="square_tank_tilt_settling",
        generate_args=(
            "--nx", "17",
            "--ny", "17",
            "--element-order", "2",
            "--fluid-taylor-hood",
            "--time-step", "0.001",
            "--time-steps", "400",
            "--verification-profile", "early_transient",
        ),
        result_name="result_400.vtu",
        expect_solver_success=False,
        expect_verifier_pass=None,
        description=(
            "Refined square-tank P2/P1 diagnostic on a quadratic background mesh. "
            "Current evidence expects fallback-free LinearCorner cut contexts "
            "after high-order-cell normal alignment, followed by nonlinear "
            "failure at the early fixed-step transient gate."
        ),
        validation_scope="expected_failure_diagnostic",
    ),
}


def parse_float(text: str) -> float | None:
    try:
        value = float(text)
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def parse_key_values(line: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for key, raw in KEY_VALUE_RE.findall(line):
        values[key] = raw.strip("'\";")
    return values


def parse_int_value(values: dict[str, str], key: str) -> int | None:
    if key not in values:
        return None
    try:
        return int(float(values[key]))
    except ValueError:
        return None


def parse_float_value(values: dict[str, str], key: str) -> float | None:
    if key not in values:
        return None
    return parse_float(values[key])


def compact_cut_context(values: dict[str, str]) -> dict[str, Any]:
    keys = (
        "provenance",
        "solution_source",
        "generated_interface_geometry",
        "implicit_cut_quadrature_backend",
        "geometry_tangent_policy",
        "implicit_cut_fallback_policy",
        "active_side",
        "retained_volume_sides",
        "active_cut_cells",
        "active_full_wet_cells",
        "active_full_dry_cells",
        "interface_fragments",
        "active_interface_fragments",
        "active_volume_regions",
        "active_quadrature_points",
        "active_min_volume_fraction",
        "active_max_volume_fraction",
        "achieved_interface_quadrature_order",
        "achieved_volume_quadrature_order",
        "implicit_cut_fallback_cells",
        "implicit_cut_backend_seconds",
        "implicit_cut_backend_seconds_min",
        "implicit_cut_backend_seconds_mean",
        "implicit_cut_backend_seconds_max",
        "implicit_cut_backend_internal_seconds",
        "implicit_cut_backend_internal_seconds_total",
        "backend_total_quadrature_point_count",
        "terminal_topology_refinements",
        "max_terminal_topology_extra_depth",
        "process_vm_kb",
        "process_rss_kb",
        "basis_cache_entries",
        "cut_context_revision",
        "cut_context_topology_key",
    )
    record: dict[str, Any] = {}
    for key in keys:
        if key not in values:
            continue
        if key in {
            "active_min_volume_fraction",
            "active_max_volume_fraction",
            "implicit_cut_backend_seconds",
            "implicit_cut_backend_seconds_min",
            "implicit_cut_backend_seconds_mean",
            "implicit_cut_backend_seconds_max",
            "implicit_cut_backend_internal_seconds",
            "implicit_cut_backend_internal_seconds_total",
        }:
            record[key] = parse_float_value(values, key)
        elif key in {
            "active_cut_cells",
            "active_full_wet_cells",
            "active_full_dry_cells",
            "interface_fragments",
            "active_interface_fragments",
            "active_volume_regions",
            "active_quadrature_points",
            "achieved_interface_quadrature_order",
            "achieved_volume_quadrature_order",
            "implicit_cut_fallback_cells",
            "backend_total_quadrature_point_count",
            "terminal_topology_refinements",
            "max_terminal_topology_extra_depth",
            "process_vm_kb",
            "process_rss_kb",
            "basis_cache_entries",
            "cut_context_revision",
        }:
            record[key] = parse_int_value(values, key)
        else:
            record[key] = values[key]
    return record


def compact_cut_volume_assembly(values: dict[str, str]) -> dict[str, Any]:
    record: dict[str, Any] = {}
    for key in (
        "side",
        "active_wet_volume",
        "cut_cell_active_wet_volume",
        "full_cell_active_wet_volume",
        "rules",
        "cut_cell_rules",
        "full_cell_rules",
        "quadrature_points",
        "null_rules",
        "zero_quadrature_rules",
        "nonfinite_measure_rules",
        "negative_measure_rules",
        "nonfinite_volume_fraction_rules",
        "min_volume_fraction",
        "max_volume_fraction",
        "min_exact_order",
        "max_exact_order",
    ):
        if key not in values:
            continue
        if key in {
            "active_wet_volume",
            "cut_cell_active_wet_volume",
            "full_cell_active_wet_volume",
            "min_volume_fraction",
            "max_volume_fraction",
        }:
            record[key] = parse_float_value(values, key)
        elif key == "side":
            record[key] = values[key]
        else:
            record[key] = parse_int_value(values, key)
    return record


def compact_wet_volume_diagnostic(values: dict[str, str]) -> dict[str, Any]:
    record: dict[str, Any] = {}
    for key in (
        "step",
        "time",
        "field",
        "domain_id",
        "marker",
        "active_side",
        "isovalue",
        "wet_volume",
        "wet_volume_frame",
        "reference_wet_volume",
        "physical_wet_volume",
        "initial_wet_volume",
        "wet_volume_drift",
        "relative_wet_volume_drift",
        "volume_rule_count",
        "physical_volume_rule_count",
        "skipped_physical_volume_rule_count",
        "cut_cell_count",
        "full_wet_cell_count",
        "full_dry_cell_count",
    ):
        if key not in values:
            continue
        if key in {"field", "domain_id", "active_side", "wet_volume_frame"}:
            record[key] = values[key]
        elif key in {"step", "marker", "volume_rule_count", "physical_volume_rule_count", "skipped_physical_volume_rule_count", "cut_cell_count", "full_wet_cell_count", "full_dry_cell_count"}:
            record[key] = parse_int_value(values, key)
        else:
            record[key] = parse_float_value(values, key)
    return record


def parse_component_norms(line: str) -> dict[str, dict[str, float | None]]:
    marker = "diagnostic=vector_component_norms"
    marker_index = line.find(marker)
    if marker_index >= 0:
        first_component = line.find(" [", marker_index)
        if first_component >= 0:
            line = line[first_component + 1:]
    components: dict[str, dict[str, float | None]] = {}
    for match in COMPONENT_NORM_RE.finditer(line):
        components[match.group("name")] = {
            "norm": parse_float(match.group("norm")),
            "mean": parse_float(match.group("mean")),
            "min": parse_float(match.group("min")),
            "max": parse_float(match.group("max")),
        }
    return components


def timing_key(label: str) -> str:
    key = label.strip().lower().replace("+", "plus")
    key = re.sub(r"[^a-z0-9]+", "_", key).strip("_")
    return f"{key}_seconds"


def update_max_metric(metrics: dict[str, Any], key: str, value: int | float | None) -> None:
    if value is None:
        return
    previous = metrics.get(key)
    metrics[key] = value if previous is None else max(previous, value)


def snapshot_step_context(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "nonlinear": copy.deepcopy(metrics.get("last_nonlinear")),
        "cut_context": copy.deepcopy(
            metrics.get("last_accepted_cut_context")
            or metrics.get("last_cut_context")
        ),
        "cut_volume_assembly_by_side": copy.deepcopy(
            metrics.get("last_cut_volume_assembly_by_side") or {}
        ),
        "vector_component_norms_by_label": copy.deepcopy(
            metrics.get("last_vector_component_norms_by_label") or {}
        ),
        "residual_block_norms": copy.deepcopy(metrics.get("last_residual_block_norms")),
        "line_search_residual_block_norms": copy.deepcopy(
            metrics.get("last_line_search_residual_block_norms")
        ),
        "active_pressure_constraint_refresh": copy.deepcopy(
            metrics.get("last_active_pressure_constraint_refresh")
        ),
        "active_side_vertex_constraint": copy.deepcopy(
            metrics.get("last_active_side_vertex_constraint")
        ),
        "newton_timing": copy.deepcopy(metrics.get("last_newton_timing")),
        "process_memory": copy.deepcopy(metrics.get("last_process_memory")),
    }


def upsert_step_history_record(
    records_by_step: dict[int, dict[str, Any]],
    step: int,
    **fields: Any,
) -> dict[str, Any]:
    record = records_by_step.setdefault(step, {"step": step})
    for key, value in fields.items():
        if value is not None:
            record[key] = value
    return record


def append_step_context(
    record: dict[str, Any],
    metrics: dict[str, Any],
) -> None:
    record.update(snapshot_step_context(metrics))


def summarize_step_history(records: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "count": len(records),
        "records": records,
    }
    if not records:
        return summary
    summary["first_record"] = records[0]
    summary["final_record"] = records[-1]
    min_volume = None
    max_fallback = 0
    max_backend_points = 0
    max_interface_fragments = 0
    max_cut_cells = 0
    max_relative_wet_volume_drift = None
    max_newton_iterations = 0
    max_linear_iterations = 0
    accepted_dt_values: list[float] = []
    accepted_time_values: list[float] = []
    for record in records:
        dt = record.get("dt")
        if isinstance(dt, (int, float)):
            accepted_dt_values.append(float(dt))
        time = record.get("time")
        if isinstance(time, (int, float)):
            accepted_time_values.append(float(time))
        cut_context = record.get("cut_context") or {}
        if isinstance(cut_context, dict):
            volume = cut_context.get("active_min_volume_fraction")
            if isinstance(volume, (int, float)):
                min_volume = volume if min_volume is None else min(min_volume, volume)
            fallback = cut_context.get("implicit_cut_fallback_cells")
            if isinstance(fallback, int):
                max_fallback = max(max_fallback, fallback)
            points = cut_context.get("backend_total_quadrature_point_count")
            if isinstance(points, int):
                max_backend_points = max(max_backend_points, points)
            fragments = cut_context.get("interface_fragments")
            if isinstance(fragments, int):
                max_interface_fragments = max(max_interface_fragments, fragments)
            cut_cells = cut_context.get("active_cut_cells")
            if isinstance(cut_cells, int):
                max_cut_cells = max(max_cut_cells, cut_cells)
        wet_volume = record.get("wet_volume") or {}
        if isinstance(wet_volume, dict):
            drift = wet_volume.get("relative_wet_volume_drift")
            if isinstance(drift, (int, float)):
                abs_drift = abs(drift)
                max_relative_wet_volume_drift = (
                    abs_drift
                    if max_relative_wet_volume_drift is None
                    else max(max_relative_wet_volume_drift, abs_drift)
                )
        newton_timing = record.get("newton_timing") or {}
        if isinstance(newton_timing, dict):
            newton_iterations = newton_timing.get("newton_iterations")
            if isinstance(newton_iterations, int):
                max_newton_iterations = max(max_newton_iterations, newton_iterations)
            linear_iterations = newton_timing.get("linear_iterations")
            if isinstance(linear_iterations, int):
                max_linear_iterations = max(max_linear_iterations, linear_iterations)

    summary["min_active_volume_fraction"] = min_volume
    summary["max_implicit_cut_fallback_cells"] = max_fallback
    summary["max_backend_total_quadrature_point_count"] = max_backend_points
    summary["max_interface_fragments"] = max_interface_fragments
    summary["max_active_cut_cells"] = max_cut_cells
    summary["max_abs_relative_wet_volume_drift"] = max_relative_wet_volume_drift
    summary["max_newton_iterations"] = max_newton_iterations
    summary["max_linear_iterations"] = max_linear_iterations
    if accepted_dt_values:
        summary["min_accepted_dt"] = min(accepted_dt_values)
        summary["max_accepted_dt"] = max(accepted_dt_values)
    if accepted_time_values:
        summary["final_accepted_time"] = accepted_time_values[-1]
    return summary


def summarize_rejected_step_history(records: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "count": len(records),
    }
    if not records:
        return summary
    dt_values = [
        float(record["dt"])
        for record in records
        if isinstance(record.get("dt"), (int, float))
    ]
    residual_values = [
        float(record["nonlinear"]["residual"])
        for record in records
        if isinstance(record.get("nonlinear"), dict)
        and isinstance(record["nonlinear"].get("residual"), (int, float))
    ]
    summary["first_record"] = records[0]
    summary["final_record"] = records[-1]
    if dt_values:
        summary["min_rejected_dt"] = min(dt_values)
        summary["max_rejected_dt"] = max(dt_values)
    if residual_values:
        summary["min_rejected_residual"] = min(residual_values)
        summary["max_rejected_residual"] = max(residual_values)
    return summary


def parse_backend_failure(line: str) -> dict[str, Any]:
    source = line
    reason_marker = "reason='"
    if reason_marker in line:
        start = line.find(reason_marker) + len(reason_marker)
        end = line.rfind("'")
        if end > start:
            source = line[start:end]
    values = parse_key_values(source)
    backend_failure: dict[str, Any] = {
        "raw": line[-4000:],
        "backend": values.get("backend"),
        "element_type": values.get("element_type"),
        "status": values.get("status"),
        "fallback_used": values.get("fallback_used"),
        "high_order_downgrade": values.get("high_order_downgrade"),
        "achieved_interface_order": values.get("achieved_interface_order"),
        "achieved_volume_order": values.get("achieved_volume_order"),
        "max_depth_limit": values.get("max_depth_limit"),
        "max_depth_reached": values.get("max_depth_reached"),
        "terminal_topology_refinements": values.get("terminal_topology_refinements"),
        "max_terminal_topology_extra_depth": values.get("max_terminal_topology_extra_depth"),
        "curved_fragment_failures": values.get("curved_fragment_failures"),
        "curved_edge_root_mismatches": values.get("curved_edge_root_mismatches"),
    }
    if "cell" in values:
        try:
            backend_failure["cell"] = int(float(values["cell"]))
        except ValueError:
            backend_failure["cell"] = values["cell"]
    return backend_failure


def copy_case(source: Path, target: Path) -> None:
    ignore = shutil.ignore_patterns(
        "__pycache__",
        "*.pyc",
        "result_*.vtu",
        "result_*.pvtu",
        "result.pvd",
        "solver_run*.log",
        "run_*.log",
        "verify_result*.json",
        "solver_record_each_step_linear.xml",
    )
    shutil.copytree(source, target, ignore=ignore)


def parse_cmake_cache(cache_path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not cache_path.exists():
        return values
    wanted = set(CMAKE_CACHE_PROVENANCE_KEYS)
    for line in cache_path.read_text(errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith(("//", "#")) or "=" not in line:
            continue
        raw_key, value = line.split("=", 1)
        key = raw_key.split(":", 1)[0]
        if key in wanted:
            values[key] = value
    return values


def collect_solver_metadata(solver: Path) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "path": str(solver),
        "launcher": "direct",
        "mpi_ranks": DEFAULT_MPI_RANKS,
        "mpi_policy": DEFAULT_MPI_POLICY,
    }
    try:
        stat = solver.stat()
    except OSError as exc:
        metadata["stat_error"] = str(exc)
        return metadata
    metadata["size_bytes"] = stat.st_size
    metadata["mtime_epoch_seconds"] = stat.st_mtime

    build_dir = solver.parent.parent if solver.parent.name == "bin" else solver.parent
    cache_path = build_dir / "CMakeCache.txt"
    metadata["build_dir"] = str(build_dir)
    metadata["cmake_cache"] = str(cache_path) if cache_path.exists() else None
    metadata["cmake_cache_values"] = parse_cmake_cache(cache_path)
    return metadata


def solver_feature_enabled(value: str | None) -> bool:
    return value is not None and value.strip().upper() in {"1", "ON", "TRUE", "YES"}


def solver_feature_preflight(
    solver_metadata: dict[str, Any],
    required_features: tuple[str, ...],
) -> dict[str, Any]:
    cache_values = solver_metadata.get("cmake_cache_values") or {}
    values = {key: cache_values.get(key) for key in required_features}
    missing_or_disabled = [
        key
        for key, value in values.items()
        if not solver_feature_enabled(value)
    ]
    return {
        "required_solver_features": list(required_features),
        "feature_values": values,
        "passed": not missing_or_disabled,
        "missing_or_disabled": missing_or_disabled,
    }


def probe_launcher(probe: Probe, solver_metadata: dict[str, Any]) -> str:
    if probe.mpi_ranks > 1:
        return "mpiexec"
    return solver_metadata.get("launcher") or "direct"


def probe_solver_command(
    probe: Probe,
    *,
    solver: Path,
    solver_metadata: dict[str, Any],
) -> list[str]:
    if probe.mpi_ranks <= 1:
        return [str(solver), "solver.xml"]
    cache_values = solver_metadata.get("cmake_cache_values") or {}
    mpiexec = cache_values.get("MPIEXEC_EXECUTABLE") or "mpiexec"
    np_flag = cache_values.get("MPIEXEC_NUMPROC_FLAG") or "-n"
    return [mpiexec, np_flag, str(probe.mpi_ranks), str(solver), "solver.xml"]


def direct_child(element: ET.Element, tag_name: str) -> ET.Element | None:
    for child in list(element):
        if strip_xml_namespace(child.tag) == tag_name:
            return child
    return None


def patch_solver_xml_for_probe(case_dir: Path, probe: Probe) -> list[dict[str, Any]]:
    if probe.mpi_ranks <= 1:
        return []

    xml_path = case_dir / "solver.xml"
    if not xml_path.exists():
        return []

    tree = ET.parse(xml_path)
    root = tree.getroot()
    patches: list[dict[str, Any]] = []

    for equation in root.iter():
        if strip_xml_namespace(equation.tag) != "Add_equation":
            continue
        equation_type = equation.attrib.get("type", "")
        solver_element = direct_child(equation, "LS")
        if solver_element is None:
            continue

        equation_type_key = equation_type.lower()
        desired_solver_type = "NS" if equation_type_key == "fluid" else "GMRES"
        old_solver_type = solver_element.attrib.get("type", "")
        if old_solver_type.lower() != desired_solver_type.lower():
            solver_element.set("type", desired_solver_type)
            patches.append(
                {
                    "equation_type": equation_type,
                    "control": "LS@type",
                    "old": old_solver_type,
                    "new": desired_solver_type,
                    "reason": "mpi_validation_requires_distributed_backend",
                }
            )

        linear_algebra = direct_child(solver_element, "Linear_algebra")
        if linear_algebra is None:
            linear_algebra = ET.SubElement(solver_element, "Linear_algebra")
            patches.append(
                {
                    "equation_type": equation_type,
                    "control": "Linear_algebra",
                    "old": None,
                    "new": "created",
                    "reason": "mpi_validation_requires_distributed_backend",
                }
            )

        old_backend = linear_algebra.attrib.get("type", "")
        if old_backend.lower() != "fsils":
            linear_algebra.set("type", "fsils")
            patches.append(
                {
                    "equation_type": equation_type,
                    "control": "Linear_algebra@type",
                    "old": old_backend,
                    "new": "fsils",
                    "reason": "mpi_validation_requires_distributed_backend",
                }
            )

        preconditioner = direct_child(linear_algebra, "Preconditioner")
        if preconditioner is None:
            preconditioner = ET.SubElement(linear_algebra, "Preconditioner")
            old_preconditioner = None
        else:
            old_preconditioner = (preconditioner.text or "").strip()
        if old_preconditioner != "fsils":
            preconditioner.text = "fsils"
            patches.append(
                {
                    "equation_type": equation_type,
                    "control": "Preconditioner",
                    "old": old_preconditioner,
                    "new": "fsils",
                    "reason": "mpi_validation_requires_distributed_backend",
                }
            )

    if patches:
        tree.write(xml_path, encoding="UTF-8", xml_declaration=True)
    return patches


def timeout_interpretation(timeout: bool, timeout_policy: str) -> str:
    if not timeout:
        return "not_observed"
    if timeout_policy == "diagnostic_timeout_not_validation_pass":
        return "expected_diagnostic_timeout_but_not_validation_pass"
    return "failing_validation_timeout"


def qualify_probe_summary(summary: dict[str, Any]) -> dict[str, Any]:
    preflight = summary.get("preflight") or {}
    generate = summary.get("generate")
    control_check = summary.get("solver_xml_control_check") or {}
    solver = summary.get("solver") or {}
    scope = summary.get("validation_scope")
    meets_expectation = bool(summary.get("meets_expectation"))
    solver_timeout = bool(solver.get("timeout"))
    counts_as_validation_pass = False

    if not preflight.get("passed", True):
        outcome = "preflight_failed"
    elif generate is None or generate.get("status") != 0 or generate.get("timeout"):
        outcome = "setup_failed"
    elif not control_check.get("passed", True):
        outcome = "setup_failed"
    elif not meets_expectation:
        if scope == "open_performance_target" and solver_timeout:
            outcome = "open_performance_target_timeout_diagnostic"
        elif scope == "open_performance_target":
            outcome = "open_performance_target_unmet"
        else:
            outcome = "unexpected_result"
    elif scope == "regression_gate":
        outcome = "validation_pass"
        counts_as_validation_pass = True
    elif scope == "expected_failure_diagnostic":
        outcome = "expected_failure_diagnostic_matched"
    elif scope == "trend_diagnostic":
        outcome = "trend_diagnostic_recorded"
    elif scope == "open_performance_target":
        outcome = "open_performance_target_observed_success_not_broad_gate"
    else:
        outcome = "diagnostic_recorded"

    summary["qualification_outcome"] = outcome
    summary["counts_as_validation_pass"] = counts_as_validation_pass
    summary["broad_gate_closure_permitted"] = (
        counts_as_validation_pass
        and not solver_timeout
        and not (solver.get("timeout_interpretation") or "").startswith("expected_diagnostic_timeout")
    )
    return summary


def summarize_qualification(probe_summaries: list[dict[str, Any]]) -> dict[str, Any]:
    outcomes: dict[str, int] = {}
    validation_passes: list[str] = []
    diagnostics: list[str] = []
    unexpected: list[str] = []
    broad_gate_closure_violations: list[str] = []
    for item in probe_summaries:
        outcome = item.get("qualification_outcome", "missing_outcome")
        outcomes[outcome] = outcomes.get(outcome, 0) + 1
        name = item.get("probe", "<unknown>")
        if item.get("counts_as_validation_pass"):
            validation_passes.append(name)
        else:
            diagnostics.append(name)
        if outcome in {"unexpected_result", "setup_failed", "preflight_failed", "open_performance_target_unmet"}:
            unexpected.append(name)
        if item.get("counts_as_validation_pass") and not item.get("broad_gate_closure_permitted"):
            broad_gate_closure_violations.append(name)
    return {
        "outcomes": outcomes,
        "validation_pass_probes": validation_passes,
        "non_validation_diagnostic_or_open_probes": diagnostics,
        "unexpected_or_unmet_probes": unexpected,
        "broad_gate_closure_violations": broad_gate_closure_violations,
        "all_broad_gate_closures_are_complete_runs": not broad_gate_closure_violations,
    }


def final_verifier_history_record(probe_summary: dict[str, Any]) -> dict[str, Any]:
    history = probe_summary.get("verifier_history") or {}
    if isinstance(history.get("final_record"), dict):
        return history["final_record"]
    verifier = probe_summary.get("verifier") or {}
    metrics = verifier.get("metrics") or {}
    return metrics if isinstance(metrics, dict) else {}


def verifier_history_records(probe_summary: dict[str, Any]) -> list[dict[str, Any]]:
    history = probe_summary.get("verifier_history") or {}
    records = history.get("records")
    if not isinstance(records, list):
        return []
    return [record for record in records if isinstance(record, dict)]


def metric_trend(
    values_by_axis: dict[int, float],
    *,
    axis_label: str = "nx",
    trend_label: str = "refinement",
) -> dict[str, Any]:
    ordered = sorted(values_by_axis)
    magnitudes = [abs(values_by_axis[key]) for key in ordered]
    trend_magnitudes = [
        0.0 if value <= MMS_REFINEMENT_TREND_ZERO_TOL else value
        for value in magnitudes
    ]
    nonincreasing = all(
        trend_magnitudes[index + 1] <= trend_magnitudes[index] * (1.0 + 1.0e-12)
        for index in range(len(trend_magnitudes) - 1)
    )
    ratios = {}
    for index, (left, right) in enumerate(zip(ordered, ordered[1:])):
        denominator = trend_magnitudes[index + 1]
        numerator = trend_magnitudes[index]
        ratios[f"{left}_to_{right}"] = (
            None if denominator == 0.0 else numerator / denominator
        )
    return {
        f"values_by_{axis_label}": {
            str(key): values_by_axis[key] for key in ordered
        },
        f"absolute_values_by_{axis_label}": {
            str(key): magnitudes[index] for index, key in enumerate(ordered)
        },
        "trend_zero_tolerance": MMS_REFINEMENT_TREND_ZERO_TOL,
        f"nonincreasing_with_{trend_label}": nonincreasing,
        "coarse_to_fine_ratios": ratios,
    }


def summarize_history_refinement_trends(
    present: dict[int, dict[str, Any]],
    probes_by_nx: dict[int, str],
) -> dict[str, Any]:
    records_by_nx = {
        nx: verifier_history_records(probe_summary)
        for nx, probe_summary in present.items()
    }
    history_counts = {str(nx): len(records) for nx, records in sorted(records_by_nx.items())}
    complete = set(present) == set(probes_by_nx)
    can_evaluate_refinement = complete and len(present) >= 2
    common_count = min(history_counts.values()) if history_counts else 0
    record_trends: list[dict[str, Any]] = []
    for record_index in range(common_count):
        metric_summaries: dict[str, Any] = {}
        for metric in MMS_REFINEMENT_TREND_METRICS:
            values: dict[int, float] = {}
            for nx, records in records_by_nx.items():
                value = records[record_index].get(metric)
                if isinstance(value, (int, float)) and math.isfinite(float(value)):
                    values[nx] = float(value)
            if len(values) >= 2:
                metric_summaries[metric] = metric_trend(values)
        coherent_metrics = [
            metric
            for metric, summary in metric_summaries.items()
            if summary.get("nonincreasing_with_refinement")
        ]
        incoherent_metrics = [
            metric
            for metric, summary in metric_summaries.items()
            if not summary.get("nonincreasing_with_refinement")
        ]
        refinement_evaluated = can_evaluate_refinement and bool(metric_summaries)
        record_trends.append(
            {
                "history_index": record_index,
                "time_by_nx": {
                    str(nx): records[record_index].get("time")
                    for nx, records in sorted(records_by_nx.items())
                },
                "passed_by_nx": {
                    str(nx): bool(records[record_index].get("passed"))
                    for nx, records in sorted(records_by_nx.items())
                },
                "metrics": metric_summaries,
                "coherent_metrics": coherent_metrics,
                "incoherent_metrics": incoherent_metrics,
                "refinement_evaluated": refinement_evaluated,
                "coherent_refinement_gate": refinement_evaluated and not incoherent_metrics,
            }
        )

    coherent_records = [
        record for record in record_trends if record.get("coherent_refinement_gate")
    ]
    incoherent_records = [
        record
        for record in record_trends
        if record.get("refinement_evaluated")
        and not record.get("coherent_refinement_gate")
    ]
    first_incoherent = incoherent_records[0] if incoherent_records else None
    return {
        "history_record_count_by_nx": history_counts,
        "common_history_record_count": common_count,
        "refinement_evaluated": can_evaluate_refinement,
        "not_evaluated_history_record_count": len(record_trends)
        - len(coherent_records)
        - len(incoherent_records),
        "coherent_history_record_count": len(coherent_records),
        "incoherent_history_record_count": len(incoherent_records),
        "first_incoherent_history_record": {
            "history_index": first_incoherent.get("history_index"),
            "time_by_nx": first_incoherent.get("time_by_nx"),
            "incoherent_metrics": first_incoherent.get("incoherent_metrics"),
        } if first_incoherent else None,
        "history_record_trends": record_trends,
    }


def summarize_mms_refinement_trends(probe_summaries: list[dict[str, Any]]) -> dict[str, Any] | None:
    by_name = {
        summary.get("probe"): summary
        for summary in probe_summaries
        if summary.get("probe")
    }
    groups: dict[str, Any] = {}
    any_group = False
    for family, probes_by_nx in MMS_REFINEMENT_TREND_PROBES.items():
        present = {
            nx: by_name[probe_name]
            for nx, probe_name in probes_by_nx.items()
            if probe_name in by_name
        }
        if not present:
            continue
        any_group = True
        metric_summaries: dict[str, Any] = {}
        for metric in MMS_REFINEMENT_TREND_METRICS:
            values: dict[int, float] = {}
            for nx, probe_summary in present.items():
                final_record = final_verifier_history_record(probe_summary)
                value = final_record.get(metric)
                if isinstance(value, (int, float)) and math.isfinite(float(value)):
                    values[nx] = float(value)
            if len(values) >= 2:
                metric_summaries[metric] = metric_trend(values)
        complete = set(present) == set(probes_by_nx)
        refinement_evaluated = complete and bool(metric_summaries)
        coherent_metrics = [
            metric
            for metric, summary in metric_summaries.items()
            if summary.get("nonincreasing_with_refinement")
        ]
        incoherent_metrics = [
            metric
            for metric, summary in metric_summaries.items()
            if not summary.get("nonincreasing_with_refinement")
        ]
        primary_present_metrics = [
            metric
            for metric in MMS_REFINEMENT_PRIMARY_GATE_METRICS
            if metric in metric_summaries
        ]
        primary_incoherent_metrics = [
            metric
            for metric in primary_present_metrics
            if not metric_summaries[metric].get("nonincreasing_with_refinement")
        ]
        primary_refinement_evaluated = (
            refinement_evaluated
            and bool(primary_present_metrics)
            and set(primary_present_metrics) == set(MMS_REFINEMENT_PRIMARY_GATE_METRICS)
        )
        groups[family] = {
            "complete_nx2_nx3_nx4": complete,
            "complete_requested_resolutions": complete,
            "requested_nx": sorted(probes_by_nx),
            "present_nx": sorted(present),
            "probes": {str(nx): probes_by_nx[nx] for nx in sorted(present)},
            "solver_success_by_nx": {
                str(nx): bool((present[nx].get("solver") or {}).get("success"))
                for nx in sorted(present)
            },
            "final_verifier_pass_by_nx": {
                str(nx): bool(final_verifier_history_record(present[nx]).get("passed"))
                for nx in sorted(present)
            },
            "metrics": metric_summaries,
            "coherent_metric_count": len(coherent_metrics),
            "incoherent_metric_count": len(incoherent_metrics),
            "coherent_metrics": coherent_metrics,
            "incoherent_metrics": incoherent_metrics,
            "refinement_evaluated": refinement_evaluated,
            "coherent_refinement_gate": refinement_evaluated and not incoherent_metrics,
            "primary_gate_metrics": list(MMS_REFINEMENT_PRIMARY_GATE_METRICS),
            "primary_gate_present_metrics": primary_present_metrics,
            "primary_gate_missing_metrics": [
                metric
                for metric in MMS_REFINEMENT_PRIMARY_GATE_METRICS
                if metric not in metric_summaries
            ],
            "primary_gate_incoherent_metrics": primary_incoherent_metrics,
            "primary_gate_coherent_metric_count": (
                len(primary_present_metrics) - len(primary_incoherent_metrics)
            ),
            "primary_gate_incoherent_metric_count": len(primary_incoherent_metrics),
            "primary_refinement_evaluated": primary_refinement_evaluated,
            "primary_coherent_refinement_gate": (
                primary_refinement_evaluated and not primary_incoherent_metrics
            ),
            "history_trends": summarize_history_refinement_trends(present, probes_by_nx),
        }
    if not any_group:
        return None
    coherent_groups = [
        name
        for name, group in groups.items()
        if group.get("coherent_refinement_gate")
    ]
    incoherent_groups = [
        name
        for name, group in groups.items()
        if group.get("refinement_evaluated")
        and not group.get("coherent_refinement_gate")
    ]
    not_evaluated_groups = [
        name
        for name, group in groups.items()
        if not group.get("refinement_evaluated")
    ]
    primary_coherent_groups = [
        name
        for name, group in groups.items()
        if group.get("primary_coherent_refinement_gate")
    ]
    primary_incoherent_groups = [
        name
        for name, group in groups.items()
        if group.get("primary_refinement_evaluated")
        and not group.get("primary_coherent_refinement_gate")
    ]
    primary_not_evaluated_groups = [
        name
        for name, group in groups.items()
        if not group.get("primary_refinement_evaluated")
    ]
    return {
        "groups": groups,
        "coherent_group_count": len(coherent_groups),
        "incoherent_group_count": len(incoherent_groups),
        "not_evaluated_group_count": len(not_evaluated_groups),
        "coherent_groups": coherent_groups,
        "incoherent_groups": incoherent_groups,
        "not_evaluated_groups": not_evaluated_groups,
        "all_complete_groups_are_coherent": not incoherent_groups,
        "primary_coherent_group_count": len(primary_coherent_groups),
        "primary_incoherent_group_count": len(primary_incoherent_groups),
        "primary_not_evaluated_group_count": len(primary_not_evaluated_groups),
        "primary_coherent_groups": primary_coherent_groups,
        "primary_incoherent_groups": primary_incoherent_groups,
        "primary_not_evaluated_groups": primary_not_evaluated_groups,
        "all_complete_groups_are_primary_coherent": not primary_incoherent_groups,
    }


def probe_generate_arg(probe_name: str, option: str) -> str | None:
    args = PROBES[probe_name].generate_args
    for index, value in enumerate(args[:-1]):
        if value == option:
            return args[index + 1]
    return None


def summarize_mms_temporal_trends(probe_summaries: list[dict[str, Any]]) -> dict[str, Any] | None:
    by_name = {
        summary.get("probe"): summary
        for summary in probe_summaries
        if summary.get("probe")
    }
    groups: dict[str, Any] = {}
    any_group = False
    for family, probes_by_time_steps in MMS_TEMPORAL_TREND_PROBES.items():
        present = {
            time_steps: by_name[probe_name]
            for time_steps, probe_name in probes_by_time_steps.items()
            if probe_name in by_name
        }
        if not present:
            continue
        any_group = True
        metric_summaries: dict[str, Any] = {}
        for metric in MMS_REFINEMENT_TREND_METRICS:
            values: dict[int, float] = {}
            for time_steps, probe_summary in present.items():
                final_record = final_verifier_history_record(probe_summary)
                value = final_record.get(metric)
                if isinstance(value, (int, float)) and math.isfinite(float(value)):
                    values[time_steps] = float(value)
            if len(values) >= 2:
                metric_summaries[metric] = metric_trend(
                    values,
                    axis_label="time_steps",
                    trend_label="temporal_refinement",
                )
        complete = set(present) == set(probes_by_time_steps)
        temporal_refinement_evaluated = complete and bool(metric_summaries)
        coherent_metrics = [
            metric
            for metric, summary in metric_summaries.items()
            if summary.get("nonincreasing_with_temporal_refinement")
        ]
        incoherent_metrics = [
            metric
            for metric, summary in metric_summaries.items()
            if not summary.get("nonincreasing_with_temporal_refinement")
        ]
        groups[family] = {
            "complete_requested_time_steps": complete,
            "requested_time_steps": sorted(probes_by_time_steps),
            "present_time_steps": sorted(present),
            "probes": {
                str(time_steps): probes_by_time_steps[time_steps]
                for time_steps in sorted(present)
            },
            "time_step_by_time_steps": {
                str(time_steps): probe_generate_arg(
                    probes_by_time_steps[time_steps],
                    "--time-step",
                )
                for time_steps in sorted(present)
            },
            "final_time_by_time_steps": {
                str(time_steps): final_verifier_history_record(present[time_steps]).get("time")
                for time_steps in sorted(present)
            },
            "solver_success_by_time_steps": {
                str(time_steps): bool((present[time_steps].get("solver") or {}).get("success"))
                for time_steps in sorted(present)
            },
            "final_verifier_pass_by_time_steps": {
                str(time_steps): bool(
                    final_verifier_history_record(present[time_steps]).get("passed")
                )
                for time_steps in sorted(present)
            },
            "metrics": metric_summaries,
            "coherent_metric_count": len(coherent_metrics),
            "incoherent_metric_count": len(incoherent_metrics),
            "coherent_metrics": coherent_metrics,
            "incoherent_metrics": incoherent_metrics,
            "temporal_refinement_evaluated": temporal_refinement_evaluated,
            "coherent_temporal_refinement_gate": (
                temporal_refinement_evaluated and not incoherent_metrics
            ),
        }
    if not any_group:
        return None
    coherent_groups = [
        name
        for name, group in groups.items()
        if group.get("coherent_temporal_refinement_gate")
    ]
    incoherent_groups = [
        name
        for name, group in groups.items()
        if group.get("temporal_refinement_evaluated")
        and not group.get("coherent_temporal_refinement_gate")
    ]
    not_evaluated_groups = [
        name
        for name, group in groups.items()
        if not group.get("temporal_refinement_evaluated")
    ]
    return {
        "groups": groups,
        "coherent_group_count": len(coherent_groups),
        "incoherent_group_count": len(incoherent_groups),
        "not_evaluated_group_count": len(not_evaluated_groups),
        "coherent_groups": coherent_groups,
        "incoherent_groups": incoherent_groups,
        "not_evaluated_groups": not_evaluated_groups,
        "all_complete_groups_are_coherent": not incoherent_groups,
    }


def strip_xml_namespace(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def relative_repo_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def normalized_xml_token(text: str | None) -> str:
    if text is None:
        return ""
    return re.sub(r"[^a-z0-9]+", "", text.strip().lower())


def enabled_xml_bool(text: str | None) -> bool:
    return normalized_xml_token(text) in {"1", "true", "on", "yes"}


def direct_child_text(element: ET.Element, tag: str) -> str | None:
    for child in list(element):
        if strip_xml_namespace(child.tag) == tag:
            text = (child.text or "").strip()
            return text if text else None
    return None


def direct_child_text_any(element: ET.Element, tags: tuple[str, ...]) -> str | None:
    for tag in tags:
        value = direct_child_text(element, tag)
        if value is not None:
            return value
    return None


def direct_child_control_records(element: ET.Element) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for child in list(element):
        tag = strip_xml_namespace(child.tag)
        record: dict[str, Any] = {
            "tag": tag,
            "normalized_tag": normalized_xml_token(tag),
        }
        text = (child.text or "").strip()
        if text:
            record["value"] = text
            record["normalized_value"] = normalized_xml_token(text)
        if child.attrib:
            record["attributes"] = dict(child.attrib)
        records.append(record)
    return records


def unsupported_free_surface_scope_reason(record: dict[str, Any]) -> str | None:
    tag = str(record.get("normalized_tag") or "")
    value = str(record.get("normalized_value") or "")
    if tag in FREE_SURFACE_UNSUPPORTED_SCOPE_TAG_TOKENS:
        return "unsupported_two_phase_or_jump_scope_tag"
    if any(fragment in tag for fragment in FREE_SURFACE_UNSUPPORTED_SCOPE_TAG_FRAGMENTS):
        return "unsupported_two_phase_or_jump_scope_tag"
    if tag in FREE_SURFACE_SCOPE_MODE_TAG_TOKENS and any(
        fragment in value for fragment in FREE_SURFACE_UNSUPPORTED_SCOPE_VALUE_FRAGMENTS
    ):
        return "unsupported_two_phase_or_jump_scope_value"
    return None


def unsupported_free_surface_scope_controls(
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    unsupported: list[dict[str, Any]] = []
    for record in records:
        reason = unsupported_free_surface_scope_reason(record)
        if reason is None:
            continue
        item = {
            "tag": record["tag"],
            "reason": reason,
        }
        if "value" in record:
            item["value"] = record["value"]
        unsupported.append(item)
    return unsupported


def collect_solver_xml_controls(xml_path: Path) -> list[dict[str, Any]]:
    if not xml_path.exists():
        return []
    try:
        root = ET.parse(xml_path).getroot()
    except ET.ParseError as exc:
        return [{"path": str(xml_path), "parse_error": str(exc)}]

    records: list[dict[str, Any]] = []

    def visit(element: ET.Element, path: list[str]) -> None:
        tag = strip_xml_namespace(element.tag)
        current_path = [*path, tag]
        if tag in SOLVER_XML_CONTROL_TAGS:
            record: dict[str, Any] = {
                "path": "/".join(current_path),
                "tag": tag,
            }
            text = (element.text or "").strip()
            if text:
                record["value"] = text
            if element.attrib:
                record["attributes"] = dict(element.attrib)
            records.append(record)
        for child in list(element):
            visit(child, current_path)

    visit(root, [])
    return records


def collect_unfitted_free_surface_entries(xml_path: Path) -> tuple[list[dict[str, Any]], str | None]:
    try:
        root = ET.parse(xml_path).getroot()
    except ET.ParseError as exc:
        return [], str(exc)

    entries: list[dict[str, Any]] = []
    for element in root.iter():
        if strip_xml_namespace(element.tag) != "Add_BC":
            continue
        if direct_child_text(element, "Type") != "Free_surface":
            continue
        if direct_child_text(element, "Implementation") != "UnfittedLevelSet":
            continue
        child_controls = direct_child_control_records(element)
        entries.append(
            {
                "path": relative_repo_path(xml_path),
                "bc_name": element.attrib.get("name"),
                "level_set_field": direct_child_text(element, "Level_set_field_name"),
                "domain_id": direct_child_text(element, "Generated_interface_domain_id"),
                "active_domain": direct_child_text(element, "Active_domain"),
                "active_domain_method": direct_child_text(element, "Active_domain_method"),
                "velocity_extension_enabled": enabled_xml_bool(
                    direct_child_text(element, "Enable_velocity_extension")
                ),
                "surface_tension": direct_child_text(element, "Surface_tension"),
                "use_level_set_curvature": direct_child_text(
                    element, "Use_level_set_curvature"
                ),
                "curvature": direct_child_text(element, "Curvature"),
                "curvature_field": direct_child_text_any(
                    element, FREE_SURFACE_CURVATURE_FIELD_TAGS
                ),
                "kinematic_enforcement": direct_child_text(
                    element, "Kinematic_enforcement"
                ),
                "normal_kinematic_policy": direct_child_text(
                    element, "Normal_kinematic_policy"
                ),
                "generated_interface_geometry": (
                    direct_child_text(element, "Generated_interface_geometry")
                    or "LinearCorner(default)"
                ),
                "implicit_cut_quadrature_backend": (
                    direct_child_text(element, "Implicit_cut_quadrature_backend")
                    or "LinearCorner(default)"
                ),
                "unsupported_scope_controls": unsupported_free_surface_scope_controls(
                    child_controls
                ),
            }
        )
    return entries, None


def audit_active_domain_literals() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    paths = sorted(
        {
            *CASE_ROOT.rglob("generate_case.py"),
            *CASE_ROOT.rglob("benchmark.json"),
            *CASE_ROOT.rglob("expected_results.json"),
        }
    )
    for path in paths:
        text = path.read_text(errors="replace")
        for pattern, source in (
            (ACTIVE_DOMAIN_XML_LITERAL_RE, "xml_template"),
            (ACTIVE_DOMAIN_JSON_LITERAL_RE, "json_metadata"),
        ):
            for match in pattern.finditer(text):
                token = match.group(1).strip()
                normalized = normalized_xml_token(token)
                records.append(
                    {
                        "path": relative_repo_path(path),
                        "source": source,
                        "token": token,
                        "normalized": normalized,
                        "accepted": normalized in ACTIVE_DOMAIN_ACCEPTED_TOKENS,
                    }
                )
    return records


def audit_case_inventory() -> dict[str, Any]:
    failures: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    entries: list[dict[str, Any]] = []
    solver_xml_paths = sorted(CASE_ROOT.rglob("solver*.xml"))

    for xml_path in solver_xml_paths:
        path_entries, parse_error = collect_unfitted_free_surface_entries(xml_path)
        if parse_error:
            failures.append(
                {
                    "path": relative_repo_path(xml_path),
                    "reason": "xml_parse_error",
                    "detail": parse_error,
                }
            )
            continue
        for entry in path_entries:
            entries.append(entry)
            active_token = normalized_xml_token(entry.get("active_domain"))
            method_token = normalized_xml_token(entry.get("active_domain_method"))
            kinematic_token = normalized_xml_token(entry.get("kinematic_enforcement"))
            if active_token not in ACTIVE_DOMAIN_ACCEPTED_TOKENS:
                failures.append(
                    {
                        "path": entry["path"],
                        "bc_name": entry.get("bc_name"),
                        "reason": "unknown_active_domain",
                        "active_domain": entry.get("active_domain"),
                    }
                )
            if method_token and method_token not in ACTIVE_DOMAIN_METHOD_ACCEPTED_TOKENS:
                failures.append(
                    {
                        "path": entry["path"],
                        "bc_name": entry.get("bc_name"),
                        "reason": "unknown_active_domain_method",
                        "active_domain_method": entry.get("active_domain_method"),
                    }
                )
            if entry["velocity_extension_enabled"]:
                if not entry.get("active_domain"):
                    failures.append(
                        {
                            "path": entry["path"],
                            "bc_name": entry.get("bc_name"),
                            "reason": "velocity_extension_without_active_domain",
                        }
                    )
                if method_token != "cutvolume":
                    failures.append(
                        {
                            "path": entry["path"],
                            "bc_name": entry.get("bc_name"),
                            "reason": "velocity_extension_requires_cut_volume_method",
                            "active_domain_method": entry.get("active_domain_method"),
                        }
                    )
                if not entry.get("level_set_field") or not entry.get("domain_id"):
                    failures.append(
                        {
                            "path": entry["path"],
                            "bc_name": entry.get("bc_name"),
                            "reason": "velocity_extension_missing_level_set_field_or_domain",
                            "level_set_field": entry.get("level_set_field"),
                            "domain_id": entry.get("domain_id"),
                        }
                    )
            unsupported_scope_controls = entry.get("unsupported_scope_controls") or []
            for control in unsupported_scope_controls:
                failures.append(
                    {
                        "path": entry["path"],
                        "bc_name": entry.get("bc_name"),
                        "reason": "unsupported_two_phase_or_jump_free_surface_scope",
                        "control": control,
                    }
                )
            if kinematic_token == "nitsche":
                failures.append(
                    {
                        "path": entry["path"],
                        "bc_name": entry.get("bc_name"),
                        "reason": "unsupported_nitsche_kinematics_for_one_sided_unfitted_scope",
                        "kinematic_enforcement": entry.get("kinematic_enforcement"),
                    }
                )

            surface_tension = parse_float(entry.get("surface_tension") or "0")
            if surface_tension is not None and abs(surface_tension) > 0.0:
                use_level_set_curvature = entry.get("use_level_set_curvature")
                uses_raw_level_set_curvature = (
                    use_level_set_curvature is None
                    or enabled_xml_bool(use_level_set_curvature)
                ) and not entry.get("curvature_field")
                has_supplied_curvature = bool(
                    entry.get("curvature") or entry.get("curvature_field")
                )
                if uses_raw_level_set_curvature:
                    failures.append(
                        {
                            "path": entry["path"],
                            "bc_name": entry.get("bc_name"),
                            "reason": "nonzero_surface_tension_uses_raw_level_set_curvature",
                            "surface_tension": entry.get("surface_tension"),
                            "use_level_set_curvature": entry.get("use_level_set_curvature"),
                            "curvature_field": entry.get("curvature_field"),
                        }
                    )
                elif not has_supplied_curvature:
                    failures.append(
                        {
                            "path": entry["path"],
                            "bc_name": entry.get("bc_name"),
                            "reason": "nonzero_surface_tension_without_supplied_or_projected_curvature",
                            "surface_tension": entry.get("surface_tension"),
                            "use_level_set_curvature": entry.get("use_level_set_curvature"),
                        }
                    )
                warnings.append(
                    {
                        "path": entry["path"],
                        "bc_name": entry.get("bc_name"),
                        "reason": "nonzero_surface_tension_outside_zero_surface_tension_gate",
                        "surface_tension": entry.get("surface_tension"),
                        "curvature": entry.get("curvature"),
                        "curvature_field": entry.get("curvature_field"),
                        "use_level_set_curvature": entry.get("use_level_set_curvature"),
                    }
                )

    active_domain_literals = audit_active_domain_literals()
    for record in active_domain_literals:
        if not record["accepted"]:
            failures.append(
                {
                    "path": record["path"],
                    "reason": "unknown_active_domain_literal",
                    "source": record["source"],
                    "active_domain": record["token"],
                }
            )

    extension_entries = [
        entry for entry in entries if entry["velocity_extension_enabled"]
    ]
    nonzero_surface_tension_entries = [
        entry
        for entry in entries
        if (parse_float(entry.get("surface_tension") or "0") or 0.0) != 0.0
    ]
    unsupported_scope_entries = [
        entry for entry in entries if entry.get("unsupported_scope_controls")
    ]
    nitsche_kinematic_entries = [
        entry
        for entry in entries
        if normalized_xml_token(entry.get("kinematic_enforcement")) == "nitsche"
    ]
    raw_curvature_surface_tension_entries = [
        entry
        for entry in nonzero_surface_tension_entries
        if (
            entry.get("use_level_set_curvature") is None
            or enabled_xml_bool(entry.get("use_level_set_curvature"))
        )
        and not entry.get("curvature_field")
    ]
    supplied_or_projected_capillary_entries = [
        entry
        for entry in nonzero_surface_tension_entries
        if not (
            (
                entry.get("use_level_set_curvature") is None
                or enabled_xml_bool(entry.get("use_level_set_curvature"))
            )
            and not entry.get("curvature_field")
        )
        and bool(entry.get("curvature") or entry.get("curvature_field"))
    ]
    return {
        "passed": not failures,
        "checked_solver_xml_count": len(solver_xml_paths),
        "unfitted_free_surface_entry_count": len(entries),
        "velocity_extension_enabled_entry_count": len(extension_entries),
        "unsupported_scope_control_entry_count": len(unsupported_scope_entries),
        "nitsche_kinematic_entry_count": len(nitsche_kinematic_entries),
        "nonzero_surface_tension_entry_count": len(nonzero_surface_tension_entries),
        "raw_curvature_surface_tension_entry_count": len(
            raw_curvature_surface_tension_entries
        ),
        "supplied_or_projected_capillary_entry_count": len(
            supplied_or_projected_capillary_entries
        ),
        "active_domain_literal_count": len(active_domain_literals),
        "failures": failures,
        "warnings": warnings,
        "entries": entries,
        "active_domain_literals": active_domain_literals,
        "runtime_guard": (
            "extension-enabled solver probes are additionally checked by "
            "free_surface_guard_checks after cut-context assembly"
        ),
    }


def check_solver_xml_controls(
    records: list[dict[str, Any]],
    required_controls: tuple[tuple[str, str], ...],
) -> dict[str, Any]:
    observed: dict[str, list[str]] = {}
    for record in records:
        tag = record.get("tag")
        value = record.get("value")
        if isinstance(tag, str) and isinstance(value, str):
            observed.setdefault(tag, []).append(value)

    checks: list[dict[str, Any]] = []
    missing_tags: list[str] = []
    mismatched_controls: list[dict[str, Any]] = []
    for tag, expected in required_controls:
        values = observed.get(tag, [])
        passed = expected in values
        check = {
            "tag": tag,
            "expected": expected,
            "observed": values,
            "passed": passed,
        }
        checks.append(check)
        if not values:
            missing_tags.append(tag)
        elif not passed:
            mismatched_controls.append(check)

    return {
        "required_controls": [
            {"tag": tag, "expected": expected}
            for tag, expected in required_controls
        ],
        "passed": not missing_tags and not mismatched_controls,
        "missing_tags": missing_tags,
        "mismatched_controls": mismatched_controls,
        "checks": checks,
    }


def selected_control_records(
    records: list[dict[str, Any]],
    tags: tuple[str, ...],
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    wanted = set(tags)
    for record in records:
        if record.get("tag") not in wanted:
            continue
        selected.append(
            {
                key: value
                for key, value in record.items()
                if key in {"path", "tag", "value", "attributes"}
            }
        )
    return selected


def solver_xml_qualification_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    linear_algebra = selected_control_records(records, ("Linear_algebra",))
    linear_algebra_types = sorted(
        {
            item.get("attributes", {}).get("type")
            for item in linear_algebra
            if isinstance(item.get("attributes"), dict)
            and item.get("attributes", {}).get("type")
        }
    )
    preconditioners = sorted(
        {
            item.get("value")
            for item in selected_control_records(records, ("Preconditioner",))
            if item.get("value")
        }
    )
    return {
        "solver_xml_available": bool(records),
        "linear_algebra_types": linear_algebra_types,
        "linear_algebra": linear_algebra,
        "linear_solver_controls": selected_control_records(records, ("Preconditioner",)),
        "preconditioners": preconditioners,
        "nonlinear_limits": selected_control_records(records, ("Max_iterations", "Tolerance")),
        "time_controls": selected_control_records(
            records,
            ("Number_of_time_steps", "Time_step_size", "Output_frequency"),
        ),
        "level_set_transport_controls": selected_control_records(
            records,
            (
                "Level_set_source",
                "Velocity_source",
                "Velocity_field_name",
                "Auto_register_velocity_field",
                "Use_wet_extension_advection_velocity",
                "Advection_velocity_from_field",
                "Constant_velocity",
            ),
        ),
        "cut_controls": selected_control_records(
            records,
            (
                "Generated_interface_geometry",
                "Implicit_cut_quadrature_backend",
                "Implicit_cut_fallback_policy",
                "Implicit_cut_max_subdivision_depth",
                "Implicit_cut_root_tolerance",
                "Generated_interface_quadrature_order",
                "Interface_quadrature_order",
                "Volume_quadrature_order",
                "Enable_velocity_extension",
                "Use_cut_metadata_scale",
                "Cut_cell_metadata_scale_cap",
                "Cut_cell_pressure_gradient_penalty",
                "Cut_cell_pressure_stabilization_policy",
            ),
        ),
    }


def solver_xml_has_enabled_velocity_extension(records: list[dict[str, Any]]) -> bool:
    return any(
        record.get("tag") == "Enable_velocity_extension"
        and str(record.get("value", "")).strip().lower() in {"1", "true", "on", "yes"}
        for record in records
    )


def cut_side_for_active_side(active_side: str | None) -> str | None:
    if active_side == "LevelSetNegative":
        return "Negative"
    if active_side == "LevelSetPositive":
        return "Positive"
    return None


def opposite_cut_side(side: str | None) -> str | None:
    if side == "Negative":
        return "Positive"
    if side == "Positive":
        return "Negative"
    return None


def check_velocity_extension_cut_volume_guard(
    solver_xml_controls: list[dict[str, Any]],
    solver_metrics: dict[str, Any] | None,
) -> dict[str, Any]:
    extension_enabled = solver_xml_has_enabled_velocity_extension(solver_xml_controls)
    if not extension_enabled:
        return {
            "passed": True,
            "extension_enabled": False,
            "status": "not_applicable",
        }

    if not solver_metrics or not solver_metrics.get("cut_context_count"):
        return {
            "passed": True,
            "extension_enabled": True,
            "status": "not_evaluated_no_cut_context",
        }

    context = (
        solver_metrics.get("last_accepted_cut_context")
        or solver_metrics.get("last_cut_context")
        or {}
    )
    active_cut_side = cut_side_for_active_side(context.get("active_side"))
    inactive_cut_side = opposite_cut_side(active_cut_side)
    side_records = solver_metrics.get("last_cut_volume_assembly_by_side") or {}
    inactive_record = side_records.get(inactive_cut_side) if inactive_cut_side else None
    inactive_rules = (
        inactive_record.get("rules")
        if isinstance(inactive_record, dict)
        else None
    )
    inactive_points = (
        inactive_record.get("quadrature_points")
        if isinstance(inactive_record, dict)
        else None
    )
    missing_consumer_count = int(
        solver_metrics.get("missing_cut_volume_consumer_diagnostic_count") or 0
    )
    no_retained_consumer_count = int(
        solver_metrics.get("no_retained_cut_volume_consumer_diagnostic_count") or 0
    )
    retained_sides = context.get("retained_volume_sides")
    retained_sides_ok = retained_sides == "active_and_inactive"
    inactive_rules_ok = (
        inactive_cut_side is not None
        and isinstance(inactive_rules, int)
        and inactive_rules > 0
        and isinstance(inactive_points, int)
        and inactive_points > 0
    )
    consumer_diagnostics_ok = (
        missing_consumer_count == 0
        and no_retained_consumer_count == 0
    )
    passed = retained_sides_ok and inactive_rules_ok and consumer_diagnostics_ok
    return {
        "passed": passed,
        "extension_enabled": True,
        "status": "passed" if passed else "failed",
        "active_side": context.get("active_side"),
        "active_cut_side": active_cut_side,
        "inactive_cut_side": inactive_cut_side,
        "retained_volume_sides": retained_sides,
        "retained_sides_ok": retained_sides_ok,
        "inactive_cut_volume_rules": inactive_rules,
        "inactive_cut_volume_quadrature_points": inactive_points,
        "inactive_cut_volume_rules_ok": inactive_rules_ok,
        "missing_cut_volume_consumer_diagnostic_count": missing_consumer_count,
        "no_retained_cut_volume_consumer_diagnostic_count": no_retained_consumer_count,
        "consumer_diagnostics_ok": consumer_diagnostics_ok,
        "last_missing_cut_volume_consumer_diagnostic": solver_metrics.get(
            "last_missing_cut_volume_consumer_diagnostic"
        ),
        "last_no_retained_cut_volume_consumer_diagnostic": solver_metrics.get(
            "last_no_retained_cut_volume_consumer_diagnostic"
        ),
    }


def probe_qualification_metadata(
    probe: Probe,
    solver_metadata: dict[str, Any],
    preflight: dict[str, Any],
    solver_xml_controls: list[dict[str, Any]],
    solver_xml_control_check: dict[str, Any] | None,
) -> dict[str, Any]:
    cache_values = solver_metadata.get("cmake_cache_values") or {}
    feature_keys = (
        "FE_ENABLE_LLVM_JIT",
        "FE_ENABLE_LLVM_JIT_RESOLVED",
        "FE_ENABLE_EIGEN",
        "FE_ENABLE_MPI",
        "FE_USE_MPI_WRAPPERS",
    )
    return {
        "solver_binary": {
            "path": solver_metadata.get("path"),
            "size_bytes": solver_metadata.get("size_bytes"),
            "mtime_epoch_seconds": solver_metadata.get("mtime_epoch_seconds"),
            "build_dir": solver_metadata.get("build_dir"),
            "cmake_cache": solver_metadata.get("cmake_cache"),
        },
        "build_features": {key: cache_values.get(key) for key in feature_keys},
        "required_solver_features": list(probe.required_solver_features),
        "preflight": preflight,
        "launcher": probe_launcher(probe, solver_metadata),
        "mpi_policy": probe.mpi_policy,
        "mpi_ranks": probe.mpi_ranks,
        "environment_overrides": dict(probe.solver_env),
        "generate_args": list(probe.generate_args),
        "required_solver_xml_controls": [
            {"tag": tag, "expected": expected}
            for tag, expected in probe.required_solver_xml_controls
        ],
        "solver_xml_control_check_passed": (
            None if solver_xml_control_check is None else solver_xml_control_check.get("passed")
        ),
        "solver_xml": solver_xml_qualification_summary(solver_xml_controls),
    }


def run_command(
    command: list[str],
    *,
    cwd: Path,
    timeout: float | None = None,
    stdout_path: Path | None = None,
    env: dict[str, str] | None = None,
) -> tuple[int, bool, str, str, float]:
    start = time.monotonic()
    try:
        if stdout_path is None:
            completed = subprocess.run(
                command,
                cwd=cwd,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout,
                check=False,
                env=env,
            )
        else:
            with stdout_path.open("w") as stream:
                completed = subprocess.run(
                    command,
                    cwd=cwd,
                    text=True,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    timeout=timeout,
                    check=False,
                    env=env,
                )
        elapsed = time.monotonic() - start
        return completed.returncode, False, completed.stdout or "", completed.stderr or "", elapsed
    except subprocess.TimeoutExpired as exc:
        elapsed = time.monotonic() - start
        return 124, True, exc.stdout or "", exc.stderr or "", elapsed


def process_environment_metadata() -> dict[str, str]:
    return {
        key: os.environ[key]
        for key in sorted(os.environ)
        if key.startswith("SVMP_")
    }


def invocation_metadata(argv: list[str], *, start_epoch_seconds: float) -> dict[str, Any]:
    return {
        "argv": argv,
        "command": shlex.join(argv),
        "cwd": str(Path.cwd()),
        "environment_overrides": process_environment_metadata(),
        "python": sys.executable,
        "start_time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(start_epoch_seconds)),
    }


def parse_solver_log(log_path: Path) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "accepted_steps": 0,
        "last_accepted_step": None,
        "last_accepted_time": None,
        "rejected_steps": 0,
        "loop_success": None,
        "loop_steps_taken": None,
        "loop_final_time": None,
        "last_nonlinear": None,
        "nonlinear_failure_count": 0,
        "cut_context_count": 0,
        "max_implicit_cut_fallback_cells": 0,
        "min_achieved_interface_quadrature_order": None,
        "min_achieved_volume_quadrature_order": None,
        "max_backend_total_quadrature_point_count": 0,
        "max_implicit_cut_backend_seconds": None,
        "max_implicit_cut_backend_internal_seconds_total": None,
        "max_terminal_topology_refinements": 0,
        "max_terminal_topology_extra_depth": 0,
        "min_active_volume_fraction": None,
        "process_memory_sample_count": 0,
        "last_process_memory": None,
        "max_process_vm_kb": None,
        "max_process_rss_kb": None,
        "max_basis_cache_entries": None,
        "newton_timing_count": 0,
        "last_newton_timing": None,
        "max_newton_timing_seconds_by_label": {},
        "top_level_timing": {},
        "last_error": None,
        "last_backend_failure": None,
        "missing_cut_volume_consumer_diagnostic_count": 0,
        "last_missing_cut_volume_consumer_diagnostic": None,
        "no_retained_cut_volume_consumer_diagnostic_count": 0,
        "last_no_retained_cut_volume_consumer_diagnostic": None,
        "line_search_trial_backend_failure_count": 0,
        "last_line_search_trial_backend_failure": None,
        "max_line_search_trial_terminal_topology_refinements": 0,
        "max_line_search_trial_terminal_topology_extra_depth": 0,
        "line_search_trial_rejection_count": 0,
        "last_cut_context": None,
        "last_accepted_cut_context": None,
        "last_cut_volume_assembly_by_side": {},
        "last_vector_component_norms_by_label": {},
        "last_residual_block_norms": None,
        "last_line_search_residual_block_norms": None,
        "active_pressure_constraint_refresh_count": 0,
        "last_active_pressure_constraint_refresh": None,
        "last_active_side_vertex_constraint": None,
        "last_wet_volume_diagnostic": None,
        "time_loop_options": None,
        "time_loop_adaptive_controller": None,
        "step_start_history": [],
        "nonlinear_history": [],
        "accepted_step_history": summarize_step_history([]),
        "rejected_step_history": [],
        "rejected_step_history_summary": summarize_rejected_step_history([]),
    }
    if not log_path.exists():
        return metrics

    accepted_records_by_step: dict[int, dict[str, Any]] = {}
    current_accepted_step: int | None = None

    for line in log_path.read_text(errors="replace").splitlines():
        if "Transient solve:" in line and "pde_udot_init=" in line:
            values = parse_key_values(line)
            metrics["time_loop_options"] = {
                "t0": parse_float_value(values, "t0"),
                "dt": parse_float_value(values, "dt"),
                "t_end": parse_float_value(values, "t_end"),
                "max_steps": parse_int_value(values, "max_steps"),
                "scheme": values.get("scheme"),
                "rho_inf": parse_float_value(values, "rho_inf"),
                "pde_udot_init": parse_int_value(values, "pde_udot_init"),
            }
        if "TimeLoop adaptive controller enabled:" in line:
            values = parse_key_values(line)
            metrics["time_loop_adaptive_controller"] = {
                "min_dt": parse_float_value(values, "min_dt"),
                "max_dt": parse_float_value(values, "max_dt"),
                "max_retries": parse_int_value(values, "max_retries"),
                "decrease_factor": parse_float_value(values, "decrease_factor"),
                "increase_factor": parse_float_value(values, "increase_factor"),
                "target_newton_iterations": parse_int_value(values, "target_newton_iterations"),
                "max_steps": parse_int_value(values, "max_steps"),
            }
        if match := TIMELOOP_STEP_START_RE.search(line):
            current_accepted_step = None
            metrics["step_start_history"].append(
                {
                    "step": int(match.group("step")),
                    "time": parse_float(match.group("time")),
                    "dt": parse_float(match.group("dt")),
                }
            )
        if match := TIMELOOP_ACCEPTED_RE.search(line):
            metrics["accepted_steps"] += 1
            current_accepted_step = int(match.group("step"))
            accepted_time = parse_float(match.group("time"))
            accepted_dt = parse_float(match.group("dt"))
            metrics["last_accepted_step"] = current_accepted_step
            metrics["last_accepted_time"] = accepted_time
            record = upsert_step_history_record(
                accepted_records_by_step,
                current_accepted_step,
                time=accepted_time,
                dt=accepted_dt,
            )
            append_step_context(record, metrics)
        if match := TIMELOOP_REJECTED_RE.search(line):
            metrics["rejected_steps"] += 1
            rejected_record = snapshot_step_context(metrics)
            rejected_record.update({
                "step": int(match.group("step")),
                "time": parse_float(match.group("time")),
                "dt": parse_float(match.group("dt")),
                "reason": match.group("reason"),
                "nonlinear": {
                    "converged": match.group("converged") == "1",
                    "iters": int(match.group("iters")),
                    "residual": parse_float(match.group("residual")),
                    "field_residual": parse_float(match.group("field_residual") or ""),
                    "aux_residual": parse_float(match.group("aux_residual") or ""),
                },
            })
            metrics["rejected_step_history"].append(rejected_record)
        if match := TIMELOOP_DONE_RE.search(line):
            metrics["loop_success"] = match.group("success") == "1"
            metrics["loop_steps_taken"] = int(match.group("steps"))
            metrics["loop_final_time"] = parse_float(match.group("final_time"))
        if match := TIMELOOP_NONLINEAR_RE.search(line):
            converged = match.group("converged") == "1"
            metrics["last_nonlinear"] = {
                "step": int(match.group("step")),
                "time": parse_float(match.group("time")),
                "converged": converged,
                "iters": int(match.group("iters")),
                "residual": parse_float(match.group("residual")),
                "field_residual": parse_float(match.group("field_residual") or ""),
                "aux_residual": parse_float(match.group("aux_residual") or ""),
            }
            if match.group("linear_converged") is not None:
                metrics["last_nonlinear"]["linear"] = {
                    "converged": match.group("linear_converged") == "1",
                    "iters": int(match.group("linear_iters")),
                    "relative_residual": parse_float(match.group("linear_relative_residual")),
                }
            if not converged:
                metrics["nonlinear_failure_count"] += 1
            metrics["nonlinear_history"].append(copy.deepcopy(metrics["last_nonlinear"]))
        if "diagnostic=cut_context_rebuild" in line:
            metrics["cut_context_count"] += 1
            values = parse_key_values(line)
            cut_context = compact_cut_context(values)
            metrics["last_cut_context"] = cut_context
            if values.get("provenance") in {"accepted", "accepted_step"}:
                metrics["last_accepted_cut_context"] = cut_context
                if current_accepted_step is not None:
                    record = upsert_step_history_record(
                        accepted_records_by_step,
                        current_accepted_step,
                    )
                    record["cut_context"] = copy.deepcopy(cut_context)
            fallback_cells = int(values.get("implicit_cut_fallback_cells", "0"))
            metrics["max_implicit_cut_fallback_cells"] = max(
                metrics["max_implicit_cut_fallback_cells"],
                fallback_cells,
            )
            for source, target in (
                ("achieved_interface_quadrature_order", "min_achieved_interface_quadrature_order"),
                ("achieved_volume_quadrature_order", "min_achieved_volume_quadrature_order"),
            ):
                if source in values:
                    current = int(float(values[source]))
                    previous = metrics[target]
                    metrics[target] = current if previous is None else min(previous, current)
            if "backend_total_quadrature_point_count" in values:
                total = int(float(values["backend_total_quadrature_point_count"]))
                metrics["max_backend_total_quadrature_point_count"] = max(
                    metrics["max_backend_total_quadrature_point_count"],
                    total,
                )
            if "implicit_cut_backend_seconds" in values:
                update_max_metric(
                    metrics,
                    "max_implicit_cut_backend_seconds",
                    parse_float_value(values, "implicit_cut_backend_seconds"),
                )
            if "implicit_cut_backend_internal_seconds_total" in values:
                update_max_metric(
                    metrics,
                    "max_implicit_cut_backend_internal_seconds_total",
                    parse_float_value(values, "implicit_cut_backend_internal_seconds_total"),
                )
            if "terminal_topology_refinements" in values:
                count = int(float(values["terminal_topology_refinements"]))
                metrics["max_terminal_topology_refinements"] = max(
                    metrics["max_terminal_topology_refinements"],
                    count,
                )
            if "max_terminal_topology_extra_depth" in values:
                depth = int(float(values["max_terminal_topology_extra_depth"]))
                metrics["max_terminal_topology_extra_depth"] = max(
                    metrics["max_terminal_topology_extra_depth"],
                    depth,
                )
            if "active_min_volume_fraction" in values:
                fraction = parse_float(values["active_min_volume_fraction"])
                if fraction is not None:
                    previous = metrics["min_active_volume_fraction"]
                    metrics["min_active_volume_fraction"] = fraction if previous is None else min(previous, fraction)
            update_max_metric(metrics, "max_process_vm_kb", parse_int_value(values, "process_vm_kb"))
            update_max_metric(metrics, "max_process_rss_kb", parse_int_value(values, "process_rss_kb"))
            update_max_metric(metrics, "max_basis_cache_entries", parse_int_value(values, "basis_cache_entries"))
        if "diagnostic=process_memory" in line:
            values = parse_key_values(line)
            record = {
                "phase": values.get("phase"),
                "rank": parse_int_value(values, "rank"),
                "op": values.get("op"),
                "process_vm_kb": parse_int_value(values, "process_vm_kb"),
                "process_rss_kb": parse_int_value(values, "process_rss_kb"),
            }
            metrics["process_memory_sample_count"] += 1
            metrics["last_process_memory"] = record
            update_max_metric(metrics, "max_process_vm_kb", record["process_vm_kb"])
            update_max_metric(metrics, "max_process_rss_kb", record["process_rss_kb"])
        if match := TIMING_LINE_RE.search(line):
            label = match.group("label").strip()
            seconds = parse_float(match.group("seconds"))
            key = timing_key(label)
            if label == "Total Newton time":
                metrics["newton_timing_count"] += 1
                metrics["last_newton_timing"] = {key: seconds}
                detail = match.group("detail") or ""
                if detail_match := NEWTON_TIMING_DETAIL_RE.search(detail):
                    metrics["last_newton_timing"].update(
                        {
                            "newton_iterations": int(detail_match.group("newton")),
                            "assemblies": int(detail_match.group("assemblies")),
                            "linear_iterations": int(detail_match.group("linear")),
                        }
                    )
            elif label in {
                "Assembly (J+r)",
                "Linear solve",
                "Solution update",
                "Constraint/ghosts",
                "Other (overhead)",
            } and metrics["last_newton_timing"] is not None:
                metrics["last_newton_timing"][key] = seconds
            elif label in {"Total time loop", "Solve (Newton+linear)", "VTK output"}:
                metrics["top_level_timing"][key] = seconds
            if seconds is not None and (
                label == "Total Newton time"
                or label in {
                    "Assembly (J+r)",
                    "Linear solve",
                    "Solution update",
                    "Constraint/ghosts",
                    "Other (overhead)",
                }
            ):
                current = metrics["max_newton_timing_seconds_by_label"].get(key)
                metrics["max_newton_timing_seconds_by_label"][key] = (
                    seconds if current is None else max(current, seconds)
                )
        if "diagnostic=cut_volume_assembly" in line:
            values = parse_key_values(line)
            record = compact_cut_volume_assembly(values)
            side = record.get("side")
            if side:
                metrics["last_cut_volume_assembly_by_side"][side] = record
                if current_accepted_step is not None:
                    accepted_record = upsert_step_history_record(
                        accepted_records_by_step,
                        current_accepted_step,
                    )
                    accepted_record["cut_volume_assembly_by_side"] = copy.deepcopy(
                        metrics["last_cut_volume_assembly_by_side"]
                    )
        if "Wet volume diagnostic" in line:
            values = parse_key_values(line)
            record = compact_wet_volume_diagnostic(values)
            metrics["last_wet_volume_diagnostic"] = record
            step = record.get("step")
            if isinstance(step, int):
                accepted_record = upsert_step_history_record(
                    accepted_records_by_step,
                    step,
                )
                accepted_record["wet_volume"] = record
        if "has no matching dCutVolume" in line:
            metrics["missing_cut_volume_consumer_diagnostic_count"] += 1
            metrics["last_missing_cut_volume_consumer_diagnostic"] = line[-4000:]
        if "Generated cut-volume consumer has no retained quadrature rules" in line:
            metrics["no_retained_cut_volume_consumer_diagnostic_count"] += 1
            metrics["last_no_retained_cut_volume_consumer_diagnostic"] = line[-4000:]
        if "diagnostic=vector_component_norms" in line:
            values = parse_key_values(line)
            label = values.get("label", "unknown")
            components = parse_component_norms(line)
            if components:
                metrics["last_vector_component_norms_by_label"][label] = components
                if current_accepted_step is not None:
                    record = upsert_step_history_record(
                        accepted_records_by_step,
                        current_accepted_step,
                    )
                    record["vector_component_norms_by_label"] = copy.deepcopy(
                        metrics["last_vector_component_norms_by_label"]
                    )
        if "diagnostic=residual_block_norms" in line:
            values = parse_key_values(line)
            record = {
                "phase": values.get("phase"),
                "field": parse_float_value(values, "field"),
                "aux": parse_float_value(values, "aux"),
                "combined": parse_float_value(values, "combined"),
            }
            metrics["last_residual_block_norms"] = record
            if values.get("phase") == "line_search":
                metrics["last_line_search_residual_block_norms"] = record
            if current_accepted_step is not None:
                accepted_record = upsert_step_history_record(
                    accepted_records_by_step,
                    current_accepted_step,
                )
                accepted_record["residual_block_norms"] = copy.deepcopy(
                    metrics["last_residual_block_norms"]
                )
                accepted_record["line_search_residual_block_norms"] = copy.deepcopy(
                    metrics["last_line_search_residual_block_norms"]
                )
        if "diagnostic=active_pressure_constraint_refresh" in line:
            values = parse_key_values(line)
            metrics["active_pressure_constraint_refresh_count"] += 1
            metrics["last_active_pressure_constraint_refresh"] = {
                "provenance": values.get("provenance"),
                "solution_source": values.get("solution_source"),
                "support_source": values.get("support_source"),
                "constraints": parse_int_value(values, "constraints"),
            }
            if (
                values.get("provenance") in {"accepted", "accepted_step"}
                and current_accepted_step is not None
            ):
                record = upsert_step_history_record(
                    accepted_records_by_step,
                    current_accepted_step,
                )
                record["active_pressure_constraint_refresh"] = copy.deepcopy(
                    metrics["last_active_pressure_constraint_refresh"]
                )
        if "diagnostic=level_set_active_side_vertex_constraint" in line:
            values = parse_key_values(line)
            metrics["last_active_side_vertex_constraint"] = {
                "field": values.get("field"),
                "level_set_field": values.get("level_set_field"),
                "active_side": values.get("active_side"),
                "support_mode": values.get("support_mode"),
                "active_support_cells": parse_int_value(values, "active_support_cells"),
                "active_support_dofs": parse_int_value(values, "active_support_dofs"),
                "inactive_dofs": parse_int_value(values, "inactive_dofs"),
                "constrained_owned_dofs": parse_int_value(values, "constrained_owned_dofs"),
                "active_sign_vertices_without_support": parse_int_value(
                    values,
                    "active_sign_vertices_without_support",
                ),
                "inactive_sign_vertices_with_support": parse_int_value(
                    values,
                    "inactive_sign_vertices_with_support",
                ),
            }
            if current_accepted_step is not None:
                record = upsert_step_history_record(
                    accepted_records_by_step,
                    current_accepted_step,
                )
                record["active_side_vertex_constraint"] = copy.deepcopy(
                    metrics["last_active_side_vertex_constraint"]
                )
        if "[svMultiPhysics] ERROR:" in line:
            metrics["last_error"] = line[-4000:]
        if "generated level-set interface backend failure" in line:
            backend_failure = parse_backend_failure(line)
            if "line search trial residual failed" in line:
                metrics["line_search_trial_rejection_count"] += 1
                metrics["line_search_trial_backend_failure_count"] += 1
                metrics["last_line_search_trial_backend_failure"] = backend_failure
                if backend_failure.get("terminal_topology_refinements") is not None:
                    metrics["max_line_search_trial_terminal_topology_refinements"] = max(
                        metrics["max_line_search_trial_terminal_topology_refinements"],
                        int(float(backend_failure["terminal_topology_refinements"])),
                    )
                if backend_failure.get("max_terminal_topology_extra_depth") is not None:
                    metrics["max_line_search_trial_terminal_topology_extra_depth"] = max(
                        metrics["max_line_search_trial_terminal_topology_extra_depth"],
                        int(float(backend_failure["max_terminal_topology_extra_depth"])),
                    )
            else:
                metrics["last_backend_failure"] = backend_failure
        elif "line search trial residual failed" in line:
            metrics["line_search_trial_rejection_count"] += 1
    accepted_history = [
        accepted_records_by_step[step]
        for step in sorted(accepted_records_by_step)
    ]
    metrics["accepted_step_history"] = summarize_step_history(accepted_history)
    metrics["rejected_step_history_summary"] = summarize_rejected_step_history(
        metrics["rejected_step_history"]
    )
    return metrics


def parse_verifier_json(stdout: str) -> dict[str, Any] | None:
    start = stdout.find("{")
    end = stdout.rfind("}")
    if start < 0 or end < start:
        return None
    try:
        return json.loads(stdout[start:end + 1])
    except json.JSONDecodeError:
        return None


def result_step(path: Path) -> int:
    match = re.search(r"_(\d+)\.p?vtu$", path.name)
    return int(match.group(1)) if match else -1


def output_results(case_dir: Path) -> list[Path]:
    return sorted(
        [*case_dir.glob("result_*.vtu"), *case_dir.glob("result_*.pvtu")],
        key=result_step,
    )


def collate_solver_time_series(case_dir: Path) -> dict[str, Any]:
    try:
        return collate_time_series(case_dir)
    except Exception as exc:  # pragma: no cover - diagnostic path only
        return {
            "generated": False,
            "reason": "collation_error",
            "error": str(exc),
        }


def resolve_probe_result_name(case_dir: Path, result_name: str | None) -> str | None:
    if result_name != LATEST_RESULT_NAME:
        return result_name
    results = output_results(case_dir)
    return results[-1].name if results else None


def compact_verifier_record(result_name: str, metrics: dict[str, Any] | None, *, status: int, timeout: bool) -> dict[str, Any]:
    record: dict[str, Any] = {
        "result": result_name,
        "status": status,
        "timeout": timeout,
        "passed": False,
        "failed_checks": None,
    }
    if metrics is None:
        return record
    for key in (
        "time",
        "verification_profile",
        "interface_pressure_check_scope",
        "interface_pressure_boundary_guard_required",
        "interface_pressure_near_boundary_guard_required",
        "interface_pressure_near_boundary_width",
        "passed",
        "failed_checks",
        "phi_l2_error",
        "phi_max_abs_error",
        "side_wall_clearance",
        "interior_phi_node_count",
        "interior_phi_l2_error",
        "interior_phi_relative_l2_error",
        "interior_phi_max_abs_error",
        "field_quadrature_order",
        "quadrature_domain_area",
        "quadrature_phi_l2_error",
        "quadrature_phi_relative_l2_error",
        "quadrature_phi_grad_l2_error",
        "quadrature_phi_grad_x_l2_error",
        "quadrature_phi_grad_y_l2_error",
        "quadrature_level_set_spatial_residual_l2_error",
        "quadrature_interior_phi_area",
        "quadrature_interior_phi_sample_count",
        "quadrature_interior_phi_l2_error",
        "quadrature_interior_phi_relative_l2_error",
        "interior_quadrature_phi_grad_l2_error",
        "interior_quadrature_phi_grad_x_l2_error",
        "interior_quadrature_phi_grad_y_l2_error",
        "interior_quadrature_level_set_spatial_residual_l2_error",
        "quadrature_interior_phi_finite_sample_count",
        "quadrature_implied_interface_height_l2_error",
        "quadrature_implied_interface_mean_error",
        "quadrature_implied_interface_amplitude_error",
        "quadrature_implied_interface_shift_error",
        "quadrature_implied_interface_shift_measured",
        "interior_quadrature_implied_interface_height_l2_error",
        "interior_quadrature_implied_interface_mean_error",
        "interior_quadrature_implied_interface_amplitude_error",
        "interior_quadrature_implied_interface_shift_error",
        "interior_quadrature_implied_interface_shift_measured",
        "interface_mean_error",
        "interface_amplitude_error",
        "interface_shift_error",
        "interface_shift_measured",
        "interface_l2_height_error",
        "interface_max_height_error",
        "area",
        "area_error",
        "area_relative_error",
        "centroid_y",
        "centroid_y_error",
        "velocity_relative_l2_error",
        "quadrature_field_area",
        "quadrature_field_finite_area",
        "quadrature_field_finite_area_fraction",
        "quadrature_velocity_relative_l2_error",
        "quadrature_pressure_relative_rms_error",
        "quadrature_pressure_relative_rms_error_after_constant_offset_removal",
        "bulk_field_clearance",
        "bulk_wet_node_count",
        "bulk_wet_fallback_to_legacy_mask",
        "bulk_velocity_relative_l2_error",
        "bulk_quadrature_field_area",
        "bulk_quadrature_field_finite_area",
        "bulk_quadrature_field_finite_area_fraction",
        "bulk_quadrature_velocity_relative_l2_error",
        "bulk_pressure_relative_rms_error",
        "bulk_pressure_relative_rms_error_after_constant_offset_removal",
        "bulk_quadrature_pressure_relative_rms_error",
        "bulk_quadrature_pressure_relative_rms_error_after_constant_offset_removal",
        "pressure_relative_rms_error",
        "pressure_relative_rms_error_after_constant_offset_removal",
        "interface_pressure_finite_count",
        "interface_pressure_interpolation_count",
        "interface_pressure_segment_count",
        "interface_pressure_finite_segment_count",
        "interface_pressure_total_segment_length",
        "interface_pressure_boundary_sample_count",
        "interface_pressure_interior_sample_count",
        "interface_pressure_near_boundary_sample_count",
        "interface_pressure_core_sample_count",
        "interface_pressure_boundary_finite_count",
        "interface_pressure_interior_finite_count",
        "interface_pressure_near_boundary_finite_count",
        "interface_pressure_core_finite_count",
        "interface_pressure_boundary_segment_count",
        "interface_pressure_interior_segment_count",
        "interface_pressure_near_boundary_segment_count",
        "interface_pressure_core_segment_count",
        "interface_pressure_boundary_total_segment_length",
        "interface_pressure_interior_total_segment_length",
        "interface_pressure_near_boundary_total_segment_length",
        "interface_pressure_core_total_segment_length",
        "interface_pressure_rms",
        "interface_pressure_boundary_rms",
        "interface_pressure_interior_rms",
        "interface_pressure_near_boundary_rms",
        "interface_pressure_core_rms",
        "interface_pressure_length_weighted_rms",
        "interface_pressure_boundary_length_weighted_rms",
        "interface_pressure_interior_length_weighted_rms",
        "interface_pressure_near_boundary_length_weighted_rms",
        "interface_pressure_core_length_weighted_rms",
        "interface_pressure_length_weighted_mean",
        "interface_pressure_boundary_length_weighted_mean",
        "interface_pressure_interior_length_weighted_mean",
        "interface_pressure_near_boundary_length_weighted_mean",
        "interface_pressure_core_length_weighted_mean",
        "interface_pressure_length_weighted_rms_after_mean_removal",
        "interface_pressure_boundary_length_weighted_rms_after_mean_removal",
        "interface_pressure_interior_length_weighted_rms_after_mean_removal",
        "interface_pressure_near_boundary_length_weighted_rms_after_mean_removal",
        "interface_pressure_core_length_weighted_rms_after_mean_removal",
        "interface_pressure_length_weighted_max_abs",
        "interface_pressure_length_weighted_max_abs_x",
        "interface_pressure_length_weighted_max_abs_y",
        "interface_pressure_length_weighted_max_abs_value",
        "interface_pressure_boundary_max_abs",
        "interface_pressure_boundary_max_abs_x",
        "interface_pressure_boundary_max_abs_y",
        "interface_pressure_boundary_max_abs_value",
        "interface_pressure_interior_max_abs",
        "interface_pressure_interior_max_abs_x",
        "interface_pressure_interior_max_abs_y",
        "interface_pressure_interior_max_abs_value",
        "interface_pressure_near_boundary_max_abs",
        "interface_pressure_near_boundary_max_abs_x",
        "interface_pressure_near_boundary_max_abs_y",
        "interface_pressure_near_boundary_max_abs_value",
        "interface_pressure_core_max_abs",
        "interface_pressure_core_max_abs_x",
        "interface_pressure_core_max_abs_y",
        "interface_pressure_core_max_abs_value",
        "interface_pressure_boundary_length_weighted_max_abs",
        "interface_pressure_boundary_length_weighted_max_abs_x",
        "interface_pressure_boundary_length_weighted_max_abs_y",
        "interface_pressure_boundary_length_weighted_max_abs_value",
        "interface_pressure_interior_length_weighted_max_abs",
        "interface_pressure_interior_length_weighted_max_abs_x",
        "interface_pressure_interior_length_weighted_max_abs_y",
        "interface_pressure_interior_length_weighted_max_abs_value",
        "interface_pressure_near_boundary_length_weighted_max_abs",
        "interface_pressure_near_boundary_length_weighted_max_abs_x",
        "interface_pressure_near_boundary_length_weighted_max_abs_y",
        "interface_pressure_near_boundary_length_weighted_max_abs_value",
        "interface_pressure_core_length_weighted_max_abs",
        "interface_pressure_core_length_weighted_max_abs_x",
        "interface_pressure_core_length_weighted_max_abs_y",
        "interface_pressure_core_length_weighted_max_abs_value",
        "interface_pressure_length_weighted_max_abs_after_mean_removal",
        "interface_pressure_max_abs_x",
        "interface_pressure_max_abs_y",
        "interface_pressure_max_abs_value",
        "interface_pressure_mean",
        "interface_pressure_min",
        "interface_pressure_max",
        "interface_pressure_rms_after_mean_removal",
        "interface_pressure_max_abs_after_mean_removal",
        "interface_stress_sample_count",
        "interface_normal_traction_residual_rms",
        "interface_normal_traction_residual_max_abs",
        "interface_viscous_normal_stress_rms",
        "interface_viscous_normal_stress_max_abs",
        "interface_tangential_traction_rms",
        "interface_tangential_traction_max_abs",
        "wet_node_count",
    ):
        if key in metrics:
            record[key] = metrics[key]
    return record


def summarize_verifier_history(records: list[dict[str, Any]]) -> dict[str, Any]:
    positive_metric_keys = (
        "phi_l2_error",
        "phi_max_abs_error",
        "quadrature_phi_l2_error",
        "quadrature_phi_relative_l2_error",
        "quadrature_phi_grad_l2_error",
        "quadrature_phi_grad_x_l2_error",
        "quadrature_phi_grad_y_l2_error",
        "quadrature_level_set_spatial_residual_l2_error",
        "interior_phi_l2_error",
        "interior_phi_relative_l2_error",
        "quadrature_interior_phi_l2_error",
        "quadrature_interior_phi_relative_l2_error",
        "interior_quadrature_phi_grad_l2_error",
        "interior_quadrature_phi_grad_x_l2_error",
        "interior_quadrature_phi_grad_y_l2_error",
        "interior_quadrature_level_set_spatial_residual_l2_error",
        "quadrature_implied_interface_height_l2_error",
        "interior_quadrature_implied_interface_height_l2_error",
        "interface_l2_height_error",
        "interface_max_height_error",
        "area_relative_error",
        "velocity_relative_l2_error",
        "quadrature_velocity_relative_l2_error",
        "bulk_velocity_relative_l2_error",
        "bulk_quadrature_velocity_relative_l2_error",
        "pressure_relative_rms_error",
        "pressure_relative_rms_error_after_constant_offset_removal",
        "quadrature_pressure_relative_rms_error",
        "quadrature_pressure_relative_rms_error_after_constant_offset_removal",
        "bulk_pressure_relative_rms_error",
        "bulk_pressure_relative_rms_error_after_constant_offset_removal",
        "bulk_quadrature_pressure_relative_rms_error",
        "bulk_quadrature_pressure_relative_rms_error_after_constant_offset_removal",
        "interface_pressure_rms",
        "interface_pressure_boundary_rms",
        "interface_pressure_interior_rms",
        "interface_pressure_near_boundary_rms",
        "interface_pressure_core_rms",
        "interface_pressure_length_weighted_rms",
        "interface_pressure_boundary_length_weighted_rms",
        "interface_pressure_interior_length_weighted_rms",
        "interface_pressure_near_boundary_length_weighted_rms",
        "interface_pressure_core_length_weighted_rms",
        "interface_pressure_length_weighted_rms_after_mean_removal",
        "interface_pressure_boundary_length_weighted_rms_after_mean_removal",
        "interface_pressure_interior_length_weighted_rms_after_mean_removal",
        "interface_pressure_near_boundary_length_weighted_rms_after_mean_removal",
        "interface_pressure_core_length_weighted_rms_after_mean_removal",
        "interface_pressure_length_weighted_max_abs",
        "interface_pressure_boundary_length_weighted_max_abs",
        "interface_pressure_interior_length_weighted_max_abs",
        "interface_pressure_near_boundary_length_weighted_max_abs",
        "interface_pressure_core_length_weighted_max_abs",
        "interface_pressure_length_weighted_max_abs_after_mean_removal",
        "interface_pressure_rms_after_mean_removal",
        "interface_pressure_max_abs_after_mean_removal",
    )
    signed_metric_keys = (
        "interface_mean_error",
        "interface_amplitude_error",
        "interface_shift_error",
        "quadrature_implied_interface_mean_error",
        "quadrature_implied_interface_amplitude_error",
        "quadrature_implied_interface_shift_error",
        "interior_quadrature_implied_interface_mean_error",
        "interior_quadrature_implied_interface_amplitude_error",
        "interior_quadrature_implied_interface_shift_error",
        "centroid_y_error",
    )
    summary: dict[str, Any] = {
        "result_count": len(records),
        "passed_count": sum(1 for record in records if record.get("passed")),
        "failed_count": sum(1 for record in records if not record.get("passed")),
        "failed_check_names": sorted(
            {
                check
                for record in records
                for check in (record.get("failed_checks") or [])
            }
        ),
        "records": records,
    }
    if records:
        summary["first_record"] = records[0]
        summary["final_record"] = records[-1]
    for key in positive_metric_keys:
        values = [record.get(key) for record in records if isinstance(record.get(key), (int, float))]
        if values:
            summary[f"max_{key}"] = max(values)
    for key in signed_metric_keys:
        values = [record.get(key) for record in records if isinstance(record.get(key), (int, float))]
        if values:
            summary[f"max_abs_{key}"] = max(abs(value) for value in values)
    return summary


def run_verifier(case_dir: Path, result_name: str) -> tuple[dict[str, Any], str]:
    command = [sys.executable, "verify_expected_results.py", result_name]
    verifier_status, verifier_timeout, verifier_stdout, verifier_stderr, verifier_seconds = run_command(
        command,
        cwd=case_dir,
        timeout=120.0,
    )
    verifier_metrics = parse_verifier_json(verifier_stdout)
    verify_path = case_dir / f"verify_{result_name}.json"
    verify_path.write_text(verifier_stdout)
    verifier_pass = verifier_status == 0 and not verifier_timeout
    return (
        {
            "command": command,
            "cwd": str(case_dir),
            "status": verifier_status,
            "timeout": verifier_timeout,
            "elapsed_seconds": verifier_seconds,
            "pass": verifier_pass,
            "stdout_json": str(verify_path),
            "stderr_tail": verifier_stderr[-2000:],
            "metrics": verifier_metrics,
        },
        verifier_stdout,
    )


def select_verifier_metrics(metrics: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(metrics, dict):
        return None
    keys = (
        "time",
        "verification_profile",
        "interface_pressure_check_scope",
        "interface_pressure_boundary_guard_required",
        "interface_pressure_near_boundary_guard_required",
        "interface_pressure_near_boundary_width",
        "passed",
        "failed_checks",
        "phi_l2_error",
        "phi_max_abs_error",
        "side_wall_clearance",
        "interior_phi_node_count",
        "interior_phi_l2_error",
        "interior_phi_relative_l2_error",
        "interior_phi_max_abs_error",
        "field_quadrature_order",
        "quadrature_domain_area",
        "quadrature_phi_l2_error",
        "quadrature_phi_relative_l2_error",
        "quadrature_phi_grad_l2_error",
        "quadrature_phi_grad_x_l2_error",
        "quadrature_phi_grad_y_l2_error",
        "quadrature_level_set_spatial_residual_l2_error",
        "quadrature_interior_phi_area",
        "quadrature_interior_phi_sample_count",
        "quadrature_interior_phi_l2_error",
        "quadrature_interior_phi_relative_l2_error",
        "interior_quadrature_phi_grad_l2_error",
        "interior_quadrature_phi_grad_x_l2_error",
        "interior_quadrature_phi_grad_y_l2_error",
        "interior_quadrature_level_set_spatial_residual_l2_error",
        "quadrature_interior_phi_finite_sample_count",
        "quadrature_implied_interface_height_l2_error",
        "quadrature_implied_interface_mean_error",
        "quadrature_implied_interface_amplitude_error",
        "quadrature_implied_interface_shift_error",
        "quadrature_implied_interface_shift_measured",
        "interior_quadrature_implied_interface_height_l2_error",
        "interior_quadrature_implied_interface_mean_error",
        "interior_quadrature_implied_interface_amplitude_error",
        "interior_quadrature_implied_interface_shift_error",
        "interior_quadrature_implied_interface_shift_measured",
        "interface_mean_error",
        "interface_amplitude_error",
        "interface_shift_error",
        "interface_slope_error",
        "interface_slope_progress_fraction",
        "interface_line_rms_residual",
        "interface_pressure_finite_count",
        "interface_pressure_interpolation_count",
        "interface_pressure_segment_count",
        "interface_pressure_finite_segment_count",
        "interface_pressure_total_segment_length",
        "interface_pressure_boundary_sample_count",
        "interface_pressure_interior_sample_count",
        "interface_pressure_near_boundary_sample_count",
        "interface_pressure_core_sample_count",
        "interface_pressure_boundary_finite_count",
        "interface_pressure_interior_finite_count",
        "interface_pressure_near_boundary_finite_count",
        "interface_pressure_core_finite_count",
        "interface_pressure_boundary_segment_count",
        "interface_pressure_interior_segment_count",
        "interface_pressure_near_boundary_segment_count",
        "interface_pressure_core_segment_count",
        "interface_pressure_boundary_total_segment_length",
        "interface_pressure_interior_total_segment_length",
        "interface_pressure_near_boundary_total_segment_length",
        "interface_pressure_core_total_segment_length",
        "interface_pressure_rms",
        "interface_pressure_max_abs",
        "interface_pressure_max_abs_x",
        "interface_pressure_max_abs_y",
        "interface_pressure_max_abs_value",
        "interface_pressure_boundary_rms",
        "interface_pressure_boundary_max_abs",
        "interface_pressure_boundary_max_abs_x",
        "interface_pressure_boundary_max_abs_y",
        "interface_pressure_boundary_max_abs_value",
        "interface_pressure_interior_rms",
        "interface_pressure_interior_max_abs",
        "interface_pressure_interior_max_abs_x",
        "interface_pressure_interior_max_abs_y",
        "interface_pressure_interior_max_abs_value",
        "interface_pressure_near_boundary_rms",
        "interface_pressure_near_boundary_max_abs",
        "interface_pressure_near_boundary_max_abs_x",
        "interface_pressure_near_boundary_max_abs_y",
        "interface_pressure_near_boundary_max_abs_value",
        "interface_pressure_core_rms",
        "interface_pressure_core_max_abs",
        "interface_pressure_core_max_abs_x",
        "interface_pressure_core_max_abs_y",
        "interface_pressure_core_max_abs_value",
        "interface_pressure_length_weighted_rms",
        "interface_pressure_length_weighted_mean",
        "interface_pressure_length_weighted_rms_after_mean_removal",
        "interface_pressure_length_weighted_max_abs",
        "interface_pressure_length_weighted_max_abs_x",
        "interface_pressure_length_weighted_max_abs_y",
        "interface_pressure_length_weighted_max_abs_value",
        "interface_pressure_boundary_length_weighted_rms",
        "interface_pressure_boundary_length_weighted_mean",
        "interface_pressure_boundary_length_weighted_rms_after_mean_removal",
        "interface_pressure_boundary_length_weighted_max_abs",
        "interface_pressure_boundary_length_weighted_max_abs_x",
        "interface_pressure_boundary_length_weighted_max_abs_y",
        "interface_pressure_boundary_length_weighted_max_abs_value",
        "interface_pressure_interior_length_weighted_rms",
        "interface_pressure_interior_length_weighted_mean",
        "interface_pressure_interior_length_weighted_rms_after_mean_removal",
        "interface_pressure_interior_length_weighted_max_abs",
        "interface_pressure_interior_length_weighted_max_abs_x",
        "interface_pressure_interior_length_weighted_max_abs_y",
        "interface_pressure_interior_length_weighted_max_abs_value",
        "interface_pressure_near_boundary_length_weighted_rms",
        "interface_pressure_near_boundary_length_weighted_mean",
        "interface_pressure_near_boundary_length_weighted_rms_after_mean_removal",
        "interface_pressure_near_boundary_length_weighted_max_abs",
        "interface_pressure_near_boundary_length_weighted_max_abs_x",
        "interface_pressure_near_boundary_length_weighted_max_abs_y",
        "interface_pressure_near_boundary_length_weighted_max_abs_value",
        "interface_pressure_core_length_weighted_rms",
        "interface_pressure_core_length_weighted_mean",
        "interface_pressure_core_length_weighted_rms_after_mean_removal",
        "interface_pressure_core_length_weighted_max_abs",
        "interface_pressure_core_length_weighted_max_abs_x",
        "interface_pressure_core_length_weighted_max_abs_y",
        "interface_pressure_core_length_weighted_max_abs_value",
        "interface_pressure_length_weighted_max_abs_after_mean_removal",
        "interface_pressure_mean",
        "interface_pressure_min",
        "interface_pressure_max",
        "interface_pressure_rms_after_mean_removal",
        "interface_pressure_max_abs_after_mean_removal",
        "interface_stress_sample_count",
        "interface_normal_traction_residual_rms",
        "interface_normal_traction_residual_max_abs",
        "interface_viscous_normal_stress_rms",
        "interface_viscous_normal_stress_max_abs",
        "interface_tangential_traction_rms",
        "interface_tangential_traction_max_abs",
        "area_error",
        "area_relative_error",
        "centroid_error",
        "velocity_max",
        "velocity_relative_l2_error",
        "quadrature_field_area",
        "quadrature_field_finite_area",
        "quadrature_field_finite_area_fraction",
        "quadrature_velocity_relative_l2_error",
        "quadrature_pressure_relative_rms_error",
        "quadrature_pressure_relative_rms_error_after_constant_offset_removal",
        "bulk_field_clearance",
        "bulk_wet_node_count",
        "bulk_wet_fallback_to_legacy_mask",
        "bulk_velocity_relative_l2_error",
        "bulk_quadrature_field_area",
        "bulk_quadrature_field_finite_area",
        "bulk_quadrature_field_finite_area_fraction",
        "bulk_quadrature_velocity_relative_l2_error",
        "bulk_pressure_relative_rms_error",
        "bulk_pressure_relative_rms_error_after_constant_offset_removal",
        "bulk_quadrature_pressure_relative_rms_error",
        "bulk_quadrature_pressure_relative_rms_error_after_constant_offset_removal",
        "pressure_relative_rms_error",
        "pressure_rms_relative_error",
        "pressure_gradient_relative_error",
        "pressure_relative_rms_error_after_constant_offset_removal",
        "wet_node_count",
    )
    return {key: metrics[key] for key in keys if key in metrics}


def strip_records_from_summary(summary: Any) -> Any:
    if not isinstance(summary, dict):
        return summary
    return {key: value for key, value in summary.items() if key != "records"}


def compact_solver_metrics(metrics: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(metrics, dict):
        return None
    keys = (
        "accepted_steps",
        "last_accepted_step",
        "last_accepted_time",
        "rejected_steps",
        "loop_success",
        "loop_steps_taken",
        "loop_final_time",
        "last_nonlinear",
        "nonlinear_failure_count",
        "cut_context_count",
        "max_implicit_cut_fallback_cells",
        "min_achieved_interface_quadrature_order",
        "min_achieved_volume_quadrature_order",
        "max_backend_total_quadrature_point_count",
        "max_implicit_cut_backend_seconds",
        "max_implicit_cut_backend_internal_seconds_total",
        "max_terminal_topology_refinements",
        "max_terminal_topology_extra_depth",
        "min_active_volume_fraction",
        "process_memory_sample_count",
        "max_process_vm_kb",
        "max_process_rss_kb",
        "max_basis_cache_entries",
        "newton_timing_count",
        "last_newton_timing",
        "top_level_timing",
        "last_error",
        "last_backend_failure",
        "line_search_trial_backend_failure_count",
        "line_search_trial_rejection_count",
        "time_loop_options",
        "time_loop_adaptive_controller",
        "accepted_step_history",
        "rejected_step_history_summary",
        "last_wet_volume_diagnostic",
        "last_accepted_cut_context",
        "last_cut_volume_assembly_by_side",
        "last_active_pressure_constraint_refresh",
        "last_active_side_vertex_constraint",
    )
    compact: dict[str, Any] = {}
    for key in keys:
        if key not in metrics:
            continue
        value = metrics[key]
        if key == "accepted_step_history":
            value = strip_records_from_summary(value)
        compact[key] = value
    return compact


def compact_probe_for_qualification(summary: dict[str, Any]) -> dict[str, Any]:
    solver = summary.get("solver") or {}
    verifier = summary.get("verifier") or {}
    qualification_metadata = summary.get("qualification_metadata") or {}
    verifier_history = summary.get("verifier_history") or {}
    return {
        "probe": summary.get("probe"),
        "description": summary.get("description"),
        "case_subdir": summary.get("case_subdir"),
        "validation_scope": summary.get("validation_scope"),
        "timeout_policy": summary.get("timeout_policy"),
        "qualification_outcome": summary.get("qualification_outcome"),
        "counts_as_validation_pass": summary.get("counts_as_validation_pass"),
        "broad_gate_closure_permitted": summary.get("broad_gate_closure_permitted"),
        "meets_expectation": summary.get("meets_expectation"),
        "work_dir": summary.get("work_dir"),
        "case_dir": summary.get("case_dir"),
        "kept_workdir": summary.get("kept_workdir"),
        "workdir_removed": summary.get("workdir_removed"),
        "generate_args": summary.get("generate_args"),
        "result_name": summary.get("result_name"),
        "resolved_result_name": summary.get("resolved_result_name"),
        "solver_env_overrides": summary.get("solver_env_overrides"),
        "solver_invocation": summary.get("solver_invocation"),
        "solver_xml_patches": summary.get("solver_xml_patches"),
        "preflight": summary.get("preflight"),
        "generate": summary.get("generate"),
        "expected": summary.get("expected"),
        "solver_xml_control_check": summary.get("solver_xml_control_check"),
        "free_surface_guard_checks": summary.get("free_surface_guard_checks"),
        "solver_xml": qualification_metadata.get("solver_xml"),
        "build_features": qualification_metadata.get("build_features"),
        "solver_binary": qualification_metadata.get("solver_binary"),
        "solver": {
            "status": solver.get("status"),
            "timeout": solver.get("timeout"),
            "elapsed_seconds": solver.get("elapsed_seconds"),
            "success": solver.get("success"),
            "timeout_interpretation": solver.get("timeout_interpretation"),
            "env": solver.get("env"),
            "log": solver.get("log"),
            "metrics": compact_solver_metrics(solver.get("metrics")),
        } if solver else None,
        "verifier": {
            "command": verifier.get("command"),
            "cwd": verifier.get("cwd"),
            "status": verifier.get("status"),
            "timeout": verifier.get("timeout"),
            "elapsed_seconds": verifier.get("elapsed_seconds"),
            "pass": verifier.get("pass"),
            "stdout_json": verifier.get("stdout_json"),
            "stderr_tail": verifier.get("stderr_tail"),
            "metrics": select_verifier_metrics(verifier.get("metrics")),
        } if verifier else None,
        "verifier_history_summary": {
            key: value for key, value in verifier_history.items() if key != "records"
        } if verifier_history else None,
    }


def compact_output_for_qualification(output: dict[str, Any]) -> dict[str, Any]:
    return {
        "invocation": output.get("invocation"),
        "json_output": output.get("json_output"),
        "timeout_seconds": output.get("timeout_seconds"),
        "work_root": output.get("work_root"),
        "keep_workdir": output.get("keep_workdir"),
        "solver": output.get("solver"),
        "solver_metadata": output.get("solver_metadata"),
        "passed_expected_outcomes": output.get("passed_expected_outcomes"),
        "passed_case_inventory_audit": output.get("passed_case_inventory_audit"),
        "passed_all_requested_checks": output.get("passed_all_requested_checks"),
        "qualification_summary": output.get("qualification_summary"),
        "mms_refinement_trends": output.get("mms_refinement_trends"),
        "mms_temporal_trends": output.get("mms_temporal_trends"),
        "case_inventory_audit": output.get("case_inventory_audit"),
        "probes": [
            compact_probe_for_qualification(probe)
            for probe in output.get("probes", [])
        ],
    }


def run_probe(
    probe: Probe,
    *,
    solver: Path,
    solver_metadata: dict[str, Any],
    work_root: Path,
    timeout: float,
    keep_workdir: bool,
) -> dict[str, Any]:
    source = CASE_ROOT / probe.case_subdir
    if not source.exists():
        raise FileNotFoundError(f"missing source case {source}")

    solver_command = probe_solver_command(
        probe,
        solver=solver,
        solver_metadata=solver_metadata,
    )
    launcher = probe_launcher(probe, solver_metadata)
    preflight = solver_feature_preflight(solver_metadata, probe.required_solver_features)
    if not preflight["passed"]:
        summary = {
            "probe": probe.name,
            "description": probe.description,
            "case_subdir": probe.case_subdir,
            "work_dir": None,
            "case_dir": None,
            "kept_workdir": False,
            "workdir_removed": False,
            "generate_args": list(probe.generate_args),
            "result_name": probe.result_name,
            "validation_scope": probe.validation_scope,
            "timeout_policy": probe.timeout_policy,
            "verify_history_enabled": probe.verify_history,
            "solver_env_overrides": dict(probe.solver_env),
            "solver_invocation": {
                "launcher": launcher,
                "mpi_ranks": probe.mpi_ranks,
                "mpi_policy": probe.mpi_policy,
                "command": solver_command,
            },
            "solver_xml_patches": [],
            "preflight": preflight,
            "qualification_metadata": probe_qualification_metadata(
                probe,
                solver_metadata,
                preflight,
                [],
                None,
            ),
            "generate": None,
            "solver_xml_controls": [],
            "solver_xml_control_check": None,
            "free_surface_guard_checks": None,
            "solver": None,
            "verifier": None,
            "verifier_history": None,
            "expected": {
                "solver_success": probe.expect_solver_success,
                "verifier_pass": probe.expect_verifier_pass,
                "required_solver_xml_controls": [
                    {"tag": tag, "expected": expected}
                    for tag, expected in probe.required_solver_xml_controls
                ],
                "timeout_policy": probe.timeout_policy,
                "validation_scope": probe.validation_scope,
                "mpi_policy": probe.mpi_policy,
                "mpi_ranks": probe.mpi_ranks,
            },
            "meets_expectation": False,
        }
        return qualify_probe_summary(summary)

    work_dir = Path(tempfile.mkdtemp(prefix=f"svmp-{probe.name}-", dir=work_root))
    case_dir = work_dir / probe.case_subdir
    copy_case(source, case_dir)

    generate_command = [sys.executable, "generate_case.py", *probe.generate_args]
    generate_status, generate_timeout, generate_stdout, generate_stderr, generate_seconds = run_command(
        generate_command,
        cwd=case_dir,
        timeout=120.0,
    )
    solver_xml_patches = []
    if generate_status == 0 and not generate_timeout:
        solver_xml_patches = patch_solver_xml_for_probe(case_dir, probe)
    solver_xml_controls = collect_solver_xml_controls(case_dir / "solver.xml")
    solver_xml_control_check = check_solver_xml_controls(
        solver_xml_controls,
        probe.required_solver_xml_controls,
    )
    summary: dict[str, Any] = {
        "probe": probe.name,
        "description": probe.description,
        "case_subdir": probe.case_subdir,
        "work_dir": str(work_dir),
        "case_dir": str(case_dir),
        "kept_workdir": keep_workdir,
        "workdir_removed": False,
        "generate_args": list(probe.generate_args),
        "result_name": probe.result_name,
        "validation_scope": probe.validation_scope,
        "timeout_policy": probe.timeout_policy,
        "verify_history_enabled": probe.verify_history,
        "solver_env_overrides": dict(probe.solver_env),
        "solver_invocation": {
            "launcher": launcher,
            "mpi_ranks": probe.mpi_ranks,
            "mpi_policy": probe.mpi_policy,
            "command": solver_command,
        },
        "solver_xml_patches": solver_xml_patches,
        "preflight": preflight,
        "qualification_metadata": probe_qualification_metadata(
            probe,
            solver_metadata,
            preflight,
            solver_xml_controls,
            solver_xml_control_check,
        ),
        "generate": {
            "command": generate_command,
            "cwd": str(case_dir),
            "status": generate_status,
            "timeout": generate_timeout,
            "elapsed_seconds": generate_seconds,
            "stdout_tail": generate_stdout[-2000:],
            "stderr_tail": generate_stderr[-2000:],
        },
        "solver_xml_controls": solver_xml_controls,
        "solver_xml_control_check": solver_xml_control_check,
        "free_surface_guard_checks": None,
        "solver": None,
        "verifier": None,
        "verifier_history": None,
        "expected": {
            "solver_success": probe.expect_solver_success,
            "verifier_pass": probe.expect_verifier_pass,
            "required_solver_xml_controls": [
                {"tag": tag, "expected": expected}
                for tag, expected in probe.required_solver_xml_controls
            ],
            "timeout_policy": probe.timeout_policy,
            "validation_scope": probe.validation_scope,
            "mpi_policy": probe.mpi_policy,
            "mpi_ranks": probe.mpi_ranks,
        },
    }
    if generate_status != 0 or generate_timeout:
        summary["meets_expectation"] = False
        return qualify_probe_summary(summary)
    if not solver_xml_control_check["passed"]:
        summary["meets_expectation"] = False
        return qualify_probe_summary(summary)

    log_path = case_dir / "solver_run.log"
    solver_env = os.environ.copy()
    solver_env.update(dict(probe.solver_env))
    solver_status, solver_timeout, _, _, solver_seconds = run_command(
        solver_command,
        cwd=case_dir,
        timeout=timeout,
        stdout_path=log_path,
        env=solver_env,
    )
    solver_metrics = parse_solver_log(log_path)
    time_series_collation = collate_solver_time_series(case_dir)
    solver_success = solver_status == 0 and not solver_timeout
    free_surface_guard_checks = check_velocity_extension_cut_volume_guard(
        solver_xml_controls,
        solver_metrics,
    )
    summary["free_surface_guard_checks"] = free_surface_guard_checks
    summary["solver"] = {
        "status": solver_status,
        "timeout": solver_timeout,
        "elapsed_seconds": solver_seconds,
        "log": str(log_path),
        "success": solver_success,
        "timeout_policy": probe.timeout_policy,
        "timeout_interpretation": timeout_interpretation(solver_timeout, probe.timeout_policy),
        "env": dict(probe.solver_env),
        "metrics": solver_metrics,
    }
    summary["time_series_collation"] = time_series_collation

    verifier_pass: bool | None = None
    resolved_result_name = resolve_probe_result_name(case_dir, probe.result_name)
    if resolved_result_name is not None:
        summary["resolved_result_name"] = resolved_result_name
    if resolved_result_name is not None and (case_dir / resolved_result_name).exists():
        summary["verifier"], _ = run_verifier(case_dir, resolved_result_name)
        verifier_pass = summary["verifier"]["pass"]

    if probe.verify_history:
        records: list[dict[str, Any]] = []
        for result_path in output_results(case_dir):
            verifier, _ = run_verifier(case_dir, result_path.name)
            records.append(
                compact_verifier_record(
                    result_path.name,
                    verifier.get("metrics"),
                    status=verifier["status"],
                    timeout=verifier["timeout"],
                )
            )
        summary["verifier_history"] = summarize_verifier_history(records)

    solver_matches = (
        not solver_timeout
        and solver_success == probe.expect_solver_success
    )
    guard_matches = free_surface_guard_checks.get("passed", True)
    verifier_matches = (
        probe.expect_verifier_pass is None
        or verifier_pass == probe.expect_verifier_pass
    )
    history_matches = True
    if probe.verify_history and probe.expect_verifier_pass is True:
        history = summary.get("verifier_history") or {}
        history_matches = (
            history.get("result_count", 0) > 0
            and history.get("failed_count", 0) == 0
        )
    summary["history_matches_expectation"] = history_matches
    summary["meets_expectation"] = (
        solver_matches
        and verifier_matches
        and history_matches
        and guard_matches
    )
    qualify_probe_summary(summary)

    if not keep_workdir:
        shutil.rmtree(work_dir)
        summary["workdir_removed"] = True
        summary["case_dir"] = None
        if summary["solver"]:
            summary["solver"]["log"] = None
        if summary["verifier"]:
            summary["verifier"]["stdout_json"] = None
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list-probes", action="store_true", help="List available probes and exit.")
    parser.add_argument(
        "--audit-case-inventory",
        action="store_true",
        help=(
            "Audit checked-in unfitted free-surface XMLs and generator "
            "metadata for recognized active-domain tokens and "
            "velocity-extension-compatible CutVolume wiring."
        ),
    )
    parser.add_argument(
        "--probe",
        action="append",
        choices=sorted(PROBES),
        help="Probe to run. Can be repeated. Defaults to no run unless --list-probes is used.",
    )
    parser.add_argument("--solver", type=Path, help="Path to the svmultiphysics executable.")
    parser.add_argument("--timeout", type=float, default=1200.0, help="Per-probe solver timeout in seconds.")
    parser.add_argument("--work-root", type=Path, default=None, help="Directory for temporary probe workdirs.")
    parser.add_argument("--keep-workdir", action="store_true", help="Keep copied case directories after running.")
    parser.add_argument("--json-output", type=Path, help="Optional path for the JSON summary.")
    parser.add_argument(
        "--compact-json-output",
        type=Path,
        help=(
            "Optional path for a compact qualification artifact that keeps "
            "commands, build metadata, solver controls, verifier metrics, and "
            "diagnostic summaries without full per-step records."
        ),
    )
    parser.add_argument(
        "--no-fail-on-unexpected",
        action="store_true",
        help="Always exit 0 after producing the summary, even if a probe misses its expected outcome.",
    )
    return parser.parse_args()


def main() -> int:
    start_epoch_seconds = time.time()
    args = parse_args()
    if args.list_probes:
        for name in sorted(PROBES):
            probe = PROBES[name]
            print(f"{name}: {probe.description}")
        return 0

    if not args.probe and not args.audit_case_inventory:
        raise SystemExit("no work selected; use --probe, --audit-case-inventory, or --list-probes")
    if args.probe and args.solver is None:
        raise SystemExit("--solver is required when running probes")
    solver = args.solver.resolve() if args.solver else None
    if solver is not None and not solver.exists():
        raise SystemExit(f"solver executable does not exist: {solver}")

    work_root = args.work_root.resolve() if args.work_root else Path(tempfile.gettempdir())
    work_root.mkdir(parents=True, exist_ok=True)
    solver_metadata = collect_solver_metadata(solver) if solver is not None else None
    case_inventory_audit = (
        audit_case_inventory() if args.audit_case_inventory else None
    )
    summaries: list[dict[str, Any]] = []
    if args.probe:
        assert solver is not None
        assert solver_metadata is not None
        summaries = [
            run_probe(
                PROBES[name],
                solver=solver,
                solver_metadata=solver_metadata,
                work_root=work_root,
                timeout=args.timeout,
                keep_workdir=args.keep_workdir,
            )
            for name in args.probe
        ]
    passed_expected_outcomes = all(item.get("meets_expectation") for item in summaries)
    passed_case_inventory_audit = (
        True if case_inventory_audit is None else bool(case_inventory_audit.get("passed"))
    )
    mms_refinement_trends = summarize_mms_refinement_trends(summaries)
    mms_temporal_trends = summarize_mms_temporal_trends(summaries)
    output = {
        "invocation": invocation_metadata(sys.argv, start_epoch_seconds=start_epoch_seconds),
        "timeout_seconds": args.timeout,
        "work_root": str(work_root),
        "keep_workdir": args.keep_workdir,
        "probes": summaries,
        "passed_expected_outcomes": passed_expected_outcomes,
        "passed_case_inventory_audit": passed_case_inventory_audit,
        "passed_all_requested_checks": (
            passed_expected_outcomes and passed_case_inventory_audit
        ),
        "qualification_summary": summarize_qualification(summaries),
    }
    if mms_refinement_trends is not None:
        output["mms_refinement_trends"] = mms_refinement_trends
    if mms_temporal_trends is not None:
        output["mms_temporal_trends"] = mms_temporal_trends
    if solver is not None:
        output["solver"] = str(solver)
        output["solver_metadata"] = solver_metadata
    if case_inventory_audit is not None:
        output["case_inventory_audit"] = case_inventory_audit
    rendered = json.dumps(output, indent=2, sort_keys=True)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        output["json_output"] = str(args.json_output)
        rendered = json.dumps(output, indent=2, sort_keys=True)
        args.json_output.write_text(rendered + "\n")
    if args.compact_json_output:
        args.compact_json_output.parent.mkdir(parents=True, exist_ok=True)
        compact = compact_output_for_qualification(output)
        compact["compact_json_output"] = str(args.compact_json_output)
        args.compact_json_output.write_text(json.dumps(compact, indent=2, sort_keys=True) + "\n")
    print(rendered)
    if args.no_fail_on_unexpected:
        return 0
    return 0 if output["passed_all_requested_checks"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

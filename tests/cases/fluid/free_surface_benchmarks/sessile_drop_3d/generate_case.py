#!/usr/bin/env python3
"""Write one resolution level of the 3D sessile-drop benchmark (tracker M4).

A liquid spherical cap sits on the flat bottom wall y = 0 with zero gravity
and exterior pressure p_ext = 0.  It starts as a spherical cap with a
non-equilibrium contact angle theta_0 and relaxes, at fixed volume, to the
spherical cap with the Young angle theta_e.  The contact angle is imposed only
by the variational Young term in the momentum equation (decision D4):
PrescribedAngle contact line on the bottom wall, Navier slip on the wetted
wall, and a strong normal-only zero velocity on that wall, as in
sessile_drop_2d.  In 3D the contact line is a curve.  The equilibrium angles
are 60, 90 and 120 degrees (--contact-angle), each started 30 degrees away
as in sessile_drop_2d: from 90 degrees for 60 and 120, from 120 for 90.

Domain: by default one quarter of the drop, x >= 0 and z >= 0, with the drop
axis on the line x = z = 0.  The planes x = 0 and z = 0 are symmetry planes:
free-slip walls (zero normal velocity only) that are not contact walls, so the
interface meets them at the mirror angle of 90 degrees through the natural
condition of the SurfaceStress form.  --domain full writes the whole drop on
the mirrored box; its mesh restricted to x, z >= 0 is the quarter mesh, which
is the symmetry check of README.md.

Time step (decisions D25 and D29): the D25 fixed physical step of the 3D
static sphere at La = 12, dt = 0.02 with the lagged normal-increment
capillary term, at both levels; the run lasts 10 viscous times with 200
outputs (D29).  --dt-rule capillary-limit uses the capillary limit of each
level instead (the 2D sessile protocol rule, without the lagged term).
--dt-divisor 2 halves the step with unchanged output times (time-step check
at R/h = 16, decision D43).

The case is written for the new OOP solver: solver.xml, an affine Tetra4
background mesh with the initial fields, the wall face files, and case.json
with every parameter that verify.py needs.  See README.md.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def _load(directory: str, name: str):
    path = HERE.parent / directory / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"{directory}_{name}_for_sessile_3d", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Tetra4 writers, the Kuhn split and the FSILS and transport input blocks.
SPHERE = _load("static_sphere_3d", "generate_case")

# ---------------------------------------------------------------------------
# Protocol constants (principle P1); README.md gives the source of each one.
# ---------------------------------------------------------------------------
LEVELS = (8, 16)                            # R/h; R/h = 16 gates (as D25)
EQUILIBRIUM_ANGLES_DEG = (60, 90, 120)      # theta_e; 90 first (D38), 60 and 120 added 2026-10-22
INITIAL_ANGLE_DEG = {60: 90.0, 90: 120.0, 120: 90.0}  # 30 degrees from theta_e, the sessile_drop_2d pairs
DOMAINS = ("quarter", "full")
DEFAULT_DOMAIN = "quarter"
DENSITY = 1.0
SURFACE_TENSION = 1.0
RADIUS = 1.0                                # radius of the equilibrium cap (length unit)
LAPLACE_NUMBER = 12.0                       # La = rho*gamma*(2R)/mu^2, as sessile_drop_2d
SLIP_LENGTH = RADIUS / 8.0                  # as sessile_drop_2d: l_s/h = 1, 2 at R/h = 8, 16
BOX_MARGIN = 0.25 * RADIUS                  # dry band around the initial and final caps
GRID_UNIT = RADIUS / min(LEVELS)            # box sides are multiples of the coarsest h
EXTERNAL_PRESSURE = 0.0
DEFAULT_VISCOUS_TIMES = 10.0                # D29
DEFAULT_SNAPSHOTS = 200                     # D29
DT_SAFETY = 1.0 / math.sqrt(2.0)            # one-sided capillary limit, as sessile_drop_2d
PROTOCOL_DT = 0.02                          # D25 fixed step at La = 12
DT_RULES = ("fixed", "capillary-limit")
DEFAULT_DT_RULE = "fixed"
DT_DIVISORS = (1, 2)
SEMI_IMPLICIT_OPTIONS = ("None", "NormalIncrement")
SEMI_IMPLICIT_FOR_RULE = {"fixed": "NormalIncrement", "capillary-limit": "None"}
KINEMATIC_RECONCILIATION = True             # as sessile_drop_2d
SIGN_DEFINITE_PATCH_BOUNDS = True           # as sessile_drop_2d
INTERFACE_PRESERVING_REDISTANCING = True    # decisions D38/D39, as sessile_drop_2d
# The method profile of the wetting protocol (decision D47, step P4;
# Physics/Options/FreeSurfaceProfiles.cpp): it states the harmonic PDE
# extension with monolithic coupling, SUPG, kinematic reconciliation
# (KINEMATIC_RECONCILIATION), sign-definite patch bounds
# (SIGN_DEFINITE_PATCH_BOUNDS), triggered redistancing
# (INTERFACE_PRESERVING_REDISTANCING), LinearCorner geometry with
# refreshed-frozen tangents and interface order 2, the cut-facet pressure
# penalty, small-cut aggregation, Surface_tension_semi_implicit=None and the
# strong Ren-E law.  A deck writes a key only when its option differs from
# the profile.
FREE_SURFACE_PROFILE = "Wetting@1"
MIN_PHI_OVER_H_WARNING = 1.0e-6
LEVEL_SET_FIELD = "phi"
INTERFACE_DOMAIN_ID = "sessile_drop_3d_surface"
CONTACT_WALL = "wall_bottom"
CONTACT_WALL_NORMAL = (0.0, -1.0, 0.0)      # out of the liquid, into the solid
# Boundary planes (name, axis, side, role); side -1 = lower box end, +1 = upper.
# Roles: contact (the wetted wall), dry (no-slip; the drop never reaches it),
# symmetry (free slip, zero normal velocity only; quarter domain only).
QUARTER_PLANES = (("sym_x", 0, -1, "symmetry"), ("wall_right", 0, 1, "dry"),
                  ("wall_bottom", 1, -1, "contact"), ("wall_top", 1, 1, "dry"),
                  ("sym_z", 2, -1, "symmetry"), ("wall_back", 2, 1, "dry"))
FULL_PLANES = (("wall_left", 0, -1, "dry"), ("wall_right", 0, 1, "dry"),
               ("wall_bottom", 1, -1, "contact"), ("wall_top", 1, 1, "dry"),
               ("wall_front", 2, -1, "dry"), ("wall_back", 2, 1, "dry"))


# ---------------------------------------------------------------------------
# Spherical-cap geometry: radius r on y = 0 with contact angle theta (through
# the liquid); the centre lies at y = -r cos(theta) on the drop axis.
# ---------------------------------------------------------------------------
def cap_volume_factor(theta: float) -> float:
    """Volume of the cap divided by pi r^3: (2 - 3 cos(theta) + cos(theta)^3) / 3."""
    c = math.cos(theta)
    return (2.0 - 3.0 * c + c ** 3) / 3.0


def cap_radius_for_volume(volume: float, theta: float) -> float:
    return (volume / (math.pi * cap_volume_factor(theta))) ** (1.0 / 3.0)


def cap_geometry(radius: float, theta: float) -> dict:
    return {
        "radius": radius,
        "angle_radians": theta,
        "centre_y": -radius * math.cos(theta),
        "base_radius": radius * math.sin(theta),
        "apex_height": radius * (1.0 - math.cos(theta)),
        "half_extent": radius if theta > 0.5 * math.pi else radius * math.sin(theta),
        "volume": math.pi * radius ** 3 * cap_volume_factor(theta),
    }


def viscosity_from_laplace(laplace: float) -> float:
    return math.sqrt(DENSITY * SURFACE_TENSION * 2.0 * RADIUS / laplace)


def capillary_dt_limit(h: float) -> float:
    """Tracker form sqrt(rho h^3 / (2 pi gamma)), before the safety factor."""
    return math.sqrt(DENSITY * h ** 3 / (2.0 * math.pi * SURFACE_TENSION))


def time_schedule(level: int, viscous_times: float, snapshots: int, *,
                  dt_rule: str = DEFAULT_DT_RULE, dt_divisor: int = 1) -> dict:
    """Base step from the rule, rounded down to `snapshots` equal output
    intervals over viscous_times viscous times; dt_divisor refines it exactly
    with unchanged output times."""
    if dt_rule not in DT_RULES:
        raise ValueError(f"--dt-rule must be one of {DT_RULES}")
    if dt_divisor not in DT_DIVISORS:
        raise ValueError(f"--dt-divisor must be one of {DT_DIVISORS}")
    h = RADIUS / level
    mu = viscosity_from_laplace(LAPLACE_NUMBER)
    viscous_time = DENSITY * RADIUS ** 2 / mu
    end_time = viscous_times * viscous_time
    dt_b = DT_SAFETY * capillary_dt_limit(h)
    dt_max = PROTOCOL_DT if dt_rule == "fixed" else dt_b
    cadence = max(1, math.ceil(end_time / (snapshots * dt_max) - 1.0e-12))
    steps = snapshots * cadence
    dt_base = end_time / steps
    return {
        "h": h, "viscosity": mu, "viscous_time": viscous_time,
        "capillary_time": math.sqrt(DENSITY * RADIUS ** 3 / SURFACE_TENSION),
        "end_time": end_time, "dt_rule": dt_rule, "dt_max": dt_max,
        "dt_capillary_limit": capillary_dt_limit(h), "dt_B": dt_b,
        "dt_base": dt_base, "dt_divisor": dt_divisor, "dt": dt_base / dt_divisor,
        "dt_over_dt_B": dt_base / dt_divisor / dt_b,
        "steps": steps * dt_divisor, "output_cadence": cadence * dt_divisor,
    }


def case_geometry(equilibrium_deg: float, initial_deg: float, domain: str) -> dict:
    theta_e, theta_0 = math.radians(equilibrium_deg), math.radians(initial_deg)
    equilibrium = cap_geometry(RADIUS, theta_e)
    initial = cap_geometry(cap_radius_for_volume(equilibrium["volume"], theta_0), theta_0)

    def grid_ceil(value: float) -> float:
        return math.ceil(value / GRID_UNIT - 1.0e-9) * GRID_UNIT

    side = grid_ceil(max(initial["half_extent"], equilibrium["half_extent"]) + BOX_MARGIN)
    height = grid_ceil(max(initial["apex_height"], equilibrium["apex_height"]) + BOX_MARGIN)
    low = 0.0 if domain == "quarter" else -side
    return {"equilibrium": equilibrium, "initial": initial,
            "box": [low, side, 0.0, height, low, side]}


# ---------------------------------------------------------------------------
# Mesh
# ---------------------------------------------------------------------------
def structured_tetra_mesh(box, level: int, planes):
    """Box split into the six Kuhn tetrahedra of each grid cube.

    The split of cube (i, j, k), indexed from the coordinate origin, is the
    reference split reflected along every axis whose cube index is odd, as in
    static_sphere_3d, so the mesh is symmetric under reflection about every
    grid plane (in particular x = 0 and z = 0), and the full-domain mesh
    restricted to x, z >= 0 is the quarter-domain mesh.  Every cell is
    positively oriented.  Returns points, cells, {plane: (triangles, parents)}
    and the cell counts.
    """
    h = RADIUS / level
    lows = np.array(box[0::2], dtype=float)
    highs = np.array(box[1::2], dtype=float)
    n = np.rint((highs - lows) / h).astype(np.int64)
    first = np.rint(lows / h).astype(np.int64)                # global index of the low plane
    if not (np.allclose(n * h, highs - lows, atol=1e-12) and np.allclose(first * h, lows, atol=1e-12)):
        raise ValueError("box planes must lie on the grid h = R/level")
    m = n + 1
    axes = [lows[a] + h * np.arange(m[a]) for a in range(3)]
    gi, gj, gk = np.meshgrid(np.arange(m[0]), np.arange(m[1]), np.arange(m[2]), indexing="ij")
    points = np.column_stack([axes[0][gi.ravel()], axes[1][gj.ravel()], axes[2][gk.ravel()]])

    def vid(i, j, k):
        return (i * m[1] + j) * m[2] + k

    ci, cj, ck = (a.ravel() for a in np.meshgrid(np.arange(n[0]), np.arange(n[1]), np.arange(n[2]),
                                                 indexing="ij"))
    base = np.column_stack([ci, cj, ck])
    mirror = (base + first[None, :]) % 2                    # parity of the global cube index
    paths = SPHERE._kuhn_paths()
    cells = np.empty((base.shape[0], 6, 4), dtype=np.int64)
    for t in range(6):
        for s in range(4):
            g = base + (paths[t, s][None, :] ^ mirror)
            cells[:, t, s] = vid(g[:, 0], g[:, 1], g[:, 2])
    cells = cells.reshape(-1, 4)
    negative = SPHERE.signed_volume(points, cells) < 0.0
    cells[negative, :2] = cells[negative, 1::-1]
    if np.any(SPHERE.signed_volume(points, cells) <= 0.0):
        raise RuntimeError("degenerate cell")

    grid = np.column_stack([cells.ravel() // (m[1] * m[2]), (cells.ravel() // m[2]) % m[1],
                            cells.ravel() % m[2]]).reshape(-1, 4, 3)
    local_faces = [tuple(v for v in range(4) if v != omit) for omit in range(4)]
    faces = {}
    for name, axis, side, _role in planes:
        index = 0 if side < 0 else n[axis]
        on_plane = grid[:, :, axis] == index                   # (cells, 4)
        tris, parents = [], []
        for lf in local_faces:
            mask = np.all(on_plane[:, lf], axis=1)
            tris.append(cells[mask][:, lf])
            parents.append(np.nonzero(mask)[0])
        tris, parents = np.concatenate(tris), np.concatenate(parents)
        others = [a for a in range(3) if a != axis]
        expected = 2 * n[others[0]] * n[others[1]]
        if tris.shape[0] != expected:
            raise RuntimeError(f"{name}: expected {expected} boundary triangles, found {tris.shape[0]}")
        order = np.argsort(parents, kind="stable")
        faces[name] = (tris[order], parents[order])
    return points, cells, faces, tuple(int(v) for v in n)


# ---------------------------------------------------------------------------
# Solver input
# ---------------------------------------------------------------------------
def solver_xml(equilibrium_deg: float, schedule: dict, steps: int, cadence: int, planes,
               semi_implicit: str, transport: str = "pde_extension",
               kinematic_reconciliation: bool = KINEMATIC_RECONCILIATION,
               sign_definite_patch_bounds: bool = SIGN_DEFINITE_PATCH_BOUNDS,
               ghost_layers: int | str = "derived",
               redistancing: bool = INTERFACE_PRESERVING_REDISTANCING) -> str:
    if semi_implicit not in SEMI_IMPLICIT_OPTIONS:
        raise ValueError(f"semi_implicit must be one of {SEMI_IMPLICIT_OPTIONS}")
    ghost_layers = SPHERE.ghost_layers_value(ghost_layers)
    ghost = "" if ghost_layers == "derived" else f"\n    <Ghost_layers>{ghost_layers}</Ghost_layers>"
    semi_implicit_bc = ("" if semi_implicit == "None" else
                        f"\n      <Surface_tension_semi_implicit>{semi_implicit}"
                        "</Surface_tension_semi_implicit>")
    # The profile turns the three accepted-step stages on; a deck states
    # the ones that are off.
    maintenance = ""
    if not kinematic_reconciliation:
        maintenance += "\n    <Enable_kinematic_reconciliation>false</Enable_kinematic_reconciliation>"
    if not redistancing:
        maintenance += ("\n    <Enable_interface_preserving_redistancing>false"
                        "</Enable_interface_preserving_redistancing>")
    if not sign_definite_patch_bounds:
        maintenance += ("\n    <Enable_sign_definite_patch_bounds>false"
                        "</Enable_sign_definite_patch_bounds>")
    walls = []
    for name, axis, _side, role in planes:
        direction = ""
        if role in ("contact", "symmetry"):
            # zero normal velocity only; the contact wall's tangential motion is
            # governed by the Navier slip of the free-surface condition, the
            # symmetry plane's by its natural condition.
            component = ["0", "0", "0"]
            component[axis] = "1"
            direction = f"\n      <Effective_direction>{' '.join(component)}</Effective_direction>"
        walls.append(f"""    <Add_BC name="{name}">
      <Type>Dir</Type>
      <Value>0.0</Value>{direction}
    </Add_BC>""")
    walls = "\n".join(walls)
    faces = "\n".join(
        f'    <Add_face name="{name}"><Face_file_path>mesh/mesh-surfaces/{name}.vtp</Face_file_path></Add_face>'
        for name, *_ in planes)
    wall_normal = " ".join(f"{c:g}" for c in CONTACT_WALL_NORMAL)
    return f"""<?xml version="1.0" encoding="UTF-8" ?>
<!-- sessile_drop_3d benchmark, theta_e = {equilibrium_deg:g} deg; generated by generate_case.py -->
<svMultiPhysicsFile version="0.1">
  <GeneralSimulationParameters>
    <Use_new_OOP_solver>true</Use_new_OOP_solver>
    <Continue_previous_simulation>false</Continue_previous_simulation>
    <Number_of_spatial_dimensions>3</Number_of_spatial_dimensions>
    <Number_of_time_steps>{steps}</Number_of_time_steps>
    <Time_step_size>{schedule['dt']:.17g}</Time_step_size>
    <Spectral_radius_of_infinite_time_step>0.50</Spectral_radius_of_infinite_time_step>
    <Searched_file_name_to_trigger_stop>STOP_SIM</Searched_file_name_to_trigger_stop>
    <Save_results_to_VTK_format>true</Save_results_to_VTK_format>
    <Combine_time_series>true</Combine_time_series>
    <Name_prefix_of_saved_VTK_files>result</Name_prefix_of_saved_VTK_files>
    <Increment_in_saving_VTK_files>{cadence}</Increment_in_saving_VTK_files>
    <Start_saving_after_time_step>{cadence}</Start_saving_after_time_step>
    <Increment_in_saving_restart_files>{steps}</Increment_in_saving_restart_files>
    <Convert_BIN_to_VTK_format>0</Convert_BIN_to_VTK_format>
    <Verbose>1</Verbose>
    <Warning>0</Warning>
    <Debug>0</Debug>
  </GeneralSimulationParameters>

  <Add_mesh name="box">
    <Mesh_file_path>mesh/mesh-complete.mesh.vtu</Mesh_file_path>{ghost}
{faces}
  </Add_mesh>

  <Add_equation type="level_set">
    <Coupled>true</Coupled>
    <Min_iterations>1</Min_iterations>
    <Max_iterations>4</Max_iterations>
    <Tolerance>1.0e-4</Tolerance>
    <Module_options>jit=true; jit_specialization=true</Module_options>
    <Level_set_field_name>{LEVEL_SET_FIELD}</Level_set_field_name>{SPHERE.level_set_velocity_overrides(transport)}{maintenance}
    <Output type="Spatial">
      <Level_set>true</Level_set>
    </Output>
    <Output type="Volume_integral">
      <Volume>true</Volume>
    </Output>
{SPHERE.fsils_gmres_block()}
  </Add_equation>

  <Add_equation type="fluid">
    <Coupled>true</Coupled>
    <Min_iterations>1</Min_iterations>
    <Max_iterations>8</Max_iterations>
    <Tolerance>1.0e-4</Tolerance>
    <Module_options>jit=true; jit_specialization=true</Module_options>
    <Backflow_stabilization_coefficient>0.0</Backflow_stabilization_coefficient>
    <Density>{DENSITY:.17g}</Density>
    <Force_x>0.0</Force_x>
    <Force_y>0.0</Force_y>
    <Force_z>0.0</Force_z>
    <Hydrostatic_pressure_initialization>false</Hydrostatic_pressure_initialization>
    <Viscosity model="Constant">
      <Value>{schedule['viscosity']:.17g}</Value>
    </Viscosity>
    <Output type="Spatial">
      <Velocity>true</Velocity>
      <Pressure>true</Pressure>
    </Output>
    <Output type="Volume_integral">
      <Volume>true</Volume>
    </Output>
{SPHERE.fsils_gmres_block()}
{walls}
    <Add_BC name="free_surface">
      <Type>Free_surface</Type>
      <Implementation>UnfittedLevelSet</Implementation>
      <Free_surface_profile>{FREE_SURFACE_PROFILE}</Free_surface_profile>
      <Level_set_field_name>{LEVEL_SET_FIELD}</Level_set_field_name>
      <Generated_interface_domain_id>{INTERFACE_DOMAIN_ID}</Generated_interface_domain_id>
      <Active_domain>LevelSetNegative</Active_domain>
      <External_pressure>{EXTERNAL_PRESSURE:.17g}</External_pressure>
      <Surface_tension>{SURFACE_TENSION:.17g}</Surface_tension>{semi_implicit_bc}
      <Contact_line_model>PrescribedAngle</Contact_line_model>
      <Contact_line_wall_face>{CONTACT_WALL}</Contact_line_wall_face>
      <Contact_line_wall_normal>{wall_normal}</Contact_line_wall_normal>
      <Contact_angle_degrees>{equilibrium_deg:.17g}</Contact_angle_degrees>
      <Wall_slip_model>Navier</Wall_slip_model>
      <Wall_slip_length>{SLIP_LENGTH:.17g}</Wall_slip_length>
    </Add_BC>
  </Add_equation>
</svMultiPhysicsFile>
"""


# ---------------------------------------------------------------------------
def generate(level: int, equilibrium_deg: float, output_dir: Path, *,
             domain: str = DEFAULT_DOMAIN, initial_deg: float | None = None,
             viscous_times: float = DEFAULT_VISCOUS_TIMES, snapshots: int = DEFAULT_SNAPSHOTS,
             dt_rule: str = DEFAULT_DT_RULE, dt_divisor: int = 1,
             semi_implicit: str | None = None, transport: str = "pde_extension",
             kinematic_reconciliation: bool = KINEMATIC_RECONCILIATION,
             sign_definite_patch_bounds: bool = SIGN_DEFINITE_PATCH_BOUNDS,
             ghost_layers: int | str = "derived",
             redistancing: bool = INTERFACE_PRESERVING_REDISTANCING,
             max_steps: int | None = None, force: bool = False) -> dict:
    if level not in LEVELS:
        raise ValueError(f"--level must be one of {LEVELS}")
    if equilibrium_deg not in EQUILIBRIUM_ANGLES_DEG:
        raise ValueError(f"--contact-angle must be one of {EQUILIBRIUM_ANGLES_DEG}")
    if domain not in DOMAINS:
        raise ValueError(f"--domain must be one of {DOMAINS}")
    if initial_deg is None:
        initial_deg = INITIAL_ANGLE_DEG[int(equilibrium_deg)]
    if not (10.0 <= initial_deg <= 170.0):
        raise ValueError("--initial-angle must lie in [10, 170] degrees")
    if not (viscous_times > 0.0 and snapshots >= 4):
        raise ValueError("need viscous_times > 0 and at least 4 snapshots")
    if semi_implicit is None:
        semi_implicit = SEMI_IMPLICIT_FOR_RULE[dt_rule]
    if output_dir.exists() and any(output_dir.iterdir()) and not force:
        raise FileExistsError(f"{output_dir} is not empty (use --force)")

    schedule = time_schedule(level, viscous_times, snapshots, dt_rule=dt_rule, dt_divisor=dt_divisor)
    steps, cadence, truncated = schedule["steps"], schedule["output_cadence"], False
    if max_steps is not None:
        if max_steps < 1:
            raise ValueError("--max-steps must be positive")
        if max_steps < steps:
            steps, cadence, truncated = max_steps, 1, True

    planes = QUARTER_PLANES if domain == "quarter" else FULL_PLANES
    geometry = case_geometry(equilibrium_deg, initial_deg, domain)
    initial, equilibrium, box = geometry["initial"], geometry["equilibrium"], geometry["box"]
    points, cells, faces, n = structured_tetra_mesh(box, level, planes)
    centre = np.array([0.0, initial["centre_y"], 0.0])
    phi = np.linalg.norm(points - centre[None, :], axis=1) - initial["radius"]
    h = schedule["h"]
    min_phi_over_h = float(np.min(np.abs(phi)) / h)
    wall = np.abs(points[:, 1]) <= 1.0e-12
    contact_vertex_gap_over_h = float(np.min(np.abs(
        np.hypot(points[wall, 0], points[wall, 2]) - initial["base_radius"])) / h)
    dry_gap = min(box[1] - max(initial["half_extent"], equilibrium["half_extent"]),
                  box[3] - max(initial["apex_height"], equilibrium["apex_height"]))
    laplace_pressure = 2.0 * SURFACE_TENSION / initial["radius"]
    symmetry_fraction = 0.25 if domain == "quarter" else 1.0

    mesh_dir = output_dir / "mesh"
    (mesh_dir / "mesh-surfaces").mkdir(parents=True, exist_ok=True)
    n_points = points.shape[0]
    SPHERE.write_vtu(mesh_dir / "mesh-complete.mesh.vtu", points, cells,
                     {"GlobalNodeID": ("Int64", np.arange(n_points)),
                      LEVEL_SET_FIELD: ("Float64", phi),
                      "Velocity": ("Float64", np.zeros((n_points, 3))),
                      "Pressure": ("Float64", np.full(n_points, EXTERNAL_PRESSURE + laplace_pressure))},
                     {"GlobalElementID": ("Int64", np.arange(cells.shape[0]))})
    for name, *_ in planes:
        tris, parents = faces[name]
        SPHERE.write_face_vtp(mesh_dir / "mesh-surfaces" / f"{name}.vtp", points, tris, parents)
    (output_dir / "solver.xml").write_text(
        solver_xml(equilibrium_deg, schedule, steps, cadence, planes, semi_implicit, transport,
                   kinematic_reconciliation, sign_definite_patch_bounds, ghost_layers,
                   redistancing),
        encoding="utf-8")

    case = {
        "benchmark": "sessile_drop_3d",
        "generator": "tests/cases/fluid/free_surface_benchmarks/sessile_drop_3d/generate_case.py",
        "level_R_over_h": level,
        "equilibrium_angle_degrees": float(equilibrium_deg),
        "initial_angle_degrees": float(initial_deg),
        "domain": domain,
        "symmetry_fraction": symmetry_fraction,
        "boundary_planes": [{"name": nm, "axis": ax, "side": sd, "role": rl}
                            for nm, ax, sd, rl in planes],
        "capillary_form": "surface_stress",
        "contact_line_model": "PrescribedAngle",
        "transport": transport,
        "kinematic_reconciliation": bool(kinematic_reconciliation),
        "sign_definite_patch_bounds": bool(sign_definite_patch_bounds),
        "surface_tension_semi_implicit": semi_implicit,
        "ghost_layers": ghost_layers,
        "laplace_number": LAPLACE_NUMBER,
        "density": DENSITY,
        "surface_tension": SURFACE_TENSION,
        "viscosity": schedule["viscosity"],
        "slip_length": SLIP_LENGTH,
        "slip_length_over_h": SLIP_LENGTH / h,
        "radius": RADIUS,
        "equilibrium_cap_nominal": equilibrium,
        "initial_cap": initial,
        "drop_axis": [0.0, 0.0],
        "initial_centre": centre.tolist(),
        "box": box,
        "cells": list(n),
        "h": h,
        "n_vertices": int(n_points),
        "n_tetrahedra": int(cells.shape[0]),
        "contact_wall_y": 0.0,
        "contact_wall_normal": list(CONTACT_WALL_NORMAL),
        "external_pressure": EXTERNAL_PRESSURE,
        "level_set_field": LEVEL_SET_FIELD,
        "velocity_field": "Velocity",
        "pressure_field": "Pressure",
        "liquid_side": "phi<0",
        "viscous_time": schedule["viscous_time"],
        "capillary_time": schedule["capillary_time"],
        "viscous_times": viscous_times,
        "end_time_protocol": schedule["end_time"],
        "dt_rule": dt_rule,
        "dt_capillary_limit": schedule["dt_capillary_limit"],
        "dt_B": schedule["dt_B"],
        "dt_base": schedule["dt_base"],
        "dt_divisor": dt_divisor,
        "dt": schedule["dt"],
        "dt_over_dt_B": schedule["dt_over_dt_B"],
        "steps_protocol": schedule["steps"],
        "steps": steps,
        "end_time": steps * schedule["dt"],
        "output_cadence": cadence,
        "snapshots": snapshots,
        "truncated": truncated,
        "min_abs_phi_over_h": min_phi_over_h,
        "initial_contact_vertex_gap_over_h": contact_vertex_gap_over_h,
        "dry_gap_over_h": float(dry_gap / h),
        "result_prefix": "result",
    }
    if redistancing:
        # Decision D39; omitted when off, so earlier cases are reproduced byte for byte.
        case["interface_preserving_redistancing"] = True
    (output_dir / "case.json").write_text(json.dumps(case, indent=2) + "\n", encoding="utf-8")
    return case


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--level", type=int, required=True, choices=LEVELS, help="R/h")
    parser.add_argument("--contact-angle", type=int, default=90, choices=EQUILIBRIUM_ANGLES_DEG)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--domain", choices=DOMAINS, default=DEFAULT_DOMAIN,
                        help="quarter (protocol; symmetry planes x = 0 and z = 0) or full "
                             "(symmetry check)")
    parser.add_argument("--initial-angle", type=float, default=None)
    parser.add_argument("--viscous-times", type=float, default=DEFAULT_VISCOUS_TIMES)
    parser.add_argument("--snapshots", type=int, default=DEFAULT_SNAPSHOTS)
    parser.add_argument("--dt-rule", choices=DT_RULES, default=DEFAULT_DT_RULE,
                        help="fixed (protocol, D25: dt = 0.02 with NormalIncrement) or "
                             "capillary-limit (each level's limit, no lagged term)")
    parser.add_argument("--dt-divisor", type=int, choices=DT_DIVISORS, default=1)
    parser.add_argument("--surface-tension-semi-implicit", choices=SEMI_IMPLICIT_OPTIONS,
                        default=None, help="default: NormalIncrement for --dt-rule fixed, "
                                           "None for capillary-limit")
    parser.add_argument("--interface-preserving-redistancing", choices=("on", "off"),
                        default="on" if INTERFACE_PRESERVING_REDISTANCING else "off",
                        help="triggered interface-preserving redistancing (D38/D39, protocol: "
                             "on; off writes the decks without it)")
    parser.add_argument("--ghost-layers", default="derived",
                        help="<Ghost_layers>: a nonnegative integer, or 'derived' (unset; default)")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="smoke runs only: the case is marked truncated")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    case = generate(args.level, args.contact_angle, args.output_dir, domain=args.domain,
                    initial_deg=args.initial_angle, viscous_times=args.viscous_times,
                    snapshots=args.snapshots, dt_rule=args.dt_rule, dt_divisor=args.dt_divisor,
                    semi_implicit=args.surface_tension_semi_implicit,
                    ghost_layers=args.ghost_layers,
                    redistancing=args.interface_preserving_redistancing == "on",
                    max_steps=args.max_steps, force=args.force)
    print(f"wrote {args.output_dir}")
    for key in ("level_R_over_h", "domain", "equilibrium_angle_degrees", "initial_angle_degrees",
                "viscosity", "slip_length_over_h", "dt_rule", "dt", "dt_over_dt_B",
                "surface_tension_semi_implicit", "steps", "output_cadence", "box", "cells",
                "n_vertices", "n_tetrahedra", "min_abs_phi_over_h",
                "initial_contact_vertex_gap_over_h", "dry_gap_over_h", "truncated"):
        print(f"  {key} = {case[key]}")
    if case["min_abs_phi_over_h"] < MIN_PHI_OVER_H_WARNING:
        print("WARNING: the sampled cap nearly touches a vertex", file=sys.stderr)
    if case["initial_contact_vertex_gap_over_h"] < MIN_PHI_OVER_H_WARNING:
        print("WARNING: the initial contact line nearly passes through a wall vertex", file=sys.stderr)
    if case["dry_gap_over_h"] < 2.0:
        print("WARNING: the drop comes within two cells of a dry box wall", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

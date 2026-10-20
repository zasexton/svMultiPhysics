#!/usr/bin/env python3
"""Compute the sessile_drop_3d metrics and apply tolerances.json.

Usage:
    verify.py RUN_DIR [RUN_DIR ...] [--tolerances FILE] [--json OUT] [--allow-truncated]
    verify.py --symmetry-check QUARTER_RUN FULL_RUN [--json OUT]

Each RUN_DIR is a case written by generate_case.py in which the solver has
run.  Runs of one (domain, dt rule, dt divisor) form a refinement study; the
criteria are applied across its levels.  The contact angle is measured along
the whole contact line: the line is divided into azimuthal sectors of
SECTOR_DEGREES around the drop axis, and in each sector a circle is fitted to
the meridional projection (r, y) of the interface points near the wall, as
the 2D benchmark fits each side (README.md).  The sector angle of the
criteria is the mean over the rotations SECTOR_ROTATION_OFFSETS of the sector
partition (decision D43); the single-partition value is reported alongside.
The time-step criterion is judged at the levels in
time_step_criterion.refined_step_levels (R/h = 16, D43); the changes at the
other levels run at both divisors are reported, not gated.
--symmetry-check compares a quarter-domain run with a full-domain run of the
same level on the shared quarter.

Exit status: 0 if every criterion passes, 1 if any criterion fails, 2 if
input data are missing or invalid.
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


SV = _load("static_sphere_3d", "verify")      # Tet4 reader, exact liquid volume, edge crossings
S2 = _load("sessile_drop_2d", "verify")       # circle fit and local contact angle of one side
GEN = _load("sessile_drop_3d", "generate_case")
DataError = SV.DataError
liquid_volume_centroid = SV.liquid_volume_centroid
interface_points = SV.interface_points
observed_order = SV.observed_order
output_series = SV.output_series

SECTOR_DEGREES = 30.0              # azimuthal width of one contact-angle sector
# Decision D43: the sector angle is the mean over 21 rotations of the sector
# partition, offsets -5, -4.5, ..., +5 degrees (offset 0 is the single partition).
SECTOR_ROTATION_HALF_WIDTH = 5.0
SECTOR_ROTATION_COUNT = 21
SECTOR_ROTATION_OFFSETS = tuple(float(v) for v in np.linspace(
    -SECTOR_ROTATION_HALF_WIDTH, SECTOR_ROTATION_HALF_WIDTH, SECTOR_ROTATION_COUNT))
ANGLE_FIT_HEIGHT = S2.ANGLE_FIT_HEIGHT   # fit window 0 <= y <= 0.25 R_ref, as in 2D
WALL_TOLERANCE = 1.0e-9
POINT_MERGE_DIGITS = 12
LOCAL_FACES = ((1, 2, 3), (0, 2, 3), (0, 1, 3), (0, 1, 2))


# ---------------------------------------------------------------------------
# Equilibrium reference and geometry
# ---------------------------------------------------------------------------
def equilibrium_cap(drop_volume: float, theta_deg: float) -> dict:
    theta = math.radians(theta_deg)
    r = GEN.cap_radius_for_volume(drop_volume, theta)
    return GEN.cap_geometry(r, theta)


def wall_contact_segments(points, tets, phi, wall_y):
    """Zero segments of phi_h on the wall faces of the tetrahedra.

    Returns the segment end points (S, 2, 3) and the parent tetrahedron of
    each segment.  A wall face is a tetrahedron face whose three vertices lie
    on y = wall_y; each face with vertices on both sides of zero carries one
    segment of the P1 contact line.
    """
    on_wall = np.abs(points[:, 1] - wall_y) <= WALL_TOLERANCE
    segments, parents = [], []
    for lf in LOCAL_FACES:
        faces = tets[:, lf]
        mask = np.all(on_wall[faces], axis=1)
        for t in np.nonzero(mask)[0]:
            tri = tets[t, list(lf)]
            f = phi[tri]
            if np.all(f < 0.0) or np.all(f >= 0.0):
                continue
            ends = []
            for a, b in ((0, 1), (1, 2), (2, 0)):
                if (f[a] < 0.0) != (f[b] < 0.0):
                    s = f[a] / (f[a] - f[b])
                    ends.append(points[tri[a]] + s * (points[tri[b]] - points[tri[a]]))
            if len(ends) == 2:
                segments.append(ends)
                parents.append(t)
    if not segments:
        raise DataError("no contact line: phi_h does not change sign on the contact wall")
    return np.asarray(segments), np.asarray(parents, dtype=np.int64)


def cell_angle(points, tet, phi) -> float:
    """Through-liquid angle of the P1 interface in one tetrahedron (degrees):
    n = grad(phi)/|grad(phi)| points out of the liquid; with the wall normal
    (0, -1, 0), cos(theta) = n_y."""
    p = points[tet]
    g = np.linalg.solve((p[1:] - p[0]), phi[tet[1:]] - phi[tet[0]])
    return math.degrees(math.acos(float(np.clip(g[1] / np.linalg.norm(g), -1.0, 1.0))))


def fit_axis(contact: np.ndarray) -> np.ndarray:
    """Centre (x, z) of the circle fitted to the contact points (full domain)."""
    xc, zc, _ = S2.fit_circle(contact[:, [0, 2]])
    return np.array([xc, zc])


def sector_masks(azimuth: np.ndarray, domain: str, offset: float = 0.0) -> list[np.ndarray]:
    """Membership of each nominal sector for the partition rotated by offset (degrees).

    Sectors are identified by their nominal centre azimuth (the centre at
    offset 0) and keep that identity in every rotation.  Quarter domain
    (azimuth 0 to 90): the outer edges stay on the symmetry planes, so a
    rotation moves only the interior boundaries (sectors [0, 30 + s),
    [30 + s, 60 + s), [60 + s, 90]).  Full domain: the partition rotates
    periodically (sector k is [-180 + 30 k + s, -150 + 30 k + s) modulo 360).
    At offset 0 both are the unrotated partition of the D38 definition.
    """
    a = np.asarray(azimuth, dtype=float)
    if domain == "quarter":
        span, shift = (0.0, 90.0), offset
    else:
        span, shift = (-180.0, 180.0), 0.0
        a = a - offset
        a = np.where(a < -180.0, a + 360.0, np.where(a > 180.0, a - 360.0, a))
    edges = np.arange(span[0], span[1] + 0.5 * SECTOR_DEGREES, SECTOR_DEGREES)
    n = len(edges) - 1
    masks = []
    for k in range(n):
        lo = edges[k] + (shift if k > 0 else 0.0)
        hi = edges[k + 1] + (shift if k + 1 < n else 0.0)
        masks.append((a >= lo) & (a < hi) if k + 1 < n else (a >= lo))
    return masks


def nominal_sectors(domain: str) -> list[tuple[float, float]]:
    """Azimuth ranges (degrees) of the unrotated sectors, in sector order."""
    span = (0.0, 90.0) if domain == "quarter" else (-180.0, 180.0)
    edges = np.arange(span[0], span[1] + 0.5 * SECTOR_DEGREES, SECTOR_DEGREES)
    return [(float(lo), float(hi)) for lo, hi in zip(edges[:-1], edges[1:])]


def drop_geometry(snap: dict, case: dict, reference_radius: float, theta_e: float) -> dict:
    """Contact line, base radius, apex height and sector contact angles.

    Each sector's "angle" is the mean over SECTOR_ROTATION_OFFSETS of the
    single-partition circle-fit angle of that sector (decision D43); the
    offset-0 value, the spread over the rotations and the offset-0 fit data
    are reported with it.  The max deviation is formed from the averaged
    sector angles: max over the nominal sectors of |angle - theta_e|.
    """
    wall_y = case["contact_wall_y"]
    points, tets, phi = snap["points"], snap["tets"], snap["phi"]
    segments, parents = wall_contact_segments(points, tets, phi, wall_y)
    contact = np.unique(np.round(segments.reshape(-1, 3), POINT_MERGE_DIGITS), axis=0)
    axis = (np.asarray(case["drop_axis"], dtype=float) if case["domain"] == "quarter"
            else fit_axis(contact))
    radial = np.hypot(contact[:, 0] - axis[0], contact[:, 2] - axis[1])
    iface = interface_points(points, tets, phi)
    apex = float(np.max(iface[:, 1]) - wall_y)

    # Sector angles: azimuth about the axis; quarter domain covers [0, 90] degrees.
    def azimuth(q):
        return np.degrees(np.arctan2(q[:, 2] - axis[1], q[:, 0] - axis[0]))
    az_iface, az_contact = azimuth(iface), azimuth(contact)
    meridional = np.column_stack([np.hypot(iface[:, 0] - axis[0], iface[:, 2] - axis[1]),
                                  iface[:, 1]])
    window = ANGLE_FIT_HEIGHT * reference_radius
    nominal = nominal_sectors(case["domain"])
    by_offset = {}
    for offset in SECTOR_ROTATION_OFFSETS:
        rows = []
        for (lo, hi), in_c, in_i in zip(nominal, sector_masks(az_contact, case["domain"], offset),
                                        sector_masks(az_iface, case["domain"], offset)):
            if not np.any(in_c):
                raise DataError(f"no contact point in the sector [{lo:g}, {hi:g}) degrees "
                                f"rotated by {offset:g} degrees")
            r_contact = float(np.mean(radial[in_c]))
            angle, n_fit = S2.local_contact_angle(meridional[in_i], r_contact, 0.0, +1,
                                                  wall_y, window)
            rows.append((angle, n_fit, r_contact))
        by_offset[offset] = rows
    sectors = []
    for k, (lo, hi) in enumerate(nominal):
        rotated = np.array([by_offset[o][k][0] for o in SECTOR_ROTATION_OFFSETS])
        single, n_fit, r_contact = by_offset[0.0][k]
        sectors.append({"azimuth_degrees": [lo, hi],
                        "nominal_centre_azimuth_degrees": 0.5 * (lo + hi),
                        "angle": float(np.mean(rotated)),
                        "angle_single_partition": single,
                        "angle_rotation_min": float(np.min(rotated)),
                        "angle_rotation_max": float(np.max(rotated)),
                        "angle_rotation_spread": float(np.max(rotated) - np.min(rotated)),
                        "angle_rotation_std": float(np.std(rotated)),
                        "fit_points": n_fit, "contact_radius": r_contact})
    angles = np.array([s["angle"] for s in sectors])
    single = np.array([s["angle_single_partition"] for s in sectors])
    lengths = np.linalg.norm(segments[:, 1] - segments[:, 0], axis=1)
    cell = np.array([cell_angle(points, tets[t], phi) for t in parents])
    return {
        "axis": axis.tolist(),
        "base_radius": float(np.mean(radial)),
        "base_radius_spread": float(np.max(radial) - np.min(radial)),
        "apex_height": apex,
        "contact_points": int(len(contact)),
        "sector_rotation_offsets_degrees": list(SECTOR_ROTATION_OFFSETS),
        "sectors": sectors,
        "contact_angle_mean": float(np.mean(angles)),
        "contact_angle_max_deviation": float(np.max(np.abs(angles - theta_e))),
        "contact_angle_mean_single_partition": float(np.mean(single)),
        "contact_angle_max_deviation_single_partition": float(np.max(np.abs(single - theta_e))),
        "contact_angle_rotation_spread_max": float(max(s["angle_rotation_spread"]
                                                       for s in sectors)),
        "cell_angle_mean": float(np.sum(lengths * cell) / np.sum(lengths)),
        "cell_angle_max_deviation": float(np.max(np.abs(cell - theta_e))),
        "contact_line_length": float(np.sum(lengths)),
    }


# ---------------------------------------------------------------------------
# Solver output
# ---------------------------------------------------------------------------
def read_snapshot(path: Path, case: dict) -> dict:
    snap = SV.read_snapshot(path, case)
    keys = np.round(snap["points"], POINT_MERGE_DIGITS)
    _, first, inverse = np.unique(keys, axis=0, return_index=True, return_inverse=True)
    if len(first) != len(snap["points"]):          # points duplicated across MPI pieces
        inverse = inverse.ravel()
        snap = {"points": snap["points"][first], "tets": inverse[snap["tets"]],
                "phi": snap["phi"][first], "velocity": snap["velocity"][first],
                "pressure": snap["pressure"][first]}
    return snap


def analyse_run(run: Path, *, allow_truncated: bool = False) -> dict:
    case_path = run / "case.json"
    if not case_path.is_file():
        raise DataError(f"{run}: no case.json")
    case = json.loads(case_path.read_text())
    if case.get("benchmark") != "sessile_drop_3d":
        raise DataError(f"{run}: not a sessile_drop_3d case")
    if case["truncated"] and not allow_truncated:
        raise DataError(f"{run}: truncated smoke run (use --allow-truncated for diagnostics)")
    series = output_series(run, case)
    if not case["truncated"] and abs(series[-1][0] - case["end_time"]) > 0.5 * case["dt"]:
        raise DataError(f"{run}: incomplete run (last output t = {series[-1][0]:g}, "
                        f"end time {case['end_time']:g})")
    initial = read_snapshot(run / "mesh" / "mesh-complete.mesh.vtu", case)
    v0, _ = liquid_volume_centroid(initial["points"], initial["tets"], initial["phi"])
    fraction = case["symmetry_fraction"]
    theta_e = case["equilibrium_angle_degrees"]
    reference = equilibrium_cap(v0 / fraction, theta_e)
    times, volumes, speeds, history = [], [], [], []
    snap = None
    for t, path in series:
        snap = read_snapshot(path, case)
        v, _ = liquid_volume_centroid(snap["points"], snap["tets"], snap["phi"])
        times.append(t)
        volumes.append(v)
        speeds.append(SV.max_liquid_speed(snap["phi"], snap["velocity"]))
        g = drop_geometry(snap, case, reference["radius"], theta_e)
        history.append({"time": t, "base_radius": g["base_radius"],
                        "contact_angle_mean": g["contact_angle_mean"],
                        "contact_angle_max_deviation": g["contact_angle_max_deviation"],
                        "contact_angle_max_deviation_single_partition":
                            g["contact_angle_max_deviation_single_partition"]})
    final = drop_geometry(snap, case, reference["radius"], theta_e)
    times, speeds = np.asarray(times), np.asarray(speeds)
    t_end = times[-1]
    third = (times >= 0.5 * t_end) & (times <= 0.75 * t_end)
    growth = float(speeds[-1] / np.max(speeds[third])) if np.any(third) else None
    return {
        "run": str(run),
        "level": case["level_R_over_h"],
        "domain": case["domain"],
        "dt_rule": case["dt_rule"],
        "dt_divisor": case["dt_divisor"],
        "dt": case["dt"],
        "truncated": case["truncated"],
        "equilibrium_angle_degrees": theta_e,
        "reference": reference,
        "initial_liquid_volume": v0,
        "final": final,
        "contact_angle_mean_error_degrees": abs(final["contact_angle_mean"] - theta_e),
        "contact_angle_max_error_degrees": final["contact_angle_max_deviation"],
        "contact_angle_max_error_degrees_single_partition":
            final["contact_angle_max_deviation_single_partition"],
        "contact_angle_rotation_spread_max_degrees": final["contact_angle_rotation_spread_max"],
        "base_radius_relative_error": abs(final["base_radius"] - reference["base_radius"])
        / reference["base_radius"],
        "apex_height_relative_error": abs(final["apex_height"] - reference["apex_height"])
        / reference["apex_height"],
        "liquid_volume_relative_drift_max": float(np.max(np.abs(np.asarray(volumes) - v0)) / v0),
        "max_speed_growth_ratio": growth,
        "max_speed_final": float(speeds[-1]),
        # mu max|u_h| / gamma at the last output (rest floor of D39)
        "capillary_number_at_end": float(case["viscosity"] * speeds[-1] / case["surface_tension"]),
        "times": times.tolist(),
        "max_speed": speeds.tolist(),
        "liquid_volume": volumes,
        "history": history,
    }


# ---------------------------------------------------------------------------
# Criteria
# ---------------------------------------------------------------------------
def evaluate_group(runs: list[dict], tolerances: dict, levels_required=None) -> list[dict]:
    by_level = {r["level"]: r for r in runs}
    results = []
    for crit in tolerances["criteria"]:
        q, limit, ok, messages = crit["quantity"], crit["limit"], True, []
        rest_gated: dict[int, bool] = {}
        at = crit["at_level"]
        levels = sorted(by_level) if at == "each" else [at]
        if levels_required is not None and at != "each" and at not in levels_required:
            results.append({"id": crit["id"], "passed": True, "evaluated": False,
                            "details": [f"not evaluated (needs R/h = {at})"]})
            continue
        if levels_required is not None and at != "each" and at not in by_level:
            # A refined-step (dt/2) study without its gating level: not evaluated.
            results.append({"id": crit["id"], "passed": True, "evaluated": False,
                            "details": [f"not evaluated (no run at R/h = {at} in this study)"]})
            continue
        for level in levels:
            if level not in by_level:
                ok = False
                messages.append(f"missing run at R/h={level}")
                continue
            value = by_level[level][q]
            if value is None:
                ok = False
                messages.append(f"R/h={level}: {q} unavailable")
                continue
            passed = value <= limit
            branch = S2.rest_branch(crit, by_level[level])
            if branch is not None:
                rest_gated[level] = branch["gated"]
                if not branch["gated"]:
                    messages.append(f"R/h={level}: {value:.4g} (not gated: {branch['text']})")
                    continue
                messages.append(f"R/h={level}: {value:.4g} {'<=' if passed else '>'} {limit:g}"
                                f" (gated: {branch['text']})")
                ok &= passed
                continue
            ok &= passed
            messages.append(f"R/h={level}: {value:.4g} {'<=' if passed else '>'} {limit:g}")
        mono = crit.get("monotone_levels")
        if crit.get("monotone") == "strictly_decreasing" and mono:
            if all(lv in by_level for lv in mono):
                values = [by_level[lv][q] for lv in mono]
                passed = all(b < a for a, b in zip(values, values[1:]))
                ok &= passed
                messages.append("decreasing over R/h=" + "/".join(map(str, mono)) +
                                (": yes" if passed else ": NO (" +
                                 ", ".join(f"{v:.3g}" for v in values) + ")"))
            elif levels_required is None:
                ok = False
                messages.append("monotonicity needs R/h=" + "/".join(map(str, mono)))
        result = {"id": crit["id"], "passed": bool(ok), "evaluated": True, "details": messages}
        if "rest_capillary_floor" in crit:
            result["rest_capillary_floor"] = crit["rest_capillary_floor"]
            result["gated_by_level"] = {str(lv): g for lv, g in sorted(rest_gated.items())}
        results.append(result)
    return results


def time_step_changes(coarse: dict, fine: dict) -> dict:
    """End-state changes between the dt and dt/2 runs of one level.

    Sectors are matched by their nominal centre azimuth.  The angle change is
    max over the sectors of |angle(dt) - angle(dt/2)| of the rotation-averaged
    sector angles (decision D43); the same with the single-partition angles is
    reported.
    """
    a, b = coarse["final"], fine["final"]
    sb = {s["nominal_centre_azimuth_degrees"]: s for s in b["sectors"]}
    if set(sb) != {s["nominal_centre_azimuth_degrees"] for s in a["sectors"]}:
        raise DataError("the dt and dt/2 runs have different sector partitions")
    by_sector = []
    for s in a["sectors"]:
        t = sb[s["nominal_centre_azimuth_degrees"]]
        by_sector.append({"nominal_centre_azimuth_degrees": s["nominal_centre_azimuth_degrees"],
                          "angle_change": s["angle"] - t["angle"],
                          "angle_change_single_partition":
                              s["angle_single_partition"] - t["angle_single_partition"]})
    return {
        "contact_angle_change_final": max(abs(c["angle_change"]) for c in by_sector),
        "base_radius_change_final": abs(a["base_radius"] - b["base_radius"]) / b["base_radius"],
        "apex_height_change_final": abs(a["apex_height"] - b["apex_height"]) / b["apex_height"],
        "contact_angle_change_final_single_partition": max(
            abs(c["angle_change_single_partition"]) for c in by_sector),
        "contact_angle_change_by_sector": by_sector,
    }


def evaluate_time_step(groups: dict, tolerances: dict) -> list[dict]:
    """Time-step criterion (dt against dt/2) for each (domain, dt rule) study.

    Gated at the levels in time_step_criterion.refined_step_levels (R/h = 16
    since decision D43); a gated level without runs at both divisors is not
    evaluated and does not fail.  The changes at every other level run at both
    divisors (R/h = 8) are reported, not gated.
    """
    crit = tolerances["time_step_criterion"]
    gated_levels = set(crit["refined_step_levels"])
    out = []
    for key, runs in groups.items():
        domain, rule, divisor = key
        if divisor != 1 or (domain, rule, 2) not in groups:
            continue
        coarse = {r["level"]: r for r in runs}
        fine = {r["level"]: r for r in groups[(domain, rule, 2)]}
        common = set(coarse) & set(fine)
        for level in sorted(common | gated_levels):
            gated = level in gated_levels
            if level not in common:
                out.append({"id": "time_step", "group": list(key), "level": level,
                            "gated": True, "evaluated": False, "passed": True,
                            "details": [f"not evaluated (needs dt and dt/2 runs at R/h = {level})"]})
                continue
            changes = time_step_changes(coarse[level], fine[level])
            details, ok = [], True
            for q in crit["quantities"]:
                value = changes[q["id"]]
                passed = value <= q["limit"]
                ok &= passed
                details.append(f"{q['id']}: {value:.3g} {'<=' if passed else '>'} {q['limit']:g}")
            details.append("single-partition angle change "
                           f"{changes['contact_angle_change_final_single_partition']:.3g}")
            out.append({"id": "time_step", "group": list(key), "level": level, "gated": gated,
                        "evaluated": True, "within_limits": bool(ok),
                        "passed": bool(ok) if gated else None,
                        "changes": changes, "details": details})
    return out


# ---------------------------------------------------------------------------
# Symmetry check: quarter domain against the full domain on the shared quarter
# ---------------------------------------------------------------------------
def symmetry_check(quarter_run: Path, full_run: Path) -> dict:
    cq = json.loads((quarter_run / "case.json").read_text())
    cf = json.loads((full_run / "case.json").read_text())
    if cq["domain"] != "quarter" or cf["domain"] != "full" or \
            cq["level_R_over_h"] != cf["level_R_over_h"]:
        raise DataError("--symmetry-check needs a quarter and a full run of the same level")
    sq, sf = output_series(quarter_run, cq), output_series(full_run, cf)
    times_f = {round(t / cf["dt"]): p for t, p in sf}
    theta_e = cq["equilibrium_angle_degrees"]
    rows = []
    for t, pq in sq:
        key = round(t / cq["dt"])
        if key not in times_f:
            continue
        a, b = read_snapshot(pq, cq), read_snapshot(times_f[key], cf)
        # Shared quarter: map the quarter vertices onto the full-domain vertices.
        index = {tuple(k): i for i, k in enumerate(np.round(b["points"], POINT_MERGE_DIGITS))}
        mapping = np.array([index[tuple(k)] for k in np.round(a["points"], POINT_MERGE_DIGITS)])
        du = np.linalg.norm(a["velocity"][:, :3] - b["velocity"][mapping, :3], axis=1)
        umax = max(float(np.max(np.linalg.norm(b["velocity"][:, :3], axis=1))), 1e-300)
        vq, _ = liquid_volume_centroid(a["points"], a["tets"], a["phi"])
        vf, _ = liquid_volume_centroid(b["points"], b["tets"], b["phi"])
        # The full-domain solution restricted to the shared quarter, measured
        # exactly as the quarter run (same cells, axis and sectors).
        corner = np.all((b["points"][b["tets"]][:, :, 0] >= -WALL_TOLERANCE) &
                        (b["points"][b["tets"]][:, :, 2] >= -WALL_TOLERANCE), axis=1)
        bq = dict(b, tets=b["tets"][corner])
        vfq, _ = liquid_volume_centroid(bq["points"], bq["tets"], bq["phi"])
        ref = equilibrium_cap(vf, theta_e)
        ga = drop_geometry(a, cq, ref["radius"], theta_e)
        gb = drop_geometry(bq, cq, ref["radius"], theta_e)
        gfull = drop_geometry(b, cf, ref["radius"], theta_e)
        rows.append({
            "time": t,
            "velocity_max_difference_over_max_speed": float(np.max(du) / umax),
            "phi_max_difference": float(np.max(np.abs(a["phi"] - b["phi"][mapping]))),
            "volume_relative_difference_shared_quarter": abs(vq - vfq) / vfq,
            "volume_relative_difference_4x_quarter_vs_full": abs(4.0 * vq - vf) / vf,
            "base_radius_difference": abs(ga["base_radius"] - gb["base_radius"]),
            "contact_angle_sector_max_difference": float(np.max(np.abs(
                np.array([s["angle"] for s in ga["sectors"]]) -
                np.array([s["angle"] for s in gb["sectors"]])))),
            "contact_angle_sector_max_difference_single_partition": float(np.max(np.abs(
                np.array([s["angle_single_partition"] for s in ga["sectors"]]) -
                np.array([s["angle_single_partition"] for s in gb["sectors"]])))),
            "full_domain_axis_offset": float(np.hypot(*gfull["axis"])),
            "full_domain_angle_spread_over_sectors": float(
                max(s["angle"] for s in gfull["sectors"]) - min(s["angle"] for s in gfull["sectors"])),
            "full_domain_angle_spread_over_sectors_single_partition": float(
                max(s["angle_single_partition"] for s in gfull["sectors"])
                - min(s["angle_single_partition"] for s in gfull["sectors"])),
        })
    if not rows:
        raise DataError("the quarter and full runs share no output time")
    worst = {k: max(r[k] for r in rows) for k in rows[0] if k != "time"}
    return {"quarter_run": str(quarter_run), "full_run": str(full_run), "outputs": rows,
            "worst": worst}


# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="*", type=Path)
    parser.add_argument("--tolerances", type=Path, default=HERE / "tolerances.json")
    parser.add_argument("--json", type=Path)
    parser.add_argument("--allow-truncated", action="store_true",
                        help="report metrics of --max-steps smoke runs (never acceptance evidence)")
    parser.add_argument("--symmetry-check", nargs=2, type=Path, metavar=("QUARTER", "FULL"))
    args = parser.parse_args(argv)
    try:
        if args.symmetry_check:
            report = symmetry_check(*args.symmetry_check)
            for r in report["outputs"]:
                print(f"t={r['time']:.5g}: |du|/max|u| {r['velocity_max_difference_over_max_speed']:.3e}"
                      f"  |dphi| {r['phi_max_difference']:.3e}"
                      f"  dV/V {r['volume_relative_difference_shared_quarter']:.3e}"
                      f"  d(base) {r['base_radius_difference']:.3e}"
                      f"  d(angle) {r['contact_angle_sector_max_difference']:.3e} deg")
            print("worst: " + ", ".join(f"{k} {v:.3e}" for k, v in report["worst"].items()))
            if args.json:
                args.json.write_text(json.dumps(report, indent=2) + "\n")
            return 0
        if not args.runs:
            parser.error("give run directories or --symmetry-check")
        tolerances = json.loads(args.tolerances.read_text())
        runs = [analyse_run(r, allow_truncated=args.allow_truncated) for r in args.runs]
    except (DataError, OSError, KeyError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    groups: dict[tuple, list] = {}
    for r in runs:
        groups.setdefault((r["domain"], r["dt_rule"], r["dt_divisor"]), []).append(r)
    # angle mean / max dev: rotation-averaged sector angles (D43, gated);
    # 1-part dev: single partition; rot sprd: largest spread over the rotations.
    print(f"{'R/h':>4} {'domain':>7} {'rule':>15} {'dt/d':>4} {'angle mean':>10} {'max dev':>8} "
          f"{'1-part dev':>10} {'rot sprd':>8} "
          f"{'cell mean':>9} {'base err':>9} {'apex err':>9} {'dV/V max':>9} {'growth':>7}")
    for r in sorted(runs, key=lambda r: (r["domain"], r["dt_rule"], r["dt_divisor"], r["level"])):
        f = r["final"]
        growth = "-" if r["max_speed_growth_ratio"] is None else f"{r['max_speed_growth_ratio']:.3f}"
        print(f"{r['level']:>4} {r['domain']:>7} {r['dt_rule']:>15} {r['dt_divisor']:>4} "
              f"{f['contact_angle_mean']:>10.3f} {f['contact_angle_max_deviation']:>8.3f} "
              f"{f['contact_angle_max_deviation_single_partition']:>10.3f} "
              f"{f['contact_angle_rotation_spread_max']:>8.3f} "
              f"{f['cell_angle_mean']:>9.3f} {r['base_radius_relative_error']:>9.2e} "
              f"{r['apex_height_relative_error']:>9.2e} {r['liquid_volume_relative_drift_max']:>9.2e} "
              f"{growth:>7}" + ("  [truncated, not acceptance evidence]" if r["truncated"] else ""))
    protocol_key = ("quarter", tolerances["protocol"]["dt_rule"], 1)
    verdicts = evaluate_group(groups.get(protocol_key, []), tolerances)
    refined = tolerances["time_step_criterion"]["refined_step_levels"]
    fine_key = ("quarter", tolerances["protocol"]["dt_rule"], 2)
    fine_verdicts = (evaluate_group(groups[fine_key], tolerances, set(refined))
                     if fine_key in groups else [])
    steps = evaluate_time_step(groups, tolerances)
    any_truncated = any(r["truncated"] for r in runs)
    gated_steps = [v for v in steps if v["gated"] and v["evaluated"]]
    all_pass = bool(verdicts) and all(v["passed"] for v in verdicts + fine_verdicts + gated_steps) \
        and not any_truncated
    print(f"\nprotocol study {protocol_key}:")
    for v in verdicts:
        print(f"  [{'PASS' if v['passed'] else 'FAIL'}] {v['id']}: " + "; ".join(v["details"]))
    for v in fine_verdicts:
        tag = ("NOT EVALUATED" if not v["evaluated"] else "PASS" if v["passed"] else "FAIL")
        print(f"  [dt/2 {tag}] {v['id']}: " + "; ".join(v["details"]))
    for v in steps:
        if not v["evaluated"]:
            tag = "NOT EVALUATED"
        elif v["gated"]:
            tag = "PASS" if v["passed"] else "FAIL"
        else:
            tag = "REPORTED, not gated (D43): " + ("within" if v["within_limits"] else "outside") \
                + " the limits"
        print(f"  [{tag}] time_step at R/h={v['level']}: " + "; ".join(v["details"]))
    if not steps:
        print("  time_step: not evaluated (needs dt and dt/2 runs)")
    if args.json:
        args.json.write_text(json.dumps({"benchmark": "sessile_drop_3d", "runs": runs,
                                         "criteria": verdicts, "criteria_dt2": fine_verdicts,
                                         "time_step": steps, "passed": all_pass}, indent=2) + "\n")
    print("\nOVERALL:", "PASS" if all_pass else "FAIL")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())

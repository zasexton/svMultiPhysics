"""Post-process sharp order-one free-surface energy histories.

The interface measure is the piecewise-linear contour serialized by VTK.  It
is exact for a saved simplex-P1 zero set and is an output linearization for a
tensor-product Q1 zero set.  The wall trace is linear for either supported
order-one space.  Kinetic energy is an explicitly labelled output-space proxy:
``|u_h|^2`` is formed at saved vertices, linearly interpolated by VTK on the
clipped liquid cells, and then integrated.  These quantities are suitable for
refinement and gross energy-growth gates, but are not the assembled method's
quadrature-exact energies and therefore are not evidence of a discrete energy
theorem.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pyvista as pv


KINETIC_DENSITY_FIELD = "__free_surface_kinetic_energy_density_proxy"


def _finite_point_scalar(dataset: pv.DataSet, name: str) -> np.ndarray:
    if name not in dataset.point_data:
        raise ValueError(f"saved state is missing point field {name!r}")
    values = np.asarray(dataset.point_data[name], dtype=float).reshape(-1)
    if values.shape[0] != dataset.n_points or not np.isfinite(values).all():
        raise ValueError(f"point field {name!r} is incomplete or non-finite")
    return values


def _finite_point_vector(dataset: pv.DataSet, name: str) -> np.ndarray:
    if name not in dataset.point_data:
        raise ValueError(f"saved state is missing point field {name!r}")
    values = np.asarray(dataset.point_data[name], dtype=float)
    if values.ndim == 1:
        values = values.reshape((-1, 1))
    if (values.shape[0] != dataset.n_points or values.shape[1] == 0 or
            not np.isfinite(values).all()):
        raise ValueError(f"point field {name!r} is incomplete or non-finite")
    return values


def _sum_cell_measure(dataset: pv.DataSet, name: str) -> float:
    sized = dataset.compute_cell_sizes(length=True, area=True, volume=True)
    if name not in sized.cell_data:
        raise ValueError(f"VTK did not provide cell measure {name!r}")
    values = np.asarray(sized.cell_data[name], dtype=float).reshape(-1)
    if not np.isfinite(values).all():
        raise ValueError(f"cell measure {name!r} is non-finite")
    return float(np.sum(values))


def interface_measure_2d(dataset: pv.DataSet,
                         level_set_name: str = "phi") -> float:
    """Return the length of the saved linearized zero contour in a planar mesh."""
    _finite_point_scalar(dataset, level_set_name)
    contour = dataset.contour(isosurfaces=[0.0], scalars=level_set_name)
    if contour.n_cells == 0 or contour.n_points == 0:
        raise ValueError("saved level set has no zero contour")
    measure = _sum_cell_measure(contour, "Length")
    if not math.isfinite(measure) or measure <= 0.0:
        raise ValueError("saved zero contour has nonpositive length")
    return measure


def liquid_kinetic_energy_proxy_2d(dataset: pv.DataSet,
                                   density: float,
                                   level_set_name: str = "phi",
                                   velocity_name: str = "Velocity") -> float:
    """Integrate a saved-output interpolation proxy for liquid kinetic energy."""
    if not math.isfinite(density) or density <= 0.0:
        raise ValueError("density must be finite and positive")
    _finite_point_scalar(dataset, level_set_name)
    velocity = _finite_point_vector(dataset, velocity_name)

    work = dataset.copy(deep=True)
    work.point_data[KINETIC_DENSITY_FIELD] = (
        0.5 * density * np.sum(velocity * velocity, axis=1)
    )
    # PyVista/VTK's scalar clip with invert=True retains phi <= 0.  Check the
    # resulting domain instead of silently accepting an empty or wrong side.
    liquid = work.clip_scalar(
        scalars=level_set_name,
        value=0.0,
        invert=True,
    )
    if liquid.n_cells == 0:
        raise ValueError("saved level set has no clipped liquid cells")
    integrated = liquid.integrate_data()
    if KINETIC_DENSITY_FIELD not in integrated.point_data:
        raise ValueError("VTK did not integrate the kinetic-energy proxy")
    values = np.asarray(
        integrated.point_data[KINETIC_DENSITY_FIELD], dtype=float
    ).reshape(-1)
    if values.size != 1 or not np.isfinite(values[0]) or values[0] < 0.0:
        raise ValueError("integrated kinetic-energy proxy is invalid")
    return float(values[0])


def _linear_negative_segment_fraction(phi0: float, phi1: float) -> float:
    if phi0 <= 0.0 and phi1 <= 0.0:
        return 1.0
    if phi0 > 0.0 and phi1 > 0.0:
        return 0.0
    denominator = phi0 - phi1
    if denominator == 0.0 or not math.isfinite(denominator):
        raise ValueError("degenerate wall level-set trace")
    crossing = phi0 / denominator
    if not math.isfinite(crossing) or crossing < 0.0 or crossing > 1.0:
        raise ValueError("wall level-set crossing lies outside its edge")
    return crossing if phi0 <= 0.0 else 1.0 - crossing


def wetted_axis_wall_measure_2d(dataset: pv.DataSet,
                                wall_axis: int = 1,
                                wall_coordinate: float = 0.0,
                                level_set_name: str = "phi",
                                tolerance: float = 1.0e-12) -> float:
    """Measure ``phi <= 0`` on a complete axis-aligned planar wall trace.

    The saved synthetic qualification meshes are structured along the wall.
    Consecutive unique wall vertices therefore define its complete P1 trace.
    A discontinuous or branched trace fails closed instead of guessing a
    wetted footprint.
    """
    if wall_axis not in (0, 1):
        raise ValueError("a planar wall axis must be 0 or 1")
    if not math.isfinite(wall_coordinate) or tolerance < 0.0:
        raise ValueError("wall coordinate/tolerance is invalid")
    phi = _finite_point_scalar(dataset, level_set_name)
    points = np.asarray(dataset.points, dtype=float)
    if points.ndim != 2 or points.shape[0] != dataset.n_points:
        raise ValueError("saved state has invalid point coordinates")
    wall = np.flatnonzero(
        np.abs(points[:, wall_axis] - wall_coordinate) <= tolerance
    )
    if wall.size < 2:
        raise ValueError("saved state does not contain a complete wall trace")
    tangent_axis = 1 - wall_axis
    order = wall[np.argsort(points[wall, tangent_axis], kind="mergesort")]
    coordinates = points[order, tangent_axis]
    if np.any(np.diff(coordinates) <= 0.0):
        raise ValueError("wall trace contains duplicate or unordered vertices")

    measure = 0.0
    for left, right in zip(order[:-1], order[1:]):
        length = float(abs(points[right, tangent_axis] - points[left, tangent_axis]))
        measure += length * _linear_negative_segment_fraction(
            float(phi[left]), float(phi[right])
        )
    if not math.isfinite(measure) or measure < 0.0:
        raise ValueError("wetted wall measure is invalid")
    return measure


def free_surface_energy_state_2d(
        dataset: pv.DataSet,
        *,
        density: float,
        surface_tension: float,
        equilibrium_contact_angle_degrees: float | None = None,
        wall_axis: int = 1,
        wall_coordinate: float = 0.0,
        level_set_name: str = "phi",
        velocity_name: str = "Velocity",
) -> dict[str, Any]:
    """Compute kinetic, interface, Young-wall, and total energy components."""
    if not math.isfinite(surface_tension) or surface_tension <= 0.0:
        raise ValueError("surface tension must be finite and positive")
    interface_measure = interface_measure_2d(dataset, level_set_name)
    kinetic = liquid_kinetic_energy_proxy_2d(
        dataset, density, level_set_name, velocity_name
    )
    wall_measure = 0.0
    wall_energy = 0.0
    if equilibrium_contact_angle_degrees is not None:
        theta = math.radians(equilibrium_contact_angle_degrees)
        if not math.isfinite(theta) or not (0.0 < theta < math.pi):
            raise ValueError("equilibrium contact angle must lie strictly in (0, 180)")
        wall_measure = wetted_axis_wall_measure_2d(
            dataset,
            wall_axis,
            wall_coordinate,
            level_set_name,
        )
        # Young's relation gives gamma_SL - gamma_SG = -gamma*cos(theta_e).
        wall_energy = -surface_tension * math.cos(theta) * wall_measure
    interface_energy = surface_tension * interface_measure
    total = kinetic + interface_energy + wall_energy
    if not all(math.isfinite(value) for value in (
            kinetic, interface_measure, interface_energy,
            wall_measure, wall_energy, total)):
        raise ValueError("free-surface energy state is non-finite")
    return {
        "kinetic_energy_proxy": kinetic,
        "interface_measure": interface_measure,
        "interface_energy": interface_energy,
        "wetted_wall_measure": wall_measure,
        "young_wall_energy": wall_energy,
        "total_energy_proxy": total,
        "kinetic_energy_contract": (
            "vertex_squared_velocity_linearly_interpolated_and_integrated_"
            "on_vtk_phi_nonpositive_clip"
        ),
        "surface_energy_contract": (
            "saved_piecewise_linear_zero_contour_exact_for_simplex_p1_"
            "output_linearization_for_tensor_product_q1"
        ),
        "wall_energy_contract": "saved_piecewise_linear_wall_trace_young_relation",
    }


def summarize_energy_history(history: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize a complete, ordered, accepted-state energy history."""
    if len(history) < 2:
        raise ValueError("energy history requires initial and accepted states")
    times: list[float] = []
    totals: list[float] = []
    for record in history:
        time = record.get("time") if isinstance(record, dict) else None
        total = record.get("total_energy_proxy") if isinstance(record, dict) else None
        if (not isinstance(time, (int, float)) or
                not isinstance(total, (int, float)) or
                not math.isfinite(float(time)) or
                not math.isfinite(float(total))):
            raise ValueError("energy history contains an incomplete/non-finite state")
        times.append(float(time))
        totals.append(float(total))
    if times[0] != 0.0 or any(right <= left for left, right in zip(times, times[1:])):
        raise ValueError("energy history times are not initial-plus-strictly-increasing")
    initial = totals[0]
    scale = max(abs(initial), 1.0e-300)
    increments = [right - left for left, right in zip(totals, totals[1:])]
    max_positive_increment = max([0.0, *increments])
    max_above_initial = max(0.0, max(totals) - initial)
    return {
        "initial_total_energy_proxy": initial,
        "final_total_energy_proxy": totals[-1],
        "signed_total_energy_change_proxy": totals[-1] - initial,
        "relative_total_energy_change_proxy": (totals[-1] - initial) / scale,
        "max_positive_step_energy_increment_proxy": max_positive_increment,
        "max_positive_step_energy_increment_relative_to_initial_proxy": (
            max_positive_increment / scale
        ),
        "max_energy_above_initial_proxy": max_above_initial,
        "max_energy_above_initial_relative_proxy": max_above_initial / scale,
        "state_count": len(history),
        "complete": True,
        "discrete_energy_theorem_claimed": False,
    }


def energy_history_gate_errors(
        summary: dict[str, Any],
        *,
        max_positive_step_increment_relative: float,
        max_above_initial_relative: float,
        require_final_not_above_initial: bool = True,
) -> list[str]:
    """Fail closed on missing histories or excessive accepted-state growth."""
    for name, value in (
            ("max_positive_step_increment_relative",
             max_positive_step_increment_relative),
            ("max_above_initial_relative", max_above_initial_relative)):
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and nonnegative")
    if summary.get("complete") is not True:
        return ["free-surface energy history is unavailable or incomplete"]
    required = {
        "relative_total_energy_change_proxy": None,
        "max_positive_step_energy_increment_relative_to_initial_proxy":
            max_positive_step_increment_relative,
        "max_energy_above_initial_relative_proxy": max_above_initial_relative,
    }
    errors: list[str] = []
    for name, maximum in required.items():
        value = summary.get(name)
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            errors.append(f"free-surface energy metric {name!r} is unavailable")
            continue
        if maximum is not None and float(value) > maximum:
            errors.append(
                f"free-surface energy metric {name} {float(value):.6g} "
                f"exceeds {maximum:.6g}"
            )
    final_change = summary.get("relative_total_energy_change_proxy")
    if (require_final_not_above_initial and
            isinstance(final_change, (int, float)) and
            math.isfinite(float(final_change)) and
            float(final_change) > max_above_initial_relative):
        errors.append(
            "free-surface final total-energy proxy increase "
            f"{float(final_change):.6g} exceeds "
            f"{max_above_initial_relative:.6g}"
        )
    return errors

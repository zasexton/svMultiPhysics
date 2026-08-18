#!/usr/bin/env python3
"""Assess Ustruct block solution fields against an exact or reference solution.

The default exact solution is the homogeneous incompressible neo-Hookean
uniaxial compression field documented in this directory's README.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Callable, Iterable


def import_vtk():
    try:
        import vtk  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "This script requires the Python VTK package. "
            "Install it or run from an environment where 'import vtk' works."
        ) from exc
    return vtk


def resolve_vtu(path: Path, prefix: str, step: int) -> Path:
    if path.is_file():
        return path

    names = [f"{prefix}_{step:03d}.vtu", f"{prefix}_{step:03d}.pvtu"]
    for name in names:
        direct = path / name
        if direct.exists():
            return direct

    legacy = path / "1-procs" / names[0]
    if legacy.exists():
        return legacy

    matches = []
    for name in names:
        matches.extend(path.rglob(name))
    matches = sorted(matches)
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise SystemExit(
            f"Could not find {names[0]!r} or {names[1]!r} under {path}"
        )
    raise SystemExit(
        "Found multiple matching VTK output files:\n"
        + "\n".join(f"  {m}" for m in matches)
        + "\nPass the VTU/PVTU file path directly."
    )


def read_grid(path: Path):
    vtk = import_vtk()
    if path.suffix == ".pvtu":
        reader = vtk.vtkXMLPUnstructuredGridReader()
    else:
        reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(str(path))
    reader.Update()
    grid = reader.GetOutput()
    if grid is None or grid.GetNumberOfPoints() == 0:
        raise SystemExit(f"Failed to read nonempty VTU grid from {path}")
    return grid


def get_array(grid, name: str):
    arr = grid.GetPointData().GetArray(name)
    if arr is None:
        available = [
            grid.GetPointData().GetArrayName(i)
            for i in range(grid.GetPointData().GetNumberOfArrays())
        ]
        raise SystemExit(
            f"Point-data array {name!r} not found. Available arrays: {available}"
        )
    return arr


def tuple_values(arr, i: int) -> list[float]:
    return [float(arr.GetComponent(i, c)) for c in range(arr.GetNumberOfComponents())]


def l2(values: Iterable[float]) -> float:
    return math.sqrt(sum(v * v for v in values))


def compute_metrics(
    grid,
    array_name: str,
    exact_at_point: Callable[[tuple[float, float, float]], list[float]],
    coordinate_decimals: int = 12,
) -> dict[str, object]:
    arr = get_array(grid, array_name)
    ncomp = arr.GetNumberOfComponents()
    groups = grouped_point_indices(grid, coordinate_decimals)
    n = len(groups)

    max_abs = 0.0
    max_exact = 0.0
    sum_err2 = 0.0
    sum_exact2 = 0.0
    component_max_abs = [0.0 for _ in range(ncomp)]
    component_sum_err2 = [0.0 for _ in range(ncomp)]

    for indices in groups.values():
        i = indices[0]
        x = grid.GetPoint(i)
        exact = exact_at_point((float(x[0]), float(x[1]), float(x[2])))
        if len(exact) != ncomp:
            raise SystemExit(
                f"Exact {array_name} has {len(exact)} components, "
                f"but VTU array has {ncomp}"
            )
        observed = tuple_values(arr, i)
        err = [observed[c] - exact[c] for c in range(ncomp)]
        err_norm = l2(err)
        exact_norm = l2(exact)
        max_abs = max(max_abs, err_norm)
        max_exact = max(max_exact, exact_norm)
        sum_err2 += err_norm * err_norm
        sum_exact2 += exact_norm * exact_norm
        for c in range(ncomp):
            component_max_abs[c] = max(component_max_abs[c], abs(err[c]))
            component_sum_err2[c] += err[c] * err[c]

    rms_abs = math.sqrt(sum_err2 / n)
    rms_exact = math.sqrt(sum_exact2 / n)
    denom_linf = max(max_exact, 1.0e-30)
    denom_rms = max(rms_exact, 1.0e-30)
    return {
        "components": ncomp,
        "points": n,
        "raw_points": grid.GetNumberOfPoints(),
        "duplicate_point_groups": sum(1 for indices in groups.values() if len(indices) > 1),
        "duplicate_points": sum(max(0, len(indices) - 1) for indices in groups.values()),
        "duplicate_linf_abs": duplicate_spread(grid, array_name, groups),
        "linf_abs": max_abs,
        "rms_abs": rms_abs,
        "linf_exact": max_exact,
        "rms_exact": rms_exact,
        "linf_rel": max_abs / denom_linf,
        "rms_rel": rms_abs / denom_rms,
        "component_linf_abs": component_max_abs,
        "component_rms_abs": [
            math.sqrt(component_sum_err2[c] / n) for c in range(ncomp)
        ],
    }


def point_key(point: tuple[float, float, float], decimals: int) -> tuple[float, float, float]:
    return tuple(round(float(v), decimals) for v in point)


def grouped_point_indices(grid, coordinate_decimals: int) -> dict[tuple[float, float, float], list[int]]:
    groups: dict[tuple[float, float, float], list[int]] = {}
    for i in range(grid.GetNumberOfPoints()):
        groups.setdefault(point_key(grid.GetPoint(i), coordinate_decimals), []).append(i)
    return groups


def duplicate_spread(grid, array_name: str, groups: dict[tuple[float, float, float], list[int]]) -> float:
    arr = get_array(grid, array_name)
    max_spread = 0.0
    for indices in groups.values():
        if len(indices) <= 1:
            continue
        base = tuple_values(arr, indices[0])
        for idx in indices[1:]:
            value = tuple_values(arr, idx)
            max_spread = max(
                max_spread,
                l2(value[c] - base[c] for c in range(len(base))),
            )
    return max_spread


def compare_to_reference(
    grid,
    reference_grid,
    array_name: str,
    coordinate_decimals: int,
) -> dict[str, object]:
    ref_arr = get_array(reference_grid, array_name)
    ref_groups = grouped_point_indices(reference_grid, coordinate_decimals)
    reference_by_point: dict[tuple[float, float, float], list[float]] = {}
    for key, indices in ref_groups.items():
        reference_by_point[key] = tuple_values(ref_arr, indices[0])

    def exact_at_point(point: tuple[float, float, float]) -> list[float]:
        key = point_key(point, coordinate_decimals)
        if key not in reference_by_point:
            raise SystemExit(
                "Could not find matching reference point for "
                f"{point} after rounding to {coordinate_decimals} decimals"
            )
        return reference_by_point[key]

    metrics = compute_metrics(grid, array_name, exact_at_point, coordinate_decimals)
    grid_groups = grouped_point_indices(grid, coordinate_decimals)
    missing = sorted(set(reference_by_point) - set(grid_groups))
    extra = sorted(set(grid_groups) - set(reference_by_point))
    if missing or extra:
        raise SystemExit(
            f"Point-coordinate mismatch for {array_name}: "
            f"missing={len(missing)} extra={len(extra)}"
        )
    metrics["reference_points"] = len(reference_by_point)
    metrics["reference_duplicate_point_groups"] = sum(
        1 for indices in ref_groups.values() if len(indices) > 1
    )
    metrics["reference_duplicate_points"] = sum(
        max(0, len(indices) - 1) for indices in ref_groups.values()
    )
    metrics["reference_duplicate_linf_abs"] = duplicate_spread(
        reference_grid, array_name, ref_groups
    )
    return metrics


def exact_displacement(lambda_z: float) -> Callable[[tuple[float, float, float]], list[float]]:
    lambda_xy = lambda_z ** (-0.5)

    def field(point: tuple[float, float, float]) -> list[float]:
        x, y, z = point
        return [
            (lambda_xy - 1.0) * x,
            (lambda_xy - 1.0) * y,
            (lambda_z - 1.0) * z,
        ]

    return field


def exact_pressure_value(
    lambda_z: float,
    youngs_modulus: float,
    poisson_ratio: float,
    override: float | None,
) -> float:
    if override is not None:
        return override
    mu = 0.5 * youngs_modulus / (1.0 + poisson_ratio)
    return mu * (lambda_z ** (-1.0) - lambda_z * lambda_z) / 3.0


def constant_scalar(value: float) -> Callable[[tuple[float, float, float]], list[float]]:
    def field(_point: tuple[float, float, float]) -> list[float]:
        return [value]

    return field


def velocity_field(
    mode: str,
    displacement_exact: Callable[[tuple[float, float, float]], list[float]],
    scale: float,
) -> Callable[[tuple[float, float, float]], list[float]] | None:
    if mode == "skip":
        return None
    if mode == "zero":
        return lambda _point: [0.0, 0.0, 0.0]
    if mode == "scaled-displacement":
        return lambda point: [scale * v for v in displacement_exact(point)]
    raise SystemExit(f"Unsupported velocity mode {mode!r}")


def print_report(results: dict[str, dict[str, object]]) -> None:
    for name, metrics in results.items():
        print(f"\n{name}")
        print(f"  points:     {metrics['points']} unique / {metrics['raw_points']} raw")
        if int(metrics["duplicate_points"]) > 0:
            print(
                "  duplicates: "
                f"{metrics['duplicate_points']} points in "
                f"{metrics['duplicate_point_groups']} groups, "
                f"max spread {metrics['duplicate_linf_abs']:.16e}"
            )
        print(f"  components: {metrics['components']}")
        print(f"  linf abs:   {metrics['linf_abs']:.16e}")
        print(f"  rms abs:    {metrics['rms_abs']:.16e}")
        print(f"  linf rel:   {metrics['linf_rel']:.16e}")
        print(f"  rms rel:    {metrics['rms_rel']:.16e}")
        print(
            "  component linf abs: "
            + ", ".join(f"{v:.16e}" for v in metrics["component_linf_abs"])
        )
        print(
            "  component rms abs:  "
            + ", ".join(f"{v:.16e}" for v in metrics["component_rms_abs"])
        )


def check_tolerance(
    results: dict[str, dict[str, object]],
    field: str,
    key: str,
    limit: float | None,
) -> bool:
    if limit is None or field not in results:
        return True
    value = float(results[field][key])
    if value <= limit:
        return True
    print(f"FAILED: {field} {key}={value:.16e} exceeds {limit:.16e}")
    return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Assess Displacement, Velocity, and Pressure fields for the "
            "Ustruct block exact-compression benchmark."
        )
    )
    parser.add_argument(
        "solution",
        type=Path,
        help="Solution VTU file or run directory containing the VTU output.",
    )
    parser.add_argument(
        "--reference-vtu",
        type=Path,
        help="Optional reference VTU. If set, fields are compared to this file by matching point coordinates.",
    )
    parser.add_argument(
        "--coordinate-decimals",
        type=int,
        default=12,
        help="Decimal places used to match points when --reference-vtu is set.",
    )
    parser.add_argument(
        "--prefix",
        default="ustruct_uniaxial_compression",
        help="VTU file prefix when a run directory is passed.",
    )
    parser.add_argument("--step", type=int, default=1, help="Output step number.")
    parser.add_argument(
        "--lambda-z",
        type=float,
        default=0.98,
        help="Exact axial stretch for the analytical affine solution.",
    )
    parser.add_argument(
        "--youngs-modulus",
        type=float,
        default=240.56596e6,
        help="Young's modulus used to compute the default exact pressure.",
    )
    parser.add_argument(
        "--poisson-ratio",
        type=float,
        default=0.4999999,
        help="Poisson ratio used to compute the default exact pressure.",
    )
    parser.add_argument(
        "--exact-pressure",
        type=float,
        help="Override the analytical mixed pressure target.",
    )
    parser.add_argument(
        "--velocity-mode",
        choices=("zero", "scaled-displacement", "skip"),
        default="zero",
        help=(
            "Exact velocity model. Use 'scaled-displacement' with "
            "--velocity-scale for transient scheme-specific checks."
        ),
    )
    parser.add_argument(
        "--velocity-scale",
        type=float,
        default=0.0,
        help="Scale used when --velocity-mode=scaled-displacement.",
    )
    parser.add_argument("--json", type=Path, help="Optional path for JSON metrics.")
    parser.add_argument("--max-displacement-linf", type=float)
    parser.add_argument("--max-displacement-rms", type=float)
    parser.add_argument("--max-velocity-linf", type=float)
    parser.add_argument("--max-velocity-rms", type=float)
    parser.add_argument("--max-pressure-linf", type=float)
    parser.add_argument("--max-pressure-rms", type=float)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    solution_path = resolve_vtu(args.solution, args.prefix, args.step)
    grid = read_grid(solution_path)

    results: dict[str, dict[str, object]] = {}
    if args.reference_vtu is not None:
        reference_path = resolve_vtu(args.reference_vtu, args.prefix, args.step)
        reference_grid = read_grid(reference_path)
        for field in ("Displacement", "Velocity", "Pressure"):
            results[field] = compare_to_reference(
                grid, reference_grid, field, args.coordinate_decimals
            )
    else:
        disp = exact_displacement(args.lambda_z)
        pressure = exact_pressure_value(
            args.lambda_z,
            args.youngs_modulus,
            args.poisson_ratio,
            args.exact_pressure,
        )
        vel = velocity_field(args.velocity_mode, disp, args.velocity_scale)

        results["Displacement"] = compute_metrics(
            grid, "Displacement", disp, args.coordinate_decimals
        )
        if vel is not None:
            results["Velocity"] = compute_metrics(
                grid, "Velocity", vel, args.coordinate_decimals
            )
        results["Pressure"] = compute_metrics(
            grid,
            "Pressure",
            constant_scalar(pressure),
            args.coordinate_decimals,
        )

    print(f"solution: {solution_path}")
    if args.reference_vtu is not None:
        print(f"reference: {resolve_vtu(args.reference_vtu, args.prefix, args.step)}")
    else:
        print(f"lambda_z: {args.lambda_z:.16e}")
        print(
            "exact pressure: "
            f"{exact_pressure_value(args.lambda_z, args.youngs_modulus, args.poisson_ratio, args.exact_pressure):.16e}"
        )
        print(f"velocity mode: {args.velocity_mode}")
    print_report(results)

    if args.json is not None:
        args.json.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    ok = True
    ok &= check_tolerance(results, "Displacement", "linf_abs", args.max_displacement_linf)
    ok &= check_tolerance(results, "Displacement", "rms_abs", args.max_displacement_rms)
    ok &= check_tolerance(results, "Velocity", "linf_abs", args.max_velocity_linf)
    ok &= check_tolerance(results, "Velocity", "rms_abs", args.max_velocity_rms)
    ok &= check_tolerance(results, "Pressure", "linf_abs", args.max_pressure_linf)
    ok &= check_tolerance(results, "Pressure", "rms_abs", args.max_pressure_rms)
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())

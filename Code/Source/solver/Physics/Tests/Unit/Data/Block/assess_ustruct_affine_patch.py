#!/usr/bin/env python3
"""Assess generated OOP Ustruct affine patch cases against exact fields."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Callable, Iterable


THIS_DIR = Path(__file__).resolve().parent
MANIFEST = THIS_DIR / "affine_patch_cases.json"


def import_vtk():
    try:
        import vtk  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "This script requires the Python VTK package. "
            "Install it or run from an environment where 'import vtk' works."
        ) from exc
    return vtk


def load_manifest() -> dict[str, object]:
    if not MANIFEST.exists():
        raise SystemExit(f"Missing manifest: {MANIFEST}")
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def resolve_case(raw: str, manifest: dict[str, object]) -> tuple[str, dict[str, object]]:
    cases = manifest.get("cases")
    if not isinstance(cases, dict):
        raise SystemExit("Manifest does not contain a 'cases' object")

    key = raw[:-4] if raw.endswith(".xml") else raw
    key = key[:-4] if key.endswith("_oop") else key
    if key in cases and isinstance(cases[key], dict):
        return key, cases[key]

    for case_id, case_data in cases.items():
        if isinstance(case_data, dict) and raw == case_data.get("xml"):
            return str(case_id), case_data

    available = "\n".join(f"  {name}" for name in sorted(cases))
    raise SystemExit(f"Unknown case {raw!r}. Available cases:\n{available}")


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

    matches: list[Path] = []
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


def compute_metrics(
    grid,
    array_name: str,
    exact_at_point: Callable[[tuple[float, float, float]], list[float]],
    coordinate_decimals: int,
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
        point = tuple(float(v) for v in grid.GetPoint(i))
        exact = exact_at_point(point)
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
        "linf_rel": max_abs / max(max_exact, 1.0e-30),
        "rms_rel": rms_abs / max(rms_exact, 1.0e-30),
        "component_linf_abs": component_max_abs,
        "component_rms_abs": [
            math.sqrt(component_sum_err2[c] / n) for c in range(ncomp)
        ],
    }


def affine_displacement(f: list[list[float]]) -> Callable[[tuple[float, float, float]], list[float]]:
    dim = len(f)

    def field(point: tuple[float, float, float]) -> list[float]:
        return [
            sum((f[i][j] - (1.0 if i == j else 0.0)) * point[j] for j in range(dim))
            for i in range(dim)
        ]

    return field


def constant_vector(values: list[float]) -> Callable[[tuple[float, float, float]], list[float]]:
    def field(_point: tuple[float, float, float]) -> list[float]:
        return values

    return field


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


def tolerance_for(args: argparse.Namespace, field: str, case: dict[str, object]) -> float | None:
    override = {
        "Displacement": args.max_displacement_linf,
        "Velocity": args.max_velocity_linf,
        "Pressure": args.max_pressure_linf,
        "Jacobian": args.max_jacobian_linf,
    }.get(field)
    if override is not None:
        return override

    defaults = case.get("default_tolerances", {})
    if not isinstance(defaults, dict):
        return None
    field_defaults = defaults.get(field, {})
    if not isinstance(field_defaults, dict):
        return None
    value = field_defaults.get("linf_abs")
    return float(value) if value is not None else None


def check_tolerances(
    args: argparse.Namespace,
    results: dict[str, dict[str, object]],
    case: dict[str, object],
) -> bool:
    if args.no_check:
        return True

    ok = True
    for field, metrics in results.items():
        limit = tolerance_for(args, field, case)
        if limit is None:
            continue
        value = float(metrics["linf_abs"])
        if value > limit:
            print(f"FAILED: {field} linf_abs={value:.16e} exceeds {limit:.16e}")
            ok = False
    return ok


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Assess an OOP Ustruct affine patch case against its exact analytical solution."
    )
    parser.add_argument(
        "case",
        help="Case id from affine_patch_cases.json, or generated XML filename.",
    )
    parser.add_argument(
        "solution",
        nargs="?",
        type=Path,
        default=Path("."),
        help="Solution VTU/PVTU file or run directory. Defaults to the current directory.",
    )
    parser.add_argument("--step", type=int, default=1, help="Output step number.")
    parser.add_argument(
        "--coordinate-decimals",
        type=int,
        default=12,
        help="Decimal places used to coalesce duplicate MPI output points.",
    )
    parser.add_argument(
        "--fields",
        nargs="+",
        default=["Displacement", "Velocity", "Pressure"],
        choices=[
            "Displacement",
            "Velocity",
            "Pressure",
            "Def_grad",
            "Jacobian",
            "Divergence",
            "Strain",
            "Stress",
            "Cauchy_stress",
            "VonMises_stress",
        ],
        help="Point-data fields to compare.",
    )
    parser.add_argument("--json", type=Path, help="Optional path for JSON metrics.")
    parser.add_argument("--no-check", action="store_true", help="Report metrics without enforcing tolerances.")
    parser.add_argument("--max-displacement-linf", type=float)
    parser.add_argument("--max-velocity-linf", type=float)
    parser.add_argument("--max-pressure-linf", type=float)
    parser.add_argument("--max-jacobian-linf", type=float)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = load_manifest()
    case_id, case = resolve_case(args.case, manifest)
    prefix = str(case["prefix"])
    solution_path = resolve_vtu(args.solution, prefix, args.step)
    grid = read_grid(solution_path)

    f = case["deformation_gradient"]
    if not isinstance(f, list):
        raise SystemExit("Case manifest has invalid deformation_gradient")
    pressure = float(case["pressure"])
    jacobian = float(case["jacobian"])
    dim = int(case.get("dim", len(f)))
    exact_outputs = case.get("exact_fields", {})
    if not isinstance(exact_outputs, dict):
        exact_outputs = {}

    exact_fields: dict[str, Callable[[tuple[float, float, float]], list[float]]] = {
        "Displacement": affine_displacement(f),  # type: ignore[arg-type]
        "Velocity": constant_vector([0.0 for _ in range(dim)]),
        "Pressure": constant_vector([pressure]),
        "Jacobian": constant_vector([jacobian]),
    }
    for name in ("Def_grad", "Divergence", "Strain", "Stress", "Cauchy_stress", "VonMises_stress"):
        values = exact_outputs.get(name)
        if isinstance(values, list):
            exact_fields[name] = constant_vector([float(v) for v in values])

    results: dict[str, dict[str, object]] = {}
    for field in args.fields:
        if field not in exact_fields:
            raise SystemExit(f"Case manifest does not provide an exact field for {field!r}")
        results[field] = compute_metrics(
            grid, field, exact_fields[field], args.coordinate_decimals
        )

    print(f"case: {case_id}")
    print(f"description: {case['description']}")
    print(f"solution: {solution_path}")
    print(f"exact pressure: {pressure:.16e}")
    print(f"exact jacobian: {jacobian:.16e}")
    print_report(results)

    if args.json is not None:
        args.json.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    return 0 if check_tolerances(args, results, case) else 2


if __name__ == "__main__":
    raise SystemExit(main())

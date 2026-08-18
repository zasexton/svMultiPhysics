#!/usr/bin/env python3
"""Gate LagrangeBasis microbenchmark CSV output.

The gate is intentionally CSV-schema driven so it can run on local benchmark
captures and CI artifacts without importing project code.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import statistics
import sys
from pathlib import Path


TOPOLOGIES = ("line", "triangle", "quad", "tet", "hex", "wedge", "pyramid")
TENSOR_TOPOLOGIES = {"line", "quad", "hex"}
SIMPLEX_TOPOLOGIES = {"triangle", "tet"}
HOT_SCALAR_OPS = ("values", "gradients", "hessians", "all")
HOT_RAW_TO_OPS = ("values", "gradients", "hessians", "all")
HOT_STRIDED_OPS = ("values", "gradients", "hessians", "all")
REQUIRED_METADATA_COLUMNS = (
    "compiler_id",
    "compiler_version",
    "build_flags",
    "cpu_model",
    "bench_threads",
    "simd_width_bytes",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check LagrangeBasis benchmark CSV rows for allocations and regressions."
    )
    parser.add_argument("--current", required=True, help="Current basis_perf_microbench CSV")
    parser.add_argument("--baseline", help="Optional accepted baseline CSV")
    parser.add_argument("--min-order", type=int, default=0)
    parser.add_argument("--max-order", type=int, default=8)
    parser.add_argument("--fail-on-hot-allocations", action="store_true")
    parser.add_argument("--max-case-slowdown", type=float, default=1.25)
    parser.add_argument("--warn-case-slowdown", type=float, default=1.10)
    parser.add_argument("--max-category-geomean-slowdown", type=float, default=1.10)
    parser.add_argument("--warn-variance-ratio", type=float, default=1.25)
    parser.add_argument("--enforce-roofline-thresholds", action="store_true")
    parser.add_argument("--tensor-roofline-threshold", type=float, default=4.0)
    parser.add_argument("--simplex-roofline-threshold", type=float, default=5.0)
    parser.add_argument("--wedge-roofline-threshold", type=float, default=5.0)
    return parser.parse_args()


def load_csv(path: Path) -> tuple[list[str], dict[str, dict[str, str]]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path} is empty or has no header")
        rows: dict[str, dict[str, str]] = {}
        for row_number, row in enumerate(reader, start=2):
            name = (row.get("case") or "").strip()
            if not name:
                raise ValueError(f"{path}:{row_number}: missing case column")
            if name in rows:
                raise ValueError(f"{path}:{row_number}: duplicate case {name!r}")
            rows[name] = row
        return list(reader.fieldnames), rows


def required_hot_cases(min_order: int, max_order: int) -> set[str]:
    required: set[str] = set()
    for topology in TOPOLOGIES:
        for order in range(min_order, max_order + 1):
            prefix = f"lagrange_{topology}_order{order}"
            for op in HOT_SCALAR_OPS:
                required.add(f"{prefix}_point_{op}")
            for op in HOT_RAW_TO_OPS:
                required.add(f"{prefix}_to_{op}")
            for op in HOT_STRIDED_OPS:
                required.add(f"{prefix}_strided_{op}")
    return required


def as_float(row: dict[str, str], column: str, default: float = 0.0) -> float:
    value = (row.get(column) or "").strip()
    if not value:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def nonempty_metadata(row: dict[str, str], column: str) -> bool:
    value = (row.get(column) or "").strip()
    return bool(value) and value.lower() != "unknown"


def parse_lagrange_case(case: str) -> tuple[str, int, str] | None:
    match = re.fullmatch(
        r"lagrange_(line|triangle|quad|tet|hex|wedge|pyramid)_order(\d+)_(.+)",
        case,
    )
    if not match:
        return None
    return match.group(1), int(match.group(2)), match.group(3)


def geomean(values: list[float]) -> float:
    return math.exp(sum(math.log(v) for v in values) / len(values))


def check_required_rows(
    rows: dict[str, dict[str, str]], required: set[str], failures: list[str]
) -> None:
    missing = sorted(required.difference(rows))
    for case in missing:
        failures.append(f"missing required Lagrange hot row: {case}")


def check_metadata(
    label: str,
    rows: dict[str, dict[str, str]],
    required: set[str],
    failures: list[str],
) -> None:
    for case in sorted(required.intersection(rows)):
        row = rows[case]
        for column in REQUIRED_METADATA_COLUMNS:
            if column not in row:
                failures.append(f"{label} row {case} is missing metadata column {column}")
            elif not nonempty_metadata(row, column):
                failures.append(f"{label} row {case} has empty/unknown metadata {column}")


def check_allocations(
    rows: dict[str, dict[str, str]],
    failures: list[str],
    fail_on_hot_allocations: bool,
) -> None:
    if not fail_on_hot_allocations:
        return
    for case, row in sorted(rows.items()):
        if not case.startswith("lagrange_"):
            continue
        if row.get("category") == "lagrange_construction":
            continue
        allocations_per_call = as_float(row, "allocations_per_call")
        if allocations_per_call != 0.0:
            failures.append(
                f"hot Lagrange row allocates: {case} allocations_per_call={allocations_per_call:g}"
            )


def check_baseline_regressions(
    current: dict[str, dict[str, str]],
    baseline: dict[str, dict[str, str]],
    required: set[str],
    args: argparse.Namespace,
    failures: list[str],
    warnings: list[str],
) -> None:
    comparable_by_category: dict[str, list[float]] = {}
    for case in sorted(required):
        if case not in current:
            continue
        if case not in baseline:
            failures.append(f"baseline is missing required row: {case}")
            continue
        current_ns = as_float(current[case], "ns_per_call")
        baseline_ns = as_float(baseline[case], "ns_per_call")
        if current_ns <= 0.0 or baseline_ns <= 0.0:
            failures.append(
                f"cannot compare {case}: current_ns={current_ns:g}, baseline_ns={baseline_ns:g}"
            )
            continue
        slowdown = current_ns / baseline_ns
        category = current[case].get("category") or "unknown"
        comparable_by_category.setdefault(category, []).append(slowdown)
        if slowdown > args.max_case_slowdown:
            failures.append(
                f"{case} slowdown {slowdown:.3f} exceeds {args.max_case_slowdown:.3f} "
                f"(current {current_ns:.6g} ns, baseline {baseline_ns:.6g} ns)"
            )
        elif slowdown > args.warn_case_slowdown:
            warnings.append(
                f"{case} slowdown {slowdown:.3f} exceeds warning threshold "
                f"{args.warn_case_slowdown:.3f}"
            )

    for category, slowdowns in sorted(comparable_by_category.items()):
        if not slowdowns:
            continue
        category_geomean = geomean(slowdowns)
        if category_geomean > args.max_category_geomean_slowdown:
            failures.append(
                f"{category} geometric-mean slowdown {category_geomean:.3f} exceeds "
                f"{args.max_category_geomean_slowdown:.3f}"
            )


def check_variance(
    rows: dict[str, dict[str, str]],
    required: set[str],
    threshold: float,
    warnings: list[str],
) -> None:
    for case in sorted(required.intersection(rows)):
        row = rows[case]
        min_ns = as_float(row, "min_ns_per_call")
        max_ns = as_float(row, "max_ns_per_call")
        if min_ns <= 0.0 or max_ns <= 0.0:
            continue
        ratio = max_ns / min_ns
        if ratio > threshold:
            warnings.append(f"{case} variance ratio {ratio:.3f} exceeds {threshold:.3f}")


def check_roofline(
    rows: dict[str, dict[str, str]],
    required: set[str],
    args: argparse.Namespace,
    failures: list[str],
    warnings: list[str],
) -> None:
    if not args.enforce_roofline_thresholds:
        return
    for case in sorted(required.intersection(rows)):
        row = rows[case]
        if row.get("category") != "lagrange_strided_batch":
            continue
        parsed = parse_lagrange_case(case)
        if parsed is None:
            continue
        topology, order, _ = parsed
        if order < 2:
            continue
        if topology == "pyramid":
            warnings.append(f"{case} roofline ratio is tracked but not gated for pyramid")
            continue
        ratio = as_float(row, "measured_to_model_bound")
        if ratio <= 0.0:
            failures.append(
                f"{case} has no positive measured_to_model_bound; supply "
                "SVMP_BASIS_BENCH_PEAK_GFLOPS and SVMP_BASIS_BENCH_STREAM_GBPS"
            )
            continue
        if topology in TENSOR_TOPOLOGIES:
            threshold = args.tensor_roofline_threshold
        elif topology in SIMPLEX_TOPOLOGIES:
            threshold = args.simplex_roofline_threshold
        else:
            threshold = args.wedge_roofline_threshold
        if ratio > threshold:
            failures.append(
                f"{case} roofline ratio {ratio:.3f} exceeds threshold {threshold:.3f}"
            )


def print_messages(kind: str, messages: list[str]) -> None:
    if not messages:
        return
    print(f"{kind}:")
    for message in messages:
        print(f"  - {message}")


def main() -> int:
    args = parse_args()
    current_path = Path(args.current)
    baseline_path = Path(args.baseline) if args.baseline else None
    failures: list[str] = []
    warnings: list[str] = []

    try:
        _, current = load_csv(current_path)
        baseline = None
        if baseline_path is not None:
            _, baseline = load_csv(baseline_path)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    required = required_hot_cases(args.min_order, args.max_order)
    check_required_rows(current, required, failures)
    check_metadata("current", current, required, failures)
    check_allocations(current, failures, args.fail_on_hot_allocations)
    check_variance(current, required, args.warn_variance_ratio, warnings)
    check_roofline(current, required, args, failures, warnings)

    if baseline is not None:
        check_required_rows(baseline, required, failures)
        check_metadata("baseline", baseline, required, failures)
        check_baseline_regressions(
            current, baseline, required, args, failures, warnings
        )

    print_messages("warnings", warnings)
    if failures:
        print_messages("failures", failures)
        return 1

    comparable = len(required.intersection(current))
    print(
        f"LagrangeBasis benchmark gate passed: {comparable}/{len(required)} "
        "required hot rows present"
    )
    if warnings:
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

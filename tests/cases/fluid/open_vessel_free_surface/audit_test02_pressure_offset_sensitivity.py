#!/usr/bin/env python3
"""Audit whether Test02 pressure errors are mostly a common pressure offset."""

from __future__ import annotations

import argparse
import csv
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np

import verify_spheric_test02_histories as verifier


PRIMARY_HEIGHT_TRACES = ("H4", "H2")
PRIMARY_PRESSURE_TRACES = ("P1", "P3", "P5", "P7")
DEFAULT_RHO = 998.2
DEFAULT_GRAVITY = 9.81


def pressure_anchor(case_dir: Path) -> dict[str, Any] | None:
    csv_path = case_dir / "pressure_gauge.csv"
    if not csv_path.exists():
        return None
    with csv_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        return None
    row = rows[0]
    anchor: dict[str, Any] = {
        "node_id": int(row["node_id"]),
        "pressure_pa": float(row["pressure"]),
    }
    benchmark_path = case_dir / "benchmark.json"
    if benchmark_path.exists():
        benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
        meta = benchmark.get("pressure_gauge", {})
        if "coordinates" in meta:
            anchor["coordinates_m"] = [float(value) for value in meta["coordinates"]]
    return anchor


def result_times(case_dir: Path, prefix: str, solver_log: Path | None) -> dict[str, float]:
    pvd_times = verifier.result_times_from_pvd(case_dir, prefix)
    if pvd_times:
        return pvd_times
    return verifier.result_times_from_solver_log(solver_log, prefix)


def load_samples(
    case_dir: Path,
    *,
    result_prefix: str,
    solver_log: Path | None,
    height_traces: tuple[str, ...],
    pressure_traces: tuple[str, ...],
) -> list[dict[str, Any]]:
    setup = verifier.parse_solver_xml(case_dir / "solver.xml")
    times = result_times(case_dir, result_prefix, solver_log)
    return [
        verifier.sample_result(
            result,
            prefix=result_prefix,
            dt=setup["time_step_size_s"],
            result_times=times,
            height_traces=height_traces,
            pressure_traces=pressure_traces,
        )
        for result in verifier.output_results(case_dir, result_prefix)
    ]


def matrices(
    samples: list[dict[str, Any]],
    reference: dict[str, np.ndarray],
    *,
    pressure_traces: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    times = np.asarray([sample["time_s"] for sample in samples], dtype=float)
    simulated = np.asarray(
        [
            [sample["pressure"][trace]["pressure_pa"] for trace in pressure_traces]
            for sample in samples
        ],
        dtype=float,
    )
    ref = np.asarray(
        [np.interp(times, reference["Time"], reference[trace]) for trace in pressure_traces],
        dtype=float,
    ).T
    return times, simulated, ref


def trace_metrics(
    times: np.ndarray,
    values: np.ndarray,
    reference_values: np.ndarray,
) -> dict[str, Any]:
    errors = values - reference_values
    peak_index = int(np.argmax(values))
    reference_peak_index = int(np.argmax(reference_values))
    return {
        "rmse_pa": float(math.sqrt(float(np.mean(errors * errors)))),
        "max_abs_error_pa": float(np.max(np.abs(errors))),
        "final_value_pa": float(values[-1]),
        "final_reference_pa": float(reference_values[-1]),
        "final_error_pa": float(errors[-1]),
        "simulated_peak_over_sample_window_pa": float(values[peak_index]),
        "simulated_peak_time_s": float(times[peak_index]),
        "reference_peak_over_sample_window_pa": float(reference_values[reference_peak_index]),
        "reference_peak_time_s": float(times[reference_peak_index]),
    }


def scenario_report(
    name: str,
    times: np.ndarray,
    values: np.ndarray,
    reference: np.ndarray,
    *,
    pressure_traces: tuple[str, ...],
    offsets: np.ndarray,
    offset_description: str,
) -> dict[str, Any]:
    corrected = values + offsets.reshape(-1, 1)
    errors = corrected - reference
    traces = {
        trace: trace_metrics(times, corrected[:, index], reference[:, index])
        for index, trace in enumerate(pressure_traces)
    }
    p1_peak = traces.get("P1", {}).get("simulated_peak_over_sample_window_pa")
    p3_peak = traces.get("P3", {}).get("simulated_peak_over_sample_window_pa")
    ref_p1_peak = traces.get("P1", {}).get("reference_peak_over_sample_window_pa")
    ref_p3_peak = traces.get("P3", {}).get("reference_peak_over_sample_window_pa")
    return {
        "name": name,
        "offset_description": offset_description,
        "offset_min_pa": float(np.min(offsets)) if offsets.size else None,
        "offset_max_pa": float(np.max(offsets)) if offsets.size else None,
        "offset_mean_pa": float(np.mean(offsets)) if offsets.size else None,
        "aggregate_rmse_pa": float(math.sqrt(float(np.mean(errors * errors)))),
        "aggregate_max_abs_error_pa": float(np.max(np.abs(errors))),
        "traces": traces,
        "p3_over_p1_peak_ratio": (
            float(p3_peak / p1_peak) if p1_peak not in (None, 0.0) and p3_peak is not None else None
        ),
        "reference_p3_over_p1_peak_ratio": (
            float(ref_p3_peak / ref_p1_peak)
            if ref_p1_peak not in (None, 0.0) and ref_p3_peak is not None
            else None
        ),
    }


def h4_hydrostatic_offsets(
    samples: list[dict[str, Any]],
    *,
    anchor: dict[str, Any] | None,
    rho: float,
    gravity: float,
) -> np.ndarray:
    if anchor is None or "coordinates_m" not in anchor:
        return np.zeros(len(samples), dtype=float)
    anchor_y = float(anchor["coordinates_m"][1])
    anchor_pressure = float(anchor["pressure_pa"])
    heights = np.asarray(
        [sample["height"]["H4"]["height_m"] for sample in samples],
        dtype=float,
    )
    expected = rho * gravity * np.maximum(0.0, heights - anchor_y)
    return expected - anchor_pressure


def audit_case(
    case_dir: Path,
    reference_csv: Path,
    *,
    result_prefix: str,
    solver_log: Path | None,
    pressure_traces: tuple[str, ...],
    rho: float,
    gravity: float,
) -> dict[str, Any]:
    reference = verifier.load_reference_csv(reference_csv)
    samples = load_samples(
        case_dir,
        result_prefix=result_prefix,
        solver_log=solver_log,
        height_traces=PRIMARY_HEIGHT_TRACES,
        pressure_traces=pressure_traces,
    )
    if not samples:
        return {
            "case_dir": str(case_dir),
            "reference_csv": str(reference_csv),
            "result_count": 0,
            "status": "no_results",
        }

    times, simulated, reference_values = matrices(
        samples,
        reference,
        pressure_traces=pressure_traces,
    )
    anchor = pressure_anchor(case_dir)
    scenarios: dict[str, Any] = {}

    raw_offsets = np.zeros(len(samples), dtype=float)
    scenarios["raw"] = scenario_report(
        "raw",
        times,
        simulated,
        reference_values,
        pressure_traces=pressure_traces,
        offsets=raw_offsets,
        offset_description="no pressure offset applied",
    )

    constant_offset_value = float(np.mean(reference_values - simulated))
    scenarios["best_constant_common_offset"] = scenario_report(
        "best_constant_common_offset",
        times,
        simulated,
        reference_values,
        pressure_traces=pressure_traces,
        offsets=np.full(len(samples), constant_offset_value, dtype=float),
        offset_description="single least-squares scalar offset over all sampled pressure traces and times",
    )

    per_time_offsets = np.mean(reference_values - simulated, axis=1)
    scenarios["best_per_time_common_offset"] = scenario_report(
        "best_per_time_common_offset",
        times,
        simulated,
        reference_values,
        pressure_traces=pressure_traces,
        offsets=per_time_offsets,
        offset_description="optimistic least-squares scalar offset recomputed independently at each saved time",
    )

    if "P1" in pressure_traces:
        p1_index = pressure_traces.index("P1")
        p1_offsets = reference_values[:, p1_index] - simulated[:, p1_index]
        scenarios["p1_aligned_per_time_offset"] = scenario_report(
            "p1_aligned_per_time_offset",
            times,
            simulated,
            reference_values,
            pressure_traces=pressure_traces,
            offsets=p1_offsets,
            offset_description="per-time scalar offset that exactly aligns P1 before applying the same offset to all pressure traces",
        )

    scenarios["h4_hydrostatic_anchor_offset"] = scenario_report(
        "h4_hydrostatic_anchor_offset",
        times,
        simulated,
        reference_values,
        pressure_traces=pressure_traces,
        offsets=h4_hydrostatic_offsets(samples, anchor=anchor, rho=rho, gravity=gravity),
        offset_description="per-time offset that would make the fixed pressure anchor follow local H4 hydrostatic pressure inferred from simulated H4 height",
    )

    raw = scenarios["raw"]
    optimistic = scenarios["best_per_time_common_offset"]
    raw_p3_ratio = raw["p3_over_p1_peak_ratio"]
    optimistic_p3_ratio = optimistic["p3_over_p1_peak_ratio"]
    reference_ratio = raw["reference_p3_over_p1_peak_ratio"]
    raw_rmse = float(raw["aggregate_rmse_pa"])
    optimistic_rmse = float(optimistic["aggregate_rmse_pa"])
    if raw_rmse > 0.0:
        rmse_reduction = 1.0 - optimistic_rmse / raw_rmse
    else:
        rmse_reduction = 0.0

    if reference_ratio and optimistic_p3_ratio is not None:
        ratio_gap = abs(optimistic_p3_ratio - reference_ratio)
    else:
        ratio_gap = None

    finding = (
        "A common pressure offset is not the primary Test02 pressure-stack fix. "
        f"The optimistic per-time common offset reduces aggregate RMSE by {rmse_reduction:.3g}, "
        f"but the P3/P1 peak ratio changes from {raw_p3_ratio:.6g} raw to "
        f"{optimistic_p3_ratio:.6g}, versus reference {reference_ratio:.6g}."
    )
    if ratio_gap is not None and ratio_gap < 0.1:
        finding = (
            "A common pressure offset materially improves the Test02 pressure-stack ratio; "
            "this should be followed by a pressure-anchor/nullspace control run. "
            f"Raw P3/P1={raw_p3_ratio:.6g}, offset P3/P1={optimistic_p3_ratio:.6g}, "
            f"reference={reference_ratio:.6g}."
        )

    return {
        "case_dir": str(case_dir),
        "reference_csv": str(reference_csv),
        "result_prefix": result_prefix,
        "result_count": len(samples),
        "sampled_time_start_s": float(times[0]),
        "sampled_time_end_s": float(times[-1]),
        "pressure_traces": list(pressure_traces),
        "pressure_anchor": anchor,
        "rho_kg_per_m3": rho,
        "gravity_m_per_s2": gravity,
        "scenarios": scenarios,
        "finding": finding,
        "status": "diagnostic_pressure_offset_sensitivity_not_validation_gate",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-dir", type=Path, required=True)
    parser.add_argument("--reference-csv", type=Path, required=True)
    parser.add_argument("--result-prefix", default="result")
    parser.add_argument("--solver-log", type=Path)
    parser.add_argument("--pressure-traces", default=",".join(PRIMARY_PRESSURE_TRACES))
    parser.add_argument("--rho", type=float, default=DEFAULT_RHO)
    parser.add_argument("--gravity", type=float, default=DEFAULT_GRAVITY)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()

    pressure_traces = tuple(
        trace.strip() for trace in args.pressure_traces.split(",") if trace.strip()
    )
    report = audit_case(
        args.case_dir,
        args.reference_csv,
        result_prefix=args.result_prefix,
        solver_log=args.solver_log,
        pressure_traces=pressure_traces,
        rho=args.rho,
        gravity=args.gravity,
    )
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

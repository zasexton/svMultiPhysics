#!/usr/bin/env python3
"""Run and summarize the new OOP solver fluid case matrix.

The runner is intentionally separate from legacy comparison scripts. It copies
each case into the output directory before executing svmultiphysics so case
fixtures are not dirtied by restart files, histor.dat, or result output.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET


DEFAULT_CASES = (
    "Channel2D:solver_perf_oop.xml",
    "Channel2D_Simple:solver_perf_oop.xml",
    "vortex_shedding:solver_perf_oop.xml",
    "iliac_artery:solver_perf_oop.xml",
    "pipe_simple:solver_perf_oop.xml",
    "pipe_RCR_3d:solver_perf_oop.xml",
)

FLOAT = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
TIME_STEPPING_RE = re.compile(
    r"Number_of_time_steps=(?P<steps>\d+)\s+"
    r"Time_step_size=(?P<dt>" + FLOAT + r")"
)
NONLINEAR_DONE_RE = re.compile(
    r"TimeLoop:\s+nonlinear_done\s+step=(?P<step>\d+)\s+"
    r"time=(?P<time>" + FLOAT + r")\s+"
    r"converged=(?P<converged>[01])\s+iters=(?P<iters>\d+)\s+"
    r"\|\|r\|\|=(?P<residual>" + FLOAT + r").*?"
    r"\(linear:\s+converged=(?P<linear_converged>[01])\s+"
    r"iters=(?P<linear_iters>\d+)\s+rel=(?P<linear_rel>" + FLOAT + r")\)"
)
STEP_ACCEPTED_RE = re.compile(
    r"TimeLoop:\s+step_accepted\s+step=(?P<step>\d+)\s+"
    r"time=(?P<time>" + FLOAT + r")\s+dt=(?P<dt>" + FLOAT + r")"
)
NEWTON_TIMING_RE = re.compile(
    r"Total Newton time:\s+(?P<seconds>" + FLOAT + r")\s+s\s+"
    r"\((?P<newton_iters>\d+)\s+Newton iters,\s+"
    r"(?P<assemblies>\d+)\s+assemblies,\s+"
    r"(?P<linear_iters>\d+)\s+linear iters\)"
)
TOTAL_LOOP_RE = re.compile(r"Total time loop:\s+(?P<seconds>" + FLOAT + r")\s+s")
CUT_CONTEXT_RE = re.compile(
    r"Active-domain cut context diagnostic=cut_context_rebuild\b(?P<body>.*)"
)
KEY_VALUE_RE = re.compile(
    r"(?P<key>[A-Za-z0-9_]+)=(?P<value>'[^']*'|\"[^\"]*\"|\S+)"
)


def log_value(raw: str) -> object:
    value = raw.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    if re.fullmatch(r"[-+]?\d+", value):
        try:
            return int(value)
        except ValueError:
            return value
    if re.fullmatch(FLOAT, value):
        try:
            return float(value)
        except ValueError:
            return value
    return value


def numeric_values(records: list[dict[str, object]], key: str) -> list[float]:
    values: list[float] = []
    for record in records:
        value = record.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            values.append(float(value))
    return values


def max_numeric(records: list[dict[str, object]], key: str) -> float | None:
    values = numeric_values(records, key)
    return max(values) if values else None


def min_numeric(records: list[dict[str, object]], key: str) -> float | None:
    values = numeric_values(records, key)
    return min(values) if values else None


def first_numeric(records: list[dict[str, object]], key: str) -> float | None:
    values = numeric_values(records, key)
    return values[0] if values else None


def last_numeric(records: list[dict[str, object]], key: str) -> float | None:
    values = numeric_values(records, key)
    return values[-1] if values else None


def max_growth_from_first(records: list[dict[str, object]], key: str) -> float | None:
    values = numeric_values(records, key)
    if not values:
        return None
    return max(values) - values[0]


def sum_numeric(records: list[dict[str, object]], key: str) -> float:
    return sum(numeric_values(records, key))


def unique_strings(records: list[dict[str, object]], key: str) -> list[str]:
    values = {
        str(record[key])
        for record in records
        if key in record and record[key] not in (None, "")
    }
    return sorted(values)


def count_strings(records: list[dict[str, object]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        if key not in record:
            continue
        value = str(record[key])
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def parse_cut_context_rebuilds(text: str) -> dict[str, object]:
    records: list[dict[str, object]] = []
    for line in text.splitlines():
        match = CUT_CONTEXT_RE.search(line)
        if not match:
            continue
        record: dict[str, object] = {}
        for kv in KEY_VALUE_RE.finditer(match.group("body")):
            record[kv.group("key")] = log_value(kv.group("value"))
        records.append(record)

    return {
        "count": len(records),
        "provenance_counts": count_strings(records, "provenance"),
        "geometry_modes": unique_strings(records, "generated_interface_geometry"),
        "implicit_cut_backends": unique_strings(records, "implicit_cut_quadrature_backend"),
        "geometry_tangent_policies": unique_strings(records, "geometry_tangent_policy"),
        "backend_seconds_sum": sum_numeric(records, "implicit_cut_backend_seconds"),
        "backend_seconds_min": min_numeric(records, "implicit_cut_backend_seconds"),
        "backend_seconds_mean": (
            sum_numeric(records, "implicit_cut_backend_seconds") / len(records)
            if records else None
        ),
        "backend_seconds_max": max_numeric(records, "implicit_cut_backend_seconds"),
        "max_active_cut_cells": max_numeric(records, "active_cut_cells"),
        "max_active_full_wet_cells": max_numeric(records, "active_full_wet_cells"),
        "max_active_full_dry_cells": max_numeric(records, "active_full_dry_cells"),
        "max_active_quadrature_points": max_numeric(records, "active_quadrature_points"),
        "max_active_volume_quadrature_point_count": max_numeric(
            records, "active_volume_quadrature_point_count"),
        "max_interface_quadrature_point_count": max_numeric(
            records, "interface_quadrature_point_count"),
        "max_active_volume_regions": max_numeric(records, "active_volume_regions"),
        "max_active_interface_fragments": max_numeric(records, "active_interface_fragments"),
        "max_implicit_cut_fallback_cells": max_numeric(records, "implicit_cut_fallback_cells"),
        "max_corner_linearized_cells": max_numeric(records, "corner_linearized_cells"),
        "min_achieved_interface_quadrature_order": min_numeric(
            records, "achieved_interface_quadrature_order"),
        "min_achieved_volume_quadrature_order": min_numeric(
            records, "achieved_volume_quadrature_order"),
        "max_process_vm_kb": max_numeric(records, "process_vm_kb"),
        "max_process_rss_kb": max_numeric(records, "process_rss_kb"),
        "first_process_rss_kb": first_numeric(records, "process_rss_kb"),
        "last_process_rss_kb": last_numeric(records, "process_rss_kb"),
        "process_rss_kb_growth": max_growth_from_first(records, "process_rss_kb"),
        "process_vm_kb_growth": max_growth_from_first(records, "process_vm_kb"),
        "max_basis_cache_entries": max_numeric(records, "basis_cache_entries"),
        "first_basis_cache_entries": first_numeric(records, "basis_cache_entries"),
        "last_basis_cache_entries": last_numeric(records, "basis_cache_entries"),
        "basis_cache_entry_growth": max_growth_from_first(records, "basis_cache_entries"),
    }


def run_identity(case_name: object, xml: object, ranks: object) -> str:
    return f"{case_name}|{xml}|{ranks}"


def load_baseline_runs(path: Path) -> dict[str, dict[str, object]]:
    summary = json.loads(path.read_text())
    runs = summary.get("runs")
    if not isinstance(runs, list):
        raise ValueError(f"baseline summary has no runs list: {path}")
    by_key: dict[str, dict[str, object]] = {}
    for run in runs:
        if not isinstance(run, dict):
            continue
        if not all(key in run for key in ("case", "xml", "ranks")):
            continue
        by_key[run_identity(run["case"], run["xml"], run["ranks"])] = run
    return by_key


def nested_metric(mapping: dict[str, object] | None, path: tuple[str, ...]) -> object:
    value: object = mapping or {}
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def numeric_metric(mapping: dict[str, object] | None, path: tuple[str, ...]) -> float | None:
    value = nested_metric(mapping, path)
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def compare_ratio_gate(
    *,
    failures: list[str],
    metric_name: str,
    actual: float | None,
    baseline: float | None,
    max_ratio: float | None,
) -> None:
    if max_ratio is None:
        return
    if max_ratio <= 0.0:
        failures.append(f"invalid {metric_name} ratio gate {max_ratio:g}")
        return
    if actual is None:
        failures.append(f"missing current {metric_name} for baseline comparison")
        return
    if baseline is None:
        failures.append(f"missing baseline {metric_name}")
        return
    if baseline == 0.0:
        if actual > 0.0:
            failures.append(
                f"{metric_name} {actual:g} exceeds zero baseline")
        return
    ratio = actual / baseline
    if ratio > max_ratio:
        failures.append(
            f"{metric_name} ratio {ratio:.3g} exceeds limit {max_ratio:g} "
            f"(current {actual:g}, baseline {baseline:g})")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_case_spec(spec: str) -> tuple[str, str]:
    if ":" in spec:
        name, xml = spec.split(":", 1)
    else:
        name, xml = spec, "solver_perf_oop.xml"
    name = name.strip()
    xml = xml.strip()
    if not name or not xml:
        raise argparse.ArgumentTypeError(f"invalid case spec: {spec!r}")
    return name, xml


def parse_ranks(text: str) -> list[int]:
    ranks: list[int] = []
    for item in re.split(r"[,\s]+", text.strip()):
        if not item:
            continue
        try:
            rank = int(item)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"invalid rank: {item!r}") from exc
        if rank < 1:
            raise argparse.ArgumentTypeError("ranks must be positive")
        ranks.append(rank)
    if not ranks:
        raise argparse.ArgumentTypeError("at least one rank is required")
    return ranks


def timestamp() -> str:
    return _dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def resolve_binary(repo: Path, binary_arg: str | None) -> Path:
    if binary_arg:
        return Path(binary_arg).expanduser().resolve()
    env_binary = os.environ.get("SVMULTIPHYSICS_BINARY") or os.environ.get("BINARY")
    if env_binary:
        return Path(env_binary).expanduser().resolve()
    candidates = (
        repo / "build/svMultiPhysics-build/bin/svmultiphysics",
        repo / "build-unit/svMultiPhysics-build/bin/svmultiphysics",
        repo / "build-fe-check/svMultiPhysics-build/bin/svmultiphysics",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


def xml_text(root: ET.Element, tag: str) -> str | None:
    node = root.find(f".//{tag}")
    if node is None or node.text is None:
        return None
    return node.text.strip()


def inspect_case_xml(xml_path: Path) -> dict[str, object]:
    root = ET.parse(xml_path).getroot()
    steps_text = xml_text(root, "Number_of_time_steps")
    dt_text = xml_text(root, "Time_step_size")
    oop_text = xml_text(root, "Use_new_OOP_solver")
    dimensions_text = xml_text(root, "Number_of_spatial_dimensions")
    return {
        "expected_steps": int(steps_text) if steps_text else None,
        "time_step_size": float(dt_text) if dt_text else None,
        "uses_oop_solver": (oop_text or "").strip().lower() == "true",
        "spatial_dimensions": int(dimensions_text) if dimensions_text else None,
    }


def clean_generated_outputs(case_dir: Path) -> None:
    for child in case_dir.iterdir():
        if child.is_dir() and re.fullmatch(r"\d+-procs", child.name):
            shutil.rmtree(child, ignore_errors=True)
            continue
        if not child.is_file():
            continue
        if child.name == "STOP_SIM":
            child.unlink(missing_ok=True)
        elif child.name == "histor.dat":
            child.unlink(missing_ok=True)
        elif child.name.startswith("result_") and child.suffix in {".vtu", ".vtp"}:
            child.unlink(missing_ok=True)
        elif child.name.startswith("restart") or child.name.startswith("stFile"):
            child.unlink(missing_ok=True)
        elif child.name.startswith("geombc.dat."):
            child.unlink(missing_ok=True)


def prepare_work_dir(src_dir: Path, dst_dir: Path) -> None:
    if dst_dir.exists():
        shutil.rmtree(dst_dir)
    shutil.copytree(src_dir, dst_dir)
    clean_generated_outputs(dst_dir)


def parse_solver_logs(stdout_path: Path, stderr_path: Path) -> dict[str, object]:
    text_parts = []
    if stdout_path.exists():
        text_parts.append(stdout_path.read_text(errors="replace"))
    if stderr_path.exists():
        text_parts.append(stderr_path.read_text(errors="replace"))
    text = "\n".join(text_parts)
    time_stepping = TIME_STEPPING_RE.search(text)
    nonlinear_entries = []
    for match in NONLINEAR_DONE_RE.finditer(text):
        nonlinear_entries.append(
            {
                "step": int(match.group("step")),
                "time": float(match.group("time")),
                "converged": int(match.group("converged")),
                "newton_iters": int(match.group("iters")),
                "residual": float(match.group("residual")),
                "linear_converged": int(match.group("linear_converged")),
                "linear_iters": int(match.group("linear_iters")),
                "linear_rel": float(match.group("linear_rel")),
            }
        )
    accepted_entries = [
        {
            "step": int(match.group("step")),
            "time": float(match.group("time")),
            "dt": float(match.group("dt")),
        }
        for match in STEP_ACCEPTED_RE.finditer(text)
    ]
    newton_entries = [
        {
            "seconds": float(match.group("seconds")),
            "newton_iters": int(match.group("newton_iters")),
            "assemblies": int(match.group("assemblies")),
            "linear_iters": int(match.group("linear_iters")),
        }
        for match in NEWTON_TIMING_RE.finditer(text)
    ]
    total_loop = TOTAL_LOOP_RE.search(text)

    final_nonlinear = nonlinear_entries[-1] if nonlinear_entries else None
    return {
        "expected_steps_from_log": int(time_stepping.group("steps")) if time_stepping else None,
        "time_step_size_from_log": float(time_stepping.group("dt")) if time_stepping else None,
        "nonlinear_done_count": len(nonlinear_entries),
        "accepted_step_count": len(accepted_entries),
        "final_accepted_step": accepted_entries[-1]["step"] if accepted_entries else None,
        "final_accepted_time": accepted_entries[-1]["time"] if accepted_entries else None,
        "final_residual": final_nonlinear["residual"] if final_nonlinear else None,
        "max_residual": max((entry["residual"] for entry in nonlinear_entries), default=None),
        "max_newton_iters": max((entry["newton_iters"] for entry in nonlinear_entries), default=None),
        "max_linear_iters": max((entry["linear_iters"] for entry in nonlinear_entries), default=None),
        "all_nonlinear_converged": all(entry["converged"] == 1 for entry in nonlinear_entries),
        "all_linear_converged": all(entry["linear_converged"] == 1 for entry in nonlinear_entries),
        "newton_time_sum_s": sum(entry["seconds"] for entry in newton_entries),
        "newton_timing_count": len(newton_entries),
        "total_newton_iters": sum(entry["newton_iters"] for entry in newton_entries),
        "total_assemblies": sum(entry["assemblies"] for entry in newton_entries),
        "total_linear_iters": sum(entry["linear_iters"] for entry in newton_entries),
        "total_time_loop_s": float(total_loop.group("seconds")) if total_loop else None,
        "high_order_downgrade_count": text.count("high_order_downgrade=true"),
        "linearized_leaf_mentions": text.count("linearized"),
        "fallback_mentions": text.count("fallback"),
        "cut_context_rebuilds": parse_cut_context_rebuilds(text),
    }


def command_for(binary: Path, xml: str, ranks: int, mpi_launcher: str) -> list[str]:
    if ranks == 1:
        return [str(binary), xml]
    return [mpi_launcher, "-n", str(ranks), str(binary), xml]


def run_solver(
    *,
    cmd: list[str],
    cwd: Path,
    stdout_path: Path,
    stderr_path: Path,
    timeout_seconds: float,
    extra_env: dict[str, str],
) -> tuple[int, bool, float]:
    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("SVMP_FSILS_GMRES_REORTH", "off")
    env.update(extra_env)
    start = time.monotonic()
    timed_out = False
    with stdout_path.open("w") as stdout_file, stderr_path.open("w") as stderr_file:
        process = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            env=env,
            stdout=stdout_file,
            stderr=stderr_file,
            text=True,
        )
        try:
            returncode = process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            process.kill()
            returncode = process.wait()
            stderr_file.write(f"\n[run_oop_case_matrix] timeout after {timeout_seconds:.1f}s\n")
    wall_s = time.monotonic() - start
    return returncode, timed_out, wall_s


def evaluate_run(
    *,
    returncode: int | None,
    timed_out: bool,
    xml_info: dict[str, object],
    parsed: dict[str, object] | None,
    run_record_for_metrics: dict[str, object],
    baseline_run: dict[str, object] | None,
    baseline_required: bool,
    require_total_time: bool,
    fail_on_high_order_downgrade: bool,
    fail_on_cut_fallback: bool,
    max_basis_cache_entries: int | None,
    max_basis_cache_entry_growth: int | None,
    max_rss_kb: int | None,
    max_rss_growth_kb: int | None,
    max_wall_ratio: float | None,
    max_solver_loop_ratio: float | None,
    max_final_residual_ratio: float | None,
    max_total_newton_iters_ratio: float | None,
    max_total_linear_iters_ratio: float | None,
    max_cut_backend_seconds_ratio: float | None,
) -> tuple[str, list[str]]:
    failures: list[str] = []
    if timed_out:
        failures.append("timed out")
    if returncode not in (0, None):
        failures.append(f"return code {returncode}")
    if parsed is None:
        failures.append("no parsed solver log")
        return "fail", failures

    expected_steps = parsed.get("expected_steps_from_log") or xml_info.get("expected_steps")
    accepted_steps = parsed.get("accepted_step_count") or 0
    nonlinear_steps = parsed.get("nonlinear_done_count") or 0
    if nonlinear_steps == 0:
        failures.append("no nonlinear convergence markers")
    if not parsed.get("all_nonlinear_converged", False):
        failures.append("a nonlinear solve did not converge")
    if not parsed.get("all_linear_converged", False):
        failures.append("a linear solve did not converge")
    if expected_steps is not None and accepted_steps != expected_steps:
        failures.append(f"accepted {accepted_steps} steps, expected {expected_steps}")
    if require_total_time and parsed.get("total_time_loop_s") is None:
        failures.append("missing total time loop marker")
    if fail_on_high_order_downgrade and (parsed.get("high_order_downgrade_count") or 0) > 0:
        failures.append("high-order downgrade detected")
    cut_context = parsed.get("cut_context_rebuilds")
    if isinstance(cut_context, dict):
        max_fallback = cut_context.get("max_implicit_cut_fallback_cells") or 0
        max_corner_linearized = cut_context.get("max_corner_linearized_cells") or 0
        if fail_on_cut_fallback and (max_fallback > 0 or max_corner_linearized > 0):
            failures.append("cut-context fallback or corner-linearized cells detected")
        max_cache = cut_context.get("max_basis_cache_entries")
        if max_basis_cache_entries is not None and max_cache is not None:
            if max_cache > max_basis_cache_entries:
                failures.append(
                    f"basis cache entries {max_cache:g} exceed limit {max_basis_cache_entries}")
        cache_growth = cut_context.get("basis_cache_entry_growth")
        if max_basis_cache_entry_growth is not None and cache_growth is not None:
            if cache_growth > max_basis_cache_entry_growth:
                failures.append(
                    f"basis cache entry growth {cache_growth:g} exceeds limit "
                    f"{max_basis_cache_entry_growth}")
        max_rss = cut_context.get("max_process_rss_kb")
        if max_rss_kb is not None and max_rss is not None and max_rss > max_rss_kb:
            failures.append(f"RSS {max_rss:g} KB exceeds limit {max_rss_kb} KB")
        rss_growth = cut_context.get("process_rss_kb_growth")
        if max_rss_growth_kb is not None and rss_growth is not None:
            if rss_growth > max_rss_growth_kb:
                failures.append(
                    f"RSS growth {rss_growth:g} KB exceeds limit "
                    f"{max_rss_growth_kb} KB")
    if baseline_required and baseline_run is None:
        failures.append("missing matching baseline run")
    if baseline_run is not None:
        compare_ratio_gate(
            failures=failures,
            metric_name="wall time",
            actual=numeric_metric(run_record_for_metrics, ("wall_seconds",)),
            baseline=numeric_metric(baseline_run, ("wall_seconds",)),
            max_ratio=max_wall_ratio,
        )
        compare_ratio_gate(
            failures=failures,
            metric_name="solver-loop time",
            actual=numeric_metric(parsed, ("total_time_loop_s",)),
            baseline=numeric_metric(baseline_run, ("parsed", "total_time_loop_s")),
            max_ratio=max_solver_loop_ratio,
        )
        compare_ratio_gate(
            failures=failures,
            metric_name="final residual",
            actual=numeric_metric(parsed, ("final_residual",)),
            baseline=numeric_metric(baseline_run, ("parsed", "final_residual")),
            max_ratio=max_final_residual_ratio,
        )
        compare_ratio_gate(
            failures=failures,
            metric_name="total Newton iterations",
            actual=numeric_metric(parsed, ("total_newton_iters",)),
            baseline=numeric_metric(baseline_run, ("parsed", "total_newton_iters")),
            max_ratio=max_total_newton_iters_ratio,
        )
        compare_ratio_gate(
            failures=failures,
            metric_name="total linear iterations",
            actual=numeric_metric(parsed, ("total_linear_iters",)),
            baseline=numeric_metric(baseline_run, ("parsed", "total_linear_iters")),
            max_ratio=max_total_linear_iters_ratio,
        )
        compare_ratio_gate(
            failures=failures,
            metric_name="implicit cut backend time",
            actual=numeric_metric(
                parsed,
                ("cut_context_rebuilds", "backend_seconds_sum"),
            ),
            baseline=numeric_metric(
                baseline_run,
                ("parsed", "cut_context_rebuilds", "backend_seconds_sum"),
            ),
            max_ratio=max_cut_backend_seconds_ratio,
        )
    return ("pass" if not failures else "fail"), failures


def write_summary_json(path: Path, summary: dict[str, object]) -> None:
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


def fmt(value: object) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def write_summary_md(path: Path, summary: dict[str, object]) -> None:
    lines = [
        "# OOP Case Matrix Qualification Summary",
        "",
        f"- Date: {summary['metadata']['date']}",
        f"- Binary: `{summary['metadata']['binary']}`",
        f"- Cases root: `{summary['metadata']['cases_root']}`",
        f"- Ranks: `{', '.join(str(rank) for rank in summary['metadata']['ranks'])}`",
        "",
        "| Status | Case | Ranks | XML | Wall s | Solver loop s | Steps | Final residual | Newton iters | Linear iters |",
        "|---|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for run in summary["runs"]:
        parsed = run.get("parsed") or {}
        lines.append(
            "| {status} | `{case}` | {ranks} | `{xml}` | {wall} | {loop} | {steps} | {residual} | {newton} | {linear} |".format(
                status=run["status"],
                case=run["case"],
                ranks=run["ranks"],
                xml=run["xml"],
                wall=fmt(run.get("wall_seconds")),
                loop=fmt(parsed.get("total_time_loop_s")),
                steps=fmt(parsed.get("accepted_step_count")),
                residual=fmt(parsed.get("final_residual")),
                newton=fmt(parsed.get("total_newton_iters")),
                linear=fmt(parsed.get("total_linear_iters")),
            )
        )
    runs_with_cut_context = [
        run for run in summary["runs"]
        if ((run.get("parsed") or {}).get("cut_context_rebuilds") or {}).get("count", 0)
    ]
    if runs_with_cut_context:
        lines.extend([
            "",
            "## Cut Context",
            "",
            "| Case | Ranks | Rebuilds | Backend s | Max cut cells | Max qpts | Max RSS KB | Max cache | Max fallback | Achieved orders |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ])
        for run in runs_with_cut_context:
            parsed = run.get("parsed") or {}
            cut_context = parsed.get("cut_context_rebuilds") or {}
            achieved = "{}/{}".format(
                fmt(cut_context.get("min_achieved_interface_quadrature_order")),
                fmt(cut_context.get("min_achieved_volume_quadrature_order")),
            )
            lines.append(
                "| `{case}` | {ranks} | {count} | {backend_s} | {cut_cells} | {qpts} | {rss} | {cache} | {fallback} | {achieved} |".format(
                    case=run["case"],
                    ranks=run["ranks"],
                    count=fmt(cut_context.get("count")),
                    backend_s=fmt(cut_context.get("backend_seconds_sum")),
                    cut_cells=fmt(cut_context.get("max_active_cut_cells")),
                    qpts=fmt(cut_context.get("max_active_quadrature_points")),
                    rss=fmt(cut_context.get("max_process_rss_kb")),
                    cache=fmt(cut_context.get("max_basis_cache_entries")),
                    fallback=fmt(cut_context.get("max_implicit_cut_fallback_cells")),
                    achieved=achieved,
                )
            )
        lines.extend([
            "",
            "## Cut Context Growth",
            "",
            "| Case | Ranks | First RSS KB | Last RSS KB | Peak RSS Growth KB | First cache | Last cache | Peak cache growth |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for run in runs_with_cut_context:
            parsed = run.get("parsed") or {}
            cut_context = parsed.get("cut_context_rebuilds") or {}
            lines.append(
                "| `{case}` | {ranks} | {first_rss} | {last_rss} | {rss_growth} | {first_cache} | {last_cache} | {cache_growth} |".format(
                    case=run["case"],
                    ranks=run["ranks"],
                    first_rss=fmt(cut_context.get("first_process_rss_kb")),
                    last_rss=fmt(cut_context.get("last_process_rss_kb")),
                    rss_growth=fmt(cut_context.get("process_rss_kb_growth")),
                    first_cache=fmt(cut_context.get("first_basis_cache_entries")),
                    last_cache=fmt(cut_context.get("last_basis_cache_entries")),
                    cache_growth=fmt(cut_context.get("basis_cache_entry_growth")),
                )
            )
    failed = [run for run in summary["runs"] if run["status"] != "pass"]
    if failed:
        lines.extend(["", "## Failures", ""])
        for run in failed:
            failures = "; ".join(run.get("failures") or ["unknown failure"])
            lines.append(f"- `{run['case']}` ranks={run['ranks']}: {failures}")
    path.write_text("\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> int:
    repo = repo_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--binary",
        help="Path to svmultiphysics. Defaults to build-unit, then build.",
    )
    parser.add_argument(
        "--cases-root",
        type=Path,
        default=repo / "tests/cases/fluid",
        help="Root containing fluid case directories.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo / "Documentation/qualification_logs" / f"oop_case_matrix_{timestamp()}",
        help="Directory for copied cases, logs, and summaries.",
    )
    parser.add_argument(
        "--case",
        action="append",
        dest="cases",
        help="Case spec NAME[:XML]. May be repeated. Defaults to the OOP qualification matrix.",
    )
    parser.add_argument(
        "--ranks",
        type=parse_ranks,
        default=parse_ranks("1,2"),
        help="Comma or space separated rank counts. Default: 1,2.",
    )
    parser.add_argument("--mpi-launcher", default="mpiexec")
    parser.add_argument("--timeout-seconds", type=float, default=900.0)
    parser.add_argument("--max-runs", type=int, help="Run only the first N matrix entries.")
    parser.add_argument("--dry-run", action="store_true", help="Validate and print the matrix without executing.")
    parser.add_argument("--stop-on-failure", action="store_true")
    parser.add_argument(
        "--allow-non-oop",
        action="store_true",
        help="Do not fail case validation when Use_new_OOP_solver is not true.",
    )
    parser.add_argument(
        "--allow-missing-total-time",
        action="store_true",
        help="Do not require the completed Total time loop marker.",
    )
    parser.add_argument(
        "--fail-on-high-order-downgrade",
        action="store_true",
        help="Fail if solver logs contain high_order_downgrade=true.",
    )
    parser.add_argument(
        "--fail-on-cut-fallback",
        action="store_true",
        help="Fail if cut-context diagnostics report fallback or corner-linearized cells.",
    )
    parser.add_argument(
        "--max-basis-cache-entries",
        type=int,
        help="Fail if parsed cut-context diagnostics exceed this basis-cache entry count.",
    )
    parser.add_argument(
        "--max-basis-cache-entry-growth",
        type=int,
        help="Fail if parsed cut-context diagnostics show peak basis-cache growth above this count.",
    )
    parser.add_argument(
        "--max-rss-kb",
        type=int,
        help="Fail if parsed cut-context diagnostics exceed this process RSS in KB.",
    )
    parser.add_argument(
        "--max-rss-growth-kb",
        type=int,
        help="Fail if parsed cut-context diagnostics show peak RSS growth above this KB count.",
    )
    parser.add_argument(
        "--baseline-summary",
        type=Path,
        help="Previous run_oop_case_matrix.py summary.json used for regression gates.",
    )
    parser.add_argument(
        "--allow-missing-baseline",
        action="store_true",
        help="Do not fail when --baseline-summary lacks a matching case/XML/ranks entry.",
    )
    parser.add_argument(
        "--max-wall-ratio",
        type=float,
        help="Fail if wall time exceeds this ratio relative to --baseline-summary.",
    )
    parser.add_argument(
        "--max-solver-loop-ratio",
        type=float,
        help="Fail if Total time loop exceeds this ratio relative to --baseline-summary.",
    )
    parser.add_argument(
        "--max-final-residual-ratio",
        type=float,
        help="Fail if final nonlinear residual exceeds this ratio relative to --baseline-summary.",
    )
    parser.add_argument(
        "--max-total-newton-iters-ratio",
        type=float,
        help="Fail if total Newton iterations exceed this ratio relative to --baseline-summary.",
    )
    parser.add_argument(
        "--max-total-linear-iters-ratio",
        type=float,
        help="Fail if total linear iterations exceed this ratio relative to --baseline-summary.",
    )
    parser.add_argument(
        "--max-cut-backend-seconds-ratio",
        type=float,
        help="Fail if parsed implicit cut backend time exceeds this ratio relative to --baseline-summary.",
    )
    parser.add_argument(
        "--env",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Extra environment variable for solver runs. May be repeated.",
    )
    args = parser.parse_args(argv)

    binary = resolve_binary(repo, args.binary)
    cases_root = args.cases_root.resolve()
    output_dir = args.output_dir.resolve()
    logs_dir = output_dir / "logs"
    work_dir = output_dir / "work"
    summary_json = output_dir / "summary.json"
    summary_md = output_dir / "summary.md"
    case_specs = [parse_case_spec(spec) for spec in (args.cases or DEFAULT_CASES)]
    extra_env: dict[str, str] = {}
    for item in args.env:
        if "=" not in item:
            parser.error(f"--env must be KEY=VALUE, got {item!r}")
        key, value = item.split("=", 1)
        if not key:
            parser.error(f"--env has an empty key: {item!r}")
        extra_env[key] = value
    baseline_runs: dict[str, dict[str, object]] | None = None
    baseline_summary_path: Path | None = None
    if args.baseline_summary is not None:
        baseline_summary_path = args.baseline_summary.resolve()
        try:
            baseline_runs = load_baseline_runs(baseline_summary_path)
        except Exception as exc:  # noqa: BLE001 - report exact baseline context.
            parser.error(f"failed to load --baseline-summary {baseline_summary_path}: {exc}")
    baseline_ratio_args = (
        args.max_wall_ratio,
        args.max_solver_loop_ratio,
        args.max_final_residual_ratio,
        args.max_total_newton_iters_ratio,
        args.max_total_linear_iters_ratio,
        args.max_cut_backend_seconds_ratio,
    )
    if baseline_runs is None and any(value is not None for value in baseline_ratio_args):
        parser.error("baseline ratio gates require --baseline-summary")

    matrix: list[tuple[str, str, int, dict[str, object]]] = []
    validation_errors: list[str] = []
    for case_name, xml in case_specs:
        src_dir = cases_root / case_name
        xml_path = src_dir / xml
        if not src_dir.is_dir():
            validation_errors.append(f"missing case directory: {src_dir}")
            continue
        if not xml_path.is_file():
            validation_errors.append(f"missing case XML: {xml_path}")
            continue
        try:
            xml_info = inspect_case_xml(xml_path)
        except Exception as exc:  # noqa: BLE001 - report exact validation context.
            validation_errors.append(f"failed to inspect {xml_path}: {exc}")
            continue
        if not args.allow_non_oop and not xml_info["uses_oop_solver"]:
            validation_errors.append(f"{xml_path} does not set Use_new_OOP_solver=true")
            continue
        for ranks in args.ranks:
            matrix.append((case_name, xml, ranks, xml_info))

    if args.max_runs is not None:
        matrix = matrix[: args.max_runs]

    if args.dry_run:
        for case_name, xml, ranks, _xml_info in matrix:
            cmd = " ".join(command_for(binary, xml, ranks, args.mpi_launcher))
            print(f"{case_name} ranks={ranks} xml={xml}: {cmd}")
        if validation_errors:
            for error in validation_errors:
                print(f"error: {error}", file=sys.stderr)
            return 2
        return 0

    if validation_errors:
        for error in validation_errors:
            print(f"error: {error}", file=sys.stderr)
        return 2
    if not matrix:
        print("error: empty run matrix", file=sys.stderr)
        return 2
    if not binary.is_file():
        print(f"error: binary not found: {binary}", file=sys.stderr)
        return 2

    output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, object] = {
        "metadata": {
            "date": _dt.datetime.now().isoformat(timespec="seconds"),
            "repo": str(repo),
            "binary": str(binary),
            "cases_root": str(cases_root),
            "output_dir": str(output_dir),
            "ranks": args.ranks,
            "timeout_seconds": args.timeout_seconds,
            "extra_env": extra_env,
            "require_total_time": not args.allow_missing_total_time,
            "fail_on_high_order_downgrade": args.fail_on_high_order_downgrade,
            "fail_on_cut_fallback": args.fail_on_cut_fallback,
            "max_basis_cache_entries": args.max_basis_cache_entries,
            "max_basis_cache_entry_growth": args.max_basis_cache_entry_growth,
            "max_rss_kb": args.max_rss_kb,
            "max_rss_growth_kb": args.max_rss_growth_kb,
            "baseline_summary": str(baseline_summary_path) if baseline_summary_path else None,
            "allow_missing_baseline": args.allow_missing_baseline,
            "max_wall_ratio": args.max_wall_ratio,
            "max_solver_loop_ratio": args.max_solver_loop_ratio,
            "max_final_residual_ratio": args.max_final_residual_ratio,
            "max_total_newton_iters_ratio": args.max_total_newton_iters_ratio,
            "max_total_linear_iters_ratio": args.max_total_linear_iters_ratio,
            "max_cut_backend_seconds_ratio": args.max_cut_backend_seconds_ratio,
        },
        "runs": [],
    }

    exit_code = 0
    for case_name, xml, ranks, xml_info in matrix:
        run_name = f"{case_name}_np{ranks}"
        case_work_dir = work_dir / run_name
        run_stdout = logs_dir / f"{run_name}.stdout.log"
        run_stderr = logs_dir / f"{run_name}.stderr.log"
        cmd = command_for(binary, xml, ranks, args.mpi_launcher)
        print(f"run case={case_name} ranks={ranks} xml={xml}", flush=True)
        prepare_work_dir(cases_root / case_name, case_work_dir)
        returncode, timed_out, wall_s = run_solver(
            cmd=cmd,
            cwd=case_work_dir,
            stdout_path=run_stdout,
            stderr_path=run_stderr,
            timeout_seconds=args.timeout_seconds,
            extra_env=extra_env,
        )
        parsed = parse_solver_logs(run_stdout, run_stderr) if run_stdout.exists() else None
        baseline_run = None
        if baseline_runs is not None:
            baseline_run = baseline_runs.get(run_identity(case_name, xml, ranks))
        status, failures = evaluate_run(
            returncode=returncode,
            timed_out=timed_out,
            xml_info=xml_info,
            parsed=parsed,
            run_record_for_metrics={"wall_seconds": wall_s},
            baseline_run=baseline_run,
            baseline_required=baseline_runs is not None and not args.allow_missing_baseline,
            require_total_time=not args.allow_missing_total_time,
            fail_on_high_order_downgrade=args.fail_on_high_order_downgrade,
            fail_on_cut_fallback=args.fail_on_cut_fallback,
            max_basis_cache_entries=args.max_basis_cache_entries,
            max_basis_cache_entry_growth=args.max_basis_cache_entry_growth,
            max_rss_kb=args.max_rss_kb,
            max_rss_growth_kb=args.max_rss_growth_kb,
            max_wall_ratio=args.max_wall_ratio,
            max_solver_loop_ratio=args.max_solver_loop_ratio,
            max_final_residual_ratio=args.max_final_residual_ratio,
            max_total_newton_iters_ratio=args.max_total_newton_iters_ratio,
            max_total_linear_iters_ratio=args.max_total_linear_iters_ratio,
            max_cut_backend_seconds_ratio=args.max_cut_backend_seconds_ratio,
        )
        if status != "pass":
            exit_code = 1
        run_record = {
            "case": case_name,
            "xml": xml,
            "ranks": ranks,
            "status": status,
            "failures": failures,
            "command": cmd,
            "returncode": returncode,
            "timed_out": timed_out,
            "wall_seconds": wall_s,
            "work_dir": str(case_work_dir),
            "stdout_log": str(run_stdout),
            "stderr_log": str(run_stderr),
            "xml_info": xml_info,
            "baseline_key": run_identity(case_name, xml, ranks),
            "baseline_matched": baseline_run is not None,
            "parsed": parsed,
        }
        summary["runs"].append(run_record)
        write_summary_json(summary_json, summary)
        write_summary_md(summary_md, summary)
        print(
            f"  {status} wall_s={wall_s:.3f} failures={'; '.join(failures) if failures else '-'}",
            flush=True,
        )
        if status != "pass" and args.stop_on_failure:
            break

    write_summary_json(summary_json, summary)
    write_summary_md(summary_md, summary)
    print(f"summary_json={summary_json}")
    print(f"summary_md={summary_md}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

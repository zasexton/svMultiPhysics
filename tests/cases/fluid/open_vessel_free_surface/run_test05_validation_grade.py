#!/usr/bin/env python3
"""Run or validate SPHERIC Test05 OOP level-set qualification evidence.

This wrapper combines the existing OOP Test05 runner/profile comparison with a
full output-history active-region audit.  It is intended for D18/D38 evidence
where a final free-surface profile alone is not enough to prove that the
unfitted active fluid region remained consistent over time.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[3]
CASE_ROOT = SCRIPT_DIR / "unfitted_level_set"
RUNNER = SCRIPT_DIR / "run_test05_velocity_growth_smoke.py"
CASE_DIRS = {
    "d18": CASE_ROOT / "spheric_test05_wet_bed_d18",
    "d38": CASE_ROOT / "spheric_test05_wet_bed_d38",
}
DEFAULT_OUT_DIR = (
    ROOT
    / "Documentation"
    / "qualification_logs"
    / "open_vessel_free_surface_remaining_20260526"
    / "validation_grade_test05_oop"
)


def parse_case_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            "expected CASE=PATH, for example d18=Documentation/.../metrics.json"
        )
    case, raw_path = value.split("=", 1)
    case = case.strip().lower()
    if case not in CASE_DIRS:
        raise argparse.ArgumentTypeError(
            f"unknown case {case!r}; expected one of {sorted(CASE_DIRS)}"
        )
    return case, Path(raw_path)


def load_probe(metrics_path: Path, case: str) -> dict[str, Any]:
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    candidates: list[dict[str, Any]]
    if isinstance(payload, dict) and isinstance(payload.get("probes"), list):
        candidates = [
            item for item in payload["probes"]
            if isinstance(item, dict) and item.get("case") == case
        ]
    elif isinstance(payload, list):
        candidates = [
            item for item in payload
            if isinstance(item, dict) and item.get("case") == case
        ]
    elif isinstance(payload, dict):
        candidates = [payload] if payload.get("case") in (case, None) else []
    else:
        candidates = []

    if not candidates:
        raise ValueError(f"{metrics_path} does not contain a metrics record for {case}")
    if len(candidates) > 1:
        raise ValueError(f"{metrics_path} contains multiple records for {case}")
    return candidates[0]


def run_solver_case(args: argparse.Namespace, case: str) -> Path:
    metrics_path = args.out_dir / f"{case}_oop_profile_metrics.json"
    command = [
        sys.executable,
        str(RUNNER),
        "--solver",
        str(args.solver),
        "--case",
        case,
        "--steps",
        str(args.steps),
        "--preserve-run-dir",
        "--qualification-log",
        str(metrics_path),
        "--linear-solver-type",
        args.linear_solver_type,
        "--linear-algebra-backend",
        args.linear_algebra_backend,
        "--linear-preconditioner",
        args.linear_preconditioner,
        "--min-max-speed",
        "0.0",
        "--min-wet-mean-speed",
        "0.0",
        "--min-gate-mean-ux",
        "-1.0",
        "--min-front-mean-ux",
        "-1.0",
        "--max-wet-fraction-volume-error",
        f"{args.max_wet_fraction_volume_error:.17g}",
        "--require-time-loop-convergence",
        "--require-reference-profile-comparison",
        "--min-reference-profile-coverage",
        f"{args.min_reference_profile_coverage:.17g}",
        "--min-reference-profile-direct-coverage",
        f"{args.min_reference_profile_direct_coverage:.17g}",
        "--max-reference-profile-rmse",
        f"{args.max_reference_profile_rmse:.17g}",
        "--max-reference-profile-mae",
        f"{args.max_reference_profile_mae:.17g}",
        "--max-reference-profile-max-abs-error",
        f"{args.max_reference_profile_max_abs_error:.17g}",
        "--reference-profile-elevated-front-clearance",
        f"{args.reference_profile_elevated_front_clearance:.17g}",
        "--max-reference-profile-elevated-front-lag",
        f"{args.max_reference_profile_elevated_front_lag:.17g}",
        "--enable-adaptive-time-loop",
        "--adaptive-time-loop-min-dt",
        f"{args.adaptive_time_loop_min_dt:.17g}",
        "--adaptive-time-loop-max-dt",
        f"{args.adaptive_time_loop_max_dt:.17g}",
        "--adaptive-time-loop-max-retries",
        str(args.adaptive_time_loop_max_retries),
        "--adaptive-time-loop-decrease-factor",
        f"{args.adaptive_time_loop_decrease_factor:.17g}",
        "--adaptive-time-loop-increase-factor",
        f"{args.adaptive_time_loop_increase_factor:.17g}",
        "--adaptive-time-loop-target-newton-iterations",
        str(args.adaptive_time_loop_target_newton_iterations),
        "--adaptive-time-loop-max-steps-multiplier",
        str(args.adaptive_time_loop_max_steps_multiplier),
        "--disable-cut-metadata-scale",
        "--trace-level-set-advection-velocity",
        "--enable-physical-history-instrumentation",
        "--timeout-seconds",
        f"{args.timeout_seconds:.17g}",
    ]
    if args.wet_extension_advection_velocity_method:
        command.extend([
            "--wet-extension-advection-velocity-method",
            args.wet_extension_advection_velocity_method,
        ])
    if args.allow_experimental_profile_linear_solver:
        command.append("--allow-experimental-profile-linear-solver")
    for extra in args.extra_runner_arg:
        command.append(extra)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print("Running:", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)
    return metrics_path


def numeric(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def gate_min(
    failures: list[str],
    probe: dict[str, Any],
    key: str,
    minimum: float,
    label: str,
) -> None:
    value = numeric(probe.get(key))
    if value is None:
        failures.append(f"{label} is unavailable")
    elif value < minimum:
        failures.append(f"{label} {value:.6g} is below {minimum:.6g}")


def gate_max(
    failures: list[str],
    probe: dict[str, Any],
    key: str,
    maximum: float,
    label: str,
) -> None:
    value = numeric(probe.get(key))
    if value is None:
        failures.append(f"{label} is unavailable")
    elif value > maximum:
        failures.append(f"{label} {value:.6g} exceeds {maximum:.6g}")


def false_wall_wet_failures(probe: dict[str, Any]) -> list[str]:
    """Require the instrumented wall history and reject its first event."""
    if (probe.get("wall_only_false_wet_applicability") ==
            "not_applicable_closed_interface"):
        if probe.get(
                "wall_only_false_wet_closed_interface_certified") is True:
            return []
        return [
            "wall-wetting instrumentation claimed a closed-interface "
            "exemption without a valid initial P1 boundary-sign certificate"
        ]
    if "wall_only_false_wet_history" not in probe or \
            "first_wall_only_false_wet" not in probe:
        return [
            "physical wall-wetting instrumentation is unavailable; "
            "the validation-grade run cannot certify the historical symptom"
        ]
    history = probe.get("wall_only_false_wet_history")
    if not isinstance(history, list) or not history:
        return ["physical wall-wetting history is empty"]
    first_event = probe.get("first_wall_only_false_wet")
    if first_event is None:
        return []
    if isinstance(first_event, dict):
        return [
            "false wall wetting detected at step "
            f"{first_event.get('step')!r}, time {first_event.get('time')!r}, "
            f"vertex {first_event.get('global_node_id')!r}"
        ]
    return [f"false wall wetting detected: {first_event!r}"]


def result_dir_and_prefix(probe: dict[str, Any]) -> tuple[Path, str]:
    result_path = probe.get("result_path")
    if isinstance(result_path, str):
        path = Path(result_path)
        prefix = path.stem.rsplit("_", 1)[0]
        return path.parent, prefix

    run_dir = probe.get("run_dir")
    if isinstance(run_dir, str):
        return Path(run_dir), "result"

    raise ValueError("metrics record has neither result_path nor run_dir")


def active_history_audit(
    args: argparse.Namespace,
    case: str,
    probe: dict[str, Any],
) -> tuple[dict[str, Any] | None, Path | None, list[str]]:
    if args.skip_active_history:
        return None, None, []

    sys.path.insert(0, str(SCRIPT_DIR))
    import audit_test05_active_region_history as active_audit

    result_dir, prefix = result_dir_and_prefix(probe)
    if not result_dir.exists():
        return None, None, [f"active-history result directory is missing: {result_dir}"]

    report = active_audit.audit_history(
        result_dir,
        case_dir=CASE_DIRS[case],
        prefix=prefix,
        max_volume_rel_drift=args.max_volume_rel_drift,
        max_cell_clip_rel_error=args.max_cell_clip_rel_error,
        max_step_rel_jump=args.max_step_rel_jump,
        fraction_tolerance=args.fraction_tolerance,
    )
    output_path = args.out_dir / f"{case}_active_region_history_audit.json"
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report, output_path, list(report.get("failures", []))


def validate_probe(
    args: argparse.Namespace,
    case: str,
    metrics_path: Path,
) -> dict[str, Any]:
    probe = load_probe(metrics_path, case)
    failures: list[str] = []

    if probe.get("passed") is False:
        failures.extend(str(item) for item in probe.get("errors", []))

    if probe.get("reference_profile_validation_passed") is not True:
        if probe.get("reference_profile_error"):
            failures.append(
                "reference profile comparison failed: "
                f"{probe['reference_profile_error']}"
            )
        else:
            failures.append("reference profile comparison did not pass")
        failures.extend(
            f"reference profile: {item}"
            for item in probe.get("reference_profile_validation_failures", [])
        )

    gate_min(
        failures,
        probe,
        "reference_profile_coverage_fraction",
        args.min_reference_profile_coverage,
        "reference profile coverage",
    )
    gate_min(
        failures,
        probe,
        "reference_profile_direct_coverage_fraction",
        args.min_reference_profile_direct_coverage,
        "reference profile direct coverage",
    )
    gate_max(
        failures,
        probe,
        "reference_profile_rmse_m",
        args.max_reference_profile_rmse,
        "reference profile RMSE",
    )
    gate_max(
        failures,
        probe,
        "reference_profile_mae_m",
        args.max_reference_profile_mae,
        "reference profile MAE",
    )
    gate_max(
        failures,
        probe,
        "reference_profile_max_abs_error_m",
        args.max_reference_profile_max_abs_error,
        "reference profile max absolute error",
    )

    elevated_front_error = numeric(probe.get("reference_profile_elevated_front_error_m"))
    if elevated_front_error is None:
        failures.append("reference profile elevated-front error is unavailable")
    elif elevated_front_error < -args.max_reference_profile_elevated_front_lag:
        failures.append(
            "reference elevated front lag "
            f"{abs(elevated_front_error):.6g} m exceeds "
            f"{args.max_reference_profile_elevated_front_lag:.6g} m"
        )

    if "wet_fraction_volume_error_vs_last_cut_context" in probe:
        gate_max(
            failures,
            probe,
            "wet_fraction_volume_error_vs_last_cut_context",
            args.max_wet_fraction_volume_error,
            "WetVolumeFraction volume error",
        )

    failures.extend(false_wall_wet_failures(probe))

    time_loop = probe.get("time_loop", {})
    summary = time_loop.get("summary") if isinstance(time_loop, dict) else None
    if not isinstance(summary, dict):
        failures.append("time-loop summary is unavailable")
    else:
        if summary.get("all_linear_converged") is not True:
            failures.append("not all linear solves converged")
        accepted = summary.get("accepted_steps")
        final_time = numeric(summary.get("final_accepted_time"))
        expected_time = args.steps * args.time_step_size
        if not isinstance(accepted, int) or accepted < args.steps:
            failures.append(
                f"accepted steps {accepted!r} below requested steps {args.steps}"
            )
        if final_time is None or final_time + 1.0e-12 < expected_time:
            failures.append(
                f"final accepted time {final_time!r} below requested "
                f"time {expected_time:.6g}"
            )

    active_report, active_path, active_failures = active_history_audit(args, case, probe)
    failures.extend(f"active history: {item}" for item in active_failures)

    profile = {
        key: probe.get(key)
        for key in (
            "reference_profile_target_time_s",
            "reference_profile_coverage_fraction",
            "reference_profile_direct_coverage_fraction",
            "reference_profile_rmse_m",
            "reference_profile_mae_m",
            "reference_profile_max_abs_error_m",
            "reference_profile_elevated_front_error_m",
        )
    }
    active_summary: dict[str, Any] | None = None
    if active_report is not None:
        active_summary = {
            "path": str(active_path),
            "passed": active_report.get("passed"),
            "result_count": active_report.get("result_count"),
            "max_abs_relative_volume_drift": active_report.get(
                "max_abs_relative_volume_drift"
            ),
            "max_abs_cell_clip_relative_error": active_report.get(
                "max_abs_cell_clip_relative_error"
            ),
            "max_abs_step_relative_volume_jump": active_report.get(
                "max_abs_step_relative_volume_jump"
            ),
            "max_active_fluid_mask_mismatch_count": active_report.get(
                "max_active_fluid_mask_mismatch_count"
            ),
            "max_wet_volume_fraction_outside_count": active_report.get(
                "max_wet_volume_fraction_outside_count"
            ),
        }

    return {
        "case": case,
        "passed": not failures,
        "failures": failures,
        "metrics_path": str(metrics_path),
        "run_dir": probe.get("run_dir"),
        "result_path": probe.get("result_path"),
        "profile": profile,
        "time_loop_summary": summary,
        "wet_fraction_volume_error_vs_last_cut_context": probe.get(
            "wet_fraction_volume_error_vs_last_cut_context"
        ),
        "first_wall_only_false_wet": probe.get("first_wall_only_false_wet"),
        "active_history": active_summary,
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# SPHERIC Test05 OOP validation-grade report",
        "",
        f"Generated UTC: {report['generated_at_utc']}",
        f"Overall passed: {report['overall_passed']}",
        "",
        "## Gates",
        "",
        f"- Wet-volume relative drift: <= {report['gates']['max_volume_rel_drift']}",
        f"- WetVolumeMeasure vs phi clipping: <= {report['gates']['max_cell_clip_rel_error']}",
        f"- Step wet-volume relative jump: <= {report['gates']['max_step_rel_jump']}",
        f"- Reference profile coverage: >= {report['gates']['min_reference_profile_coverage']}",
        f"- Reference profile direct coverage: >= {report['gates']['min_reference_profile_direct_coverage']}",
        f"- Reference profile RMSE: <= {report['gates']['max_reference_profile_rmse']} m",
        f"- Reference profile MAE: <= {report['gates']['max_reference_profile_mae']} m",
        f"- Reference profile max abs error: <= {report['gates']['max_reference_profile_max_abs_error']} m",
        f"- Reference elevated-front lag: <= {report['gates']['max_reference_profile_elevated_front_lag']} m",
        "",
        "## Cases",
        "",
        "| Case | Pass | Profile RMSE (m) | Profile max abs (m) | Active drift | Active mismatches | Failures |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for case in report["cases"]:
        profile = case.get("profile") or {}
        active = case.get("active_history") or {}
        drift = active.get("max_abs_relative_volume_drift") or {}
        failures = "; ".join(case.get("failures") or [])
        failures = failures.replace("\n", " ")
        lines.append(
            "| {case} | {passed} | {rmse} | {max_abs} | {drift} | {mismatch} | {failures} |".format(
                case=case["case"],
                passed=case["passed"],
                rmse=profile.get("reference_profile_rmse_m"),
                max_abs=profile.get("reference_profile_max_abs_error_m"),
                drift=drift.get("value"),
                mismatch=active.get("max_active_fluid_mask_mismatch_count"),
                failures=failures or "-",
            )
        )
    lines.append("")
    if report.get("dualsphysics_root"):
        lines.extend([
            "## DualSPHysics",
            "",
            f"DualSPHysics root recorded for cross-code smoke checks: `{report['dualsphysics_root']}`.",
            "This report treats the digitized SPHERIC profiles and active-region history as the validation gates.",
            "",
        ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_report(args: argparse.Namespace, case_reports: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "generated_at_utc": _dt.datetime.now(_dt.UTC).isoformat(),
        "overall_passed": all(item["passed"] for item in case_reports),
        "runner": str(RUNNER),
        "dualsphysics_root": (
            str(args.dualsphysics_root)
            if args.dualsphysics_root and args.dualsphysics_root.exists()
            else None
        ),
        "gates": {
            "steps": args.steps,
            "time_step_size": args.time_step_size,
            "max_volume_rel_drift": args.max_volume_rel_drift,
            "max_cell_clip_rel_error": args.max_cell_clip_rel_error,
            "max_step_rel_jump": args.max_step_rel_jump,
            "fraction_tolerance": args.fraction_tolerance,
            "max_wet_fraction_volume_error": args.max_wet_fraction_volume_error,
            "min_reference_profile_coverage": args.min_reference_profile_coverage,
            "min_reference_profile_direct_coverage": args.min_reference_profile_direct_coverage,
            "max_reference_profile_rmse": args.max_reference_profile_rmse,
            "max_reference_profile_mae": args.max_reference_profile_mae,
            "max_reference_profile_max_abs_error": args.max_reference_profile_max_abs_error,
            "max_reference_profile_elevated_front_lag": args.max_reference_profile_elevated_front_lag,
        },
        "cases": case_reports,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solver", type=Path)
    parser.add_argument("--run-solver", action="store_true")
    parser.add_argument("--cases", choices=sorted(CASE_DIRS), nargs="+", default=["d18", "d38"])
    parser.add_argument("--metrics", action="append", type=parse_case_path, default=[])
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--steps",
        type=int,
        default=562,
        help=(
            "accepted steps to audit; the default reaches t=0.281 s at the "
            "benchmark dt, spanning the historical D18 false-wall-wet onset "
            "near t=0.235 s"
        ),
    )
    parser.add_argument("--time-step-size", type=float, default=5.0e-4)
    parser.add_argument("--timeout-seconds", type=float, default=86400.0)
    parser.add_argument("--linear-solver-type", default="direct")
    parser.add_argument("--linear-algebra-backend", default="eigen")
    parser.add_argument("--linear-preconditioner", default="none")
    parser.add_argument("--allow-experimental-profile-linear-solver", action="store_true", default=True)
    parser.add_argument("--wet-extension-advection-velocity-method", default=None)
    parser.add_argument("--extra-runner-arg", action="append", default=[])
    parser.add_argument("--adaptive-time-loop-min-dt", type=float, default=1.5625e-5)
    parser.add_argument("--adaptive-time-loop-max-dt", type=float, default=5.0e-4)
    parser.add_argument("--adaptive-time-loop-max-retries", type=int, default=8)
    parser.add_argument("--adaptive-time-loop-decrease-factor", type=float, default=0.5)
    parser.add_argument("--adaptive-time-loop-increase-factor", type=float, default=1.0)
    parser.add_argument("--adaptive-time-loop-target-newton-iterations", type=int, default=2)
    parser.add_argument("--adaptive-time-loop-max-steps-multiplier", type=int, default=16)
    parser.add_argument("--max-volume-rel-drift", type=float, default=5.0e-4)
    parser.add_argument("--max-cell-clip-rel-error", type=float, default=1.0e-8)
    parser.add_argument("--max-step-rel-jump", type=float, default=5.0e-4)
    parser.add_argument("--fraction-tolerance", type=float, default=1.0e-12)
    parser.add_argument("--max-wet-fraction-volume-error", type=float, default=1.0e-8)
    parser.add_argument("--min-reference-profile-coverage", type=float, default=0.95)
    parser.add_argument("--min-reference-profile-direct-coverage", type=float, default=0.25)
    parser.add_argument("--max-reference-profile-rmse", type=float, default=0.12)
    parser.add_argument("--max-reference-profile-mae", type=float, default=0.10)
    parser.add_argument("--max-reference-profile-max-abs-error", type=float, default=0.18)
    parser.add_argument("--reference-profile-elevated-front-clearance", type=float, default=0.010)
    parser.add_argument("--max-reference-profile-elevated-front-lag", type=float, default=0.30)
    parser.add_argument("--skip-active-history", action="store_true")
    parser.add_argument("--dualsphysics-root", type=Path, default=Path("/tmp/svmp_sph_baseline/DualSPHysics"))
    args = parser.parse_args()

    if args.run_solver and args.solver is None:
        parser.error("--run-solver requires --solver")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    metrics_by_case = dict(args.metrics)
    case_reports = []
    for case in args.cases:
        metrics_path = metrics_by_case.get(case)
        if args.run_solver:
            metrics_path = run_solver_case(args, case)
        if metrics_path is None:
            parser.error(f"no metrics path supplied for {case}; use --metrics {case}=PATH or --run-solver")
        case_reports.append(validate_probe(args, case, metrics_path))

    report = build_report(args, case_reports)
    json_path = args.out_dir / "test05_validation_grade_report.json"
    md_path = args.out_dir / "test05_validation_grade_report.md"
    json_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_markdown(report, md_path)
    print(json.dumps({"report": str(json_path), "passed": report["overall_passed"]}, indent=2))
    return 0 if report["overall_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

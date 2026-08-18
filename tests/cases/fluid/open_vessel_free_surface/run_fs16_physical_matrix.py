#!/usr/bin/env python3
"""Run the reproducible FS-16 free-surface physical-verification matrix.

The matrix deliberately keeps failed/nonconverged probes in the report.  A
geometry/postprocessing check is never promoted to production-solve evidence.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
RUNNER = HERE / "run_test05_velocity_growth_smoke.py"
MIN_ACCEPTED_REFINEMENT_RATE = 0.25
REFINEMENT_EXACTNESS_FLOOR = 1.0e-12
DYNAMIC_CONTACT_TIMEOUT_SECONDS_AT_N16 = 1200


def dynamic_contact_timeout_seconds(resolution: int) -> int:
    """Return a mesh-scaled liveness guard, not a physical acceptance gate."""
    if resolution <= 0:
        raise ValueError("dynamic-contact resolution must be positive")
    # The dominant production tangent assembly scales with the number of Q1
    # cells.  Preserve 300 s for n=8, provide 1200 s for n=16, and scale by
    # cell count thereafter (4800 s at n=32).
    return max(
        300,
        math.ceil(
            DYNAMIC_CONTACT_TIMEOUT_SECONDS_AT_N16 *
            (float(resolution) / 16.0) ** 2),
    )


def metric(probe: dict[str, Any], name: str) -> str:
    value = probe.get(name)
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def run_one(spec: dict[str, Any], solver: Path, output_dir: Path) -> dict[str, Any]:
    name = str(spec["name"])
    qualification_log = output_dir / f"{name}.json"
    stdout_log = output_dir / f"{name}.stdout.log"
    command = [
        sys.executable,
        str(RUNNER),
        "--solver",
        str(solver),
        "--qualification-log",
        str(qualification_log),
        "--preserve-run-dir",
        *[str(value) for value in spec["args"]],
    ]
    completed = subprocess.run(
        command,
        cwd=HERE.parents[3],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    stdout_log.write_text(completed.stdout, encoding="utf-8")
    payload: dict[str, Any] = {}
    if qualification_log.exists():
        try:
            payload = json.loads(qualification_log.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            payload = {"parse_error": str(exc)}
    probes = payload.get("probes", []) if isinstance(payload, dict) else []
    probe = probes[-1] if probes and isinstance(probes[-1], dict) else {}
    return {
        "name": name,
        "kind": spec["kind"],
        "command": command,
        "returncode": completed.returncode,
        "qualification_log": str(qualification_log),
        "stdout_log": str(stdout_log),
        "production_solve_converged": bool(
            completed.returncode == 0 and probe.get("passed") is True),
        "probe": probe,
    }


def matrix_specs(meshes: list[int], quick: bool) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    static_steps = 1 if quick else 3
    dynamic_steps = 3 if quick else 20
    for resolution in meshes:
        for angle in (60, 90, 120):
            specs.append({
                "name": f"sessile_theta{angle}_n{resolution}",
                "kind": "static_sessile",
                "args": [
                    "--case", "sessile2d",
                    "--steps", static_steps,
                    "--time-step-size", 1.0e-3,
                    "--synthetic-nx", resolution,
                    "--synthetic-ny", resolution,
                    "--contact-angle-degrees", angle,
                    "--timeout-seconds", 300,
                ],
            })
        # Equal-area +/-5 degree perturbations isolate the line response
        # without changing liquid mass or introducing the large startup shape
        # imbalance of the former fixed-radius +/-20 degree pair.
        for label, initial_angle in (("advancing", 95), ("receding", 85)):
            specs.append({
                "name": f"dynamic_{label}_n{resolution}",
                "kind": "dynamic_contact",
                "args": [
                    "--case", "dynamiccontact2d",
                    "--steps", dynamic_steps,
                    "--time-step-size", 1.0e-3,
                    "--synthetic-nx", resolution,
                    "--synthetic-ny", resolution,
                    "--contact-angle-degrees", 90,
                    "--dynamic-initial-contact-angle-degrees", initial_angle,
                    "--contact-line-mobility", 1.0,
                    "--wall-slip-length", 0.1,
                    "--timeout-seconds",
                    dynamic_contact_timeout_seconds(resolution),
                ],
            })
            # Rotate the same equal-area Ren--E problem onto the true vertical
            # x=0 tank side.  This is a separate production qualification, not
            # an alias of the bottom-wall result: wall-normal impermeability is
            # x-directed, tangential wetting motion is y-directed, and the raw
            # constitutive/geometric gates remain enabled by the common runner.
            specs.append({
                "name": f"dynamic_left_wall_{label}_n{resolution}",
                "kind": "dynamic_contact_sidewall",
                "args": [
                    "--case", "dynamiccontact2d",
                    "--steps", dynamic_steps,
                    "--time-step-size", 1.0e-3,
                    "--synthetic-nx", resolution,
                    "--synthetic-ny", resolution,
                    "--contact-angle-degrees", 90,
                    "--dynamic-initial-contact-angle-degrees", initial_angle,
                    "--dynamic-contact-wall", "wall_left",
                    "--contact-line-mobility", 1.0,
                    "--wall-slip-length", 0.1,
                    "--timeout-seconds",
                    dynamic_contact_timeout_seconds(resolution),
                ],
            })
    for resolution in (8, 16, 32):
        specs.extend([
            {
                "name": f"capillary_droplet_equilibrium_n{resolution}",
                "kind": "capillary_equilibrium",
                "args": [
                    "--high-order-capillary-droplet-equilibrium-smoke",
                    "--synthetic-nx", resolution,
                    "--synthetic-ny", resolution,
                ],
            },
            {
                "name": f"capillary_wave_n{resolution}",
                "kind": "capillary_wave",
                "args": [
                    "--high-order-capillary-wave-smoke",
                    "--steps", 100,
                    "--time-step-size", 1.0e-3,
                    "--surface-tension", 500.0,
                    "--synthetic-nx", resolution,
                    "--synthetic-ny", resolution,
                ],
            },
        ])
    return specs


def _sessile_contact(probe: dict[str, Any]) -> dict[str, Any]:
    contact = probe.get("benchmark", {}).get("sessile_contact", {})
    return contact if isinstance(contact, dict) else {}


def _final_contact_fluid_speed(probe: dict[str, Any]) -> float | None:
    value = probe.get("sessile_final_contact_fluid_outward_speed")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    state = probe.get("final_sessile_state")
    if isinstance(state, dict):
        value = state.get("contact_fluid_outward_speed")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    return None


def symmetric_dynamic_response_records(
        results: list[dict[str, Any]]) -> list[dict[str, float | int]]:
    """Center +/- angle responses on the same-area equilibrium control.

    The raw dynamic speed remains the pass/fail physical observable.  This
    centered triad is diagnostic: it separates an even parasitic equilibrium
    drift from the odd Ren--E response without subtracting that drift to make
    an individual production solve pass.
    """
    by_resolution: dict[int, dict[str, dict[str, Any]]] = {}
    for result in results:
        probe = result.get("probe", {})
        if not isinstance(probe, dict):
            continue
        resolution = probe.get("benchmark", {}).get(
            "mesh_resolution", {}).get("nx")
        if not isinstance(resolution, int) or isinstance(resolution, bool):
            continue
        contact = _sessile_contact(probe)
        initial = contact.get("initial_contact_angle_degrees")
        equilibrium = contact.get("equilibrium_contact_angle_degrees")
        if not all(isinstance(value, (int, float)) and not isinstance(value, bool)
                   for value in (initial, equilibrium)):
            continue
        role: str | None = None
        if result.get("kind") == "static_sessile" and math.isclose(
                float(initial), float(equilibrium), abs_tol=1.0e-12):
            if math.isclose(float(equilibrium), 90.0, abs_tol=1.0e-12):
                role = "equilibrium"
        elif result.get("kind") == "dynamic_contact":
            role = "advancing" if float(initial) > float(equilibrium) else "receding"
        if role is not None:
            by_resolution.setdefault(resolution, {})[role] = probe

    records: list[dict[str, float | int]] = []
    for resolution, probes in sorted(by_resolution.items()):
        if set(probes) != {"equilibrium", "advancing", "receding"}:
            continue
        equilibrium = probes["equilibrium"]
        advancing = probes["advancing"]
        receding = probes["receding"]
        u_eq = _final_contact_fluid_speed(equilibrium)
        u_adv = _final_contact_fluid_speed(advancing)
        u_rec = _final_contact_fluid_speed(receding)
        v_adv = advancing.get("ren_e_predicted_final_contact_line_speed")
        v_rec = receding.get("ren_e_predicted_final_contact_line_speed")
        if (u_eq is None or u_adv is None or u_rec is None or
                not isinstance(v_adv, (int, float)) or isinstance(v_adv, bool) or
                not isinstance(v_rec, (int, float)) or isinstance(v_rec, bool)):
            continue
        predicted_odd = 0.5 * (float(v_adv) - float(v_rec))
        if abs(predicted_odd) <= 1.0e-14:
            continue
        observed_odd = 0.5 * (u_adv - u_rec)
        pair_even = 0.5 * (u_adv + u_rec)
        centered_advancing = u_adv - u_eq
        centered_receding = u_rec - u_eq
        centered_scale = max(
            abs(centered_advancing) + abs(centered_receding), 1.0e-300)
        record: dict[str, float | int] = {
            "resolution": resolution,
            "h": 1.0 / float(resolution),
            "equilibrium_bias_speed": u_eq,
            "advancing_raw_speed": u_adv,
            "receding_raw_speed": u_rec,
            "pair_even_speed": pair_even,
            "pair_even_vs_equilibrium_absolute_defect": abs(pair_even - u_eq),
            "centered_advancing_response": centered_advancing,
            "centered_receding_response": centered_receding,
            "centered_odd_response": observed_odd,
            "predicted_odd_response": predicted_odd,
            "centered_odd_response_relative_error": (
                abs(observed_odd - predicted_odd) / abs(predicted_odd)),
            "centered_antisymmetry_relative_defect": (
                abs(centered_advancing + centered_receding) / centered_scale),
        }
        adv_area = _sessile_contact(advancing).get("expected_initial_liquid_area")
        rec_area = _sessile_contact(receding).get("expected_initial_liquid_area")
        eq_area = _sessile_contact(equilibrium).get("expected_initial_liquid_area")
        if all(isinstance(value, (int, float)) and not isinstance(value, bool)
               for value in (adv_area, rec_area, eq_area)):
            area_scale = max(abs(float(eq_area)), 1.0e-300)
            record["maximum_reference_area_relative_mismatch"] = max(
                abs(float(adv_area) - float(eq_area)),
                abs(float(rec_area) - float(eq_area)),
            ) / area_scale
        records.append(record)
    return records


def refinement_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    requested: dict[str, tuple[str, tuple[str, ...]]] = {
        "capillary_equilibrium": ("capillary_equilibrium", (
            "capillary_pressure_jump_relative_error",
            "max_speed",
        )),
        "capillary_wave": ("capillary_wave", (
            "capillary_wave_omega_relative_error",
            "capillary_wave_profile_relative_error",
            "capillary_wave_apparent_damping_vs_inviscid",
            "free_surface_energy_max_energy_above_initial_relative_proxy",
            "free_surface_energy_relative_total_energy_change_proxy",
        )),
    }
    for angle in (60, 90, 120):
        requested[f"static_sessile_theta{angle}"] = ("static_sessile", (
            "sessile_final_contact_angle_absolute_error_degrees",
            "sessile_final_pressure_jump_relative_error",
            "sessile_final_liquid_area_relative_error",
            "sessile_final_parasitic_capillary_number",
            "free_surface_energy_max_energy_above_initial_relative_proxy",
            "free_surface_energy_relative_total_energy_change_proxy",
        ))
    for motion in ("advancing", "receding"):
        requested[f"dynamic_contact_{motion}"] = ("dynamic_contact", (
            "ren_e_contact_fluid_speed_relative_error",
            "ren_e_geometric_speed_relative_error",
            "sessile_final_contact_angle_absolute_error_degrees",
            "free_surface_energy_max_energy_above_initial_relative_proxy",
            "free_surface_energy_relative_total_energy_change_proxy",
        ))
        requested[f"dynamic_contact_sidewall_{motion}"] = (
            "dynamic_contact_sidewall", (
                "ren_e_contact_fluid_speed_relative_error",
                "ren_e_geometric_speed_relative_error",
                "sessile_final_contact_angle_absolute_error_degrees",
                "free_surface_energy_max_energy_above_initial_relative_proxy",
                "free_surface_energy_relative_total_energy_change_proxy",
            ))
    summary: dict[str, Any] = {}
    for group_name, (kind, metric_names) in requested.items():
        kind_results = [result for result in results if result["kind"] == kind]
        if group_name.startswith("static_sessile_theta"):
            angle = int(group_name.removeprefix("static_sessile_theta"))
            kind_results = [
                result for result in kind_results
                if result["probe"].get("benchmark", {}).get(
                    "sessile_contact", {}).get(
                        "equilibrium_contact_angle_degrees") == angle
            ]
        elif group_name.startswith("dynamic_contact_"):
            advancing = group_name.endswith("advancing")
            filtered = []
            for result in kind_results:
                contact = result["probe"].get("benchmark", {}).get(
                    "sessile_contact", {})
                initial = contact.get("initial_contact_angle_degrees")
                equilibrium = contact.get("equilibrium_contact_angle_degrees")
                if isinstance(initial, (int, float)) and isinstance(
                        equilibrium, (int, float)):
                    if (initial > equilibrium) == advancing:
                        filtered.append(result)
            kind_results = filtered
        metric_summary: dict[str, Any] = {}
        for metric_name in metric_names:
            samples = []
            for result in kind_results:
                probe = result["probe"]
                resolution = (
                    probe.get("benchmark", {}).get("mesh_resolution", {}).get("nx")
                    if isinstance(probe, dict) else None
                )
                value = probe.get(metric_name) if isinstance(probe, dict) else None
                if isinstance(resolution, (int, float)) and isinstance(value, (int, float)):
                    samples.append({
                        "probe": result["name"],
                        "resolution": int(resolution),
                        "h": 1.0 / float(resolution),
                        "value": float(value),
                        "absolute_value": abs(float(value)),
                    })
            samples.sort(key=lambda sample: sample["resolution"])
            rates = []
            for coarse, fine in zip(samples[:-1], samples[1:]):
                coarse_error = coarse["absolute_value"]
                fine_error = fine["absolute_value"]
                rate = None
                if coarse_error > 0.0 and fine_error > 0.0:
                    rate = math.log(coarse_error / fine_error) / math.log(
                        coarse["h"] / fine["h"])
                rates.append({
                    "coarse_resolution": coarse["resolution"],
                    "fine_resolution": fine["resolution"],
                    "observed_rate": rate,
                })
            if samples:
                metric_summary[metric_name] = {"samples": samples, "rates": rates}
        if metric_summary:
            summary[group_name] = metric_summary

    symmetric_records = symmetric_dynamic_response_records(results)
    if symmetric_records:
        symmetric_metrics: dict[str, Any] = {}
        metric_names = (
            "equilibrium_bias_speed",
            "pair_even_vs_equilibrium_absolute_defect",
            "centered_odd_response_relative_error",
            "centered_antisymmetry_relative_defect",
            "maximum_reference_area_relative_mismatch",
        )
        for metric_name in metric_names:
            samples = [
                {
                    "probe": "equal_area_advancing_equilibrium_receding_triad",
                    "resolution": int(record["resolution"]),
                    "h": float(record["h"]),
                    "value": float(record[metric_name]),
                    "absolute_value": abs(float(record[metric_name])),
                }
                for record in symmetric_records if metric_name in record
            ]
            rates = []
            for coarse, fine in zip(samples[:-1], samples[1:]):
                rate = None
                if coarse["absolute_value"] > 0.0 and fine["absolute_value"] > 0.0:
                    rate = math.log(
                        coarse["absolute_value"] / fine["absolute_value"]
                    ) / math.log(coarse["h"] / fine["h"])
                rates.append({
                    "coarse_resolution": coarse["resolution"],
                    "fine_resolution": fine["resolution"],
                    "observed_rate": rate,
                })
            symmetric_metrics[metric_name] = {"samples": samples, "rates": rates}
        summary["dynamic_contact_symmetric_triad"] = symmetric_metrics
    return summary


def refinement_acceptance(convergence: dict[str, Any]) -> dict[str, Any]:
    """Require demonstrable error reduction, not merely multiple mesh runs."""
    required: dict[str, tuple[str, ...]] = {
        "capillary_equilibrium": (
            "capillary_pressure_jump_relative_error",
            "max_speed",
        ),
        "capillary_wave": (
            "capillary_wave_omega_relative_error",
            "capillary_wave_profile_relative_error",
        ),
        "dynamic_contact_advancing": (
            "ren_e_contact_fluid_speed_relative_error",
            "ren_e_geometric_speed_relative_error",
        ),
        "dynamic_contact_receding": (
            "ren_e_contact_fluid_speed_relative_error",
            "ren_e_geometric_speed_relative_error",
        ),
        "dynamic_contact_sidewall_advancing": (
            "ren_e_contact_fluid_speed_relative_error",
            "ren_e_geometric_speed_relative_error",
        ),
        "dynamic_contact_sidewall_receding": (
            "ren_e_contact_fluid_speed_relative_error",
            "ren_e_geometric_speed_relative_error",
        ),
    }
    for angle in (60, 90, 120):
        required[f"static_sessile_theta{angle}"] = (
            "sessile_final_contact_angle_absolute_error_degrees",
            "sessile_final_pressure_jump_relative_error",
            "sessile_final_liquid_area_relative_error",
            "sessile_final_parasitic_capillary_number",
        )

    errors: list[str] = []
    accepted_rates: list[dict[str, Any]] = []
    for group_name, metric_names in required.items():
        group = convergence.get(group_name)
        if not isinstance(group, dict):
            errors.append(f"missing refinement group {group_name}")
            continue
        for metric_name in metric_names:
            record = group.get(metric_name)
            if not isinstance(record, dict):
                errors.append(f"{group_name}/{metric_name}: metric is unavailable")
                continue
            samples = record.get("samples", [])
            rates = record.get("rates", [])
            if not isinstance(samples, list) or len(samples) < 2:
                errors.append(
                    f"{group_name}/{metric_name}: fewer than two solved mesh samples"
                )
                continue
            by_resolution = {
                sample.get("resolution"): sample
                for sample in samples
                if isinstance(sample, dict)
            }
            if not isinstance(rates, list) or not rates:
                errors.append(f"{group_name}/{metric_name}: no refinement rate")
                continue
            for rate_record in rates:
                coarse_resolution = rate_record.get("coarse_resolution")
                fine_resolution = rate_record.get("fine_resolution")
                coarse = by_resolution.get(coarse_resolution, {})
                fine = by_resolution.get(fine_resolution, {})
                fine_error = fine.get("absolute_value")
                observed_rate = rate_record.get("observed_rate")
                label = (
                    f"{group_name}/{metric_name} "
                    f"{coarse_resolution}->{fine_resolution}"
                )
                if (isinstance(fine_error, (int, float)) and
                        float(fine_error) <= REFINEMENT_EXACTNESS_FLOOR):
                    accepted_rates.append({
                        "metric": label,
                        "observed_rate": observed_rate,
                        "accepted_as_numerically_exact": True,
                    })
                    continue
                if not isinstance(observed_rate, (int, float)):
                    errors.append(f"{label}: refinement rate is unavailable")
                elif float(observed_rate) < MIN_ACCEPTED_REFINEMENT_RATE:
                    errors.append(
                        f"{label}: observed rate {float(observed_rate):.6g} is below "
                        f"{MIN_ACCEPTED_REFINEMENT_RATE:.6g}"
                    )
                else:
                    accepted_rates.append({
                        "metric": label,
                        "observed_rate": float(observed_rate),
                        "accepted_as_numerically_exact": False,
                    })
    return {
        "passed": not errors,
        "minimum_observed_rate": MIN_ACCEPTED_REFINEMENT_RATE,
        "numerical_exactness_floor": REFINEMENT_EXACTNESS_FLOOR,
        "accepted_rates": accepted_rates,
        "errors": errors,
    }


def write_markdown(path: Path,
                   solver: Path,
                   results: list[dict[str, Any]],
                   convergence: dict[str, Any],
                   refinement_qualification: dict[str, Any]) -> None:
    lines = [
        "# FS-16 production physical-verification matrix",
        "",
        f"Production executable: `{solver}`",
        "",
        "A row is production evidence only when `converged=yes`. Instrumentation-only "
        "geometry checks and failed solves are retained but do not qualify FS-16.",
        "",
        "| probe | kind | converged | angle error (deg) | pressure-jump rel. error | "
        "parasitic Ca | contact-fluid/predicted Ren--E speed | sign | errors |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for result in results:
        probe = result["probe"]
        errors = probe.get("errors", []) if isinstance(probe, dict) else []
        if not errors and result["returncode"] != 0:
            errors = [f"runner return code {result['returncode']}"]
        lines.append(
            "| {name} | {kind} | {converged} | {angle} | {pressure} | {speed} | "
            "{ratio} | {sign} | {errors} |".format(
                name=result["name"],
                kind=result["kind"],
                converged="yes" if result["production_solve_converged"] else "no",
                angle=metric(probe, "sessile_final_contact_angle_absolute_error_degrees"),
                pressure=metric(probe, "sessile_final_pressure_jump_relative_error"),
                speed=metric(probe, "sessile_final_parasitic_capillary_number"),
                ratio=metric(probe, "ren_e_speed_ratio_measured_to_predicted"),
                sign=metric(probe, "ren_e_speed_sign_agrees"),
                errors="; ".join(str(error).replace("|", "\\|") for error in errors) or "none",
            )
        )
    lines.extend([
        "",
        "## Refinement evidence",
        "",
        "Rates use `log(error_coarse/error_fine)/log(h_coarse/h_fine)`. Negative "
        "rates are retained and are disqualifying evidence, not hidden.",
        f"The a-priori acceptance rate is {MIN_ACCEPTED_REFINEMENT_RATE:g}; a fine-grid "
        f"error at or below {REFINEMENT_EXACTNESS_FLOOR:g} is accepted as numerically exact.",
        "",
    ])
    for kind, metrics in convergence.items():
        lines.extend([f"### {kind}", ""])
        for name, data in metrics.items():
            samples = ", ".join(
                f"n={sample['resolution']}: {sample['value']:.6g}"
                for sample in data["samples"]
            )
            rates = ", ".join(
                f"{rate['coarse_resolution']}->{rate['fine_resolution']}: "
                + ("n/a" if rate["observed_rate"] is None
                   else f"{rate['observed_rate']:.4g}")
                for rate in data["rates"]
            )
            lines.append(f"- `{name}` — {samples}; rates: {rates or 'n/a'}")
        lines.append("")
    lines.extend([
        "## Refinement qualification",
        "",
        "Passed: " + ("yes" if refinement_qualification.get("passed") else "no"),
        "",
    ])
    refinement_errors = refinement_qualification.get("errors", [])
    if refinement_errors:
        lines.extend(f"- {error}" for error in refinement_errors)
        lines.append("")
    lines.extend([
        "Exact commands, preserved run directories, full parsed metrics, and stdout paths "
        "are in `matrix_summary.json`.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solver", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mesh", type=int, action="append")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    solver = args.solver.resolve()
    if not solver.is_file():
        raise FileNotFoundError(solver)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    meshes = args.mesh or ([8, 16] if args.quick else [16, 32])
    results = [run_one(spec, solver, output_dir)
               for spec in matrix_specs(meshes, args.quick)]
    convergence = refinement_summary(results)
    refinement_qualification = refinement_acceptance(convergence)
    all_solves_converged = all(
        result["production_solve_converged"] for result in results)
    summary = {
        "solver": str(solver),
        "all_production_solves_converged": all_solves_converged,
        "refinement_qualification": refinement_qualification,
        "fs16_physical_qualification_passed": (
            all_solves_converged and refinement_qualification["passed"]),
        "results": results,
        "refinement": convergence,
    }
    (output_dir / "matrix_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(
        output_dir / "README.md",
        solver,
        results,
        convergence,
        refinement_qualification,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["fs16_physical_qualification_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

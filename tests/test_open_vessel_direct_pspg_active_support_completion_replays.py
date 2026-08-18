import importlib.util
import json
from pathlib import Path
import sys


def _load_audit_module():
    repo = Path(__file__).resolve().parents[1]
    script = (
        repo
        / "tests"
        / "cases"
        / "fluid"
        / "open_vessel_free_surface"
        / "audit_direct_pspg_active_support_completion_replays.py"
    )
    spec = importlib.util.spec_from_file_location(
        "audit_direct_pspg_active_support_completion_replays", script
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def _pressure_audit(
    *,
    threshold: float,
    update_pa: float,
    support_class: str,
    point_index: int,
    triggered: bool = True,
    full_wet_update_pa: float | None = None,
    cut_update_pa: float | None = None,
    tiny_cut_update_pa: float | None = None,
) -> dict:
    stats = {
        "active_or_wet_supported": {"max_abs_delta_pa": update_pa},
    }
    if full_wet_update_pa is not None:
        stats["full_wet_supported"] = {
            "max_abs_delta_pa": full_wet_update_pa
        }
    if cut_update_pa is not None:
        stats["cut_supported"] = {"max_abs_delta_pa": cut_update_pa}
    if tiny_cut_update_pa is not None:
        stats["tiny_cut_supported"] = {
            "max_abs_delta_pa": tiny_cut_update_pa
        }
    return {
        "absolute_threshold_pa": threshold,
        "finding": "triggered" if triggered else "not triggered",
        "status": (
            "diagnostic_pressure_update_guard_triggered"
            if triggered
            else "diagnostic_pressure_update_guard_no_threshold_trigger"
        ),
        "transitions": [
            {
                "delta_statistics_by_category": stats,
                "max_by_category": {
                    "active_or_wet_supported": {
                        "abs_pressure_delta_pa": update_pa,
                        "point_index": point_index,
                        "pressure_delta_pa": update_pa,
                        "support_class": support_class,
                    }
                },
            }
        ],
    }


def _support_audit(
    *,
    candidate_count: int,
    neighbor_count: int,
    edge_count: int,
    edge_weight: float,
    min_weight: float,
    max_weight: float,
    max_active_neighbors: int,
    max_update_pa: float,
    max_update_global_dof: int,
) -> dict:
    return {
        "latest_pressure_graph_completion": {
            "values": {
                "applied": 1,
                "candidate_row_count": candidate_count,
                "candidate_with_existing_pressure_edge_count": 0,
                "candidate_with_laplacian_pressure_edge_count": 0,
                "edge_count": edge_count,
                "edge_weight": edge_weight,
                "max_active_neighbors": max_active_neighbors,
                "max_completion_edge_weight": max_weight,
                "min_completion_edge_weight": min_weight,
                "mode": "active_support_completion",
                "neighbor_row_count": neighbor_count,
                "requested_mode": "active_support_completion",
                "weak_coupling_and_self_candidate_count": 0,
                "weak_coupling_candidate_count": 6,
                "weak_self_candidate_count": candidate_count - 6,
                "zero_coupling_candidate_count": 0,
                "zero_self_candidate_count": 0,
            }
        },
        "latest_pressure_update_support_diagnostic": {
            "values": {
                "max_abs_update": max_update_pa,
                "max_update_global_dof": max_update_global_dof,
                "positive_coupling_max_abs_update": max_update_pa,
                "positive_self_max_abs_update": max_update_pa / 2.0,
                "weak_coupling_max_abs_update": max_update_pa / 2.0,
                "weak_self_max_abs_update": max_update_pa,
                "zero_coupling_max_abs_update": 0.0,
            }
        },
    }


def _log_text(*, residual: float) -> str:
    return "\n".join(
        [
            "[svMultiPhysics::Application] TimeLoop: entering loop.run()",
            (
                "[svMultiPhysics::Application] TimeLoop: nonlinear_done step=0 "
                "time=0.9 converged=1 iters=1 "
                f"||r||={residual:.16e} ||r_field||={residual:.16e} "
                "||r_aux||=0.0000000000000000e+00 "
                "(linear: converged=1 iters=1 rel=7.0e-14)"
            ),
            (
                "[svMultiPhysics::Application] TimeLoop: loop.run() returned "
                "success=1 steps_taken=1 final_time=0.900625 message=''"
            ),
        ]
    )


def _write_required_inputs(audit, root: Path, *, uncapped_test10_triggered=True):
    values = {
        ("active_support_neigh64", "test02"): {
            "threshold": 100000.0,
            "update": 186507.92759082434,
            "support_class": "full_wet_supported",
            "point": 1172,
            "full": 186507.92759082434,
            "cut": 158968.08756291826,
            "tiny": 158968.08756291826,
            "candidate": 304,
            "neighbor": 368,
            "edge": 19456,
            "weight": 1.978976717737339e-11,
            "min_weight": 1.978976717737339e-11,
            "max_weight": 1.978976717737339e-11,
            "max_neighbors": 64,
            "global_dof": 10676,
        },
        ("active_support_neigh64", "test10"): {
            "threshold": 100.0,
            "update": 201.1556177587019,
            "support_class": "cut_supported",
            "point": 609,
            "full": 145.34506703510738,
            "cut": 201.1556177587019,
            "tiny": None,
            "candidate": 68,
            "neighbor": 132,
            "edge": 4352,
            "weight": 6.335956786031046e-10,
            "min_weight": 6.335956786031046e-10,
            "max_weight": 6.335956786031046e-10,
            "max_neighbors": 64,
            "global_dof": 3837,
        },
        ("active_support_all", "test02"): {
            "threshold": 100000.0,
            "update": 155956.10179486268,
            "support_class": "full_wet_supported",
            "point": 1172,
            "full": 155956.10179486268,
            "cut": 127471.53305378903,
            "tiny": 127471.53305378903,
            "candidate": 304,
            "neighbor": 879,
            "edge": 220856,
            "weight": 2.8850685634439568e-12,
            "min_weight": 1.4425342817219784e-12,
            "max_weight": 2.8850685634439568e-12,
            "max_neighbors": -1,
            "global_dof": 10676,
        },
        ("active_support_all", "test10"): {
            "threshold": 100.0,
            "update": 203.0459932023828,
            "support_class": "cut_supported",
            "point": 609,
            "full": 132.0798509906606,
            "cut": 203.0459932023828,
            "tiny": None,
            "candidate": 68,
            "neighbor": 251,
            "edge": 14722,
            "weight": 3.2440098744478956e-10,
            "min_weight": 1.6220049372239478e-10,
            "max_weight": 3.2440098744478956e-10,
            "max_neighbors": -1,
            "global_dof": 3837,
        },
    }
    for variant_key, variant in audit.VARIANTS.items():
        for label, spec in variant["cases"].items():
            data = values[(variant_key, label)]
            triggered = not (
                variant_key == "active_support_all"
                and label == "test10"
                and not uncapped_test10_triggered
            )
            _write_json(
                root / spec["pressure"],
                _pressure_audit(
                    threshold=data["threshold"],
                    update_pa=data["update"],
                    support_class=data["support_class"],
                    point_index=data["point"],
                    triggered=triggered,
                    full_wet_update_pa=data["full"],
                    cut_update_pa=data["cut"],
                    tiny_cut_update_pa=data["tiny"],
                ),
            )
            _write_json(
                root / spec["support"],
                _support_audit(
                    candidate_count=data["candidate"],
                    neighbor_count=data["neighbor"],
                    edge_count=data["edge"],
                    edge_weight=data["weight"],
                    min_weight=data["min_weight"],
                    max_weight=data["max_weight"],
                    max_active_neighbors=data["max_neighbors"],
                    max_update_pa=data["update"],
                    max_update_global_dof=data["global_dof"],
                ),
            )
            log_path = root / spec["log"]
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text(_log_text(residual=1.0e-3) + "\n")


def test_active_support_completion_replays_rule_out_raw_completion(tmp_path):
    audit = _load_audit_module()
    _write_required_inputs(audit, tmp_path)

    report = audit.build_report(tmp_path)

    assert report["finding"] == (
        "direct_pspg_active_support_completion_replays_rule_out_raw_"
        "active_support_completion"
    )
    assert report["status"] == (
        "raw_active_support_completion_directional_but_insufficient"
    )
    assert report["all_replays_guard_triggered"]
    assert report["all_replays_accepted_one_step"]
    assert report["all_neighbor_settings_confirmed"]
    assert report["case_updates_pa"]["active_support_neigh64"]["test02"] == (
        186507.92759082434
    )
    assert report["case_updates_pa"]["active_support_all"]["test10"] == (
        203.0459932023828
    )
    assert report["cap_removal"]["cap64_neighbor_cap_limited_all_cases"]
    assert report["cap_removal"]["uncapped_still_triggers_all_cases"]
    assert report["cap_removal"]["by_case"]["test02"][
        "uncapped_minus_cap64_update_pa"
    ] < 0.0
    assert report["cap_removal"]["by_case"]["test10"][
        "uncapped_minus_cap64_update_pa"
    ] > 0.0
    assert "formulation-derived physical support" in report["next_requirement"]


def test_active_support_completion_replays_report_missing_inputs(tmp_path):
    audit = _load_audit_module()

    report = audit.build_report(tmp_path)

    assert report["finding"] == (
        "direct_pspg_active_support_completion_replays_missing_evidence"
    )
    assert report["status"] == "regenerate_active_support_completion_replays"
    assert len(report["missing_evidence"]) == 12


def test_active_support_completion_clear_requires_transfer_check(tmp_path):
    audit = _load_audit_module()
    _write_required_inputs(audit, tmp_path, uncapped_test10_triggered=False)

    report = audit.build_report(tmp_path)

    assert report["finding"] == (
        "direct_pspg_active_support_completion_replays_need_transfer_check"
    )
    assert report["status"] == (
        "active_support_completion_requires_cross_case_validation"
    )
    assert report["all_replays_guard_triggered"] is False
    assert report["cap_removal"]["uncapped_still_triggers_all_cases"] is False

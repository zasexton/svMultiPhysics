import importlib.util
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
        / "audit_direct_pspg_formulation_vocabulary_support.py"
    )
    spec = importlib.util.spec_from_file_location(
        "audit_direct_pspg_formulation_vocabulary_support",
        script,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_form_vocabulary_audit_requires_topology_api_for_direct_pspg(tmp_path):
    audit = _load_audit_module()
    _write(
        tmp_path / audit.FORM_FILES["cut_cell_forms"],
        """
        FormExpr cutVolumeFraction();
        FormExpr cutSideIndicator();
        FormExpr cutEmbeddedNormal();
        FormExpr cutStabilizationScale();
        struct CutCellFormTerminals {};
        """,
    )
    _write(
        tmp_path / audit.FORM_FILES["form_expr"],
        """
        FormExpr dx() const;
        FormExpr ds(int marker) const;
        FormExpr dS(int marker) const;
        FormExpr dI(int marker) const;
        FormExpr dCutVolume(int marker, CutVolumeSide side) const;
        """,
    )
    _write(
        tmp_path / audit.FORM_FILES["vocabulary"],
        """
        Current public cut-cell helpers include cutVolumeFraction(),
        cutSideIndicator(), cutEmbeddedNormal(), and cutStabilizationScale().
        """,
    )
    _write(
        tmp_path / audit.FORM_FILES["navier_stokes_vms"],
        """
        const auto pspg_pressure_gradient_support_scale =
            1.0 / cutVolumeFraction();
        const auto vms_pspg_pressure_gradient_integrand =
            pspg_pressure_gradient_support_scale * grad(p);
        auto continuity = integrateOnActiveVolume(
            vms_pspg_pressure_gradient_integrand, active_volume_domain);
        auto cut = continuity.dCutVolume(1, CutVolumeSide::Negative);
        FormExpr vms_pspg_boundary_pressure_gradient_form;
        FormExpr vms_pspg_boundary_tangential_pressure_gradient_form;
        """,
    )

    report = audit.build_report(tmp_path)

    assert report["finding"] == (
        "form_vocabulary_lacks_direct_pspg_support_topology_handles"
    )
    assert report["status"] == "requires_fe_forms_or_assembly_api_extension"
    assert "cutVolumeFraction" in report["public_cut_cell_helpers"]
    assert "dCutVolume" in report["public_measures"]
    assert report["direct_pspg_expression_summary"] == {
        "direct_pressure_gradient_integrand_installed": True,
        "active_volume_measure_used": True,
        "cut_volume_fraction_scale_available": True,
        "free_surface_boundary_terms_separate": True,
    }
    assert all(report["required_topology_handles_missing"].values())
    assert report["missing_required_topology_handle_count"] == (
        report["required_topology_handle_count"]
    )
    assert "another scalar form multiplier" in report["next_requirement"]

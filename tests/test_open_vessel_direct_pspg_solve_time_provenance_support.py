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
        / "audit_direct_pspg_solve_time_provenance_support.py"
    )
    spec = importlib.util.spec_from_file_location(
        "audit_direct_pspg_solve_time_provenance_support",
        script,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _standard_assembler_fixture(*, include_source_filter=True):
    source_filter = (
        """
[[nodiscard]] std::string cutVolumeDirectPspgSupportCouplingSourceComponentFilter()
{
    static const std::string value = []() {
        const char* raw = std::getenv(
            "SVMP_FE_CUT_VOLUME_DIRECT_PSPG_SUPPORT_COUPLING_SOURCE_COMPONENT");
        if (raw == nullptr || raw[0] == '\\0') {
            return std::string{"navier_stokes_vms_pspg_pressure_gradient"};
        }
        return std::string(raw);
    }();
    return value;
}
"""
        if include_source_filter
        else ""
    )
    return (
        """
[[nodiscard]] bool cutVolumeDirectPspgSupportCouplingProvenanceDiagnosticEnabled() noexcept
{
    static const bool enabled = envFlagEnabled(
        "SVMP_FE_CUT_VOLUME_DIRECT_PSPG_SUPPORT_COUPLING_PROVENANCE_DIAGNOSTIC");
    return enabled;
}

[[nodiscard]] std::string cutVolumeDirectPspgSupportCouplingOperatorFilter()
{
    static const std::string value = []() {
        const char* raw = std::getenv(
            "SVMP_FE_CUT_VOLUME_DIRECT_PSPG_SUPPORT_COUPLING_OPERATOR");
        if (raw == nullptr || raw[0] == '\\0') {
            return std::string{"equations"};
        }
        return std::string(raw);
    }();
    return value;
}

[[nodiscard]] std::size_t cutVolumeLocalMatrixColumnSupportMaxColumns() noexcept
{
    return 16u;
}
"""
        + source_filter
        + """
void logCutVolumeDirectPspgSupportCouplingProvenance(
    const AssemblyDiagnosticContext& diagnostic_context)
{
    if (!fieldNameEquals(diagnostic_context.test_field_name, "pressure")) {
        return;
    }
    const bool pressure_pressure_block =
        fieldNameEquals(diagnostic_context.trial_field_name, "pressure");
    const bool pressure_velocity_block =
        fieldNameEquals(diagnostic_context.trial_field_name, "velocity");
    const char* block_name = pressure_pressure_block ? "pressure_pressure"
                                                     : "pressure_velocity";
    std::ostringstream oss;
    oss << "StandardAssembler: diagnostic=cut_volume_direct_pspg_support_coupling_provenance"
        << " source_component='" << diagnostic_context.source_component_tag << "'"
        << " block=" << block_name
        << " source_edge_count=" << 2
        << " two_hop_completion_count=" << 1
        << " local_clustering=" << 0.5
        << " sampled_col_count=" << 4
        << " sample_truncated=" << 0
        << " sample_sorted_by=abs_desc"
        << " diag_in_sample=" << 1
        << " sampled_col_local_indices=0|1|2|3"
        << " sampled_col_dofs=10|11|12|13"
        << " sampled_col_values=1|-0.5|0.25|-0.125"
        << " sampled_col_abs_values=1|0.5|0.25|0.125"
        << " sampled_col_signs=1|-1|1|-1"
        << " pressure_update_sign_used=0"
        << " diagnostic_only=1";
    const std::size_t max_column_samples =
        cutVolumeLocalMatrixColumnSupportMaxColumns();
}

void assembleLegacy()
{
            logCutVolumeDirectPspgSupportCouplingProvenance(
                *diagnostic_context_);
            applyCutVolumeDirectPspgTopologyPolicy(
                *diagnostic_context_);
        insertLocalForCell(cell_id, row_dof_map_);
}

void assembleFused()
{
                logCutVolumeDirectPspgSupportCouplingProvenance(
                    *active_diagnostic_context);
                applyCutVolumeDirectPspgTopologyPolicy(
                    *active_diagnostic_context);
            insertLocalForCell(cell_id, t.row_dof_map);
}
"""
    )


def _navier_stokes_fixture():
    return """
void install()
{
    direct_pspg_install.source_component_tag =
        "navier_stokes_vms_pspg_pressure_gradient";
    direct_pspg_install.extra_trial_fields.push_back(u_id);
}
"""


def test_solve_time_direct_pspg_provenance_is_ready_when_scoped_and_preupdate():
    audit = _load_audit_module()
    report = audit.build_report(
        standard_assembler_text=_standard_assembler_fixture(),
        navier_stokes_text=_navier_stokes_fixture(),
    )

    assert report["finding"] == (
        "solve_time_direct_pspg_support_coupling_provenance_ready"
    )
    assert report["status"] == "diagnostic_ready_replay_pending"
    assert report["missing_features"] == []
    assert report["features"]["records_pressure_update_sign_not_used"]
    assert report["features"]["emits_pressure_velocity_block"]
    assert report["features"]["emits_sampled_column_payload"]
    assert report["features"]["uses_bounded_column_sample"]
    assert report["features"]["records_sample_order_and_diag_membership"]
    assert "SVMP_FE_CUT_VOLUME_DIRECT_PSPG_SUPPORT_COUPLING_PROVENANCE_DIAGNOSTIC" in (
        report["diagnostic_env"]
    )


def test_solve_time_direct_pspg_provenance_requires_source_component_scope():
    audit = _load_audit_module()
    report = audit.build_report(
        standard_assembler_text=_standard_assembler_fixture(
            include_source_filter=False
        ),
        navier_stokes_text=_navier_stokes_fixture(),
    )

    assert report["finding"] == (
        "solve_time_direct_pspg_support_coupling_provenance_incomplete"
    )
    assert "source_component_filter_env_present" in report["missing_features"]

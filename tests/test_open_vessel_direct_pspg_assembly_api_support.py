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
        / "audit_direct_pspg_assembly_api_support.py"
    )
    spec = importlib.util.spec_from_file_location(
        "audit_direct_pspg_assembly_api_support",
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


def test_assembly_api_audit_reports_policy_hook_with_composite_provenance_pending(tmp_path):
    audit = _load_audit_module()
    _write(
        tmp_path / audit.SOURCE_FILES["assembler_header"],
        """
        /**
         * Disabled-by-default assembly policies may also use this provenance.
         */
        struct AssemblyDiagnosticContext {
            std::string operator_tag{};
            std::string source_component_tag{};
            std::string test_field_name{};
            std::string trial_field_name{};
        };
        """,
    )
    _write(
        tmp_path / audit.SOURCE_FILES["fesystem_header"],
        """
        struct PlannedCutVolumeTerm {
            int marker{0};
            geometry::CutIntegrationSide side{geometry::CutIntegrationSide::Negative};
            FieldId test_field{INVALID_FIELD_ID};
            FieldId trial_field{INVALID_FIELD_ID};
            std::string source_component_tag{};
            const spaces::FunctionSpace* test_space{nullptr};
            const spaces::FunctionSpace* trial_space{nullptr};
            assembly::AssemblyKernel* kernel{nullptr};
            const dofs::DofMap* row_dof_map{nullptr};
            const dofs::DofMap* col_dof_map{nullptr};
            GlobalIndex row_dof_offset{0};
            GlobalIndex col_dof_offset{0};
            bool matrix_capable{false};
            bool vector_capable{false};
        };
        """,
    )
    _write(
        tmp_path / audit.SOURCE_FILES["fesystem_cpp"],
        """
        void FESystem::addCutVolumeKernel(
            OperatorTag op,
            InterfaceId interface_marker,
            geometry::CutIntegrationSide side,
            FieldId test_field,
            FieldId trial_field,
            std::shared_ptr<assembly::AssemblyKernel> kernel,
            std::string source_component_tag)
        {
            def.cut_volumes.push_back(CutVolumeTerm{
                interface_marker,
                side,
                test_field,
                trial_field,
                std::move(source_component_tag),
                std::move(kernel)});
        }
        """,
    )
    _write(
        tmp_path / audit.SOURCE_FILES["forms_installer"],
        """
        void registerKernel(std::string source_component_tag = {}) {
            system.addCutVolumeKernel(
                op,
                region.marker,
                toGeometrySide(region.side),
                test_field,
                trial_field,
                kernel,
                source_component_tag);
        }
        """,
    )
    _write(
        tmp_path / audit.SOURCE_FILES["operator_registry"],
        """
        struct CutVolumeTerm {
            int marker{0};
            geometry::CutIntegrationSide side{geometry::CutIntegrationSide::Negative};
            FieldId test_field{INVALID_FIELD_ID};
            FieldId trial_field{INVALID_FIELD_ID};
            std::string source_component_tag{};
            std::shared_ptr<assembly::AssemblyKernel> kernel;
        };
        """,
    )
    _write(
        tmp_path / audit.SOURCE_FILES["standard_assembler"],
        """
        enum class CutVolumeDirectPspgTopologyPolicy {
            Off,
            LocalSchurCompletion,
            LocalEdgeBalance,
            LocalSchurEdgeBalance
        };
        auto default_policy = CutVolumeDirectPspgTopologyPolicy::Off;
        auto policy_env = "SVMP_FE_CUT_VOLUME_DIRECT_PSPG_TOPOLOGY_POLICY";
        auto full_cell_env =
            "SVMP_FE_CUT_VOLUME_DIRECT_PSPG_TOPOLOGY_APPLY_FULL_CELL";
        auto op_env = "SVMP_FE_CUT_VOLUME_DIRECT_PSPG_TOPOLOGY_OPERATOR";
        auto op_default = std::string{"equations"};
        auto source_env =
            "SVMP_FE_CUT_VOLUME_DIRECT_PSPG_TOPOLOGY_SOURCE_COMPONENT";
        auto source_default =
            "navier_stokes_vms_pspg_pressure_gradient";
        auto schur_name = "local_schur_completion";
        auto edge_name = "local_edge_balance";
        std::string cutVolumeDirectPspgTopologyOperatorFilter();
        std::string cutVolumeDirectPspgTopologySourceComponentFilter();
        void addNullPreservingPressureEdge(KernelOutput& output) {
            output.local_matrix[matrix_index(i, i)] += weight;
            output.local_matrix[matrix_index(i, j)] -= weight;
            std::fprintf(stderr, "constant_pressure_null_preserving=1");
        }
        void applyCutVolumeDirectPspgTopologyPolicy(
            const AssemblyDiagnosticContext& diagnostic_context,
            KernelOutput& output)
        {
            fieldNameEquals(diagnostic_context.test_field_name, "pressure");
            fieldNameEquals(diagnostic_context.trial_field_name, "pressure");
            if (row_dofs[i] != col_dofs[i]) return;
            if (rule.full_cell_equivalent) return;
            addNullPreservingPressureEdge(output);
            std::fprintf(
                stderr,
                "cut_volume_direct_pspg_topology_policy "
                "solve_affecting=1 diagnostic_only=0 matrix_mutated=");
        }
        void assembleCutVolumes() {
            std::fprintf(stderr, "source_component='");
            logCutVolumeDirectPspgLocalSchurDiagnostic();
            logCutVolumeDirectPspgLocalEdgeBalanceDiagnostic();
            std::fprintf(stderr, "diagnostic_only=1");
            std::fprintf(stderr, "cut_volume_direct_pspg_local_schur_completion");
            std::fprintf(stderr, "cut_volume_direct_pspg_local_edge_balance");
            applyCutVolumeDirectPspgTopologyPolicy(
                diagnostic_context,
                interface_marker,
                side,
                rule,
                nullptr,
                rule_index,
                cell_id,
                active_rule->num_points(),
                kernel_output_,
                row_dofs,
                col_dofs);
            }
            stage_start = cut_now();
            insertLocalForCell();
        }
        """,
    )
    _write(
        tmp_path / audit.SOURCE_FILES["system_assembly"],
        """
        auto make_cut_volume_diagnostic_context = [&] {
            return AssemblyDiagnosticContext{
                request.op,
                entry.term->source_component_tag,
                test_field.name,
                trial_field.name};
        };
        std::fprintf(stderr, "source_component='");
        auto same_cut_volume_insertion_key = [](const auto& lhs, const auto& rhs) {
            const auto& a = *lhs.term;
            const auto& b = *rhs.term;
            return a.source_component_tag == b.source_component_tag;
        };
        if (insertion_group.size() == 1u) {
            fused_term.diagnostic_context = diagnostic_context;
        } else {
            fused_term.diagnostic_context = diagnostic_context;
        }
        """,
    )
    _write(
        tmp_path / audit.SOURCE_FILES["system_setup"],
        """
        void FESystem::buildAssemblyPlans() {
            plan.cut_volume_terms.push_back(PlannedCutVolumeTerm{
                term.marker,
                term.side,
                term.test_field,
                term.trial_field,
                term.source_component_tag,
                test_field.space.get()});
        }
        """,
    )
    _write(
        tmp_path / audit.SOURCE_FILES["navier_stokes_vms"],
        """
        const auto residual = momentum_form + continuity_form;
        (void)FE::systems::installFormulation(
            system, "equations", {u_id, p_id}, residual, install);
        auto direct_pspg_install = install;
        direct_pspg_install.source_component_tag =
            "navier_stokes_vms_pspg_pressure_gradient";
        direct_pspg_install.extra_trial_fields.push_back(u_id);
        (void)FE::systems::installFormulation(
            system,
            "equations",
            {p_id},
            vms_pspg_pressure_gradient_form,
            direct_pspg_install);
        if (pressureRowContributionDiagnosticEnabled()) {
            (void)FE::systems::installFormulation(
                system,
                "equations_diagnostic_ns_vms_pspg_pressure_gradient",
                {p_id},
                vms_pspg_pressure_gradient_integrand,
                diagnostic_install);
        }
        """,
    )

    report = audit.build_report(tmp_path)

    assert report["finding"] == (
        "assembly_api_has_direct_pspg_topology_policy_hook_replay_pending"
    )
    assert report["status"] == "topology_policy_hook_available_replay_pending"
    assert report["assembly_diagnostic_context_fields"] == [
        "operator_tag",
        "source_component_tag",
        "test_field_name",
        "trial_field_name",
    ]
    assert "source_component_tag" in report["planned_cut_volume_term_fields"]
    assert report["assembly_api_features"] == {
        "diagnostic_context_is_documented_non_mutating": False,
        "diagnostic_context_only_operator_and_fields": False,
        "diagnostic_context_has_source_component_tag": True,
        "diagnostic_context_lacks_topology_policy_handle": True,
        "cut_volume_context_built_from_request_op_and_fields": True,
        "cut_volume_context_includes_source_component_tag": True,
        "fused_composite_terms_may_drop_per_term_diagnostic_context": False,
        "fused_composite_terms_preserve_source_component_diagnostic_context": True,
        "planned_cut_volume_term_lacks_source_component_tag": False,
        "planned_cut_volume_term_has_source_component_tag": True,
        "operator_registry_cut_volume_term_has_source_component_tag": True,
        "add_cut_volume_kernel_lacks_source_component_argument": False,
        "add_cut_volume_kernel_has_source_component_argument": True,
        "forms_installer_forwards_cut_volume_only_op_fields_kernel": False,
        "forms_installer_forwards_source_component_tag_to_cut_volumes": True,
        "system_setup_preserves_cut_volume_source_component_tag": True,
        "diagnostic_logs_include_source_component_tag": True,
        "direct_pspg_local_topology_diagnostics_log_before_insert": True,
        "direct_pspg_local_topology_diagnostics_mark_diagnostic_only": True,
        "direct_pspg_topology_policy_api_env_gated": True,
        "direct_pspg_topology_policy_scoped_to_equations_operator": True,
        "direct_pspg_topology_policy_scoped_to_production_source_component": True,
        "direct_pspg_topology_policy_requires_pressure_pressure_block": True,
        "direct_pspg_topology_policy_default_partial_cut_only": True,
        "direct_pspg_topology_policy_constant_null_preserving": True,
        "direct_pspg_topology_policy_log_marks_solve_affecting": True,
        "direct_pspg_topology_policy_mutates_before_global_insert": True,
        "production_direct_pspg_subterm_has_source_component_tag": True,
        "production_direct_pspg_split_preserves_velocity_tangent": True,
        "production_direct_pspg_subterm_lacks_source_component_tag": False,
        "production_equations_installed": True,
        "direct_pspg_diagnostic_operator_installed": True,
        "pressure_row_contribution_diagnostics_env_gated": True,
    }
    assert report["required_api_handles_missing"] == {
        "add_cut_volume_kernel_source_component_argument": False,
        "assembly_diagnostic_source_component_context": False,
        "composite_term_provenance_for_fused_cut_volume_blocks": False,
        "direct_pspg_topology_policy_api": False,
        "forms_installer_source_component_forwarding": False,
        "legacy_planned_cut_volume_source_component_tag_absent": False,
        "planned_cut_volume_source_component_tag": False,
        "production_subterm_provenance_tag": False,
        "solve_affecting_local_matrix_mutation_hook": False,
        "system_setup_source_component_propagation": False,
    }
    assert report["missing_required_api_handle_count"] == 0
    assert report["required_api_handle_count"] == 10
    assert "fused-composite path" in report["conclusion"]
    assert "Run short Test02/Test10 replay windows" in report["next_requirement"]

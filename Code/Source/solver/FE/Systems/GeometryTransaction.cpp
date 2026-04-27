/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Systems/GeometryTransaction.h"

#include <sstream>
#include <vector>

namespace svmp {
namespace FE {
namespace systems {
namespace {

void append_reason(std::vector<std::string>& reasons, bool condition, const char* reason)
{
    if (condition) {
        reasons.emplace_back(reason);
    }
}

std::string join_reasons(const std::vector<std::string>& reasons)
{
    if (reasons.empty()) {
        return "compatible";
    }
    std::ostringstream os;
    for (std::size_t i = 0; i < reasons.size(); ++i) {
        if (i != 0u) {
            os << "; ";
        }
        os << reasons[i];
    }
    return os.str();
}

} // namespace

bool OperatorInvalidationDecision::any_change() const noexcept
{
    return geometry_changed ||
           reference_rebase_changed ||
           topology_changed ||
           ownership_changed ||
           numbering_changed ||
           mesh_field_layout_changed ||
           label_changed ||
           active_configuration_changed ||
           fe_space_changed ||
           fe_dof_layout_changed ||
           fe_constraint_layout_changed ||
           fe_block_layout_changed ||
           transaction_state_changed;
}

bool OperatorInvalidationDecision::requires_matrix_free_refresh() const noexcept
{
    return refresh_matrix_free_geometry || requires_full_layout_rebuild();
}

bool OperatorInvalidationDecision::requires_full_layout_rebuild() const noexcept
{
    return rebuild_dof_layout || rebuild_sparsity_pattern;
}

const char* to_string(GeometryTransactionState state) noexcept
{
    switch (state) {
        case GeometryTransactionState::Committed:
            return "committed";
        case GeometryTransactionState::Trial:
            return "trial";
        case GeometryTransactionState::Accepted:
            return "accepted";
        case GeometryTransactionState::RolledBack:
            return "rolled-back";
    }
    return "unknown";
}

const char* to_string(GeometryConfigurationUse use) noexcept
{
    switch (use) {
        case GeometryConfigurationUse::Reference:
            return "reference";
        case GeometryConfigurationUse::CommittedCurrent:
            return "committed-current";
        case GeometryConfigurationUse::TrialCurrent:
            return "trial-current";
        case GeometryConfigurationUse::AcceptedCurrent:
            return "accepted-current";
        case GeometryConfigurationUse::RolledBackCurrent:
            return "rolled-back-current";
    }
    return "unknown";
}

OperatorInvalidationDecision decideOperatorInvalidation(
    const OperatorRevisionSnapshot& cached,
    const OperatorRevisionSnapshot& current,
    bool allow_lagged_jacobian_on_geometry_change)
{
    OperatorInvalidationDecision d;
    std::vector<std::string> reasons;

    if (!cached.valid || !current.valid) {
        d.geometry_changed = true;
        d.refresh_assembled_matrix = true;
        d.refresh_matrix_free_geometry = true;
        d.rebuild_preconditioner = true;
        d.rebuild_multigrid_hierarchy = true;
        d.reject_lagged_jacobian = !allow_lagged_jacobian_on_geometry_change;
        d.rebuild_interface_maps = true;
        d.invalidate_restart_metadata = true;
        d.reason = "missing revision snapshot";
        return d;
    }

    const bool both_have_mesh = cached.mesh.valid && current.mesh.valid;
    if (both_have_mesh) {
        d.geometry_changed =
            cached.mesh.geometry != current.mesh.geometry ||
            cached.mesh.reference_geometry != current.mesh.reference_geometry ||
            cached.mesh.current_geometry != current.mesh.current_geometry;
        d.reference_rebase_changed =
            cached.mesh.reference_rebase != current.mesh.reference_rebase;
        d.geometry_changed = d.geometry_changed || d.reference_rebase_changed;
        d.topology_changed = cached.mesh.topology != current.mesh.topology;
        d.ownership_changed = cached.mesh.ownership != current.mesh.ownership;
        d.numbering_changed = cached.mesh.numbering != current.mesh.numbering;
        d.mesh_field_layout_changed = cached.mesh.field_layout != current.mesh.field_layout;
        d.label_changed = cached.mesh.labels != current.mesh.labels;
        d.active_configuration_changed =
            cached.mesh.active_configuration != current.mesh.active_configuration ||
            cached.geometry_use != current.geometry_use;
    } else if (cached.mesh.valid != current.mesh.valid) {
        d.geometry_changed = true;
        d.topology_changed = true;
    }

    d.fe_space_changed = cached.fe_layout.space != current.fe_layout.space;
    d.fe_dof_layout_changed = cached.fe_layout.dof_layout != current.fe_layout.dof_layout;
    d.fe_constraint_layout_changed =
        cached.fe_layout.constraint_layout != current.fe_layout.constraint_layout;
    d.fe_block_layout_changed = cached.fe_layout.block_layout != current.fe_layout.block_layout;
    d.transaction_state_changed = cached.geometry_state != current.geometry_state;

    const bool mesh_layout_changed =
        d.topology_changed ||
        d.ownership_changed ||
        d.numbering_changed ||
        d.mesh_field_layout_changed;
    const bool fe_layout_changed =
        d.fe_space_changed ||
        d.fe_dof_layout_changed ||
        d.fe_block_layout_changed;
    const bool constraint_or_label_changed =
        d.fe_constraint_layout_changed ||
        d.label_changed;
    const bool geometry_domain_changed =
        d.geometry_changed ||
        d.active_configuration_changed ||
        d.transaction_state_changed;

    d.rebuild_dof_layout = mesh_layout_changed || fe_layout_changed;
    d.rebuild_sparsity_pattern =
        d.rebuild_dof_layout ||
        d.fe_constraint_layout_changed ||
        d.fe_block_layout_changed;

    d.refresh_assembled_matrix =
        geometry_domain_changed ||
        mesh_layout_changed ||
        fe_layout_changed ||
        constraint_or_label_changed;
    d.refresh_matrix_free_geometry =
        geometry_domain_changed ||
        mesh_layout_changed ||
        fe_layout_changed;
    d.rebuild_preconditioner =
        d.refresh_assembled_matrix ||
        d.rebuild_sparsity_pattern;
    d.rebuild_multigrid_hierarchy =
        d.geometry_changed ||
        d.topology_changed ||
        d.ownership_changed ||
        d.numbering_changed ||
        d.fe_space_changed ||
        d.fe_dof_layout_changed ||
        d.fe_constraint_layout_changed ||
        d.fe_block_layout_changed;
    d.reject_lagged_jacobian =
        (d.refresh_assembled_matrix || d.rebuild_sparsity_pattern) &&
        !(allow_lagged_jacobian_on_geometry_change &&
          geometry_domain_changed &&
          !mesh_layout_changed &&
          !fe_layout_changed &&
          !constraint_or_label_changed);
    d.rebuild_interface_maps =
        geometry_domain_changed ||
        d.topology_changed ||
        d.ownership_changed ||
        d.numbering_changed ||
        d.label_changed;
    d.rebuild_constraint_state =
        d.topology_changed ||
        d.numbering_changed ||
        d.fe_constraint_layout_changed ||
        d.label_changed;
    d.invalidate_restart_metadata = d.any_change();

    append_reason(reasons, d.geometry_changed, "geometry revision changed");
    append_reason(reasons, d.reference_rebase_changed, "reference rebase epoch changed");
    append_reason(reasons, d.active_configuration_changed, "active geometry configuration changed");
    append_reason(reasons, d.transaction_state_changed, "geometry transaction state changed");
    append_reason(reasons, d.topology_changed, "mesh topology changed");
    append_reason(reasons, d.ownership_changed, "mesh ownership changed");
    append_reason(reasons, d.numbering_changed, "mesh numbering changed");
    append_reason(reasons, d.mesh_field_layout_changed, "mesh field layout changed");
    append_reason(reasons, d.label_changed, "mesh labels changed");
    append_reason(reasons, d.fe_space_changed, "FE space layout changed");
    append_reason(reasons, d.fe_dof_layout_changed, "FE DOF layout changed");
    append_reason(reasons, d.fe_constraint_layout_changed, "FE constraint layout changed");
    append_reason(reasons, d.fe_block_layout_changed, "FE block layout changed");
    d.reason = join_reasons(reasons);

    return d;
}

std::string GeometryTransactionDiagnostics::summary() const
{
    std::ostringstream os;
    os << "geometry_transaction state=" << to_string(state)
       << " geometry_use=" << to_string(geometry_use);
    if (!last_event.empty()) {
        os << " event=" << last_event;
    }
    if (current.mesh.valid) {
        os << " mesh_revisions={geometry:" << current.mesh.geometry
           << ", reference:" << current.mesh.reference_geometry
           << ", current:" << current.mesh.current_geometry
           << ", rebase:" << current.mesh.reference_rebase
           << ", topology:" << current.mesh.topology
           << ", ownership:" << current.mesh.ownership
           << ", numbering:" << current.mesh.numbering
           << ", field_layout:" << current.mesh.field_layout
           << ", labels:" << current.mesh.labels
           << ", active_configuration:" << current.mesh.active_configuration
           << '}';
    }
    os << " fe_layout={space:" << current.fe_layout.space
       << ", dof:" << current.fe_layout.dof_layout
       << ", constraints:" << current.fe_layout.constraint_layout
       << ", blocks:" << current.fe_layout.block_layout
       << '}';
    return os.str();
}

} // namespace systems
} // namespace FE
} // namespace svmp

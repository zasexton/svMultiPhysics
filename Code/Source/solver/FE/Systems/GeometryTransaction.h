/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_SYSTEMS_GEOMETRY_TRANSACTION_H
#define SVMP_FE_SYSTEMS_GEOMETRY_TRANSACTION_H

#include "Core/Types.h"

#include <cstdint>
#include <string>

namespace svmp {
namespace FE {
namespace systems {

struct FELayoutRevisionState {
    std::uint64_t space{0};
    std::uint64_t dof_layout{0};
    std::uint64_t constraint_layout{0};
    std::uint64_t block_layout{0};
};

enum class GeometryTransactionState : std::uint8_t {
    Committed,
    Trial,
    Accepted,
    RolledBack
};

enum class GeometryConfigurationUse : std::uint8_t {
    Reference,
    CommittedCurrent,
    TrialCurrent,
    AcceptedCurrent,
    RolledBackCurrent
};

struct MeshRevisionSnapshot {
    bool valid{false};
    std::uint64_t geometry{0};
    std::uint64_t reference_geometry{0};
    std::uint64_t current_geometry{0};
    std::uint64_t reference_rebase{0};
    std::uint64_t topology{0};
    std::uint64_t ownership{0};
    std::uint64_t numbering{0};
    std::uint64_t field_layout{0};
    std::uint64_t labels{0};
    std::uint64_t active_configuration{0};
};

struct OperatorRevisionSnapshot {
    bool valid{false};
    MeshRevisionSnapshot mesh{};
    FELayoutRevisionState fe_layout{};
    std::uint64_t system_layout_key{0};
    GeometryTransactionState geometry_state{GeometryTransactionState::Committed};
    GeometryConfigurationUse geometry_use{GeometryConfigurationUse::Reference};
};

struct OperatorInvalidationDecision {
    bool geometry_changed{false};
    bool reference_rebase_changed{false};
    bool topology_changed{false};
    bool ownership_changed{false};
    bool numbering_changed{false};
    bool mesh_field_layout_changed{false};
    bool label_changed{false};
    bool active_configuration_changed{false};
    bool fe_space_changed{false};
    bool fe_dof_layout_changed{false};
    bool fe_constraint_layout_changed{false};
    bool fe_block_layout_changed{false};
    bool transaction_state_changed{false};

    bool refresh_assembled_matrix{false};
    bool refresh_matrix_free_geometry{false};
    bool rebuild_dof_layout{false};
    bool rebuild_sparsity_pattern{false};
    bool rebuild_preconditioner{false};
    bool rebuild_multigrid_hierarchy{false};
    bool reject_lagged_jacobian{false};
    bool rebuild_interface_maps{false};
    bool rebuild_constraint_state{false};
    bool invalidate_restart_metadata{false};

    std::string reason{};

    [[nodiscard]] bool any_change() const noexcept;
    [[nodiscard]] bool requires_matrix_free_refresh() const noexcept;
    [[nodiscard]] bool requires_full_layout_rebuild() const noexcept;
};

struct GeometryTransactionDiagnostics {
    GeometryTransactionState state{GeometryTransactionState::Committed};
    GeometryConfigurationUse geometry_use{GeometryConfigurationUse::Reference};
    OperatorRevisionSnapshot started_from{};
    OperatorRevisionSnapshot current{};
    std::string last_event{};

    [[nodiscard]] std::string summary() const;
};

[[nodiscard]] const char* to_string(GeometryTransactionState state) noexcept;
[[nodiscard]] const char* to_string(GeometryConfigurationUse use) noexcept;

[[nodiscard]] OperatorInvalidationDecision decideOperatorInvalidation(
    const OperatorRevisionSnapshot& cached,
    const OperatorRevisionSnapshot& current,
    bool allow_lagged_jacobian_on_geometry_change = false);

} // namespace systems
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_SYSTEMS_GEOMETRY_TRANSACTION_H

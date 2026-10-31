#pragma once

/**
 * @file
 * @ingroup fe_level_set
 * @brief Algebraic rows for a frozen-topology wet-side velocity extension.
 *
 * The transported level set must not consume a state-dependent prescribed
 * coefficient in a monolithic solve: doing so drops the velocity-to-level-set
 * chain rule.  This kernel makes the extension velocity an algebraic unknown
 * E and installs the sparse graph equations used to construct it,
 *
 *   E_i - sum_j P_ij u_j - sum_k L_ik E_k = 0.
 *
 * The graph/topology and its phi-dependent regression weights are deliberately
 * refreshed between nonlinear solves and frozen during each solve.  For that
 * frozen map, both the residual and Jacobian below are exact.
 *
 * Rows and dependencies name mesh vertices, which include the edge, face and
 * interior nodes of high-order cells.  A vertex maps to the field's vertex
 * DOFs, and a high-order node to the edge, face or cell-interior DOFs whose
 * Lagrange node it is (FESystem::fieldDofsAtMeshVertices); both maps cover
 * the locally present ghost entities, so a row and its dependencies resolve
 * on the rank that owns the row's vertex.
 */

#include "Core/Types.h"
#include "Systems/GlobalKernel.h"

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace svmp::FE::level_set {

enum class VelocityExtensionDependencyField : std::uint8_t {
    SourceVelocity,
    ExtensionVelocity,
};

struct VelocityExtensionDependency {
    VelocityExtensionDependencyField field{
        VelocityExtensionDependencyField::ExtensionVelocity};
    // Rank-local native mesh vertex index (a corner or a high-order node).
    // EntityDofMap::getVertexDofs() and FESystem::fieldDofsAtMeshVertices()
    // accept this local identity, including a locally present ghost vertex;
    // it is deliberately not a globally unique vertex gid.
    GlobalIndex vertex{INVALID_GLOBAL_INDEX};
    int component{0};
    Real coefficient{0.0};
};

struct VelocityExtensionConstraintRow {
    // Rank-local native mesh vertex index.  Rows are emitted only by this
    // vertex's owner, while dependencies may name locally present ghosts.
    GlobalIndex vertex{INVALID_GLOBAL_INDEX};
    int component{0};
    std::vector<VelocityExtensionDependency> dependencies{};
};

struct LevelSetVelocityExtensionConstraintConfig {
    FieldId extension_field{INVALID_FIELD_ID};
    FieldId source_velocity_field{INVALID_FIELD_ID};
    int components{0};
    std::string operator_tag{"equations"};
};

class LevelSetVelocityExtensionConstraintKernel final
    : public systems::GlobalKernel {
public:
    explicit LevelSetVelocityExtensionConstraintKernel(
        LevelSetVelocityExtensionConstraintConfig config);

    [[nodiscard]] std::string name() const override
    {
        return "LevelSetVelocityExtensionConstraintKernel";
    }

    [[nodiscard]] FieldId extensionField() const noexcept
    {
        return config_.extension_field;
    }

    [[nodiscard]] FieldId sourceVelocityField() const noexcept
    {
        return config_.source_velocity_field;
    }

    [[nodiscard]] std::uint64_t frozenMapRevision() const noexcept
    {
        return frozen_map_revision_;
    }

    [[nodiscard]] bool hasFrozenMap() const noexcept { return map_is_valid_; }

    /** The installed rows, as setFrozenRows() normalized them. */
    [[nodiscard]] const std::vector<VelocityExtensionConstraintRow>&
    frozenRows() const noexcept
    {
        return rows_;
    }

    /** Replace the rank-local owned rows for the next nonlinear solve. */
    void setFrozenRows(std::vector<VelocityExtensionConstraintRow> rows,
                       std::uint64_t revision);

    /**
     * Invalidate a previously installed map before rebuilding its nonsmooth
     * active/cut support.  A failed rebuild must not leave an older interface
     * map available to a later assembly.
     */
    void invalidateFrozenMap() noexcept
    {
        rows_.clear();
        map_is_valid_ = false;
    }

    /** The installed map as it is: rows, revision and validity. */
    struct FrozenMapState {
        std::vector<VelocityExtensionConstraintRow> rows{};
        std::uint64_t revision{0};
        bool valid{false};
    };

    [[nodiscard]] FrozenMapState frozenMapState() const
    {
        return FrozenMapState{rows_, frozen_map_revision_, map_is_valid_};
    }

    /**
     * Reinstall a map taken with frozenMapState() exactly, for example after
     * a probe solve that refreshed the map at another state.
     */
    void restoreFrozenMapState(FrozenMapState state) noexcept
    {
        rows_ = std::move(state.rows);
        frozen_map_revision_ = state.revision;
        map_is_valid_ = state.valid;
    }

    void addSparsityCouplings(
        const systems::FESystem& system,
        sparsity::SparsityPattern& pattern) const override;

    [[nodiscard]] std::vector<analysis::ContributionDescriptor>
    analysisContributions() const override;

    [[nodiscard]] assembly::AssemblyResult assemble(
        const systems::FESystem& system,
        const systems::AssemblyRequest& request,
        const systems::SystemStateView& state,
        assembly::GlobalSystemView* matrix_out,
        assembly::GlobalSystemView* vector_out) override;

private:
    struct NodeDofTables;

    /** The field-local DOFs of the components at a mesh vertex. */
    [[nodiscard]] std::span<const GlobalIndex> vertexLocalDofs(
        const systems::FESystem& system,
        FieldId field,
        GlobalIndex vertex) const;

    LevelSetVelocityExtensionConstraintConfig config_{};
    std::vector<VelocityExtensionConstraintRow> rows_{};
    std::uint64_t frozen_map_revision_{0};
    bool map_is_valid_{false};
    // The DOFs at the mesh vertices without vertex DOFs (the edge, face and
    // interior nodes of high-order cells), built on first use for the
    // system's DOF layout.
    mutable std::shared_ptr<NodeDofTables> node_dofs_{};
};

[[nodiscard]] std::shared_ptr<LevelSetVelocityExtensionConstraintKernel>
findLevelSetVelocityExtensionConstraintKernel(
    const systems::FESystem& system,
    const std::string& operator_tag,
    FieldId extension_field);

} // namespace svmp::FE::level_set

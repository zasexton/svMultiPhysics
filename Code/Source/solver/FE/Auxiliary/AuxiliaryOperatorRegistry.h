#ifndef SVMP_FE_AUXILIARY_OPERATOR_REGISTRY_H
#define SVMP_FE_AUXILIARY_OPERATOR_REGISTRY_H

/**
 * @file AuxiliaryOperatorRegistry.h
 * @brief Registry for auxiliary operators and mixed system layout metadata.
 *
 * Owns all registered auxiliary operators and provides:
 * - Operator registration (aux-only, field-to-aux, aux-to-field, aux-to-aux)
 * - Coupling graph construction from registered operators
 * - Solver-facing block-layout metadata for mixed field/auxiliary systems
 * - Monolithic unknown layout composition
 *
 * ## Mixed system layout
 *
 * `FESystem` composes FE field unknown layouts and auxiliary-specific unknown
 * layouts into one mixed system layout for monolithic assembly and solves.
 * The `AuxiliaryUnknownLayout` struct describes the auxiliary contribution
 * to the mixed system, using auxiliary-specific indexing rather than
 * reusing FE field DOF maps.
 *
 * ## Partitioned vs Monolithic
 *
 * - `Partitioned` blocks are advanced independently via local steppers;
 *   they do NOT appear in the mixed system layout.
 * - `Monolithic` blocks contribute unknowns to the assembled system;
 *   they appear in `AuxiliaryUnknownLayout` and have entries in the
 *   coupling graph.
 */

#include "Core/Types.h"
#include "Core/FEException.h"

#include "Backends/Utils/BackendOptions.h"

#include "Auxiliary/AuxiliaryStateTypes.h"
#include "Auxiliary/AuxiliaryOperatorBuilder.h"
#include "Auxiliary/AuxiliaryCouplingGraph.h"
#include "Systems/SystemsExceptions.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace svmp {
namespace FE {
namespace systems {

/**
 * @brief Block classification for solver/preconditioner strategies.
 */
enum class AuxiliaryBlockRole : std::uint8_t {
    /// Standard block (no special solver treatment).
    Standard,

    /// Block suitable for Schur complement elimination.
    SchurEliminable,

    /// Block that should be treated as a constraint (Lagrange multiplier-like).
    Constraint,

    /// Block that needs special preconditioning (e.g., ill-conditioned).
    SpecialPrecondition
};

// ---------------------------------------------------------------------------
//  Auxiliary unknown layout for monolithic blocks
// ---------------------------------------------------------------------------

/**
 * @brief Layout metadata for one monolithic auxiliary block in the
 *        mixed system.
 */
struct AuxiliaryBlockUnknownLayout {
    /// Block name.
    std::string name{};

    /// Offset of this block's unknowns in the auxiliary segment
    /// of the mixed system vector.
    std::size_t offset{0};

    /// Number of unknowns contributed by this block
    /// (entity_count × component_stride for fixed-stride).
    std::size_t n_unknowns{0};

    /// Component stride per entity.
    int stride{0};

    /// Number of entities.
    std::size_t entity_count{0};

    /// The scope of this block.
    AuxiliaryStateScope scope{AuxiliaryStateScope::Global};

    /// Solver-oriented block metadata carried into the mixed layout.
    AuxiliaryBlockRole role{AuxiliaryBlockRole::Standard};
    backends::BlockRole backend_role{backends::BlockRole::AuxiliaryField};
    bool block_diagonal_suitable{true};
    bool schur_eliminable{false};
    std::string schur_complement_partner{};
    std::vector<std::vector<int>> constraint_groups{};

    /// Backend-facing assembly/ownership contract.  Current monolithic
    /// auxiliary rows are assembled through bordered/reduced storage unless a
    /// future scope-specific path explicitly upgrades them to native backend
    /// rows with concrete ownership.
    backends::MixedBlockAssemblyMode assembly_mode{
        backends::MixedBlockAssemblyMode::BorderedReduced};
    backends::MixedRowOwnershipPolicy row_ownership{
        backends::MixedRowOwnershipPolicy::Unspecified};
    int single_owner_rank{-1};
};

/**
 * @brief Combined unknown layout for all monolithic auxiliary blocks.
 *
 * Describes how auxiliary unknowns are arranged in the auxiliary
 * segment of the mixed system vector.
 */
struct AuxiliaryUnknownLayout {
    /// Per-block layouts in registration order.
    std::vector<AuxiliaryBlockUnknownLayout> blocks{};

    /// Total number of auxiliary unknowns across all monolithic blocks.
    std::size_t total_aux_unknowns{0};

    /// Offset where auxiliary unknowns start in the mixed system vector.
    /// (After all FE field DOFs.)
    std::size_t mixed_system_offset{0};
};

/**
 * @brief Solver-facing metadata for the full mixed field + auxiliary system.
 */
struct MixedSystemLayout {
    /// Total FE field unknowns.
    std::size_t n_field_unknowns{0};

    /// Total auxiliary unknowns (monolithic only).
    std::size_t n_aux_unknowns{0};

    /// Total mixed system size.
    std::size_t total_unknowns{0};

    /// Auxiliary unknown layout details.
    AuxiliaryUnknownLayout aux_layout{};
};

// ---------------------------------------------------------------------------
//  Solver-facing block metadata
// ---------------------------------------------------------------------------

/**
 * @brief Solver-facing metadata for one monolithic auxiliary block.
 */
struct AuxiliaryBlockSolverMetadata {
    std::string block_name{};
    AuxiliaryBlockRole role{AuxiliaryBlockRole::Standard};

    /// Estimated condition number (0 = unknown).
    double estimated_condition{0.0};

    /// Estimated coupling density (fraction of nonzeros, 0-1).
    double coupling_density{0.0};

    /// Whether this block is suitable for block-diagonal preconditioning.
    bool block_diagonal_suitable{true};

    /// Whether this block can be eliminated via Schur complement.
    bool schur_eliminable{false};

    /// Name of the complementary block for Schur elimination.
    std::string schur_complement_partner{};
};

/**
 * @brief Sparsity and coupling diagnostics for the auxiliary system.
 */
struct AuxiliaryCouplingDiagnostics {
    /// Total number of coupling edges in the graph.
    std::size_t n_coupling_edges{0};

    /// Number of field-to-auxiliary edges.
    std::size_t n_field_to_aux{0};

    /// Number of auxiliary-to-field edges.
    std::size_t n_aux_to_field{0};

    /// Number of auxiliary-to-auxiliary edges.
    std::size_t n_aux_to_aux{0};

    /// Maximum coupling degree of any auxiliary block.
    std::size_t max_coupling_degree{0};

    /// Whether the coupling graph is acyclic.
    bool is_acyclic{true};

    /// Diagnostic messages.
    std::vector<std::string> messages{};
};

// ---------------------------------------------------------------------------
//  Registry
// ---------------------------------------------------------------------------

/**
 * @brief Registry for auxiliary operators and mixed system composition.
 */
class AuxiliaryOperatorRegistry {
public:
    AuxiliaryOperatorRegistry() = default;

    // -----------------------------------------------------------------
    //  Operator registration
    // -----------------------------------------------------------------

    /**
     * @brief Register an auxiliary operator from a built descriptor.
     *
     * Also updates the coupling graph with the appropriate edge.
     */
    void registerOperator(const AuxiliaryOperatorDescriptor& desc);

    /// Number of registered operators.
    [[nodiscard]] std::size_t operatorCount() const noexcept
    {
        return operators_.size();
    }

    /// Whether an operator with the given name exists.
    [[nodiscard]] bool hasOperator(std::string_view name) const noexcept;

    /// Get operator descriptor by name.
    [[nodiscard]] const AuxiliaryOperatorDescriptor& getOperator(
        std::string_view name) const;

    /// All registered operator names.
    [[nodiscard]] std::vector<std::string> operatorNames() const;

    // -----------------------------------------------------------------
    //  Coupling graph
    // -----------------------------------------------------------------

    /// Access the coupling graph (built from registered operators).
    [[nodiscard]] const AuxiliaryCouplingGraph& couplingGraph() const noexcept
    {
        return coupling_graph_;
    }

    // -----------------------------------------------------------------
    //  Monolithic unknown registration
    // -----------------------------------------------------------------

    /**
     * @brief Register a monolithic auxiliary block as contributing
     *        unknowns to the mixed system.
     *
     * @param name          Block name.
     * @param entity_count  Number of entities.
     * @param stride        Components per entity.
     * @param scope         Block scope.
     */
    void registerMonolithicUnknowns(std::string_view name,
                                    std::size_t entity_count,
                                    int stride,
                                    AuxiliaryStateScope scope,
                                    const AuxiliaryBlockSolverMetadata* solver_meta = nullptr,
                                    std::vector<std::vector<int>> constraint_groups = {});

    // -----------------------------------------------------------------
    //  Layout composition
    // -----------------------------------------------------------------

    /**
     * @brief Finalize the auxiliary unknown layout.
     *
     * Computes offsets within the auxiliary segment.
     * Must be called after all monolithic unknowns are registered.
     */
    void finalizeLayout();

    /**
     * @brief Compose the mixed system layout given FE field unknown count.
     *
     * @param n_field_unknowns Total number of FE field DOFs.
     * @return Full mixed system layout.
     */
    [[nodiscard]] MixedSystemLayout composeMixedLayout(
        std::size_t n_field_unknowns) const;

    /// Get the finalized auxiliary unknown layout.
    [[nodiscard]] const AuxiliaryUnknownLayout& auxiliaryLayout() const noexcept
    {
        return aux_layout_;
    }

    /// Whether layout has been finalized.
    [[nodiscard]] bool isLayoutFinalized() const noexcept
    {
        return layout_finalized_;
    }

    // -----------------------------------------------------------------
    //  Solver-facing metadata
    // -----------------------------------------------------------------

    /**
     * @brief Set solver metadata for a monolithic block.
     */
    void setBlockSolverMetadata(std::string_view block_name,
                                 AuxiliaryBlockSolverMetadata meta);

    /**
     * @brief Get solver metadata for a block (empty name returns default).
     */
    [[nodiscard]] const AuxiliaryBlockSolverMetadata* getBlockSolverMetadata(
        std::string_view block_name) const;

    /// Get layout block metadata by name after registration.
    [[nodiscard]] const AuxiliaryBlockUnknownLayout* findLayoutBlock(
        std::string_view block_name) const noexcept;

    [[nodiscard]] std::vector<std::string> constraintLikeBlocks() const;
    [[nodiscard]] std::vector<std::string> schurEliminableBlocks() const;
    [[nodiscard]] std::vector<std::string> specialPreconditionBlocks() const;

    /**
     * @brief Compute coupling diagnostics from the current graph.
     */
    [[nodiscard]] AuxiliaryCouplingDiagnostics computeCouplingDiagnostics() const;

    // -----------------------------------------------------------------
    //  Lifecycle
    // -----------------------------------------------------------------

    void clear();

private:
    std::vector<AuxiliaryOperatorDescriptor> operators_{};
    std::unordered_map<std::string, std::size_t> op_name_to_index_{};

    AuxiliaryCouplingGraph coupling_graph_{};

    AuxiliaryUnknownLayout aux_layout_{};
    bool layout_finalized_{false};

    std::unordered_map<std::string, AuxiliaryBlockSolverMetadata> block_solver_meta_{};
};

} // namespace systems
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_AUXILIARY_OPERATOR_REGISTRY_H

/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_BACKENDS_BACKEND_OPTIONS_H
#define SVMP_FE_BACKENDS_BACKEND_OPTIONS_H

#include "Core/Types.h"
#include "Backends/Interfaces/BackendKind.h"

#include <algorithm>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace svmp {
namespace FE {
namespace backends {

enum class SolverMethod : std::uint8_t {
    Direct,
    CG,
    BiCGSTAB,
    GMRES,
    PGMRES,
    FGMRES,
    BlockSchur
};

enum class PreconditionerType : std::uint8_t {
    None,
    Diagonal,
    ILU,
    AMG,
    RowColumnScaling,
    FieldSplit
};

enum class FieldSplitKind : std::uint8_t {
    Auto,
    Additive,
    Multiplicative,
    Schur
};

enum class FsilsResidualCheckPolicy : std::uint8_t {
    Always,
    RetryOnly,
    DebugOnly
};

enum class FsilsBlockSchurSchurPreconditioner : std::uint8_t {
    Auto,
    DiagL,
    BlockDiagL,
    ILUL,
    AlgebraicSchur
};

enum class FsilsBlockSchurMomentumApproximation : std::uint8_t {
    Auto,
    DiagK,
    BlockDiagK,
    ILUK,
    ASM
};

struct FieldSplitOptions {
    /// `Auto` lets the backend choose additive / multiplicative / Schur mode
    /// from mixed-layout metadata when available.
    FieldSplitKind kind{FieldSplitKind::Auto};
    std::vector<std::string> split_names{}; // Optional, defaults to field0/field1/...
};

/**
 * @brief Generic solver role that a block can play
 *
 * Replaces physics-specific names like "momentum" and "constraint" with
 * generic roles that apply to any saddle-point or multi-field formulation.
 * The mapping from physics-specific names to roles is done by backend-specific
 * adapters, not by the FE public API.
 */
enum class BlockRole : std::uint8_t {
    /// No special role; treated as a generic field block
    Generic = 0,

    /// Primary field in a saddle-point system (e.g., velocity, displacement)
    PrimaryField,

    /// Constraint/multiplier field in a saddle-point system (e.g., pressure, Lagrange multiplier)
    ConstraintField,

    /// Auxiliary field (e.g., temperature in thermo-mechanics, damage variable)
    AuxiliaryField,
};

[[nodiscard]] inline std::string_view blockRoleToString(BlockRole role) noexcept {
    switch (role) {
        case BlockRole::Generic: return "Generic";
        case BlockRole::PrimaryField: return "PrimaryField";
        case BlockRole::ConstraintField: return "ConstraintField";
        case BlockRole::AuxiliaryField: return "AuxiliaryField";
    }
    return "Unknown";
}

/// Describes a single DOF block within a multi-field system.
struct BlockDescriptor {
    std::string name;             ///< Field name (e.g., "velocity", "pressure", "temperature")
    int start_component{0};       ///< First per-node component index for this block
    int n_components{0};          ///< Number of per-node components in this block
    BlockRole role{BlockRole::Generic};  ///< Solver role for this block
};

/// Describes the per-node DOF block structure of a multi-field system.
///
/// Used by block-aware solvers and preconditioners (block Jacobi, field-split,
/// Schur complement, stage scaling) without requiring physics-specific knowledge.
///
/// The blocks are ordered by start_component and must partition [0, dof_per_node)
/// without gaps or overlaps.
struct BlockLayout {
    std::vector<BlockDescriptor> blocks{};

    /// Optional: indices into blocks[] identifying the saddle-point pair.
    /// Only meaningful for BlockSchur / Schur-complement solvers.
    /// momentum_block: field-A (e.g., velocity) — rows scaled, Schur complement field-A solve
    /// constraint_block: field-B (e.g., pressure) — cols scaled, Schur complement field-B solve
    std::optional<int> momentum_block{};
    std::optional<int> constraint_block{};

    /// Convenience: total DOF per node implied by blocks.
    [[nodiscard]] int totalComponents() const noexcept {
        int total = 0;
        for (const auto& b : blocks) { total += b.n_components; }
        return total;
    }

    /// Look up block by name (returns nullptr if not found).
    [[nodiscard]] const BlockDescriptor* findBlock(std::string_view name) const noexcept {
        for (const auto& b : blocks) {
            if (b.name == name) return &b;
        }
        return nullptr;
    }

    /// Look up block index by name (returns nullopt if not found).
    [[nodiscard]] std::optional<int> findBlockIndex(std::string_view name) const noexcept {
        for (std::size_t i = 0; i < blocks.size(); ++i) {
            if (blocks[i].name == name) {
                return static_cast<int>(i);
            }
        }
        return std::nullopt;
    }

    /// Look up the first block with a given role (returns nullptr if not found).
    [[nodiscard]] const BlockDescriptor* findBlockByRole(BlockRole role) const noexcept {
        for (const auto& b : blocks) {
            if (b.role == role) return &b;
        }
        return nullptr;
    }

    /// Find the primary field block (saddle-point field-A). Falls back to momentum_block.
    [[nodiscard]] const BlockDescriptor* primaryFieldBlock() const noexcept {
        if (auto* p = findBlockByRole(BlockRole::PrimaryField)) return p;
        if (momentum_block && *momentum_block >= 0 && *momentum_block < static_cast<int>(blocks.size()))
            return &blocks[static_cast<std::size_t>(*momentum_block)];
        return nullptr;
    }

    /// Find the constraint field block (saddle-point field-B). Falls back to constraint_block.
    [[nodiscard]] const BlockDescriptor* constraintFieldBlock() const noexcept {
        if (auto* p = findBlockByRole(BlockRole::ConstraintField)) return p;
        if (constraint_block && *constraint_block >= 0 && *constraint_block < static_cast<int>(blocks.size()))
            return &blocks[static_cast<std::size_t>(*constraint_block)];
        return nullptr;
    }

    /// Check if saddle-point annotation is present and valid.
    [[nodiscard]] bool hasSaddlePoint() const noexcept {
        return momentum_block.has_value() && constraint_block.has_value()
            && *momentum_block >= 0 && *momentum_block < static_cast<int>(blocks.size())
            && *constraint_block >= 0 && *constraint_block < static_cast<int>(blocks.size());
    }
};

enum class MixedBlockKind : std::uint8_t {
    Field = 0,
    Auxiliary
};

/// Describes one absolute-offset block in a mixed field + auxiliary system.
struct MixedBlockDescriptor {
    std::string name{};
    GlobalIndex offset{0};
    GlobalIndex size{0};
    BlockRole role{BlockRole::Generic};
    MixedBlockKind kind{MixedBlockKind::Field};
    bool block_diagonal_suitable{true};
    bool special_precondition{false};
    bool schur_eliminable{false};
    std::string schur_complement_partner{};
};

/// Describes a mixed global block layout using absolute offsets and sizes.
struct MixedBlockLayout {
    GlobalIndex field_unknowns{0};
    GlobalIndex auxiliary_unknowns{0};
    GlobalIndex total_unknowns{0};
    std::vector<MixedBlockDescriptor> blocks{};

    /// Optional indices identifying a saddle-point pair in blocks[].
    std::optional<int> primary_block{};
    std::optional<int> constraint_block{};

    [[nodiscard]] const MixedBlockDescriptor* findBlock(std::string_view name) const noexcept {
        for (const auto& b : blocks) {
            if (b.name == name) {
                return &b;
            }
        }
        return nullptr;
    }

    [[nodiscard]] std::optional<int> findBlockIndex(std::string_view name) const noexcept {
        for (std::size_t i = 0; i < blocks.size(); ++i) {
            if (blocks[i].name == name) {
                return static_cast<int>(i);
            }
        }
        return std::nullopt;
    }

    [[nodiscard]] const MixedBlockDescriptor* findBlockByRole(BlockRole role) const noexcept {
        for (const auto& b : blocks) {
            if (b.role == role) {
                return &b;
            }
        }
        return nullptr;
    }

    [[nodiscard]] const MixedBlockDescriptor* findBlockByExtent(GlobalIndex offset,
                                                                GlobalIndex size) const noexcept {
        for (const auto& b : blocks) {
            if (b.offset == offset && b.size == size) {
                return &b;
            }
        }
        return nullptr;
    }

    [[nodiscard]] const MixedBlockDescriptor* primaryFieldBlock() const noexcept {
        if (auto* p = findBlockByRole(BlockRole::PrimaryField)) return p;
        if (primary_block && *primary_block >= 0 && *primary_block < static_cast<int>(blocks.size())) {
            return &blocks[static_cast<std::size_t>(*primary_block)];
        }
        return nullptr;
    }

    [[nodiscard]] const MixedBlockDescriptor* constraintFieldBlock() const noexcept {
        if (auto* p = findBlockByRole(BlockRole::ConstraintField)) return p;
        if (constraint_block && *constraint_block >= 0 &&
            *constraint_block < static_cast<int>(blocks.size())) {
            return &blocks[static_cast<std::size_t>(*constraint_block)];
        }
        return nullptr;
    }

    [[nodiscard]] bool hasSaddlePoint() const noexcept {
        return primary_block.has_value() && constraint_block.has_value()
            && *primary_block >= 0 && *primary_block < static_cast<int>(blocks.size())
            && *constraint_block >= 0 && *constraint_block < static_cast<int>(blocks.size());
    }

    [[nodiscard]] bool matchesTotalUnknowns(GlobalIndex n) const noexcept {
        return total_unknowns == n;
    }

    [[nodiscard]] std::size_t countBlocksByRole(BlockRole role) const noexcept {
        std::size_t count = 0;
        for (const auto& b : blocks) {
            if (b.role == role) {
                ++count;
            }
        }
        return count;
    }

    [[nodiscard]] bool hasSpecialPreconditionBlocks() const noexcept {
        return std::any_of(blocks.begin(), blocks.end(), [](const auto& block) {
            return block.special_precondition;
        });
    }

    [[nodiscard]] bool hasSchurEliminableBlocks() const noexcept {
        return std::any_of(blocks.begin(), blocks.end(), [](const auto& block) {
            return block.schur_eliminable;
        });
    }
};

/**
 * @brief Time integration metadata for a field
 *
 * Describes how a field participates in time integration by derivative order
 * rather than physics-specific naming (displacement/velocity/acceleration).
 *
 * - First-order systems: fields have max_derivative_order = 1 (e.g., heat, Stokes)
 * - Second-order systems: fields have max_derivative_order = 2 (e.g., elastodynamics)
 * - Mixed-order systems: different fields may have different orders
 */
struct FieldTimeMetadata {
    FieldId field{INVALID_FIELD_ID};
    std::string name;

    /// Maximum time derivative order this field undergoes (0 = steady, 1 = first-order, 2 = second-order)
    int max_derivative_order{0};

    /// Whether this field needs time-history storage beyond the current and previous step
    int history_depth{1};

    /// Optional: time integration scheme override for this field (empty = use global scheme)
    std::string scheme_override{};
};

/**
 * @brief Time integration descriptor for a multi-field system
 *
 * Groups per-field time metadata so the time integrator layer can allocate
 * scheme-specific state storage without physics-specific naming.
 */
struct TimeIntegrationDescriptor {
    std::vector<FieldTimeMetadata> fields{};

    /// Global time integration scheme (e.g., "BDF2", "Newmark", "GenAlpha")
    std::string global_scheme{};

    /// Maximum derivative order across all fields
    [[nodiscard]] int maxDerivativeOrder() const noexcept {
        int mx = 0;
        for (const auto& f : fields) mx = std::max(mx, f.max_derivative_order);
        return mx;
    }

    /// Maximum history depth across all fields
    [[nodiscard]] int maxHistoryDepth() const noexcept {
        int mx = 0;
        for (const auto& f : fields) mx = std::max(mx, f.history_depth);
        return mx;
    }
};

struct SolverOptions {
    SolverMethod method{SolverMethod::Direct};
    PreconditionerType preconditioner{PreconditionerType::None};

    Real rel_tol{1e-10};
    Real abs_tol{0.0};
    int max_iter{1000};
    int krylov_dim{0}; // Optional (backend-specific; e.g., FSILS GMRES RI.sD)
    bool use_initial_guess{false};

    FieldSplitOptions fieldsplit{};

    /// Optional block layout describing the per-node DOF structure of a multi-field system.
    /// Populated from FieldDofMap metadata; used by block-aware solvers (BlockSchur, FieldSplit).
    std::optional<BlockLayout> block_layout{};

    /// Optional absolute-offset mixed layout for field + auxiliary systems.
    /// This can describe blocks that are not representable as per-node components.
    std::optional<MixedBlockLayout> mixed_block_layout{};

    /// Explicit saddle-point block names (optional). If set, used to identify momentum/constraint
    /// blocks by name instead of the auto-detection heuristic (first multi-component = momentum,
    /// first single-component = constraint). Set from solver XML configuration.
    std::string momentum_block_name{};    ///< e.g., "velocity" or "displacement"
    std::string constraint_block_name{};  ///< e.g., "pressure"

    /// Generic role-to-name mapping (takes precedence over momentum/constraint names when set).
    /// Maps BlockRole -> block name. Used by backend adapters to resolve role-based queries.
    std::vector<std::pair<BlockRole, std::string>> block_role_names{};

    [[nodiscard]] std::string_view resolveBlockNameForRole(BlockRole role) const noexcept {
        for (const auto& [mapped_role, name] : block_role_names) {
            if (mapped_role == role && !name.empty()) {
                return name;
            }
        }

        if (role == BlockRole::PrimaryField && !momentum_block_name.empty()) {
            return momentum_block_name;
        }
        if (role == BlockRole::ConstraintField && !constraint_block_name.empty()) {
            return constraint_block_name;
        }

        if (mixed_block_layout.has_value()) {
            if (const auto* blk = mixed_block_layout->findBlockByRole(role)) {
                return blk->name;
            }
            if (role == BlockRole::PrimaryField) {
                if (const auto* blk = mixed_block_layout->primaryFieldBlock()) {
                    return blk->name;
                }
            } else if (role == BlockRole::ConstraintField) {
                if (const auto* blk = mixed_block_layout->constraintFieldBlock()) {
                    return blk->name;
                }
            }
        }

        if (block_layout.has_value()) {
            if (const auto* blk = block_layout->findBlockByRole(role)) {
                return blk->name;
            }
            if (role == BlockRole::PrimaryField) {
                if (const auto* blk = block_layout->primaryFieldBlock()) {
                    return blk->name;
                }
            } else if (role == BlockRole::ConstraintField) {
                if (const auto* blk = block_layout->constraintFieldBlock()) {
                    return blk->name;
                }
            }
        }

        return {};
    }

    /// Optional time integration descriptor for multi-field systems.
    std::optional<TimeIntegrationDescriptor> time_integration{};

    // Backend-specific pass-through (optional).
    // PETSc: used with KSPSetOptionsPrefix()/KSPSetFromOptions().
    // Example (CLI): -my_solver_ksp_type gmres
    // Example (prefix): petsc_options_prefix="my_solver_"
    std::string petsc_options_prefix{};

    // Trilinos: optional Teuchos::ParameterList XML file for solver configuration.
    std::string trilinos_xml_file{};

    // FSILS: best-effort row/column scaling toggle.
    bool fsils_use_rcs{false};

    // FSILS: post-solve true residual validation policy.
    // - Always: always compute a full residual with the original operator.
    // - RetryOnly: skip the extra A*x on clean first-pass success; validate recovery paths.
    // - DebugOnly: validate only in debug/trace mode, plus any recovery path.
    FsilsResidualCheckPolicy fsils_residual_check_policy{FsilsResidualCheckPolicy::RetryOnly};

    // FSILS: BlockSchur sub-solver parameters.
    // GM: field-A (momentum) solve, CG: Schur complement (constraint) solve.
    std::optional<int> fsils_blockschur_gm_max_iter{};
    std::optional<int> fsils_blockschur_cg_max_iter{};
    std::optional<Real> fsils_blockschur_gm_rel_tol{};
    std::optional<Real> fsils_blockschur_cg_rel_tol{};
    /// `Auto` resolves to a concrete Schur-side preconditioner from auxiliary
    /// mixed-layout metadata when available.
    FsilsBlockSchurSchurPreconditioner fsils_blockschur_schur_preconditioner{
        FsilsBlockSchurSchurPreconditioner::Auto};
    /// `Auto` resolves to a concrete momentum approximation from auxiliary
    /// mixed-layout metadata when available.
    FsilsBlockSchurMomentumApproximation fsils_blockschur_momentum_approximation{
        FsilsBlockSchurMomentumApproximation::Auto};

    // Backend-specific pass-through key/value list (optional).
    // - PETSc: key maps to an option name (with or without '-' prefix).
    // - Trilinos: key maps to a Teuchos::ParameterList entry.
    std::vector<std::pair<std::string, std::string>> passthrough{};
};

struct SolverReport {
    bool converged{false};
    int iterations{0};
    Real initial_residual_norm{0.0};
    Real final_residual_norm{0.0};
    Real relative_residual{0.0};
    double setup_time_seconds{0.0};
    double validation_time_seconds{0.0};
    double collective_time_seconds{0.0};
    std::uint64_t collective_calls{0};
    std::uint64_t collective_words{0};
    int blockschur_outer_iterations{0};
    std::uint64_t blockschur_collective_calls_max_per_outer{0};
    double blockschur_collective_time_max_per_outer{0.0};
    int blockschur_momentum_solve_calls{0};
    int blockschur_momentum_iterations{0};
    int blockschur_momentum_restart_cycles{0};
    double blockschur_momentum_solve_time_seconds{0.0};
    std::uint64_t blockschur_momentum_collective_calls{0};
    std::uint64_t blockschur_momentum_collective_words{0};
    double blockschur_momentum_collective_time_seconds{0.0};
    int blockschur_schur_solve_calls{0};
    int blockschur_schur_iterations{0};
    double blockschur_schur_setup_time_seconds{0.0};
    double blockschur_schur_solve_time_seconds{0.0};
    std::uint64_t blockschur_schur_collective_calls{0};
    std::uint64_t blockschur_schur_collective_words{0};
    double blockschur_schur_collective_time_seconds{0.0};
    std::string message{};
};

[[nodiscard]] std::string_view solverMethodToString(SolverMethod m) noexcept;
[[nodiscard]] std::string_view preconditionerToString(PreconditionerType pc) noexcept;
[[nodiscard]] std::string_view fieldSplitKindToString(FieldSplitKind kind) noexcept;
[[nodiscard]] std::string_view
fsilsBlockSchurPreconditionerToString(FsilsBlockSchurSchurPreconditioner pc) noexcept;
[[nodiscard]] std::string_view
fsilsBlockSchurMomentumApproximationToString(FsilsBlockSchurMomentumApproximation approx) noexcept;

/**
 * @brief Apply backend-specific policy defaults derived from block metadata.
 *
 * This uses propagated field/auxiliary mixed-layout metadata to:
 * - materialize stable role-to-name mappings,
 * - choose concrete block-preconditioner variants from `Auto` settings,
 * - optionally upgrade PETSc block solves to `FieldSplit` when a block operator
 *   is available and auxiliary metadata marks a block for special treatment.
 */
[[nodiscard]] SolverOptions normalizeSolverOptionsForBackend(
    const SolverOptions& options,
    BackendKind backend,
    bool block_operator_available = false);

} // namespace backends
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_BACKENDS_BACKEND_OPTIONS_H

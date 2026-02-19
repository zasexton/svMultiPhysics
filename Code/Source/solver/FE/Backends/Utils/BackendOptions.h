/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_BACKENDS_BACKEND_OPTIONS_H
#define SVMP_FE_BACKENDS_BACKEND_OPTIONS_H

#include "Core/Types.h"

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
    Additive,
    Multiplicative,
    Schur
};

struct FieldSplitOptions {
    FieldSplitKind kind{FieldSplitKind::Additive};
    std::vector<std::string> split_names{}; // Optional, defaults to field0/field1/...
};

/// Describes a single DOF block within a multi-field system.
struct BlockDescriptor {
    std::string name;             ///< Field name (e.g., "velocity", "pressure", "temperature")
    int start_component{0};       ///< First per-node component index for this block
    int n_components{0};          ///< Number of per-node components in this block
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

    /// Check if saddle-point annotation is present and valid.
    [[nodiscard]] bool hasSaddlePoint() const noexcept {
        return momentum_block.has_value() && constraint_block.has_value()
            && *momentum_block >= 0 && *momentum_block < static_cast<int>(blocks.size())
            && *constraint_block >= 0 && *constraint_block < static_cast<int>(blocks.size());
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

    // Backend-specific pass-through (optional).
    // PETSc: used with KSPSetOptionsPrefix()/KSPSetFromOptions().
    // Example (CLI): -my_solver_ksp_type gmres
    // Example (prefix): petsc_options_prefix="my_solver_"
    std::string petsc_options_prefix{};

    // Trilinos: optional Teuchos::ParameterList XML file for solver configuration.
    std::string trilinos_xml_file{};

    // FSILS: best-effort row/column scaling toggle.
    bool fsils_use_rcs{false};

    // FSILS: BlockSchur sub-solver parameters.
    // GM: field-A (momentum) solve, CG: Schur complement (constraint) solve.
    std::optional<int> fsils_blockschur_gm_max_iter{};
    std::optional<int> fsils_blockschur_cg_max_iter{};
    std::optional<Real> fsils_blockschur_gm_rel_tol{};
    std::optional<Real> fsils_blockschur_cg_rel_tol{};

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
    std::string message{};
};

[[nodiscard]] std::string_view solverMethodToString(SolverMethod m) noexcept;
[[nodiscard]] std::string_view preconditionerToString(PreconditionerType pc) noexcept;
[[nodiscard]] std::string_view fieldSplitKindToString(FieldSplitKind kind) noexcept;

} // namespace backends
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_BACKENDS_BACKEND_OPTIONS_H

/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Backends/Utils/BackendOptions.h"

#include <algorithm>

namespace svmp {
namespace FE {
namespace backends {

std::string_view solverMethodToString(SolverMethod m) noexcept
{
    switch (m) {
        case SolverMethod::Direct: return "direct";
        case SolverMethod::CG: return "cg";
        case SolverMethod::BiCGSTAB: return "bicgstab";
        case SolverMethod::GMRES: return "gmres";
        case SolverMethod::PGMRES: return "pgmres";
        case SolverMethod::FGMRES: return "fgmres";
        case SolverMethod::BlockSchur: return "block-schur";
        default: return "unknown";
    }
}

std::string_view preconditionerToString(PreconditionerType pc) noexcept
{
    switch (pc) {
        case PreconditionerType::None: return "none";
        case PreconditionerType::Diagonal: return "diagonal";
        case PreconditionerType::ILU: return "ilu";
        case PreconditionerType::AMG: return "amg";
        case PreconditionerType::RowColumnScaling: return "row-column-scaling";
        case PreconditionerType::FieldSplit: return "field-split";
        default: return "unknown";
    }
}

std::string_view fieldSplitKindToString(FieldSplitKind kind) noexcept
{
    switch (kind) {
        case FieldSplitKind::Auto: return "auto";
        case FieldSplitKind::Additive: return "additive";
        case FieldSplitKind::Multiplicative: return "multiplicative";
        case FieldSplitKind::Schur: return "schur";
        default: return "unknown";
    }
}

std::string_view fsilsBlockSchurPreconditionerToString(FsilsBlockSchurSchurPreconditioner pc) noexcept
{
    switch (pc) {
        case FsilsBlockSchurSchurPreconditioner::Auto: return "auto";
        case FsilsBlockSchurSchurPreconditioner::DiagL: return "diag-l";
        case FsilsBlockSchurSchurPreconditioner::BlockDiagL: return "blockdiag-l";
        case FsilsBlockSchurSchurPreconditioner::ILUL: return "ilu-l";
        case FsilsBlockSchurSchurPreconditioner::AlgebraicSchur: return "algebraic-shat";
        default: return "unknown";
    }
}

std::string_view
fsilsBlockSchurMomentumApproximationToString(FsilsBlockSchurMomentumApproximation approx) noexcept
{
    switch (approx) {
        case FsilsBlockSchurMomentumApproximation::Auto: return "auto";
        case FsilsBlockSchurMomentumApproximation::DiagK: return "diag-k";
        case FsilsBlockSchurMomentumApproximation::BlockDiagK: return "blockdiag-k";
        case FsilsBlockSchurMomentumApproximation::ILUK: return "ilu-k";
        case FsilsBlockSchurMomentumApproximation::ASM: return "asm-k";
        default: return "unknown";
    }
}

namespace {

[[nodiscard]] bool hasExplicitRoleName(const SolverOptions& options,
                                       BlockRole role) noexcept
{
    return std::any_of(options.block_role_names.begin(),
                       options.block_role_names.end(),
                       [&](const auto& entry) {
                           return entry.first == role && !entry.second.empty();
                       });
}

template <class Descriptor>
[[nodiscard]] const Descriptor* uniqueBlockByRole(const std::vector<Descriptor>& blocks,
                                                  BlockRole role) noexcept
{
    const Descriptor* match = nullptr;
    for (const auto& block : blocks) {
        if (block.role != role) {
            continue;
        }
        if (match != nullptr) {
            return nullptr;
        }
        match = &block;
    }
    return match;
}

void addMetadataRoleMappings(SolverOptions& options)
{
    for (const auto role : {BlockRole::PrimaryField,
                            BlockRole::ConstraintField,
                            BlockRole::AuxiliaryField}) {
        if (hasExplicitRoleName(options, role)) {
            continue;
        }

        std::string_view name{};
        if (options.mixed_block_layout.has_value()) {
            if (const auto* block = uniqueBlockByRole(options.mixed_block_layout->blocks, role)) {
                name = block->name;
            }
        }

        if (name.empty() && options.block_layout.has_value()) {
            if (const auto* block = uniqueBlockByRole(options.block_layout->blocks, role)) {
                name = block->name;
            }
        }

        if (!name.empty()) {
            options.block_role_names.emplace_back(role, std::string(name));
        }
    }

    if (options.momentum_block_name.empty()) {
        const auto name = options.resolveBlockNameForRole(BlockRole::PrimaryField);
        if (!name.empty()) {
            options.momentum_block_name = std::string(name);
        }
    }

    if (options.constraint_block_name.empty()) {
        const auto name = options.resolveBlockNameForRole(BlockRole::ConstraintField);
        if (!name.empty()) {
            options.constraint_block_name = std::string(name);
        }
    }
}

[[nodiscard]] std::size_t inferredBlockCount(const SolverOptions& options) noexcept
{
    if (options.mixed_block_layout.has_value()) {
        return options.mixed_block_layout->blocks.size();
    }
    if (options.block_layout.has_value()) {
        return options.block_layout->blocks.size();
    }
    return 0;
}

[[nodiscard]] bool hasSaddlePointMetadata(const SolverOptions& options) noexcept
{
    if (options.mixed_block_layout.has_value() && options.mixed_block_layout->hasSaddlePoint()) {
        return true;
    }
    return options.block_layout.has_value() && options.block_layout->hasSaddlePoint();
}

[[nodiscard]] bool hasSpecialPreconditionBlocks(const SolverOptions& options) noexcept
{
    return options.mixed_block_layout.has_value() &&
           options.mixed_block_layout->hasSpecialPreconditionBlocks();
}

[[nodiscard]] bool hasSchurEliminableBlocks(const SolverOptions& options) noexcept
{
    return options.mixed_block_layout.has_value() &&
           options.mixed_block_layout->hasSchurEliminableBlocks();
}

[[nodiscard]] bool hasNonBlockDiagonalAuxiliaryBlocks(const SolverOptions& options) noexcept
{
    if (!options.mixed_block_layout.has_value()) {
        return false;
    }

    return std::any_of(options.mixed_block_layout->blocks.begin(),
                       options.mixed_block_layout->blocks.end(),
                       [](const auto& block) {
                           return block.kind == MixedBlockKind::Auxiliary &&
                                  !block.block_diagonal_suitable;
                       });
}

[[nodiscard]] int primaryComponents(const SolverOptions& options) noexcept
{
    if (!options.block_layout.has_value()) {
        return 0;
    }
    if (const auto* block = options.block_layout->primaryFieldBlock()) {
        return block->n_components;
    }
    return 0;
}

[[nodiscard]] int constraintComponents(const SolverOptions& options) noexcept
{
    if (!options.block_layout.has_value()) {
        return 0;
    }
    if (const auto* block = options.block_layout->constraintFieldBlock()) {
        return block->n_components;
    }
    return 0;
}

[[nodiscard]] FieldSplitKind chooseAutoFieldSplitKind(const SolverOptions& options) noexcept
{
    const auto block_count = inferredBlockCount(options);
    if ((options.method == SolverMethod::BlockSchur || hasSaddlePointMetadata(options)) &&
        block_count == 2) {
        return FieldSplitKind::Schur;
    }

    if (hasSpecialPreconditionBlocks(options) ||
        hasSchurEliminableBlocks(options) ||
        block_count > 2) {
        return FieldSplitKind::Multiplicative;
    }

    return FieldSplitKind::Additive;
}

[[nodiscard]] FsilsBlockSchurSchurPreconditioner chooseAutoFsilsSchurPreconditioner(
    const SolverOptions& options) noexcept
{
    if (hasSpecialPreconditionBlocks(options) ||
        hasNonBlockDiagonalAuxiliaryBlocks(options) ||
        hasSchurEliminableBlocks(options)) {
        return FsilsBlockSchurSchurPreconditioner::AlgebraicSchur;
    }

    if (constraintComponents(options) > 1) {
        return FsilsBlockSchurSchurPreconditioner::BlockDiagL;
    }

    return FsilsBlockSchurSchurPreconditioner::DiagL;
}

[[nodiscard]] FsilsBlockSchurMomentumApproximation chooseAutoFsilsMomentumApproximation(
    const SolverOptions& options) noexcept
{
    if (hasSpecialPreconditionBlocks(options)) {
        return FsilsBlockSchurMomentumApproximation::ASM;
    }

    if (hasNonBlockDiagonalAuxiliaryBlocks(options)) {
        return FsilsBlockSchurMomentumApproximation::ILUK;
    }

    if (hasSchurEliminableBlocks(options) || primaryComponents(options) > 1) {
        return FsilsBlockSchurMomentumApproximation::BlockDiagK;
    }

    return FsilsBlockSchurMomentumApproximation::DiagK;
}

} // namespace

SolverOptions normalizeSolverOptionsForBackend(const SolverOptions& options,
                                               BackendKind backend,
                                               bool block_operator_available)
{
    SolverOptions normalized = options;
    addMetadataRoleMappings(normalized);

    switch (backend) {
        case BackendKind::PETSc:
            if (block_operator_available &&
                normalized.method != SolverMethod::Direct &&
                (normalized.preconditioner == PreconditionerType::None ||
                 normalized.preconditioner == PreconditionerType::Diagonal) &&
                inferredBlockCount(normalized) >= 2 &&
                (hasSpecialPreconditionBlocks(normalized) ||
                 hasSchurEliminableBlocks(normalized))) {
                normalized.preconditioner = PreconditionerType::FieldSplit;
            }

            if ((normalized.preconditioner == PreconditionerType::FieldSplit ||
                 normalized.method == SolverMethod::BlockSchur) &&
                normalized.fieldsplit.kind == FieldSplitKind::Auto) {
                normalized.fieldsplit.kind = chooseAutoFieldSplitKind(normalized);
            }
            break;

        case BackendKind::FSILS:
            if (hasSpecialPreconditionBlocks(normalized) &&
                normalized.preconditioner != PreconditionerType::RowColumnScaling) {
                normalized.fsils_use_rcs = true;
            }

            if (normalized.method == SolverMethod::BlockSchur) {
                if (normalized.fsils_blockschur_schur_preconditioner ==
                    FsilsBlockSchurSchurPreconditioner::Auto) {
                    normalized.fsils_blockschur_schur_preconditioner =
                        chooseAutoFsilsSchurPreconditioner(normalized);
                }

                if (normalized.fsils_blockschur_momentum_approximation ==
                    FsilsBlockSchurMomentumApproximation::Auto) {
                    normalized.fsils_blockschur_momentum_approximation =
                        chooseAutoFsilsMomentumApproximation(normalized);
                }
            }
            break;

        case BackendKind::Eigen:
        case BackendKind::Trilinos:
            if ((normalized.preconditioner == PreconditionerType::FieldSplit ||
                 normalized.method == SolverMethod::BlockSchur) &&
                normalized.fieldsplit.kind == FieldSplitKind::Auto) {
                normalized.fieldsplit.kind = chooseAutoFieldSplitKind(normalized);
            }
            break;
    }

    return normalized;
}

} // namespace backends
} // namespace FE
} // namespace svmp

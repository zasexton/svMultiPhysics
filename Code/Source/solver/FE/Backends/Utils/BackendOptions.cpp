/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Backends/Utils/BackendOptions.h"

#include "Core/FEException.h"

#include <algorithm>
#include <string>

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

std::string_view mixedBlockAssemblyModeToString(MixedBlockAssemblyMode mode) noexcept
{
    switch (mode) {
        case MixedBlockAssemblyMode::Unspecified: return "unspecified";
        case MixedBlockAssemblyMode::NativeOwnedRows: return "native-owned-rows";
        case MixedBlockAssemblyMode::BorderedReduced: return "bordered-reduced";
        case MixedBlockAssemblyMode::LocalCondensed: return "local-condensed";
        case MixedBlockAssemblyMode::DirectOnlyLowered: return "direct-only-lowered";
        case MixedBlockAssemblyMode::MetadataOnly: return "metadata-only";
        default: return "unknown";
    }
}

std::string_view mixedRowOwnershipPolicyToString(MixedRowOwnershipPolicy policy) noexcept
{
    switch (policy) {
        case MixedRowOwnershipPolicy::Unspecified: return "unspecified";
        case MixedRowOwnershipPolicy::SingleOwner: return "single-owner";
        case MixedRowOwnershipPolicy::BackendDofOwner: return "backend-dof-owner";
        case MixedRowOwnershipPolicy::CellOwner: return "cell-owner";
        case MixedRowOwnershipPolicy::QuadraturePointOwner: return "quadrature-point-owner";
        case MixedRowOwnershipPolicy::RegionOwner: return "region-owner";
        case MixedRowOwnershipPolicy::FacetOwner: return "facet-owner";
        default: return "unknown";
    }
}

std::string validateFsilsMixedLayoutContract(const MixedBlockLayout& layout,
                                             int dof_per_node)
{
    if (const auto* block = layout.firstAuxiliaryBlockWithoutExplicitAssemblyContract()) {
        return "FSILS mixed auxiliary block '" + block->name +
               "' has no explicit assembly mode";
    }

    for (const auto& block : layout.blocks) {
        if (block.kind != MixedBlockKind::Auxiliary ||
            block.assembly_mode != MixedBlockAssemblyMode::NativeOwnedRows) {
            continue;
        }
        if (block.row_ownership == MixedRowOwnershipPolicy::Unspecified) {
            return "FSILS native auxiliary block '" + block.name +
                   "' has no explicit row ownership policy";
        }
        if (block.row_ownership == MixedRowOwnershipPolicy::SingleOwner &&
            block.single_owner_rank < 0) {
            return "FSILS native auxiliary block '" + block.name +
                   "' has no valid single-owner rank";
        }
        if (block.row_ownership != MixedRowOwnershipPolicy::SingleOwner) {
            if (static_cast<GlobalIndex>(block.row_owner_ranks.size()) != block.size) {
                return "FSILS native auxiliary block '" + block.name +
                       "' has no concrete row-owner map";
            }
            if (std::any_of(block.row_owner_ranks.begin(),
                            block.row_owner_ranks.end(),
                            [](int owner) { return owner < 0; })) {
                return "FSILS native auxiliary block '" + block.name +
                       "' has invalid row-owner ranks";
            }
        } else if (!block.row_owner_ranks.empty()) {
            if (static_cast<GlobalIndex>(block.row_owner_ranks.size()) != block.size) {
                return "FSILS native auxiliary block '" + block.name +
                       "' has inconsistent single-owner row map";
            }
            if (std::any_of(block.row_owner_ranks.begin(),
                            block.row_owner_ranks.end(),
                            [&](int owner) {
                                return owner != block.single_owner_rank;
                            })) {
                return "FSILS native auxiliary block '" + block.name +
                       "' has inconsistent single-owner row map";
            }
        }
    }

    if (!layout.hasNativeAuxiliaryRows()) {
        return {};
    }

    if (dof_per_node <= 0) {
        return "FSILS native auxiliary rows require dof_per_node > 0";
    }

    std::vector<int> component_coverage(static_cast<std::size_t>(dof_per_node), 0);
    std::optional<GlobalIndex> common_node_count{};
    for (const auto& block : layout.blocks) {
        if (!block.usesNativeOwnedRows()) {
            continue;
        }
        if (block.node_component_start < 0 || block.node_component_count <= 0) {
            return "FSILS native mixed block '" + block.name +
                   "' is missing nodal component-range metadata";
        }
        if (block.node_component_start + block.node_component_count > dof_per_node) {
            return "FSILS native mixed block '" + block.name +
                   "' component range exceeds dof_per_node";
        }
        if (block.size <= 0 ||
            (block.size % static_cast<GlobalIndex>(block.node_component_count)) != 0) {
            return "FSILS native mixed block '" + block.name +
                   "' size is not a whole number of nodal component blocks";
        }
        const GlobalIndex block_node_count =
            block.size / static_cast<GlobalIndex>(block.node_component_count);
        if (block_node_count <= 0) {
            return "FSILS native mixed block '" + block.name +
                   "' has no represented nodes";
        }
        if (!common_node_count.has_value()) {
            common_node_count = block_node_count;
        } else if (*common_node_count != block_node_count) {
            return "FSILS native mixed block '" + block.name +
                   "' size is inconsistent with the common nodal-interleaved layout";
        }
        for (int c = block.node_component_start;
             c < block.node_component_start + block.node_component_count; ++c) {
            auto& count = component_coverage[static_cast<std::size_t>(c)];
            ++count;
            if (count > 1) {
                return "FSILS native mixed block '" + block.name +
                       "' overlaps another nodal component range";
            }
        }
    }

    for (int c = 0; c < dof_per_node; ++c) {
        if (component_coverage[static_cast<std::size_t>(c)] != 1) {
            return "FSILS native mixed blocks do not partition [0, dof_per_node)";
        }
    }

    if (common_node_count.has_value() &&
        layout.total_unknowns != *common_node_count * static_cast<GlobalIndex>(dof_per_node)) {
        return "FSILS native mixed layout total_unknowns is inconsistent with "
               "common_node_count * dof_per_node";
    }

    return {};
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
    const SolverOptions&) noexcept
{
    return FsilsBlockSchurSchurPreconditioner::AlgebraicSchur;
}

[[nodiscard]] FsilsBlockSchurMomentumApproximation chooseAutoFsilsMomentumApproximation(
    const SolverOptions& options) noexcept
{
    if (hasSpecialPreconditionBlocks(options)) {
        return FsilsBlockSchurMomentumApproximation::ASM;
    }

    return FsilsBlockSchurMomentumApproximation::ILUK;
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
            if (normalized.mixed_block_layout.has_value()) {
                if (const auto* block =
                        normalized.mixed_block_layout
                            ->firstAuxiliaryBlockWithoutExplicitAssemblyContract()) {
                    FE_THROW(InvalidArgumentException,
                             "normalizeSolverOptionsForBackend(FSILS): mixed auxiliary block '" +
                                 block->name + "' has no explicit assembly mode");
                }
                if (const auto* block =
                        normalized.mixed_block_layout
                            ->firstNativeAuxiliaryBlockWithoutExplicitRowOwnership()) {
                    FE_THROW(InvalidArgumentException,
                             "normalizeSolverOptionsForBackend(FSILS): native auxiliary block '" +
                                 block->name + "' has no concrete row ownership");
                }
            }

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

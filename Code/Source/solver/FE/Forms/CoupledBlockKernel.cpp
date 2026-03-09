/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Forms/CoupledBlockKernel.h"

#include "Core/Logger.h"
#include "Forms/FormKernels.h"
#include "Forms/JIT/JITCompiler.h"
#include "Forms/JIT/JITKernelWrapper.h"

#include <algorithm>
#include <cstdlib>

namespace svmp {
namespace FE {
namespace forms {

CoupledBlockKernel::CoupledBlockKernel(
    std::vector<BlockSpec> blocks,
    std::shared_ptr<jit::JITCompiler> compiler,
    JITOptions options)
    : blocks_(std::move(blocks)),
      compiler_(std::move(compiler)),
      options_(std::move(options))
{
    scratch_block_views_.resize(blocks_.size());
}

assembly::RequiredData CoupledBlockKernel::getRequiredData() const
{
    // Union of all blocks' requirements
    auto result = assembly::RequiredData::None;
    for (const auto& b : blocks_) {
        if (b.fallback_kernel) {
            result |= b.fallback_kernel->getRequiredData();
        }
    }
    return result;
}

std::vector<assembly::FieldRequirement> CoupledBlockKernel::fieldRequirements() const
{
    // Merge all blocks' field requirements, deduplicating by field ID
    std::vector<assembly::FieldRequirement> merged;
    for (const auto& b : blocks_) {
        if (!b.fallback_kernel) continue;
        for (const auto& req : b.fallback_kernel->fieldRequirements()) {
            auto it = std::find_if(merged.begin(), merged.end(),
                                   [&](const assembly::FieldRequirement& r) { return r.field == req.field; });
            if (it != merged.end()) {
                it->required |= req.required;
            } else {
                merged.push_back(req);
            }
        }
    }
    return merged;
}

bool CoupledBlockKernel::hasCell() const noexcept
{
    return std::any_of(blocks_.begin(), blocks_.end(),
                       [](const BlockSpec& b) { return b.fallback_kernel && b.fallback_kernel->hasCell(); });
}

void CoupledBlockKernel::computeCell(
    const assembly::AssemblyContext& ctx,
    assembly::KernelOutput& output)
{
    // Fallback: dispatch to first block's fallback kernel.
    // The assembler's coupled path handles multi-block dispatch;
    // this single-element path is only used as a compatibility shim.
    if (!blocks_.empty() && blocks_[0].fallback_kernel) {
        blocks_[0].fallback_kernel->computeCell(ctx, output);
    }
}

void CoupledBlockKernel::computeCellBatch(
    std::span<const assembly::AssemblyContext* const> contexts,
    std::span<assembly::KernelOutput> outputs)
{
    // Fallback: dispatch to first block's fallback kernel.
    if (!blocks_.empty() && blocks_[0].fallback_kernel) {
        blocks_[0].fallback_kernel->computeCellBatch(contexts, outputs);
    }
}

std::string CoupledBlockKernel::name() const
{
    return "CoupledBlockKernel[" + std::to_string(blocks_.size()) + " blocks]";
}

int CoupledBlockKernel::maxTemporalDerivativeOrder() const noexcept
{
    int max_order = 0;
    for (const auto& b : blocks_) {
        if (b.fallback_kernel) {
            max_order = std::max(max_order, b.fallback_kernel->maxTemporalDerivativeOrder());
        }
    }
    return max_order;
}

bool CoupledBlockKernel::isMatrixOnly() const noexcept
{
    return std::all_of(blocks_.begin(), blocks_.end(),
                       [](const BlockSpec& b) { return b.want_matrix && !b.want_vector; });
}

bool CoupledBlockKernel::isVectorOnly() const noexcept
{
    return std::all_of(blocks_.begin(), blocks_.end(),
                       [](const BlockSpec& b) { return !b.want_matrix && b.want_vector; });
}

void CoupledBlockKernel::maybeCompileMonolithic()
{
    if (attempted_monolithic_) return;
    attempted_monolithic_ = true;
    has_monolithic_jit_ = false;

    if (!std::getenv("SVMP_USE_MONOLITHIC")) {
        FE_LOG_DEBUG("CoupledBlockKernel: monolithic JIT disabled (set SVMP_USE_MONOLITHIC=1 to enable)");
        return;
    }

    if (!compiler_) {
        FE_LOG_DEBUG("CoupledBlockKernel: no JIT compiler, using per-block fallback");
        return;
    }

    // Extract FormIRs from fallback kernels.
    // Chain: fallback_kernel -> JITKernelWrapper -> SymbolicNonlinearFormKernel -> tangentIR()/residualIR()
    std::vector<jit::JITCompiler::MonolithicBlockSpec> block_specs;
    block_specs.reserve(blocks_.size());

    for (std::size_t i = 0; i < blocks_.size(); ++i) {
        const auto& bs = blocks_[i];
        if (!bs.fallback_kernel) {
            FE_LOG_DEBUG("CoupledBlockKernel: block has no fallback kernel, skipping monolithic");
            return;
        }

        // Unwrap JITKernelWrapper to get the underlying kernel
        const assembly::AssemblyKernel* inner = bs.fallback_kernel.get();
        if (const auto* wrapper = dynamic_cast<const jit::JITKernelWrapper*>(inner)) {
            inner = &wrapper->fallbackKernel();
        }

        const auto* sym_kernel = dynamic_cast<const SymbolicNonlinearFormKernel*>(inner);
        if (!sym_kernel) {
            FE_LOG_DEBUG("CoupledBlockKernel: block is not SymbolicNonlinearFormKernel, skipping monolithic");
            return;
        }

        jit::JITCompiler::MonolithicBlockSpec spec;
        spec.tangent_ir = &sym_kernel->tangentIR();
        spec.residual_ir = &sym_kernel->residualIR();
        spec.want_matrix = bs.want_matrix;
        spec.want_vector = bs.want_vector;

        block_specs.push_back(spec);
    }

    // Attempt monolithic compilation
    const auto result = compiler_->compileMonolithic(block_specs);
    if (!result.ok) {
        FE_LOG_DEBUG("CoupledBlockKernel: monolithic compilation not available, using per-block fallback");
        return;
    }

    if (result.kernels.empty() || result.kernels[0].address == 0) {
        FE_LOG_DEBUG("CoupledBlockKernel: monolithic compilation produced no kernel");
        return;
    }

    monolithic_cell_addr_ = result.kernels[0].address;
    has_monolithic_jit_ = true;
    FE_LOG_INFO("CoupledBlockKernel: monolithic JIT compiled successfully");
}

void CoupledBlockKernel::maybeCompilePairwise()
{
    if (attempted_pairwise_) return;
    attempted_pairwise_ = true;
    has_pairwise_jit_ = false;

    if (!std::getenv("SVMP_USE_PAIRWISE")) {
        FE_LOG_DEBUG("CoupledBlockKernel: pairwise JIT disabled (set SVMP_USE_PAIRWISE=1 to enable)");
        return;
    }

    if (has_monolithic_jit_) {
        FE_LOG_DEBUG("CoupledBlockKernel: monolithic already available, skipping pairwise");
        return;
    }

    if (!compiler_) {
        FE_LOG_DEBUG("CoupledBlockKernel: no JIT compiler, skipping pairwise");
        return;
    }

    if (blocks_.size() < 2) {
        FE_LOG_DEBUG("CoupledBlockKernel: fewer than 2 blocks, skipping pairwise");
        return;
    }

    // Extract FormIRs for all blocks (same logic as maybeCompileMonolithic).
    std::vector<jit::JITCompiler::MonolithicBlockSpec> all_specs;
    all_specs.reserve(blocks_.size());

    for (std::size_t i = 0; i < blocks_.size(); ++i) {
        const auto& bs = blocks_[i];
        if (!bs.fallback_kernel) {
            FE_LOG_DEBUG("CoupledBlockKernel: block has no fallback kernel, skipping pairwise");
            return;
        }

        const assembly::AssemblyKernel* inner = bs.fallback_kernel.get();
        if (const auto* wrapper = dynamic_cast<const jit::JITKernelWrapper*>(inner)) {
            inner = &wrapper->fallbackKernel();
        }

        const auto* sym_kernel = dynamic_cast<const SymbolicNonlinearFormKernel*>(inner);
        if (!sym_kernel) {
            FE_LOG_DEBUG("CoupledBlockKernel: block is not SymbolicNonlinearFormKernel, skipping pairwise");
            return;
        }

        jit::JITCompiler::MonolithicBlockSpec spec;
        spec.tangent_ir = &sym_kernel->tangentIR();
        spec.residual_ir = &sym_kernel->residualIR();
        spec.want_matrix = bs.want_matrix;
        spec.want_vector = bs.want_vector;
        all_specs.push_back(spec);
    }

    // Group blocks by trial space (col_dof_map + col_dof_offset).
    // Blocks in the same trial group share the same solution evaluation
    // and benefit from QP intermediate caching in the coupled codegen.
    trial_groups_.clear();
    std::vector<int> group_of(blocks_.size(), -1);
    for (std::size_t i = 0; i < blocks_.size(); ++i) {
        if (group_of[i] >= 0) continue;
        const auto gidx = static_cast<int>(trial_groups_.size());
        trial_groups_.emplace_back();
        auto& grp = trial_groups_.back();
        grp.block_indices.push_back(i);
        group_of[i] = gidx;

        for (std::size_t j = i + 1; j < blocks_.size(); ++j) {
            if (group_of[j] >= 0) continue;
            if (blocks_[i].col_dof_map == blocks_[j].col_dof_map &&
                blocks_[i].col_dof_offset == blocks_[j].col_dof_offset) {
                grp.block_indices.push_back(j);
                group_of[j] = gidx;
            }
        }
    }

    // If every group has only 1 block, pairwise doesn't help — fall through to per-block.
    if (std::all_of(trial_groups_.begin(), trial_groups_.end(),
                    [](const TrialGroup& g) { return g.block_indices.size() <= 1; })) {
        FE_LOG_DEBUG("CoupledBlockKernel: all trial groups have 1 block, pairwise not beneficial");
        trial_groups_.clear();
        return;
    }

    FE_LOG_INFO("CoupledBlockKernel: pairwise compilation: " +
                std::to_string(trial_groups_.size()) + " trial groups from " +
                std::to_string(blocks_.size()) + " blocks");

    // Compile each trial group as a separate monolithic kernel.
    for (std::size_t gi = 0; gi < trial_groups_.size(); ++gi) {
        auto& grp = trial_groups_[gi];

        std::vector<jit::JITCompiler::MonolithicBlockSpec> group_specs;
        group_specs.reserve(grp.block_indices.size());
        for (const auto bi : grp.block_indices) {
            group_specs.push_back(all_specs[bi]);
        }

        const std::string sym = "pairwise_group" + std::to_string(gi);
        FE_LOG_INFO("CoupledBlockKernel: compiling pairwise group " + std::to_string(gi) +
                    " with " + std::to_string(group_specs.size()) + " blocks");

        const auto result = compiler_->compileMonolithic(group_specs);
        if (!result.ok || result.kernels.empty() || result.kernels[0].address == 0) {
            FE_LOG_DEBUG("CoupledBlockKernel: pairwise group " + std::to_string(gi) +
                         " compilation failed, falling back to per-block");
            trial_groups_.clear();
            return;
        }

        grp.kernel_addr = result.kernels[0].address;
    }

    has_pairwise_jit_ = true;
    FE_LOG_INFO("CoupledBlockKernel: pairwise JIT compiled successfully (" +
                std::to_string(trial_groups_.size()) + " groups)");
}

} // namespace forms
} // namespace FE
} // namespace svmp

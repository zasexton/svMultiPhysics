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

} // namespace forms
} // namespace FE
} // namespace svmp

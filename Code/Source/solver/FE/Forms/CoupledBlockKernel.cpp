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

void CoupledBlockKernel::primeAllBlocksColocated()
{
    if (blocks_.empty() || !compiler_) {
        return;
    }

    // Collect FormIR and JITKernelWrapper from each block's fallback kernel.
    struct BlockInfo {
        std::size_t block_index{0};
        jit::JITKernelWrapper* wrapper{nullptr};
        const FormIR* ir{nullptr};
    };
    std::vector<BlockInfo> infos;
    infos.reserve(blocks_.size());

    for (std::size_t i = 0; i < blocks_.size(); ++i) {
        const auto& bs = blocks_[i];
        if (!bs.fallback_kernel) {
            continue;
        }

        auto* jit_wrapper = dynamic_cast<jit::JITKernelWrapper*>(bs.fallback_kernel.get());
        if (!jit_wrapper) {
            continue;
        }

        // Ensure the wrapper has been compiled so we can access its FormIR.
        jit_wrapper->ensureCompiled();

        // Extract FormIR from the underlying interpreter kernel.
        const auto& fallback = jit_wrapper->fallbackKernel();
        const FormIR* ir = nullptr;

        if (const auto* fk = dynamic_cast<const FormKernel*>(&fallback)) {
            ir = &fk->ir();
        } else if (const auto* snk = dynamic_cast<const SymbolicNonlinearFormKernel*>(&fallback)) {
            ir = &snk->tangentIR();
        } else if (const auto* lfk = dynamic_cast<const LinearFormKernel*>(&fallback)) {
            ir = &lfk->bilinearIR();
        }

        if (!ir || !ir->isCompiled()) {
            continue;
        }

        infos.push_back(BlockInfo{i, jit_wrapper, ir});
    }

    if (infos.size() < 2u) {
        // Colocation only benefits when there are multiple kernels.
        return;
    }

    // Build colocated specs from the FormIRs.
    std::vector<jit::JITCompiler::ColocatedKernelSpec> specs;
    specs.reserve(infos.size());

    for (const auto& info : infos) {
        jit::JITCompiler::ColocatedKernelSpec spec;
        spec.ir = info.ir;

        // Collect cell-domain term indices from the FormIR.
        for (std::size_t t = 0; t < info.ir->terms().size(); ++t) {
            if (info.ir->terms()[t].domain == IntegralDomain::Cell) {
                spec.term_indices.push_back(t);
            }
        }

        if (spec.term_indices.empty()) {
            continue;
        }

        spec.domain = IntegralDomain::Cell;
        specs.push_back(std::move(spec));
    }

    if (specs.size() < 2u) {
        return;
    }

    // Compile all block kernels into a single module.
    std::vector<jit::JITCompiler::ColocatedKernelResult> results;
    auto compile_result = compiler_->compileColocated(specs, results);

    if (!compile_result.ok || results.size() != specs.size()) {
        FE_LOG_WARNING("CoupledBlockKernel: colocated compilation failed: " + compile_result.message);
        return;
    }

    // Inject resolved addresses into each block's JITKernelWrapper.
    std::size_t result_idx = 0;
    for (const auto& info : infos) {
        if (result_idx >= results.size()) {
            break;
        }

        // Check that this info contributed a spec (had cell terms).
        bool had_cell_terms = false;
        for (std::size_t t = 0; t < info.ir->terms().size(); ++t) {
            if (info.ir->terms()[t].domain == IntegralDomain::Cell) {
                had_cell_terms = true;
                break;
            }
        }

        if (!had_cell_terms) {
            continue;
        }

        if (results[result_idx].address != 0) {
            info.wrapper->setExternalCellAddress(results[result_idx].address);
        }
        ++result_idx;
    }

    FE_LOG_INFO("CoupledBlockKernel: colocated " + std::to_string(infos.size()) +
                " block kernels into single module");
}

} // namespace forms
} // namespace FE
} // namespace svmp

/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Forms/MixedBlockKernelSet.h"

#include "Core/Logger.h"
#include "Forms/FormKernels.h"
#include "Forms/JIT/HardwareProfile.h"
#include "Forms/JIT/JITCompiler.h"
#include "Forms/JIT/JITKernelWrapper.h"
#include "Forms/JIT/KernelIR.h"

#include <algorithm>

namespace svmp {
namespace FE {
namespace forms {

MixedBlockKernelSet::MixedBlockKernelSet(
    std::vector<BlockSpec> blocks,
    std::shared_ptr<jit::JITCompiler> compiler,
    JITOptions options)
    : blocks_(std::move(blocks)),
      compiler_(std::move(compiler)),
      options_(std::move(options))
{
}

assembly::RequiredData MixedBlockKernelSet::getRequiredData() const
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

std::vector<assembly::FieldRequirement> MixedBlockKernelSet::fieldRequirements() const
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

assembly::MaterialStateSpec MixedBlockKernelSet::materialStateSpec() const noexcept
{
    assembly::MaterialStateSpec merged{};
    for (const auto& b : blocks_) {
        if (!b.fallback_kernel) {
            continue;
        }
        const auto spec = b.fallback_kernel->materialStateSpec();
        if (spec.bytes_per_qpt > merged.bytes_per_qpt) {
            merged.bytes_per_qpt = spec.bytes_per_qpt;
            merged.variables = spec.variables;
            merged.frame_transform_hook = spec.frame_transform_hook;
        } else if (spec.bytes_per_qpt == merged.bytes_per_qpt &&
                   merged.variables.empty() && !spec.variables.empty()) {
            merged.variables = spec.variables;
            merged.frame_transform_hook = spec.frame_transform_hook;
        }
        merged.alignment = std::max(merged.alignment, spec.alignment);
    }
    return merged;
}

std::vector<params::Spec> MixedBlockKernelSet::parameterSpecs() const
{
    std::vector<params::Spec> specs;
    for (const auto& b : blocks_) {
        if (!b.fallback_kernel) {
            continue;
        }
        const auto block_specs = b.fallback_kernel->parameterSpecs();
        specs.insert(specs.end(), block_specs.begin(), block_specs.end());
    }
    return specs;
}

void MixedBlockKernelSet::resolveParameterSlots(
    const std::function<std::optional<std::uint32_t>(std::string_view)>& slot_of_real_param)
{
    for (auto& b : blocks_) {
        if (b.fallback_kernel) {
            b.fallback_kernel->resolveParameterSlots(slot_of_real_param);
        }
    }
}

void MixedBlockKernelSet::resolveInlinableConstitutives()
{
    for (auto& b : blocks_) {
        if (b.fallback_kernel) {
            b.fallback_kernel->resolveInlinableConstitutives();
        }
    }
}

bool MixedBlockKernelSet::hasCell() const noexcept
{
    return std::any_of(blocks_.begin(), blocks_.end(),
                       [](const BlockSpec& b) { return b.fallback_kernel && b.fallback_kernel->hasCell(); });
}

void MixedBlockKernelSet::computeCell(
    const assembly::AssemblyContext& ctx,
    assembly::KernelOutput& output)
{
    // Compatibility fallback when invoked outside the explicit mixed-block
    // assembler path. The production path dispatches block-by-block in
    // StandardAssembler.
    if (!blocks_.empty() && blocks_[0].fallback_kernel) {
        blocks_[0].fallback_kernel->computeCell(ctx, output);
    }
}

void MixedBlockKernelSet::computeCellBatch(
    std::span<const assembly::AssemblyContext* const> contexts,
    std::span<assembly::KernelOutput> outputs)
{
    // Compatibility fallback mirroring computeCell().
    if (!blocks_.empty() && blocks_[0].fallback_kernel) {
        blocks_[0].fallback_kernel->computeCellBatch(contexts, outputs);
    }
}

std::string MixedBlockKernelSet::name() const
{
    return "MixedBlockKernelSet[" + std::to_string(blocks_.size()) + " blocks]";
}

int MixedBlockKernelSet::maxTemporalDerivativeOrder() const noexcept
{
    int max_order = 0;
    for (const auto& b : blocks_) {
        if (b.fallback_kernel) {
            max_order = std::max(max_order, b.fallback_kernel->maxTemporalDerivativeOrder());
        }
    }
    return max_order;
}

bool MixedBlockKernelSet::hasExplicitTimeDependency() const noexcept
{
    for (const auto& b : blocks_) {
        if (b.fallback_kernel && b.fallback_kernel->hasExplicitTimeDependency()) {
            return true;
        }
    }
    return false;
}

bool MixedBlockKernelSet::isMatrixOnly() const noexcept
{
    return std::all_of(blocks_.begin(), blocks_.end(),
                       [](const BlockSpec& b) { return b.want_matrix && !b.want_vector; });
}

bool MixedBlockKernelSet::isVectorOnly() const noexcept
{
    return std::all_of(blocks_.begin(), blocks_.end(),
                       [](const BlockSpec& b) { return !b.want_matrix && b.want_vector; });
}

void MixedBlockKernelSet::primeColocatedTextLayout()
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

    // Partition specs into groups whose estimated .text fits within L1i.
    // This prevents instruction cache thrashing for large coupled systems.
    const auto& hw = jit::hardwareProfile();
    const std::uint64_t text_budget = hw.colocationTextBudgetBytes();
    // Estimate per-spec .text size from actual KernelIR op counts when
    // available (FormIR terms → lowerToKernelIR → optimize → opCount).
    // Falls back to a conservative heuristic if lowering fails or no
    // FormIR is available.
    // Use telemetry-calibrated bytes/op when available, else default.
    const auto kBytesPerOp = jit::bytesPerOpCalibration().calibratedBytesPerOp(
        options_.specialization.bytes_per_op_estimate);
    const auto kRawBytesPerOp = static_cast<std::uint64_t>(options_.specialization.raw_bytes_per_op_estimate);
    constexpr auto kFallbackOpsPerTerm = jit::HardwareProfile::kFallbackOpsPerTerm;
    auto estimateSpecSize = [&](const jit::JITCompiler::ColocatedKernelSpec& spec) -> std::uint64_t {
        if (!spec.ir) {
            return static_cast<std::uint64_t>(spec.term_indices.size()) * kFallbackOpsPerTerm * kBytesPerOp;
        }
        std::uint64_t total_ops = 0;
        const auto& terms = spec.ir->terms();
        for (const auto idx : spec.term_indices) {
            if (idx >= terms.size()) {
                total_ops += kFallbackOpsPerTerm;
                continue;
            }
            try {
                auto lowered = jit::lowerToKernelIR(terms[idx].integrand);
                lowered.ir.optimize();
                total_ops += lowered.ir.opCount();
            } catch (...) {
                total_ops += kFallbackOpsPerTerm;
            }
        }

        if (const auto* sz = spec.specialization) {
            if (sz->domain == spec.domain &&
                sz->n_qpts_minus.has_value() &&
                sz->n_test_dofs_minus.has_value()) {
                std::uint64_t expand = static_cast<std::uint64_t>(*sz->n_qpts_minus) *
                                       static_cast<std::uint64_t>(*sz->n_test_dofs_minus);
                if (spec.ir->kind() == FormKind::Bilinear) {
                    expand *= static_cast<std::uint64_t>(sz->n_trial_dofs_minus.value_or(1u));
                }
                return total_ops * std::max<std::uint64_t>(1u, expand) * kRawBytesPerOp;
            }
        }

        return total_ops * kBytesPerOp;
    };

    // Greedy first-fit partitioning.
    struct Partition {
        std::vector<std::size_t> spec_indices;  // indices into specs[]
        std::uint64_t estimated_bytes{0};
    };
    std::vector<Partition> partitions;
    for (std::size_t si = 0; si < specs.size(); ++si) {
        const auto est = estimateSpecSize(specs[si]);
        bool placed = false;
        for (auto& part : partitions) {
            if (part.estimated_bytes + est <= text_budget) {
                part.spec_indices.push_back(si);
                part.estimated_bytes += est;
                placed = true;
                break;
            }
        }
        if (!placed) {
            partitions.push_back({{si}, est});
        }
    }

    // Compile each partition as a separate colocated module.
    std::vector<jit::JITCompiler::ColocatedKernelResult> results(specs.size());
    std::size_t modules_compiled = 0;
    for (const auto& part : partitions) {
        if (part.spec_indices.size() < 2u) {
            // Single-kernel partition — no colocation benefit.
            continue;
        }

        std::vector<jit::JITCompiler::ColocatedKernelSpec> group_specs;
        group_specs.reserve(part.spec_indices.size());
        for (const auto si : part.spec_indices) {
            group_specs.push_back(specs[si]);
        }

        std::vector<jit::JITCompiler::ColocatedKernelResult> group_results;
        auto compile_result = compiler_->compileColocated(group_specs, group_results);

        if (!compile_result.ok || group_results.size() != group_specs.size()) {
            FE_LOG_WARNING("MixedBlockKernelSet: colocated partition failed: " + compile_result.message);
            continue;
        }

        for (std::size_t gi = 0; gi < group_results.size(); ++gi) {
            results[part.spec_indices[gi]] = group_results[gi];
        }
        ++modules_compiled;
    }

    if (modules_compiled == 0) {
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

    FE_LOG_INFO("MixedBlockKernelSet: colocated " + std::to_string(infos.size()) +
                " block kernels into " + std::to_string(modules_compiled) +
                " module(s) (L1i budget " + std::to_string(text_budget) + " bytes)");
}

} // namespace forms
} // namespace FE
} // namespace svmp

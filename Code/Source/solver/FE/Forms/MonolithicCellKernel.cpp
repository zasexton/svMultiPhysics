/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Forms/MonolithicCellKernel.h"

#include "Core/FEException.h"
#include "Core/KernelTrace.h"
#include "Forms/FormKernels.h"
#include "Forms/JIT/JITCompiler.h"
#include "Forms/JIT/JITKernelWrapper.h"

#include <algorithm>
#include <sstream>

namespace svmp {
namespace FE {
namespace forms {

namespace {

const assembly::AssemblyKernel* unwrapKernel(const std::shared_ptr<assembly::AssemblyKernel>& kernel) noexcept
{
    if (!kernel) {
        return nullptr;
    }
    if (const auto* jit = dynamic_cast<const jit::JITKernelWrapper*>(kernel.get())) {
        return &jit->fallbackKernel();
    }
    return kernel.get();
}

const FormIR* tangentIRFromKernel(const assembly::AssemblyKernel* kernel) noexcept
{
    if (kernel == nullptr) {
        return nullptr;
    }
    if (const auto* form = dynamic_cast<const FormKernel*>(kernel)) {
        return form->ir().kind() == FormKind::Bilinear ? &form->ir() : nullptr;
    }
    if (const auto* linear = dynamic_cast<const LinearFormKernel*>(kernel)) {
        return &linear->bilinearIR();
    }
    if (const auto* symbolic = dynamic_cast<const SymbolicNonlinearFormKernel*>(kernel)) {
        return symbolic->tangentIR().isCompiled() ? &symbolic->tangentIR() : nullptr;
    }
    return nullptr;
}

const FormIR* residualIRFromKernel(const assembly::AssemblyKernel* kernel) noexcept
{
    if (kernel == nullptr) {
        return nullptr;
    }
    if (const auto* form = dynamic_cast<const FormKernel*>(kernel)) {
        return form->ir().kind() == FormKind::Linear ? &form->ir() : nullptr;
    }
    if (const auto* linear = dynamic_cast<const LinearFormKernel*>(kernel)) {
        return linear->linearIR() ? &*linear->linearIR() : nullptr;
    }
    if (const auto* symbolic = dynamic_cast<const SymbolicNonlinearFormKernel*>(kernel)) {
        return &symbolic->residualIR();
    }
    if (const auto* nonlinear = dynamic_cast<const NonlinearFormKernel*>(kernel)) {
        return &nonlinear->residualIR();
    }
    return nullptr;
}

void rewriteResidualTrialToCurrentState(FormIR& ir)
{
    if (ir.kind() != FormKind::Residual) {
        return;
    }

    const auto transform = [&](const FormExprNode& n) -> std::optional<FormExpr> {
        if (n.type() != FormExprType::TrialFunction) {
            return std::nullopt;
        }
        const auto* sig = n.spaceSignature();
        FE_THROW_IF(!sig, InvalidArgumentException,
                    "MonolithicCellKernel: TrialFunction missing SpaceSignature during residual rewrite");
        return FormExpr::stateField(CURRENT_SOLUTION_FIELD_ID, *sig, n.toString());
    };

    ir.transformIntegrands(transform);
}

template <class Getter>
std::vector<assembly::FieldRequirement> mergeFieldRequirements(
    const std::vector<MonolithicCellKernel::BlockSpec>& blocks,
    Getter&& getter)
{
    std::vector<assembly::FieldRequirement> merged;
    for (const auto& block : blocks) {
        const auto* kernel = getter(block);
        if (kernel == nullptr) {
            continue;
        }
        for (const auto& req : kernel->fieldRequirements()) {
            auto it = std::find_if(
                merged.begin(), merged.end(),
                [&](const assembly::FieldRequirement& existing) { return existing.field == req.field; });
            if (it == merged.end()) {
                merged.push_back(req);
            } else {
                it->required = it->required | req.required;
            }
        }
    }
    return merged;
}

template <class Getter>
assembly::MaterialStateSpec mergeMaterialStateSpec(
    const std::vector<MonolithicCellKernel::BlockSpec>& blocks,
    Getter&& getter) noexcept
{
    assembly::MaterialStateSpec merged{};
    for (const auto& block : blocks) {
        const auto* kernel = getter(block);
        if (kernel == nullptr) {
            continue;
        }
        const auto spec = kernel->materialStateSpec();
        merged.bytes_per_qpt = std::max(merged.bytes_per_qpt, spec.bytes_per_qpt);
        merged.alignment = std::max(merged.alignment, spec.alignment);
    }
    return merged;
}

template <class Getter>
std::vector<params::Spec> gatherParameterSpecs(
    const std::vector<MonolithicCellKernel::BlockSpec>& blocks,
    Getter&& getter)
{
    std::vector<params::Spec> specs;
    for (const auto& block : blocks) {
        const auto* kernel = getter(block);
        if (kernel == nullptr) {
            continue;
        }
        const auto block_specs = kernel->parameterSpecs();
        specs.insert(specs.end(), block_specs.begin(), block_specs.end());
    }
    return specs;
}

void resolveParameterSlotsInIR(
    FormIR& ir,
    const std::function<std::optional<std::uint32_t>(std::string_view)>& slot_of_real_param,
    std::string_view where)
{
    const auto transform = [&](const FormExprNode& n) -> std::optional<FormExpr> {
        if (n.type() != FormExprType::ParameterSymbol) {
            return std::nullopt;
        }
        const auto key = n.symbolName();
        FE_THROW_IF(!key || key->empty(), InvalidArgumentException,
                    std::string(where) + ": ParameterSymbol node missing name");
        const auto slot = slot_of_real_param(*key);
        FE_THROW_IF(!slot.has_value(), InvalidArgumentException,
                    std::string(where) + ": could not resolve parameter slot for '" +
                        std::string(*key) + "'");
        return FormExpr::parameterRef(*slot);
    };

    ir.transformIntegrands(transform);
}

} // namespace

MonolithicCellKernel::MonolithicCellKernel(
    std::vector<BlockSpec> blocks,
    std::shared_ptr<jit::JITCompiler> compiler,
    JITOptions options)
    : blocks_(std::move(blocks))
    , compiler_(std::move(compiler))
    , options_(std::move(options))
{
    for (auto& block : blocks_) {
        if (block.residual_ir) {
            rewriteResidualTrialToCurrentState(*block.residual_ir);
        }
    }
}

assembly::RequiredData MonolithicCellKernel::getRequiredData() const
{
    auto result = assembly::RequiredData::None;
    for (const auto& block : blocks_) {
        if (block.fallback_kernel) {
            result = result | block.fallback_kernel->getRequiredData();
        }
    }
    return result;
}

std::vector<assembly::FieldRequirement> MonolithicCellKernel::fieldRequirements() const
{
    return mergeFieldRequirements(
        blocks_,
        [](const BlockSpec& block) -> const assembly::AssemblyKernel* {
            return block.fallback_kernel.get();
        });
}

assembly::MaterialStateSpec MonolithicCellKernel::materialStateSpec() const noexcept
{
    return mergeMaterialStateSpec(
        blocks_,
        [](const BlockSpec& block) -> const assembly::AssemblyKernel* {
            return block.fallback_kernel.get();
        });
}

std::vector<params::Spec> MonolithicCellKernel::parameterSpecs() const
{
    return gatherParameterSpecs(
        blocks_,
        [](const BlockSpec& block) -> const assembly::AssemblyKernel* {
            return block.fallback_kernel.get();
        });
}

void MonolithicCellKernel::resolveParameterSlots(
    const std::function<std::optional<std::uint32_t>(std::string_view)>& slot_of_real_param)
{
    for (auto& block : blocks_) {
        if (block.fallback_kernel) {
            block.fallback_kernel->resolveParameterSlots(slot_of_real_param);
        }
        if (block.tangent_ir) {
            resolveParameterSlotsInIR(*block.tangent_ir, slot_of_real_param, "MonolithicCellKernel::tangent_ir");
        }
        if (block.residual_ir) {
            resolveParameterSlotsInIR(*block.residual_ir, slot_of_real_param, "MonolithicCellKernel::residual_ir");
        }
    }
    invalidateCompiledDispatch();
}

void MonolithicCellKernel::resolveInlinableConstitutives()
{
    for (auto& block : blocks_) {
        if (block.fallback_kernel) {
            block.fallback_kernel->resolveInlinableConstitutives();
        }
    }
}

bool MonolithicCellKernel::hasCell() const noexcept
{
    return std::any_of(
        blocks_.begin(), blocks_.end(),
        [](const BlockSpec& block) {
            return (block.fallback_kernel && block.fallback_kernel->hasCell()) ||
                   (block.tangent_ir && block.tangent_ir->hasCellTerms()) ||
                   (block.residual_ir && block.residual_ir->hasCellTerms());
        });
}

void MonolithicCellKernel::computeCell(
    const assembly::AssemblyContext& ctx,
    assembly::KernelOutput& output)
{
    for (const auto& block : blocks_) {
        if (block.fallback_kernel && block.fallback_kernel->hasCell()) {
            block.fallback_kernel->computeCell(ctx, output);
            return;
        }
    }
}

void MonolithicCellKernel::computeCellBatch(
    std::span<const assembly::AssemblyContext* const> contexts,
    std::span<assembly::KernelOutput> outputs)
{
    for (const auto& block : blocks_) {
        if (block.fallback_kernel && block.fallback_kernel->hasCell()) {
            block.fallback_kernel->computeCellBatch(contexts, outputs);
            return;
        }
    }
}

std::string MonolithicCellKernel::name() const
{
    return "MonolithicCellKernel[" + std::to_string(blocks_.size()) + " blocks]";
}

int MonolithicCellKernel::maxTemporalDerivativeOrder() const noexcept
{
    int max_order = 0;
    for (const auto& block : blocks_) {
        if (block.fallback_kernel) {
            max_order = std::max(max_order, block.fallback_kernel->maxTemporalDerivativeOrder());
        }
        if (block.tangent_ir) {
            max_order = std::max(max_order, block.tangent_ir->maxTimeDerivativeOrder());
        }
        if (block.residual_ir) {
            max_order = std::max(max_order, block.residual_ir->maxTimeDerivativeOrder());
        }
    }
    return max_order;
}

bool MonolithicCellKernel::isMatrixOnly() const noexcept
{
    return std::all_of(
        blocks_.begin(), blocks_.end(),
        [](const BlockSpec& block) { return block.want_matrix && !block.want_vector; });
}

bool MonolithicCellKernel::isVectorOnly() const noexcept
{
    return std::all_of(
        blocks_.begin(), blocks_.end(),
        [](const BlockSpec& block) { return !block.want_matrix && block.want_vector; });
}

void MonolithicCellKernel::invalidateCompiledDispatch()
{
    std::lock_guard<std::mutex> lock(compile_mutex_);
    compile_attempted_ = false;
    compiled_address_ = 0;
    compile_message_.clear();
}

void MonolithicCellKernel::ensureCompiled() const
{
    std::lock_guard<std::mutex> lock(compile_mutex_);
    if (compile_attempted_) {
        return;
    }
    compile_attempted_ = true;

    if (!compiler_) {
        compile_message_ = "MonolithicCellKernel: no JIT compiler configured";
        return;
    }
    if (blocks_.empty()) {
        compile_message_ = "MonolithicCellKernel: no blocks to compile";
        return;
    }

    std::vector<jit::JITCompiler::MonolithicBlockSpec> specs;
    specs.reserve(blocks_.size());

    for (std::size_t i = 0; i < blocks_.size(); ++i) {
        const auto& block = blocks_[i];
        const auto* fallback = unwrapKernel(block.fallback_kernel);
        if (dynamic_cast<const NonlinearFormKernel*>(fallback) != nullptr) {
            compile_message_ =
                "MonolithicCellKernel: AD-backed NonlinearFormKernel blocks use exact per-block fallback";
            return;
        }
        const FormIR* tangent_ir = block.tangent_ir ? &*block.tangent_ir : tangentIRFromKernel(fallback);
        const FormIR* residual_ir = block.residual_ir ? &*block.residual_ir : residualIRFromKernel(fallback);

        if (block.want_matrix) {
            if (tangent_ir == nullptr) {
                compile_message_ = "MonolithicCellKernel: missing tangent IR for block " + std::to_string(i);
                return;
            }
            if (!tangent_ir->isCompiled()) {
                compile_message_ = "MonolithicCellKernel: tangent IR for block " + std::to_string(i) +
                                   " is not compiled";
                return;
            }
        }

        if (block.want_vector) {
            if (residual_ir == nullptr) {
                compile_message_ = "MonolithicCellKernel: missing residual IR for block " + std::to_string(i);
                return;
            }
            if (!residual_ir->isCompiled()) {
                compile_message_ = "MonolithicCellKernel: residual IR for block " + std::to_string(i) +
                                   " is not compiled";
                return;
            }
        }

        specs.push_back(jit::JITCompiler::MonolithicBlockSpec{
            .tangent_ir = tangent_ir,
            .residual_ir = residual_ir,
            .want_matrix = block.want_matrix,
            .want_vector = block.want_vector,
        });
    }

    if (core::kernelTraceEnabled(core::KernelTraceChannel::Selection)) {
        std::ostringstream oss;
        oss << "MonolithicCellKernel: compiling " << blocks_.size() << " block(s)";
        core::kernelTraceLog(core::KernelTraceChannel::Selection, oss.str());
    }

    const auto result = compiler_->compileMonolithic(specs);
    if (!result.ok || result.kernels.empty() || result.kernels.front().address == 0) {
        compile_message_ = result.message.empty()
            ? "MonolithicCellKernel: JIT compilation failed"
            : result.message;
        return;
    }

    compiled_address_ = result.kernels.front().address;
    compile_message_ = result.message;

    if (core::kernelTraceEnabled(core::KernelTraceChannel::Selection)) {
        std::ostringstream oss;
        oss << "MonolithicCellKernel: compiled symbol='" << result.kernels.front().symbol
            << "' address=" << compiled_address_;
        core::kernelTraceLog(core::KernelTraceChannel::Selection, oss.str());
    }
}

} // namespace forms
} // namespace FE
} // namespace svmp

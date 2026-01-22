/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Forms/JIT/JITKernelWrapper.h"

#include "Core/FEException.h"
#include "Core/Logger.h"

#include "Forms/FormKernels.h"
#include "Forms/JIT/JITValidation.h"

#include <optional>
#include <sstream>
#include <utility>

#ifndef SVMP_FE_ENABLE_LLVM_JIT
#define SVMP_FE_ENABLE_LLVM_JIT 0
#endif

namespace svmp {
namespace FE {
namespace forms {
namespace jit {

namespace {

struct IRView {
    const FormIR* ir{nullptr};
    std::string_view label{};
};

[[nodiscard]] std::vector<IRView> gatherIRs(const assembly::AssemblyKernel& kernel)
{
    std::vector<IRView> out;

    if (const auto* k = dynamic_cast<const FormKernel*>(&kernel)) {
        out.push_back(IRView{.ir = &k->ir(), .label = "FormKernel"});
        return out;
    }

    if (const auto* k = dynamic_cast<const LinearFormKernel*>(&kernel)) {
        out.push_back(IRView{.ir = &k->bilinearIR(), .label = "LinearFormKernel(bilinear)"});
        if (k->linearIR().has_value()) {
            out.push_back(IRView{.ir = &*k->linearIR(), .label = "LinearFormKernel(linear)"});
        }
        return out;
    }

    if (const auto* k = dynamic_cast<const NonlinearFormKernel*>(&kernel)) {
        out.push_back(IRView{.ir = &k->residualIR(), .label = "NonlinearFormKernel(residual)"});
        return out;
    }

    return out;
}

[[nodiscard]] std::string formatValidationIssue(const ValidationIssue& issue)
{
    std::ostringstream oss;
    oss << issue.message;
    if (!issue.subexpr.empty()) {
        oss << " (subexpr: " << issue.subexpr << ")";
    }
    return oss.str();
}

} // namespace

JITKernelWrapper::JITKernelWrapper(std::shared_ptr<assembly::AssemblyKernel> fallback,
                                   JITOptions options)
    : fallback_(std::move(fallback)),
      options_(std::move(options))
{
    FE_CHECK_NOT_NULL(fallback_.get(), "JITKernelWrapper: fallback kernel");
}

assembly::RequiredData JITKernelWrapper::getRequiredData() const
{
    return fallback_->getRequiredData();
}

std::vector<assembly::FieldRequirement> JITKernelWrapper::fieldRequirements() const
{
    return fallback_->fieldRequirements();
}

assembly::MaterialStateSpec JITKernelWrapper::materialStateSpec() const noexcept
{
    return fallback_->materialStateSpec();
}

std::vector<params::Spec> JITKernelWrapper::parameterSpecs() const
{
    return fallback_->parameterSpecs();
}

void JITKernelWrapper::resolveParameterSlots(
    const std::function<std::optional<std::uint32_t>(std::string_view)>& slot_of_real_param)
{
    fallback_->resolveParameterSlots(slot_of_real_param);
    markDirty();
    maybeCompile();
}

void JITKernelWrapper::resolveInlinableConstitutives()
{
    fallback_->resolveInlinableConstitutives();
    markDirty();
}

bool JITKernelWrapper::hasCell() const noexcept { return fallback_->hasCell(); }
bool JITKernelWrapper::hasBoundaryFace() const noexcept { return fallback_->hasBoundaryFace(); }
bool JITKernelWrapper::hasInteriorFace() const noexcept { return fallback_->hasInteriorFace(); }
bool JITKernelWrapper::hasInterfaceFace() const noexcept { return fallback_->hasInterfaceFace(); }

void JITKernelWrapper::computeCell(const assembly::AssemblyContext& ctx,
                                   assembly::KernelOutput& output)
{
    maybeCompile();
    fallback_->computeCell(ctx, output);
}

void JITKernelWrapper::computeBoundaryFace(const assembly::AssemblyContext& ctx,
                                           int boundary_marker,
                                           assembly::KernelOutput& output)
{
    maybeCompile();
    fallback_->computeBoundaryFace(ctx, boundary_marker, output);
}

void JITKernelWrapper::computeInteriorFace(const assembly::AssemblyContext& ctx_minus,
                                           const assembly::AssemblyContext& ctx_plus,
                                           assembly::KernelOutput& output_minus,
                                           assembly::KernelOutput& output_plus,
                                           assembly::KernelOutput& coupling_minus_plus,
                                           assembly::KernelOutput& coupling_plus_minus)
{
    maybeCompile();
    fallback_->computeInteriorFace(ctx_minus, ctx_plus,
                                  output_minus, output_plus,
                                  coupling_minus_plus, coupling_plus_minus);
}

void JITKernelWrapper::computeInterfaceFace(const assembly::AssemblyContext& ctx_minus,
                                            const assembly::AssemblyContext& ctx_plus,
                                            int interface_marker,
                                            assembly::KernelOutput& output_minus,
                                            assembly::KernelOutput& output_plus,
                                            assembly::KernelOutput& coupling_minus_plus,
                                            assembly::KernelOutput& coupling_plus_minus)
{
    maybeCompile();
    fallback_->computeInterfaceFace(ctx_minus, ctx_plus, interface_marker,
                                   output_minus, output_plus,
                                   coupling_minus_plus, coupling_plus_minus);
}

std::string JITKernelWrapper::name() const
{
    return "Forms::JITKernelWrapper(" + fallback_->name() + ")";
}

int JITKernelWrapper::maxTemporalDerivativeOrder() const noexcept
{
    return fallback_->maxTemporalDerivativeOrder();
}

bool JITKernelWrapper::isSymmetric() const noexcept
{
    return fallback_->isSymmetric();
}

bool JITKernelWrapper::isMatrixOnly() const noexcept
{
    return fallback_->isMatrixOnly();
}

bool JITKernelWrapper::isVectorOnly() const noexcept
{
    return fallback_->isVectorOnly();
}

void JITKernelWrapper::markDirty() noexcept
{
    std::lock_guard<std::mutex> lock(jit_mutex_);
    ++revision_;
    compiled_revision_ = static_cast<std::uint64_t>(-1);
    attempted_revision_ = static_cast<std::uint64_t>(-1);
}

void JITKernelWrapper::maybeCompile()
{
    if (!options_.enable) {
        return;
    }

    std::lock_guard<std::mutex> lock(jit_mutex_);

    if (compiled_revision_ == revision_) {
        return;
    }
    if (attempted_revision_ == revision_) {
        return;
    }
    attempted_revision_ = revision_;

    const auto irs = gatherIRs(*fallback_);
    if (!irs.empty() && !warned_validation_) {
        ValidationOptions vopt;
        vopt.strictness = Strictness::AllowExternalCalls;

        for (const auto& v : irs) {
            if (v.ir == nullptr) continue;
            const auto r = canCompile(*v.ir, vopt);
            if (!r.ok) {
                warned_validation_ = true;
                std::string msg = "JIT: kernel '" + fallback_->name() + "' is not currently JIT-compatible";
                if (!v.label.empty()) {
                    msg += " [" + std::string(v.label) + "]";
                }
                if (r.first_issue.has_value()) {
                    msg += ": " + formatValidationIssue(*r.first_issue);
                }
                FE_LOG_WARNING(msg);
                break;
            }
        }
    }

    if (warned_unavailable_) {
        return;
    }

#if SVMP_FE_ENABLE_LLVM_JIT
    FE_LOG_WARNING("JIT requested for kernel '" + fallback_->name() +
                   "', but the LLVM backend is not implemented yet (phase 0); using interpreter.");
#else
    FE_LOG_WARNING("JIT requested for kernel '" + fallback_->name() +
                   "', but FE was built without LLVM JIT support; using interpreter.");
#endif
    warned_unavailable_ = true;
}

} // namespace jit
} // namespace forms
} // namespace FE
} // namespace svmp


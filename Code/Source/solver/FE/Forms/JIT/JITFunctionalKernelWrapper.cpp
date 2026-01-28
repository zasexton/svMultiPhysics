/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Forms/JIT/JITFunctionalKernelWrapper.h"

#include "Assembly/JIT/KernelArgs.h"
#include "Core/FEException.h"
#include "Core/Logger.h"

#include "Forms/JIT/JITCompiler.h"
#include "Forms/JIT/JITValidation.h"

#include <utility>

namespace svmp {
namespace FE {
namespace forms {
namespace jit {

namespace {

using JITFn = void (*)(const void*);

inline void callJIT(std::uintptr_t addr, const void* args) noexcept
{
    if (addr == 0 || args == nullptr) {
        return;
    }
    reinterpret_cast<JITFn>(addr)(args);
}

inline void overrideFunctionalWeights(assembly::jit::KernelSideArgsV3& side) noexcept
{
    side.time_derivative_term_weight = 1.0;
    side.non_time_derivative_term_weight = 1.0;
    side.dt1_term_weight = 1.0;
    side.dt2_term_weight = 1.0;
}

} // namespace

JITFunctionalKernelWrapper::JITFunctionalKernelWrapper(std::shared_ptr<assembly::FunctionalKernel> fallback,
                                                       FormExpr integrand,
                                                       Domain domain,
                                                       JITOptions options)
    : fallback_(std::move(fallback)),
      integrand_(std::move(integrand)),
      domain_(domain),
      options_(std::move(options))
{
    FE_CHECK_NOT_NULL(fallback_.get(), "JITFunctionalKernelWrapper: fallback kernel");
    FE_THROW_IF(!integrand_.isValid(), InvalidArgumentException,
                "JITFunctionalKernelWrapper: invalid integrand");
}

assembly::RequiredData JITFunctionalKernelWrapper::getRequiredData() const noexcept
{
    return fallback_->getRequiredData();
}

std::vector<assembly::FieldRequirement> JITFunctionalKernelWrapper::fieldRequirements() const
{
    return fallback_->fieldRequirements();
}

bool JITFunctionalKernelWrapper::hasCell() const noexcept { return fallback_->hasCell(); }
bool JITFunctionalKernelWrapper::hasBoundaryFace() const noexcept { return fallback_->hasBoundaryFace(); }

Real JITFunctionalKernelWrapper::evaluateCell(const assembly::AssemblyContext& ctx, LocalIndex q)
{
    return fallback_->evaluateCell(ctx, q);
}

Real JITFunctionalKernelWrapper::evaluateBoundaryFace(const assembly::AssemblyContext& ctx,
                                                      LocalIndex q,
                                                      int boundary_marker)
{
    return fallback_->evaluateBoundaryFace(ctx, q, boundary_marker);
}

Real JITFunctionalKernelWrapper::evaluateCellTotal(const assembly::AssemblyContext& ctx)
{
    if (domain_ != Domain::Cell) {
        return fallback_->evaluateCellTotal(ctx);
    }

    maybeCompile();
    if (!canUseJIT()) {
        return fallback_->evaluateCellTotal(ctx);
    }

    try {
        thread_local assembly::KernelOutput out;
        out.reserve(/*n_test=*/1, /*n_trial=*/ctx.numTrialDofs(), /*need_matrix=*/false, /*need_vector=*/true);
        out.clear();

        const auto checks = assembly::jit::PackingChecks{.validate_alignment = true};
        auto args = assembly::jit::packCellKernelArgsV3(ctx, out, checks);

        args.side.n_test_dofs = 1u;
        args.output.n_test_dofs = 1u;
        overrideFunctionalWeights(args.side);

        callJIT(addr_, &args);
        return out.local_vector.empty() ? Real(0.0) : out.local_vector[0];
    } catch (const std::exception& e) {
        markRuntimeFailureOnce("evaluateCellTotal", e.what());
        return fallback_->evaluateCellTotal(ctx);
    } catch (...) {
        markRuntimeFailureOnce("evaluateCellTotal", "unknown exception");
        return fallback_->evaluateCellTotal(ctx);
    }
}

Real JITFunctionalKernelWrapper::evaluateBoundaryFaceTotal(const assembly::AssemblyContext& ctx,
                                                           int boundary_marker)
{
    if (domain_ != Domain::BoundaryFace) {
        return fallback_->evaluateBoundaryFaceTotal(ctx, boundary_marker);
    }

    maybeCompile();
    if (!canUseJIT()) {
        return fallback_->evaluateBoundaryFaceTotal(ctx, boundary_marker);
    }

    try {
        thread_local assembly::KernelOutput out;
        out.reserve(/*n_test=*/1, /*n_trial=*/ctx.numTrialDofs(), /*need_matrix=*/false, /*need_vector=*/true);
        out.clear();

        const auto checks = assembly::jit::PackingChecks{.validate_alignment = true};
        auto args = assembly::jit::packBoundaryFaceKernelArgsV3(ctx, boundary_marker, out, checks);

        args.side.n_test_dofs = 1u;
        args.output.n_test_dofs = 1u;
        overrideFunctionalWeights(args.side);

        callJIT(addr_, &args);
        return out.local_vector.empty() ? Real(0.0) : out.local_vector[0];
    } catch (const std::exception& e) {
        markRuntimeFailureOnce("evaluateBoundaryFaceTotal", e.what());
        return fallback_->evaluateBoundaryFaceTotal(ctx, boundary_marker);
    } catch (...) {
        markRuntimeFailureOnce("evaluateBoundaryFaceTotal", "unknown exception");
        return fallback_->evaluateBoundaryFaceTotal(ctx, boundary_marker);
    }
}

std::string JITFunctionalKernelWrapper::name() const
{
    return "Forms::JITFunctionalKernelWrapper(" + fallback_->name() + ")";
}

bool JITFunctionalKernelWrapper::canUseJIT() const noexcept
{
    return options_.enable && compiled_ && addr_ != 0 && !runtime_failed_;
}

void JITFunctionalKernelWrapper::markRuntimeFailureOnce(std::string_view where, std::string_view msg) noexcept
{
    std::lock_guard<std::mutex> lock(jit_mutex_);
    runtime_failed_ = true;
    if (warned_runtime_failure_) {
        return;
    }
    warned_runtime_failure_ = true;

    std::string full =
        "JIT: runtime failure in " + std::string(where) + " for functional kernel '" + fallback_->name() + "'";
    if (!msg.empty()) {
        full += ": " + std::string(msg);
    }
    FE_LOG_WARNING(full);
}

void JITFunctionalKernelWrapper::maybeCompile()
{
    if (!options_.enable) {
        return;
    }

    std::lock_guard<std::mutex> lock(jit_mutex_);
    if (compiled_ || attempted_) {
        return;
    }
    attempted_ = true;

    if (!compiler_) {
        compiler_ = JITCompiler::getOrCreate(options_);
    }
    if (!compiler_) {
        if (!warned_unavailable_) {
            warned_unavailable_ = true;
            FE_LOG_WARNING("JIT requested for functional kernel '" + fallback_->name() +
                           "', but JITCompiler could not be created; using interpreter.");
        }
        return;
    }

    ValidationOptions vopt;
    vopt.strictness = Strictness::AllowExternalCalls;

    const IntegralDomain jit_domain =
        (domain_ == Domain::Cell) ? IntegralDomain::Cell : IntegralDomain::Boundary;

    const auto r = compiler_->compileFunctional(integrand_, jit_domain, vopt);
    if (!r.ok || r.kernels.empty()) {
        if (!warned_compile_failure_) {
            warned_compile_failure_ = true;
            std::string msg = "JIT: failed to compile functional kernel '" + fallback_->name() + "'";
            if (!r.message.empty()) {
                msg += ": " + r.message;
            }
            FE_LOG_WARNING(msg);
        }
        compile_message_ = r.message;
        return;
    }

    // Synthetic functional kernels are emitted as a single (domain, marker=-1) kernel group.
    addr_ = 0;
    for (const auto& k : r.kernels) {
        if (k.domain == jit_domain) {
            addr_ = k.address;
            break;
        }
    }

    if (addr_ == 0) {
        if (!warned_compile_failure_) {
            warned_compile_failure_ = true;
            FE_LOG_WARNING("JIT: functional compile produced no runnable kernel address for '" + fallback_->name() + "'");
        }
        return;
    }

    compiled_ = true;
    compile_message_.clear();
}

} // namespace jit
} // namespace forms
} // namespace FE
} // namespace svmp

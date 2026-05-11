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

#include "Forms/JIT/HardwareProfile.h"
#include "Forms/JIT/JITCompiler.h"
#include "Forms/JIT/JITValidation.h"

#include <algorithm>
#include <utility>
#include <vector>

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

[[nodiscard]] std::size_t effectiveSimdBatchWidth(const JITOptions& options) noexcept
{
    if (!options.simd_batch) {
        return 1u;
    }
    const auto hardware_width = static_cast<std::size_t>(hardwareProfile().simdDoubles());
    return (hardware_width == 2u) ? 2u : 1u;
}

inline void overrideFunctionalWeights(assembly::jit::KernelSideArgsV6& side) noexcept
{
    side.n_test_dofs = 1u;
    side.time_derivative_term_weight = 1.0;
    side.non_time_derivative_term_weight = 1.0;
    side.dt_term_weights.fill(Real(1.0));
    side.max_time_derivative_order = 0u;
}

[[nodiscard]] inline assembly::jit::KernelOutputViewV6
makeFunctionalOutputView(Real* value, std::uint32_t n_trial_dofs) noexcept
{
    assembly::jit::KernelOutputViewV6 out;
    out.element_matrix = nullptr;
    out.element_vector = value;
    out.n_test_dofs = 1u;
    out.n_trial_dofs = n_trial_dofs;
    return out;
}

[[nodiscard]] inline assembly::jit::KernelSideArgsV6
packFunctionalSide(const assembly::AssemblyContext& ctx) noexcept
{
    const auto checks = assembly::jit::PackingChecks{.validate_alignment = false};
    auto side = assembly::jit::detail::packSideArgsV6(ctx, std::nullopt, std::nullopt, checks);
    overrideFunctionalWeights(side);
    return side;
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

    maybeCompile(ctx);
    if (!canUseJIT()) {
        return fallback_->evaluateCellTotal(ctx);
    }

    try {
        Real value = 0.0;
        auto side = packFunctionalSide(ctx);
        auto output = makeFunctionalOutputView(&value,
                                               static_cast<std::uint32_t>(ctx.numTrialDofs()));

        if (options_.vectorize) {
            const auto padded_n = static_cast<std::uint32_t>(effectiveSimdBatchWidth(options_));

            thread_local std::vector<assembly::jit::KernelSideArgsV6> sides;
            thread_local std::vector<assembly::jit::KernelOutputViewV6> outputs;
            thread_local std::vector<Real> pad_values;
            sides.resize(padded_n);
            outputs.resize(padded_n);
            pad_values.assign(padded_n > 0u ? padded_n - 1u : 0u, Real(0.0));

            sides[0] = side;
            outputs[0] = output;
            for (std::uint32_t i = 1; i < padded_n; ++i) {
                sides[i] = side;
                outputs[i] = makeFunctionalOutputView(&pad_values[static_cast<std::size_t>(i - 1u)],
                                                       static_cast<std::uint32_t>(ctx.numTrialDofs()));
            }

            assembly::jit::CellKernelBatchArgsV1 batch_args;
            batch_args.batch_size = padded_n;
            batch_args.sides = sides.data();
            batch_args.outputs = outputs.data();
            callJIT(addr_, &batch_args);
        } else {
            assembly::jit::CellKernelArgsV6 args;
            args.side = side;
            args.output = output;
            callJIT(addr_, &args);
        }
        return value;
    } catch (const std::exception& e) {
        markRuntimeFailureOnce("evaluateCellTotal", e.what());
        return fallback_->evaluateCellTotal(ctx);
    } catch (...) {
        markRuntimeFailureOnce("evaluateCellTotal", "unknown exception");
        return fallback_->evaluateCellTotal(ctx);
    }
}

bool JITFunctionalKernelWrapper::supportsCellTotalBatch() const noexcept
{
    return domain_ == Domain::Cell;
}

void JITFunctionalKernelWrapper::evaluateCellTotalBatch(
    std::span<const assembly::AssemblyContext* const> contexts,
    std::span<Real> totals)
{
    const std::size_t n = std::min(contexts.size(), totals.size());
    if (n == 0u) {
        return;
    }
    if (domain_ != Domain::Cell) {
        fallback_->evaluateCellTotalBatch(contexts.first(n), totals.first(n));
        return;
    }

    const assembly::AssemblyContext* first_ctx = nullptr;
    for (std::size_t i = 0; i < n; ++i) {
        if (contexts[i] != nullptr) {
            first_ctx = contexts[i];
            break;
        }
    }
    if (first_ctx == nullptr) {
        std::fill_n(totals.begin(), n, Real(0.0));
        return;
    }

    maybeCompile(*first_ctx);
    if (!canUseJIT()) {
        fallback_->evaluateCellTotalBatch(contexts.first(n), totals.first(n));
        return;
    }

    try {
        if (options_.vectorize) {
            const auto simd_w = effectiveSimdBatchWidth(options_);
            const auto padded_n = (simd_w >= 2u)
                ? ((n + simd_w - 1u) / simd_w) * simd_w
                : n;

            thread_local std::vector<assembly::jit::KernelSideArgsV6> sides;
            thread_local std::vector<assembly::jit::KernelOutputViewV6> outputs;
            thread_local std::vector<Real> pad_values;
            sides.resize(padded_n);
            outputs.resize(padded_n);
            pad_values.assign(std::max(padded_n > n ? padded_n - n : 0u, std::size_t{1}), Real(0.0));

            std::size_t fill_src = static_cast<std::size_t>(-1);
            for (std::size_t i = 0; i < n; ++i) {
                totals[i] = 0.0;
                if (contexts[i] == nullptr) {
                    sides[i] = {};
                    outputs[i] = {};
                    continue;
                }
                sides[i] = packFunctionalSide(*contexts[i]);
                outputs[i] = makeFunctionalOutputView(&totals[i],
                                                       static_cast<std::uint32_t>(contexts[i]->numTrialDofs()));
                if (fill_src == static_cast<std::size_t>(-1)) {
                    fill_src = i;
                }
            }

            FE_THROW_IF(fill_src == static_cast<std::size_t>(-1), InvalidArgumentException,
                        "JITFunctionalKernelWrapper::evaluateCellTotalBatch: no valid contexts");

            for (std::size_t i = n; i < padded_n; ++i) {
                sides[i] = sides[fill_src];
                outputs[i] = makeFunctionalOutputView(&pad_values[i - n],
                                                       static_cast<std::uint32_t>(contexts[fill_src]->numTrialDofs()));
            }
            for (std::size_t i = 0; i < n; ++i) {
                if (contexts[i] == nullptr) {
                    sides[i] = sides[fill_src];
                    outputs[i] = makeFunctionalOutputView(&pad_values[0],
                                                           static_cast<std::uint32_t>(contexts[fill_src]->numTrialDofs()));
                }
            }

            assembly::jit::CellKernelBatchArgsV1 batch_args;
            batch_args.batch_size = static_cast<std::uint32_t>(padded_n);
            batch_args.sides = sides.data();
            batch_args.outputs = outputs.data();
            callJIT(addr_, &batch_args);
        } else {
            for (std::size_t i = 0; i < n; ++i) {
                totals[i] = 0.0;
                if (contexts[i] == nullptr) {
                    continue;
                }
                assembly::jit::CellKernelArgsV6 args;
                args.side = packFunctionalSide(*contexts[i]);
                args.output = makeFunctionalOutputView(&totals[i],
                                                       static_cast<std::uint32_t>(contexts[i]->numTrialDofs()));
                callJIT(addr_, &args);
            }
        }
    } catch (const std::exception& e) {
        markRuntimeFailureOnce("evaluateCellTotalBatch", e.what());
        fallback_->evaluateCellTotalBatch(contexts.first(n), totals.first(n));
    } catch (...) {
        markRuntimeFailureOnce("evaluateCellTotalBatch", "unknown exception");
        fallback_->evaluateCellTotalBatch(contexts.first(n), totals.first(n));
    }
}

Real JITFunctionalKernelWrapper::evaluateBoundaryFaceTotal(const assembly::AssemblyContext& ctx,
                                                           int boundary_marker)
{
    if (domain_ != Domain::BoundaryFace) {
        return fallback_->evaluateBoundaryFaceTotal(ctx, boundary_marker);
    }

    maybeCompile(ctx);
    if (!canUseJIT()) {
        return fallback_->evaluateBoundaryFaceTotal(ctx, boundary_marker);
    }

    try {
        thread_local assembly::KernelOutput out;
        out.reserve(/*n_test=*/1, /*n_trial=*/ctx.numTrialDofs(), /*need_matrix=*/false, /*need_vector=*/true);
        out.clear();

        const auto checks = assembly::jit::PackingChecks{.validate_alignment = true};
        auto args = assembly::jit::packBoundaryFaceKernelArgsV6(ctx, boundary_marker, out, checks);

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

void JITFunctionalKernelWrapper::maybeCompile(const assembly::AssemblyContext& ctx)
{
    if (!options_.enable) {
        return;
    }

    std::lock_guard<std::mutex> lock(jit_mutex_);
    if (compiled_ || attempted_) {
        return;
    }
    attempted_ = true;
    compiled_dim_ = static_cast<std::uint32_t>(ctx.dimension());

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

    const auto r = compiler_->compileFunctional(integrand_, jit_domain, compiled_dim_, vopt);
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

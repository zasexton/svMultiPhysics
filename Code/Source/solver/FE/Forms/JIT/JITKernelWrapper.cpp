/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Forms/JIT/JITKernelWrapper.h"

#include "Assembly/JIT/KernelArgs.h"
#include "Core/FEException.h"
#include "Core/Logger.h"

#include "Forms/FormKernels.h"
#include "Forms/JIT/JITCompiler.h"
#include "Forms/JIT/JITValidation.h"

#include <cstddef>
#include <utility>

#ifndef SVMP_FE_ENABLE_LLVM_JIT
#define SVMP_FE_ENABLE_LLVM_JIT 0
#endif

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

} // namespace

JITKernelWrapper::JITKernelWrapper(std::shared_ptr<assembly::AssemblyKernel> fallback,
                                   JITOptions options)
    : fallback_(std::move(fallback)),
      options_(std::move(options))
{
    FE_CHECK_NOT_NULL(fallback_.get(), "JITKernelWrapper: fallback kernel");

    if (dynamic_cast<const FormKernel*>(fallback_.get()) != nullptr) {
        kind_ = WrappedKind::FormKernel;
    } else if (dynamic_cast<const LinearFormKernel*>(fallback_.get()) != nullptr) {
        kind_ = WrappedKind::LinearFormKernel;
    } else if (dynamic_cast<const SymbolicNonlinearFormKernel*>(fallback_.get()) != nullptr) {
        kind_ = WrappedKind::SymbolicNonlinearFormKernel;
    } else if (dynamic_cast<const NonlinearFormKernel*>(fallback_.get()) != nullptr) {
        kind_ = WrappedKind::NonlinearFormKernel;
    } else {
        kind_ = WrappedKind::Unknown;
    }
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
    if (!canUseJIT()) {
        fallback_->computeCell(ctx, output);
        return;
    }

    try {
        const auto checks = assembly::jit::PackingChecks{.validate_alignment = true};

    if (kind_ == WrappedKind::FormKernel) {
        const auto* k = dynamic_cast<const FormKernel*>(fallback_.get());
        if (!k) {
            fallback_->computeCell(ctx, output);
            return;
        }

        const bool want_matrix = (k->ir().kind() == FormKind::Bilinear);
        const bool want_vector = (k->ir().kind() == FormKind::Linear);

        output.reserve(ctx.numTestDofs(), ctx.numTrialDofs(), want_matrix, want_vector);
        output.clear();

        const auto& updates = k->inlinedStateUpdates().cell;
        if (!updates.empty()) {
            for (LocalIndex q = 0; q < ctx.numQuadraturePoints(); ++q) {
                applyInlinedMaterialStateUpdatesReal(ctx, nullptr, k->ir().kind(),
                                                     k->constitutiveStateLayout(),
                                                     updates,
                                                     Side::Minus, q);
            }
	        }

	        const auto args = assembly::jit::packCellKernelArgsV6(ctx, output, checks);
	        const auto disp = getSpecializedDispatch(KernelRole::Form, k->ir(), IntegralDomain::Cell, ctx, nullptr);
	        const auto& compiled = disp ? *disp : compiled_form_;
	        callJIT(compiled.cell, &args);

        output.has_matrix = want_matrix;
        output.has_vector = want_vector;
        return;
    }

    if (kind_ == WrappedKind::LinearFormKernel) {
        const auto* k = dynamic_cast<const LinearFormKernel*>(fallback_.get());
        if (!k) {
            fallback_->computeCell(ctx, output);
            return;
        }

        const bool want_matrix = !k->isVectorOnly();
        const bool want_vector = !k->isMatrixOnly();

        output.reserve(ctx.numTestDofs(), ctx.numTrialDofs(), want_matrix, want_vector);
        output.clear();

        const auto& updates = k->inlinedStateUpdates().cell;
        if (!updates.empty()) {
            for (LocalIndex q = 0; q < ctx.numQuadraturePoints(); ++q) {
                applyInlinedMaterialStateUpdatesReal(ctx, nullptr, FormKind::Bilinear,
                                                     k->constitutiveStateLayout(),
                                                     updates,
                                                     Side::Minus, q);
            }
	        }

	        // 1) Jacobian (bilinear part).
	        const auto disp_bi =
	            getSpecializedDispatch(KernelRole::Bilinear, k->bilinearIR(), IntegralDomain::Cell, ctx, nullptr);
	        const auto& compiled_bi = disp_bi ? *disp_bi : compiled_bilinear_;
	        if (want_matrix) {
	            const auto args_bi = assembly::jit::packCellKernelArgsV6(ctx, output, checks);
	            callJIT(compiled_bi.cell, &args_bi);
	        }

	        // 2) Residual vector = (linear part) + (K*u).
	        if (want_vector) {
            const auto coeffs = ctx.solutionCoefficients();
	        FE_THROW_IF(coeffs.size() < static_cast<std::size_t>(ctx.numTrialDofs()), InvalidArgumentException,
	                        "JITKernelWrapper(LinearFormKernel)::computeCell: missing solution coefficients");

	            if (has_compiled_linear_ && k->linearIR().has_value()) {
	                const auto disp_lin =
	                    getSpecializedDispatch(KernelRole::Linear, *k->linearIR(), IntegralDomain::Cell, ctx, nullptr);
	                const auto& compiled_lin = disp_lin ? *disp_lin : compiled_linear_;
	                const auto args_lin = assembly::jit::packCellKernelArgsV6(ctx, output, checks);
	                callJIT(compiled_lin.cell, &args_lin);
	            }

            // K*u contribution.
            if (want_matrix) {
                for (LocalIndex i = 0; i < ctx.numTestDofs(); ++i) {
                    Real sum = 0.0;
                    for (LocalIndex j = 0; j < ctx.numTrialDofs(); ++j) {
                        sum += output.matrixEntry(i, j) * coeffs[static_cast<std::size_t>(j)];
                    }
                    output.vectorEntry(i) += sum;
                }
	            } else {
	                assembly::KernelOutput tmp;
	                tmp.reserve(ctx.numTestDofs(), ctx.numTrialDofs(), /*need_matrix=*/true, /*need_vector=*/false);
	                tmp.clear();

	                const auto args_bi = assembly::jit::packCellKernelArgsV6(ctx, tmp, checks);
	                callJIT(compiled_bi.cell, &args_bi);

                for (LocalIndex i = 0; i < ctx.numTestDofs(); ++i) {
                    Real sum = 0.0;
                    for (LocalIndex j = 0; j < ctx.numTrialDofs(); ++j) {
                        sum += tmp.matrixEntry(i, j) * coeffs[static_cast<std::size_t>(j)];
                    }
                    output.vectorEntry(i) += sum;
                }
            }
        }

        output.has_matrix = want_matrix;
        output.has_vector = want_vector;
        return;
    }

    if (kind_ == WrappedKind::SymbolicNonlinearFormKernel) {
        const auto* k = dynamic_cast<const SymbolicNonlinearFormKernel*>(fallback_.get());
        if (!k) {
            fallback_->computeCell(ctx, output);
            return;
        }

        const bool want_matrix = !k->isVectorOnly();
        const bool want_vector = !k->isMatrixOnly();

        output.reserve(ctx.numTestDofs(), ctx.numTrialDofs(), want_matrix, want_vector);
        output.clear();

        const auto& updates = k->inlinedStateUpdates().cell;
        if (!updates.empty()) {
            for (LocalIndex q = 0; q < ctx.numQuadraturePoints(); ++q) {
                applyInlinedMaterialStateUpdatesReal(ctx, nullptr, FormKind::Residual,
                                                     k->constitutiveStateLayout(),
                                                     updates,
                                                     Side::Minus, q);
            }
        }

	        if (want_matrix) {
	            if (compiled_tangent_.cell == 0) {
	                fallback_->computeCell(ctx, output);
	                return;
	            }
	            const auto args = assembly::jit::packCellKernelArgsV6(ctx, output, checks);
	            const auto disp =
	                getSpecializedDispatch(KernelRole::Tangent, k->tangentIR(), IntegralDomain::Cell, ctx, nullptr);
	            const auto& compiled = disp ? *disp : compiled_tangent_;
	            callJIT(compiled.cell, &args);
	        }
	        if (want_vector) {
	            if (compiled_residual_.cell == 0) {
	                fallback_->computeCell(ctx, output);
	                return;
	            }
	            const auto args = assembly::jit::packCellKernelArgsV6(ctx, output, checks);
	            const auto disp =
	                getSpecializedDispatch(KernelRole::Residual, k->residualIR(), IntegralDomain::Cell, ctx, nullptr);
	            const auto& compiled = disp ? *disp : compiled_residual_;
	            callJIT(compiled.cell, &args);
	        }

        output.has_matrix = want_matrix;
        output.has_vector = want_vector;
        return;
    }

    fallback_->computeCell(ctx, output);
    } catch (const std::exception& e) {
        markRuntimeFailureOnce("computeCell", e.what());
        fallback_->computeCell(ctx, output);
    } catch (...) {
        markRuntimeFailureOnce("computeCell", "unknown exception");
        fallback_->computeCell(ctx, output);
    }
}

void JITKernelWrapper::computeCellBatch(std::span<const assembly::AssemblyContext* const> contexts,
                                        std::span<assembly::KernelOutput> outputs)
{
    const std::size_t n = (contexts.size() < outputs.size()) ? contexts.size() : outputs.size();
    if (n == 0u) {
        return;
    }

    maybeCompile();
    if (!canUseJIT()) {
        if (fallback_->supportsCellBatch()) {
            fallback_->computeCellBatch(
                std::span<const assembly::AssemblyContext* const>(contexts.data(), n),
                std::span<assembly::KernelOutput>(outputs.data(), n));
            return;
        }

        for (std::size_t idx = 0; idx < n; ++idx) {
            if (contexts[idx] == nullptr) {
                continue;
            }
            fallback_->computeCell(*contexts[idx], outputs[idx]);
        }
        return;
    }

    // --- Batch fast-path: hoist per-batch invariant work ---
    //
    // All elements in a batch share the same topology, DOF counts, and kernel
    // configuration.  The per-element computeCell() redundantly performs:
    //   - kind_ dispatch + dynamic_cast (identical every element)
    //   - getSpecializedDispatch hash lookup (same key every element)
    //   - output.reserve() with potential allocation (same sizes every element)
    //   - individual try/catch per element
    //
    // The fast-path below does each of these once per batch, then runs a tight
    // pack-and-call loop for each element.

    // Find the first non-null context to derive batch-invariant properties.
    const assembly::AssemblyContext* first_ctx = nullptr;
    for (std::size_t idx = 0; idx < n; ++idx) {
        if (contexts[idx] != nullptr) {
            first_ctx = contexts[idx];
            break;
        }
    }
    if (first_ctx == nullptr) {
        return;
    }

    try {
        const auto checks = assembly::jit::PackingChecks{.validate_alignment = true};

        // ---- FormKernel (pure bilinear or pure linear) ----
        if (kind_ == WrappedKind::FormKernel) {
            const auto* k = dynamic_cast<const FormKernel*>(fallback_.get());
            if (k) {
                const bool want_matrix = (k->ir().kind() == FormKind::Bilinear);
                const bool want_vector = (k->ir().kind() == FormKind::Linear);
                const auto& updates = k->inlinedStateUpdates().cell;
                const bool has_updates = !updates.empty();

                // Resolve specialization once for the whole batch.
                const auto disp = getSpecializedDispatch(
                    KernelRole::Form, k->ir(), IntegralDomain::Cell, *first_ctx, nullptr);
                const auto& compiled = disp ? *disp : compiled_form_;

                // Pre-reserve all outputs (Opt B): allocates to correct size
                // so the per-element loop only needs to clear.
                const auto n_test = first_ctx->numTestDofs();
                const auto n_trial = first_ctx->numTrialDofs();
                for (std::size_t idx = 0; idx < n; ++idx) {
                    outputs[idx].reserve(n_test, n_trial, want_matrix, want_vector);
                }

                // Tight per-element loop.
                for (std::size_t idx = 0; idx < n; ++idx) {
                    if (contexts[idx] == nullptr) {
                        continue;
                    }
                    const auto& ctx = *contexts[idx];
                    auto& output = outputs[idx];
                    output.clear();

                    if (has_updates) {
                        for (LocalIndex q = 0; q < ctx.numQuadraturePoints(); ++q) {
                            applyInlinedMaterialStateUpdatesReal(
                                ctx, nullptr, k->ir().kind(),
                                k->constitutiveStateLayout(), updates, Side::Minus, q);
                        }
                    }

                    const auto args = assembly::jit::packCellKernelArgsV6(ctx, output, checks);
                    callJIT(compiled.cell, &args);

                    output.has_matrix = want_matrix;
                    output.has_vector = want_vector;
                }
                return;
            }
        }

        // ---- LinearFormKernel (bilinear + optional linear + K*u) ----
        if (kind_ == WrappedKind::LinearFormKernel) {
            const auto* k = dynamic_cast<const LinearFormKernel*>(fallback_.get());
            if (k) {
                const bool want_matrix = !k->isVectorOnly();
                const bool want_vector = !k->isMatrixOnly();
                const auto& updates = k->inlinedStateUpdates().cell;
                const bool has_updates = !updates.empty();

                // Resolve specializations once for the whole batch.
                const auto disp_bi = getSpecializedDispatch(
                    KernelRole::Bilinear, k->bilinearIR(), IntegralDomain::Cell, *first_ctx, nullptr);
                const auto& compiled_bi = disp_bi ? *disp_bi : compiled_bilinear_;

                std::shared_ptr<const CompiledDispatch> disp_lin;
                const CompiledDispatch* compiled_lin_ptr = nullptr;
                if (has_compiled_linear_ && k->linearIR().has_value()) {
                    disp_lin = getSpecializedDispatch(
                        KernelRole::Linear, *k->linearIR(), IntegralDomain::Cell, *first_ctx, nullptr);
                    compiled_lin_ptr = disp_lin ? disp_lin.get() : &compiled_linear_;
                }

                // Pre-reserve all outputs.
                const auto n_test = first_ctx->numTestDofs();
                const auto n_trial = first_ctx->numTrialDofs();
                for (std::size_t idx = 0; idx < n; ++idx) {
                    outputs[idx].reserve(n_test, n_trial, want_matrix, want_vector);
                }

                // Scratch output for K*u when matrix-only mode needs a temporary.
                assembly::KernelOutput tmp;
                if (want_vector && !want_matrix) {
                    tmp.reserve(n_test, n_trial, /*need_matrix=*/true, /*need_vector=*/false);
                }

                // Tight per-element loop.
                for (std::size_t idx = 0; idx < n; ++idx) {
                    if (contexts[idx] == nullptr) {
                        continue;
                    }
                    const auto& ctx = *contexts[idx];
                    auto& output = outputs[idx];
                    output.clear();

                    if (has_updates) {
                        for (LocalIndex q = 0; q < ctx.numQuadraturePoints(); ++q) {
                            applyInlinedMaterialStateUpdatesReal(
                                ctx, nullptr, FormKind::Bilinear,
                                k->constitutiveStateLayout(), updates, Side::Minus, q);
                        }
                    }

                    // 1) Jacobian (bilinear part).
                    if (want_matrix) {
                        const auto args_bi = assembly::jit::packCellKernelArgsV6(ctx, output, checks);
                        callJIT(compiled_bi.cell, &args_bi);
                    }

                    // 2) Residual vector = (linear part) + (K*u).
                    if (want_vector) {
                        const auto coeffs = ctx.solutionCoefficients();
                        FE_THROW_IF(coeffs.size() < static_cast<std::size_t>(ctx.numTrialDofs()),
                                    InvalidArgumentException,
                                    "JITKernelWrapper(LinearFormKernel)::computeCellBatch: "
                                    "missing solution coefficients");

                        if (compiled_lin_ptr != nullptr) {
                            const auto args_lin = assembly::jit::packCellKernelArgsV6(ctx, output, checks);
                            callJIT(compiled_lin_ptr->cell, &args_lin);
                        }

                        // K*u contribution.
                        if (want_matrix) {
                            for (LocalIndex i = 0; i < ctx.numTestDofs(); ++i) {
                                Real sum = 0.0;
                                for (LocalIndex j = 0; j < ctx.numTrialDofs(); ++j) {
                                    sum += output.matrixEntry(i, j) *
                                           coeffs[static_cast<std::size_t>(j)];
                                }
                                output.vectorEntry(i) += sum;
                            }
                        } else {
                            tmp.clear();
                            const auto args_bi = assembly::jit::packCellKernelArgsV6(ctx, tmp, checks);
                            callJIT(compiled_bi.cell, &args_bi);

                            for (LocalIndex i = 0; i < ctx.numTestDofs(); ++i) {
                                Real sum = 0.0;
                                for (LocalIndex j = 0; j < ctx.numTrialDofs(); ++j) {
                                    sum += tmp.matrixEntry(i, j) *
                                           coeffs[static_cast<std::size_t>(j)];
                                }
                                output.vectorEntry(i) += sum;
                            }
                        }
                    }

                    output.has_matrix = want_matrix;
                    output.has_vector = want_vector;
                }
                return;
            }
        }

        // ---- SymbolicNonlinearFormKernel (tangent + residual) ----
        if (kind_ == WrappedKind::SymbolicNonlinearFormKernel) {
            const auto* k = dynamic_cast<const SymbolicNonlinearFormKernel*>(fallback_.get());
            if (k) {
                const bool want_matrix = !k->isVectorOnly();
                const bool want_vector = !k->isMatrixOnly();

                // Verify compiled addresses are available before committing to
                // the batch fast-path.
                const bool can_batch = (!want_matrix || compiled_tangent_.cell != 0) &&
                                       (!want_vector || compiled_residual_.cell != 0);

                if (can_batch) {
                    const auto& updates = k->inlinedStateUpdates().cell;
                    const bool has_updates = !updates.empty();

                    // Resolve specializations once for the whole batch.
                    std::shared_ptr<const CompiledDispatch> disp_tan;
                    const CompiledDispatch* compiled_tan_ptr = &compiled_tangent_;
                    if (want_matrix) {
                        disp_tan = getSpecializedDispatch(
                            KernelRole::Tangent, k->tangentIR(), IntegralDomain::Cell, *first_ctx, nullptr);
                        if (disp_tan) compiled_tan_ptr = disp_tan.get();
                    }

                    std::shared_ptr<const CompiledDispatch> disp_res;
                    const CompiledDispatch* compiled_res_ptr = &compiled_residual_;
                    if (want_vector) {
                        disp_res = getSpecializedDispatch(
                            KernelRole::Residual, k->residualIR(), IntegralDomain::Cell, *first_ctx, nullptr);
                        if (disp_res) compiled_res_ptr = disp_res.get();
                    }

                    // Pre-reserve all outputs.
                    const auto n_test = first_ctx->numTestDofs();
                    const auto n_trial = first_ctx->numTrialDofs();
                    for (std::size_t idx = 0; idx < n; ++idx) {
                        outputs[idx].reserve(n_test, n_trial, want_matrix, want_vector);
                    }

                    // Tight per-element loop.
                    for (std::size_t idx = 0; idx < n; ++idx) {
                        if (contexts[idx] == nullptr) {
                            continue;
                        }
                        const auto& ctx = *contexts[idx];
                        auto& output = outputs[idx];
                        output.clear();

                        if (has_updates) {
                            for (LocalIndex q = 0; q < ctx.numQuadraturePoints(); ++q) {
                                applyInlinedMaterialStateUpdatesReal(
                                    ctx, nullptr, FormKind::Residual,
                                    k->constitutiveStateLayout(), updates, Side::Minus, q);
                            }
                        }

                        const auto args = assembly::jit::packCellKernelArgsV6(ctx, output, checks);
                        if (want_matrix) {
                            callJIT(compiled_tan_ptr->cell, &args);
                        }
                        if (want_vector) {
                            callJIT(compiled_res_ptr->cell, &args);
                        }

                        output.has_matrix = want_matrix;
                        output.has_vector = want_vector;
                    }
                    return;
                }
                // can_batch == false: fall through to per-element fallback.
            }
        }

    } catch (const std::exception& e) {
        markRuntimeFailureOnce("computeCellBatch", e.what());
    } catch (...) {
        markRuntimeFailureOnce("computeCellBatch", "unknown exception");
    }

    // Fallback: per-element dispatch (unknown kind, failed cast, or exception).
    for (std::size_t idx = 0; idx < n; ++idx) {
        if (contexts[idx] == nullptr) {
            continue;
        }
        computeCell(*contexts[idx], outputs[idx]);
    }
}

void JITKernelWrapper::computeBoundaryFace(const assembly::AssemblyContext& ctx,
                                           int boundary_marker,
                                           assembly::KernelOutput& output)
{
    maybeCompile();
    if (!canUseJIT()) {
        fallback_->computeBoundaryFace(ctx, boundary_marker, output);
        return;
    }

    try {
        const auto checks = assembly::jit::PackingChecks{.validate_alignment = true};

    if (kind_ == WrappedKind::FormKernel) {
        const auto* k = dynamic_cast<const FormKernel*>(fallback_.get());
        if (!k) {
            fallback_->computeBoundaryFace(ctx, boundary_marker, output);
            return;
        }

        const bool want_matrix = (k->ir().kind() == FormKind::Bilinear);
        const bool want_vector = (k->ir().kind() == FormKind::Linear);

        output.reserve(ctx.numTestDofs(), ctx.numTrialDofs(), want_matrix, want_vector);
        output.clear();

        const auto& updates_all = k->inlinedStateUpdates().boundary_all;
        const auto* updates_marker = [&]() -> const std::vector<MaterialStateUpdate>* {
            const auto it = k->inlinedStateUpdates().boundary_by_marker.find(boundary_marker);
            return (it == k->inlinedStateUpdates().boundary_by_marker.end()) ? nullptr : &it->second;
        }();

        if (!updates_all.empty() || (updates_marker != nullptr && !updates_marker->empty())) {
            for (LocalIndex q = 0; q < ctx.numQuadraturePoints(); ++q) {
                if (!updates_all.empty()) {
                    applyInlinedMaterialStateUpdatesReal(ctx, nullptr, k->ir().kind(),
                                                         k->constitutiveStateLayout(),
                                                         updates_all,
                                                         Side::Minus, q);
                }
                if (updates_marker != nullptr && !updates_marker->empty()) {
                    applyInlinedMaterialStateUpdatesReal(ctx, nullptr, k->ir().kind(),
                                                         k->constitutiveStateLayout(),
                                                         *updates_marker,
                                                         Side::Minus, q);
                }
            }
	        }

	        const auto args = assembly::jit::packBoundaryFaceKernelArgsV6(ctx, boundary_marker, output, checks);
	        const auto disp =
	            getSpecializedDispatch(KernelRole::Form, k->ir(), IntegralDomain::Boundary, ctx, nullptr);
	        const auto& compiled = disp ? *disp : compiled_form_;

	        callJIT(compiled.boundary_all, &args);
	        if (const auto it = compiled.boundary_by_marker.find(boundary_marker);
	            it != compiled.boundary_by_marker.end()) {
	            callJIT(it->second, &args);
	        }

        output.has_matrix = want_matrix;
        output.has_vector = want_vector;
        return;
    }

    if (kind_ == WrappedKind::LinearFormKernel) {
        const auto* k = dynamic_cast<const LinearFormKernel*>(fallback_.get());
        if (!k) {
            fallback_->computeBoundaryFace(ctx, boundary_marker, output);
            return;
        }

        const bool want_matrix = !k->isVectorOnly();
        const bool want_vector = !k->isMatrixOnly();

        output.reserve(ctx.numTestDofs(), ctx.numTrialDofs(), want_matrix, want_vector);
        output.clear();

        const auto& updates_all = k->inlinedStateUpdates().boundary_all;
        const auto* updates_marker = [&]() -> const std::vector<MaterialStateUpdate>* {
            const auto it = k->inlinedStateUpdates().boundary_by_marker.find(boundary_marker);
            return (it == k->inlinedStateUpdates().boundary_by_marker.end()) ? nullptr : &it->second;
        }();

        if (!updates_all.empty() || (updates_marker != nullptr && !updates_marker->empty())) {
            for (LocalIndex q = 0; q < ctx.numQuadraturePoints(); ++q) {
                if (!updates_all.empty()) {
                    applyInlinedMaterialStateUpdatesReal(ctx, nullptr, FormKind::Bilinear,
                                                         k->constitutiveStateLayout(),
                                                         updates_all,
                                                         Side::Minus, q);
                }
                if (updates_marker != nullptr && !updates_marker->empty()) {
                    applyInlinedMaterialStateUpdatesReal(ctx, nullptr, FormKind::Bilinear,
                                                         k->constitutiveStateLayout(),
                                                         *updates_marker,
                                                         Side::Minus, q);
                }
            }
	        }

	        const auto disp_bi =
	            getSpecializedDispatch(KernelRole::Bilinear, k->bilinearIR(), IntegralDomain::Boundary, ctx, nullptr);
	        const auto& compiled_bi = disp_bi ? *disp_bi : compiled_bilinear_;

	        // 1) Jacobian (bilinear boundary part).
	        if (want_matrix) {
	            const auto args_bi = assembly::jit::packBoundaryFaceKernelArgsV6(ctx, boundary_marker, output, checks);
	            callJIT(compiled_bi.boundary_all, &args_bi);
	            if (const auto it = compiled_bi.boundary_by_marker.find(boundary_marker);
	                it != compiled_bi.boundary_by_marker.end()) {
	                callJIT(it->second, &args_bi);
	            }
	        }

        // 2) Residual vector = (linear boundary part) + (K*u).
        if (want_vector) {
            const auto coeffs = ctx.solutionCoefficients();
            FE_THROW_IF(coeffs.size() < static_cast<std::size_t>(ctx.numTrialDofs()), InvalidArgumentException,
                        "JITKernelWrapper(LinearFormKernel)::computeBoundaryFace: missing solution coefficients");

	            if (has_compiled_linear_ && k->linearIR().has_value()) {
	                const auto disp_lin = getSpecializedDispatch(KernelRole::Linear,
	                                                             *k->linearIR(),
	                                                             IntegralDomain::Boundary,
	                                                             ctx,
	                                                             nullptr);
	                const auto& compiled_lin = disp_lin ? *disp_lin : compiled_linear_;
	                const auto args_lin = assembly::jit::packBoundaryFaceKernelArgsV6(ctx, boundary_marker, output, checks);
	                callJIT(compiled_lin.boundary_all, &args_lin);
	                if (const auto it = compiled_lin.boundary_by_marker.find(boundary_marker);
	                    it != compiled_lin.boundary_by_marker.end()) {
	                    callJIT(it->second, &args_lin);
	                }
	            }

            if (want_matrix) {
                for (LocalIndex i = 0; i < ctx.numTestDofs(); ++i) {
                    Real sum = 0.0;
                    for (LocalIndex j = 0; j < ctx.numTrialDofs(); ++j) {
                        sum += output.matrixEntry(i, j) * coeffs[static_cast<std::size_t>(j)];
                    }
                    output.vectorEntry(i) += sum;
                }
            } else {
                assembly::KernelOutput tmp;
                tmp.reserve(ctx.numTestDofs(), ctx.numTrialDofs(), /*need_matrix=*/true, /*need_vector=*/false);
                tmp.clear();

	                const auto args_bi = assembly::jit::packBoundaryFaceKernelArgsV6(ctx, boundary_marker, tmp, checks);
	                callJIT(compiled_bi.boundary_all, &args_bi);
	                if (const auto it = compiled_bi.boundary_by_marker.find(boundary_marker);
	                    it != compiled_bi.boundary_by_marker.end()) {
	                    callJIT(it->second, &args_bi);
	                }

                for (LocalIndex i = 0; i < ctx.numTestDofs(); ++i) {
                    Real sum = 0.0;
                    for (LocalIndex j = 0; j < ctx.numTrialDofs(); ++j) {
                        sum += tmp.matrixEntry(i, j) * coeffs[static_cast<std::size_t>(j)];
                    }
                    output.vectorEntry(i) += sum;
                }
            }
        }

        output.has_matrix = want_matrix;
        output.has_vector = want_vector;
        return;
    }

    if (kind_ == WrappedKind::SymbolicNonlinearFormKernel) {
        const auto* k = dynamic_cast<const SymbolicNonlinearFormKernel*>(fallback_.get());
        if (!k) {
            fallback_->computeBoundaryFace(ctx, boundary_marker, output);
            return;
        }

        const bool want_matrix = !k->isVectorOnly();
        const bool want_vector = !k->isMatrixOnly();

        output.reserve(ctx.numTestDofs(), ctx.numTrialDofs(), want_matrix, want_vector);
        output.clear();

        const auto& updates_all = k->inlinedStateUpdates().boundary_all;
        const auto* updates_marker = [&]() -> const std::vector<MaterialStateUpdate>* {
            const auto it = k->inlinedStateUpdates().boundary_by_marker.find(boundary_marker);
            return (it == k->inlinedStateUpdates().boundary_by_marker.end()) ? nullptr : &it->second;
        }();

        if (!updates_all.empty() || (updates_marker != nullptr && !updates_marker->empty())) {
            for (LocalIndex q = 0; q < ctx.numQuadraturePoints(); ++q) {
                if (!updates_all.empty()) {
                    applyInlinedMaterialStateUpdatesReal(ctx, nullptr, FormKind::Residual,
                                                         k->constitutiveStateLayout(),
                                                         updates_all,
                                                         Side::Minus, q);
                }
                if (updates_marker != nullptr && !updates_marker->empty()) {
                    applyInlinedMaterialStateUpdatesReal(ctx, nullptr, FormKind::Residual,
                                                         k->constitutiveStateLayout(),
                                                         *updates_marker,
                                                         Side::Minus, q);
                }
            }
        }

	        const auto args = assembly::jit::packBoundaryFaceKernelArgsV6(ctx, boundary_marker, output, checks);

	        if (want_matrix) {
	            const auto disp =
	                getSpecializedDispatch(KernelRole::Tangent, k->tangentIR(), IntegralDomain::Boundary, ctx, nullptr);
	            const auto& compiled = disp ? *disp : compiled_tangent_;

	            callJIT(compiled.boundary_all, &args);
	            if (const auto it = compiled.boundary_by_marker.find(boundary_marker);
	                it != compiled.boundary_by_marker.end()) {
	                callJIT(it->second, &args);
	            }
	        }

	        if (want_vector) {
	            const auto disp = getSpecializedDispatch(
	                KernelRole::Residual, k->residualIR(), IntegralDomain::Boundary, ctx, nullptr);
	            const auto& compiled = disp ? *disp : compiled_residual_;

	            callJIT(compiled.boundary_all, &args);
	            if (const auto it = compiled.boundary_by_marker.find(boundary_marker);
	                it != compiled.boundary_by_marker.end()) {
	                callJIT(it->second, &args);
	            }
	        }

        output.has_matrix = want_matrix;
        output.has_vector = want_vector;
        return;
    }

    fallback_->computeBoundaryFace(ctx, boundary_marker, output);
    } catch (const std::exception& e) {
        markRuntimeFailureOnce("computeBoundaryFace", e.what());
        fallback_->computeBoundaryFace(ctx, boundary_marker, output);
    } catch (...) {
        markRuntimeFailureOnce("computeBoundaryFace", "unknown exception");
        fallback_->computeBoundaryFace(ctx, boundary_marker, output);
    }
}

void JITKernelWrapper::computeInteriorFace(const assembly::AssemblyContext& ctx_minus,
                                           const assembly::AssemblyContext& ctx_plus,
                                           assembly::KernelOutput& output_minus,
                                           assembly::KernelOutput& output_plus,
                                           assembly::KernelOutput& coupling_minus_plus,
                                           assembly::KernelOutput& coupling_plus_minus)
{
    maybeCompile();
    if (!canUseJIT()) {
        fallback_->computeInteriorFace(ctx_minus, ctx_plus,
                                       output_minus, output_plus,
                                       coupling_minus_plus, coupling_plus_minus);
        return;
    }

    try {
        const auto checks = assembly::jit::PackingChecks{.validate_alignment = true};

    if (kind_ == WrappedKind::FormKernel) {
        const auto* k = dynamic_cast<const FormKernel*>(fallback_.get());
        if (!k || k->ir().kind() != FormKind::Bilinear) {
            fallback_->computeInteriorFace(ctx_minus, ctx_plus,
                                           output_minus, output_plus,
                                           coupling_minus_plus, coupling_plus_minus);
            return;
        }

        const auto n_test_minus = ctx_minus.numTestDofs();
        const auto n_trial_minus = ctx_minus.numTrialDofs();
        const auto n_test_plus = ctx_plus.numTestDofs();
        const auto n_trial_plus = ctx_plus.numTrialDofs();

        output_minus.reserve(n_test_minus, n_trial_minus, true, false);
        output_plus.reserve(n_test_plus, n_trial_plus, true, false);
        coupling_minus_plus.reserve(n_test_minus, n_trial_plus, true, false);
        coupling_plus_minus.reserve(n_test_plus, n_trial_minus, true, false);

        output_minus.clear();
        output_plus.clear();
        coupling_minus_plus.clear();
        coupling_plus_minus.clear();

        const auto& updates = k->inlinedStateUpdates().interior_face;
        if (!updates.empty()) {
            for (LocalIndex q = 0; q < ctx_minus.numQuadraturePoints(); ++q) {
                applyInlinedMaterialStateUpdatesReal(ctx_minus, &ctx_plus, FormKind::Bilinear,
                                                     k->constitutiveStateLayout(),
                                                     updates,
                                                     Side::Minus, q);

                const auto* base_minus = ctx_minus.materialStateWorkBase();
                const auto* base_plus = ctx_plus.materialStateWorkBase();
                if (base_plus != nullptr && base_plus != base_minus) {
                    applyInlinedMaterialStateUpdatesReal(ctx_minus, &ctx_plus, FormKind::Bilinear,
                                                         k->constitutiveStateLayout(),
                                                         updates,
                                                         Side::Plus, q);
                }
            }
        }

	        const auto args =
	            assembly::jit::packInteriorFaceKernelArgsV6(ctx_minus, ctx_plus,
	                                                        output_minus, output_plus,
	                                                        coupling_minus_plus, coupling_plus_minus,
	                                                        checks);
	        const auto disp =
	            getSpecializedDispatch(KernelRole::Form, k->ir(), IntegralDomain::InteriorFace, ctx_minus, &ctx_plus);
	        const auto& compiled = disp ? *disp : compiled_form_;
	        callJIT(compiled.interior_face, &args);

        output_minus.has_matrix = true;
        output_minus.has_vector = false;
        output_plus.has_matrix = true;
        output_plus.has_vector = false;
        coupling_minus_plus.has_matrix = true;
        coupling_minus_plus.has_vector = false;
        coupling_plus_minus.has_matrix = true;
        coupling_plus_minus.has_vector = false;
        return;
    }

    if (kind_ == WrappedKind::SymbolicNonlinearFormKernel) {
        const auto* k = dynamic_cast<const SymbolicNonlinearFormKernel*>(fallback_.get());
        if (!k) {
            fallback_->computeInteriorFace(ctx_minus, ctx_plus,
                                           output_minus, output_plus,
                                           coupling_minus_plus, coupling_plus_minus);
            return;
        }

        const bool want_matrix = !k->isVectorOnly();
        const bool want_vector = !k->isMatrixOnly();

        const auto n_test_minus = ctx_minus.numTestDofs();
        const auto n_trial_minus = ctx_minus.numTrialDofs();
        const auto n_test_plus = ctx_plus.numTestDofs();
        const auto n_trial_plus = ctx_plus.numTrialDofs();

        output_minus.reserve(n_test_minus, n_trial_minus, want_matrix, want_vector);
        output_plus.reserve(n_test_plus, n_trial_plus, want_matrix, want_vector);
        coupling_minus_plus.reserve(n_test_minus, n_trial_plus, want_matrix, /*need_vector=*/false);
        coupling_plus_minus.reserve(n_test_plus, n_trial_minus, want_matrix, /*need_vector=*/false);

        output_minus.clear();
        output_plus.clear();
        coupling_minus_plus.clear();
        coupling_plus_minus.clear();

        const auto& updates = k->inlinedStateUpdates().interior_face;
        if (!updates.empty()) {
            for (LocalIndex q = 0; q < ctx_minus.numQuadraturePoints(); ++q) {
                applyInlinedMaterialStateUpdatesReal(ctx_minus, &ctx_plus, FormKind::Residual,
                                                     k->constitutiveStateLayout(),
                                                     updates,
                                                     Side::Minus, q);

                const auto* base_minus = ctx_minus.materialStateWorkBase();
                const auto* base_plus = ctx_plus.materialStateWorkBase();
                if (base_plus != nullptr && base_plus != base_minus) {
                    applyInlinedMaterialStateUpdatesReal(ctx_minus, &ctx_plus, FormKind::Residual,
                                                         k->constitutiveStateLayout(),
                                                         updates,
                                                         Side::Plus, q);
                }
            }
        }

	        const auto args =
	            assembly::jit::packInteriorFaceKernelArgsV6(ctx_minus, ctx_plus,
	                                                        output_minus, output_plus,
	                                                        coupling_minus_plus, coupling_plus_minus,
	                                                        checks);
	        if (want_matrix) {
	            const auto disp =
	                getSpecializedDispatch(KernelRole::Tangent, k->tangentIR(), IntegralDomain::InteriorFace, ctx_minus, &ctx_plus);
	            const auto& compiled = disp ? *disp : compiled_tangent_;
	            callJIT(compiled.interior_face, &args);
	        }
	        if (want_vector) {
	            const auto disp =
	                getSpecializedDispatch(KernelRole::Residual, k->residualIR(), IntegralDomain::InteriorFace, ctx_minus, &ctx_plus);
	            const auto& compiled = disp ? *disp : compiled_residual_;
	            callJIT(compiled.interior_face, &args);
	        }

        output_minus.has_matrix = want_matrix;
        output_minus.has_vector = want_vector;
        output_plus.has_matrix = want_matrix;
        output_plus.has_vector = want_vector;
        coupling_minus_plus.has_matrix = want_matrix;
        coupling_minus_plus.has_vector = false;
        coupling_plus_minus.has_matrix = want_matrix;
        coupling_plus_minus.has_vector = false;
        return;
    }

    fallback_->computeInteriorFace(ctx_minus, ctx_plus,
                                   output_minus, output_plus,
                                   coupling_minus_plus, coupling_plus_minus);
    } catch (const std::exception& e) {
        markRuntimeFailureOnce("computeInteriorFace", e.what());
        fallback_->computeInteriorFace(ctx_minus, ctx_plus,
                                       output_minus, output_plus,
                                       coupling_minus_plus, coupling_plus_minus);
    } catch (...) {
        markRuntimeFailureOnce("computeInteriorFace", "unknown exception");
        fallback_->computeInteriorFace(ctx_minus, ctx_plus,
                                       output_minus, output_plus,
                                       coupling_minus_plus, coupling_plus_minus);
    }
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
    if (!canUseJIT()) {
        fallback_->computeInterfaceFace(ctx_minus, ctx_plus, interface_marker,
                                        output_minus, output_plus,
                                        coupling_minus_plus, coupling_plus_minus);
        return;
    }

    try {
        const auto checks = assembly::jit::PackingChecks{.validate_alignment = true};

    if (kind_ == WrappedKind::FormKernel) {
        const auto* k = dynamic_cast<const FormKernel*>(fallback_.get());
        if (!k || k->ir().kind() != FormKind::Bilinear) {
            fallback_->computeInterfaceFace(ctx_minus, ctx_plus, interface_marker,
                                            output_minus, output_plus,
                                            coupling_minus_plus, coupling_plus_minus);
            return;
        }

        const auto n_test_minus = ctx_minus.numTestDofs();
        const auto n_trial_minus = ctx_minus.numTrialDofs();
        const auto n_test_plus = ctx_plus.numTestDofs();
        const auto n_trial_plus = ctx_plus.numTrialDofs();

        output_minus.reserve(n_test_minus, n_trial_minus, true, false);
        output_plus.reserve(n_test_plus, n_trial_plus, true, false);
        coupling_minus_plus.reserve(n_test_minus, n_trial_plus, true, false);
        coupling_plus_minus.reserve(n_test_plus, n_trial_minus, true, false);

        output_minus.clear();
        output_plus.clear();
        coupling_minus_plus.clear();
        coupling_plus_minus.clear();

        const auto& updates = k->inlinedStateUpdates().interface_face;
        if (!updates.empty()) {
            for (LocalIndex q = 0; q < ctx_minus.numQuadraturePoints(); ++q) {
                applyInlinedMaterialStateUpdatesReal(ctx_minus, &ctx_plus, FormKind::Bilinear,
                                                     k->constitutiveStateLayout(),
                                                     updates,
                                                     Side::Minus, q);

                const auto* base_minus = ctx_minus.materialStateWorkBase();
                const auto* base_plus = ctx_plus.materialStateWorkBase();
                if (base_plus != nullptr && base_plus != base_minus) {
                    applyInlinedMaterialStateUpdatesReal(ctx_minus, &ctx_plus, FormKind::Bilinear,
                                                         k->constitutiveStateLayout(),
                                                         updates,
                                                         Side::Plus, q);
                }
            }
        }

	        const auto args =
	            assembly::jit::packInterfaceFaceKernelArgsV6(ctx_minus, ctx_plus, interface_marker,
	                                                         output_minus, output_plus,
	                                                         coupling_minus_plus, coupling_plus_minus,
	                                                         checks);
	        const auto disp =
	            getSpecializedDispatch(KernelRole::Form, k->ir(), IntegralDomain::InterfaceFace, ctx_minus, &ctx_plus);
	        const auto& compiled = disp ? *disp : compiled_form_;

	        callJIT(compiled.interface_all, &args);
	        if (const auto it = compiled.interface_by_marker.find(interface_marker);
	            it != compiled.interface_by_marker.end()) {
	            callJIT(it->second, &args);
	        }

        output_minus.has_matrix = true;
        output_minus.has_vector = false;
        output_plus.has_matrix = true;
        output_plus.has_vector = false;
        coupling_minus_plus.has_matrix = true;
        coupling_minus_plus.has_vector = false;
        coupling_plus_minus.has_matrix = true;
        coupling_plus_minus.has_vector = false;
        return;
    }

    if (kind_ == WrappedKind::SymbolicNonlinearFormKernel) {
        const auto* k = dynamic_cast<const SymbolicNonlinearFormKernel*>(fallback_.get());
        if (!k) {
            fallback_->computeInterfaceFace(ctx_minus, ctx_plus, interface_marker,
                                            output_minus, output_plus,
                                            coupling_minus_plus, coupling_plus_minus);
            return;
        }

        const bool want_matrix = !k->isVectorOnly();
        const bool want_vector = !k->isMatrixOnly();

        const auto n_test_minus = ctx_minus.numTestDofs();
        const auto n_trial_minus = ctx_minus.numTrialDofs();
        const auto n_test_plus = ctx_plus.numTestDofs();
        const auto n_trial_plus = ctx_plus.numTrialDofs();

        output_minus.reserve(n_test_minus, n_trial_minus, want_matrix, want_vector);
        output_plus.reserve(n_test_plus, n_trial_plus, want_matrix, want_vector);
        coupling_minus_plus.reserve(n_test_minus, n_trial_plus, want_matrix, /*need_vector=*/false);
        coupling_plus_minus.reserve(n_test_plus, n_trial_minus, want_matrix, /*need_vector=*/false);

        output_minus.clear();
        output_plus.clear();
        coupling_minus_plus.clear();
        coupling_plus_minus.clear();

        const auto& updates = k->inlinedStateUpdates().interface_face;
        if (!updates.empty()) {
            for (LocalIndex q = 0; q < ctx_minus.numQuadraturePoints(); ++q) {
                applyInlinedMaterialStateUpdatesReal(ctx_minus, &ctx_plus, FormKind::Residual,
                                                     k->constitutiveStateLayout(),
                                                     updates,
                                                     Side::Minus, q);

                const auto* base_minus = ctx_minus.materialStateWorkBase();
                const auto* base_plus = ctx_plus.materialStateWorkBase();
                if (base_plus != nullptr && base_plus != base_minus) {
                    applyInlinedMaterialStateUpdatesReal(ctx_minus, &ctx_plus, FormKind::Residual,
                                                         k->constitutiveStateLayout(),
                                                         updates,
                                                         Side::Plus, q);
                }
            }
        }

        const auto args =
            assembly::jit::packInterfaceFaceKernelArgsV6(ctx_minus, ctx_plus, interface_marker,
                                                         output_minus, output_plus,
                                                         coupling_minus_plus, coupling_plus_minus,
                                                         checks);

	        if (want_matrix) {
	            const auto disp =
	                getSpecializedDispatch(KernelRole::Tangent, k->tangentIR(), IntegralDomain::InterfaceFace, ctx_minus, &ctx_plus);
	            const auto& compiled = disp ? *disp : compiled_tangent_;

	            callJIT(compiled.interface_all, &args);
	            if (const auto it = compiled.interface_by_marker.find(interface_marker);
	                it != compiled.interface_by_marker.end()) {
	                callJIT(it->second, &args);
	            }
	        }

	        if (want_vector) {
	            const auto disp =
	                getSpecializedDispatch(KernelRole::Residual, k->residualIR(), IntegralDomain::InterfaceFace, ctx_minus, &ctx_plus);
	            const auto& compiled = disp ? *disp : compiled_residual_;

	            callJIT(compiled.interface_all, &args);
	            if (const auto it = compiled.interface_by_marker.find(interface_marker);
	                it != compiled.interface_by_marker.end()) {
	                callJIT(it->second, &args);
	            }
	        }

        output_minus.has_matrix = want_matrix;
        output_minus.has_vector = want_vector;
        output_plus.has_matrix = want_matrix;
        output_plus.has_vector = want_vector;
        coupling_minus_plus.has_matrix = want_matrix;
        coupling_minus_plus.has_vector = false;
        coupling_plus_minus.has_matrix = want_matrix;
        coupling_plus_minus.has_vector = false;
        return;
    }

    fallback_->computeInterfaceFace(ctx_minus, ctx_plus, interface_marker,
                                    output_minus, output_plus,
                                    coupling_minus_plus, coupling_plus_minus);
    } catch (const std::exception& e) {
        markRuntimeFailureOnce("computeInterfaceFace", e.what());
        fallback_->computeInterfaceFace(ctx_minus, ctx_plus, interface_marker,
                                        output_minus, output_plus,
                                        coupling_minus_plus, coupling_plus_minus);
    } catch (...) {
        markRuntimeFailureOnce("computeInterfaceFace", "unknown exception");
        fallback_->computeInterfaceFace(ctx_minus, ctx_plus, interface_marker,
                                        output_minus, output_plus,
                                        coupling_minus_plus, coupling_plus_minus);
    }
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

    compiled_form_ = CompiledDispatch{};
    compiled_bilinear_ = CompiledDispatch{};
    compiled_linear_ = CompiledDispatch{};
    compiled_residual_ = CompiledDispatch{};
    compiled_tangent_ = CompiledDispatch{};
    has_compiled_linear_ = false;

    specialized_dispatch_.clear();
    attempted_specializations_.clear();
    warned_specialization_failure_ = false;

    warned_compile_failure_ = false;
    runtime_failed_ = false;
    warned_runtime_failure_ = false;
}

bool JITKernelWrapper::canUseJIT() const noexcept
{
    return options_.enable && compiled_revision_ == revision_ && !runtime_failed_;
}

std::shared_ptr<const JITKernelWrapper::CompiledDispatch> JITKernelWrapper::getSpecializedDispatch(
    KernelRole role,
    const FormIR& ir,
    IntegralDomain domain,
    const assembly::AssemblyContext& ctx_minus,
    const assembly::AssemblyContext* ctx_plus)
{
    if (!options_.enable || !options_.specialization.enable) {
        return nullptr;
    }
    if (!compiler_) {
        return nullptr;
    }

    const bool face_domain = (domain == IntegralDomain::InteriorFace || domain == IntegralDomain::InterfaceFace);
    if (face_domain && ctx_plus == nullptr) {
        return nullptr;
    }

    JITCompileSpecialization spec;
    spec.domain = domain;
    bool any = false;

    if (options_.specialization.specialize_n_qpts) {
        const auto n_qpts_minus = static_cast<std::uint32_t>(ctx_minus.numQuadraturePoints());
        if (n_qpts_minus > 0u && n_qpts_minus <= options_.specialization.max_specialized_n_qpts) {
            spec.n_qpts_minus = n_qpts_minus;
            any = true;
        }

        if (face_domain) {
            const auto n_qpts_plus = static_cast<std::uint32_t>(ctx_plus->numQuadraturePoints());
            if (n_qpts_plus != n_qpts_minus) {
                return nullptr;
            }
            if (n_qpts_plus > 0u && n_qpts_plus <= options_.specialization.max_specialized_n_qpts) {
                spec.n_qpts_plus = n_qpts_plus;
                any = true;
            }
        }
    }

    if (options_.specialization.specialize_dofs) {
        const auto n_test_minus = static_cast<std::uint32_t>(ctx_minus.numTestDofs());
        const auto n_trial_minus = static_cast<std::uint32_t>(ctx_minus.numTrialDofs());
        const bool ok_minus = n_test_minus > 0u && n_trial_minus > 0u &&
                              n_test_minus <= options_.specialization.max_specialized_dofs &&
                              n_trial_minus <= options_.specialization.max_specialized_dofs;

        bool ok_plus = true;
        std::uint32_t n_test_plus = 0;
        std::uint32_t n_trial_plus = 0;
        if (face_domain) {
            n_test_plus = static_cast<std::uint32_t>(ctx_plus->numTestDofs());
            n_trial_plus = static_cast<std::uint32_t>(ctx_plus->numTrialDofs());
            ok_plus = n_test_plus > 0u && n_trial_plus > 0u &&
                      n_test_plus <= options_.specialization.max_specialized_dofs &&
                      n_trial_plus <= options_.specialization.max_specialized_dofs;
        }

        if (ok_minus && ok_plus) {
            spec.n_test_dofs_minus = n_test_minus;
            spec.n_trial_dofs_minus = n_trial_minus;
            any = true;
            if (face_domain) {
                spec.n_test_dofs_plus = n_test_plus;
                spec.n_trial_dofs_plus = n_trial_plus;
            }
        }
    }

    if (!any) {
        return nullptr;
    }

    SpecializationKey key;
    key.role = role;
    key.domain = domain;

    if (spec.n_qpts_minus) {
        key.has_n_qpts_minus = true;
        key.n_qpts_minus = *spec.n_qpts_minus;
    }
    if (spec.n_test_dofs_minus) {
        key.has_n_test_dofs_minus = true;
        key.n_test_dofs_minus = *spec.n_test_dofs_minus;
    }
    if (spec.n_trial_dofs_minus) {
        key.has_n_trial_dofs_minus = true;
        key.n_trial_dofs_minus = *spec.n_trial_dofs_minus;
    }

    if (spec.n_qpts_plus) {
        key.has_n_qpts_plus = true;
        key.n_qpts_plus = *spec.n_qpts_plus;
    }
    if (spec.n_test_dofs_plus) {
        key.has_n_test_dofs_plus = true;
        key.n_test_dofs_plus = *spec.n_test_dofs_plus;
    }
    if (spec.n_trial_dofs_plus) {
        key.has_n_trial_dofs_plus = true;
        key.n_trial_dofs_plus = *spec.n_trial_dofs_plus;
    }

    std::uint64_t my_revision = 0;
    {
        std::lock_guard<std::mutex> lock(jit_mutex_);
        if (compiled_revision_ != revision_) {
            return nullptr;
        }
        my_revision = revision_;

        if (const auto it = specialized_dispatch_.find(key); it != specialized_dispatch_.end()) {
            return it->second;
        }

        if (attempted_specializations_.find(key) != attempted_specializations_.end()) {
            return nullptr;
        }

        std::size_t count = 0;
        for (const auto& [k, _] : specialized_dispatch_) {
            if (k.role == role && k.domain == domain) {
                ++count;
            }
        }
        if (count >= options_.specialization.max_variants_per_kernel) {
            return nullptr;
        }

        attempted_specializations_.insert(key);
    }

    ValidationOptions vopt;
    vopt.strictness = Strictness::AllowExternalCalls;

    const auto r = compiler_->compileSpecialized(ir, spec, vopt);
    if (!r.ok) {
        bool should_warn = false;
        {
            std::lock_guard<std::mutex> lock(jit_mutex_);
            if (!warned_specialization_failure_) {
                warned_specialization_failure_ = true;
                should_warn = true;
            }
        }
        if (should_warn) {
            std::string msg = "JIT: failed to compile specialized variant for kernel '" + fallback_->name() + "'";
            if (!r.message.empty()) {
                msg += ": " + r.message;
            }
            FE_LOG_WARNING(msg);
        }
        return nullptr;
    }

    auto disp = std::make_shared<CompiledDispatch>();
    disp->ok = r.ok;
    disp->cacheable = r.cacheable;
    disp->message = r.message;
    disp->boundary_by_marker.reserve(r.kernels.size());
    disp->interface_by_marker.reserve(r.kernels.size());

    for (const auto& k : r.kernels) {
        switch (k.domain) {
            case IntegralDomain::Cell:
                disp->cell = k.address;
                break;
            case IntegralDomain::Boundary:
                if (k.boundary_marker < 0) {
                    disp->boundary_all = k.address;
                } else {
                    disp->boundary_by_marker[k.boundary_marker] = k.address;
                }
                break;
            case IntegralDomain::InteriorFace:
                disp->interior_face = k.address;
                break;
            case IntegralDomain::InterfaceFace:
                if (k.interface_marker < 0) {
                    disp->interface_all = k.address;
                } else {
                    disp->interface_by_marker[k.interface_marker] = k.address;
                }
                break;
        }
    }

    {
        std::lock_guard<std::mutex> lock(jit_mutex_);
        if (revision_ != my_revision || compiled_revision_ != my_revision) {
            return nullptr;
        }
        specialized_dispatch_[key] = disp;
    }

    return disp;
}

void JITKernelWrapper::markRuntimeFailureOnce(std::string_view where, std::string_view msg) noexcept
{
    std::lock_guard<std::mutex> lock(jit_mutex_);
    runtime_failed_ = true;
    if (warned_runtime_failure_) {
        return;
    }
    warned_runtime_failure_ = true;

    std::string full = "JIT: runtime failure in " + std::string(where) + " for kernel '" + fallback_->name() + "'";
    if (!msg.empty()) {
        full += ": " + std::string(msg);
    }
    FE_LOG_WARNING(full);
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

    // We currently only JIT-accelerate kernels that are backed by FE/Forms IR.
    if (kind_ == WrappedKind::Unknown || kind_ == WrappedKind::NonlinearFormKernel) {
        if (!warned_unavailable_) {
            warned_unavailable_ = true;
            FE_LOG_WARNING("JIT requested for kernel '" + fallback_->name() +
                           "', but this kernel type is not JIT-accelerated yet; using interpreter.");
        }
        return;
    }

    if (!compiler_) {
        compiler_ = JITCompiler::getOrCreate(options_);
    }
    if (!compiler_) {
        if (!warned_unavailable_) {
            warned_unavailable_ = true;
            FE_LOG_WARNING("JIT requested for kernel '" + fallback_->name() +
                           "', but JITCompiler could not be created; using interpreter.");
        }
        return;
    }

    ValidationOptions vopt;
    vopt.strictness = Strictness::AllowExternalCalls;

    const auto fillDispatch = [&](CompiledDispatch& out, const JITCompileResult& r) {
        out = CompiledDispatch{};
        out.ok = r.ok;
        out.cacheable = r.cacheable;
        out.message = r.message;
        out.boundary_by_marker.reserve(r.kernels.size());
        out.interface_by_marker.reserve(r.kernels.size());

        for (const auto& k : r.kernels) {
            switch (k.domain) {
                case IntegralDomain::Cell:
                    out.cell = k.address;
                    break;
                case IntegralDomain::Boundary:
                    if (k.boundary_marker < 0) {
                        out.boundary_all = k.address;
                    } else {
                        out.boundary_by_marker[k.boundary_marker] = k.address;
                    }
                    break;
                case IntegralDomain::InteriorFace:
                    out.interior_face = k.address;
                    break;
                case IntegralDomain::InterfaceFace:
                    if (k.interface_marker < 0) {
                        out.interface_all = k.address;
                    } else {
                        out.interface_by_marker[k.interface_marker] = k.address;
                    }
                    break;
            }
        }
    };

    auto warnCompileFailureOnce = [&](std::string_view what, std::string_view msg) {
        if (warned_compile_failure_) {
            return;
        }
        warned_compile_failure_ = true;
        std::string full = "JIT: failed to compile " + std::string(what) + " for kernel '" + fallback_->name() + "'";
        if (!msg.empty()) {
            full += ": " + std::string(msg);
        }
        FE_LOG_WARNING(full);
    };

    if (kind_ == WrappedKind::FormKernel) {
        const auto* k = dynamic_cast<const FormKernel*>(fallback_.get());
        if (!k) {
            warnCompileFailureOnce("FormKernel", "dynamic_cast failed");
            return;
        }

        const auto r = compiler_->compile(k->ir(), vopt);
        if (!r.ok) {
            warnCompileFailureOnce("FormKernel", r.message);
            return;
        }
        fillDispatch(compiled_form_, r);
        compiled_revision_ = revision_;
        return;
    }

    if (kind_ == WrappedKind::LinearFormKernel) {
        const auto* k = dynamic_cast<const LinearFormKernel*>(fallback_.get());
        if (!k) {
            warnCompileFailureOnce("LinearFormKernel", "dynamic_cast failed");
            return;
        }

        const bool want_matrix = !k->isVectorOnly();
        const bool want_vector = !k->isMatrixOnly();

        const auto r_bi = compiler_->compile(k->bilinearIR(), vopt);
        if (!r_bi.ok) {
            warnCompileFailureOnce("LinearFormKernel(bilinear)", r_bi.message);
            return;
        }
        fillDispatch(compiled_bilinear_, r_bi);

        has_compiled_linear_ = false;
        if (want_vector && k->linearIR().has_value()) {
            const auto r_lin = compiler_->compile(*k->linearIR(), vopt);
            if (!r_lin.ok) {
                warnCompileFailureOnce("LinearFormKernel(linear)", r_lin.message);
                return;
            }
            fillDispatch(compiled_linear_, r_lin);
            has_compiled_linear_ = true;
        }

        compiled_revision_ = revision_;
        return;
    }

    if (kind_ == WrappedKind::SymbolicNonlinearFormKernel) {
        const auto* k = dynamic_cast<const SymbolicNonlinearFormKernel*>(fallback_.get());
        if (!k) {
            warnCompileFailureOnce("SymbolicNonlinearFormKernel", "dynamic_cast failed");
            return;
        }

        const bool want_matrix = !k->isVectorOnly();
        const bool want_vector = !k->isMatrixOnly();

        if (want_vector) {
            const auto r_res = compiler_->compile(k->residualIR(), vopt);
            if (!r_res.ok) {
                warnCompileFailureOnce("SymbolicNonlinearFormKernel(residual)", r_res.message);
                return;
            }
            fillDispatch(compiled_residual_, r_res);
        } else {
            compiled_residual_ = CompiledDispatch{};
        }

        if (want_matrix) {
            const auto r_tan = compiler_->compile(k->tangentIR(), vopt);
            if (!r_tan.ok) {
                warnCompileFailureOnce("SymbolicNonlinearFormKernel(tangent)", r_tan.message);
                return;
            }
            fillDispatch(compiled_tangent_, r_tan);
        } else {
            compiled_tangent_ = CompiledDispatch{};
        }

        compiled_revision_ = revision_;
        return;
    }
}

} // namespace jit
} // namespace forms
} // namespace FE
} // namespace svmp

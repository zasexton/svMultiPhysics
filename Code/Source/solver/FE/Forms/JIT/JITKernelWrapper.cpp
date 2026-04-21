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
#include "Forms/FormCompiler.h"
#include "Forms/SymbolicDifferentiation.h"
#include "Forms/JIT/HardwareProfile.h"
#include "Forms/JIT/JITCompiler.h"
#include "Forms/JIT/JITValidation.h"

#include <cstddef>
#include <cstdlib>
#include <sstream>
#include <utility>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifndef SVMP_FE_ENABLE_LLVM_JIT
#define SVMP_FE_ENABLE_LLVM_JIT 0
#endif

namespace svmp {
namespace FE {
namespace forms {
namespace jit {

namespace {

using JITFn = void (*)(const void*);

struct RequestedKernelOutputs {
    bool matrix{false};
    bool vector{false};
};

[[nodiscard]] RequestedKernelOutputs requestedKernelOutputs(
    const assembly::KernelOutput& output,
    bool can_matrix,
    bool can_vector) noexcept
{
    bool want_matrix = output.has_matrix || !output.local_matrix.empty();
    bool want_vector = output.has_vector || !output.local_vector.empty();
    if (!want_matrix && !want_vector) {
        want_matrix = can_matrix;
        want_vector = can_vector;
    }

    RequestedKernelOutputs requested;
    requested.matrix = can_matrix && want_matrix;
    requested.vector = can_vector && want_vector;
    return requested;
}

[[nodiscard]] bool traceSpecializationEnabled() noexcept
{
    static const bool enabled = [] {
        const char* value = std::getenv("SVMP_JIT_TRACE_SPECIALIZATION");
        return value != nullptr && value[0] != '\0' && value[0] != '0';
    }();
    return enabled;
}

[[nodiscard]] const char* integralDomainName(IntegralDomain domain) noexcept
{
    switch (domain) {
        case IntegralDomain::Cell:
            return "Cell";
        case IntegralDomain::Boundary:
            return "Boundary";
        case IntegralDomain::InteriorFace:
            return "InteriorFace";
        case IntegralDomain::InterfaceFace:
            return "InterfaceFace";
    }
    return "Unknown";
}

[[nodiscard]] bool exprContainsRuntimeCoefficient(const FormExprNode& node) noexcept
{
    if (node.type() == FormExprType::Coefficient) {
        return true;
    }
    for (const auto& child : node.childrenShared()) {
        if (child && exprContainsRuntimeCoefficient(*child)) {
            return true;
        }
    }
    return false;
}

[[nodiscard]] bool domainUsesRuntimeCoefficient(const FormIR& ir,
                                                IntegralDomain domain,
                                                int marker = -1) noexcept
{
    for (const auto& term : ir.terms()) {
        if (term.domain != domain) {
            continue;
        }
        if (domain == IntegralDomain::Boundary &&
            marker >= 0 &&
            term.boundary_marker >= 0 &&
            term.boundary_marker != marker) {
            continue;
        }
        const auto* root = term.integrand.node();
        if (root && exprContainsRuntimeCoefficient(*root)) {
            return true;
        }
    }
    return false;
}

void traceSpecialization(const JITKernelWrapper* wrapper,
                         const assembly::AssemblyKernel& fallback,
                         std::uint64_t revision,
                         std::string_view detail)
{
    if (!traceSpecializationEnabled()) {
        return;
    }

    std::ostringstream oss;
    oss << "JIT specialization trace: wrapper=" << static_cast<const void*>(wrapper)
        << " kernel='" << fallback.name() << "' rev=" << revision;
    if (!detail.empty()) {
        oss << ' ' << detail;
    }
    FE_LOG_INFO(oss.str());
}

inline void callJIT(std::uintptr_t addr, const void* args) noexcept
{
    if (addr == 0 || args == nullptr) {
        return;
    }
    reinterpret_cast<JITFn>(addr)(args);
}

[[nodiscard]] inline assembly::jit::KernelOutputViewV6 makeOutputViewV6(
    assembly::KernelOutput& output,
    Real* matrix_override = nullptr,
    Real* vector_override = nullptr) noexcept
{
    auto view = assembly::jit::detail::packOutputViewV6(output);
    if (matrix_override != nullptr) {
        view.element_matrix = matrix_override;
    }
    if (vector_override != nullptr) {
        view.element_vector = vector_override;
    }
    return view;
}

struct PreparedCellBatchTemplateV6 {
    std::uint32_t abi_version{assembly::jit::kKernelArgsABIVersionV6};
    assembly::jit::KernelSideArgsV6 side{};
};

[[nodiscard]] inline PreparedCellBatchTemplateV6 makePreparedCellBatchTemplateV6(
    const assembly::AssemblyContext& ctx,
    assembly::KernelOutput& output,
    assembly::jit::PackingChecks checks = {}) noexcept
{
    const auto args = assembly::jit::packCellKernelArgsV6(ctx, output, checks);
    PreparedCellBatchTemplateV6 prepared;
    prepared.abi_version = args.abi_version;
    prepared.side = args.side;
    return prepared;
}

/// Patch per-cell fields of a KernelSideArgsV6 that was templated from another
/// element in the same batch.
inline void patchCellSideArgsV6(assembly::jit::KernelSideArgsV6& s,
                                const assembly::AssemblyContext& ctx) noexcept
{
    using assembly::jit::detail::flattenXYZ;
    using assembly::jit::detail::flattenMat3;

    // Per-cell scalar fields
    s.cell_id = ctx.cellId();
    s.cell_domain_id = static_cast<std::int32_t>(ctx.cellDomainId());
    s.cell_diameter = ctx.cellDiameter();
    s.cell_volume = (s.context_type == static_cast<std::uint32_t>(assembly::ContextType::Cell))
                        ? ctx.cellVolume() : Real(0.0);
    s.facet_area = (s.context_type == static_cast<std::uint32_t>(assembly::ContextType::Cell))
                        ? Real(0.0) : ctx.facetArea();

    // Quadrature/geometry pointers
    s.quad_weights = ctx.quadratureWeights().empty() ? nullptr : ctx.quadratureWeights().data();
    s.integration_weights = ctx.integrationWeights().empty() ? nullptr : ctx.integrationWeights().data();
    s.quad_points_xyz = flattenXYZ(ctx.quadraturePoints());
    s.physical_points_xyz = flattenXYZ(ctx.physicalPoints());

    s.jacobians = flattenMat3(ctx.jacobians());
    s.inverse_jacobians = flattenMat3(ctx.inverseJacobians());
    s.jacobian_dets = ctx.jacobianDets().empty() ? nullptr : ctx.jacobianDets().data();
    s.normals_xyz = flattenXYZ(ctx.normals());

    const auto interleaved = ctx.interleavedQPointGeometryRaw();
    s.interleaved_qpoint_geometry = interleaved.empty() ? nullptr : interleaved.data();
    s.interleaved_qpoint_geometry_stride_reals =
        s.interleaved_qpoint_geometry == nullptr ? 0u
                                                 : assembly::AssemblyContext::kInterleavedQPointGeometryStride;

    // Basis table pointers
    s.test_basis_values = ctx.testBasisValuesRaw().empty() ? nullptr : ctx.testBasisValuesRaw().data();
    s.test_phys_gradients_xyz = flattenXYZ(ctx.testPhysicalGradientsRaw());
    s.test_phys_hessians = flattenMat3(ctx.testPhysicalHessiansRaw());

    s.trial_basis_values = ctx.trialBasisValuesRaw().empty() ? nullptr : ctx.trialBasisValuesRaw().data();
    s.trial_phys_gradients_xyz = flattenXYZ(ctx.trialPhysicalGradientsRaw());
    s.trial_phys_hessians = flattenMat3(ctx.trialPhysicalHessiansRaw());

    s.test_basis_vector_values_xyz = flattenXYZ(ctx.testBasisVectorValuesRaw());
    s.test_basis_vector_jacobians = flattenMat3(ctx.testBasisVectorJacobiansRaw());
    s.test_basis_curls_xyz = flattenXYZ(ctx.testBasisCurlsRaw());
    s.test_basis_divergences = ctx.testBasisDivergencesRaw().empty() ? nullptr : ctx.testBasisDivergencesRaw().data();

    s.trial_basis_vector_values_xyz = flattenXYZ(ctx.trialBasisVectorValuesRaw());
    s.trial_basis_vector_jacobians = flattenMat3(ctx.trialBasisVectorJacobiansRaw());
    s.trial_basis_curls_xyz = flattenXYZ(ctx.trialBasisCurlsRaw());
    s.trial_basis_divergences = ctx.trialBasisDivergencesRaw().empty() ? nullptr : ctx.trialBasisDivergencesRaw().data();

    // Solution coefficient pointers
    s.solution_coefficients = ctx.solutionCoefficients().empty() ? nullptr : ctx.solutionCoefficients().data();

    if (s.num_history_steps > 0u) {
        s.history_weights = ctx.historyWeights().data();
    }
    for (std::size_t i = 0; i < static_cast<std::size_t>(s.num_previous_solutions); ++i) {
        const auto coeffs = ctx.previousSolutionCoefficientsRaw(static_cast<int>(i + 1));
        const Real* coeff_ptr = coeffs.empty() ? nullptr : coeffs.data();
        s.previous_solution_coefficients[i] = coeff_ptr;
        s.history_solution_coefficients[i] = coeff_ptr;
    }

    // Field solution / parameter pointers
    const auto field_table = ctx.jitFieldSolutionTable();
    s.field_solutions = field_table.empty() ? nullptr : field_table.data();
    s.num_field_solutions = static_cast<std::uint32_t>(field_table.size());

    s.jit_constants = ctx.jitConstants().empty() ? nullptr : ctx.jitConstants().data();
    s.num_jit_constants = static_cast<std::uint32_t>(ctx.jitConstants().size());

    s.coupled_integrals = ctx.coupledIntegrals().empty() ? nullptr : ctx.coupledIntegrals().data();
    s.num_coupled_integrals = static_cast<std::uint32_t>(ctx.coupledIntegrals().size());

    s.coupled_aux = ctx.coupledAuxState().empty() ? nullptr : ctx.coupledAuxState().data();
    s.num_coupled_aux = static_cast<std::uint32_t>(ctx.coupledAuxState().size());

    // Material state pointers (sizes/alignment are batch-invariant)
    s.material_state_old_base = ctx.materialStateOldBase();
    s.material_state_work_base = ctx.materialStateWorkBase();
    s.user_data = ctx.userData();
}

inline void prepareCellBatchEntryV6(
    const PreparedCellBatchTemplateV6& prepared,
    const assembly::AssemblyContext& ctx,
    assembly::KernelOutput& output,
    assembly::jit::KernelSideArgsV6& side_out,
    assembly::jit::KernelOutputViewV6& output_out,
    Real* matrix_override = nullptr,
    Real* vector_override = nullptr) noexcept
{
    side_out = prepared.side;
    patchCellSideArgsV6(side_out, ctx);
    output_out = makeOutputViewV6(output, matrix_override, vector_override);
}

inline void prepareCellArgsV6(const PreparedCellBatchTemplateV6& prepared,
                              assembly::jit::CellKernelArgsV6& args,
                              const assembly::AssemblyContext& ctx,
                              assembly::KernelOutput& output,
                              Real* matrix_override = nullptr,
                              Real* vector_override = nullptr) noexcept
{
    args.abi_version = prepared.abi_version;
    args.side = prepared.side;
    patchCellSideArgsV6(args.side, ctx);
    args.output = makeOutputViewV6(output, matrix_override, vector_override);
}

inline void accumulateDenseMatVecRaw(const Real* matrix_ptr,
                                     LocalIndex n_test,
                                     LocalIndex n_trial,
                                     const Real* x_ptr,
                                     Real* y_ptr) noexcept
{
    const auto trial_count = static_cast<std::size_t>(n_trial);
    const auto test_count = static_cast<std::size_t>(n_test);

    for (std::size_t i = 0; i < test_count; ++i) {
        const Real* row = matrix_ptr + i * trial_count;
        Real sum = 0.0;
        for (std::size_t j = 0; j < trial_count; ++j) {
            sum += row[j] * x_ptr[j];
        }
        y_ptr[i] += sum;
    }
}

inline void accumulateDenseMatVec(std::span<const Real> matrix,
                                  LocalIndex n_test,
                                  LocalIndex n_trial,
                                  std::span<const Real> x,
                                  std::span<Real> y) noexcept
{
    accumulateDenseMatVecRaw(matrix.data(), n_test, n_trial, x.data(), y.data());
}

inline void accumulateKernelOutputMatrixTimesSolution(assembly::KernelOutput& output,
                                                      const Real* coeffs,
                                                      LocalIndex n_test,
                                                      LocalIndex n_trial) noexcept
{
    accumulateDenseMatVecRaw(
        output.local_matrix.data(),
        n_test,
        n_trial,
        coeffs,
        output.local_vector.data());
}

inline void accumulateKernelMatrixTimesSolution(const assembly::AssemblyContext& ctx,
                                                std::span<const Real> matrix,
                                                std::span<Real> vector,
                                                LocalIndex n_test,
                                                LocalIndex n_trial,
                                                const char* where)
{
    const auto coeffs = ctx.solutionCoefficients();
    FE_THROW_IF(coeffs.size() < static_cast<std::size_t>(n_trial),
                InvalidArgumentException,
                where);
    accumulateDenseMatVec(matrix, n_test, n_trial, coeffs.first(static_cast<std::size_t>(n_trial)), vector);
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

    if (kind_ == WrappedKind::SymbolicNonlinearFormKernel ||
        kind_ == WrappedKind::NonlinearFormKernel) {
        // Nonlinear JIT kernels are currently more robust on the scalar ABI
        // path than on the SIMD-padded batch path used by linear kernels.
        options_.vectorize = false;
        options_.simd_batch = false;
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
    markDirty("resolveParameterSlots");
    maybeCompile();
}

void JITKernelWrapper::resolveInlinableConstitutives()
{
    fallback_->resolveInlinableConstitutives();
    markDirty("resolveInlinableConstitutives");
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
        const auto checks = assembly::jit::PackingChecks{.validate_alignment = false};

        // When vectorize=true, JIT kernels expect CellKernelBatchArgsV1 (batch ABI).
        // Wrap callJIT to pack a batch-of-1 with stack-local scratch so computeCell
        // is thread-safe (no shared member vectors).
        auto callJITCell = [&](std::uintptr_t addr, const assembly::jit::CellKernelArgsV6& v6_args) {
            if (options_.vectorize) {
                // SIMD batch: pad to simd_w elements (the JIT kernel reads simd_w at a time).
                const auto sw = options_.simd_batch
                    ? static_cast<std::uint32_t>(jit::hardwareProfile().simdDoubles())
                    : 1u;
                const auto pn = (sw >= 2 && options_.simd_batch) ? sw : 1u;

                thread_local std::vector<assembly::jit::KernelSideArgsV6> local_sides;
                thread_local std::vector<assembly::jit::KernelOutputViewV6> local_outs;
                thread_local std::vector<Real> pad_m_cell, pad_v_cell;
                local_sides.resize(pn);
                local_outs.resize(pn);

                local_sides[0] = v6_args.side;
                local_outs[0] = v6_args.output;
                if (pn > 1) {
                    pad_m_cell.assign(
                        static_cast<std::size_t>(v6_args.output.n_test_dofs) *
                        static_cast<std::size_t>(v6_args.output.n_trial_dofs), 0.0);
                    pad_v_cell.assign(v6_args.output.n_test_dofs, 0.0);
                    for (std::uint32_t pi = 1; pi < pn; ++pi) {
                        local_sides[pi] = v6_args.side;
                        local_outs[pi] = v6_args.output;
                        local_outs[pi].element_matrix = pad_m_cell.data();
                        local_outs[pi].element_vector = pad_v_cell.data();
                    }
                }

                assembly::jit::CellKernelBatchArgsV1 batch_args;
                batch_args.batch_size = pn;
                batch_args.sides = local_sides.data();
                batch_args.outputs = local_outs.data();
                callJIT(addr, &batch_args);
            } else {
                callJIT(addr, &v6_args);
            }
        };

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
	        callJITCell(compiled.cell, args);

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
	            callJITCell(compiled_bi.cell, args_bi);
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
	                callJITCell(compiled_lin.cell, args_lin);
	            }

            // K*u contribution.
            if (want_matrix) {
                accumulateKernelOutputMatrixTimesSolution(
                    output, coeffs.data(), ctx.numTestDofs(), ctx.numTrialDofs());
	            } else {
		                assembly::KernelOutput tmp;
		                tmp.reserve(ctx.numTestDofs(), ctx.numTrialDofs(), /*need_matrix=*/true, /*need_vector=*/false);
		                tmp.clear();

	                const auto args_bi = assembly::jit::packCellKernelArgsV6(ctx, tmp, checks);
	                callJITCell(compiled_bi.cell, args_bi);

                accumulateDenseMatVecRaw(
                    tmp.local_matrix.data(),
                    ctx.numTestDofs(),
                    ctx.numTrialDofs(),
                    coeffs.data(),
                    output.local_vector.data());
            }
        }

        output.has_matrix = want_matrix;
        output.has_vector = want_vector;
        return;
    }

    // ---- Nonlinear kernels (Symbolic or AD-based fallback) ----
    if (kind_ == WrappedKind::SymbolicNonlinearFormKernel || kind_ == WrappedKind::NonlinearFormKernel) {
        const auto* k_sym = dynamic_cast<const SymbolicNonlinearFormKernel*>(fallback_.get());
        const auto* k_nl = dynamic_cast<const NonlinearFormKernel*>(fallback_.get());

        if (!k_sym && !k_nl) {
            fallback_->computeCell(ctx, output);
            return;
        }

        const auto requested = requestedKernelOutputs(
            output,
            k_sym ? !k_sym->isVectorOnly() : !k_nl->isVectorOnly(),
            k_sym ? !k_sym->isMatrixOnly() : !k_nl->isMatrixOnly());
        const bool want_matrix = requested.matrix;
        const bool want_vector = requested.vector;
        const auto& updates = k_sym ? k_sym->inlinedStateUpdates().cell : k_nl->inlinedStateUpdates().cell;
        const auto* state_layout = k_sym ? k_sym->constitutiveStateLayout() : k_nl->constitutiveStateLayout();
        const auto& residual_ir = k_sym ? k_sym->residualIR() : k_nl->residualIR();

        output.reserve(ctx.numTestDofs(), ctx.numTrialDofs(), want_matrix, want_vector);
        output.clear();

        if (!updates.empty()) {
            for (LocalIndex q = 0; q < ctx.numQuadraturePoints(); ++q) {
                applyInlinedMaterialStateUpdatesReal(ctx, nullptr, FormKind::Residual,
                                                     state_layout, updates,
                                                     Side::Minus, q);
            }
        }

        // Symbolic nonlinear kernels use the exact tangent and residual
        // kernels directly. The old fused cell-kernel experiment was removed
        // after it regressed instruction-cache pressure and is no longer part
        // of the production dispatch path.
        if (want_matrix) {
            if (compiled_tangent_.cell == 0) {
                fallback_->computeCell(ctx, output);
                return;
            }
            const auto args = assembly::jit::packCellKernelArgsV6(ctx, output, checks);
            const auto disp = k_sym ? getSpecializedDispatch(KernelRole::Tangent, k_sym->tangentIR(), IntegralDomain::Cell, ctx, nullptr)
                                    : nullptr;
            const auto& compiled = disp ? *disp : compiled_tangent_;
            callJITCell(compiled.cell, args);
        }

        if (want_vector) {
            if (compiled_residual_.cell == 0) {
                fallback_->computeCell(ctx, output);
                return;
            }
            const auto args = assembly::jit::packCellKernelArgsV6(ctx, output, checks);
            const auto disp = getSpecializedDispatch(KernelRole::Residual, residual_ir, IntegralDomain::Cell, ctx, nullptr);
            const auto& compiled = disp ? *disp : compiled_residual_;
            callJITCell(compiled.cell, args);
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
    std::size_t first_idx = 0;
    for (std::size_t idx = 0; idx < n; ++idx) {
        if (contexts[idx] != nullptr) {
            first_ctx = contexts[idx];
            first_idx = idx;
            break;
        }
    }
    if (first_ctx == nullptr) {
        return;
    }

    try {
        const auto checks = assembly::jit::PackingChecks{.validate_alignment = false};

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

                // Pre-reserve all outputs: allocates to correct size
                // so the per-element loop only needs to clear.
                const auto n_test = first_ctx->numTestDofs();
                const auto n_trial = first_ctx->numTrialDofs();
                for (std::size_t idx = 0; idx < n; ++idx) {
                    outputs[idx].reserveNoZero(n_test, n_trial, want_matrix, want_vector);
                }

                // Pack template from first element (time integration stencils
                // are batch-invariant and the most expensive part to recompute).
                const auto prepared =
                    makePreparedCellBatchTemplateV6(*first_ctx, outputs[first_idx], checks);

                // Tight per-element loop.
                if (options_.vectorize) {
                    // SIMD batch padding: round up to next multiple of simd_w
                    // so the JIT kernel's SIMD loop doesn't read past the end.
                    const auto simd_w = options_.simd_batch
                        ? static_cast<std::size_t>(jit::hardwareProfile().simdDoubles())
                        : std::size_t{1};
                    const auto padded_n = (simd_w >= 2 && options_.simd_batch)
                        ? ((n + simd_w - 1) / simd_w) * simd_w
                        : n;

                    // Thread-local scratch: avoids per-call heap allocation.
                    // resize() is a no-op when capacity >= padded_n (hot path).
                    thread_local std::vector<assembly::jit::KernelSideArgsV6> batch_sides;
                    thread_local std::vector<assembly::jit::KernelOutputViewV6> batch_outputs;
                    batch_sides.resize(padded_n);
                    batch_outputs.resize(padded_n);
                    // Zero-init only the padding slots (hot elements overwritten below).
                    for (std::size_t idx = n; idx < padded_n; ++idx) {
                        batch_sides[idx] = {};
                        batch_outputs[idx] = {};
                    }

                    // Scratch output for padding elements (writes are harmless).
                    thread_local std::vector<Real> pad_matrix_scratch, pad_vector_scratch;

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

                        prepareCellBatchEntryV6(
                            prepared, ctx, output, batch_sides[idx], batch_outputs[idx]);

                        output.has_matrix = want_matrix;
                        output.has_vector = want_vector;
                    }

                    // Fill holes and pad for SIMD batch.
                    {
                        pad_matrix_scratch.assign(
                            static_cast<std::size_t>(n_test) * static_cast<std::size_t>(n_trial), 0.0);
                        pad_vector_scratch.assign(static_cast<std::size_t>(n_test), 0.0);
                        std::size_t fill_src = 0;
                        for (std::size_t idx = 0; idx < n; ++idx) {
                            if (batch_sides[idx].integration_weights != nullptr) {
                                fill_src = idx; break;
                            }
                        }
                        for (std::size_t idx = 0; idx < padded_n; ++idx) {
                            if (batch_sides[idx].integration_weights == nullptr) {
                                batch_sides[idx] = batch_sides[fill_src];
                                batch_outputs[idx].element_matrix = pad_matrix_scratch.data();
                                batch_outputs[idx].element_vector = pad_vector_scratch.data();
                                batch_outputs[idx].n_test_dofs = batch_outputs[fill_src].n_test_dofs;
                                batch_outputs[idx].n_trial_dofs = batch_outputs[fill_src].n_trial_dofs;
                            }
                        }
                    }

                    assembly::jit::CellKernelBatchArgsV1 batch_args;
                    batch_args.batch_size = static_cast<std::uint32_t>(padded_n);
                    batch_args.sides = batch_sides.data();
                    batch_args.outputs = batch_outputs.data();

                    callJIT(compiled.cell, &batch_args);

                } else {
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

                        assembly::jit::CellKernelArgsV6 args;
                        prepareCellArgsV6(prepared, args, ctx, output);
                        callJIT(compiled.cell, &args);

                        output.has_matrix = want_matrix;
                        output.has_vector = want_vector;
                    }
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
                    outputs[idx].reserveNoZero(n_test, n_trial, want_matrix, want_vector);
                }

                // Scratch output for K*u when matrix-only mode needs a temporary.
                assembly::KernelOutput tmp;
                if (want_vector && !want_matrix) {
                    tmp.reserve(n_test, n_trial, /*need_matrix=*/true, /*need_vector=*/false);
                }

                // Pack template from first element (avoids stencil recomputation).
                const auto prepared =
                    makePreparedCellBatchTemplateV6(*first_ctx, outputs[first_idx], checks);

                // Tight per-element loop.
                if (options_.vectorize) {
                    // SIMD batch padding (same as FormKernel path above).
                    const auto simd_w = options_.simd_batch
                        ? static_cast<std::size_t>(jit::hardwareProfile().simdDoubles())
                        : std::size_t{1};
                    const auto padded_n = (simd_w >= 2 && options_.simd_batch)
                        ? ((n + simd_w - 1) / simd_w) * simd_w
                        : n;

                    // Thread-local scratch: avoids per-call heap allocation.
                    thread_local std::vector<assembly::jit::KernelSideArgsV6> batch_sides;
                    thread_local std::vector<assembly::jit::KernelOutputViewV6> batch_outputs;
                    batch_sides.resize(padded_n);
                    batch_outputs.resize(padded_n);

                    thread_local std::vector<Real> pad_matrix_scratch2, pad_vector_scratch2;
                    thread_local std::vector<Real> batch_matrix_scratch2;
                    const std::size_t matrix_stride =
                        static_cast<std::size_t>(n_test) * static_cast<std::size_t>(n_trial);
                    if (want_vector && !want_matrix) {
                        batch_matrix_scratch2.assign(matrix_stride * padded_n, 0.0);
                    }

                    for (std::size_t idx = 0; idx < n; ++idx) {
                        if (contexts[idx] == nullptr) continue;
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

                        prepareCellBatchEntryV6(
                            prepared,
                            ctx,
                            output,
                            batch_sides[idx],
                            batch_outputs[idx],
                            (want_vector && !want_matrix)
                                ? (batch_matrix_scratch2.data() + idx * matrix_stride)
                                : nullptr);
                    }

                    // Fill holes and pad remaining slots for SIMD batch.
                    if (padded_n > 0u && n > 0u) {
                        pad_matrix_scratch2.assign(
                            static_cast<std::size_t>(n_test) * static_cast<std::size_t>(n_trial), 0.0);
                        pad_vector_scratch2.assign(static_cast<std::size_t>(n_test), 0.0);
                        std::size_t fill_src = 0;
                        for (std::size_t idx = 0; idx < n; ++idx) {
                            if (batch_sides[idx].integration_weights != nullptr) {
                                fill_src = idx;
                                break;
                            }
                        }
                        for (std::size_t idx = 0; idx < padded_n; ++idx) {
                            if (idx >= n || batch_sides[idx].integration_weights == nullptr) {
                                batch_sides[idx] = batch_sides[fill_src];
                                batch_outputs[idx] = batch_outputs[fill_src];
                                batch_outputs[idx].element_matrix = pad_matrix_scratch2.data();
                                batch_outputs[idx].element_vector = pad_vector_scratch2.data();
                            }
                        }
                    }

                    assembly::jit::CellKernelBatchArgsV1 batch_args;
                    batch_args.batch_size = static_cast<std::uint32_t>(padded_n);
                    batch_args.sides = batch_sides.data();
                    batch_args.outputs = batch_outputs.data();

                    if (want_matrix) {
                        callJIT(compiled_bi.cell, &batch_args);
                    }

                    if (want_vector) {
                        if (compiled_lin_ptr != nullptr) {
                            callJIT(compiled_lin_ptr->cell, &batch_args);
                        }

                        // K*u contribution.
                        for (std::size_t idx = 0; idx < n; ++idx) {
                            if (contexts[idx] == nullptr) continue;
                            const auto& ctx = *contexts[idx];
                            auto& output = outputs[idx];
                            const auto coeffs = ctx.solutionCoefficients();

                            if (want_matrix) {
                                accumulateKernelOutputMatrixTimesSolution(
                                    output, coeffs.data(), ctx.numTestDofs(), ctx.numTrialDofs());
                            } else {
                                const Real* matrix_ptr =
                                    batch_matrix_scratch2.data() + idx * matrix_stride;
                                accumulateKernelMatrixTimesSolution(
                                    ctx,
                                    std::span<const Real>(matrix_ptr, matrix_stride),
                                    std::span<Real>(output.local_vector),
                                    ctx.numTestDofs(),
                                    ctx.numTrialDofs(),
                                    "JITKernelWrapper(LinearFormKernel)::computeCellBatch: "
                                    "missing solution coefficients");
                            }
                        }
                    }

                    for (std::size_t idx = 0; idx < n; ++idx) {
                        if (contexts[idx] == nullptr) continue;
                        outputs[idx].has_matrix = want_matrix;
                        outputs[idx].has_vector = want_vector;
                    }
                } else {
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

                        // Patch per-cell fields from template.
                        assembly::jit::CellKernelArgsV6 args;
                        prepareCellArgsV6(prepared, args, ctx, output);

                        // 1) Jacobian (bilinear part).
                        if (want_matrix) {
                            callJIT(compiled_bi.cell, &args);
                        }

                        // 2) Residual vector = (linear part) + (K*u).
                        if (want_vector) {
                            const auto coeffs = ctx.solutionCoefficients();
                            FE_THROW_IF(coeffs.size() < static_cast<std::size_t>(ctx.numTrialDofs()),
                                        InvalidArgumentException,
                                        "JITKernelWrapper(LinearFormKernel)::computeCellBatch: "
                                        "missing solution coefficients");

                            if (compiled_lin_ptr != nullptr) {
                                callJIT(compiled_lin_ptr->cell, &args);
                            }

                            // K*u contribution.
                            if (want_matrix) {
                                accumulateKernelOutputMatrixTimesSolution(
                                    output, coeffs.data(), ctx.numTestDofs(), ctx.numTrialDofs());
                            } else {
                                tmp.clear();
                                args.output = makeOutputViewV6(tmp);
                                callJIT(compiled_bi.cell, &args);

                                accumulateDenseMatVecRaw(
                                    tmp.local_matrix.data(),
                                    ctx.numTestDofs(),
                                    ctx.numTrialDofs(),
                                    coeffs.data(),
                                    output.local_vector.data());
                            }
                        }

                        output.has_matrix = want_matrix;
                        output.has_vector = want_vector;
                    }
                }
                return;
            }
        }

        // ---- Nonlinear kernels (Symbolic or AD-based fallback) ----
        if (kind_ == WrappedKind::SymbolicNonlinearFormKernel || kind_ == WrappedKind::NonlinearFormKernel) {
            const auto* k_sym = dynamic_cast<const SymbolicNonlinearFormKernel*>(fallback_.get());
            const auto* k_nl = dynamic_cast<const NonlinearFormKernel*>(fallback_.get());

            if (k_sym || k_nl) {
                const auto requested = requestedKernelOutputs(
                    outputs[first_idx],
                    k_sym ? !k_sym->isVectorOnly() : !k_nl->isVectorOnly(),
                    k_sym ? !k_sym->isMatrixOnly() : !k_nl->isMatrixOnly());
                const bool want_matrix = requested.matrix;
                const bool want_vector = requested.vector;
                const auto& updates = k_sym ? k_sym->inlinedStateUpdates().cell : k_nl->inlinedStateUpdates().cell;
                const auto* state_layout = k_sym ? k_sym->constitutiveStateLayout() : k_nl->constitutiveStateLayout();
                const auto& residual_ir = k_sym ? k_sym->residualIR() : k_nl->residualIR();

                // Verify compiled addresses are available before committing to
                // the batch fast-path.
                const bool can_batch = (!want_matrix || compiled_tangent_.cell != 0) &&
                                       (!want_vector || compiled_residual_.cell != 0);

                if (can_batch) {
                    const bool has_updates = !updates.empty();

                    // Resolve specializations once for the whole batch.
                    std::shared_ptr<const CompiledDispatch> disp_tan;
                    const CompiledDispatch* compiled_tan_ptr = &compiled_tangent_;
                    if (want_matrix) {
                        // Note: for NonlinearFormKernel, we don't have the tangent IR for specialization
                        // lookups readily available here if it was created inside maybeCompile.
                        // For now, we fall back to the generic tangent if k_nl.
                        if (k_sym) {
                            disp_tan = getSpecializedDispatch(
                                KernelRole::Tangent, k_sym->tangentIR(), IntegralDomain::Cell, *first_ctx, nullptr);
                            if (disp_tan) compiled_tan_ptr = disp_tan.get();
                        }
                    }

                    std::shared_ptr<const CompiledDispatch> disp_res;
                    const CompiledDispatch* compiled_res_ptr = &compiled_residual_;
                    if (want_vector) {
                        disp_res = getSpecializedDispatch(
                            KernelRole::Residual, residual_ir, IntegralDomain::Cell, *first_ctx, nullptr);
                        if (disp_res) compiled_res_ptr = disp_res.get();
                    }

                    // Pre-reserve all outputs.
                    const auto n_test = first_ctx->numTestDofs();
                    const auto n_trial = first_ctx->numTrialDofs();
                    for (std::size_t idx = 0; idx < n; ++idx) {
                        outputs[idx].reserveNoZero(n_test, n_trial, want_matrix, want_vector);
                    }

                    // Pack template from first element (avoids stencil recomputation).
                    const auto prepared =
                        makePreparedCellBatchTemplateV6(*first_ctx, outputs[first_idx], checks);

                    // Tight per-element loop.
                    // When OpenMP is available and the batch is large enough,
                    // elements are processed in parallel using thread-local
                    // sub-batches.  The JIT function is a pure function of its
                    // input/output pointers and is safe for concurrent calls.

                    const auto tan_addr   = want_matrix ? compiled_tan_ptr->cell : std::uintptr_t{0};
                    const auto res_addr   = want_vector ? compiled_res_ptr->cell : std::uintptr_t{0};

                    if (options_.vectorize) {
#ifdef _OPENMP
                        // Parallel sub-batch dispatch: each thread packs and
                        // processes its own contiguous sub-batch using the batch
                        // ABI (CellKernelBatchArgsV1) that the vectorized JIT
                        // kernel expects.
                        const int omp_threads = omp_get_max_threads();
                        if (omp_threads > 1 && n >= 4) {
                            #pragma omp parallel num_threads(omp_threads)
                            {
                                const int tid = omp_get_thread_num();
                                const int nt  = omp_get_num_threads();
                                const std::size_t chunk = (n + static_cast<std::size_t>(nt) - 1)
                                                          / static_cast<std::size_t>(nt);
                                const std::size_t lo = std::min(static_cast<std::size_t>(tid) * chunk, n);
                                const std::size_t hi = std::min(lo + chunk, n);
                                const std::size_t sub_n = hi - lo;

                                if (sub_n > 0) {
                                    // Thread-local scratch for batch packing.
                                    // SIMD batch padding.
                                    const auto simd_w_omp = options_.simd_batch
                                        ? static_cast<std::size_t>(jit::hardwareProfile().simdDoubles())
                                        : std::size_t{1};
                                    const auto padded_sub_n = (simd_w_omp >= 2 && options_.simd_batch)
                                        ? ((sub_n + simd_w_omp - 1) / simd_w_omp) * simd_w_omp
                                        : sub_n;
                                    std::vector<assembly::jit::KernelSideArgsV6> local_sides(padded_sub_n);
                                    std::vector<assembly::jit::KernelOutputViewV6> local_outputs(padded_sub_n);

                                    for (std::size_t idx = lo; idx < hi; ++idx) {
                                        if (contexts[idx] == nullptr) continue;
                                        const auto& ctx = *contexts[idx];
                                        auto& output = outputs[idx];
                                        output.clear();

                                        if (has_updates) {
                                            for (LocalIndex q = 0; q < ctx.numQuadraturePoints(); ++q) {
                                                applyInlinedMaterialStateUpdatesReal(
                                                    ctx, nullptr, FormKind::Residual,
                                                    state_layout, updates, Side::Minus, q);
                                            }
                                        }

                                        prepareCellBatchEntryV6(
                                            prepared,
                                            ctx,
                                            output,
                                            local_sides[idx - lo],
                                            local_outputs[idx - lo]);

                                        output.has_matrix = want_matrix;
                                        output.has_vector = want_vector;
                                    }

                                    // Pad remaining slots for SIMD batch
                                    if (padded_sub_n > sub_n) {
                                        std::size_t last_v = sub_n - 1;
                                        while (last_v > 0 && local_sides[last_v].integration_weights == nullptr) {
                                            --last_v;
                                        }
                                        thread_local std::vector<Real> pad_m_omp, pad_v_omp;
                                        pad_m_omp.assign(
                                            static_cast<std::size_t>(n_test) * static_cast<std::size_t>(n_trial), 0.0);
                                        pad_v_omp.assign(static_cast<std::size_t>(n_test), 0.0);
                                        for (std::size_t pi = sub_n; pi < padded_sub_n; ++pi) {
                                            local_sides[pi] = local_sides[last_v];
                                            local_outputs[pi] = local_outputs[last_v];
                                            local_outputs[pi].element_matrix = pad_m_omp.data();
                                            local_outputs[pi].element_vector = pad_v_omp.data();
                                        }
                                    }

                                    assembly::jit::CellKernelBatchArgsV1 batch_args;
                                    batch_args.batch_size = static_cast<std::uint32_t>(padded_sub_n);
                                    batch_args.sides   = local_sides.data();
                                    batch_args.outputs = local_outputs.data();

                                    if (tan_addr != 0) callJIT(tan_addr, &batch_args);
                                    if (res_addr != 0) callJIT(res_addr, &batch_args);
                                }
                            }
                        } else
#endif
                        {
                            // Serial vectorize path: stack-local scratch
                            // (thread-safe for concurrent calls).
                            // SIMD batch padding: round up to next SIMD width.
                            const auto simd_w_nl = options_.simd_batch
                                ? static_cast<std::size_t>(jit::hardwareProfile().simdDoubles())
                                : std::size_t{1};
                            const auto padded_n_nl = (simd_w_nl >= 2 && options_.simd_batch)
                                ? ((n + simd_w_nl - 1) / simd_w_nl) * simd_w_nl
                                : n;

                            std::vector<assembly::jit::KernelSideArgsV6> batch_sides(padded_n_nl);
                            std::vector<assembly::jit::KernelOutputViewV6> batch_outputs(padded_n_nl);
                            thread_local std::vector<Real> pad_mat_nl, pad_vec_nl;

                            for (std::size_t idx = 0; idx < n; ++idx) {
                                if (contexts[idx] == nullptr) continue;
                                const auto& ctx = *contexts[idx];
                                auto& output = outputs[idx];
                                output.clear();

                                if (has_updates) {
                                    for (LocalIndex q = 0; q < ctx.numQuadraturePoints(); ++q) {
                                        applyInlinedMaterialStateUpdatesReal(
                                            ctx, nullptr, FormKind::Residual,
                                            state_layout, updates, Side::Minus, q);
                                    }
                                }

                                prepareCellBatchEntryV6(
                                    prepared, ctx, output, batch_sides[idx], batch_outputs[idx]);

                                output.has_matrix = want_matrix;
                                output.has_vector = want_vector;
                            }

                            // Fill holes and pad for SIMD batch.
                            // The SIMD kernel reads ALL batch_size entries — null-context
                            // holes must be filled with valid data (reads are safe, writes
                            // go to scratch buffers).
                            {
                                pad_mat_nl.assign(
                                    static_cast<std::size_t>(n_test) * static_cast<std::size_t>(n_trial), 0.0);
                                pad_vec_nl.assign(static_cast<std::size_t>(n_test), 0.0);

                                // Find first valid entry to use as fill source.
                                std::size_t fill_src = 0;
                                for (std::size_t idx = 0; idx < n; ++idx) {
                                    if (batch_sides[idx].integration_weights != nullptr) {
                                        fill_src = idx;
                                        break;
                                    }
                                }
                                // Fill holes within [0, n) and padding slots [n, padded_n_nl).
                                for (std::size_t idx = 0; idx < padded_n_nl; ++idx) {
                                    if (batch_sides[idx].integration_weights == nullptr) {
                                        batch_sides[idx] = batch_sides[fill_src];
                                        batch_outputs[idx].element_matrix = pad_mat_nl.data();
                                        batch_outputs[idx].element_vector = pad_vec_nl.data();
                                        batch_outputs[idx].n_test_dofs = batch_outputs[fill_src].n_test_dofs;
                                        batch_outputs[idx].n_trial_dofs = batch_outputs[fill_src].n_trial_dofs;
                                    }
                                }
                            }

                            assembly::jit::CellKernelBatchArgsV1 batch_args;
                            batch_args.batch_size = static_cast<std::uint32_t>(padded_n_nl);
                            batch_args.sides   = batch_sides.data();
                            batch_args.outputs = batch_outputs.data();

                            if (tan_addr != 0) callJIT(tan_addr, &batch_args);
                            if (res_addr != 0) callJIT(res_addr, &batch_args);
                        }
                    } else {
                        // Non-vectorize per-element path.
#ifdef _OPENMP
                        #pragma omp parallel for schedule(static) if(n >= 4)
#endif
                        for (std::size_t idx = 0; idx < n; ++idx) {
                            if (contexts[idx] == nullptr) continue;
                            const auto& ctx = *contexts[idx];
                            auto& output = outputs[idx];
                            output.clear();

                            if (has_updates) {
                                for (LocalIndex q = 0; q < ctx.numQuadraturePoints(); ++q) {
                                    applyInlinedMaterialStateUpdatesReal(
                                        ctx, nullptr, FormKind::Residual,
                                        state_layout, updates, Side::Minus, q);
                                }
                            }

                            assembly::jit::CellKernelArgsV6 args;
                            prepareCellArgsV6(prepared, args, ctx, output);
                            if (tan_addr != 0) callJIT(tan_addr, &args);
                            if (res_addr != 0) callJIT(res_addr, &args);

                            output.has_matrix = want_matrix;
                            output.has_vector = want_vector;
                        }
                    }
                    return;
                }
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
        const auto checks = assembly::jit::PackingChecks{.validate_alignment = false};

    if (kind_ == WrappedKind::FormKernel) {
        const auto* k = dynamic_cast<const FormKernel*>(fallback_.get());
        if (!k) {
            fallback_->computeBoundaryFace(ctx, boundary_marker, output);
            return;
        }
        if (domainUsesRuntimeCoefficient(k->ir(), IntegralDomain::Boundary, boundary_marker)) {
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
        if (domainUsesRuntimeCoefficient(k->bilinearIR(), IntegralDomain::Boundary, boundary_marker) ||
            (k->linearIR().has_value() &&
             domainUsesRuntimeCoefficient(*k->linearIR(), IntegralDomain::Boundary, boundary_marker))) {
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
                accumulateKernelOutputMatrixTimesSolution(
                    output, coeffs.data(), ctx.numTestDofs(), ctx.numTrialDofs());
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

                accumulateDenseMatVecRaw(
                    tmp.local_matrix.data(),
                    ctx.numTestDofs(),
                    ctx.numTrialDofs(),
                    coeffs.data(),
                    output.local_vector.data());
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
        if (domainUsesRuntimeCoefficient(k->residualIR(), IntegralDomain::Boundary, boundary_marker) ||
            domainUsesRuntimeCoefficient(k->tangentIR(), IntegralDomain::Boundary, boundary_marker)) {
            fallback_->computeBoundaryFace(ctx, boundary_marker, output);
            return;
        }
        const auto requested = requestedKernelOutputs(
            output,
            !k->isVectorOnly(),
            !k->isMatrixOnly());
        const bool want_matrix = requested.matrix;
        const bool want_vector = requested.vector;

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
        const auto checks = assembly::jit::PackingChecks{.validate_alignment = false};

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

        const auto minus_requested = requestedKernelOutputs(
            output_minus,
            !k->isVectorOnly(),
            !k->isMatrixOnly());
        const auto plus_requested = requestedKernelOutputs(
            output_plus,
            !k->isVectorOnly(),
            !k->isMatrixOnly());
        const auto mp_requested = requestedKernelOutputs(
            coupling_minus_plus,
            !k->isVectorOnly(),
            /*can_vector=*/false);
        const auto pm_requested = requestedKernelOutputs(
            coupling_plus_minus,
            !k->isVectorOnly(),
            /*can_vector=*/false);
        const bool want_matrix =
            minus_requested.matrix || plus_requested.matrix || mp_requested.matrix || pm_requested.matrix;
        const bool want_vector = minus_requested.vector || plus_requested.vector;

        const auto n_test_minus = ctx_minus.numTestDofs();
        const auto n_trial_minus = ctx_minus.numTrialDofs();
        const auto n_test_plus = ctx_plus.numTestDofs();
        const auto n_trial_plus = ctx_plus.numTrialDofs();

        output_minus.reserve(n_test_minus, n_trial_minus, minus_requested.matrix, minus_requested.vector);
        output_plus.reserve(n_test_plus, n_trial_plus, plus_requested.matrix, plus_requested.vector);
        coupling_minus_plus.reserve(n_test_minus, n_trial_plus, mp_requested.matrix, /*need_vector=*/false);
        coupling_plus_minus.reserve(n_test_plus, n_trial_minus, pm_requested.matrix, /*need_vector=*/false);

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

        output_minus.has_matrix = minus_requested.matrix;
        output_minus.has_vector = minus_requested.vector;
        output_plus.has_matrix = plus_requested.matrix;
        output_plus.has_vector = plus_requested.vector;
        coupling_minus_plus.has_matrix = mp_requested.matrix;
        coupling_minus_plus.has_vector = false;
        coupling_plus_minus.has_matrix = pm_requested.matrix;
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
        const auto checks = assembly::jit::PackingChecks{.validate_alignment = false};

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

        const auto minus_requested = requestedKernelOutputs(
            output_minus,
            !k->isVectorOnly(),
            !k->isMatrixOnly());
        const auto plus_requested = requestedKernelOutputs(
            output_plus,
            !k->isVectorOnly(),
            !k->isMatrixOnly());
        const auto mp_requested = requestedKernelOutputs(
            coupling_minus_plus,
            !k->isVectorOnly(),
            /*can_vector=*/false);
        const auto pm_requested = requestedKernelOutputs(
            coupling_plus_minus,
            !k->isVectorOnly(),
            /*can_vector=*/false);
        const bool want_matrix =
            minus_requested.matrix || plus_requested.matrix || mp_requested.matrix || pm_requested.matrix;
        const bool want_vector = minus_requested.vector || plus_requested.vector;

        const auto n_test_minus = ctx_minus.numTestDofs();
        const auto n_trial_minus = ctx_minus.numTrialDofs();
        const auto n_test_plus = ctx_plus.numTestDofs();
        const auto n_trial_plus = ctx_plus.numTrialDofs();

        output_minus.reserve(n_test_minus, n_trial_minus, minus_requested.matrix, minus_requested.vector);
        output_plus.reserve(n_test_plus, n_trial_plus, plus_requested.matrix, plus_requested.vector);
        coupling_minus_plus.reserve(n_test_minus, n_trial_plus, mp_requested.matrix, /*need_vector=*/false);
        coupling_plus_minus.reserve(n_test_plus, n_trial_minus, pm_requested.matrix, /*need_vector=*/false);

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

        output_minus.has_matrix = minus_requested.matrix;
        output_minus.has_vector = minus_requested.vector;
        output_plus.has_matrix = plus_requested.matrix;
        output_plus.has_vector = plus_requested.vector;
        coupling_minus_plus.has_matrix = mp_requested.matrix;
        coupling_minus_plus.has_vector = false;
        coupling_plus_minus.has_matrix = pm_requested.matrix;
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

void JITKernelWrapper::primeCellSpecializations(std::span<const CellSpecializationHint> hints)
{
    maybeCompile();
    if (traceSpecializationEnabled()) {
        std::uint64_t revision = 0;
        std::uint64_t compiled_revision = 0;
        bool compiler_ready = false;
        bool jit_ready = false;
        std::uintptr_t residual_cell = 0;
        std::uintptr_t tangent_cell = 0;
        {
            std::lock_guard<std::mutex> lock(jit_mutex_);
            revision = revision_;
            compiled_revision = compiled_revision_;
            compiler_ready = static_cast<bool>(compiler_);
            jit_ready = options_.enable && compiled_revision_ == revision_ && !runtime_failed_;
            residual_cell = compiled_residual_.cell;
            tangent_cell = compiled_tangent_.cell;
        }

        std::ostringstream detail;
        detail << "event=prime_begin hints=" << hints.size()
               << " compiled_revision=" << compiled_revision
               << " compiler_ready=" << (compiler_ready ? 1 : 0)
               << " jit_ready=" << (jit_ready ? 1 : 0)
               << " residual_cell=" << residual_cell
               << " tangent_cell=" << tangent_cell;
        traceSpecialization(this, *fallback_, revision, detail.str());
    }

    if (!canUseJIT() || !options_.specialization.enable || !compiler_) {
        if (traceSpecializationEnabled()) {
            std::uint64_t revision = 0;
            {
                std::lock_guard<std::mutex> lock(jit_mutex_);
                revision = revision_;
            }

            std::ostringstream detail;
            detail << "event=prime_skip reason=jit_unavailable"
                   << " can_use_jit=" << (canUseJIT() ? 1 : 0)
                   << " specialization_enabled=" << (options_.specialization.enable ? 1 : 0)
                   << " compiler_ready=" << (compiler_ ? 1 : 0);
            traceSpecialization(this, *fallback_, revision, detail.str());
        }
        return;
    }

    const auto prime_role = [&](KernelRole role,
                                const FormIR& ir,
                                const CellSpecializationHint& hint) {
        const auto role_name = [role]() noexcept {
            switch (role) {
                case KernelRole::Form:
                    return "Form";
                case KernelRole::Bilinear:
                    return "Bilinear";
                case KernelRole::Linear:
                    return "Linear";
                case KernelRole::Residual:
                    return "Residual";
                case KernelRole::Tangent:
                    return "Tangent";
            }
            return "Unknown";
        };

        JITCompileSpecialization spec;
        spec.domain = IntegralDomain::Cell;
        spec.is_affine = hint.is_affine;
        if (hint.is_affine) {
            primed_is_affine_ = true;
        }
        bool any = false;

        if (options_.specialization.specialize_n_qpts &&
            hint.n_qpts > 0u &&
            hint.n_qpts <= options_.specialization.max_specialized_n_qpts) {
            spec.n_qpts_minus = hint.n_qpts;
            any = true;
        }

        if (options_.specialization.specialize_dofs &&
            hint.n_test_dofs > 0u &&
            hint.n_trial_dofs > 0u &&
            hint.n_test_dofs <= options_.specialization.max_specialized_dofs &&
            hint.n_trial_dofs <= options_.specialization.max_specialized_dofs) {
            spec.n_test_dofs_minus = hint.n_test_dofs;
            spec.n_trial_dofs_minus = hint.n_trial_dofs;
            any = true;
        }

        if (!any) {
            if (traceSpecializationEnabled()) {
                std::uint64_t revision = 0;
                {
                    std::lock_guard<std::mutex> lock(jit_mutex_);
                    revision = revision_;
                }

                std::ostringstream detail;
                detail << "event=prime_skip reason=no_specializable_shape"
                       << " role=" << role_name()
                       << " n_qpts=" << hint.n_qpts
                       << " n_test_dofs=" << hint.n_test_dofs
                       << " n_trial_dofs=" << hint.n_trial_dofs;
                traceSpecialization(this, *fallback_, revision, detail.str());
            }
            return;
        }

        if (traceSpecializationEnabled()) {
            std::uint64_t revision = 0;
            {
                std::lock_guard<std::mutex> lock(jit_mutex_);
                revision = revision_;
            }

            std::ostringstream detail;
            detail << "event=prime_request role=" << role_name()
                   << " domain=" << integralDomainName(spec.domain)
                   << " n_qpts=" << hint.n_qpts
                   << " n_test_dofs=" << hint.n_test_dofs
                   << " n_trial_dofs=" << hint.n_trial_dofs;
            traceSpecialization(this, *fallback_, revision, detail.str());
        }

        (void)compileSpecializedDispatch(role, ir, spec, "prime");
    };

    if (kind_ == WrappedKind::FormKernel) {
        if (const auto* k = dynamic_cast<const FormKernel*>(fallback_.get())) {
            for (const auto& hint : hints) {
                prime_role(KernelRole::Form, k->ir(), hint);
            }
        }
        return;
    }

    if (kind_ == WrappedKind::LinearFormKernel) {
        if (const auto* k = dynamic_cast<const LinearFormKernel*>(fallback_.get())) {
            for (const auto& hint : hints) {
                prime_role(KernelRole::Bilinear, k->bilinearIR(), hint);
                if (has_compiled_linear_ && k->linearIR().has_value()) {
                    prime_role(KernelRole::Linear, *k->linearIR(), hint);
                }
            }
        }
        return;
    }

    if (kind_ == WrappedKind::SymbolicNonlinearFormKernel) {
        if (const auto* k = dynamic_cast<const SymbolicNonlinearFormKernel*>(fallback_.get())) {
            for (const auto& hint : hints) {
                if (compiled_residual_.cell != 0) {
                    prime_role(KernelRole::Residual, k->residualIR(), hint);
                } else if (traceSpecializationEnabled()) {
                    std::uint64_t revision = 0;
                    {
                        std::lock_guard<std::mutex> lock(jit_mutex_);
                        revision = revision_;
                    }
                    traceSpecialization(this, *fallback_, revision,
                                        "event=prime_skip reason=missing_generic_dispatch role=Residual");
                }
                if (compiled_tangent_.cell != 0) {
                    prime_role(KernelRole::Tangent, k->tangentIR(), hint);
                } else if (traceSpecializationEnabled()) {
                    std::uint64_t revision = 0;
                    {
                        std::lock_guard<std::mutex> lock(jit_mutex_);
                        revision = revision_;
                    }
                    traceSpecialization(this, *fallback_, revision,
                                        "event=prime_skip reason=missing_generic_dispatch role=Tangent");
                }
            }
        }
        return;
    }

    if (kind_ == WrappedKind::NonlinearFormKernel) {
        if (const auto* k = dynamic_cast<const NonlinearFormKernel*>(fallback_.get())) {
            for (const auto& hint : hints) {
                if (compiled_residual_.cell != 0) {
                    prime_role(KernelRole::Residual, k->residualIR(), hint);
                } else if (traceSpecializationEnabled()) {
                    std::uint64_t revision = 0;
                    {
                        std::lock_guard<std::mutex> lock(jit_mutex_);
                        revision = revision_;
                    }
                    traceSpecialization(this, *fallback_, revision,
                                        "event=prime_skip reason=missing_generic_dispatch role=Residual");
                }
                if (traceSpecializationEnabled()) {
                    std::uint64_t revision = 0;
                    {
                        std::lock_guard<std::mutex> lock(jit_mutex_);
                        revision = revision_;
                    }
                    traceSpecialization(this, *fallback_, revision,
                                        "event=prime_skip reason=tangent_ir_unavailable role=Tangent");
                }
            }
        }
    }
}

void JITKernelWrapper::primeBoundarySpecializations(std::span<const BoundarySpecializationHint> hints)
{
    maybeCompile();
    if (traceSpecializationEnabled()) {
        std::uint64_t revision = 0;
        std::uint64_t compiled_revision = 0;
        bool compiler_ready = false;
        bool jit_ready = false;
        std::uintptr_t residual_boundary = 0;
        std::uintptr_t tangent_boundary = 0;
        {
            std::lock_guard<std::mutex> lock(jit_mutex_);
            revision = revision_;
            compiled_revision = compiled_revision_;
            compiler_ready = static_cast<bool>(compiler_);
            jit_ready = options_.enable && compiled_revision_ == revision_ && !runtime_failed_;
            residual_boundary = compiled_residual_.boundary_all;
            tangent_boundary = compiled_tangent_.boundary_all;
        }

        std::ostringstream detail;
        detail << "event=prime_begin domain=Boundary hints=" << hints.size()
               << " compiled_revision=" << compiled_revision
               << " compiler_ready=" << (compiler_ready ? 1 : 0)
               << " jit_ready=" << (jit_ready ? 1 : 0)
               << " residual_boundary=" << residual_boundary
               << " tangent_boundary=" << tangent_boundary;
        traceSpecialization(this, *fallback_, revision, detail.str());
    }

    if (!canUseJIT() || !options_.specialization.enable || !compiler_) {
        if (traceSpecializationEnabled()) {
            std::uint64_t revision = 0;
            {
                std::lock_guard<std::mutex> lock(jit_mutex_);
                revision = revision_;
            }

            std::ostringstream detail;
            detail << "event=prime_skip domain=Boundary reason=jit_unavailable"
                   << " can_use_jit=" << (canUseJIT() ? 1 : 0)
                   << " specialization_enabled=" << (options_.specialization.enable ? 1 : 0)
                   << " compiler_ready=" << (compiler_ ? 1 : 0);
            traceSpecialization(this, *fallback_, revision, detail.str());
        }
        return;
    }

    const auto prime_role = [&](KernelRole role,
                                const FormIR& ir,
                                const BoundarySpecializationHint& hint) {
        const auto role_name = [role]() noexcept {
            switch (role) {
                case KernelRole::Form:
                    return "Form";
                case KernelRole::Bilinear:
                    return "Bilinear";
                case KernelRole::Linear:
                    return "Linear";
                case KernelRole::Residual:
                    return "Residual";
                case KernelRole::Tangent:
                    return "Tangent";
            }
            return "Unknown";
        };

        JITCompileSpecialization spec;
        spec.domain = IntegralDomain::Boundary;
        bool any = false;

        if (options_.specialization.specialize_n_qpts &&
            hint.n_qpts > 0u &&
            hint.n_qpts <= options_.specialization.max_specialized_n_qpts) {
            spec.n_qpts_minus = hint.n_qpts;
            any = true;
        }

        if (options_.specialization.specialize_dofs &&
            hint.n_test_dofs > 0u &&
            hint.n_trial_dofs > 0u &&
            hint.n_test_dofs <= options_.specialization.max_specialized_dofs &&
            hint.n_trial_dofs <= options_.specialization.max_specialized_dofs) {
            spec.n_test_dofs_minus = hint.n_test_dofs;
            spec.n_trial_dofs_minus = hint.n_trial_dofs;
            any = true;
        }

        if (!any) {
            if (traceSpecializationEnabled()) {
                std::uint64_t revision = 0;
                {
                    std::lock_guard<std::mutex> lock(jit_mutex_);
                    revision = revision_;
                }

                std::ostringstream detail;
                detail << "event=prime_skip domain=Boundary reason=no_specializable_shape"
                       << " role=" << role_name()
                       << " marker=" << hint.boundary_marker
                       << " n_qpts=" << hint.n_qpts
                       << " n_test_dofs=" << hint.n_test_dofs
                       << " n_trial_dofs=" << hint.n_trial_dofs;
                traceSpecialization(this, *fallback_, revision, detail.str());
            }
            return;
        }

        if (traceSpecializationEnabled()) {
            std::uint64_t revision = 0;
            {
                std::lock_guard<std::mutex> lock(jit_mutex_);
                revision = revision_;
            }

            std::ostringstream detail;
            detail << "event=prime_request role=" << role_name()
                   << " domain=" << integralDomainName(spec.domain)
                   << " marker=" << hint.boundary_marker
                   << " n_qpts=" << hint.n_qpts
                   << " n_test_dofs=" << hint.n_test_dofs
                   << " n_trial_dofs=" << hint.n_trial_dofs;
            traceSpecialization(this, *fallback_, revision, detail.str());
        }

        (void)compileSpecializedDispatch(role, ir, spec, "prime");
    };

    const auto has_boundary_dispatch = [](const CompiledDispatch& dispatch) {
        return dispatch.boundary_all != 0 || !dispatch.boundary_by_marker.empty();
    };

    if (kind_ == WrappedKind::FormKernel) {
        if (const auto* k = dynamic_cast<const FormKernel*>(fallback_.get())) {
            for (const auto& hint : hints) {
                prime_role(KernelRole::Form, k->ir(), hint);
            }
        }
        return;
    }

    if (kind_ == WrappedKind::LinearFormKernel) {
        if (const auto* k = dynamic_cast<const LinearFormKernel*>(fallback_.get())) {
            for (const auto& hint : hints) {
                prime_role(KernelRole::Bilinear, k->bilinearIR(), hint);
                if (has_compiled_linear_ && k->linearIR().has_value()) {
                    prime_role(KernelRole::Linear, *k->linearIR(), hint);
                }
            }
        }
        return;
    }

    if (kind_ == WrappedKind::SymbolicNonlinearFormKernel) {
        if (const auto* k = dynamic_cast<const SymbolicNonlinearFormKernel*>(fallback_.get())) {
            for (const auto& hint : hints) {
                if (has_boundary_dispatch(compiled_residual_)) {
                    prime_role(KernelRole::Residual, k->residualIR(), hint);
                } else if (traceSpecializationEnabled()) {
                    std::uint64_t revision = 0;
                    {
                        std::lock_guard<std::mutex> lock(jit_mutex_);
                        revision = revision_;
                    }
                    std::ostringstream detail;
                    detail << "event=prime_skip domain=Boundary reason=missing_generic_dispatch"
                           << " role=Residual marker=" << hint.boundary_marker;
                    traceSpecialization(this, *fallback_, revision, detail.str());
                }

                if (has_boundary_dispatch(compiled_tangent_)) {
                    prime_role(KernelRole::Tangent, k->tangentIR(), hint);
                } else if (traceSpecializationEnabled()) {
                    std::uint64_t revision = 0;
                    {
                        std::lock_guard<std::mutex> lock(jit_mutex_);
                        revision = revision_;
                    }
                    std::ostringstream detail;
                    detail << "event=prime_skip domain=Boundary reason=missing_generic_dispatch"
                           << " role=Tangent marker=" << hint.boundary_marker;
                    traceSpecialization(this, *fallback_, revision, detail.str());
                }
            }
        }
    }
}

void JITKernelWrapper::markDirty(std::string_view reason) noexcept
{
    std::uint64_t revision = 0;
    std::lock_guard<std::mutex> lock(jit_mutex_);
    ++revision_;
    revision = revision_;
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
    traced_specialization_hits_.clear();
    traced_specialization_compiles_.clear();
    traced_specialization_skips_.clear();
    warned_specialization_failure_ = false;

    warned_compile_failure_ = false;
    runtime_failed_ = false;
    warned_runtime_failure_ = false;

    const std::string detail = "event=dirty reason=" + std::string(reason);
    traceSpecialization(this, *fallback_, revision, detail);
}

bool JITKernelWrapper::canUseJIT() const noexcept
{
    return options_.enable && compiled_revision_ == revision_ && !runtime_failed_;
}

void JITKernelWrapper::ensureCompiled()
{
    maybeCompile();
}

bool JITKernelWrapper::isJITReady() const noexcept
{
    return canUseJIT();
}

void JITKernelWrapper::setExternalCellAddress(std::uintptr_t addr)
{
    if (addr == 0) {
        return;
    }

    // Ensure the fallback kernel has been analyzed so we know the wrapped kind.
    // Must be called BEFORE acquiring jit_mutex_ since maybeCompile() also locks it.
    maybeCompile();

    std::lock_guard<std::mutex> lock(jit_mutex_);

    // Inject the address into whichever compiled dispatch is appropriate.
    // For coupled blocks, the tangent (bilinear) cell kernel is the primary target.
    if (compiled_tangent_.ok && compiled_tangent_.cell != 0) {
        compiled_tangent_.cell = addr;
    } else if (compiled_bilinear_.ok && compiled_bilinear_.cell != 0) {
        compiled_bilinear_.cell = addr;
    } else if (compiled_form_.ok && compiled_form_.cell != 0) {
        compiled_form_.cell = addr;
    } else {
        // No compiled dispatch available yet — set up a minimal one.
        // This handles the case where the wrapper hasn't been compiled at all.
        compiled_form_.ok = true;
        compiled_form_.cell = addr;

        // Also set tangent/bilinear if appropriate for the kernel kind.
        if (kind_ == WrappedKind::FormKernel || kind_ == WrappedKind::SymbolicNonlinearFormKernel) {
            compiled_tangent_.ok = true;
            compiled_tangent_.cell = addr;
            compiled_bilinear_.ok = true;
            compiled_bilinear_.cell = addr;
        }
    }
}

std::shared_ptr<const JITKernelWrapper::CompiledDispatch> JITKernelWrapper::compileSpecializedDispatch(
    KernelRole role,
    const FormIR& ir,
    const JITCompileSpecialization& specialization,
    std::string_view trigger)
{
    if (!options_.enable || !options_.specialization.enable || !compiler_) {
        return nullptr;
    }

    SpecializationKey key;
    key.role = role;
    key.domain = specialization.domain;

    if (specialization.n_qpts_minus) {
        key.has_n_qpts_minus = true;
        key.n_qpts_minus = *specialization.n_qpts_minus;
    }
    if (specialization.n_test_dofs_minus) {
        key.has_n_test_dofs_minus = true;
        key.n_test_dofs_minus = *specialization.n_test_dofs_minus;
    }
    if (specialization.n_trial_dofs_minus) {
        key.has_n_trial_dofs_minus = true;
        key.n_trial_dofs_minus = *specialization.n_trial_dofs_minus;
    }
    if (specialization.n_qpts_plus) {
        key.has_n_qpts_plus = true;
        key.n_qpts_plus = *specialization.n_qpts_plus;
    }
    if (specialization.n_test_dofs_plus) {
        key.has_n_test_dofs_plus = true;
        key.n_test_dofs_plus = *specialization.n_test_dofs_plus;
    }
    if (specialization.n_trial_dofs_plus) {
        key.has_n_trial_dofs_plus = true;
        key.n_trial_dofs_plus = *specialization.n_trial_dofs_plus;
    }

    const auto role_name = [role]() noexcept {
        switch (role) {
            case KernelRole::Form:
                return "Form";
            case KernelRole::Bilinear:
                return "Bilinear";
            case KernelRole::Linear:
                return "Linear";
            case KernelRole::Residual:
                return "Residual";
            case KernelRole::Tangent:
                return "Tangent";
        }
        return "Unknown";
    };

    const auto describe_key = [&]() {
        std::ostringstream oss;
        oss << "role=" << role_name()
            << " domain=" << integralDomainName(key.domain)
            << " minus[qpts=" << (key.has_n_qpts_minus ? std::to_string(key.n_qpts_minus) : "*")
            << ",test=" << (key.has_n_test_dofs_minus ? std::to_string(key.n_test_dofs_minus) : "*")
            << ",trial=" << (key.has_n_trial_dofs_minus ? std::to_string(key.n_trial_dofs_minus) : "*") << "]";
        if (key.domain == IntegralDomain::InteriorFace || key.domain == IntegralDomain::InterfaceFace) {
            oss << " plus[qpts=" << (key.has_n_qpts_plus ? std::to_string(key.n_qpts_plus) : "*")
                << ",test=" << (key.has_n_test_dofs_plus ? std::to_string(key.n_test_dofs_plus) : "*")
                << ",trial=" << (key.has_n_trial_dofs_plus ? std::to_string(key.n_trial_dofs_plus) : "*") << "]";
        }
        return oss.str();
    };

    std::uint64_t my_revision = 0;
    std::string trace_detail;
    {
        std::lock_guard<std::mutex> lock(jit_mutex_);
        if (compiled_revision_ != revision_) {
            if (traceSpecializationEnabled()) {
                std::ostringstream oss;
                oss << "event=skip trigger=" << trigger
                    << " reason=generic_revision_mismatch"
                    << " compiled_revision=" << compiled_revision_
                    << " current_revision=" << revision_
                    << " " << describe_key();
                trace_detail = oss.str();
            }
            my_revision = revision_;
            if (!trace_detail.empty()) {
                traceSpecialization(this, *fallback_, my_revision, trace_detail);
            }
            return nullptr;
        }
        my_revision = revision_;

        if (const auto it = specialized_dispatch_.find(key); it != specialized_dispatch_.end()) {
            if (traceSpecializationEnabled() && traced_specialization_hits_.insert(key).second) {
                std::ostringstream oss;
                oss << "event=hit trigger=" << trigger << ' ' << describe_key();
                trace_detail = oss.str();
            }
            const auto result = it->second;
            if (!trace_detail.empty()) {
                traceSpecialization(this, *fallback_, my_revision, trace_detail);
            }
            return result;
        }

        if (attempted_specializations_.find(key) != attempted_specializations_.end()) {
            if (traceSpecializationEnabled() && traced_specialization_skips_.insert(key).second) {
                std::ostringstream oss;
                oss << "event=skip trigger=" << trigger
                    << " reason=already_attempted "
                    << describe_key();
                trace_detail = oss.str();
            }
            if (!trace_detail.empty()) {
                traceSpecialization(this, *fallback_, my_revision, trace_detail);
            }
            return nullptr;
        }

        std::size_t count = 0;
        for (const auto& [k, _] : specialized_dispatch_) {
            if (k.role == role && k.domain == specialization.domain) {
                ++count;
            }
        }
        if (count >= options_.specialization.max_variants_per_kernel) {
            if (traceSpecializationEnabled() && traced_specialization_skips_.insert(key).second) {
                std::ostringstream oss;
                oss << "event=skip trigger=" << trigger
                    << " reason=max_variants_per_kernel"
                    << " count=" << count
                    << " limit=" << options_.specialization.max_variants_per_kernel
                    << ' ' << describe_key();
                trace_detail = oss.str();
            }
            if (!trace_detail.empty()) {
                traceSpecialization(this, *fallback_, my_revision, trace_detail);
            }
            return nullptr;
        }

        attempted_specializations_.insert(key);
    }

    ValidationOptions vopt;
    vopt.strictness = Strictness::AllowExternalCalls;

    const auto r = compiler_->compileSpecialized(ir, specialization, vopt);
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
        if (traceSpecializationEnabled()) {
            std::ostringstream detail;
            detail << "event=compile_failed trigger=" << trigger
                   << " message='" << r.message << "' "
                   << describe_key();
            traceSpecialization(this, *fallback_, my_revision, detail.str());
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
            if (traceSpecializationEnabled() && traced_specialization_skips_.insert(key).second) {
                std::ostringstream oss;
                oss << "event=skip trigger=" << trigger
                    << " reason=revision_changed_after_compile"
                    << " compiled_revision=" << compiled_revision_
                    << " current_revision=" << revision_
                    << ' ' << describe_key();
                trace_detail = oss.str();
            }
            if (!trace_detail.empty()) {
                traceSpecialization(this, *fallback_, my_revision, trace_detail);
            }
            return nullptr;
        }
        specialized_dispatch_[key] = disp;
        if (traceSpecializationEnabled() && traced_specialization_compiles_.insert(key).second) {
            std::ostringstream oss;
            oss << "event=compile trigger=" << trigger
                << " cell=" << disp->cell
                << " interior_face=" << disp->interior_face
                << ' ' << describe_key();
            trace_detail = oss.str();
        }
    }

    if (!trace_detail.empty()) {
        traceSpecialization(this, *fallback_, my_revision, trace_detail);
    }

    return disp;
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
        if (traceSpecializationEnabled()) {
            std::uint64_t revision = 0;
            {
                std::lock_guard<std::mutex> lock(jit_mutex_);
                revision = revision_;
            }
            std::ostringstream detail;
            detail << "event=runtime_skip reason=missing_plus_context"
                   << " role=" << static_cast<int>(role)
                   << " domain=" << integralDomainName(domain);
            traceSpecialization(this, *fallback_, revision, detail.str());
        }
        return nullptr;
    }

    JITCompileSpecialization spec;
    spec.domain = domain;
    spec.is_affine = primed_is_affine_;
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
                if (traceSpecializationEnabled()) {
                    std::uint64_t revision = 0;
                    {
                        std::lock_guard<std::mutex> lock(jit_mutex_);
                        revision = revision_;
                    }

                    std::ostringstream detail;
                    detail << "event=runtime_skip reason=face_qpts_mismatch"
                           << " domain=" << integralDomainName(domain)
                           << " n_qpts_minus=" << n_qpts_minus
                           << " n_qpts_plus=" << n_qpts_plus;
                    traceSpecialization(this, *fallback_, revision, detail.str());
                }
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
        if (traceSpecializationEnabled()) {
            std::uint64_t revision = 0;
            {
                std::lock_guard<std::mutex> lock(jit_mutex_);
                revision = revision_;
            }

            std::ostringstream detail;
            detail << "event=runtime_skip reason=no_specializable_shape"
                   << " domain=" << integralDomainName(domain)
                   << " n_qpts_minus=" << ctx_minus.numQuadraturePoints()
                   << " n_test_minus=" << ctx_minus.numTestDofs()
                   << " n_trial_minus=" << ctx_minus.numTrialDofs();
            if (face_domain) {
                detail << " n_qpts_plus=" << ctx_plus->numQuadraturePoints()
                       << " n_test_plus=" << ctx_plus->numTestDofs()
                       << " n_trial_plus=" << ctx_plus->numTrialDofs();
            }
            traceSpecialization(this, *fallback_, revision, detail.str());
        }
        return nullptr;
    }

    return compileSpecializedDispatch(role, ir, spec, "runtime");
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
    if (kind_ == WrappedKind::Unknown) {
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
        traceSpecialization(this, *fallback_, compiled_revision_,
                            "event=generic_compile kind=FormKernel cell=" + std::to_string(compiled_form_.cell));
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
        {
            std::ostringstream detail;
            detail << "event=generic_compile kind=LinearFormKernel"
                   << " bilinear_cell=" << compiled_bilinear_.cell
                   << " linear_cell=" << compiled_linear_.cell
                   << " has_linear=" << (has_compiled_linear_ ? 1 : 0);
            traceSpecialization(this, *fallback_, compiled_revision_, detail.str());
        }
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
        {
            std::ostringstream detail;
            detail << "event=generic_compile kind=SymbolicNonlinearFormKernel"
                   << " residual_cell=" << compiled_residual_.cell
                   << " tangent_cell=" << compiled_tangent_.cell;
            traceSpecialization(this, *fallback_, compiled_revision_, detail.str());
        }
        return;
    }

    if (kind_ == WrappedKind::NonlinearFormKernel) {
        const auto* k = dynamic_cast<const NonlinearFormKernel*>(fallback_.get());
        if (!k) {
            warnCompileFailureOnce("NonlinearFormKernel", "dynamic_cast failed");
            return;
        }

        const bool want_matrix = !k->isVectorOnly();
        const bool want_vector = !k->isMatrixOnly();

        // 1) Compile residual
        if (want_vector) {
            const auto r_res = compiler_->compile(k->residualIR(), vopt);
            if (!r_res.ok) {
                warnCompileFailureOnce("NonlinearFormKernel(residual)", r_res.message);
                return;
            }
            fillDispatch(compiled_residual_, r_res);
        } else {
            compiled_residual_ = CompiledDispatch{};
        }

        // Keep the original exact AD tangent on matrix assemblies.
        //
        // The residual still benefits from LLVM JIT on the hot residual-only
        // path, but the coupled Navier-Stokes monolithic cases are more
        // sensitive to tangent fidelity than to tangent assembly throughput.
        compiled_tangent_ = CompiledDispatch{};

        compiled_revision_ = revision_;
        {
            std::ostringstream detail;
            detail << "event=generic_compile kind=NonlinearFormKernel"
                   << " residual_cell=" << compiled_residual_.cell
                   << " tangent_cell=" << compiled_tangent_.cell
                   << " exact_tangent_fallback=1";
            traceSpecialization(this, *fallback_, compiled_revision_, detail.str());
        }
        return;
    }
}

} // namespace jit
} // namespace forms
} // namespace FE
} // namespace svmp

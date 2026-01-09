#ifndef SVMP_FE_ASSEMBLY_JIT_KERNEL_ARGS_H
#define SVMP_FE_ASSEMBLY_JIT_KERNEL_ARGS_H

/**
 * @file KernelArgs.h
 * @brief Versioned POD kernel ABI structs for future JIT backends
 *
 * These structs define a stable, flat ABI for element kernels so that a future
 * LLVM JIT backend can call into compiled kernels without depending on C++
 * object layouts.
 *
 * NOTE: This header intentionally contains no LLVM dependencies.
 */

#include "Assembly/AssemblyContext.h"
#include "Assembly/AssemblyKernel.h"
#include "Core/Types.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <type_traits>

namespace svmp {
namespace FE {
namespace assembly {
namespace jit {

/// Bump when the ABI layout changes incompatibly.
inline constexpr std::uint32_t kKernelArgsABIVersionV1 = 1u;

/// Maximum number of previous solution coefficient vectors passed to kernels.
/// Indexing convention: k=1 corresponds to u^{n-1}.
inline constexpr std::size_t kMaxPreviousSolutionsV1 = 8u;

struct KernelOutputViewV1 {
    Real* element_matrix{nullptr};  // row-major [n_test * n_trial], or nullptr
    Real* element_vector{nullptr};  // [n_test], or nullptr
    std::uint32_t n_test_dofs{0};
    std::uint32_t n_trial_dofs{0};
};

struct KernelSideArgsV1 {
    // Context metadata
    std::uint32_t context_type{static_cast<std::uint32_t>(ContextType::Cell)};
    std::uint32_t dim{0};

    std::uint32_t n_qpts{0};
    std::uint32_t n_test_dofs{0};
    std::uint32_t n_trial_dofs{0};

    std::uint32_t test_field_type{0};
    std::uint32_t trial_field_type{0};
    std::uint32_t test_value_dim{1};
    std::uint32_t trial_value_dim{1};

    GlobalIndex cell_id{-1};
    GlobalIndex face_id{-1};
    std::uint32_t local_face_id{0};
    std::int32_t boundary_marker{-1};

    // Time
    Real time{0.0};
    Real dt{0.0};

    // Entity measures (0 when not provided)
    Real cell_diameter{0.0};
    Real cell_volume{0.0};
    Real facet_area{0.0};

    // Quadrature
    const Real* quad_weights{nullptr};          // [n_qpts]
    const Real* integration_weights{nullptr};   // [n_qpts]
    const Real* quad_points_xyz{nullptr};       // [n_qpts * 3]
    const Real* physical_points_xyz{nullptr};   // [n_qpts * 3]

    // Geometry
    const Real* jacobians{nullptr};             // [n_qpts * 9]
    const Real* inverse_jacobians{nullptr};     // [n_qpts * 9]
    const Real* jacobian_dets{nullptr};         // [n_qpts]
    const Real* normals_xyz{nullptr};           // [n_qpts * 3] (faces), or nullptr

    // Basis tables (row-major in i, then q; vectors/matrices are flattened)
    const Real* test_basis_values{nullptr};         // [n_test_dofs * n_qpts]
    const Real* test_phys_gradients_xyz{nullptr};   // [n_test_dofs * n_qpts * 3]
    const Real* test_phys_hessians{nullptr};        // [n_test_dofs * n_qpts * 9]

    const Real* trial_basis_values{nullptr};        // [n_trial_dofs * n_qpts]
    const Real* trial_phys_gradients_xyz{nullptr};  // [n_trial_dofs * n_qpts * 3]
    const Real* trial_phys_hessians{nullptr};       // [n_trial_dofs * n_qpts * 9]

    // Solution coefficients (optional; required for TrialFunction/StateField lowering)
    const Real* solution_coefficients{nullptr};     // [n_trial_dofs]

    // Previous solution coefficients (optional; used by PreviousSolutionRef)
    std::uint32_t num_previous_solutions{0};
    std::array<const Real*, kMaxPreviousSolutionsV1> previous_solution_coefficients{};

    // Parameter slots / coupled values (optional)
    const Real* jit_constants{nullptr};
    std::uint32_t num_jit_constants{0};

    const Real* coupled_integrals{nullptr};
    std::uint32_t num_coupled_integrals{0};

    const Real* coupled_aux{nullptr};
    std::uint32_t num_coupled_aux{0};

    // Symbolic time-derivative coefficients (optional; populated when TimeIntegrationContext is present)
    Real dt1_coeff0{0.0};
    Real dt2_coeff0{0.0};
    Real time_derivative_term_weight{1.0};
    Real non_time_derivative_term_weight{1.0};
    Real dt1_term_weight{1.0};
    Real dt2_term_weight{1.0};

    // Per-qpt material state (optional; used by constitutive models and history variables)
    const std::byte* material_state_old_base{nullptr};
    std::byte* material_state_work_base{nullptr};
    std::uint64_t material_state_bytes_per_qpt{0};
    std::uint64_t material_state_stride_bytes{0};

    // Opaque pointer for external-call trampolines (coefficients, constitutive models, etc.)
    const void* user_data{nullptr};
};

struct CellKernelArgsV1 {
    std::uint32_t abi_version{kKernelArgsABIVersionV1};
    KernelSideArgsV1 side{};
    KernelOutputViewV1 output{};
};

struct BoundaryFaceKernelArgsV1 {
    std::uint32_t abi_version{kKernelArgsABIVersionV1};
    KernelSideArgsV1 side{};
    KernelOutputViewV1 output{};
};

struct InteriorFaceKernelArgsV1 {
    std::uint32_t abi_version{kKernelArgsABIVersionV1};
    KernelSideArgsV1 minus{};
    KernelSideArgsV1 plus{};

    KernelOutputViewV1 output_minus{};
    KernelOutputViewV1 output_plus{};
    KernelOutputViewV1 coupling_minus_plus{};
    KernelOutputViewV1 coupling_plus_minus{};
};

static_assert(std::is_standard_layout_v<KernelOutputViewV1>);
static_assert(std::is_trivially_copyable_v<KernelOutputViewV1>);
static_assert(std::is_standard_layout_v<KernelSideArgsV1>);
static_assert(std::is_trivially_copyable_v<KernelSideArgsV1>);
static_assert(std::is_standard_layout_v<CellKernelArgsV1>);
static_assert(std::is_trivially_copyable_v<CellKernelArgsV1>);
static_assert(std::is_standard_layout_v<BoundaryFaceKernelArgsV1>);
static_assert(std::is_trivially_copyable_v<BoundaryFaceKernelArgsV1>);
static_assert(std::is_standard_layout_v<InteriorFaceKernelArgsV1>);
static_assert(std::is_trivially_copyable_v<InteriorFaceKernelArgsV1>);

namespace detail {

[[nodiscard]] inline const Real* flattenXYZ(std::span<const std::array<Real, 3>> a) noexcept
{
    return a.empty() ? nullptr : a.data()->data();
}

[[nodiscard]] inline const Real* flattenMat3(std::span<const AssemblyContext::Matrix3x3> mats) noexcept
{
    if (mats.empty()) return nullptr;
    return &(*mats.data())[0][0];
}

[[nodiscard]] inline KernelOutputViewV1 packOutputView(KernelOutput& output) noexcept
{
    KernelOutputViewV1 out;
    out.element_matrix = output.local_matrix.empty() ? nullptr : output.local_matrix.data();
    out.element_vector = output.local_vector.empty() ? nullptr : output.local_vector.data();
    out.n_test_dofs = static_cast<std::uint32_t>(output.n_test_dofs);
    out.n_trial_dofs = static_cast<std::uint32_t>(output.n_trial_dofs);
    return out;
}

[[nodiscard]] inline KernelSideArgsV1 packSideArgs(const AssemblyContext& ctx,
                                                   std::optional<int> override_boundary_marker = std::nullopt) noexcept
{
    KernelSideArgsV1 out;

    out.context_type = static_cast<std::uint32_t>(ctx.contextType());
    out.dim = static_cast<std::uint32_t>(ctx.dimension());

    out.n_qpts = static_cast<std::uint32_t>(ctx.numQuadraturePoints());
    out.n_test_dofs = static_cast<std::uint32_t>(ctx.numTestDofs());
    out.n_trial_dofs = static_cast<std::uint32_t>(ctx.numTrialDofs());

    out.test_field_type = static_cast<std::uint32_t>(ctx.testFieldType());
    out.trial_field_type = static_cast<std::uint32_t>(ctx.trialFieldType());
    out.test_value_dim = static_cast<std::uint32_t>(ctx.testValueDimension());
    out.trial_value_dim = static_cast<std::uint32_t>(ctx.trialValueDimension());

    out.cell_id = ctx.cellId();
    out.face_id = ctx.faceId();
    out.local_face_id = static_cast<std::uint32_t>(ctx.localFaceId());
    out.boundary_marker = static_cast<std::int32_t>(override_boundary_marker ? *override_boundary_marker
                                                                              : ctx.boundaryMarker());

    out.time = ctx.time();
    out.dt = ctx.timeStep();

    out.cell_diameter = ctx.cellDiameter();
    out.cell_volume = (ctx.contextType() == ContextType::Cell) ? ctx.cellVolume() : Real(0.0);
    out.facet_area = (ctx.contextType() == ContextType::Cell) ? Real(0.0) : ctx.facetArea();

    out.quad_weights = ctx.quadratureWeights().empty() ? nullptr : ctx.quadratureWeights().data();
    out.integration_weights = ctx.integrationWeights().empty() ? nullptr : ctx.integrationWeights().data();
    out.quad_points_xyz = flattenXYZ(ctx.quadraturePoints());
    out.physical_points_xyz = flattenXYZ(ctx.physicalPoints());

    out.jacobians = flattenMat3(ctx.jacobians());
    out.inverse_jacobians = flattenMat3(ctx.inverseJacobians());
    out.jacobian_dets = ctx.jacobianDets().empty() ? nullptr : ctx.jacobianDets().data();
    out.normals_xyz = flattenXYZ(ctx.normals());

    out.test_basis_values = ctx.testBasisValuesRaw().empty() ? nullptr : ctx.testBasisValuesRaw().data();
    out.test_phys_gradients_xyz = flattenXYZ(ctx.testPhysicalGradientsRaw());
    out.test_phys_hessians = flattenMat3(ctx.testPhysicalHessiansRaw());

    out.trial_basis_values = ctx.trialBasisValuesRaw().empty() ? nullptr : ctx.trialBasisValuesRaw().data();
    out.trial_phys_gradients_xyz = flattenXYZ(ctx.trialPhysicalGradientsRaw());
    out.trial_phys_hessians = flattenMat3(ctx.trialPhysicalHessiansRaw());

    out.solution_coefficients =
        ctx.solutionCoefficients().empty() ? nullptr : ctx.solutionCoefficients().data();

    const auto history = ctx.previousSolutionHistoryCount();
    out.num_previous_solutions = static_cast<std::uint32_t>(std::min(history, kMaxPreviousSolutionsV1));
    for (std::size_t i = 0; i < out.previous_solution_coefficients.size(); ++i) {
        out.previous_solution_coefficients[i] = nullptr;
    }
    for (std::size_t i = 0; i < static_cast<std::size_t>(out.num_previous_solutions); ++i) {
        const auto coeffs = ctx.previousSolutionCoefficientsRaw(static_cast<int>(i + 1));
        out.previous_solution_coefficients[i] = coeffs.empty() ? nullptr : coeffs.data();
    }

    out.jit_constants = ctx.jitConstants().empty() ? nullptr : ctx.jitConstants().data();
    out.num_jit_constants = static_cast<std::uint32_t>(ctx.jitConstants().size());

    out.coupled_integrals = ctx.coupledIntegrals().empty() ? nullptr : ctx.coupledIntegrals().data();
    out.num_coupled_integrals = static_cast<std::uint32_t>(ctx.coupledIntegrals().size());

    out.coupled_aux = ctx.coupledAuxState().empty() ? nullptr : ctx.coupledAuxState().data();
    out.num_coupled_aux = static_cast<std::uint32_t>(ctx.coupledAuxState().size());

    if (const auto* ti = ctx.timeIntegrationContext()) {
        if (const auto* s1 = ti->stencil(1)) out.dt1_coeff0 = s1->coeff(0);
        if (const auto* s2 = ti->stencil(2)) out.dt2_coeff0 = s2->coeff(0);
        out.time_derivative_term_weight = ti->time_derivative_term_weight;
        out.non_time_derivative_term_weight = ti->non_time_derivative_term_weight;
        out.dt1_term_weight = ti->dt1_term_weight;
        out.dt2_term_weight = ti->dt2_term_weight;
    }

    out.material_state_old_base = ctx.materialStateOldBase();
    out.material_state_work_base = ctx.materialStateWorkBase();
    out.material_state_bytes_per_qpt = static_cast<std::uint64_t>(ctx.materialStateBytesPerQpt());
    out.material_state_stride_bytes = static_cast<std::uint64_t>(ctx.materialStateStrideBytes());

    out.user_data = ctx.userData();
    return out;
}

} // namespace detail

[[nodiscard]] inline CellKernelArgsV1 packCellKernelArgsV1(const AssemblyContext& ctx,
                                                           KernelOutput& output) noexcept
{
    CellKernelArgsV1 out;
    out.side = detail::packSideArgs(ctx);
    out.output = detail::packOutputView(output);
    return out;
}

[[nodiscard]] inline BoundaryFaceKernelArgsV1 packBoundaryFaceKernelArgsV1(const AssemblyContext& ctx,
                                                                           int boundary_marker,
                                                                           KernelOutput& output) noexcept
{
    BoundaryFaceKernelArgsV1 out;
    out.side = detail::packSideArgs(ctx, boundary_marker);
    out.output = detail::packOutputView(output);
    return out;
}

[[nodiscard]] inline InteriorFaceKernelArgsV1 packInteriorFaceKernelArgsV1(const AssemblyContext& ctx_minus,
                                                                           const AssemblyContext& ctx_plus,
                                                                           KernelOutput& output_minus,
                                                                           KernelOutput& output_plus,
                                                                           KernelOutput& coupling_minus_plus,
                                                                           KernelOutput& coupling_plus_minus) noexcept
{
    InteriorFaceKernelArgsV1 out;
    out.minus = detail::packSideArgs(ctx_minus);
    out.plus = detail::packSideArgs(ctx_plus);

    out.output_minus = detail::packOutputView(output_minus);
    out.output_plus = detail::packOutputView(output_plus);
    out.coupling_minus_plus = detail::packOutputView(coupling_minus_plus);
    out.coupling_plus_minus = detail::packOutputView(coupling_plus_minus);
    return out;
}

} // namespace jit
} // namespace assembly
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_ASSEMBLY_JIT_KERNEL_ARGS_H

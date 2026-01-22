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
#include "Assembly/JIT/FieldSolutions.h"
#include "Core/FEConfig.h"
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
inline constexpr std::uint32_t kKernelArgsABIVersionV2 = 2u;
inline constexpr std::uint32_t kKernelArgsABIVersionV3 = 3u;

/// Maximum number of previous solution coefficient vectors passed to kernels.
/// Indexing convention: k=1 corresponds to u^{n-1}.
inline constexpr std::size_t kMaxPreviousSolutionsV1 = 8u;
inline constexpr std::size_t kMaxPreviousSolutionsV2 = kMaxPreviousSolutionsV1;
inline constexpr std::size_t kMaxPreviousSolutionsV3 = kMaxPreviousSolutionsV1;

struct PackingChecks {
    bool validate_alignment{false};
};

/**
 * Kernel output contract (all ABI versions):
 * - The caller provides output buffers sized to `n_test_dofs` and `n_trial_dofs`.
 * - The caller zero-initializes these buffers before invoking the kernel.
 * - Kernels accumulate (+=) contributions into the buffers.
 * - Matrices use row-major storage: entry(i,j) is at `i*n_trial_dofs + j`.
 */
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

// =============================================================================
// ABI v2 (adds marker/domain id + vector-basis tables)
// =============================================================================

struct KernelOutputViewV2 {
    Real* element_matrix{nullptr};  // row-major [n_test * n_trial], or nullptr
    Real* element_vector{nullptr};  // [n_test], or nullptr
    std::uint32_t n_test_dofs{0};
    std::uint32_t n_trial_dofs{0};
};

struct KernelSideArgsV2 {
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

    std::uint32_t test_uses_vector_basis{0};   // 0/1
    std::uint32_t trial_uses_vector_basis{0};  // 0/1

    GlobalIndex cell_id{-1};
    GlobalIndex face_id{-1};
    std::uint32_t local_face_id{0};

    // Markers
    std::int32_t boundary_marker{-1};
    std::int32_t interface_marker{-1};
    std::int32_t cell_domain_id{0};

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

    // Scalar basis tables (row-major in i, then q; vectors/matrices are flattened)
    const Real* test_basis_values{nullptr};         // [n_test_dofs * n_qpts]
    const Real* test_phys_gradients_xyz{nullptr};   // [n_test_dofs * n_qpts * 3]
    const Real* test_phys_hessians{nullptr};        // [n_test_dofs * n_qpts * 9]

    const Real* trial_basis_values{nullptr};        // [n_trial_dofs * n_qpts]
    const Real* trial_phys_gradients_xyz{nullptr};  // [n_trial_dofs * n_qpts * 3]
    const Real* trial_phys_hessians{nullptr};       // [n_trial_dofs * n_qpts * 9]

    // Vector basis tables (H(curl)/H(div); row-major in i, then q)
    const Real* test_basis_vector_values_xyz{nullptr};  // [n_test_dofs * n_qpts * 3]
    const Real* test_basis_curls_xyz{nullptr};          // [n_test_dofs * n_qpts * 3]
    const Real* test_basis_divergences{nullptr};        // [n_test_dofs * n_qpts]

    const Real* trial_basis_vector_values_xyz{nullptr}; // [n_trial_dofs * n_qpts * 3]
    const Real* trial_basis_curls_xyz{nullptr};         // [n_trial_dofs * n_qpts * 3]
    const Real* trial_basis_divergences{nullptr};       // [n_trial_dofs * n_qpts]

    // Solution coefficients (optional; required for TrialFunction/StateField lowering)
    const Real* solution_coefficients{nullptr};     // [n_trial_dofs]

    // Previous solution coefficients (optional; used by PreviousSolutionRef)
    std::uint32_t num_previous_solutions{0};
    std::array<const Real*, kMaxPreviousSolutionsV2> previous_solution_coefficients{};

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

struct CellKernelArgsV2 {
    std::uint32_t abi_version{kKernelArgsABIVersionV2};
    KernelSideArgsV2 side{};
    KernelOutputViewV2 output{};
};

struct BoundaryFaceKernelArgsV2 {
    std::uint32_t abi_version{kKernelArgsABIVersionV2};
    KernelSideArgsV2 side{};
    KernelOutputViewV2 output{};
};

struct InteriorFaceKernelArgsV2 {
    std::uint32_t abi_version{kKernelArgsABIVersionV2};
    KernelSideArgsV2 minus{};
    KernelSideArgsV2 plus{};

    KernelOutputViewV2 output_minus{};
    KernelOutputViewV2 output_plus{};
    KernelOutputViewV2 coupling_minus_plus{};
    KernelOutputViewV2 coupling_plus_minus{};
};

static_assert(std::is_standard_layout_v<KernelOutputViewV2>);
static_assert(std::is_trivially_copyable_v<KernelOutputViewV2>);
static_assert(std::is_standard_layout_v<KernelSideArgsV2>);
static_assert(std::is_trivially_copyable_v<KernelSideArgsV2>);
static_assert(std::is_standard_layout_v<CellKernelArgsV2>);
static_assert(std::is_trivially_copyable_v<CellKernelArgsV2>);
static_assert(std::is_standard_layout_v<BoundaryFaceKernelArgsV2>);
static_assert(std::is_trivially_copyable_v<BoundaryFaceKernelArgsV2>);
static_assert(std::is_standard_layout_v<InteriorFaceKernelArgsV2>);
static_assert(std::is_trivially_copyable_v<InteriorFaceKernelArgsV2>);

// =============================================================================
// ABI v3 (adds multi-field DiscreteField/StateField tables)
// =============================================================================

struct KernelOutputViewV3 {
    Real* element_matrix{nullptr};  // row-major [n_test * n_trial], or nullptr
    Real* element_vector{nullptr};  // [n_test], or nullptr
    std::uint32_t n_test_dofs{0};
    std::uint32_t n_trial_dofs{0};
};

struct KernelSideArgsV3 {
    // Context metadata
    std::uint32_t context_type{static_cast<std::uint32_t>(ContextType::Cell)};
    std::uint32_t dim{0};
    std::uint32_t pointer_alignment_bytes{0};  // alignment guarantee for Real-valued pointer fields in this struct

    std::uint32_t n_qpts{0};
    std::uint32_t n_test_dofs{0};
    std::uint32_t n_trial_dofs{0};

    std::uint32_t test_field_type{0};
    std::uint32_t trial_field_type{0};
    std::uint32_t test_value_dim{1};
    std::uint32_t trial_value_dim{1};

    std::uint32_t test_uses_vector_basis{0};   // 0/1
    std::uint32_t trial_uses_vector_basis{0};  // 0/1

    GlobalIndex cell_id{-1};
    GlobalIndex face_id{-1};
    std::uint32_t local_face_id{0};

    // Markers
    std::int32_t boundary_marker{-1};
    std::int32_t interface_marker{-1};
    std::int32_t cell_domain_id{0};

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

    // Scalar basis tables (row-major in i, then q; vectors/matrices are flattened)
    const Real* test_basis_values{nullptr};         // [n_test_dofs * n_qpts]
    const Real* test_phys_gradients_xyz{nullptr};   // [n_test_dofs * n_qpts * 3]
    const Real* test_phys_hessians{nullptr};        // [n_test_dofs * n_qpts * 9]

    const Real* trial_basis_values{nullptr};        // [n_trial_dofs * n_qpts]
    const Real* trial_phys_gradients_xyz{nullptr};  // [n_trial_dofs * n_qpts * 3]
    const Real* trial_phys_hessians{nullptr};       // [n_trial_dofs * n_qpts * 9]

    // Vector basis tables (H(curl)/H(div); row-major in i, then q)
    const Real* test_basis_vector_values_xyz{nullptr};  // [n_test_dofs * n_qpts * 3]
    const Real* test_basis_curls_xyz{nullptr};          // [n_test_dofs * n_qpts * 3]
    const Real* test_basis_divergences{nullptr};        // [n_test_dofs * n_qpts]

    const Real* trial_basis_vector_values_xyz{nullptr}; // [n_trial_dofs * n_qpts * 3]
    const Real* trial_basis_curls_xyz{nullptr};         // [n_trial_dofs * n_qpts * 3]
    const Real* trial_basis_divergences{nullptr};       // [n_trial_dofs * n_qpts]

    // Solution coefficients (optional; required for TrialFunction/StateField lowering)
    const Real* solution_coefficients{nullptr};     // [n_trial_dofs]

    // Previous solution coefficients (optional; used by PreviousSolutionRef)
    std::uint32_t num_previous_solutions{0};
    std::array<const Real*, kMaxPreviousSolutionsV3> previous_solution_coefficients{};

    // Multi-field discrete/state fields evaluated at quadrature points (optional)
    const FieldSolutionEntryV1* field_solutions{nullptr};
    std::uint32_t num_field_solutions{0};

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
    std::uint64_t material_state_alignment_bytes{0};

    // Opaque pointer for external-call trampolines (coefficients, constitutive models, etc.)
    const void* user_data{nullptr};
};

struct CellKernelArgsV3 {
    std::uint32_t abi_version{kKernelArgsABIVersionV3};
    KernelSideArgsV3 side{};
    KernelOutputViewV3 output{};
};

struct BoundaryFaceKernelArgsV3 {
    std::uint32_t abi_version{kKernelArgsABIVersionV3};
    KernelSideArgsV3 side{};
    KernelOutputViewV3 output{};
};

struct InteriorFaceKernelArgsV3 {
    std::uint32_t abi_version{kKernelArgsABIVersionV3};
    KernelSideArgsV3 minus{};
    KernelSideArgsV3 plus{};

    KernelOutputViewV3 output_minus{};
    KernelOutputViewV3 output_plus{};
    KernelOutputViewV3 coupling_minus_plus{};
    KernelOutputViewV3 coupling_plus_minus{};
};

static_assert(std::is_standard_layout_v<KernelOutputViewV3>);
static_assert(std::is_trivially_copyable_v<KernelOutputViewV3>);
static_assert(std::is_standard_layout_v<KernelSideArgsV3>);
static_assert(std::is_trivially_copyable_v<KernelSideArgsV3>);
static_assert(std::is_standard_layout_v<CellKernelArgsV3>);
static_assert(std::is_trivially_copyable_v<CellKernelArgsV3>);
static_assert(std::is_standard_layout_v<BoundaryFaceKernelArgsV3>);
static_assert(std::is_trivially_copyable_v<BoundaryFaceKernelArgsV3>);
static_assert(std::is_standard_layout_v<InteriorFaceKernelArgsV3>);
static_assert(std::is_trivially_copyable_v<InteriorFaceKernelArgsV3>);

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

[[nodiscard]] inline KernelOutputViewV2 packOutputViewV2(KernelOutput& output) noexcept
{
    KernelOutputViewV2 out;
    out.element_matrix = output.local_matrix.empty() ? nullptr : output.local_matrix.data();
    out.element_vector = output.local_vector.empty() ? nullptr : output.local_vector.data();
    out.n_test_dofs = static_cast<std::uint32_t>(output.n_test_dofs);
    out.n_trial_dofs = static_cast<std::uint32_t>(output.n_trial_dofs);
    return out;
}

[[nodiscard]] inline KernelSideArgsV2 packSideArgsV2(const AssemblyContext& ctx,
                                                     std::optional<int> override_boundary_marker = std::nullopt,
                                                     std::optional<int> override_interface_marker = std::nullopt) noexcept
{
    KernelSideArgsV2 out;

    out.context_type = static_cast<std::uint32_t>(ctx.contextType());
    out.dim = static_cast<std::uint32_t>(ctx.dimension());

    out.n_qpts = static_cast<std::uint32_t>(ctx.numQuadraturePoints());
    out.n_test_dofs = static_cast<std::uint32_t>(ctx.numTestDofs());
    out.n_trial_dofs = static_cast<std::uint32_t>(ctx.numTrialDofs());

    out.test_field_type = static_cast<std::uint32_t>(ctx.testFieldType());
    out.trial_field_type = static_cast<std::uint32_t>(ctx.trialFieldType());
    out.test_value_dim = static_cast<std::uint32_t>(ctx.testValueDimension());
    out.trial_value_dim = static_cast<std::uint32_t>(ctx.trialValueDimension());

    out.test_uses_vector_basis = ctx.testUsesVectorBasis() ? 1u : 0u;
    out.trial_uses_vector_basis = ctx.trialUsesVectorBasis() ? 1u : 0u;

    out.cell_id = ctx.cellId();
    out.face_id = ctx.faceId();
    out.local_face_id = static_cast<std::uint32_t>(ctx.localFaceId());

    out.boundary_marker = static_cast<std::int32_t>(override_boundary_marker ? *override_boundary_marker
                                                                              : ctx.boundaryMarker());
    out.interface_marker = static_cast<std::int32_t>(override_interface_marker ? *override_interface_marker : -1);
    out.cell_domain_id = static_cast<std::int32_t>(ctx.cellDomainId());

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

    out.test_basis_vector_values_xyz = flattenXYZ(ctx.testBasisVectorValuesRaw());
    out.test_basis_curls_xyz = flattenXYZ(ctx.testBasisCurlsRaw());
    out.test_basis_divergences = ctx.testBasisDivergencesRaw().empty() ? nullptr : ctx.testBasisDivergencesRaw().data();

    out.trial_basis_vector_values_xyz = flattenXYZ(ctx.trialBasisVectorValuesRaw());
    out.trial_basis_curls_xyz = flattenXYZ(ctx.trialBasisCurlsRaw());
    out.trial_basis_divergences = ctx.trialBasisDivergencesRaw().empty() ? nullptr : ctx.trialBasisDivergencesRaw().data();

    out.solution_coefficients =
        ctx.solutionCoefficients().empty() ? nullptr : ctx.solutionCoefficients().data();

    const auto history = ctx.previousSolutionHistoryCount();
    out.num_previous_solutions = static_cast<std::uint32_t>(std::min(history, kMaxPreviousSolutionsV2));
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

[[nodiscard]] inline KernelOutputViewV3 packOutputViewV3(KernelOutput& output) noexcept
{
    KernelOutputViewV3 out;
    out.element_matrix = output.local_matrix.empty() ? nullptr : output.local_matrix.data();
    out.element_vector = output.local_vector.empty() ? nullptr : output.local_vector.data();
    out.n_test_dofs = static_cast<std::uint32_t>(output.n_test_dofs);
    out.n_trial_dofs = static_cast<std::uint32_t>(output.n_trial_dofs);
    return out;
}

[[nodiscard]] inline KernelSideArgsV3 packSideArgsV3(const AssemblyContext& ctx,
                                                     std::optional<int> override_boundary_marker = std::nullopt,
                                                     std::optional<int> override_interface_marker = std::nullopt,
                                                     ::svmp::FE::assembly::jit::PackingChecks checks = {}) noexcept
{
    KernelSideArgsV3 out;

    out.context_type = static_cast<std::uint32_t>(ctx.contextType());
    out.dim = static_cast<std::uint32_t>(ctx.dimension());
    out.pointer_alignment_bytes = static_cast<std::uint32_t>(jit::kJITPointerAlignmentBytes);

    out.n_qpts = static_cast<std::uint32_t>(ctx.numQuadraturePoints());
    out.n_test_dofs = static_cast<std::uint32_t>(ctx.numTestDofs());
    out.n_trial_dofs = static_cast<std::uint32_t>(ctx.numTrialDofs());

    out.test_field_type = static_cast<std::uint32_t>(ctx.testFieldType());
    out.trial_field_type = static_cast<std::uint32_t>(ctx.trialFieldType());
    out.test_value_dim = static_cast<std::uint32_t>(ctx.testValueDimension());
    out.trial_value_dim = static_cast<std::uint32_t>(ctx.trialValueDimension());

    out.test_uses_vector_basis = ctx.testUsesVectorBasis() ? 1u : 0u;
    out.trial_uses_vector_basis = ctx.trialUsesVectorBasis() ? 1u : 0u;

    out.cell_id = ctx.cellId();
    out.face_id = ctx.faceId();
    out.local_face_id = static_cast<std::uint32_t>(ctx.localFaceId());

    out.boundary_marker = static_cast<std::int32_t>(override_boundary_marker ? *override_boundary_marker
                                                                              : ctx.boundaryMarker());
    out.interface_marker = static_cast<std::int32_t>(override_interface_marker ? *override_interface_marker : -1);
    out.cell_domain_id = static_cast<std::int32_t>(ctx.cellDomainId());

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

    out.test_basis_vector_values_xyz = flattenXYZ(ctx.testBasisVectorValuesRaw());
    out.test_basis_curls_xyz = flattenXYZ(ctx.testBasisCurlsRaw());
    out.test_basis_divergences = ctx.testBasisDivergencesRaw().empty() ? nullptr : ctx.testBasisDivergencesRaw().data();

    out.trial_basis_vector_values_xyz = flattenXYZ(ctx.trialBasisVectorValuesRaw());
    out.trial_basis_curls_xyz = flattenXYZ(ctx.trialBasisCurlsRaw());
    out.trial_basis_divergences = ctx.trialBasisDivergencesRaw().empty() ? nullptr : ctx.trialBasisDivergencesRaw().data();

    out.solution_coefficients =
        ctx.solutionCoefficients().empty() ? nullptr : ctx.solutionCoefficients().data();

    const auto history = ctx.previousSolutionHistoryCount();
    out.num_previous_solutions = static_cast<std::uint32_t>(std::min(history, kMaxPreviousSolutionsV3));
    for (std::size_t i = 0; i < out.previous_solution_coefficients.size(); ++i) {
        out.previous_solution_coefficients[i] = nullptr;
    }
    for (std::size_t i = 0; i < static_cast<std::size_t>(out.num_previous_solutions); ++i) {
        const auto coeffs = ctx.previousSolutionCoefficientsRaw(static_cast<int>(i + 1));
        out.previous_solution_coefficients[i] = coeffs.empty() ? nullptr : coeffs.data();
    }

    const auto field_table = ctx.jitFieldSolutionTable();
    out.field_solutions = field_table.empty() ? nullptr : field_table.data();
    out.num_field_solutions = static_cast<std::uint32_t>(field_table.size());

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
    out.material_state_alignment_bytes = static_cast<std::uint64_t>(ctx.materialStateAlignmentBytes());

    out.user_data = ctx.userData();

    if (checks.validate_alignment) {
        constexpr auto isPowerOfTwo = [](std::size_t x) noexcept {
            return x != 0u && (x & (x - 1u)) == 0u;
        };
        constexpr auto isAligned = [](const void* ptr, std::size_t alignment) noexcept {
            if (ptr == nullptr) return true;
            if (alignment == 0u) return false;
            return (reinterpret_cast<std::uintptr_t>(ptr) % alignment) == 0u;
        };

        const auto align = static_cast<std::size_t>(out.pointer_alignment_bytes);
        FE_ASSERT_MSG(isPowerOfTwo(align), "KernelArgsV3: pointer_alignment_bytes must be power-of-two");

        auto assertAligned = [&](const void* ptr, const char* msg) noexcept {
            FE_ASSERT_MSG(isAligned(ptr, align), msg);
        };

        assertAligned(out.quad_weights, "KernelArgsV3: quad_weights not aligned");
        assertAligned(out.integration_weights, "KernelArgsV3: integration_weights not aligned");
        assertAligned(out.quad_points_xyz, "KernelArgsV3: quad_points_xyz not aligned");
        assertAligned(out.physical_points_xyz, "KernelArgsV3: physical_points_xyz not aligned");

        assertAligned(out.jacobians, "KernelArgsV3: jacobians not aligned");
        assertAligned(out.inverse_jacobians, "KernelArgsV3: inverse_jacobians not aligned");
        assertAligned(out.jacobian_dets, "KernelArgsV3: jacobian_dets not aligned");
        assertAligned(out.normals_xyz, "KernelArgsV3: normals_xyz not aligned");

        assertAligned(out.test_basis_values, "KernelArgsV3: test_basis_values not aligned");
        assertAligned(out.test_phys_gradients_xyz, "KernelArgsV3: test_phys_gradients_xyz not aligned");
        assertAligned(out.test_phys_hessians, "KernelArgsV3: test_phys_hessians not aligned");

        assertAligned(out.trial_basis_values, "KernelArgsV3: trial_basis_values not aligned");
        assertAligned(out.trial_phys_gradients_xyz, "KernelArgsV3: trial_phys_gradients_xyz not aligned");
        assertAligned(out.trial_phys_hessians, "KernelArgsV3: trial_phys_hessians not aligned");

        assertAligned(out.test_basis_vector_values_xyz, "KernelArgsV3: test_basis_vector_values_xyz not aligned");
        assertAligned(out.test_basis_curls_xyz, "KernelArgsV3: test_basis_curls_xyz not aligned");
        assertAligned(out.test_basis_divergences, "KernelArgsV3: test_basis_divergences not aligned");

        assertAligned(out.trial_basis_vector_values_xyz, "KernelArgsV3: trial_basis_vector_values_xyz not aligned");
        assertAligned(out.trial_basis_curls_xyz, "KernelArgsV3: trial_basis_curls_xyz not aligned");
        assertAligned(out.trial_basis_divergences, "KernelArgsV3: trial_basis_divergences not aligned");

        assertAligned(out.solution_coefficients, "KernelArgsV3: solution_coefficients not aligned");
        for (const auto* p : out.previous_solution_coefficients) {
            assertAligned(p, "KernelArgsV3: previous_solution_coefficients not aligned");
        }

        assertAligned(out.jit_constants, "KernelArgsV3: jit_constants not aligned");
        assertAligned(out.coupled_integrals, "KernelArgsV3: coupled_integrals not aligned");
        assertAligned(out.coupled_aux, "KernelArgsV3: coupled_aux not aligned");

        if (!field_table.empty()) {
            FE_ASSERT_MSG(isAligned(out.field_solutions, alignof(FieldSolutionEntryV1)),
                          "KernelArgsV3: field_solutions table pointer not aligned");
            for (const auto& e : field_table) {
                assertAligned(e.values, "KernelArgsV3: field_solutions.values not aligned");
                assertAligned(e.gradients_xyz, "KernelArgsV3: field_solutions.gradients_xyz not aligned");
                assertAligned(e.hessians, "KernelArgsV3: field_solutions.hessians not aligned");
                assertAligned(e.laplacians, "KernelArgsV3: field_solutions.laplacians not aligned");

                assertAligned(e.vector_values_xyz, "KernelArgsV3: field_solutions.vector_values_xyz not aligned");
                assertAligned(e.jacobians, "KernelArgsV3: field_solutions.jacobians not aligned");
                assertAligned(e.component_hessians, "KernelArgsV3: field_solutions.component_hessians not aligned");
                assertAligned(e.component_laplacians, "KernelArgsV3: field_solutions.component_laplacians not aligned");

                assertAligned(e.history_values, "KernelArgsV3: field_solutions.history_values not aligned");
                assertAligned(e.history_vector_values_xyz, "KernelArgsV3: field_solutions.history_vector_values_xyz not aligned");
            }
        }

        const auto state_align = static_cast<std::size_t>(out.material_state_alignment_bytes);
        if (out.material_state_work_base != nullptr) {
            FE_ASSERT_MSG(isPowerOfTwo(state_align), "KernelArgsV3: material_state_alignment_bytes must be power-of-two");
            FE_ASSERT_MSG(isAligned(out.material_state_old_base, state_align), "KernelArgsV3: material_state_old_base not aligned");
            FE_ASSERT_MSG(isAligned(out.material_state_work_base, state_align), "KernelArgsV3: material_state_work_base not aligned");
            FE_ASSERT_MSG(out.material_state_stride_bytes % state_align == 0u, "KernelArgsV3: material_state_stride_bytes not aligned");
        }
    }
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

[[nodiscard]] inline CellKernelArgsV2 packCellKernelArgsV2(const AssemblyContext& ctx,
                                                           KernelOutput& output) noexcept
{
    CellKernelArgsV2 out;
    out.side = detail::packSideArgsV2(ctx);
    out.output = detail::packOutputViewV2(output);
    return out;
}

[[nodiscard]] inline BoundaryFaceKernelArgsV2 packBoundaryFaceKernelArgsV2(const AssemblyContext& ctx,
                                                                           int boundary_marker,
                                                                           KernelOutput& output) noexcept
{
    BoundaryFaceKernelArgsV2 out;
    out.side = detail::packSideArgsV2(ctx, boundary_marker);
    out.output = detail::packOutputViewV2(output);
    return out;
}

[[nodiscard]] inline InteriorFaceKernelArgsV2 packInteriorFaceKernelArgsV2(const AssemblyContext& ctx_minus,
                                                                           const AssemblyContext& ctx_plus,
                                                                           KernelOutput& output_minus,
                                                                           KernelOutput& output_plus,
                                                                           KernelOutput& coupling_minus_plus,
                                                                           KernelOutput& coupling_plus_minus) noexcept
{
    InteriorFaceKernelArgsV2 out;
    out.minus = detail::packSideArgsV2(ctx_minus);
    out.plus = detail::packSideArgsV2(ctx_plus);

    out.output_minus = detail::packOutputViewV2(output_minus);
    out.output_plus = detail::packOutputViewV2(output_plus);
    out.coupling_minus_plus = detail::packOutputViewV2(coupling_minus_plus);
    out.coupling_plus_minus = detail::packOutputViewV2(coupling_plus_minus);
    return out;
}

[[nodiscard]] inline InteriorFaceKernelArgsV2 packInterfaceFaceKernelArgsV2(const AssemblyContext& ctx_minus,
                                                                            const AssemblyContext& ctx_plus,
                                                                            int interface_marker,
                                                                            KernelOutput& output_minus,
                                                                            KernelOutput& output_plus,
                                                                            KernelOutput& coupling_minus_plus,
                                                                            KernelOutput& coupling_plus_minus) noexcept
{
    InteriorFaceKernelArgsV2 out;
    out.minus = detail::packSideArgsV2(ctx_minus, std::nullopt, interface_marker);
    out.plus = detail::packSideArgsV2(ctx_plus, std::nullopt, interface_marker);

    out.output_minus = detail::packOutputViewV2(output_minus);
    out.output_plus = detail::packOutputViewV2(output_plus);
    out.coupling_minus_plus = detail::packOutputViewV2(coupling_minus_plus);
    out.coupling_plus_minus = detail::packOutputViewV2(coupling_plus_minus);
    return out;
}

[[nodiscard]] inline CellKernelArgsV3 packCellKernelArgsV3(const AssemblyContext& ctx,
                                                          KernelOutput& output,
                                                          PackingChecks checks = {}) noexcept
{
    CellKernelArgsV3 out;
    out.side = detail::packSideArgsV3(ctx, std::nullopt, std::nullopt, checks);
    out.output = detail::packOutputViewV3(output);
    return out;
}

[[nodiscard]] inline BoundaryFaceKernelArgsV3 packBoundaryFaceKernelArgsV3(const AssemblyContext& ctx,
                                                                          int boundary_marker,
                                                                          KernelOutput& output,
                                                                          PackingChecks checks = {}) noexcept
{
    BoundaryFaceKernelArgsV3 out;
    out.side = detail::packSideArgsV3(ctx, boundary_marker, std::nullopt, checks);
    out.output = detail::packOutputViewV3(output);
    return out;
}

[[nodiscard]] inline InteriorFaceKernelArgsV3 packInteriorFaceKernelArgsV3(const AssemblyContext& ctx_minus,
                                                                          const AssemblyContext& ctx_plus,
                                                                          KernelOutput& output_minus,
                                                                          KernelOutput& output_plus,
                                                                          KernelOutput& coupling_minus_plus,
                                                                          KernelOutput& coupling_plus_minus,
                                                                          PackingChecks checks = {}) noexcept
{
    InteriorFaceKernelArgsV3 out;
    out.minus = detail::packSideArgsV3(ctx_minus, std::nullopt, std::nullopt, checks);
    out.plus = detail::packSideArgsV3(ctx_plus, std::nullopt, std::nullopt, checks);

    out.output_minus = detail::packOutputViewV3(output_minus);
    out.output_plus = detail::packOutputViewV3(output_plus);
    out.coupling_minus_plus = detail::packOutputViewV3(coupling_minus_plus);
    out.coupling_plus_minus = detail::packOutputViewV3(coupling_plus_minus);
    return out;
}

[[nodiscard]] inline InteriorFaceKernelArgsV3 packInterfaceFaceKernelArgsV3(const AssemblyContext& ctx_minus,
                                                                           const AssemblyContext& ctx_plus,
                                                                           int interface_marker,
                                                                           KernelOutput& output_minus,
                                                                           KernelOutput& output_plus,
                                                                           KernelOutput& coupling_minus_plus,
                                                                           KernelOutput& coupling_plus_minus,
                                                                           PackingChecks checks = {}) noexcept
{
    InteriorFaceKernelArgsV3 out;
    out.minus = detail::packSideArgsV3(ctx_minus, std::nullopt, interface_marker, checks);
    out.plus = detail::packSideArgsV3(ctx_plus, std::nullopt, interface_marker, checks);

    out.output_minus = detail::packOutputViewV3(output_minus);
    out.output_plus = detail::packOutputViewV3(output_plus);
    out.coupling_minus_plus = detail::packOutputViewV3(coupling_minus_plus);
    out.coupling_plus_minus = detail::packOutputViewV3(coupling_plus_minus);
    return out;
}

} // namespace jit
} // namespace assembly
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_ASSEMBLY_JIT_KERNEL_ARGS_H

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
inline constexpr std::uint32_t kKernelArgsABIVersionV4 = 4u;
inline constexpr std::uint32_t kKernelArgsABIVersionV5 = 5u;
inline constexpr std::uint32_t kKernelArgsABIVersionV6 = 6u;

/// Maximum number of previous solution coefficient vectors passed to kernels.
/// Indexing convention: k=1 corresponds to u^{n-1}.
inline constexpr std::size_t kMaxPreviousSolutionsV1 = 8u;
inline constexpr std::size_t kMaxPreviousSolutionsV2 = kMaxPreviousSolutionsV1;
inline constexpr std::size_t kMaxPreviousSolutionsV3 = kMaxPreviousSolutionsV1;
inline constexpr std::size_t kMaxPreviousSolutionsV4 = kMaxPreviousSolutionsV1;
inline constexpr std::size_t kMaxPreviousSolutionsV5 = kMaxPreviousSolutionsV1;
inline constexpr std::size_t kMaxPreviousSolutionsV6 = kMaxPreviousSolutionsV5;

/// Maximum supported continuous-time derivative order for dt(·,k) coefficient stencils in the ABI.
inline constexpr std::size_t kMaxTimeDerivativeOrderV4 = 8u;
inline constexpr std::size_t kMaxTimeDerivativeOrderV5 = kMaxTimeDerivativeOrderV4;
inline constexpr std::size_t kMaxTimeDerivativeOrderV6 = kMaxTimeDerivativeOrderV5;

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

// =============================================================================
// ABI v4 (generalized dt(·,k) coefficients and term weights)
// =============================================================================

struct KernelOutputViewV4 {
    Real* element_matrix{nullptr};  // row-major [n_test * n_trial], or nullptr
    Real* element_vector{nullptr};  // [n_test], or nullptr
    std::uint32_t n_test_dofs{0};
    std::uint32_t n_trial_dofs{0};
};

struct KernelSideArgsV4 {
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
    std::array<const Real*, kMaxPreviousSolutionsV4> previous_solution_coefficients{};

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
    Real time_derivative_term_weight{1.0};
    Real non_time_derivative_term_weight{1.0};

    // Time derivative stencils for dt(·,k), k=1..kMaxTimeDerivativeOrderV4.
    // Coefficients follow the convention:
    //   d^k u / dt^k ≈ sum_{j=0..kMaxPreviousSolutionsV4} a[j] * u^{n-j}
    // where j=0 is the current state u^n and j>=1 are historical states.
    std::array<std::array<Real, kMaxPreviousSolutionsV4 + 1>, kMaxTimeDerivativeOrderV4> dt_stencil_coeffs{};

    // Optional per-derivative scaling weights (dt term splitting).
    // Term weighting: time_derivative_term_weight * dt_term_weights[k-1].
    std::array<Real, kMaxTimeDerivativeOrderV4> dt_term_weights{
        Real(1.0), Real(1.0), Real(1.0), Real(1.0), Real(1.0), Real(1.0), Real(1.0), Real(1.0)};

    // Maximum dt order provided in dt_stencil_coeffs/dt_term_weights (0 => no transient context).
    std::uint32_t max_time_derivative_order{0};

    // Per-qpt material state (optional; used by constitutive models and history variables)
    const std::byte* material_state_old_base{nullptr};
    std::byte* material_state_work_base{nullptr};
    std::uint64_t material_state_bytes_per_qpt{0};
    std::uint64_t material_state_stride_bytes{0};
    std::uint64_t material_state_alignment_bytes{0};

    // Opaque pointer for external-call trampolines (coefficients, constitutive models, etc.)
    const void* user_data{nullptr};
};

struct CellKernelArgsV4 {
    std::uint32_t abi_version{kKernelArgsABIVersionV4};
    KernelSideArgsV4 side{};
    KernelOutputViewV4 output{};
};

struct BoundaryFaceKernelArgsV4 {
    std::uint32_t abi_version{kKernelArgsABIVersionV4};
    KernelSideArgsV4 side{};
    KernelOutputViewV4 output{};
};

struct InteriorFaceKernelArgsV4 {
    std::uint32_t abi_version{kKernelArgsABIVersionV4};
    KernelSideArgsV4 minus{};
    KernelSideArgsV4 plus{};

    KernelOutputViewV4 output_minus{};
    KernelOutputViewV4 output_plus{};
    KernelOutputViewV4 coupling_minus_plus{};
    KernelOutputViewV4 coupling_plus_minus{};
};

static_assert(std::is_standard_layout_v<KernelOutputViewV4>);
static_assert(std::is_trivially_copyable_v<KernelOutputViewV4>);
static_assert(std::is_standard_layout_v<KernelSideArgsV4>);
static_assert(std::is_trivially_copyable_v<KernelSideArgsV4>);
static_assert(std::is_standard_layout_v<CellKernelArgsV4>);
static_assert(std::is_trivially_copyable_v<CellKernelArgsV4>);
static_assert(std::is_standard_layout_v<BoundaryFaceKernelArgsV4>);
static_assert(std::is_trivially_copyable_v<BoundaryFaceKernelArgsV4>);
static_assert(std::is_standard_layout_v<InteriorFaceKernelArgsV4>);
static_assert(std::is_trivially_copyable_v<InteriorFaceKernelArgsV4>);

// =============================================================================
// ABI v5 (history weights + history coefficient pointers for history operators)
// =============================================================================

struct KernelOutputViewV5 {
    Real* element_matrix{nullptr};  // row-major [n_test * n_trial], or nullptr
    Real* element_vector{nullptr};  // [n_test], or nullptr
    std::uint32_t n_test_dofs{0};
    std::uint32_t n_trial_dofs{0};
};

struct KernelSideArgsV5 {
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

    // Previous solution coefficients (optional; used by PreviousSolutionRef and dt(·))
    std::uint32_t num_previous_solutions{0};
    std::array<const Real*, kMaxPreviousSolutionsV5> previous_solution_coefficients{};

    // History operator data (optional; used by HistoryWeightedSum/HistoryConvolution when no explicit weights are present)
    std::uint32_t num_history_steps{0};
    const Real* history_weights{nullptr};  // [num_history_steps]
    std::array<const Real*, kMaxPreviousSolutionsV5> history_solution_coefficients{};

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
    Real time_derivative_term_weight{1.0};
    Real non_time_derivative_term_weight{1.0};

    // Time derivative stencils for dt(·,k), k=1..kMaxTimeDerivativeOrderV5.
    // Coefficients follow the convention:
    //   d^k u / dt^k ≈ sum_{j=0..kMaxPreviousSolutionsV5} a[j] * u^{n-j}
    // where j=0 is the current state u^n and j>=1 are historical states.
    std::array<std::array<Real, kMaxPreviousSolutionsV5 + 1>, kMaxTimeDerivativeOrderV5> dt_stencil_coeffs{};

    // Optional per-derivative scaling weights (dt term splitting).
    // Term weighting: time_derivative_term_weight * dt_term_weights[k-1].
    std::array<Real, kMaxTimeDerivativeOrderV5> dt_term_weights{
        Real(1.0), Real(1.0), Real(1.0), Real(1.0), Real(1.0), Real(1.0), Real(1.0), Real(1.0)};

    // Maximum dt order provided in dt_stencil_coeffs/dt_term_weights (0 => no transient context).
    std::uint32_t max_time_derivative_order{0};

    // Per-qpt material state (optional; used by constitutive models and history variables)
    const std::byte* material_state_old_base{nullptr};
    std::byte* material_state_work_base{nullptr};
    std::uint64_t material_state_bytes_per_qpt{0};
    std::uint64_t material_state_stride_bytes{0};
    std::uint64_t material_state_alignment_bytes{0};

    // Opaque pointer for external-call trampolines (coefficients, constitutive models, etc.)
    const void* user_data{nullptr};
};

struct CellKernelArgsV5 {
    std::uint32_t abi_version{kKernelArgsABIVersionV5};
    KernelSideArgsV5 side{};
    KernelOutputViewV5 output{};
};

struct BoundaryFaceKernelArgsV5 {
    std::uint32_t abi_version{kKernelArgsABIVersionV5};
    KernelSideArgsV5 side{};
    KernelOutputViewV5 output{};
};

struct InteriorFaceKernelArgsV5 {
    std::uint32_t abi_version{kKernelArgsABIVersionV5};
    KernelSideArgsV5 minus{};
    KernelSideArgsV5 plus{};

    KernelOutputViewV5 output_minus{};
    KernelOutputViewV5 output_plus{};
    KernelOutputViewV5 coupling_minus_plus{};
    KernelOutputViewV5 coupling_plus_minus{};
};

static_assert(std::is_standard_layout_v<KernelOutputViewV5>);
static_assert(std::is_trivially_copyable_v<KernelOutputViewV5>);
static_assert(std::is_standard_layout_v<KernelSideArgsV5>);
static_assert(std::is_trivially_copyable_v<KernelSideArgsV5>);
static_assert(std::is_standard_layout_v<CellKernelArgsV5>);
static_assert(std::is_trivially_copyable_v<CellKernelArgsV5>);
static_assert(std::is_standard_layout_v<BoundaryFaceKernelArgsV5>);
static_assert(std::is_trivially_copyable_v<BoundaryFaceKernelArgsV5>);
static_assert(std::is_standard_layout_v<InteriorFaceKernelArgsV5>);
static_assert(std::is_trivially_copyable_v<InteriorFaceKernelArgsV5>);

// =============================================================================
// ABI v6 (interleaved quadrature-point geometry payload for JIT locality)
// =============================================================================

struct KernelOutputViewV6 {
    Real* element_matrix{nullptr};  // row-major [n_test * n_trial], or nullptr
    Real* element_vector{nullptr};  // [n_test], or nullptr
    std::uint32_t n_test_dofs{0};
    std::uint32_t n_trial_dofs{0};
};

struct KernelSideArgsV6 {
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

    // Geometry (legacy split arrays)
    const Real* jacobians{nullptr};             // [n_qpts * 9]
    const Real* inverse_jacobians{nullptr};     // [n_qpts * 9]
    const Real* jacobian_dets{nullptr};         // [n_qpts]
    const Real* normals_xyz{nullptr};           // [n_qpts * 3] (faces), or nullptr

    // Geometry (interleaved by quadrature point; optional but preferred)
    const Real* interleaved_qpoint_geometry{nullptr}; // [n_qpts * interleaved_qpoint_geometry_stride_reals]
    std::uint32_t interleaved_qpoint_geometry_stride_reals{0};
    std::uint32_t interleaved_qpoint_geometry_physical_offset{AssemblyContext::kInterleavedQPointPhysicalOffset};
    std::uint32_t interleaved_qpoint_geometry_jacobian_offset{AssemblyContext::kInterleavedQPointJacobianOffset};
    std::uint32_t interleaved_qpoint_geometry_inverse_jacobian_offset{AssemblyContext::kInterleavedQPointInverseJacobianOffset};
    std::uint32_t interleaved_qpoint_geometry_det_offset{AssemblyContext::kInterleavedQPointDetOffset};
    std::uint32_t interleaved_qpoint_geometry_normal_offset{AssemblyContext::kInterleavedQPointNormalOffset};

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

    // Previous solution coefficients (optional; used by PreviousSolutionRef and dt(·))
    std::uint32_t num_previous_solutions{0};
    std::array<const Real*, kMaxPreviousSolutionsV6> previous_solution_coefficients{};

    // History operator data (optional; used by HistoryWeightedSum/HistoryConvolution when no explicit weights are present)
    std::uint32_t num_history_steps{0};
    const Real* history_weights{nullptr};  // [num_history_steps]
    std::array<const Real*, kMaxPreviousSolutionsV6> history_solution_coefficients{};

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
    Real time_derivative_term_weight{1.0};
    Real non_time_derivative_term_weight{1.0};

    // Time derivative stencils for dt(·,k), k=1..kMaxTimeDerivativeOrderV6.
    // Coefficients follow the convention:
    //   d^k u / dt^k ≈ sum_{j=0..kMaxPreviousSolutionsV6} a[j] * u^{n-j}
    // where j=0 is the current state u^n and j>=1 are historical states.
    std::array<std::array<Real, kMaxPreviousSolutionsV6 + 1>, kMaxTimeDerivativeOrderV6> dt_stencil_coeffs{};

    // Optional per-derivative scaling weights (dt term splitting).
    // Term weighting: time_derivative_term_weight * dt_term_weights[k-1].
    std::array<Real, kMaxTimeDerivativeOrderV6> dt_term_weights{
        Real(1.0), Real(1.0), Real(1.0), Real(1.0), Real(1.0), Real(1.0), Real(1.0), Real(1.0)};

    // Maximum dt order provided in dt_stencil_coeffs/dt_term_weights (0 => no transient context).
    std::uint32_t max_time_derivative_order{0};

    // Per-qpt material state (optional; used by constitutive models and history variables)
    const std::byte* material_state_old_base{nullptr};
    std::byte* material_state_work_base{nullptr};
    std::uint64_t material_state_bytes_per_qpt{0};
    std::uint64_t material_state_stride_bytes{0};
    std::uint64_t material_state_alignment_bytes{0};

    // Opaque pointer for external-call trampolines (coefficients, constitutive models, etc.)
    const void* user_data{nullptr};
};

struct CellKernelArgsV6 {
    std::uint32_t abi_version{kKernelArgsABIVersionV6};
    KernelSideArgsV6 side{};
    KernelOutputViewV6 output{};
};

struct BoundaryFaceKernelArgsV6 {
    std::uint32_t abi_version{kKernelArgsABIVersionV6};
    KernelSideArgsV6 side{};
    KernelOutputViewV6 output{};
};

struct InteriorFaceKernelArgsV6 {
    std::uint32_t abi_version{kKernelArgsABIVersionV6};
    KernelSideArgsV6 minus{};
    KernelSideArgsV6 plus{};

    KernelOutputViewV6 output_minus{};
    KernelOutputViewV6 output_plus{};
    KernelOutputViewV6 coupling_minus_plus{};
    KernelOutputViewV6 coupling_plus_minus{};
};

static_assert(std::is_standard_layout_v<KernelOutputViewV6>);
static_assert(std::is_trivially_copyable_v<KernelOutputViewV6>);
static_assert(std::is_standard_layout_v<KernelSideArgsV6>);
static_assert(std::is_trivially_copyable_v<KernelSideArgsV6>);
static_assert(std::is_standard_layout_v<CellKernelArgsV6>);
static_assert(std::is_trivially_copyable_v<CellKernelArgsV6>);
static_assert(std::is_standard_layout_v<BoundaryFaceKernelArgsV6>);
static_assert(std::is_trivially_copyable_v<BoundaryFaceKernelArgsV6>);
static_assert(std::is_standard_layout_v<InteriorFaceKernelArgsV6>);
static_assert(std::is_trivially_copyable_v<InteriorFaceKernelArgsV6>);

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

[[nodiscard]] inline KernelOutputViewV4 packOutputViewV4(KernelOutput& output) noexcept
{
    KernelOutputViewV4 out;
    out.element_matrix = output.local_matrix.empty() ? nullptr : output.local_matrix.data();
    out.element_vector = output.local_vector.empty() ? nullptr : output.local_vector.data();
    out.n_test_dofs = static_cast<std::uint32_t>(output.n_test_dofs);
    out.n_trial_dofs = static_cast<std::uint32_t>(output.n_trial_dofs);
    return out;
}

[[nodiscard]] inline KernelOutputViewV5 packOutputViewV5(KernelOutput& output) noexcept
{
    KernelOutputViewV5 out;
    out.element_matrix = output.local_matrix.empty() ? nullptr : output.local_matrix.data();
    out.element_vector = output.local_vector.empty() ? nullptr : output.local_vector.data();
    out.n_test_dofs = static_cast<std::uint32_t>(output.n_test_dofs);
    out.n_trial_dofs = static_cast<std::uint32_t>(output.n_trial_dofs);
    return out;
}

[[nodiscard]] inline KernelOutputViewV6 packOutputViewV6(KernelOutput& output) noexcept
{
    KernelOutputViewV6 out;
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

[[nodiscard]] inline KernelSideArgsV4 packSideArgsV4(const AssemblyContext& ctx,
                                                     std::optional<int> override_boundary_marker = std::nullopt,
                                                     std::optional<int> override_interface_marker = std::nullopt,
                                                     ::svmp::FE::assembly::jit::PackingChecks checks = {}) noexcept
{
    KernelSideArgsV4 out;

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
    out.num_previous_solutions = static_cast<std::uint32_t>(std::min(history, kMaxPreviousSolutionsV4));
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
        out.time_derivative_term_weight = ti->time_derivative_term_weight;
        out.non_time_derivative_term_weight = ti->non_time_derivative_term_weight;

        std::uint32_t max_order = 0;
        for (std::size_t order = 1; order <= kMaxTimeDerivativeOrderV4; ++order) {
            const auto* stencil = ti->stencil(static_cast<int>(order));
            if (!stencil) {
                continue;
            }
            max_order = static_cast<std::uint32_t>(order);
            for (std::size_t j = 0; j <= kMaxPreviousSolutionsV4; ++j) {
                out.dt_stencil_coeffs[order - 1u][j] = stencil->coeff(static_cast<int>(j));
            }
            out.dt_term_weights[order - 1u] = ti->derivativeTermWeight(static_cast<int>(order));
        }
        out.max_time_derivative_order = max_order;
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
        FE_ASSERT_MSG(isPowerOfTwo(align), "KernelArgsV4: pointer_alignment_bytes must be power-of-two");

        auto assertAligned = [&](const void* ptr, const char* msg) noexcept {
            FE_ASSERT_MSG(isAligned(ptr, align), msg);
        };

        assertAligned(out.quad_weights, "KernelArgsV4: quad_weights not aligned");
        assertAligned(out.integration_weights, "KernelArgsV4: integration_weights not aligned");
        assertAligned(out.quad_points_xyz, "KernelArgsV4: quad_points_xyz not aligned");
        assertAligned(out.physical_points_xyz, "KernelArgsV4: physical_points_xyz not aligned");

        assertAligned(out.jacobians, "KernelArgsV4: jacobians not aligned");
        assertAligned(out.inverse_jacobians, "KernelArgsV4: inverse_jacobians not aligned");
        assertAligned(out.jacobian_dets, "KernelArgsV4: jacobian_dets not aligned");
        assertAligned(out.normals_xyz, "KernelArgsV4: normals_xyz not aligned");

        assertAligned(out.test_basis_values, "KernelArgsV4: test_basis_values not aligned");
        assertAligned(out.test_phys_gradients_xyz, "KernelArgsV4: test_phys_gradients_xyz not aligned");
        assertAligned(out.test_phys_hessians, "KernelArgsV4: test_phys_hessians not aligned");

        assertAligned(out.trial_basis_values, "KernelArgsV4: trial_basis_values not aligned");
        assertAligned(out.trial_phys_gradients_xyz, "KernelArgsV4: trial_phys_gradients_xyz not aligned");
        assertAligned(out.trial_phys_hessians, "KernelArgsV4: trial_phys_hessians not aligned");

        assertAligned(out.test_basis_vector_values_xyz, "KernelArgsV4: test_basis_vector_values_xyz not aligned");
        assertAligned(out.test_basis_curls_xyz, "KernelArgsV4: test_basis_curls_xyz not aligned");
        assertAligned(out.test_basis_divergences, "KernelArgsV4: test_basis_divergences not aligned");

        assertAligned(out.trial_basis_vector_values_xyz, "KernelArgsV4: trial_basis_vector_values_xyz not aligned");
        assertAligned(out.trial_basis_curls_xyz, "KernelArgsV4: trial_basis_curls_xyz not aligned");
        assertAligned(out.trial_basis_divergences, "KernelArgsV4: trial_basis_divergences not aligned");

        assertAligned(out.solution_coefficients, "KernelArgsV4: solution_coefficients not aligned");
        for (const auto* p : out.previous_solution_coefficients) {
            assertAligned(p, "KernelArgsV4: previous_solution_coefficients not aligned");
        }

        assertAligned(out.jit_constants, "KernelArgsV4: jit_constants not aligned");
        assertAligned(out.coupled_integrals, "KernelArgsV4: coupled_integrals not aligned");
        assertAligned(out.coupled_aux, "KernelArgsV4: coupled_aux not aligned");

        if (!field_table.empty()) {
            FE_ASSERT_MSG(isAligned(out.field_solutions, alignof(FieldSolutionEntryV1)),
                          "KernelArgsV4: field_solutions table pointer not aligned");
            for (const auto& e : field_table) {
                assertAligned(e.values, "KernelArgsV4: field_solutions.values not aligned");
                assertAligned(e.gradients_xyz, "KernelArgsV4: field_solutions.gradients_xyz not aligned");
                assertAligned(e.hessians, "KernelArgsV4: field_solutions.hessians not aligned");
                assertAligned(e.laplacians, "KernelArgsV4: field_solutions.laplacians not aligned");

                assertAligned(e.vector_values_xyz, "KernelArgsV4: field_solutions.vector_values_xyz not aligned");
                assertAligned(e.jacobians, "KernelArgsV4: field_solutions.jacobians not aligned");
                assertAligned(e.component_hessians, "KernelArgsV4: field_solutions.component_hessians not aligned");
                assertAligned(e.component_laplacians, "KernelArgsV4: field_solutions.component_laplacians not aligned");

                assertAligned(e.history_values, "KernelArgsV4: field_solutions.history_values not aligned");
                assertAligned(e.history_vector_values_xyz, "KernelArgsV4: field_solutions.history_vector_values_xyz not aligned");
            }
        }

        const auto state_align = static_cast<std::size_t>(out.material_state_alignment_bytes);
        if (out.material_state_work_base != nullptr) {
            FE_ASSERT_MSG(isPowerOfTwo(state_align), "KernelArgsV4: material_state_alignment_bytes must be power-of-two");
            FE_ASSERT_MSG(isAligned(out.material_state_old_base, state_align), "KernelArgsV4: material_state_old_base not aligned");
            FE_ASSERT_MSG(isAligned(out.material_state_work_base, state_align), "KernelArgsV4: material_state_work_base not aligned");
            FE_ASSERT_MSG(out.material_state_stride_bytes % state_align == 0u, "KernelArgsV4: material_state_stride_bytes not aligned");
        }
    }

    return out;
}

[[nodiscard]] inline KernelSideArgsV5 packSideArgsV5(const AssemblyContext& ctx,
                                                     std::optional<int> override_boundary_marker = std::nullopt,
                                                     std::optional<int> override_interface_marker = std::nullopt,
                                                     ::svmp::FE::assembly::jit::PackingChecks checks = {}) noexcept
{
    KernelSideArgsV5 out;

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
    out.num_previous_solutions = static_cast<std::uint32_t>(std::min(history, kMaxPreviousSolutionsV5));
    for (std::size_t i = 0; i < out.previous_solution_coefficients.size(); ++i) {
        out.previous_solution_coefficients[i] = nullptr;
    }
    for (std::size_t i = 0; i < static_cast<std::size_t>(out.num_previous_solutions); ++i) {
        const auto coeffs = ctx.previousSolutionCoefficientsRaw(static_cast<int>(i + 1));
        out.previous_solution_coefficients[i] = coeffs.empty() ? nullptr : coeffs.data();
    }

    const auto weights = ctx.historyWeights();
    out.num_history_steps =
        static_cast<std::uint32_t>(std::min<std::size_t>({weights.size(), history, kMaxPreviousSolutionsV5}));
    out.history_weights = (out.num_history_steps == 0u) ? nullptr : weights.data();
    for (std::size_t i = 0; i < out.history_solution_coefficients.size(); ++i) {
        out.history_solution_coefficients[i] = nullptr;
    }
    for (std::size_t i = 0; i < static_cast<std::size_t>(out.num_previous_solutions); ++i) {
        const auto coeffs = ctx.previousSolutionCoefficientsRaw(static_cast<int>(i + 1));
        out.history_solution_coefficients[i] = coeffs.empty() ? nullptr : coeffs.data();
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
        out.time_derivative_term_weight = ti->time_derivative_term_weight;
        out.non_time_derivative_term_weight = ti->non_time_derivative_term_weight;

        std::uint32_t max_order = 0;
        for (std::size_t order = 1; order <= kMaxTimeDerivativeOrderV5; ++order) {
            const auto* stencil = ti->stencil(static_cast<int>(order));
            if (!stencil) {
                continue;
            }
            max_order = static_cast<std::uint32_t>(order);
            for (std::size_t j = 0; j <= kMaxPreviousSolutionsV5; ++j) {
                out.dt_stencil_coeffs[order - 1u][j] = stencil->coeff(static_cast<int>(j));
            }
            out.dt_term_weights[order - 1u] = ti->derivativeTermWeight(static_cast<int>(order));
        }
        out.max_time_derivative_order = max_order;
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
        FE_ASSERT_MSG(isPowerOfTwo(align), "KernelArgsV5: pointer_alignment_bytes must be power-of-two");

        auto assertAligned = [&](const void* ptr, const char* msg) noexcept {
            FE_ASSERT_MSG(isAligned(ptr, align), msg);
        };

        assertAligned(out.quad_weights, "KernelArgsV5: quad_weights not aligned");
        assertAligned(out.integration_weights, "KernelArgsV5: integration_weights not aligned");
        assertAligned(out.quad_points_xyz, "KernelArgsV5: quad_points_xyz not aligned");
        assertAligned(out.physical_points_xyz, "KernelArgsV5: physical_points_xyz not aligned");

        assertAligned(out.jacobians, "KernelArgsV5: jacobians not aligned");
        assertAligned(out.inverse_jacobians, "KernelArgsV5: inverse_jacobians not aligned");
        assertAligned(out.jacobian_dets, "KernelArgsV5: jacobian_dets not aligned");
        assertAligned(out.normals_xyz, "KernelArgsV5: normals_xyz not aligned");

        assertAligned(out.test_basis_values, "KernelArgsV5: test_basis_values not aligned");
        assertAligned(out.test_phys_gradients_xyz, "KernelArgsV5: test_phys_gradients_xyz not aligned");
        assertAligned(out.test_phys_hessians, "KernelArgsV5: test_phys_hessians not aligned");

        assertAligned(out.trial_basis_values, "KernelArgsV5: trial_basis_values not aligned");
        assertAligned(out.trial_phys_gradients_xyz, "KernelArgsV5: trial_phys_gradients_xyz not aligned");
        assertAligned(out.trial_phys_hessians, "KernelArgsV5: trial_phys_hessians not aligned");

        assertAligned(out.test_basis_vector_values_xyz, "KernelArgsV5: test_basis_vector_values_xyz not aligned");
        assertAligned(out.test_basis_curls_xyz, "KernelArgsV5: test_basis_curls_xyz not aligned");
        assertAligned(out.test_basis_divergences, "KernelArgsV5: test_basis_divergences not aligned");

        assertAligned(out.trial_basis_vector_values_xyz, "KernelArgsV5: trial_basis_vector_values_xyz not aligned");
        assertAligned(out.trial_basis_curls_xyz, "KernelArgsV5: trial_basis_curls_xyz not aligned");
        assertAligned(out.trial_basis_divergences, "KernelArgsV5: trial_basis_divergences not aligned");

        assertAligned(out.solution_coefficients, "KernelArgsV5: solution_coefficients not aligned");
        for (const auto* p : out.previous_solution_coefficients) {
            assertAligned(p, "KernelArgsV5: previous_solution_coefficients not aligned");
        }

        assertAligned(out.history_weights, "KernelArgsV5: history_weights not aligned");
        for (const auto* p : out.history_solution_coefficients) {
            assertAligned(p, "KernelArgsV5: history_solution_coefficients not aligned");
        }

        assertAligned(out.jit_constants, "KernelArgsV5: jit_constants not aligned");
        assertAligned(out.coupled_integrals, "KernelArgsV5: coupled_integrals not aligned");
        assertAligned(out.coupled_aux, "KernelArgsV5: coupled_aux not aligned");

        if (!field_table.empty()) {
            FE_ASSERT_MSG(isAligned(out.field_solutions, alignof(FieldSolutionEntryV1)),
                          "KernelArgsV5: field_solutions table pointer not aligned");
            for (const auto& e : field_table) {
                assertAligned(e.values, "KernelArgsV5: field_solutions.values not aligned");
                assertAligned(e.gradients_xyz, "KernelArgsV5: field_solutions.gradients_xyz not aligned");
                assertAligned(e.hessians, "KernelArgsV5: field_solutions.hessians not aligned");
                assertAligned(e.laplacians, "KernelArgsV5: field_solutions.laplacians not aligned");

                assertAligned(e.vector_values_xyz, "KernelArgsV5: field_solutions.vector_values_xyz not aligned");
                assertAligned(e.jacobians, "KernelArgsV5: field_solutions.jacobians not aligned");
                assertAligned(e.component_hessians, "KernelArgsV5: field_solutions.component_hessians not aligned");
                assertAligned(e.component_laplacians, "KernelArgsV5: field_solutions.component_laplacians not aligned");

                assertAligned(e.history_values, "KernelArgsV5: field_solutions.history_values not aligned");
                assertAligned(e.history_vector_values_xyz, "KernelArgsV5: field_solutions.history_vector_values_xyz not aligned");
            }
        }

        const auto state_align = static_cast<std::size_t>(out.material_state_alignment_bytes);
        if (out.material_state_work_base != nullptr) {
            FE_ASSERT_MSG(isPowerOfTwo(state_align), "KernelArgsV5: material_state_alignment_bytes must be power-of-two");
            FE_ASSERT_MSG(isAligned(out.material_state_old_base, state_align), "KernelArgsV5: material_state_old_base not aligned");
            FE_ASSERT_MSG(isAligned(out.material_state_work_base, state_align), "KernelArgsV5: material_state_work_base not aligned");
            FE_ASSERT_MSG(out.material_state_stride_bytes % state_align == 0u, "KernelArgsV5: material_state_stride_bytes not aligned");
        }
    }

    return out;
}

[[nodiscard]] inline KernelSideArgsV6 packSideArgsV6(const AssemblyContext& ctx,
                                                     std::optional<int> override_boundary_marker = std::nullopt,
                                                     std::optional<int> override_interface_marker = std::nullopt,
                                                     ::svmp::FE::assembly::jit::PackingChecks checks = {}) noexcept
{
    KernelSideArgsV6 out;

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

    const auto interleaved = ctx.interleavedQPointGeometryRaw();
    out.interleaved_qpoint_geometry = interleaved.empty() ? nullptr : interleaved.data();
    out.interleaved_qpoint_geometry_stride_reals =
        out.interleaved_qpoint_geometry == nullptr ? 0u : AssemblyContext::kInterleavedQPointGeometryStride;
    out.interleaved_qpoint_geometry_physical_offset = AssemblyContext::kInterleavedQPointPhysicalOffset;
    out.interleaved_qpoint_geometry_jacobian_offset = AssemblyContext::kInterleavedQPointJacobianOffset;
    out.interleaved_qpoint_geometry_inverse_jacobian_offset = AssemblyContext::kInterleavedQPointInverseJacobianOffset;
    out.interleaved_qpoint_geometry_det_offset = AssemblyContext::kInterleavedQPointDetOffset;
    out.interleaved_qpoint_geometry_normal_offset = AssemblyContext::kInterleavedQPointNormalOffset;

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
    out.num_previous_solutions = static_cast<std::uint32_t>(std::min(history, kMaxPreviousSolutionsV6));
    for (std::size_t i = 0; i < out.previous_solution_coefficients.size(); ++i) {
        out.previous_solution_coefficients[i] = nullptr;
    }
    for (std::size_t i = 0; i < static_cast<std::size_t>(out.num_previous_solutions); ++i) {
        const auto coeffs = ctx.previousSolutionCoefficientsRaw(static_cast<int>(i + 1));
        out.previous_solution_coefficients[i] = coeffs.empty() ? nullptr : coeffs.data();
    }

    const auto weights = ctx.historyWeights();
    out.num_history_steps =
        static_cast<std::uint32_t>(std::min<std::size_t>({weights.size(), history, kMaxPreviousSolutionsV6}));
    out.history_weights = (out.num_history_steps == 0u) ? nullptr : weights.data();
    for (std::size_t i = 0; i < out.history_solution_coefficients.size(); ++i) {
        out.history_solution_coefficients[i] = nullptr;
    }
    for (std::size_t i = 0; i < static_cast<std::size_t>(out.num_previous_solutions); ++i) {
        const auto coeffs = ctx.previousSolutionCoefficientsRaw(static_cast<int>(i + 1));
        out.history_solution_coefficients[i] = coeffs.empty() ? nullptr : coeffs.data();
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
        out.time_derivative_term_weight = ti->time_derivative_term_weight;
        out.non_time_derivative_term_weight = ti->non_time_derivative_term_weight;

        std::uint32_t max_order = 0;
        for (std::size_t order = 1; order <= kMaxTimeDerivativeOrderV6; ++order) {
            const auto* stencil = ti->stencil(static_cast<int>(order));
            if (!stencil) {
                continue;
            }
            max_order = static_cast<std::uint32_t>(order);
            for (std::size_t j = 0; j <= kMaxPreviousSolutionsV6; ++j) {
                out.dt_stencil_coeffs[order - 1u][j] = stencil->coeff(static_cast<int>(j));
            }
            out.dt_term_weights[order - 1u] = ti->derivativeTermWeight(static_cast<int>(order));
        }
        out.max_time_derivative_order = max_order;
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
        FE_ASSERT_MSG(isPowerOfTwo(align), "KernelArgsV6: pointer_alignment_bytes must be power-of-two");

        auto assertAligned = [&](const void* ptr, const char* msg) noexcept {
            FE_ASSERT_MSG(isAligned(ptr, align), msg);
        };

        assertAligned(out.quad_weights, "KernelArgsV6: quad_weights not aligned");
        assertAligned(out.integration_weights, "KernelArgsV6: integration_weights not aligned");
        assertAligned(out.quad_points_xyz, "KernelArgsV6: quad_points_xyz not aligned");
        assertAligned(out.physical_points_xyz, "KernelArgsV6: physical_points_xyz not aligned");

        assertAligned(out.jacobians, "KernelArgsV6: jacobians not aligned");
        assertAligned(out.inverse_jacobians, "KernelArgsV6: inverse_jacobians not aligned");
        assertAligned(out.jacobian_dets, "KernelArgsV6: jacobian_dets not aligned");
        assertAligned(out.normals_xyz, "KernelArgsV6: normals_xyz not aligned");
        assertAligned(out.interleaved_qpoint_geometry, "KernelArgsV6: interleaved_qpoint_geometry not aligned");

        assertAligned(out.test_basis_values, "KernelArgsV6: test_basis_values not aligned");
        assertAligned(out.test_phys_gradients_xyz, "KernelArgsV6: test_phys_gradients_xyz not aligned");
        assertAligned(out.test_phys_hessians, "KernelArgsV6: test_phys_hessians not aligned");

        assertAligned(out.trial_basis_values, "KernelArgsV6: trial_basis_values not aligned");
        assertAligned(out.trial_phys_gradients_xyz, "KernelArgsV6: trial_phys_gradients_xyz not aligned");
        assertAligned(out.trial_phys_hessians, "KernelArgsV6: trial_phys_hessians not aligned");

        assertAligned(out.test_basis_vector_values_xyz, "KernelArgsV6: test_basis_vector_values_xyz not aligned");
        assertAligned(out.test_basis_curls_xyz, "KernelArgsV6: test_basis_curls_xyz not aligned");
        assertAligned(out.test_basis_divergences, "KernelArgsV6: test_basis_divergences not aligned");

        assertAligned(out.trial_basis_vector_values_xyz, "KernelArgsV6: trial_basis_vector_values_xyz not aligned");
        assertAligned(out.trial_basis_curls_xyz, "KernelArgsV6: trial_basis_curls_xyz not aligned");
        assertAligned(out.trial_basis_divergences, "KernelArgsV6: trial_basis_divergences not aligned");

        assertAligned(out.solution_coefficients, "KernelArgsV6: solution_coefficients not aligned");
        for (const auto* p : out.previous_solution_coefficients) {
            assertAligned(p, "KernelArgsV6: previous_solution_coefficients not aligned");
        }

        assertAligned(out.history_weights, "KernelArgsV6: history_weights not aligned");
        for (const auto* p : out.history_solution_coefficients) {
            assertAligned(p, "KernelArgsV6: history_solution_coefficients not aligned");
        }

        assertAligned(out.jit_constants, "KernelArgsV6: jit_constants not aligned");
        assertAligned(out.coupled_integrals, "KernelArgsV6: coupled_integrals not aligned");
        assertAligned(out.coupled_aux, "KernelArgsV6: coupled_aux not aligned");

        if (!field_table.empty()) {
            FE_ASSERT_MSG(isAligned(out.field_solutions, alignof(FieldSolutionEntryV1)),
                          "KernelArgsV6: field_solutions table pointer not aligned");
            for (const auto& e : field_table) {
                assertAligned(e.values, "KernelArgsV6: field_solutions.values not aligned");
                assertAligned(e.gradients_xyz, "KernelArgsV6: field_solutions.gradients_xyz not aligned");
                assertAligned(e.hessians, "KernelArgsV6: field_solutions.hessians not aligned");
                assertAligned(e.laplacians, "KernelArgsV6: field_solutions.laplacians not aligned");

                assertAligned(e.vector_values_xyz, "KernelArgsV6: field_solutions.vector_values_xyz not aligned");
                assertAligned(e.jacobians, "KernelArgsV6: field_solutions.jacobians not aligned");
                assertAligned(e.component_hessians, "KernelArgsV6: field_solutions.component_hessians not aligned");
                assertAligned(e.component_laplacians, "KernelArgsV6: field_solutions.component_laplacians not aligned");

                assertAligned(e.history_values, "KernelArgsV6: field_solutions.history_values not aligned");
                assertAligned(e.history_vector_values_xyz, "KernelArgsV6: field_solutions.history_vector_values_xyz not aligned");
            }
        }

        const auto state_align = static_cast<std::size_t>(out.material_state_alignment_bytes);
        if (out.material_state_work_base != nullptr) {
            FE_ASSERT_MSG(isPowerOfTwo(state_align), "KernelArgsV6: material_state_alignment_bytes must be power-of-two");
            FE_ASSERT_MSG(isAligned(out.material_state_old_base, state_align), "KernelArgsV6: material_state_old_base not aligned");
            FE_ASSERT_MSG(isAligned(out.material_state_work_base, state_align), "KernelArgsV6: material_state_work_base not aligned");
            FE_ASSERT_MSG(out.material_state_stride_bytes % state_align == 0u, "KernelArgsV6: material_state_stride_bytes not aligned");
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

[[nodiscard]] inline CellKernelArgsV4 packCellKernelArgsV4(const AssemblyContext& ctx,
                                                           KernelOutput& output,
                                                           PackingChecks checks = {}) noexcept
{
    CellKernelArgsV4 out;
    out.side = detail::packSideArgsV4(ctx, std::nullopt, std::nullopt, checks);
    out.output = detail::packOutputViewV4(output);
    return out;
}

[[nodiscard]] inline BoundaryFaceKernelArgsV4 packBoundaryFaceKernelArgsV4(const AssemblyContext& ctx,
                                                                           int boundary_marker,
                                                                           KernelOutput& output,
                                                                           PackingChecks checks = {}) noexcept
{
    BoundaryFaceKernelArgsV4 out;
    out.side = detail::packSideArgsV4(ctx, boundary_marker, std::nullopt, checks);
    out.output = detail::packOutputViewV4(output);
    return out;
}

[[nodiscard]] inline InteriorFaceKernelArgsV4 packInteriorFaceKernelArgsV4(const AssemblyContext& ctx_minus,
                                                                           const AssemblyContext& ctx_plus,
                                                                           KernelOutput& output_minus,
                                                                           KernelOutput& output_plus,
                                                                           KernelOutput& coupling_minus_plus,
                                                                           KernelOutput& coupling_plus_minus,
                                                                           PackingChecks checks = {}) noexcept
{
    InteriorFaceKernelArgsV4 out;
    out.minus = detail::packSideArgsV4(ctx_minus, std::nullopt, std::nullopt, checks);
    out.plus = detail::packSideArgsV4(ctx_plus, std::nullopt, std::nullopt, checks);

    out.output_minus = detail::packOutputViewV4(output_minus);
    out.output_plus = detail::packOutputViewV4(output_plus);
    out.coupling_minus_plus = detail::packOutputViewV4(coupling_minus_plus);
    out.coupling_plus_minus = detail::packOutputViewV4(coupling_plus_minus);
    return out;
}

[[nodiscard]] inline InteriorFaceKernelArgsV4 packInterfaceFaceKernelArgsV4(const AssemblyContext& ctx_minus,
                                                                            const AssemblyContext& ctx_plus,
                                                                            int interface_marker,
                                                                            KernelOutput& output_minus,
                                                                            KernelOutput& output_plus,
                                                                            KernelOutput& coupling_minus_plus,
                                                                            KernelOutput& coupling_plus_minus,
                                                                            PackingChecks checks = {}) noexcept
{
    InteriorFaceKernelArgsV4 out;
    out.minus = detail::packSideArgsV4(ctx_minus, std::nullopt, interface_marker, checks);
    out.plus = detail::packSideArgsV4(ctx_plus, std::nullopt, interface_marker, checks);

    out.output_minus = detail::packOutputViewV4(output_minus);
    out.output_plus = detail::packOutputViewV4(output_plus);
    out.coupling_minus_plus = detail::packOutputViewV4(coupling_minus_plus);
    out.coupling_plus_minus = detail::packOutputViewV4(coupling_plus_minus);
    return out;
}

[[nodiscard]] inline CellKernelArgsV5 packCellKernelArgsV5(const AssemblyContext& ctx,
                                                           KernelOutput& output,
                                                           PackingChecks checks = {}) noexcept
{
    CellKernelArgsV5 out;
    out.side = detail::packSideArgsV5(ctx, std::nullopt, std::nullopt, checks);
    out.output = detail::packOutputViewV5(output);
    return out;
}

[[nodiscard]] inline BoundaryFaceKernelArgsV5 packBoundaryFaceKernelArgsV5(const AssemblyContext& ctx,
                                                                           int boundary_marker,
                                                                           KernelOutput& output,
                                                                           PackingChecks checks = {}) noexcept
{
    BoundaryFaceKernelArgsV5 out;
    out.side = detail::packSideArgsV5(ctx, boundary_marker, std::nullopt, checks);
    out.output = detail::packOutputViewV5(output);
    return out;
}

[[nodiscard]] inline InteriorFaceKernelArgsV5 packInteriorFaceKernelArgsV5(const AssemblyContext& ctx_minus,
                                                                           const AssemblyContext& ctx_plus,
                                                                           KernelOutput& output_minus,
                                                                           KernelOutput& output_plus,
                                                                           KernelOutput& coupling_minus_plus,
                                                                           KernelOutput& coupling_plus_minus,
                                                                           PackingChecks checks = {}) noexcept
{
    InteriorFaceKernelArgsV5 out;
    out.minus = detail::packSideArgsV5(ctx_minus, std::nullopt, std::nullopt, checks);
    out.plus = detail::packSideArgsV5(ctx_plus, std::nullopt, std::nullopt, checks);

    out.output_minus = detail::packOutputViewV5(output_minus);
    out.output_plus = detail::packOutputViewV5(output_plus);
    out.coupling_minus_plus = detail::packOutputViewV5(coupling_minus_plus);
    out.coupling_plus_minus = detail::packOutputViewV5(coupling_plus_minus);
    return out;
}

[[nodiscard]] inline InteriorFaceKernelArgsV5 packInterfaceFaceKernelArgsV5(const AssemblyContext& ctx_minus,
                                                                            const AssemblyContext& ctx_plus,
                                                                            int interface_marker,
                                                                            KernelOutput& output_minus,
                                                                            KernelOutput& output_plus,
                                                                            KernelOutput& coupling_minus_plus,
                                                                            KernelOutput& coupling_plus_minus,
                                                                            PackingChecks checks = {}) noexcept
{
    InteriorFaceKernelArgsV5 out;
    out.minus = detail::packSideArgsV5(ctx_minus, std::nullopt, interface_marker, checks);
    out.plus = detail::packSideArgsV5(ctx_plus, std::nullopt, interface_marker, checks);

    out.output_minus = detail::packOutputViewV5(output_minus);
    out.output_plus = detail::packOutputViewV5(output_plus);
    out.coupling_minus_plus = detail::packOutputViewV5(coupling_minus_plus);
    out.coupling_plus_minus = detail::packOutputViewV5(coupling_plus_minus);
    return out;
}

[[nodiscard]] inline CellKernelArgsV6 packCellKernelArgsV6(const AssemblyContext& ctx,
                                                           KernelOutput& output,
                                                           PackingChecks checks = {}) noexcept
{
    CellKernelArgsV6 out;
    out.side = detail::packSideArgsV6(ctx, std::nullopt, std::nullopt, checks);
    out.output = detail::packOutputViewV6(output);
    return out;
}

[[nodiscard]] inline BoundaryFaceKernelArgsV6 packBoundaryFaceKernelArgsV6(const AssemblyContext& ctx,
                                                                           int boundary_marker,
                                                                           KernelOutput& output,
                                                                           PackingChecks checks = {}) noexcept
{
    BoundaryFaceKernelArgsV6 out;
    out.side = detail::packSideArgsV6(ctx, boundary_marker, std::nullopt, checks);
    out.output = detail::packOutputViewV6(output);
    return out;
}

[[nodiscard]] inline InteriorFaceKernelArgsV6 packInteriorFaceKernelArgsV6(const AssemblyContext& ctx_minus,
                                                                           const AssemblyContext& ctx_plus,
                                                                           KernelOutput& output_minus,
                                                                           KernelOutput& output_plus,
                                                                           KernelOutput& coupling_minus_plus,
                                                                           KernelOutput& coupling_plus_minus,
                                                                           PackingChecks checks = {}) noexcept
{
    InteriorFaceKernelArgsV6 out;
    out.minus = detail::packSideArgsV6(ctx_minus, std::nullopt, std::nullopt, checks);
    out.plus = detail::packSideArgsV6(ctx_plus, std::nullopt, std::nullopt, checks);

    out.output_minus = detail::packOutputViewV6(output_minus);
    out.output_plus = detail::packOutputViewV6(output_plus);
    out.coupling_minus_plus = detail::packOutputViewV6(coupling_minus_plus);
    out.coupling_plus_minus = detail::packOutputViewV6(coupling_plus_minus);
    return out;
}

[[nodiscard]] inline InteriorFaceKernelArgsV6 packInterfaceFaceKernelArgsV6(const AssemblyContext& ctx_minus,
                                                                            const AssemblyContext& ctx_plus,
                                                                            int interface_marker,
                                                                            KernelOutput& output_minus,
                                                                            KernelOutput& output_plus,
                                                                            KernelOutput& coupling_minus_plus,
                                                                            KernelOutput& coupling_plus_minus,
                                                                            PackingChecks checks = {}) noexcept
{
    InteriorFaceKernelArgsV6 out;
    out.minus = detail::packSideArgsV6(ctx_minus, std::nullopt, interface_marker, checks);
    out.plus = detail::packSideArgsV6(ctx_plus, std::nullopt, interface_marker, checks);

    out.output_minus = detail::packOutputViewV6(output_minus);
    out.output_plus = detail::packOutputViewV6(output_plus);
    out.coupling_minus_plus = detail::packOutputViewV6(coupling_minus_plus);
    out.coupling_plus_minus = detail::packOutputViewV6(coupling_plus_minus);
    return out;
}

} // namespace jit
} // namespace assembly
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_ASSEMBLY_JIT_KERNEL_ARGS_H

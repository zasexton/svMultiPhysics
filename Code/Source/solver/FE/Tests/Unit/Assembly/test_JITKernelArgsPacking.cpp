/**
 * @file test_JITKernelArgsPacking.cpp
 * @brief Unit tests for Assembly/JIT KernelArgs ABI packing
 */

#include <gtest/gtest.h>

#include "Assembly/AssemblyContext.h"
#include "Assembly/JIT/KernelArgs.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace svmp {
namespace FE {
namespace assembly {
namespace test {

namespace {

static void setupTwoDofTwoQptScalarContext(AssemblyContext& ctx)
{
    std::vector<AssemblyContext::Point3D> quad_pts = {{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}};
    std::vector<Real> weights = {0.5, 0.5};
    ctx.setQuadratureData(std::span<const AssemblyContext::Point3D>(quad_pts.data(), quad_pts.size()),
                          std::span<const Real>(weights.data(), weights.size()));

    const LocalIndex n_dofs = 2;

    std::vector<Real> values = {
        1.0, 0.5,   // phi_0 at q=0,1
        0.0, 0.5    // phi_1 at q=0,1
    };

    std::vector<AssemblyContext::Vector3D> grads = {
        {-1.0, 0.0, 0.0}, {-1.0, 0.0, 0.0},
        { 1.0, 0.0, 0.0}, { 1.0, 0.0, 0.0}
    };

    ctx.setTestBasisData(n_dofs,
                         std::span<const Real>(values.data(), values.size()),
                         std::span<const AssemblyContext::Vector3D>(grads.data(), grads.size()));
    ctx.setPhysicalGradients(std::span<const AssemblyContext::Vector3D>(grads.data(), grads.size()),
                             std::span<const AssemblyContext::Vector3D>(grads.data(), grads.size()));

    std::vector<Real> int_wts = {0.25, 0.25};
    ctx.setIntegrationWeights(std::span<const Real>(int_wts.data(), int_wts.size()));
}

} // namespace

TEST(JITKernelArgsPacking, PacksCellKernelArgsPointersAndCounts)
{
    AssemblyContext ctx;
    ctx.reserve(/*max_dofs=*/8, /*max_qpts=*/8, /*dim=*/3);
    setupTwoDofTwoQptScalarContext(ctx);

    ctx.setCellDomainId(42);
    ctx.setTime(Real(1.25));
    ctx.setTimeStep(Real(0.1));

    std::vector<Real, AlignedAllocator<Real, kFEPreferredAlignmentBytes>> jit_constants = {Real(3.0), Real(4.0)};
    ctx.setJITConstants(std::span<const Real>(jit_constants.data(), jit_constants.size()));

    std::vector<Real, AlignedAllocator<Real, kFEPreferredAlignmentBytes>> coupled_integrals = {Real(7.0)};
    std::vector<Real, AlignedAllocator<Real, kFEPreferredAlignmentBytes>> coupled_aux = {Real(8.0), Real(9.0)};
    ctx.setCoupledValues(std::span<const Real>(coupled_integrals.data(), coupled_integrals.size()),
                         std::span<const Real>(coupled_aux.data(), coupled_aux.size()));

    std::vector<Real> prev_coeffs = {Real(10.0), Real(11.0)};
    ctx.setPreviousSolutionCoefficientsK(/*k=*/1, prev_coeffs);

    KernelOutput out;
    out.reserve(ctx.numTestDofs(), ctx.numTrialDofs(), /*need_matrix=*/true, /*need_vector=*/true);

    auto args = jit::packCellKernelArgsV1(ctx, out);

    EXPECT_EQ(args.abi_version, jit::kKernelArgsABIVersionV1);

    EXPECT_EQ(args.side.n_qpts, 2u);
    EXPECT_EQ(args.side.n_test_dofs, 2u);
    EXPECT_EQ(args.side.n_trial_dofs, 2u);

    EXPECT_EQ(args.side.quad_weights, ctx.quadratureWeights().data());
    EXPECT_EQ(args.side.integration_weights, ctx.integrationWeights().data());
    EXPECT_EQ(args.side.quad_points_xyz, ctx.quadraturePoints().data()->data());

    EXPECT_EQ(args.side.test_basis_values, ctx.testBasisValuesRaw().data());
    EXPECT_EQ(args.side.trial_basis_values, ctx.trialBasisValuesRaw().data());

    EXPECT_EQ(args.side.jit_constants, jit_constants.data());
    EXPECT_EQ(args.side.num_jit_constants, jit_constants.size());

    EXPECT_EQ(args.side.coupled_integrals, coupled_integrals.data());
    EXPECT_EQ(args.side.num_coupled_integrals, coupled_integrals.size());
    EXPECT_EQ(args.side.coupled_aux, coupled_aux.data());
    EXPECT_EQ(args.side.num_coupled_aux, coupled_aux.size());

    EXPECT_EQ(args.side.num_previous_solutions, 1u);
    EXPECT_EQ(args.side.previous_solution_coefficients[0], ctx.previousSolutionCoefficientsRaw(/*k=*/1).data());

    EXPECT_EQ(args.output.element_matrix, out.local_matrix.data());
    EXPECT_EQ(args.output.element_vector, out.local_vector.data());

    auto args2 = jit::packCellKernelArgsV2(ctx, out);
    EXPECT_EQ(args2.abi_version, jit::kKernelArgsABIVersionV2);

    EXPECT_EQ(args2.side.n_qpts, 2u);
    EXPECT_EQ(args2.side.n_test_dofs, 2u);
    EXPECT_EQ(args2.side.n_trial_dofs, 2u);

    EXPECT_EQ(args2.side.cell_domain_id, 42);
    EXPECT_EQ(args2.side.interface_marker, -1);
    EXPECT_EQ(args2.side.test_uses_vector_basis, 0u);
    EXPECT_EQ(args2.side.trial_uses_vector_basis, 0u);

    EXPECT_EQ(args2.side.quad_weights, ctx.quadratureWeights().data());
    EXPECT_EQ(args2.side.integration_weights, ctx.integrationWeights().data());
    EXPECT_EQ(args2.side.quad_points_xyz, ctx.quadraturePoints().data()->data());

    EXPECT_EQ(args2.side.test_basis_values, ctx.testBasisValuesRaw().data());
    EXPECT_EQ(args2.side.trial_basis_values, ctx.trialBasisValuesRaw().data());
    EXPECT_EQ(args2.side.test_basis_vector_values_xyz, nullptr);
    EXPECT_EQ(args2.side.trial_basis_vector_values_xyz, nullptr);

    EXPECT_EQ(args2.side.jit_constants, jit_constants.data());
    EXPECT_EQ(args2.side.num_jit_constants, jit_constants.size());

    EXPECT_EQ(args2.side.coupled_integrals, coupled_integrals.data());
    EXPECT_EQ(args2.side.num_coupled_integrals, coupled_integrals.size());
    EXPECT_EQ(args2.side.coupled_aux, coupled_aux.data());
    EXPECT_EQ(args2.side.num_coupled_aux, coupled_aux.size());

    EXPECT_EQ(args2.side.num_previous_solutions, 1u);
    EXPECT_EQ(args2.side.previous_solution_coefficients[0], ctx.previousSolutionCoefficientsRaw(/*k=*/1).data());

    EXPECT_EQ(args2.output.element_matrix, out.local_matrix.data());
    EXPECT_EQ(args2.output.element_vector, out.local_vector.data());
}

TEST(JITKernelArgsPacking, PacksMultiFieldSolutionTableV3AndChecksAlignment)
{
    AssemblyContext ctx;
    ctx.reserve(/*max_dofs=*/8, /*max_qpts=*/8, /*dim=*/3);
    setupTwoDofTwoQptScalarContext(ctx);

    std::vector<Real, AlignedAllocator<Real, kFEPreferredAlignmentBytes>> jit_constants = {Real(3.0), Real(4.0)};
    ctx.setJITConstants(std::span<const Real>(jit_constants.data(), jit_constants.size()));

    // Bind one additional scalar field at quadrature points with one history state.
    const FieldId field = static_cast<FieldId>(123);
    std::vector<Real> field_values = {Real(2.0), Real(3.0)};
    std::vector<AssemblyContext::Vector3D> field_gradients = {
        AssemblyContext::Vector3D{Real(1.0), Real(0.0), Real(0.0)},
        AssemblyContext::Vector3D{Real(2.0), Real(0.0), Real(0.0)}};
    ctx.setFieldSolutionScalar(field,
                               std::span<const Real>(field_values.data(), field_values.size()),
                               std::span<const AssemblyContext::Vector3D>(field_gradients.data(), field_gradients.size()));

    std::vector<Real> field_prev_values = {Real(9.0), Real(10.0)};
    ctx.setFieldPreviousSolutionScalarK(field, /*k=*/1,
                                        std::span<const Real>(field_prev_values.data(), field_prev_values.size()));

    KernelOutput out;
    out.reserve(ctx.numTestDofs(), ctx.numTrialDofs(), /*need_matrix=*/true, /*need_vector=*/true);

    jit::PackingChecks checks;
    checks.validate_alignment = true;
    const auto args3 = jit::packCellKernelArgsV3(ctx, out, checks);

    EXPECT_EQ(args3.abi_version, jit::kKernelArgsABIVersionV3);
    EXPECT_EQ(args3.side.pointer_alignment_bytes, static_cast<std::uint32_t>(kFEPreferredAlignmentBytes));

    EXPECT_NE(args3.side.field_solutions, nullptr);
    EXPECT_EQ(args3.side.num_field_solutions, 1u);

    const auto& e0 = args3.side.field_solutions[0];
    EXPECT_EQ(e0.field_id, static_cast<std::int32_t>(field));
    EXPECT_EQ(e0.field_type, static_cast<std::uint32_t>(FieldType::Scalar));
    EXPECT_EQ(e0.value_dim, 1u);

    EXPECT_NE(e0.values, nullptr);
    EXPECT_NE(e0.gradients_xyz, nullptr);
    EXPECT_EQ(e0.history_count, 1u);
    EXPECT_NE(e0.history_values, nullptr);

    const auto align = static_cast<std::uintptr_t>(args3.side.pointer_alignment_bytes);
    EXPECT_NE(align, 0u);
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(args3.side.quad_weights) % align, 0u);
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(args3.side.integration_weights) % align, 0u);
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(args3.side.test_basis_values) % align, 0u);
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(args3.side.trial_basis_values) % align, 0u);
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(args3.side.jit_constants) % align, 0u);
}

TEST(JITKernelArgsPacking, PacksTimeIntegrationStencilsV5)
{
    AssemblyContext ctx;
    ctx.reserve(/*max_dofs=*/8, /*max_qpts=*/8, /*dim=*/3);
    setupTwoDofTwoQptScalarContext(ctx);

    // Provide some history solution slots so packers can bind previous_solution_coefficients pointers.
    std::vector<Real> prev1 = {Real(10.0), Real(11.0)};
    std::vector<Real> prev2 = {Real(12.0), Real(13.0)};
    std::vector<Real> prev3 = {Real(14.0), Real(15.0)};
    ctx.setPreviousSolutionCoefficientsK(/*k=*/1, prev1);
    ctx.setPreviousSolutionCoefficientsK(/*k=*/2, prev2);
    ctx.setPreviousSolutionCoefficientsK(/*k=*/3, prev3);

    // Build a time-integration context with dt1, dt2, and dt3 stencils.
    TimeIntegrationContext ti;
    ti.integrator_name = "TestTI";
    ti.time_derivative_term_weight = Real(0.5);
    ti.non_time_derivative_term_weight = Real(2.0);
    ti.dt1_term_weight = Real(3.0);
    ti.dt2_term_weight = Real(4.0);
    ti.dt_extra_term_weight = {Real(5.0)}; // dt3

    ti.dt1 = TimeDerivativeStencil{.order = 1, .a = {Real(2.0), Real(-2.0)}};
    ti.dt2 = TimeDerivativeStencil{.order = 2, .a = {Real(4.0), Real(-8.0), Real(4.0)}};
    ti.dt_extra.resize(1);
    ti.dt_extra[0] = TimeDerivativeStencil{.order = 3, .a = {Real(8.0), Real(-24.0), Real(24.0), Real(-8.0)}};

    ctx.setTimeIntegrationContext(&ti);

    std::vector<Real, AlignedAllocator<Real, kFEPreferredAlignmentBytes>> hist_w = {Real(0.25), Real(0.125)};
    ctx.setHistoryWeights(std::span<const Real>(hist_w.data(), hist_w.size()));

    KernelOutput out;
    out.reserve(ctx.numTestDofs(), ctx.numTrialDofs(), /*need_matrix=*/true, /*need_vector=*/true);

    jit::PackingChecks checks;
    checks.validate_alignment = true;
    const auto args5 = jit::packCellKernelArgsV5(ctx, out, checks);

    EXPECT_EQ(args5.abi_version, jit::kKernelArgsABIVersionV5);
    EXPECT_EQ(args5.side.max_time_derivative_order, 3u);

    EXPECT_DOUBLE_EQ(args5.side.time_derivative_term_weight, 0.5);
    EXPECT_DOUBLE_EQ(args5.side.non_time_derivative_term_weight, 2.0);

    EXPECT_DOUBLE_EQ(args5.side.dt_term_weights[0], 3.0);
    EXPECT_DOUBLE_EQ(args5.side.dt_term_weights[1], 4.0);
    EXPECT_DOUBLE_EQ(args5.side.dt_term_weights[2], 5.0);

    EXPECT_DOUBLE_EQ(args5.side.dt_stencil_coeffs[0][0], 2.0);
    EXPECT_DOUBLE_EQ(args5.side.dt_stencil_coeffs[0][1], -2.0);

    EXPECT_DOUBLE_EQ(args5.side.dt_stencil_coeffs[1][0], 4.0);
    EXPECT_DOUBLE_EQ(args5.side.dt_stencil_coeffs[1][1], -8.0);
    EXPECT_DOUBLE_EQ(args5.side.dt_stencil_coeffs[1][2], 4.0);

    EXPECT_DOUBLE_EQ(args5.side.dt_stencil_coeffs[2][0], 8.0);
    EXPECT_DOUBLE_EQ(args5.side.dt_stencil_coeffs[2][1], -24.0);
    EXPECT_DOUBLE_EQ(args5.side.dt_stencil_coeffs[2][2], 24.0);
    EXPECT_DOUBLE_EQ(args5.side.dt_stencil_coeffs[2][3], -8.0);

    EXPECT_EQ(args5.side.num_history_steps, 2u);
    EXPECT_EQ(args5.side.history_weights, hist_w.data());
    EXPECT_EQ(args5.side.history_solution_coefficients[0], ctx.previousSolutionCoefficientsRaw(/*k=*/1).data());
    EXPECT_EQ(args5.side.history_solution_coefficients[1], ctx.previousSolutionCoefficientsRaw(/*k=*/2).data());
    EXPECT_EQ(args5.side.history_solution_coefficients[2], ctx.previousSolutionCoefficientsRaw(/*k=*/3).data());
}

TEST(JITKernelArgsPacking, PacksMaterialStateAlignmentV3)
{
    AssemblyContext ctx;
    ctx.reserve(/*max_dofs=*/8, /*max_qpts=*/8, /*dim=*/3);
    setupTwoDofTwoQptScalarContext(ctx);

    constexpr std::size_t bytes_per_qpt = 32u;
    constexpr std::size_t stride_bytes = kFEPreferredAlignmentBytes;
    static_assert(stride_bytes >= bytes_per_qpt);

    std::vector<std::byte, AlignedAllocator<std::byte, kFEPreferredAlignmentBytes>> state(
        stride_bytes * static_cast<std::size_t>(ctx.numQuadraturePoints()));

    ctx.setMaterialState(state.data(), state.data(), bytes_per_qpt, stride_bytes, kFEPreferredAlignmentBytes);

    KernelOutput out;
    out.reserve(ctx.numTestDofs(), ctx.numTrialDofs(), /*need_matrix=*/false, /*need_vector=*/false);

    jit::PackingChecks checks;
    checks.validate_alignment = true;
    const auto args3 = jit::packCellKernelArgsV3(ctx, out, checks);

    EXPECT_EQ(args3.side.material_state_old_base, state.data());
    EXPECT_EQ(args3.side.material_state_work_base, state.data());
    EXPECT_EQ(args3.side.material_state_bytes_per_qpt, bytes_per_qpt);
    EXPECT_EQ(args3.side.material_state_stride_bytes, stride_bytes);
    EXPECT_EQ(args3.side.material_state_alignment_bytes, kFEPreferredAlignmentBytes);
}

TEST(JITKernelArgsPacking, PacksInterfaceMarkerOverrideV2)
{
    AssemblyContext ctx_minus;
    ctx_minus.reserve(/*max_dofs=*/8, /*max_qpts=*/8, /*dim=*/3);
    setupTwoDofTwoQptScalarContext(ctx_minus);

    AssemblyContext ctx_plus;
    ctx_plus.reserve(/*max_dofs=*/8, /*max_qpts=*/8, /*dim=*/3);
    setupTwoDofTwoQptScalarContext(ctx_plus);

    KernelOutput out_minus, out_plus, coupling_mp, coupling_pm;
    out_minus.reserve(ctx_minus.numTestDofs(), ctx_minus.numTrialDofs(), /*need_matrix=*/true, /*need_vector=*/false);
    out_plus.reserve(ctx_plus.numTestDofs(), ctx_plus.numTrialDofs(), /*need_matrix=*/true, /*need_vector=*/false);
    coupling_mp.reserve(ctx_minus.numTestDofs(), ctx_plus.numTrialDofs(), /*need_matrix=*/true, /*need_vector=*/false);
    coupling_pm.reserve(ctx_plus.numTestDofs(), ctx_minus.numTrialDofs(), /*need_matrix=*/true, /*need_vector=*/false);

    auto args = jit::packInterfaceFaceKernelArgsV2(ctx_minus, ctx_plus, /*interface_marker=*/7,
                                                   out_minus, out_plus, coupling_mp, coupling_pm);
    EXPECT_EQ(args.abi_version, jit::kKernelArgsABIVersionV2);
    EXPECT_EQ(args.minus.interface_marker, 7);
    EXPECT_EQ(args.plus.interface_marker, 7);
}

} // namespace test
} // namespace assembly
} // namespace FE
} // namespace svmp

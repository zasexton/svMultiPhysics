/**
 * @file test_JITKernelArgsPacking.cpp
 * @brief Unit tests for Assembly/JIT KernelArgs ABI packing
 */

#include <gtest/gtest.h>

#include "Assembly/AssemblyContext.h"
#include "Assembly/JIT/KernelArgs.h"

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
    ctx.setQuadratureData(quad_pts, weights);

    const LocalIndex n_dofs = 2;

    std::vector<Real> values = {
        1.0, 0.5,   // phi_0 at q=0,1
        0.0, 0.5    // phi_1 at q=0,1
    };

    std::vector<AssemblyContext::Vector3D> grads = {
        {-1.0, 0.0, 0.0}, {-1.0, 0.0, 0.0},
        { 1.0, 0.0, 0.0}, { 1.0, 0.0, 0.0}
    };

    ctx.setTestBasisData(n_dofs, values, grads);
    ctx.setPhysicalGradients(grads, grads);

    std::vector<Real> int_wts = {0.25, 0.25};
    ctx.setIntegrationWeights(int_wts);
}

} // namespace

TEST(JITKernelArgsPacking, PacksCellKernelArgsPointersAndCounts)
{
    AssemblyContext ctx;
    ctx.reserve(/*max_dofs=*/8, /*max_qpts=*/8, /*dim=*/3);
    setupTwoDofTwoQptScalarContext(ctx);

    ctx.setTime(Real(1.25));
    ctx.setTimeStep(Real(0.1));

    std::vector<Real> jit_constants = {Real(3.0), Real(4.0)};
    ctx.setJITConstants(jit_constants);

    std::vector<Real> coupled_integrals = {Real(7.0)};
    std::vector<Real> coupled_aux = {Real(8.0), Real(9.0)};
    ctx.setCoupledValues(coupled_integrals, coupled_aux);

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
}

} // namespace test
} // namespace assembly
} // namespace FE
} // namespace svmp

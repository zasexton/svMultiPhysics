/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

/**
 * @file test_NonlinearAssemblyDriver_EndToEnd.cpp
 * @brief End-to-end tests for NonlinearAssemblyDriver with real spaces + dof map.
 */

#include <gtest/gtest.h>

#include "Assembly/GlobalSystemView.h"
#include "Assembly/NonlinearAssemblyDriver.h"
#include "Spaces/H1Space.h"
#include "Tests/Unit/Forms/FormsTestHelpers.h"

#include <vector>

namespace svmp {
namespace FE {
namespace assembly {
namespace testing {

namespace {

class PoissonKernel final : public INonlinearKernel {
public:
    void computeResidual(AssemblyContext& context, KernelOutput& output) override
    {
        const auto coeffs = context.solutionCoefficients();
        const auto n_test = context.numTestDofs();
        const auto n_trial = context.numTrialDofs();
        FE_THROW_IF(coeffs.size() < static_cast<std::size_t>(n_trial),
                    "PoissonKernel: missing solution coefficients");

        for (LocalIndex q = 0; q < context.numQuadraturePoints(); ++q) {
            const Real w = context.integrationWeight(q);

            for (LocalIndex i = 0; i < n_test; ++i) {
                const auto gi = context.physicalGradient(i, q);

                Real acc = 0.0;
                for (LocalIndex j = 0; j < n_trial; ++j) {
                    const auto gj = context.trialPhysicalGradient(j, q);
                    const Real kij = (gi[0] * gj[0] + gi[1] * gj[1] + gi[2] * gj[2]) * w;
                    acc += kij * coeffs[static_cast<std::size_t>(j)];
                }
                output.vectorEntry(i) += acc;
            }
        }
    }

    void computeJacobian(AssemblyContext& context, KernelOutput& output) override
    {
        const auto n_test = context.numTestDofs();
        const auto n_trial = context.numTrialDofs();

        for (LocalIndex q = 0; q < context.numQuadraturePoints(); ++q) {
            const Real w = context.integrationWeight(q);
            for (LocalIndex i = 0; i < n_test; ++i) {
                const auto gi = context.physicalGradient(i, q);
                for (LocalIndex j = 0; j < n_trial; ++j) {
                    const auto gj = context.trialPhysicalGradient(j, q);
                    output.matrixEntry(i, j) += (gi[0] * gj[0] + gi[1] * gj[1] + gi[2] * gj[2]) * w;
                }
            }
        }
    }

    void computeBoth(AssemblyContext& context, KernelOutput& output) override
    {
        const auto coeffs = context.solutionCoefficients();
        const auto n_test = context.numTestDofs();
        const auto n_trial = context.numTrialDofs();
        FE_THROW_IF(coeffs.size() < static_cast<std::size_t>(n_trial),
                    "PoissonKernel: missing solution coefficients");

        for (LocalIndex q = 0; q < context.numQuadraturePoints(); ++q) {
            const Real w = context.integrationWeight(q);
            for (LocalIndex i = 0; i < n_test; ++i) {
                const auto gi = context.physicalGradient(i, q);

                for (LocalIndex j = 0; j < n_trial; ++j) {
                    const auto gj = context.trialPhysicalGradient(j, q);
                    const Real kij = (gi[0] * gj[0] + gi[1] * gj[1] + gi[2] * gj[2]) * w;
                    output.matrixEntry(i, j) += kij;
                    output.vectorEntry(i) += kij * coeffs[static_cast<std::size_t>(j)];
                }
            }
        }
    }

    [[nodiscard]] bool hasOptimizedBoth() const noexcept override { return true; }

    [[nodiscard]] RequiredData getResidualRequiredData() const noexcept override
    {
        return RequiredData::IntegrationWeights | RequiredData::PhysicalGradients | RequiredData::SolutionCoefficients;
    }

    [[nodiscard]] RequiredData getJacobianRequiredData() const noexcept override { return getResidualRequiredData(); }
};

void expectResidualMatchesMatrixTimesU(const GlobalSystemView& J,
                                      const GlobalSystemView& R,
                                      const std::vector<Real>& U,
                                      Real tol)
{
    const GlobalIndex n = J.numRows();
    ASSERT_EQ(n, R.numRows());
    ASSERT_EQ(static_cast<std::size_t>(n), U.size());

    for (GlobalIndex i = 0; i < n; ++i) {
        Real acc = 0.0;
        for (GlobalIndex j = 0; j < n; ++j) {
            acc += J.getMatrixEntry(i, j) * U[static_cast<std::size_t>(j)];
        }
        EXPECT_NEAR(R.getVectorEntry(i), acc, tol);
    }
}

} // namespace

TEST(NonlinearAssemblyDriverEndToEndTest, AssemblesMultiElementPoissonAndResidualMatchesMatrixTimesU)
{
    forms::test::TwoTetraSharedFaceMeshAccess mesh;

    // Shared-vertex DOF numbering (P1): cell0 {0,1,2,3}, cell1 {1,2,3,4}.
    dofs::DofMap dof_map(2, 5, 4);
    dof_map.setCellDofs(0, std::vector<GlobalIndex>{0, 1, 2, 3});
    dof_map.setCellDofs(1, std::vector<GlobalIndex>{1, 2, 3, 4});
    dof_map.setNumDofs(5);
    dof_map.setNumLocalDofs(5);
    dof_map.finalize();

    spaces::H1Space space(ElementType::Tetra4, /*order=*/1);
    PoissonKernel kernel;

    NonlinearAssemblyDriver driver;
    driver.setMesh(mesh);
    driver.setDofMap(dof_map);
    driver.setSpace(space);
    driver.setKernel(kernel);

    std::vector<Real> U = {0.12, -0.05, 0.08, -0.02, 0.07};
    driver.setCurrentSolution(U);

    DenseMatrixView J(/*n_dofs=*/5);
    DenseVectorView R(/*n_dofs=*/5);
    J.zero();
    R.zero();

    (void)driver.assembleBoth(J, R);

    expectResidualMatchesMatrixTimesU(J, R, U, /*tol=*/1e-12);
}

TEST(NonlinearAssemblyDriverEndToEndTest, VerifyJacobianFDPassesForLinearPoisson)
{
    forms::test::SingleTetraMeshAccess mesh;
    auto dof_map = forms::test::createSingleTetraDofMap();
    spaces::H1Space space(ElementType::Tetra4, /*order=*/1);
    PoissonKernel kernel;

    NonlinearAssemblyDriver driver;
    driver.setMesh(mesh);
    driver.setDofMap(dof_map);
    driver.setSpace(space);
    driver.setKernel(kernel);

    std::vector<Real> U = {0.12, -0.05, 0.08, -0.02};
    driver.setCurrentSolution(U);

    DenseMatrixView J(/*n_dofs=*/4);
    J.zero();
    (void)driver.assembleJacobian(J);

    const auto res = driver.verifyJacobianFD(J, /*tol=*/1e-12);
    EXPECT_TRUE(res.passed) << res.message;
}

TEST(NonlinearAssemblyDriverEndToEndTest, FiniteDifferenceJacobianMatchesAnalyticForLinearPoisson)
{
    forms::test::SingleTetraMeshAccess mesh;
    auto dof_map = forms::test::createSingleTetraDofMap();
    spaces::H1Space space(ElementType::Tetra4, /*order=*/1);
    PoissonKernel kernel;

    NonlinearAssemblyOptions opts;
    opts.jacobian_strategy = JacobianStrategy::FiniteDifference;
    opts.fd_relative_perturbation = false;
    opts.fd_perturbation = 1e-7;

    std::vector<Real> U = {0.12, -0.05, 0.08, -0.02};

    DenseMatrixView J_analytic(/*n_dofs=*/4);
    J_analytic.zero();
    {
        NonlinearAssemblyDriver driver;
        driver.setMesh(mesh);
        driver.setDofMap(dof_map);
        driver.setSpace(space);
        driver.setKernel(kernel);
        driver.setCurrentSolution(U);
        (void)driver.assembleJacobian(J_analytic);
    }

    DenseMatrixView J_fd(/*n_dofs=*/4);
    J_fd.zero();
    {
        NonlinearAssemblyDriver driver(opts);
        driver.setMesh(mesh);
        driver.setDofMap(dof_map);
        driver.setSpace(space);
        driver.setKernel(kernel);
        driver.setCurrentSolution(U);
        (void)driver.assembleJacobian(J_fd);
    }

    for (GlobalIndex i = 0; i < 4; ++i) {
        for (GlobalIndex j = 0; j < 4; ++j) {
            EXPECT_NEAR(J_fd.getMatrixEntry(i, j), J_analytic.getMatrixEntry(i, j), 1e-12);
        }
    }
}

} // namespace testing
} // namespace assembly
} // namespace FE
} // namespace svmp


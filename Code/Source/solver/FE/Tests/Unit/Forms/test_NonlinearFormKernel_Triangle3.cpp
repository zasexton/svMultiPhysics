/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

/**
 * @file test_NonlinearFormKernel_Triangle3.cpp
 * @brief Unit tests extending nonlinear residual/Jacobian FD checks beyond Tetra4
 */

#include <gtest/gtest.h>

#include "Assembly/GlobalSystemView.h"
#include "Assembly/StandardAssembler.h"
#include "Forms/FormCompiler.h"
#include "Forms/FormKernels.h"
#include "Spaces/H1Space.h"
#include "Tests/Unit/Forms/FormsTestHelpers.h"

#include <array>
#include <vector>

namespace svmp {
namespace FE {
namespace forms {
namespace test {

TEST(NonlinearFormKernelTriangle3Test, JacobianMatchesCentralDifferences)
{
    SingleTriangleMeshAccess mesh;
    auto dof_map = createSingleTriangleDofMap();
    spaces::H1Space space(ElementType::Triangle3, 1);

    FormCompiler compiler;
    const auto u = FormExpr::trialFunction(space, "u");
    const auto v = FormExpr::testFunction(space, "v");
    const auto residual = (u * u * v).dx();

    auto ir = compiler.compileResidual(residual);
    NonlinearFormKernel kernel(std::move(ir), ADMode::Forward);

    assembly::StandardAssembler assembler;
    assembler.setDofMap(dof_map);

    std::vector<Real> U = {0.1, -0.2, 0.3};
    assembler.setCurrentSolution(U);

    assembly::DenseMatrixView J(3);
    assembly::DenseVectorView R(3);
    J.zero();
    R.zero();

    (void)assembler.assembleBoth(mesh, space, space, kernel, J, R);

    std::array<Real, 3> R0{};
    for (GlobalIndex i = 0; i < 3; ++i) {
        R0[static_cast<std::size_t>(i)] = R.getVectorEntry(i);
    }

    // Central-difference check: J(:,j) ~= (R(U+eps e_j) - R(U-eps e_j)) / (2 eps)
    const Real eps = 1e-6;
    for (GlobalIndex j = 0; j < 3; ++j) {
        auto U_plus = U;
        auto U_minus = U;
        U_plus[static_cast<std::size_t>(j)] += eps;
        U_minus[static_cast<std::size_t>(j)] -= eps;

        assembler.setCurrentSolution(U_plus);
        assembly::DenseVectorView Rp(3);
        Rp.zero();
        (void)assembler.assembleVector(mesh, space, kernel, Rp);

        assembler.setCurrentSolution(U_minus);
        assembly::DenseVectorView Rm(3);
        Rm.zero();
        (void)assembler.assembleVector(mesh, space, kernel, Rm);

        for (GlobalIndex i = 0; i < 3; ++i) {
            const Real fd = (Rp.getVectorEntry(i) - Rm.getVectorEntry(i)) / (2.0 * eps);
            EXPECT_NEAR(J.getMatrixEntry(i, j), fd, 1e-10);
        }
    }
}

} // namespace test
} // namespace forms
} // namespace FE
} // namespace svmp


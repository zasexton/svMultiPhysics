/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

/**
 * @file test_NonlinearFormKernel_DGHelpers.cpp
 * @brief Central-difference Jacobian verification for DG vocabulary helper operators.
 */

#include <gtest/gtest.h>

#include "Assembly/GlobalSystemView.h"
#include "Assembly/StandardAssembler.h"
#include "Forms/FormCompiler.h"
#include "Forms/FormKernels.h"
#include "Forms/Vocabulary.h"
#include "Spaces/H1Space.h"
#include "Tests/Unit/Forms/FormsTestHelpers.h"

#include <vector>

namespace svmp {
namespace FE {
namespace forms {
namespace test {

namespace {

void expectInteriorFaceJacobianMatchesCentralFD(const assembly::IMeshAccess& mesh,
                                               const dofs::DofMap& dof_map,
                                               const spaces::FunctionSpace& space,
                                               assembly::AssemblyKernel& kernel_both,
                                               assembly::AssemblyKernel& kernel_vec,
                                               const std::vector<Real>& U,
                                               Real eps,
                                               Real tol)
{
    assembly::StandardAssembler assembler;
    assembler.setDofMap(dof_map);
    assembler.setCurrentSolution(U);

    const auto n = dof_map.getNumDofs();

    assembly::DenseMatrixView J(n);
    assembly::DenseVectorView R(n);
    J.zero();
    R.zero();

    const auto result = assembler.assembleInteriorFaces(mesh, space, space, kernel_both, J, &R);
    EXPECT_EQ(result.interior_faces_assembled, 1);

    for (GlobalIndex j = 0; j < n; ++j) {
        auto U_plus = U;
        auto U_minus = U;
        U_plus[static_cast<std::size_t>(j)] += eps;
        U_minus[static_cast<std::size_t>(j)] -= eps;

        assembler.setCurrentSolution(U_plus);
        assembly::DenseMatrixView M_dummy_p(n);
        assembly::DenseVectorView Rp(n);
        M_dummy_p.zero();
        Rp.zero();
        (void)assembler.assembleInteriorFaces(mesh, space, space, kernel_vec, M_dummy_p, &Rp);

        assembler.setCurrentSolution(U_minus);
        assembly::DenseMatrixView M_dummy_m(n);
        assembly::DenseVectorView Rm(n);
        M_dummy_m.zero();
        Rm.zero();
        (void)assembler.assembleInteriorFaces(mesh, space, space, kernel_vec, M_dummy_m, &Rm);

        for (GlobalIndex i = 0; i < n; ++i) {
            SCOPED_TRACE(::testing::Message() << "i=" << i << ", j=" << j);
            const Real fd = (Rp.getVectorEntry(i) - Rm.getVectorEntry(i)) / (2.0 * eps);
            EXPECT_NEAR(J.getMatrixEntry(i, j), fd, tol);
        }
    }
}

} // namespace

TEST(NonlinearFormKernelDGHelpersTest, UpwindValueJacobianMatchesCentralDifferences)
{
    TwoTetraSharedFaceMeshAccess mesh;
    auto dof_map = createTwoTetraDG_DofMap();
    spaces::H1Space space(ElementType::Tetra4, /*order=*/1);

    FormCompiler compiler;
    const auto u = TrialFunction(space, "u");
    const auto v = TestFunction(space, "v");

    const auto beta = as_vector({FormExpr::constant(Real(1.0)),
                                 FormExpr::constant(Real(1.0)),
                                 FormExpr::constant(Real(1.0))});

    // Minimal upwind DG residual: ∫ u_upwind(u, beta) * [[v]] dS
    const auto residual = (upwindValue(u, beta) * jump(v)).dS();

    auto ir = compiler.compileResidual(residual);
    auto ir_vec = compiler.compileResidual(residual);
    NonlinearFormKernel kernel_both(std::move(ir), ADMode::Forward, NonlinearKernelOutput::Both);
    NonlinearFormKernel kernel_vec(std::move(ir_vec), ADMode::Forward, NonlinearKernelOutput::VectorOnly);

    std::vector<Real> U = {0.12, -0.05, 0.08, 0.02, -0.07, 0.2, 0.05, -0.15};
    expectInteriorFaceJacobianMatchesCentralFD(mesh, dof_map, space,
                                              kernel_both, kernel_vec,
                                              U, /*eps=*/1e-6, /*tol=*/5e-7);
}

TEST(NonlinearFormKernelDGHelpersTest, HarmonicAverageJacobianMatchesCentralDifferences)
{
    TwoTetraSharedFaceMeshAccess mesh;
    auto dof_map = createTwoTetraDG_DofMap();
    spaces::H1Space space(ElementType::Tetra4, /*order=*/1);

    FormCompiler compiler;
    const auto u = TrialFunction(space, "u");
    const auto v = TestFunction(space, "v");

    // Face coefficient from harmonic average of a positive nonlinear diffusion k(u)=1+u^2.
    const auto k = FormExpr::constant(Real(1.0)) + u * u;
    const auto residual = (harmonicAverage(k) * jump(v)).dS();

    auto ir = compiler.compileResidual(residual);
    auto ir_vec = compiler.compileResidual(residual);
    NonlinearFormKernel kernel_both(std::move(ir), ADMode::Forward, NonlinearKernelOutput::Both);
    NonlinearFormKernel kernel_vec(std::move(ir_vec), ADMode::Forward, NonlinearKernelOutput::VectorOnly);

    std::vector<Real> U = {0.12, -0.05, 0.08, 0.02, -0.07, 0.2, 0.05, -0.15};
    expectInteriorFaceJacobianMatchesCentralFD(mesh, dof_map, space,
                                              kernel_both, kernel_vec,
                                              U, /*eps=*/1e-6, /*tol=*/5e-7);
}

} // namespace test
} // namespace forms
} // namespace FE
} // namespace svmp


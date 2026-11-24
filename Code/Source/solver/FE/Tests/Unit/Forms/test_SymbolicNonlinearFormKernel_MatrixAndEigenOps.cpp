/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

/**
 * @file test_SymbolicNonlinearFormKernel_MatrixAndEigenOps.cpp
 * @brief End-to-end FD Jacobian checks for matrix/eigen operators via symbolic tangents.
 */

#include <gtest/gtest.h>

#include "Assembly/GlobalSystemView.h"
#include "Assembly/StandardAssembler.h"
#include "Dofs/DofMap.h"
#include "Forms/FormCompiler.h"
#include "Forms/FormKernels.h"
#include "Forms/Vocabulary.h"
#include "Spaces/H1Space.h"
#include "Tests/Unit/Forms/FormsTestHelpers.h"

#include <cmath>
#include <exception>
#include <vector>

namespace svmp {
namespace FE {
namespace forms {
namespace test {

namespace {

[[nodiscard]] dofs::DofMap makeSingleCellDofMap(GlobalIndex n_dofs)
{
    dofs::DofMap dof_map(1, n_dofs, static_cast<LocalIndex>(n_dofs));
    std::vector<GlobalIndex> cell_dofs(static_cast<std::size_t>(n_dofs));
    for (GlobalIndex i = 0; i < n_dofs; ++i) {
        cell_dofs[static_cast<std::size_t>(i)] = i;
    }
    dof_map.setCellDofs(0, cell_dofs);
    dof_map.setNumDofs(n_dofs);
    dof_map.setNumLocalDofs(n_dofs);
    dof_map.finalize();
    return dof_map;
}

void expectSymbolicJacobianMatchesCentralDifferences(const assembly::IMeshAccess& mesh,
                                                     const spaces::FunctionSpace& space,
                                                     const FormExpr& residual,
                                                     const std::vector<Real>& U,
                                                     Real eps,
                                                     Real tol)
{
    const auto n_dofs = static_cast<GlobalIndex>(space.dofs_per_element());
    ASSERT_EQ(static_cast<GlobalIndex>(U.size()), n_dofs);

    FormCompiler compiler;
    auto ir = compiler.compileResidual(residual);
    SymbolicNonlinearFormKernel kernel(std::move(ir), NonlinearKernelOutput::Both);
    kernel.resolveInlinableConstitutives();

    auto dof_map = makeSingleCellDofMap(n_dofs);
    assembly::StandardAssembler assembler;
    assembler.setDofMap(dof_map);
    assembler.setCurrentSolution(U);

    assembly::DenseMatrixView J(n_dofs);
    assembly::DenseVectorView R(n_dofs);
    J.zero();
    R.zero();
    (void)assembler.assembleBoth(mesh, space, space, kernel, J, R);

    for (GlobalIndex j = 0; j < n_dofs; ++j) {
        auto U_plus = U;
        auto U_minus = U;
        U_plus[static_cast<std::size_t>(j)] += eps;
        U_minus[static_cast<std::size_t>(j)] -= eps;

        assembler.setCurrentSolution(U_plus);
        assembly::DenseVectorView Rp(n_dofs);
        Rp.zero();
        (void)assembler.assembleVector(mesh, space, kernel, Rp);

        assembler.setCurrentSolution(U_minus);
        assembly::DenseVectorView Rm(n_dofs);
        Rm.zero();
        (void)assembler.assembleVector(mesh, space, kernel, Rm);

        for (GlobalIndex i = 0; i < n_dofs; ++i) {
            const Real fd = (Rp.getVectorEntry(i) - Rm.getVectorEntry(i)) / (2.0 * eps);
            EXPECT_NEAR(J.getMatrixEntry(i, j), fd, tol);
        }
    }
}

FormExpr spd2x2FromGradU(const FormExpr& u)
{
    // SPD matrix: A = 2 I + (∇u)(∇u)^T  (2x2 on 2D cells)
    const auto I = FormExpr::identity(2);
    const auto g = grad(u);
    return (2.0 * I) + outer(g, g);
}

FormExpr spd3x3FromGradU(const FormExpr& u)
{
    // SPD matrix: A = diag(2,3,5) + (∇u)(∇u)^T (3x3 on 3D cells).
    // Use distinct baseline eigenvalues to avoid repeated-eigenvalue kinks in symmetricEigenvalue().
    const auto z = FormExpr::constant(Real(0.0));
    const auto A0 = FormExpr::asTensor({
        {FormExpr::constant(Real(2.0)), z, z},
        {z, FormExpr::constant(Real(3.0)), z},
        {z, z, FormExpr::constant(Real(5.0))},
    });
    const auto g = grad(u);
    return A0 + outer(g, g);
}

} // namespace

TEST(SymbolicMatrixFunctionJacobianTest, MatrixExpInFormMatchesCentralDifferences)
{
    SingleTriangleMeshAccess mesh;
    spaces::H1Space space(ElementType::Triangle3, /*order=*/1);

    const auto u = TrialFunction(space, "u");
    const auto v = TestFunction(space, "v");
    const auto A = spd2x2FromGradU(u);

    const auto residual = (A.matrixExp().trace() * v).dx();
    const std::vector<Real> U = {0.12, -0.08, 0.15};

    expectSymbolicJacobianMatchesCentralDifferences(mesh, space, residual, U, /*eps=*/1e-6, /*tol=*/5e-6);
}

TEST(SymbolicMatrixFunctionJacobianTest, MatrixLogInFormMatchesCentralDifferences)
{
    SingleTriangleMeshAccess mesh;
    spaces::H1Space space(ElementType::Triangle3, /*order=*/1);

    const auto u = TrialFunction(space, "u");
    const auto v = TestFunction(space, "v");
    const auto A = spd2x2FromGradU(u);

    const auto residual = (A.matrixLog().trace() * v).dx();
    const std::vector<Real> U = {0.12, -0.08, 0.15};

    expectSymbolicJacobianMatchesCentralDifferences(mesh, space, residual, U, /*eps=*/1e-6, /*tol=*/5e-6);
}

TEST(SymbolicMatrixFunctionJacobianTest, MatrixSqrtInFormMatchesCentralDifferences)
{
    SingleTriangleMeshAccess mesh;
    spaces::H1Space space(ElementType::Triangle3, /*order=*/1);

    const auto u = TrialFunction(space, "u");
    const auto v = TestFunction(space, "v");
    const auto A = spd2x2FromGradU(u);

    const auto residual = (A.matrixSqrt().trace() * v).dx();
    const std::vector<Real> U = {0.12, -0.08, 0.15};

    expectSymbolicJacobianMatchesCentralDifferences(mesh, space, residual, U, /*eps=*/1e-6, /*tol=*/5e-6);
}

TEST(SymbolicMatrixFunctionJacobianTest, MatrixPowInFormMatchesCentralDifferences)
{
    SingleTriangleMeshAccess mesh;
    spaces::H1Space space(ElementType::Triangle3, /*order=*/1);

    const auto u = TrialFunction(space, "u");
    const auto v = TestFunction(space, "v");
    const auto A = spd2x2FromGradU(u);

    const auto residual = (A.matrixPow(FormExpr::constant(Real(2.0))).trace() * v).dx();
    const std::vector<Real> U = {0.12, -0.08, 0.15};

    expectSymbolicJacobianMatchesCentralDifferences(mesh, space, residual, U, /*eps=*/1e-6, /*tol=*/5e-6);
}

TEST(SymbolicEigenJacobianTest, SymmetricEigenvalueInFormMatchesCentralDifferences)
{
    SingleTriangleMeshAccess mesh;
    spaces::H1Space space(ElementType::Triangle3, /*order=*/1);

    const auto u = TrialFunction(space, "u");
    const auto v = TestFunction(space, "v");
    const auto A = spd2x2FromGradU(u);

    const auto residual = (A.symmetricEigenvalue(0) * v).dx();
    const std::vector<Real> U = {0.12, -0.08, 0.15};

    expectSymbolicJacobianMatchesCentralDifferences(mesh, space, residual, U, /*eps=*/1e-6, /*tol=*/5e-6);
}

TEST(SymbolicMatrixFunctionJacobianTest, MatrixExpInForm_Tetra4_3x3_MatchesCentralDifferences)
{
    SingleTetraMeshAccess mesh;
    spaces::H1Space space(ElementType::Tetra4, /*order=*/1);

    const auto u = TrialFunction(space, "u");
    const auto v = TestFunction(space, "v");
    const auto A = spd3x3FromGradU(u);

    const auto residual = (A.matrixExp().trace() * v).dx();
    const std::vector<Real> U = {0.12, -0.08, 0.15, 0.02};

    expectSymbolicJacobianMatchesCentralDifferences(mesh, space, residual, U, /*eps=*/1e-6, /*tol=*/5e-6);
}

TEST(SymbolicMatrixFunctionJacobianTest, MatrixLogInForm_Tetra4_3x3_MatchesCentralDifferences)
{
    SingleTetraMeshAccess mesh;
    spaces::H1Space space(ElementType::Tetra4, /*order=*/1);

    const auto u = TrialFunction(space, "u");
    const auto v = TestFunction(space, "v");
    const auto A = spd3x3FromGradU(u);

    const auto residual = (A.matrixLog().trace() * v).dx();
    const std::vector<Real> U = {0.12, -0.08, 0.15, 0.02};

    expectSymbolicJacobianMatchesCentralDifferences(mesh, space, residual, U, /*eps=*/1e-6, /*tol=*/5e-6);
}

TEST(SymbolicEigenJacobianTest, SymmetricEigenvalue_Tetra4_3x3_Index0_MatchesCentralDifferences)
{
    SingleTetraMeshAccess mesh;
    spaces::H1Space space(ElementType::Tetra4, /*order=*/1);

    const auto u = TrialFunction(space, "u");
    const auto v = TestFunction(space, "v");
    const auto A = spd3x3FromGradU(u);

    const auto residual = (A.symmetricEigenvalue(0) * v).dx();
    const std::vector<Real> U = {0.12, -0.08, 0.15, 0.02};

    expectSymbolicJacobianMatchesCentralDifferences(mesh, space, residual, U, /*eps=*/1e-6, /*tol=*/5e-6);
}

TEST(SymbolicEigenJacobianTest, SymmetricEigenvalue_Tetra4_3x3_Index1_MatchesCentralDifferences)
{
    SingleTetraMeshAccess mesh;
    spaces::H1Space space(ElementType::Tetra4, /*order=*/1);

    const auto u = TrialFunction(space, "u");
    const auto v = TestFunction(space, "v");
    const auto A = spd3x3FromGradU(u);

    const auto residual = (A.symmetricEigenvalue(1) * v).dx();
    const std::vector<Real> U = {0.12, -0.08, 0.15, 0.02};

    expectSymbolicJacobianMatchesCentralDifferences(mesh, space, residual, U, /*eps=*/1e-6, /*tol=*/5e-6);
}

TEST(SymbolicEigenJacobianTest, SymmetricEigenvalue_Tetra4_3x3_Index2_MatchesCentralDifferences)
{
    SingleTetraMeshAccess mesh;
    spaces::H1Space space(ElementType::Tetra4, /*order=*/1);

    const auto u = TrialFunction(space, "u");
    const auto v = TestFunction(space, "v");
    const auto A = spd3x3FromGradU(u);

    const auto residual = (A.symmetricEigenvalue(2) * v).dx();
    const std::vector<Real> U = {0.12, -0.08, 0.15, 0.02};

    expectSymbolicJacobianMatchesCentralDifferences(mesh, space, residual, U, /*eps=*/1e-6, /*tol=*/5e-6);
}

/**
 * eigenvalue(A, i) and A.symmetricEigenvalue(i) are the eigenvalues of the
 * symmetric part (A + A^T)/2, in descending order.  For A = [[1, 4], [0, 2]]
 * the eigenvalues of A are 2 and 1; those of its symmetric part [[1, 2], [2, 2]]
 * are (3 + sqrt(17))/2 and (3 - sqrt(17))/2.  An index outside [0, n) is
 * refused by the evaluator, as by the JIT validator.
 */
TEST(SymbolicEigenJacobianTest, EigenvalueOfANonsymmetricMatrixIsThatOfItsSymmetricPart)
{
    SingleTriangleMeshAccess mesh;
    spaces::H1Space space(ElementType::Triangle3, /*order=*/1);
    auto dof_map = makeSingleCellDofMap(3);

    const auto u = TrialFunction(space, "u");
    const auto v = TestFunction(space, "v");
    const auto A = FormExpr::asTensor({{FormExpr::constant(1.0), FormExpr::constant(4.0)},
                                       {FormExpr::constant(0.0), FormExpr::constant(2.0)}});

    // The entries of the mass matrix of one triangle sum to its area, 1/2.
    const auto assembled = [&](const FormExpr& lambda) {
        FormCompiler compiler;
        FormKernel kernel(compiler.compileBilinear((lambda * u * v).dx()));
        assembly::StandardAssembler assembler;
        assembler.setDofMap(dof_map);
        assembly::DenseMatrixView M(3);
        M.zero();
        (void)assembler.assembleMatrix(mesh, space, space, kernel, M);
        Real sum = 0.0;
        for (GlobalIndex i = 0; i < 3; ++i) {
            for (GlobalIndex j = 0; j < 3; ++j) {
                sum += M.getMatrixEntry(i, j);
            }
        }
        return sum / 0.5;
    };

    const Real largest = 0.5 * (3.0 + std::sqrt(17.0));
    const Real smallest = 0.5 * (3.0 - std::sqrt(17.0));
    EXPECT_NEAR(assembled(eigenvalue(A, 0)), largest, 1e-12);
    EXPECT_NEAR(assembled(eigenvalue(A, 1)), smallest, 1e-12);
    EXPECT_NEAR(assembled(A.symmetricEigenvalue(0)), largest, 1e-12);
    EXPECT_NEAR(assembled(A.symmetricEigenvalue(1)), smallest, 1e-12);

    for (const int which : {-1, 2}) {
        EXPECT_THROW((void)assembled(eigenvalue(A, which)), std::exception) << "index " << which;
        EXPECT_THROW((void)assembled(A.symmetricEigenvalue(which)), std::exception) << "index " << which;
        EXPECT_THROW((void)assembled(A.symmetricEigenvector(which).component(0)), std::exception)
            << "index " << which;
    }
}

} // namespace test
} // namespace forms
} // namespace FE
} // namespace svmp

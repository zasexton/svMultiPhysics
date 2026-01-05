/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

/**
 * @file test_DynamicValueShapes.cpp
 * @brief Unit tests for dynamic-sized Forms values (asVector/asTensor)
 */

#include <gtest/gtest.h>

#include "Assembly/StandardAssembler.h"
#include "Forms/FormCompiler.h"
#include "Forms/FormKernels.h"
#include "Spaces/H1Space.h"
#include "Tests/Unit/Forms/FormsTestHelpers.h"

#include <vector>

namespace svmp {
namespace FE {
namespace forms {
namespace test {

TEST(DynamicValueShapesTest, BilinearAssemblySupportsInnerOfLongVectors)
{
    SingleTetraMeshAccess mesh;
    auto dof_map = createSingleTetraDofMap();
    spaces::H1Space space(ElementType::Tetra4, 1);

    FormCompiler compiler;
    const auto u = FormExpr::trialFunction(space, "u");
    const auto v = FormExpr::testFunction(space, "v");

    const auto a = FormExpr::asVector({FormExpr::constant(1.0),
                                       FormExpr::constant(2.0),
                                       FormExpr::constant(3.0),
                                       FormExpr::constant(4.0)});
    const auto b = FormExpr::asVector({FormExpr::constant(5.0),
                                       FormExpr::constant(6.0),
                                       FormExpr::constant(7.0),
                                       FormExpr::constant(8.0)});
    const auto scale = inner(a, b); // 70

    const auto form = (scale * u * v).dx();
    auto ir = compiler.compileBilinear(form);
    FormKernel kernel(std::move(ir));

    assembly::StandardAssembler assembler;
    assembler.setDofMap(dof_map);

    assembly::DenseMatrixView mat(4);
    mat.zero();
    (void)assembler.assembleMatrix(mesh, space, space, kernel, mat);

    const Real V = 1.0 / 6.0;
    const Real diag = 70.0 * (V / 10.0);
    const Real off = 70.0 * (V / 20.0);
    for (GlobalIndex i = 0; i < 4; ++i) {
        for (GlobalIndex j = 0; j < 4; ++j) {
            const Real expected = (i == j) ? diag : off;
            EXPECT_NEAR(mat.getMatrixEntry(i, j), expected, 1e-12);
        }
    }
}

TEST(DynamicValueShapesTest, ADJacobianSupportsInnerOfLongVectors)
{
    SingleTetraMeshAccess mesh;
    auto dof_map = createSingleTetraDofMap();
    spaces::H1Space space(ElementType::Tetra4, 1);

    FormCompiler compiler;
    const auto u = FormExpr::trialFunction(space, "u");
    const auto v = FormExpr::testFunction(space, "v");

    const auto a = FormExpr::asVector({FormExpr::constant(1.0),
                                       FormExpr::constant(2.0),
                                       FormExpr::constant(3.0),
                                       FormExpr::constant(4.0)});
    const auto b = FormExpr::asVector({FormExpr::constant(5.0),
                                       FormExpr::constant(6.0),
                                       FormExpr::constant(7.0),
                                       FormExpr::constant(8.0)});
    const auto scale = inner(a, b); // 70

    const auto residual = (scale * u * v).dx();
    auto ir = compiler.compileResidual(residual);
    NonlinearFormKernel kernel(std::move(ir), ADMode::Forward);

    assembly::StandardAssembler assembler;
    assembler.setDofMap(dof_map);

    std::vector<Real> U = {0.2, -0.1, 0.05, 0.4};
    assembler.setCurrentSolution(U);

    assembly::DenseMatrixView J(4);
    assembly::DenseVectorView R(4);
    J.zero();
    R.zero();
    (void)assembler.assembleBoth(mesh, space, space, kernel, J, R);

    const Real V = 1.0 / 6.0;
    const Real diag = 70.0 * (V / 10.0);
    const Real off = 70.0 * (V / 20.0);

    for (GlobalIndex i = 0; i < 4; ++i) {
        for (GlobalIndex j = 0; j < 4; ++j) {
            const Real expected = (i == j) ? diag : off;
            EXPECT_NEAR(J.getMatrixEntry(i, j), expected, 1e-12);
        }
    }

    for (GlobalIndex i = 0; i < 4; ++i) {
        Real expected = 0.0;
        for (GlobalIndex j = 0; j < 4; ++j) {
            expected += ((i == j) ? diag : off) * U[static_cast<std::size_t>(j)];
        }
        EXPECT_NEAR(R.getVectorEntry(i), expected, 1e-12);
    }
}

TEST(DynamicValueShapesTest, ADJacobianSupportsComponentOfDynamicTensor)
{
    SingleTetraMeshAccess mesh;
    auto dof_map = createSingleTetraDofMap();
    spaces::H1Space space(ElementType::Tetra4, 1);

    FormCompiler compiler;
    const auto u = FormExpr::trialFunction(space, "u");
    const auto v = FormExpr::testFunction(space, "v");

    const auto A = FormExpr::asTensor({
        {FormExpr::constant(1.0), FormExpr::constant(2.0)},
        {FormExpr::constant(3.0), FormExpr::constant(4.0)},
        {FormExpr::constant(5.0), FormExpr::constant(6.0)},
        {FormExpr::constant(7.0), FormExpr::constant(8.0)},
    });
    const auto scale = component(A, 3, 1); // 8

    const auto residual = (scale * u * v).dx();
    auto ir = compiler.compileResidual(residual);
    NonlinearFormKernel kernel(std::move(ir), ADMode::Forward);

    assembly::StandardAssembler assembler;
    assembler.setDofMap(dof_map);

    std::vector<Real> U = {0.2, -0.1, 0.05, 0.4};
    assembler.setCurrentSolution(U);

    assembly::DenseMatrixView J(4);
    assembly::DenseVectorView R(4);
    J.zero();
    R.zero();
    (void)assembler.assembleBoth(mesh, space, space, kernel, J, R);

    const Real V = 1.0 / 6.0;
    const Real diag = 8.0 * (V / 10.0);
    const Real off = 8.0 * (V / 20.0);

    for (GlobalIndex i = 0; i < 4; ++i) {
        for (GlobalIndex j = 0; j < 4; ++j) {
            const Real expected = (i == j) ? diag : off;
            EXPECT_NEAR(J.getMatrixEntry(i, j), expected, 1e-12);
        }
    }

    for (GlobalIndex i = 0; i < 4; ++i) {
        Real expected = 0.0;
        for (GlobalIndex j = 0; j < 4; ++j) {
            expected += ((i == j) ? diag : off) * U[static_cast<std::size_t>(j)];
        }
        EXPECT_NEAR(R.getVectorEntry(i), expected, 1e-12);
    }
}

} // namespace test
} // namespace forms
} // namespace FE
} // namespace svmp


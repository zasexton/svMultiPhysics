/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

/**
 * @file test_Complex.cpp
 * @brief Unit tests for FE/Forms complex-valued vocabulary helpers
 */

#include <gtest/gtest.h>

#include "Assembly/StandardAssembler.h"
#include "Forms/Complex.h"
#include "Forms/FormCompiler.h"
#include "Forms/FormKernels.h"
#include "Spaces/H1Space.h"
#include "Tests/Unit/Forms/FormsTestHelpers.h"

#include <cmath>
#include <complex>

namespace svmp {
namespace FE {
namespace forms {
namespace test {

namespace {

Real singleTetraVolume()
{
    return 1.0 / 6.0;
}

Real singleTetraP1BasisIntegral()
{
    return singleTetraVolume() / 4.0;
}

assembly::DenseVectorView assembleCellLinear(const FormExpr& scalar_expr,
                                             dofs::DofMap& dof_map,
                                             const assembly::IMeshAccess& mesh,
                                             const spaces::FunctionSpace& space)
{
    FormCompiler compiler;
    const auto v = FormExpr::testFunction(space, "v");
    const auto form = (scalar_expr * v).dx();
    auto ir = compiler.compileLinear(form);
    FormKernel kernel(std::move(ir));

    assembly::StandardAssembler assembler;
    assembler.setDofMap(dof_map);

    assembly::DenseVectorView vec(static_cast<GlobalIndex>(dof_map.getNumDofs()));
    vec.zero();
    (void)assembler.assembleVector(mesh, space, kernel, vec);
    return vec;
}

assembly::DenseMatrixView assembleCellBilinear(const FormExpr& bilinear_form,
                                               dofs::DofMap& dof_map,
                                               const assembly::IMeshAccess& mesh,
                                               const spaces::FunctionSpace& space)
{
    FormCompiler compiler;
    auto ir = compiler.compileBilinear(bilinear_form);
    FormKernel kernel(std::move(ir));

    assembly::StandardAssembler assembler;
    assembler.setDofMap(dof_map);

    assembly::DenseMatrixView mat(static_cast<GlobalIndex>(dof_map.getNumDofs()));
    mat.zero();
    (void)assembler.assembleMatrix(mesh, space, space, kernel, mat);
    return mat;
}

} // namespace

TEST(ComplexTest, ComplexScalarConstantReal)
{
    const auto z = ComplexScalar::constant(1.25, -2.5);
    ASSERT_TRUE(z.isValid());
    ASSERT_NE(z.re.node(), nullptr);
    ASSERT_NE(z.im.node(), nullptr);

    EXPECT_EQ(z.re.node()->type(), FormExprType::Constant);
    EXPECT_EQ(z.im.node()->type(), FormExprType::Constant);

    ASSERT_TRUE(z.re.node()->constantValue().has_value());
    ASSERT_TRUE(z.im.node()->constantValue().has_value());
    EXPECT_DOUBLE_EQ(*z.re.node()->constantValue(), 1.25);
    EXPECT_DOUBLE_EQ(*z.im.node()->constantValue(), -2.5);
}

TEST(ComplexTest, ComplexScalarConstantStdComplex)
{
    const std::complex<Real> z0{3.0, -4.0};
    const auto z = ComplexScalar::constant(z0);
    ASSERT_TRUE(z.isValid());

    ASSERT_TRUE(z.re.node()->constantValue().has_value());
    ASSERT_TRUE(z.im.node()->constantValue().has_value());
    EXPECT_DOUBLE_EQ(*z.re.node()->constantValue(), 3.0);
    EXPECT_DOUBLE_EQ(*z.im.node()->constantValue(), -4.0);
}

TEST(ComplexTest, ImaginaryUnit)
{
    const auto z = I();
    ASSERT_TRUE(z.isValid());

    ASSERT_TRUE(z.re.node()->constantValue().has_value());
    ASSERT_TRUE(z.im.node()->constantValue().has_value());
    EXPECT_DOUBLE_EQ(*z.re.node()->constantValue(), 0.0);
    EXPECT_DOUBLE_EQ(*z.im.node()->constantValue(), 1.0);
}

TEST(ComplexTest, ComplexConjugate)
{
    SingleTetraMeshAccess mesh;
    auto dof_map = createSingleTetraDofMap();
    spaces::H1Space space(ElementType::Tetra4, 1);

    const auto z = ComplexScalar::constant(2.0, 3.0);
    const auto c = conj(z);

    const Real scale = singleTetraP1BasisIntegral();
    auto re = assembleCellLinear(c.re, dof_map, mesh, space);
    auto im = assembleCellLinear(c.im, dof_map, mesh, space);

    for (GlobalIndex i = 0; i < 4; ++i) {
        EXPECT_NEAR(re.getVectorEntry(i), 2.0 * scale, 1e-12);
        EXPECT_NEAR(im.getVectorEntry(i), -3.0 * scale, 1e-12);
    }
}

TEST(ComplexTest, ComplexAdditionSubtractionNegation)
{
    SingleTetraMeshAccess mesh;
    auto dof_map = createSingleTetraDofMap();
    spaces::H1Space space(ElementType::Tetra4, 1);

    const auto a = ComplexScalar::constant(2.0, 3.0);
    const auto b = ComplexScalar::constant(4.0, -5.0);

    const auto sum = a + b;
    const auto diff = a - b;
    const auto neg_a = -a;

    const Real scale = singleTetraP1BasisIntegral();

    auto sum_re = assembleCellLinear(sum.re, dof_map, mesh, space);
    auto sum_im = assembleCellLinear(sum.im, dof_map, mesh, space);
    auto diff_re = assembleCellLinear(diff.re, dof_map, mesh, space);
    auto diff_im = assembleCellLinear(diff.im, dof_map, mesh, space);
    auto neg_re = assembleCellLinear(neg_a.re, dof_map, mesh, space);
    auto neg_im = assembleCellLinear(neg_a.im, dof_map, mesh, space);

    for (GlobalIndex i = 0; i < 4; ++i) {
        EXPECT_NEAR(sum_re.getVectorEntry(i), (2.0 + 4.0) * scale, 1e-12);
        EXPECT_NEAR(sum_im.getVectorEntry(i), (3.0 - 5.0) * scale, 1e-12);
        EXPECT_NEAR(diff_re.getVectorEntry(i), (2.0 - 4.0) * scale, 1e-12);
        EXPECT_NEAR(diff_im.getVectorEntry(i), (3.0 + 5.0) * scale, 1e-12);
        EXPECT_NEAR(neg_re.getVectorEntry(i), (-2.0) * scale, 1e-12);
        EXPECT_NEAR(neg_im.getVectorEntry(i), (-3.0) * scale, 1e-12);
    }
}

TEST(ComplexTest, ComplexMultiplication)
{
    SingleTetraMeshAccess mesh;
    auto dof_map = createSingleTetraDofMap();
    spaces::H1Space space(ElementType::Tetra4, 1);

    const auto a = ComplexScalar::constant(2.0, 3.0);
    const auto b = ComplexScalar::constant(4.0, -5.0);
    const auto prod = a * b;

    const Real expected_re = 2.0 * 4.0 - 3.0 * (-5.0);
    const Real expected_im = 2.0 * (-5.0) + 3.0 * 4.0;
    const Real scale = singleTetraP1BasisIntegral();

    auto re = assembleCellLinear(prod.re, dof_map, mesh, space);
    auto im = assembleCellLinear(prod.im, dof_map, mesh, space);

    for (GlobalIndex i = 0; i < 4; ++i) {
        EXPECT_NEAR(re.getVectorEntry(i), expected_re * scale, 1e-12);
        EXPECT_NEAR(im.getVectorEntry(i), expected_im * scale, 1e-12);
    }
}

TEST(ComplexTest, ComplexScalarTimesFormExprAndReal)
{
    SingleTetraMeshAccess mesh;
    auto dof_map = createSingleTetraDofMap();
    spaces::H1Space space(ElementType::Tetra4, 1);

    const auto z = ComplexScalar::constant(2.0, 3.0);
    const auto s = FormExpr::constant(4.0);

    const auto left = z * s;
    const auto right = s * z;
    const auto rscale_left = 3.0 * z;
    const auto rscale_right = z * 3.0;

    const Real scale = singleTetraP1BasisIntegral();

    auto left_re = assembleCellLinear(left.re, dof_map, mesh, space);
    auto left_im = assembleCellLinear(left.im, dof_map, mesh, space);
    auto right_re = assembleCellLinear(right.re, dof_map, mesh, space);
    auto right_im = assembleCellLinear(right.im, dof_map, mesh, space);
    auto rsl_re = assembleCellLinear(rscale_left.re, dof_map, mesh, space);
    auto rsl_im = assembleCellLinear(rscale_left.im, dof_map, mesh, space);
    auto rsr_re = assembleCellLinear(rscale_right.re, dof_map, mesh, space);
    auto rsr_im = assembleCellLinear(rscale_right.im, dof_map, mesh, space);

    for (GlobalIndex i = 0; i < 4; ++i) {
        EXPECT_NEAR(left_re.getVectorEntry(i), (2.0 * 4.0) * scale, 1e-12);
        EXPECT_NEAR(left_im.getVectorEntry(i), (3.0 * 4.0) * scale, 1e-12);
        EXPECT_NEAR(right_re.getVectorEntry(i), (2.0 * 4.0) * scale, 1e-12);
        EXPECT_NEAR(right_im.getVectorEntry(i), (3.0 * 4.0) * scale, 1e-12);

        EXPECT_NEAR(rsl_re.getVectorEntry(i), (3.0 * 2.0) * scale, 1e-12);
        EXPECT_NEAR(rsl_im.getVectorEntry(i), (3.0 * 3.0) * scale, 1e-12);
        EXPECT_NEAR(rsr_re.getVectorEntry(i), (3.0 * 2.0) * scale, 1e-12);
        EXPECT_NEAR(rsr_im.getVectorEntry(i), (3.0 * 3.0) * scale, 1e-12);
    }
}

TEST(ComplexTest, ComplexLinearFormValidationAndToRealBlock2x1)
{
    const ComplexLinearForm f_good{FormExpr::constant(1.0), FormExpr::constant(2.0)};
    EXPECT_TRUE(f_good.isValid());

    const ComplexLinearForm f_bad{FormExpr::constant(1.0), FormExpr{}};
    EXPECT_FALSE(f_bad.isValid());

    const auto blocks = toRealBlock2x1(f_good);
    EXPECT_EQ(blocks.numTestFields(), 2u);
    EXPECT_TRUE(blocks.hasBlock(0));
    EXPECT_TRUE(blocks.hasBlock(1));
    EXPECT_EQ(blocks.block(0).toString(), f_good.re.toString());
    EXPECT_EQ(blocks.block(1).toString(), f_good.im.toString());
}

TEST(ComplexTest, ToRealBlock2x2Symmetry)
{
    SingleTetraMeshAccess mesh;
    auto dof_map = createSingleTetraDofMap();
    spaces::H1Space space(ElementType::Tetra4, 1);

    const auto u = FormExpr::trialFunction(space, "u");
    const auto v = FormExpr::testFunction(space, "v");

    const auto a_re = (u * v).dx();
    const auto a_im = (FormExpr::constant(2.0) * u * v).dx();
    const ComplexBilinearForm a{a_re, a_im};
    const auto blocks = toRealBlock2x2(a);

    const auto A_re = assembleCellBilinear(a_re, dof_map, mesh, space);
    const auto A_im = assembleCellBilinear(a_im, dof_map, mesh, space);

    const auto A00 = assembleCellBilinear(blocks.block(0, 0), dof_map, mesh, space);
    const auto A01 = assembleCellBilinear(blocks.block(0, 1), dof_map, mesh, space);
    const auto A10 = assembleCellBilinear(blocks.block(1, 0), dof_map, mesh, space);
    const auto A11 = assembleCellBilinear(blocks.block(1, 1), dof_map, mesh, space);

    for (GlobalIndex r = 0; r < 4; ++r) {
        for (GlobalIndex c = 0; c < 4; ++c) {
            EXPECT_NEAR(A00.getMatrixEntry(r, c), A_re.getMatrixEntry(r, c), 1e-12);
            EXPECT_NEAR(A11.getMatrixEntry(r, c), A_re.getMatrixEntry(r, c), 1e-12);
            EXPECT_NEAR(A10.getMatrixEntry(r, c), A_im.getMatrixEntry(r, c), 1e-12);
            EXPECT_NEAR(A01.getMatrixEntry(r, c), -A_im.getMatrixEntry(r, c), 1e-12);
        }
    }
}

} // namespace test
} // namespace forms
} // namespace FE
} // namespace svmp


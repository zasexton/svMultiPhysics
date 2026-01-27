/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include <gtest/gtest.h>

#include "Forms/FormExpr.h"
#include "Forms/Index.h"
#include "Forms/Tensor/SymmetryOptimizer.h"

namespace svmp::FE::forms::tensor {

TEST(SymmetryOptimizer, CanonicalizeSymmetricPairAndPacking)
{
    const auto sym = TensorSymmetry::symmetric2();
    auto c = canonicalizeComponent(sym, {2, 0});
    EXPECT_TRUE(c.ok);
    EXPECT_FALSE(c.is_zero);
    EXPECT_EQ(c.sign, 1);
    ASSERT_EQ(c.indices.size(), 2u);
    EXPECT_EQ(c.indices[0], 0);
    EXPECT_EQ(c.indices[1], 2);

    EXPECT_EQ(packedIndexSymmetricPair(0, 0, 3), 0);
    EXPECT_EQ(packedIndexSymmetricPair(0, 1, 3), 1);
    EXPECT_EQ(packedIndexSymmetricPair(0, 2, 3), 2);
    EXPECT_EQ(packedIndexSymmetricPair(1, 1, 3), 3);
    EXPECT_EQ(packedIndexSymmetricPair(1, 2, 3), 4);
    EXPECT_EQ(packedIndexSymmetricPair(2, 2, 3), 5);
}

TEST(SymmetryOptimizer, CanonicalizeAntisymmetricPairSignZeroAndPacking)
{
    const auto skew = TensorSymmetry::antisymmetric2();
    auto c = canonicalizeComponent(skew, {1, 0});
    EXPECT_TRUE(c.ok);
    EXPECT_FALSE(c.is_zero);
    EXPECT_EQ(c.sign, -1);
    ASSERT_EQ(c.indices.size(), 2u);
    EXPECT_EQ(c.indices[0], 0);
    EXPECT_EQ(c.indices[1], 1);

    auto d = canonicalizeComponent(skew, {1, 1});
    EXPECT_TRUE(d.ok);
    EXPECT_TRUE(d.is_zero);

    EXPECT_EQ(packedIndexAntisymmetricPair(0, 1, 3), 0);
    EXPECT_EQ(packedIndexAntisymmetricPair(0, 2, 3), 1);
    EXPECT_EQ(packedIndexAntisymmetricPair(1, 2, 3), 2);
}

TEST(SymmetryOptimizer, ElasticityVoigtPackingIsDeterministic)
{
    const auto elast = TensorSymmetry::elasticity();
    auto c = canonicalizeComponent(elast, {1, 0, 2, 1});
    EXPECT_TRUE(c.ok);
    EXPECT_FALSE(c.is_zero);
    EXPECT_EQ(c.sign, 1);
    ASSERT_EQ(c.indices.size(), 4u);
    EXPECT_EQ(c.indices[0], 0);
    EXPECT_EQ(c.indices[1], 1);
    EXPECT_EQ(c.indices[2], 1);
    EXPECT_EQ(c.indices[3], 2);

    const int idx = packedIndexElasticityVoigt(1, 0, 2, 1, 3);
    EXPECT_GE(idx, 0);
    EXPECT_LT(idx, 21); // 3D full elasticity has 21 independent components.
}

TEST(SymmetryOptimizer, LowerWithSymmetryCanonicalizesIndexedAccess)
{
    const auto A = FormExpr::asTensor({
        {FormExpr::constant(1.0), FormExpr::constant(2.0), FormExpr::constant(3.0)},
        {FormExpr::constant(4.0), FormExpr::constant(5.0), FormExpr::constant(6.0)},
        {FormExpr::constant(7.0), FormExpr::constant(8.0), FormExpr::constant(9.0)},
    });

    forms::Index i("i");
    forms::Index j("j");

    const auto sym_swapped = A.sym()(j, i);
    const auto r1 = lowerWithSymmetry(sym_swapped);
    ASSERT_TRUE(r1.ok);
    ASSERT_TRUE(r1.expr.isValid());
    ASSERT_NE(r1.expr.node(), nullptr);
    ASSERT_EQ(r1.expr.node()->type(), FormExprType::IndexedAccess);
    const auto ids1 = r1.expr.node()->indexIds().value();
    EXPECT_LT(ids1[0], ids1[1]); // canonical by id

    const auto skew_diag = A.skew()(i, i);
    const auto r2 = lowerWithSymmetry(skew_diag);
    ASSERT_TRUE(r2.ok);
    ASSERT_TRUE(r2.expr.isValid());
    ASSERT_NE(r2.expr.node(), nullptr);
    EXPECT_EQ(r2.expr.node()->type(), FormExprType::Constant);
    EXPECT_NEAR(r2.expr.node()->constantValue().value_or(1.0), 0.0, 0.0);

    const auto skew_swapped = A.skew()(j, i);
    const auto r3 = lowerWithSymmetry(skew_swapped);
    ASSERT_TRUE(r3.ok);
    ASSERT_TRUE(r3.expr.isValid());
    ASSERT_NE(r3.expr.node(), nullptr);
    EXPECT_EQ(r3.expr.node()->type(), FormExprType::Negate);
}

} // namespace svmp::FE::forms::tensor


/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include <gtest/gtest.h>

#include "Forms/FormExpr.h"
#include "Forms/Index.h"
#include "Forms/Tensor/TensorCSE.h"

namespace svmp::FE::forms::tensor {

TEST(TensorCSE, DetectsRepeatedContractionUpToIndexRenaming)
{
    const auto A = FormExpr::asVector({FormExpr::constant(1.0), FormExpr::constant(2.0), FormExpr::constant(3.0)});
    const auto B = FormExpr::asVector({FormExpr::constant(4.0), FormExpr::constant(5.0), FormExpr::constant(6.0)});

    forms::Index i("i");
    forms::Index j("j");

    const auto s1 = A(i) * B(i);
    const auto s2 = A(j) * B(j);
    const auto expr = s1 + s2;

    TensorCSEOptions opts;
    opts.min_subtree_nodes = 1; // allow small repeated terms
    const auto plan = planTensorCSE(expr, opts);
    ASSERT_TRUE(plan.ok);

    // One temporary for the repeated contraction.
    ASSERT_EQ(plan.temporaries.size(), 1u);
    EXPECT_EQ(plan.temporaries[0].use_count, 2u);
}

TEST(TensorCSE, OrdersTemporariesWithDependenciesFirst)
{
    const auto A = FormExpr::asTensor({
        {FormExpr::constant(1.0), FormExpr::constant(2.0), FormExpr::constant(3.0)},
        {FormExpr::constant(0.5), FormExpr::constant(-1.0), FormExpr::constant(4.0)},
        {FormExpr::constant(1.5), FormExpr::constant(0.0), FormExpr::constant(2.0)},
    });
    const auto F = FormExpr::identity(3) + A;

    const auto expr = log(det(F)) + log(det(F));

    TensorCSEOptions opts;
    opts.min_subtree_nodes = 1;
    const auto plan = planTensorCSE(expr, opts);
    ASSERT_TRUE(plan.ok);

    // Expect two temporaries: det(F) and log(det(F)).
    ASSERT_EQ(plan.temporaries.size(), 2u);
    EXPECT_EQ(plan.temporaries[0].expr.node()->type(), FormExprType::Determinant);
    EXPECT_EQ(plan.temporaries[1].expr.node()->type(), FormExprType::Log);
    EXPECT_EQ(plan.temporaries[0].use_count, 2u);
    EXPECT_EQ(plan.temporaries[1].use_count, 2u);
}

} // namespace svmp::FE::forms::tensor


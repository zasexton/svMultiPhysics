/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Forms/Tensor/TensorSimplify.h"

#include "Forms/FormExpr.h"
#include "Forms/Tensor/TensorIndex.h"

#include <gtest/gtest.h>

namespace svmp::FE::forms::tensor {

namespace {

[[nodiscard]] std::pair<int, IndexVariance> getSingleIndexMetadata(const FormExpr& expr)
{
    if (!expr.isValid() || expr.node() == nullptr) {
        ADD_FAILURE() << "Expected valid FormExpr with a node";
        return {-1, IndexVariance::None};
    }
    if (expr.node()->type() != FormExprType::IndexedAccess) {
        ADD_FAILURE() << "Expected IndexedAccess node";
        return {-1, IndexVariance::None};
    }
    if (expr.node()->indexRank().value_or(0) != 1) {
        ADD_FAILURE() << "Expected rank-1 IndexedAccess";
        return {-1, IndexVariance::None};
    }

    const auto ids_opt = expr.node()->indexIds();
    const auto vars_opt = expr.node()->indexVariances();
    if (!ids_opt.has_value() || !vars_opt.has_value()) {
        ADD_FAILURE() << "IndexedAccess missing id/variance metadata";
        return {-1, IndexVariance::None};
    }
    const auto ids = *ids_opt;
    const auto vars = *vars_opt;
    return {ids[0], vars[0]};
}

} // namespace

TEST(TensorSimplify, DeltaTraceToDimension)
{
    TensorIndex iU;
    iU.id = 7;
    iU.name = "i";
    iU.variance = IndexVariance::Upper;
    iU.dimension = 3;

    TensorIndex iL = iU;
    iL.variance = IndexVariance::Lower;

    const auto expr = FormExpr::identity()(iU, iL);
    TensorSimplifyOptions opts;
    opts.max_passes = 4;
    opts.canonicalize_terms = false;

    const auto r = simplifyTensorExpr(expr, opts);
    EXPECT_TRUE(r.ok);
    ASSERT_NE(r.expr.node(), nullptr);
    EXPECT_EQ(r.expr.node()->type(), FormExprType::Constant);
    EXPECT_DOUBLE_EQ(r.expr.node()->constantValue().value_or(-1.0), 3.0);
    EXPECT_EQ(r.stats.delta_traces, 1u);
    EXPECT_TRUE(r.changed);
}

TEST(TensorSimplify, DeltaSubstitutionRewritesIndexAndVariance)
{
    const auto A = FormExpr::asVector({FormExpr::constant(1.0), FormExpr::constant(2.0), FormExpr::constant(3.0)});

    TensorIndex iU;
    iU.id = 1;
    iU.name = "i";
    iU.variance = IndexVariance::Upper;
    iU.dimension = 3;

    TensorIndex jL;
    jL.id = 2;
    jL.name = "j";
    jL.variance = IndexVariance::Lower;
    jL.dimension = 3;

    TensorIndex jU = jL;
    jU.variance = IndexVariance::Upper;

    const auto delta = FormExpr::identity()(iU, jL); // δ^i_j
    const auto expr = delta * A(jU);

    TensorSimplifyOptions opts;
    opts.max_passes = 6;
    opts.canonicalize_terms = false;

    const auto r = simplifyTensorExpr(expr, opts);
    EXPECT_TRUE(r.ok);
    EXPECT_EQ(r.stats.delta_substitutions, 1u);
    EXPECT_TRUE(r.changed);

    const auto [id0, var0] = getSingleIndexMetadata(r.expr);
    EXPECT_EQ(id0, iU.id);
    EXPECT_EQ(var0, IndexVariance::Upper);
}

TEST(TensorSimplify, DeltaComposition)
{
    TensorIndex iU;
    iU.id = 10;
    iU.name = "i";
    iU.variance = IndexVariance::Upper;
    iU.dimension = 3;

    TensorIndex jL;
    jL.id = 11;
    jL.name = "j";
    jL.variance = IndexVariance::Lower;
    jL.dimension = 3;

    TensorIndex jU = jL;
    jU.variance = IndexVariance::Upper;

    TensorIndex kL;
    kL.id = 12;
    kL.name = "k";
    kL.variance = IndexVariance::Lower;
    kL.dimension = 3;

    const auto d1 = FormExpr::identity()(iU, jL);
    const auto d2 = FormExpr::identity()(jU, kL);
    const auto expr = d1 * d2;

    TensorSimplifyOptions opts;
    opts.max_passes = 6;
    opts.canonicalize_terms = false;

    const auto r = simplifyTensorExpr(expr, opts);
    EXPECT_TRUE(r.ok);
    EXPECT_EQ(r.stats.delta_compositions, 1u);
    ASSERT_NE(r.expr.node(), nullptr);
    EXPECT_EQ(r.expr.node()->type(), FormExprType::IndexedAccess);
    EXPECT_EQ(r.expr.node()->indexRank().value_or(0), 2);

    const auto ids = r.expr.node()->indexIds().value();
    const auto vars = r.expr.node()->indexVariances().value();
    EXPECT_EQ(ids[0], iU.id);
    EXPECT_EQ(ids[1], kL.id);
    EXPECT_EQ(vars[0], IndexVariance::Upper);
    EXPECT_EQ(vars[1], IndexVariance::Lower);
}

TEST(TensorSimplify, SymmetryAnnihilation)
{
    const auto A = FormExpr::asTensor({
        {FormExpr::constant(1.0), FormExpr::constant(2.0)},
        {FormExpr::constant(3.0), FormExpr::constant(4.0)},
    });
    const auto B = FormExpr::asTensor({
        {FormExpr::constant(5.0), FormExpr::constant(6.0)},
        {FormExpr::constant(7.0), FormExpr::constant(8.0)},
    });

    TensorIndex i;
    i.id = 21;
    i.name = "i";
    i.variance = IndexVariance::None;
    i.dimension = 2;

    TensorIndex j;
    j.id = 22;
    j.name = "j";
    j.variance = IndexVariance::None;
    j.dimension = 2;

    const auto expr = A.sym()(i, j) * B.skew()(i, j);

    TensorSimplifyOptions opts;
    opts.max_passes = 4;
    opts.canonicalize_terms = false;

    const auto r = simplifyTensorExpr(expr, opts);
    EXPECT_TRUE(r.ok);
    EXPECT_EQ(r.stats.symmetry_zeroes, 1u);
    ASSERT_NE(r.expr.node(), nullptr);
    EXPECT_EQ(r.expr.node()->type(), FormExprType::Constant);
    EXPECT_DOUBLE_EQ(r.expr.node()->constantValue().value_or(1.0), 0.0);
}

TEST(TensorSimplify, EpsilonIdentityViaCross)
{
    const auto a = FormExpr::asVector({FormExpr::parameter("a0"), FormExpr::parameter("a1"), FormExpr::parameter("a2")});
    const auto b = FormExpr::asVector({FormExpr::parameter("b0"), FormExpr::parameter("b1"), FormExpr::parameter("b2")});
    const auto c = FormExpr::asVector({FormExpr::parameter("c0"), FormExpr::parameter("c1"), FormExpr::parameter("c2")});
    const auto d = FormExpr::asVector({FormExpr::parameter("d0"), FormExpr::parameter("d1"), FormExpr::parameter("d2")});

    const auto expr = inner(cross(a, b), cross(c, d));

    TensorSimplifyOptions opts;
    opts.max_passes = 4;
    opts.canonicalize_terms = false;

    const auto r = simplifyTensorExpr(expr, opts);
    EXPECT_TRUE(r.ok);
    EXPECT_EQ(r.stats.epsilon_identities, 1u);

    const auto expected = (inner(a, c) * inner(b, d)) - (inner(a, d) * inner(b, c));
    EXPECT_EQ(r.expr.toString(), expected.toString());
}

TEST(TensorSimplify, MetricLoweringViaDelta)
{
    const auto v = FormExpr::asVector({FormExpr::constant(1.0), FormExpr::constant(2.0), FormExpr::constant(3.0)});

    TensorIndex iL;
    iL.id = 31;
    iL.name = "i";
    iL.variance = IndexVariance::Lower;
    iL.dimension = 3;

    TensorIndex jL;
    jL.id = 32;
    jL.name = "j";
    jL.variance = IndexVariance::Lower;
    jL.dimension = 3;

    TensorIndex jU = jL;
    jU.variance = IndexVariance::Upper;

    const auto g = FormExpr::identity()(iL, jL); // g_ij (identity metric)
    const auto expr = g * v(jU);                 // g_ij v^j -> v_i

    TensorSimplifyOptions opts;
    opts.max_passes = 6;
    opts.canonicalize_terms = false;

    const auto r = simplifyTensorExpr(expr, opts);
    EXPECT_TRUE(r.ok);
    EXPECT_EQ(r.stats.delta_substitutions, 1u);

    const auto [id0, var0] = getSingleIndexMetadata(r.expr);
    EXPECT_EQ(id0, iL.id);
    EXPECT_EQ(var0, IndexVariance::Lower);
}

} // namespace svmp::FE::forms::tensor

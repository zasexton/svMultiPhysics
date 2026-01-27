/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Forms/FormExpr.h"
#include "Forms/Index.h"
#include "Forms/Tensor/TensorCanonicalize.h"
#include "Forms/Tensor/TensorContraction.h"
#include "Forms/Tensor/TensorIndex.h"

#include <gtest/gtest.h>

namespace svmp::FE::forms::tensor {

TEST(TensorPrettyPrinting, SymmetryTags)
{
    const auto A = FormExpr::coefficient("A", [](Real, Real, Real) {
        std::array<std::array<Real, 3>, 3> m{};
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                m[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] = (i == j) ? 1.0 : 0.0;
            }
        }
        return m;
    });

    forms::Index i("i");
    forms::Index j("j");

    EXPECT_EQ(A.sym()(i, j).toString(), "A_{(ij)}");
    EXPECT_EQ(A.skew()(i, j).toString(), "A_{[ij]}");
}

TEST(TensorPrettyPrinting, KroneckerDelta)
{
    forms::Index i("i");
    forms::Index j("j");
    EXPECT_EQ(FormExpr::identity()(i, j).toString(), "\u03B4_{ij}");
}

TEST(TensorPrettyPrinting, LeviCivitaCrossProduct)
{
    const auto a = FormExpr::coefficient("a", [](Real, Real, Real) {
        return std::array<Real, 3>{1.0, 2.0, 3.0};
    });
    const auto b = FormExpr::coefficient("b", [](Real, Real, Real) {
        return std::array<Real, 3>{4.0, 5.0, 6.0};
    });

    forms::Index i("i");
    EXPECT_EQ(cross(a, b)(i).toString(), "(\u03B5_{ijk} * a_{j} * b_{k})");
}

TEST(TensorPrettyPrinting, LeviCivitaCurl)
{
    const auto u = FormExpr::coefficient("u", [](Real, Real, Real) {
        return std::array<Real, 3>{1.0, 0.0, 0.0};
    });

    forms::Index i("i");
    EXPECT_EQ(curl(u)(i).toString(), "(\u03B5_{ijk} * grad(u)_{kj})");
}

TEST(TensorPrettyPrinting, CanonicalIndexString)
{
    const auto A = FormExpr::coefficient("A", [](Real, Real, Real) {
        return std::array<Real, 3>{1.0, 2.0, 3.0};
    });
    const auto B = FormExpr::coefficient("B", [](Real, Real, Real) {
        return std::array<Real, 3>{4.0, 5.0, 6.0};
    });

    forms::Index p;
    forms::Index q;
    const auto expr = A(p) * B(q);

    EXPECT_EQ(toCanonicalString(expr), "(A_{i0} * B_{i1})");
}

TEST(TensorContraction, VarianceValidationRejectsLowerLower)
{
    const auto A = FormExpr::coefficient("A", [](Real, Real, Real) {
        return std::array<Real, 3>{1.0, 2.0, 3.0};
    });
    const auto B = FormExpr::coefficient("B", [](Real, Real, Real) {
        return std::array<Real, 3>{4.0, 5.0, 6.0};
    });

    TensorIndex i;
    i.id = 0;
    i.name = "i";
    i.variance = IndexVariance::Lower;
    i.dimension = 3;

    const auto expr = A(i) * B(i);
    const auto a = analyzeContractions(expr);
    EXPECT_FALSE(a.ok);
    EXPECT_NE(a.message.find("covariant"), std::string::npos);
}

TEST(TensorContraction, VarianceValidationAllowsLowerUpper)
{
    const auto A = FormExpr::coefficient("A", [](Real, Real, Real) {
        return std::array<Real, 3>{1.0, 2.0, 3.0};
    });
    const auto B = FormExpr::coefficient("B", [](Real, Real, Real) {
        return std::array<Real, 3>{4.0, 5.0, 6.0};
    });

    TensorIndex i_lower;
    i_lower.id = 0;
    i_lower.name = "i";
    i_lower.variance = IndexVariance::Lower;
    i_lower.dimension = 3;

    TensorIndex i_upper = i_lower;
    i_upper.variance = IndexVariance::Upper;

    const auto expr = A(i_lower) * B(i_upper);
    const auto a = analyzeContractions(expr);
    EXPECT_TRUE(a.ok);
    EXPECT_TRUE(a.free_indices.empty());
    EXPECT_EQ(a.bound_indices.size(), 1u);
}

} // namespace svmp::FE::forms::tensor

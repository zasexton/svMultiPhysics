/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Forms/Index.h"
#include "Forms/SymbolicDifferentiation.h"
#include "Spaces/H1Space.h"

#include <gtest/gtest.h>

namespace svmp::FE::forms::tensor {

TEST(TensorDifferentiation, DifferentiateResidualAcceptsIndexedAccess)
{
    spaces::H1Space space(ElementType::Tetra4, 1);

    const auto u = FormExpr::trialFunction(space, "u");
    const auto v = FormExpr::testFunction(space, "v");

    const auto A = FormExpr::asVector({u, 2.0 * u, 3.0 * u});
    forms::Index i("i");

    const auto residual = (A(i) * v).dx();

    EXPECT_TRUE(checkSymbolicDifferentiability(residual).ok);
    EXPECT_NO_THROW({
        const auto tangent = differentiateResidual(residual);
        EXPECT_TRUE(tangent.isValid());
        EXPECT_NE(tangent.toString().find("_{i}"), std::string::npos);
    });
}

} // namespace svmp::FE::forms::tensor

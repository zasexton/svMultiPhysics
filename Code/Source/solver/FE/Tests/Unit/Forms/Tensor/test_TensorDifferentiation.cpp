/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Forms/Index.h"
#include "Forms/SymbolicDifferentiation.h"
#include "Forms/Tensor/TensorContraction.h"
#include "Forms/Tensor/TensorDifferentiation.h"
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

TEST(TensorDifferentiationNewPhysics, MatrixFunctionDerivativesPreserveIndexedAccess)
{
    spaces::H1Space space(ElementType::Tetra4, 1);

    const auto u = FormExpr::trialFunction(space, "u");
    const auto v = FormExpr::testFunction(space, "v");

    // SPD-ish symmetric 2x2 matrix built from a scalar field (keeps log/sqrt well-defined in practice).
    const auto A = FormExpr::identity(2) * (u * u + FormExpr::constant(Real(2.0)));

    forms::Index i("i");
    const auto integrand = A.matrixExp()(i, i) * v;
    const auto residual = integrand.dx();

    EXPECT_NO_THROW({
        const auto tangent = differentiateTensorResidual(residual);
        EXPECT_TRUE(tangent.isValid());
        EXPECT_NE(tangent.toString().find("expm_dd("), std::string::npos);

        // Smoke-check: tensor contraction analysis should accept the index-notation form.
        const auto dI = forms::differentiateResidual(integrand);
        const auto a = analyzeContractions(dI);
        EXPECT_TRUE(a.ok);
    });
}

} // namespace svmp::FE::forms::tensor

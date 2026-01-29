/**
 * @file test_JITValidation.cpp
 * @brief Unit tests for FE/Forms JIT compatibility validation + KernelIR hashing
 */

#include <gtest/gtest.h>

#include "Forms/Complex.h"
#include "Forms/FormCompiler.h"
#include "Forms/FormExpr.h"
#include "Forms/JIT/JITValidation.h"
#include "Forms/JIT/KernelIR.h"
#include "Spaces/H1Space.h"

using svmp::FE::Real;

TEST(JITValidation, AcceptsSlotBasedIntegrandInStrictMode)
{
    using namespace svmp::FE::forms;

    const auto expr = FormExpr::parameterRef(0) + FormExpr::constant(Real(2.0));
    const auto r = jit::canCompile(expr, jit::ValidationOptions{.strictness = jit::Strictness::Strict});
    EXPECT_TRUE(r.ok);
    EXPECT_TRUE(r.cacheable);
}

TEST(JITValidation, RejectsParameterSymbolInStrictMode)
{
    using namespace svmp::FE::forms;

    const auto expr = FormExpr::parameter("mu") + FormExpr::constant(Real(1.0));
    const auto r = jit::canCompile(expr, jit::ValidationOptions{.strictness = jit::Strictness::Strict});
    EXPECT_FALSE(r.ok);
    ASSERT_TRUE(r.first_issue.has_value());
    EXPECT_EQ(r.first_issue->type, FormExprType::ParameterSymbol);
}

TEST(JITValidation, RejectsCoupledPlaceholderSymbols)
{
    using namespace svmp::FE::forms;

    const auto expr = FormExpr::boundaryIntegralValue("Q");
    const auto r = jit::canCompile(expr, jit::ValidationOptions{.strictness = jit::Strictness::Strict});
    EXPECT_FALSE(r.ok);
    ASSERT_TRUE(r.first_issue.has_value());
    EXPECT_EQ(r.first_issue->type, FormExprType::BoundaryIntegralSymbol);
}

TEST(JITValidation, RejectsStdFunctionCoefficientInStrictMode)
{
    using namespace svmp::FE::forms;

    const auto f = FormExpr::coefficient("f", [](Real, Real, Real) { return Real(1.0); });
    const auto expr = f + FormExpr::constant(Real(2.0));

    const auto r = jit::canCompile(expr, jit::ValidationOptions{.strictness = jit::Strictness::Strict});
    EXPECT_FALSE(r.ok);
    ASSERT_TRUE(r.first_issue.has_value());
    EXPECT_EQ(r.first_issue->type, FormExprType::Coefficient);
}

TEST(JITValidation, AllowsExternalCallsInRelaxedModeButMarksNonCacheable)
{
    using namespace svmp::FE::forms;

    const auto f = FormExpr::coefficient("f", [](Real, Real, Real) { return Real(1.0); });
    const auto expr = f + FormExpr::constant(Real(2.0));

    const auto r = jit::canCompile(expr, jit::ValidationOptions{.strictness = jit::Strictness::AllowExternalCalls});
    EXPECT_TRUE(r.ok);
    EXPECT_FALSE(r.cacheable);
}

TEST(KernelIR, StableHashIgnoresCommutativeOperandOrder)
{
    using namespace svmp::FE::forms;

    const auto a = FormExpr::parameterRef(0);
    const auto b = FormExpr::constant(Real(1.0));

    const auto e1 = a + b;
    const auto e2 = b + a;

    const auto ir1 = jit::lowerToKernelIR(e1).ir;
    const auto ir2 = jit::lowerToKernelIR(e2).ir;

    EXPECT_EQ(ir1.stableHash64(), ir2.stableHash64());
}

TEST(JITValidation, RejectsMatrixFunctionOnNonSquareMatrix)
{
    using namespace svmp::FE::forms;

    const auto A = FormExpr::asTensor({
        {FormExpr::constant(Real(1.0)), FormExpr::constant(Real(0.0)), FormExpr::constant(Real(0.0))},
        {FormExpr::constant(Real(0.0)), FormExpr::constant(Real(1.0)), FormExpr::constant(Real(0.0))},
    });

    const auto expr = A.matrixExp();
    const auto r = jit::canCompile(expr, jit::ValidationOptions{.strictness = jit::Strictness::Strict});
    EXPECT_FALSE(r.ok);
    ASSERT_TRUE(r.first_issue.has_value());
    EXPECT_EQ(r.first_issue->type, FormExprType::MatrixExponential);
}

TEST(JITValidation, RejectsMatrixLogarithmOnNonSPDConstantMatrix)
{
    using namespace svmp::FE::forms;

    const auto A = FormExpr::asTensor({
        {FormExpr::constant(Real(-1.0)), FormExpr::constant(Real(0.0))},
        {FormExpr::constant(Real(0.0)), FormExpr::constant(Real(1.0))},
    });

    const auto expr = A.matrixLog();
    const auto r = jit::canCompile(expr, jit::ValidationOptions{.strictness = jit::Strictness::Strict});
    EXPECT_FALSE(r.ok);
    ASSERT_TRUE(r.first_issue.has_value());
    EXPECT_EQ(r.first_issue->type, FormExprType::MatrixLogarithm);
}

TEST(JITValidation, RejectsMatrixSqrtOnNonSPDConstantMatrix)
{
    using namespace svmp::FE::forms;

    const auto A = FormExpr::asTensor({
        {FormExpr::constant(Real(-1.0)), FormExpr::constant(Real(0.0))},
        {FormExpr::constant(Real(0.0)), FormExpr::constant(Real(1.0))},
    });

    const auto expr = A.matrixSqrt();
    const auto r = jit::canCompile(expr, jit::ValidationOptions{.strictness = jit::Strictness::Strict});
    EXPECT_FALSE(r.ok);
    ASSERT_TRUE(r.first_issue.has_value());
    EXPECT_EQ(r.first_issue->type, FormExprType::MatrixSqrt);
}

TEST(JITValidation, RejectsMatrixPowerOnNonSPDConstantMatrixWhenExponentNonZero)
{
    using namespace svmp::FE::forms;

    const auto A = FormExpr::asTensor({
        {FormExpr::constant(Real(-1.0)), FormExpr::constant(Real(0.0))},
        {FormExpr::constant(Real(0.0)), FormExpr::constant(Real(1.0))},
    });

    const auto expr = A.matrixPow(FormExpr::constant(Real(1.0)));
    const auto r = jit::canCompile(expr, jit::ValidationOptions{.strictness = jit::Strictness::Strict});
    EXPECT_FALSE(r.ok);
    ASSERT_TRUE(r.first_issue.has_value());
    EXPECT_EQ(r.first_issue->type, FormExprType::MatrixPower);
}

TEST(JITValidation, AllowsMatrixPowerOnNonSPDConstantMatrixWhenExponentZero)
{
    using namespace svmp::FE::forms;

    const auto A = FormExpr::asTensor({
        {FormExpr::constant(Real(-1.0)), FormExpr::constant(Real(0.0))},
        {FormExpr::constant(Real(0.0)), FormExpr::constant(Real(1.0))},
    });

    const auto expr = A.matrixPow(FormExpr::constant(Real(0.0)));
    const auto r = jit::canCompile(expr, jit::ValidationOptions{.strictness = jit::Strictness::Strict});
    EXPECT_TRUE(r.ok);
}

TEST(JITValidation, RejectsMatrixLogDirectionalDerivativeOnNonSPDConstantMatrix)
{
    using namespace svmp::FE::forms;

    const auto A = FormExpr::asTensor({
        {FormExpr::constant(Real(-1.0)), FormExpr::constant(Real(0.0))},
        {FormExpr::constant(Real(0.0)), FormExpr::constant(Real(1.0))},
    });
    const auto dA = FormExpr::identity(2);

    const auto expr = FormExpr::matrixLogDirectionalDerivative(A, dA);
    const auto r = jit::canCompile(expr, jit::ValidationOptions{.strictness = jit::Strictness::Strict});
    EXPECT_FALSE(r.ok);
    ASSERT_TRUE(r.first_issue.has_value());
    EXPECT_EQ(r.first_issue->type, FormExprType::MatrixLogarithmDirectionalDerivative);
}

TEST(JITValidation, RejectsMatrixPowerDirectionalDerivativeWithNonScalarExponent)
{
    using namespace svmp::FE::forms;

    const auto A = FormExpr::identity(2);
    const auto dA = FormExpr::identity(2);
    const auto p = FormExpr::asVector({FormExpr::constant(Real(1.0)), FormExpr::constant(Real(2.0))});

    const auto expr = FormExpr::matrixPowDirectionalDerivative(A, dA, p);
    const auto r = jit::canCompile(expr, jit::ValidationOptions{.strictness = jit::Strictness::Strict});
    EXPECT_FALSE(r.ok);
    ASSERT_TRUE(r.first_issue.has_value());
    EXPECT_EQ(r.first_issue->type, FormExprType::MatrixPowerDirectionalDerivative);
}

TEST(JITValidation, AllowsMatrixPowerDirectionalDerivativeWithZeroExponentEvenForNonSPDConstantMatrix)
{
    using namespace svmp::FE::forms;

    const auto A = FormExpr::asTensor({
        {FormExpr::constant(Real(-1.0)), FormExpr::constant(Real(0.0))},
        {FormExpr::constant(Real(0.0)), FormExpr::constant(Real(1.0))},
    });
    const auto dA = FormExpr::identity(2);
    const auto p = FormExpr::constant(Real(0.0));

    const auto expr = FormExpr::matrixPowDirectionalDerivative(A, dA, p);
    const auto r = jit::canCompile(expr, jit::ValidationOptions{.strictness = jit::Strictness::Strict});
    EXPECT_TRUE(r.ok);
}

TEST(JITValidation, ComplexBlockLiftingProducesJITValidBlocks)
{
    using namespace svmp::FE::forms;

    svmp::FE::spaces::H1Space space(svmp::FE::ElementType::Tetra4, 1);
    const auto u = FormExpr::trialFunction(space, "u");
    const auto v = FormExpr::testFunction(space, "v");

    const auto z = ComplexScalar::constant(Real(2.0), Real(-3.0));
    const auto w = z * (u * v);
    const ComplexBilinearForm a{w.re.dx(), w.im.dx()};
    const auto blocks = toRealBlock2x2(a);

    EXPECT_EQ(blocks.block(0, 0).toString(), blocks.block(1, 1).toString());
    EXPECT_EQ(blocks.block(0, 1).toString(), (-blocks.block(1, 0)).toString());

    FormCompiler compiler;
    for (std::size_t i = 0; i < blocks.numTestFields(); ++i) {
        for (std::size_t j = 0; j < blocks.numTrialFields(); ++j) {
            auto ir = compiler.compileBilinear(blocks.block(i, j));
            const auto r = jit::canCompile(ir, jit::ValidationOptions{.strictness = jit::Strictness::Strict});
            EXPECT_TRUE(r.ok) << (r.first_issue ? r.first_issue->message : "");
        }
    }
}

TEST(JITValidation, RejectsMatrixPowerWithNonScalarExponent)
{
    using namespace svmp::FE::forms;

    const auto A = FormExpr::identity(2);
    const auto p = FormExpr::asVector({FormExpr::constant(Real(1.0)), FormExpr::constant(Real(2.0))});

    const auto expr = A.matrixPow(p);
    const auto r = jit::canCompile(expr, jit::ValidationOptions{.strictness = jit::Strictness::Strict});
    EXPECT_FALSE(r.ok);
    ASSERT_TRUE(r.first_issue.has_value());
    EXPECT_EQ(r.first_issue->type, FormExprType::MatrixPower);
}

TEST(JITValidation, RejectsSmoothMinWithNonScalarInputs)
{
    using namespace svmp::FE::forms;

    const auto a = FormExpr::asVector({FormExpr::constant(Real(1.0)), FormExpr::constant(Real(2.0))});
    const auto b = FormExpr::constant(Real(0.0));
    const auto eps = FormExpr::constant(Real(1e-3));

    const auto expr = a.smoothMin(b, eps);
    const auto r = jit::canCompile(expr, jit::ValidationOptions{.strictness = jit::Strictness::Strict});
    EXPECT_FALSE(r.ok);
    ASSERT_TRUE(r.first_issue.has_value());
    EXPECT_EQ(r.first_issue->type, FormExprType::SmoothMin);
}

TEST(JITValidation, RejectsHistoryOperatorWithNonScalarWeights)
{
    using namespace svmp::FE::forms;

    const auto w = FormExpr::asVector({FormExpr::constant(Real(1.0)), FormExpr::constant(Real(2.0))});
    const auto expr = FormExpr::historyWeightedSum({w});

    const auto r = jit::canCompile(expr, jit::ValidationOptions{.strictness = jit::Strictness::Strict});
    EXPECT_FALSE(r.ok);
    ASSERT_TRUE(r.first_issue.has_value());
    EXPECT_EQ(r.first_issue->type, FormExprType::HistoryWeightedSum);
}

TEST(JITValidation, RejectsHistoryOperatorWithTooManyWeights)
{
    using namespace svmp::FE::forms;

    std::vector<FormExpr> weights;
    for (int i = 0; i < 9; ++i) {
        weights.push_back(FormExpr::constant(Real(1.0)));
    }
    const auto expr = FormExpr::historyWeightedSum(weights);

    const auto r = jit::canCompile(expr, jit::ValidationOptions{.strictness = jit::Strictness::Strict});
    EXPECT_FALSE(r.ok);
    ASSERT_TRUE(r.first_issue.has_value());
    EXPECT_EQ(r.first_issue->type, FormExprType::HistoryWeightedSum);
}

TEST(JITValidation, RejectsTimeDerivativeOrderTooHigh)
{
    using namespace svmp::FE::forms;

    const auto expr = FormExpr::parameterRef(0).dt(9);
    const auto r = jit::canCompile(expr, jit::ValidationOptions{.strictness = jit::Strictness::Strict});
    EXPECT_FALSE(r.ok);
    ASSERT_TRUE(r.first_issue.has_value());
    EXPECT_EQ(r.first_issue->type, FormExprType::TimeDerivative);
}

TEST(JITValidation, RejectsEigenIndexOutOfRange)
{
    using namespace svmp::FE::forms;

    const auto A = FormExpr::identity(2);
    const auto expr = A.eigenvalue(2);

    const auto r = jit::canCompile(expr, jit::ValidationOptions{.strictness = jit::Strictness::Strict});
    EXPECT_FALSE(r.ok);
    ASSERT_TRUE(r.first_issue.has_value());
    EXPECT_EQ(r.first_issue->type, FormExprType::Eigenvalue);
}

/**
 * @file test_JITValidation.cpp
 * @brief Unit tests for FE/Forms JIT compatibility validation + KernelIR hashing
 */

#include <gtest/gtest.h>

#include "Forms/FormExpr.h"
#include "Forms/JIT/JITValidation.h"
#include "Forms/JIT/KernelIR.h"

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


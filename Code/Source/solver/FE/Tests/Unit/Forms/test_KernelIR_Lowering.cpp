/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include <gtest/gtest.h>

#include "Forms/FormExpr.h"
#include "Forms/JIT/KernelIR.h"

#include <bit>
#include <cstdint>

namespace svmp {
namespace FE {
namespace forms {
namespace test {

TEST(KernelIRLowering, LowerConstantPreservesValue)
{
    const Real value = Real(1.25);
    const auto expr = FormExpr::constant(value);

    const auto r = jit::lowerToKernelIR(expr);
    ASSERT_FALSE(r.ir.ops.empty());

    const auto& op = r.ir.ops.at(static_cast<std::size_t>(r.ir.root));
    EXPECT_EQ(op.type, FormExprType::Constant);

    const double roundtrip = std::bit_cast<double>(op.imm0);
    EXPECT_DOUBLE_EQ(roundtrip, static_cast<double>(value));
}

TEST(KernelIRLowering, LoweringContainsExpectedOps)
{
    const auto a = FormExpr::parameterRef(0);
    const auto b = FormExpr::constant(Real(2.0));
    const auto c = FormExpr::parameterRef(1);

    const auto expr = (a + b) * c;
    const auto r = jit::lowerToKernelIR(expr);

    bool saw_add = false;
    bool saw_mul = false;
    for (const auto& op : r.ir.ops) {
        saw_add = saw_add || (op.type == FormExprType::Add);
        saw_mul = saw_mul || (op.type == FormExprType::Multiply);
    }
    EXPECT_TRUE(saw_add);
    EXPECT_TRUE(saw_mul);
}

TEST(KernelIRLowering, CSEDeduplicatesCommonSubexpressions)
{
    const auto a = FormExpr::parameterRef(0);
    const auto b = FormExpr::constant(Real(1.0));
    const auto sub = a + b;
    const auto expr = sub + sub;

    jit::KernelIRBuildOptions with_cse;
    with_cse.cse = true;
    with_cse.canonicalize_commutative = true;

    jit::KernelIRBuildOptions without_cse = with_cse;
    without_cse.cse = false;

    const auto r_cse = jit::lowerToKernelIR(expr, with_cse);
    const auto r_no = jit::lowerToKernelIR(expr, without_cse);

    EXPECT_LT(r_cse.ir.opCount(), r_no.ir.opCount());
}

TEST(KernelIRHashing, DifferentExprsDifferentHash)
{
    const auto a = FormExpr::parameterRef(0);
    const auto b = FormExpr::constant(Real(1.0));

    const auto h1 = jit::lowerToKernelIR(a + b).ir.stableHash64();
    const auto h2 = jit::lowerToKernelIR(a - b).ir.stableHash64();

    EXPECT_NE(h1, h2);
}

} // namespace test
} // namespace forms
} // namespace FE
} // namespace svmp


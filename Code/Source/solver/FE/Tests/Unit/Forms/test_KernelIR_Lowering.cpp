/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include <gtest/gtest.h>

#include "Forms/FormExpr.h"
#include "Forms/JIT/KernelIR.h"
#include "Forms/JIT/HardwareProfile.h"

#include <bit>
#include <cmath>
#include <cstdint>

namespace svmp {
namespace FE {
namespace forms {
namespace test {

// ============================================================================
// Lowering tests
// ============================================================================

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

TEST(KernelIRLowering, AuxiliaryInputRefPreservesSlotIndex)
{
    const std::uint32_t slot = 7;
    const auto r = jit::lowerToKernelIR(FormExpr::auxiliaryInputRef(slot));
    ASSERT_FALSE(r.ir.ops.empty());

    const auto& op = r.ir.ops.at(static_cast<std::size_t>(r.ir.root));
    EXPECT_EQ(op.type, FormExprType::AuxiliaryInputRef);
    EXPECT_EQ(op.imm0, static_cast<std::uint64_t>(slot));
}

TEST(KernelIRLowering, AuxiliaryOutputRefPreservesSlotIndex)
{
    const std::uint32_t slot = 11;
    const auto r = jit::lowerToKernelIR(FormExpr::auxiliaryOutputRef(slot));
    ASSERT_FALSE(r.ir.ops.empty());

    const auto& op = r.ir.ops.at(static_cast<std::size_t>(r.ir.root));
    EXPECT_EQ(op.type, FormExprType::AuxiliaryOutputRef);
    EXPECT_EQ(op.imm0, static_cast<std::uint64_t>(slot));
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

// ============================================================================
// Optimize: zero propagation (shape-aware)
// ============================================================================

TEST(KernelIROptimize, ConstantFoldingMultiply)
{
    // const * const → const
    const auto expr = FormExpr::constant(Real(3.0)) * FormExpr::constant(Real(4.0));
    auto r = jit::lowerToKernelIR(expr);
    const auto before = r.ir.opCount();
    const auto eliminated = r.ir.optimize();
    EXPECT_GT(eliminated, 0u);

    const auto& root_op = r.ir.ops[r.ir.root];
    EXPECT_EQ(root_op.type, FormExprType::Constant);
    EXPECT_DOUBLE_EQ(std::bit_cast<double>(root_op.imm0), 12.0);
}

TEST(KernelIROptimize, ConstantFoldingAdd)
{
    const auto expr = FormExpr::constant(Real(1.5)) + FormExpr::constant(Real(2.5));
    auto r = jit::lowerToKernelIR(expr);
    r.ir.optimize();

    const auto& root_op = r.ir.ops[r.ir.root];
    EXPECT_EQ(root_op.type, FormExprType::Constant);
    EXPECT_DOUBLE_EQ(std::bit_cast<double>(root_op.imm0), 4.0);
}

TEST(KernelIROptimize, IdentityMultiplyOneTimesX)
{
    // 1 * X → X
    const auto x = FormExpr::parameterRef(0);
    const auto expr = FormExpr::constant(Real(1.0)) * x;
    auto r = jit::lowerToKernelIR(expr);
    const auto before = r.ir.opCount();
    r.ir.optimize();

    // Root should now be ParameterRef directly (no Multiply)
    const auto& root_op = r.ir.ops[r.ir.root];
    EXPECT_EQ(root_op.type, FormExprType::ParameterRef);
    // DCE should remove the Constant(1.0) and Multiply ops
    EXPECT_LT(r.ir.opCount(), before);
}

TEST(KernelIROptimize, ZeroPlusXEliminated)
{
    // 0 + X → X
    const auto x = FormExpr::parameterRef(0);
    const auto expr = FormExpr::typedZero() + x;
    auto r = jit::lowerToKernelIR(expr);
    const auto before = r.ir.opCount();
    r.ir.optimize();

    const auto& root_op = r.ir.ops[r.ir.root];
    EXPECT_EQ(root_op.type, FormExprType::ParameterRef);
    EXPECT_LT(r.ir.opCount(), before);
}

TEST(KernelIROptimize, XMinusZeroEliminated)
{
    // X - 0 → X
    const auto x = FormExpr::parameterRef(0);
    const auto expr = x - FormExpr::typedZero();
    auto r = jit::lowerToKernelIR(expr);
    r.ir.optimize();

    const auto& root_op = r.ir.ops[r.ir.root];
    EXPECT_EQ(root_op.type, FormExprType::ParameterRef);
}

TEST(KernelIROptimize, InnerProductWithZeroIsScalarZero)
{
    // inner(0, X) → Constant(0.0)
    const auto x = FormExpr::parameterRef(0);
    const auto expr = inner(FormExpr::typedZero(), x);
    auto r = jit::lowerToKernelIR(expr);
    r.ir.optimize();

    const auto& root_op = r.ir.ops[r.ir.root];
    EXPECT_EQ(root_op.type, FormExprType::Constant);
    EXPECT_DOUBLE_EQ(std::bit_cast<double>(root_op.imm0), 0.0);
}

TEST(KernelIROptimize, GradientOfZeroPreservesGradientOp)
{
    // grad(0) should keep the Gradient op (shape-changing: scalar→vector)
    // NOT collapse to TypedZero
    const auto expr = grad(FormExpr::typedZero());
    auto r = jit::lowerToKernelIR(expr);
    r.ir.optimize();

    const auto& root_op = r.ir.ops[r.ir.root];
    EXPECT_EQ(root_op.type, FormExprType::Gradient)
        << "Gradient(0) must keep Gradient op for shape inference";
}

TEST(KernelIROptimize, NegateOfZeroCollapses)
{
    // -0 → TypedZero (shape-preserving, safe to collapse)
    const auto expr = -FormExpr::typedZero();
    auto r = jit::lowerToKernelIR(expr);
    r.ir.optimize();

    const auto& root_op = r.ir.ops[r.ir.root];
    EXPECT_EQ(root_op.type, FormExprType::TypedZero);
}

TEST(KernelIROptimize, TimeDerivativeOfZeroCollapses)
{
    // dt(0) → TypedZero (TimeDerivative is truly shape-preserving:
    // scalar→scalar for shapeless TypedZero input)
    const auto expr = dt(FormExpr::typedZero());
    auto r = jit::lowerToKernelIR(expr);
    r.ir.optimize();

    const auto& root_op = r.ir.ops[r.ir.root];
    EXPECT_EQ(root_op.type, FormExprType::TypedZero);
}

TEST(KernelIROptimize, RestrictMinusOfZeroKeepsOp)
{
    // minus(0) must keep the RestrictMinus node for shape context.
    const auto expr = minus(FormExpr::typedZero());
    auto r = jit::lowerToKernelIR(expr);
    r.ir.optimize();

    const auto& root_op = r.ir.ops[r.ir.root];
    EXPECT_EQ(root_op.type, FormExprType::RestrictMinus)
        << "RestrictMinus(0) must keep op for shape inference context";
}

TEST(KernelIROptimize, RestrictPlusOfZeroKeepsOp)
{
    // plus(0) must keep the RestrictPlus node for shape context.
    const auto expr = plus(FormExpr::typedZero());
    auto r = jit::lowerToKernelIR(expr);
    r.ir.optimize();

    const auto& root_op = r.ir.ops[r.ir.root];
    EXPECT_EQ(root_op.type, FormExprType::RestrictPlus)
        << "RestrictPlus(0) must keep op for shape inference context";
}

TEST(KernelIROptimize, JumpOfZeroKeepsOp)
{
    // jump(0) must keep the Jump node — collapsing to bare TypedZero
    // loses the rank context needed when the result is used in
    // vector/matrix operations downstream.
    const auto expr = jump(FormExpr::typedZero());
    auto r = jit::lowerToKernelIR(expr);
    r.ir.optimize();

    const auto& root_op = r.ir.ops[r.ir.root];
    EXPECT_EQ(root_op.type, FormExprType::Jump)
        << "Jump(0) must keep Jump op for shape inference context";
}

TEST(KernelIROptimize, AverageOfZeroKeepsOp)
{
    // avg(0) must keep the Average node for shape inference.
    const auto expr = avg(FormExpr::typedZero());
    auto r = jit::lowerToKernelIR(expr);
    r.ir.optimize();

    const auto& root_op = r.ir.ops[r.ir.root];
    EXPECT_EQ(root_op.type, FormExprType::Average)
        << "Average(0) must keep Average op for shape inference context";
}

TEST(KernelIROptimize, AsVectorOfZerosKeepsOp)
{
    // AsVector(0, 0, 0) should stay as AsVector, not collapse to TypedZero.
    // The op carries vector-rank information (3 components → 3-vector).
    const auto expr = FormExpr::asVector({
        FormExpr::typedZero(), FormExpr::typedZero(), FormExpr::typedZero()
    });
    auto r = jit::lowerToKernelIR(expr);
    r.ir.optimize();

    const auto& root_op = r.ir.ops[r.ir.root];
    EXPECT_EQ(root_op.type, FormExprType::AsVector)
        << "AsVector(0,0,0) must keep AsVector for shape (3-vector, not scalar)";
}

TEST(KernelIROptimize, OuterProductWithZeroKeepsOp)
{
    // outer(0, X) should keep OuterProduct op (shape-changing: loses dim info)
    const auto x = FormExpr::parameterRef(0);
    const auto expr = outer(FormExpr::typedZero(), x);
    auto r = jit::lowerToKernelIR(expr);
    r.ir.optimize();

    const auto& root_op = r.ir.ops[r.ir.root];
    EXPECT_EQ(root_op.type, FormExprType::OuterProduct)
        << "OuterProduct(0, X) must keep OuterProduct for shape inference";
}

TEST(KernelIROptimize, ComponentOfZeroIsScalarZero)
{
    // component(0, 0) → Constant(0.0) (always scalar result)
    const auto expr = component(FormExpr::typedZero(), 0);
    auto r = jit::lowerToKernelIR(expr);
    r.ir.optimize();

    const auto& root_op = r.ir.ops[r.ir.root];
    EXPECT_EQ(root_op.type, FormExprType::Constant);
    EXPECT_DOUBLE_EQ(std::bit_cast<double>(root_op.imm0), 0.0);
}

TEST(KernelIROptimize, PowerZeroExponentIsOne)
{
    // X^0 → 1
    const auto x = FormExpr::parameterRef(0);
    const auto expr = pow(x, FormExpr::constant(Real(0.0)));
    auto r = jit::lowerToKernelIR(expr);
    r.ir.optimize();

    const auto& root_op = r.ir.ops[r.ir.root];
    EXPECT_EQ(root_op.type, FormExprType::Constant);
    EXPECT_DOUBLE_EQ(std::bit_cast<double>(root_op.imm0), 1.0);
}

TEST(KernelIROptimize, DoubleNegationEliminated)
{
    // --X → X
    const auto x = FormExpr::parameterRef(0);
    const auto expr = -(-x);
    auto r = jit::lowerToKernelIR(expr);
    const auto before = r.ir.opCount();
    r.ir.optimize();

    const auto& root_op = r.ir.ops[r.ir.root];
    EXPECT_EQ(root_op.type, FormExprType::ParameterRef);
    EXPECT_LT(r.ir.opCount(), before);
}

// ============================================================================
// Optimize: post-rewrite CSE
// ============================================================================

TEST(KernelIROptimize, PostRewriteCSEMergesDuplicateConstants)
{
    // After folding, two subtrees that both produce Constant(0.0) should
    // be merged by CSE, reducing the reachable op count.
    const auto a = FormExpr::parameterRef(0);
    const auto b = FormExpr::parameterRef(1);
    // inner(0, a) → Constant(0.0) and inner(0, b) → Constant(0.0)
    // Adding them: 0 + 0 → Constant(0.0) after folding
    const auto expr = inner(FormExpr::typedZero(), a) + inner(FormExpr::typedZero(), b);
    auto r = jit::lowerToKernelIR(expr);
    r.ir.optimize();

    // Result should be a single Constant(0.0)
    const auto& root_op = r.ir.ops[r.ir.root];
    EXPECT_EQ(root_op.type, FormExprType::Constant);
    EXPECT_DOUBLE_EQ(std::bit_cast<double>(root_op.imm0), 0.0);
}

// ============================================================================
// Optimize: DCE
// ============================================================================

TEST(KernelIROptimize, DCERemovesUnreachableOps)
{
    // Build an expression where some subtrees become dead after folding.
    // 0 * (a + b + c) → 0, making the entire (a+b+c) subtree dead.
    // But with our shape-aware fix, 0*X is NOT collapsed — it stays as
    // Multiply(0, X).  DCE won't remove X since it's still referenced.
    // Instead test: 0 + X → X, making the 0 constant unreachable.
    const auto a = FormExpr::parameterRef(0);
    const auto expr = FormExpr::typedZero() + a;
    auto r = jit::lowerToKernelIR(expr);
    const auto before = r.ir.opCount();
    r.ir.optimize();

    // After 0+X → X aliasing + DCE, only the ParameterRef should remain.
    EXPECT_LT(r.ir.opCount(), before);
    EXPECT_EQ(r.ir.ops[r.ir.root].type, FormExprType::ParameterRef);
}

// ============================================================================
// SubtreeCosts
// ============================================================================

TEST(KernelIRCost, LeafCostIsOne)
{
    const auto expr = FormExpr::parameterRef(0);
    const auto r = jit::lowerToKernelIR(expr);
    const auto costs = r.ir.subtreeCosts();

    ASSERT_EQ(costs.size(), r.ir.opCount());
    EXPECT_EQ(costs[r.ir.root], 1u);
}

TEST(KernelIRCost, BinaryOpCostAddsChildren)
{
    const auto a = FormExpr::parameterRef(0);
    const auto b = FormExpr::parameterRef(1);
    const auto expr = a + b;
    const auto r = jit::lowerToKernelIR(expr);
    const auto costs = r.ir.subtreeCosts();

    // Root is Add: cost = cost(a) + cost(b) + 1 = 1 + 1 + 1 = 3
    EXPECT_EQ(costs[r.ir.root], 3u);
}

TEST(KernelIRCost, TranscendentalOpsInflated)
{
    // sqrt(x) should cost more than plain x+y due to libm call overhead.
    const auto x = FormExpr::parameterRef(0);
    const auto y = FormExpr::parameterRef(1);

    const auto sqrt_expr = sqrt(x);
    const auto add_expr = x + y;

    const auto r_sqrt = jit::lowerToKernelIR(sqrt_expr);
    const auto r_add = jit::lowerToKernelIR(add_expr);
    const auto costs_sqrt = r_sqrt.ir.subtreeCosts();
    const auto costs_add = r_add.ir.subtreeCosts();

    // sqrt(x): cost(x)=1 + 1(op) + 4(transcendental) = 6
    // x + y:   cost(x)=1 + cost(y)=1 + 1(op) = 3
    EXPECT_GT(costs_sqrt[r_sqrt.ir.root], costs_add[r_add.ir.root]);
}

// ============================================================================
// HardwareProfile
// ============================================================================

TEST(HardwareProfile, DiscoverReturnsReasonableDefaults)
{
    const auto& hp = jit::hardwareProfile();
    EXPECT_GT(hp.l1d.size_bytes, 0u);
    EXPECT_GT(hp.l1i.size_bytes, 0u);
    EXPECT_GT(hp.l2.size_bytes, 0u);
    EXPECT_GE(hp.simd_width_bytes, 16u);  // At least SSE2
}

TEST(HardwareProfile, DerivedBudgetsArePositive)
{
    const auto& hp = jit::hardwareProfile();
    EXPECT_GT(hp.qpCacheBudgetDoubles(), 0u);
    EXPECT_GT(hp.colocationTextBudgetBytes(), 0u);
    EXPECT_GT(hp.maxUnrollTripCount(), 0u);
    EXPECT_GT(hp.defaultTileSize(), 0u);
}

TEST(HardwareProfile, ProfitabilityThresholdsAreReasonable)
{
    const auto& hp = jit::hardwareProfile();
    // Trial-only caching thresholds
    EXPECT_GE(hp.trialOnlyMinSavings(), 4u);
    EXPECT_LE(hp.trialOnlyMinSavings(), 16u);
    EXPECT_GE(hp.trialOnlyMinOps(), 2u);
    EXPECT_LE(hp.trialOnlyMinOps(), 4u);
    // Cross-block CSE threshold
    EXPECT_GE(hp.crossBlockCostThreshold(), 2u);
    EXPECT_LE(hp.crossBlockCostThreshold(), 8u);
    // Default test DOF estimate
    EXPECT_GE(hp.defaultTestDofEstimate(), 2u);
    EXPECT_LE(hp.defaultTestDofEstimate(), 8u);
}

TEST(KernelIROptimize, ConstantFoldingThenCSEMergesEquivalentSums)
{
    // Build a+b and b+a where a and b are distinct constants.
    // Constant folding makes both 2+3→5 and 3+2→5, then CSE merges
    // the duplicate Constant(5) nodes.  The outer Add(5,5) folds to 10.
    const auto a = FormExpr::constant(2.0);
    const auto b = FormExpr::constant(3.0);
    const auto ab = a + b;
    const auto ba = b + a;
    const auto expr = ab + ba;  // Add( Add(2,3), Add(3,2) )

    auto r = jit::lowerToKernelIR(expr);
    const auto before = r.ir.opCount();
    r.ir.optimize();
    const auto after = r.ir.opCount();

    // After constant folding + CSE + DCE, everything collapses to
    // Constant(10.0).
    const auto& root_op = r.ir.ops[r.ir.root];
    EXPECT_EQ(root_op.type, FormExprType::Constant);
    EXPECT_LT(after, before) << "Constant folding + CSE should reduce op count";
}

TEST(HardwareProfile, CumulativeCacheBudgetIsPositive)
{
    const auto& hp = jit::hardwareProfile();
    EXPECT_GT(hp.qpCacheBudgetBytes(), 0u);
    // Budget should be <= L1d size
    EXPECT_LE(hp.qpCacheBudgetBytes(), hp.l1d.size_bytes);
}

TEST(HardwareProfile, BytesPerOpCalibrationDefaultsToStatic)
{
    // Before any samples, should return the static default.
    jit::BytesPerOpCalibration cal;
    EXPECT_EQ(cal.calibratedBytesPerOp(), jit::HardwareProfile::kBytesPerOp);
    // Custom fallback is honored when uncalibrated.
    EXPECT_EQ(cal.calibratedBytesPerOp(42u), 42u);
    // After one sample, still not enough (need kMinSamples=2)
    cal.recordSample(5800, 100);  // 58 bytes/op
    EXPECT_EQ(cal.calibratedBytesPerOp(), jit::HardwareProfile::kBytesPerOp);
    // After two samples, should return the measured average
    cal.recordSample(4200, 100);  // 42 bytes/op → average (5800+4200)/(100+100) = 50
    EXPECT_EQ(cal.calibratedBytesPerOp(), 50u);
    // Custom fallback is ignored once calibrated.
    EXPECT_EQ(cal.calibratedBytesPerOp(999u), 50u);
}

TEST(HardwareProfile, RawBytesPerOpCalibrationUsesCallerFallback)
{
    jit::BytesPerOpCalibration cal;
    EXPECT_EQ(cal.calibratedBytesPerOp(jit::HardwareProfile::kRawBytesPerOp),
              jit::HardwareProfile::kRawBytesPerOp);

    cal.recordSample(900, 100);   // 9 bytes/op, still uncalibrated
    EXPECT_EQ(cal.calibratedBytesPerOp(jit::HardwareProfile::kRawBytesPerOp),
              jit::HardwareProfile::kRawBytesPerOp);

    cal.recordSample(1100, 100);  // average 10 bytes/op
    EXPECT_EQ(cal.calibratedBytesPerOp(jit::HardwareProfile::kRawBytesPerOp), 10u);
}

TEST(HardwareProfile, ContinuousProfitabilityFormulas)
{
    using HP = jit::HardwareProfile;

    // 16KB L1d — tight cache
    HP hp16;
    hp16.l1d.size_bytes = 16u * 1024u;
    EXPECT_EQ(hp16.trialOnlyMinSavings(), 12u);
    EXPECT_EQ(hp16.trialOnlyMinOps(), 3u);
    EXPECT_EQ(hp16.crossBlockCostThreshold(), 6u);
    EXPECT_EQ(hp16.defaultTestDofEstimate(), 3u);

    // 32KB L1d — typical desktop
    HP hp32;
    hp32.l1d.size_bytes = 32u * 1024u;
    EXPECT_EQ(hp32.trialOnlyMinSavings(), 8u);
    EXPECT_EQ(hp32.trialOnlyMinOps(), 2u);
    EXPECT_EQ(hp32.crossBlockCostThreshold(), 4u);
    EXPECT_EQ(hp32.defaultTestDofEstimate(), 4u);

    // 48KB L1d — large cache
    HP hp48;
    hp48.l1d.size_bytes = 48u * 1024u;
    EXPECT_EQ(hp48.trialOnlyMinSavings(), 6u);   // raw=4, clamped to 6
    EXPECT_EQ(hp48.trialOnlyMinOps(), 2u);
    EXPECT_EQ(hp48.crossBlockCostThreshold(), 3u); // raw=2, clamped to 3

    // 64KB L1d — very large cache
    HP hp64;
    hp64.l1d.size_bytes = 64u * 1024u;
    EXPECT_EQ(hp64.trialOnlyMinSavings(), 6u);   // floor clamp
    EXPECT_EQ(hp64.crossBlockCostThreshold(), 3u);  // floor clamp
}

// ============================================================================
// Term-group planning
// ============================================================================

TEST(TermGroupPlanning, SingleTermFitsInBudget)
{
    // A single small term fits within budget — no split.
    std::vector<std::size_t> ops = {100};
    auto plan = jit::planTermGroups(ops, /*budget_bytes=*/24576, /*bytes_per_op=*/58);
    EXPECT_FALSE(plan.needs_split);
    ASSERT_EQ(plan.groups.size(), 1u);
    EXPECT_EQ(plan.groups[0].first_term, 0u);
    EXPECT_EQ(plan.groups[0].num_terms, 1u);
    EXPECT_EQ(plan.groups[0].estimated_text_bytes, 100u * 58u);
}

TEST(TermGroupPlanning, AllTermsFitInBudget_NoSplit)
{
    // 3 terms whose combined .text fits within budget.
    std::vector<std::size_t> ops = {50, 60, 70};
    const auto total = (50u + 60u + 70u) * 58u; // 10440 < 24576
    auto plan = jit::planTermGroups(ops, 24576, 58);
    EXPECT_FALSE(plan.needs_split);
    ASSERT_EQ(plan.groups.size(), 1u);
    EXPECT_EQ(plan.groups[0].num_terms, 3u);
    EXPECT_EQ(plan.total_estimated_bytes, total);
}

TEST(TermGroupPlanning, LargeBlockSplitsIntoGroups)
{
    // 4 terms, each ~200 ops × 58 bpo = 11600 bytes.
    // Budget = 24576. Terms 0+1 = 23200 fits, term 2 would exceed → split.
    std::vector<std::size_t> ops = {200, 200, 200, 200};
    auto plan = jit::planTermGroups(ops, 24576, 58);
    EXPECT_TRUE(plan.needs_split);
    ASSERT_GE(plan.groups.size(), 2u);

    // Verify all terms are covered contiguously.
    std::size_t total_terms = 0;
    for (std::size_t g = 0; g < plan.groups.size(); ++g) {
        EXPECT_EQ(plan.groups[g].first_term, total_terms);
        total_terms += plan.groups[g].num_terms;
        // Each group should fit within budget (except single oversized terms).
        EXPECT_GT(plan.groups[g].num_terms, 0u);
    }
    EXPECT_EQ(total_terms, 4u);
}

TEST(TermGroupPlanning, SingleOversizedTermGetsOwnGroup)
{
    // One term that exceeds the budget by itself.
    std::vector<std::size_t> ops = {50, 500, 50};
    auto plan = jit::planTermGroups(ops, 24576, 58);
    // 500 × 58 = 29000 > 24576, so it must get its own group.
    EXPECT_TRUE(plan.needs_split);

    // Find the group containing term 1 (the 500-op term).
    bool found_big_term = false;
    for (const auto& g : plan.groups) {
        if (g.first_term <= 1 && g.first_term + g.num_terms > 1) {
            found_big_term = true;
            // If the oversized term got its own group, num_terms == 1.
            if (g.estimated_text_bytes > 24576) {
                EXPECT_EQ(g.num_terms, 1u);
            }
        }
    }
    EXPECT_TRUE(found_big_term);
}

TEST(TermGroupPlanning, PreservesContiguousOrder)
{
    // Verify groups are contiguous and non-overlapping.
    std::vector<std::size_t> ops = {100, 100, 100, 100, 100, 100, 100, 100};
    auto plan = jit::planTermGroups(ops, 12000, 58); // 100×58=5800 per term
    EXPECT_TRUE(plan.needs_split);

    std::size_t expected_first = 0;
    for (const auto& g : plan.groups) {
        EXPECT_EQ(g.first_term, expected_first);
        EXPECT_GT(g.num_terms, 0u);
        expected_first += g.num_terms;
    }
    EXPECT_EQ(expected_first, 8u);
}

TEST(TermGroupPlanning, EmptyTermsReturnsNoSplit)
{
    std::vector<std::size_t> ops;
    auto plan = jit::planTermGroups(ops, 24576, 58);
    EXPECT_FALSE(plan.needs_split);
    EXPECT_TRUE(plan.groups.empty());
}

// ============================================================================
// Optimize: scalar-zero folds for contraction ops
// ============================================================================

TEST(KernelIROptimize, DivergenceOfZeroIsScalarZero)
{
    // div(0) → Constant(0.0)  (contraction: always scalar output)
    const auto expr = div(FormExpr::typedZero());
    auto r = jit::lowerToKernelIR(expr);
    r.ir.optimize();

    const auto& root_op = r.ir.ops[r.ir.root];
    EXPECT_EQ(root_op.type, FormExprType::Constant);
    EXPECT_DOUBLE_EQ(std::bit_cast<double>(root_op.imm0), 0.0);
}

TEST(KernelIROptimize, TraceOfZeroIsScalarZero)
{
    // trace(0) → Constant(0.0)  (sum of diagonal of zero matrix)
    const auto expr = trace(FormExpr::typedZero());
    auto r = jit::lowerToKernelIR(expr);
    r.ir.optimize();

    const auto& root_op = r.ir.ops[r.ir.root];
    EXPECT_EQ(root_op.type, FormExprType::Constant);
    EXPECT_DOUBLE_EQ(std::bit_cast<double>(root_op.imm0), 0.0);
}

TEST(KernelIROptimize, DeterminantOfZeroIsScalarZero)
{
    // det(0) → Constant(0.0)  (determinant of zero matrix)
    const auto expr = det(FormExpr::typedZero());
    auto r = jit::lowerToKernelIR(expr);
    r.ir.optimize();

    const auto& root_op = r.ir.ops[r.ir.root];
    EXPECT_EQ(root_op.type, FormExprType::Constant);
    EXPECT_DOUBLE_EQ(std::bit_cast<double>(root_op.imm0), 0.0);
}

// ============================================================================
// Optimize: expanded unary constant folding
// ============================================================================

TEST(KernelIROptimize, ConstantFoldingExp)
{
    const auto expr = exp(FormExpr::constant(Real(1.0)));
    auto r = jit::lowerToKernelIR(expr);
    r.ir.optimize();

    const auto& root_op = r.ir.ops[r.ir.root];
    EXPECT_EQ(root_op.type, FormExprType::Constant);
    const double expected_exp = std::exp(1.0);
    EXPECT_DOUBLE_EQ(std::bit_cast<double>(root_op.imm0), expected_exp);
}

TEST(KernelIROptimize, ConstantFoldingLog)
{
    const auto expr = log(FormExpr::constant(Real(2.0)));
    auto r = jit::lowerToKernelIR(expr);
    r.ir.optimize();

    const auto& root_op = r.ir.ops[r.ir.root];
    EXPECT_EQ(root_op.type, FormExprType::Constant);
    const double expected_log = std::log(2.0);
    EXPECT_DOUBLE_EQ(std::bit_cast<double>(root_op.imm0), expected_log);
}

TEST(KernelIROptimize, ConstantFoldingLogNegativeNotFolded)
{
    // log(-1) should NOT be folded (domain error)
    const auto expr = log(FormExpr::constant(Real(-1.0)));
    auto r = jit::lowerToKernelIR(expr);
    r.ir.optimize();

    const auto& root_op = r.ir.ops[r.ir.root];
    EXPECT_EQ(root_op.type, FormExprType::Log)
        << "log(negative) must not be constant-folded";
}

TEST(KernelIROptimize, ConstantFoldingSqrt)
{
    const auto expr = sqrt(FormExpr::constant(Real(4.0)));
    auto r = jit::lowerToKernelIR(expr);
    r.ir.optimize();

    const auto& root_op = r.ir.ops[r.ir.root];
    EXPECT_EQ(root_op.type, FormExprType::Constant);
    EXPECT_DOUBLE_EQ(std::bit_cast<double>(root_op.imm0), 2.0);
}

TEST(KernelIROptimize, ConstantFoldingSqrtZero)
{
    const auto expr = sqrt(FormExpr::constant(Real(0.0)));
    auto r = jit::lowerToKernelIR(expr);
    r.ir.optimize();

    const auto& root_op = r.ir.ops[r.ir.root];
    EXPECT_EQ(root_op.type, FormExprType::Constant);
    EXPECT_DOUBLE_EQ(std::bit_cast<double>(root_op.imm0), 0.0);
}

TEST(KernelIROptimize, ConstantFoldingAbs)
{
    const auto expr = abs(FormExpr::constant(Real(-3.5)));
    auto r = jit::lowerToKernelIR(expr);
    r.ir.optimize();

    const auto& root_op = r.ir.ops[r.ir.root];
    EXPECT_EQ(root_op.type, FormExprType::Constant);
    EXPECT_DOUBLE_EQ(std::bit_cast<double>(root_op.imm0), 3.5);
}

// ============================================================================
// Optimize: expanded binary constant folding
// ============================================================================

TEST(KernelIROptimize, ConstantFoldingMinimum)
{
    const auto expr = min(FormExpr::constant(Real(3.0)), FormExpr::constant(Real(5.0)));
    auto r = jit::lowerToKernelIR(expr);
    r.ir.optimize();

    const auto& root_op = r.ir.ops[r.ir.root];
    EXPECT_EQ(root_op.type, FormExprType::Constant);
    EXPECT_DOUBLE_EQ(std::bit_cast<double>(root_op.imm0), 3.0);
}

TEST(KernelIROptimize, ConstantFoldingMaximum)
{
    const auto expr = max(FormExpr::constant(Real(3.0)), FormExpr::constant(Real(5.0)));
    auto r = jit::lowerToKernelIR(expr);
    r.ir.optimize();

    const auto& root_op = r.ir.ops[r.ir.root];
    EXPECT_EQ(root_op.type, FormExprType::Constant);
    EXPECT_DOUBLE_EQ(std::bit_cast<double>(root_op.imm0), 5.0);
}

TEST(KernelIROptimize, ConstantFoldingLess)
{
    const auto a = FormExpr::constant(Real(2.0));
    const auto b = FormExpr::constant(Real(5.0));
    const auto expr = a.lt(b);
    auto r = jit::lowerToKernelIR(expr);
    r.ir.optimize();

    const auto& root_op = r.ir.ops[r.ir.root];
    EXPECT_EQ(root_op.type, FormExprType::Constant);
    EXPECT_DOUBLE_EQ(std::bit_cast<double>(root_op.imm0), 1.0);  // 2 < 5 → true
}

TEST(KernelIROptimize, ConstantFoldingGreaterFalse)
{
    const auto a = FormExpr::constant(Real(2.0));
    const auto b = FormExpr::constant(Real(5.0));
    const auto expr = a.gt(b);
    auto r = jit::lowerToKernelIR(expr);
    r.ir.optimize();

    const auto& root_op = r.ir.ops[r.ir.root];
    EXPECT_EQ(root_op.type, FormExprType::Constant);
    EXPECT_DOUBLE_EQ(std::bit_cast<double>(root_op.imm0), 0.0);  // 2 > 5 → false
}

// ============================================================================
// Optimize: Conditional with constant condition
// ============================================================================

TEST(KernelIROptimize, ConditionalTrueSelectsThenBranch)
{
    // conditional(1.0, a, b) → a  (condition > 0.0 → then branch)
    const auto cond = FormExpr::constant(Real(1.0));
    const auto a = FormExpr::parameterRef(0);
    const auto b = FormExpr::parameterRef(1);
    const auto expr = conditional(cond, a, b);
    auto r = jit::lowerToKernelIR(expr);
    r.ir.optimize();

    const auto& root_op = r.ir.ops[r.ir.root];
    EXPECT_EQ(root_op.type, FormExprType::ParameterRef);
    EXPECT_EQ(root_op.imm0, 0u);  // slot 0 = a
}

TEST(KernelIROptimize, ConditionalFalseSelectsElseBranch)
{
    // conditional(0.0, a, b) → b  (condition not > 0.0 → else branch)
    const auto cond = FormExpr::constant(Real(0.0));
    const auto a = FormExpr::parameterRef(0);
    const auto b = FormExpr::parameterRef(1);
    const auto expr = conditional(cond, a, b);
    auto r = jit::lowerToKernelIR(expr);
    r.ir.optimize();

    const auto& root_op = r.ir.ops[r.ir.root];
    EXPECT_EQ(root_op.type, FormExprType::ParameterRef);
    EXPECT_EQ(root_op.imm0, 1u);  // slot 1 = b
}

TEST(KernelIROptimize, ConditionalNegativeSelectsElseBranch)
{
    // conditional(-5.0, a, b) → b  (negative is not > 0.0 → else branch)
    const auto cond = FormExpr::constant(Real(-5.0));
    const auto a = FormExpr::parameterRef(0);
    const auto b = FormExpr::parameterRef(1);
    const auto expr = conditional(cond, a, b);
    auto r = jit::lowerToKernelIR(expr);
    r.ir.optimize();

    const auto& root_op = r.ir.ops[r.ir.root];
    EXPECT_EQ(root_op.type, FormExprType::ParameterRef);
    EXPECT_EQ(root_op.imm0, 1u);  // slot 1 = b
}

// ============================================================================
// Optimize: compound folds (constant folding + propagation in single pass)
// ============================================================================

TEST(KernelIROptimize, CompoundExpLogFolding)
{
    // exp(log(2.0)) → 2.0 via log(2.0)→const, exp(const)→const
    const auto expr = exp(log(FormExpr::constant(Real(2.0))));
    auto r = jit::lowerToKernelIR(expr);
    r.ir.optimize();

    const auto& root_op = r.ir.ops[r.ir.root];
    EXPECT_EQ(root_op.type, FormExprType::Constant);
    EXPECT_NEAR(std::bit_cast<double>(root_op.imm0), 2.0, 1e-14);
}

// ============================================================================
// HardwareProfile hash stability
// ============================================================================

TEST(HardwareProfile, StableHashDiffersForDifferentCaches)
{
    jit::HardwareProfile a;
    a.l1d.size_bytes = 32 * 1024;
    a.l1i.size_bytes = 32 * 1024;
    a.l2.size_bytes = 256 * 1024;
    a.l3.size_bytes = 8 * 1024 * 1024;
    a.simd_width_bytes = 16;

    jit::HardwareProfile b = a;
    b.l1i.size_bytes = 64 * 1024;  // different L1i

    EXPECT_NE(a.stableHash64(), b.stableHash64());
}

TEST(HardwareProfile, StableHashSameForIdenticalProfiles)
{
    jit::HardwareProfile a;
    a.l1d.size_bytes = 32 * 1024;
    a.l1i.size_bytes = 32 * 1024;
    a.l2.size_bytes = 256 * 1024;
    a.l3.size_bytes = 8 * 1024 * 1024;
    a.simd_width_bytes = 16;

    jit::HardwareProfile b = a;
    EXPECT_EQ(a.stableHash64(), b.stableHash64());
}

// ============================================================================
// Pass 1.3: Algebraic strength reduction tests
// ============================================================================

// --- Power specialization (Tier 1) ---

TEST(KernelIRStrengthReduction, PowerTwo_BecomesMultiply)
{
    const auto x = FormExpr::parameterRef(0);
    auto r = jit::lowerToKernelIR(pow(x, FormExpr::constant(Real(2.0))));
    r.ir.optimize();
    const auto& root = r.ir.ops[r.ir.root];
    EXPECT_EQ(root.type, FormExprType::Multiply);
    // Both children should be the same op (X * X)
    const auto c0 = r.ir.children[root.first_child];
    const auto c1 = r.ir.children[root.first_child + 1];
    EXPECT_EQ(c0, c1);
}

TEST(KernelIRStrengthReduction, PowerHalf_BecomesSqrt)
{
    const auto x = FormExpr::parameterRef(0);
    auto r = jit::lowerToKernelIR(pow(x, FormExpr::constant(Real(0.5))));
    r.ir.optimize();
    EXPECT_EQ(r.ir.ops[r.ir.root].type, FormExprType::Sqrt);
}

TEST(KernelIRStrengthReduction, PowerMinusOne_BecomesDivide)
{
    const auto x = FormExpr::parameterRef(0);
    auto r = jit::lowerToKernelIR(pow(x, FormExpr::constant(Real(-1.0))));
    r.ir.optimize();
    const auto& root = r.ir.ops[r.ir.root];
    EXPECT_EQ(root.type, FormExprType::Divide);
    // Numerator should be Constant(1.0)
    const auto num = r.ir.children[root.first_child];
    EXPECT_EQ(r.ir.ops[num].type, FormExprType::Constant);
    EXPECT_DOUBLE_EQ(std::bit_cast<double>(r.ir.ops[num].imm0), 1.0);
}

TEST(KernelIRStrengthReduction, PowerMinusHalf_BecomesDivSqrt)
{
    const auto x = FormExpr::parameterRef(0);
    auto r = jit::lowerToKernelIR(pow(x, FormExpr::constant(Real(-0.5))));
    ASSERT_FALSE(r.ir.ops.empty());
    r.ir.optimize();
    ASSERT_LT(r.ir.root, r.ir.ops.size());
    const auto& root = r.ir.ops[r.ir.root];
    EXPECT_EQ(root.type, FormExprType::Divide);
    if (root.type == FormExprType::Divide) {
        const auto den = r.ir.children[root.first_child + 1];
        EXPECT_EQ(r.ir.ops[den].type, FormExprType::Sqrt);
    }
}

TEST(KernelIRStrengthReduction, PowerThree_BecomesMulMul)
{
    const auto x = FormExpr::parameterRef(0);
    auto r = jit::lowerToKernelIR(pow(x, FormExpr::constant(Real(3.0))));
    r.ir.optimize();
    const auto& root = r.ir.ops[r.ir.root];
    EXPECT_EQ(root.type, FormExprType::Multiply);
    // One child should be Mul(X,X), the other should be X
}

TEST(KernelIRStrengthReduction, PowerFour_BecomesSqSq)
{
    const auto x = FormExpr::parameterRef(0);
    auto r = jit::lowerToKernelIR(pow(x, FormExpr::constant(Real(4.0))));
    r.ir.optimize();
    const auto& root = r.ir.ops[r.ir.root];
    EXPECT_EQ(root.type, FormExprType::Multiply);
    // Both children should be the same Mul(X,X) op
    const auto c0 = r.ir.children[root.first_child];
    const auto c1 = r.ir.children[root.first_child + 1];
    EXPECT_EQ(c0, c1);
    EXPECT_EQ(r.ir.ops[c0].type, FormExprType::Multiply);
}

TEST(KernelIRStrengthReduction, PowerMinusTwo_BecomesDivMulMul)
{
    const auto x = FormExpr::parameterRef(0);
    auto r = jit::lowerToKernelIR(pow(x, FormExpr::constant(Real(-2.0))));
    r.ir.optimize();
    const auto& root = r.ir.ops[r.ir.root];
    EXPECT_EQ(root.type, FormExprType::Divide);
}

TEST(KernelIRStrengthReduction, PowerMinusThree)
{
    const auto x = FormExpr::parameterRef(0);
    auto r = jit::lowerToKernelIR(pow(x, FormExpr::constant(Real(-3.0))));
    r.ir.optimize();
    EXPECT_EQ(r.ir.ops[r.ir.root].type, FormExprType::Divide);
}

TEST(KernelIRStrengthReduction, PowerMinusFour)
{
    const auto x = FormExpr::parameterRef(0);
    auto r = jit::lowerToKernelIR(pow(x, FormExpr::constant(Real(-4.0))));
    r.ir.optimize();
    EXPECT_EQ(r.ir.ops[r.ir.root].type, FormExprType::Divide);
}

// --- Multiply/Divide by -1 (Tier 2) ---

TEST(KernelIRStrengthReduction, MultiplyByMinusOne_Left_BecomesNegate)
{
    const auto x = FormExpr::parameterRef(0);
    auto r = jit::lowerToKernelIR(FormExpr::constant(Real(-1.0)) * x);
    r.ir.optimize();
    EXPECT_EQ(r.ir.ops[r.ir.root].type, FormExprType::Negate);
}

TEST(KernelIRStrengthReduction, MultiplyByMinusOne_Right_BecomesNegate)
{
    const auto x = FormExpr::parameterRef(0);
    auto r = jit::lowerToKernelIR(x * FormExpr::constant(Real(-1.0)));
    r.ir.optimize();
    EXPECT_EQ(r.ir.ops[r.ir.root].type, FormExprType::Negate);
}

TEST(KernelIRStrengthReduction, DivideByMinusOne_BecomesNegate)
{
    const auto x = FormExpr::parameterRef(0);
    auto r = jit::lowerToKernelIR(x / FormExpr::constant(Real(-1.0)));
    r.ir.optimize();
    EXPECT_EQ(r.ir.ops[r.ir.root].type, FormExprType::Negate);
}

// --- Divide by exact-reciprocal constant (Tier 2) ---

TEST(KernelIRStrengthReduction, DivideByTwo_BecomesMultiplyByHalf)
{
    const auto x = FormExpr::parameterRef(0);
    auto r = jit::lowerToKernelIR(x / FormExpr::constant(Real(2.0)));
    r.ir.optimize();
    const auto& root = r.ir.ops[r.ir.root];
    EXPECT_EQ(root.type, FormExprType::Multiply);
    // Second child should be Constant(0.5)
    const auto c1 = r.ir.children[root.first_child + 1];
    EXPECT_EQ(r.ir.ops[c1].type, FormExprType::Constant);
    EXPECT_DOUBLE_EQ(std::bit_cast<double>(r.ir.ops[c1].imm0), 0.5);
}

TEST(KernelIRStrengthReduction, DivideByThree_BecomesMultiply)
{
    // 3.0 * (1.0/3.0) == 1.0 in double precision (IEEE754 rounding),
    // so x/3 is rewritten to x*(1/3).
    const auto x = FormExpr::parameterRef(0);
    auto r = jit::lowerToKernelIR(x / FormExpr::constant(Real(3.0)));
    r.ir.optimize();
    EXPECT_EQ(r.ir.ops[r.ir.root].type, FormExprType::Multiply);
}

TEST(KernelIRStrengthReduction, DivideBy49_StaysDivide)
{
    // 49.0 * (1.0/49.0) != 1.0 in double precision, so should NOT be rewritten
    const auto x = FormExpr::parameterRef(0);
    auto r = jit::lowerToKernelIR(x / FormExpr::constant(Real(49.0)));
    r.ir.optimize();
    EXPECT_EQ(r.ir.ops[r.ir.root].type, FormExprType::Divide);
}

// --- Same-operand: NOT collapsed for NaN safety (Note 2) ---

TEST(KernelIRStrengthReduction, SubtractSelf_NotCollapsed)
{
    // Sub(X,X) is NOT collapsed to 0 because X could be NaN or inf
    const auto x = FormExpr::parameterRef(0);
    auto r = jit::lowerToKernelIR(x - x);
    r.ir.optimize();
    // Should remain as Subtract (or collapse via CSE to single ref, but NOT Constant(0))
    // With CSE, both children reference the same ParameterRef, but the Subtract stays.
    const auto& root = r.ir.ops[r.ir.root];
    EXPECT_NE(root.type, FormExprType::Constant);
}

TEST(KernelIRStrengthReduction, DivideSelf_NotCollapsed)
{
    // Div(X,X) is NOT collapsed to 1 because X could be 0 or NaN
    const auto x = FormExpr::parameterRef(0);
    auto r = jit::lowerToKernelIR(x / x);
    r.ir.optimize();
    const auto& root = r.ir.ops[r.ir.root];
    EXPECT_NE(root.type, FormExprType::Constant);
}

// --- Add(X, Negate(X)) is NOT collapsed ---
// KernelIR has no shape information; collapsing to TypedZero (unshaped)
// would lose tensor rank for vector/matrix operands.

TEST(KernelIRStrengthReduction, AddNegateX_NotCollapsed)
{
    const auto x = FormExpr::parameterRef(0);
    auto r = jit::lowerToKernelIR(x + (-x));
    r.ir.optimize();
    const auto& root = r.ir.ops[r.ir.root];
    // Should remain as Add (LLVM folds element-wise)
    EXPECT_EQ(root.type, FormExprType::Add);
}

// --- Tensor identities (Tier 4) ---
// These tests build KernelIR manually with a matrix-typed leaf
// (AsTensor) so the reductions are exercised on actual tensor ops,
// not just scalar placeholders.

namespace {
/// Build a 2x2 AsTensor IR leaf from 4 ParameterRefs, then wrap it
/// with the given unary op(s).  Returns the IR with root pointing to
/// the outermost op.
jit::KernelIR buildUnaryOnMatrix(std::initializer_list<FormExprType> chain)
{
    jit::KernelIR ir;
    // 4 scalar leaves: p0..p3
    for (int i = 0; i < 4; ++i) {
        jit::KernelIROp p{};
        p.type = FormExprType::ParameterRef;
        p.imm0 = static_cast<std::uint64_t>(i);
        ir.ops.push_back(p);
    }
    // AsTensor([p0,p1],[p2,p3]) — 4 children, imm0 encodes 2 rows
    jit::KernelIROp at{};
    at.type = FormExprType::AsTensor;
    at.first_child = static_cast<std::uint32_t>(ir.children.size());
    at.child_count = 4;
    at.imm0 = 2; // n_rows
    ir.ops.push_back(at);
    ir.children.insert(ir.children.end(), {0, 1, 2, 3});

    std::uint32_t prev = 4; // AsTensor index
    for (auto ty : chain) {
        jit::KernelIROp u{};
        u.type = ty;
        u.first_child = static_cast<std::uint32_t>(ir.children.size());
        u.child_count = 1;
        ir.ops.push_back(u);
        ir.children.push_back(prev);
        prev = static_cast<std::uint32_t>(ir.ops.size() - 1);
    }
    ir.root = prev;
    return ir;
}
} // namespace

TEST(KernelIRStrengthReduction, TransposeTranspose_Eliminated)
{
    auto ir = buildUnaryOnMatrix({FormExprType::Transpose, FormExprType::Transpose});
    ir.optimize();
    // Root should be the AsTensor (both Transposes eliminated)
    EXPECT_EQ(ir.ops[ir.root].type, FormExprType::AsTensor);
}

TEST(KernelIRStrengthReduction, SymPartIdempotent)
{
    auto ir = buildUnaryOnMatrix({FormExprType::SymmetricPart, FormExprType::SymmetricPart});
    ir.optimize();
    const auto& root = ir.ops[ir.root];
    EXPECT_EQ(root.type, FormExprType::SymmetricPart);
    const auto child = ir.children[root.first_child];
    EXPECT_EQ(ir.ops[child].type, FormExprType::AsTensor);
}

TEST(KernelIRStrengthReduction, SymOfTranspose_DropTranspose)
{
    auto ir = buildUnaryOnMatrix({FormExprType::Transpose, FormExprType::SymmetricPart});
    ir.optimize();
    const auto& root = ir.ops[ir.root];
    EXPECT_EQ(root.type, FormExprType::SymmetricPart);
    const auto child = ir.children[root.first_child];
    EXPECT_EQ(ir.ops[child].type, FormExprType::AsTensor);
}

TEST(KernelIRStrengthReduction, TraceOfTranspose_DropTranspose)
{
    auto ir = buildUnaryOnMatrix({FormExprType::Transpose, FormExprType::Trace});
    ir.optimize();
    const auto& root = ir.ops[ir.root];
    EXPECT_EQ(root.type, FormExprType::Trace);
    const auto child = ir.children[root.first_child];
    EXPECT_EQ(ir.ops[child].type, FormExprType::AsTensor);
}

TEST(KernelIRStrengthReduction, TraceOfSymPart_DropSymPart)
{
    auto ir = buildUnaryOnMatrix({FormExprType::SymmetricPart, FormExprType::Trace});
    ir.optimize();
    const auto& root = ir.ops[ir.root];
    EXPECT_EQ(root.type, FormExprType::Trace);
    const auto child = ir.children[root.first_child];
    EXPECT_EQ(ir.ops[child].type, FormExprType::AsTensor);
}

TEST(KernelIRStrengthReduction, DetOfTranspose_DropTranspose)
{
    auto ir = buildUnaryOnMatrix({FormExprType::Transpose, FormExprType::Determinant});
    ir.optimize();
    const auto& root = ir.ops[ir.root];
    EXPECT_EQ(root.type, FormExprType::Determinant);
    const auto child = ir.children[root.first_child];
    EXPECT_EQ(ir.ops[child].type, FormExprType::AsTensor);
}

TEST(KernelIRStrengthReduction, NegateSubtract_SwapsOperands)
{
    const auto a = FormExpr::parameterRef(0);
    const auto b = FormExpr::parameterRef(1);
    auto r = jit::lowerToKernelIR(-(a - b));
    r.ir.optimize();
    const auto& root = r.ir.ops[r.ir.root];
    EXPECT_EQ(root.type, FormExprType::Subtract);
    // Children should be (b, a), not (a, b)
    const auto c0 = r.ir.children[root.first_child];
    const auto c1 = r.ir.children[root.first_child + 1];
    EXPECT_EQ(r.ir.ops[c0].imm0, 1u); // ParameterRef(1) = b
    EXPECT_EQ(r.ir.ops[c1].imm0, 0u); // ParameterRef(0) = a
}

// --- Compound rewrite via second sweep (Note 3) ---

TEST(KernelIRStrengthReduction, DivideByMinusOne_CompoundToNegate)
{
    // x / (-1) should become Negate(x) directly (not Mul(x, -1))
    const auto x = FormExpr::parameterRef(0);
    auto r = jit::lowerToKernelIR(x / FormExpr::constant(Real(-1.0)));
    r.ir.optimize();
    EXPECT_EQ(r.ir.ops[r.ir.root].type, FormExprType::Negate);
}

// --- Inverse(Inverse) NOT collapsed (Note 2, FP safety) ---

TEST(KernelIRStrengthReduction, InverseInverse_NotCollapsed)
{
    // Inverse(Inverse(X)) is NOT collapsed because Inverse of a singular
    // matrix returns inf/NaN, and Inverse(inf) ≠ X.
    // Build the IR manually: Inverse(Inverse(ParameterRef(0)))
    jit::KernelIR ir;
    jit::KernelIROp param{};
    param.type = FormExprType::ParameterRef;
    param.imm0 = 0;
    ir.ops.push_back(param);

    jit::KernelIROp inv1{};
    inv1.type = FormExprType::Inverse;
    inv1.first_child = static_cast<std::uint32_t>(ir.children.size());
    inv1.child_count = 1;
    ir.ops.push_back(inv1);
    ir.children.push_back(0); // child = param

    jit::KernelIROp inv2{};
    inv2.type = FormExprType::Inverse;
    inv2.first_child = static_cast<std::uint32_t>(ir.children.size());
    inv2.child_count = 1;
    ir.ops.push_back(inv2);
    ir.children.push_back(1); // child = inv1

    ir.root = 2;
    ir.optimize();

    // Should remain as Inverse (NOT collapsed to ParameterRef)
    EXPECT_EQ(ir.ops[ir.root].type, FormExprType::Inverse);
}

// --- Factor extraction (Pass 1.5) ---

TEST(KernelIRStrengthReduction, ConstantFactorExtraction_BasicAdd)
{
    // C*a + C*b → C*(a+b) when C is a constant
    const auto a = FormExpr::parameterRef(0);
    const auto b = FormExpr::parameterRef(1);
    const auto c = FormExpr::constant(Real(3.14));
    auto r = jit::lowerToKernelIR(c * a + c * b);
    const auto before = r.ir.opCount();
    r.ir.optimize();
    // Should have fewer ops (one Multiply eliminated)
    EXPECT_LT(r.ir.opCount(), before);
    // Root should be Multiply(C, Add(a, b))
    const auto& root = r.ir.ops[r.ir.root];
    EXPECT_EQ(root.type, FormExprType::Multiply);
}

TEST(KernelIRStrengthReduction, FactorExtraction_SingleUseOnly)
{
    // C*a + C*b where C*a is also used elsewhere — should NOT factor
    const auto a = FormExpr::parameterRef(0);
    const auto b = FormExpr::parameterRef(1);
    const auto c = FormExpr::constant(Real(2.0));
    const auto ca = c * a;
    // ca used twice: in the Add and in an outer Multiply
    auto r = jit::lowerToKernelIR((ca + c * b) * ca);
    r.ir.optimize();
    // Root should still be Multiply (factoring should NOT have happened
    // because ca is multi-use)
    EXPECT_EQ(r.ir.ops[r.ir.root].type, FormExprType::Multiply);
}

// --- kBytesPerOp updated default ---

TEST(HardwareProfile, DefaultBytesPerOpIs300)
{
    EXPECT_EQ(jit::HardwareProfile::kBytesPerOp, 300u);
}

} // namespace test
} // namespace forms
} // namespace FE
} // namespace svmp

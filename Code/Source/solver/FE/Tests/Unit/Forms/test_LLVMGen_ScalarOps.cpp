/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Tests/Unit/Forms/RuntimeScalarFunctionalTest.h"

namespace svmp::FE::forms::test {

class LLVMGenScalarOps : public RuntimeScalarFunctionalTest {};

TEST_F(LLVMGenScalarOps, PowerComputesExponentiation)
{
    expectSamples(a.pow(b), {{{2, 3}, 8}, {{4, 0.5}, 2}, {{2, -3}, 0.125}});
}

TEST_F(LLVMGenScalarOps, MinimumSelectsSmaller)
{
    expectSamples(a.min(b), {{{-1, 2}, -1}, {{4, -3}, -3}, {{2, 2}, 2}});
}

TEST_F(LLVMGenScalarOps, MaximumSelectsLarger)
{
    expectSamples(a.max(b), {{{-1, 2}, 2}, {{4, -3}, 4}, {{2, 2}, 2}});
}

TEST_F(LLVMGenScalarOps, ConditionalSelectsCorrectBranch)
{
    expectSamples(a.lt(b).conditional(a + b, a - b), {{{1, 2}, 3}, {{3, 1}, 2}, {{2, 2}, 0}});
}

TEST_F(LLVMGenScalarOps, AbsoluteValueMatchesInterpreter)
{
    expectSamples(a.abs(), {{{-2, 0}, 2}, {{1.5, 0}, 1.5}, {{0, 0}, 0}});
}

TEST_F(LLVMGenScalarOps, SignMatchesInterpreter)
{
    expectSamples(a.sign(), {{{-2, 0}, -1}, {{1.5, 0}, 1}, {{0, 0}, 0}});
}

TEST_F(LLVMGenScalarOps, SqrtMatchesInterpreter)
{
    expectSamples(a.sqrt(), {{{0, 0}, 0}, {{4, 0}, 2}, {{2, 0}, std::sqrt(2.0)}});
}

TEST_F(LLVMGenScalarOps, ExpMatchesInterpreter)
{
    expectSamples(a.exp(), {{{-0.5, 0}, std::exp(-0.5)}, {{2, 0}, std::exp(2.0)}});
}

TEST_F(LLVMGenScalarOps, LogMatchesInterpreter)
{
    expectSamples(a.log(), {{{0.5, 0}, std::log(0.5)}, {{1, 0}, 0}, {{2, 0}, std::log(2.0)}});
}

TEST_F(LLVMGenScalarOps, ComparisonOperatorsReturnCorrectValues)
{
    // A distinct bit for every predicate detects swapped or canceled comparisons.
    const auto mask = a.lt(b) + 2 * a.le(b) + 4 * a.gt(b) +
                      8 * a.ge(b) + 16 * a.eq(b) + 32 * a.ne(b);
    expectSamples(mask, {{{1, 2}, 35}, {{2, 2}, 26}, {{3, 2}, 44}});
}

} // namespace svmp::FE::forms::test

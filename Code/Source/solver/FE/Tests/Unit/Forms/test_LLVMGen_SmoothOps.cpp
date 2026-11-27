/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Tests/Unit/Forms/RuntimeScalarFunctionalTest.h"

namespace svmp::FE::forms::test {

class LLVMGenSmoothOps : public RuntimeScalarFunctionalTest {};

namespace {
constexpr Real epsilon = 0.01;
}

TEST_F(LLVMGenSmoothOps, SmoothAbsoluteValueApproximatesAbs)
{
    const auto radius = std::hypot(2.0, epsilon);
    expectSamples(a.smoothAbs(b), {{{-2, epsilon}, radius}, {{0, epsilon}, epsilon}, {{2, epsilon}, radius}});
}

TEST_F(LLVMGenSmoothOps, SmoothSignApproximatesSign)
{
    const Real sign = 2.0 / std::hypot(2.0, epsilon);
    expectSamples(a.smoothSign(b), {{{-2, epsilon}, -sign}, {{0, epsilon}, 0}, {{2, epsilon}, sign}});
}

TEST_F(LLVMGenSmoothOps, SmoothHeavisideApproximatesStep)
{
    const Real sign = 2.0 / std::hypot(2.0, epsilon);
    expectSamples(a.smoothHeaviside(b),
                  {{{-2, epsilon}, (1 - sign) / 2}, {{0, epsilon}, 0.5}, {{2, epsilon}, (1 + sign) / 2}});
}

TEST_F(LLVMGenSmoothOps, SmoothMinApproximatesMinWithEpsilon)
{
    const Real expected = (3 - std::hypot(1.0, epsilon)) / 2;
    expectSamples(a.smoothMin(b, FormExpr::constant(epsilon)),
                  {{{1, 2}, expected}, {{2, 1}, expected}, {{1, 1}, 1 - epsilon / 2}});
}

TEST_F(LLVMGenSmoothOps, SmoothMaxApproximatesMaxWithEpsilon)
{
    const Real expected = (3 + std::hypot(1.0, epsilon)) / 2;
    expectSamples(a.smoothMax(b, FormExpr::constant(epsilon)),
                  {{{1, 2}, expected}, {{2, 1}, expected}, {{1, 1}, 1 + epsilon / 2}});
}

} // namespace svmp::FE::forms::test

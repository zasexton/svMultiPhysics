/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include <gtest/gtest.h>

#include "FE/Interfaces/detail/ProducerArithmeticAssessment.h"

#include <boost/multiprecision/cpp_int.hpp>

#include <bit>
#include <cerrno>
#include <cfenv>
#include <cmath>
#include <cstdint>
#include <limits>
#include <thread>

namespace {

using namespace svmp::FE::interfaces::detail;
using svmp::FE::Real;
using boost::multiprecision::cpp_int;

constexpr std::uint64_t sign_mask = UINT64_C(0x8000000000000000);
constexpr std::uint64_t fraction_mask = UINT64_C(0x000fffffffffffff);

std::uint64_t bits(Real value)
{
    return std::bit_cast<std::uint64_t>(value);
}

Real fromBits(std::uint64_t value)
{
    return std::bit_cast<Real>(value);
}

cpp_int independentlyScaledMagnitude(Real value)
{
    const auto magnitude = bits(value) & ~sign_mask;
    const auto exponent = magnitude >> 52u;
    const auto fraction = magnitude & fraction_mask;
    if (exponent == 0u) {
        return cpp_int(fraction);
    }
    return (cpp_int(UINT64_C(1) << 52u) + fraction)
           << static_cast<unsigned>(exponent - 1u);
}

void expectProductInequality(Real left,
                             Real right,
                             const ArithmeticInterval& bracket)
{
    const cpp_int exact = independentlyScaledMagnitude(left) *
                          independentlyScaledMagnitude(right);
    const cpp_int lower = independentlyScaledMagnitude(bracket.lower) << 1074u;
    const cpp_int upper = independentlyScaledMagnitude(bracket.upper) << 1074u;
    EXPECT_LE(lower, exact);
    EXPECT_GE(upper, exact);
}

void expectDivisionInequality(Real numerator,
                              Real denominator,
                              const ArithmeticInterval& bracket)
{
    const cpp_int exact = independentlyScaledMagnitude(numerator) << 1074u;
    const cpp_int denominator_integer =
        independentlyScaledMagnitude(denominator);
    const cpp_int lower = independentlyScaledMagnitude(bracket.lower) *
                          denominator_integer;
    const cpp_int upper = independentlyScaledMagnitude(bracket.upper) *
                          denominator_integer;
    EXPECT_LE(lower, exact);
    EXPECT_GE(upper, exact);
}

void expectSqrtInequality(Real input,
                          const ArithmeticInterval& bracket)
{
    const cpp_int exact = independentlyScaledMagnitude(input) << 1074u;
    const cpp_int lower_magnitude =
        independentlyScaledMagnitude(bracket.lower);
    const cpp_int upper_magnitude =
        independentlyScaledMagnitude(bracket.upper);
    EXPECT_LE(lower_magnitude * lower_magnitude, exact);
    EXPECT_GE(upper_magnitude * upper_magnitude, exact);
}

struct HardwareState {
    std::uint32_t mxcsr{0u};
    std::uint16_t x87_control{0u};
    std::uint16_t x87_status{0u};
};

HardwareState readHardwareState()
{
    HardwareState state;
#if defined(__x86_64__)
    asm volatile("stmxcsr %0" : "=m"(state.mxcsr));
    asm volatile("fnstcw %0" : "=m"(state.x87_control));
    asm volatile("fnstsw %0" : "=am"(state.x87_status));
#endif
    return state;
}

void writeMxcsr(std::uint32_t value)
{
#if defined(__x86_64__)
    asm volatile("ldmxcsr %0" : : "m"(value));
#else
    static_cast<void>(value);
#endif
}

void writeX87Control(std::uint16_t value)
{
#if defined(__x86_64__)
    asm volatile("fldcw %0" : : "m"(value));
#else
    static_cast<void>(value);
#endif
}

void clearX87Status()
{
#if defined(__x86_64__)
    asm volatile("fnclex");
#endif
}

void expectHardwareStateEqual(const HardwareState& actual,
                              const HardwareState& expected)
{
    EXPECT_EQ(actual.mxcsr, expected.mxcsr);
    EXPECT_EQ(actual.x87_control, expected.x87_control);
    EXPECT_EQ(actual.x87_status, expected.x87_status);
}

void expectBits(const ArithmeticInterval& interval,
                std::uint64_t lower,
                std::uint64_t upper)
{
    EXPECT_EQ(bits(interval.lower), lower);
    EXPECT_EQ(bits(interval.upper), upper);
}

Real normalWithExponentAndFraction(int exponent, std::uint64_t fraction)
{
    return fromBits((static_cast<std::uint64_t>(exponent + 1023) << 52u) |
                    fraction);
}

void expectTightDivisionBracket(const IntervalAssessment& result,
                                Real numerator,
                                Real denominator)
{
    ASSERT_TRUE(result.available());
    const bool negative = ((bits(numerator) ^ bits(denominator)) &
                           sign_mask) != 0u;
    const auto lower_bits = negative
                                ? bits(result.interval.upper) & ~sign_mask
                                : bits(result.interval.lower);
    const auto upper_bits = negative
                                ? bits(result.interval.lower) & ~sign_mask
                                : bits(result.interval.upper);
    if (negative) {
        EXPECT_NE(bits(result.interval.lower) & sign_mask, 0u);
        EXPECT_NE(bits(result.interval.upper) & sign_mask, 0u);
    } else {
        EXPECT_EQ(bits(result.interval.lower) & sign_mask, 0u);
        EXPECT_EQ(bits(result.interval.upper) & sign_mask, 0u);
    }

    const cpp_int exact = independentlyScaledMagnitude(numerator) << 1074u;
    const cpp_int denominator_integer =
        independentlyScaledMagnitude(denominator);
    const cpp_int lower = independentlyScaledMagnitude(fromBits(lower_bits)) *
                          denominator_integer;
    const cpp_int upper = independentlyScaledMagnitude(fromBits(upper_bits)) *
                          denominator_integer;
    EXPECT_LE(lower, exact);
    EXPECT_GE(upper, exact);
    if (lower_bits == upper_bits) {
        EXPECT_EQ(lower, exact);
        return;
    }

    EXPECT_EQ(upper_bits, lower_bits + 1u);
    EXPECT_LT(lower, exact);
    EXPECT_GT(upper, exact);
    for (const auto endpoint : {lower_bits, upper_bits}) {
        const auto exponent = endpoint >> 52u;
        EXPECT_NE(exponent, 0u);
        EXPECT_NE(exponent, UINT64_C(0x7ff));
    }
}

void expectDivisionRangeFailure(const IntervalAssessment& result,
                                Real numerator,
                                Real denominator)
{
    EXPECT_FALSE(result.available());
    EXPECT_EQ(result.failure, ArithmeticFailure::ArithmeticRange);
    expectBits(result.interval, 0u, 0u);
    const cpp_int exact = independentlyScaledMagnitude(numerator) << 1074u;
    const cpp_int denominator_integer =
        independentlyScaledMagnitude(denominator);
    const cpp_int minimum = independentlyScaledMagnitude(
                                fromBits(UINT64_C(0x0010000000000000))) *
                            denominator_integer;
    const cpp_int maximum = independentlyScaledMagnitude(
                                fromBits(UINT64_C(0x7fefffffffffffff))) *
                            denominator_integer;
    EXPECT_TRUE(exact < minimum || exact > maximum);
}

void expectTightSqrtBracket(const IntervalAssessment& result, Real input)
{
    ASSERT_TRUE(result.available());
    const auto lower_bits = bits(result.interval.lower);
    const auto upper_bits = bits(result.interval.upper);
    EXPECT_EQ(lower_bits & sign_mask, 0u);
    EXPECT_EQ(upper_bits & sign_mask, 0u);

    const cpp_int exact = independentlyScaledMagnitude(input) << 1074u;
    const cpp_int lower = independentlyScaledMagnitude(result.interval.lower);
    const cpp_int upper = independentlyScaledMagnitude(result.interval.upper);
    EXPECT_LE(lower * lower, exact);
    EXPECT_GE(upper * upper, exact);
    if (lower_bits == upper_bits) {
        EXPECT_EQ(lower * lower, exact);
        return;
    }

    EXPECT_EQ(upper_bits, lower_bits + 1u);
    EXPECT_LT(lower * lower, exact);
    EXPECT_GT(upper * upper, exact);
    for (const auto endpoint : {lower_bits, upper_bits}) {
        const auto exponent = endpoint >> 52u;
        EXPECT_NE(exponent, 0u);
        EXPECT_NE(exponent, UINT64_C(0x7ff));
    }
}

void expectTightDyadicBracket(const IntervalAssessment& result,
                              const cpp_int& exact,
                              unsigned scale,
                              bool negative = false)
{
    ASSERT_TRUE(result.available());
    ASSERT_GE(scale, 1074u);
    ASSERT_GT(exact, 0);

    const auto lower_bits = negative
                                ? bits(result.interval.upper) & ~sign_mask
                                : bits(result.interval.lower);
    const auto upper_bits = negative
                                ? bits(result.interval.lower) & ~sign_mask
                                : bits(result.interval.upper);
    if (negative) {
        EXPECT_NE(bits(result.interval.lower) & sign_mask, 0u);
        EXPECT_NE(bits(result.interval.upper) & sign_mask, 0u);
    } else {
        EXPECT_EQ(lower_bits & sign_mask, 0u);
        EXPECT_EQ(upper_bits & sign_mask, 0u);
    }

    const cpp_int lower = independentlyScaledMagnitude(fromBits(lower_bits))
                          << (scale - 1074u);
    const cpp_int upper = independentlyScaledMagnitude(fromBits(upper_bits))
                          << (scale - 1074u);
    EXPECT_LE(lower, exact);
    EXPECT_GE(upper, exact);

    if (lower_bits == upper_bits) {
        EXPECT_EQ(lower, exact);
        return;
    }

    EXPECT_EQ(upper_bits, lower_bits + 1u);
    EXPECT_LT(lower, exact);
    EXPECT_GT(upper, exact);
    for (const auto endpoint : {lower_bits, upper_bits}) {
        const auto exponent = endpoint >> 52u;
        EXPECT_NE(exponent, 0u);
        EXPECT_NE(exponent, UINT64_C(0x7ff));
    }
}

void expectPositiveZeroSingleton(const IntervalAssessment& result)
{
    ASSERT_TRUE(result.available());
    expectBits(result.interval, 0u, 0u);
}

void expectDyadicRangeFailure(const IntervalAssessment& result,
                              const cpp_int& exact,
                              unsigned scale)
{
    EXPECT_FALSE(result.available());
    EXPECT_EQ(result.failure, ArithmeticFailure::ArithmeticRange);
    const cpp_int minimum =
        independentlyScaledMagnitude(fromBits(UINT64_C(0x0010000000000000)))
        << (scale - 1074u);
    const cpp_int maximum =
        independentlyScaledMagnitude(fromBits(UINT64_C(0x7fefffffffffffff)))
        << (scale - 1074u);
    EXPECT_TRUE(exact > 0 && (exact < minimum || exact > maximum));
}

TEST(ProducerArithmeticAssessment, DivideRoundsOneFifthOutward)
{
    const auto quotient = assessIntervalOperation(
        IntervalOperation::Divide, {1.0, 1.0}, {5.0, 5.0});
    ASSERT_TRUE(quotient.available());
    expectBits(quotient.interval,
               UINT64_C(0x3fc9999999999999),
               UINT64_C(0x3fc999999999999a));
}

TEST(ProducerArithmeticAssessment, SqrtTwoUsesTheCorrectSides)
{
    const auto irrational = assessIntervalSqrt({2.0, 2.0});
    ASSERT_TRUE(irrational.available());
    expectBits(irrational.interval,
               UINT64_C(0x3ff6a09e667f3bcc),
               UINT64_C(0x3ff6a09e667f3bcd));
}

TEST(ProducerArithmeticAssessment, ExactZeroAndIdentitiesStaySingletons)
{
    const auto sum = assessIntervalOperation(
        IntervalOperation::Add, {-0.0, 0.0}, {1.0, 1.0});
    ASSERT_TRUE(sum.available());
    expectBits(sum.interval,
               UINT64_C(0x3ff0000000000000),
               UINT64_C(0x3ff0000000000000));

    const auto product = assessIntervalOperation(
        IntervalOperation::Multiply, {-1.0, -1.0}, {1.0, 1.0});
    ASSERT_TRUE(product.available());
    expectBits(product.interval,
               UINT64_C(0xbff0000000000000),
               UINT64_C(0xbff0000000000000));
}

TEST(ProducerArithmeticAssessment, SubtractionPreservesIntervalUncertainty)
{
    const auto difference = assessIntervalOperation(
        IntervalOperation::Subtract, {1.0, 2.0}, {1.0, 2.0});
    ASSERT_TRUE(difference.available());
    expectBits(difference.interval,
               UINT64_C(0xbff0000000000000),
               UINT64_C(0x3ff0000000000000));
}

TEST(ProducerArithmeticAssessment, SignedEndpointAdditionUsesRealOrder)
{
    const auto sum = assessIntervalOperation(
        IntervalOperation::Add, {-2.0, -1.0}, {0.5, 1.0});
    ASSERT_TRUE(sum.available());
    expectBits(sum.interval,
               UINT64_C(0xbff8000000000000),
               UINT64_C(0x0000000000000000));
}

TEST(ProducerArithmeticAssessment, DivisionRejectsDenominatorContainingZero)
{
    const auto quotient = assessIntervalOperation(
        IntervalOperation::Divide, {1.0, 1.0}, {-1.0, 1.0});
    EXPECT_FALSE(quotient.available());
    EXPECT_EQ(quotient.failure, ArithmeticFailure::UnresolvedDenominator);
}

TEST(ProducerArithmeticAssessment, SquareIsSignAware)
{
    const auto crossing = assessIntervalSquare({-2.0, 3.0});
    ASSERT_TRUE(crossing.available());
    expectBits(crossing.interval,
               UINT64_C(0x0000000000000000),
               UINT64_C(0x4022000000000000));

    const auto negative = assessIntervalSquare({-3.0, -2.0});
    ASSERT_TRUE(negative.available());
    expectBits(negative.interval,
               UINT64_C(0x4010000000000000),
               UINT64_C(0x4022000000000000));
}

TEST(ProducerArithmeticAssessment, InvalidOrderingAndNonfiniteBitsReject)
{
    const auto reversed = assessIntervalOperation(
        IntervalOperation::Add, {2.0, 1.0}, {0.0, 0.0});
    EXPECT_FALSE(reversed.available());
    EXPECT_EQ(reversed.failure, ArithmeticFailure::InvalidInput);

    for (const auto nonfinite : {
             UINT64_C(0x7ff0000000000000),
             UINT64_C(0x7ff8000000000001),
             UINT64_C(0x7ff0000000000001),
             UINT64_C(0xfff8000000000001)}) {
        const auto result = assessIntervalSqrt(
            {fromBits(nonfinite), fromBits(nonfinite)});
        EXPECT_FALSE(result.available());
        EXPECT_EQ(result.failure, ArithmeticFailure::InvalidInput);
    }
}

TEST(ProducerArithmeticAssessment, NormalBoundariesPassAndRangesReject)
{
    const auto minimum_normal = fromBits(UINT64_C(0x0010000000000000));
    const auto maximum_finite = fromBits(UINT64_C(0x7fefffffffffffff));
    const auto subnormal = fromBits(UINT64_C(0x0000000000000001));

    const auto exact_minimum = assessIntervalOperation(
        IntervalOperation::Multiply,
        {minimum_normal, minimum_normal},
        {1.0, 1.0});
    ASSERT_TRUE(exact_minimum.available());
    expectBits(exact_minimum.interval,
               UINT64_C(0x0010000000000000),
               UINT64_C(0x0010000000000000));

    const auto exact_maximum = assessIntervalOperation(
        IntervalOperation::Multiply,
        {maximum_finite, maximum_finite},
        {1.0, 1.0});
    ASSERT_TRUE(exact_maximum.available());
    expectBits(exact_maximum.interval,
               UINT64_C(0x7fefffffffffffff),
               UINT64_C(0x7fefffffffffffff));

    const auto bad_input = assessIntervalOperation(
        IntervalOperation::Add, {subnormal, subnormal}, {0.0, 0.0});
    EXPECT_FALSE(bad_input.available());
    EXPECT_EQ(bad_input.failure, ArithmeticFailure::InvalidInput);

    const auto underflow = assessIntervalOperation(
        IntervalOperation::Multiply,
        {minimum_normal, minimum_normal},
        {0.5, 0.5});
    EXPECT_FALSE(underflow.available());
    EXPECT_EQ(underflow.failure, ArithmeticFailure::ArithmeticRange);

    const auto overflow = assessIntervalOperation(
        IntervalOperation::Multiply,
        {maximum_finite, maximum_finite},
        {2.0, 2.0});
    EXPECT_FALSE(overflow.available());
    EXPECT_EQ(overflow.failure, ArithmeticFailure::ArithmeticRange);
}

TEST(ProducerArithmeticAssessment,
     DyadicDirectOracleCoversZeroReflectionAndBinadeCarry)
{
    const Real one = fromBits(UINT64_C(0x3ff0000000000000));
    const Real negative_one = fromBits(UINT64_C(0xbff0000000000000));
    const Real two_to_minus_53 = fromBits(UINT64_C(0x3ca0000000000000));
    const Real two_to_minus_54 = fromBits(UINT64_C(0x3c90000000000000));

    expectPositiveZeroSingleton(assessIntervalOperation(
        IntervalOperation::Add, {-0.0, -0.0}, {0.0, 0.0}));
    expectPositiveZeroSingleton(assessIntervalOperation(
        IntervalOperation::Multiply, {0.0, 0.0}, {negative_one, negative_one}));
    expectPositiveZeroSingleton(assessIntervalOperation(
        IntervalOperation::Add, {one, one}, {negative_one, negative_one}));

    const auto predecessor = assessIntervalOperation(
        IntervalOperation::Subtract,
        {one, one},
        {two_to_minus_53, two_to_minus_53});
    expectTightDyadicBracket(
        predecessor,
        independentlyScaledMagnitude(one) -
            independentlyScaledMagnitude(two_to_minus_53),
        1074u);
    expectBits(predecessor.interval,
               UINT64_C(0x3fefffffffffffff),
               UINT64_C(0x3fefffffffffffff));

    const auto crossing = assessIntervalOperation(
        IntervalOperation::Subtract,
        {one, one},
        {two_to_minus_54, two_to_minus_54});
    expectTightDyadicBracket(
        crossing,
        independentlyScaledMagnitude(one) -
            independentlyScaledMagnitude(two_to_minus_54),
        1074u);
    expectBits(crossing.interval,
               UINT64_C(0x3fefffffffffffff),
               UINT64_C(0x3ff0000000000000));

    const auto reflected = assessIntervalOperation(
        IntervalOperation::Subtract,
        {negative_one, negative_one},
        {two_to_minus_54, two_to_minus_54});
    expectTightDyadicBracket(
        reflected,
        independentlyScaledMagnitude(one) +
            independentlyScaledMagnitude(two_to_minus_54),
        1074u,
        true);
}

TEST(ProducerArithmeticAssessment,
     DyadicDirectOracleCoversRangeAndAllDiscardedLimbPositions)
{
    const Real minimum = fromBits(UINT64_C(0x0010000000000000));
    const Real minimum_successor = fromBits(UINT64_C(0x0010000000000001));
    const Real half = fromBits(UINT64_C(0x3fe0000000000000));
    const Real one_predecessor = fromBits(UINT64_C(0x3fefffffffffffff));

    const auto exact_subnormal = assessIntervalOperation(
        IntervalOperation::Subtract,
        {minimum_successor, minimum_successor},
        {minimum, minimum});
    expectDyadicRangeFailure(exact_subnormal, cpp_int(1), 1074u);

    for (const Real factor : {half, one_predecessor}) {
        const auto below_minimum = assessIntervalOperation(
            IntervalOperation::Multiply,
            {minimum, minimum},
            {factor, factor});
        expectDyadicRangeFailure(
            below_minimum,
            independentlyScaledMagnitude(minimum) *
                independentlyScaledMagnitude(factor),
            2148u);
    }

    for (const int exponent : {1, 2, 3, 13, 14}) {
        const Real leading = normalWithExponentAndFraction(exponent, 0u);
        const auto sticky = assessIntervalOperation(
            IntervalOperation::Add,
            {leading, leading},
            {minimum, minimum});
        expectTightDyadicBracket(
            sticky,
            independentlyScaledMagnitude(leading) +
                independentlyScaledMagnitude(minimum),
            1074u);

        const Real one_ulp =
            normalWithExponentAndFraction(exponent - 52, 0u);
        const auto exact = assessIntervalOperation(
            IntervalOperation::Add,
            {leading, leading},
            {one_ulp, one_ulp});
        expectTightDyadicBracket(
            exact,
            independentlyScaledMagnitude(leading) +
                independentlyScaledMagnitude(one_ulp),
            1074u);
    }
}

TEST(ProducerArithmeticAssessment,
     DyadicProductOracleCoversRangeAndExtractionBoundaries)
{
    const Real minimum = fromBits(UINT64_C(0x0010000000000000));
    const Real minimum_successor = fromBits(UINT64_C(0x0010000000000001));
    const Real half = fromBits(UINT64_C(0x3fe0000000000000));
    const Real one = fromBits(UINT64_C(0x3ff0000000000000));
    const Real one_successor = fromBits(UINT64_C(0x3ff0000000000001));

    const auto exact_minimum = assessIntervalOperation(
        IntervalOperation::Multiply,
        {minimum, minimum},
        {one, one});
    expectTightDyadicBracket(
        exact_minimum,
        independentlyScaledMagnitude(minimum) *
            independentlyScaledMagnitude(one),
        2148u);

    const auto above_minimum = assessIntervalOperation(
        IntervalOperation::Multiply,
        {minimum_successor, minimum_successor},
        {one_successor, one_successor});
    expectTightDyadicBracket(
        above_minimum,
        independentlyScaledMagnitude(minimum_successor) *
            independentlyScaledMagnitude(one_successor),
        2148u);

    const auto half_minimum = assessIntervalOperation(
        IntervalOperation::Multiply,
        {minimum, minimum},
        {half, half});
    expectDyadicRangeFailure(
        half_minimum,
        independentlyScaledMagnitude(minimum) *
            independentlyScaledMagnitude(half),
        2148u);

    const auto minimum_squared = assessIntervalOperation(
        IntervalOperation::Multiply,
        {minimum, minimum},
        {minimum, minimum});
    expectDyadicRangeFailure(
        minimum_squared,
        independentlyScaledMagnitude(minimum) *
            independentlyScaledMagnitude(minimum),
        2148u);

    for (const int exponent : {15, 16, 17, 27, 28}) {
        const Real left = normalWithExponentAndFraction(exponent, 1u);
        const auto product = assessIntervalOperation(
            IntervalOperation::Multiply,
            {left, left},
            {one_successor, one_successor});
        expectTightDyadicBracket(
            product,
            independentlyScaledMagnitude(left) *
                independentlyScaledMagnitude(one_successor),
            2148u);
    }
}

TEST(ProducerArithmeticAssessment,
     DyadicOracleCoversMaximumFiniteTailAndExponentRejection)
{
    const Real minimum = fromBits(UINT64_C(0x0010000000000000));
    const Real one = fromBits(UINT64_C(0x3ff0000000000000));
    const Real maximum = fromBits(UINT64_C(0x7fefffffffffffff));

    const auto exact_maximum = assessIntervalOperation(
        IntervalOperation::Multiply,
        {maximum, maximum},
        {one, one});
    expectTightDyadicBracket(
        exact_maximum,
        independentlyScaledMagnitude(maximum) *
            independentlyScaledMagnitude(one),
        2148u);

    const auto maximum_tail = assessIntervalOperation(
        IntervalOperation::Add,
        {maximum, maximum},
        {minimum, minimum});
    expectDyadicRangeFailure(
        maximum_tail,
        independentlyScaledMagnitude(maximum) +
            independentlyScaledMagnitude(minimum),
        1074u);

    const auto maximum_span = assessIntervalOperation(
        IntervalOperation::Subtract,
        {maximum, maximum},
        {minimum, minimum});
    expectTightDyadicBracket(
        maximum_span,
        independentlyScaledMagnitude(maximum) -
            independentlyScaledMagnitude(minimum),
        1074u);

    const Real product_left = normalWithExponentAndFraction(512, 1u);
    const Real product_right =
        normalWithExponentAndFraction(511, fraction_mask - 1u);
    const cpp_int exact_product =
        independentlyScaledMagnitude(product_left) *
        independentlyScaledMagnitude(product_right);
    const cpp_int maximum_scaled =
        independentlyScaledMagnitude(maximum) << 1074u;
    const cpp_int two_to_1024_scaled = cpp_int(1) << 3172u;
    EXPECT_EQ(exact_product,
              two_to_1024_scaled - (cpp_int(1) << 3068u));
    EXPECT_GT(exact_product, maximum_scaled);
    EXPECT_LT(exact_product, two_to_1024_scaled);
    const auto product_tail = assessIntervalOperation(
        IntervalOperation::Multiply,
        {product_left, product_left},
        {product_right, product_right});
    expectDyadicRangeFailure(
        product_tail,
        exact_product,
        2148u);

    const auto maximum_sum = assessIntervalOperation(
        IntervalOperation::Add,
        {maximum, maximum},
        {maximum, maximum});
    expectDyadicRangeFailure(
        maximum_sum,
        independentlyScaledMagnitude(maximum) +
            independentlyScaledMagnitude(maximum),
        1074u);

    const auto maximum_product = assessIntervalOperation(
        IntervalOperation::Multiply,
        {maximum, maximum},
        {maximum, maximum});
    expectDyadicRangeFailure(
        maximum_product,
        independentlyScaledMagnitude(maximum) *
            independentlyScaledMagnitude(maximum),
        2148u);
}

TEST(ProducerArithmeticAssessment,
     CompactDivisionOracleCoversNormalizedBranchesSignsAndRemainders)
{
    const Real one = fromBits(UINT64_C(0x3ff0000000000000));
    const Real one_successor = fromBits(UINT64_C(0x3ff0000000000001));
    const Real one_predecessor = fromBits(UINT64_C(0x3fefffffffffffff));
    const Real maximum = fromBits(UINT64_C(0x7fefffffffffffff));
    const Real maximum_predecessor = fromBits(UINT64_C(0x7feffffffffffffe));

    for (const auto operands : {
             std::array<Real, 2>{1.0, 3.0},
             std::array<Real, 2>{1.0, 5.0},
             std::array<Real, 2>{3.0, 2.0},
             std::array<Real, 2>{one_successor, one},
             std::array<Real, 2>{maximum, maximum},
             std::array<Real, 2>{one, one_successor},
             std::array<Real, 2>{one, one_predecessor},
             std::array<Real, 2>{maximum_predecessor, one_predecessor}}) {
        const auto result = assessIntervalOperation(
            IntervalOperation::Divide,
            {operands[0], operands[0]},
            {operands[1], operands[1]});
        expectTightDivisionBracket(result, operands[0], operands[1]);
    }

    const auto below_one = assessIntervalOperation(
        IntervalOperation::Divide,
        {one, one},
        {one_successor, one_successor});
    expectBits(below_one.interval,
               UINT64_C(0x3feffffffffffffe),
               UINT64_C(0x3fefffffffffffff));
    const auto above_one = assessIntervalOperation(
        IntervalOperation::Divide,
        {one, one},
        {one_predecessor, one_predecessor});
    expectBits(above_one.interval,
               UINT64_C(0x3ff0000000000000),
               UINT64_C(0x3ff0000000000001));
    const auto last_finite_tail = assessIntervalOperation(
        IntervalOperation::Divide,
        {maximum_predecessor, maximum_predecessor},
        {one_predecessor, one_predecessor});
    expectBits(last_finite_tail.interval,
               UINT64_C(0x7feffffffffffffe),
               UINT64_C(0x7fefffffffffffff));

    for (const auto operands : {
             std::array<Real, 2>{-1.0, 5.0},
             std::array<Real, 2>{1.0, -5.0},
             std::array<Real, 2>{-1.0, -5.0}}) {
        const auto result = assessIntervalOperation(
            IntervalOperation::Divide,
            {operands[0], operands[0]},
            {operands[1], operands[1]});
        expectTightDivisionBracket(result, operands[0], operands[1]);
    }

    expectPositiveZeroSingleton(assessIntervalOperation(
        IntervalOperation::Divide, {0.0, 0.0}, {3.0, 3.0}));
    expectPositiveZeroSingleton(assessIntervalOperation(
        IntervalOperation::Divide, {-0.0, -0.0}, {-3.0, -3.0}));
}

TEST(ProducerArithmeticAssessment,
     CompactDivisionOracleCoversNormalRangeAndIntervalExtrema)
{
    const Real minimum = fromBits(UINT64_C(0x0010000000000000));
    const Real minimum_successor = fromBits(UINT64_C(0x0010000000000001));
    const Real maximum = fromBits(UINT64_C(0x7fefffffffffffff));
    const Real one = fromBits(UINT64_C(0x3ff0000000000000));
    const Real one_successor = fromBits(UINT64_C(0x3ff0000000000001));
    const Real one_predecessor = fromBits(UINT64_C(0x3fefffffffffffff));

    const auto exact_minimum = assessIntervalOperation(
        IntervalOperation::Divide,
        {minimum, minimum},
        {one, one});
    expectTightDivisionBracket(exact_minimum, minimum, one);
    expectBits(exact_minimum.interval,
               UINT64_C(0x0010000000000000),
               UINT64_C(0x0010000000000000));

    const auto exact_above_minimum = assessIntervalOperation(
        IntervalOperation::Divide,
        {minimum_successor, minimum_successor},
        {one, one});
    expectTightDivisionBracket(exact_above_minimum, minimum_successor, one);
    expectBits(exact_above_minimum.interval,
               UINT64_C(0x0010000000000001),
               UINT64_C(0x0010000000000001));

    const auto recovered_minimum = assessIntervalOperation(
        IntervalOperation::Divide,
        {minimum_successor, minimum_successor},
        {one_successor, one_successor});
    expectTightDivisionBracket(
        recovered_minimum, minimum_successor, one_successor);
    expectBits(recovered_minimum.interval,
               UINT64_C(0x0010000000000000),
               UINT64_C(0x0010000000000000));

    const auto exact_maximum = assessIntervalOperation(
        IntervalOperation::Divide,
        {maximum, maximum},
        {one, one});
    expectTightDivisionBracket(exact_maximum, maximum, one);
    expectBits(exact_maximum.interval,
               UINT64_C(0x7fefffffffffffff),
               UINT64_C(0x7fefffffffffffff));

    const auto below_minimum = assessIntervalOperation(
        IntervalOperation::Divide,
        {minimum, minimum},
        {one_successor, one_successor});
    expectDivisionRangeFailure(below_minimum, minimum, one_successor);
    const auto above_maximum = assessIntervalOperation(
        IntervalOperation::Divide,
        {maximum, maximum},
        {one_predecessor, one_predecessor});
    expectDivisionRangeFailure(above_maximum, maximum, one_predecessor);

    const auto positive_extrema = assessIntervalOperation(
        IntervalOperation::Divide, {1.0, 3.0}, {2.0, 4.0});
    ASSERT_TRUE(positive_extrema.available());
    expectBits(positive_extrema.interval,
               UINT64_C(0x3fd0000000000000),
               UINT64_C(0x3ff8000000000000));
    const auto negative_extrema = assessIntervalOperation(
        IntervalOperation::Divide, {1.0, 3.0}, {-4.0, -2.0});
    ASSERT_TRUE(negative_extrema.available());
    expectBits(negative_extrema.interval,
               UINT64_C(0xbff8000000000000),
               UINT64_C(0xbfd0000000000000));

    for (const ArithmeticInterval denominator : {
             ArithmeticInterval{-1.0, 1.0},
             ArithmeticInterval{0.0, 0.0},
             ArithmeticInterval{-0.0, -0.0}}) {
        const auto result = assessIntervalOperation(
            IntervalOperation::Divide, {1.0, 1.0}, denominator);
        EXPECT_FALSE(result.available());
        EXPECT_EQ(result.failure, ArithmeticFailure::UnresolvedDenominator);
        expectBits(result.interval, 0u, 0u);
    }

    const Real infinity = fromBits(UINT64_C(0x7ff0000000000000));
    const Real subnormal = fromBits(UINT64_C(0x0000000000000001));
    for (const Real invalid : {infinity, subnormal}) {
        const auto result = assessIntervalOperation(
            IntervalOperation::Divide,
            {invalid, invalid},
            {0.0, 0.0});
        EXPECT_FALSE(result.available());
        EXPECT_EQ(result.failure, ArithmeticFailure::InvalidInput);
        expectBits(result.interval, 0u, 0u);
    }
}

TEST(ProducerArithmeticAssessment,
     CompactSqrtOracleCoversParityResidualsAndBinadeCarry)
{
    const Real minimum = fromBits(UINT64_C(0x0010000000000000));
    const Real maximum = fromBits(UINT64_C(0x7fefffffffffffff));
    const Real one_predecessor = fromBits(UINT64_C(0x3fefffffffffffff));
    const Real one_successor = fromBits(UINT64_C(0x3ff0000000000001));
    const Real four_predecessor = fromBits(UINT64_C(0x400fffffffffffff));
    const Real four_successor = fromBits(UINT64_C(0x4010000000000001));
    const Real negative_odd_exponent =
        normalWithExponentAndFraction(-1021, 0u);

    for (const Real input : {
             1.0,
             4.0,
             2.25,
             2.0,
             minimum,
             negative_odd_exponent,
             one_predecessor,
             one_successor,
             four_predecessor,
             four_successor,
             maximum}) {
        const auto result = assessIntervalSqrt({input, input});
        expectTightSqrtBracket(result, input);
    }

    expectBits(assessIntervalSqrt({1.0, 1.0}).interval,
               UINT64_C(0x3ff0000000000000),
               UINT64_C(0x3ff0000000000000));
    expectBits(assessIntervalSqrt({4.0, 4.0}).interval,
               UINT64_C(0x4000000000000000),
               UINT64_C(0x4000000000000000));
    expectBits(assessIntervalSqrt({2.25, 2.25}).interval,
               UINT64_C(0x3ff8000000000000),
               UINT64_C(0x3ff8000000000000));
    expectBits(assessIntervalSqrt({minimum, minimum}).interval,
               UINT64_C(0x2000000000000000),
               UINT64_C(0x2000000000000000));
    expectBits(assessIntervalSqrt({maximum, maximum}).interval,
               UINT64_C(0x5fefffffffffffff),
               UINT64_C(0x5ff0000000000000));
}

TEST(ProducerArithmeticAssessment,
     CompactSqrtOracleCoversNormalBoundariesValidationAndIntervalExtrema)
{
    const Real minimum = fromBits(UINT64_C(0x0010000000000000));
    const Real maximum = fromBits(UINT64_C(0x7fefffffffffffff));
    const auto extrema = assessIntervalSqrt({minimum, maximum});
    ASSERT_TRUE(extrema.available());
    expectBits(extrema.interval,
               UINT64_C(0x2000000000000000),
               UINT64_C(0x5ff0000000000000));
    expectTightSqrtBracket(
        assessIntervalSqrt({minimum, minimum}), minimum);
    expectTightSqrtBracket(
        assessIntervalSqrt({maximum, maximum}), maximum);

    expectPositiveZeroSingleton(assessIntervalSqrt({0.0, 0.0}));
    expectPositiveZeroSingleton(assessIntervalSqrt({-0.0, 0.0}));

    const Real subnormal = fromBits(UINT64_C(0x0000000000000001));
    const Real infinity = fromBits(UINT64_C(0x7ff0000000000000));
    for (const ArithmeticInterval input : {
             ArithmeticInterval{-1.0, -1.0},
             ArithmeticInterval{2.0, 1.0},
             ArithmeticInterval{subnormal, subnormal},
             ArithmeticInterval{infinity, infinity}}) {
        const auto result = assessIntervalSqrt(input);
        EXPECT_FALSE(result.available());
        EXPECT_EQ(result.failure, ArithmeticFailure::InvalidInput);
        expectBits(result.interval, 0u, 0u);
    }
}

TEST(ProducerArithmeticAssessment,
     NonDyadicForwardAndReverseRootsResolveTheSameRemovedOrigin)
{
    const Real forward_emitted = 0.2;
    const Real forward_denominator = -1.0 - 4.0;
    const Real forward_quotient = -1.0 / forward_denominator;
    OriginalEdgeObservation forward;
    forward.a = {0.0, 0.0, 0.0};
    forward.b = {1.0, 0.0, 0.0};
    forward.emitted = {forward_emitted, 0.0, 0.0};
    forward.phi_a = -1.0;
    forward.phi_b = 4.0;
    forward.signed_band = 1.0e-12;
    forward.actual_signed_a = -1.0;
    forward.actual_signed_b = 4.0;
    forward.actual_denominator = forward_denominator;
    forward.actual_quotient = forward_quotient;
    forward.actual_clamped = forward_quotient;
    forward.helper_denominator_guard = true;
    forward.division_taken = true;

    const Real reverse_quotient = 0.8;
    const Real reverse_emitted = 1.0 - reverse_quotient;
    const Real reverse_denominator = 4.0 - (-1.0);
    OriginalEdgeObservation reverse;
    reverse.a = {1.0, 0.0, 0.0};
    reverse.b = {0.0, 0.0, 0.0};
    reverse.emitted = {reverse_emitted, 0.0, 0.0};
    reverse.phi_a = 4.0;
    reverse.phi_b = -1.0;
    reverse.signed_band = 1.0e-12;
    reverse.actual_signed_a = 4.0;
    reverse.actual_signed_b = -1.0;
    reverse.actual_denominator = reverse_denominator;
    reverse.actual_quotient = reverse_quotient;
    reverse.actual_clamped = reverse_quotient;
    reverse.helper_denominator_guard = true;
    reverse.division_taken = true;

    const auto forward_assessment = assessOriginalEdge(forward);
    const auto reverse_assessment = assessOriginalEdge(reverse);
    ASSERT_TRUE(forward_assessment.available());
    ASSERT_TRUE(reverse_assessment.available());
    EXPECT_LE(forward_assessment.ideal[0].lower, 0.2);
    EXPECT_GE(forward_assessment.ideal[0].upper, 0.2);
    EXPECT_LE(reverse_assessment.ideal[0].lower, 0.2);
    EXPECT_GE(reverse_assessment.ideal[0].upper, 0.2);
    EXPECT_TRUE(std::isfinite(forward_assessment.radius[0]));
    EXPECT_TRUE(std::isfinite(reverse_assessment.radius[0]));

    const Real actual_distance =
        std::abs(forward_emitted - reverse_emitted);
    const auto distance = assessDistance(
        forward_assessment,
        forward.emitted,
        reverse_assessment,
        reverse.emitted,
        1.0e-12,
        3u,
        OriginRelation::SameOriginal,
        {true, actual_distance, true, true});
    ASSERT_TRUE(distance.available());
    EXPECT_LT(distance.hull.upper, 1.0e-12);
}

TEST(ProducerArithmeticAssessment,
     HelperMarginRejectsTwoToMinus109ButDirectPathRemainsAvailable)
{
    const Real half_denominator = std::ldexp(1.0, -110);
    const Real denominator = std::ldexp(-1.0, -109);
    OriginalEdgeObservation input;
    input.a = {0.0, 0.0, 0.0};
    input.b = {1.0, 0.0, 0.0};
    input.emitted = {0.5, 0.0, 0.0};
    input.phi_a = -half_denominator;
    input.phi_b = half_denominator;
    input.signed_band = std::ldexp(1.0, -120);
    input.actual_signed_a = -half_denominator;
    input.actual_signed_b = half_denominator;
    input.actual_denominator = denominator;
    input.actual_quotient = 0.5;
    input.actual_clamped = 0.5;
    input.division_taken = true;

    input.helper_denominator_guard = true;
    const auto guarded = assessOriginalEdge(input);
    EXPECT_FALSE(guarded.available());
    EXPECT_EQ(guarded.failure, ArithmeticFailure::UnresolvedDenominator);

    input.helper_denominator_guard = false;
    const auto direct = assessOriginalEdge(input);
    EXPECT_TRUE(direct.available());
}

TEST(ProducerArithmeticAssessment,
     EdgeAssessmentRejectsBandEqualityAndChangedProducerBranches)
{
    OriginalEdgeObservation input;
    input.a = {0.0, 0.0, 0.0};
    input.b = {1.0, 0.0, 0.0};
    input.emitted = {0.5, 0.0, 0.0};
    input.phi_a = -1.0;
    input.phi_b = 1.0;
    input.signed_band = 1.0;
    input.actual_signed_a = -1.0;
    input.actual_signed_b = 1.0;
    input.actual_denominator = -2.0;
    input.actual_quotient = 0.5;
    input.actual_clamped = 0.5;
    input.division_taken = true;

    auto assessment = assessOriginalEdge(input);
    EXPECT_FALSE(assessment.available());
    EXPECT_EQ(assessment.failure, ArithmeticFailure::UnresolvedBand);

    input.signed_band = 1.0e-12;
    input.actual_signed_a = 1.0;
    assessment = assessOriginalEdge(input);
    EXPECT_FALSE(assessment.available());
    EXPECT_EQ(assessment.failure, ArithmeticFailure::ChangedBranch);

    input.actual_signed_a = -1.0;
    input.division_taken = false;
    assessment = assessOriginalEdge(input);
    EXPECT_FALSE(assessment.available());
    EXPECT_EQ(assessment.failure, ArithmeticFailure::ChangedBranch);

    input.division_taken = true;
    input.actual_clamped = 0.25;
    assessment = assessOriginalEdge(input);
    EXPECT_FALSE(assessment.available());
    EXPECT_EQ(assessment.failure, ArithmeticFailure::ChangedBranch);

    input.actual_clamped = 0.5;
    input.canonicalization_changed = true;
    assessment = assessOriginalEdge(input);
    EXPECT_FALSE(assessment.available());
    EXPECT_EQ(assessment.failure, ArithmeticFailure::ChangedBranch);
}

TEST(ProducerArithmeticAssessment,
     EdgeAssessmentFailsClosedForQuotientUnderflowAndDenominatorOverflow)
{
    const Real tiny = std::ldexp(1.0, -600);
    const Real large = std::ldexp(1.0, 500);
    OriginalEdgeObservation underflow;
    underflow.a = {0.0, 0.0, 0.0};
    underflow.b = {1.0, 0.0, 0.0};
    underflow.emitted = {0.0, 0.0, 0.0};
    underflow.phi_a = tiny;
    underflow.phi_b = -large;
    underflow.signed_band = std::ldexp(1.0, -700);
    underflow.actual_signed_a = tiny;
    underflow.actual_signed_b = -large;
    underflow.actual_denominator = large;
    underflow.actual_quotient = 0.0;
    underflow.actual_clamped = 0.0;
    underflow.division_taken = true;
    const auto tiny_quotient = assessOriginalEdge(underflow);
    EXPECT_FALSE(tiny_quotient.available());
    EXPECT_EQ(tiny_quotient.failure, ArithmeticFailure::UnresolvedInterior);

    const Real huge = std::ldexp(1.0, 1023);
    OriginalEdgeObservation overflow = underflow;
    overflow.phi_a = huge;
    overflow.phi_b = -huge;
    overflow.actual_signed_a = huge;
    overflow.actual_signed_b = -huge;
    overflow.actual_denominator =
        fromBits(UINT64_C(0x7ff0000000000000));
    const auto huge_denominator = assessOriginalEdge(overflow);
    EXPECT_FALSE(huge_denominator.available());
    EXPECT_EQ(huge_denominator.failure, ArithmeticFailure::InvalidInput);
}

TEST(ProducerArithmeticAssessment,
     CornerDistanceDistinguishesOriginsDimensionsAndObservedBooleans)
{
    const AssessmentPoint zero{0.0, 0.0, 0.0};
    const AssessmentPoint one{1.0, 0.0, 0.0};
    const AssessmentPoint third_only{0.0, 0.0, 1.0};
    const auto zero_corner = assessOriginalCorner(zero, zero);
    const auto one_corner = assessOriginalCorner(one, one);
    const auto third_corner = assessOriginalCorner(third_only, third_only);
    ASSERT_TRUE(zero_corner.available());
    ASSERT_TRUE(one_corner.available());
    ASSERT_TRUE(third_corner.available());
    expectBits(zero_corner.ideal[0], 0u, 0u);
    EXPECT_EQ(bits(zero_corner.radius[0]), 0u);

    const auto surviving = assessDistance(
        zero_corner, zero, one_corner, one, std::ldexp(1.0, -20), 3u,
        OriginRelation::DistinctOriginal, {});
    EXPECT_TRUE(surviving.available());

    const auto captured_distance = assessDistance(
        zero_corner, zero, one_corner, one, 0.5, 3u,
        OriginRelation::DistinctOriginal, {true, 2.0, false, false});
    ASSERT_TRUE(captured_distance.available());
    EXPECT_EQ(bits(captured_distance.hull.upper),
              UINT64_C(0x4000000000000000));

    const auto two_dimensional = assessDistance(
        zero_corner, zero, third_corner, third_only,
        std::ldexp(1.0, -20), 2u,
        OriginRelation::DistinctOriginal, {});
    EXPECT_FALSE(two_dimensional.available());
    EXPECT_EQ(two_dimensional.failure, ArithmeticFailure::UnresolvedDistance);

    const auto three_dimensional = assessDistance(
        zero_corner, zero, third_corner, third_only,
        std::ldexp(1.0, -20), 3u,
        OriginRelation::DistinctOriginal, {});
    EXPECT_TRUE(three_dimensional.available());

    const auto merged_distinct = assessDistance(
        zero_corner, zero, one_corner, one, 2.0, 3u,
        OriginRelation::DistinctOriginal, {true, 1.0, true, true});
    EXPECT_FALSE(merged_distinct.available());
    EXPECT_EQ(merged_distinct.failure, ArithmeticFailure::DistinctOriginMerge);

    const auto bad_boolean = assessDistance(
        zero_corner, zero, one_corner, one, 2.0, 3u,
        OriginRelation::DistinctOriginal, {true, 1.0, true, false});
    EXPECT_FALSE(bad_boolean.available());
    EXPECT_EQ(bad_boolean.failure, ArithmeticFailure::InvalidInput);

    const auto unknown = assessDistance(
        zero_corner, zero, one_corner, one, 2.0, 3u,
        OriginRelation::Unknown, {true, 1.0, true, true});
    EXPECT_FALSE(unknown.available());
    EXPECT_EQ(unknown.failure, ArithmeticFailure::UnknownOrigin);
}

TEST(ProducerArithmeticAssessment,
     DistanceRejectsRetainedRepeatsToleranceEqualityAndTinySquares)
{
    const AssessmentPoint zero{0.0, 0.0, 0.0};
    const Real tolerance = std::ldexp(1.0, -20);
    const AssessmentPoint at_tolerance{tolerance, 0.0, 0.0};
    const auto exact = assessOriginalCorner(zero, zero);
    const auto displaced_same = assessOriginalCorner(zero, at_tolerance);
    ASSERT_TRUE(exact.available());
    ASSERT_TRUE(displaced_same.available());

    const auto retained = assessDistance(
        exact, zero, exact, zero, tolerance, 3u,
        OriginRelation::SameOriginal, {true, 0.0, false, false});
    EXPECT_FALSE(retained.available());
    EXPECT_EQ(retained.failure, ArithmeticFailure::RetainedRepeat);

    const auto equality = assessDistance(
        exact, zero, displaced_same, at_tolerance, tolerance, 3u,
        OriginRelation::SameOriginal,
        {true, tolerance, true, true});
    EXPECT_FALSE(equality.available());
    EXPECT_EQ(equality.failure, ArithmeticFailure::UnresolvedDistance);

    const Real tiny = std::ldexp(1.0, -600);
    const AssessmentPoint tiny_point{tiny, 0.0, 0.0};
    const auto tiny_corner = assessOriginalCorner(tiny_point, tiny_point);
    ASSERT_TRUE(tiny_corner.available());
    const auto tiny_distance = assessDistance(
        exact, zero, tiny_corner, tiny_point, std::ldexp(1.0, -700), 3u,
        OriginRelation::DistinctOriginal,
        {true, tiny, false, false});
    EXPECT_FALSE(tiny_distance.available());
    EXPECT_EQ(tiny_distance.failure, ArithmeticFailure::UnresolvedDistance);
}

TEST(ProducerArithmeticAssessment,
     MaximumWidthSearchesSatisfyIndependentIntegerInequalities)
{
    const Real maximum = fromBits(UINT64_C(0x7fefffffffffffff));
    const Real minimum = fromBits(UINT64_C(0x0010000000000000));

    const auto span = assessIntervalOperation(
        IntervalOperation::Add,
        {maximum, maximum},
        {-minimum, -minimum});
    ASSERT_TRUE(span.available());
    expectBits(span.interval,
               UINT64_C(0x7feffffffffffffe),
               UINT64_C(0x7fefffffffffffff));
    const cpp_int exact_span = independentlyScaledMagnitude(maximum) -
                               independentlyScaledMagnitude(minimum);
    EXPECT_LE(independentlyScaledMagnitude(span.interval.lower), exact_span);
    EXPECT_GE(independentlyScaledMagnitude(span.interval.upper), exact_span);

    const auto maximum_divided_by_itself = assessIntervalOperation(
        IntervalOperation::Divide,
        {maximum, maximum},
        {maximum, maximum});
    ASSERT_TRUE(maximum_divided_by_itself.available());
    expectBits(maximum_divided_by_itself.interval,
               UINT64_C(0x3ff0000000000000),
               UINT64_C(0x3ff0000000000000));
    expectDivisionInequality(
        maximum, maximum, maximum_divided_by_itself.interval);

    const auto maximum_square_root = assessIntervalSqrt({maximum, maximum});
    ASSERT_TRUE(maximum_square_root.available());
    expectSqrtInequality(maximum, maximum_square_root.interval);

    const auto ordinary_product = assessIntervalOperation(
        IntervalOperation::Multiply, {0.2, 0.2}, {5.0, 5.0});
    ASSERT_TRUE(ordinary_product.available());
    expectProductInequality(0.2, 5.0, ordinary_product.interval);
}

TEST(ProducerArithmeticAssessment,
     IntegerKernelPreservesPreexistingStatusErrnoAndSignalingNanRejection)
{
    fenv_t saved_environment;
    ASSERT_EQ(fegetenv(&saved_environment), 0);
    const int saved_errno = errno;
    ASSERT_EQ(feclearexcept(FE_ALL_EXCEPT), 0);
    ASSERT_EQ(feraiseexcept(FE_DIVBYZERO | FE_INEXACT), 0);
    errno = EDOM;
    const auto before_success = readHardwareState();
    const auto success = assessIntervalSqrt({2.0, 2.0});
    const auto after_success = readHardwareState();
    const int errno_after_success = errno;

    const auto signaling_nan = fromBits(UINT64_C(0x7ff0000000000001));
    const auto before_rejection = readHardwareState();
    const auto rejection = assessIntervalSqrt(
        {signaling_nan, signaling_nan});
    const auto after_rejection = readHardwareState();
    const int errno_after_rejection = errno;
    const Real maximum = fromBits(UINT64_C(0x7fefffffffffffff));
    const auto before_range_rejection = readHardwareState();
    const auto range_rejection = assessIntervalOperation(
        IntervalOperation::Multiply,
        {maximum, maximum},
        {2.0, 2.0});
    const auto after_range_rejection = readHardwareState();
    const int errno_after_range_rejection = errno;
    const Real minimum = fromBits(UINT64_C(0x0010000000000000));
    const auto before_division_range = readHardwareState();
    const auto division_range = assessIntervalOperation(
        IntervalOperation::Divide,
        {minimum, minimum},
        {2.0, 2.0});
    const auto after_division_range = readHardwareState();
    const int errno_after_division_range = errno;
    const auto before_denominator_rejection = readHardwareState();
    const auto denominator_rejection = assessIntervalOperation(
        IntervalOperation::Divide,
        {0.0, 0.0},
        {-0.0, -0.0});
    const auto after_denominator_rejection = readHardwareState();
    const int errno_after_denominator_rejection = errno;
    const auto before_zero_root = readHardwareState();
    const auto zero_root = assessIntervalSqrt({-0.0, -0.0});
    const auto after_zero_root = readHardwareState();
    const int errno_after_zero_root = errno;
    ASSERT_EQ(fesetenv(&saved_environment), 0);
    errno = saved_errno;

    ASSERT_TRUE(success.available());
    expectBits(success.interval,
               UINT64_C(0x3ff6a09e667f3bcc),
               UINT64_C(0x3ff6a09e667f3bcd));
    EXPECT_FALSE(rejection.available());
    EXPECT_EQ(rejection.failure, ArithmeticFailure::InvalidInput);
    EXPECT_FALSE(range_rejection.available());
    EXPECT_EQ(range_rejection.failure, ArithmeticFailure::ArithmeticRange);
    EXPECT_FALSE(division_range.available());
    EXPECT_EQ(division_range.failure, ArithmeticFailure::ArithmeticRange);
    EXPECT_FALSE(denominator_rejection.available());
    EXPECT_EQ(denominator_rejection.failure,
              ArithmeticFailure::UnresolvedDenominator);
    expectPositiveZeroSingleton(zero_root);
    expectHardwareStateEqual(after_success, before_success);
    expectHardwareStateEqual(after_rejection, before_rejection);
    expectHardwareStateEqual(after_range_rejection,
                             before_range_rejection);
    expectHardwareStateEqual(after_division_range,
                             before_division_range);
    expectHardwareStateEqual(after_denominator_rejection,
                             before_denominator_rejection);
    expectHardwareStateEqual(after_zero_root, before_zero_root);
    EXPECT_EQ(errno_after_success, EDOM);
    EXPECT_EQ(errno_after_rejection, EDOM);
    EXPECT_EQ(errno_after_range_rejection, EDOM);
    EXPECT_EQ(errno_after_division_range, EDOM);
    EXPECT_EQ(errno_after_denominator_rejection, EDOM);
    EXPECT_EQ(errno_after_zero_root, EDOM);
}

TEST(ProducerArithmeticAssessment,
     IntegerKernelIsModeIndependentUnderDirectedRoundingAndFtzDaz)
{
    constexpr std::array<int, 4> modes{
        FE_TONEAREST, FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO};
    for (const int mode : modes) {
        fenv_t saved_environment;
        ASSERT_EQ(fegetenv(&saved_environment), 0);
        const int saved_errno = errno;
        ASSERT_EQ(fesetround(mode), 0);
        errno = ERANGE;
        const auto before = readHardwareState();
        const auto result = assessIntervalOperation(
            IntervalOperation::Divide, {1.0, 1.0}, {5.0, 5.0});
        const auto after = readHardwareState();
        const int errno_after = errno;
        ASSERT_EQ(fesetenv(&saved_environment), 0);
        errno = saved_errno;

        ASSERT_TRUE(result.available());
        expectBits(result.interval,
                   UINT64_C(0x3fc9999999999999),
                   UINT64_C(0x3fc999999999999a));
        expectHardwareStateEqual(after, before);
        EXPECT_EQ(errno_after, ERANGE);
    }

    fenv_t saved_environment;
    ASSERT_EQ(fegetenv(&saved_environment), 0);
    const int saved_errno = errno;
    auto ftz_daz = readHardwareState().mxcsr | UINT32_C(0x8040);
    writeMxcsr(ftz_daz);
    errno = EILSEQ;
    const auto before = readHardwareState();
    const auto result = assessIntervalSqrt({2.0, 2.0});
    const auto after = readHardwareState();
    const int errno_after = errno;
    ASSERT_EQ(fesetenv(&saved_environment), 0);
    errno = saved_errno;

    ASSERT_TRUE(result.available());
    expectBits(result.interval,
               UINT64_C(0x3ff6a09e667f3bcc),
               UINT64_C(0x3ff6a09e667f3bcd));
    expectHardwareStateEqual(after, before);
    EXPECT_EQ(errno_after, EILSEQ);
}

TEST(ProducerArithmeticAssessment,
     IntegerKernelPreservesSafelyUnmaskedAndPerThreadModes)
{
    fenv_t saved_environment;
    ASSERT_EQ(fegetenv(&saved_environment), 0);
    const int saved_errno = errno;
    ASSERT_EQ(feclearexcept(FE_ALL_EXCEPT), 0);
    clearX87Status();
    auto unmasked = readHardwareState();
    unmasked.mxcsr &= ~UINT32_C(0x00001f80);
    unmasked.mxcsr &= ~UINT32_C(0x0000003f);
    unmasked.x87_control =
        static_cast<std::uint16_t>(unmasked.x87_control &
                                   static_cast<std::uint16_t>(~UINT16_C(0x003f)));
    writeMxcsr(unmasked.mxcsr);
    writeX87Control(unmasked.x87_control);
    errno = ENOENT;
    const auto before = readHardwareState();
    const auto result = assessIntervalOperation(
        IntervalOperation::Multiply, {2.0, 2.0}, {3.0, 3.0});
    const auto after = readHardwareState();
    const int errno_after = errno;
    const auto result_lower = bits(result.interval.lower);
    const auto result_upper = bits(result.interval.upper);
    const auto result_failure = result.failure;
    ASSERT_EQ(fesetenv(&saved_environment), 0);
    errno = saved_errno;

    EXPECT_EQ(result_failure, ArithmeticFailure::None);
    EXPECT_EQ(result_lower, UINT64_C(0x4018000000000000));
    EXPECT_EQ(result_upper, UINT64_C(0x4018000000000000));
    expectHardwareStateEqual(after, before);
    EXPECT_EQ(errno_after, ENOENT);

    struct ThreadResult {
        std::uint64_t lower{0u};
        std::uint64_t upper{0u};
        HardwareState before{};
        HardwareState after{};
        ArithmeticFailure failure{ArithmeticFailure::InvalidInput};
    };
    std::array<ThreadResult, 2> thread_results{};
    std::array<std::thread, 2> threads{
        std::thread([&thread_results]() {
            fenv_t saved;
            static_cast<void>(fegetenv(&saved));
            static_cast<void>(fesetround(FE_DOWNWARD));
            thread_results[0].before = readHardwareState();
            const auto value = assessIntervalSqrt({2.0, 2.0});
            thread_results[0].lower = bits(value.interval.lower);
            thread_results[0].upper = bits(value.interval.upper);
            thread_results[0].failure = value.failure;
            thread_results[0].after = readHardwareState();
            static_cast<void>(fesetenv(&saved));
        }),
        std::thread([&thread_results]() {
            fenv_t saved;
            static_cast<void>(fegetenv(&saved));
            static_cast<void>(fesetround(FE_UPWARD));
            thread_results[1].before = readHardwareState();
            const auto value = assessIntervalSqrt({2.0, 2.0});
            thread_results[1].lower = bits(value.interval.lower);
            thread_results[1].upper = bits(value.interval.upper);
            thread_results[1].failure = value.failure;
            thread_results[1].after = readHardwareState();
            static_cast<void>(fesetenv(&saved));
        })};
    for (auto& thread : threads) {
        thread.join();
    }
    for (const auto& thread_result : thread_results) {
        EXPECT_EQ(thread_result.failure, ArithmeticFailure::None);
        EXPECT_EQ(thread_result.lower, UINT64_C(0x3ff6a09e667f3bcc));
        EXPECT_EQ(thread_result.upper, UINT64_C(0x3ff6a09e667f3bcd));
        expectHardwareStateEqual(thread_result.after, thread_result.before);
    }
}

} // namespace

/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Systems/ExactAccumulator.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <vector>

namespace {

using svmp::FE::systems::ExactAccumulator;

double exactSum(const std::vector<double>& terms)
{
    ExactAccumulator accumulator;
    accumulator.add(terms);
    return accumulator.value();
}

std::uint64_t bitsOf(double value)
{
    return std::bit_cast<std::uint64_t>(value);
}

// Terms over most of the double range, with both signs.
std::vector<double> wideTerms(std::size_t count, std::uint32_t seed)
{
    std::mt19937_64 random(seed);
    std::uniform_real_distribution<double> mantissa(0.5, 1.0);
    std::uniform_int_distribution<int> exponent(-1060, 1000);
    std::vector<double> terms(count);
    for (auto& term : terms) {
        term = std::ldexp(mantissa(random), exponent(random));
        if (random() & 1u) {
            term = -term;
        }
    }
    return terms;
}

// Terms k * 2^e with |k| < 2^53 and e in [-73, -33]: every bit lies between
// 2^-73 and 2^20, and 20000 terms sum below 2^35, so the bits of all partial
// sums span fewer than 113 bits and a quadruple-precision sum is exact.
std::vector<double> narrowTerms(std::size_t count, std::uint32_t seed)
{
    std::mt19937_64 random(seed);
    std::uniform_int_distribution<std::int64_t> integer(
        -((std::int64_t{1} << 53) - 1), (std::int64_t{1} << 53) - 1);
    std::uniform_int_distribution<int> exponent(-73, -33);
    std::vector<double> terms(count);
    for (auto& term : terms) {
        term = std::ldexp(static_cast<double>(integer(random)), exponent(random));
    }
    return terms;
}

// Neumaier-compensated long double sum, rounded to double.
double longDoubleReference(const std::vector<double>& terms)
{
    long double sum = 0.0L;
    long double compensation = 0.0L;
    for (const double term : terms) {
        const long double t = term;
        const long double next = sum + t;
        if (std::fabs(sum) >= std::fabs(t)) {
            compensation += (sum - next) + t;
        } else {
            compensation += (t - next) + sum;
        }
        sum = next;
    }
    return static_cast<double>(sum + compensation);
}

} // namespace

TEST(ExactAccumulator, KnownSumsAreExact)
{
    EXPECT_EQ(exactSum({1e308, 1.0, -1e308}), 1.0);
    const double two53 = std::ldexp(1.0, 53);
    EXPECT_EQ(exactSum({two53, 1.0, 1.0}), two53 + 2.0);
    EXPECT_EQ(exactSum({1.0, two53, 1.0}), two53 + 2.0);
    // Ten copies of the double nearest 0.1 sum to 1 + 5.55e-17, which rounds
    // to 1 (a running double sum gives 0.9999999999999999).
    EXPECT_EQ(exactSum(std::vector<double>(10, 0.1)), 1.0);
    EXPECT_EQ(exactSum({0.5, std::ldexp(1.0, -1074), -0.5}), std::ldexp(1.0, -1074));
}

TEST(ExactAccumulator, RoundsToNearestWithTiesToEven)
{
    const double two53 = std::ldexp(1.0, 53);
    // Exactly halfway: to the even neighbour.
    EXPECT_EQ(exactSum({two53, 1.0}), two53);
    EXPECT_EQ(exactSum({two53 + 2.0, 1.0}), two53 + 4.0);
    // Just above or below halfway.
    EXPECT_EQ(exactSum({two53, 1.0, std::ldexp(1.0, -60)}), two53 + 2.0);
    EXPECT_EQ(exactSum({two53 + 2.0, 1.0, -std::ldexp(1.0, -60)}), two53 + 2.0);
    // Negative sums round symmetrically.
    EXPECT_EQ(exactSum({-two53, -1.0}), -two53);
    EXPECT_EQ(exactSum({-two53, -1.0, -std::ldexp(1.0, -60)}), -two53 - 2.0);
    // A carry out of the mantissa moves to the next binade.
    const double below_two = std::nextafter(2.0, 0.0);
    EXPECT_EQ(exactSum({below_two, std::ldexp(1.0, -53)}), 2.0);
}

TEST(ExactAccumulator, SubnormalsAndRangeLimits)
{
    const double tiny = std::numeric_limits<double>::denorm_min();
    const double smallest_normal = std::numeric_limits<double>::min();
    const double largest = std::numeric_limits<double>::max();
    EXPECT_EQ(exactSum({tiny, tiny, tiny}), 3.0 * tiny);
    EXPECT_EQ(exactSum({smallest_normal, -tiny}), std::nextafter(smallest_normal, 0.0));
    EXPECT_EQ(exactSum({largest, largest, -largest}), largest);
    EXPECT_EQ(exactSum({largest, largest}), std::numeric_limits<double>::infinity());
    EXPECT_EQ(exactSum({-largest, -largest}), -std::numeric_limits<double>::infinity());
    EXPECT_EQ(exactSum({largest, std::ldexp(1.0, 969)}), largest);  // below halfway to 2^1024
    EXPECT_EQ(exactSum({largest, std::ldexp(1.0, 970)}), std::numeric_limits<double>::infinity());
}

TEST(ExactAccumulator, ZerosAndNonFiniteTerms)
{
    EXPECT_EQ(bitsOf(exactSum({})), bitsOf(0.0));
    EXPECT_EQ(bitsOf(exactSum({-0.0})), bitsOf(0.0));
    EXPECT_EQ(bitsOf(exactSum({1.0, -1.0})), bitsOf(0.0));
    const double inf = std::numeric_limits<double>::infinity();
    EXPECT_TRUE(std::isnan(exactSum({1.0, std::numeric_limits<double>::quiet_NaN()})));
    EXPECT_TRUE(std::isnan(exactSum({inf, -inf})));
    EXPECT_EQ(exactSum({inf, 1.0, inf}), inf);
    EXPECT_EQ(exactSum({-inf, 1.0}), -inf);
}

TEST(ExactAccumulator, CancellationLeavesTheExactResidue)
{
    auto terms = wideTerms(4000, 7u);
    std::vector<double> all = terms;
    for (const double term : terms) {
        all.push_back(-term);
    }
    const double residue = std::ldexp(0.7853981633974483, -900);
    all.push_back(residue);
    std::mt19937_64 random(11u);
    std::shuffle(all.begin(), all.end(), random);
    EXPECT_EQ(bitsOf(exactSum(all)), bitsOf(residue));
}

TEST(ExactAccumulator, OrderAndPartitionDoNotChangeTheValue)
{
    auto terms = wideTerms(5000, 3u);
    const double reference = exactSum(terms);
    ASSERT_TRUE(std::isfinite(reference));
    std::mt19937_64 random(5u);
    for (int trial = 0; trial < 20; ++trial) {
        std::shuffle(terms.begin(), terms.end(), random);
        EXPECT_EQ(bitsOf(exactSum(terms)), bitsOf(reference)) << "permutation " << trial;

        // Partial accumulators over a random partition, merged in random order
        // and through their slots.
        const std::size_t parts = 1 + static_cast<std::size_t>(random() % 7);
        std::vector<ExactAccumulator> partial(parts);
        for (const double term : terms) {
            partial[random() % parts].add(term);
        }
        std::shuffle(partial.begin(), partial.end(), random);
        ExactAccumulator merged;
        std::array<std::int64_t, ExactAccumulator::kSlots> slot_sum{};
        for (const auto& part : partial) {
            merged += part;
            const auto slots = part.slots();
            for (std::size_t k = 0; k < slots.size(); ++k) {
                slot_sum[k] += slots[k];
            }
        }
        EXPECT_EQ(bitsOf(merged.value()), bitsOf(reference)) << "partition " << trial;
        EXPECT_EQ(bitsOf(ExactAccumulator::fromSlots(slot_sum).value()), bitsOf(reference))
            << "slot sum " << trial;
        EXPECT_EQ(merged.slots(), ExactAccumulator::fromSlots(slot_sum).slots());
    }
}

TEST(ExactAccumulator, MatchesAQuadruplePrecisionReference)
{
    for (std::uint32_t seed = 1; seed <= 5; ++seed) {
        const auto terms = narrowTerms(20000, seed);
        const double exact = exactSum(terms);
        const double reference = longDoubleReference(terms);
        if (std::numeric_limits<long double>::digits >= 113) {
            // The terms span fewer than 113 bits: the reference is the exact
            // sum rounded once.
            EXPECT_EQ(bitsOf(exact), bitsOf(reference)) << "seed " << seed;
        } else {
            EXPECT_LE(std::abs(exact - reference),
                      std::abs(std::nextafter(reference, 2.0 * reference) - reference))
                << "seed " << seed;
        }
    }
}

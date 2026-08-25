// SPDX-FileCopyrightText: Copyright (c) Stanford University, The Regents of the University of California, and others.
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @file test_QuadratureGenerators.cpp
 * @brief Shared test support for bounded one-dimensional quadrature generators.
 */

#include <gtest/gtest.h>

#include "FE/Common/FEException.h"
#include "FE/Quadrature/QuadratureRule.h"

#include <cmath>
#include <cstddef>
#include <exception>
#include <string_view>
#include <type_traits>
#include <utility>

using namespace svmp::FE;
using namespace svmp::FE::quadrature;

namespace {

constexpr double kStructureTolerance = 1.0e-12;
constexpr double kMomentTolerance = 2.0e-12;

enum class LineEndpointPolicy {
    Excluded,
    Included,
};

double analytic_line_monomial_integral(std::size_t power)
{
    if (power % 2u != 0u) {
        return 0.0;
    }
    return 2.0 / (static_cast<double>(power) + 1.0);
}

double accumulate_line_moment(const QuadratureRule& rule, std::size_t power)
{
    long double sum = 0.0L;
    long double correction = 0.0L;

    for (std::size_t point_index = 0;
         point_index < rule.num_points();
         ++point_index) {
        const long double coordinate =
            static_cast<long double>(rule.point(point_index)[0]);
        const long double term =
            static_cast<long double>(rule.weight(point_index)) *
            std::pow(coordinate, static_cast<int>(power));
        const long double next_sum = sum + term;

        if (std::abs(sum) >= std::abs(term)) {
            correction += (sum - next_sum) + term;
        } else {
            correction += (term - next_sum) + sum;
        }
        sum = next_sum;
    }

    return static_cast<double>(sum + correction);
}

void expect_common_line_metadata(
    const QuadratureRule& rule,
    std::size_t expected_num_points,
    int expected_exactness)
{
    EXPECT_EQ(rule.cell_family(), svmp::CellFamily::Line);
    EXPECT_EQ(rule.dimension(), 1u);
    EXPECT_DOUBLE_EQ(rule.reference_cell_measure(), 2.0);
    EXPECT_EQ(rule.polynomial_exactness(), expected_exactness);
    ASSERT_EQ(rule.num_points(), expected_num_points);
    ASSERT_EQ(rule.points().size(), expected_num_points);
    ASSERT_EQ(rule.weights().size(), expected_num_points);

    for (std::size_t point_index = 0;
         point_index < rule.num_points();
         ++point_index) {
        SCOPED_TRACE(::testing::Message() << "point index=" << point_index);
        EXPECT_DOUBLE_EQ(rule.point(point_index)[1], 0.0);
        EXPECT_DOUBLE_EQ(rule.point(point_index)[2], 0.0);
    }
}

void expect_line_rule_invariants(
    const QuadratureRule& rule,
    LineEndpointPolicy endpoint_policy,
    double tolerance = kStructureTolerance)
{
    ASSERT_GT(rule.num_points(), 0u);
    ASSERT_EQ(rule.points().size(), rule.weights().size());

    for (std::size_t point_index = 0;
         point_index < rule.num_points();
         ++point_index) {
        SCOPED_TRACE(::testing::Message() << "point index=" << point_index);

        const double coordinate = rule.point(point_index)[0];
        const double weight = rule.weight(point_index);
        EXPECT_TRUE(std::isfinite(coordinate));
        EXPECT_TRUE(std::isfinite(weight));
        EXPECT_GT(weight, 0.0);

        if (endpoint_policy == LineEndpointPolicy::Included) {
            EXPECT_GE(coordinate, -1.0);
            EXPECT_LE(coordinate, 1.0);
            if (point_index > 0u &&
                point_index + 1u < rule.num_points()) {
                EXPECT_GT(coordinate, -1.0);
                EXPECT_LT(coordinate, 1.0);
            }
        } else {
            EXPECT_GT(coordinate, -1.0);
            EXPECT_LT(coordinate, 1.0);
        }

        if (point_index > 0u) {
            EXPECT_LT(
                rule.point(point_index - 1u)[0],
                coordinate);
        }

        const std::size_t mirror_index =
            rule.num_points() - 1u - point_index;
        EXPECT_NEAR(
            coordinate,
            -rule.point(mirror_index)[0],
            tolerance);
        EXPECT_NEAR(weight, rule.weight(mirror_index), tolerance);
    }

    if (endpoint_policy == LineEndpointPolicy::Included) {
        ASSERT_GE(rule.num_points(), 2u);
        EXPECT_DOUBLE_EQ(rule.point(0)[0], -1.0);
        EXPECT_DOUBLE_EQ(rule.point(rule.num_points() - 1u)[0], 1.0);
    }

    if (rule.num_points() % 2u == 1u) {
        EXPECT_DOUBLE_EQ(rule.point(rule.num_points() / 2u)[0], 0.0);
    }

    EXPECT_NEAR(
        accumulate_line_moment(rule, 0u),
        rule.reference_cell_measure(),
        tolerance);
}

void expect_advertised_line_exactness(
    const QuadratureRule& rule,
    double tolerance = kMomentTolerance)
{
    ASSERT_GE(rule.polynomial_exactness(), 0);

    for (int power = 0;
         power <= rule.polynomial_exactness();
         ++power) {
        SCOPED_TRACE(::testing::Message() << "monomial power=" << power);
        const std::size_t nonnegative_power =
            static_cast<std::size_t>(power);
        EXPECT_NEAR(
            accumulate_line_moment(rule, nonnegative_power),
            analytic_line_monomial_integral(nonnegative_power),
            tolerance);
    }
}

template <typename ExceptionType, typename Function>
void expect_exception_with_message(
    Function&& function,
    std::string_view expected_substring)
{
    static_assert(std::is_base_of_v<std::exception, ExceptionType>);
    ASSERT_FALSE(expected_substring.empty());

    try {
        std::forward<Function>(function)();
        FAIL() << "Expected requested exception containing: "
               << expected_substring;
    } catch (const ExceptionType& exception) {
        const std::string_view actual_message{exception.what()};
        EXPECT_NE(
            actual_message.find(expected_substring),
            std::string_view::npos)
            << "actual message: " << actual_message;
    } catch (const std::exception& exception) {
        FAIL() << "Received a different exception type: " << exception.what();
    } catch (...) {
        FAIL() << "Received an unknown exception type";
    }
}

} // namespace

TEST(QuadratureGeneratorTestSupport, ExercisesSharedLineRuleChecks)
{
    const double abscissa = std::sqrt(3.0 / 5.0);
    const QuadratureRule interior_rule(
        svmp::CellFamily::Line,
        5,
        {{-abscissa, 0.0, 0.0},
         {0.0, 0.0, 0.0},
         {abscissa, 0.0, 0.0}},
        {5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0});
    const QuadratureRule endpoint_rule(
        svmp::CellFamily::Line,
        1,
        {{-1.0, 0.0, 0.0}, {1.0, 0.0, 0.0}},
        {1.0, 1.0});

    EXPECT_DOUBLE_EQ(analytic_line_monomial_integral(0u), 2.0);
    EXPECT_DOUBLE_EQ(analytic_line_monomial_integral(1u), 0.0);
    EXPECT_DOUBLE_EQ(analytic_line_monomial_integral(2u), 2.0 / 3.0);
    EXPECT_DOUBLE_EQ(analytic_line_monomial_integral(255u), 0.0);
    EXPECT_NEAR(
        accumulate_line_moment(interior_rule, 255u),
        0.0,
        kMomentTolerance);

    expect_common_line_metadata(interior_rule, 3u, 5);
    expect_line_rule_invariants(
        interior_rule,
        LineEndpointPolicy::Excluded);
    expect_advertised_line_exactness(interior_rule);

    expect_common_line_metadata(endpoint_rule, 2u, 1);
    expect_line_rule_invariants(
        endpoint_rule,
        LineEndpointPolicy::Included);
    expect_advertised_line_exactness(endpoint_rule);
    expect_exception_with_message<InvalidArgumentException>(
        [] {
            (void)QuadratureRule(
                svmp::CellFamily::Line, 1, {}, {});
        },
        "at least one point");
}

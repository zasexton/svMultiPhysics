// SPDX-FileCopyrightText: Copyright (c) Stanford University, The Regents of the University of California, and others.
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @file GaussQuadrature.cpp
 * @brief Bounded generation and validation of Gauss-Legendre line rules.
 * @ingroup FE_Quadrature
 */

#include "FE/Quadrature/GaussQuadrature.h"

#include "FE/Common/FEException.h"

#include <cmath>
#include <cstddef>
#include <sstream>
#include <string_view>
#include <utility>
#include <vector>

namespace svmp::FE::quadrature {
namespace {

constexpr int kMaximumNewtonIterations = 100;
constexpr double kNewtonCorrectionTolerance = 1.0e-14;
constexpr double kRuleValidationTolerance = 1.0e-12;

struct LegendreEvaluation {
    double value;
    double derivative;
};

LegendreEvaluation evaluate_legendre_with_derivative(
    int degree,
    double coordinate) noexcept
{
    double previous_value = 1.0;
    double previous_derivative = 0.0;

    if (degree == 0) {
        return {previous_value, previous_derivative};
    }

    double value = coordinate;
    double derivative = 1.0;
    for (int recurrence_degree = 2;
         recurrence_degree <= degree;
         ++recurrence_degree) {
        const double degree_value =
            static_cast<double>(recurrence_degree);
        const double recurrence_factor =
            static_cast<double>(2 * recurrence_degree - 1);
        const double next_value =
            (recurrence_factor * coordinate * value -
             static_cast<double>(recurrence_degree - 1) * previous_value) /
            degree_value;
        const double next_derivative =
            (recurrence_factor * (value + coordinate * derivative) -
             static_cast<double>(recurrence_degree - 1) *
                 previous_derivative) /
            degree_value;

        previous_value = value;
        previous_derivative = derivative;
        value = next_value;
        derivative = next_derivative;
    }

    return {value, derivative};
}

[[noreturn]] void raise_generation_failure(
    int num_points,
    int root_index,
    int iteration,
    std::string_view diagnostic_name,
    double diagnostic_value,
    std::string_view detail)
{
    std::ostringstream message;
    message << "Gauss-Legendre generator: " << detail
            << ", num_points=" << num_points
            << ", root_index=" << root_index
            << ", " << diagnostic_name << '=' << diagnostic_value;

    const double residual = std::isfinite(diagnostic_value)
                                ? std::abs(diagnostic_value)
                                : 0.0;
    svmp::raise<ConvergenceException>(
        message.str(), iteration, residual);
}

void validate_num_points(int num_points)
{
    if (num_points < 1 ||
        num_points > max_gauss_legendre_points()) {
        std::ostringstream message;
        message << "Gauss-Legendre generator: num_points must be in [1, "
                << max_gauss_legendre_points() << ']';
        svmp::raise<InvalidArgumentException>(message.str());
    }
}

void validate_generated_rule(
    int num_points,
    const std::vector<QuadPoint>& points,
    const std::vector<double>& weights)
{
    const std::size_t expected_size =
        static_cast<std::size_t>(num_points);
    if (points.size() != expected_size) {
        raise_generation_failure(
            num_points,
            -1,
            -1,
            "generated_point_count",
            static_cast<double>(points.size()),
            "generated point storage has the wrong size");
    }
    if (weights.size() != expected_size) {
        raise_generation_failure(
            num_points,
            -1,
            -1,
            "generated_weight_count",
            static_cast<double>(weights.size()),
            "generated weight storage has the wrong size");
    }

    long double weight_sum = 0.0L;
    long double weight_sum_correction = 0.0L;
    for (std::size_t point_index = 0;
         point_index < expected_size;
         ++point_index) {
        const QuadPoint& point = points[point_index];
        const double coordinate = point[0];
        const double weight = weights[point_index];
        const int root_index = static_cast<int>(point_index);

        if (!std::isfinite(coordinate)) {
            raise_generation_failure(
                num_points,
                root_index,
                -1,
                "coordinate",
                coordinate,
                "generated a non-finite point");
        }
        if (point[1] != 0.0 || point[2] != 0.0) {
            const double inactive_coordinate =
                point[1] != 0.0 ? point[1] : point[2];
            raise_generation_failure(
                num_points,
                root_index,
                -1,
                "inactive_coordinate",
                inactive_coordinate,
                "generated a nonzero inactive coordinate");
        }
        if (!std::isfinite(weight) || weight <= 0.0) {
            raise_generation_failure(
                num_points,
                root_index,
                -1,
                "weight",
                weight,
                "generated a non-finite or non-positive weight");
        }
        if (coordinate <= -1.0 || coordinate >= 1.0) {
            raise_generation_failure(
                num_points,
                root_index,
                -1,
                "coordinate",
                coordinate,
                "generated a point outside the open reference interval");
        }
        if (point_index > 0u) {
            const double spacing =
                coordinate - points[point_index - 1u][0];
            if (!std::isfinite(spacing) || spacing <= 0.0) {
                raise_generation_failure(
                    num_points,
                    root_index,
                    -1,
                    "point_spacing",
                    spacing,
                    "generated points are not strictly increasing");
            }
        }

        const std::size_t mirror_index =
            expected_size - 1u - point_index;
        const double point_symmetry_error =
            std::abs(coordinate + points[mirror_index][0]);
        if (!std::isfinite(point_symmetry_error) ||
            point_symmetry_error > kRuleValidationTolerance) {
            raise_generation_failure(
                num_points,
                root_index,
                -1,
                "point_symmetry_error",
                point_symmetry_error,
                "generated points are not symmetric");
        }
        const double weight_symmetry_error =
            std::abs(weight - weights[mirror_index]);
        if (!std::isfinite(weight_symmetry_error) ||
            weight_symmetry_error > kRuleValidationTolerance) {
            raise_generation_failure(
                num_points,
                root_index,
                -1,
                "weight_symmetry_error",
                weight_symmetry_error,
                "generated weights are not symmetric");
        }

        const long double weight_term = static_cast<long double>(weight);
        const long double next_weight_sum = weight_sum + weight_term;
        if (std::abs(weight_sum) >= std::abs(weight_term)) {
            weight_sum_correction +=
                (weight_sum - next_weight_sum) + weight_term;
        } else {
            weight_sum_correction +=
                (weight_term - next_weight_sum) + weight_sum;
        }
        weight_sum = next_weight_sum;
    }

    if (expected_size % 2u == 1u) {
        const std::size_t center_index = expected_size / 2u;
        if (points[center_index][0] != 0.0) {
            raise_generation_failure(
                num_points,
                static_cast<int>(center_index),
                -1,
                "center_coordinate",
                points[center_index][0],
                "generated an inexact center point");
        }
    }

    const long double corrected_weight_sum =
        weight_sum + weight_sum_correction;
    const long double measure_error =
        std::abs(corrected_weight_sum - 2.0L);
    if (!std::isfinite(corrected_weight_sum) ||
        measure_error >
            static_cast<long double>(kRuleValidationTolerance)) {
        raise_generation_failure(
            num_points,
            -1,
            -1,
            "measure_error",
            static_cast<double>(measure_error),
            "generated weights do not reproduce the reference measure");
    }
}

} // namespace

QuadratureRule make_gauss_legendre_rule(int num_points)
{
    validate_num_points(num_points);

    const std::size_t point_count =
        static_cast<std::size_t>(num_points);
    std::vector<QuadPoint> points(
        point_count, QuadPoint::Zero());
    std::vector<double> weights(point_count);

    const double pi = std::acos(-1.0);
    const int roots_to_refine = (num_points + 1) / 2;
    for (int root_index = 0;
         root_index < roots_to_refine;
         ++root_index) {
        double root = std::cos(
            pi * (static_cast<double>(root_index) + 0.75) /
            (static_cast<double>(num_points) + 0.5));
        if (!std::isfinite(root)) {
            raise_generation_failure(
                num_points,
                root_index,
                -1,
                "initial_root",
                root,
                "computed a non-finite asymptotic root seed");
        }

        bool converged = false;
        double correction = 0.0;
        int iterations_used = 0;
        for (int iteration = 1;
             iteration <= kMaximumNewtonIterations;
             ++iteration) {
            iterations_used = iteration;
            const LegendreEvaluation evaluation =
                evaluate_legendre_with_derivative(num_points, root);
            if (!std::isfinite(evaluation.value)) {
                raise_generation_failure(
                    num_points,
                    root_index,
                    iteration,
                    "polynomial_value",
                    evaluation.value,
                    "encountered a non-finite Legendre value");
            }
            if (!std::isfinite(evaluation.derivative) ||
                evaluation.derivative == 0.0) {
                raise_generation_failure(
                    num_points,
                    root_index,
                    iteration,
                    "polynomial_derivative",
                    evaluation.derivative,
                    "encountered an invalid Legendre derivative");
            }

            correction = evaluation.value / evaluation.derivative;
            if (!std::isfinite(correction)) {
                raise_generation_failure(
                    num_points,
                    root_index,
                    iteration,
                    "newton_correction",
                    correction,
                    "computed a non-finite Newton correction");
            }

            const double updated_root = root - correction;
            if (!std::isfinite(updated_root)) {
                raise_generation_failure(
                    num_points,
                    root_index,
                    iteration,
                    "updated_root",
                    updated_root,
                    "computed a non-finite Newton update");
            }
            root = updated_root;

            if (std::abs(correction) <=
                kNewtonCorrectionTolerance) {
                converged = true;
                break;
            }
        }

        if (!converged) {
            raise_generation_failure(
                num_points,
                root_index,
                kMaximumNewtonIterations,
                "newton_correction",
                correction,
                "Newton refinement did not converge");
        }

        const std::size_t left_index =
            static_cast<std::size_t>(root_index);
        const std::size_t right_index =
            point_count - 1u - left_index;
        if (left_index == right_index) {
            root = 0.0;
        }
        if (!std::isfinite(root) || root < 0.0 || root >= 1.0 ||
            (left_index != right_index && root == 0.0)) {
            raise_generation_failure(
                num_points,
                root_index,
                iterations_used,
                "refined_root",
                root,
                "refined root is outside the expected half interval");
        }

        const LegendreEvaluation final_evaluation =
            evaluate_legendre_with_derivative(num_points, root);
        if (!std::isfinite(final_evaluation.value)) {
            raise_generation_failure(
                num_points,
                root_index,
                iterations_used,
                "final_polynomial_value",
                final_evaluation.value,
                "refined root produced a non-finite Legendre value");
        }
        if (!std::isfinite(final_evaluation.derivative) ||
            final_evaluation.derivative == 0.0) {
            raise_generation_failure(
                num_points,
                root_index,
                iterations_used,
                "final_polynomial_derivative",
                final_evaluation.derivative,
                "refined root produced an invalid Legendre derivative");
        }

        const double final_correction =
            final_evaluation.value / final_evaluation.derivative;
        if (!std::isfinite(final_correction) ||
            std::abs(final_correction) >
                kNewtonCorrectionTolerance) {
            raise_generation_failure(
                num_points,
                root_index,
                iterations_used,
                "final_correction",
                final_correction,
                "refined root failed final correction validation");
        }

        const double interval_factor =
            (1.0 - root) * (1.0 + root);
        const double denominator =
            interval_factor * final_evaluation.derivative *
            final_evaluation.derivative;
        if (!std::isfinite(interval_factor) || interval_factor <= 0.0 ||
            !std::isfinite(denominator) || denominator <= 0.0) {
            raise_generation_failure(
                num_points,
                root_index,
                iterations_used,
                "weight_denominator",
                denominator,
                "refined root produced an invalid weight denominator");
        }

        const double weight = 2.0 / denominator;
        if (!std::isfinite(weight) || weight <= 0.0) {
            raise_generation_failure(
                num_points,
                root_index,
                iterations_used,
                "weight",
                weight,
                "refined root produced a non-finite or non-positive weight");
        }

        if (left_index == right_index) {
            points[left_index][0] = 0.0;
            weights[left_index] = weight;
        } else {
            points[left_index][0] = -root;
            points[right_index][0] = root;
            weights[left_index] = weight;
            weights[right_index] = weight;
        }
    }

    validate_generated_rule(num_points, points, weights);

    const int polynomial_exactness = 2 * num_points - 1;
    return QuadratureRule(
        svmp::CellFamily::Line,
        polynomial_exactness,
        std::move(points),
        std::move(weights));
}

} // namespace svmp::FE::quadrature

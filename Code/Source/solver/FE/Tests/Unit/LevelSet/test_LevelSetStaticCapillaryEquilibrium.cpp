#include <gtest/gtest.h>

#include "LevelSet/LevelSetStaticCapillaryEquilibrium.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace FE = svmp::FE;
namespace level_set = svmp::FE::level_set;
using EvaluationPurpose =
    level_set::LevelSetStaticCapillaryEvaluationPurpose;

std::uint64_t coefficientRevision(std::span<const FE::Real> coefficients)
{
    static_assert(sizeof(FE::Real) == sizeof(std::uint64_t));
    std::uint64_t revision = 1469598103934665603ull;
    for (const auto value : coefficients) {
        revision ^= std::bit_cast<std::uint64_t>(value);
        revision *= 1099511628211ull;
    }
    return revision == 0u ? 1u : revision;
}

level_set::LevelSetStaticCapillaryEquilibriumEvaluation
quadraticCapillaryEvaluation(
    std::span<const FE::Real> coefficients,
    EvaluationPurpose purpose,
    bool topology_barrier = false,
    bool constraint_barrier = false,
    bool provide_functional_derivatives = false)
{
    level_set::LevelSetStaticCapillaryEquilibriumEvaluation evaluation;
    if (coefficients.size() != 2u) {
        evaluation.diagnostic = "unexpected_coefficient_count";
        return evaluation;
    }
    const FE::Real x = coefficients[0];
    const FE::Real y = coefficients[1];
    if (!std::isfinite(x) || !std::isfinite(y)) {
        evaluation.diagnostic = "nonfinite_candidate";
        return evaluation;
    }

    const FE::Real energy_x = x - FE::Real{1.0};
    const FE::Real energy_y = y - FE::Real{2.0};
    const FE::Real gradient_x = FE::Real{2.0} * energy_x;
    const FE::Real gradient_y = FE::Real{2.0} * energy_y;
    const FE::Real multiplier =
        -(gradient_x + gradient_y) / FE::Real{2.0};
    const FE::Real residual_x = gradient_x + multiplier;
    const FE::Real residual_y = gradient_y + multiplier;
    const FE::Real residual_norm =
        std::hypot(residual_x, residual_y);
    const FE::Real gradient_norm =
        std::hypot(gradient_x, gradient_y);

    evaluation.success = true;
    evaluation.snapshot_revision_key =
        coefficientRevision(coefficients);
    evaluation.cut_topology_key =
        topology_barrier && x < FE::Real{0.9} ? 22u : 11u;
    evaluation.constraint_semantics_key =
        constraint_barrier && x < FE::Real{0.9} ? 44u : 33u;
    evaluation.surface_wall_energy =
        energy_x * energy_x + energy_y * energy_y;
    evaluation.liquid_volume = x + y;
    evaluation.functional_derivatives_available =
        provide_functional_derivatives;
    if (provide_functional_derivatives) {
        evaluation.physical_potential_derivative = {
            gradient_x, gradient_y};
        evaluation.liquid_volume_derivative = {
            FE::Real{1.0}, FE::Real{1.0}};
    }
    evaluation.pressure_representability_available =
        purpose == EvaluationPurpose::AcceptanceCertificate;
    evaluation.pressure_representability_converged =
        evaluation.pressure_representability_available;
    evaluation.pressure_representability_residual_norm =
        evaluation.pressure_representability_available
            ? residual_norm
            : std::numeric_limits<FE::Real>::quiet_NaN();
    evaluation.pressure_representability_relative_distance =
        evaluation.pressure_representability_available
            ? residual_norm /
                  std::max(FE::Real{1.0}, gradient_norm)
            : std::numeric_limits<FE::Real>::quiet_NaN();
    evaluation.production_residual_norm =
        evaluation.pressure_representability_available
            ? residual_norm
            : std::numeric_limits<FE::Real>::quiet_NaN();
    evaluation.constant_pressure_kkt_available =
        purpose == EvaluationPurpose::AcceptanceCertificate;
    if (evaluation.constant_pressure_kkt_available) {
        evaluation.constant_pressure_kkt_residual_norm =
            residual_norm;
        evaluation.constant_pressure_kkt_relative_distance =
            residual_norm /
            std::max(FE::Real{1.0}, gradient_norm);
    }
    evaluation.production_force_projection_applied = false;
    evaluation.diagnostic = "available";
    return evaluation;
}

level_set::LevelSetStaticCapillaryEquilibriumOptions
quadraticOptions()
{
    level_set::LevelSetStaticCapillaryEquilibriumOptions options;
    options.target_liquid_volume = 3.0;
    options.volume_tolerance = 1.0e-10;
    options.projected_gradient_tolerance = 1.0e-7;
    options.pressure_representability_max_residual_norm = 1.0e-7;
    options.pressure_representability_max_relative_distance = 1.0e-7;
    options.physical_equilibrium_max_residual_norm = 1.0e-7;
    options.constant_pressure_kkt_max_residual_norm =
        1.0e-7;
    options.constant_pressure_kkt_max_relative_distance =
        1.0e-7;
    options.finite_difference_reference_coefficient_scale = 1.0;
    options.finite_difference_relative_step = 1.0e-6;
    options.minimum_finite_difference_step = 1.0e-12;
    options.finite_difference_max_shrinks = 12;
    options.max_iterations = 80;
    options.max_line_search_iterations = 24;
    options.projected_gradient_inverse_stiffness = 1.0;
    options.tangent_trust_radius = 0.5;
    options.maximum_coefficient_update_linf = 1.0;
    options.line_search_shrink = 0.5;
    options.armijo_fraction = 1.0e-4;
    options.minimum_volume_merit_penalty = 2.0;
    return options;
}

TEST(LevelSetStaticCapillaryEquilibrium,
     ConvergesAtFixedVolumeAndAssignsOnlyTheAcceptedCandidate)
{
    const std::vector<FE::Real> input{2.5, 0.5};
    const std::array<std::size_t, 2> active{0u, 1u};
    std::vector<FE::Real> accepted{91.0, 92.0, 93.0};

    const auto result =
        level_set::minimizeLevelSetStaticCapillaryEquilibrium(
            quadraticOptions(),
            input,
            active,
            [](std::span<const FE::Real> coefficients,
               EvaluationPurpose purpose) {
                return quadraticCapillaryEvaluation(
                    coefficients, purpose);
            },
            accepted);

    ASSERT_TRUE(result.success) << result.diagnostic;
    EXPECT_TRUE(result.converged);
    EXPECT_TRUE(result.accepted_coefficients_assigned);
    ASSERT_EQ(accepted.size(), 2u);
    EXPECT_NEAR(accepted[0], 1.0, 1.0e-7);
    EXPECT_NEAR(accepted[1], 2.0, 1.0e-7);
    EXPECT_DOUBLE_EQ(input[0], 2.5);
    EXPECT_DOUBLE_EQ(input[1], 0.5);
    EXPECT_NEAR(result.final_liquid_volume, 3.0, 1.0e-10);
    EXPECT_NEAR(result.final_surface_wall_energy, 0.0, 1.0e-13);
    EXPECT_LE(result.final_projected_gradient_norm, 1.0e-7);
    EXPECT_TRUE(result.final_constant_pressure_kkt_available);
    EXPECT_LE(
        result.final_constant_pressure_kkt_relative_distance,
        1.0e-7);
    EXPECT_GT(result.functional_evaluations, 1u);
    EXPECT_EQ(result.acceptance_certificate_evaluations, 1u);
    EXPECT_GT(result.limited_memory_updates, 0u);
    EXPECT_GT(result.limited_memory_peak_history, 0u);
    EXPECT_LE(
        result.limited_memory_peak_history,
        static_cast<std::size_t>(
            quadraticOptions().limited_memory_history_size));
    EXPECT_EQ(result.cut_topology_key, 11u);
    EXPECT_EQ(result.constraint_semantics_key, 33u);
    RecordProperty(
        "static_capillary_minimizer_production_force_projected", 0);
}

TEST(LevelSetStaticCapillaryEquilibrium,
     UsesSuppliedFunctionalDerivativesWithoutDifferenceTrials)
{
    const std::vector<FE::Real> input{2.5, 0.5};
    const std::array<std::size_t, 2> active{0u, 1u};
    std::vector<FE::Real> accepted;

    const auto result =
        level_set::minimizeLevelSetStaticCapillaryEquilibrium(
            quadraticOptions(),
            input,
            active,
            [](std::span<const FE::Real> coefficients,
               EvaluationPurpose purpose) {
                return quadraticCapillaryEvaluation(
                    coefficients,
                    purpose,
                    /*topology_barrier=*/false,
                    /*constraint_barrier=*/false,
                    /*provide_functional_derivatives=*/true);
            },
            accepted);

    ASSERT_TRUE(result.success) << result.diagnostic;
    ASSERT_EQ(accepted.size(), 2u);
    EXPECT_NEAR(accepted[0], 1.0, 1.0e-7);
    EXPECT_NEAR(accepted[1], 2.0, 1.0e-7);
    EXPECT_GT(result.analytic_derivative_evaluations, 0u);
    EXPECT_EQ(result.finite_difference_fourth_order_components, 0u);
    EXPECT_DOUBLE_EQ(result.minimum_finite_difference_step_used, 0.0);
    EXPECT_DOUBLE_EQ(result.maximum_finite_difference_step_used, 0.0);
}

TEST(LevelSetStaticCapillaryEquilibrium,
     RefinesParameterStationarityUntilThePhysicalCertificatePasses)
{
    const std::vector<FE::Real> input{1.05, 1.95};
    const std::array<std::size_t, 2> active{0u, 1u};
    std::vector<FE::Real> accepted;
    auto options = quadraticOptions();
    options.projected_gradient_tolerance = 0.2;
    options.pressure_representability_max_residual_norm = 1.0e-6;
    options.pressure_representability_max_relative_distance = 1.0e-6;
    options.physical_equilibrium_max_residual_norm = 1.0e-6;
    options.constant_pressure_kkt_max_residual_norm = 1.0e-6;
    options.constant_pressure_kkt_max_relative_distance = 1.0e-6;
    options.projected_gradient_inverse_stiffness = 0.25;
    options.limited_memory_history_size = 0;

    const auto result =
        level_set::minimizeLevelSetStaticCapillaryEquilibrium(
            options,
            input,
            active,
            [](std::span<const FE::Real> coefficients,
               EvaluationPurpose purpose) {
                return quadraticCapillaryEvaluation(
                    coefficients,
                    purpose,
                    /*topology_barrier=*/false,
                    /*constraint_barrier=*/false,
                    /*provide_functional_derivatives=*/true);
            },
            accepted);

    ASSERT_TRUE(result.success) << result.diagnostic;
    ASSERT_EQ(accepted.size(), 2u);
    EXPECT_NEAR(accepted[0], 1.0, 1.0e-6);
    EXPECT_NEAR(accepted[1], 2.0, 1.0e-6);
    EXPECT_GT(result.iterations, 0);
    EXPECT_EQ(result.acceptance_certificate_evaluations, 2u);
    EXPECT_LE(result.final_projected_gradient_norm, 1.0e-6);
    EXPECT_LE(result.final_production_residual_norm, 1.0e-6);
}

TEST(LevelSetStaticCapillaryEquilibrium,
     ZeroLimitedMemoryHistoryRetainsTheSafeguardedGradientRoute)
{
    const std::vector<FE::Real> input{2.5, 0.5};
    const std::array<std::size_t, 2> active{0u, 1u};
    std::vector<FE::Real> accepted;
    auto options = quadraticOptions();
    options.limited_memory_history_size = 0;

    const auto result =
        level_set::minimizeLevelSetStaticCapillaryEquilibrium(
            options,
            input,
            active,
            [](std::span<const FE::Real> coefficients,
               EvaluationPurpose purpose) {
                return quadraticCapillaryEvaluation(
                    coefficients, purpose);
            },
            accepted);

    ASSERT_TRUE(result.success) << result.diagnostic;
    ASSERT_EQ(accepted.size(), 2u);
    EXPECT_NEAR(accepted[0], 1.0, 1.0e-7);
    EXPECT_NEAR(accepted[1], 2.0, 1.0e-7);
    EXPECT_EQ(result.limited_memory_updates, 0u);
    EXPECT_EQ(result.limited_memory_peak_history, 0u);
}

TEST(LevelSetStaticCapillaryEquilibrium,
     ExactDerivativesResolveRoundoffScaleMeritDecrease)
{
    const std::vector<FE::Real> input{2.5, 0.5};
    const std::array<std::size_t, 2> active{0u, 1u};
    std::vector<FE::Real> accepted;
    constexpr FE::Real energy_scale{1.0e-16};
    auto options = quadraticOptions();
    options.projected_gradient_tolerance = 1.0e-20;
    options.projected_gradient_inverse_stiffness =
        FE::Real{0.5} / energy_scale;
    options.tangent_trust_radius = 4.0;
    options.maximum_coefficient_update_linf = 4.0;
    options.limited_memory_history_size = 0;
    options.max_iterations = 4;

    const auto result =
        level_set::minimizeLevelSetStaticCapillaryEquilibrium(
            options,
            input,
            active,
            [](std::span<const FE::Real> coefficients,
               EvaluationPurpose purpose) {
                auto evaluation = quadraticCapillaryEvaluation(
                    coefficients,
                    purpose,
                    /*topology_barrier=*/false,
                    /*constraint_barrier=*/false,
                    /*provide_functional_derivatives=*/true);
                evaluation.surface_wall_energy =
                    coefficients[0] < FE::Real{2.0}
                        ? std::nextafter(
                              FE::Real{1.0},
                              std::numeric_limits<FE::Real>::infinity())
                        : FE::Real{1.0};
                for (auto& derivative :
                     evaluation.physical_potential_derivative) {
                    derivative *= energy_scale;
                }
                if (evaluation.pressure_representability_available) {
                    evaluation.pressure_representability_residual_norm *=
                        energy_scale;
                    evaluation.pressure_representability_relative_distance *=
                        energy_scale;
                    evaluation.production_residual_norm *= energy_scale;
                    evaluation.constant_pressure_kkt_residual_norm *=
                        energy_scale;
                    evaluation.constant_pressure_kkt_relative_distance *=
                        energy_scale;
                }
                return evaluation;
            },
            accepted);

    ASSERT_TRUE(result.success) << result.diagnostic;
    ASSERT_EQ(accepted.size(), 2u);
    EXPECT_NEAR(accepted[0], 1.0, 1.0e-14);
    EXPECT_NEAR(accepted[1], 2.0, 1.0e-14);
    EXPECT_GT(result.derivative_resolution_step_acceptances, 0u);
    EXPECT_GT(result.analytic_derivative_evaluations, 0u);
    EXPECT_EQ(result.finite_difference_fourth_order_components, 0u);
}

TEST(LevelSetStaticCapillaryEquilibrium,
     RejectsMalformedExactDerivativesBeforeChangingOutput)
{
    const std::vector<FE::Real> input{2.5, 0.5};
    const std::array<std::size_t, 2> active{0u, 1u};
    const std::vector<FE::Real> sentinel{41.0, 42.0, 43.0};
    std::vector<FE::Real> accepted = sentinel;

    const auto result =
        level_set::minimizeLevelSetStaticCapillaryEquilibrium(
            quadraticOptions(),
            input,
            active,
            [](std::span<const FE::Real> coefficients,
               EvaluationPurpose purpose) {
                auto evaluation = quadraticCapillaryEvaluation(
                    coefficients,
                    purpose,
                    /*topology_barrier=*/false,
                    /*constraint_barrier=*/false,
                    /*provide_functional_derivatives=*/true);
                evaluation.liquid_volume_derivative.pop_back();
                return evaluation;
            },
            accepted);

    EXPECT_FALSE(result.success);
    EXPECT_EQ(accepted, sentinel);
    EXPECT_NE(
        result.diagnostic.find(
            "candidate_evaluator_returned_invalid_functional_derivatives"),
        std::string::npos);
}

TEST(LevelSetStaticCapillaryEquilibrium,
     GravitationalPotentialCanOwnTheObjectiveWithAConstrainedPressureCertificate)
{
    const std::vector<FE::Real> input{2.5, 0.5};
    const std::array<std::size_t, 2> active{0u, 1u};
    std::vector<FE::Real> accepted{71.0, 72.0, 73.0};

    const auto result =
        level_set::minimizeLevelSetStaticCapillaryEquilibrium(
            quadraticOptions(),
            input,
            active,
            [](std::span<const FE::Real> coefficients,
               EvaluationPurpose purpose) {
                auto evaluation = quadraticCapillaryEvaluation(
                    coefficients, purpose);
                evaluation.gravitational_potential_energy =
                    evaluation.surface_wall_energy;
                evaluation.surface_wall_energy = FE::Real{0.0};
                evaluation.constant_pressure_kkt_required = false;
                evaluation.constant_pressure_kkt_available = false;
                evaluation.constant_pressure_kkt_residual_norm =
                    std::numeric_limits<FE::Real>::quiet_NaN();
                evaluation.constant_pressure_kkt_relative_distance =
                    std::numeric_limits<FE::Real>::quiet_NaN();
                return evaluation;
            },
            accepted);

    ASSERT_TRUE(result.success) << result.diagnostic;
    ASSERT_EQ(accepted.size(), 2u);
    EXPECT_NEAR(accepted[0], 1.0, 1.0e-7);
    EXPECT_NEAR(accepted[1], 2.0, 1.0e-7);
    EXPECT_DOUBLE_EQ(result.final_surface_wall_energy, 0.0);
    EXPECT_NEAR(
        result.final_gravitational_potential_energy, 0.0, 1.0e-13);
    EXPECT_NEAR(
        result.final_physical_potential_energy, 0.0, 1.0e-13);
    EXPECT_TRUE(result.final_pressure_representability_available);
    EXPECT_TRUE(result.final_pressure_representability_converged);
    EXPECT_FALSE(result.final_pressure_representability_breakdown);
    EXPECT_FALSE(result.final_constant_pressure_kkt_required);
    EXPECT_FALSE(result.final_constant_pressure_kkt_available);
}

TEST(LevelSetStaticCapillaryEquilibrium,
     BacktracksAcrossTopologyChangingTrialsWithoutPublishingThem)
{
    const std::vector<FE::Real> input{2.5, 0.5};
    const std::array<std::size_t, 2> active{0u, 1u};
    std::vector<FE::Real> accepted{17.0};
    auto options = quadraticOptions();
    options.tangent_trust_radius = 4.0;
    options.maximum_coefficient_update_linf = 4.0;

    const auto result =
        level_set::minimizeLevelSetStaticCapillaryEquilibrium(
            options,
            input,
            active,
            [](std::span<const FE::Real> coefficients,
               EvaluationPurpose purpose) {
                return quadraticCapillaryEvaluation(
                    coefficients,
                    purpose,
                    /*topology_barrier=*/true);
            },
            accepted);

    ASSERT_TRUE(result.success) << result.diagnostic;
    ASSERT_EQ(accepted.size(), 2u);
    EXPECT_NEAR(accepted[0], 1.0, 1.0e-7);
    EXPECT_NEAR(accepted[1], 2.0, 1.0e-7);
    EXPECT_GT(result.topology_change_rejections, 0u);
    EXPECT_GT(result.line_search_rejections, 0u);
    EXPECT_EQ(result.cut_topology_key, 11u);
    EXPECT_EQ(result.constraint_semantics_key, 33u);
}

TEST(LevelSetStaticCapillaryEquilibrium,
     ExactDerivativesAdvanceAcrossABoundedTopologyEpoch)
{
    const std::vector<FE::Real> input{0.8, 2.2};
    const std::array<std::size_t, 2> active{0u, 1u};
    std::vector<FE::Real> accepted{19.0, 20.0, 21.0};
    auto options = quadraticOptions();
    options.allow_topology_epoch_transitions = true;
    options.max_topology_epoch_transitions = 2;
    options.max_iterations = 120;

    const auto result =
        level_set::minimizeLevelSetStaticCapillaryEquilibrium(
            options,
            input,
            active,
            [](std::span<const FE::Real> coefficients,
               EvaluationPurpose purpose) {
                return quadraticCapillaryEvaluation(
                    coefficients,
                    purpose,
                    /*topology_barrier=*/true,
                    /*constraint_barrier=*/false,
                    /*provide_functional_derivatives=*/true);
            },
            accepted);

    ASSERT_TRUE(result.success) << result.diagnostic;
    ASSERT_EQ(accepted.size(), 2u);
    EXPECT_NEAR(accepted[0], 1.0, 1.0e-7);
    EXPECT_NEAR(accepted[1], 2.0, 1.0e-7);
    EXPECT_EQ(result.topology_epoch_transitions, 1u);
    EXPECT_GT(result.topology_change_rejections, 0u);
    EXPECT_EQ(result.cut_topology_key, 11u);
    EXPECT_EQ(result.constraint_semantics_key, 33u);
}

TEST(LevelSetStaticCapillaryEquilibrium,
     TopologyEpochPermissionDoesNotAdmitConstraintOnlyChanges)
{
    const std::vector<FE::Real> input{0.8, 2.2};
    const std::array<std::size_t, 2> active{0u, 1u};
    const std::vector<FE::Real> sentinel{23.0, 24.0, 25.0};
    std::vector<FE::Real> accepted = sentinel;
    auto options = quadraticOptions();
    options.allow_topology_epoch_transitions = true;
    options.max_topology_epoch_transitions = 2;
    options.max_iterations = 120;

    const auto result =
        level_set::minimizeLevelSetStaticCapillaryEquilibrium(
            options,
            input,
            active,
            [](std::span<const FE::Real> coefficients,
               EvaluationPurpose purpose) {
                return quadraticCapillaryEvaluation(
                    coefficients,
                    purpose,
                    /*topology_barrier=*/false,
                    /*constraint_barrier=*/true,
                    /*provide_functional_derivatives=*/true);
            },
            accepted);

    EXPECT_FALSE(result.success);
    EXPECT_EQ(accepted, sentinel);
    EXPECT_EQ(result.topology_epoch_transitions, 0u);
    EXPECT_EQ(result.topology_change_rejections, 0u);
    EXPECT_GT(result.constraint_change_rejections, 0u);
    EXPECT_NE(
        result.diagnostic.find(
            "candidate_constraint_semantics_changed"),
        std::string::npos);
}

TEST(LevelSetStaticCapillaryEquilibrium,
     BacktracksAcrossConstraintChangingTrialsWithoutPublishingThem)
{
    const std::vector<FE::Real> input{2.5, 0.5};
    const std::array<std::size_t, 2> active{0u, 1u};
    std::vector<FE::Real> accepted{27.0};
    auto options = quadraticOptions();
    options.tangent_trust_radius = 4.0;
    options.maximum_coefficient_update_linf = 4.0;

    const auto result =
        level_set::minimizeLevelSetStaticCapillaryEquilibrium(
            options,
            input,
            active,
            [](std::span<const FE::Real> coefficients,
               EvaluationPurpose purpose) {
                return quadraticCapillaryEvaluation(
                    coefficients,
                    purpose,
                    /*topology_barrier=*/false,
                    /*constraint_barrier=*/true);
            },
            accepted);

    ASSERT_TRUE(result.success) << result.diagnostic;
    ASSERT_EQ(accepted.size(), 2u);
    EXPECT_NEAR(accepted[0], 1.0, 1.0e-7);
    EXPECT_NEAR(accepted[1], 2.0, 1.0e-7);
    EXPECT_EQ(result.topology_change_rejections, 0u);
    EXPECT_GT(result.constraint_change_rejections, 0u);
    EXPECT_GT(result.line_search_rejections, 0u);
    EXPECT_EQ(result.constraint_semantics_key, 33u);
}

TEST(LevelSetStaticCapillaryEquilibrium,
     RejectsAParameterStationaryGeometryThatFailsTheAbsolutePhysicalKktGate)
{
    const std::vector<FE::Real> input{1.0, 2.0};
    const std::array<std::size_t, 2> active{0u, 1u};
    const std::vector<FE::Real> sentinel{41.0, 42.0, 43.0};
    std::vector<FE::Real> accepted = sentinel;

    const auto result =
        level_set::minimizeLevelSetStaticCapillaryEquilibrium(
            quadraticOptions(),
            input,
            active,
            [](std::span<const FE::Real> coefficients,
               EvaluationPurpose purpose) {
                auto evaluation =
                    quadraticCapillaryEvaluation(
                        coefficients, purpose);
                evaluation.constant_pressure_kkt_residual_norm =
                    0.25;
                evaluation.constant_pressure_kkt_relative_distance =
                    0.0;
                return evaluation;
            },
            accepted);

    EXPECT_FALSE(result.success);
    EXPECT_FALSE(result.converged);
    EXPECT_FALSE(result.accepted_coefficients_assigned);
    EXPECT_EQ(
        result.diagnostic,
        "constant_pressure_kkt_gate_failed_at_parameter_stationary_geometry");
    EXPECT_EQ(accepted, sentinel);
}

TEST(LevelSetStaticCapillaryEquilibrium,
     RejectsAParameterStationaryGeometryThatFailsTheScaledPhysicalKktGate)
{
    const std::vector<FE::Real> input{1.0, 2.0};
    const std::array<std::size_t, 2> active{0u, 1u};
    const std::vector<FE::Real> sentinel{44.0, 45.0, 46.0};
    std::vector<FE::Real> accepted = sentinel;

    const auto result =
        level_set::minimizeLevelSetStaticCapillaryEquilibrium(
            quadraticOptions(),
            input,
            active,
            [](std::span<const FE::Real> coefficients,
               EvaluationPurpose purpose) {
                auto evaluation =
                    quadraticCapillaryEvaluation(
                        coefficients, purpose);
                evaluation.constant_pressure_kkt_residual_norm =
                    0.0;
                evaluation.constant_pressure_kkt_relative_distance =
                    0.25;
                return evaluation;
            },
            accepted);

    EXPECT_FALSE(result.success);
    EXPECT_FALSE(result.converged);
    EXPECT_FALSE(result.accepted_coefficients_assigned);
    EXPECT_EQ(
        result.diagnostic,
        "constant_pressure_kkt_gate_failed_at_parameter_stationary_geometry");
    EXPECT_EQ(accepted, sentinel);
}

TEST(LevelSetStaticCapillaryEquilibrium,
     RejectsAParameterStationaryGeometryThatFailsTheAbsolutePressureRangeGate)
{
    const std::vector<FE::Real> input{1.0, 2.0};
    const std::array<std::size_t, 2> active{0u, 1u};
    const std::vector<FE::Real> sentinel{45.0, 46.0, 47.0};
    std::vector<FE::Real> accepted = sentinel;

    const auto result =
        level_set::minimizeLevelSetStaticCapillaryEquilibrium(
            quadraticOptions(),
            input,
            active,
            [](std::span<const FE::Real> coefficients,
               EvaluationPurpose purpose) {
                auto evaluation = quadraticCapillaryEvaluation(
                    coefficients, purpose);
                evaluation.pressure_representability_residual_norm = 0.25;
                evaluation.pressure_representability_relative_distance = 0.0;
                evaluation.production_residual_norm = 0.0;
                return evaluation;
            },
            accepted);

    EXPECT_FALSE(result.success);
    EXPECT_FALSE(result.converged);
    EXPECT_FALSE(result.accepted_coefficients_assigned);
    EXPECT_EQ(
        result.diagnostic,
        "pressure_representability_gate_failed_at_parameter_stationary_geometry");
    EXPECT_EQ(accepted, sentinel);
}

TEST(LevelSetStaticCapillaryEquilibrium,
     RejectsAParameterStationaryGeometryThatFailsTheProductionResidualGate)
{
    const std::vector<FE::Real> input{1.0, 2.0};
    const std::array<std::size_t, 2> active{0u, 1u};
    const std::vector<FE::Real> sentinel{48.0, 49.0, 50.0};
    std::vector<FE::Real> accepted = sentinel;

    const auto result =
        level_set::minimizeLevelSetStaticCapillaryEquilibrium(
            quadraticOptions(),
            input,
            active,
            [](std::span<const FE::Real> coefficients,
               EvaluationPurpose purpose) {
                auto evaluation = quadraticCapillaryEvaluation(
                    coefficients, purpose);
                evaluation.pressure_representability_residual_norm = 0.0;
                evaluation.pressure_representability_relative_distance = 0.0;
                evaluation.production_residual_norm = 0.25;
                return evaluation;
            },
            accepted);

    EXPECT_FALSE(result.success);
    EXPECT_FALSE(result.converged);
    EXPECT_FALSE(result.accepted_coefficients_assigned);
    EXPECT_EQ(
        result.diagnostic,
        "physical_equilibrium_residual_gate_failed_at_parameter_stationary_geometry");
    EXPECT_EQ(accepted, sentinel);
}

TEST(LevelSetStaticCapillaryEquilibrium,
     RejectsAnAcceptanceCertificateThatChangesTheFunctionals)
{
    const std::vector<FE::Real> input{1.0, 2.0};
    const std::array<std::size_t, 2> active{0u, 1u};
    const std::vector<FE::Real> sentinel{47.0, 48.0};
    std::vector<FE::Real> accepted = sentinel;

    const auto result =
        level_set::minimizeLevelSetStaticCapillaryEquilibrium(
            quadraticOptions(),
            input,
            active,
            [](std::span<const FE::Real> coefficients,
               EvaluationPurpose purpose) {
                auto evaluation =
                    quadraticCapillaryEvaluation(
                        coefficients, purpose);
                if (purpose ==
                    EvaluationPurpose::AcceptanceCertificate) {
                    evaluation.surface_wall_energy += 1.0e-12;
                }
                return evaluation;
            },
            accepted);

    EXPECT_FALSE(result.success);
    EXPECT_EQ(
        result.diagnostic,
        "acceptance_certificate_functionals_not_reproducible");
    EXPECT_EQ(result.acceptance_certificate_evaluations, 1u);
    EXPECT_EQ(accepted, sentinel);
}

TEST(LevelSetStaticCapillaryEquilibrium,
     RejectsAnAcceptanceCertificateThatChangesConstraintSemantics)
{
    const std::vector<FE::Real> input{1.0, 2.0};
    const std::array<std::size_t, 2> active{0u, 1u};
    const std::vector<FE::Real> sentinel{49.0, 50.0};
    std::vector<FE::Real> accepted = sentinel;

    const auto result =
        level_set::minimizeLevelSetStaticCapillaryEquilibrium(
            quadraticOptions(),
            input,
            active,
            [](std::span<const FE::Real> coefficients,
               EvaluationPurpose purpose) {
                auto evaluation =
                    quadraticCapillaryEvaluation(
                        coefficients, purpose);
                if (purpose ==
                    EvaluationPurpose::AcceptanceCertificate) {
                    ++evaluation.constraint_semantics_key;
                }
                return evaluation;
            },
            accepted);

    EXPECT_FALSE(result.success);
    EXPECT_EQ(
        result.diagnostic,
        "acceptance_certificate_constraint_semantics_changed");
    EXPECT_EQ(result.constraint_change_rejections, 1u);
    EXPECT_EQ(accepted, sentinel);
}

TEST(LevelSetStaticCapillaryEquilibrium,
     RejectsAnAcceptanceCertificateThatChangesCutTopology)
{
    const std::vector<FE::Real> input{1.0, 2.0};
    const std::array<std::size_t, 2> active{0u, 1u};
    const std::vector<FE::Real> sentinel{55.0, 56.0};
    std::vector<FE::Real> accepted = sentinel;

    const auto result =
        level_set::minimizeLevelSetStaticCapillaryEquilibrium(
            quadraticOptions(),
            input,
            active,
            [](std::span<const FE::Real> coefficients,
               EvaluationPurpose purpose) {
                auto evaluation =
                    quadraticCapillaryEvaluation(
                        coefficients, purpose);
                if (purpose ==
                    EvaluationPurpose::AcceptanceCertificate) {
                    ++evaluation.cut_topology_key;
                }
                return evaluation;
            },
            accepted);

    EXPECT_FALSE(result.success);
    EXPECT_EQ(
        result.diagnostic,
        "acceptance_certificate_cut_topology_changed");
    EXPECT_EQ(result.topology_change_rejections, 1u);
    EXPECT_EQ(accepted, sentinel);
}

TEST(LevelSetStaticCapillaryEquilibrium,
     RejectsAParameterStationaryGeometryWithoutAPhysicalKktCertificate)
{
    const std::vector<FE::Real> input{1.0, 2.0};
    const std::array<std::size_t, 2> active{0u, 1u};
    const std::vector<FE::Real> sentinel{53.0, 54.0};
    std::vector<FE::Real> accepted = sentinel;

    const auto result =
        level_set::minimizeLevelSetStaticCapillaryEquilibrium(
            quadraticOptions(),
            input,
            active,
            [](std::span<const FE::Real> coefficients,
               EvaluationPurpose purpose) {
                auto evaluation =
                    quadraticCapillaryEvaluation(
                        coefficients, purpose);
                evaluation.constant_pressure_kkt_available = false;
                return evaluation;
            },
            accepted);

    EXPECT_FALSE(result.success);
    EXPECT_EQ(
        result.diagnostic,
        "constant_pressure_kkt_unavailable_at_parameter_stationary_geometry");
    EXPECT_EQ(result.acceptance_certificate_evaluations, 1u);
    EXPECT_EQ(accepted, sentinel);
}

TEST(LevelSetStaticCapillaryEquilibrium,
     RejectsProductionForceProjectionIntroducedOnlyByTheCertificate)
{
    const std::vector<FE::Real> input{1.0, 2.0};
    const std::array<std::size_t, 2> active{0u, 1u};
    const std::vector<FE::Real> sentinel{57.0, 58.0};
    std::vector<FE::Real> accepted = sentinel;

    const auto result =
        level_set::minimizeLevelSetStaticCapillaryEquilibrium(
            quadraticOptions(),
            input,
            active,
            [](std::span<const FE::Real> coefficients,
               EvaluationPurpose purpose) {
                auto evaluation =
                    quadraticCapillaryEvaluation(
                        coefficients, purpose);
                evaluation.production_force_projection_applied =
                    purpose ==
                    EvaluationPurpose::AcceptanceCertificate;
                return evaluation;
            },
            accepted);

    EXPECT_FALSE(result.success);
    EXPECT_EQ(
        result.diagnostic,
        "production_force_projection_is_forbidden");
    EXPECT_EQ(result.acceptance_certificate_evaluations, 1u);
    EXPECT_EQ(accepted, sentinel);
}

TEST(LevelSetStaticCapillaryEquilibrium,
     LeavesOutputUntouchedWhenFixedTopologyDifferencesCannotBeEvaluated)
{
    const std::vector<FE::Real> input{2.5, 0.5};
    const std::array<std::size_t, 2> active{0u, 1u};
    const std::vector<FE::Real> sentinel{51.0, 52.0};
    std::vector<FE::Real> accepted = sentinel;

    const auto result =
        level_set::minimizeLevelSetStaticCapillaryEquilibrium(
            quadraticOptions(),
            input,
            active,
            [&input](std::span<const FE::Real> coefficients,
                     EvaluationPurpose purpose) {
                if (coefficients.size() != input.size() ||
                    coefficients[0] != input[0] ||
                    coefficients[1] != input[1]) {
                    level_set::LevelSetStaticCapillaryEquilibriumEvaluation
                        rejected;
                    rejected.diagnostic =
                        "synthetic_candidate_rejection";
                    return rejected;
                }
                return quadraticCapillaryEvaluation(
                    coefficients, purpose);
            },
            accepted);

    EXPECT_FALSE(result.success);
    EXPECT_EQ(accepted, sentinel);
    EXPECT_NE(
        result.diagnostic.find(
            "fixed_topology_functional_derivative_unavailable"),
        std::string::npos);
    EXPECT_NE(
        result.diagnostic.find(
            "synthetic_candidate_rejection"),
        std::string::npos);
}

TEST(LevelSetStaticCapillaryEquilibrium,
     RejectsAnyEvaluatorThatProjectsTheProductionForce)
{
    const std::vector<FE::Real> input{2.5, 0.5};
    const std::array<std::size_t, 2> active{0u, 1u};
    const std::vector<FE::Real> sentinel{61.0};
    std::vector<FE::Real> accepted = sentinel;

    const auto result =
        level_set::minimizeLevelSetStaticCapillaryEquilibrium(
            quadraticOptions(),
            input,
            active,
            [](std::span<const FE::Real> coefficients,
               EvaluationPurpose purpose) {
                auto evaluation =
                    quadraticCapillaryEvaluation(
                        coefficients, purpose);
                evaluation.production_force_projection_applied = true;
                return evaluation;
            },
            accepted);

    EXPECT_FALSE(result.success);
    EXPECT_EQ(
        result.diagnostic,
        "production_force_projection_is_forbidden");
    EXPECT_EQ(accepted, sentinel);
}

TEST(LevelSetStaticCapillaryEquilibrium,
     InvalidActiveIndexSetFailsBeforeChangingOutput)
{
    const std::vector<FE::Real> input{2.5, 0.5};
    const std::array<std::size_t, 2> duplicated_active{0u, 0u};
    const std::vector<FE::Real> sentinel{71.0, 72.0};
    std::vector<FE::Real> accepted = sentinel;

    EXPECT_THROW(
        (void)level_set::minimizeLevelSetStaticCapillaryEquilibrium(
            quadraticOptions(),
            input,
            duplicated_active,
            [](std::span<const FE::Real> coefficients,
               EvaluationPurpose purpose) {
                return quadraticCapillaryEvaluation(
                    coefficients, purpose);
            },
            accepted),
        std::invalid_argument);
    EXPECT_EQ(accepted, sentinel);

    auto zero_volume_options = quadraticOptions();
    zero_volume_options.target_liquid_volume = 0.0;
    const std::array<std::size_t, 2> valid_active{0u, 1u};
    EXPECT_THROW(
        (void)level_set::minimizeLevelSetStaticCapillaryEquilibrium(
            zero_volume_options,
            input,
            valid_active,
            [](std::span<const FE::Real> coefficients,
               EvaluationPurpose purpose) {
                return quadraticCapillaryEvaluation(
                    coefficients, purpose);
            },
            accepted),
        std::invalid_argument);
    EXPECT_EQ(accepted, sentinel);

    auto invalid_epoch_options = quadraticOptions();
    invalid_epoch_options.max_topology_epoch_transitions = -1;
    EXPECT_THROW(
        (void)level_set::minimizeLevelSetStaticCapillaryEquilibrium(
            invalid_epoch_options,
            input,
            valid_active,
            [](std::span<const FE::Real> coefficients,
               EvaluationPurpose purpose) {
                return quadraticCapillaryEvaluation(
                    coefficients, purpose);
            },
            accepted),
        std::invalid_argument);
    EXPECT_EQ(accepted, sentinel);
}

} // namespace

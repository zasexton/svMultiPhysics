#include <gtest/gtest.h>

#include "LevelSet/LevelSetVolumeConstrainedMinimizer.h"

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
    level_set::LevelSetVolumeConstrainedEvaluationPurpose;

// The static cap as the Navier-Stokes driver states it: two energy
// components and three pressure certificates, in this order, with the limits
// of its options.
constexpr std::size_t kSurfaceWallEnergy = 0u;
constexpr std::size_t kGravitationalPotentialEnergy = 1u;
constexpr std::size_t kPressureRepresentability = 0u;
constexpr std::size_t kPhysicalEquilibrium = 1u;
constexpr std::size_t kConstantPressureKkt = 2u;

struct CapillaryOptions : level_set::LevelSetVolumeConstrainedMinimizerOptions {
    FE::Real pressure_representability_max_residual_norm{1.0e-10};
    FE::Real pressure_representability_max_relative_distance{1.0e-8};
    FE::Real physical_equilibrium_max_residual_norm{1.0e-10};
    FE::Real constant_pressure_kkt_max_residual_norm{1.0e-10};
    FE::Real constant_pressure_kkt_max_relative_distance{1.0e-8};
};

level_set::LevelSetVolumeConstrainedProblem
capillaryProblem(const CapillaryOptions& options)
{
    level_set::LevelSetVolumeConstrainedProblem problem;
    problem.energy_components = {"surface_wall_energy",
                                 "gravitational_potential_energy"};
    problem.acceptance_checks.resize(3u);
    auto& pressure = problem.acceptance_checks[kPressureRepresentability];
    pressure.name = "pressure_representability";
    pressure.limits = {options.pressure_representability_max_residual_norm,
                       options.pressure_representability_max_relative_distance};
    pressure.invalid_state_diagnostic =
        "candidate_evaluator_returned_invalid_pressure_representability_state";
    pressure.unavailable_diagnostic =
        "pressure_representability_unavailable_at_parameter_stationary_geometry";
    pressure.failed_diagnostic =
        "pressure_representability_gate_failed_at_parameter_stationary_geometry";
    auto& equilibrium = problem.acceptance_checks[kPhysicalEquilibrium];
    equilibrium.name = "physical_equilibrium_residual";
    equilibrium.limits = {options.physical_equilibrium_max_residual_norm};
    equilibrium.failed_diagnostic =
        "physical_equilibrium_residual_gate_failed_at_parameter_stationary_geometry";
    auto& kkt = problem.acceptance_checks[kConstantPressureKkt];
    kkt.name = "constant_pressure_kkt";
    kkt.limits = {options.constant_pressure_kkt_max_residual_norm,
                  options.constant_pressure_kkt_max_relative_distance};
    kkt.invalid_state_diagnostic =
        "candidate_evaluator_returned_invalid_constant_pressure_kkt_state";
    kkt.unavailable_diagnostic =
        "constant_pressure_kkt_unavailable_at_parameter_stationary_geometry";
    kkt.failed_diagnostic =
        "constant_pressure_kkt_gate_failed_at_parameter_stationary_geometry";
    problem.forbidden_substitution_evaluation_diagnostic =
        "candidate_evaluator_projected_production_force";
    problem.forbidden_substitution_diagnostic =
        "production_force_projection_is_forbidden";
    return problem;
}

// A successful evaluation starts from a NaN surface energy, zero gravity
// energy and unmeasured certificates.
level_set::LevelSetVolumeConstrainedEvaluation capillaryEvaluation()
{
    constexpr auto nan = std::numeric_limits<FE::Real>::quiet_NaN();
    level_set::LevelSetVolumeConstrainedEvaluation evaluation;
    evaluation.energy_components = {nan, FE::Real{0.0}};
    evaluation.checks.resize(3u);
    evaluation.checks[kPressureRepresentability].converged = false;
    evaluation.checks[kPressureRepresentability].values = {nan, nan};
    evaluation.checks[kPhysicalEquilibrium].available = true;
    evaluation.checks[kPhysicalEquilibrium].values = {nan};
    evaluation.checks[kConstantPressureKkt].values = {nan, nan};
    return evaluation;
}

level_set::LevelSetVolumeConstrainedMinimizerResult minimizeCapillary(
    const CapillaryOptions& options,
    std::span<const FE::Real> input_coefficients,
    std::span<const std::size_t> active_coefficient_indices,
    const level_set::LevelSetVolumeConstrainedEvaluator& evaluator,
    std::vector<FE::Real>& accepted_coefficients)
{
    return level_set::minimizeLevelSetVolumeConstrained(
        options,
        capillaryProblem(options),
        input_coefficients,
        active_coefficient_indices,
        evaluator,
        accepted_coefficients);
}

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

level_set::LevelSetVolumeConstrainedEvaluation
quadraticCapillaryEvaluation(
    std::span<const FE::Real> coefficients,
    EvaluationPurpose purpose,
    bool topology_barrier = false,
    bool constraint_barrier = false,
    bool provide_functional_derivatives = false)
{
    auto evaluation = capillaryEvaluation();
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
    evaluation.energy_components[kSurfaceWallEnergy] =
        energy_x * energy_x + energy_y * energy_y;
    evaluation.liquid_volume = x + y;
    evaluation.functional_derivatives_available =
        provide_functional_derivatives;
    if (provide_functional_derivatives) {
        evaluation.energy_derivative = {
            gradient_x, gradient_y};
        evaluation.liquid_volume_derivative = {
            FE::Real{1.0}, FE::Real{1.0}};
    }
    evaluation.checks[kPressureRepresentability].available =
        purpose == EvaluationPurpose::AcceptanceCertificate;
    evaluation.checks[kPressureRepresentability].converged =
        evaluation.checks[kPressureRepresentability].available;
    evaluation.checks[kPressureRepresentability].values[0] =
        evaluation.checks[kPressureRepresentability].available
            ? residual_norm
            : std::numeric_limits<FE::Real>::quiet_NaN();
    evaluation.checks[kPressureRepresentability].values[1] =
        evaluation.checks[kPressureRepresentability].available
            ? residual_norm /
                  std::max(FE::Real{1.0}, gradient_norm)
            : std::numeric_limits<FE::Real>::quiet_NaN();
    evaluation.checks[kPhysicalEquilibrium].values[0] =
        evaluation.checks[kPressureRepresentability].available
            ? residual_norm
            : std::numeric_limits<FE::Real>::quiet_NaN();
    evaluation.checks[kConstantPressureKkt].available =
        purpose == EvaluationPurpose::AcceptanceCertificate;
    if (evaluation.checks[kConstantPressureKkt].available) {
        evaluation.checks[kConstantPressureKkt].values[0] =
            residual_norm;
        evaluation.checks[kConstantPressureKkt].values[1] =
            residual_norm /
            std::max(FE::Real{1.0}, gradient_norm);
    }
    evaluation.forbidden_substitution_applied = false;
    evaluation.diagnostic = "available";
    return evaluation;
}

CapillaryOptions
quadraticOptions()
{
    CapillaryOptions options;
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

level_set::LevelSetVolumeConstrainedEvaluation
piecewiseTransitionEvaluation(
    std::span<const FE::Real> coefficients,
    FE::Real positive_energy_slope)
{
    auto evaluation = capillaryEvaluation();
    if (coefficients.size() != 2u) {
        evaluation.diagnostic = "unexpected_coefficient_count";
        return evaluation;
    }
    constexpr FE::Real epsilon{1.0e-6};
    const FE::Real x = coefficients[0];
    const FE::Real y = coefficients[1];
    evaluation.success = true;
    evaluation.snapshot_revision_key = coefficientRevision(coefficients);
    evaluation.cut_topology_key = x <= FE::Real{0.0} ? 101u : 202u;
    evaluation.constraint_semantics_key = 303u;
    evaluation.energy_components[kSurfaceWallEnergy] =
        x <= FE::Real{0.0}
            ? FE::Real{1.0} - x
            : FE::Real{1.0} + positive_energy_slope * epsilon * x;
    evaluation.liquid_volume = x + y;
    evaluation.functional_derivatives_available = true;
    evaluation.energy_derivative = {
        x <= FE::Real{0.0}
            ? FE::Real{-1.0}
            : positive_energy_slope * epsilon,
        FE::Real{0.0}};
    evaluation.liquid_volume_derivative = {
        FE::Real{1.0}, FE::Real{1.0}};
    evaluation.diagnostic = "available";
    return evaluation;
}

CapillaryOptions
piecewiseTransitionOptions(int line_search_trials, FE::Real shrink = 0.5)
{
    auto options = quadraticOptions();
    options.projected_gradient_tolerance = 1.0e-12;
    options.max_iterations = 1;
    options.max_line_search_iterations = line_search_trials;
    options.allow_topology_epoch_transitions = true;
    options.max_topology_epoch_transitions = 2;
    options.projected_gradient_inverse_stiffness = 1.0;
    options.tangent_trust_radius = 2.0;
    options.maximum_coefficient_update_linf = 2.0;
    options.line_search_shrink = shrink;
    options.armijo_fraction = 1.0e-4;
    options.limited_memory_history_size = 0;
    return options;
}

TEST(LevelSetVolumeConstrainedMinimizer,
     RecordsUnpublishedStateAndModelForExhaustedLineSearch)
{
    const std::vector<FE::Real> input{17.0, 0.0, 18.0, 19.0, 3.0};
    const std::array<std::size_t, 2> active{4u, 1u};
    const std::vector<FE::Real> sentinel{101.0, 102.0, 103.0};
    std::vector<FE::Real> accepted = sentinel;
    std::vector<std::vector<FE::Real>> evaluated_coefficients;
    const auto evaluator = [&evaluated_coefficients](
                               std::span<const FE::Real> c,
                               EvaluationPurpose) {
        evaluated_coefficients.emplace_back(c.begin(), c.end());
        const std::array<FE::Real, 2> compact{c[1], c[4]};
        auto e = piecewiseTransitionEvaluation(compact, FE::Real{1.0});
        const auto ge = e.energy_derivative;
        const auto gv = e.liquid_volume_derivative;
        e.snapshot_revision_key = coefficientRevision(c);
        e.energy_derivative.assign(c.size(), FE::Real{0.0});
        e.liquid_volume_derivative.assign(c.size(), FE::Real{0.0});
        e.energy_derivative[1] = ge[0];
        e.energy_derivative[4] = ge[1];
        e.liquid_volume_derivative[1] = gv[0];
        e.liquid_volume_derivative[4] = gv[1];
        return e;
    };

    const auto result =
        minimizeCapillary(
            piecewiseTransitionOptions(4),
            input,
            active,
            evaluator,
            accepted);

    EXPECT_FALSE(result.success);
    EXPECT_EQ(
        result.diagnostic,
        "capillary_merit_line_search_failed:candidate_cut_topology_changed");
    EXPECT_EQ(result.functional_evaluations, 5u);
    EXPECT_EQ(result.line_search_rejections, 4u);
    EXPECT_EQ(result.line_search_trace_total_attempt_count, 4u);
    EXPECT_EQ(result.line_search_trace_omitted_count, 0u);
    EXPECT_EQ(accepted, sentinel);

    const auto& state = result.line_search_failure_state;
    EXPECT_EQ(
        state.status,
        level_set::LevelSetVolumeConstrainedFailureStateStatus::Available);
    EXPECT_EQ(state.coefficient_count, input.size());
    EXPECT_EQ(state.active_coefficient_count, active.size());
    EXPECT_EQ(state.accepted_iteration_index, 0u);
    EXPECT_EQ(state.snapshot_revision_key, coefficientRevision(input));
    EXPECT_EQ(state.cut_topology_key, 101u);
    EXPECT_EQ(state.constraint_semantics_key, 303u);
    EXPECT_DOUBLE_EQ(state.current_merit, 1.0);
    EXPECT_DOUBLE_EQ(state.merit_penalty, 3.0);
    ASSERT_EQ(state.coefficients, input);
    ASSERT_EQ(
        state.active_coefficient_indices,
        (std::vector<std::size_t>{1u, 4u}));
    EXPECT_EQ(
        state.energy_gradient,
        (std::vector<FE::Real>{-1.0, 0.0}));
    EXPECT_EQ(
        state.volume_gradient,
        (std::vector<FE::Real>{1.0, 1.0}));
    ASSERT_EQ(
        state.direction,
        (std::vector<FE::Real>{0.5, -0.5}));

    ASSERT_EQ(evaluated_coefficients.size(), 5u);
    ASSERT_EQ(result.line_search_trace.size(), 4u);
    for (std::size_t trial = 0u;
         trial < result.line_search_trace.size();
         ++trial) {
        auto reconstructed = state.coefficients;
        for (std::size_t local = 0u;
             local < state.active_coefficient_indices.size();
             ++local) {
            reconstructed[state.active_coefficient_indices[local]] +=
                result.line_search_trace[trial].step_size *
                state.direction[local];
        }
        EXPECT_EQ(reconstructed, evaluated_coefficients[trial + 1u]);
    }
}

TEST(LevelSetVolumeConstrainedMinimizer,
     FailureStateCapacityDoesNotChangeTheNumericalOutcome)
{
    std::vector<FE::Real> input(
        level_set::kLevelSetVolumeConstrainedFailureStateCoefficientCapacity +
            1u,
        FE::Real{17.0});
    input[1] = FE::Real{0.0};
    input.back() = FE::Real{3.0};
    const std::array<std::size_t, 2> active{input.size() - 1u, 1u};
    const std::vector<FE::Real> sentinel{104.0, 105.0};
    std::vector<FE::Real> accepted = sentinel;
    const auto evaluator = [](std::span<const FE::Real> c,
                              EvaluationPurpose) {
        const std::array<FE::Real, 2> compact{c[1], c.back()};
        auto e = piecewiseTransitionEvaluation(compact, FE::Real{1.0});
        const auto ge = e.energy_derivative;
        const auto gv = e.liquid_volume_derivative;
        e.snapshot_revision_key = coefficientRevision(c);
        e.energy_derivative.assign(c.size(), FE::Real{0.0});
        e.liquid_volume_derivative.assign(c.size(), FE::Real{0.0});
        e.energy_derivative[1] = ge[0];
        e.energy_derivative.back() = ge[1];
        e.liquid_volume_derivative[1] = gv[0];
        e.liquid_volume_derivative.back() = gv[1];
        return e;
    };

    const auto result =
        minimizeCapillary(
            piecewiseTransitionOptions(4),
            input,
            active,
            evaluator,
            accepted);

    EXPECT_FALSE(result.success);
    EXPECT_EQ(
        result.diagnostic,
        "capillary_merit_line_search_failed:candidate_cut_topology_changed");
    EXPECT_EQ(result.functional_evaluations, 5u);
    EXPECT_EQ(result.line_search_rejections, 4u);
    EXPECT_EQ(result.line_search_trace_total_attempt_count, 4u);
    EXPECT_EQ(result.line_search_trace_omitted_count, 0u);
    EXPECT_EQ(accepted, sentinel);

    const auto& state = result.line_search_failure_state;
    EXPECT_EQ(
        state.status,
        level_set::LevelSetVolumeConstrainedFailureStateStatus::CapacityExceeded);
    EXPECT_EQ(state.coefficient_count, input.size());
    EXPECT_EQ(state.active_coefficient_count, active.size());
    EXPECT_TRUE(state.coefficients.empty());
    EXPECT_TRUE(state.active_coefficient_indices.empty());
    EXPECT_TRUE(state.energy_gradient.empty());
    EXPECT_TRUE(state.volume_gradient.empty());
    EXPECT_TRUE(state.direction.empty());
}

TEST(LevelSetVolumeConstrainedMinimizer,
     DoesNotRecordFailureStateForSuccessOrPreSearchExit)
{
    level_set::LevelSetVolumeConstrainedFailureState manual;
    EXPECT_EQ(
        manual.status,
        level_set::LevelSetVolumeConstrainedFailureStateStatus::NotApplicable);
    EXPECT_TRUE(manual.coefficients.empty());
    EXPECT_TRUE(manual.active_coefficient_indices.empty());
    EXPECT_TRUE(manual.energy_gradient.empty());
    EXPECT_TRUE(manual.volume_gradient.empty());
    EXPECT_TRUE(manual.direction.empty());

    const std::vector<FE::Real> input{2.5, 0.5};
    const std::array<std::size_t, 2> active{0u, 1u};
    std::vector<FE::Real> accepted;
    const auto exact_evaluator = [](
                                     std::span<const FE::Real> coefficients,
                                     EvaluationPurpose purpose) {
        return quadraticCapillaryEvaluation(
            coefficients,
            purpose,
            /*topology_barrier=*/false,
            /*constraint_barrier=*/false,
            /*provide_functional_derivatives=*/true);
    };

    const auto success =
        minimizeCapillary(
            quadraticOptions(),
            input,
            active,
            exact_evaluator,
            accepted);
    ASSERT_TRUE(success.success) << success.diagnostic;
    EXPECT_TRUE(success.accepted_coefficients_assigned);
    EXPECT_EQ(
        success.line_search_failure_state.status,
        level_set::LevelSetVolumeConstrainedFailureStateStatus::NotApplicable);
    EXPECT_TRUE(success.line_search_failure_state.coefficients.empty());
    EXPECT_TRUE(
        success.line_search_failure_state.active_coefficient_indices.empty());
    EXPECT_TRUE(success.line_search_failure_state.energy_gradient.empty());
    EXPECT_TRUE(success.line_search_failure_state.volume_gradient.empty());
    EXPECT_TRUE(success.line_search_failure_state.direction.empty());

    auto limited_options = quadraticOptions();
    limited_options.max_iterations = 1;
    limited_options.projected_gradient_inverse_stiffness = 0.25;
    limited_options.limited_memory_history_size = 0;
    const std::vector<FE::Real> sentinel{106.0, 107.0, 108.0};
    accepted = sentinel;
    const auto pre_search_exit =
        minimizeCapillary(
            limited_options,
            input,
            active,
            exact_evaluator,
            accepted);
    EXPECT_FALSE(pre_search_exit.success);
    EXPECT_EQ(pre_search_exit.iterations, 1);
    EXPECT_EQ(
        pre_search_exit.diagnostic,
        "static_capillary_equilibrium_iteration_limit_reached");
    EXPECT_EQ(accepted, sentinel);
    EXPECT_EQ(
        pre_search_exit.line_search_failure_state.status,
        level_set::LevelSetVolumeConstrainedFailureStateStatus::NotApplicable);
    EXPECT_TRUE(pre_search_exit.line_search_failure_state.coefficients.empty());
    EXPECT_TRUE(
        pre_search_exit.line_search_failure_state
            .active_coefficient_indices.empty());
    EXPECT_TRUE(
        pre_search_exit.line_search_failure_state.energy_gradient.empty());
    EXPECT_TRUE(
        pre_search_exit.line_search_failure_state.volume_gradient.empty());
    EXPECT_TRUE(pre_search_exit.line_search_failure_state.direction.empty());
}

TEST(LevelSetVolumeConstrainedMinimizer,
     RecordsCrossTopologyDecreaseThatMissesTheArmijoBound)
{
    const std::vector<FE::Real> input{0.0, 3.0};
    const std::array<std::size_t, 2> active{0u, 1u};
    const std::vector<FE::Real> sentinel{81.0, 82.0, 83.0};
    std::vector<FE::Real> accepted = sentinel;

    const auto result =
        minimizeCapillary(
            piecewiseTransitionOptions(4),
            input,
            active,
            [](std::span<const FE::Real> coefficients,
               EvaluationPurpose) {
                return piecewiseTransitionEvaluation(
                    coefficients, FE::Real{-1.0});
            },
            accepted);

    EXPECT_FALSE(result.success);
    EXPECT_EQ(
        result.diagnostic,
        "capillary_merit_line_search_failed:candidate_cut_topology_changed");
    EXPECT_EQ(accepted, sentinel);
    EXPECT_EQ(result.functional_evaluations, 5u);
    EXPECT_EQ(result.line_search_trace_total_attempt_count, 4u);
    EXPECT_EQ(result.line_search_trace_omitted_count, 0u);
    ASSERT_EQ(result.line_search_trace.size(), 4u);
    FE::Real expected_step = 1.0;
    for (std::size_t i = 0u; i < result.line_search_trace.size(); ++i) {
        const auto& record = result.line_search_trace[i];
        EXPECT_EQ(record.accepted_iteration_index, 0u);
        EXPECT_EQ(record.line_search_trial_index, i);
        EXPECT_EQ(
            record.phase,
            level_set::LevelSetVolumeConstrainedLineSearchPhase::Trial);
        EXPECT_EQ(
            record.disposition,
            level_set::LevelSetVolumeConstrainedLineSearchDisposition::
                TopologyRejected);
        EXPECT_TRUE(record.evaluation_available);
        EXPECT_FALSE(record.used_limited_memory_direction);
        EXPECT_FALSE(record.used_projected_gradient_fallback_direction);
        EXPECT_DOUBLE_EQ(record.step_size, expected_step);
        EXPECT_EQ(record.current_cut_topology_key, 101u);
        EXPECT_EQ(record.trial_cut_topology_key, 202u);
        EXPECT_EQ(record.current_constraint_semantics_key, 303u);
        EXPECT_EQ(record.trial_constraint_semantics_key, 303u);
        EXPECT_DOUBLE_EQ(record.current_merit, 1.0);
        EXPECT_LT(record.armijo_bound, record.trial_merit);
        EXPECT_LT(record.trial_merit, record.current_merit);
        EXPECT_DOUBLE_EQ(record.predicted_merit_decrease, 0.5);
        EXPECT_DOUBLE_EQ(record.trial_volume_error, 0.0);
        expected_step *= 0.5;
    }
}

TEST(LevelSetVolumeConstrainedMinimizer,
     RecordsCrossTopologyMeritIncrease)
{
    const std::vector<FE::Real> input{0.0, 3.0};
    const std::array<std::size_t, 2> active{0u, 1u};
    const std::vector<FE::Real> sentinel{84.0, 85.0};
    std::vector<FE::Real> accepted = sentinel;

    const auto result =
        minimizeCapillary(
            piecewiseTransitionOptions(4),
            input,
            active,
            [](std::span<const FE::Real> coefficients,
               EvaluationPurpose) {
                return piecewiseTransitionEvaluation(
                    coefficients, FE::Real{1.0});
            },
            accepted);

    EXPECT_FALSE(result.success);
    EXPECT_EQ(
        result.diagnostic,
        "capillary_merit_line_search_failed:candidate_cut_topology_changed");
    EXPECT_EQ(accepted, sentinel);
    ASSERT_EQ(result.line_search_trace.size(), 4u);
    for (const auto& record : result.line_search_trace) {
        EXPECT_EQ(
            record.disposition,
            level_set::LevelSetVolumeConstrainedLineSearchDisposition::
                TopologyRejected);
        EXPECT_TRUE(record.evaluation_available);
        EXPECT_GT(record.trial_merit, record.current_merit);
        EXPECT_GT(record.trial_merit, record.armijo_bound);
    }
}

TEST(LevelSetVolumeConstrainedMinimizer,
     RetainsTheLatestSixtyFourLineSearchTrialsInOrder)
{
    const std::vector<FE::Real> input{0.0, 3.0};
    const std::array<std::size_t, 2> active{0u, 1u};
    std::vector<FE::Real> accepted{86.0};

    const auto result =
        minimizeCapillary(
            piecewiseTransitionOptions(70, FE::Real{0.99}),
            input,
            active,
            [](std::span<const FE::Real> coefficients,
               EvaluationPurpose) {
                return piecewiseTransitionEvaluation(
                    coefficients, FE::Real{-1.0});
            },
            accepted);

    EXPECT_FALSE(result.success);
    EXPECT_EQ(result.functional_evaluations, 71u);
    EXPECT_EQ(result.line_search_trace_total_attempt_count, 70u);
    EXPECT_EQ(result.line_search_trace_omitted_count, 6u);
    ASSERT_EQ(
        result.line_search_trace.size(),
        level_set::kLevelSetVolumeConstrainedLineSearchTraceCapacity);
    EXPECT_EQ(result.line_search_trace.front().line_search_trial_index, 6u);
    EXPECT_EQ(result.line_search_trace.back().line_search_trial_index, 69u);
    for (std::size_t i = 1u; i < result.line_search_trace.size(); ++i) {
        EXPECT_EQ(
            result.line_search_trace[i].line_search_trial_index,
            result.line_search_trace[i - 1u].line_search_trial_index + 1u);
    }
    EXPECT_NEAR(
        result.line_search_trace.back().step_size,
        std::pow(FE::Real{0.99}, FE::Real{69.0}),
        1.0e-15);
    EXPECT_EQ(
        result.line_search_trace.back().disposition,
        level_set::LevelSetVolumeConstrainedLineSearchDisposition::
            TopologyRejected);
}

TEST(LevelSetVolumeConstrainedMinimizer,
     MarksUnavailableTrialMeasurementsWithoutFiniteSubstitutes)
{
    const std::vector<FE::Real> input{0.0, 3.0};
    const std::array<std::size_t, 2> active{0u, 1u};
    std::vector<FE::Real> accepted{87.0};

    const auto result =
        minimizeCapillary(
            piecewiseTransitionOptions(1),
            input,
            active,
            [](std::span<const FE::Real> coefficients,
               EvaluationPurpose) {
                if (coefficients[0] > FE::Real{0.0}) {
                    level_set::LevelSetVolumeConstrainedEvaluation
                        unavailable;
                    unavailable.diagnostic = "synthetic_trial_unavailable";
                    return unavailable;
                }
                return piecewiseTransitionEvaluation(
                    coefficients, FE::Real{-1.0});
            },
            accepted);

    EXPECT_FALSE(result.success);
    EXPECT_EQ(result.functional_evaluations, 2u);
    ASSERT_EQ(result.line_search_trace.size(), 1u);
    const auto& record = result.line_search_trace.front();
    EXPECT_EQ(
        record.disposition,
        level_set::LevelSetVolumeConstrainedLineSearchDisposition::
            UnavailableOrForbidden);
    EXPECT_FALSE(record.evaluation_available);
    EXPECT_EQ(record.current_cut_topology_key, 101u);
    EXPECT_EQ(record.trial_cut_topology_key, 0u);
    EXPECT_EQ(record.current_constraint_semantics_key, 303u);
    EXPECT_EQ(record.trial_constraint_semantics_key, 0u);
    EXPECT_TRUE(std::isnan(record.trial_merit));
    EXPECT_TRUE(std::isnan(record.trial_volume_error));
    EXPECT_TRUE(std::isfinite(record.current_merit));
    EXPECT_TRUE(std::isnan(record.armijo_bound));
    EXPECT_TRUE(std::isfinite(record.predicted_merit_decrease));
}

TEST(LevelSetVolumeConstrainedMinimizer,
     RecordsTheActualRejectedTopologyReproductionWithoutAnotherProbe)
{
    const std::vector<FE::Real> input{0.0, 3.0};
    const std::array<std::size_t, 2> active{0u, 1u};
    std::vector<FE::Real> accepted{88.0};
    std::size_t positive_candidate_evaluations = 0u;

    const auto result =
        minimizeCapillary(
            piecewiseTransitionOptions(4),
            input,
            active,
            [&positive_candidate_evaluations](
                std::span<const FE::Real> coefficients,
                EvaluationPurpose) {
                auto evaluation = piecewiseTransitionEvaluation(
                    coefficients, FE::Real{-1000000.0});
                if (coefficients[0] > FE::Real{0.0} &&
                    ++positive_candidate_evaluations == 2u) {
                    evaluation.energy_components[kSurfaceWallEnergy] += FE::Real{0.125};
                }
                return evaluation;
            },
            accepted);

    EXPECT_FALSE(result.success);
    EXPECT_EQ(result.diagnostic, "topology_transition_not_reproducible");
    EXPECT_EQ(result.functional_evaluations, 3u);
    EXPECT_EQ(positive_candidate_evaluations, 2u);
    EXPECT_EQ(result.line_search_trace_total_attempt_count, 2u);
    ASSERT_EQ(result.line_search_trace.size(), 2u);
    const auto& deferred = result.line_search_trace[0];
    const auto& reproduced = result.line_search_trace[1];
    EXPECT_EQ(
        deferred.disposition,
        level_set::LevelSetVolumeConstrainedLineSearchDisposition::Deferred);
    EXPECT_EQ(
        reproduced.phase,
        level_set::LevelSetVolumeConstrainedLineSearchPhase::
            DeferredReproduction);
    EXPECT_EQ(
        reproduced.disposition,
        level_set::LevelSetVolumeConstrainedLineSearchDisposition::
            ReproductionRejected);
    EXPECT_TRUE(reproduced.evaluation_available);
    EXPECT_DOUBLE_EQ(deferred.trial_merit, 0.5);
    EXPECT_DOUBLE_EQ(reproduced.trial_merit, 0.625);
    EXPECT_EQ(
        reproduced.line_search_trial_index,
        deferred.line_search_trial_index);
}

TEST(LevelSetVolumeConstrainedMinimizer,
     ReplacesLimitedMemoryAttemptEvidenceWithFallbackAttemptEvidence)
{
    const std::vector<FE::Real> input{2.5, 0.5};
    const std::array<std::size_t, 2> active{0u, 1u};
    const std::vector<FE::Real> sentinel{89.0, 90.0};
    std::vector<FE::Real> accepted = sentinel;
    bool first_trial_returned = false;
    auto options = quadraticOptions();
    options.max_line_search_iterations = 2;

    const auto result =
        minimizeCapillary(
            options,
            input,
            active,
            [&input, &first_trial_returned](
                std::span<const FE::Real> coefficients,
                EvaluationPurpose purpose) {
                const bool initial_candidate =
                    coefficients[0] == input[0] &&
                    coefficients[1] == input[1];
                if (purpose == EvaluationPurpose::FunctionalTrial &&
                    !initial_candidate && first_trial_returned) {
                    level_set::LevelSetVolumeConstrainedEvaluation
                        unavailable;
                    unavailable.diagnostic =
                        "synthetic_trial_unavailable";
                    return unavailable;
                }
                if (purpose == EvaluationPurpose::FunctionalTrial &&
                    !initial_candidate) {
                    first_trial_returned = true;
                }
                return quadraticCapillaryEvaluation(
                    coefficients,
                    purpose,
                    /*topology_barrier=*/false,
                    /*constraint_barrier=*/false,
                    /*provide_functional_derivatives=*/true);
            },
            accepted);

    EXPECT_FALSE(result.success);
    EXPECT_EQ(accepted, sentinel);
    EXPECT_EQ(result.iterations, 1);
    EXPECT_EQ(result.functional_evaluations, 6u);
    EXPECT_EQ(result.limited_memory_updates, 1u);
    EXPECT_EQ(result.projected_gradient_fallbacks, 1u);
    EXPECT_EQ(result.line_search_trace_total_attempt_count, 2u);
    EXPECT_EQ(result.line_search_trace_omitted_count, 0u);
    ASSERT_EQ(result.line_search_trace.size(), 2u);
    for (const auto& record : result.line_search_trace) {
        EXPECT_EQ(record.accepted_iteration_index, 1u);
        EXPECT_FALSE(record.used_limited_memory_direction);
        EXPECT_TRUE(record.used_projected_gradient_fallback_direction);
        EXPECT_EQ(
            record.disposition,
            level_set::LevelSetVolumeConstrainedLineSearchDisposition::
                UnavailableOrForbidden);
    }
}

TEST(LevelSetVolumeConstrainedMinimizer,
     ClearsPriorAttemptBeforeNewDirectionTerminatesWithoutATrial)
{
    const std::vector<FE::Real> input{0.0, 3.0};
    const std::array<std::size_t, 2> active{0u, 1u};
    const std::vector<FE::Real> sentinel{93.0, 94.0};
    std::vector<FE::Real> accepted = sentinel;
    auto options = quadraticOptions();
    options.projected_gradient_tolerance = 0.0;
    options.max_iterations = 3;
    options.projected_gradient_inverse_stiffness = 1.0e210;
    options.limited_memory_history_size = 0;

    const auto result =
        minimizeCapillary(
            options,
            input,
            active,
            [](std::span<const FE::Real> coefficients,
               EvaluationPurpose) {
                auto evaluation = capillaryEvaluation();
                const bool initial =
                    coefficients[0] == FE::Real{0.0};
                evaluation.success = true;
                evaluation.snapshot_revision_key =
                    coefficientRevision(coefficients);
                evaluation.cut_topology_key = 401u;
                evaluation.constraint_semantics_key = 402u;
                evaluation.energy_components[kSurfaceWallEnergy] =
                    initial ? FE::Real{1.0} : FE::Real{0.5};
                evaluation.liquid_volume =
                    coefficients[0] + coefficients[1];
                evaluation.functional_derivatives_available = true;
                evaluation.energy_derivative = {
                    initial ? FE::Real{-1.0e-100} : FE::Real{1.0e100},
                    FE::Real{0.0}};
                evaluation.liquid_volume_derivative = {
                    FE::Real{1.0}, FE::Real{1.0}};
                evaluation.diagnostic = "available";
                return evaluation;
            },
            accepted);

    EXPECT_FALSE(result.success);
    EXPECT_EQ(
        result.diagnostic,
        "static_capillary_equilibrium_tangent_step_is_nonfinite");
    EXPECT_EQ(result.iterations, 1);
    EXPECT_EQ(result.functional_evaluations, 2u);
    EXPECT_EQ(accepted, sentinel);
    EXPECT_EQ(result.line_search_trace_total_attempt_count, 0u);
    EXPECT_EQ(result.line_search_trace_omitted_count, 0u);
    EXPECT_TRUE(result.line_search_trace.empty());
}

TEST(LevelSetVolumeConstrainedMinimizer,
     ConvergesAtFixedVolumeAndAssignsOnlyTheAcceptedCandidate)
{
    const std::vector<FE::Real> input{2.5, 0.5};
    const std::array<std::size_t, 2> active{0u, 1u};
    std::vector<FE::Real> accepted{91.0, 92.0, 93.0};

    const auto result =
        minimizeCapillary(
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
    EXPECT_NEAR(result.final_energy_components[kSurfaceWallEnergy], 0.0, 1.0e-13);
    EXPECT_LE(result.final_projected_gradient_norm, 1.0e-7);
    EXPECT_TRUE(result.final_checks[kConstantPressureKkt].available);
    EXPECT_LE(
        result.final_checks[kConstantPressureKkt].values[1],
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

TEST(LevelSetVolumeConstrainedMinimizer,
     UsesSuppliedFunctionalDerivativesWithoutDifferenceTrials)
{
    const std::vector<FE::Real> input{2.5, 0.5};
    const std::array<std::size_t, 2> active{0u, 1u};
    std::vector<FE::Real> accepted;

    const auto result =
        minimizeCapillary(
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

TEST(LevelSetVolumeConstrainedMinimizer,
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
        minimizeCapillary(
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
    EXPECT_LE(result.final_checks[kPhysicalEquilibrium].values[0], 1.0e-6);
}

TEST(LevelSetVolumeConstrainedMinimizer,
     ZeroLimitedMemoryHistoryRetainsTheSafeguardedGradientRoute)
{
    const std::vector<FE::Real> input{2.5, 0.5};
    const std::array<std::size_t, 2> active{0u, 1u};
    std::vector<FE::Real> accepted;
    auto options = quadraticOptions();
    options.limited_memory_history_size = 0;

    const auto result =
        minimizeCapillary(
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

TEST(LevelSetVolumeConstrainedMinimizer,
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
        minimizeCapillary(
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
                evaluation.energy_components[kSurfaceWallEnergy] =
                    coefficients[0] < FE::Real{2.0}
                        ? std::nextafter(
                              FE::Real{1.0},
                              std::numeric_limits<FE::Real>::infinity())
                        : FE::Real{1.0};
                for (auto& derivative :
                     evaluation.energy_derivative) {
                    derivative *= energy_scale;
                }
                if (evaluation.checks[kPressureRepresentability].available) {
                    evaluation.checks[kPressureRepresentability].values[0] *=
                        energy_scale;
                    evaluation.checks[kPressureRepresentability].values[1] *=
                        energy_scale;
                    evaluation.checks[kPhysicalEquilibrium].values[0] *= energy_scale;
                    evaluation.checks[kConstantPressureKkt].values[0] *=
                        energy_scale;
                    evaluation.checks[kConstantPressureKkt].values[1] *=
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

TEST(LevelSetVolumeConstrainedMinimizer,
     RejectsMalformedExactDerivativesBeforeChangingOutput)
{
    const std::vector<FE::Real> input{2.5, 0.5};
    const std::array<std::size_t, 2> active{0u, 1u};
    const std::vector<FE::Real> sentinel{41.0, 42.0, 43.0};
    std::vector<FE::Real> accepted = sentinel;

    const auto result =
        minimizeCapillary(
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

TEST(LevelSetVolumeConstrainedMinimizer,
     GravitationalPotentialCanOwnTheObjectiveWithAConstrainedPressureCertificate)
{
    const std::vector<FE::Real> input{2.5, 0.5};
    const std::array<std::size_t, 2> active{0u, 1u};
    std::vector<FE::Real> accepted{71.0, 72.0, 73.0};

    const auto result =
        minimizeCapillary(
            quadraticOptions(),
            input,
            active,
            [](std::span<const FE::Real> coefficients,
               EvaluationPurpose purpose) {
                auto evaluation = quadraticCapillaryEvaluation(
                    coefficients, purpose);
                evaluation.energy_components[kGravitationalPotentialEnergy] =
                    evaluation.energy_components[kSurfaceWallEnergy];
                evaluation.energy_components[kSurfaceWallEnergy] = FE::Real{0.0};
                evaluation.checks[kConstantPressureKkt].required = false;
                evaluation.checks[kConstantPressureKkt].available = false;
                evaluation.checks[kConstantPressureKkt].values[0] =
                    std::numeric_limits<FE::Real>::quiet_NaN();
                evaluation.checks[kConstantPressureKkt].values[1] =
                    std::numeric_limits<FE::Real>::quiet_NaN();
                return evaluation;
            },
            accepted);

    ASSERT_TRUE(result.success) << result.diagnostic;
    ASSERT_EQ(accepted.size(), 2u);
    EXPECT_NEAR(accepted[0], 1.0, 1.0e-7);
    EXPECT_NEAR(accepted[1], 2.0, 1.0e-7);
    EXPECT_DOUBLE_EQ(result.final_energy_components[kSurfaceWallEnergy], 0.0);
    EXPECT_NEAR(
        result.final_energy_components[kGravitationalPotentialEnergy], 0.0, 1.0e-13);
    EXPECT_NEAR(
        result.final_energy, 0.0, 1.0e-13);
    EXPECT_TRUE(result.final_checks[kPressureRepresentability].available);
    EXPECT_TRUE(result.final_checks[kPressureRepresentability].converged);
    EXPECT_FALSE(result.final_checks[kPressureRepresentability].breakdown);
    EXPECT_FALSE(result.final_checks[kConstantPressureKkt].required);
    EXPECT_FALSE(result.final_checks[kConstantPressureKkt].available);
}

TEST(LevelSetVolumeConstrainedMinimizer,
     BacktracksAcrossTopologyChangingTrialsWithoutPublishingThem)
{
    const std::vector<FE::Real> input{2.5, 0.5};
    const std::array<std::size_t, 2> active{0u, 1u};
    std::vector<FE::Real> accepted{17.0};
    auto options = quadraticOptions();
    options.tangent_trust_radius = 4.0;
    options.maximum_coefficient_update_linf = 4.0;

    const auto result =
        minimizeCapillary(
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

TEST(LevelSetVolumeConstrainedMinimizer,
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
        minimizeCapillary(
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
    EXPECT_LE(result.functional_evaluations, 8u);
    EXPECT_EQ(result.cut_topology_key, 11u);
    EXPECT_EQ(result.constraint_semantics_key, 33u);
}

TEST(LevelSetVolumeConstrainedMinimizer,
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
        minimizeCapillary(
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

TEST(LevelSetVolumeConstrainedMinimizer,
     BacktracksAcrossConstraintChangingTrialsWithoutPublishingThem)
{
    const std::vector<FE::Real> input{2.5, 0.5};
    const std::array<std::size_t, 2> active{0u, 1u};
    std::vector<FE::Real> accepted{27.0};
    auto options = quadraticOptions();
    options.tangent_trust_radius = 4.0;
    options.maximum_coefficient_update_linf = 4.0;

    const auto result =
        minimizeCapillary(
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

TEST(LevelSetVolumeConstrainedMinimizer,
     RejectsAParameterStationaryGeometryThatFailsTheAbsolutePhysicalKktGate)
{
    const std::vector<FE::Real> input{1.0, 2.0};
    const std::array<std::size_t, 2> active{0u, 1u};
    const std::vector<FE::Real> sentinel{41.0, 42.0, 43.0};
    std::vector<FE::Real> accepted = sentinel;

    const auto result =
        minimizeCapillary(
            quadraticOptions(),
            input,
            active,
            [](std::span<const FE::Real> coefficients,
               EvaluationPurpose purpose) {
                auto evaluation =
                    quadraticCapillaryEvaluation(
                        coefficients, purpose);
                evaluation.checks[kConstantPressureKkt].values[0] =
                    0.25;
                evaluation.checks[kConstantPressureKkt].values[1] =
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

TEST(LevelSetVolumeConstrainedMinimizer,
     RejectsAParameterStationaryGeometryThatFailsTheScaledPhysicalKktGate)
{
    const std::vector<FE::Real> input{1.0, 2.0};
    const std::array<std::size_t, 2> active{0u, 1u};
    const std::vector<FE::Real> sentinel{44.0, 45.0, 46.0};
    std::vector<FE::Real> accepted = sentinel;

    const auto result =
        minimizeCapillary(
            quadraticOptions(),
            input,
            active,
            [](std::span<const FE::Real> coefficients,
               EvaluationPurpose purpose) {
                auto evaluation =
                    quadraticCapillaryEvaluation(
                        coefficients, purpose);
                evaluation.checks[kConstantPressureKkt].values[0] =
                    0.0;
                evaluation.checks[kConstantPressureKkt].values[1] =
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

TEST(LevelSetVolumeConstrainedMinimizer,
     RejectsAParameterStationaryGeometryThatFailsTheAbsolutePressureRangeGate)
{
    const std::vector<FE::Real> input{1.0, 2.0};
    const std::array<std::size_t, 2> active{0u, 1u};
    const std::vector<FE::Real> sentinel{45.0, 46.0, 47.0};
    std::vector<FE::Real> accepted = sentinel;

    const auto result =
        minimizeCapillary(
            quadraticOptions(),
            input,
            active,
            [](std::span<const FE::Real> coefficients,
               EvaluationPurpose purpose) {
                auto evaluation = quadraticCapillaryEvaluation(
                    coefficients, purpose);
                evaluation.checks[kPressureRepresentability].values[0] = 0.25;
                evaluation.checks[kPressureRepresentability].values[1] = 0.0;
                evaluation.checks[kPhysicalEquilibrium].values[0] = 0.0;
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

TEST(LevelSetVolumeConstrainedMinimizer,
     RejectsAParameterStationaryGeometryThatFailsTheProductionResidualGate)
{
    const std::vector<FE::Real> input{1.0, 2.0};
    const std::array<std::size_t, 2> active{0u, 1u};
    const std::vector<FE::Real> sentinel{48.0, 49.0, 50.0};
    std::vector<FE::Real> accepted = sentinel;

    const auto result =
        minimizeCapillary(
            quadraticOptions(),
            input,
            active,
            [](std::span<const FE::Real> coefficients,
               EvaluationPurpose purpose) {
                auto evaluation = quadraticCapillaryEvaluation(
                    coefficients, purpose);
                evaluation.checks[kPressureRepresentability].values[0] = 0.0;
                evaluation.checks[kPressureRepresentability].values[1] = 0.0;
                evaluation.checks[kPhysicalEquilibrium].values[0] = 0.25;
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

TEST(LevelSetVolumeConstrainedMinimizer,
     RejectsAnAcceptanceCertificateThatChangesTheFunctionals)
{
    const std::vector<FE::Real> input{1.0, 2.0};
    const std::array<std::size_t, 2> active{0u, 1u};
    const std::vector<FE::Real> sentinel{47.0, 48.0};
    std::vector<FE::Real> accepted = sentinel;

    const auto result =
        minimizeCapillary(
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
                    evaluation.energy_components[kSurfaceWallEnergy] += 1.0e-12;
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

TEST(LevelSetVolumeConstrainedMinimizer,
     RejectsAnAcceptanceCertificateThatChangesConstraintSemantics)
{
    const std::vector<FE::Real> input{1.0, 2.0};
    const std::array<std::size_t, 2> active{0u, 1u};
    const std::vector<FE::Real> sentinel{49.0, 50.0};
    std::vector<FE::Real> accepted = sentinel;

    const auto result =
        minimizeCapillary(
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

TEST(LevelSetVolumeConstrainedMinimizer,
     RejectsAnAcceptanceCertificateThatChangesCutTopology)
{
    const std::vector<FE::Real> input{1.0, 2.0};
    const std::array<std::size_t, 2> active{0u, 1u};
    const std::vector<FE::Real> sentinel{55.0, 56.0};
    std::vector<FE::Real> accepted = sentinel;

    const auto result =
        minimizeCapillary(
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

TEST(LevelSetVolumeConstrainedMinimizer,
     RejectsAParameterStationaryGeometryWithoutAPhysicalKktCertificate)
{
    const std::vector<FE::Real> input{1.0, 2.0};
    const std::array<std::size_t, 2> active{0u, 1u};
    const std::vector<FE::Real> sentinel{53.0, 54.0};
    std::vector<FE::Real> accepted = sentinel;

    const auto result =
        minimizeCapillary(
            quadraticOptions(),
            input,
            active,
            [](std::span<const FE::Real> coefficients,
               EvaluationPurpose purpose) {
                auto evaluation =
                    quadraticCapillaryEvaluation(
                        coefficients, purpose);
                evaluation.checks[kConstantPressureKkt].available = false;
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

TEST(LevelSetVolumeConstrainedMinimizer,
     RejectsProductionForceProjectionIntroducedOnlyByTheCertificate)
{
    const std::vector<FE::Real> input{1.0, 2.0};
    const std::array<std::size_t, 2> active{0u, 1u};
    const std::vector<FE::Real> sentinel{57.0, 58.0};
    std::vector<FE::Real> accepted = sentinel;

    const auto result =
        minimizeCapillary(
            quadraticOptions(),
            input,
            active,
            [](std::span<const FE::Real> coefficients,
               EvaluationPurpose purpose) {
                auto evaluation =
                    quadraticCapillaryEvaluation(
                        coefficients, purpose);
                evaluation.forbidden_substitution_applied =
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

TEST(LevelSetVolumeConstrainedMinimizer,
     LeavesOutputUntouchedWhenFixedTopologyDifferencesCannotBeEvaluated)
{
    const std::vector<FE::Real> input{2.5, 0.5};
    const std::array<std::size_t, 2> active{0u, 1u};
    const std::vector<FE::Real> sentinel{51.0, 52.0};
    std::vector<FE::Real> accepted = sentinel;

    const auto result =
        minimizeCapillary(
            quadraticOptions(),
            input,
            active,
            [&input](std::span<const FE::Real> coefficients,
                     EvaluationPurpose purpose) {
                if (coefficients.size() != input.size() ||
                    coefficients[0] != input[0] ||
                    coefficients[1] != input[1]) {
                    level_set::LevelSetVolumeConstrainedEvaluation
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

TEST(LevelSetVolumeConstrainedMinimizer,
     RejectsAnyEvaluatorThatProjectsTheProductionForce)
{
    const std::vector<FE::Real> input{2.5, 0.5};
    const std::array<std::size_t, 2> active{0u, 1u};
    const std::vector<FE::Real> sentinel{61.0};
    std::vector<FE::Real> accepted = sentinel;

    const auto result =
        minimizeCapillary(
            quadraticOptions(),
            input,
            active,
            [](std::span<const FE::Real> coefficients,
               EvaluationPurpose purpose) {
                auto evaluation =
                    quadraticCapillaryEvaluation(
                        coefficients, purpose);
                evaluation.forbidden_substitution_applied = true;
                return evaluation;
            },
            accepted);

    EXPECT_FALSE(result.success);
    EXPECT_EQ(
        result.diagnostic,
        "production_force_projection_is_forbidden");
    EXPECT_EQ(accepted, sentinel);
}

TEST(LevelSetVolumeConstrainedMinimizer,
     InvalidActiveIndexSetFailsBeforeChangingOutput)
{
    const std::vector<FE::Real> input{2.5, 0.5};
    const std::array<std::size_t, 2> duplicated_active{0u, 0u};
    const std::vector<FE::Real> sentinel{71.0, 72.0};
    std::vector<FE::Real> accepted = sentinel;

    EXPECT_THROW(
        (void)minimizeCapillary(
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
        (void)minimizeCapillary(
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
        (void)minimizeCapillary(
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

TEST(LevelSetVolumeConstrainedMinimizer,
     MinimizesAnyNamedEnergyComponentsUnderDeclaredCertificates)
{
    // Three named components and one certificate: the stationarity residual
    // |dE/dx - dE/dy| of E = (x - 1)^2 + (y - 2)^2 + 1/2 on x + y = 3.
    level_set::LevelSetVolumeConstrainedProblem problem;
    problem.energy_components = {"x_part", "y_part", "offset"};
    level_set::LevelSetVolumeConstrainedAcceptanceCheck stationarity;
    stationarity.name = "stationarity";
    stationarity.limits = {1.0e-7};
    stationarity.unavailable_diagnostic = "stationarity_unavailable";
    stationarity.failed_diagnostic = "stationarity_failed";
    problem.acceptance_checks = {stationarity};
    problem.forbidden_substitution_evaluation_diagnostic = "forbidden_evaluation";
    problem.forbidden_substitution_diagnostic = "forbidden";

    const auto evaluation =
        [](std::span<const FE::Real> c, bool certificate_available,
           std::size_t check_count) {
            level_set::LevelSetVolumeConstrainedEvaluation e;
            e.success = true;
            e.snapshot_revision_key = coefficientRevision(c);
            e.cut_topology_key = 1u;
            e.constraint_semantics_key = 2u;
            const FE::Real dx = c[0] - FE::Real{1.0};
            const FE::Real dy = c[1] - FE::Real{2.0};
            e.energy_components = {dx * dx, dy * dy, FE::Real{0.5}};
            e.liquid_volume = c[0] + c[1];
            e.functional_derivatives_available = true;
            e.energy_derivative = {FE::Real{2.0} * dx, FE::Real{2.0} * dy};
            e.liquid_volume_derivative = {FE::Real{1.0}, FE::Real{1.0}};
            level_set::LevelSetVolumeConstrainedCheckValue check;
            check.available = certificate_available;
            check.values = {std::abs(FE::Real{2.0} * dx - FE::Real{2.0} * dy)};
            e.checks.assign(check_count, check);
            return e;
        };
    const level_set::LevelSetVolumeConstrainedMinimizerOptions options =
        quadraticOptions();
    const std::vector<FE::Real> input{2.5, 0.5};
    const std::vector<std::size_t> active{0u, 1u};

    std::vector<FE::Real> accepted;
    const auto result = level_set::minimizeLevelSetVolumeConstrained(
        options,
        problem,
        input,
        active,
        [&](std::span<const FE::Real> c, EvaluationPurpose purpose) {
            return evaluation(
                c, purpose == EvaluationPurpose::AcceptanceCertificate, 1u);
        },
        accepted);
    ASSERT_TRUE(result.success) << result.diagnostic;
    ASSERT_EQ(accepted.size(), 2u);
    EXPECT_NEAR(accepted[0], 1.0, 1.0e-7);
    EXPECT_NEAR(accepted[1], 2.0, 1.0e-7);
    ASSERT_EQ(result.initial_energy_components.size(), 3u);
    ASSERT_EQ(result.final_energy_components.size(), 3u);
    EXPECT_EQ(result.initial_energy,
              (result.initial_energy_components[0] +
               result.initial_energy_components[1]) +
                  result.initial_energy_components[2]);
    EXPECT_NEAR(result.final_energy, 0.5, 1.0e-12);
    ASSERT_EQ(result.final_checks.size(), 1u);
    EXPECT_TRUE(result.final_checks[0].available);
    EXPECT_LE(result.final_checks[0].values[0], 1.0e-7);

    const auto failure = [&](bool certificate_available,
                             std::size_t check_count) {
        std::vector<FE::Real> unchanged{7.0};
        const auto failed = level_set::minimizeLevelSetVolumeConstrained(
            options,
            problem,
            input,
            active,
            [&](std::span<const FE::Real> c, EvaluationPurpose) {
                return evaluation(c, certificate_available, check_count);
            },
            unchanged);
        EXPECT_FALSE(failed.success);
        EXPECT_EQ(unchanged, std::vector<FE::Real>{7.0});
        return failed.diagnostic;
    };
    EXPECT_EQ(failure(false, 1u), "stationarity_unavailable");
    EXPECT_EQ(failure(true, 0u),
              "initial_candidate_evaluation_failed:"
              "candidate_evaluator_returned_invalid_acceptance_checks");

    auto unnamed = problem;
    unnamed.energy_components.clear();
    std::vector<FE::Real> ignored;
    EXPECT_THROW((void)level_set::minimizeLevelSetVolumeConstrained(
                     options,
                     unnamed,
                     input,
                     active,
                     [&](std::span<const FE::Real> c, EvaluationPurpose) {
                         return evaluation(c, true, 1u);
                     },
                     ignored),
                 std::invalid_argument);
}

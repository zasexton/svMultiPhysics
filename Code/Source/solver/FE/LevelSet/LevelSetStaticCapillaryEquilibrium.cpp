#include "LevelSet/LevelSetStaticCapillaryEquilibrium.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <deque>
#include <exception>
#include <optional>
#include <stdexcept>
#include <utility>

namespace svmp::FE::level_set {
namespace {

[[nodiscard]] bool finiteNonnegative(Real value) noexcept
{
    return std::isfinite(value) && value >= Real{0.0};
}

[[nodiscard]] bool finitePositive(Real value) noexcept
{
    return std::isfinite(value) && value > Real{0.0};
}

void validateOptions(
    const LevelSetStaticCapillaryEquilibriumOptions& options,
    std::span<const Real> input_coefficients,
    std::span<const std::size_t> active_coefficient_indices,
    const LevelSetStaticCapillaryEquilibriumEvaluator& evaluator)
{
    if (!evaluator) {
        throw std::invalid_argument(
            "static capillary equilibrium minimization requires an evaluator");
    }
    if (input_coefficients.empty()) {
        throw std::invalid_argument(
            "static capillary equilibrium minimization requires coefficients");
    }
    if (active_coefficient_indices.empty()) {
        throw std::invalid_argument(
            "static capillary equilibrium minimization requires active coefficient indices");
    }
    if (!std::all_of(
            input_coefficients.begin(),
            input_coefficients.end(),
            [](Real value) { return std::isfinite(value); })) {
        throw std::invalid_argument(
            "static capillary equilibrium minimization requires finite coefficients");
    }
    if (!finitePositive(options.target_liquid_volume) ||
        !finitePositive(options.volume_tolerance) ||
        !finiteNonnegative(options.projected_gradient_tolerance) ||
        !finiteNonnegative(
            options.pressure_representability_max_residual_norm) ||
        !finiteNonnegative(
            options.pressure_representability_max_relative_distance) ||
        !finiteNonnegative(
            options.physical_equilibrium_max_residual_norm) ||
        !finiteNonnegative(
            options.constant_pressure_kkt_max_residual_norm) ||
        !finiteNonnegative(
            options.constant_pressure_kkt_max_relative_distance) ||
        !finitePositive(
            options.finite_difference_reference_coefficient_scale) ||
        !finitePositive(options.finite_difference_relative_step) ||
        !finitePositive(options.minimum_finite_difference_step) ||
        options.finite_difference_max_shrinks < 0 ||
        options.max_iterations <= 0 ||
        options.max_line_search_iterations <= 0 ||
        options.max_topology_epoch_transitions < 0 ||
        !finitePositive(
            options.projected_gradient_inverse_stiffness) ||
        !finitePositive(options.tangent_trust_radius) ||
        !finitePositive(options.maximum_coefficient_update_linf) ||
        !finitePositive(options.line_search_shrink) ||
        !(options.line_search_shrink < Real{1.0}) ||
        !finitePositive(options.armijo_fraction) ||
        !(options.armijo_fraction < Real{1.0}) ||
        options.limited_memory_history_size < 0 ||
        !finitePositive(
            options.limited_memory_curvature_tolerance) ||
        !finitePositive(options.minimum_volume_merit_penalty)) {
        throw std::invalid_argument(
            "static capillary equilibrium minimization received invalid tolerances or iteration controls");
    }

    std::vector<std::size_t> sorted_indices(
        active_coefficient_indices.begin(),
        active_coefficient_indices.end());
    std::sort(sorted_indices.begin(), sorted_indices.end());
    if (sorted_indices.back() >= input_coefficients.size() ||
        std::adjacent_find(sorted_indices.begin(), sorted_indices.end()) !=
            sorted_indices.end()) {
        throw std::invalid_argument(
            "static capillary equilibrium minimization active coefficient indices are duplicated or out of range");
    }
}

[[nodiscard]] Real dot(
    std::span<const Real> left,
    std::span<const Real> right)
{
    if (left.size() != right.size()) {
        throw std::invalid_argument(
            "static capillary equilibrium minimization vector sizes differ");
    }
    long double value = 0.0L;
    for (std::size_t i = 0u; i < left.size(); ++i) {
        value += static_cast<long double>(left[i]) *
                 static_cast<long double>(right[i]);
    }
    return static_cast<Real>(value);
}

[[nodiscard]] Real norm(std::span<const Real> values)
{
    return std::sqrt(std::max(Real{0.0}, dot(values, values)));
}

void projectOrthogonalTo(
    std::span<Real> values,
    std::span<const Real> normal,
    Real normal_squared)
{
    const Real normal_component = dot(values, normal) / normal_squared;
    for (std::size_t i = 0u; i < values.size(); ++i) {
        values[i] -= normal_component * normal[i];
    }
}

[[nodiscard]] Real physicalPotentialEnergy(
    const LevelSetStaticCapillaryEquilibriumEvaluation& evaluation) noexcept
{
    return evaluation.surface_wall_energy +
           evaluation.gravitational_potential_energy;
}

} // namespace

LevelSetStaticCapillaryEquilibriumResult
minimizeLevelSetStaticCapillaryEquilibrium(
    const LevelSetStaticCapillaryEquilibriumOptions& options,
    std::span<const Real> input_coefficients,
    std::span<const std::size_t> active_coefficient_indices,
    const LevelSetStaticCapillaryEquilibriumEvaluator& evaluator,
    std::vector<Real>& accepted_coefficients)
{
    validateOptions(
        options,
        input_coefficients,
        active_coefficient_indices,
        evaluator);

    LevelSetStaticCapillaryEquilibriumResult result;
    std::vector<std::size_t> active_indices(
        active_coefficient_indices.begin(),
        active_coefficient_indices.end());
    std::sort(active_indices.begin(), active_indices.end());
    std::vector<Real> coefficients(
        input_coefficients.begin(), input_coefficients.end());

    bool forbidden_projection_seen = false;
    std::string last_evaluation_diagnostic;
    const auto evaluate =
        [&](std::span<const Real> candidate,
            LevelSetStaticCapillaryEvaluationPurpose purpose,
            LevelSetStaticCapillaryEquilibriumEvaluation& evaluation) {
            ++result.functional_evaluations;
            if (purpose ==
                LevelSetStaticCapillaryEvaluationPurpose::
                    AcceptanceCertificate) {
                ++result.acceptance_certificate_evaluations;
            }
            try {
                evaluation = evaluator(candidate, purpose);
            } catch (const std::exception& error) {
                last_evaluation_diagnostic =
                    std::string("candidate_evaluator_exception:") +
                    error.what();
                return false;
            } catch (...) {
                last_evaluation_diagnostic =
                    "candidate_evaluator_unknown_exception";
                return false;
            }

            if (!evaluation.success) {
                last_evaluation_diagnostic =
                    evaluation.diagnostic.empty()
                        ? "candidate_evaluator_rejected"
                        : evaluation.diagnostic;
                return false;
            }
            if (evaluation.production_force_projection_applied) {
                forbidden_projection_seen = true;
                last_evaluation_diagnostic =
                    "candidate_evaluator_projected_production_force";
                return false;
            }
            if (evaluation.snapshot_revision_key == 0u ||
                evaluation.cut_topology_key == 0u ||
                evaluation.constraint_semantics_key == 0u ||
                !std::isfinite(evaluation.surface_wall_energy) ||
                !std::isfinite(
                    evaluation.gravitational_potential_energy) ||
                !std::isfinite(physicalPotentialEnergy(evaluation)) ||
                !finiteNonnegative(evaluation.liquid_volume)) {
                last_evaluation_diagnostic =
                    "candidate_evaluator_returned_invalid_functional_state";
                return false;
            }
            if (evaluation.functional_derivatives_available &&
                (evaluation.physical_potential_derivative.size() !=
                     candidate.size() ||
                 evaluation.liquid_volume_derivative.size() !=
                     candidate.size() ||
                 !std::all_of(
                     evaluation.physical_potential_derivative.begin(),
                     evaluation.physical_potential_derivative.end(),
                     [](Real value) { return std::isfinite(value); }) ||
                 !std::all_of(
                     evaluation.liquid_volume_derivative.begin(),
                     evaluation.liquid_volume_derivative.end(),
                     [](Real value) { return std::isfinite(value); }))) {
                last_evaluation_diagnostic =
                    "candidate_evaluator_returned_invalid_functional_derivatives";
                return false;
            }
            if (evaluation.pressure_representability_available &&
                (!finiteNonnegative(
                     evaluation.pressure_representability_residual_norm) ||
                 !finiteNonnegative(
                     evaluation
                         .pressure_representability_relative_distance))) {
                last_evaluation_diagnostic =
                    "candidate_evaluator_returned_invalid_pressure_representability_state";
                return false;
            }
            if (evaluation.constant_pressure_kkt_available &&
                (!finiteNonnegative(
                     evaluation.constant_pressure_kkt_residual_norm) ||
                 !finiteNonnegative(
                     evaluation.constant_pressure_kkt_relative_distance))) {
                last_evaluation_diagnostic =
                    "candidate_evaluator_returned_invalid_constant_pressure_kkt_state";
                return false;
            }
            last_evaluation_diagnostic.clear();
            return true;
        };

    LevelSetStaticCapillaryEquilibriumEvaluation current;
    if (!evaluate(
            coefficients,
            LevelSetStaticCapillaryEvaluationPurpose::FunctionalTrial,
            current)) {
        result.diagnostic =
            forbidden_projection_seen
                ? "production_force_projection_is_forbidden"
                : "initial_candidate_evaluation_failed:" +
                      last_evaluation_diagnostic;
        return result;
    }

    std::uint64_t topology_epoch_key =
        current.cut_topology_key;
    std::uint64_t constraint_epoch_key =
        current.constraint_semantics_key;
    result.initial_snapshot_revision_key =
        current.snapshot_revision_key;
    result.final_snapshot_revision_key =
        current.snapshot_revision_key;
    result.cut_topology_key = topology_epoch_key;
    result.constraint_semantics_key =
        constraint_epoch_key;
    result.initial_surface_wall_energy =
        current.surface_wall_energy;
    result.final_surface_wall_energy =
        current.surface_wall_energy;
    result.initial_gravitational_potential_energy =
        current.gravitational_potential_energy;
    result.final_gravitational_potential_energy =
        current.gravitational_potential_energy;
    result.initial_physical_potential_energy =
        physicalPotentialEnergy(current);
    result.final_physical_potential_energy =
        physicalPotentialEnergy(current);
    result.initial_liquid_volume = current.liquid_volume;
    result.final_liquid_volume = current.liquid_volume;

    struct LimitedMemoryCorrection {
        std::vector<Real> parameter_step{};
        std::vector<Real> gradient_step{};
        Real inverse_curvature{0.0};
    };
    std::deque<LimitedMemoryCorrection> limited_memory;
    std::vector<Real> previous_active_coefficients;
    std::vector<Real> previous_projected_gradient;
    bool pending_limited_memory_update = false;
    Real effective_projected_gradient_tolerance =
        options.projected_gradient_tolerance;
    std::string pending_acceptance_gate_diagnostic;

    const auto exactProjectedGradientNorm =
        [&](const LevelSetStaticCapillaryEquilibriumEvaluation& evaluation,
            Real& projected_norm) {
            if (!evaluation.functional_derivatives_available) {
                return false;
            }
            std::vector<Real> energy(active_indices.size(), Real{0.0});
            std::vector<Real> volume(active_indices.size(), Real{0.0});
            for (std::size_t local = 0u;
                 local < active_indices.size();
                 ++local) {
                const auto index = active_indices[local];
                energy[local] =
                    evaluation.physical_potential_derivative[index];
                volume[local] =
                    evaluation.liquid_volume_derivative[index];
            }
            const Real volume_squared = dot(volume, volume);
            if (!finitePositive(volume_squared)) {
                return false;
            }
            const Real multiplier =
                -dot(energy, volume) / volume_squared;
            if (!std::isfinite(multiplier)) {
                return false;
            }
            for (std::size_t local = 0u;
                 local < energy.size();
                 ++local) {
                energy[local] += multiplier * volume[local];
            }
            projected_norm = norm(energy);
            return finiteNonnegative(projected_norm);
        };

    const auto updateFinalEvaluation =
        [&](const LevelSetStaticCapillaryEquilibriumEvaluation& evaluation) {
            result.final_snapshot_revision_key =
                evaluation.snapshot_revision_key;
            result.final_surface_wall_energy =
                evaluation.surface_wall_energy;
            result.final_gravitational_potential_energy =
                evaluation.gravitational_potential_energy;
            result.final_physical_potential_energy =
                physicalPotentialEnergy(evaluation);
            result.final_liquid_volume = evaluation.liquid_volume;
            result.final_volume_error =
                evaluation.liquid_volume -
                options.target_liquid_volume;
            result.final_pressure_representability_available =
                evaluation.pressure_representability_available;
            result.final_pressure_representability_converged =
                evaluation.pressure_representability_converged;
            result.final_pressure_representability_breakdown =
                evaluation.pressure_representability_breakdown;
            result.final_pressure_representability_residual_norm =
                evaluation.pressure_representability_residual_norm;
            result.final_pressure_representability_relative_distance =
                evaluation.pressure_representability_relative_distance;
            result.final_production_residual_norm =
                evaluation.production_residual_norm;
            result.final_constant_pressure_kkt_required =
                evaluation.constant_pressure_kkt_required;
            result.final_constant_pressure_kkt_available =
                evaluation.constant_pressure_kkt_available;
            result.final_constant_pressure_kkt_residual_norm =
                evaluation.constant_pressure_kkt_residual_norm;
            result.final_constant_pressure_kkt_relative_distance =
                evaluation.constant_pressure_kkt_relative_distance;
        };
    updateFinalEvaluation(current);

    for (;;) {
        std::vector<Real> energy_gradient(active_indices.size(), Real{0.0});
        std::vector<Real> volume_gradient(active_indices.size(), Real{0.0});
        bool gradient_available = true;
        if (current.functional_derivatives_available) {
            for (std::size_t local = 0u; local < active_indices.size();
                 ++local) {
                const auto index = active_indices[local];
                energy_gradient[local] =
                    current.physical_potential_derivative[index];
                volume_gradient[local] =
                    current.liquid_volume_derivative[index];
            }
            ++result.analytic_derivative_evaluations;
        } else {
            for (std::size_t local = 0u; local < active_indices.size();
                 ++local) {
                const auto index = active_indices[local];
                const Real coefficient_scale = std::max(
                    options.finite_difference_reference_coefficient_scale,
                    std::abs(coefficients[index]));
                // The fourth-order stencil permits a larger, less
                // subtraction-sensitive configured step than the former
                // two-point formula. Topology checks below shrink it
                // deterministically when the nominal stencil is too wide.
                Real difference_step =
                    std::max(options.minimum_finite_difference_step,
                             options.finite_difference_relative_step *
                                 coefficient_scale);
                bool component_available = false;
                for (int attempt = 0;
                     attempt <= options.finite_difference_max_shrinks;
                     ++attempt) {
                    constexpr std::array<Real, 4> offsets{
                        {Real{-2.0}, Real{-1.0}, Real{1.0}, Real{2.0}}};
                    std::array<LevelSetStaticCapillaryEquilibriumEvaluation,
                               offsets.size()>
                        states;
                    std::array<bool, offsets.size()> available{};
                    std::array<std::string, offsets.size()> diagnostics;
                    for (std::size_t sample = 0u; sample < offsets.size();
                         ++sample) {
                        std::vector<Real> perturbed = coefficients;
                        perturbed[index] += offsets[sample] * difference_step;
                        available[sample] =
                            evaluate(perturbed,
                                     LevelSetStaticCapillaryEvaluationPurpose::
                                         FunctionalTrial,
                                     states[sample]);
                        diagnostics[sample] = last_evaluation_diagnostic;
                    }
                    if (forbidden_projection_seen) {
                        result.diagnostic =
                            "production_force_projection_is_forbidden";
                        return result;
                    }
                    const bool all_available =
                        std::all_of(available.begin(), available.end(),
                                    [](bool value) { return value; });
                    const bool topology_preserved =
                        all_available &&
                        std::all_of(states.begin(), states.end(),
                                    [&](const auto& state) {
                                        return state.cut_topology_key ==
                                               topology_epoch_key;
                                    });
                    const bool constraints_preserved =
                        all_available &&
                        std::all_of(states.begin(), states.end(),
                                    [&](const auto& state) {
                                        return state.constraint_semantics_key ==
                                               constraint_epoch_key;
                                    });
                    if (topology_preserved && constraints_preserved) {
                        const std::array<Real, 4> energies{
                            {physicalPotentialEnergy(states[0]),
                             physicalPotentialEnergy(states[1]),
                             physicalPotentialEnergy(states[2]),
                             physicalPotentialEnergy(states[3])}};
                        const std::array<Real, 4> volumes{
                            {states[0].liquid_volume, states[1].liquid_volume,
                             states[2].liquid_volume, states[3].liquid_volume}};
                        const auto fourth_order_derivative =
                            [&](const std::array<Real, 4>& values) {
                                return (values[0] - Real{8.0} * values[1] +
                                        Real{8.0} * values[2] - values[3]) /
                                       (Real{12.0} * difference_step);
                            };
                        const auto second_order_derivative =
                            [&](const std::array<Real, 4>& values) {
                                return (values[2] - values[1]) /
                                       (Real{2.0} * difference_step);
                            };
                        const Real energy_derivative =
                            fourth_order_derivative(energies);
                        const Real volume_derivative =
                            fourth_order_derivative(volumes);
                        if (std::isfinite(energy_derivative) &&
                            std::isfinite(volume_derivative)) {
                            const Real second_order_energy =
                                second_order_derivative(energies);
                            const Real second_order_volume =
                                second_order_derivative(volumes);
                            const auto relative_correction =
                                [](Real high_order, Real low_order) {
                                    return std::abs(high_order - low_order) /
                                           std::max({Real{1.0},
                                                     std::abs(high_order),
                                                     std::abs(low_order)});
                                };
                            energy_gradient[local] = energy_derivative;
                            volume_gradient[local] = volume_derivative;
                            if (result
                                    .finite_difference_fourth_order_components ==
                                0u) {
                                result.minimum_finite_difference_step_used =
                                    difference_step;
                            } else {
                                result.minimum_finite_difference_step_used =
                                    std::min(
                                        result
                                            .minimum_finite_difference_step_used,
                                        difference_step);
                            }
                            ++result.finite_difference_fourth_order_components;
                            result.maximum_finite_difference_step_used =
                                std::max(
                                    result.maximum_finite_difference_step_used,
                                    difference_step);
                            result
                                .maximum_energy_derivative_relative_correction =
                                std::max(
                                    result
                                        .maximum_energy_derivative_relative_correction,
                                    relative_correction(energy_derivative,
                                                        second_order_energy));
                            result
                                .maximum_volume_derivative_relative_correction =
                                std::max(
                                    result
                                        .maximum_volume_derivative_relative_correction,
                                    relative_correction(volume_derivative,
                                                        second_order_volume));
                            component_available = true;
                            break;
                        }
                        last_evaluation_diagnostic =
                            "candidate_fourth_order_difference_is_nonfinite";
                    } else if (all_available && !topology_preserved) {
                        ++result.topology_change_rejections;
                        last_evaluation_diagnostic =
                            "candidate_cut_topology_changed";
                    } else if (all_available && !constraints_preserved) {
                        ++result.constraint_change_rejections;
                        last_evaluation_diagnostic =
                            "candidate_constraint_semantics_changed";
                    } else {
                        last_evaluation_diagnostic =
                            "candidate_stencil_sample_failed";
                        for (std::size_t sample = 0u; sample < available.size();
                             ++sample) {
                            if (!available[sample]) {
                                last_evaluation_diagnostic +=
                                    ":offset=" +
                                    std::to_string(offsets[sample]) + ":" +
                                    diagnostics[sample];
                                break;
                            }
                        }
                    }

                    if (attempt < options.finite_difference_max_shrinks) {
                        const Real smaller_step =
                            std::max(options.minimum_finite_difference_step,
                                     difference_step * Real{0.5});
                        if (smaller_step == difference_step) {
                            break;
                        }
                        difference_step = smaller_step;
                        ++result.finite_difference_step_shrinks;
                    }
                }
                if (!component_available) {
                    gradient_available = false;
                    break;
                }
            }
        }
        if (!gradient_available) {
            result.diagnostic =
                "fixed_topology_functional_derivative_unavailable:" +
                last_evaluation_diagnostic;
            return result;
        }

        const Real volume_gradient_squared =
            dot(volume_gradient, volume_gradient);
        if (!std::isfinite(volume_gradient_squared) ||
            !(volume_gradient_squared > Real{0.0})) {
            result.diagnostic = "liquid_volume_gradient_is_zero_or_nonfinite";
            return result;
        }
        const Real energy_volume_cross = dot(energy_gradient, volume_gradient);
        const Real volume_multiplier =
            -energy_volume_cross / volume_gradient_squared;
        if (!std::isfinite(volume_multiplier)) {
            result.diagnostic = "volume_multiplier_is_nonfinite";
            return result;
        }

        std::vector<Real> projected_gradient(active_indices.size(), Real{0.0});
        for (std::size_t i = 0u; i < active_indices.size(); ++i) {
            projected_gradient[i] =
                energy_gradient[i] +
                volume_multiplier * volume_gradient[i];
        }
        const Real projected_gradient_norm =
            norm(projected_gradient);
        if (!finiteNonnegative(projected_gradient_norm)) {
            result.diagnostic =
                "projected_energy_gradient_is_nonfinite";
            return result;
        }

        if (pending_limited_memory_update &&
            options.limited_memory_history_size > 0) {
            std::vector<Real> parameter_step(
                active_indices.size(), Real{0.0});
            std::vector<Real> gradient_step(
                active_indices.size(), Real{0.0});
            for (std::size_t i = 0u;
                 i < active_indices.size();
                 ++i) {
                parameter_step[i] =
                    coefficients[active_indices[i]] -
                    previous_active_coefficients[i];
                gradient_step[i] =
                    projected_gradient[i] -
                    previous_projected_gradient[i];
            }
            // Both secant vectors must lie in the current linearized volume
            // tangent space. This keeps the inverse-Hessian action from
            // introducing a multiplier component when that tangent rotates.
            projectOrthogonalTo(
                parameter_step,
                volume_gradient,
                volume_gradient_squared);
            projectOrthogonalTo(
                gradient_step,
                volume_gradient,
                volume_gradient_squared);
            const Real step_norm = norm(parameter_step);
            const Real gradient_step_norm = norm(gradient_step);
            const Real curvature = dot(
                parameter_step, gradient_step);
            const Real curvature_floor =
                options.limited_memory_curvature_tolerance *
                step_norm * gradient_step_norm;
            if (finitePositive(step_norm) &&
                finitePositive(gradient_step_norm) &&
                std::isfinite(curvature) &&
                curvature > curvature_floor) {
                if (limited_memory.size() ==
                    static_cast<std::size_t>(
                        options.limited_memory_history_size)) {
                    limited_memory.pop_front();
                }
                limited_memory.push_back({
                    std::move(parameter_step),
                    std::move(gradient_step),
                    Real{1.0} / curvature});
                ++result.limited_memory_updates;
                result.limited_memory_peak_history = std::max(
                    result.limited_memory_peak_history,
                    limited_memory.size());
            } else if (!limited_memory.empty()) {
                limited_memory.clear();
                ++result.limited_memory_resets;
            }
        }
        pending_limited_memory_update = false;

        updateFinalEvaluation(current);
        result.final_volume_multiplier = volume_multiplier;
        result.final_projected_gradient_norm =
            projected_gradient_norm;
        const Real volume_error =
            current.liquid_volume -
            options.target_liquid_volume;
        const bool volume_converged =
            std::abs(volume_error) <= options.volume_tolerance;
        const bool parameter_gradient_converged =
            projected_gradient_norm <=
            effective_projected_gradient_tolerance;
        if (volume_converged &&
            parameter_gradient_converged) {
            LevelSetStaticCapillaryEquilibriumEvaluation
                certificate;
            if (!evaluate(
                    coefficients,
                    LevelSetStaticCapillaryEvaluationPurpose::
                        AcceptanceCertificate,
                    certificate)) {
                result.diagnostic =
                    forbidden_projection_seen
                        ? "production_force_projection_is_forbidden"
                        : "acceptance_certificate_evaluation_failed:" +
                              last_evaluation_diagnostic;
                return result;
            }
            if (certificate.cut_topology_key !=
                topology_epoch_key) {
                ++result.topology_change_rejections;
                result.diagnostic =
                    "acceptance_certificate_cut_topology_changed";
                return result;
            }
            if (certificate.constraint_semantics_key !=
                constraint_epoch_key) {
                ++result.constraint_change_rejections;
                result.diagnostic =
                    "acceptance_certificate_constraint_semantics_changed";
                return result;
            }
            if (certificate.surface_wall_energy !=
                    current.surface_wall_energy ||
                certificate.gravitational_potential_energy !=
                    current.gravitational_potential_energy ||
                certificate.liquid_volume !=
                    current.liquid_volume) {
                result.diagnostic =
                    "acceptance_certificate_functionals_not_reproducible";
                return result;
            }

            current = std::move(certificate);
            updateFinalEvaluation(current);
            result.final_volume_multiplier = volume_multiplier;
            result.final_projected_gradient_norm =
                projected_gradient_norm;
            const bool certified_volume_converged =
                std::abs(
                    current.liquid_volume -
                    options.target_liquid_volume) <=
                options.volume_tolerance;
            if (!certified_volume_converged) {
                result.diagnostic =
                    "acceptance_certificate_volume_gate_failed";
                return result;
            }
            if (!current.pressure_representability_available) {
                result.diagnostic =
                    "pressure_representability_unavailable_at_parameter_stationary_geometry";
                return result;
            }
            if (!current.pressure_representability_converged ||
                current.pressure_representability_breakdown) {
                result.diagnostic =
                    "pressure_representability_gate_failed_at_parameter_stationary_geometry";
                return result;
            }
            if (!finiteNonnegative(current.production_residual_norm)) {
                result.diagnostic =
                    "physical_equilibrium_residual_gate_failed_at_parameter_stationary_geometry";
                return result;
            }
            if (current.constant_pressure_kkt_required &&
                !current.constant_pressure_kkt_available) {
                result.diagnostic =
                    "constant_pressure_kkt_unavailable_at_parameter_stationary_geometry";
                return result;
            }

            Real tightening_factor = Real{0.1};
            std::string failed_gate_diagnostic;
            const auto accountForFailedGate =
                [&](Real value,
                    Real limit,
                    const char* diagnostic) {
                    if (value <= limit) {
                        return;
                    }
                    if (failed_gate_diagnostic.empty()) {
                        failed_gate_diagnostic = diagnostic;
                    }
                    // Predict the parameter tolerance from the observed gate
                    // ratio, retain a factor-of-two margin, and always request
                    // at least one order of magnitude of additional descent.
                    const Real gate_factor =
                        limit > Real{0.0}
                            ? Real{0.5} * limit / value
                            : Real{0.0};
                    tightening_factor =
                        std::min(tightening_factor, gate_factor);
                };
            accountForFailedGate(
                current.pressure_representability_residual_norm,
                options.pressure_representability_max_residual_norm,
                "pressure_representability_gate_failed_at_parameter_stationary_geometry");
            accountForFailedGate(
                current.pressure_representability_relative_distance,
                options.pressure_representability_max_relative_distance,
                "pressure_representability_gate_failed_at_parameter_stationary_geometry");
            accountForFailedGate(
                current.production_residual_norm,
                options.physical_equilibrium_max_residual_norm,
                "physical_equilibrium_residual_gate_failed_at_parameter_stationary_geometry");
            if (current.constant_pressure_kkt_required) {
                accountForFailedGate(
                    current.constant_pressure_kkt_residual_norm,
                    options.constant_pressure_kkt_max_residual_norm,
                    "constant_pressure_kkt_gate_failed_at_parameter_stationary_geometry");
                accountForFailedGate(
                    current.constant_pressure_kkt_relative_distance,
                    options.constant_pressure_kkt_max_relative_distance,
                    "constant_pressure_kkt_gate_failed_at_parameter_stationary_geometry");
            }

            if (!failed_gate_diagnostic.empty()) {
                // A difference gradient cannot reliably distinguish further
                // descent from stencil and subtraction error at this point.
                if (!current.functional_derivatives_available) {
                    result.diagnostic = failed_gate_diagnostic;
                    return result;
                }
                if (result.iterations >= options.max_iterations) {
                    result.diagnostic = failed_gate_diagnostic;
                    return result;
                }
                const Real stationarity_scale =
                    norm(energy_gradient) +
                    std::abs(volume_multiplier) * norm(volume_gradient);
                const Real numerical_stationarity_floor =
                    Real{64.0} *
                    std::numeric_limits<Real>::epsilon() *
                    stationarity_scale;
                const Real requested_tolerance =
                    std::max(
                        numerical_stationarity_floor,
                        projected_gradient_norm * tightening_factor);
                if (!(projected_gradient_norm >
                      numerical_stationarity_floor) ||
                    !(requested_tolerance < projected_gradient_norm)) {
                    result.diagnostic = failed_gate_diagnostic;
                    return result;
                }
                effective_projected_gradient_tolerance =
                    std::min(
                        effective_projected_gradient_tolerance,
                        requested_tolerance);
                pending_acceptance_gate_diagnostic =
                    failed_gate_diagnostic;
            } else {
                std::vector<Real> committed = coefficients;
                accepted_coefficients.swap(committed);
                result.success = true;
                result.converged = true;
                result.accepted_coefficients_assigned = true;
                result.diagnostic =
                    "fixed_volume_discrete_capillary_equilibrium_converged";
                return result;
            }
        }
        if (result.iterations >= options.max_iterations) {
            if (pending_acceptance_gate_diagnostic.empty()) {
                result.diagnostic =
                    "static_capillary_equilibrium_iteration_limit_reached";
            } else {
                result.diagnostic =
                    "acceptance_certificate_refinement_iteration_limit_reached:" +
                    pending_acceptance_gate_diagnostic;
            }
            return result;
        }

        std::vector<Real> tangent_direction = projected_gradient;
        bool used_limited_memory = !limited_memory.empty();
        if (used_limited_memory) {
            std::vector<Real> alpha(limited_memory.size(), Real{0.0});
            for (std::size_t reverse = limited_memory.size();
                 reverse > 0u;
                 --reverse) {
                const std::size_t i = reverse - 1u;
                const auto& correction = limited_memory[i];
                alpha[i] = correction.inverse_curvature * dot(
                    correction.parameter_step, tangent_direction);
                for (std::size_t j = 0u;
                     j < tangent_direction.size();
                     ++j) {
                    tangent_direction[j] -=
                        alpha[i] * correction.gradient_step[j];
                }
            }
            const auto& newest = limited_memory.back();
            const Real newest_gradient_squared = dot(
                newest.gradient_step, newest.gradient_step);
            Real initial_inverse_scale =
                newest_gradient_squared > Real{0.0}
                    ? dot(
                          newest.parameter_step,
                          newest.gradient_step) /
                          newest_gradient_squared
                    : options.projected_gradient_inverse_stiffness;
            const Real configured_scale =
                options.projected_gradient_inverse_stiffness;
            initial_inverse_scale = std::clamp(
                initial_inverse_scale,
                configured_scale * Real{1.0e-6},
                configured_scale * Real{1.0e6});
            for (Real& value : tangent_direction) {
                value *= initial_inverse_scale;
            }
            for (std::size_t i = 0u;
                 i < limited_memory.size();
                 ++i) {
                const auto& correction = limited_memory[i];
                const Real beta = correction.inverse_curvature * dot(
                    correction.gradient_step, tangent_direction);
                for (std::size_t j = 0u;
                     j < tangent_direction.size();
                     ++j) {
                    tangent_direction[j] +=
                        correction.parameter_step[j] *
                        (alpha[i] - beta);
                }
            }
        } else {
            for (Real& value : tangent_direction) {
                value *=
                    options.projected_gradient_inverse_stiffness;
            }
        }
        for (Real& value : tangent_direction) {
            value = -value;
        }
        projectOrthogonalTo(
            tangent_direction,
            volume_gradient,
            volume_gradient_squared);
        Real tangent_direction_norm = norm(tangent_direction);
        const Real descent_floor =
            std::numeric_limits<Real>::epsilon() *
            projected_gradient_norm * tangent_direction_norm;
        if (!finitePositive(tangent_direction_norm) ||
            !(dot(projected_gradient, tangent_direction) <
              -descent_floor)) {
            tangent_direction = projected_gradient;
            for (Real& value : tangent_direction) {
                value *=
                    -options.projected_gradient_inverse_stiffness;
            }
            tangent_direction_norm = norm(tangent_direction);
            limited_memory.clear();
            ++result.limited_memory_resets;
            ++result.projected_gradient_fallbacks;
        }
        if (!finitePositive(tangent_direction_norm)) {
            result.diagnostic =
                "static_capillary_equilibrium_tangent_step_is_nonfinite";
            return result;
        }
        if (tangent_direction_norm >
            options.tangent_trust_radius) {
            const Real scale =
                options.tangent_trust_radius /
                tangent_direction_norm;
            for (Real& value : tangent_direction) {
                value *= scale;
            }
        }
        std::vector<Real> direction(
            active_indices.size(), Real{0.0});
        for (std::size_t i = 0u;
             i < active_indices.size();
             ++i) {
            const Real volume_correction =
                -volume_error * volume_gradient[i] /
                volume_gradient_squared;
            direction[i] =
                tangent_direction[i] + volume_correction;
        }
        Real direction_linf = Real{0.0};
        for (const Real value : direction) {
            if (!std::isfinite(value)) {
                result.diagnostic =
                    "static_capillary_equilibrium_direction_is_nonfinite";
                return result;
            }
            direction_linf =
                std::max(direction_linf, std::abs(value));
        }
        if (!finitePositive(direction_linf)) {
            result.diagnostic =
                "static_capillary_equilibrium_has_no_admissible_descent_direction";
            return result;
        }

        Real alpha = std::min(
            Real{1.0},
            options.maximum_coefficient_update_linf /
                direction_linf);
        const Real merit_penalty =
            options.minimum_volume_merit_penalty +
            Real{2.0} * std::abs(volume_multiplier);
        if (!finitePositive(merit_penalty)) {
            result.diagnostic =
                "static_capillary_equilibrium_merit_penalty_is_nonfinite";
            return result;
        }
        const Real volume_error_sign =
            volume_error > Real{0.0}
                ? Real{1.0}
                : (volume_error < Real{0.0}
                       ? Real{-1.0}
                       : Real{0.0});
        const Real directional_energy =
            dot(energy_gradient, direction);
        const Real directional_volume =
            dot(volume_gradient, direction);
        const Real directional_merit =
            directional_energy +
            merit_penalty * volume_error_sign *
                directional_volume;
        const Real predicted_merit_decrease =
            -directional_merit;
        if (!finitePositive(predicted_merit_decrease)) {
            result.diagnostic =
                "static_capillary_equilibrium_merit_direction_is_not_descent";
            return result;
        }
        const Real current_merit =
            physicalPotentialEnergy(current) +
            merit_penalty * std::abs(volume_error);

        struct TopologyTransitionTrial {
            Real alpha{0.0};
            std::vector<Real> coefficients{};
            LevelSetStaticCapillaryEquilibriumEvaluation evaluation{};
        };
        bool step_accepted = false;
        bool accepted_topology_transition = false;
        std::vector<Real> accepted_trial;
        LevelSetStaticCapillaryEquilibriumEvaluation accepted_state;
        std::optional<TopologyTransitionTrial>
            deferred_topology_transition;
        for (int line_search_iteration = 0;
             line_search_iteration <
             options.max_line_search_iterations;
             ++line_search_iteration) {
            std::vector<Real> trial = coefficients;
            for (std::size_t i = 0u;
                 i < active_indices.size();
                 ++i) {
                trial[active_indices[i]] +=
                    alpha * direction[i];
            }
            LevelSetStaticCapillaryEquilibriumEvaluation trial_state;
            const bool trial_available =
                evaluate(
                    trial,
                    LevelSetStaticCapillaryEvaluationPurpose::
                        FunctionalTrial,
                    trial_state);
            if (forbidden_projection_seen) {
                result.diagnostic =
                    "production_force_projection_is_forbidden";
                return result;
            }
            if (trial_available) {
                const Real trial_volume_error =
                    trial_state.liquid_volume -
                    options.target_liquid_volume;
                const Real trial_merit =
                    physicalPotentialEnergy(trial_state) +
                    merit_penalty *
                        std::abs(trial_volume_error);
                const Real armijo_bound =
                    current_merit -
                    options.armijo_fraction * alpha *
                        predicted_merit_decrease;
                const bool topology_changed =
                    trial_state.cut_topology_key !=
                    topology_epoch_key;
                const bool constraints_changed =
                    trial_state.constraint_semantics_key !=
                    constraint_epoch_key;
                if (topology_changed) {
                    ++result.topology_change_rejections;
                    const bool transition_limit_available =
                        result.topology_epoch_transitions <
                        static_cast<std::size_t>(
                            options.max_topology_epoch_transitions);
                    if (options.allow_topology_epoch_transitions &&
                        transition_limit_available &&
                        current.functional_derivatives_available &&
                        trial_state.functional_derivatives_available &&
                        std::isfinite(trial_merit) &&
                        trial_merit <= armijo_bound) {
                        deferred_topology_transition =
                            TopologyTransitionTrial{
                                alpha,
                                std::move(trial),
                                std::move(trial_state)};
                        last_evaluation_diagnostic =
                            "candidate_cut_topology_transition_deferred";
                    } else if (
                        options.allow_topology_epoch_transitions &&
                        !transition_limit_available) {
                        last_evaluation_diagnostic =
                            "candidate_cut_topology_transition_limit_reached";
                    } else {
                        last_evaluation_diagnostic =
                            "candidate_cut_topology_changed";
                    }
                } else if (constraints_changed) {
                    ++result.constraint_change_rejections;
                    last_evaluation_diagnostic =
                        "candidate_constraint_semantics_changed";
                } else {
                    Real trial_projected_gradient_norm =
                        std::numeric_limits<Real>::quiet_NaN();
                    const bool trial_exact_gradient_available =
                        current.functional_derivatives_available &&
                        exactProjectedGradientNorm(
                            trial_state,
                            trial_projected_gradient_norm);
                    const Real merit_resolution =
                        Real{16.0} *
                        std::numeric_limits<Real>::epsilon() *
                        std::max({
                            Real{1.0},
                            std::abs(current_merit),
                            std::abs(trial_merit)});
                    const bool derivative_resolution_step =
                        trial_exact_gradient_available &&
                        std::isfinite(trial_merit) &&
                        trial_merit > armijo_bound &&
                        alpha * predicted_merit_decrease <=
                            merit_resolution &&
                        trial_merit <=
                            current_merit + merit_resolution &&
                        std::abs(trial_volume_error) <=
                            std::max(
                                std::abs(volume_error),
                                options.volume_tolerance) &&
                        trial_projected_gradient_norm <
                            projected_gradient_norm;
                    if (std::isfinite(trial_merit) &&
                        (trial_merit <= armijo_bound ||
                         derivative_resolution_step)) {
                        accepted_trial = std::move(trial);
                        accepted_state = std::move(trial_state);
                        step_accepted = true;
                        if (derivative_resolution_step) {
                            ++result
                                  .derivative_resolution_step_acceptances;
                        }
                        break;
                    }
                    last_evaluation_diagnostic =
                        std::isfinite(trial_merit)
                            ? "candidate_capillary_merit_decrease_insufficient"
                            : "candidate_capillary_merit_is_nonfinite";
                }
            }

            ++result.line_search_rejections;
            alpha *= options.line_search_shrink;
        }
        if (!step_accepted) {
            if (used_limited_memory) {
                limited_memory.clear();
                ++result.limited_memory_resets;
                ++result.projected_gradient_fallbacks;
                continue;
            }
            if (deferred_topology_transition.has_value()) {
                LevelSetStaticCapillaryEquilibriumEvaluation
                    reproduced_state;
                if (!evaluate(
                        deferred_topology_transition->coefficients,
                        LevelSetStaticCapillaryEvaluationPurpose::
                            FunctionalTrial,
                        reproduced_state)) {
                    result.diagnostic =
                        "topology_transition_reproduction_failed:" +
                        last_evaluation_diagnostic;
                    return result;
                }
                const auto& deferred_state =
                    deferred_topology_transition->evaluation;
                const Real reproduced_volume_error =
                    reproduced_state.liquid_volume -
                    options.target_liquid_volume;
                const Real reproduced_merit =
                    physicalPotentialEnergy(reproduced_state) +
                    merit_penalty *
                        std::abs(reproduced_volume_error);
                const Real reproduced_armijo_bound =
                    current_merit -
                    options.armijo_fraction *
                        deferred_topology_transition->alpha *
                        predicted_merit_decrease;
                const bool transition_reproduced =
                    reproduced_state.functional_derivatives_available &&
                    reproduced_state.cut_topology_key !=
                        topology_epoch_key &&
                    reproduced_state.cut_topology_key ==
                        deferred_state.cut_topology_key &&
                    reproduced_state.constraint_semantics_key ==
                        deferred_state.constraint_semantics_key &&
                    reproduced_state.surface_wall_energy ==
                        deferred_state.surface_wall_energy &&
                    reproduced_state.gravitational_potential_energy ==
                        deferred_state.gravitational_potential_energy &&
                    reproduced_state.liquid_volume ==
                        deferred_state.liquid_volume &&
                    std::isfinite(reproduced_merit) &&
                    reproduced_merit <= reproduced_armijo_bound;
                if (!transition_reproduced) {
                    result.diagnostic =
                        "topology_transition_not_reproducible";
                    return result;
                }
                alpha = deferred_topology_transition->alpha;
                accepted_trial = std::move(
                    deferred_topology_transition->coefficients);
                accepted_state = std::move(reproduced_state);
                accepted_topology_transition = true;
                step_accepted = true;
            } else {
                if (pending_acceptance_gate_diagnostic.empty()) {
                    result.diagnostic =
                        (options.allow_topology_epoch_transitions
                             ? "capillary_merit_line_search_failed:"
                             : "fixed_topology_capillary_merit_line_search_failed:") +
                        last_evaluation_diagnostic;
                } else {
                    result.diagnostic =
                        "acceptance_certificate_refinement_failed:" +
                        pending_acceptance_gate_diagnostic + ":" +
                        last_evaluation_diagnostic;
                }
                return result;
            }
        }

        coefficients = std::move(accepted_trial);
        current = std::move(accepted_state);
        if (accepted_topology_transition) {
            topology_epoch_key = current.cut_topology_key;
            constraint_epoch_key =
                current.constraint_semantics_key;
            result.cut_topology_key = topology_epoch_key;
            result.constraint_semantics_key =
                constraint_epoch_key;
            limited_memory.clear();
            previous_active_coefficients.clear();
            previous_projected_gradient.clear();
            pending_limited_memory_update = false;
            ++result.topology_epoch_transitions;
        } else {
            previous_active_coefficients.resize(active_indices.size());
            for (std::size_t i = 0u;
                 i < active_indices.size();
                 ++i) {
                previous_active_coefficients[i] =
                    coefficients[active_indices[i]] -
                    alpha * direction[i];
            }
            previous_projected_gradient = projected_gradient;
            pending_limited_memory_update = true;
        }
        ++result.iterations;
        updateFinalEvaluation(current);
    }
}

} // namespace svmp::FE::level_set

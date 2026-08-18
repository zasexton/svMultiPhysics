#include "LevelSet/LevelSetStaticCapillaryEquilibrium.h"

#include <algorithm>
#include <cmath>
#include <exception>
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
        !finitePositive(
            options.projected_gradient_inverse_stiffness) ||
        !finitePositive(options.tangent_trust_radius) ||
        !finitePositive(options.maximum_coefficient_update_linf) ||
        !finitePositive(options.line_search_shrink) ||
        !(options.line_search_shrink < Real{1.0}) ||
        !finitePositive(options.armijo_fraction) ||
        !(options.armijo_fraction < Real{1.0}) ||
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
                !finiteNonnegative(evaluation.liquid_volume)) {
                last_evaluation_diagnostic =
                    "candidate_evaluator_returned_invalid_functional_state";
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

    const std::uint64_t fixed_topology_key =
        current.cut_topology_key;
    const std::uint64_t fixed_constraint_semantics_key =
        current.constraint_semantics_key;
    result.initial_snapshot_revision_key =
        current.snapshot_revision_key;
    result.final_snapshot_revision_key =
        current.snapshot_revision_key;
    result.cut_topology_key = fixed_topology_key;
    result.constraint_semantics_key =
        fixed_constraint_semantics_key;
    result.initial_surface_wall_energy =
        current.surface_wall_energy;
    result.final_surface_wall_energy =
        current.surface_wall_energy;
    result.initial_liquid_volume = current.liquid_volume;
    result.final_liquid_volume = current.liquid_volume;

    const auto updateFinalEvaluation =
        [&](const LevelSetStaticCapillaryEquilibriumEvaluation& evaluation) {
            result.final_snapshot_revision_key =
                evaluation.snapshot_revision_key;
            result.final_surface_wall_energy =
                evaluation.surface_wall_energy;
            result.final_liquid_volume = evaluation.liquid_volume;
            result.final_volume_error =
                evaluation.liquid_volume -
                options.target_liquid_volume;
            result.final_constant_pressure_kkt_available =
                evaluation.constant_pressure_kkt_available;
            result.final_constant_pressure_kkt_residual_norm =
                evaluation.constant_pressure_kkt_residual_norm;
            result.final_constant_pressure_kkt_relative_distance =
                evaluation.constant_pressure_kkt_relative_distance;
        };
    updateFinalEvaluation(current);

    for (;;) {
        std::vector<Real> energy_gradient(
            active_indices.size(), Real{0.0});
        std::vector<Real> volume_gradient(
            active_indices.size(), Real{0.0});
        bool gradient_available = true;
        for (std::size_t local = 0u;
             local < active_indices.size();
             ++local) {
            const auto index = active_indices[local];
            Real difference_step = std::max(
                options.minimum_finite_difference_step,
                options.finite_difference_relative_step *
                    std::max(
                        options
                            .finite_difference_reference_coefficient_scale,
                        std::abs(coefficients[index])));
            bool component_available = false;
            for (int attempt = 0;
                 attempt <= options.finite_difference_max_shrinks;
                 ++attempt) {
                std::vector<Real> plus = coefficients;
                std::vector<Real> minus = coefficients;
                plus[index] += difference_step;
                minus[index] -= difference_step;

                LevelSetStaticCapillaryEquilibriumEvaluation plus_state;
                LevelSetStaticCapillaryEquilibriumEvaluation minus_state;
                const bool plus_available = evaluate(
                    plus,
                    LevelSetStaticCapillaryEvaluationPurpose::
                        FunctionalTrial,
                    plus_state);
                const std::string plus_diagnostic =
                    last_evaluation_diagnostic;
                const bool minus_available = evaluate(
                    minus,
                    LevelSetStaticCapillaryEvaluationPurpose::
                        FunctionalTrial,
                    minus_state);
                const std::string minus_diagnostic =
                    last_evaluation_diagnostic;
                if (forbidden_projection_seen) {
                    result.diagnostic =
                        "production_force_projection_is_forbidden";
                    return result;
                }
                const bool topology_preserved =
                    plus_available && minus_available &&
                    plus_state.cut_topology_key ==
                        fixed_topology_key &&
                    minus_state.cut_topology_key ==
                        fixed_topology_key;
                const bool constraints_preserved =
                    plus_available && minus_available &&
                    plus_state.constraint_semantics_key ==
                        fixed_constraint_semantics_key &&
                    minus_state.constraint_semantics_key ==
                        fixed_constraint_semantics_key;
                if (topology_preserved && constraints_preserved) {
                    const Real denominator =
                        Real{2.0} * difference_step;
                    const Real energy_derivative =
                        (plus_state.surface_wall_energy -
                         minus_state.surface_wall_energy) /
                        denominator;
                    const Real volume_derivative =
                        (plus_state.liquid_volume -
                         minus_state.liquid_volume) /
                        denominator;
                    if (std::isfinite(energy_derivative) &&
                        std::isfinite(volume_derivative)) {
                        energy_gradient[local] = energy_derivative;
                        volume_gradient[local] = volume_derivative;
                        component_available = true;
                        break;
                    }
                    last_evaluation_diagnostic =
                        "candidate_central_difference_is_nonfinite";
                } else if (
                    (plus_available &&
                     plus_state.cut_topology_key !=
                         fixed_topology_key) ||
                    (minus_available &&
                     minus_state.cut_topology_key !=
                         fixed_topology_key)) {
                    ++result.topology_change_rejections;
                    last_evaluation_diagnostic =
                        "candidate_cut_topology_changed";
                } else if (
                    (plus_available &&
                     plus_state.constraint_semantics_key !=
                         fixed_constraint_semantics_key) ||
                    (minus_available &&
                     minus_state.constraint_semantics_key !=
                         fixed_constraint_semantics_key)) {
                    ++result.constraint_change_rejections;
                    last_evaluation_diagnostic =
                        "candidate_constraint_semantics_changed";
                } else if (!plus_available && !minus_available) {
                    last_evaluation_diagnostic =
                        "candidate_plus_failed:" +
                        plus_diagnostic +
                        ",candidate_minus_failed:" +
                        minus_diagnostic;
                } else if (!plus_available) {
                    last_evaluation_diagnostic =
                        "candidate_plus_failed:" +
                        plus_diagnostic;
                } else if (!minus_available) {
                    last_evaluation_diagnostic =
                        "candidate_minus_failed:" +
                        minus_diagnostic;
                }

                if (attempt <
                    options.finite_difference_max_shrinks) {
                    const Real smaller_step = std::max(
                        options.minimum_finite_difference_step,
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
        if (!gradient_available) {
            result.diagnostic =
                "fixed_topology_central_difference_unavailable:" +
                last_evaluation_diagnostic;
            return result;
        }

        const Real volume_gradient_squared =
            dot(volume_gradient, volume_gradient);
        if (!std::isfinite(volume_gradient_squared) ||
            !(volume_gradient_squared > Real{0.0})) {
            result.diagnostic =
                "liquid_volume_gradient_is_zero_or_nonfinite";
            return result;
        }
        const Real energy_volume_cross =
            dot(energy_gradient, volume_gradient);
        const Real volume_multiplier =
            -energy_volume_cross / volume_gradient_squared;
        if (!std::isfinite(volume_multiplier)) {
            result.diagnostic =
                "volume_multiplier_is_nonfinite";
            return result;
        }

        std::vector<Real> projected_gradient(
            active_indices.size(), Real{0.0});
        for (std::size_t i = 0u;
             i < active_indices.size();
             ++i) {
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
            options.projected_gradient_tolerance;
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
                fixed_topology_key) {
                ++result.topology_change_rejections;
                result.diagnostic =
                    "acceptance_certificate_cut_topology_changed";
                return result;
            }
            if (certificate.constraint_semantics_key !=
                fixed_constraint_semantics_key) {
                ++result.constraint_change_rejections;
                result.diagnostic =
                    "acceptance_certificate_constraint_semantics_changed";
                return result;
            }
            if (certificate.surface_wall_energy !=
                    current.surface_wall_energy ||
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
            const bool constant_pressure_kkt_converged =
                current.constant_pressure_kkt_available &&
                finiteNonnegative(
                    current.constant_pressure_kkt_residual_norm) &&
                current.constant_pressure_kkt_residual_norm <=
                    options.constant_pressure_kkt_max_residual_norm &&
                finiteNonnegative(
                    current.constant_pressure_kkt_relative_distance) &&
                current.constant_pressure_kkt_relative_distance <=
                    options
                        .constant_pressure_kkt_max_relative_distance;
            if (!constant_pressure_kkt_converged) {
                result.diagnostic =
                    current.constant_pressure_kkt_available
                        ? "constant_pressure_kkt_gate_failed_at_parameter_stationary_geometry"
                        : "constant_pressure_kkt_unavailable_at_parameter_stationary_geometry";
                return result;
            }

            std::vector<Real> committed = coefficients;
            accepted_coefficients.swap(committed);
            result.success = true;
            result.converged = true;
            result.accepted_coefficients_assigned = true;
            result.diagnostic =
                "fixed_volume_discrete_capillary_equilibrium_converged";
            return result;
        }
        if (result.iterations >= options.max_iterations) {
            result.diagnostic =
                "static_capillary_equilibrium_iteration_limit_reached";
            return result;
        }

        const Real unconstrained_tangent_step =
            options.projected_gradient_inverse_stiffness *
            projected_gradient_norm;
        if (!finiteNonnegative(unconstrained_tangent_step)) {
            result.diagnostic =
                "static_capillary_equilibrium_tangent_step_is_nonfinite";
            return result;
        }
        const Real tangent_step =
            std::min(
                options.tangent_trust_radius,
                unconstrained_tangent_step);
        std::vector<Real> direction(
            active_indices.size(), Real{0.0});
        for (std::size_t i = 0u;
             i < active_indices.size();
             ++i) {
            const Real tangent =
                projected_gradient_norm > Real{0.0}
                    ? -tangent_step * projected_gradient[i] /
                          projected_gradient_norm
                    : Real{0.0};
            const Real volume_correction =
                -volume_error * volume_gradient[i] /
                volume_gradient_squared;
            direction[i] = tangent + volume_correction;
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
            current.surface_wall_energy +
            merit_penalty * std::abs(volume_error);

        bool step_accepted = false;
        std::vector<Real> accepted_trial;
        LevelSetStaticCapillaryEquilibriumEvaluation accepted_state;
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
            if (trial_available &&
                trial_state.cut_topology_key !=
                    fixed_topology_key) {
                ++result.topology_change_rejections;
                last_evaluation_diagnostic =
                    "candidate_cut_topology_changed";
            } else if (
                trial_available &&
                trial_state.constraint_semantics_key !=
                    fixed_constraint_semantics_key) {
                ++result.constraint_change_rejections;
                last_evaluation_diagnostic =
                    "candidate_constraint_semantics_changed";
            } else if (trial_available) {
                const Real trial_volume_error =
                    trial_state.liquid_volume -
                    options.target_liquid_volume;
                const Real trial_merit =
                    trial_state.surface_wall_energy +
                    merit_penalty *
                        std::abs(trial_volume_error);
                const Real armijo_bound =
                    current_merit -
                    options.armijo_fraction * alpha *
                        predicted_merit_decrease;
                if (std::isfinite(trial_merit) &&
                    trial_merit <= armijo_bound) {
                    accepted_trial = std::move(trial);
                    accepted_state = std::move(trial_state);
                    step_accepted = true;
                    break;
                }
                last_evaluation_diagnostic =
                    std::isfinite(trial_merit)
                        ? "candidate_capillary_merit_decrease_insufficient"
                        : "candidate_capillary_merit_is_nonfinite";
            }

            ++result.line_search_rejections;
            alpha *= options.line_search_shrink;
        }
        if (!step_accepted) {
            result.diagnostic =
                "fixed_topology_capillary_merit_line_search_failed:" +
                last_evaluation_diagnostic;
            return result;
        }

        coefficients = std::move(accepted_trial);
        current = std::move(accepted_state);
        ++result.iterations;
        updateFinalEvaluation(current);
    }
}

} // namespace svmp::FE::level_set

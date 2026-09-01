/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Interfaces/IncompressibleTwoFluidDiagnostics.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace svmp::FE::interfaces {
namespace {

using Vector = std::array<Real, 3>;
using Matrix = FreeSurfaceDiscreteFunctionalPhysicalGradient;

[[nodiscard]] bool finiteVector(const Vector& value) noexcept
{
    return std::all_of(value.begin(), value.end(), [](Real entry) {
        return std::isfinite(entry);
    });
}

[[nodiscard]] bool finiteMatrix(const Matrix& value) noexcept
{
    return std::all_of(value.begin(), value.end(), [](const auto& row) {
        return finiteVector(row);
    });
}

[[nodiscard]] Real dot(const Vector& lhs, const Vector& rhs) noexcept
{
    Real value{0.0};
    for (std::size_t component = 0u; component < 3u; ++component) {
        value += lhs[component] * rhs[component];
    }
    return value;
}

[[nodiscard]] Vector subtract(const Vector& lhs, const Vector& rhs) noexcept
{
    Vector value{};
    for (std::size_t component = 0u; component < 3u; ++component) {
        value[component] = lhs[component] - rhs[component];
    }
    return value;
}

[[nodiscard]] Vector traction(
    const Matrix& gradient,
    Real pressure,
    Real viscosity,
    const Vector& normal,
    int dimension) noexcept
{
    Vector value{};
    for (int row = 0; row < dimension; ++row) {
        Real component = -pressure * normal[static_cast<std::size_t>(row)];
        for (int column = 0; column < dimension; ++column) {
            component += viscosity *
                         (gradient[static_cast<std::size_t>(row)]
                                  [static_cast<std::size_t>(column)] +
                          gradient[static_cast<std::size_t>(column)]
                                  [static_cast<std::size_t>(row)]) *
                         normal[static_cast<std::size_t>(column)];
        }
        value[static_cast<std::size_t>(row)] = component;
    }
    return value;
}

void validateParameters(
    const IncompressibleTwoFluidDiagnosticParameters& parameters)
{
    const auto finite_positive = [](Real value) {
        return std::isfinite(value) && value > Real{0.0};
    };
    if ((parameters.dimension != 2 && parameters.dimension != 3) ||
        parameters.interface_marker < 0 ||
        !finite_positive(parameters.negative_density) ||
        !finite_positive(parameters.positive_density) ||
        !finite_positive(parameters.negative_viscosity) ||
        !finite_positive(parameters.positive_viscosity) ||
        !finite_positive(parameters.nitsche_gamma) ||
        !std::isfinite(parameters.surface_tension) ||
        parameters.surface_tension < Real{0.0} ||
        (parameters.prescribed_pressure_jump.has_value() &&
         !std::isfinite(*parameters.prescribed_pressure_jump))) {
        throw std::invalid_argument(
            "incompressible two-fluid diagnostics require a valid marker, dimension, material coefficients, Nitsche coefficient, surface tension, and optional pressure-jump target");
    }
}

void validateEvaluators(
    const IncompressibleTwoFluidPhaseEvaluators& phase,
    const char* name)
{
    if (!phase.velocity.canEvaluateValue() ||
        !phase.velocity.canEvaluatePhysicalGradient() ||
        !phase.pressure.canEvaluateValue()) {
        throw std::invalid_argument(
            std::string("incompressible two-fluid diagnostics require value and physical-gradient evaluators for the ") +
            name + " phase");
    }
}

[[nodiscard]] Real stableHarmonicMean(Real lhs, Real rhs) noexcept
{
    const Real maximum = std::max(lhs, rhs);
    const Real minimum = std::min(lhs, rhs);
    return minimum * (Real{2.0} / (Real{1.0} + minimum / maximum));
}

[[nodiscard]] std::array<Real, 4> interfaceWeights(
    const IncompressibleTwoFluidDiagnosticParameters& parameters) noexcept
{
    const Real maximum = std::max(
        parameters.negative_viscosity,
        parameters.positive_viscosity);
    const Real negative_scaled = parameters.negative_viscosity / maximum;
    const Real positive_scaled = parameters.positive_viscosity / maximum;
    const Real denominator = negative_scaled + positive_scaled;
    return {
        positive_scaled / denominator,
        negative_scaled / denominator,
        negative_scaled / denominator,
        positive_scaled / denominator};
}

void accumulatePhasePoint(
    IncompressibleTwoFluidPhaseDiagnosticAccumulator& phase,
    const FreeSurfaceDiscreteFunctionalVectorEvaluator& velocity,
    const FreeSurfaceGeometryRuleRecord& record,
    std::size_t point_index,
    Real weight)
{
    const auto& reference = record.reference_rule.points[point_index];
    const auto value = velocity.value(
        record.reference_rule.provenance.parent_entity,
        reference.parent_coordinate,
        record.reference_rule.provenance);
    if (!finiteVector(value)) {
        throw std::invalid_argument(
            "incompressible two-fluid phase velocity evaluator returned a non-finite value");
    }
    ++phase.owned_quadrature_point_count;
    phase.volume += weight;
    for (std::size_t component = 0u; component < 3u; ++component) {
        phase.velocity_integral[component] += value[component] * weight;
    }
    phase.velocity_squared_integral += dot(value, value) * weight;
}

void requireFiniteAccumulator(
    const IncompressibleTwoFluidDiagnosticAccumulator& value)
{
    const std::array<Real, 19> scalars{
        value.interface_measure,
        value.velocity_jump_squared,
        value.normal_velocity_jump_squared,
        value.tangential_velocity_jump_squared,
        value.negative_normal_flux,
        value.positive_normal_flux,
        value.traction_jump_normal_integral,
        value.traction_jump_squared,
        value.pressure_jump_integral,
        value.pressure_jump_squared,
        value.surface_energy_work,
        value.nitsche_consistency_work,
        value.nitsche_adjoint_work,
        value.nitsche_penalty_work,
        value.negative_phase.volume,
        value.negative_phase.velocity_squared_integral,
        value.positive_phase.volume,
        value.positive_phase.velocity_squared_integral,
        static_cast<Real>(value.owned_interface_quadrature_point_count)};
    if (!std::all_of(scalars.begin(), scalars.end(), [](Real scalar) {
            return std::isfinite(scalar);
        }) ||
        !finiteVector(value.negative_traction_integral) ||
        !finiteVector(value.positive_traction_integral) ||
        !finiteVector(value.traction_jump_integral) ||
        !finiteVector(value.negative_phase.velocity_integral) ||
        !finiteVector(value.positive_phase.velocity_integral) ||
        (value.prescribed_stress_jump_residual_squared.has_value() &&
         !std::isfinite(*value.prescribed_stress_jump_residual_squared)) ||
        (value.prescribed_pressure_jump_error_squared.has_value() &&
         !std::isfinite(*value.prescribed_pressure_jump_error_squared)) ||
        (value.transient_penalty_effective_dt.has_value() &&
         (!std::isfinite(*value.transient_penalty_effective_dt) ||
          !(*value.transient_penalty_effective_dt > Real{0.0})))) {
        throw std::invalid_argument(
            "incompressible two-fluid diagnostics produced a non-finite accumulator");
    }
}

} // namespace

IncompressibleTwoFluidDiagnosticAccumulator
evaluateLocalIncompressibleTwoFluidDiagnostics(
    const FreeSurfaceGeometrySnapshot& snapshot,
    const IncompressibleTwoFluidDiagnosticParameters& parameters,
    const IncompressibleTwoFluidPhaseEvaluators& negative_phase,
    const IncompressibleTwoFluidPhaseEvaluators& positive_phase,
    const IncompressibleTwoFluidCellMeasureEvaluator& cell_measure,
    Real effective_dt)
{
    validateParameters(parameters);
    validateEvaluators(negative_phase, "negative");
    validateEvaluators(positive_phase, "positive");
    if (!snapshot.revision().complete() ||
        snapshot.revision().interface_marker != parameters.interface_marker ||
        !cell_measure.canEvaluate() || !std::isfinite(effective_dt) ||
        (parameters.include_transient_penalty && !(effective_dt > Real{0.0}))) {
        throw std::invalid_argument(
            "incompressible two-fluid diagnostics require a matching revision-complete snapshot, parent-cell measure evaluator, and valid effective step");
    }

    const auto weights = interfaceWeights(parameters);
    const Real harmonic_viscosity = stableHarmonicMean(
        parameters.negative_viscosity,
        parameters.positive_viscosity);
    const Real harmonic_density = stableHarmonicMean(
        parameters.negative_density,
        parameters.positive_density);

    IncompressibleTwoFluidDiagnosticAccumulator state;
    state.snapshot_revision_key = snapshot.revision().snapshot_revision_key;
    if (parameters.include_transient_penalty) {
        state.transient_penalty_effective_dt = effective_dt;
    }
    if (parameters.prescribed_pressure_jump.has_value()) {
        state.prescribed_stress_jump_residual_squared = Real{0.0};
        state.prescribed_pressure_jump_error_squared = Real{0.0};
    }

    for (const auto& record : snapshot.rules()) {
        if (!record.locally_owned ||
            record.retention != FreeSurfaceGeometryRetention::Retained) {
            continue;
        }
        const bool negative_volume =
            record.role == FreeSurfaceGeometryRuleRole::NegativeVolume;
        const bool positive_volume =
            record.role == FreeSurfaceGeometryRuleRole::PositiveVolume;
        if (negative_volume || positive_volume) {
            const auto expected_side = negative_volume
                                           ? geometry::CutIntegrationSide::Negative
                                           : geometry::CutIntegrationSide::Positive;
            if (record.reference_rule.kind !=
                    geometry::CutQuadratureKind::Volume ||
                record.physical_rule.kind !=
                    geometry::CutQuadratureKind::Volume ||
                record.reference_rule.side != expected_side ||
                record.physical_rule.side != expected_side ||
                record.reference_rule.points.size() !=
                    record.physical_rule.points.size()) {
                throw std::invalid_argument(
                    "incompressible two-fluid diagnostics encountered an inconsistent phase-volume rule");
            }
            for (std::size_t point_index = 0u;
                 point_index < record.physical_rule.points.size();
                 ++point_index) {
                const auto& physical =
                    record.physical_rule.points[point_index];
                const Real weight = physical.physical_weight;
                if (!std::isfinite(weight) || !(weight > Real{0.0}) ||
                    !finiteVector(physical.physical_point)) {
                    throw std::invalid_argument(
                        "incompressible two-fluid diagnostics encountered invalid phase-volume quadrature");
                }
                accumulatePhasePoint(
                    negative_volume ? state.negative_phase
                                    : state.positive_phase,
                    negative_volume ? negative_phase.velocity
                                    : positive_phase.velocity,
                    record,
                    point_index,
                    weight);
            }
            continue;
        }
        if (record.role != FreeSurfaceGeometryRuleRole::Interface) {
            continue;
        }
        if (record.reference_rule.kind !=
                geometry::CutQuadratureKind::Interface ||
            record.physical_rule.kind !=
                geometry::CutQuadratureKind::Interface ||
            record.reference_rule.points.size() !=
                record.physical_rule.points.size() ||
            !std::isfinite(record.physical_rule.physical_measure) ||
            !(record.physical_rule.physical_measure > Real{0.0})) {
            throw std::invalid_argument(
                "incompressible two-fluid diagnostics encountered an inconsistent interface rule");
        }
        const auto parent = record.reference_rule.provenance.parent_entity;
        const Real parent_measure = cell_measure.physical_cell_measure(parent);
        const Real h_normal =
            Real{2.0} * parent_measure /
            record.physical_rule.physical_measure;
        if (!std::isfinite(parent_measure) || !(parent_measure > Real{0.0}) ||
            !std::isfinite(h_normal) || !(h_normal > Real{0.0})) {
            throw std::invalid_argument(
                "incompressible two-fluid diagnostics require a positive finite parent-cell measure and normal length scale");
        }
        Real penalty_scale = harmonic_viscosity / h_normal;
        if (parameters.include_transient_penalty) {
            penalty_scale += harmonic_density * h_normal / effective_dt;
        }
        penalty_scale *= parameters.nitsche_gamma;

        for (std::size_t point_index = 0u;
             point_index < record.physical_rule.points.size();
             ++point_index) {
            const auto& physical = record.physical_rule.points[point_index];
            const auto& reference = record.reference_rule.points[point_index];
            const Real weight = physical.physical_weight;
            if (!std::isfinite(weight) || !(weight > Real{0.0}) ||
                !finiteVector(physical.physical_point) ||
                !finiteVector(physical.normal) ||
                !finiteVector(reference.parent_coordinate)) {
                throw std::invalid_argument(
                    "incompressible two-fluid diagnostics encountered invalid interface quadrature");
            }

            const auto u_negative = negative_phase.velocity.value(
                parent,
                reference.parent_coordinate,
                record.reference_rule.provenance);
            const auto u_positive = positive_phase.velocity.value(
                parent,
                reference.parent_coordinate,
                record.reference_rule.provenance);
            const auto grad_negative =
                negative_phase.velocity.physical_gradient(
                    parent,
                    reference.parent_coordinate,
                    record.reference_rule.provenance);
            const auto grad_positive =
                positive_phase.velocity.physical_gradient(
                    parent,
                    reference.parent_coordinate,
                    record.reference_rule.provenance);
            const Real p_negative = negative_phase.pressure.value(
                parent,
                reference.parent_coordinate,
                record.reference_rule.provenance);
            const Real p_positive = positive_phase.pressure.value(
                parent,
                reference.parent_coordinate,
                record.reference_rule.provenance);
            if (!finiteVector(u_negative) || !finiteVector(u_positive) ||
                !finiteMatrix(grad_negative) || !finiteMatrix(grad_positive) ||
                !std::isfinite(p_negative) || !std::isfinite(p_positive)) {
                throw std::invalid_argument(
                    "incompressible two-fluid diagnostics field evaluator returned a non-finite interface value");
            }

            const auto jump = subtract(u_negative, u_positive);
            const Real normal_jump = dot(jump, physical.normal);
            const Real jump_squared = dot(jump, jump);
            const Real normal_jump_squared = normal_jump * normal_jump;
            Real tangential_jump_squared = jump_squared - normal_jump_squared;
            const Real projection_tolerance =
                Real{128.0} * std::numeric_limits<Real>::epsilon() *
                std::max(Real{1.0}, jump_squared);
            if (tangential_jump_squared < -projection_tolerance) {
                throw std::invalid_argument(
                    "incompressible two-fluid diagnostics velocity-jump projection identity failed");
            }
            tangential_jump_squared =
                std::max(Real{0.0}, tangential_jump_squared);

            const auto t_negative = traction(
                grad_negative,
                p_negative,
                parameters.negative_viscosity,
                physical.normal,
                parameters.dimension);
            const auto t_positive = traction(
                grad_positive,
                p_positive,
                parameters.positive_viscosity,
                physical.normal,
                parameters.dimension);
            const auto traction_jump = subtract(t_negative, t_positive);
            Vector weighted_traction{};
            Matrix complementary_gradient{};
            for (int row = 0; row < parameters.dimension; ++row) {
                const auto r = static_cast<std::size_t>(row);
                weighted_traction[r] =
                    weights[0] * t_negative[r] +
                    weights[1] * t_positive[r];
                for (int column = 0; column < parameters.dimension; ++column) {
                    const auto c = static_cast<std::size_t>(column);
                    complementary_gradient[r][c] =
                        weights[2] * grad_negative[r][c] +
                        weights[3] * grad_positive[r][c];
                }
            }
            Real projector_gradient_contraction{0.0};
            for (int row = 0; row < parameters.dimension; ++row) {
                for (int column = 0; column < parameters.dimension; ++column) {
                    const auto r = static_cast<std::size_t>(row);
                    const auto c = static_cast<std::size_t>(column);
                    const Real projector =
                        (row == column ? Real{1.0} : Real{0.0}) -
                        physical.normal[r] * physical.normal[c];
                    projector_gradient_contraction +=
                        projector * complementary_gradient[r][c];
                }
            }

            const Real pressure_jump = p_negative - p_positive;
            const Real consistency_density = -dot(weighted_traction, jump);
            ++state.owned_interface_quadrature_point_count;
            state.interface_measure += weight;
            state.velocity_jump_squared += jump_squared * weight;
            state.normal_velocity_jump_squared +=
                normal_jump_squared * weight;
            state.tangential_velocity_jump_squared +=
                tangential_jump_squared * weight;
            state.negative_normal_flux +=
                dot(u_negative, physical.normal) * weight;
            state.positive_normal_flux +=
                dot(u_positive, physical.normal) * weight;
            for (std::size_t component = 0u; component < 3u; ++component) {
                state.negative_traction_integral[component] +=
                    t_negative[component] * weight;
                state.positive_traction_integral[component] +=
                    t_positive[component] * weight;
                state.traction_jump_integral[component] +=
                    traction_jump[component] * weight;
            }
            state.traction_jump_squared +=
                dot(traction_jump, traction_jump) * weight;
            state.traction_jump_normal_integral +=
                dot(traction_jump, physical.normal) * weight;
            state.pressure_jump_integral += pressure_jump * weight;
            state.pressure_jump_squared +=
                pressure_jump * pressure_jump * weight;
            state.surface_energy_work +=
                parameters.surface_tension *
                projector_gradient_contraction * weight;
            state.nitsche_consistency_work += consistency_density * weight;
            state.nitsche_adjoint_work += consistency_density * weight;
            state.nitsche_penalty_work +=
                penalty_scale * jump_squared * weight;

            if (parameters.prescribed_pressure_jump.has_value()) {
                const Real target = *parameters.prescribed_pressure_jump;
                const Real pressure_error = pressure_jump - target;
                *state.prescribed_pressure_jump_error_squared +=
                    pressure_error * pressure_error * weight;
                Vector stress_residual = traction_jump;
                for (int component = 0;
                     component < parameters.dimension;
                     ++component) {
                    const auto c = static_cast<std::size_t>(component);
                    stress_residual[c] += target * physical.normal[c];
                }
                *state.prescribed_stress_jump_residual_squared +=
                    dot(stress_residual, stress_residual) * weight;
            }
        }
    }
    requireFiniteAccumulator(state);
    return state;
}

IncompressibleTwoFluidDiagnosticState
finalizeIncompressibleTwoFluidDiagnostics(
    const IncompressibleTwoFluidDiagnosticAccumulator& accumulator,
    const IncompressibleTwoFluidDiagnosticParameters& parameters)
{
    validateParameters(parameters);
    requireFiniteAccumulator(accumulator);
    if (accumulator.snapshot_revision_key == 0u ||
        accumulator.owned_interface_quadrature_point_count == 0u ||
        !(accumulator.interface_measure > Real{0.0}) ||
        !(accumulator.negative_phase.volume > Real{0.0}) ||
        !(accumulator.positive_phase.volume > Real{0.0}) ||
        accumulator.velocity_jump_squared < Real{0.0} ||
        accumulator.normal_velocity_jump_squared < Real{0.0} ||
        accumulator.tangential_velocity_jump_squared < Real{0.0} ||
        accumulator.traction_jump_squared < Real{0.0} ||
        accumulator.pressure_jump_squared < Real{0.0} ||
        accumulator.nitsche_penalty_work < Real{0.0} ||
        accumulator.negative_phase.velocity_squared_integral < Real{0.0} ||
        accumulator.positive_phase.velocity_squared_integral < Real{0.0} ||
        accumulator.negative_phase.owned_quadrature_point_count == 0u ||
        accumulator.positive_phase.owned_quadrature_point_count == 0u ||
        accumulator.prescribed_pressure_jump_error_squared.has_value() !=
            parameters.prescribed_pressure_jump.has_value() ||
        accumulator.prescribed_stress_jump_residual_squared.has_value() !=
            parameters.prescribed_pressure_jump.has_value() ||
        accumulator.transient_penalty_effective_dt.has_value() !=
            parameters.include_transient_penalty) {
        throw std::invalid_argument(
            "incompressible two-fluid diagnostics require positive two-phase coverage, nonnegative squared measures, and exact target applicability");
    }
    const Real projection_residual =
        accumulator.velocity_jump_squared -
        accumulator.normal_velocity_jump_squared -
        accumulator.tangential_velocity_jump_squared;
    const Real projection_tolerance =
        Real{512.0} * std::numeric_limits<Real>::epsilon() *
        std::max({Real{1.0},
                  std::abs(accumulator.velocity_jump_squared),
                  std::abs(accumulator.normal_velocity_jump_squared),
                  std::abs(accumulator.tangential_velocity_jump_squared)});
    if (std::abs(projection_residual) > projection_tolerance) {
        throw std::invalid_argument(
            "incompressible two-fluid diagnostics global velocity-jump projection identity failed");
    }
    const auto moment_tolerance = [](Real lhs, Real rhs) {
        return Real{2048.0} * std::numeric_limits<Real>::epsilon() *
               std::max({Real{1.0}, std::abs(lhs), std::abs(rhs)});
    };
    const Real flux_jump =
        accumulator.negative_normal_flux -
        accumulator.positive_normal_flux;
    const Real flux_bound =
        accumulator.interface_measure *
        accumulator.normal_velocity_jump_squared;
    if (flux_jump * flux_jump >
        flux_bound + moment_tolerance(flux_jump * flux_jump, flux_bound)) {
        throw std::invalid_argument(
            "incompressible two-fluid diagnostics normal-flux moment bound failed");
    }
    const Real pressure_bound =
        accumulator.interface_measure * accumulator.pressure_jump_squared;
    const Real pressure_moment_squared =
        accumulator.pressure_jump_integral *
        accumulator.pressure_jump_integral;
    if (pressure_moment_squared >
        pressure_bound +
            moment_tolerance(pressure_moment_squared, pressure_bound)) {
        throw std::invalid_argument(
            "incompressible two-fluid diagnostics pressure-jump moment bound failed");
    }
    const Real traction_moment_squared =
        dot(accumulator.traction_jump_integral,
            accumulator.traction_jump_integral);
    const Real traction_bound =
        accumulator.interface_measure * accumulator.traction_jump_squared;
    if (traction_moment_squared >
        traction_bound +
            moment_tolerance(traction_moment_squared, traction_bound)) {
        throw std::invalid_argument(
            "incompressible two-fluid diagnostics traction-jump moment bound failed");
    }
    for (std::size_t component = 0u; component < 3u; ++component) {
        const Real expected =
            accumulator.negative_traction_integral[component] -
            accumulator.positive_traction_integral[component];
        if (std::abs(accumulator.traction_jump_integral[component] - expected) >
            moment_tolerance(
                accumulator.traction_jump_integral[component], expected)) {
            throw std::invalid_argument(
                "incompressible two-fluid diagnostics traction-jump integral identity failed");
        }
    }
    const auto validate_phase_moment = [&](const auto& phase) {
        const Real velocity_moment_squared =
            dot(phase.velocity_integral, phase.velocity_integral);
        const Real velocity_bound =
            phase.volume * phase.velocity_squared_integral;
        return velocity_moment_squared <=
               velocity_bound +
                   moment_tolerance(velocity_moment_squared, velocity_bound);
    };
    if (!validate_phase_moment(accumulator.negative_phase) ||
        !validate_phase_moment(accumulator.positive_phase)) {
        throw std::invalid_argument(
            "incompressible two-fluid diagnostics phase momentum-energy moment bound failed");
    }
    if (parameters.prescribed_pressure_jump.has_value()) {
        const Real target = *parameters.prescribed_pressure_jump;
        const Real expected_pressure_error =
            accumulator.pressure_jump_squared -
            Real{2.0} * target * accumulator.pressure_jump_integral +
            target * target * accumulator.interface_measure;
        const Real expected_stress_error =
            accumulator.traction_jump_squared +
            Real{2.0} * target *
                accumulator.traction_jump_normal_integral +
            target * target * accumulator.interface_measure;
        if (std::abs(
                *accumulator.prescribed_pressure_jump_error_squared -
                expected_pressure_error) >
                moment_tolerance(
                    *accumulator.prescribed_pressure_jump_error_squared,
                    expected_pressure_error) ||
            std::abs(
                *accumulator.prescribed_stress_jump_residual_squared -
                expected_stress_error) >
                moment_tolerance(
                    *accumulator.prescribed_stress_jump_residual_squared,
                    expected_stress_error)) {
            throw std::invalid_argument(
                "incompressible two-fluid diagnostics prescribed-jump moment identity failed");
        }
    }

    IncompressibleTwoFluidDiagnosticState state;
    state.snapshot_revision_key = accumulator.snapshot_revision_key;
    state.transient_penalty_effective_dt =
        accumulator.transient_penalty_effective_dt;
    state.interface_quadrature_point_count =
        accumulator.owned_interface_quadrature_point_count;
    state.interface_measure = accumulator.interface_measure;
    state.velocity_jump_squared = accumulator.velocity_jump_squared;
    state.normal_velocity_jump_squared =
        accumulator.normal_velocity_jump_squared;
    state.tangential_velocity_jump_squared =
        accumulator.tangential_velocity_jump_squared;
    state.negative_normal_flux = accumulator.negative_normal_flux;
    state.positive_normal_flux = accumulator.positive_normal_flux;
    state.normal_flux_jump =
        accumulator.negative_normal_flux - accumulator.positive_normal_flux;
    state.negative_mass_flux =
        parameters.negative_density * accumulator.negative_normal_flux;
    state.positive_mass_flux =
        parameters.positive_density * accumulator.positive_normal_flux;
    state.negative_traction_integral =
        accumulator.negative_traction_integral;
    state.positive_traction_integral =
        accumulator.positive_traction_integral;
    state.traction_jump_integral = accumulator.traction_jump_integral;
    state.traction_jump_normal_integral =
        accumulator.traction_jump_normal_integral;
    state.traction_jump_squared = accumulator.traction_jump_squared;
    state.prescribed_stress_jump_residual_squared =
        accumulator.prescribed_stress_jump_residual_squared;
    state.pressure_jump_integral = accumulator.pressure_jump_integral;
    state.mean_pressure_jump =
        accumulator.pressure_jump_integral / accumulator.interface_measure;
    state.pressure_jump_squared = accumulator.pressure_jump_squared;
    state.prescribed_pressure_jump_error_squared =
        accumulator.prescribed_pressure_jump_error_squared;
    state.surface_energy_work = accumulator.surface_energy_work;
    state.nitsche_consistency_work = accumulator.nitsche_consistency_work;
    state.nitsche_adjoint_work = accumulator.nitsche_adjoint_work;
    state.nitsche_penalty_work = accumulator.nitsche_penalty_work;

    const auto finalize_phase = [](const auto& source,
                                   geometry::CutIntegrationSide side,
                                   Real density) {
        IncompressibleTwoFluidPhaseDiagnosticState phase;
        phase.side = side;
        phase.quadrature_point_count = source.owned_quadrature_point_count;
        phase.density = density;
        phase.volume = source.volume;
        phase.mass = density * source.volume;
        for (std::size_t component = 0u; component < 3u; ++component) {
            phase.momentum[component] =
                density * source.velocity_integral[component];
        }
        phase.kinetic_energy =
            Real{0.5} * density * source.velocity_squared_integral;
        return phase;
    };
    state.negative_phase = finalize_phase(
        accumulator.negative_phase,
        geometry::CutIntegrationSide::Negative,
        parameters.negative_density);
    state.positive_phase = finalize_phase(
        accumulator.positive_phase,
        geometry::CutIntegrationSide::Positive,
        parameters.positive_density);

    const std::array<Real, 14> derived{
        state.normal_flux_jump,
        state.negative_mass_flux,
        state.positive_mass_flux,
        state.mean_pressure_jump,
        state.negative_phase.mass,
        state.negative_phase.momentum[0],
        state.negative_phase.momentum[1],
        state.negative_phase.momentum[2],
        state.negative_phase.kinetic_energy,
        state.positive_phase.mass,
        state.positive_phase.momentum[0],
        state.positive_phase.momentum[1],
        state.positive_phase.momentum[2],
        state.positive_phase.kinetic_energy};
    if (!std::all_of(derived.begin(), derived.end(), [](Real value) {
            return std::isfinite(value);
        })) {
        throw std::invalid_argument(
            "incompressible two-fluid diagnostics produced a non-finite finalized state");
    }
    return state;
}

namespace {

[[nodiscard]] Real vectorNorm(const std::array<Real, 3>& value) noexcept
{
    const long double squared =
        static_cast<long double>(value[0]) * value[0] +
        static_cast<long double>(value[1]) * value[1] +
        static_cast<long double>(value[2]) * value[2];
    return static_cast<Real>(std::sqrt(std::max(0.0L, squared)));
}

[[nodiscard]] bool reconciliationNear(Real lhs, Real rhs) noexcept
{
    const Real scale = std::max({Real{1.0}, std::abs(lhs), std::abs(rhs)});
    return std::abs(lhs - rhs) <=
           Real{64.0} * std::numeric_limits<Real>::epsilon() * scale;
}

[[nodiscard]] IncompressibleTwoFluidPhaseMomentumReconciliation
buildPhaseMomentumReconciliation(
    const IncompressibleTwoFluidPhaseDiagnosticState& raw,
    const IncompressibleTwoFluidPhaseDiagnosticState& corrected,
    Real relative_tolerance)
{
    if (raw.side != corrected.side || !std::isfinite(raw.density) ||
        !std::isfinite(corrected.density) ||
        !(raw.density > Real{0.0}) || raw.density != corrected.density ||
        !std::isfinite(raw.volume) ||
        !std::isfinite(corrected.volume) || !std::isfinite(raw.mass) ||
        !std::isfinite(corrected.mass) || raw.volume < Real{0.0} ||
        corrected.volume < Real{0.0} || raw.mass < Real{0.0} ||
        corrected.mass < Real{0.0} ||
        !reconciliationNear(raw.mass, raw.density * raw.volume) ||
        !reconciliationNear(
            corrected.mass, corrected.density * corrected.volume) ||
        !finiteVector(raw.momentum) ||
        !finiteVector(corrected.momentum)) {
        throw std::invalid_argument(
            "incompressible two-fluid momentum reconciliation received invalid phase states");
    }
    IncompressibleTwoFluidPhaseMomentumReconciliation phase;
    phase.side = raw.side;
    phase.density = raw.density;
    phase.raw_volume = raw.volume;
    phase.corrected_volume = corrected.volume;
    phase.raw_mass = raw.mass;
    phase.corrected_mass = corrected.mass;
    phase.raw_momentum = raw.momentum;
    phase.corrected_momentum = corrected.momentum;
    for (std::size_t component = 0u; component < 3u; ++component) {
        phase.momentum_delta[component] =
            corrected.momentum[component] - raw.momentum[component];
    }
    phase.momentum_delta_norm = vectorNorm(phase.momentum_delta);
    phase.momentum_reference_norm =
        std::max(vectorNorm(raw.momentum), vectorNorm(corrected.momentum));
    phase.allowed_momentum_delta =
        relative_tolerance * phase.momentum_reference_norm;
    phase.satisfied =
        phase.momentum_delta_norm <= phase.allowed_momentum_delta;
    return phase;
}

void validatePhaseMomentumReconciliation(
    const IncompressibleTwoFluidPhaseMomentumReconciliation& phase,
    geometry::CutIntegrationSide expected_side,
    Real relative_tolerance)
{
    if (phase.side != expected_side || !std::isfinite(phase.density) ||
        !(phase.density > Real{0.0}) ||
        !std::isfinite(phase.raw_volume) ||
        !std::isfinite(phase.corrected_volume) ||
        !std::isfinite(phase.raw_mass) ||
        !std::isfinite(phase.corrected_mass) ||
        phase.raw_volume < Real{0.0} ||
        phase.corrected_volume < Real{0.0} ||
        phase.raw_mass < Real{0.0} || phase.corrected_mass < Real{0.0} ||
        !reconciliationNear(
            phase.raw_mass, phase.density * phase.raw_volume) ||
        !reconciliationNear(
            phase.corrected_mass,
            phase.density * phase.corrected_volume) ||
        !finiteVector(phase.raw_momentum) ||
        !finiteVector(phase.corrected_momentum) ||
        !finiteVector(phase.momentum_delta) ||
        !std::isfinite(phase.momentum_delta_norm) ||
        !std::isfinite(phase.momentum_reference_norm) ||
        !std::isfinite(phase.allowed_momentum_delta) ||
        phase.momentum_delta_norm < Real{0.0} ||
        phase.momentum_reference_norm < Real{0.0} ||
        phase.allowed_momentum_delta < Real{0.0}) {
        throw std::invalid_argument(
            "incompressible two-fluid momentum reconciliation phase record is invalid");
    }
    std::array<Real, 3> expected_delta{};
    for (std::size_t component = 0u; component < 3u; ++component) {
        expected_delta[component] =
            phase.corrected_momentum[component] -
            phase.raw_momentum[component];
        if (!reconciliationNear(
                phase.momentum_delta[component],
                expected_delta[component])) {
            throw std::invalid_argument(
                "incompressible two-fluid momentum reconciliation delta is inconsistent");
        }
    }
    const Real expected_delta_norm = vectorNorm(expected_delta);
    const Real expected_reference_norm = std::max(
        vectorNorm(phase.raw_momentum),
        vectorNorm(phase.corrected_momentum));
    const Real expected_allowed =
        relative_tolerance * expected_reference_norm;
    const bool expected_satisfied =
        expected_delta_norm <= expected_allowed;
    if (!reconciliationNear(
            phase.momentum_delta_norm, expected_delta_norm) ||
        !reconciliationNear(
            phase.momentum_reference_norm, expected_reference_norm) ||
        !reconciliationNear(
            phase.allowed_momentum_delta, expected_allowed) ||
        phase.satisfied != expected_satisfied) {
        throw std::invalid_argument(
            "incompressible two-fluid momentum reconciliation phase gate is inconsistent");
    }
}

} // namespace

IncompressibleTwoFluidMomentumReconciliation
buildIncompressibleTwoFluidMomentumReconciliation(
    int interface_marker,
    const FreeSurfaceGeometryRevision& raw_geometry_revision,
    const IncompressibleTwoFluidDiagnosticState& raw,
    std::uint64_t raw_algebraic_revision,
    const FreeSurfaceGeometryRevision& corrected_geometry_revision,
    const IncompressibleTwoFluidDiagnosticState& corrected,
    std::uint64_t corrected_algebraic_revision,
    Real relative_tolerance)
{
    if (interface_marker < 0 || !std::isfinite(relative_tolerance) ||
        !(relative_tolerance > Real{0.0}) ||
        raw_algebraic_revision == 0u ||
        corrected_algebraic_revision == 0u ||
        raw.snapshot_revision_key == 0u ||
        corrected.snapshot_revision_key == 0u ||
        !raw_geometry_revision.complete() ||
        !corrected_geometry_revision.complete() ||
        raw_geometry_revision.interface_marker != interface_marker ||
        corrected_geometry_revision.interface_marker != interface_marker ||
        raw_geometry_revision.snapshot_revision_key !=
            raw.snapshot_revision_key ||
        corrected_geometry_revision.snapshot_revision_key !=
            corrected.snapshot_revision_key ||
        raw_geometry_revision.source_id !=
            corrected_geometry_revision.source_id ||
        raw_geometry_revision.domain_id !=
            corrected_geometry_revision.domain_id ||
        raw_geometry_revision.isovalue !=
            corrected_geometry_revision.isovalue ||
        raw_geometry_revision.source_layout_revision !=
            corrected_geometry_revision.source_layout_revision ||
        raw_geometry_revision.mesh_geometry_revision !=
            corrected_geometry_revision.mesh_geometry_revision ||
        raw_geometry_revision.mesh_topology_revision !=
            corrected_geometry_revision.mesh_topology_revision ||
        raw_geometry_revision.ownership_revision !=
            corrected_geometry_revision.ownership_revision ||
        raw_geometry_revision.numbering_revision !=
            corrected_geometry_revision.numbering_revision ||
        raw_geometry_revision.quadrature_policy_key !=
            corrected_geometry_revision.quadrature_policy_key) {
        throw std::invalid_argument(
            "incompressible two-fluid momentum reconciliation provenance is invalid");
    }
    IncompressibleTwoFluidMomentumReconciliation reconciliation;
    reconciliation.interface_marker = interface_marker;
    reconciliation.raw_geometry_revision = raw_geometry_revision;
    reconciliation.corrected_geometry_revision = corrected_geometry_revision;
    reconciliation.raw_algebraic_revision = raw_algebraic_revision;
    reconciliation.corrected_algebraic_revision =
        corrected_algebraic_revision;
    reconciliation.relative_tolerance = relative_tolerance;
    reconciliation.velocity_update_applied = false;
    reconciliation.negative_phase = buildPhaseMomentumReconciliation(
        raw.negative_phase, corrected.negative_phase, relative_tolerance);
    reconciliation.positive_phase = buildPhaseMomentumReconciliation(
        raw.positive_phase, corrected.positive_phase, relative_tolerance);
    reconciliation.satisfied =
        reconciliation.negative_phase.satisfied &&
        reconciliation.positive_phase.satisfied;
    validateIncompressibleTwoFluidMomentumReconciliation(reconciliation);
    return reconciliation;
}

void validateIncompressibleTwoFluidMomentumReconciliation(
    const IncompressibleTwoFluidMomentumReconciliation& reconciliation)
{
    const auto& raw_geometry = reconciliation.raw_geometry_revision;
    const auto& corrected_geometry =
        reconciliation.corrected_geometry_revision;
    if (reconciliation.interface_marker < 0 ||
        !std::isfinite(reconciliation.relative_tolerance) ||
        !(reconciliation.relative_tolerance > Real{0.0}) ||
        reconciliation.raw_algebraic_revision == 0u ||
        reconciliation.corrected_algebraic_revision == 0u ||
        reconciliation.velocity_update_applied ||
        !raw_geometry.complete() || !corrected_geometry.complete() ||
        raw_geometry.interface_marker != reconciliation.interface_marker ||
        corrected_geometry.interface_marker !=
            reconciliation.interface_marker ||
        raw_geometry.snapshot_revision_key == 0u ||
        corrected_geometry.snapshot_revision_key == 0u ||
        raw_geometry.source_id != corrected_geometry.source_id ||
        raw_geometry.domain_id != corrected_geometry.domain_id ||
        raw_geometry.isovalue != corrected_geometry.isovalue ||
        raw_geometry.source_layout_revision !=
            corrected_geometry.source_layout_revision ||
        raw_geometry.mesh_geometry_revision !=
            corrected_geometry.mesh_geometry_revision ||
        raw_geometry.mesh_topology_revision !=
            corrected_geometry.mesh_topology_revision ||
        raw_geometry.ownership_revision !=
            corrected_geometry.ownership_revision ||
        raw_geometry.numbering_revision !=
            corrected_geometry.numbering_revision ||
        raw_geometry.quadrature_policy_key !=
            corrected_geometry.quadrature_policy_key) {
        throw std::invalid_argument(
            "incompressible two-fluid momentum reconciliation metadata is invalid");
    }
    validatePhaseMomentumReconciliation(
        reconciliation.negative_phase,
        geometry::CutIntegrationSide::Negative,
        reconciliation.relative_tolerance);
    validatePhaseMomentumReconciliation(
        reconciliation.positive_phase,
        geometry::CutIntegrationSide::Positive,
        reconciliation.relative_tolerance);
    if (reconciliation.satisfied !=
        (reconciliation.negative_phase.satisfied &&
         reconciliation.positive_phase.satisfied)) {
        throw std::invalid_argument(
            "incompressible two-fluid momentum reconciliation aggregate gate is inconsistent");
    }
}

} // namespace svmp::FE::interfaces

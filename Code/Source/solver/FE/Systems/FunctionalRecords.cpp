/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Systems/FunctionalRecords.h"

#include "Core/FEException.h"
#include "Core/Logger.h"
#include "Spaces/FunctionSpace.h"
#include "Systems/FreeSurfaceFunctionalRounding.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>

namespace svmp {
namespace FE {
namespace systems {

namespace {

bool sameFreeSurfaceGeometryRevision(
    const interfaces::FreeSurfaceGeometryRevision& lhs,
    const interfaces::FreeSurfaceGeometryRevision& rhs) noexcept
{
    return lhs.sameSourceState(rhs) &&
           lhs.snapshot_revision_key == rhs.snapshot_revision_key;
}

bool sameFreeSurfaceFunctionalParameters(
    const interfaces::FreeSurfaceDiscreteFunctionalParameters& lhs,
    const interfaces::FreeSurfaceDiscreteFunctionalParameters& rhs) noexcept
{
    if (lhs.liquid_side != rhs.liquid_side ||
        lhs.surface_tension != rhs.surface_tension ||
        lhs.volume_multiplier != rhs.volume_multiplier ||
        lhs.young_wall_coefficients.size() !=
            rhs.young_wall_coefficients.size() ||
        lhs.dynamic_contact_coefficients.size() !=
            rhs.dynamic_contact_coefficients.size()) {
        return false;
    }
    for (std::size_t i = 0; i < lhs.young_wall_coefficients.size(); ++i) {
        const auto& left = lhs.young_wall_coefficients[i];
        const auto& right = rhs.young_wall_coefficients[i];
        if (left.boundary_marker != right.boundary_marker ||
            left.equilibrium_contact_angle_radians !=
                right.equilibrium_contact_angle_radians) {
            return false;
        }
    }
    for (std::size_t i = 0; i < lhs.dynamic_contact_coefficients.size();
         ++i) {
        const auto& left = lhs.dynamic_contact_coefficients[i];
        const auto& right = rhs.dynamic_contact_coefficients[i];
        if (left.boundary_marker != right.boundary_marker ||
            left.equilibrium_contact_angle_radians !=
                right.equilibrium_contact_angle_radians ||
            left.law != right.law ||
            left.mobility != right.mobility ||
            left.slip_length != right.slip_length ||
            left.dynamic_viscosity != right.dynamic_viscosity) {
            return false;
        }
    }
    return true;
}

bool sameFreeSurfaceActiveVolumeEnergyParameters(
    const interfaces::FreeSurfaceActiveVolumeEnergyParameters& lhs,
    const interfaces::FreeSurfaceActiveVolumeEnergyParameters& rhs) noexcept
{
    return lhs.liquid_side == rhs.liquid_side &&
           lhs.density == rhs.density &&
           lhs.gravitational_acceleration ==
               rhs.gravitational_acceleration &&
           lhs.gravitational_reference_point ==
               rhs.gravitational_reference_point;
}

bool sameFreeSurfaceActiveVolumeEnergyState(
    const interfaces::FreeSurfaceActiveVolumeEnergyState& lhs,
    const interfaces::FreeSurfaceActiveVolumeEnergyState& rhs) noexcept
{
    return lhs.snapshot_revision_key == rhs.snapshot_revision_key &&
           lhs.liquid_side == rhs.liquid_side &&
           lhs.density == rhs.density &&
           lhs.gravitational_acceleration ==
               rhs.gravitational_acceleration &&
           lhs.gravitational_reference_point ==
               rhs.gravitational_reference_point &&
           lhs.owned_quadrature_point_count ==
               rhs.owned_quadrature_point_count &&
           lhs.owned_liquid_volume == rhs.owned_liquid_volume &&
           lhs.kinetic_energy == rhs.kinetic_energy &&
           lhs.gravitational_energy == rhs.gravitational_energy &&
           lhs.gravitational_potential_power ==
               rhs.gravitational_potential_power &&
           lhs.total_energy == rhs.total_energy;
}

bool sameFreeSurfaceActiveVolumeDissipationParameters(
    const interfaces::FreeSurfaceActiveVolumeDissipationParameters& lhs,
    const interfaces::FreeSurfaceActiveVolumeDissipationParameters& rhs)
    noexcept
{
    return lhs.liquid_side == rhs.liquid_side &&
           lhs.dynamic_viscosity == rhs.dynamic_viscosity;
}

bool sameFreeSurfaceActiveVolumeDissipationState(
    const interfaces::FreeSurfaceActiveVolumeDissipationState& lhs,
    const interfaces::FreeSurfaceActiveVolumeDissipationState& rhs) noexcept
{
    return lhs.snapshot_revision_key == rhs.snapshot_revision_key &&
           lhs.liquid_side == rhs.liquid_side &&
           lhs.dynamic_viscosity == rhs.dynamic_viscosity &&
           lhs.owned_quadrature_point_count ==
               rhs.owned_quadrature_point_count &&
           lhs.owned_liquid_volume == rhs.owned_liquid_volume &&
           lhs.bulk_viscous_dissipation_rate ==
               rhs.bulk_viscous_dissipation_rate;
}

bool sameFreeSurfaceExternalPressurePowerParameters(
    const interfaces::FreeSurfaceExternalPressurePowerParameters& lhs,
    const interfaces::FreeSurfaceExternalPressurePowerParameters& rhs)
    noexcept
{
    return lhs.liquid_side == rhs.liquid_side &&
           lhs.external_pressure == rhs.external_pressure;
}

bool sameFreeSurfaceDiscreteFunctionalDeclaration(
    const FreeSurfaceDiscreteFunctionalDeclaration& lhs,
    const FreeSurfaceDiscreteFunctionalDeclaration& rhs) noexcept
{
    if (lhs.interface_marker != rhs.interface_marker ||
        lhs.level_set_field != rhs.level_set_field ||
        lhs.curvature_field != rhs.curvature_field ||
        lhs.velocity_field != rhs.velocity_field ||
        lhs.geometry_domain_id != rhs.geometry_domain_id ||
        !sameFreeSurfaceFunctionalParameters(
            lhs.parameters, rhs.parameters) ||
        lhs.active_volume_energy_parameters.has_value() !=
            rhs.active_volume_energy_parameters.has_value() ||
        lhs.static_conservative_body_force_complete !=
            rhs.static_conservative_body_force_complete ||
        lhs.active_volume_dissipation_parameters.has_value() !=
            rhs.active_volume_dissipation_parameters.has_value() ||
        lhs.external_pressure_power_parameters.has_value() !=
            rhs.external_pressure_power_parameters.has_value() ||
        lhs.endpoint_functional_power_enabled !=
            rhs.endpoint_functional_power_enabled ||
        lhs.capillary_balance_method != rhs.capillary_balance_method ||
        lhs.capillary_balance_qualification !=
            rhs.capillary_balance_qualification ||
        lhs.owner_component != rhs.owner_component) {
        return false;
    }
    return (!lhs.active_volume_energy_parameters.has_value() ||
            sameFreeSurfaceActiveVolumeEnergyParameters(
                *lhs.active_volume_energy_parameters,
                *rhs.active_volume_energy_parameters)) &&
           (!lhs.active_volume_dissipation_parameters.has_value() ||
            sameFreeSurfaceActiveVolumeDissipationParameters(
                *lhs.active_volume_dissipation_parameters,
                *rhs.active_volume_dissipation_parameters)) &&
           (!lhs.external_pressure_power_parameters.has_value() ||
            sameFreeSurfaceExternalPressurePowerParameters(
                *lhs.external_pressure_power_parameters,
                *rhs.external_pressure_power_parameters));
}

bool sameFreeSurfaceExternalPressurePowerState(
    const interfaces::FreeSurfaceExternalPressurePowerState& lhs,
    const interfaces::FreeSurfaceExternalPressurePowerState& rhs) noexcept
{
    return lhs.snapshot_revision_key == rhs.snapshot_revision_key &&
           lhs.liquid_side == rhs.liquid_side &&
           lhs.external_pressure == rhs.external_pressure &&
           lhs.owned_quadrature_point_count ==
               rhs.owned_quadrature_point_count &&
           lhs.owned_liquid_gas_area ==
               rhs.owned_liquid_gas_area &&
           lhs.outward_liquid_volume_flux_rate ==
               rhs.outward_liquid_volume_flux_rate &&
           lhs.external_pressure_power ==
               rhs.external_pressure_power;
}

bool sameFreeSurfaceBackwardEulerKineticWorkState(
    const interfaces::FreeSurfaceBackwardEulerKineticWorkState& lhs,
    const interfaces::FreeSurfaceBackwardEulerKineticWorkState& rhs) noexcept
{
    return lhs.snapshot_revision_key == rhs.snapshot_revision_key &&
           lhs.previous_velocity_revision ==
               rhs.previous_velocity_revision &&
           lhs.endpoint_velocity_revision ==
               rhs.endpoint_velocity_revision &&
           lhs.liquid_side == rhs.liquid_side &&
           lhs.density == rhs.density &&
           lhs.owned_quadrature_point_count ==
               rhs.owned_quadrature_point_count &&
           lhs.owned_liquid_volume == rhs.owned_liquid_volume &&
           lhs.kinetic_energy_before_on_endpoint_domain ==
               rhs.kinetic_energy_before_on_endpoint_domain &&
           lhs.kinetic_energy_after == rhs.kinetic_energy_after &&
           lhs.kinetic_energy_change_on_endpoint_domain ==
               rhs.kinetic_energy_change_on_endpoint_domain &&
           lhs.step_integrated_inertia_work ==
               rhs.step_integrated_inertia_work &&
           lhs.time_discretization_loss ==
               rhs.time_discretization_loss &&
           lhs.identity_residual == rhs.identity_residual;
}

bool sameFreeSurfaceFunctionalState(
    const interfaces::FreeSurfaceDiscreteFunctionalState& lhs,
    const interfaces::FreeSurfaceDiscreteFunctionalState& rhs) noexcept
{
    if (lhs.snapshot_revision_key != rhs.snapshot_revision_key ||
        lhs.liquid_side != rhs.liquid_side ||
        lhs.surface_tension != rhs.surface_tension ||
        lhs.volume_multiplier != rhs.volume_multiplier ||
        lhs.owned_liquid_volume != rhs.owned_liquid_volume ||
        lhs.owned_liquid_gas_area != rhs.owned_liquid_gas_area ||
        lhs.owned_wetted_wall_area != rhs.owned_wetted_wall_area ||
        lhs.owned_contact_measure != rhs.owned_contact_measure ||
        lhs.liquid_gas_surface_energy != rhs.liquid_gas_surface_energy ||
        lhs.young_wall_energy != rhs.young_wall_energy ||
        lhs.volume_constraint_potential !=
            rhs.volume_constraint_potential ||
        lhs.total_potential != rhs.total_potential ||
        lhs.walls.size() != rhs.walls.size()) {
        return false;
    }
    for (std::size_t i = 0; i < lhs.walls.size(); ++i) {
        const auto& left = lhs.walls[i];
        const auto& right = rhs.walls[i];
        if (left.boundary_marker != right.boundary_marker ||
            left.equilibrium_contact_angle_radians !=
                right.equilibrium_contact_angle_radians ||
            left.owned_wetted_wall_area !=
                right.owned_wetted_wall_area ||
            left.owned_contact_measure != right.owned_contact_measure ||
            left.young_wall_energy != right.young_wall_energy) {
            return false;
        }
    }
    return true;
}

bool sameFreeSurfaceFunctionalVariationState(
    const interfaces::FreeSurfaceDiscreteFunctionalVariationState& lhs,
    const interfaces::FreeSurfaceDiscreteFunctionalVariationState& rhs)
    noexcept
{
    if (lhs.snapshot_revision_key != rhs.snapshot_revision_key ||
        lhs.liquid_side != rhs.liquid_side ||
        lhs.surface_tension != rhs.surface_tension ||
        lhs.volume_multiplier != rhs.volume_multiplier ||
        lhs.owned_liquid_volume_variation !=
            rhs.owned_liquid_volume_variation ||
        lhs.owned_liquid_gas_area_variation !=
            rhs.owned_liquid_gas_area_variation ||
        lhs.owned_wetted_wall_area_variation !=
            rhs.owned_wetted_wall_area_variation ||
        lhs.liquid_gas_surface_energy_variation !=
            rhs.liquid_gas_surface_energy_variation ||
        lhs.young_wall_energy_variation !=
            rhs.young_wall_energy_variation ||
        lhs.volume_constraint_potential_variation !=
            rhs.volume_constraint_potential_variation ||
        lhs.total_potential_variation !=
            rhs.total_potential_variation ||
        lhs.walls.size() != rhs.walls.size()) {
        return false;
    }
    for (std::size_t i = 0u; i < lhs.walls.size(); ++i) {
        const auto& left = lhs.walls[i];
        const auto& right = rhs.walls[i];
        if (left.boundary_marker != right.boundary_marker ||
            left.equilibrium_contact_angle_radians !=
                right.equilibrium_contact_angle_radians ||
            left.owned_wetted_wall_area_variation !=
                right.owned_wetted_wall_area_variation ||
            left.young_wall_energy_variation !=
                right.young_wall_energy_variation) {
            return false;
        }
    }
    return true;
}

bool sameFreeSurfaceDynamicContactWallState(
    const interfaces::FreeSurfaceDynamicContactWallState& lhs,
    const interfaces::FreeSurfaceDynamicContactWallState& rhs) noexcept
{
    return lhs.boundary_marker == rhs.boundary_marker &&
           lhs.equilibrium_contact_angle_radians ==
               rhs.equilibrium_contact_angle_radians &&
           lhs.law == rhs.law &&
           lhs.mobility == rhs.mobility &&
           lhs.slip_length == rhs.slip_length &&
           lhs.dynamic_viscosity == rhs.dynamic_viscosity &&
           lhs.owned_quadrature_point_count ==
               rhs.owned_quadrature_point_count &&
           lhs.owned_advancing_point_count ==
               rhs.owned_advancing_point_count &&
           lhs.owned_receding_point_count ==
               rhs.owned_receding_point_count &&
           lhs.owned_stationary_point_count ==
               rhs.owned_stationary_point_count &&
           lhs.owned_contact_measure == rhs.owned_contact_measure &&
           lhs.dynamic_angle_integral == rhs.dynamic_angle_integral &&
           lhs.dynamic_cosine_integral == rhs.dynamic_cosine_integral &&
           lhs.contact_speed_integral == rhs.contact_speed_integral &&
           lhs.contact_speed_squared_integral ==
               rhs.contact_speed_squared_integral &&
           lhs.constitutive_residual_integral ==
               rhs.constitutive_residual_integral &&
           lhs.absolute_constitutive_residual_integral ==
               rhs.absolute_constitutive_residual_integral &&
           lhs.line_friction_dissipation ==
               rhs.line_friction_dissipation &&
           lhs.owned_wetted_wall_quadrature_point_count ==
               rhs.owned_wetted_wall_quadrature_point_count &&
           lhs.owned_wetted_wall_measure ==
               rhs.owned_wetted_wall_measure &&
           lhs.wall_slip_speed_integral ==
               rhs.wall_slip_speed_integral &&
           lhs.wall_slip_speed_squared_integral ==
               rhs.wall_slip_speed_squared_integral &&
           lhs.wall_slip_dissipation == rhs.wall_slip_dissipation &&
           lhs.wall_tangential_velocity_integral ==
               rhs.wall_tangential_velocity_integral &&
           lhs.contact_position_integral ==
               rhs.contact_position_integral &&
           lhs.wall_normal_integral == rhs.wall_normal_integral &&
           lhs.footprint_direction_integral ==
               rhs.footprint_direction_integral &&
           lhs.contact_line_tangent_integral ==
               rhs.contact_line_tangent_integral &&
           lhs.mean_dynamic_angle_radians ==
               rhs.mean_dynamic_angle_radians &&
           lhs.mean_dynamic_cosine == rhs.mean_dynamic_cosine &&
           lhs.mean_contact_speed == rhs.mean_contact_speed &&
           lhs.mean_constitutive_residual ==
               rhs.mean_constitutive_residual &&
           lhs.mean_absolute_constitutive_residual ==
               rhs.mean_absolute_constitutive_residual &&
           lhs.mean_wall_slip_speed == rhs.mean_wall_slip_speed &&
           lhs.mean_wall_tangential_velocity ==
               rhs.mean_wall_tangential_velocity &&
           lhs.mean_contact_position == rhs.mean_contact_position &&
           lhs.mean_wall_normal == rhs.mean_wall_normal &&
           lhs.mean_footprint_direction ==
               rhs.mean_footprint_direction &&
           lhs.mean_contact_line_tangent ==
               rhs.mean_contact_line_tangent &&
           lhs.motion == rhs.motion;
}

bool validFreeSurfaceFirstOrderGeneralizedAlphaProvenance(
    const FreeSurfaceFirstOrderGeneralizedAlphaProvenance& provenance) noexcept
{
    if (!std::isfinite(provenance.alpha_m) ||
        !std::isfinite(provenance.alpha_f) ||
        !std::isfinite(provenance.gamma) ||
        !std::isfinite(provenance.dt) ||
        provenance.alpha_f < Real{0.5} || provenance.alpha_f > Real{1.0} ||
        provenance.alpha_m < Real{0.5} || provenance.alpha_m > Real{1.5} ||
        !(provenance.gamma > Real{0.0}) || !(provenance.dt > Real{0.0})) {
        return false;
    }
    // Retain exactly the first-order rho-infinity family produced by TimeLoop.
    // alpha_m may exceed one and reaches 1.5 at rho_inf == 0.
    const Real expected_alpha_m =
        Real{2.0} * provenance.alpha_f - Real{0.5};
    const Real expected_gamma = provenance.alpha_f;
    const Real scale = std::max(
        {Real{1.0},
         std::abs(provenance.alpha_m),
         std::abs(provenance.alpha_f),
         std::abs(provenance.gamma),
         std::abs(expected_alpha_m),
         std::abs(expected_gamma)});
    const Real tolerance =
        Real{512.0} * std::numeric_limits<Real>::epsilon() * scale;
    return std::abs(provenance.alpha_m - expected_alpha_m) <= tolerance &&
           std::abs(provenance.gamma - expected_gamma) <= tolerance;
}

bool sameFreeSurfaceAcceptedContactStageState(
    const FreeSurfaceAcceptedContactStageState& lhs,
    const FreeSurfaceAcceptedContactStageState& rhs) noexcept
{
    if (lhs.stage_time != rhs.stage_time ||
        lhs.stage_alpha_f != rhs.stage_alpha_f ||
        lhs.first_order_generalized_alpha !=
            rhs.first_order_generalized_alpha ||
        lhs.previous_state_revision != rhs.previous_state_revision ||
        lhs.endpoint_state_revision != rhs.endpoint_state_revision ||
        lhs.stage_state_revision != rhs.stage_state_revision ||
        !sameFreeSurfaceGeometryRevision(lhs.geometry_revision,
                                         rhs.geometry_revision) ||
        lhs.state.snapshot_revision_key !=
            rhs.state.snapshot_revision_key ||
        lhs.state.liquid_side != rhs.state.liquid_side ||
        lhs.state.surface_tension != rhs.state.surface_tension ||
        lhs.state.owned_contact_measure !=
            rhs.state.owned_contact_measure ||
        lhs.state.owned_wetted_wall_measure !=
            rhs.state.owned_wetted_wall_measure ||
        lhs.state.line_friction_dissipation !=
            rhs.state.line_friction_dissipation ||
        lhs.state.wall_slip_dissipation !=
            rhs.state.wall_slip_dissipation ||
        lhs.state.total_dissipation != rhs.state.total_dissipation ||
        lhs.state.walls.size() != rhs.state.walls.size()) {
        return false;
    }
    for (std::size_t i = 0; i < lhs.state.walls.size(); ++i) {
        if (!sameFreeSurfaceDynamicContactWallState(
                lhs.state.walls[i], rhs.state.walls[i])) {
            return false;
        }
    }
    return true;
}

bool sameFreeSurfaceAcceptedContactLineKinematics(
    const FreeSurfaceAcceptedContactLineKinematics& lhs,
    const FreeSurfaceAcceptedContactLineKinematics& rhs) noexcept
{
    return lhs.boundary_marker == rhs.boundary_marker &&
           lhs.previous_accepted_step == rhs.previous_accepted_step &&
           lhs.previous_accepted_time == rhs.previous_accepted_time &&
           lhs.previous_stage_time == rhs.previous_stage_time &&
           lhs.previous_stage_state_revision ==
               rhs.previous_stage_state_revision &&
           lhs.previous_snapshot_revision_key ==
               rhs.previous_snapshot_revision_key &&
           lhs.stage_time_interval == rhs.stage_time_interval &&
           lhs.previous_mean_contact_position ==
               rhs.previous_mean_contact_position &&
           lhs.projection_direction == rhs.projection_direction &&
           lhs.projected_contact_centroid_speed ==
               rhs.projected_contact_centroid_speed &&
           lhs.mean_fluid_contact_speed ==
               rhs.mean_fluid_contact_speed &&
           lhs.fluid_minus_geometric_contact_speed ==
               rhs.fluid_minus_geometric_contact_speed;
}

bool sameFreeSurfaceAcceptedContactLineKinematics(
    std::span<const FreeSurfaceAcceptedContactLineKinematics> lhs,
    std::span<const FreeSurfaceAcceptedContactLineKinematics> rhs) noexcept
{
    if (lhs.size() != rhs.size()) {
        return false;
    }
    for (std::size_t index = 0u; index < lhs.size(); ++index) {
        if (!sameFreeSurfaceAcceptedContactLineKinematics(
                lhs[index], rhs[index])) {
            return false;
        }
    }
    return true;
}

// Identities between a few scalar operations, and sums whose term count is
// not carried by the accepted state (rule counts, MPI rank counts), keep this
// fixed bound.  Sums of the same terms with known counts use the derived
// bound of FreeSurfaceFunctionalRounding.h instead.
bool freeSurfaceFunctionalValueNear(Real actual, Real expected) noexcept
{
    const auto scale = std::max(
        {Real{1.0}, std::abs(actual), std::abs(expected)});
    return std::abs(actual - expected) <=
           Real{512.0} * std::numeric_limits<Real>::epsilon() * scale;
}

} // namespace

std::vector<FreeSurfaceAcceptedContactLineKinematics>
deriveAcceptedContactLineKinematics(
    const FreeSurfaceDiscreteFunctionalHistoryRecord& previous,
    std::uint64_t accepted_step,
    const FreeSurfaceAcceptedContactStageState& current)
{
    std::vector<FreeSurfaceAcceptedContactLineKinematics> result;
    if (accepted_step <= previous.accepted_step ||
        !previous.contact_stage.has_value()) {
        return result;
    }
    const auto& prior = *previous.contact_stage;
    const Real stage_time_interval = current.stage_time - prior.stage_time;
    if (!std::isfinite(previous.accepted_time) ||
        !std::isfinite(prior.stage_time) ||
        !std::isfinite(current.stage_time) ||
        !std::isfinite(stage_time_interval) ||
        !(stage_time_interval > Real{0.0}) ||
        prior.stage_state_revision == 0u ||
        prior.geometry_revision.snapshot_revision_key == 0u) {
        return result;
    }

    const auto finite_array = [](const auto& values) {
        return std::all_of(values.begin(), values.end(), [](Real value) {
            return std::isfinite(value);
        });
    };
    const auto vector_norm = [](const auto& values) {
        Real norm_squared = Real{0.0};
        for (const auto value : values) {
            norm_squared += value * value;
        }
        return std::sqrt(norm_squared);
    };
    const Real direction_tolerance =
        Real{64.0} * std::numeric_limits<Real>::epsilon();
    result.reserve(current.state.walls.size());
    for (const auto& wall : current.state.walls) {
        const auto previous_wall = std::find_if(
            prior.state.walls.begin(),
            prior.state.walls.end(),
            [&](const auto& candidate) {
                return candidate.boundary_marker == wall.boundary_marker;
            });
        if (previous_wall == prior.state.walls.end() ||
            !(wall.owned_contact_measure > Real{0.0}) ||
            !(previous_wall->owned_contact_measure > Real{0.0}) ||
            !wall.mean_contact_speed.has_value() ||
            !previous_wall->mean_contact_speed.has_value() ||
            !finite_array(wall.mean_contact_position) ||
            !finite_array(previous_wall->mean_contact_position) ||
            !finite_array(wall.mean_footprint_direction) ||
            !finite_array(previous_wall->mean_footprint_direction)) {
            continue;
        }
        const Real current_direction_norm =
            vector_norm(wall.mean_footprint_direction);
        const Real previous_direction_norm =
            vector_norm(previous_wall->mean_footprint_direction);
        if (!std::isfinite(current_direction_norm) ||
            !std::isfinite(previous_direction_norm) ||
            !(current_direction_norm > direction_tolerance) ||
            !(previous_direction_norm > direction_tolerance)) {
            continue;
        }
        Real direction_alignment = Real{0.0};
        for (std::size_t component = 0u; component < 3u; ++component) {
            direction_alignment +=
                wall.mean_footprint_direction[component] /
                    current_direction_norm *
                previous_wall->mean_footprint_direction[component] /
                    previous_direction_norm;
        }
        if (!std::isfinite(direction_alignment) ||
            !(direction_alignment > Real{0.0})) {
            continue;
        }
        std::array<Real, 3> projection_direction{};
        for (std::size_t component = 0u; component < 3u; ++component) {
            projection_direction[component] =
                wall.mean_footprint_direction[component] /
                    current_direction_norm +
                previous_wall->mean_footprint_direction[component] /
                    previous_direction_norm;
        }
        const Real projection_norm = vector_norm(projection_direction);
        if (!std::isfinite(projection_norm) ||
            !(projection_norm > direction_tolerance)) {
            continue;
        }
        Real projected_displacement = Real{0.0};
        for (std::size_t component = 0u; component < 3u; ++component) {
            projection_direction[component] /= projection_norm;
            projected_displacement +=
                (wall.mean_contact_position[component] -
                 previous_wall->mean_contact_position[component]) *
                projection_direction[component];
        }
        const Real geometric_speed =
            projected_displacement / stage_time_interval;
        const Real fluid_speed = *wall.mean_contact_speed;
        const Real mismatch = fluid_speed - geometric_speed;
        if (!std::isfinite(geometric_speed) ||
            !std::isfinite(fluid_speed) || !std::isfinite(mismatch)) {
            continue;
        }
        result.push_back(
            FreeSurfaceAcceptedContactLineKinematics{
                .boundary_marker = wall.boundary_marker,
                .previous_accepted_step = previous.accepted_step,
                .previous_accepted_time = previous.accepted_time,
                .previous_stage_time = prior.stage_time,
                .previous_stage_state_revision =
                    prior.stage_state_revision,
                .previous_snapshot_revision_key =
                    prior.geometry_revision.snapshot_revision_key,
                .stage_time_interval = stage_time_interval,
                .previous_mean_contact_position =
                    previous_wall->mean_contact_position,
                .projection_direction = projection_direction,
                .projected_contact_centroid_speed = geometric_speed,
                .mean_fluid_contact_speed = fluid_speed,
                .fluid_minus_geometric_contact_speed = mismatch,
            });
    }
    return result;
}

FunctionalRecords::FunctionalRecords(FESystem& system)
    : system_(&system),
      functionals_(system.attachment<FunctionalRecordStore>().functionals),
      functional_declarations_(
          system.attachment<FunctionalRecordStore>().functionals.declarations),
      work_channel_declarations_(
          system.attachment<FunctionalRecordStore>().work_channels),
      external_boundary_energy_declaration_(
          system.attachment<FunctionalRecordStore>().external_boundary_energy),
      functional_history_(
          system.attachment<FunctionalRecordStore>().functionals.history),
      field_registry_{&system},
      mesh_access_{&system}
{
}

ConstFunctionalRecords::ConstFunctionalRecords(
    const FESystem& system) noexcept
    : store_(system.findAttachment<FunctionalRecordStore>())
{
}

std::span<const FreeSurfaceResidualWorkDeclaration>
ConstFunctionalRecords::workChannelDeclarations() const noexcept
{
  if (store_ == nullptr) {
    return {};
  }
  return store_->work_channels;
}

const std::optional<FreeSurfaceExternalBoundaryEnergyDeclaration>&
ConstFunctionalRecords::externalBoundaryEnergyDeclaration()
    const noexcept
{
  static const std::optional<FreeSurfaceExternalBoundaryEnergyDeclaration>
      none{};
  if (store_ == nullptr) {
    return none;
  }
  return store_->external_boundary_energy;
}

std::span<const FreeSurfaceDiscreteFunctionalDeclaration>
ConstFunctionalRecords::functionalDeclarations()
    const noexcept
{
  if (store_ == nullptr) {
    return {};
  }
  return store_->functionals.declarations;
}

std::span<const FreeSurfaceDiscreteFunctionalHistoryRecord>
ConstFunctionalRecords::acceptedFunctionalHistory() const noexcept
{
  if (store_ == nullptr) {
    return {};
  }
  return store_->functionals.history;
}

FunctionalRecords functionalRecords(FESystem& system)
{
  return FunctionalRecords(system);
}

ConstFunctionalRecords functionalRecords(const FESystem& system) noexcept
{
  return ConstFunctionalRecords(system);
}

void FunctionalRecords::declareWorkChannel(
    FreeSurfaceResidualWorkDeclaration declaration)
{
    FE_THROW_IF(
        isSetup(),
        InvalidArgumentException,
        "FESystem::declareFreeSurfaceResidualWork: declarations cannot change after setup");
    FE_THROW_IF(
        std::string_view(
            freeSurfaceResidualWorkChannelName(declaration.channel)) ==
            "unknown",
        InvalidArgumentException,
        "FESystem::declareFreeSurfaceResidualWork: channel is invalid");
    FE_THROW_IF(
        declaration.owner_component.find_first_not_of(" \t\r\n") ==
            std::string::npos,
        InvalidArgumentException,
        "FESystem::declareFreeSurfaceResidualWork: owner component must be nonempty");
    const bool produced =
        declaration.applicability ==
        FreeSurfaceResidualWorkApplicability::Produced;
    const bool inapplicable =
        declaration.applicability ==
        FreeSurfaceResidualWorkApplicability::NotApplicable;
    FE_THROW_IF(
        !produced && !inapplicable,
        InvalidArgumentException,
        "FESystem::declareFreeSurfaceResidualWork: applicability is invalid");
    FE_THROW_IF(
        produced && declaration.operator_tag.empty(),
        InvalidArgumentException,
        "FESystem::declareFreeSurfaceResidualWork: a produced channel requires an operator tag");
    FE_THROW_IF(
        produced && !hasOperator(declaration.operator_tag),
        InvalidArgumentException,
        "FESystem::declareFreeSurfaceResidualWork: produced operator '" +
            declaration.operator_tag + "' is not installed");
    FE_THROW_IF(
        inapplicable && !declaration.operator_tag.empty(),
        InvalidArgumentException,
        "FESystem::declareFreeSurfaceResidualWork: an inapplicable channel cannot name an operator");
    const auto duplicate = std::find_if(
        work_channel_declarations_.begin(),
        work_channel_declarations_.end(),
        [&](const auto& existing) {
            return existing.channel == declaration.channel;
        });
    FE_THROW_IF(
        duplicate != work_channel_declarations_.end(),
        InvalidArgumentException,
        "FESystem::declareFreeSurfaceResidualWork: channel '" +
            std::string(freeSurfaceResidualWorkChannelName(
                declaration.channel)) +
            "' already has an owner");
    work_channel_declarations_.push_back(
        std::move(declaration));
}

void FunctionalRecords::declareExternalBoundaryEnergy(
    FreeSurfaceExternalBoundaryEnergyDeclaration declaration)
{
    FE_THROW_IF(
        isSetup(),
        InvalidArgumentException,
        "FESystem::declareFreeSurfaceExternalBoundaryEnergy: declaration cannot change after setup");
    FE_THROW_IF(
        external_boundary_energy_declaration_.has_value(),
        InvalidArgumentException,
        "FESystem::declareFreeSurfaceExternalBoundaryEnergy: the system already has an external-boundary energy owner");
    FE_THROW_IF(
        declaration.owner_component.find_first_not_of(" \t\r\n") ==
            std::string::npos,
        InvalidArgumentException,
        "FESystem::declareFreeSurfaceExternalBoundaryEnergy: owner component must be nonempty");
    const auto valid_applicability = [](auto applicability) {
        return applicability ==
                   FreeSurfaceExternalBoundaryEnergyApplicability::
                       Unresolved ||
            applicability ==
                   FreeSurfaceExternalBoundaryEnergyApplicability::
                       NotApplicable;
    };
    FE_THROW_IF(
        !valid_applicability(declaration.imposed_traction) ||
            !valid_applicability(declaration.open_boundary_flux),
        InvalidArgumentException,
        "FESystem::declareFreeSurfaceExternalBoundaryEnergy: channel applicability is invalid");
    external_boundary_energy_declaration_ =
        std::move(declaration);
}

void FunctionalRecords::declareFunctional(
    FreeSurfaceDiscreteFunctionalDeclaration declaration)
{
    FE_THROW_IF(
        !functional_history_.empty(),
        InvalidArgumentException,
        "FESystem::declareFreeSurfaceDiscreteFunctional: declarations "
        "cannot change after accepted history has begun");
    FE_THROW_IF(
        declaration.interface_marker < 0,
        InvalidArgumentException,
        "FESystem::declareFreeSurfaceDiscreteFunctional: interface marker "
        "must be nonnegative");
    FE_THROW_IF(
        !field_registry_.has(declaration.level_set_field),
        InvalidArgumentException,
        "FESystem::declareFreeSurfaceDiscreteFunctional: unknown level-set "
        "field");
    const auto& level_set = field_registry_.get(declaration.level_set_field);
    FE_THROW_IF(
        level_set.scope != FieldScope::VolumeCell || level_set.space == nullptr ||
            level_set.components != 1 ||
            level_set.space->value_dimension() != 1,
        InvalidArgumentException,
        "FESystem::declareFreeSurfaceDiscreteFunctional: level-set field "
        "must be a scalar volume field");
    FE_THROW_IF(
        declaration.geometry_domain_id.empty(),
        InvalidArgumentException,
        "FESystem::declareFreeSurfaceDiscreteFunctional: geometry domain "
        "identifier must be nonempty");
    FE_THROW_IF(
        declaration.owner_component.empty(),
        InvalidArgumentException,
        "FESystem::declareFreeSurfaceDiscreteFunctional: owner component "
        "must be nonempty");
    const bool capillary_method_stationarity =
        declaration.capillary_balance_method ==
        FreeSurfaceCapillaryBalanceMethod::
            DiscreteEnergyVolumeStationarity;
    const bool capillary_method_kinematic_traction =
        declaration.capillary_balance_method ==
        FreeSurfaceCapillaryBalanceMethod::
            KinematicAreaGradientEnergyTraction;
    const bool capillary_method_selected =
        capillary_method_stationarity ||
        capillary_method_kinematic_traction;
    const bool capillary_method_unselected =
        declaration.capillary_balance_method ==
        FreeSurfaceCapillaryBalanceMethod::Unselected;
    FE_THROW_IF(
        !capillary_method_selected && !capillary_method_unselected,
        InvalidArgumentException,
        "FESystem::declareFreeSurfaceDiscreteFunctional: unknown capillary "
        "balance method");
    const bool capillary_qualification_unselected =
        declaration.capillary_balance_qualification ==
        FreeSurfaceCapillaryBalanceQualification::Unselected;
    const bool capillary_qualification_prerequisite =
        declaration.capillary_balance_qualification ==
        FreeSurfaceCapillaryBalanceQualification::PrerequisiteOnly;
    const bool capillary_qualification_complete =
        declaration.capillary_balance_qualification ==
        FreeSurfaceCapillaryBalanceQualification::Qualified;
    FE_THROW_IF(
        !capillary_qualification_unselected &&
            !capillary_qualification_prerequisite &&
            !capillary_qualification_complete,
        InvalidArgumentException,
        "FESystem::declareFreeSurfaceDiscreteFunctional: unknown capillary "
        "balance qualification");
    FE_THROW_IF(
        capillary_method_unselected !=
            capillary_qualification_unselected,
        InvalidArgumentException,
        "FESystem::declareFreeSurfaceDiscreteFunctional: capillary balance "
        "method and qualification must be selected together");
    FE_THROW_IF(
        capillary_qualification_complete,
        InvalidArgumentException,
        "FESystem::declareFreeSurfaceDiscreteFunctional: no capillary "
        "balance method is qualified in the current implementation");
    FE_THROW_IF(
        declaration.endpoint_functional_power_enabled &&
            !capillary_method_selected,
        InvalidArgumentException,
        "FESystem::declareFreeSurfaceDiscreteFunctional: endpoint "
        "functional power requires the discrete-energy capillary method");
    FE_THROW_IF(
        capillary_method_kinematic_traction &&
            !declaration.endpoint_functional_power_enabled,
        InvalidArgumentException,
        "FESystem::declareFreeSurfaceDiscreteFunctional: the "
        "kinematic-area-gradient traction method requires endpoint "
        "functional-power accounting");
    FE_THROW_IF(
        capillary_method_kinematic_traction &&
            !field_registry_.has(declaration.curvature_field),
        InvalidArgumentException,
        "FESystem::declareFreeSurfaceDiscreteFunctional: the kinematic-area-gradient traction method requires a declared curvature field");
    FE_THROW_IF(
        capillary_method_kinematic_traction &&
            declaration.curvature_field == declaration.level_set_field,
        InvalidArgumentException,
        "FESystem::declareFreeSurfaceDiscreteFunctional: the kinematic-area-gradient traction level-set and curvature fields must be distinct");
    FE_THROW_IF(
        !capillary_method_kinematic_traction &&
            declaration.curvature_field != INVALID_FIELD_ID,
        InvalidArgumentException,
        "FESystem::declareFreeSurfaceDiscreteFunctional: a curvature field is valid only for the kinematic-area-gradient traction method");
    if (capillary_method_kinematic_traction) {
        const auto& curvature =
            field_registry_.get(declaration.curvature_field);
        FE_THROW_IF(
            curvature.scope != FieldScope::VolumeCell ||
                curvature.space == nullptr || curvature.components != 1 ||
                curvature.space->value_dimension() != 1 ||
                curvature.source_kind != FieldSourceKind::PrescribedData,
            InvalidArgumentException,
            "FESystem::declareFreeSurfaceDiscreteFunctional: the kinematic-area-gradient traction curvature must be a prescribed scalar volume field");
        const int dimension = mesh_access_ ? mesh_access_->dimension() : 0;
        const bool supported_simplex =
            (dimension == 2 && level_set.space->element_type() ==
                                   ElementType::Triangle3) ||
            (dimension == 3 && level_set.space->element_type() ==
                                   ElementType::Tetra4);
        FE_THROW_IF(
            !supported_simplex ||
                level_set.space->polynomial_order() != 1 ||
                level_set.space->continuity() != Continuity::C0 ||
                curvature.space->space_type() !=
                    level_set.space->space_type() ||
                curvature.space->field_type() != FieldType::Scalar ||
                curvature.space->topological_dimension() !=
                    level_set.space->topological_dimension() ||
                curvature.space->polynomial_order() != 1 ||
                curvature.space->continuity() != Continuity::C0 ||
                curvature.space->element_type() !=
                    level_set.space->element_type() ||
                curvature.space->dofs_per_element() !=
                    level_set.space->dofs_per_element(),
            InvalidArgumentException,
            "FESystem::declareFreeSurfaceDiscreteFunctional: the kinematic-area-gradient traction requires matching scalar affine C0 P1 Triangle3 or Tetra4 level-set and curvature spaces");
        FE_THROW_IF(
            !field_registry_.has(declaration.velocity_field),
            InvalidArgumentException,
            "FESystem::declareFreeSurfaceDiscreteFunctional: the kinematic-area-gradient traction requires a registered velocity field");
        const auto& velocity =
            field_registry_.get(declaration.velocity_field);
        FE_THROW_IF(
            velocity.scope != FieldScope::VolumeCell ||
                velocity.source_kind != FieldSourceKind::Unknown ||
                velocity.space == nullptr ||
                velocity.space->space_type() != spaces::SpaceType::Product ||
                velocity.space->field_type() != FieldType::Vector ||
                velocity.space->value_dimension() != dimension ||
                velocity.components != dimension ||
                velocity.space->topological_dimension() !=
                    level_set.space->topological_dimension() ||
                velocity.space->polynomial_order() != 1 ||
                velocity.space->continuity() != Continuity::C0 ||
                velocity.space->element_type() !=
                    level_set.space->element_type() ||
                velocity.space->dofs_per_element() !=
                    level_set.space->dofs_per_element() *
                        static_cast<std::size_t>(dimension),
            InvalidArgumentException,
            "FESystem::declareFreeSurfaceDiscreteFunctional: the kinematic-area-gradient traction requires a velocity field built from the same affine C0 P1 simplex scalar basis as the level-set and curvature fields");
    }
    auto& parameters = declaration.parameters;
    FE_THROW_IF(
        parameters.liquid_side != geometry::CutIntegrationSide::Negative &&
            parameters.liquid_side != geometry::CutIntegrationSide::Positive,
        InvalidArgumentException,
        "FESystem::declareFreeSurfaceDiscreteFunctional: liquid side must "
        "be Negative or Positive");
    FE_THROW_IF(
        !std::isfinite(parameters.surface_tension) ||
            parameters.surface_tension < Real{0.0},
        InvalidArgumentException,
        "FESystem::declareFreeSurfaceDiscreteFunctional: surface tension "
        "must be finite and nonnegative");
    FE_THROW_IF(
        !std::isfinite(parameters.volume_multiplier),
        InvalidArgumentException,
        "FESystem::declareFreeSurfaceDiscreteFunctional: volume multiplier "
        "must be finite");
    if (declaration.active_volume_energy_parameters.has_value()) {
        const auto& energy =
            *declaration.active_volume_energy_parameters;
        const auto finite_vector = [](const auto& values) {
            return std::all_of(
                values.begin(), values.end(), [](Real value) {
                    return std::isfinite(value);
                });
        };
        FE_THROW_IF(
            energy.liquid_side != parameters.liquid_side ||
                !std::isfinite(energy.density) ||
                !(energy.density > Real{0.0}) ||
                !finite_vector(energy.gravitational_acceleration) ||
                !finite_vector(energy.gravitational_reference_point),
            InvalidArgumentException,
            "FESystem::declareFreeSurfaceDiscreteFunctional: active-volume "
            "energy requires the declared liquid side, positive finite "
            "density, and finite gravitational data");
    }
    FE_THROW_IF(
        declaration.static_conservative_body_force_complete &&
            !declaration.active_volume_energy_parameters.has_value(),
        InvalidArgumentException,
        "FESystem::declareFreeSurfaceDiscreteFunctional: a complete static "
        "conservative body-force declaration requires active-volume energy "
        "parameters");
    if (declaration.active_volume_dissipation_parameters.has_value()) {
        const auto& dissipation =
            *declaration.active_volume_dissipation_parameters;
        FE_THROW_IF(
            dissipation.liquid_side != parameters.liquid_side ||
                !std::isfinite(dissipation.dynamic_viscosity) ||
                !(dissipation.dynamic_viscosity > Real{0.0}),
            InvalidArgumentException,
            "FESystem::declareFreeSurfaceDiscreteFunctional: active-volume "
            "dissipation requires the declared liquid side and positive "
            "finite dynamic viscosity");
    }
    if (declaration.external_pressure_power_parameters.has_value()) {
        const auto& pressure =
            *declaration.external_pressure_power_parameters;
        FE_THROW_IF(
            pressure.liquid_side != parameters.liquid_side ||
                !std::isfinite(pressure.external_pressure),
            InvalidArgumentException,
            "FESystem::declareFreeSurfaceDiscreteFunctional: exterior-"
            "pressure power requires the declared liquid side and finite "
            "pressure");
    }
    std::sort(
        parameters.young_wall_coefficients.begin(),
        parameters.young_wall_coefficients.end(),
        [](const auto& lhs, const auto& rhs) {
            return lhs.boundary_marker < rhs.boundary_marker;
        });
    constexpr Real pi = Real{3.141592653589793238462643383279502884};
    for (std::size_t i = 0; i < parameters.young_wall_coefficients.size();
         ++i) {
        const auto& coefficient = parameters.young_wall_coefficients[i];
        FE_THROW_IF(
            coefficient.boundary_marker < 0 ||
                !std::isfinite(
                    coefficient.equilibrium_contact_angle_radians) ||
                !(coefficient.equilibrium_contact_angle_radians >
                  Real{0.0}) ||
                !(coefficient.equilibrium_contact_angle_radians < pi),
            InvalidArgumentException,
            "FESystem::declareFreeSurfaceDiscreteFunctional: every Young "
            "wall coefficient requires a nonnegative marker and a finite "
            "angle strictly in (0, pi)");
        FE_THROW_IF(
            i != 0u &&
                parameters.young_wall_coefficients[i - 1u]
                        .boundary_marker == coefficient.boundary_marker,
            InvalidArgumentException,
            "FESystem::declareFreeSurfaceDiscreteFunctional: duplicate "
            "Young wall coefficient marker");
    }
    std::sort(
        parameters.dynamic_contact_coefficients.begin(),
        parameters.dynamic_contact_coefficients.end(),
        [](const auto& lhs, const auto& rhs) {
            return lhs.boundary_marker < rhs.boundary_marker;
        });
    for (std::size_t i = 0;
         i < parameters.dynamic_contact_coefficients.size(); ++i) {
        const auto& coefficient =
            parameters.dynamic_contact_coefficients[i];
        const bool prescribed_angle =
            coefficient.law ==
            interfaces::FreeSurfaceContactLaw::PrescribedAngle;
        const bool dynamic_ren_e =
            coefficient.law ==
            interfaces::FreeSurfaceContactLaw::DynamicRenE;
        FE_THROW_IF(
            coefficient.boundary_marker < 0 ||
                !std::isfinite(
                    coefficient.equilibrium_contact_angle_radians) ||
                !(coefficient.equilibrium_contact_angle_radians >
                  Real{0.0}) ||
                !(coefficient.equilibrium_contact_angle_radians < pi) ||
                !std::isfinite(coefficient.mobility) ||
                (!prescribed_angle && !dynamic_ren_e) ||
                (dynamic_ren_e &&
                 !(coefficient.mobility > Real{0.0})) ||
                (prescribed_angle &&
                 coefficient.mobility != Real{0.0}) ||
                !std::isfinite(coefficient.slip_length) ||
                !(coefficient.slip_length > Real{0.0}) ||
                !std::isfinite(coefficient.dynamic_viscosity) ||
                !(coefficient.dynamic_viscosity > Real{0.0}),
            InvalidArgumentException,
            "FESystem::declareFreeSurfaceDiscreteFunctional: every contact "
            "coefficient requires a supported law, a nonnegative marker, a "
            "finite angle strictly in (0, pi), law-consistent mobility, "
            "finite positive slip length, and finite positive dynamic "
            "viscosity");
        FE_THROW_IF(
            i != 0u &&
                parameters.dynamic_contact_coefficients[i - 1u]
                        .boundary_marker == coefficient.boundary_marker,
            InvalidArgumentException,
            "FESystem::declareFreeSurfaceDiscreteFunctional: duplicate "
            "contact-stage coefficient marker");
        const auto young = std::lower_bound(
            parameters.young_wall_coefficients.begin(),
            parameters.young_wall_coefficients.end(),
            coefficient.boundary_marker,
            [](const auto& candidate, int marker) {
                return candidate.boundary_marker < marker;
            });
        FE_THROW_IF(
            young == parameters.young_wall_coefficients.end() ||
                young->boundary_marker != coefficient.boundary_marker ||
                young->equilibrium_contact_angle_radians !=
                    coefficient.equilibrium_contact_angle_radians,
            InvalidArgumentException,
            "FESystem::declareFreeSurfaceDiscreteFunctional: contact-stage "
            "coefficients require matching Young wall coefficients");
    }
    if (declaration.active_volume_energy_parameters.has_value() ||
        declaration.active_volume_dissipation_parameters.has_value() ||
        declaration.external_pressure_power_parameters.has_value() ||
        declaration.endpoint_functional_power_enabled ||
        !parameters.dynamic_contact_coefficients.empty()) {
        FE_THROW_IF(
            !field_registry_.has(declaration.velocity_field),
            InvalidArgumentException,
            "FESystem::declareFreeSurfaceDiscreteFunctional: active-volume "
            "energy/dissipation, exterior-pressure/endpoint functional "
            "power, and dynamic-contact telemetry require a registered "
            "velocity field");
        const auto& velocity =
            field_registry_.get(declaration.velocity_field);
        const int dimension = mesh_access_ ? mesh_access_->dimension() : 0;
        FE_THROW_IF(
            dimension < 1 || dimension > 3 ||
                velocity.scope != FieldScope::VolumeCell ||
                velocity.space == nullptr ||
                velocity.components < dimension ||
                velocity.space->value_dimension() < dimension ||
                !fieldParticipatesInUnknownVector(
                    declaration.velocity_field),
            InvalidArgumentException,
            "FESystem::declareFreeSurfaceDiscreteFunctional: active-volume "
            "energy/dissipation, exterior-pressure/endpoint functional "
            "power, and dynamic-contact telemetry require an unknown vector "
            "volume field");
    }
    const auto conflict = std::find_if(
        functional_declarations_.begin(),
        functional_declarations_.end(),
        [&](const auto& existing) {
            return existing.interface_marker == declaration.interface_marker;
        });
    FE_THROW_IF(
        conflict != functional_declarations_.end(),
        InvalidArgumentException,
        "FESystem::declareFreeSurfaceDiscreteFunctional: interface marker " +
            std::to_string(declaration.interface_marker) +
            " already has functional owner '" + conflict->owner_component +
            "'");
    const bool owns_active_volume =
        declaration.active_volume_energy_parameters.has_value() ||
        declaration.active_volume_dissipation_parameters.has_value();
    const auto active_volume_conflict = std::find_if(
        functional_declarations_.begin(),
        functional_declarations_.end(),
        [](const auto& existing) {
            return existing.active_volume_energy_parameters.has_value() ||
                   existing.active_volume_dissipation_parameters.has_value();
        });
    FE_THROW_IF(
        owns_active_volume &&
            active_volume_conflict !=
                functional_declarations_.end(),
        InvalidArgumentException,
        "FESystem::declareFreeSurfaceDiscreteFunctional: active-volume "
        "energy/dissipation owner is already assigned to interface marker " +
            std::to_string(active_volume_conflict->interface_marker) +
            " by '" + active_volume_conflict->owner_component +
            "'; the current one-phase accounting contract permits exactly "
            "one bulk owner");

    const int declared_marker = declaration.interface_marker;
    invalidateSetup();
    functionals_.declare(std::move(declaration));
    std::sort(
        functional_declarations_.begin(),
        functional_declarations_.end(),
        [](const auto& lhs, const auto& rhs) {
            return lhs.interface_marker < rhs.interface_marker;
        });
    const auto& stored = *std::find_if(
        functional_declarations_.begin(),
        functional_declarations_.end(),
        [&](const auto& candidate) {
            return candidate.interface_marker ==
                   declared_marker;
        });
    std::ostringstream message;
    message << std::setprecision(17)
            << "FESystem: declared free-surface discrete functional"
            << " diagnostic=free_surface_discrete_functional_declaration"
            << " interface_marker=" << stored.interface_marker
            << " level_set_field=" << stored.level_set_field
            << " curvature_field=" << stored.curvature_field
            << " velocity_field=" << stored.velocity_field
            << " geometry_domain_id='" << stored.geometry_domain_id << "'"
            << " liquid_side="
            << (stored.parameters.liquid_side ==
                        geometry::CutIntegrationSide::Negative
                    ? "Negative"
                    : "Positive")
            << " surface_tension=" << stored.parameters.surface_tension
            << " volume_multiplier=" << stored.parameters.volume_multiplier
            << " young_wall_count="
            << stored.parameters.young_wall_coefficients.size()
            << " contact_stage_count="
            << stored.parameters.dynamic_contact_coefficients.size()
            << " active_volume_energy="
            << (stored.active_volume_energy_parameters.has_value()
                    ? "enabled"
                    : "disabled")
            << " static_conservative_body_force_complete="
            << (stored.static_conservative_body_force_complete ? 1 : 0)
            << " active_volume_dissipation="
            << (stored.active_volume_dissipation_parameters.has_value()
                    ? "enabled"
                    : "disabled")
            << " external_pressure_power="
            << (stored.external_pressure_power_parameters.has_value()
                    ? "enabled"
                    : "disabled")
            << " endpoint_functional_power="
            << (stored.endpoint_functional_power_enabled
                    ? "enabled"
                    : "disabled")
            << " capillary_balance_method="
            << (stored.capillary_balance_method ==
                        FreeSurfaceCapillaryBalanceMethod::
                            DiscreteEnergyVolumeStationarity
                    ? "discrete_energy_volume_stationarity"
                    : stored.capillary_balance_method ==
                              FreeSurfaceCapillaryBalanceMethod::
                                  KinematicAreaGradientEnergyTraction
                          ? "kinematic_area_gradient_energy_traction"
                          : "unselected")
            << " capillary_balance_qualification="
            << (stored.capillary_balance_qualification ==
                        FreeSurfaceCapillaryBalanceQualification::Qualified
                    ? "qualified"
                    : stored.capillary_balance_qualification ==
                              FreeSurfaceCapillaryBalanceQualification::
                                  PrerequisiteOnly
                          ? "prerequisite_only"
                          : "unselected")
            << " owner='" << stored.owner_component << "'";
    if (stored.active_volume_energy_parameters.has_value()) {
        const auto& energy =
            *stored.active_volume_energy_parameters;
        message
            << " active_volume_density=" << energy.density
            << " gravitational_acceleration=("
            << energy.gravitational_acceleration[0] << ","
            << energy.gravitational_acceleration[1] << ","
            << energy.gravitational_acceleration[2] << ")"
            << " gravitational_reference_point=("
            << energy.gravitational_reference_point[0] << ","
            << energy.gravitational_reference_point[1] << ","
            << energy.gravitational_reference_point[2] << ")";
    }
    for (const auto& wall : stored.parameters.young_wall_coefficients) {
        message << " young_wall_marker=" << wall.boundary_marker
                << " equilibrium_contact_angle_radians="
                << wall.equilibrium_contact_angle_radians;
    }
    for (const auto& contact :
         stored.parameters.dynamic_contact_coefficients) {
        message << " contact_stage_marker=" << contact.boundary_marker
                << " contact_stage_law="
                << (contact.law ==
                            interfaces::FreeSurfaceContactLaw::DynamicRenE
                        ? "DynamicRenE"
                        : "PrescribedAngle")
                << " contact_stage_equilibrium_angle_radians="
                << contact.equilibrium_contact_angle_radians
                << " contact_stage_mobility=" << contact.mobility
                << " contact_stage_slip_length=" << contact.slip_length
                << " contact_stage_viscosity="
                << contact.dynamic_viscosity;
    }
    FE_LOG_INFO(message.str());
}

void FunctionalRecords::recordAcceptedFunctionals(
    std::uint64_t accepted_step,
    Real accepted_time,
    Real dt,
    std::uint64_t pre_maintenance_endpoint_state_revision,
    std::uint64_t state_revision,
    std::span<const AcceptedFreeSurfaceDiscreteFunctionalState> states,
    std::optional<std::uint64_t> extension_map_revision)
{
    if (functional_declarations_.empty()) {
        FE_THROW_IF(
            !states.empty(),
            InvalidArgumentException,
            "FESystem::recordAcceptedFreeSurfaceDiscreteFunctionals: "
            "states were supplied without declarations");
        return;
    }
    FE_THROW_IF(
        !std::isfinite(accepted_time) || !std::isfinite(dt) ||
            dt < Real{0.0} ||
            pre_maintenance_endpoint_state_revision == 0u ||
            state_revision == 0u ||
            (extension_map_revision.has_value() &&
             *extension_map_revision == 0u),
        InvalidArgumentException,
        "FESystem::recordAcceptedFreeSurfaceDiscreteFunctionals: accepted "
        "time and dt must be finite, dt must be nonnegative, and the "
        "pre-maintenance endpoint, accepted-state, and any present "
        "extension-map revisions must be nonzero");
    FE_THROW_IF(
        !isSetup(),
        InvalidArgumentException,
        "FESystem::recordAcceptedFreeSurfaceDiscreteFunctionals: system "
        "setup must be complete");
    FE_THROW_IF(
        states.size() !=
            functional_declarations_.size(),
        InvalidArgumentException,
        "FESystem::recordAcceptedFreeSurfaceDiscreteFunctionals: expected "
        "exactly one state for every declaration");

    const auto require_near = [](Real actual,
                                 Real expected,
                                 std::string_view quantity) {
        FE_THROW_IF(
            !freeSurfaceFunctionalValueNear(actual, expected),
            InvalidArgumentException,
            "FESystem::recordAcceptedFreeSurfaceDiscreteFunctionals: " +
                std::string(quantity) + " is inconsistent");
    };
    // Two sums of the same nonnegative terms, each with its term count; their
    // difference is bounded by floating-point summation error analysis.
    const auto require_same_terms_near = [](Real actual,
                                            std::uint64_t actual_terms,
                                            Real expected,
                                            std::uint64_t expected_terms,
                                            std::string_view quantity) {
        FE_THROW_IF(
            !freeSurfaceNonnegativeSummationsAgree(
                actual, actual_terms, expected, expected_terms),
            InvalidArgumentException,
            "FESystem::recordAcceptedFreeSurfaceDiscreteFunctionals: " +
                std::string(quantity) + " is inconsistent");
    };
    constexpr Real contact_pi =
        Real{3.141592653589793238462643383279502884};
    for (std::size_t i = 0; i < states.size(); ++i) {
        const auto& accepted = states[i];
        const auto& declaration =
            functional_declarations_[i];
        FE_THROW_IF(
            accepted.interface_marker != declaration.interface_marker,
            InvalidArgumentException,
            "FESystem::recordAcceptedFreeSurfaceDiscreteFunctionals: "
            "states must be unique and sorted by declared interface marker");
        const auto& revision = accepted.geometry_revision;
        const auto& state = accepted.state;
        FE_THROW_IF(
            !revision.complete(),
            InvalidArgumentException,
            "FESystem::recordAcceptedFreeSurfaceDiscreteFunctionals: "
            "geometry revision is incomplete");
        const auto expected_source =
            "field:" + std::to_string(declaration.level_set_field);
        FE_THROW_IF(
            revision.interface_marker != declaration.interface_marker ||
                revision.source_id != expected_source ||
                revision.domain_id != declaration.geometry_domain_id ||
                accepted.cut_topology_revision == 0u ||
                revision.snapshot_revision_key !=
                    state.snapshot_revision_key,
            InvalidArgumentException,
            "FESystem::recordAcceptedFreeSurfaceDiscreteFunctionals: "
            "geometry provenance does not match the declaration and state");
        FE_THROW_IF(
            state.liquid_side != declaration.parameters.liquid_side ||
                state.surface_tension !=
                    declaration.parameters.surface_tension ||
                state.volume_multiplier !=
                    declaration.parameters.volume_multiplier,
            InvalidArgumentException,
            "FESystem::recordAcceptedFreeSurfaceDiscreteFunctionals: state "
            "coefficients do not match the declaration");
        const auto finite_nonnegative = [](Real value) {
            return std::isfinite(value) && value >= Real{0.0};
        };
        FE_THROW_IF(
            !finite_nonnegative(state.owned_liquid_volume) ||
                !finite_nonnegative(state.owned_liquid_gas_area) ||
                !finite_nonnegative(state.owned_wetted_wall_area) ||
                !finite_nonnegative(state.owned_contact_measure) ||
                !std::isfinite(state.liquid_gas_surface_energy) ||
                !std::isfinite(state.young_wall_energy) ||
                !std::isfinite(state.volume_constraint_potential) ||
                !std::isfinite(state.total_potential),
            InvalidArgumentException,
            "FESystem::recordAcceptedFreeSurfaceDiscreteFunctionals: state "
            "contains a non-finite value or a negative measure");

        Real wetted_area_sum = Real{0.0};
        Real contact_measure_sum = Real{0.0};
        Real wall_energy_sum = Real{0.0};
        std::size_t configured_wall_count = 0u;
        for (std::size_t wall_index = 0; wall_index < state.walls.size();
             ++wall_index) {
            const auto& wall = state.walls[wall_index];
            FE_THROW_IF(
                wall.boundary_marker < 0 ||
                    (wall_index != 0u &&
                     state.walls[wall_index - 1u].boundary_marker >=
                         wall.boundary_marker) ||
                    !finite_nonnegative(wall.owned_wetted_wall_area) ||
                    !finite_nonnegative(wall.owned_contact_measure) ||
                    !std::isfinite(wall.young_wall_energy),
                InvalidArgumentException,
                "FESystem::recordAcceptedFreeSurfaceDiscreteFunctionals: "
                "wall states must be finite, nonnegative in measure, unique, "
                "and sorted by marker");
            const auto coefficient = std::lower_bound(
                declaration.parameters.young_wall_coefficients.begin(),
                declaration.parameters.young_wall_coefficients.end(),
                wall.boundary_marker,
                [](const auto& candidate, int marker) {
                    return candidate.boundary_marker < marker;
                });
            const bool configured =
                coefficient !=
                    declaration.parameters.young_wall_coefficients.end() &&
                coefficient->boundary_marker == wall.boundary_marker;
            FE_THROW_IF(
                configured !=
                    wall.equilibrium_contact_angle_radians.has_value(),
                InvalidArgumentException,
                "FESystem::recordAcceptedFreeSurfaceDiscreteFunctionals: "
                "wall angle presence does not match the declaration");
            Real expected_wall_energy = Real{0.0};
            if (configured) {
                ++configured_wall_count;
                FE_THROW_IF(
                    *wall.equilibrium_contact_angle_radians !=
                        coefficient->equilibrium_contact_angle_radians,
                    InvalidArgumentException,
                    "FESystem::recordAcceptedFreeSurfaceDiscreteFunctionals: "
                    "wall angle does not match the declaration");
                expected_wall_energy =
                    -state.surface_tension *
                    std::cos(*wall.equilibrium_contact_angle_radians) *
                    wall.owned_wetted_wall_area;
            }
            require_near(wall.young_wall_energy,
                         expected_wall_energy,
                         "per-wall Young energy");
            wetted_area_sum += wall.owned_wetted_wall_area;
            contact_measure_sum += wall.owned_contact_measure;
            wall_energy_sum += wall.young_wall_energy;
        }
        FE_THROW_IF(
            configured_wall_count !=
                declaration.parameters.young_wall_coefficients.size(),
            InvalidArgumentException,
            "FESystem::recordAcceptedFreeSurfaceDiscreteFunctionals: a "
            "configured Young wall is absent from the state");
        require_near(state.owned_wetted_wall_area,
                     wetted_area_sum,
                     "total wetted-wall area");
        require_near(state.owned_contact_measure,
                     contact_measure_sum,
                     "total contact measure");
        require_near(state.liquid_gas_surface_energy,
                     state.surface_tension *
                         state.owned_liquid_gas_area,
                     "liquid-gas surface energy");
        require_near(state.young_wall_energy,
                     wall_energy_sum,
                     "total Young wall energy");
        require_near(state.volume_constraint_potential,
                     state.volume_multiplier *
                         state.owned_liquid_volume,
                     "volume-constraint potential");
        require_near(state.total_potential,
                     state.liquid_gas_surface_energy +
                         state.young_wall_energy +
                         state.volume_constraint_potential,
                     "total potential");

        const bool endpoint_functional_power_declared =
            declaration.endpoint_functional_power_enabled;
        FE_THROW_IF(
            accepted.endpoint_functional_power.has_value() !=
                endpoint_functional_power_declared,
            InvalidArgumentException,
            "FESystem::recordAcceptedFreeSurfaceDiscreteFunctionals: "
            "endpoint functional-power presence does not match the "
            "capillary balance method");
        if (accepted.endpoint_functional_power.has_value()) {
            const auto& power =
                *accepted.endpoint_functional_power;
            FE_THROW_IF(
                power.snapshot_revision_key !=
                        revision.snapshot_revision_key ||
                    power.liquid_side != state.liquid_side ||
                    power.surface_tension != state.surface_tension ||
                    power.volume_multiplier != state.volume_multiplier ||
                    power.walls.size() != state.walls.size() ||
                    !std::isfinite(
                        power.owned_liquid_volume_variation) ||
                    !std::isfinite(
                        power.owned_liquid_gas_area_variation) ||
                    !std::isfinite(
                        power.owned_wetted_wall_area_variation) ||
                    !std::isfinite(
                        power.liquid_gas_surface_energy_variation) ||
                    !std::isfinite(
                        power.young_wall_energy_variation) ||
                    !std::isfinite(
                        power.volume_constraint_potential_variation) ||
                    !std::isfinite(
                        power.total_potential_variation),
                InvalidArgumentException,
                "FESystem::recordAcceptedFreeSurfaceDiscreteFunctionals: "
                "endpoint functional power is inconsistent or non-finite");
            Real wetted_wall_variation_sum = Real{0.0};
            Real young_wall_power_sum = Real{0.0};
            for (std::size_t wall_index = 0u;
                 wall_index < power.walls.size(); ++wall_index) {
                const auto& wall_power = power.walls[wall_index];
                const auto& wall_state = state.walls[wall_index];
                FE_THROW_IF(
                    wall_power.boundary_marker !=
                            wall_state.boundary_marker ||
                        wall_power.equilibrium_contact_angle_radians !=
                            wall_state
                                .equilibrium_contact_angle_radians ||
                        !std::isfinite(
                            wall_power
                                .owned_wetted_wall_area_variation) ||
                        !std::isfinite(
                            wall_power.young_wall_energy_variation),
                    InvalidArgumentException,
                    "FESystem::recordAcceptedFreeSurfaceDiscreteFunctionals: "
                    "endpoint wall-functional power is inconsistent or "
                    "non-finite");
                wetted_wall_variation_sum +=
                    wall_power.owned_wetted_wall_area_variation;
                young_wall_power_sum +=
                    wall_power.young_wall_energy_variation;
            }
            require_near(
                power.owned_wetted_wall_area_variation,
                wetted_wall_variation_sum,
                "endpoint wetted-wall area power");
            require_near(
                power.liquid_gas_surface_energy_variation,
                power.surface_tension *
                    power.owned_liquid_gas_area_variation,
                "endpoint liquid-gas surface-energy power");
            require_near(
                power.young_wall_energy_variation,
                young_wall_power_sum,
                "endpoint Young wall-energy power");
            require_near(
                power.volume_constraint_potential_variation,
                power.volume_multiplier *
                    power.owned_liquid_volume_variation,
                "endpoint volume-constraint power");
            require_near(
                power.total_potential_variation,
                power.liquid_gas_surface_energy_variation +
                    power.young_wall_energy_variation +
                    power.volume_constraint_potential_variation,
                "endpoint total functional power");
        }

        const bool active_volume_energy_declared =
            declaration.active_volume_energy_parameters.has_value();
        FE_THROW_IF(
            accepted.active_volume_energy.has_value() !=
                active_volume_energy_declared,
            InvalidArgumentException,
            "FESystem::recordAcceptedFreeSurfaceDiscreteFunctionals: "
            "active-volume energy presence does not match the declaration");
        if (accepted.active_volume_energy.has_value()) {
            const auto& energy = *accepted.active_volume_energy;
            const auto& energy_parameters =
                *declaration.active_volume_energy_parameters;
            FE_THROW_IF(
                energy.snapshot_revision_key !=
                        revision.snapshot_revision_key ||
                    energy.liquid_side != energy_parameters.liquid_side ||
                    energy.density != energy_parameters.density ||
                    energy.gravitational_acceleration !=
                        energy_parameters.gravitational_acceleration ||
                    energy.gravitational_reference_point !=
                        energy_parameters.gravitational_reference_point ||
                    !finite_nonnegative(energy.owned_liquid_volume) ||
                    !finite_nonnegative(energy.kinetic_energy) ||
                    !std::isfinite(energy.gravitational_energy) ||
                    !std::isfinite(
                        energy.gravitational_potential_power) ||
                    !std::isfinite(energy.total_energy),
                InvalidArgumentException,
                "FESystem::recordAcceptedFreeSurfaceDiscreteFunctionals: "
                "active-volume energy does not match its declaration or "
                "geometry revision");
            // The state sums the per-rule measures and the energy sums
            // every weight of those rules: the same terms, counted by the
            // energy's quadrature points on both sides.
            require_same_terms_near(
                energy.owned_liquid_volume,
                energy.owned_quadrature_point_count,
                state.owned_liquid_volume,
                energy.owned_quadrature_point_count,
                "active-volume energy liquid measure");
            require_near(
                energy.total_energy,
                energy.kinetic_energy + energy.gravitational_energy,
                "active-volume total energy");
        }

        const bool active_volume_dissipation_declared =
            declaration.active_volume_dissipation_parameters.has_value();
        FE_THROW_IF(
            accepted.active_volume_dissipation.has_value() !=
                active_volume_dissipation_declared,
            InvalidArgumentException,
            "FESystem::recordAcceptedFreeSurfaceDiscreteFunctionals: "
            "active-volume dissipation presence does not match the "
            "declaration");
        if (accepted.active_volume_dissipation.has_value()) {
            const auto& dissipation =
                *accepted.active_volume_dissipation;
            const auto& dissipation_parameters =
                *declaration.active_volume_dissipation_parameters;
            FE_THROW_IF(
                dissipation.snapshot_revision_key !=
                        revision.snapshot_revision_key ||
                    dissipation.liquid_side !=
                        dissipation_parameters.liquid_side ||
                    dissipation.dynamic_viscosity !=
                        dissipation_parameters.dynamic_viscosity ||
                    !finite_nonnegative(
                        dissipation.owned_liquid_volume) ||
                    !finite_nonnegative(
                        dissipation.bulk_viscous_dissipation_rate),
                InvalidArgumentException,
                "FESystem::recordAcceptedFreeSurfaceDiscreteFunctionals: "
                "active-volume dissipation does not match its declaration "
                "or geometry revision");
            require_same_terms_near(
                dissipation.owned_liquid_volume,
                dissipation.owned_quadrature_point_count,
                state.owned_liquid_volume,
                dissipation.owned_quadrature_point_count,
                "active-volume dissipation liquid measure");
            if (accepted.active_volume_energy.has_value()) {
                const auto& energy = *accepted.active_volume_energy;
                FE_THROW_IF(
                    dissipation.owned_quadrature_point_count !=
                        energy.owned_quadrature_point_count,
                    InvalidArgumentException,
                    "FESystem::recordAcceptedFreeSurfaceDiscreteFunctionals: "
                    "active-volume energy and dissipation quadrature "
                    "coverage differ");
                require_same_terms_near(
                    dissipation.owned_liquid_volume,
                    dissipation.owned_quadrature_point_count,
                    energy.owned_liquid_volume,
                    energy.owned_quadrature_point_count,
                    "active-volume energy/dissipation liquid measure");
            }
        }

        const bool external_pressure_power_declared =
            declaration.external_pressure_power_parameters.has_value();
        FE_THROW_IF(
            accepted.external_pressure_power.has_value() !=
                external_pressure_power_declared,
            InvalidArgumentException,
            "FESystem::recordAcceptedFreeSurfaceDiscreteFunctionals: "
            "exterior-pressure power presence does not match the "
            "declaration");
        if (accepted.external_pressure_power.has_value()) {
            const auto& pressure =
                *accepted.external_pressure_power;
            const auto& pressure_parameters =
                *declaration.external_pressure_power_parameters;
            FE_THROW_IF(
                pressure.snapshot_revision_key !=
                        revision.snapshot_revision_key ||
                    pressure.liquid_side !=
                        pressure_parameters.liquid_side ||
                    pressure.external_pressure !=
                        pressure_parameters.external_pressure ||
                    !finite_nonnegative(
                        pressure.owned_liquid_gas_area) ||
                    !std::isfinite(
                        pressure.outward_liquid_volume_flux_rate) ||
                    !std::isfinite(
                        pressure.external_pressure_power),
                InvalidArgumentException,
                "FESystem::recordAcceptedFreeSurfaceDiscreteFunctionals: "
                "exterior-pressure power does not match its declaration or "
                "geometry revision");
            require_same_terms_near(
                pressure.owned_liquid_gas_area,
                pressure.owned_quadrature_point_count,
                state.owned_liquid_gas_area,
                pressure.owned_quadrature_point_count,
                "exterior-pressure surface measure");
            require_near(
                pressure.external_pressure_power,
                -pressure.external_pressure *
                    pressure.outward_liquid_volume_flux_rate,
                "exterior-pressure power identity");
        }

        FE_THROW_IF(
            accepted.backward_euler_kinetic_work.has_value() &&
                !active_volume_energy_declared,
            InvalidArgumentException,
            "FESystem::recordAcceptedFreeSurfaceDiscreteFunctionals: "
            "backward-Euler kinetic work requires a declared active-volume "
            "energy");
        if (accepted.backward_euler_kinetic_work.has_value()) {
            const auto& work =
                *accepted.backward_euler_kinetic_work;
            const auto& energy =
                *accepted.active_volume_energy;
            const auto& energy_parameters =
                *declaration.active_volume_energy_parameters;
            FE_THROW_IF(
                work.snapshot_revision_key !=
                        revision.snapshot_revision_key ||
                    work.previous_velocity_revision == 0u ||
                    work.endpoint_velocity_revision == 0u ||
                    work.liquid_side != energy_parameters.liquid_side ||
                    work.density != energy_parameters.density ||
                    work.owned_quadrature_point_count !=
                        energy.owned_quadrature_point_count ||
                    !finite_nonnegative(work.owned_liquid_volume) ||
                    !finite_nonnegative(
                        work.kinetic_energy_before_on_endpoint_domain) ||
                    !finite_nonnegative(work.kinetic_energy_after) ||
                    !std::isfinite(
                        work.kinetic_energy_change_on_endpoint_domain) ||
                    !std::isfinite(
                        work.step_integrated_inertia_work) ||
                    !finite_nonnegative(
                        work.time_discretization_loss) ||
                    !std::isfinite(work.identity_residual),
                InvalidArgumentException,
                "FESystem::recordAcceptedFreeSurfaceDiscreteFunctionals: "
                "backward-Euler kinetic work does not match its declaration "
                "or contains an invalid value");
            require_same_terms_near(
                work.owned_liquid_volume,
                work.owned_quadrature_point_count,
                state.owned_liquid_volume,
                work.owned_quadrature_point_count,
                "backward-Euler kinetic-work liquid measure");
            require_same_terms_near(
                work.owned_liquid_volume,
                work.owned_quadrature_point_count,
                energy.owned_liquid_volume,
                energy.owned_quadrature_point_count,
                "backward-Euler and stored-energy liquid measures");
            require_same_terms_near(
                work.kinetic_energy_after,
                work.owned_quadrature_point_count,
                energy.kinetic_energy,
                energy.owned_quadrature_point_count,
                "backward-Euler endpoint kinetic energy");
            require_near(
                work.kinetic_energy_change_on_endpoint_domain,
                work.kinetic_energy_after -
                    work.kinetic_energy_before_on_endpoint_domain,
                "backward-Euler endpoint-domain kinetic-energy change");
            require_near(
                work.identity_residual,
                work.step_integrated_inertia_work -
                    work.kinetic_energy_change_on_endpoint_domain -
                    work.time_discretization_loss,
                "backward-Euler kinetic identity residual");
            require_near(
                work.identity_residual,
                Real{0.0},
                "backward-Euler kinetic identity");
            if (dt == Real{0.0}) {
                FE_THROW_IF(
                    work.previous_velocity_revision !=
                        work.endpoint_velocity_revision,
                    InvalidArgumentException,
                    "FESystem::recordAcceptedFreeSurfaceDiscreteFunctionals: "
                    "zero-duration kinetic baseline endpoint revisions "
                    "differ");
                require_same_terms_near(
                    work.kinetic_energy_before_on_endpoint_domain,
                    work.owned_quadrature_point_count,
                    work.kinetic_energy_after,
                    work.owned_quadrature_point_count,
                    "zero-duration kinetic baseline");
                require_near(
                    work.step_integrated_inertia_work,
                    Real{0.0},
                    "zero-duration inertia work");
                require_near(
                    work.time_discretization_loss,
                    Real{0.0},
                    "zero-duration time-discretization loss");
            }
        }

        const bool contact_stage_declared =
            !declaration.parameters.dynamic_contact_coefficients.empty();
        FE_THROW_IF(
            accepted.contact_stage.has_value() !=
                contact_stage_declared,
            InvalidArgumentException,
            "FESystem::recordAcceptedFreeSurfaceDiscreteFunctionals: "
            "contact-stage state presence does not match the declaration");
        if (accepted.contact_stage.has_value()) {
            const auto& contact_stage = *accepted.contact_stage;
            const auto& contact_revision =
                contact_stage.geometry_revision;
            const auto& contact_state = contact_stage.state;
            FE_THROW_IF(
                !std::isfinite(contact_stage.stage_time) ||
                    !std::isfinite(contact_stage.stage_alpha_f) ||
                    !(contact_stage.stage_alpha_f > Real{0.0}) ||
                    contact_stage.stage_alpha_f > Real{1.0} ||
                    contact_stage.previous_state_revision == 0u ||
                    contact_stage.endpoint_state_revision == 0u ||
                    contact_stage.stage_state_revision == 0u,
                InvalidArgumentException,
                "FESystem::recordAcceptedFreeSurfaceDiscreteFunctionals: "
                "contact-stage provenance is incomplete");
            FE_THROW_IF(
                contact_stage.endpoint_state_revision !=
                    pre_maintenance_endpoint_state_revision,
                InvalidArgumentException,
                "FESystem::recordAcceptedFreeSurfaceDiscreteFunctionals: "
                "contact-stage endpoint revision does not match the "
                "pre-maintenance endpoint revision");
            require_near(
                contact_stage.stage_time,
                accepted_time -
                    (Real{1.0} - contact_stage.stage_alpha_f) * dt,
                "contact-stage time");
            FE_THROW_IF(
                contact_stage.stage_alpha_f < Real{1.0} &&
                    !contact_stage.first_order_generalized_alpha.has_value(),
                InvalidArgumentException,
                "FESystem::recordAcceptedFreeSurfaceDiscreteFunctionals: "
                "a non-endpoint contact stage requires authentic first-order "
                "generalized-alpha provenance");
            if (contact_stage.first_order_generalized_alpha.has_value()) {
                const auto& generalized_alpha =
                    *contact_stage.first_order_generalized_alpha;
                FE_THROW_IF(
                    !validFreeSurfaceFirstOrderGeneralizedAlphaProvenance(
                        generalized_alpha),
                    InvalidArgumentException,
                    "FESystem::recordAcceptedFreeSurfaceDiscreteFunctionals: "
                    "contact-stage first-order generalized-alpha provenance "
                    "is outside the supported rho-infinity family");
                require_near(
                    generalized_alpha.alpha_f,
                    contact_stage.stage_alpha_f,
                    "contact-stage generalized-alpha alpha_f");
                require_near(
                    generalized_alpha.dt,
                    dt,
                    "contact-stage generalized-alpha dt");
            }
            FE_THROW_IF(
                !contact_revision.complete() ||
                    contact_revision.interface_marker !=
                        declaration.interface_marker ||
                    contact_revision.source_id != expected_source ||
                    contact_revision.domain_id !=
                        declaration.geometry_domain_id ||
                    contact_revision.snapshot_revision_key !=
                        contact_state.snapshot_revision_key,
                InvalidArgumentException,
                "FESystem::recordAcceptedFreeSurfaceDiscreteFunctionals: "
                "contact-stage geometry provenance does not match the "
                "declaration and state");
            FE_THROW_IF(
                contact_state.liquid_side !=
                        declaration.parameters.liquid_side ||
                    contact_state.surface_tension !=
                        declaration.parameters.surface_tension ||
                    !finite_nonnegative(
                        contact_state.owned_contact_measure) ||
                    !finite_nonnegative(
                        contact_state.owned_wetted_wall_measure) ||
                    !finite_nonnegative(
                        contact_state.line_friction_dissipation) ||
                    !finite_nonnegative(
                        contact_state.wall_slip_dissipation) ||
                    !finite_nonnegative(
                        contact_state.total_dissipation) ||
                    contact_state.walls.size() !=
                        declaration.parameters.dynamic_contact_coefficients
                            .size(),
                InvalidArgumentException,
                "FESystem::recordAcceptedFreeSurfaceDiscreteFunctionals: "
                "contact-stage coefficients do not match the declaration");

            Real dynamic_contact_measure_sum = Real{0.0};
            Real dynamic_wetted_wall_measure_sum = Real{0.0};
            Real line_dissipation_sum = Real{0.0};
            Real wall_slip_dissipation_sum = Real{0.0};
            for (std::size_t contact_index = 0;
                 contact_index < contact_state.walls.size();
                 ++contact_index) {
                const auto& wall = contact_state.walls[contact_index];
                const auto& coefficient =
                    declaration.parameters.dynamic_contact_coefficients
                        [contact_index];
                const auto point_count =
                    wall.owned_advancing_point_count +
                    wall.owned_receding_point_count +
                    wall.owned_stationary_point_count;
                const auto finite = [](Real value) {
                    return std::isfinite(value);
                };
                FE_THROW_IF(
                    wall.boundary_marker != coefficient.boundary_marker ||
                        wall.equilibrium_contact_angle_radians !=
                            coefficient
                                .equilibrium_contact_angle_radians ||
                        wall.law != coefficient.law ||
                        wall.mobility != coefficient.mobility ||
                        wall.slip_length != coefficient.slip_length ||
                        wall.dynamic_viscosity !=
                            coefficient.dynamic_viscosity ||
                        point_count !=
                            wall.owned_quadrature_point_count ||
                        !finite_nonnegative(
                            wall.owned_contact_measure) ||
                        ((wall.owned_contact_measure > Real{0.0}) !=
                         (wall.owned_quadrature_point_count > 0u)) ||
                        !finite(wall.dynamic_angle_integral) ||
                        !finite(wall.dynamic_cosine_integral) ||
                        !finite(wall.contact_speed_integral) ||
                        !finite_nonnegative(
                            wall.contact_speed_squared_integral) ||
                        !finite(wall.constitutive_residual_integral) ||
                        !finite(
                            wall.absolute_constitutive_residual_integral) ||
                        wall.absolute_constitutive_residual_integral <
                            Real{0.0} ||
                        !finite_nonnegative(
                            wall.line_friction_dissipation) ||
                        !finite_nonnegative(
                            wall.owned_wetted_wall_measure) ||
                        ((wall.owned_wetted_wall_measure > Real{0.0}) !=
                         (wall.owned_wetted_wall_quadrature_point_count >
                          0u)) ||
                        !finite_nonnegative(
                            wall.wall_slip_speed_integral) ||
                        !finite_nonnegative(
                            wall.wall_slip_speed_squared_integral) ||
                        !finite_nonnegative(
                            wall.wall_slip_dissipation) ||
                        !std::all_of(
                            wall.wall_tangential_velocity_integral.begin(),
                            wall.wall_tangential_velocity_integral.end(),
                            finite) ||
                        !std::all_of(
                            wall.contact_position_integral.begin(),
                            wall.contact_position_integral.end(), finite) ||
                        !std::all_of(
                            wall.wall_normal_integral.begin(),
                            wall.wall_normal_integral.end(), finite) ||
                        !std::all_of(
                            wall.footprint_direction_integral.begin(),
                            wall.footprint_direction_integral.end(), finite) ||
                        !std::all_of(
                            wall.contact_line_tangent_integral.begin(),
                            wall.contact_line_tangent_integral.end(), finite),
                    InvalidArgumentException,
                    "FESystem::recordAcceptedFreeSurfaceDiscreteFunctionals: "
                    "contact-stage wall state is inconsistent or non-finite");
                const Real measure_tolerance =
                    Real{1024.0} *
                    std::numeric_limits<Real>::epsilon() *
                    std::max(Real{1.0}, wall.owned_contact_measure);
                const Real residual_tolerance =
                    Real{1024.0} *
                    std::numeric_limits<Real>::epsilon() *
                    std::max(
                        Real{1.0},
                        std::abs(wall.constitutive_residual_integral));
                const auto vector_norm = [](const auto& values) {
                    Real norm_squared = Real{0.0};
                    for (const auto value : values) {
                        norm_squared += value * value;
                    }
                    return std::sqrt(norm_squared);
                };
                const Real speed_cauchy_scale = std::max(
                    {Real{1.0},
                     std::abs(wall.contact_speed_squared_integral *
                              wall.owned_contact_measure),
                     wall.contact_speed_integral *
                         wall.contact_speed_integral});
                const Real wall_slip_cauchy_scale = std::max(
                    {Real{1.0},
                     wall.wall_slip_speed_squared_integral *
                         wall.owned_wetted_wall_measure,
                     wall.wall_slip_speed_integral *
                         wall.wall_slip_speed_integral,
                     vector_norm(
                         wall.wall_tangential_velocity_integral) *
                         vector_norm(
                             wall.wall_tangential_velocity_integral)});
                FE_THROW_IF(
                    wall.dynamic_angle_integral < Real{0.0} ||
                        wall.dynamic_angle_integral >
                            contact_pi * wall.owned_contact_measure +
                                measure_tolerance ||
                        std::abs(wall.dynamic_cosine_integral) >
                            wall.owned_contact_measure +
                                measure_tolerance ||
                        vector_norm(wall.wall_normal_integral) >
                            wall.owned_contact_measure +
                                measure_tolerance ||
                        vector_norm(
                            wall.footprint_direction_integral) >
                            wall.owned_contact_measure +
                                measure_tolerance ||
                        vector_norm(
                            wall.contact_line_tangent_integral) >
                            wall.owned_contact_measure +
                                measure_tolerance ||
                        wall.contact_speed_squared_integral *
                                    wall.owned_contact_measure -
                                wall.contact_speed_integral *
                                    wall.contact_speed_integral <
                            -Real{1024.0} *
                                std::numeric_limits<Real>::epsilon() *
                                speed_cauchy_scale ||
                        wall.absolute_constitutive_residual_integral +
                                residual_tolerance <
                            std::abs(
                                wall.constitutive_residual_integral) ||
                        wall.wall_slip_speed_squared_integral *
                                    wall.owned_wetted_wall_measure -
                                wall.wall_slip_speed_integral *
                                    wall.wall_slip_speed_integral <
                            -Real{1024.0} *
                                std::numeric_limits<Real>::epsilon() *
                                wall_slip_cauchy_scale ||
                        wall.wall_slip_speed_squared_integral *
                                    wall.owned_wetted_wall_measure -
                                vector_norm(
                                    wall.wall_tangential_velocity_integral) *
                                    vector_norm(
                                        wall.wall_tangential_velocity_integral) <
                            -Real{1024.0} *
                                std::numeric_limits<Real>::epsilon() *
                                wall_slip_cauchy_scale,
                    InvalidArgumentException,
                    "FESystem::recordAcceptedFreeSurfaceDiscreteFunctionals: "
                    "contact-stage integral bounds are inconsistent");
                const bool dynamic_ren_e =
                    wall.law ==
                    interfaces::FreeSurfaceContactLaw::DynamicRenE;
                const Real line_friction_residual = dynamic_ren_e
                    ? wall.contact_speed_integral / wall.mobility
                    : Real{0.0};
                const Real expected_line_dissipation = dynamic_ren_e
                    ? wall.contact_speed_squared_integral / wall.mobility
                    : Real{0.0};
                require_near(
                    wall.constitutive_residual_integral,
                    line_friction_residual -
                        contact_state.surface_tension *
                            (std::cos(
                                 wall.equilibrium_contact_angle_radians) *
                                 wall.owned_contact_measure -
                             wall.dynamic_cosine_integral),
                    "contact-stage law residual integral");
                require_near(
                    wall.line_friction_dissipation,
                    expected_line_dissipation,
                    "contact-stage line-friction dissipation identity");
                require_near(
                    wall.wall_slip_dissipation,
                    wall.dynamic_viscosity / wall.slip_length *
                        wall.wall_slip_speed_squared_integral,
                    "contact-stage wall-slip dissipation identity");

                const bool has_measure =
                    wall.owned_contact_measure > Real{0.0};
                const bool all_means_present =
                    wall.mean_dynamic_angle_radians.has_value() &&
                    wall.mean_dynamic_cosine.has_value() &&
                    wall.mean_contact_speed.has_value() &&
                    wall.mean_constitutive_residual.has_value() &&
                    wall.mean_absolute_constitutive_residual.has_value();
                const bool any_means_present =
                    wall.mean_dynamic_angle_radians.has_value() ||
                    wall.mean_dynamic_cosine.has_value() ||
                    wall.mean_contact_speed.has_value() ||
                    wall.mean_constitutive_residual.has_value() ||
                    wall.mean_absolute_constitutive_residual.has_value();
                FE_THROW_IF(
                    all_means_present != has_measure ||
                        any_means_present != has_measure,
                    InvalidArgumentException,
                    "FESystem::recordAcceptedFreeSurfaceDiscreteFunctionals: "
                    "contact-stage mean presence does not match its measure");
                if (has_measure) {
                    const Real inverse_measure =
                        Real{1.0} / wall.owned_contact_measure;
                    require_near(
                        *wall.mean_dynamic_angle_radians,
                        wall.dynamic_angle_integral * inverse_measure,
                        "mean dynamic contact angle");
                    require_near(
                        *wall.mean_dynamic_cosine,
                        wall.dynamic_cosine_integral * inverse_measure,
                        "mean dynamic contact cosine");
                    require_near(
                        *wall.mean_contact_speed,
                        wall.contact_speed_integral * inverse_measure,
                        "mean contact speed");
                    require_near(
                        *wall.mean_constitutive_residual,
                        wall.constitutive_residual_integral *
                            inverse_measure,
                        "mean contact constitutive residual");
                    require_near(
                        *wall.mean_absolute_constitutive_residual,
                        wall.absolute_constitutive_residual_integral *
                            inverse_measure,
                        "mean absolute contact constitutive residual");
                    for (std::size_t component = 0; component < 3u;
                         ++component) {
                        require_near(
                            wall.mean_contact_position[component],
                            wall.contact_position_integral[component] *
                                inverse_measure,
                            "mean contact position");
                        require_near(
                            wall.mean_wall_normal[component],
                            wall.wall_normal_integral[component] *
                                inverse_measure,
                            "mean contact wall frame");
                        require_near(
                            wall.mean_footprint_direction[component],
                            wall.footprint_direction_integral[component] *
                                inverse_measure,
                            "mean contact footprint frame");
                        require_near(
                            wall.mean_contact_line_tangent[component],
                            wall.contact_line_tangent_integral[component] *
                                inverse_measure,
                            "mean oriented contact-line tangent");
                    }
                } else {
                    for (const auto value : wall.mean_contact_position) {
                        require_near(value, Real{0.0},
                                     "absent contact position");
                    }
                    for (const auto value : wall.mean_wall_normal) {
                        require_near(value, Real{0.0},
                                     "absent contact wall frame");
                    }
                    for (const auto value :
                         wall.mean_footprint_direction) {
                        require_near(value, Real{0.0},
                                     "absent contact footprint frame");
                    }
                    for (const auto value :
                         wall.mean_contact_line_tangent) {
                        require_near(
                            value,
                            Real{0.0},
                            "absent oriented contact-line tangent");
                    }
                }
                const bool has_wetted_wall_measure =
                    wall.owned_wetted_wall_measure > Real{0.0};
                FE_THROW_IF(
                    wall.mean_wall_slip_speed.has_value() !=
                        has_wetted_wall_measure,
                    InvalidArgumentException,
                    "FESystem::recordAcceptedFreeSurfaceDiscreteFunctionals: "
                    "contact-stage wall-slip mean presence does not match "
                    "its sharp wetted-wall measure");
                if (has_wetted_wall_measure) {
                    const Real inverse_wall_measure =
                        Real{1.0} / wall.owned_wetted_wall_measure;
                    require_near(
                        *wall.mean_wall_slip_speed,
                        wall.wall_slip_speed_integral *
                            inverse_wall_measure,
                        "mean wall-slip speed");
                    for (std::size_t component = 0; component < 3u;
                         ++component) {
                        require_near(
                            wall.mean_wall_tangential_velocity[component],
                            wall.wall_tangential_velocity_integral[component] *
                                inverse_wall_measure,
                            "mean wall-tangential velocity");
                    }
                } else {
                    for (const auto value :
                         wall.mean_wall_tangential_velocity) {
                        require_near(value,
                                     Real{0.0},
                                     "absent wall-tangential velocity");
                    }
                }
                auto expected_motion =
                    interfaces::FreeSurfaceContactMotion::Absent;
                if (has_measure) {
                    if (wall.owned_advancing_point_count != 0u &&
                        wall.owned_receding_point_count != 0u) {
                        expected_motion =
                            interfaces::FreeSurfaceContactMotion::Mixed;
                    } else if (wall.owned_advancing_point_count != 0u) {
                        expected_motion =
                            interfaces::FreeSurfaceContactMotion::Advancing;
                    } else if (wall.owned_receding_point_count != 0u) {
                        expected_motion =
                            interfaces::FreeSurfaceContactMotion::Receding;
                    } else {
                        expected_motion = interfaces::
                            FreeSurfaceContactMotion::Stationary;
                    }
                }
                FE_THROW_IF(
                    wall.motion != expected_motion,
                    InvalidArgumentException,
                    "FESystem::recordAcceptedFreeSurfaceDiscreteFunctionals: "
                    "contact-stage motion label is inconsistent");
                dynamic_contact_measure_sum +=
                    wall.owned_contact_measure;
                dynamic_wetted_wall_measure_sum +=
                    wall.owned_wetted_wall_measure;
                line_dissipation_sum +=
                    wall.line_friction_dissipation;
                wall_slip_dissipation_sum +=
                    wall.wall_slip_dissipation;
            }
            // Contact-stage totals are sums of the nonnegative per-wall
            // values, as are the sums above.
            const auto contact_wall_count =
                static_cast<std::uint64_t>(contact_state.walls.size());
            require_same_terms_near(contact_state.owned_contact_measure,
                                    contact_wall_count,
                                    dynamic_contact_measure_sum,
                                    contact_wall_count,
                                    "contact-stage total measure");
            require_same_terms_near(contact_state.line_friction_dissipation,
                                    contact_wall_count,
                                    line_dissipation_sum,
                                    contact_wall_count,
                                    "contact-stage line-friction dissipation");
            require_same_terms_near(contact_state.owned_wetted_wall_measure,
                                    contact_wall_count,
                                    dynamic_wetted_wall_measure_sum,
                                    contact_wall_count,
                                    "contact-stage sharp wetted-wall measure");
            require_same_terms_near(contact_state.wall_slip_dissipation,
                                    contact_wall_count,
                                    wall_slip_dissipation_sum,
                                    contact_wall_count,
                                    "contact-stage wall-slip dissipation");
            require_near(contact_state.total_dissipation,
                         contact_state.line_friction_dissipation +
                             contact_state.wall_slip_dissipation,
                         "contact-stage total contact and wall dissipation");
        }
        FE_THROW_IF(
            !accepted.contact_stage.has_value() &&
                !accepted.contact_line_kinematics.empty(),
            InvalidArgumentException,
            "FESystem::recordAcceptedFreeSurfaceDiscreteFunctionals: "
            "contact-line kinematics require an accepted contact stage");
        if (accepted.contact_stage.has_value()) {
            const auto& contact_stage = *accepted.contact_stage;
            for (std::size_t kinematics_index = 0u;
                 kinematics_index <
                     accepted.contact_line_kinematics.size();
                 ++kinematics_index) {
                const auto& kinematics =
                    accepted.contact_line_kinematics[kinematics_index];
                const auto wall = std::find_if(
                    contact_stage.state.walls.begin(),
                    contact_stage.state.walls.end(),
                    [&](const auto& candidate) {
                        return candidate.boundary_marker ==
                               kinematics.boundary_marker;
                    });
                const auto finite_array = [](const auto& values) {
                    return std::all_of(
                        values.begin(), values.end(), [](Real value) {
                            return std::isfinite(value);
                        });
                };
                FE_THROW_IF(
                    kinematics.boundary_marker < 0 ||
                        (kinematics_index != 0u &&
                         accepted.contact_line_kinematics
                                 [kinematics_index - 1u]
                                     .boundary_marker >=
                             kinematics.boundary_marker) ||
                        wall == contact_stage.state.walls.end() ||
                        !(wall->owned_contact_measure > Real{0.0}) ||
                        !wall->mean_contact_speed.has_value() ||
                        kinematics.previous_accepted_step >=
                            accepted_step ||
                        !std::isfinite(
                            kinematics.previous_accepted_time) ||
                        kinematics.previous_accepted_time > accepted_time ||
                        !std::isfinite(
                            kinematics.previous_stage_time) ||
                        kinematics.previous_stage_state_revision == 0u ||
                        kinematics.previous_snapshot_revision_key == 0u ||
                        !std::isfinite(
                            kinematics.stage_time_interval) ||
                        !(kinematics.stage_time_interval > Real{0.0}) ||
                        !finite_array(
                            kinematics.previous_mean_contact_position) ||
                        !finite_array(
                            kinematics.projection_direction) ||
                        !std::isfinite(
                            kinematics.projected_contact_centroid_speed) ||
                        !std::isfinite(
                            kinematics.mean_fluid_contact_speed) ||
                        !std::isfinite(
                            kinematics.fluid_minus_geometric_contact_speed),
                    InvalidArgumentException,
                    "FESystem::recordAcceptedFreeSurfaceDiscreteFunctionals: "
                    "contact-line geometric kinematics are incomplete or "
                    "non-finite");
                Real projection_norm_squared = Real{0.0};
                Real projected_displacement = Real{0.0};
                for (std::size_t component = 0u; component < 3u;
                     ++component) {
                    projection_norm_squared +=
                        kinematics.projection_direction[component] *
                        kinematics.projection_direction[component];
                    projected_displacement +=
                        (wall->mean_contact_position[component] -
                         kinematics
                             .previous_mean_contact_position[component]) *
                        kinematics.projection_direction[component];
                }
                require_near(
                    std::sqrt(projection_norm_squared),
                    Real{1.0},
                    "contact-line kinematics projection direction");
                require_near(
                    kinematics.stage_time_interval,
                    contact_stage.stage_time -
                        kinematics.previous_stage_time,
                    "contact-line kinematics stage interval");
                require_near(
                    kinematics.projected_contact_centroid_speed,
                    projected_displacement /
                        kinematics.stage_time_interval,
                    "projected contact-centroid speed");
                require_near(
                    kinematics.mean_fluid_contact_speed,
                    *wall->mean_contact_speed,
                    "contact-line kinematics fluid speed");
                require_near(
                    kinematics.fluid_minus_geometric_contact_speed,
                    kinematics.mean_fluid_contact_speed -
                        kinematics.projected_contact_centroid_speed,
                    "fluid/geometric contact-speed mismatch");
            }
        }
    }

    const auto declaration_count =
        functional_declarations_.size();
    if (functional_history_.empty()) {
        FE_THROW_IF(
            std::any_of(
                states.begin(), states.end(), [](const auto& state) {
                    return !state.contact_line_kinematics.empty();
                }),
            InvalidArgumentException,
            "FESystem::recordAcceptedFreeSurfaceDiscreteFunctionals: "
            "initial accepted contact history cannot contain geometric "
            "kinematics without a previous accepted stage");
    }
    if (!functional_history_.empty()) {
        FE_THROW_IF(
            functional_history_.size() <
                    declaration_count ||
                functional_history_.size() %
                        declaration_count !=
                    0u,
            InvalidArgumentException,
            "FESystem::recordAcceptedFreeSurfaceDiscreteFunctionals: "
            "accepted history is not a complete declaration-aligned group");
        const auto history_begin =
            functional_history_.size() -
            declaration_count;
        const auto& latest =
            functional_history_[history_begin];
        for (std::size_t i = 0u; i < declaration_count; ++i) {
            const auto& record =
                functional_history_[history_begin + i];
            FE_THROW_IF(
                record.accepted_step != latest.accepted_step ||
                    record.accepted_time != latest.accepted_time ||
                    record.dt != latest.dt ||
                    record.pre_maintenance_endpoint_state_revision !=
                        latest.pre_maintenance_endpoint_state_revision ||
                    record.state_revision != latest.state_revision ||
                    record.extension_map_revision !=
                        latest.extension_map_revision ||
                    !sameFreeSurfaceDiscreteFunctionalDeclaration(
                        record.declaration,
                        functional_declarations_[i]),
                InvalidArgumentException,
                "FESystem::recordAcceptedFreeSurfaceDiscreteFunctionals: "
                "latest accepted history rows do not share one provenance "
                "and declaration ordering");
        }
        FE_THROW_IF(
            accepted_step < latest.accepted_step ||
                accepted_time < latest.accepted_time,
            InvalidArgumentException,
            "FESystem::recordAcceptedFreeSurfaceDiscreteFunctionals: "
            "accepted history must be monotone");
        if (accepted_step > latest.accepted_step) {
            for (std::size_t i = 0u; i < declaration_count; ++i) {
                const auto& previous =
                    functional_history_[history_begin +
                                                              i];
                const auto expected = states[i].contact_stage.has_value()
                    ? deriveAcceptedContactLineKinematics(
                          previous,
                          accepted_step,
                          *states[i].contact_stage)
                    : std::vector<
                          FreeSurfaceAcceptedContactLineKinematics>{};
                FE_THROW_IF(
                    !sameFreeSurfaceAcceptedContactLineKinematics(
                        states[i].contact_line_kinematics, expected),
                    InvalidArgumentException,
                    "FESystem::recordAcceptedFreeSurfaceDiscreteFunctionals: "
                    "contact-line kinematics do not bind to the latest "
                    "accepted stage and contact-position history");
            }
        }
        if (accepted_step == latest.accepted_step) {
            FE_THROW_IF(
                accepted_time != latest.accepted_time || dt != latest.dt ||
                    pre_maintenance_endpoint_state_revision !=
                        latest.pre_maintenance_endpoint_state_revision ||
                    state_revision != latest.state_revision ||
                    extension_map_revision !=
                        latest.extension_map_revision,
                InvalidArgumentException,
                "FESystem::recordAcceptedFreeSurfaceDiscreteFunctionals: "
                "an accepted step cannot be recorded with conflicting "
                "provenance");
            for (std::size_t i = 0; i < declaration_count; ++i) {
                const auto& record =
                    functional_history_[history_begin +
                                                              i];
                FE_THROW_IF(
                    record.extension_map_revision !=
                            latest.extension_map_revision ||
                        record.declaration.interface_marker !=
                            functional_declarations_[i]
                                .interface_marker ||
                        record.declaration.level_set_field !=
                            functional_declarations_[i]
                                .level_set_field ||
                        record.declaration.curvature_field !=
                            functional_declarations_[i]
                                .curvature_field ||
                        record.declaration.velocity_field !=
                            functional_declarations_[i]
                                .velocity_field ||
                        record.declaration.geometry_domain_id !=
                            functional_declarations_[i]
                                .geometry_domain_id ||
                        record.declaration.capillary_balance_method !=
                            functional_declarations_[i]
                                .capillary_balance_method ||
                        record.declaration
                                .capillary_balance_qualification !=
                            functional_declarations_[i]
                                .capillary_balance_qualification ||
                        record.declaration
                                .endpoint_functional_power_enabled !=
                            functional_declarations_[i]
                                .endpoint_functional_power_enabled ||
                        record.declaration
                                .static_conservative_body_force_complete !=
                            functional_declarations_[i]
                                .static_conservative_body_force_complete ||
                        record.declaration.owner_component !=
                            functional_declarations_[i]
                                .owner_component ||
                        !sameFreeSurfaceFunctionalParameters(
                            record.declaration.parameters,
                            functional_declarations_[i]
                                .parameters) ||
                        record.declaration.active_volume_energy_parameters
                                .has_value() !=
                            functional_declarations_[i]
                                .active_volume_energy_parameters
                                .has_value() ||
                        (record.declaration.active_volume_energy_parameters
                             .has_value() &&
                         !sameFreeSurfaceActiveVolumeEnergyParameters(
                             *record.declaration
                                  .active_volume_energy_parameters,
                             *functional_declarations_[i]
                                  .active_volume_energy_parameters)) ||
                        record.declaration
                                .active_volume_dissipation_parameters
                                .has_value() !=
                            functional_declarations_[i]
                                .active_volume_dissipation_parameters
                                .has_value() ||
                        (record.declaration
                             .active_volume_dissipation_parameters
                             .has_value() &&
                         !sameFreeSurfaceActiveVolumeDissipationParameters(
                             *record.declaration
                                  .active_volume_dissipation_parameters,
                             *functional_declarations_[i]
                                  .active_volume_dissipation_parameters)) ||
                        record.declaration
                                .external_pressure_power_parameters
                                .has_value() !=
                            functional_declarations_[i]
                                .external_pressure_power_parameters
                                .has_value() ||
                        (record.declaration
                             .external_pressure_power_parameters
                             .has_value() &&
                         !sameFreeSurfaceExternalPressurePowerParameters(
                             *record.declaration
                                  .external_pressure_power_parameters,
                             *functional_declarations_[i]
                                  .external_pressure_power_parameters)) ||
                        !sameFreeSurfaceGeometryRevision(
                            record.geometry_revision,
                            states[i].geometry_revision) ||
                        record.cut_topology_revision !=
                            states[i].cut_topology_revision ||
                        record.geometry_pruning_occurred !=
                            states[i].geometry_pruning_occurred ||
                        !sameFreeSurfaceFunctionalState(
                            record.state, states[i].state) ||
                        record.endpoint_functional_power.has_value() !=
                            states[i]
                                .endpoint_functional_power
                                .has_value() ||
                        (record.endpoint_functional_power.has_value() &&
                         !sameFreeSurfaceFunctionalVariationState(
                             *record.endpoint_functional_power,
                             *states[i].endpoint_functional_power)) ||
                        record.active_volume_energy.has_value() !=
                            states[i].active_volume_energy.has_value() ||
                        (record.active_volume_energy.has_value() &&
                         !sameFreeSurfaceActiveVolumeEnergyState(
                             *record.active_volume_energy,
                             *states[i].active_volume_energy)) ||
                        record.active_volume_dissipation.has_value() !=
                            states[i]
                                .active_volume_dissipation
                                .has_value() ||
                        (record.active_volume_dissipation.has_value() &&
                         !sameFreeSurfaceActiveVolumeDissipationState(
                             *record.active_volume_dissipation,
                             *states[i].active_volume_dissipation)) ||
                        record.external_pressure_power.has_value() !=
                            states[i]
                                .external_pressure_power
                                .has_value() ||
                        (record.external_pressure_power.has_value() &&
                         !sameFreeSurfaceExternalPressurePowerState(
                             *record.external_pressure_power,
                             *states[i].external_pressure_power)) ||
                        record.backward_euler_kinetic_work.has_value() !=
                            states[i]
                                .backward_euler_kinetic_work
                                .has_value() ||
                        (record.backward_euler_kinetic_work.has_value() &&
                         !sameFreeSurfaceBackwardEulerKineticWorkState(
                             *record.backward_euler_kinetic_work,
                             *states[i].backward_euler_kinetic_work)) ||
                        record.contact_stage.has_value() !=
                            states[i].contact_stage.has_value() ||
                        (record.contact_stage.has_value() &&
                         !sameFreeSurfaceAcceptedContactStageState(
                             *record.contact_stage,
                             *states[i].contact_stage)) ||
                        !sameFreeSurfaceAcceptedContactLineKinematics(
                            record.contact_line_kinematics,
                            states[i].contact_line_kinematics),
                    InvalidArgumentException,
                    "FESystem::recordAcceptedFreeSurfaceDiscreteFunctionals: "
                    "an accepted step cannot be replayed with different "
                    "functional state");
            }
            return;
        }
    }

    // Grow geometrically: reserving exactly one more group reallocated and
    // moved the whole history on every accepted step.
    functionals_.growHistoryFor(states.size());
    for (std::size_t i = 0; i < states.size(); ++i) {
        const auto& accepted = states[i];
        const auto& declaration =
            functional_declarations_[i];
        functionals_.appendHistory(
            FreeSurfaceDiscreteFunctionalHistoryRecord{
                .accepted_step = accepted_step,
                .accepted_time = accepted_time,
                .dt = dt,
                .pre_maintenance_endpoint_state_revision =
                    pre_maintenance_endpoint_state_revision,
                .state_revision = state_revision,
                .extension_map_revision =
                    extension_map_revision,
                .declaration = declaration,
                .geometry_revision = accepted.geometry_revision,
                .cut_topology_revision =
                    accepted.cut_topology_revision,
                .geometry_pruning_occurred =
                    accepted.geometry_pruning_occurred,
                .state = accepted.state,
                .endpoint_functional_power =
                    accepted.endpoint_functional_power,
                .active_volume_energy =
                    accepted.active_volume_energy,
                .active_volume_dissipation =
                    accepted.active_volume_dissipation,
                .external_pressure_power =
                    accepted.external_pressure_power,
                .backward_euler_kinetic_work =
                    accepted.backward_euler_kinetic_work,
                .contact_stage = accepted.contact_stage,
                .contact_line_kinematics =
                    accepted.contact_line_kinematics,
            });
        const auto& revision = accepted.geometry_revision;
        const auto& state = accepted.state;
        std::ostringstream message;
        message << std::setprecision(17)
                << "FESystem: accepted free-surface discrete functional"
                << " diagnostic=free_surface_discrete_functional_history"
                << " accepted_step=" << accepted_step
                << " accepted_time=" << accepted_time
                << " dt=" << dt
                << " pre_maintenance_endpoint_state_revision="
                << pre_maintenance_endpoint_state_revision
                << " state_revision=" << state_revision
                << " extension_map_revision_available="
                << (extension_map_revision.has_value()
                        ? "true"
                        : "false")
                << " extension_map_revision="
                << extension_map_revision.value_or(0u)
                << " interface_marker=" << declaration.interface_marker
                << " level_set_field=" << declaration.level_set_field
                << " curvature_field=" << declaration.curvature_field
                << " velocity_field=" << declaration.velocity_field
                << " capillary_balance_method="
                << (declaration.capillary_balance_method ==
                            FreeSurfaceCapillaryBalanceMethod::
                                DiscreteEnergyVolumeStationarity
                        ? "discrete_energy_volume_stationarity"
                        : declaration.capillary_balance_method ==
                                  FreeSurfaceCapillaryBalanceMethod::
                                      KinematicAreaGradientEnergyTraction
                              ? "kinematic_area_gradient_energy_traction"
                              : "unselected")
                << " capillary_balance_qualification="
                << (declaration.capillary_balance_qualification ==
                            FreeSurfaceCapillaryBalanceQualification::Qualified
                        ? "qualified"
                        : declaration.capillary_balance_qualification ==
                                  FreeSurfaceCapillaryBalanceQualification::
                                      PrerequisiteOnly
                              ? "prerequisite_only"
                              : "unselected")
                << " geometry_source_id='" << revision.source_id << "'"
                << " geometry_domain_id='" << revision.domain_id << "'"
                << " geometry_isovalue=" << revision.isovalue
                << " source_layout_revision="
                << revision.source_layout_revision
                << " source_value_revision="
                << revision.source_value_revision
                << " mesh_geometry_revision="
                << revision.mesh_geometry_revision
                << " mesh_topology_revision="
                << revision.mesh_topology_revision
                << " ownership_revision=" << revision.ownership_revision
                << " numbering_revision=" << revision.numbering_revision
                << " quadrature_policy_key="
                << revision.quadrature_policy_key
                << " snapshot_revision="
                << revision.snapshot_revision_key
                << " cut_topology_revision="
                << accepted.cut_topology_revision
                << " geometry_pruning_occurred="
                << (accepted.geometry_pruning_occurred
                        ? "true"
                        : "false")
                << " liquid_side="
                << (state.liquid_side ==
                            geometry::CutIntegrationSide::Negative
                        ? "Negative"
                        : "Positive")
                << " surface_tension=" << state.surface_tension
                << " volume_multiplier=" << state.volume_multiplier
                << " liquid_volume=" << state.owned_liquid_volume
                << " liquid_gas_area=" << state.owned_liquid_gas_area
                << " wetted_wall_area=" << state.owned_wetted_wall_area
                << " contact_measure=" << state.owned_contact_measure
                << " liquid_gas_surface_energy="
                << state.liquid_gas_surface_energy
                << " young_wall_energy=" << state.young_wall_energy
                << " volume_constraint_potential="
                << state.volume_constraint_potential
                << " total_potential=" << state.total_potential
                << " wall_count=" << state.walls.size()
                << " owner='" << declaration.owner_component << "'";
        if (accepted.endpoint_functional_power.has_value()) {
            const auto& power =
                *accepted.endpoint_functional_power;
            message
                << " endpoint_liquid_volume_power="
                << power.owned_liquid_volume_variation
                << " endpoint_liquid_gas_area_power="
                << power.owned_liquid_gas_area_variation
                << " endpoint_wetted_wall_area_power="
                << power.owned_wetted_wall_area_variation
                << " endpoint_liquid_gas_surface_energy_power="
                << power.liquid_gas_surface_energy_variation
                << " endpoint_young_wall_energy_power="
                << power.young_wall_energy_variation
                << " endpoint_volume_constraint_power="
                << power.volume_constraint_potential_variation
                << " endpoint_total_functional_power="
                << power.total_potential_variation;
        }
        if (accepted.active_volume_energy.has_value()) {
            const auto& energy = *accepted.active_volume_energy;
            message
                << " active_volume_qpoints="
                << energy.owned_quadrature_point_count
                << " active_volume_liquid_measure="
                << energy.owned_liquid_volume
                << " kinetic_energy=" << energy.kinetic_energy
                << " gravitational_energy="
                << energy.gravitational_energy
                << " gravitational_potential_power="
                << energy.gravitational_potential_power
                << " active_volume_total_energy="
                << energy.total_energy;
        }
        if (accepted.active_volume_dissipation.has_value()) {
            const auto& dissipation =
                *accepted.active_volume_dissipation;
            message
                << " active_volume_dissipation_qpoints="
                << dissipation.owned_quadrature_point_count
                << " active_volume_dissipation_liquid_measure="
                << dissipation.owned_liquid_volume
                << " bulk_viscous_dissipation_rate="
                << dissipation.bulk_viscous_dissipation_rate;
        }
        if (accepted.external_pressure_power.has_value()) {
            const auto& pressure =
                *accepted.external_pressure_power;
            message
                << " exterior_pressure_surface_qpoints="
                << pressure.owned_quadrature_point_count
                << " exterior_pressure_surface_area="
                << pressure.owned_liquid_gas_area
                << " exterior_pressure_outward_liquid_flux_rate="
                << pressure.outward_liquid_volume_flux_rate
                << " exterior_pressure=" << pressure.external_pressure
                << " exterior_pressure_power="
                << pressure.external_pressure_power;
        }
        if (accepted.backward_euler_kinetic_work.has_value()) {
            const auto& work =
                *accepted.backward_euler_kinetic_work;
            message
                << " backward_euler_previous_velocity_revision="
                << work.previous_velocity_revision
                << " backward_euler_endpoint_velocity_revision="
                << work.endpoint_velocity_revision
                << " backward_euler_kinetic_before_on_endpoint_domain="
                << work.kinetic_energy_before_on_endpoint_domain
                << " backward_euler_kinetic_after="
                << work.kinetic_energy_after
                << " backward_euler_kinetic_change_on_endpoint_domain="
                << work.kinetic_energy_change_on_endpoint_domain
                << " backward_euler_step_integrated_inertia_work="
                << work.step_integrated_inertia_work
                << " backward_euler_time_discretization_loss="
                << work.time_discretization_loss
                << " backward_euler_kinetic_identity_residual="
                << work.identity_residual;
        }
        for (const auto& wall : state.walls) {
            message << " wall_marker=" << wall.boundary_marker;
            if (wall.equilibrium_contact_angle_radians.has_value()) {
                message << " wall_equilibrium_contact_angle_radians="
                        << *wall.equilibrium_contact_angle_radians;
            } else {
                message << " wall_equilibrium_contact_angle_radians=none";
            }
            message << " wall_wetted_area="
                    << wall.owned_wetted_wall_area
                    << " wall_contact_measure="
                    << wall.owned_contact_measure
                    << " wall_young_energy=" << wall.young_wall_energy;
        }
        if (accepted.contact_stage.has_value()) {
            const auto& contact_stage = *accepted.contact_stage;
            message
                << " contact_stage_time=" << contact_stage.stage_time
                << " contact_stage_alpha_f="
                << contact_stage.stage_alpha_f
                << " contact_stage_previous_state_revision="
                << contact_stage.previous_state_revision
                << " contact_stage_endpoint_state_revision="
                << contact_stage.endpoint_state_revision
                << " contact_stage_state_revision="
                << contact_stage.stage_state_revision
                << " contact_stage_snapshot_revision="
                << contact_stage.geometry_revision.snapshot_revision_key
                << " contact_stage_source_value_revision="
                << contact_stage.geometry_revision.source_value_revision
                << " contact_stage_measure="
                << contact_stage.state.owned_contact_measure
                << " contact_stage_wetted_wall_measure="
                << contact_stage.state.owned_wetted_wall_measure
                << " contact_stage_line_friction_dissipation="
                << contact_stage.state.line_friction_dissipation
                << " contact_stage_wall_slip_dissipation="
                << contact_stage.state.wall_slip_dissipation
                << " contact_stage_total_dissipation="
                << contact_stage.state.total_dissipation;
            if (contact_stage.first_order_generalized_alpha.has_value()) {
                const auto& generalized_alpha =
                    *contact_stage.first_order_generalized_alpha;
                message
                    << " contact_stage_generalized_alpha_alpha_m="
                    << generalized_alpha.alpha_m
                    << " contact_stage_generalized_alpha_alpha_f="
                    << generalized_alpha.alpha_f
                    << " contact_stage_generalized_alpha_gamma="
                    << generalized_alpha.gamma
                    << " contact_stage_generalized_alpha_dt="
                    << generalized_alpha.dt;
            } else {
                message << " contact_stage_generalized_alpha=none";
            }
            for (const auto& contact : contact_stage.state.walls) {
                const char* motion = "Absent";
                switch (contact.motion) {
                case interfaces::FreeSurfaceContactMotion::Absent:
                    break;
                case interfaces::FreeSurfaceContactMotion::Stationary:
                    motion = "Stationary";
                    break;
                case interfaces::FreeSurfaceContactMotion::Advancing:
                    motion = "Advancing";
                    break;
                case interfaces::FreeSurfaceContactMotion::Receding:
                    motion = "Receding";
                    break;
                case interfaces::FreeSurfaceContactMotion::Mixed:
                    motion = "Mixed";
                    break;
                }
                message
                    << " contact_wall_marker=" << contact.boundary_marker
                    << " contact_law="
                    << (contact.law ==
                                interfaces::FreeSurfaceContactLaw::DynamicRenE
                            ? "DynamicRenE"
                            : "PrescribedAngle")
                    << " contact_equilibrium_angle_radians="
                    << contact.equilibrium_contact_angle_radians
                    << " contact_mobility=" << contact.mobility
                    << " contact_slip_length=" << contact.slip_length
                    << " contact_dynamic_viscosity="
                    << contact.dynamic_viscosity
                    << " contact_qpoints="
                    << contact.owned_quadrature_point_count
                    << " contact_advancing_qpoints="
                    << contact.owned_advancing_point_count
                    << " contact_receding_qpoints="
                    << contact.owned_receding_point_count
                    << " contact_stationary_qpoints="
                    << contact.owned_stationary_point_count
                    << " contact_measure="
                    << contact.owned_contact_measure
                    << " contact_motion=" << motion
                    << " contact_line_friction_dissipation="
                    << contact.line_friction_dissipation
                    << " contact_speed_squared_integral="
                    << contact.contact_speed_squared_integral
                    << " contact_wetted_wall_qpoints="
                    << contact.owned_wetted_wall_quadrature_point_count
                    << " contact_wetted_wall_measure="
                    << contact.owned_wetted_wall_measure
                    << " contact_wall_slip_speed_integral="
                    << contact.wall_slip_speed_integral
                    << " contact_wall_slip_speed_squared_integral="
                    << contact.wall_slip_speed_squared_integral
                    << " contact_wall_slip_dissipation="
                    << contact.wall_slip_dissipation;
                const auto kinematics = std::find_if(
                    accepted.contact_line_kinematics.begin(),
                    accepted.contact_line_kinematics.end(),
                    [&](const auto& candidate) {
                        return candidate.boundary_marker ==
                               contact.boundary_marker;
                    });
                message << " contact_geometric_kinematics_available="
                        << (kinematics !=
                                    accepted.contact_line_kinematics.end()
                                ? "true"
                                : "false");
                if (kinematics !=
                    accepted.contact_line_kinematics.end()) {
                    message
                        << " contact_geometric_previous_accepted_step="
                        << kinematics->previous_accepted_step
                        << " contact_geometric_previous_accepted_time="
                        << kinematics->previous_accepted_time
                        << " contact_geometric_previous_stage_time="
                        << kinematics->previous_stage_time
                        << " contact_geometric_previous_stage_state_revision="
                        << kinematics->previous_stage_state_revision
                        << " contact_geometric_previous_snapshot_revision="
                        << kinematics->previous_snapshot_revision_key
                        << " contact_geometric_stage_time_interval="
                        << kinematics->stage_time_interval
                        << " contact_projected_centroid_speed="
                        << kinematics->projected_contact_centroid_speed
                        << " contact_geometric_mean_fluid_speed="
                        << kinematics->mean_fluid_contact_speed
                        << " contact_fluid_minus_geometric_speed="
                        << kinematics->fluid_minus_geometric_contact_speed;
                }
                if (contact.mean_dynamic_angle_radians.has_value()) {
                    message
                        << " contact_mean_dynamic_angle_radians="
                        << *contact.mean_dynamic_angle_radians
                        << " contact_mean_dynamic_cosine="
                        << *contact.mean_dynamic_cosine
                        << " contact_mean_speed="
                        << *contact.mean_contact_speed
                        << " contact_mean_constitutive_residual="
                        << *contact.mean_constitutive_residual
                        << " contact_mean_absolute_constitutive_residual="
                        << *contact.mean_absolute_constitutive_residual;
                } else {
                    message << " contact_mean_dynamic_angle_radians=none"
                            << " contact_mean_dynamic_cosine=none"
                            << " contact_mean_speed=none"
                            << " contact_mean_constitutive_residual=none"
                            << " contact_mean_absolute_constitutive_residual=none";
                }
                if (contact.mean_wall_slip_speed.has_value()) {
                    message << " contact_mean_wall_slip_speed="
                            << *contact.mean_wall_slip_speed;
                } else {
                    message << " contact_mean_wall_slip_speed=none";
                }
                for (std::size_t component = 0; component < 3u;
                     ++component) {
                    message
                        << " contact_mean_position_" << component << "="
                        << contact.mean_contact_position[component]
                        << " contact_mean_wall_normal_" << component << "="
                        << contact.mean_wall_normal[component]
                        << " contact_mean_footprint_direction_" << component
                        << "="
                        << contact.mean_footprint_direction[component]
                        << " contact_mean_contact_line_tangent_"
                        << component << "="
                        << contact.mean_contact_line_tangent[component]
                        << " contact_mean_wall_tangential_velocity_"
                        << component << "="
                        << contact.mean_wall_tangential_velocity[component];
                    if (kinematics !=
                        accepted.contact_line_kinematics.end()) {
                        message
                            << " contact_geometric_previous_mean_position_"
                            << component << "="
                            << kinematics
                                   ->previous_mean_contact_position[component]
                            << " contact_geometric_projection_direction_"
                            << component << "="
                            << kinematics->projection_direction[component];
                    }
                }
            }
        }
        FE_LOG_INFO(message.str());
    }
}


} // namespace systems
} // namespace FE
} // namespace svmp

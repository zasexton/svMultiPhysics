/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Physics/Formulations/NavierStokes/FreeSurface/SnapshotFunctionals.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <numbers>
#include <stdexcept>
#include <utility>

namespace svmp::Physics::formulations::navier_stokes::free_surface {

using namespace svmp::FE;
using namespace svmp::FE::interfaces;

namespace {

[[nodiscard]] bool finitePoint(const std::array<Real, 3>& point) noexcept
{
    return std::isfinite(point[0]) && std::isfinite(point[1]) &&
           std::isfinite(point[2]);
}

[[nodiscard]] Real dot(const std::array<Real, 3>& a,
                       const std::array<Real, 3>& b) noexcept
{
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

[[nodiscard]] std::array<Real, 3> crossPoint(
    const std::array<Real, 3>& a,
    const std::array<Real, 3>& b) noexcept
{
    return {{
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    }};
}

[[nodiscard]] Real norm(const std::array<Real, 3>& point) noexcept
{
    return std::sqrt(dot(point, point));
}

} // namespace

FreeSurfaceDiscreteFunctionalState freeSurfaceDiscreteFunctionalLayout(
    const LevelSetGeometrySnapshot& snapshot,
    const FreeSurfaceDiscreteFunctionalParameters& parameters)
{
    if (!snapshot.revision().complete()) {
        throw std::invalid_argument(
            "free-surface functional requires a revision-complete snapshot");
    }
    if (!std::isfinite(parameters.surface_tension) ||
        parameters.surface_tension < Real{0.0}) {
        throw std::invalid_argument(
            "free-surface functional requires finite nonnegative surface tension");
    }
    if (!std::isfinite(parameters.volume_multiplier)) {
        throw std::invalid_argument(
            "free-surface functional requires a finite volume multiplier");
    }
    std::map<int, FreeSurfaceDiscreteWallFunctionalState> walls;
    for (const auto& coefficient : parameters.young_wall_coefficients) {
        if (coefficient.boundary_marker < 0 ||
            !std::isfinite(
                coefficient.equilibrium_contact_angle_radians) ||
            !(coefficient.equilibrium_contact_angle_radians > Real{0.0}) ||
            !(coefficient.equilibrium_contact_angle_radians <
              std::numbers::pi_v<Real>)) {
            throw std::invalid_argument(
                "free-surface functional wall coefficient requires a nonnegative marker and an angle strictly between zero and pi");
        }
        FreeSurfaceDiscreteWallFunctionalState wall;
        wall.boundary_marker = coefficient.boundary_marker;
        wall.equilibrium_contact_angle_radians =
            coefficient.equilibrium_contact_angle_radians;
        if (!walls.emplace(coefficient.boundary_marker, wall).second) {
            throw std::invalid_argument(
                "free-surface functional has duplicate coefficients for one physical boundary");
        }
    }
    const auto ensure_geometry_wall = [&walls](int boundary_marker) {
        auto& wall = walls[boundary_marker];
        wall.boundary_marker = boundary_marker;
    };
    for (const auto& contact : snapshot.contactDomains()) {
        ensure_geometry_wall(contact.boundaryMarker());
    }
    for (const auto& active : snapshot.activeBoundaryDomains()) {
        ensure_geometry_wall(active.request().boundary_marker);
    }

    const auto wall_role =
        parameters.active_side == geometry::CutIntegrationSide::Negative
            ? LevelSetGeometryRuleRole::NegativeExteriorBoundary
            : LevelSetGeometryRuleRole::PositiveExteriorBoundary;

    FreeSurfaceDiscreteFunctionalState state;
    state.snapshot_revision_key =
        snapshot.revision().snapshot_revision_key;
    state.active_side = parameters.active_side;
    state.surface_tension = parameters.surface_tension;
    state.volume_multiplier = parameters.volume_multiplier;

    // Every wall the owned retained rules touch is a wall of the layout.
    for (const auto& record : snapshot.rules()) {
        if (!record.locally_owned ||
            record.retention != LevelSetGeometryRetention::Retained) {
            continue;
        }
        if (record.role == wall_role) {
            if (record.physical_boundary_marker < 0) {
                throw std::invalid_argument(
                    "free-surface functional encountered exterior-boundary geometry without a physical marker");
            }
            ensure_geometry_wall(record.physical_boundary_marker);
        } else if (record.role == LevelSetGeometryRuleRole::Contact) {
            if (record.physical_boundary_marker < 0) {
                throw std::invalid_argument(
                    "free-surface functional encountered contact geometry without a physical marker");
            }
            ensure_geometry_wall(record.physical_boundary_marker);
        }
    }

    state.walls.reserve(walls.size());
    for (auto& [boundary_marker, wall] : walls) {
        (void)boundary_marker;
        state.walls.push_back(wall);
    }
    return state;
}

FreeSurfaceDiscreteFunctionalVariationState
freeSurfaceDiscreteFunctionalVariationLayout(
    const LevelSetGeometrySnapshot& snapshot,
    const FreeSurfaceDiscreteFunctionalParameters& parameters)
{
    if (parameters.active_side != geometry::CutIntegrationSide::Negative &&
        parameters.active_side != geometry::CutIntegrationSide::Positive) {
        throw std::invalid_argument(
            "free-surface functional first variation requires a physical liquid side");
    }

    // The layout's validation and wall canonicalization, so the walls of
    // the variation are the walls of the record.
    const auto layout = freeSurfaceDiscreteFunctionalLayout(snapshot, parameters);

    FreeSurfaceDiscreteFunctionalVariationState state;
    state.snapshot_revision_key = layout.snapshot_revision_key;
    state.active_side = layout.active_side;
    state.surface_tension = layout.surface_tension;
    state.volume_multiplier = layout.volume_multiplier;
    state.walls.reserve(layout.walls.size());
    for (const auto& layout_wall : layout.walls) {
        state.walls.push_back(
            FreeSurfaceDiscreteWallFunctionalVariationState{
                .boundary_marker = layout_wall.boundary_marker,
                .equilibrium_contact_angle_radians =
                    layout_wall.equilibrium_contact_angle_radians,
            });
    }
    return state;
}

void finalizeFreeSurfaceDynamicContactState(
    FreeSurfaceDynamicContactState& state)
{
    state.owned_contact_measure = Real{0.0};
    state.owned_wetted_wall_measure = Real{0.0};
    state.line_friction_dissipation = Real{0.0};
    state.wall_slip_dissipation = Real{0.0};
    state.total_dissipation = Real{0.0};
    for (auto& wall : state.walls) {
        const bool prescribed_angle =
            wall.law == FreeSurfaceContactLaw::PrescribedAngle;
        const bool dynamic_ren_e =
            wall.law == FreeSurfaceContactLaw::DynamicRenE;
        const auto count = wall.owned_advancing_point_count +
                           wall.owned_receding_point_count +
                           wall.owned_stationary_point_count;
        const auto finite_array = [](const auto& values) {
            return std::all_of(values.begin(), values.end(), [](Real value) {
                return std::isfinite(value);
            });
        };
        if (wall.boundary_marker < 0 ||
            !std::isfinite(wall.equilibrium_contact_angle_radians) ||
            !(wall.equilibrium_contact_angle_radians > Real{0.0}) ||
            !(wall.equilibrium_contact_angle_radians <
              std::numbers::pi_v<Real>) ||
            !std::isfinite(wall.mobility) ||
            (!prescribed_angle && !dynamic_ren_e) ||
            (dynamic_ren_e && !(wall.mobility > Real{0.0})) ||
            (prescribed_angle && wall.mobility != Real{0.0}) ||
            !std::isfinite(wall.slip_length) ||
            !(wall.slip_length > Real{0.0}) ||
            !std::isfinite(wall.dynamic_viscosity) ||
            !(wall.dynamic_viscosity > Real{0.0}) ||
            count != wall.owned_quadrature_point_count ||
            !std::isfinite(wall.owned_contact_measure) ||
            wall.owned_contact_measure < Real{0.0} ||
            ((wall.owned_contact_measure > Real{0.0}) !=
             (wall.owned_quadrature_point_count > 0u)) ||
            !std::isfinite(wall.dynamic_angle_integral) ||
            !std::isfinite(wall.dynamic_cosine_integral) ||
            !std::isfinite(wall.contact_speed_integral) ||
            !std::isfinite(wall.contact_speed_squared_integral) ||
            wall.contact_speed_squared_integral < Real{0.0} ||
            !std::isfinite(wall.constitutive_residual_integral) ||
            !std::isfinite(
                wall.absolute_constitutive_residual_integral) ||
            wall.absolute_constitutive_residual_integral < Real{0.0} ||
            !std::isfinite(wall.line_friction_dissipation) ||
            wall.line_friction_dissipation < Real{0.0} ||
            !std::isfinite(wall.owned_wetted_wall_measure) ||
            wall.owned_wetted_wall_measure < Real{0.0} ||
            ((wall.owned_wetted_wall_measure > Real{0.0}) !=
             (wall.owned_wetted_wall_quadrature_point_count > 0u)) ||
            !std::isfinite(wall.wall_slip_speed_integral) ||
            wall.wall_slip_speed_integral < Real{0.0} ||
            !std::isfinite(wall.wall_slip_speed_squared_integral) ||
            wall.wall_slip_speed_squared_integral < Real{0.0} ||
            !std::isfinite(wall.wall_slip_dissipation) ||
            wall.wall_slip_dissipation < Real{0.0} ||
            !finite_array(wall.wall_tangential_velocity_integral) ||
            !finite_array(wall.contact_position_integral) ||
            !finite_array(wall.wall_normal_integral) ||
            !finite_array(wall.footprint_direction_integral) ||
            !finite_array(wall.contact_line_tangent_integral)) {
            throw std::invalid_argument(
                "free-surface dynamic-contact state has invalid additive data");
        }
        const Real wall_slip_cauchy_scale = std::max(
            {Real{1.0},
             wall.wall_slip_speed_squared_integral *
                 wall.owned_wetted_wall_measure,
             wall.wall_slip_speed_integral *
                 wall.wall_slip_speed_integral});
        Real tangential_integral_norm_squared = Real{0.0};
        for (const auto component :
             wall.wall_tangential_velocity_integral) {
            tangential_integral_norm_squared += component * component;
        }
        const Real wall_slip_cauchy_tolerance =
            Real{1024.0} * std::numeric_limits<Real>::epsilon() *
            wall_slip_cauchy_scale;
        if (wall.wall_slip_speed_squared_integral *
                    wall.owned_wetted_wall_measure -
                wall.wall_slip_speed_integral *
                    wall.wall_slip_speed_integral <
                -wall_slip_cauchy_tolerance ||
            wall.wall_slip_speed_squared_integral *
                        wall.owned_wetted_wall_measure -
                    tangential_integral_norm_squared <
                -wall_slip_cauchy_tolerance) {
            throw std::invalid_argument(
                "free-surface dynamic-contact wall-slip integrals violate their measure bounds");
        }
        wall.line_friction_dissipation = dynamic_ren_e
            ? wall.contact_speed_squared_integral / wall.mobility
            : Real{0.0};
        wall.wall_slip_dissipation =
            wall.dynamic_viscosity / wall.slip_length *
            wall.wall_slip_speed_squared_integral;
        state.owned_contact_measure += wall.owned_contact_measure;
        state.owned_wetted_wall_measure +=
            wall.owned_wetted_wall_measure;
        state.line_friction_dissipation +=
            wall.line_friction_dissipation;
        state.wall_slip_dissipation += wall.wall_slip_dissipation;
        if (wall.owned_wetted_wall_measure > Real{0.0}) {
            const Real inverse_wall_measure =
                Real{1.0} / wall.owned_wetted_wall_measure;
            wall.mean_wall_slip_speed =
                wall.wall_slip_speed_integral * inverse_wall_measure;
            for (std::size_t component = 0; component < 3u; ++component) {
                wall.mean_wall_tangential_velocity[component] =
                    wall.wall_tangential_velocity_integral[component] *
                    inverse_wall_measure;
            }
        } else {
            wall.mean_wall_slip_speed.reset();
            wall.mean_wall_tangential_velocity = {{0.0, 0.0, 0.0}};
        }
        if (!(wall.owned_contact_measure > Real{0.0})) {
            wall.mean_dynamic_angle_radians.reset();
            wall.mean_dynamic_cosine.reset();
            wall.mean_contact_speed.reset();
            wall.mean_constitutive_residual.reset();
            wall.mean_absolute_constitutive_residual.reset();
            wall.mean_contact_position = {{0.0, 0.0, 0.0}};
            wall.mean_wall_normal = {{0.0, 0.0, 0.0}};
            wall.mean_footprint_direction = {{0.0, 0.0, 0.0}};
            wall.mean_contact_line_tangent = {{0.0, 0.0, 0.0}};
            wall.motion = FreeSurfaceContactMotion::Absent;
            continue;
        }
        const Real inverse_measure =
            Real{1.0} / wall.owned_contact_measure;
        wall.mean_dynamic_angle_radians =
            wall.dynamic_angle_integral * inverse_measure;
        wall.mean_dynamic_cosine =
            wall.dynamic_cosine_integral * inverse_measure;
        wall.mean_contact_speed =
            wall.contact_speed_integral * inverse_measure;
        wall.mean_constitutive_residual =
            wall.constitutive_residual_integral * inverse_measure;
        wall.mean_absolute_constitutive_residual =
            wall.absolute_constitutive_residual_integral * inverse_measure;
        for (std::size_t component = 0; component < 3u; ++component) {
            wall.mean_contact_position[component] =
                wall.contact_position_integral[component] *
                inverse_measure;
            wall.mean_wall_normal[component] =
                wall.wall_normal_integral[component] * inverse_measure;
            wall.mean_footprint_direction[component] =
                wall.footprint_direction_integral[component] *
                inverse_measure;
            wall.mean_contact_line_tangent[component] =
                wall.contact_line_tangent_integral[component] *
                inverse_measure;
        }
        if (wall.owned_advancing_point_count != 0u &&
            wall.owned_receding_point_count != 0u) {
            wall.motion = FreeSurfaceContactMotion::Mixed;
        } else if (wall.owned_advancing_point_count != 0u) {
            wall.motion = FreeSurfaceContactMotion::Advancing;
        } else if (wall.owned_receding_point_count != 0u) {
            wall.motion = FreeSurfaceContactMotion::Receding;
        } else {
            wall.motion = FreeSurfaceContactMotion::Stationary;
        }
    }
    state.total_dissipation = state.line_friction_dissipation +
                              state.wall_slip_dissipation;
    if (!std::isfinite(state.owned_contact_measure) ||
        !std::isfinite(state.owned_wetted_wall_measure) ||
        !std::isfinite(state.line_friction_dissipation) ||
        !std::isfinite(state.wall_slip_dissipation) ||
        !std::isfinite(state.total_dissipation)) {
        throw std::invalid_argument(
            "free-surface dynamic-contact state produced a non-finite total");
    }
}

FreeSurfaceDynamicContactState evaluateFreeSurfaceDynamicContactState(
    const LevelSetGeometrySnapshot& snapshot,
    const FreeSurfaceDiscreteFunctionalParameters& parameters,
    const SnapshotVectorEvaluator& velocity)
{
    if (!snapshot.revision().complete()) {
        throw std::invalid_argument(
            "free-surface dynamic-contact evaluation requires a revision-complete snapshot");
    }
    if (parameters.active_side != geometry::CutIntegrationSide::Negative &&
        parameters.active_side != geometry::CutIntegrationSide::Positive) {
        throw std::invalid_argument(
            "free-surface dynamic-contact evaluation requires a physical liquid side");
    }
    if (!std::isfinite(parameters.surface_tension) ||
        parameters.surface_tension < Real{0.0}) {
        throw std::invalid_argument(
            "free-surface dynamic-contact evaluation requires finite nonnegative surface tension");
    }
    if (!parameters.dynamic_contact_coefficients.empty() &&
        !velocity.canEvaluateValue()) {
        throw std::invalid_argument(
            "free-surface dynamic-contact evaluation requires a velocity evaluator");
    }

    std::map<int, FreeSurfaceDynamicContactWallState> walls;
    for (const auto& coefficient :
         parameters.dynamic_contact_coefficients) {
        const bool prescribed_angle =
            coefficient.law == FreeSurfaceContactLaw::PrescribedAngle;
        const bool dynamic_ren_e =
            coefficient.law == FreeSurfaceContactLaw::DynamicRenE;
        if (coefficient.boundary_marker < 0 ||
            !std::isfinite(
                coefficient.equilibrium_contact_angle_radians) ||
            !(coefficient.equilibrium_contact_angle_radians > Real{0.0}) ||
            !(coefficient.equilibrium_contact_angle_radians <
              std::numbers::pi_v<Real>) ||
            !std::isfinite(coefficient.mobility) ||
            (!prescribed_angle && !dynamic_ren_e) ||
            (dynamic_ren_e && !(coefficient.mobility > Real{0.0})) ||
            (prescribed_angle && coefficient.mobility != Real{0.0}) ||
            !std::isfinite(coefficient.slip_length) ||
            !(coefficient.slip_length > Real{0.0}) ||
            !std::isfinite(coefficient.dynamic_viscosity) ||
            !(coefficient.dynamic_viscosity > Real{0.0})) {
            throw std::invalid_argument(
                "free-surface contact coefficient requires a supported law, a nonnegative marker, an angle strictly between zero and pi, law-consistent mobility, positive slip length, and positive dynamic viscosity");
        }
        const auto young = std::find_if(
            parameters.young_wall_coefficients.begin(),
            parameters.young_wall_coefficients.end(),
            [&](const auto& candidate) {
                return candidate.boundary_marker ==
                       coefficient.boundary_marker;
            });
        if (young == parameters.young_wall_coefficients.end() ||
            young->equilibrium_contact_angle_radians !=
                coefficient.equilibrium_contact_angle_radians) {
            throw std::invalid_argument(
                "free-surface dynamic-contact coefficient requires a matching Young wall coefficient");
        }
        FreeSurfaceDynamicContactWallState wall;
        wall.boundary_marker = coefficient.boundary_marker;
        wall.equilibrium_contact_angle_radians =
            coefficient.equilibrium_contact_angle_radians;
        wall.law = coefficient.law;
        wall.mobility = coefficient.mobility;
        wall.slip_length = coefficient.slip_length;
        wall.dynamic_viscosity = coefficient.dynamic_viscosity;
        if (!walls.emplace(coefficient.boundary_marker, wall).second) {
            throw std::invalid_argument(
                "free-surface dynamic-contact evaluation has duplicate wall coefficients");
        }
    }

    FreeSurfaceDynamicContactState state;
    state.snapshot_revision_key =
        snapshot.revision().snapshot_revision_key;
    state.active_side = parameters.active_side;
    state.surface_tension = parameters.surface_tension;
    const auto wall_role =
        parameters.active_side == geometry::CutIntegrationSide::Negative
            ? LevelSetGeometryRuleRole::NegativeExteriorBoundary
            : LevelSetGeometryRuleRole::PositiveExteriorBoundary;
    for (const auto& record : snapshot.rules()) {
        if (!record.locally_owned ||
            record.retention != LevelSetGeometryRetention::Retained ||
            (record.role != LevelSetGeometryRuleRole::Contact &&
             record.role != wall_role)) {
            continue;
        }
        const auto found = walls.find(record.physical_boundary_marker);
        if (found == walls.end()) {
            continue;
        }
        auto& wall = found->second;
        if (record.reference_rule.points.size() !=
            record.physical_rule.points.size()) {
            throw std::invalid_argument(
                "free-surface dynamic-contact rule has inconsistent reference and physical point counts");
        }
        if (record.role == wall_role) {
            for (std::size_t point_index = 0;
                 point_index < record.physical_rule.points.size();
                 ++point_index) {
                const auto& physical =
                    record.physical_rule.points[point_index];
                const auto& reference =
                    record.reference_rule.points[point_index];
                const Real weight = physical.physical_weight;
                if (!std::isfinite(weight) || !(weight > Real{0.0}) ||
                    !finitePoint(physical.boundary_normal)) {
                    throw std::invalid_argument(
                        "free-surface dynamic-contact wall rule has invalid physical geometry");
                }
                auto wall_normal = physical.boundary_normal;
                const Real wall_normal_norm = norm(wall_normal);
                if (!(wall_normal_norm > Real{0.0}) ||
                    !std::isfinite(wall_normal_norm)) {
                    throw std::invalid_argument(
                        "free-surface dynamic-contact wall rule has a degenerate normal");
                }
                for (auto& component : wall_normal) {
                    component /= wall_normal_norm;
                }
                const auto value = velocity.value(
                    record.reference_rule.provenance.parent_entity,
                    reference.parent_coordinate,
                    record.reference_rule.provenance);
                if (!finitePoint(value)) {
                    throw std::invalid_argument(
                        "free-surface dynamic-contact wall velocity evaluator returned a non-finite value");
                }
                std::array<Real, 3> tangential_velocity{};
                const Real normal_velocity = dot(value, wall_normal);
                Real tangential_speed_squared = Real{0.0};
                for (std::size_t component = 0; component < 3u;
                     ++component) {
                    tangential_velocity[component] =
                        value[component] -
                        normal_velocity * wall_normal[component];
                    tangential_speed_squared +=
                        tangential_velocity[component] *
                        tangential_velocity[component];
                }
                if (!std::isfinite(tangential_speed_squared) ||
                    tangential_speed_squared < Real{0.0}) {
                    throw std::invalid_argument(
                        "free-surface dynamic-contact wall slip is non-finite");
                }
                const Real tangential_speed =
                    std::sqrt(tangential_speed_squared);
                ++wall.owned_wetted_wall_quadrature_point_count;
                wall.owned_wetted_wall_measure += weight;
                wall.wall_slip_speed_integral +=
                    tangential_speed * weight;
                wall.wall_slip_speed_squared_integral +=
                    tangential_speed_squared * weight;
                wall.wall_slip_dissipation +=
                    wall.dynamic_viscosity / wall.slip_length *
                    tangential_speed_squared * weight;
                for (std::size_t component = 0; component < 3u;
                     ++component) {
                    wall.wall_tangential_velocity_integral[component] +=
                        tangential_velocity[component] * weight;
                }
            }
            continue;
        }
        const Real line_friction =
            wall.law == FreeSurfaceContactLaw::DynamicRenE
                ? Real{1.0} / wall.mobility
                : Real{0.0};
        const Real equilibrium_cosine =
            std::cos(wall.equilibrium_contact_angle_radians);
        for (std::size_t point_index = 0;
             point_index < record.physical_rule.points.size();
             ++point_index) {
            const auto& physical =
                record.physical_rule.points[point_index];
            const auto& reference =
                record.reference_rule.points[point_index];
            const Real weight = physical.physical_weight;
            if (!std::isfinite(weight) || !(weight > Real{0.0}) ||
                !finitePoint(physical.normal) ||
                !finitePoint(physical.boundary_normal)) {
                throw std::invalid_argument(
                    "free-surface dynamic-contact rule has invalid physical geometry");
            }
            auto liquid_normal = physical.normal;
            if (parameters.active_side ==
                geometry::CutIntegrationSide::Positive) {
                for (auto& component : liquid_normal) {
                    component = -component;
                }
            }
            const auto wall_normal = physical.boundary_normal;
            const Real liquid_normal_norm = norm(liquid_normal);
            const Real wall_normal_norm = norm(wall_normal);
            if (!(liquid_normal_norm > Real{0.0}) ||
                !(wall_normal_norm > Real{0.0}) ||
                !std::isfinite(liquid_normal_norm) ||
                !std::isfinite(wall_normal_norm)) {
                throw std::invalid_argument(
                    "free-surface dynamic-contact rule has a degenerate normal");
            }
            for (auto& component : liquid_normal) {
                component /= liquid_normal_norm;
            }
            std::array<Real, 3> unit_wall_normal = wall_normal;
            for (auto& component : unit_wall_normal) {
                component /= wall_normal_norm;
            }
            const Real normal_dot =
                std::clamp(dot(liquid_normal, unit_wall_normal),
                           Real{-1.0}, Real{1.0});
            const Real dynamic_cosine = -normal_dot;
            const Real dynamic_angle =
                std::acos(std::clamp(dynamic_cosine,
                                     Real{-1.0}, Real{1.0}));
            std::array<Real, 3> footprint_direction{};
            for (std::size_t component = 0; component < 3u; ++component) {
                footprint_direction[component] =
                    liquid_normal[component] -
                    normal_dot * unit_wall_normal[component];
            }
            const Real footprint_norm = norm(footprint_direction);
            if (!(footprint_norm > Real{0.0}) ||
                !std::isfinite(footprint_norm)) {
                throw std::invalid_argument(
                    "free-surface dynamic-contact rule has no wall-tangential footprint direction");
            }
            for (auto& component : footprint_direction) {
                component /= footprint_norm;
            }
            auto contact_line_tangent =
                crossPoint(unit_wall_normal, footprint_direction);
            const Real tangent_norm = norm(contact_line_tangent);
            if (!(tangent_norm > Real{0.0}) ||
                !std::isfinite(tangent_norm)) {
                throw std::invalid_argument(
                    "free-surface dynamic-contact rule has no oriented contact-line tangent");
            }
            for (auto& component : contact_line_tangent) {
                component /= tangent_norm;
            }
            const auto value = velocity.value(
                record.reference_rule.provenance.parent_entity,
                reference.parent_coordinate,
                record.reference_rule.provenance);
            if (!finitePoint(value)) {
                throw std::invalid_argument(
                    "free-surface dynamic-contact velocity evaluator returned a non-finite value");
            }
            const Real contact_speed = dot(value, footprint_direction);
            const Real constitutive_residual =
                line_friction * contact_speed -
                parameters.surface_tension *
                    (equilibrium_cosine - dynamic_cosine);

            ++wall.owned_quadrature_point_count;
            if (contact_speed > Real{0.0}) {
                ++wall.owned_advancing_point_count;
            } else if (contact_speed < Real{0.0}) {
                ++wall.owned_receding_point_count;
            } else {
                ++wall.owned_stationary_point_count;
            }
            wall.owned_contact_measure += weight;
            wall.dynamic_angle_integral += dynamic_angle * weight;
            wall.dynamic_cosine_integral += dynamic_cosine * weight;
            wall.contact_speed_integral += contact_speed * weight;
            wall.contact_speed_squared_integral +=
                contact_speed * contact_speed * weight;
            wall.constitutive_residual_integral +=
                constitutive_residual * weight;
            wall.absolute_constitutive_residual_integral +=
                std::abs(constitutive_residual) * weight;
            wall.line_friction_dissipation +=
                line_friction * contact_speed * contact_speed * weight;
            for (std::size_t component = 0; component < 3u; ++component) {
                wall.contact_position_integral[component] +=
                    physical.physical_point[component] * weight;
                wall.wall_normal_integral[component] +=
                    unit_wall_normal[component] * weight;
                wall.footprint_direction_integral[component] +=
                    footprint_direction[component] * weight;
                wall.contact_line_tangent_integral[component] +=
                    contact_line_tangent[component] * weight;
            }
        }
    }
    state.walls.reserve(walls.size());
    for (auto& [marker, wall] : walls) {
        (void)marker;
        state.walls.push_back(std::move(wall));
    }
    finalizeFreeSurfaceDynamicContactState(state);
    return state;
}

} // namespace svmp::Physics::formulations::navier_stokes::free_surface

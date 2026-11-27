/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#pragma once

#include "FE/Interfaces/LevelSetGeometrySnapshot.h"

#include <array>
#include <cstdint>
#include <optional>
#include <vector>

namespace svmp::Physics::formulations::navier_stokes::free_surface {

/**
 * Coefficients for the snapshot-owned capillary functional
 *
 *   F_h = gamma A_lg,h
 *         - sum_w gamma cos(theta_e,w) A_sl,h,w + lambda V_h.
 *
 * A boundary has no Young contribution when it has no coefficient entry.
 * The volume multiplier uses the displayed plus-sign convention; callers
 * are responsible for mapping their pressure sign convention to lambda.
 */
struct FreeSurfaceYoungWallCoefficient {
    int boundary_marker{-1};
    FE::Real equilibrium_contact_angle_radians{0.0};
};

enum class FreeSurfaceContactLaw : std::uint8_t {
    PrescribedAngle,
    DynamicRenE
};

struct FreeSurfaceDynamicContactCoefficient {
    int boundary_marker{-1};
    FE::Real equilibrium_contact_angle_radians{0.0};
    FreeSurfaceContactLaw law{FreeSurfaceContactLaw::DynamicRenE};
    FE::Real mobility{0.0};
    FE::Real slip_length{0.0};
    FE::Real dynamic_viscosity{0.0};
};

struct FreeSurfaceDiscreteFunctionalParameters {
    FE::geometry::CutIntegrationSide active_side{
        FE::geometry::CutIntegrationSide::Negative};
    FE::Real surface_tension{0.0};
    std::vector<FreeSurfaceYoungWallCoefficient> young_wall_coefficients{};
    std::vector<FreeSurfaceDynamicContactCoefficient>
        dynamic_contact_coefficients{};
    FE::Real volume_multiplier{0.0};
};

struct FreeSurfaceDiscreteWallFunctionalState {
    int boundary_marker{-1};
    std::optional<FE::Real> equilibrium_contact_angle_radians{};
    FE::Real owned_wetted_wall_area{0.0};
    FE::Real owned_contact_measure{0.0};
    FE::Real young_wall_energy{0.0};
};

/**
 * The capillary functional of one immutable snapshot, as an accepted record
 * holds it.
 *
 * freeSurfaceDiscreteFunctionalLayout gives its walls and parameters; the
 * values are the measured 0-forms the formulation declares (D49 steps E3 and
 * E4), which the recorder evaluates (LevelSet/AcceptedFunctionalRecorder).
 */
struct FreeSurfaceDiscreteFunctionalState {
    std::uint64_t snapshot_revision_key{0};
    FE::geometry::CutIntegrationSide active_side{
        FE::geometry::CutIntegrationSide::Negative};
    FE::Real surface_tension{0.0};
    FE::Real volume_multiplier{0.0};
    std::vector<FreeSurfaceDiscreteWallFunctionalState> walls{};
    FE::Real owned_liquid_volume{0.0};
    FE::Real owned_liquid_gas_area{0.0};
    FE::Real owned_wetted_wall_area{0.0};
    FE::Real owned_contact_measure{0.0};
    FE::Real liquid_gas_surface_energy{0.0};
    FE::Real young_wall_energy{0.0};
    FE::Real volume_constraint_potential{0.0};
    FE::Real total_potential{0.0};
};

struct FreeSurfaceDiscreteWallFunctionalVariationState {
    int boundary_marker{-1};
    std::optional<FE::Real> equilibrium_contact_angle_radians{};
    FE::Real owned_wetted_wall_area_variation{0.0};
    FE::Real young_wall_energy_variation{0.0};
};

/**
 * First variation of the immutable snapshot's capillary functional under a
 * deformation (the endpoint functional power of a record).
 *
 * Every rate is a declared measured 0-form: the liquid-gas area and liquid
 * volume rates (D49 step E3) and the wetted-wall and Young-energy rates of
 * each wall, the formulation's wall functionals over the wall's contact lines
 * with the rule's wall normal.  freeSurfaceDiscreteFunctionalVariationLayout
 * gives the record its walls.
 */
struct FreeSurfaceDiscreteFunctionalVariationState {
    std::uint64_t snapshot_revision_key{0};
    FE::geometry::CutIntegrationSide active_side{
        FE::geometry::CutIntegrationSide::Negative};
    FE::Real surface_tension{0.0};
    FE::Real volume_multiplier{0.0};
    std::vector<FreeSurfaceDiscreteWallFunctionalVariationState> walls{};
    FE::Real owned_liquid_volume_variation{0.0};
    FE::Real owned_liquid_gas_area_variation{0.0};
    FE::Real owned_wetted_wall_area_variation{0.0};
    FE::Real liquid_gas_surface_energy_variation{0.0};
    FE::Real young_wall_energy_variation{0.0};
    FE::Real volume_constraint_potential_variation{0.0};
    FE::Real total_potential_variation{0.0};
};

enum class FreeSurfaceContactMotion : std::uint8_t {
    Absent,
    Stationary,
    Advancing,
    Receding,
    Mixed
};

/**
 * Rank-owned contact-law sample on one wall and one geometry snapshot.
 *
 * The integral fields are additive across ranks.  The mean fields are derived
 * from those integrals and are recomputed after communicator reduction.  The
 * mean frame vectors retain their magnitude: a value below one records frame
 * variation instead of disguising it through renormalization.  Prescribed
 * angle records set mobility and line-friction dissipation to zero while
 * retaining sharp wetted-wall Navier dissipation and geometric angle error.
 */
struct FreeSurfaceDynamicContactWallState {
    int boundary_marker{-1};
    FE::Real equilibrium_contact_angle_radians{0.0};
    FreeSurfaceContactLaw law{FreeSurfaceContactLaw::DynamicRenE};
    FE::Real mobility{0.0};
    FE::Real slip_length{0.0};
    FE::Real dynamic_viscosity{0.0};
    std::size_t owned_quadrature_point_count{0u};
    std::size_t owned_advancing_point_count{0u};
    std::size_t owned_receding_point_count{0u};
    std::size_t owned_stationary_point_count{0u};
    FE::Real owned_contact_measure{0.0};
    FE::Real dynamic_angle_integral{0.0};
    FE::Real dynamic_cosine_integral{0.0};
    FE::Real contact_speed_integral{0.0};
    FE::Real contact_speed_squared_integral{0.0};
    FE::Real constitutive_residual_integral{0.0};
    FE::Real absolute_constitutive_residual_integral{0.0};
    FE::Real line_friction_dissipation{0.0};
    std::size_t owned_wetted_wall_quadrature_point_count{0u};
    FE::Real owned_wetted_wall_measure{0.0};
    FE::Real wall_slip_speed_integral{0.0};
    FE::Real wall_slip_speed_squared_integral{0.0};
    FE::Real wall_slip_dissipation{0.0};
    std::array<FE::Real, 3> wall_tangential_velocity_integral{
        {0.0, 0.0, 0.0}};
    std::array<FE::Real, 3> contact_position_integral{{0.0, 0.0, 0.0}};
    std::array<FE::Real, 3> wall_normal_integral{{0.0, 0.0, 0.0}};
    std::array<FE::Real, 3> footprint_direction_integral{{0.0, 0.0, 0.0}};
    std::array<FE::Real, 3> contact_line_tangent_integral{{0.0, 0.0, 0.0}};
    std::optional<FE::Real> mean_dynamic_angle_radians{};
    std::optional<FE::Real> mean_dynamic_cosine{};
    std::optional<FE::Real> mean_contact_speed{};
    std::optional<FE::Real> mean_constitutive_residual{};
    std::optional<FE::Real> mean_absolute_constitutive_residual{};
    std::optional<FE::Real> mean_wall_slip_speed{};
    std::array<FE::Real, 3> mean_wall_tangential_velocity{
        {0.0, 0.0, 0.0}};
    std::array<FE::Real, 3> mean_contact_position{{0.0, 0.0, 0.0}};
    std::array<FE::Real, 3> mean_wall_normal{{0.0, 0.0, 0.0}};
    std::array<FE::Real, 3> mean_footprint_direction{{0.0, 0.0, 0.0}};
    std::array<FE::Real, 3> mean_contact_line_tangent{{0.0, 0.0, 0.0}};
    FreeSurfaceContactMotion motion{FreeSurfaceContactMotion::Absent};
};

struct FreeSurfaceDynamicContactState {
    std::uint64_t snapshot_revision_key{0};
    FE::geometry::CutIntegrationSide active_side{
        FE::geometry::CutIntegrationSide::Negative};
    FE::Real surface_tension{0.0};
    std::vector<FreeSurfaceDynamicContactWallState> walls{};
    FE::Real owned_contact_measure{0.0};
    FE::Real owned_wetted_wall_measure{0.0};
    FE::Real line_friction_dissipation{0.0};
    FE::Real wall_slip_dissipation{0.0};
    FE::Real total_dissipation{0.0};
};

struct FreeSurfaceActiveVolumeEnergyParameters {
    FE::geometry::CutIntegrationSide active_side{
        FE::geometry::CutIntegrationSide::Negative};
    FE::Real density{0.0};
    std::array<FE::Real, 3> gravitational_acceleration{{0.0, 0.0, 0.0}};
    std::array<FE::Real, 3> gravitational_reference_point{{0.0, 0.0, 0.0}};
};

/**
 * Kinetic and gravitational energy of the active liquid, as an accepted
 * record holds it; the values are declared measured 0-forms (D49 steps E3
 * and E4), the point count is the snapshot's (ownedSnapshotRulePointCount).
 *
 * The gravitational potential density is
 *
 *   -rho g dot (x - x_ref).
 *
 * Its endpoint material-domain variation rate under the supplied velocity is
 *
 *   -rho int g dot u.
 *
 * Ghost and pruned rules are excluded. Distributed callers reduce the owned
 * scalar fields exactly once on the snapshot communicator.
 */
struct FreeSurfaceActiveVolumeEnergyState {
    std::uint64_t snapshot_revision_key{0};
    FE::geometry::CutIntegrationSide active_side{
        FE::geometry::CutIntegrationSide::Negative};
    FE::Real density{0.0};
    std::array<FE::Real, 3> gravitational_acceleration{{0.0, 0.0, 0.0}};
    std::array<FE::Real, 3> gravitational_reference_point{{0.0, 0.0, 0.0}};
    std::size_t owned_quadrature_point_count{0u};
    FE::Real owned_liquid_volume{0.0};
    FE::Real kinetic_energy{0.0};
    FE::Real gravitational_energy{0.0};
    FE::Real gravitational_potential_power{0.0};
    FE::Real total_energy{0.0};
};

struct FreeSurfaceActiveVolumeDissipationParameters {
    FE::geometry::CutIntegrationSide active_side{
        FE::geometry::CutIntegrationSide::Negative};
    FE::Real dynamic_viscosity{0.0};
};

/**
 * Endpoint bulk-viscous dissipation of the active liquid, as an accepted
 * record holds it (a declared measured 0-form).
 *
 * The rate is the production Newtonian form tested by the endpoint velocity,
 *
 *   2 mu int sym(grad u) : sym(grad u).
 *
 * Ghost and pruned rules are excluded. Distributed callers reduce the owned
 * scalar fields exactly once on the snapshot communicator.
 */
struct FreeSurfaceActiveVolumeDissipationState {
    std::uint64_t snapshot_revision_key{0};
    FE::geometry::CutIntegrationSide active_side{
        FE::geometry::CutIntegrationSide::Negative};
    FE::Real dynamic_viscosity{0.0};
    std::size_t owned_quadrature_point_count{0u};
    FE::Real owned_liquid_volume{0.0};
    FE::Real bulk_viscous_dissipation_rate{0.0};
};

struct FreeSurfaceExternalPressurePowerParameters {
    FE::geometry::CutIntegrationSide active_side{
        FE::geometry::CutIntegrationSide::Negative};
    FE::Real external_pressure{0.0};
};

/**
 * Prescribed exterior-pressure power on the liquid-gas interface, as an
 * accepted record holds it (declared measured 0-forms).
 *
 * With the snapshot normal directed outward from the selected liquid, the
 * work rate added to the modeled liquid is
 *
 *   -p_external int u dot n_liquid.
 */
struct FreeSurfaceExternalPressurePowerState {
    std::uint64_t snapshot_revision_key{0};
    FE::geometry::CutIntegrationSide active_side{
        FE::geometry::CutIntegrationSide::Negative};
    FE::Real external_pressure{0.0};
    std::size_t owned_quadrature_point_count{0u};
    FE::Real owned_liquid_gas_area{0.0};
    FE::Real outward_liquid_volume_flux_rate{0.0};
    FE::Real external_pressure_power{0.0};
};

/**
 * Backward-Euler kinetic-work identity on the active liquid of the endpoint
 * snapshot, as an accepted record holds it: the declared kinetic energy
 * evaluated on the previous state and on the step change (D49 step E3).
 *
 * Both velocity endpoints are evaluated on that one domain. The identity is
 *
 *   rho int (u_after-u_before) dot u_after
 *     = K_after - K_before_on_after_domain
 *       + rho/2 int |u_after-u_before|^2.
 *
 * The final term is the nonnegative backward-Euler time-discretization loss.
 * Comparing K_before_on_after_domain with the stored energy on the preceding
 * snapshot exposes the separate domain-change coupling term.
 * The two velocity revisions bind the callback values to their declared
 * field slices without conflating unrelated level-set representation changes.
 */
struct FreeSurfaceBackwardEulerKineticWorkState {
    std::uint64_t snapshot_revision_key{0};
    std::uint64_t previous_velocity_revision{0};
    std::uint64_t endpoint_velocity_revision{0};
    FE::geometry::CutIntegrationSide active_side{
        FE::geometry::CutIntegrationSide::Negative};
    FE::Real density{0.0};
    std::size_t owned_quadrature_point_count{0u};
    FE::Real owned_liquid_volume{0.0};
    FE::Real kinetic_energy_before_on_endpoint_domain{0.0};
    FE::Real kinetic_energy_after{0.0};
    FE::Real kinetic_energy_change_on_endpoint_domain{0.0};
    FE::Real step_integrated_inertia_work{0.0};
    FE::Real time_discretization_loss{0.0};
    FE::Real identity_residual{0.0};
};

/** Recompute non-additive mean and motion fields after rank reduction. */
void finalizeFreeSurfaceDynamicContactState(
    FreeSurfaceDynamicContactState& state);

[[nodiscard]] FreeSurfaceDynamicContactState
evaluateFreeSurfaceDynamicContactState(
    const FE::interfaces::LevelSetGeometrySnapshot& snapshot,
    const FreeSurfaceDiscreteFunctionalParameters& parameters,
    const FE::interfaces::SnapshotVectorEvaluator& velocity);

/**
 * The walls and parameters of the snapshot's capillary functional, every
 * value zero: the walls with an equilibrium angle and every wall the
 * snapshot's domains or its retained, rank-owned rules touch, in marker
 * order.  Validates the parameters.  The values are the declared measured
 * 0-forms (D49 steps E3 and E4).
 */
[[nodiscard]] FreeSurfaceDiscreteFunctionalState
freeSurfaceDiscreteFunctionalLayout(
    const FE::interfaces::LevelSetGeometrySnapshot& snapshot,
    const FreeSurfaceDiscreteFunctionalParameters& parameters);

/**
 * The walls and parameters of the first variation of the snapshot's capillary
 * functional (the walls of freeSurfaceDiscreteFunctionalLayout), every rate
 * zero.  The rates are declared measured 0-forms (D49 steps E3 and E4).
 */
[[nodiscard]] FreeSurfaceDiscreteFunctionalVariationState
freeSurfaceDiscreteFunctionalVariationLayout(
    const FE::interfaces::LevelSetGeometrySnapshot& snapshot,
    const FreeSurfaceDiscreteFunctionalParameters& parameters);

} // namespace svmp::Physics::formulations::navier_stokes::free_surface

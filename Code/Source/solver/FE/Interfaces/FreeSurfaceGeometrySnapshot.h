/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_INTERFACES_FREESURFACEGEOMETRYSNAPSHOT_H
#define SVMP_FE_INTERFACES_FREESURFACEGEOMETRYSNAPSHOT_H

/**
 * @file FreeSurfaceGeometrySnapshot.h
 * @brief Immutable, revision-complete geometry state for one level-set surface.
 */

#include "Geometry/CutQuadratureMapping.h"
#include "Interfaces/GeneratedActiveBoundaryDomain.h"
#include "Interfaces/GeneratedInterfaceBoundaryIntersectionDomain.h"
#include "Interfaces/LevelSetInterfaceDomain.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

namespace svmp::FE {
namespace assembly {
class IMeshAccess;
}
namespace interfaces {

class FreeSurfaceGeometrySnapshot;

enum class FreeSurfaceGeometryRuleRole : std::uint8_t {
    NegativeVolume,
    PositiveVolume,
    Interface,
    NegativeExteriorBoundary,
    PositiveExteriorBoundary,
    Contact
};

enum class FreeSurfaceGeometryRetention : std::uint8_t {
    Retained,
    PrunedSmallVolume
};

enum class FreeSurfaceGeometryMomentCertificateSource : std::uint8_t {
    ParentReferenceCell,
    RegionMeasureCentroid,
    PiecewiseAffineGeometry,
    BackendReferenceQuadrature,
    StoredGeneratedGeometry
};

struct FreeSurfaceGeometryMonomialMoment {
    std::array<int, 3> exponents{{0, 0, 0}};
    Real value{0.0};
};

struct FreeSurfaceGeometryMomentCertificate {
    int polynomial_order{-1};
    int ambient_dimension{0};
    FreeSurfaceGeometryMomentCertificateSource source{
        FreeSurfaceGeometryMomentCertificateSource::RegionMeasureCentroid};
    bool phase_sign_certified{false};
    std::vector<FreeSurfaceGeometryMonomialMoment> moments{};
};

struct FreeSurfaceGeometryRevision {
    std::string source_id{};
    std::string domain_id{};
    int interface_marker{-1};
    Real isovalue{0.0};
    std::uint64_t source_layout_revision{0};
    std::uint64_t source_value_revision{0};
    std::uint64_t mesh_geometry_revision{0};
    std::uint64_t mesh_topology_revision{0};
    std::uint64_t ownership_revision{0};
    std::uint64_t numbering_revision{0};
    std::uint64_t quadrature_policy_key{0};
    std::uint64_t snapshot_revision_key{0};

    [[nodiscard]] bool complete() const noexcept;
    [[nodiscard]] bool sameSourceState(
        const FreeSurfaceGeometryRevision& other) const noexcept;
};

/** Rank-local mesh epochs retained for live-consumer staleness checks. */
struct FreeSurfaceGeometryLocalMeshRevision {
    std::uint64_t mesh_geometry_revision{0};
    std::uint64_t mesh_topology_revision{0};
    std::uint64_t ownership_revision{0};
    std::uint64_t numbering_revision{0};
};

struct FreeSurfaceGeometryRuleRecord {
    FreeSurfaceGeometryRuleRole role{FreeSurfaceGeometryRuleRole::Interface};
    FreeSurfaceGeometryRetention retention{
        FreeSurfaceGeometryRetention::Retained};
    int physical_boundary_marker{-1};
    geometry::CutQuadratureRule reference_rule{};
    geometry::MappedCutQuadratureRule physical_rule{};
    bool locally_owned{false};
    std::vector<std::uint64_t> source_fragment_stable_ids{};
    std::string topology_id{};
    // Epoch-free semantic descriptor copied from the authoritative generated
    // source.  Unlike source_fragment_stable_ids/component_id, this key must
    // not incorporate a value revision or rank/local parent identity.  Zero
    // remains the explicit legacy/manual-source "unclassified" value; the
    // production active-cut publisher requires a nonzero key.
    std::uint64_t source_topology_key{0};
    std::int64_t component_id{-1};
    FreeSurfaceGeometryMomentCertificate moment_certificate{};
};

/** Epoch-free topology descriptor used for authoritative interface sources. */
[[nodiscard]] std::uint64_t freeSurfaceGeometrySourceTopologyKey(
    const CutInterfaceFragment& fragment,
    ElementType parent_element_type,
    Real tolerance);

struct FreeSurfaceGeometrySnapshotPolicy {
    Real tolerance{1.0e-12};
    Real minimum_retained_volume_fraction{1.0e-8};
    int minimum_achieved_quadrature_order{0};
    bool require_complete_exterior_boundary_partition{true};
};

struct FreeSurfaceGeometryScalarEvaluator {
    std::function<Real(GlobalIndex,
                       const std::array<Real, 3>&,
                       const geometry::CutQuadratureProvenance&)>
        value{};
    std::function<std::array<Real, 3>(GlobalIndex,
                                      const std::array<Real, 3>&,
                                      const geometry::CutQuadratureProvenance&)>
        reference_gradient{};

    [[nodiscard]] bool canEvaluateValue() const noexcept {
        return static_cast<bool>(value);
    }
};

/**
 * Rank-local adapter used by the snapshot validator to gather owned-rule
 * identities and revision components across the communicator.  The FE
 * geometry layer owns the identity, content, uniqueness, and canonicalization
 * checks; callers only provide the byte-free collective operation needed to
 * exchange unsigned integer values.
 */
struct FreeSurfaceGeometryOwnershipCollective {
    int rank{0};
    int size{1};
    std::function<std::vector<std::uint64_t>(
        std::span<const std::uint64_t>)>
        all_gather_owned_rule_identity_values{};
    std::function<std::vector<std::uint64_t>(
        std::span<const std::uint64_t>)>
        all_gather_revision_values{};
};

struct FreeSurfaceGeometryValidationLedger {
    std::size_t rule_count{0};
    std::size_t retained_rule_count{0};
    std::size_t pruned_rule_count{0};
    std::size_t quadrature_point_count{0};
    std::size_t owned_rule_count{0};
    std::size_t global_owned_rule_count{0};
    std::size_t contact_fragment_count{0};
    std::size_t referenced_surface_fragment_count{0};
    std::size_t orphan_contact_fragment_count{0};
    std::size_t missing_contact_fragment_count{0};
    std::size_t stale_revision_count{0};
    std::size_t invalid_phase_point_count{0};
    std::size_t represented_phase_point_count{0};
    std::size_t represented_phase_disagreement_count{0};
    std::size_t outside_parent_point_count{0};
    std::size_t invalid_weight_count{0};
    std::size_t false_achieved_order_count{0};
    std::size_t certified_rule_count{0};
    std::size_t parent_cell_moment_certificate_count{0};
    std::size_t centroid_moment_certificate_count{0};
    std::size_t piecewise_affine_moment_certificate_count{0};
    std::size_t backend_reference_moment_certificate_count{0};
    std::size_t stored_generated_moment_certificate_count{0};
    std::size_t validated_rule_polynomial_moment_count{0};
    std::size_t validated_polynomial_moment_count{0};
    std::size_t invalid_global_identity_count{0};
    std::size_t duplicate_rule_identity_count{0};
    Real unpruned_negative_reference_volume{0.0};
    Real unpruned_positive_reference_volume{0.0};
    Real unpruned_negative_physical_volume{0.0};
    Real unpruned_positive_physical_volume{0.0};
    Real owned_unpruned_negative_reference_volume{0.0};
    Real owned_unpruned_positive_reference_volume{0.0};
    Real owned_unpruned_negative_physical_volume{0.0};
    Real owned_unpruned_positive_physical_volume{0.0};
    Real retained_negative_reference_volume{0.0};
    Real retained_positive_reference_volume{0.0};
    Real retained_negative_physical_volume{0.0};
    Real retained_positive_physical_volume{0.0};
    Real owned_retained_negative_reference_volume{0.0};
    Real owned_retained_positive_reference_volume{0.0};
    Real owned_retained_negative_physical_volume{0.0};
    Real owned_retained_positive_physical_volume{0.0};
    Real interface_reference_measure{0.0};
    Real interface_physical_measure{0.0};
    Real contact_reference_measure{0.0};
    Real contact_physical_measure{0.0};
    Real maximum_root_residual{0.0};
    Real maximum_normal_angular_error{0.0};
    Real maximum_represented_phase_disagreement{0.0};
    Real maximum_constant_moment_error{0.0};
    Real maximum_polynomial_moment_error{0.0};
    Real maximum_polynomial_moment_scaled_error{0.0};
    Real maximum_volume_partition_error{0.0};
    Real maximum_boundary_partition_error{0.0};
};

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
    Real equilibrium_contact_angle_radians{0.0};
};

enum class FreeSurfaceContactLaw : std::uint8_t {
    PrescribedAngle,
    DynamicRenE
};

struct FreeSurfaceDynamicContactCoefficient {
    int boundary_marker{-1};
    Real equilibrium_contact_angle_radians{0.0};
    FreeSurfaceContactLaw law{FreeSurfaceContactLaw::DynamicRenE};
    Real mobility{0.0};
    Real slip_length{0.0};
    Real dynamic_viscosity{0.0};
};

struct FreeSurfaceDiscreteFunctionalParameters {
    geometry::CutIntegrationSide liquid_side{
        geometry::CutIntegrationSide::Negative};
    Real surface_tension{0.0};
    std::vector<FreeSurfaceYoungWallCoefficient> young_wall_coefficients{};
    std::vector<FreeSurfaceDynamicContactCoefficient>
        dynamic_contact_coefficients{};
    Real volume_multiplier{0.0};
};

struct FreeSurfaceDiscreteWallFunctionalState {
    int boundary_marker{-1};
    std::optional<Real> equilibrium_contact_angle_radians{};
    Real owned_wetted_wall_area{0.0};
    Real owned_contact_measure{0.0};
    Real young_wall_energy{0.0};
};

/**
 * Rank-owned contribution to one immutable snapshot's capillary functional.
 *
 * Ghost rules are deliberately excluded.  Distributed callers sum each
 * scalar across the snapshot communicator exactly once.
 */
struct FreeSurfaceDiscreteFunctionalState {
    std::uint64_t snapshot_revision_key{0};
    geometry::CutIntegrationSide liquid_side{
        geometry::CutIntegrationSide::Negative};
    Real surface_tension{0.0};
    Real volume_multiplier{0.0};
    std::vector<FreeSurfaceDiscreteWallFunctionalState> walls{};
    Real owned_liquid_volume{0.0};
    Real owned_liquid_gas_area{0.0};
    Real owned_wetted_wall_area{0.0};
    Real owned_contact_measure{0.0};
    Real liquid_gas_surface_energy{0.0};
    Real young_wall_energy{0.0};
    Real volume_constraint_potential{0.0};
    Real total_potential{0.0};
};

using FreeSurfaceDiscreteFunctionalPhysicalGradient =
    std::array<std::array<Real, 3>, 3>;

/**
 * Physical deformation used to differentiate the snapshot-owned functional.
 *
 * The callback coordinate is the parent reference coordinate of the
 * authoritative rule point.  `value` returns the physical deformation there,
 * while `physical_gradient[i][j]` is d(value_i)/d(x_j).  The deformation is
 * understood to preserve the fixed physical wall: its contact trace is wall
 * tangential and every non-contact exterior flux vanishes.
 */
struct FreeSurfaceDiscreteFunctionalDeformationEvaluator {
    std::function<std::array<Real, 3>(
        GlobalIndex,
        const std::array<Real, 3>&,
        const geometry::CutQuadratureProvenance&)>
        value{};
    std::function<FreeSurfaceDiscreteFunctionalPhysicalGradient(
        GlobalIndex,
        const std::array<Real, 3>&,
        const geometry::CutQuadratureProvenance&)>
        physical_gradient{};

    [[nodiscard]] bool canEvaluateValue() const noexcept {
        return static_cast<bool>(value);
    }
    [[nodiscard]] bool canEvaluatePhysicalGradient() const noexcept {
        return static_cast<bool>(physical_gradient);
    }
};

struct FreeSurfaceDiscreteWallFunctionalVariationState {
    int boundary_marker{-1};
    std::optional<Real> equilibrium_contact_angle_radians{};
    Real owned_wetted_wall_area_variation{0.0};
    Real young_wall_energy_variation{0.0};
};

/**
 * Rank-owned first variation of the immutable snapshot's capillary functional.
 *
 * The liquid--gas area uses the surface-divergence identity on retained
 * interface rules.  Wetted-wall area uses the outward wetted-footprint flux
 * on retained contact rules, and liquid volume uses the outward-liquid flux
 * on retained interface rules.  Ghost rules are deliberately excluded.
 */
struct FreeSurfaceDiscreteFunctionalVariationState {
    std::uint64_t snapshot_revision_key{0};
    geometry::CutIntegrationSide liquid_side{
        geometry::CutIntegrationSide::Negative};
    Real surface_tension{0.0};
    Real volume_multiplier{0.0};
    std::vector<FreeSurfaceDiscreteWallFunctionalVariationState> walls{};
    Real owned_liquid_volume_variation{0.0};
    Real owned_liquid_gas_area_variation{0.0};
    Real owned_wetted_wall_area_variation{0.0};
    Real liquid_gas_surface_energy_variation{0.0};
    Real young_wall_energy_variation{0.0};
    Real volume_constraint_potential_variation{0.0};
    Real total_potential_variation{0.0};
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
    Real equilibrium_contact_angle_radians{0.0};
    FreeSurfaceContactLaw law{FreeSurfaceContactLaw::DynamicRenE};
    Real mobility{0.0};
    Real slip_length{0.0};
    Real dynamic_viscosity{0.0};
    std::size_t owned_quadrature_point_count{0u};
    std::size_t owned_advancing_point_count{0u};
    std::size_t owned_receding_point_count{0u};
    std::size_t owned_stationary_point_count{0u};
    Real owned_contact_measure{0.0};
    Real dynamic_angle_integral{0.0};
    Real dynamic_cosine_integral{0.0};
    Real contact_speed_integral{0.0};
    Real contact_speed_squared_integral{0.0};
    Real constitutive_residual_integral{0.0};
    Real absolute_constitutive_residual_integral{0.0};
    Real line_friction_dissipation{0.0};
    std::size_t owned_wetted_wall_quadrature_point_count{0u};
    Real owned_wetted_wall_measure{0.0};
    Real wall_slip_speed_integral{0.0};
    Real wall_slip_speed_squared_integral{0.0};
    Real wall_slip_dissipation{0.0};
    std::array<Real, 3> wall_tangential_velocity_integral{
        {0.0, 0.0, 0.0}};
    std::array<Real, 3> contact_position_integral{{0.0, 0.0, 0.0}};
    std::array<Real, 3> wall_normal_integral{{0.0, 0.0, 0.0}};
    std::array<Real, 3> footprint_direction_integral{{0.0, 0.0, 0.0}};
    std::array<Real, 3> contact_line_tangent_integral{{0.0, 0.0, 0.0}};
    std::optional<Real> mean_dynamic_angle_radians{};
    std::optional<Real> mean_dynamic_cosine{};
    std::optional<Real> mean_contact_speed{};
    std::optional<Real> mean_constitutive_residual{};
    std::optional<Real> mean_absolute_constitutive_residual{};
    std::optional<Real> mean_wall_slip_speed{};
    std::array<Real, 3> mean_wall_tangential_velocity{
        {0.0, 0.0, 0.0}};
    std::array<Real, 3> mean_contact_position{{0.0, 0.0, 0.0}};
    std::array<Real, 3> mean_wall_normal{{0.0, 0.0, 0.0}};
    std::array<Real, 3> mean_footprint_direction{{0.0, 0.0, 0.0}};
    std::array<Real, 3> mean_contact_line_tangent{{0.0, 0.0, 0.0}};
    FreeSurfaceContactMotion motion{FreeSurfaceContactMotion::Absent};
};

struct FreeSurfaceDynamicContactState {
    std::uint64_t snapshot_revision_key{0};
    geometry::CutIntegrationSide liquid_side{
        geometry::CutIntegrationSide::Negative};
    Real surface_tension{0.0};
    std::vector<FreeSurfaceDynamicContactWallState> walls{};
    Real owned_contact_measure{0.0};
    Real owned_wetted_wall_measure{0.0};
    Real line_friction_dissipation{0.0};
    Real wall_slip_dissipation{0.0};
    Real total_dissipation{0.0};
};

struct FreeSurfaceDiscreteFunctionalVectorEvaluator {
    std::function<std::array<Real, 3>(
        GlobalIndex,
        const std::array<Real, 3>&,
        const geometry::CutQuadratureProvenance&)>
        value{};
    std::function<FreeSurfaceDiscreteFunctionalPhysicalGradient(
        GlobalIndex,
        const std::array<Real, 3>&,
        const geometry::CutQuadratureProvenance&)>
        physical_gradient{};

    [[nodiscard]] bool canEvaluateValue() const noexcept {
        return static_cast<bool>(value);
    }
    [[nodiscard]] bool canEvaluatePhysicalGradient() const noexcept {
        return static_cast<bool>(physical_gradient);
    }
};

struct FreeSurfaceActiveVolumeEnergyParameters {
    geometry::CutIntegrationSide liquid_side{
        geometry::CutIntegrationSide::Negative};
    Real density{0.0};
    std::array<Real, 3> gravitational_acceleration{{0.0, 0.0, 0.0}};
    std::array<Real, 3> gravitational_reference_point{{0.0, 0.0, 0.0}};
};

/**
 * Rank-owned kinetic and gravitational energy on one snapshot's retained
 * active-liquid volume rules.
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
    geometry::CutIntegrationSide liquid_side{
        geometry::CutIntegrationSide::Negative};
    Real density{0.0};
    std::array<Real, 3> gravitational_acceleration{{0.0, 0.0, 0.0}};
    std::array<Real, 3> gravitational_reference_point{{0.0, 0.0, 0.0}};
    std::size_t owned_quadrature_point_count{0u};
    Real owned_liquid_volume{0.0};
    Real kinetic_energy{0.0};
    Real gravitational_energy{0.0};
    Real gravitational_potential_power{0.0};
    Real total_energy{0.0};
};

[[nodiscard]] FreeSurfaceActiveVolumeEnergyState
evaluateFreeSurfaceActiveVolumeEnergy(
    const FreeSurfaceGeometrySnapshot& snapshot,
    const FreeSurfaceActiveVolumeEnergyParameters& parameters,
    const FreeSurfaceDiscreteFunctionalVectorEvaluator& velocity);

struct FreeSurfaceActiveVolumeDissipationParameters {
    geometry::CutIntegrationSide liquid_side{
        geometry::CutIntegrationSide::Negative};
    Real dynamic_viscosity{0.0};
};

/**
 * Rank-owned endpoint bulk-viscous dissipation on the retained active-liquid
 * volume rules.
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
    geometry::CutIntegrationSide liquid_side{
        geometry::CutIntegrationSide::Negative};
    Real dynamic_viscosity{0.0};
    std::size_t owned_quadrature_point_count{0u};
    Real owned_liquid_volume{0.0};
    Real bulk_viscous_dissipation_rate{0.0};
};

[[nodiscard]] FreeSurfaceActiveVolumeDissipationState
evaluateFreeSurfaceActiveVolumeDissipation(
    const FreeSurfaceGeometrySnapshot& snapshot,
    const FreeSurfaceActiveVolumeDissipationParameters& parameters,
    const FreeSurfaceDiscreteFunctionalVectorEvaluator& velocity);

struct FreeSurfaceExternalPressurePowerParameters {
    geometry::CutIntegrationSide liquid_side{
        geometry::CutIntegrationSide::Negative};
    Real external_pressure{0.0};
};

/**
 * Rank-owned prescribed exterior-pressure power on the retained liquid--gas
 * interface rules.
 *
 * With the snapshot normal directed outward from the selected liquid, the
 * work rate added to the modeled liquid is
 *
 *   -p_external int u dot n_liquid.
 */
struct FreeSurfaceExternalPressurePowerState {
    std::uint64_t snapshot_revision_key{0};
    geometry::CutIntegrationSide liquid_side{
        geometry::CutIntegrationSide::Negative};
    Real external_pressure{0.0};
    std::size_t owned_quadrature_point_count{0u};
    Real owned_liquid_gas_area{0.0};
    Real outward_liquid_volume_flux_rate{0.0};
    Real external_pressure_power{0.0};
};

[[nodiscard]] FreeSurfaceExternalPressurePowerState
evaluateFreeSurfaceExternalPressurePower(
    const FreeSurfaceGeometrySnapshot& snapshot,
    const FreeSurfaceExternalPressurePowerParameters& parameters,
    const FreeSurfaceDiscreteFunctionalVectorEvaluator& velocity);

/**
 * Rank-owned backward-Euler kinetic-work identity evaluated on the retained
 * active-liquid volume rules of the endpoint snapshot.
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
    geometry::CutIntegrationSide liquid_side{
        geometry::CutIntegrationSide::Negative};
    Real density{0.0};
    std::size_t owned_quadrature_point_count{0u};
    Real owned_liquid_volume{0.0};
    Real kinetic_energy_before_on_endpoint_domain{0.0};
    Real kinetic_energy_after{0.0};
    Real kinetic_energy_change_on_endpoint_domain{0.0};
    Real step_integrated_inertia_work{0.0};
    Real time_discretization_loss{0.0};
    Real identity_residual{0.0};
};

[[nodiscard]] FreeSurfaceBackwardEulerKineticWorkState
evaluateFreeSurfaceBackwardEulerKineticWork(
    const FreeSurfaceGeometrySnapshot& endpoint_snapshot,
    geometry::CutIntegrationSide liquid_side,
    Real density,
    std::uint64_t previous_velocity_revision,
    std::uint64_t endpoint_velocity_revision,
    const FreeSurfaceDiscreteFunctionalVectorEvaluator& previous_velocity,
    const FreeSurfaceDiscreteFunctionalVectorEvaluator& endpoint_velocity);

/** Recompute non-additive mean and motion fields after rank reduction. */
void finalizeFreeSurfaceDynamicContactState(
    FreeSurfaceDynamicContactState& state);

[[nodiscard]] FreeSurfaceDynamicContactState
evaluateFreeSurfaceDynamicContactState(
    const FreeSurfaceGeometrySnapshot& snapshot,
    const FreeSurfaceDiscreteFunctionalParameters& parameters,
    const FreeSurfaceDiscreteFunctionalVectorEvaluator& velocity);

class FreeSurfaceGeometrySnapshot {
public:
    FreeSurfaceGeometrySnapshot(const FreeSurfaceGeometrySnapshot&) = delete;
    FreeSurfaceGeometrySnapshot& operator=(const FreeSurfaceGeometrySnapshot&) =
        delete;

    [[nodiscard]] const FreeSurfaceGeometryRevision& revision() const noexcept;
    [[nodiscard]] const FreeSurfaceGeometryLocalMeshRevision&
    localMeshRevision() const noexcept;
    [[nodiscard]] const FreeSurfaceGeometrySnapshotPolicy& policy() const noexcept;
    [[nodiscard]] const LevelSetInterfaceDomain& interfaceDomain() const noexcept;
    [[nodiscard]] const std::vector<GeneratedInterfaceBoundaryIntersectionDomain>&
    contactDomains() const noexcept;
    [[nodiscard]] const std::vector<GeneratedActiveBoundaryDomain>&
    activeBoundaryDomains() const noexcept;
    [[nodiscard]] const std::vector<FreeSurfaceGeometryRuleRecord>&
    rules() const noexcept;
    [[nodiscard]] const FreeSurfaceGeometryValidationLedger& ledger() const noexcept;

    [[nodiscard]] std::vector<const FreeSurfaceGeometryRuleRecord*>
    retainedRules(FreeSurfaceGeometryRuleRole role) const;
    [[nodiscard]] std::size_t residentBytes() const noexcept;

private:
    friend std::shared_ptr<const FreeSurfaceGeometrySnapshot>
    buildFreeSurfaceGeometrySnapshot(
        LevelSetInterfaceDomain,
        std::vector<GeneratedInterfaceBoundaryIntersectionDomain>,
        std::vector<GeneratedActiveBoundaryDomain>,
        const assembly::IMeshAccess&,
        FreeSurfaceGeometrySnapshotPolicy,
        FreeSurfaceGeometryScalarEvaluator,
        std::string,
        FreeSurfaceGeometryOwnershipCollective);

    FreeSurfaceGeometrySnapshot(
        FreeSurfaceGeometryRevision revision,
        FreeSurfaceGeometryLocalMeshRevision local_mesh_revision,
        FreeSurfaceGeometrySnapshotPolicy policy,
        LevelSetInterfaceDomain interface_domain,
        std::vector<GeneratedInterfaceBoundaryIntersectionDomain> contact_domains,
        std::vector<GeneratedActiveBoundaryDomain> active_boundary_domains,
        std::vector<FreeSurfaceGeometryRuleRecord> rules,
        FreeSurfaceGeometryValidationLedger ledger);

    FreeSurfaceGeometryRevision revision_{};
    FreeSurfaceGeometryLocalMeshRevision local_mesh_revision_{};
    FreeSurfaceGeometrySnapshotPolicy policy_{};
    LevelSetInterfaceDomain interface_domain_{};
    std::vector<GeneratedInterfaceBoundaryIntersectionDomain> contact_domains_{};
    std::vector<GeneratedActiveBoundaryDomain> active_boundary_domains_{};
    std::vector<FreeSurfaceGeometryRuleRecord> rules_{};
    FreeSurfaceGeometryValidationLedger ledger_{};
};

[[nodiscard]] FreeSurfaceDiscreteFunctionalState
evaluateFreeSurfaceDiscreteFunctional(
    const FreeSurfaceGeometrySnapshot& snapshot,
    const FreeSurfaceDiscreteFunctionalParameters& parameters);

[[nodiscard]] FreeSurfaceDiscreteFunctionalVariationState
evaluateFreeSurfaceDiscreteFunctionalFirstVariation(
    const FreeSurfaceGeometrySnapshot& snapshot,
    const FreeSurfaceDiscreteFunctionalParameters& parameters,
    const FreeSurfaceDiscreteFunctionalDeformationEvaluator& deformation);

[[nodiscard]] std::shared_ptr<const FreeSurfaceGeometrySnapshot>
buildFreeSurfaceGeometrySnapshot(
    LevelSetInterfaceDomain interface_domain,
    std::vector<GeneratedInterfaceBoundaryIntersectionDomain> contact_domains,
    std::vector<GeneratedActiveBoundaryDomain> active_boundary_domains,
    const assembly::IMeshAccess& mesh,
    FreeSurfaceGeometrySnapshotPolicy policy = {},
    FreeSurfaceGeometryScalarEvaluator scalar = {},
    std::string domain_id = {},
    FreeSurfaceGeometryOwnershipCollective ownership_collective = {});

struct FreeSurfaceGeometrySnapshotCacheStatistics {
    std::size_t live_snapshot_count{0};
    std::size_t live_resident_bytes{0};
    std::size_t peak_live_snapshot_count{0};
    std::size_t peak_live_resident_bytes{0};
    std::size_t hit_count{0};
    std::size_t miss_count{0};
    std::size_t expired_eviction_count{0};
};

class FreeSurfaceGeometrySnapshotCache {
public:
    [[nodiscard]] std::shared_ptr<const FreeSurfaceGeometrySnapshot> find(
        std::uint64_t revision_key);
    void insert(std::shared_ptr<const FreeSurfaceGeometrySnapshot> snapshot);
    void evictExpired();
    [[nodiscard]] FreeSurfaceGeometrySnapshotCacheStatistics statistics();

private:
    std::unordered_map<std::uint64_t,
                       std::weak_ptr<const FreeSurfaceGeometrySnapshot>>
        snapshots_{};
    FreeSurfaceGeometrySnapshotCacheStatistics statistics_{};
};

} // namespace interfaces
} // namespace svmp::FE

#endif // SVMP_FE_INTERFACES_FREESURFACEGEOMETRYSNAPSHOT_H

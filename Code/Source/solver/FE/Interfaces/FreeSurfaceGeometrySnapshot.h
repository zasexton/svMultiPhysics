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
    std::int64_t component_id{-1};
    FreeSurfaceGeometryMomentCertificate moment_certificate{};
};

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
    std::size_t outside_parent_point_count{0};
    std::size_t invalid_weight_count{0};
    std::size_t false_achieved_order_count{0};
    std::size_t certified_rule_count{0};
    std::size_t parent_cell_moment_certificate_count{0};
    std::size_t centroid_moment_certificate_count{0};
    std::size_t piecewise_affine_moment_certificate_count{0};
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

struct FreeSurfaceDynamicContactCoefficient {
    int boundary_marker{-1};
    Real equilibrium_contact_angle_radians{0.0};
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

enum class FreeSurfaceContactMotion : std::uint8_t {
    Absent,
    Stationary,
    Advancing,
    Receding,
    Mixed
};

/**
 * Rank-owned Ren--E contact-law sample on one wall and one geometry snapshot.
 *
 * The integral fields are additive across ranks.  The mean fields are derived
 * from those integrals and are recomputed after communicator reduction.  The
 * mean frame vectors retain their magnitude: a value below one records frame
 * variation instead of disguising it through renormalization.
 */
struct FreeSurfaceDynamicContactWallState {
    int boundary_marker{-1};
    Real equilibrium_contact_angle_radians{0.0};
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

    [[nodiscard]] bool canEvaluateValue() const noexcept {
        return static_cast<bool>(value);
    }
};

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
        FreeSurfaceGeometrySnapshotPolicy policy,
        LevelSetInterfaceDomain interface_domain,
        std::vector<GeneratedInterfaceBoundaryIntersectionDomain> contact_domains,
        std::vector<GeneratedActiveBoundaryDomain> active_boundary_domains,
        std::vector<FreeSurfaceGeometryRuleRecord> rules,
        FreeSurfaceGeometryValidationLedger ledger);

    FreeSurfaceGeometryRevision revision_{};
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

/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_INTERFACES_LEVELSETGEOMETRYSNAPSHOT_H
#define SVMP_FE_INTERFACES_LEVELSETGEOMETRYSNAPSHOT_H

/**
 * @file LevelSetGeometrySnapshot.h
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

class LevelSetGeometrySnapshot;

enum class LevelSetGeometryRuleRole : std::uint8_t {
    NegativeVolume,
    PositiveVolume,
    Interface,
    NegativeExteriorBoundary,
    PositiveExteriorBoundary,
    Contact
};

enum class LevelSetGeometryRetention : std::uint8_t {
    Retained,
    PrunedSmallVolume
};

enum class LevelSetGeometryMomentCertificateSource : std::uint8_t {
    ParentReferenceCell,
    RegionMeasureCentroid,
    PiecewiseAffineGeometry,
    BackendReferenceQuadrature,
    StoredGeneratedGeometry
};

struct LevelSetGeometryMonomialMoment {
    std::array<int, 3> exponents{{0, 0, 0}};
    Real value{0.0};
};

struct LevelSetGeometryMomentCertificate {
    int polynomial_order{-1};
    int ambient_dimension{0};
    LevelSetGeometryMomentCertificateSource source{
        LevelSetGeometryMomentCertificateSource::RegionMeasureCentroid};
    bool phase_sign_certified{false};
    std::vector<LevelSetGeometryMonomialMoment> moments{};
};

struct LevelSetGeometryRevision {
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
    LevelSetCoefficientClassificationPolicy coefficient_classification_policy{
        LevelSetCoefficientClassificationPolicy::LegacyAbsoluteBand};
    Real coefficient_classification_band{1.0e-12};
    std::uint64_t snapshot_revision_key{0};

    [[nodiscard]] bool complete() const noexcept;
    [[nodiscard]] bool sameSourceState(
        const LevelSetGeometryRevision& other) const noexcept;
};

/** Rank-local mesh epochs retained for live-consumer staleness checks. */
struct LevelSetGeometryLocalMeshRevision {
    std::uint64_t mesh_geometry_revision{0};
    std::uint64_t mesh_topology_revision{0};
    std::uint64_t ownership_revision{0};
    std::uint64_t numbering_revision{0};

    [[nodiscard]] friend bool operator==(
        const LevelSetGeometryLocalMeshRevision&,
        const LevelSetGeometryLocalMeshRevision&) = default;
};

struct LevelSetGeometryRuleRecord {
    LinearCornerStrictBranch construction_observation{LinearCornerStrictBranch::Unchecked};
    LevelSetGeometryRuleRole role{LevelSetGeometryRuleRole::Interface};
    LevelSetGeometryRetention retention{
        LevelSetGeometryRetention::Retained};
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
    LevelSetGeometryMomentCertificate moment_certificate{};
    // Classification-only storage of a validated full-cell volume rule (see
    // LevelSetGeometrySnapshotPolicy::classification_only_full_cell_side).
    // The point lists of reference_rule and physical_rule and the parent-cell
    // moments of moment_certificate are empty; every other field keeps its
    // value, and classification_only_content_digest is the content digest of
    // the materialized record.
    // materializeLevelSetGeometryRuleRecord() recomputes the points.
    bool classification_only{false};
    std::uint64_t classification_only_content_digest{0};
};

/** Epoch-free topology descriptor used for authoritative interface sources. */
[[nodiscard]] std::uint64_t levelSetGeometrySourceTopologyKey(
    const CutInterfaceFragment& fragment,
    ElementType parent_element_type,
    Real tolerance);

struct LevelSetGeometrySnapshotPolicy {
    Real tolerance{1.0e-12};
    Real minimum_retained_volume_fraction{1.0e-8};
    int minimum_achieved_quadrature_order{0};
    bool require_complete_exterior_boundary_partition{true};
    // Volume side (Negative or Positive) whose full-cell rules are stored
    // classification-only after validation, typically the dry side of a
    // a one-sided level-set domain.  A rule qualifies when it is the only volume
    // rule of its parent cell.  Its points are validated and digested exactly
    // as before and then released, so the snapshot revision is unchanged.
    // Consumers that integrate over this side's volume points rematerialize
    // them (materializeLevelSetGeometryRuleRecord()) or fail closed.
    // Unset keeps every point.
    std::optional<geometry::CutIntegrationSide>
        classification_only_full_cell_side{};
    // Store the full-cell rules of both sides classification-only.  The
    // active-volume energy, dissipation and backward-Euler kinetic-work
    // evaluators then rematerialize them from the mesh they are given.
    bool classification_only_full_cells_on_both_sides{false};
    // Volume side (Negative or Positive) whose interface and contact rules
    // are pruned with it: in a parent cell where every volume rule of this
    // side is pruned as a small volume, its interface and contact rules are
    // pruned too, as its exterior-boundary rules are, so the retained
    // geometry of this side stays closed. Set it to the liquid side of a
    // a one-sided level-set domain. Unset keeps every interface and contact rule,
    // since a two-phase interface also bounds the other phase.
    std::optional<geometry::CutIntegrationSide> interface_retention_side{};
};

struct LevelSetGeometryScalarEvaluator {
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

    // Optional: an evaluator of the same scalar that another thread may use
    // concurrently with this one (evaluators that cache per-cell data are
    // not thread-safe).  The snapshot build is threaded only when it is set.
    std::function<LevelSetGeometryScalarEvaluator()> make_concurrent_copy{};
};

/**
 * Rank-local adapter used by the snapshot validator to gather owned-rule
 * identities and revision components across the communicator.  The FE
 * geometry layer owns the identity, content, uniqueness, and canonicalization
 * checks; callers only provide the byte-free collective operation needed to
 * exchange unsigned integer values.
 */
struct LevelSetGeometryOwnershipCollective {
    int rank{0};
    int size{1};
    std::function<std::vector<std::uint64_t>(
        std::span<const std::uint64_t>)>
        all_gather_owned_rule_identity_values{};
    std::function<std::vector<std::uint64_t>(
        std::span<const std::uint64_t>)>
        all_gather_revision_values{};
};

struct LevelSetGeometryValidationLedger {
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

using SnapshotVectorGradient =
    std::array<std::array<Real, 3>, 3>;

struct SnapshotVectorEvaluator {
    std::function<std::array<Real, 3>(
        GlobalIndex,
        const std::array<Real, 3>&,
        const geometry::CutQuadratureProvenance&)>
        value{};
    std::function<SnapshotVectorGradient(
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

/**
 * Reuse of full-cell volume records from one snapshot build to the next.
 *
 * Successive geometry rebuilds of an outer fixed point move the level set
 * only slightly.  Cells away from the interface stay full on one side, and
 * the record of a full-cell volume rule depends only on the rule's
 * classification fields, its parent cell (type, geometry, identity,
 * ownership) and the snapshot policy, never on the level-set magnitudes.
 * Its source identities, and the content digest that covers them, embed the
 * source value revision and change with every build.
 *
 * A cache passed to buildLevelSetGeometrySnapshot() keeps the records of
 * the previous build together with the content-only part of their
 * validation (ledger contributions and the identity-free part of the content
 * digest).  A later build under the same mesh revisions, communicator and
 * policy copies the record of a full-cell volume rule whose inputs compare
 * equal bit for bit (every rule field except the revision-dependent
 * identities, plus the source topology key and construction observation of
 * its region), re-stamps the identities, replays the content-only ledger
 * contributions and re-evaluates the level-set dependent checks.  Every
 * other record is built as before, so the snapshot is bitwise identical to
 * one built without the cache (compareLevelSetGeometrySnapshots()).
 */
class LevelSetGeometrySnapshotReuseCache {
public:
    struct Statistics {
        bool context_matched{false};
        std::size_t full_cell_records_reused{0};
        std::size_t full_cell_records_built{0};
        std::size_t other_records_built{0};
    };
    struct State;

    LevelSetGeometrySnapshotReuseCache();
    ~LevelSetGeometrySnapshotReuseCache();
    LevelSetGeometrySnapshotReuseCache(
        LevelSetGeometrySnapshotReuseCache&&) noexcept;
    LevelSetGeometrySnapshotReuseCache& operator=(
        LevelSetGeometrySnapshotReuseCache&&) noexcept;
    LevelSetGeometrySnapshotReuseCache(
        const LevelSetGeometrySnapshotReuseCache&) = delete;
    LevelSetGeometrySnapshotReuseCache& operator=(
        const LevelSetGeometrySnapshotReuseCache&) = delete;

    /** Forget the previous build; the next build reuses nothing. */
    void clear() noexcept;
    [[nodiscard]] bool empty() const noexcept;
    /** Reuse counters of the most recent build that used this cache. */
    [[nodiscard]] const Statistics& lastBuild() const noexcept {
        return last_build_;
    }

    /** Internal state of buildLevelSetGeometrySnapshot(). */
    [[nodiscard]] State& state() noexcept { return *state_; }
    void setLastBuild(const Statistics& statistics) noexcept {
        last_build_ = statistics;
    }

private:
    std::unique_ptr<State> state_;
    Statistics last_build_{};
};

class LevelSetGeometrySnapshot {
public:
    LevelSetGeometrySnapshot(const LevelSetGeometrySnapshot&) = delete;
    LevelSetGeometrySnapshot& operator=(const LevelSetGeometrySnapshot&) =
        delete;

    [[nodiscard]] const LevelSetGeometryRevision& revision() const noexcept;
    [[nodiscard]] const LevelSetGeometryLocalMeshRevision&
    localMeshRevision() const noexcept;
    [[nodiscard]] const LevelSetGeometrySnapshotPolicy& policy() const noexcept;
    [[nodiscard]] const LevelSetInterfaceDomain& interfaceDomain() const noexcept;
    [[nodiscard]] const std::vector<GeneratedInterfaceBoundaryIntersectionDomain>&
    contactDomains() const noexcept;
    [[nodiscard]] const std::vector<GeneratedActiveBoundaryDomain>&
    activeBoundaryDomains() const noexcept;
    [[nodiscard]] const std::vector<LevelSetGeometryRuleRecord>&
    rules() const noexcept;
    [[nodiscard]] const LevelSetGeometryValidationLedger& ledger() const noexcept;

    [[nodiscard]] std::vector<const LevelSetGeometryRuleRecord*>
    retainedRules(LevelSetGeometryRuleRole role) const;
    [[nodiscard]] std::size_t residentBytes() const noexcept;

private:
    friend std::shared_ptr<const LevelSetGeometrySnapshot>
    buildLevelSetGeometrySnapshot(
        LevelSetInterfaceDomain,
        std::vector<GeneratedInterfaceBoundaryIntersectionDomain>,
        std::vector<GeneratedActiveBoundaryDomain>,
        const assembly::IMeshAccess&,
        LevelSetGeometrySnapshotPolicy,
        LevelSetGeometryScalarEvaluator,
        std::string,
        LevelSetGeometryOwnershipCollective,
        LevelSetGeometrySnapshotReuseCache*,
        int);

    LevelSetGeometrySnapshot(
        LevelSetGeometryRevision revision,
        LevelSetGeometryLocalMeshRevision local_mesh_revision,
        LevelSetGeometrySnapshotPolicy policy,
        LevelSetInterfaceDomain interface_domain,
        std::vector<GeneratedInterfaceBoundaryIntersectionDomain> contact_domains,
        std::vector<GeneratedActiveBoundaryDomain> active_boundary_domains,
        std::vector<LevelSetGeometryRuleRecord> rules,
        LevelSetGeometryValidationLedger ledger);

    LevelSetGeometryRevision revision_{};
    LevelSetGeometryLocalMeshRevision local_mesh_revision_{};
    LevelSetGeometrySnapshotPolicy policy_{};
    LevelSetInterfaceDomain interface_domain_{};
    std::vector<GeneratedInterfaceBoundaryIntersectionDomain> contact_domains_{};
    std::vector<GeneratedActiveBoundaryDomain> active_boundary_domains_{};
    std::vector<LevelSetGeometryRuleRecord> rules_{};
    LevelSetGeometryValidationLedger ledger_{};
};

/**
 * The number of quadrature points of the retained, rank-owned rules of
 * @p role (classification-only volume records are rematerialized, which
 * needs @p mesh).
 */
[[nodiscard]] std::size_t ownedSnapshotRulePointCount(
    const LevelSetGeometrySnapshot& snapshot,
    LevelSetGeometryRuleRole role,
    const assembly::IMeshAccess* mesh = nullptr);

[[nodiscard]] std::shared_ptr<const LevelSetGeometrySnapshot>
buildLevelSetGeometrySnapshot(
    LevelSetInterfaceDomain interface_domain,
    std::vector<GeneratedInterfaceBoundaryIntersectionDomain> contact_domains,
    std::vector<GeneratedActiveBoundaryDomain> active_boundary_domains,
    const assembly::IMeshAccess& mesh,
    LevelSetGeometrySnapshotPolicy policy = {},
    LevelSetGeometryScalarEvaluator scalar = {},
    std::string domain_id = {},
    LevelSetGeometryOwnershipCollective ownership_collective = {});

/**
 * As above, reusing the full-cell volume records of the previous build kept
 * in reuse_cache (see LevelSetGeometrySnapshotReuseCache) and storing this
 * build for the next one.  A null cache builds every record.  The per-rule
 * work (record construction, rule checks, content digests) runs on `threads`
 * threads (FE/Core/DeterministicParallel.h); participants other than the
 * caller evaluate the level set through scalar.make_concurrent_copy(), and
 * without it the build is serial.  The result is bitwise identical for any
 * cache state and thread count.
 */
[[nodiscard]] std::shared_ptr<const LevelSetGeometrySnapshot>
buildLevelSetGeometrySnapshot(
    LevelSetInterfaceDomain interface_domain,
    std::vector<GeneratedInterfaceBoundaryIntersectionDomain> contact_domains,
    std::vector<GeneratedActiveBoundaryDomain> active_boundary_domains,
    const assembly::IMeshAccess& mesh,
    LevelSetGeometrySnapshotPolicy policy,
    LevelSetGeometryScalarEvaluator scalar,
    std::string domain_id,
    LevelSetGeometryOwnershipCollective ownership_collective,
    LevelSetGeometrySnapshotReuseCache* reuse_cache,
    int threads = 1);

/**
 * Bitwise comparison of two snapshots: revision, local mesh revision,
 * policy, ledger and every rule record field (points included).  Returns an
 * empty string when they are identical, otherwise a description of the
 * first difference.  The source domains are not compared.
 */
[[nodiscard]] std::string compareLevelSetGeometrySnapshots(
    const LevelSetGeometrySnapshot& a,
    const LevelSetGeometrySnapshot& b);

/**
 * Recompute the reference and physical points of a classification-only
 * record on the mesh it was built on.  The result equals the record stored
 * before its points were released, bit for bit.  Materialized records are
 * returned unchanged.
 */
[[nodiscard]] LevelSetGeometryRuleRecord
materializeLevelSetGeometryRuleRecord(
    const LevelSetGeometryRuleRecord& record,
    const assembly::IMeshAccess& mesh);

/** Content digest that enters the snapshot revision key. */
[[nodiscard]] std::uint64_t levelSetGeometryRuleContentDigest(
    const LevelSetGeometryRuleRecord& record) noexcept;

struct LevelSetGeometrySnapshotCacheStatistics {
    std::size_t live_snapshot_count{0};
    std::size_t live_resident_bytes{0};
    std::size_t peak_live_snapshot_count{0};
    std::size_t peak_live_resident_bytes{0};
    std::size_t hit_count{0};
    std::size_t miss_count{0};
    std::size_t expired_eviction_count{0};
};

class LevelSetGeometrySnapshotCache {
public:
    [[nodiscard]] std::shared_ptr<const LevelSetGeometrySnapshot> find(
        std::uint64_t revision_key);
    void insert(std::shared_ptr<const LevelSetGeometrySnapshot> snapshot);
    void evictExpired();
    [[nodiscard]] LevelSetGeometrySnapshotCacheStatistics statistics();

private:
    // A snapshot is immutable, so its resident size is measured once, when
    // it is inserted.
    struct Entry {
        std::weak_ptr<const LevelSetGeometrySnapshot> snapshot{};
        std::size_t resident_bytes{0};
    };
    std::unordered_map<std::uint64_t, Entry> snapshots_{};
    LevelSetGeometrySnapshotCacheStatistics statistics_{};
};

/// Test access to internals of the snapshot validation.
namespace snapshot_detail {

/// The containment certificate of the represented-phase check: true, with
/// the barycentric coordinates of `point`, when `point` lies in `simplex`
/// (every coordinate within [-t, 1 + t], t = max(tolerance, 128 eps)).
/// The snapshot always lets a double-precision pre-check reject points that
/// the extended-precision solve certainly rejects;
/// `allow_double_rejection` = false disables it, so that a test can show the
/// certificate is the same either way.
[[nodiscard]] bool representedSimplexCoordinates(
    const CutInterfaceReferenceSimplex& simplex,
    const std::array<Real, 3>& point,
    int ambient_dimension,
    Real tolerance,
    std::array<Real, 4>& coordinates,
    bool allow_double_rejection) noexcept;

/// The pre-check itself, for a tetrahedron: true only when the
/// extended-precision solve rejects `point` for `coordinate_tolerance`.
[[nodiscard]] bool tetrahedronCertainlyRejectsPoint(
    const std::array<std::array<Real, 3>, 4>& vertices,
    const std::array<Real, 3>& point,
    Real coordinate_tolerance) noexcept;

/// The volume-partition check of the snapshot validation: every global cell
/// has one local parent, its negative and positive volume measures sum to its
/// parent measure (the maximum error goes to `ledger`), and the polynomial
/// moments of every cell with more than one rule (or a materialized one) are
/// validated, cells in ascending global id.
void validateVolumePartition(
    const std::vector<LevelSetGeometryRuleRecord>& records,
    const assembly::IMeshAccess& mesh,
    const LevelSetGeometrySnapshotPolicy& policy,
    LevelSetGeometryValidationLedger& ledger);

} // namespace snapshot_detail

} // namespace interfaces
} // namespace svmp::FE

#endif // SVMP_FE_INTERFACES_LEVELSETGEOMETRYSNAPSHOT_H

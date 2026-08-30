/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_CONSTRAINTS_SMALLCUTAGGREGATIONCONSTRAINT_H
#define SVMP_FE_CONSTRAINTS_SMALLCUTAGGREGATIONCONSTRAINT_H

/**
 * @file SmallCutAggregationConstraint.h
 * @brief AgFEM-style small-cut aggregation for unfitted active domains.
 *
 * Vertices of cut cells whose support contains no full-active cell are
 * ill-posed: their basis functions see only fragments of the physical
 * domain, which is the source of cut-position-dependent conditioning that
 * ghost penalties otherwise have to patch over. This constraint slaves each
 * such vertex DOF to the polynomial extension of a nearby well-posed root
 * cell (the closest full-active cell reachable through face-adjacent cut
 * cells):
 *
 *   u(x_v) = sum_k N_k^{root}(xi_root(x_v)) * u_k(root)
 *
 * evaluated per field component. Cell classification comes from the retained
 * cut-context volume rules, root selection is a deterministic breadth-first
 * search, and the weights are the root cell's own basis functions
 * extrapolated to the constrained vertex. A fixed guard contract bounds graph
 * path length, reference-domain extrapolation, individual coefficients, and
 * row L1 amplification; a globally rooted candidate with no guarded proposal
 * fails closed.
 * With aggregation active the velocity ghost penalty is unnecessary for
 * conditioning (see
 * Documentation/plan_ghost_penalty_eigen_calibration_20260611.md).
 *
 * Current scope: nodal Lagrange H1 scalar or Product fields on
 * ISO-parametric meshes, i.e. the mesh nodes of every band cell carry the
 * field's full nodal layout corners-first (P1/Q1, and Q2 on 9-node quad
 * topologies: midside mesh nodes become slaves and midside master entries
 * are emitted; verified by test_SmallCutAggregationConstraint). DOF identity
 * is resolved by cell-local nodal pairing, cross-validated against the
 * EntityDofMap wherever it covers the node (corner vertices).
 * SUB-parametric fields — a field whose nodal layout is larger than the
 * cell's mesh-node set, e.g. H1Space(Quad4 topology, order=2) on a 4-node
 * quad mesh — are rejected with a clear error: their edge/interior dofs are
 * not mesh nodes, so candidate discovery could never slave them and the
 * emitted corner extensions would not be a partition of unity.
 *
 * Under-aggregation policy: candidates without a reachable full-active root
 * are isolated cut ISLANDS — a routine geometric condition in breaking
 * free-surface flows (the 312-step d18 gate sees up to ~48 per refresh once
 * the surface fragments) — and are pinned homogeneously (their content is
 * sub-resolution and an isolated region's pressure level is locally
 * indeterminate), never fatal. Mapping-inversion failures and (defensively)
 * constraint lines whose master entries all vanish are MACHINERY failures
 * and throw at apply time instead of leaving unstabilized small-cut DOFs in
 * the system. In MPI, cut/full-support and boundary-exclusion facts are
 * combined by global field DOF before candidates are selected; ownership-
 * filtered cut rules and owner-only wall labels therefore cannot create
 * different slave sets. The distributed band graph is reconstructed from
 * exactly-one-owner declarations of every classified cell's sorted corner
 * field-DOF face signatures; this removes any mutual-cell-halo requirement
 * for conforming manifold C0 meshes. Nonconforming/hanging subface topology
 * is outside the current aggregation scope because a coarse face and its
 * children do not share one canonical signature. A candidate is pinned as a
 * true rootless island only when the two-sided cut context classifies every
 * communicator-owned mesh cell (inactive-full cells are retained with zero
 * class flags); an induced/truncated band instead fails closed. Consequently,
 * direct FE callers must retain a volume-rule classification for every owned
 * mesh cell and both level-set sides (the ActiveAndInactive retention mode).
 * An ActiveOnly cut context does not satisfy this aggregation precondition
 * and fails closed. The initial constraint application inside FESystem::setup
 * may precede cut-context generation and is therefore empty; any post-setup
 * rebuild/refresh requires the generated context and fails closed if it is
 * absent. The configured interface marker must also occur in the generated
 * volume context on at least one communicator rank. Assembly must follow that
 * generated-context rebuild. Ranks that can see a candidate propose roots,
 * and a communicator-wide choice uses (BFS distance, globally unique physical
 * root-cell ID) as its deterministic key. Algebraic master DOF numbers are
 * excluded because owner-contiguous numbering changes under repartitioning.
 * Equivalent providers of the selected physical root must agree on the
 * extension weights. The canonical line is installed on every rank where its
 * slave is relevant only after proving that the slave has exactly one owner
 * and every nonzero master is relevant there. Distributed mesh adapters must
 * expose globally unique cell IDs; missing IDs, master availability, or
 * inconsistent component/weight data fails closed before the generic
 * owner-wins parallel merge. Increase mesh/DOF overlap in the latter case.
 * SVMP_AGGREGATION_ALLOW_UNAGGREGATED=1 restores the
 * legacy open handling of local island/machinery failures for debugging, but
 * does not bypass this communicator-consistency safety check; the
 * SVMP_AGGREGATION_MAX_LINES debug cap disables both while engaged.
 * SVMP_AGGREGATION_LINEAR_EXTENSION=1 is an A/B knob that restricts the
 * extension to the root's linear corner sub-basis (engaged only when that
 * sub-basis is strictly smaller than the field basis).
 *
 * Strong-BC precedence contract: vertices on the caller-supplied excluded
 * boundary markers are never slaved, and DOFs already constrained when this
 * constraint applies are skipped via isConstrained(). If a later strong
 * Dirichlet constraint reaches an aggregated slave through another code path,
 * AffineConstraints gives the strong value precedence by replacing the
 * master-bearing line with the Dirichlet line.
 */

#include "Constraints/SystemConstraint.h"
#include "Core/Types.h"
#include "Geometry/CutQuadrature.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace svmp {
namespace FE {
namespace constraints {

namespace detail {

struct SmallCutAggregationPendingProlongation;

/**
 * Partition-invariant identity used to choose among aggregation roots.
 *
 * Global algebraic DOF numbers are intentionally allowed to change with an
 * owner-contiguous partition.  They therefore cannot participate in the root
 * ordering.  A distributed caller must supply a globally unique physical cell
 * identifier.
 */
struct SmallCutAggregationPhysicalRootKey {
    std::size_t distance{std::numeric_limits<std::size_t>::max()};
    GlobalIndex cell_gid{INVALID_GLOBAL_INDEX};
};

[[nodiscard]] constexpr bool smallCutAggregationPhysicalRootLess(
    const SmallCutAggregationPhysicalRootKey& lhs,
    int lhs_provider_rank,
    const SmallCutAggregationPhysicalRootKey& rhs,
    int rhs_provider_rank) noexcept
{
    if (lhs.distance != rhs.distance) {
        return lhs.distance < rhs.distance;
    }
    if (lhs.cell_gid != rhs.cell_gid) {
        return lhs.cell_gid < rhs.cell_gid;
    }
    return lhs_provider_rank < rhs_provider_rank;
}

[[nodiscard]] constexpr bool smallCutAggregationSamePhysicalRoot(
    const SmallCutAggregationPhysicalRootKey& lhs,
    const SmallCutAggregationPhysicalRootKey& rhs) noexcept
{
    return lhs.distance == rhs.distance && lhs.cell_gid == rhs.cell_gid;
}

} // namespace detail

struct SmallCutAggregationGuardOptions {
    std::size_t maximum_root_path_length{8u};
    Real maximum_reference_extrapolation_distance{4.0};
    Real maximum_absolute_coefficient{16.0};
    Real maximum_row_l1_norm{32.0};
};

enum class SmallCutAggregationActiveFeatureDisposition {
    Rooted,
    Rootless,
};

/**
 * Canonical face-connected retained active-cell component.
 *
 * The stable feature ID is the minimum physical cell GID in the component.
 * The whole-feature digest and the separately domain-tagged full-active and
 * cut-cell digests mix sorted physical cell GIDs to expose membership and
 * equal-count class churn. These 64-bit digests are audit aids and not
 * collision-free identity proofs. Connectivity is
 * deliberately the background-cell face graph: it neither resolves multiple
 * selected-side regions within one cut cell nor proves that selected-side
 * geometry crosses a shared face.
 */
struct SmallCutAggregationActiveFeatureReport {
    GlobalIndex stable_feature_id{INVALID_GLOBAL_INDEX};
    std::uint64_t canonical_cell_gid_digest{0u};
    std::uint64_t canonical_full_active_cell_gid_digest{0u};
    std::uint64_t canonical_cut_cell_gid_digest{0u};
    SmallCutAggregationActiveFeatureDisposition disposition{
        SmallCutAggregationActiveFeatureDisposition::Rootless};
    std::size_t canonical_cell_count{0u};
    std::size_t canonical_full_active_cell_count{0u};
    std::size_t canonical_cut_cell_count{0u};
    Real canonical_retained_physical_volume{0.0};
};

/**
 * One communicator-canonical active-feature comparison across successive
 * successful publications with no intervening failed refresh.
 *
 * Absent-side disposition and volume fields are placeholders and must be read
 * only when the corresponding `present_*` flag is true. A classification
 * change means that the stable feature retained the same minimum-cell
 * identity while its cell membership, full/cut partition, whole-cell digest,
 * or class-specific digest changed. Continuous retained-volume change alone
 * is deliberately not a topology change.
 */
struct SmallCutAggregationActiveFeatureTransitionReport {
    GlobalIndex stable_feature_id{INVALID_GLOBAL_INDEX};
    bool present_before{false};
    bool present_after{false};
    std::uint64_t canonical_cell_gid_digest_before{0u};
    std::uint64_t canonical_cell_gid_digest_after{0u};
    std::uint64_t canonical_full_active_cell_gid_digest_before{0u};
    std::uint64_t canonical_full_active_cell_gid_digest_after{0u};
    std::uint64_t canonical_cut_cell_gid_digest_before{0u};
    std::uint64_t canonical_cut_cell_gid_digest_after{0u};
    SmallCutAggregationActiveFeatureDisposition disposition_before{
        SmallCutAggregationActiveFeatureDisposition::Rootless};
    SmallCutAggregationActiveFeatureDisposition disposition_after{
        SmallCutAggregationActiveFeatureDisposition::Rootless};
    std::size_t canonical_cell_count_before{0u};
    std::size_t canonical_cell_count_after{0u};
    std::size_t canonical_full_active_cell_count_before{0u};
    std::size_t canonical_full_active_cell_count_after{0u};
    std::size_t canonical_cut_cell_count_before{0u};
    std::size_t canonical_cut_cell_count_after{0u};
    Real canonical_retained_physical_volume_before{0.0};
    Real canonical_retained_physical_volume_after{0.0};
    bool cell_classification_changed{false};
    bool disposition_changed{false};
};

enum class SmallCutAggregationGeometryIdentityKind : std::uint8_t {
    Unavailable,
    GeneratedPublicationSource,
    AuthoritativeFreeSurfaceSnapshot,
};

/**
 * Rank-invariant source/snapshot identity for one published refresh.
 *
 * Authoritative snapshot revisions use the snapshot's distributed revision
 * keys. The generated-publication fallback contains only source and policy
 * fields whose publication contract is communicator-wide; its rank-local
 * request mesh epochs are deliberately retained in LocalPublicationLineage
 * instead. `communicator_fingerprint_consensus_validated` is set only after
 * the final all-rank 64-bit fingerprint consensus. That consensus is
 * collision-prone audit evidence, not exact cross-rank comparison of every
 * string and scalar. An unavailable identity is explicit and is never
 * promoted to a canonical geometry identity.
 */
struct SmallCutAggregationCanonicalGeometryIdentity {
    SmallCutAggregationGeometryIdentityKind kind{
        SmallCutAggregationGeometryIdentityKind::Unavailable};
    bool available{false};
    bool communicator_fingerprint_consensus_validated{false};
    std::string source_id{};
    std::string domain_id{};
    int interface_marker{-1};
    Real isovalue{0.0};
    std::uint64_t source_layout_revision{0u};
    std::uint64_t source_value_revision{0u};
    std::uint64_t quadrature_policy_key{0u};
    std::uint64_t snapshot_revision_key{0u};
    std::uint64_t distributed_mesh_geometry_revision{0u};
    std::uint64_t distributed_mesh_topology_revision{0u};
    std::uint64_t distributed_ownership_revision{0u};
    std::uint64_t distributed_numbering_revision{0u};
    std::uint64_t canonical_fingerprint{0u};
};

/**
 * Explicitly rank-local lineage captured while preparing one publication.
 *
 * These values are diagnostic cache/lifecycle stamps. They identify the
 * local source view used for before/after auditing but are not canonicalized
 * and must not be interpreted as communicator-global epochs.
 */
struct SmallCutAggregationLocalPublicationLineage {
    std::uint64_t successful_publication_ordinal{0u};
    std::uint64_t cut_context_content_revision{0u};
    bool has_snapshot_local_mesh_revision{false};
    std::uint64_t snapshot_local_mesh_geometry_revision{0u};
    std::uint64_t snapshot_local_mesh_topology_revision{0u};
    std::uint64_t snapshot_local_ownership_revision{0u};
    std::uint64_t snapshot_local_numbering_revision{0u};
    bool has_generated_publication_request{false};
    std::uint64_t generated_request_mesh_geometry_revision{0u};
    std::uint64_t generated_request_mesh_topology_revision{0u};
    std::uint64_t generated_request_ownership_revision{0u};
    bool mesh_revision_tracking_available{false};
    std::uint64_t mesh_geometry_revision{0u};
    std::uint64_t mesh_topology_revision{0u};
    std::uint64_t mesh_ownership_revision{0u};
    std::uint64_t mesh_numbering_revision{0u};
    std::uint64_t mesh_field_layout_revision{0u};
    std::uint64_t mesh_label_revision{0u};
    std::uint64_t mesh_active_configuration_epoch{0u};
    std::uint64_t fe_space_revision{0u};
    std::uint64_t fe_dof_layout_revision{0u};
    std::uint64_t fe_constraint_layout_revision{0u};
    std::uint64_t fe_block_layout_revision{0u};
    std::uint64_t affine_constraint_build_revision{0u};
};

/**
 * Transition report for successive successful publications of one
 * field/marker/side aggregation constraint with no intervening failed
 * refresh.
 *
 * Its feature/slave ledger is communicator-canonical because it is derived
 * only from the already communicator-canonical feature and slave sets. The
 * vector is sorted by stable feature ID and contains the union of previous
 * and current features, including unchanged persistent features. The local
 * lineage subfields are intentionally rank-local and need not match across
 * ranks. This is
 * topology/constraint telemetry only: it does not compare assembled
 * operators or solved states and therefore cannot establish node-crossing
 * solution continuity by itself. Entered/exited feature identities, feature
 * and cell-class counts, dispositions, and slave-set changes are exact within
 * the canonical ledgers. Same-count membership changes additionally rely on
 * the 64-bit whole/class cell-GID digests and are therefore collision-prone
 * audit evidence, not a collision-free topology oracle. The before/after
 * geometry identity and local lineage fields identify the compared source
 * publications when source provenance is available; unavailable provenance
 * remains explicit. Local lineage epochs remain rank-local even when their
 * publication ordinals happen to agree.
 */
struct SmallCutAggregationTopologyTransitionReport {
    SmallCutAggregationCanonicalGeometryIdentity geometry_identity_before{};
    SmallCutAggregationCanonicalGeometryIdentity geometry_identity_after{};
    SmallCutAggregationLocalPublicationLineage local_lineage_before{};
    SmallCutAggregationLocalPublicationLineage local_lineage_after{};
    std::uint64_t canonical_feature_class_fingerprint_before{0u};
    std::uint64_t canonical_feature_class_fingerprint_after{0u};
    std::uint64_t canonical_slave_set_fingerprint_before{0u};
    std::uint64_t canonical_slave_set_fingerprint_after{0u};
    std::size_t canonical_active_feature_count_before{0u};
    std::size_t canonical_active_feature_count_after{0u};
    std::size_t canonical_features_entered{0u};
    std::size_t canonical_features_exited{0u};
    std::size_t canonical_features_persisted{0u};
    std::size_t canonical_feature_classification_changes{0u};
    std::size_t canonical_features_became_rooted{0u};
    std::size_t canonical_features_became_rootless{0u};
    std::size_t canonical_aggregate_slaves_before{0u};
    std::size_t canonical_aggregate_slaves_after{0u};
    std::size_t canonical_aggregate_slaves_entered{0u};
    std::size_t canonical_aggregate_slaves_left{0u};
    Real canonical_rootless_active_physical_volume_before{0.0};
    Real canonical_rootless_active_physical_volume_after{0.0};
    Real canonical_rootless_active_physical_volume_delta{0.0};
    bool canonical_topology_changed{false};
    std::vector<SmallCutAggregationActiveFeatureTransitionReport>
        canonical_feature_transitions{};
};

/**
 * Result of the most recent successful aggregation refresh.
 *
 * Feature/slave/count fields form the communicator-canonical ledger; the
 * separately labeled lineage is rank-local. Candidate fields count vertices,
 * while aggregate, pin, and suppression
 * fields count DOFs. Active-feature fields count communicator-global,
 * face-connected components of cells with retained selected-side volume. A
 * structurally rooted feature contains at least one full-active cell; a
 * structurally rootless feature contains only cut cells. This classification
 * is independent of experimental runtime root-rejection options. Rootless
 * physical volume is selected from one validated canonical declaration per
 * physical cell and is repeated in each field-specific report. It is retained
 * geometry affected by the homogeneous support-removal policy, not a claim
 * that liquid volume was conservatively transferred or geometrically deleted.
 * Guard maxima describe provider-visible proposal or bounded traversal
 * attempts. Root search is restricted to the configured path neighborhood;
 * roots outside that envelope are never row candidates. A root-path rejection
 * counts a candidate in a structurally root-eligible feature for which the
 * bounded search found no admissible root. It does not enumerate every remote
 * full-active cell in that feature.
 * Extrapolation and line rejection counts are communicator sums of rank-local
 * attempts; root-path rejection counts are candidate-level guard failures.
 * None of these fields counts unique physical roots.
 */
struct SmallCutAggregationRefreshReport {
    FieldId field{INVALID_FIELD_ID};
    geometry::CutIntegrationSide active_side{
        geometry::CutIntegrationSide::Negative};
    int interface_marker{-1};
    SmallCutAggregationCanonicalGeometryIdentity geometry_identity{};
    SmallCutAggregationLocalPublicationLineage local_lineage{};
    std::uint64_t canonical_feature_class_fingerprint{0u};
    std::uint64_t canonical_slave_set_fingerprint{0u};
    std::size_t maximum_root_path_length{0u};
    std::size_t maximum_observed_root_path{0u};
    std::size_t root_path_guard_rejections{0u};
    Real maximum_reference_extrapolation_distance{0.0};
    Real maximum_observed_reference_extrapolation{0.0};
    std::size_t extrapolation_guard_rejections{0u};
    Real maximum_absolute_coefficient{0.0};
    Real maximum_observed_absolute_coefficient{0.0};
    Real maximum_row_l1_norm{0.0};
    Real maximum_observed_row_l1_norm{0.0};
    std::size_t line_guard_rejections{0u};
    std::size_t canonical_candidate_vertices{0u};
    std::size_t canonical_rooted_candidate_vertices{0u};
    std::size_t canonical_rootless_candidate_vertices{0u};
    std::size_t canonical_owned_aggregate_dofs{0u};
    std::size_t canonical_owned_pinned_dofs{0u};
    std::size_t canonical_strong_suppressed_dofs{0u};
    std::size_t canonical_active_feature_count{0u};
    std::size_t canonical_rooted_active_feature_count{0u};
    std::size_t canonical_rootless_active_feature_count{0u};
    Real canonical_rootless_active_physical_volume{0.0};
    std::vector<SmallCutAggregationActiveFeatureReport>
        canonical_active_features{};
    // Empty on the first successful refresh and after any failed refresh.
    // Otherwise compares the immediately preceding successful publication of
    // this constraint instance. Geometry identity availability is explicit;
    // low-level contexts without source publication provenance are never
    // mislabeled as canonical geometry snapshots.
    std::optional<SmallCutAggregationTopologyTransitionReport>
        canonical_topology_transition{};
};

/**
 * How a candidate row left the aggregation resolver before the complete
 * constraint set was synchronized and closed.
 */
enum class SmallCutAggregationProvisionalRowKind : std::uint8_t {
    RootedExtension,
    RootlessHomogeneousPin,
    SupportedFreeIdentity,
    UnaggregatedFreeIdentity,
    PreexistingConstraint,
};

/**
 * Exact tangent-row shape after all constraints were synchronized and closed.
 */
enum class SmallCutAggregationFinalRowKind : std::uint8_t {
    Identity,
    MasterBearing,
    HomogeneousPin,
    FixedValue,
};

enum class SmallCutAggregationActiveCellKind : std::uint8_t {
    FullActive,
    Cut,
};

enum class SmallCutAggregationPatchKind : std::uint8_t {
    Rooted,
    Rootless,
    SupportedCut,
};

/**
 * One communicator-canonical active background cell used by aggregation.
 *
 * Field DOFs, neighbor cell IDs, and retained-rule identity keys are sorted.
 * A retained-rule key is the rule's nonzero cut-topology revision, scoped by
 * this report's marker/side and the physical cell GID; the scalar key is not
 * globally unique on its own. Zero is an explicit unavailable-identity
 * placeholder retained for audit, and any such placeholder makes the report
 * ineligible for trace-bound certification. The retained volume is the
 * selected-side physical measure used by the active-volume assembly.
 */
struct SmallCutAggregationProlongationCell {
    GlobalIndex cell_gid{INVALID_GLOBAL_INDEX};
    int owner_rank{-1};
    int retained_measure_provider_rank{-1};
    SmallCutAggregationActiveCellKind kind{
        SmallCutAggregationActiveCellKind::Cut};
    GlobalIndex active_feature_id{INVALID_GLOBAL_INDEX};
    Real retained_physical_volume{0.0};
    std::vector<std::uint64_t> retained_rule_stable_ids{};
    std::vector<GlobalIndex> field_dofs{};
    std::vector<GlobalIndex> active_face_neighbor_cell_gids{};
};

/**
 * One candidate row with both producer lineage and the exact closed tangent.
 *
 * Root fields are invalid for non-rooted rows. Provisional entries are the
 * normalized polynomial-extension row selected by the aggregation resolver;
 * final entries are terminal unconstrained masters copied after transitive
 * closure. Identity rows explicitly contain (slave_dof, 1).
 */
struct SmallCutAggregationProlongationRow {
    GlobalIndex candidate_dof{INVALID_GLOBAL_INDEX};
    std::size_t component{0u};
    GlobalIndex slave_dof{INVALID_GLOBAL_INDEX};
    int slave_owner_rank{-1};
    SmallCutAggregationProvisionalRowKind provisional_kind{
        SmallCutAggregationProvisionalRowKind::UnaggregatedFreeIdentity};
    SmallCutAggregationFinalRowKind final_kind{
        SmallCutAggregationFinalRowKind::Identity};
    bool preconstrained_at_apply{false};
    GlobalIndex root_cell_gid{INVALID_GLOBAL_INDEX};
    int root_cell_owner_rank{-1};
    int root_provider_rank{-1};
    std::size_t root_distance{std::numeric_limits<std::size_t>::max()};
    std::vector<ConstraintEntry> provisional_entries{};
    std::vector<ConstraintEntry> final_entries{};
    Real final_inhomogeneity{0.0};
};

/**
 * Nonpartitioning patch induced by candidate rows with a common root.
 *
 * A rooted patch contains its full-active root and every cut cell containing
 * one of its canonical slaves. Support cells additionally include active
 * cells carrying terminal masters introduced by constraint closure. A
 * supported-cut patch covers a cut cell with no candidate-induced patch and
 * includes its active face neighbors as support. Active feature IDs list
 * every face-connected active-cell component represented by the patch. This
 * is a vector because C0 DOFs at a vertex or edge can couple otherwise
 * face-disconnected components. A cut cell may therefore occur in more than
 * one patch.
 */
struct SmallCutAggregationProlongationPatch {
    SmallCutAggregationPatchKind kind{
        SmallCutAggregationPatchKind::Rooted};
    GlobalIndex root_cell_gid{INVALID_GLOBAL_INDEX};
    int root_cell_owner_rank{-1};
    std::vector<GlobalIndex> active_feature_ids{};
    std::vector<GlobalIndex> member_cell_gids{};
    std::vector<GlobalIndex> support_cell_gids{};
    std::vector<GlobalIndex> slave_dofs{};
};

struct SmallCutAggregationProlongationRevision {
    int local_rank{0};
    int communicator_size{1};
    ConstraintRevisionSnapshot constraint{};
    std::uint64_t affine_constraint_layout_revision{0u};
    std::uint64_t cut_context_content_revision{0u};
    bool has_free_surface_snapshot_revision{false};
    std::uint64_t free_surface_snapshot_revision{0u};
    bool has_source_value_revision{false};
    std::uint64_t source_value_revision{0u};
};

/**
 * Immutable, revision-bound description of the actual aggregate
 * prolongation installed in an FESystem.
 *
 * Rows, cells, and patches are communicator-canonical and sorted. Revision
 * fields are local cache-validity stamps and are intentionally excluded from
 * the canonical digest because owner-filtered contexts and locally relevant
 * constraint storage may advance their counters differently by rank. The
 * digest covers the exact canonical contents, including IEEE floating-point
 * bit patterns. It is communicator-canonical for one algebraic partition,
 * not partition-invariant across rank counts, because algebraic DOFs and
 * owner ranks are explicit contents.
 * `trace_bound_eligible` is false whenever a debugging fail-open path or an
 * unresolved external master-bearing row prevents a closed aggregate-patch
 * interpretation.
 */
struct SmallCutAggregationProlongationReport {
    FieldId field{INVALID_FIELD_ID};
    geometry::CutIntegrationSide active_side{
        geometry::CutIntegrationSide::Negative};
    int interface_marker{-1};
    bool slave_all_cut{false};
    bool linear_extension{false};
    bool allow_unaggregated{false};
    bool trace_bound_eligible{false};
    SmallCutAggregationGuardOptions guards{};
    SmallCutAggregationProlongationRevision revision{};
    std::uint64_t canonical_content_digest{0u};
    std::vector<SmallCutAggregationProlongationRow> rows{};
    std::vector<SmallCutAggregationProlongationCell> active_cells{};
    std::vector<SmallCutAggregationProlongationPatch> patches{};
};

class SmallCutAggregationConstraint final : public ISystemConstraint {
public:
    class LifecycleCheckpoint {
    public:
        LifecycleCheckpoint(const LifecycleCheckpoint&) = default;
        LifecycleCheckpoint& operator=(const LifecycleCheckpoint&) = default;

    private:
        friend class SmallCutAggregationConstraint;
        LifecycleCheckpoint() = default;

        std::vector<GlobalIndex> previous_canonical_slaves{};
        std::optional<SmallCutAggregationRefreshReport>
            completed_refresh_report{};
        std::shared_ptr<
            const detail::SmallCutAggregationPendingProlongation>
            pending_prolongation{};
        std::uint64_t successful_publication_ordinal{0u};
    };

    /// @param excluded_boundary_markers Boundary markers whose face vertices
    ///        carry strong Dirichlet data and must never become aggregation
    ///        slaves (their BC takes precedence).
    /// @param excluded_vertices Mesh vertices that must never become
    ///        aggregation slaves. Used for nodal pins whose constraint must
    ///        win over aggregation — e.g. a pressure gauge pin removes the
    ///        global pressure constant, which a homogeneous aggregation line
    ///        cannot do.
    SmallCutAggregationConstraint(FieldId field,
                                  geometry::CutIntegrationSide active_side,
                                  int interface_marker,
                                  std::vector<int> excluded_boundary_markers = {},
                                  std::vector<GlobalIndex> excluded_vertices = {},
                                  SmallCutAggregationGuardOptions guards = {});

    void apply(const systems::FESystem& system, AffineConstraints& constraints) override;

    bool updateValues(const systems::FESystem& system,
                      AffineConstraints& constraints,
                      double time,
                      double dt) override;

    [[nodiscard]] bool isTimeDependent() const noexcept override { return false; }

    [[nodiscard]] ConstraintDependencyDeclaration dependencyDeclaration() const override;

    [[nodiscard]] systems::SetupStorageRequirements storageRequirements() const noexcept override;

    /**
     * Report for the current successful canonical refresh, if one exists.
     *
     * A refresh clears the prior report before validation. Failed refreshes,
     * the initial context-free setup pass, and debug line-cap bypasses leave
     * this empty.
     */
    [[nodiscard]] const std::optional<SmallCutAggregationRefreshReport>&
    completedRefreshReport() const noexcept
    {
        return completed_refresh_report_;
    }

private:
    friend class systems::FESystem;

    /**
     * Materialize the immutable prolongation report from a fully closed
     * system constraint set. Returns null when the most recent apply pass was
     * context-free, failed, or used the debug line-cap bypass.
     */
    [[nodiscard]] std::shared_ptr<
        const SmallCutAggregationProlongationReport>
    finalizeProlongationReport(
        const systems::FESystem& system,
        const AffineConstraints& closed_constraints) const;

    /**
     * Internal transaction support for refresh/churn and pending metadata.
     */
    [[nodiscard]] std::shared_ptr<const LifecycleCheckpoint>
    captureLifecycleCheckpoint() const;
    void restoreLifecycleCheckpoint(const LifecycleCheckpoint& checkpoint);

    FieldId field_{INVALID_FIELD_ID};
    geometry::CutIntegrationSide active_side_{geometry::CutIntegrationSide::Negative};
    int interface_marker_{-1};
    std::vector<int> excluded_boundary_markers_{};
    std::vector<GlobalIndex> excluded_vertices_{};
    SmallCutAggregationGuardOptions guards_{};
    // Constraint-instance state avoids the stale address/pointer-reuse hazard
    // of a process-global churn cache keyed by &FESystem.
    std::vector<GlobalIndex> previous_canonical_slaves_{};
    std::optional<SmallCutAggregationRefreshReport> completed_refresh_report_{};
    std::shared_ptr<const detail::SmallCutAggregationPendingProlongation>
        pending_prolongation_{};
    std::uint64_t successful_publication_ordinal_{0u};
};

} // namespace constraints
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_CONSTRAINTS_SMALLCUTAGGREGATIONCONSTRAINT_H

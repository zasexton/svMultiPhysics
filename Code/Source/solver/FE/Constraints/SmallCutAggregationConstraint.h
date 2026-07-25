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
#include <limits>
#include <string>
#include <vector>

namespace svmp {
namespace FE {
namespace constraints {

namespace detail {

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

class SmallCutAggregationConstraint final : public ISystemConstraint {
public:
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

private:
    FieldId field_{INVALID_FIELD_ID};
    geometry::CutIntegrationSide active_side_{geometry::CutIntegrationSide::Negative};
    int interface_marker_{-1};
    std::vector<int> excluded_boundary_markers_{};
    std::vector<GlobalIndex> excluded_vertices_{};
    SmallCutAggregationGuardOptions guards_{};
    // Constraint-instance state avoids the stale address/pointer-reuse hazard
    // of a process-global churn cache keyed by &FESystem.
    std::vector<GlobalIndex> previous_canonical_slaves_{};
};

} // namespace constraints
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_CONSTRAINTS_SMALLCUTAGGREGATIONCONSTRAINT_H

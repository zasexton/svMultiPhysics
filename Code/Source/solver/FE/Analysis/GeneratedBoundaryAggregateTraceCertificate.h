/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_ANALYSIS_GENERATEDBOUNDARYAGGREGATETRACECERTIFICATE_H
#define SVMP_FE_ANALYSIS_GENERATEDBOUNDARYAGGREGATETRACECERTIFICATE_H

/**
 * @file GeneratedBoundaryAggregateTraceCertificate.h
 * @brief Collective finite-dimensional trace bounds on aggregate patches.
 */

#include "Core/Types.h"
#include "Math/DenseLinearAlgebra.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace svmp {
namespace FE {

namespace systems {
class FESystem;
}

namespace analysis {

enum class GeneratedBoundaryRigidModeQuotientStatus : std::uint8_t {
    NotApplicable = 0u,
    NoCandidate = 1u,
    ReproductionNotExact = 2u,
    CandidateRankNotProven = 3u,
    NonzeroPencilAction = 4u,
    Applied = 5u,
};

/**
 * Scope of one on-demand generated-boundary trace certification.
 *
 * The current implementation deliberately accepts only reference-frame,
 * affine P1 product velocity fields on linear triangles or tetrahedra.
 * A certified aggregation patch must represent exactly one face-connected
 * active feature, because the first implementation supplies one rigid-mode
 * family per patch. The call is collective over the field communicator.
 * `maximum_reduced_dimension` may lower, but may not raise, the implementation
 * hard limit of 128 terminal tangent DOFs per certified patch. Additional
 * fixed implementation caps bound active cells, patches, retained and
 * boundary rules, quadrature points, raw patch DOFs, serialized words, and
 * modeled dense memory. The exact binary64 coordinate quotient currently has
 * its own hard dimension cap of 32 after structural rigid modes are removed;
 * exceeding any cap fails collectively before the corresponding unbounded
 * allocation or dense solve.
 */
struct GeneratedBoundaryAggregateTraceCertificationOptions {
    FieldId field{INVALID_FIELD_ID};
    int physical_boundary_marker{-1};
    int volume_interface_marker{-1};
    int generated_active_boundary_marker{-1};
    Real dynamic_viscosity{0.0};
    std::size_t maximum_reduced_dimension{128u};
};

/**
 * Certificate for one boundary-carrying aggregate support patch.
 *
 * A localized patch is either a singleton full-active boundary cell or a
 * boundary-parent/aggregate-root support used when the complete canonical
 * aggregate cannot fit the fixed exact quotient cap. Existing aggregation
 * patches retain their canonical index. The generalized bound is for
 *
 *   integral_boundary (h_normal / mu) |(2 mu eps(v)) n|^2
 *       <= C_patch
 *          integral_support 2 mu eps(v) : eps(v)
 *
 * on the exact closed tangent rows of this patch. The implementation divides
 * both forms by the common positive factor `2 mu` before certification,
 * preserving the quotient while avoiding viscosity-dependent overflow and
 * underflow. Acceptance is controlled by the exact factorized dyadic proof
 * stored in `generalized_bound.exact_dyadic`; formed dense matrices and their
 * floating eigensolver values are diagnostics only. Consequently,
 * `generalized_bound.conservative_upper_bound` is the directly proven exact
 * dyadic upper bound rather than the gauge-dependent padded floating bound.
 */
struct GeneratedBoundaryAggregateTracePatchCertificate {
    std::size_t canonical_patch_index{
        std::numeric_limits<std::size_t>::max()};
    bool localized_support_patch{false};
    GlobalIndex root_cell_gid{INVALID_GLOBAL_INDEX};
    std::vector<GlobalIndex> support_cell_gids{};
    std::vector<std::uint64_t> boundary_rule_stable_ids{};
    std::size_t raw_support_dof_count{0u};
    std::size_t terminal_tangent_dof_count{0u};
    std::size_t rigid_mode_candidate_count{0u};
    std::size_t structural_rigid_mode_count{0u};
    std::size_t rigid_mode_constraint_rank{0u};
    GeneratedBoundaryRigidModeQuotientStatus
        rigid_mode_quotient_status{
            GeneratedBoundaryRigidModeQuotientStatus::
                NotApplicable};
    // The tolerance and residual describe the floating candidate search.
    // Exact raw Gram-factor action, exact tangent reproduction, and a modular
    // full-column-rank proof are required before a candidate is applied.
    Real rigid_mode_reproduction_tolerance{0.0};
    Real maximum_rigid_mode_reproduction_residual{0.0};
    bool exact_rigid_factor_action_proven{false};
    std::size_t maximum_cell_support_overlap{0u};
    Real retained_support_physical_volume{0.0};
    Real generated_boundary_physical_measure{0.0};
    math::DensePsdGeneralizedEigenvalueBound generalized_bound{};
};

/**
 * Immutable result of one successful collective certification.
 *
 * Boundary rules are assigned exactly once to a canonical or localized
 * patch. Because aggregation patches are nonpartitioning, the global bound
 * is the maximum, over active cells, of the outward-rounded sum of the patch
 * bounds whose denominator supports contain that cell. This is generally
 * tighter than `maximum_patch_bound * maximum_support_overlap`, while the
 * latter quantities remain available as diagnostics.
 *
 * This report does not select, mutate, or qualify a production penalty.
 */
struct GeneratedBoundaryAggregateTraceCertificate {
    FieldId field{INVALID_FIELD_ID};
    int physical_boundary_marker{-1};
    int volume_interface_marker{-1};
    int generated_active_boundary_marker{-1};
    Real dynamic_viscosity{0.0};
    int communicator_size{1};
    std::uint64_t aggregation_content_digest{0u};
    // Canonical only within this communicator/algebraic partition.
    std::uint64_t canonical_certificate_digest{0u};
    // These cache-validity stamps are rank-local by report contract and are
    // intentionally excluded from the canonical certificate digest.
    std::uint64_t cut_context_content_revision{0u};
    std::uint64_t free_surface_snapshot_revision{0u};
    std::uint64_t source_value_revision{0u};
    std::uint64_t affine_constraint_layout_revision{0u};
    std::size_t active_cell_count{0u};
    std::size_t generated_boundary_rule_count{0u};
    std::size_t certified_patch_count{0u};
    std::size_t localized_support_patch_count{0u};
    std::size_t maximum_support_overlap{0u};
    std::size_t maximum_terminal_tangent_dimension{0u};
    Real retained_active_physical_volume{0.0};
    Real generated_boundary_physical_measure{0.0};
    Real maximum_patch_conservative_upper_bound{0.0};
    Real global_conservative_upper_bound{0.0};
    std::vector<GeneratedBoundaryAggregateTracePatchCertificate> patches{};
};

/**
 * Reject a certificate whose stored canonical digest is absent or stale.
 *
 * The recomputation covers the complete canonical certificate payload,
 * including exact factorized-proof diagnostics. Rank-local cache-validity
 * revisions remain excluded by the certificate contract.
 */
void validateGeneratedBoundaryAggregateTraceCertificateDigest(
    const GeneratedBoundaryAggregateTraceCertificate& certificate);

/**
 * Certify the generated-boundary viscous trace bound on the live aggregate
 * tangent space.
 *
 * Every rank in the field communicator must call this routine. It validates
 * current cut/snapshot/constraint revisions, reconstructs closed tangent rows
 * from their canonical DOF owners, gathers cell and boundary contributions
 * from their declared providers, and performs identical deterministic dense
 * certification on every rank. Unsupported or ambiguous states throw
 * collectively.
 */
[[nodiscard]] GeneratedBoundaryAggregateTraceCertificate
certifyGeneratedBoundaryAggregateTrace(
    const systems::FESystem& system,
    const GeneratedBoundaryAggregateTraceCertificationOptions& options);

} // namespace analysis
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_ANALYSIS_GENERATEDBOUNDARYAGGREGATETRACECERTIFICATE_H

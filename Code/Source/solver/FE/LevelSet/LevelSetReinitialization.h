#pragma once

/**
 * @file
 * @ingroup fe_level_set
 * @brief Signed-distance repair utilities for level-set fields.
 */

#include "Assembly/Assembler.h"
#include "Core/Types.h"
#include "Dofs/DofHandler.h"
#include "LevelSet/LevelSetOptions.h"
#include "Systems/FESystem.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace svmp::FE::level_set {

enum class LevelSetWallContactConstraintKind : std::uint8_t {
    PrescribedAngle,
    AcceptedDynamicAngle
};

/**
 * Wall-contact law and accepted physical frame for redistance projection.
 *
 * The parent identity is partition independent.  A caller may provide only
 * the locally owned records; the distributed projection forms one sorted,
 * duplicate-free communicator snapshot before changing any coefficient.
 */
struct LevelSetWallContactConstraint {
    LevelSetWallContactConstraintKind kind{
        LevelSetWallContactConstraintKind::PrescribedAngle};
    int interface_marker{-1};
    int boundary_marker{-1};
    GlobalIndex parent_cell_global_id{INVALID_GLOBAL_INDEX};
    std::uint64_t geometry_revision{0u};
    // PrescribedAngle uses the convention
    //   grad(phi) . physical_wall_normal
    //       = -cos(target_angle_radians) |grad(phi)|.
    // The caller supplies the physical wall-normal orientation used by this
    // equation; reversing it changes the prescribed-angle convention.
    Real target_angle_radians{0.0};
    std::array<Real, 3> physical_wall_normal{{0.0, 0.0, 0.0}};
    // The accepted contact point and oriented contact-line tangent form the
    // physical frame held fixed by the prescribed update.  In two dimensions
    // the tangent is the oriented out-of-plane direction.
    std::array<Real, 3> accepted_contact_point{{0.0, 0.0, 0.0}};
    std::array<Real, 3> accepted_contact_line_tangent{{0.0, 0.0, 0.0}};
};

struct LevelSetSignedDistanceRepairResult {
    bool success{false};
    LevelSetReinitializationMethod method{LevelSetReinitializationMethod::Projection};
    std::size_t repaired_dofs{0};
    std::size_t preserved_dofs{0};
    std::size_t interface_fragments{0};
    std::size_t cut_cells{0};
    std::size_t interface_displacement_samples{0};
    Real max_abs_update{0.0};
    Real max_distance{0.0};
    Real max_interface_displacement{0.0};
    Real l2_interface_displacement{0.0};
    int iterations{0};
    bool converged{false};
    bool zero_set_bound_satisfied{false};
    Real max_iteration_residual{0.0};
    // Maximum nodal discrepancy from the nearest-interface signed-distance
    // target after all geometric constraints are applied.  The two following
    // fields separate the freely repairable error from the irreducible error
    // of the accepted wall-contact constrained optimum.
    Real max_signed_distance_error{0.0};
    Real max_unconstrained_signed_distance_error{0.0};
    Real max_wall_constrained_signed_distance_error{0.0};
    std::size_t wall_contact_constraints{0u};
    std::size_t wall_contact_cells{0u};
    std::size_t wall_contact_dofs{0u};
    bool wall_contact_constraints_satisfied{false};
    Real max_wall_contact_scale_residual{0.0};
    Real max_prescribed_contact_value_residual{0.0};
    Real max_prescribed_contact_angle_error_radians{0.0};
    Real max_contact_line_displacement{0.0};
    Real max_contact_angle_change_radians{0.0};
    // Retained for output compatibility.  The corrected algorithm no longer
    // freezes a preservation band, so this is always zero.
    Real preserve_band_width{0.0};
    std::string diagnostic{};
};

/**
 * Repair nodal level-set coefficients by projecting mesh nodes to generated
 * linear interface primitives while bounding motion of the original discrete
 * zero set.  Linear cut-cell DOFs relax toward local signed distance and a
 * geometric line search limits every original cut-edge crossing.  Connected
 * high-order cut patches use a common positive scaling so unresolved
 * polynomial roots are preserved exactly.  DOFs away from cut cells relax
 * toward signed distance.
 *
 * A fixed continuous P1/Q1 space cannot, in general, both preserve an
 * arbitrary piecewise-planar/curved discrete zero set exactly and represent a
 * spatially varying positive redistance multiplier: the product usually lies
 * outside the original finite-element space.  If the geometric displacement
 * bound prevents the signed-distance tolerance from being reached, this
 * routine returns a bounded candidate with success=true and converged=false;
 * production callers must leave the accepted state unchanged in that case.
 * For scalar H1 nodal fields, cell-local DOFs are repaired when their local
 * ordering can be paired one-to-one with the mesh cell node ordering; remaining
 * vertex DOFs are repaired as a fallback.
 *
 * Supported cuts are linear Line2/Line3, Triangle3/Triangle6, Quad4/Quad8/Quad9,
 * and Tetra4/Tetra10 corner cuts. Other element types are skipped. This is an
 * iterative, displacement-bounded projection rather than a
 * Hamilton-Jacobi PDE.  max_iterations and pseudo_time_step_scale control the
 * relaxation and the result reports geometric edge-crossing displacement.
 * On a distributed DOF layout, accepted coefficients and DOF coordinates are
 * taken from their unique owners, cut primitives and zero-crossing guards are
 * gathered from owned cells, and every rank evaluates the same immutable
 * global projection snapshot.  Locally owned wall-contact constraints are
 * gathered and canonicalized the same way.  AcceptedDynamicAngle patches are
 * projected onto a positive common coefficient scale, leaving their accepted
 * finite-element crossing and unit normal unchanged.  PrescribedAngle patches
 * instead receive a unit-gradient affine target through the accepted contact
 * point and oriented contact-line frame, satisfying the declared wall-normal
 * angle relation.  Consequently, convergence applies to the unconstrained
 * signed-distance error and the appropriate constrained optimum; the total
 * discrepancy remains visible in the result.  The output candidate is
 * assigned only after all collective validation and projection work
 * completes.
 */
[[nodiscard]] LevelSetSignedDistanceRepairResult
repairLevelSetSignedDistanceByProjection(
    const assembly::IMeshAccess& mesh,
    const dofs::DofHandler& level_set_dofs,
    const LevelSetReinitializationOptions& options,
    std::span<const Real> input_coefficients,
    std::vector<Real>& repaired_coefficients,
    std::span<const LevelSetWallContactConstraint>
        wall_contact_constraints = {});

[[nodiscard]] LevelSetSignedDistanceRepairResult
repairLevelSetSignedDistanceByProjection(
    const systems::FESystem& system,
    FieldId level_set_field,
    const LevelSetReinitializationOptions& options,
    std::span<const Real> input_solution,
    std::vector<Real>& repaired_solution,
    std::span<const LevelSetWallContactConstraint>
        wall_contact_constraints = {});

} // namespace svmp::FE::level_set

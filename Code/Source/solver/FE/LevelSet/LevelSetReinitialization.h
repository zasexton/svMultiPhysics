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

#include <cstddef>
#include <span>
#include <string>
#include <vector>

namespace svmp::FE::level_set {

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
    // target after the zero-set-preserving constraint is applied.
    Real max_signed_distance_error{0.0};
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
 * Projection reinitialization currently supports single-rank communicators
 * only. Both overloads reject a multi-rank DOF communicator before modifying
 * the output candidate because primitive construction and coefficient binding
 * are rank-local.
 */
[[nodiscard]] LevelSetSignedDistanceRepairResult
repairLevelSetSignedDistanceByProjection(
    const assembly::IMeshAccess& mesh,
    const dofs::DofHandler& level_set_dofs,
    const LevelSetReinitializationOptions& options,
    std::span<const Real> input_coefficients,
    std::vector<Real>& repaired_coefficients);

[[nodiscard]] LevelSetSignedDistanceRepairResult
repairLevelSetSignedDistanceByProjection(
    const systems::FESystem& system,
    FieldId level_set_field,
    const LevelSetReinitializationOptions& options,
    std::span<const Real> input_solution,
    std::vector<Real>& repaired_solution);

} // namespace svmp::FE::level_set

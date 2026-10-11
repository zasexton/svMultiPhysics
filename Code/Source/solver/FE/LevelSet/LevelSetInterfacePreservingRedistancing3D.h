#pragma once

/**
 * @file
 * @ingroup fe_level_set
 * @brief Redistancing of a P1 level set on affine Tetra4 cells to the exact
 *        distance of its own zero surface, with the open surface edges
 *        continued across the boundary (the 3D counterpart of
 *        LevelSetInterfacePreservingRedistancing.h).
 */

#include "LevelSet/LevelSetInterfacePreservingRedistancing.h"

#include <cstddef>
#include <span>
#include <vector>

namespace svmp::FE::level_set {

/**
 * @brief Replace every nodal value by the signed distance to the zero surface.
 *
 * The zero surface is the piecewise-linear zero set of the P1 field: one
 * triangle per tetrahedron with one or three vertices below the isovalue,
 * and the quadrilateral of a tetrahedron with two vertices below split into
 * two triangles along a canonical diagonal (the one through the crossing of
 * the edge with the smallest DOF pair).  Edge crossings are computed in the
 * orientation of increasing DOF number, and a root on a vertex is that vertex
 * exactly, so neighbouring cells share their crossings bitwise; triangles
 * with two coincident corners are dropped.  A surface edge used by a single
 * triangle lies on the mesh boundary (a contact line on a wall, or the
 * intersection with a symmetry or open boundary).  There the triangle is
 * continued beyond the edge by a half-strip in its own plane, perpendicular
 * to the edge, so that near a contact line the new field is the distance to
 * the continued interface plane and its gradient keeps the discrete contact
 * angle.  Each value is isovalue -/+ min(distance to triangles, distance to
 * half-strips), with the sign of the input value.
 *
 * The result has the same fields as the 2D one; interface_segments counts the
 * surface triangles and open_ends the open edges.  Supported cells are affine
 * Tetra4 with one scalar DOF per vertex.  Every rank contributes the
 * triangles of its owned cells, the triangle list is gathered on the DOF
 * communicator, and each value is the minimum over that common list, which
 * does not depend on the order of the list, so the result does not depend on
 * the partition.  There is no numerical parameter.
 */
[[nodiscard]] LevelSetZeroPolygonRedistancingResult
redistanceLevelSetToZeroSurface(const assembly::IMeshAccess& mesh,
                                const dofs::DofHandler& level_set_dofs,
                                Real isovalue,
                                std::span<const Real> level_set,
                                std::vector<Real>& redistanced_level_set);

/// All DOFs of the Tetra4 cells that contain at least one of @p dofs.
[[nodiscard]] std::vector<std::size_t> levelSetDofCellClosure3D(
    const assembly::IMeshAccess& mesh,
    const dofs::DofHandler& level_set_dofs,
    std::span<const std::size_t> dofs);

/// levelSetInterfaceMotionScale on Tetra4 cells (three velocity components).
[[nodiscard]] LevelSetInterfaceMotionScale levelSetInterfaceMotionScale3D(
    const assembly::IMeshAccess& mesh,
    const dofs::DofHandler& level_set_dofs,
    const dofs::DofHandler& velocity_dofs,
    Real isovalue,
    std::span<const Real> level_set,
    std::span<const Real> velocity);

/**
 * @brief Change of the zero surface between two P1 fields on the same mesh.
 *
 * max_crossing_shift and cells_with_changed_cut as in 2D; max_open_end_shift
 * is the largest distance from the midpoint of an open edge of the first
 * field to the nearest open edge of the second (the contact-line shift), and
 * open_ends_first/second count the open edges.
 */
[[nodiscard]] LevelSetZeroPolygonChange measureLevelSetZeroSurfaceChange(
    const assembly::IMeshAccess& mesh,
    const dofs::DofHandler& level_set_dofs,
    Real isovalue,
    std::span<const Real> first,
    std::span<const Real> second);

} // namespace svmp::FE::level_set

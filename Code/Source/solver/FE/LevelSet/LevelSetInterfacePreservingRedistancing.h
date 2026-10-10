#pragma once

/**
 * @file
 * @ingroup fe_level_set
 * @brief Redistancing of a P1 level set to the exact distance of its own zero
 *        polygon, with the open polygon ends continued across the boundary.
 */

#include "Assembly/Assembler.h"
#include "Core/Types.h"
#include "Dofs/DofHandler.h"

#include <cstddef>
#include <span>
#include <string>
#include <vector>

namespace svmp::FE::level_set {

/**
 * @brief Outcome of one redistancing.  All values are communicator-global.
 */
struct LevelSetZeroPolygonRedistancingResult {
    bool success{false};
    // True when at least one coefficient changed.
    bool applied{false};
    std::size_t interface_segments{0u};
    // Polygon end points that lie on one segment only (on the mesh boundary,
    // e.g. contact points); each is continued by a ray.
    std::size_t open_ends{0u};
    std::size_t redistanced_dofs{0u};
    Real max_abs_change{0.0};
    std::string diagnostic{};
};

/**
 * @brief Replace every nodal value by the signed distance to the zero polygon.
 *
 * The zero polygon is the piecewise-linear zero set of the P1 field (one
 * segment per cut triangle, edge crossings computed in a canonical edge
 * orientation so that neighbouring cells share them bitwise).  An end point
 * that belongs to a single segment lies on the mesh boundary (a contact point
 * on a wall, or an interface leaving through an open boundary); there the
 * end segment is continued by a ray beyond the boundary, so that near a
 * contact point the new field is the distance to the continued interface
 * line and its gradient keeps the discrete contact angle.  Each value is
 * isovalue -/+ min(distance to segments, distance to rays), with the sign of
 * the input value (negative for values below the isovalue).
 *
 * Where the zero polygon is straight inside a cell, the new values keep its
 * edge crossings exactly; elsewhere the crossings move by second-order
 * amounts.  Callers that must keep the interface (for example the liquid
 * measure) correct the result locally afterwards, e.g. with
 * reconcileLevelSetWithKinematicFlux() and a zero velocity, which restores the
 * lumped interface position node by node and leaves pure rescalings
 * untouched.
 *
 * Supported cells are affine Triangle3 with one scalar DOF per vertex.  The
 * coefficient span is the complete (replicated) field slice.  On a
 * distributed mesh every rank contributes the segments of its owned cells,
 * the segment list is gathered on the DOF communicator, and each value is a
 * minimum over that common list, so the result does not depend on the
 * partition.  There is no numerical parameter.
 */
[[nodiscard]] LevelSetZeroPolygonRedistancingResult
redistanceLevelSetToZeroPolygon(const assembly::IMeshAccess& mesh,
                                const dofs::DofHandler& level_set_dofs,
                                Real isovalue,
                                std::span<const Real> level_set,
                                std::vector<Real>& redistanced_level_set);

/**
 * @brief Change of the zero polygon between two P1 fields on the same mesh.
 */
struct LevelSetZeroPolygonChange {
    bool success{false};
    // Largest displacement of an edge crossing present in both fields.
    Real max_crossing_shift{0.0};
    // Largest distance from an open end (boundary crossing) of the first
    // field to the nearest open end of the second.
    Real max_open_end_shift{0.0};
    // Owned cells whose set of cut edges differs.
    std::size_t cells_with_changed_cut{0u};
    std::size_t open_ends_first{0u};
    std::size_t open_ends_second{0u};
    std::string diagnostic{};
};

[[nodiscard]] LevelSetZeroPolygonChange measureLevelSetZeroPolygonChange(
    const assembly::IMeshAccess& mesh,
    const dofs::DofHandler& level_set_dofs,
    Real isovalue,
    std::span<const Real> first,
    std::span<const Real> second);

} // namespace svmp::FE::level_set

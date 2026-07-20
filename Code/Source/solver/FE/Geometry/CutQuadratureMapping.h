/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_GEOMETRY_CUTQUADRATUREMAPPING_H
#define SVMP_FE_GEOMETRY_CUTQUADRATUREMAPPING_H

/**
 * @file CutQuadratureMapping.h
 * @brief One pointwise reference-to-physical map for retained cut quadrature.
 */

#include "Geometry/CutQuadrature.h"
#include "Geometry/GeometryMapping.h"

#include <array>
#include <memory>
#include <vector>

namespace svmp::FE {
namespace assembly {
class IMeshAccess;
}
namespace geometry {

struct MappedCutQuadraturePoint {
    std::array<Real, 3> reference_point{{0.0, 0.0, 0.0}};
    std::array<Real, 3> physical_point{{0.0, 0.0, 0.0}};
    CutGeometryJacobian jacobian{};
    CutGeometryJacobian inverse_jacobian{};
    Real absolute_jacobian_determinant{0.0};
    Real reference_weight{0.0};
    Real physical_weight{0.0};
    std::array<Real, 3> normal{{0.0, 0.0, 0.0}};
    std::array<Real, 3> boundary_normal{{0.0, 0.0, 0.0}};
    std::array<Real, 3> tangent{{0.0, 0.0, 0.0}};
};

struct MappedCutQuadratureRule {
    CutQuadratureKind kind{CutQuadratureKind::Volume};
    CutIntegrationSide side{CutIntegrationSide::Negative};
    int geometric_dimension{-1};
    MeshIndex parent_entity{static_cast<MeshIndex>(-1)};
    int marker{-1};
    std::uint64_t source_stable_id{0};
    std::uint64_t cut_topology_revision{0};
    std::uint64_t source_value_revision{0};
    Real reference_measure{0.0};
    Real physical_measure{0.0};
    std::vector<MappedCutQuadraturePoint> points{};
};

/** Build the same geometry mapping used by FE assembly for one parent cell. */
[[nodiscard]] std::shared_ptr<GeometryMapping> makeCutCellGeometryMapping(
    const assembly::IMeshAccess& mesh,
    GlobalIndex cell_id);

[[nodiscard]] Real physicalCellMeasureFromMapping(
    const assembly::IMeshAccess& mesh,
    GlobalIndex cell_id,
    int quadrature_order = -1);

/**
 * Map every retained point of a cut rule through one pointwise geometry path.
 * Reference-frame weights receive the correct volume, codimension-one, or
 * codimension-two metric. Current-frame weights are retained, while the
 * reference coordinate is still mapped to record J and |det J|.
 */
[[nodiscard]] MappedCutQuadratureRule mapCutQuadratureRuleToPhysical(
    const assembly::IMeshAccess& mesh,
    const CutQuadratureRule& rule);

[[nodiscard]] Real physicalCutQuadratureMeasure(
    const assembly::IMeshAccess& mesh,
    const CutQuadratureRule& rule);

} // namespace geometry
} // namespace svmp::FE

#endif // SVMP_FE_GEOMETRY_CUTQUADRATUREMAPPING_H

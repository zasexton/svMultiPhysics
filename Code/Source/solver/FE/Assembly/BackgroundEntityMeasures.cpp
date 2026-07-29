/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Assembly/BackgroundEntityMeasures.h"

#include "Assembly/Assembler.h"
#include "Core/FEException.h"
#include "Elements/ReferenceElement.h"
#include "Geometry/FrameGeometry.h"
#include "Quadrature/QuadratureFactory.h"
#include "Quadrature/QuadratureRule.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <vector>

namespace svmp {
namespace FE {
namespace assembly {

BackgroundEntityMeasures computeBackgroundEntityMeasures(
    const IMeshAccess& mesh,
    GlobalIndex cell_id,
    GlobalIndex parent_face_id,
    int test_polynomial_order,
    int trial_polynomial_order)
{
    const auto cell_type = mesh.getCellType(cell_id);
    const auto local_face_id =
        mesh.getLocalFaceIndex(parent_face_id, cell_id);
    FE_THROW_IF(
        local_face_id == INVALID_LOCAL_INDEX,
        FEException,
        "computeBackgroundEntityMeasures: invalid parent face");

    const auto reference_element =
        elements::ReferenceElement::create(cell_type);
    const auto& face_nodes = reference_element.face_nodes(
        static_cast<std::size_t>(local_face_id));
    ElementType face_type = ElementType::Unknown;
    switch (face_nodes.size()) {
        case 2u:
            face_type = ElementType::Line2;
            break;
        case 3u:
            face_type = ElementType::Triangle3;
            break;
        case 4u:
            face_type = ElementType::Quad4;
            break;
        default:
            FE_THROW(
                FEException,
                "computeBackgroundEntityMeasures: unsupported parent-face "
                "topology");
    }

    const int measure_order = std::max(
        {test_polynomial_order,
         trial_polynomial_order,
         mesh.getCellGeometryOrder(cell_id)});
    const int quadrature_order =
        quadrature::QuadratureFactory::recommended_order(
            measure_order, false);
    const auto cell_quadrature_rule =
        quadrature::QuadratureFactory::create(
            cell_type, quadrature_order);
    const auto face_quadrature_rule =
        quadrature::QuadratureFactory::create(
            face_type, quadrature_order);

    std::vector<std::array<Real, 3>> coordinates;
    mesh.getCellCoordinates(cell_id, coordinates);
    const auto cell_geometry = geometry::evaluateCellFrame(
        cell_type, *cell_quadrature_rule, coordinates);
    FE_THROW_IF(
        cell_geometry.measures.size() !=
            cell_quadrature_rule->num_points(),
        FEException,
        "computeBackgroundEntityMeasures: background-cell geometry has an "
        "incompatible quadrature size");

    const auto face_geometry = geometry::evaluateFaceFrame(
        cell_type,
        local_face_id,
        face_type,
        *face_quadrature_rule,
        coordinates,
        {},
        false,
        false);
    FE_THROW_IF(
        face_geometry.surface_measures.size() !=
            face_quadrature_rule->num_points(),
        FEException,
        "computeBackgroundEntityMeasures: parent-face geometry has an "
        "incompatible quadrature size");

    BackgroundEntityMeasures measures;
    for (std::size_t a = 0u; a < coordinates.size(); ++a) {
        for (std::size_t b = a + 1u;
             b < coordinates.size();
             ++b) {
            const Real dx =
                coordinates[a][0] - coordinates[b][0];
            const Real dy =
                coordinates[a][1] - coordinates[b][1];
            const Real dz =
                coordinates[a][2] - coordinates[b][2];
            measures.cell_diameter = std::max(
                measures.cell_diameter,
                std::sqrt(dx * dx + dy * dy + dz * dz));
        }
    }
    for (std::size_t q = 0u;
         q < cell_quadrature_rule->num_points();
         ++q) {
        measures.physical_cell_measure +=
            cell_quadrature_rule->weight(q) *
            cell_geometry.measures[q];
    }
    for (std::size_t q = 0u;
         q < face_quadrature_rule->num_points();
         ++q) {
        measures.physical_parent_face_measure +=
            face_quadrature_rule->weight(q) *
            face_geometry.surface_measures[q];
    }
    FE_THROW_IF(
        !std::isfinite(measures.cell_diameter) ||
            !(measures.cell_diameter > Real{0.0}) ||
            !std::isfinite(measures.physical_cell_measure) ||
            !(measures.physical_cell_measure > Real{0.0}) ||
            !std::isfinite(
                measures.physical_parent_face_measure) ||
            !(measures.physical_parent_face_measure >
              Real{0.0}),
        FEException,
        "computeBackgroundEntityMeasures: nonpositive background entity "
        "measures");

    measures.h_normal =
        Real{2.0} * measures.physical_cell_measure /
        measures.physical_parent_face_measure;
    FE_THROW_IF(
        !std::isfinite(measures.h_normal) ||
            !(measures.h_normal > Real{0.0}),
        FEException,
        "computeBackgroundEntityMeasures: invalid facet-normal "
        "background-cell size");
    return measures;
}

} // namespace assembly
} // namespace FE
} // namespace svmp

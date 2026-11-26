/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#include "MappingFactory.h"

#include "Basis/BasisTraits.h"
#include "Basis/NodeOrderingConventions.h"
#include "Geometry/FrameGeometry.h"

namespace svmp::FE::geometry {
namespace {

ElementType vertexElementType(ElementType type)
{
    switch (type) {
    case ElementType::Quad8:     return ElementType::Quad4;
    case ElementType::Hex20:     return ElementType::Hex8;
    case ElementType::Wedge15:   return ElementType::Wedge6;
    case ElementType::Pyramid13: return ElementType::Pyramid5;
    default:                     return basis::canonical_lagrange_type(type);
    }
}

} // namespace

std::shared_ptr<GeometryMapping> MappingFactory::create(const MappingRequest& req,
                                                        const std::vector<math::Vector<Real, 3>>& nodes) {
    if (req.basis) {
        return std::make_shared<IsoparametricMapping>(req.basis, nodes);
    }

    const auto type = req.element_type;
    const int order = req.geometry_order == -1 ? defaultGeometryOrder(type) : req.geometry_order;
    FE_THROW_IF(order < 0, FEException, "MappingFactory: invalid geometry order");
    const auto vertex_type = vertexElementType(type);
    if (order == 1 && vertex_type != type) {
        const auto vertex_basis = std::make_shared<basis::LagrangeBasis>(vertex_type, 1);
        const auto parent_nodes = basis::ReferenceNodeLayout::num_nodes(type);
        FE_THROW_IF(nodes.size() != vertex_basis->size() && nodes.size() != parent_nodes, FEException,
                    "MappingFactory: vertex geometry requires the vertices or the complete parent node set");
        std::vector<math::Vector<Real, 3>> vertices(nodes.begin(), nodes.begin() + vertex_basis->size());
        return std::make_shared<SubparametricMapping>(vertex_basis, std::move(vertices), order, parent_nodes);
    }
    if (req.use_affine && order == 1 &&
        (type == ElementType::Line2 || type == ElementType::Triangle3 || type == ElementType::Tetra4)) {
        return std::make_shared<LinearMapping>(type, nodes);
    }

    std::shared_ptr<basis::BasisFunction> geometry_basis;
    if (order == 0) {
        geometry_basis = std::make_shared<basis::LagrangeBasis>(vertex_type, order);
    } else if (type == ElementType::Quad8 || type == ElementType::Hex20 ||
               type == ElementType::Wedge15 || type == ElementType::Pyramid13) {
        FE_THROW_IF(order != 2, FEException, "MappingFactory: serendipity geometry requires order one or two");
        geometry_basis = std::make_shared<basis::SerendipityBasis>(type, order);
    } else {
        geometry_basis = std::make_shared<basis::LagrangeBasis>(type, order);
    }
    FE_THROW_IF(geometry_basis->size() != nodes.size(), FEException,
                "MappingFactory: node count does not match geometry basis");
    if (order < 1) {
        return std::make_shared<SubparametricMapping>(geometry_basis, nodes, order);
    }
    if (order > 1) {
        return std::make_shared<SuperparametricMapping>(geometry_basis, nodes, order);
    }
    return std::make_shared<IsoparametricMapping>(geometry_basis, nodes);
}

} // namespace svmp::FE::geometry

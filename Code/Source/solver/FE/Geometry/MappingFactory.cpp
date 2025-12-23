/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#include "MappingFactory.h"

namespace svmp {
namespace FE {
namespace geometry {

std::shared_ptr<GeometryMapping> MappingFactory::create(const MappingRequest& req,
                                                        const std::vector<math::Vector<Real, 3>>& nodes) {
    // Prefer user-provided basis when present
    if (req.basis) {
        return std::make_shared<IsoparametricMapping>(req.basis, nodes);
    }

    // Affine for simplices
    if (req.use_affine && (req.element_type == ElementType::Line2 ||
                           req.element_type == ElementType::Triangle3 ||
                           req.element_type == ElementType::Tetra4)) {
        return std::make_shared<LinearMapping>(req.element_type, nodes);
    }

    // Build a geometry basis and geometry node set. For some higher-order
    // elements we intentionally use a lower-order geometry mapping (subparametric)
    // by selecting only the vertex nodes.
    std::shared_ptr<basis::BasisFunction> geom_basis;
    std::vector<math::Vector<Real, 3>> geom_nodes = nodes;

    if (req.element_type == ElementType::Quad8 || req.element_type == ElementType::Hex20) {
        // geometry_mode=true: may use reduced polynomial order for robust mapping
        geom_basis = std::make_shared<basis::SerendipityBasis>(
            req.element_type,
            req.geometry_order > 0 ? req.geometry_order : 2,
            true);
    } else if (req.element_type == ElementType::Wedge15) {
        // Geometry: use linear Wedge6 mapping based on the six vertices.
        if (nodes.size() < 6) {
            throw FEException("MappingFactory: Wedge15 geometry requires at least 6 vertex nodes",
                              __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
        }
        geom_basis = std::make_shared<basis::LagrangeBasis>(ElementType::Wedge6, 1);
        geom_nodes.assign(nodes.begin(), nodes.begin() + 6);
    } else if (req.element_type == ElementType::Pyramid13) {
        // Geometry: use linear Pyramid5 mapping based on the five vertices.
        if (nodes.size() < 5) {
            throw FEException("MappingFactory: Quadratic pyramid geometry requires at least 5 vertex nodes",
                              __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
        }
        geom_basis = std::make_shared<basis::LagrangeBasis>(ElementType::Pyramid5, 1);
        geom_nodes.assign(nodes.begin(), nodes.begin() + 5);
    } else if (req.element_type == ElementType::Pyramid14) {
        // Geometry: use full quadratic rational Pyramid14 mapping with all 14 nodes.
        if (nodes.size() < 14) {
            throw FEException("MappingFactory: Pyramid14 geometry requires 14 nodes",
                              __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
        }
        geom_basis = std::make_shared<basis::LagrangeBasis>(ElementType::Pyramid14, 2);
        geom_nodes.assign(nodes.begin(), nodes.begin() + 14);
    } else {
        geom_basis = std::make_shared<basis::LagrangeBasis>(req.element_type, req.geometry_order);
    }

    if (static_cast<int>(geom_basis->size()) != static_cast<int>(geom_nodes.size())) {
        throw FEException("MappingFactory: node count does not match geometry basis",
                          __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
    }

    if (req.geometry_order < 1) {
        return std::make_shared<SubparametricMapping>(geom_basis, geom_nodes, req.geometry_order);
    }
    if (req.geometry_order > 1) {
        return std::make_shared<SuperparametricMapping>(geom_basis, geom_nodes, req.geometry_order);
    }
    return std::make_shared<IsoparametricMapping>(geom_basis, geom_nodes);
}

} // namespace geometry
} // namespace FE
} // namespace svmp

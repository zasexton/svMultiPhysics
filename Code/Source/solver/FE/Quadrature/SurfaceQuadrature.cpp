/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#include "SurfaceQuadrature.h"
#include <algorithm>
#include <stdexcept>

namespace svmp {
namespace FE {
namespace quadrature {

namespace {

svmp::CellFamily face_family(ElementType elem_type, int face_id) {
    switch (elem_type) {
        case ElementType::Line2:
        case ElementType::Line3:
            return svmp::CellFamily::Point;
        case ElementType::Triangle3:
        case ElementType::Triangle6:
            if (face_id < 0 || face_id > 2) {
                throw FEException("Triangle face_id must be in [0,2]",
                                  __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
            }
            return svmp::CellFamily::Line;
        case ElementType::Quad4:
        case ElementType::Quad8:
        case ElementType::Quad9:
            if (face_id < 0 || face_id > 3) {
                throw FEException("Quadrilateral face_id must be in [0,3]",
                                  __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
            }
            return svmp::CellFamily::Line;
        case ElementType::Tetra4:
        case ElementType::Tetra10:
            if (face_id < 0 || face_id > 3) {
                throw FEException("Tetrahedron face_id must be in [0,3]",
                                  __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
            }
            return svmp::CellFamily::Triangle;
        case ElementType::Hex8:
        case ElementType::Hex20:
        case ElementType::Hex27:
            if (face_id < 0 || face_id > 5) {
                throw FEException("Hexahedron face_id must be in [0,5]",
                                  __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
            }
            return svmp::CellFamily::Quad;
        case ElementType::Wedge6:
        case ElementType::Wedge15:
        case ElementType::Wedge18:
            if (face_id < 0 || face_id > 4) {
                throw FEException("Wedge face_id must be in [0,4]",
                                  __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
            }
            return (face_id < 2) ? svmp::CellFamily::Triangle : svmp::CellFamily::Quad;
        case ElementType::Pyramid5:
        case ElementType::Pyramid13:
        case ElementType::Pyramid14:
            if (face_id < 0 || face_id > 4) {
                throw FEException("Pyramid face_id must be in [0,4]",
                                  __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
            }
            return (face_id == 0) ? svmp::CellFamily::Quad : svmp::CellFamily::Triangle;
        default:
            throw FEException("Unsupported element type for face quadrature",
                              __FILE__, __LINE__, __func__, FEStatus::InvalidElement);
	}
}

int effective_order_for_type(int order, QuadratureType type) {
    if (order < 1) {
        order = 1;
    }
    if (type == QuadratureType::Reduced) {
        return std::max(1, order - 1);
    }
    return order;
}

} // namespace

std::unique_ptr<QuadratureRule> SurfaceQuadrature::face_rule(
    ElementType elem_type, int face_id, int order, QuadratureType type) {

    svmp::CellFamily family = face_family(elem_type, face_id);
    switch (family) {
        case svmp::CellFamily::Line:
            if (type == QuadratureType::GaussLobatto) {
                // Need 2n-3 >= order  ->  n >= ceil((order+3)/2)
                return std::make_unique<GaussLobattoQuadrature1D>(std::max(2, (order + 4) / 2));
            }
            {
                const int eff = effective_order_for_type(order, type);
                return std::make_unique<GaussQuadrature1D>(std::max(1, (eff + 2) / 2));
            }
        case svmp::CellFamily::Triangle:
            return std::make_unique<TriangleQuadrature>(effective_order_for_type(order, type));
        case svmp::CellFamily::Quad:
            return std::make_unique<QuadrilateralQuadrature>(order, type);
        default:
            // Point face (line element end) uses a single weight of 1
            {
                std::vector<QuadPoint> pts{QuadPoint{Real(0), Real(0), Real(0)}};
                std::vector<Real> wts{Real(1)};
                class PointRule : public QuadratureRule {
                public:
                    PointRule() : QuadratureRule(svmp::CellFamily::Point, 0, 0) {
                        set_data({QuadPoint{Real(0), Real(0), Real(0)}}, {Real(1)});
                    }
                };
                return std::make_unique<PointRule>();
            }
    }
}

std::unique_ptr<QuadratureRule> SurfaceQuadrature::edge_rule(
    svmp::CellFamily face_family, int edge_id, int order, QuadratureType type) {

    (void)edge_id; // canonical orientation; no special handling currently
    switch (face_family) {
        case svmp::CellFamily::Line:
            {
                const int eff = effective_order_for_type(order, type);
                return std::make_unique<GaussQuadrature1D>(std::max(1, (eff + 2) / 2));
            }
        case svmp::CellFamily::Triangle:
        case svmp::CellFamily::Quad:
            if (type == QuadratureType::GaussLobatto) {
                // Need 2n-3 >= order  ->  n >= ceil((order+3)/2)
                return std::make_unique<GaussLobattoQuadrature1D>(std::max(2, (order + 4) / 2));
            }
            {
                const int eff = effective_order_for_type(order, type);
                return std::make_unique<GaussQuadrature1D>(std::max(1, (eff + 2) / 2));
            }
        default:
            throw FEException("Unsupported face family for edge quadrature",
                              __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
    }
}

} // namespace quadrature
} // namespace FE
} // namespace svmp

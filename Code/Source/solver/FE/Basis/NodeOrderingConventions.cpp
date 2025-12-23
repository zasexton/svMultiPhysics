/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#include "NodeOrderingConventions.h"
#include "Core/FEException.h"

namespace svmp {
namespace FE {
namespace basis {

math::Vector<Real, 3> NodeOrdering::get_node_coords(ElementType elem_type, std::size_t local_node) {
    switch (elem_type) {
        // 1D Elements
        case ElementType::Line2:
            switch (local_node) {
                case 0: return {Real(-1), Real(0), Real(0)};
                case 1: return {Real(+1), Real(0), Real(0)};
                default: break;
            }
            break;

        case ElementType::Line3:
            switch (local_node) {
                case 0: return {Real(-1), Real(0), Real(0)};
                case 1: return {Real(+1), Real(0), Real(0)};
                case 2: return {Real(0), Real(0), Real(0)};
                default: break;
            }
            break;

        // 2D Quad Elements
        case ElementType::Quad4:
            switch (local_node) {
                case 0: return {Real(-1), Real(-1), Real(0)};
                case 1: return {Real(+1), Real(-1), Real(0)};
                case 2: return {Real(+1), Real(+1), Real(0)};
                case 3: return {Real(-1), Real(+1), Real(0)};
                default: break;
            }
            break;

        case ElementType::Quad8:
            switch (local_node) {
                case 0: return {Real(-1), Real(-1), Real(0)};
                case 1: return {Real(+1), Real(-1), Real(0)};
                case 2: return {Real(+1), Real(+1), Real(0)};
                case 3: return {Real(-1), Real(+1), Real(0)};
                case 4: return {Real(0), Real(-1), Real(0)};
                case 5: return {Real(+1), Real(0), Real(0)};
                case 6: return {Real(0), Real(+1), Real(0)};
                case 7: return {Real(-1), Real(0), Real(0)};
                default: break;
            }
            break;

        case ElementType::Quad9:
            switch (local_node) {
                case 0: return {Real(-1), Real(-1), Real(0)};
                case 1: return {Real(+1), Real(-1), Real(0)};
                case 2: return {Real(+1), Real(+1), Real(0)};
                case 3: return {Real(-1), Real(+1), Real(0)};
                case 4: return {Real(0), Real(-1), Real(0)};
                case 5: return {Real(+1), Real(0), Real(0)};
                case 6: return {Real(0), Real(+1), Real(0)};
                case 7: return {Real(-1), Real(0), Real(0)};
                case 8: return {Real(0), Real(0), Real(0)};
                default: break;
            }
            break;

        // 2D Triangle Elements
        case ElementType::Triangle3:
            switch (local_node) {
                case 0: return {Real(0), Real(0), Real(0)};
                case 1: return {Real(1), Real(0), Real(0)};
                case 2: return {Real(0), Real(1), Real(0)};
                default: break;
            }
            break;

        case ElementType::Triangle6:
            switch (local_node) {
                case 0: return {Real(0), Real(0), Real(0)};
                case 1: return {Real(1), Real(0), Real(0)};
                case 2: return {Real(0), Real(1), Real(0)};
                case 3: return {Real(0.5), Real(0), Real(0)};
                case 4: return {Real(0.5), Real(0.5), Real(0)};
                case 5: return {Real(0), Real(0.5), Real(0)};
                default: break;
            }
            break;

        // 3D Hex Elements
        case ElementType::Hex8:
            switch (local_node) {
                case 0: return {Real(-1), Real(-1), Real(-1)};
                case 1: return {Real(+1), Real(-1), Real(-1)};
                case 2: return {Real(+1), Real(+1), Real(-1)};
                case 3: return {Real(-1), Real(+1), Real(-1)};
                case 4: return {Real(-1), Real(-1), Real(+1)};
                case 5: return {Real(+1), Real(-1), Real(+1)};
                case 6: return {Real(+1), Real(+1), Real(+1)};
                case 7: return {Real(-1), Real(+1), Real(+1)};
                default: break;
            }
            break;

        case ElementType::Hex20:
            switch (local_node) {
                // Corners
                case 0: return {Real(-1), Real(-1), Real(-1)};
                case 1: return {Real(+1), Real(-1), Real(-1)};
                case 2: return {Real(+1), Real(+1), Real(-1)};
                case 3: return {Real(-1), Real(+1), Real(-1)};
                case 4: return {Real(-1), Real(-1), Real(+1)};
                case 5: return {Real(+1), Real(-1), Real(+1)};
                case 6: return {Real(+1), Real(+1), Real(+1)};
                case 7: return {Real(-1), Real(+1), Real(+1)};
                // Mid-edge on bottom
                case 8:  return {Real(0), Real(-1), Real(-1)};
                case 9:  return {Real(+1), Real(0), Real(-1)};
                case 10: return {Real(0), Real(+1), Real(-1)};
                case 11: return {Real(-1), Real(0), Real(-1)};
                // Mid-edge on top
                case 12: return {Real(0), Real(-1), Real(+1)};
                case 13: return {Real(+1), Real(0), Real(+1)};
                case 14: return {Real(0), Real(+1), Real(+1)};
                case 15: return {Real(-1), Real(0), Real(+1)};
                // Vertical mid-edges
                case 16: return {Real(-1), Real(-1), Real(0)};
                case 17: return {Real(+1), Real(-1), Real(0)};
                case 18: return {Real(+1), Real(+1), Real(0)};
                case 19: return {Real(-1), Real(+1), Real(0)};
                default: break;
            }
            break;

        case ElementType::Hex27:
            switch (local_node) {
                // Corners (same as Hex8)
                case 0: return {Real(-1), Real(-1), Real(-1)};
                case 1: return {Real(+1), Real(-1), Real(-1)};
                case 2: return {Real(+1), Real(+1), Real(-1)};
                case 3: return {Real(-1), Real(+1), Real(-1)};
                case 4: return {Real(-1), Real(-1), Real(+1)};
                case 5: return {Real(+1), Real(-1), Real(+1)};
                case 6: return {Real(+1), Real(+1), Real(+1)};
                case 7: return {Real(-1), Real(+1), Real(+1)};
                // Mid-edge on bottom (same order as Hex20)
                case 8:  return {Real(0), Real(-1), Real(-1)};
                case 9:  return {Real(+1), Real(0), Real(-1)};
                case 10: return {Real(0), Real(+1), Real(-1)};
                case 11: return {Real(-1), Real(0), Real(-1)};
                // Mid-edge on top (same order as Hex20)
                case 12: return {Real(0), Real(-1), Real(+1)};
                case 13: return {Real(+1), Real(0), Real(+1)};
                case 14: return {Real(0), Real(+1), Real(+1)};
                case 15: return {Real(-1), Real(0), Real(+1)};
                // Vertical mid-edges (same order as Hex20)
                case 16: return {Real(-1), Real(-1), Real(0)};
                case 17: return {Real(+1), Real(-1), Real(0)};
                case 18: return {Real(+1), Real(+1), Real(0)};
                case 19: return {Real(-1), Real(+1), Real(0)};
                // Face centers
                case 20: return {Real(0), Real(0), Real(-1)}; // bottom
                case 21: return {Real(0), Real(0), Real(+1)}; // top
                case 22: return {Real(0), Real(-1), Real(0)}; // y = -1
                case 23: return {Real(+1), Real(0), Real(0)}; // x = +1
                case 24: return {Real(0), Real(+1), Real(0)}; // y = +1
                case 25: return {Real(-1), Real(0), Real(0)}; // x = -1
                // Cell center
                case 26: return {Real(0), Real(0), Real(0)};
                default: break;
            }
            break;

        // 3D Tetrahedron Elements
        case ElementType::Tetra4:
            switch (local_node) {
                case 0: return {Real(0), Real(0), Real(0)};
                case 1: return {Real(1), Real(0), Real(0)};
                case 2: return {Real(0), Real(1), Real(0)};
                case 3: return {Real(0), Real(0), Real(1)};
                default: break;
            }
            break;

        case ElementType::Tetra10:
            switch (local_node) {
                case 0: return {Real(0), Real(0), Real(0)};
                case 1: return {Real(1), Real(0), Real(0)};
                case 2: return {Real(0), Real(1), Real(0)};
                case 3: return {Real(0), Real(0), Real(1)};
                case 4: return {Real(0.5), Real(0), Real(0)};
                case 5: return {Real(0.5), Real(0.5), Real(0)};
                case 6: return {Real(0), Real(0.5), Real(0)};
                case 7: return {Real(0), Real(0), Real(0.5)};
                case 8: return {Real(0.5), Real(0), Real(0.5)};
                case 9: return {Real(0), Real(0.5), Real(0.5)};
                default: break;
            }
            break;

        // 3D Wedge Elements
        case ElementType::Wedge6:
            switch (local_node) {
                case 0: return {Real(0), Real(0), Real(-1)};
                case 1: return {Real(1), Real(0), Real(-1)};
                case 2: return {Real(0), Real(1), Real(-1)};
                case 3: return {Real(0), Real(0), Real(+1)};
                case 4: return {Real(1), Real(0), Real(+1)};
                case 5: return {Real(0), Real(1), Real(+1)};
                default: break;
            }
            break;

        case ElementType::Wedge15:
            switch (local_node) {
                // Corners
                case 0: return {Real(0), Real(0), Real(-1)};
                case 1: return {Real(1), Real(0), Real(-1)};
                case 2: return {Real(0), Real(1), Real(-1)};
                case 3: return {Real(0), Real(0), Real(+1)};
                case 4: return {Real(1), Real(0), Real(+1)};
                case 5: return {Real(0), Real(1), Real(+1)};
                // Bottom mid-edges
                case 6: return {Real(0.5), Real(0), Real(-1)};
                case 7: return {Real(0.5), Real(0.5), Real(-1)};
                case 8: return {Real(0), Real(0.5), Real(-1)};
                // Top mid-edges
                case 9:  return {Real(0.5), Real(0), Real(+1)};
                case 10: return {Real(0.5), Real(0.5), Real(+1)};
                case 11: return {Real(0), Real(0.5), Real(+1)};
                // Vertical mid-edges
                case 12: return {Real(0), Real(0), Real(0)};
                case 13: return {Real(1), Real(0), Real(0)};
                case 14: return {Real(0), Real(1), Real(0)};
                default: break;
            }
            break;

        case ElementType::Wedge18:
            switch (local_node) {
                // Corners (same as Wedge6)
                case 0: return {Real(0), Real(0), Real(-1)};
                case 1: return {Real(1), Real(0), Real(-1)};
                case 2: return {Real(0), Real(1), Real(-1)};
                case 3: return {Real(0), Real(0), Real(+1)};
                case 4: return {Real(1), Real(0), Real(+1)};
                case 5: return {Real(0), Real(1), Real(+1)};
                // Bottom mid-edges (same as Wedge15)
                case 6: return {Real(0.5), Real(0), Real(-1)};
                case 7: return {Real(0.5), Real(0.5), Real(-1)};
                case 8: return {Real(0), Real(0.5), Real(-1)};
                // Top mid-edges (same as Wedge15)
                case 9:  return {Real(0.5), Real(0), Real(+1)};
                case 10: return {Real(0.5), Real(0.5), Real(+1)};
                case 11: return {Real(0), Real(0.5), Real(+1)};
                // Vertical mid-edges (same as Wedge15)
                case 12: return {Real(0), Real(0), Real(0)};
                case 13: return {Real(1), Real(0), Real(0)};
                case 14: return {Real(0), Real(1), Real(0)};
                // Quadrilateral face centers
                case 15: return {Real(0.5), Real(0.0), Real(0.0)};
                case 16: return {Real(0.5), Real(0.5), Real(0.0)};
                case 17: return {Real(0.0), Real(0.5), Real(0.0)};
                default: break;
            }
            break;

        // 3D Pyramid Elements
        case ElementType::Pyramid5:
            switch (local_node) {
                case 0: return {Real(-1), Real(-1), Real(0)};
                case 1: return {Real(+1), Real(-1), Real(0)};
                case 2: return {Real(+1), Real(+1), Real(0)};
                case 3: return {Real(-1), Real(+1), Real(0)};
                case 4: return {Real(0), Real(0), Real(1)};
                default: break;
            }
            break;

        case ElementType::Pyramid13:
            switch (local_node) {
                // Base corners
                case 0: return {Real(-1), Real(-1), Real(0)};
                case 1: return {Real(+1), Real(-1), Real(0)};
                case 2: return {Real(+1), Real(+1), Real(0)};
                case 3: return {Real(-1), Real(+1), Real(0)};
                // Apex
                case 4: return {Real(0), Real(0), Real(1)};
                // Base mid-edges
                case 5: return {Real(0), Real(-1), Real(0)};
                case 6: return {Real(+1), Real(0), Real(0)};
                case 7: return {Real(0), Real(+1), Real(0)};
                case 8: return {Real(-1), Real(0), Real(0)};
                // Mid-edges to apex
                case 9:  return {Real(-0.5), Real(-0.5), Real(0.5)};
                case 10: return {Real(+0.5), Real(-0.5), Real(0.5)};
                case 11: return {Real(+0.5), Real(+0.5), Real(0.5)};
                case 12: return {Real(-0.5), Real(+0.5), Real(0.5)};
                default: break;
            }
            break;

        case ElementType::Pyramid14:
            switch (local_node) {
                // Base corners (same as Pyramid5)
                case 0: return {Real(-1), Real(-1), Real(0)};
                case 1: return {Real(+1), Real(-1), Real(0)};
                case 2: return {Real(+1), Real(+1), Real(0)};
                case 3: return {Real(-1), Real(+1), Real(0)};
                // Apex
                case 4: return {Real(0), Real(0), Real(1)};
                // Base mid-edges (same as Pyramid13)
                case 5: return {Real(0), Real(-1), Real(0)};
                case 6: return {Real(+1), Real(0), Real(0)};
                case 7: return {Real(0), Real(+1), Real(0)};
                case 8: return {Real(-1), Real(0), Real(0)};
                // Mid-edges to apex (same as Pyramid13)
                case 9:  return {Real(-0.5), Real(-0.5), Real(0.5)};
                case 10: return {Real(+0.5), Real(-0.5), Real(0.5)};
                case 11: return {Real(+0.5), Real(+0.5), Real(0.5)};
                case 12: return {Real(-0.5), Real(+0.5), Real(0.5)};
                // Base center
                case 13: return {Real(0), Real(0), Real(0)};
                default: break;
            }
            break;

        default:
            break;
    }

    throw FEException("Invalid element type or node index in NodeOrdering::get_node_coords",
                      __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
}

std::size_t NodeOrdering::num_nodes(ElementType elem_type) {
    switch (elem_type) {
        case ElementType::Line2:      return 2;
        case ElementType::Line3:      return 3;
        case ElementType::Triangle3:  return 3;
        case ElementType::Triangle6:  return 6;
        case ElementType::Quad4:      return 4;
        case ElementType::Quad8:      return 8;
        case ElementType::Quad9:      return 9;
        case ElementType::Tetra4:     return 4;
        case ElementType::Tetra10:    return 10;
        case ElementType::Hex8:       return 8;
        case ElementType::Hex20:      return 20;
        case ElementType::Hex27:      return 27;
        case ElementType::Wedge6:     return 6;
        case ElementType::Wedge15:    return 15;
        case ElementType::Wedge18:    return 18;
        case ElementType::Pyramid5:   return 5;
        case ElementType::Pyramid13:  return 13;
        case ElementType::Pyramid14:  return 14;
        default:
            throw FEException("Unknown element type in NodeOrdering::num_nodes",
                              __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
    }
}

bool NodeOrdering::is_simplex(ElementType elem_type) {
    switch (elem_type) {
        case ElementType::Triangle3:
        case ElementType::Triangle6:
        case ElementType::Tetra4:
        case ElementType::Tetra10:
            return true;
        default:
            return false;
    }
}

bool NodeOrdering::is_tensor_product(ElementType elem_type) {
    switch (elem_type) {
        case ElementType::Line2:
        case ElementType::Line3:
        case ElementType::Quad4:
        case ElementType::Quad8:
        case ElementType::Quad9:
        case ElementType::Hex8:
        case ElementType::Hex20:
        case ElementType::Hex27:
            return true;
        default:
            return false;
    }
}

} // namespace basis
} // namespace FE
} // namespace svmp

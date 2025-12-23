/* Copyright (c) Stanford University, The Regents of the University of
 * California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#include "Elements/ReferenceElement.h"

#include "Core/FEException.h"
#include "Quadrature/QuadratureFactory.h"

namespace svmp {
namespace FE {
namespace elements {

using svmp::FE::basis::NodeOrdering;
using svmp::FE::quadrature::QuadratureFactory;
using svmp::FE::QuadratureType;

namespace {

ElementType canonical(ElementType type) {
    switch (type) {
        case ElementType::Line3:     return ElementType::Line2;
        case ElementType::Triangle6: return ElementType::Triangle3;
        case ElementType::Quad8:
        case ElementType::Quad9:     return ElementType::Quad4;
        case ElementType::Tetra10:   return ElementType::Tetra4;
        case ElementType::Hex20:
        case ElementType::Hex27:     return ElementType::Hex8;
        case ElementType::Wedge15:
        case ElementType::Wedge18:   return ElementType::Wedge6;
        case ElementType::Pyramid13:
        case ElementType::Pyramid14: return ElementType::Pyramid5;
        default:                     return type;
    }
}

Real compute_reference_measure(ElementType type) {
    // Use a low-order Gauss rule on the canonical element to obtain the
    // reference measure in a robust, implementation-consistent way.
    ElementType base = canonical(type);
    if (base == ElementType::Point1) {
        return Real(1);
    }
    auto quad = QuadratureFactory::create(base, 1, QuadratureType::GaussLegendre, /*use_cache=*/false);
    return quad->reference_measure();
}

void build_edges_and_faces(ElementType base,
                           std::vector<std::vector<LocalIndex>>& edges,
                           std::vector<std::vector<LocalIndex>>& faces) {
    edges.clear();
    faces.clear();

    switch (base) {
        case ElementType::Line2:
            // Two boundary points
            faces = {{0}, {1}};
            break;

        case ElementType::Triangle3:
            // Edges: (0-1), (1-2), (2-0)
            edges = {{0,1}, {1,2}, {2,0}};
            faces = edges; // 2D: "faces" are boundary edges (codimension-1 facets)
            break;

        case ElementType::Quad4:
            // Edges: (0-1), (1-2), (2-3), (3-0)
            edges = {{0,1}, {1,2}, {2,3}, {3,0}};
            faces = edges; // 2D: "faces" are boundary edges (codimension-1 facets)
            break;

        case ElementType::Tetra4:
            // Edges
            edges = {
                {0,1}, {1,2}, {2,0},
                {0,3}, {1,3}, {2,3}
            };
            // Faces: triangles
            faces = {
                {0,1,2},
                {0,1,3},
                {1,2,3},
                {0,2,3}
            };
            break;

        case ElementType::Hex8:
            // Edges (12)
            edges = {
                {0,1}, {1,2}, {2,3}, {3,0}, // bottom
                {4,5}, {5,6}, {6,7}, {7,4}, // top
                {0,4}, {1,5}, {2,6}, {3,7}  // vertical
            };
            // Faces (6 quads)
            faces = {
                {0,1,2,3}, // bottom z=-1
                {4,5,6,7}, // top    z=+1
                {0,1,5,4}, // front
                {1,2,6,5}, // right
                {2,3,7,6}, // back
                {3,0,4,7}  // left
            };
            break;

        case ElementType::Wedge6:
            // Edges (9)
            edges = {
                {0,1}, {1,2}, {2,0}, // bottom tri
                {3,4}, {4,5}, {5,3}, // top tri
                {0,3}, {1,4}, {2,5}  // vertical
            };
            // Faces: 2 triangles + 3 quads
            faces = {
                {0,1,2},       // bottom
                {3,4,5},       // top
                {0,1,4,3},     // quad
                {1,2,5,4},     // quad
                {2,0,3,5}      // quad
            };
            break;

        case ElementType::Pyramid5:
            // Edges (8): base + vertical
            edges = {
                {0,1}, {1,2}, {2,3}, {3,0}, // base
                {0,4}, {1,4}, {2,4}, {3,4}  // vertical
            };
            // Faces: quad base + 4 triangles
            faces = {
                {0,1,2,3},   // base
                {0,1,4},
                {1,2,4},
                {2,3,4},
                {3,0,4}
            };
            break;

        case ElementType::Point1:
        default:
            // No edges/faces for a point; other unsupported types should be
            // handled by the caller.
            break;
    }
}

} // anonymous namespace

const std::vector<LocalIndex>& ReferenceElement::edge_nodes(std::size_t edge_id) const {
    if (edge_id >= edges_.size()) {
        throw FEException("ReferenceElement: edge_id out of range",
                          __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
    }
    return edges_[edge_id];
}

const std::vector<LocalIndex>& ReferenceElement::face_nodes(std::size_t face_id) const {
    if (face_id >= faces_.size()) {
        throw FEException("ReferenceElement: face_id out of range",
                          __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
    }
    return faces_[face_id];
}

ReferenceElement ReferenceElement::create(ElementType type) {
    ReferenceElement ref;
    ref.type_ = type;
    ref.dimension_ = element_dimension(type);
    if (ref.dimension_ < 0) {
        throw FEException("ReferenceElement: unknown element dimension",
                          __FILE__, __LINE__, __func__, FEStatus::InvalidElement);
    }

    // Node count comes from NodeOrdering
    ref.num_nodes_ = basis::NodeOrdering::num_nodes(type);

    // Build topology based on canonical linear element
    const ElementType base = canonical(type);
    build_edges_and_faces(base, ref.edges_, ref.faces_);

    // Reference measure is shared between high-order variants and their base
    ref.reference_measure_ = compute_reference_measure(type);
    return ref;
}

} // namespace elements
} // namespace FE
} // namespace svmp

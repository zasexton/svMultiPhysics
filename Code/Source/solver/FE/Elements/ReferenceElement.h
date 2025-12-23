/* Copyright (c) Stanford University, The Regents of the University of
 * California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#ifndef SVMP_FE_ELEMENTS_REFERENCEELEMENT_H
#define SVMP_FE_ELEMENTS_REFERENCEELEMENT_H

/**
 * @file ReferenceElement.h
 * @brief Reference element topology and geometric properties
 *
 * This utility provides immutable metadata for reference elements:
 *  - spatial dimension
 *  - number of nodes
 *  - edge and face connectivity in local node indices
 *  - reference measure (length/area/volume)
 *
 * It is purely FE-level and does not depend on the Mesh library. Node
 * coordinates are obtained from `basis::NodeOrderingConventions`.
 */

#include "Core/Types.h"
#include "Basis/NodeOrderingConventions.h"

#include <vector>

namespace svmp {
namespace FE {
namespace elements {

/**
 * @brief Topological and geometric metadata for a reference element
 */
class ReferenceElement {
public:
    ReferenceElement() = default;

    ElementType type() const noexcept { return type_; }
    int dimension() const noexcept { return dimension_; }

    /// Total number of nodes for this element type
    std::size_t num_nodes() const noexcept { return num_nodes_; }

    /// Number of topological edges (1D sub-entities)
    std::size_t num_edges() const noexcept { return edges_.size(); }

    /// Number of topological faces (codimension-1 sub-entities)
    std::size_t num_faces() const noexcept { return faces_.size(); }

    /// Reference measure: length (1D), area (2D), or volume (3D)
    Real reference_measure() const noexcept { return reference_measure_; }

    /// Node indices for a given edge (0 <= edge_id < num_edges())
    const std::vector<LocalIndex>& edge_nodes(std::size_t edge_id) const;

    /// Node indices for a given face (0 <= face_id < num_faces())
    const std::vector<LocalIndex>& face_nodes(std::size_t face_id) const;

    /**
     * @brief Factory: build metadata for the given element type
     *
     * High-order variants (e.g., Triangle6, Tetra10, Hex27) share the same
     * topological connectivity and reference measure as their linear parent
     * while exposing the correct `num_nodes()` through the FE Basis layer.
     */
    static ReferenceElement create(ElementType type);

private:
    ElementType type_{ElementType::Unknown};
    int dimension_{-1};
    std::size_t num_nodes_{0};
    Real reference_measure_{Real(0)};

    std::vector<std::vector<LocalIndex>> edges_;
    std::vector<std::vector<LocalIndex>> faces_;
};

} // namespace elements
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_ELEMENTS_REFERENCEELEMENT_H


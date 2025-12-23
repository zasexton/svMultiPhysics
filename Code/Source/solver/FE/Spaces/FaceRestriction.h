/* Copyright (c) Stanford University, The Regents of the
 * University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#ifndef SVMP_FE_SPACES_FACERESTRICTION_H
#define SVMP_FE_SPACES_FACERESTRICTION_H

/**
 * @file FaceRestriction.h
 * @brief Face DOF restriction and scatter operators
 *
 * This module provides utilities for extracting and scattering DOF values
 * between elements and their faces/edges/vertices. These operations are
 * essential for:
 *  - DG/HDG methods that operate on face DOFs
 *  - Trace spaces that restrict to boundaries
 *  - Flux assembly across element interfaces
 *  - Boundary condition enforcement
 *
 * The restriction operator extracts DOF values from an element that live
 * on a specific face. The scatter operator does the reverse, distributing
 * face DOF values back to the element.
 */

#include "Core/Types.h"
#include "Spaces/FunctionSpace.h"
#include <vector>
#include <map>
#include <memory>

namespace svmp {
namespace FE {
namespace spaces {

/**
 * @brief Describes the topology of faces/edges/vertices for an element type
 */
struct ElementTopology {
    int num_vertices;
    int num_edges;
    int num_faces;
    int dimension;

    // Face topology: for each face, list of local vertex indices (CCW orientation)
    std::vector<std::vector<int>> face_vertices;

    // Edge topology: for each edge, pair of local vertex indices
    std::vector<std::pair<int, int>> edge_vertices;

    // Face types for 3D elements (triangle or quad faces)
    std::vector<ElementType> face_types;
};

/**
 * @brief Face restriction operator for extracting/scattering DOFs
 *
 * This class provides mappings between element DOFs and face/edge/vertex DOFs
 * for various element types and polynomial orders. The mappings depend on:
 *  - Element type (triangle, quad, tet, hex, etc.)
 *  - Polynomial order
 *
 * Global continuity/DOF sharing is handled by higher-level modules; this class
 * only provides element-local index mappings.
 */
class FaceRestriction {
public:
    /**
     * @brief Construct face restriction for a function space
     *
     * @param space The function space to restrict
     */
    explicit FaceRestriction(const FunctionSpace& space);

    /**
     * @brief Construct face restriction for element type and order
     *
     * @param elem_type Element type
     * @param polynomial_order Polynomial order
     * @param continuity Continuity classification (stored for bookkeeping)
     */
    FaceRestriction(ElementType elem_type, int polynomial_order,
                    Continuity continuity = Continuity::C0);

    /**
     * @brief Get local DOF indices that live on a face
     *
     * Returns the indices of DOFs from the element that are associated
     * with the specified face. For nodal bases, this includes all DOFs whose
     * reference-node coordinates lie on that face (vertices, edge nodes, and
     * face nodes when present). This mapping is purely element-local and does
     * not depend on global continuity/DOF sharing.
     *
     * @param face_id Local face index (0 to num_faces-1)
     * @return Vector of local DOF indices on that face
     */
    std::vector<int> face_dofs(int face_id) const;

    /**
     * @brief Get local DOF indices on an edge (for 3D elements)
     *
     * @param edge_id Local edge index (0 to num_edges-1)
     * @return Vector of local DOF indices on that edge
     */
    std::vector<int> edge_dofs(int edge_id) const;

    /**
     * @brief Get local DOF indices at a vertex
     *
     * @param vertex_id Local vertex index (0 to num_vertices-1)
     * @return Vector of local DOF indices at that vertex (typically 0 or 1)
     */
    std::vector<int> vertex_dofs(int vertex_id) const;

    /**
     * @brief Get interior DOF indices (not on any face/edge/vertex)
     *
     * @return Vector of interior DOF indices
     */
    std::vector<int> interior_dofs() const;

    /**
     * @brief Extract face values from element DOF values
     *
     * @param element_values All DOF values on element
     * @param face_id Local face index
     * @return DOF values restricted to face
     */
    std::vector<Real> restrict_to_face(
        const std::vector<Real>& element_values,
        int face_id) const;

    /**
     * @brief Extract edge values from element DOF values
     *
     * @param element_values All DOF values on element
     * @param edge_id Local edge index
     * @return DOF values restricted to edge
     */
    std::vector<Real> restrict_to_edge(
        const std::vector<Real>& element_values,
        int edge_id) const;

    /**
     * @brief Scatter face values back to element
     *
     * Adds face DOF values to the corresponding positions in the element
     * DOF vector. Does not zero out element_values first.
     *
     * @param face_values DOF values on face
     * @param face_id Local face index
     * @param[in,out] element_values Element DOF vector to scatter into
     */
    void scatter_from_face(
        const std::vector<Real>& face_values,
        int face_id,
        std::vector<Real>& element_values) const;

    /**
     * @brief Scatter edge values back to element
     *
     * @param edge_values DOF values on edge
     * @param edge_id Local edge index
     * @param[in,out] element_values Element DOF vector to scatter into
     */
    void scatter_from_edge(
        const std::vector<Real>& edge_values,
        int edge_id,
        std::vector<Real>& element_values) const;

    /**
     * @brief Get number of DOFs on a face
     *
     * @param face_id Local face index
     * @return Number of DOFs on that face
     */
    std::size_t num_face_dofs(int face_id) const;

    /**
     * @brief Get number of DOFs on an edge
     *
     * @param edge_id Local edge index
     * @return Number of DOFs on that edge
     */
    std::size_t num_edge_dofs(int edge_id) const;

    /**
     * @brief Get total number of DOFs per element
     */
    std::size_t num_element_dofs() const { return num_dofs_; }

    /**
     * @brief Reference-space coordinates of element DOF nodes
     *
     * These are the nodal coordinates used internally to classify DOFs as
     * belonging to faces/edges/vertices. For nodal scalar bases, these are
     * the reference coordinates of each DOF (one coordinate per DOF index).
     */
    const std::vector<math::Vector<Real, 3>>& dof_nodes() const noexcept { return dof_nodes_; }

    /**
     * @brief Get the element topology
     */
    const ElementTopology& topology() const { return topology_; }

    /**
     * @brief Get the element type
     */
    ElementType element_type() const { return elem_type_; }

    /**
     * @brief Get the polynomial order
     */
    int polynomial_order() const { return order_; }

    /**
     * @brief Get the continuity type
     */
    Continuity continuity() const { return continuity_; }

    /**
     * @brief Check if the element type is supported
     */
    static bool is_supported(ElementType elem_type);

private:
    void initialize_topology();
    void compute_dof_maps();

    ElementType elem_type_;
    int order_;
    Continuity continuity_;
    std::size_t num_dofs_;

    ElementTopology topology_;

    // DOF maps: face_id -> list of local DOF indices
    std::vector<std::vector<int>> face_dof_maps_;

    // DOF maps: edge_id -> list of local DOF indices
    std::vector<std::vector<int>> edge_dof_maps_;

    // DOF maps: vertex_id -> list of local DOF indices (typically single DOF)
    std::vector<std::vector<int>> vertex_dof_maps_;

    // Interior DOFs (not on any face/edge/vertex)
    std::vector<int> interior_dof_map_;

    // Reference-space coordinates for each local DOF index.
    std::vector<math::Vector<Real, 3>> dof_nodes_;
};

/**
 * @brief Factory for creating FaceRestriction objects
 */
class FaceRestrictionFactory {
public:
    /**
     * @brief Get or create a FaceRestriction for given parameters
     *
     * Uses a cache to avoid recomputing DOF maps for the same configuration.
     *
     * @param elem_type Element type
     * @param order Polynomial order
     * @param continuity Continuity type
     * @return Shared pointer to FaceRestriction
     */
    static std::shared_ptr<const FaceRestriction> get(
        ElementType elem_type,
        int order,
        Continuity continuity = Continuity::C0);

    /**
     * @brief Get or create a FaceRestriction for a function space
     *
     * @param space The function space
     * @return Shared pointer to FaceRestriction
     */
    static std::shared_ptr<const FaceRestriction> get(const FunctionSpace& space);

    /**
     * @brief Clear the cache
     */
    static void clear_cache();

private:
    struct CacheKey {
        ElementType elem_type;
        int order;
        Continuity continuity;

        bool operator<(const CacheKey& other) const;
    };

    static std::map<CacheKey, std::shared_ptr<const FaceRestriction>> cache_;
};

} // namespace spaces
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_SPACES_FACERESTRICTION_H

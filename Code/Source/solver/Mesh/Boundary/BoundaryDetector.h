/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject
 * to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
 * IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
 * TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
 * PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER
 * OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef SVMP_BOUNDARY_DETECTOR_H
#define SVMP_BOUNDARY_DETECTOR_H

#include "../Core/MeshTypes.h"
#include "../Core/MeshBase.h"
#include "../Topology/CellShape.h"
#include "../Topology/CellTopology.h"
#include "BoundaryKey.h"
#include "BoundaryComponent.h"

#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <map>
#include <memory>
#include <array>

namespace svmp {

/**
 * @brief Topological boundary detection for finite cell complexes
 *
 * Detects the boundary of an n-dimensional mesh as the set of (n-1)-faces
 * incident to exactly one n-cell. Implements both:
 *
 * 1. Incidence counting (manifold-agnostic)
 *    - Count how many n-cells reference each (n-1)-face
 *    - Boundary faces: count = 1
 *    - Interior manifold faces: count = 2
 *    - Non-manifold seams: count > 2
 *
 * 2. Chain complex view (over Z2)
 *    - Build cell-face incidence matrix A_n
 *    - Compute b = (A_n mod 2) * 1 mod 2
 *    - Boundary faces have b_i = 1
 *
 * Features:
 * - Mixed element mesh support
 * - Non-manifold detection
 * - Connected component extraction
 * - Periodic mesh handling
 */
class BoundaryDetector {
public:
    /**
     * @brief Boundary (n-1)-face incidence information
     *
     * Tracks how many n-cells share each (n-1)-boundary entity.
     * The term "boundary" is dimension-agnostic:
     * - In 3D: boundaries are faces (2D surfaces)
     * - In 2D: boundaries are edges (1D curves)
     * - In 1D: boundaries are vertices (0D points)
     */
    struct BoundaryIncidence {
        BoundaryKey key;                                // Canonical boundary representation
        std::vector<index_t> incident_cells;            // Cells sharing this boundary
        std::vector<std::vector<index_t>> oriented_vertices;  // Oriented vertex lists (one per incident cell)
        int count = 0;                                  // Number of incident cells

        bool is_boundary() const { return count == 1; }
        bool is_interior() const { return count == 2; }
        bool is_nonmanifold() const { return count > 2; }

        // Get oriented vertices for boundary (outward-pointing normal convention)
        std::vector<index_t> boundary_orientation() const {
            if (is_boundary() && !oriented_vertices.empty()) {
                return oriented_vertices[0];
            }
            return key.vertices();  // Fallback to canonical
        }
    };

    /**
     * @brief Boundary detection result
     */
    struct BoundaryInfo {
        // Generic (n-1)-entity indices grouped by classification
        std::vector<index_t> boundary_entities;                   // Boundary (n-1)-entities (incidence = 1)
        std::vector<index_t> interior_entities;                   // Interior (n-1)-entities (incidence = 2)
        std::vector<index_t> nonmanifold_entities;                // Non-manifold (n-1)-entities (incidence > 2)

        // Type of each boundary entity (aligned with boundary_entities)
        // Vertex for 1D, Edge for 2D, Face for 3D
        std::vector<EntityKind> boundary_types;

        // Vertex indices that lie on any boundary (helpful for BCs)
        std::unordered_set<index_t> boundary_vertices;

        // Connected components of the boundary graph
        std::vector<BoundaryComponent> components;

        // Oriented vertex lists for boundary entities (RHR in 3D / CCW in 2D)
        std::vector<std::vector<index_t>> oriented_boundary_entities;

        bool has_boundary() const { return !boundary_entities.empty(); }
        bool is_closed() const { return boundary_entities.empty(); }
        bool has_nonmanifold() const { return !nonmanifold_entities.empty(); }
        size_t n_components() const { return components.size(); }
    };

    // ---- Construction ----
    explicit BoundaryDetector(const MeshBase& mesh);

    // ---- Main detection methods ----

    /**
     * @brief Detect topological boundary using incidence counting
     * @return Boundary information including faces, vertices, and components
     */
    BoundaryInfo detect_boundary();

    /**
     * @brief Detect boundary using chain complex approach (mod 2)
     * @return Boundary face indices
     */
    std::vector<index_t> detect_boundary_chain_complex();

    /**
     * @brief Get boundary incidence counts for all (n-1)-boundaries
     * @return Map from boundary key to incidence information
     */
    std::unordered_map<BoundaryKey, BoundaryIncidence, BoundaryKey::Hash> compute_boundary_incidence();

    /**
     * @brief Extract connected components of the boundary
     * @param boundary_faces Indices of boundary faces
     * @return Vector of connected boundary components
     */
    std::vector<BoundaryComponent> extract_boundary_components(
        const std::vector<index_t>& boundary_entities);

    // ---- Utilities ----

    /**
     * @brief Extract (n-1)-faces of a single n-cell
     * @param cell_id Cell index
     * @return Vector of boundary keys for this cell
     */
    // Extract (n-1) boundary entities incident to a given n-cell (codimension-1)
    std::vector<BoundaryKey> extract_cell_codim1(index_t cell_id) const;

    /**
     * @brief Extract (n-2)-faces (edges) of an (n-1)-face
     * @param face_vertices Vertices defining the face
     * @return Vector of edge keys
     */
    // Extract (n-2) sub-entities from an (n-1) boundary entity (codimension-2)
    std::vector<BoundaryKey> extract_codim2_from_codim1(const std::vector<index_t>& entity_vertices) const;

    /**
     * @brief Check if mesh is topologically closed (no boundary)
     * @return true if all faces have even incidence
     */
    bool is_closed_mesh();

    /**
     * @brief Detect and report non-manifold features
     * @return Indices of non-manifold faces (incidence > 2)
     */
    // Find non-manifold (n-1) entities (incidence > 2)
    std::vector<index_t> detect_nonmanifold_codim1();


    // ---- Accessors ----
    const MeshBase& mesh() const { return mesh_; }

private:
    const MeshBase& mesh_;

    // Helper for extracting edges from 2D/3D faces
    // Helper: extract (n-2) sub-entities from a cyclic (ring) list of vertices
    std::vector<BoundaryKey> extract_codim2_from_ring(const std::vector<index_t>& vertices) const;

    // BFS for connected components
    void bfs_boundary_component(
        index_t start_entity,
        const std::vector<index_t>& boundary_entities,
        const std::unordered_map<BoundaryKey, std::vector<index_t>, BoundaryKey::Hash>& sub_to_entities,
        std::unordered_set<index_t>& visited,
        BoundaryComponent& component) const;
};

} // namespace svmp

#endif // SVMP_BOUNDARY_DETECTOR_H

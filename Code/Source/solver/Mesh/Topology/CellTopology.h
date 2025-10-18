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

#ifndef SVMP_CELL_TOPOLOGY_H
#define SVMP_CELL_TOPOLOGY_H

#include "../Core/MeshTypes.h"
#include "CellShape.h"
#include <vector>
#include <array>
#include <stdexcept>

namespace svmp {

/**
 * @brief Canonical topological connectivity information for standard cell types
 *
 * This class provides reference topology for standard finite element cells:
 * - Face definitions (which local vertex indices form each face)
 * - Edge definitions (which local vertex indices form each edge)
 * - Oriented face/edge definitions (for outward normals via right-hand rule)
 *
 * **Design Philosophy:**
 * All topological connectivity information for cell types is centralized here.
 * Adding a new cell type requires only updating this class, not modifying
 * every algorithm that uses cell topology.
 *
 * **Vertex Ordering Convention:**
 * All definitions use local vertex indices [0, 1, 2, ...] referring to
 * positions in a cell's vertex array.
 *
 * **Orientation Convention (Right-Hand Rule):**
 * Oriented faces are ordered such that the cross product of the first two
 * edge vectors points outward from the cell interior.
 */
class CellTopology {
public:
    // Lightweight zero-allocation views over topology tables
    struct FaceListView {
        const index_t* indices = nullptr;   // flattened vertex indices
        const int* offsets = nullptr;       // size = face_count + 1
        int face_count = 0;
    };

    struct EdgeListView {
        const index_t* pairs_flat = nullptr; // flattened [v0, v1, v0, v1, ...]
        int edge_count = 0;
    };

    /**
     * @brief Get canonical (n-1)-face definitions for a cell type
     *
     * Returns faces with vertices in canonical (sorted) order for topology detection.
     *
     * @param family Cell type
     * @return Vector of faces, each face is a vector of local vertex indices
     * @throws std::runtime_error if cell type not supported
     */
    static std::vector<std::vector<index_t>> get_boundary_faces(CellFamily family);

    /**
     * @brief Get oriented (n-1)-face definitions for a cell type
     *
     * Returns faces with vertices ordered per right-hand rule (outward normals).
     *
     * @param family Cell type
     * @return Vector of faces, each face is a vector of local vertex indices
     * @throws std::runtime_error if cell type not supported
     */
    static std::vector<std::vector<index_t>> get_oriented_boundary_faces(CellFamily family);

    // Zero-allocation oriented face view for fixed families (Tet/Hex/Tri/Quad/Wedge/Pyramid)
    static FaceListView get_oriented_boundary_faces_view(CellFamily family);

    // Zero-allocation canonical (sorted within-face) face view for fixed families
    // Canonical views are useful for topology detection (orientation-independent keys)
    static FaceListView get_boundary_faces_canonical_view(CellFamily family);

    /**
     * @brief Get edge definitions for a cell type
     *
     * @param family Cell type
     * @return Vector of edges, each edge is a pair of local vertex indices
     * @throws std::runtime_error if cell type not supported
     */
    static std::vector<std::array<index_t, 2>> get_edges(CellFamily family);

    // Zero-allocation edge view for fixed families (Tet/Hex/Tri/Quad/Wedge/Pyramid)
    static EdgeListView get_edges_view(CellFamily family);

    // ----------------------------
    // High-order (p>2) scaffolding
    // ----------------------------
    enum class HighOrderKind { Lagrange, Serendipity };

    enum class HONodeRole { Corner, Edge, Face, Volume };

    struct HighOrderNodeRole {
        HONodeRole role;
        // For Corner: idx0=local corner id
        // For Edge:   idx0=edge index, idx1=step index (1..p-1), idx2=total steps (p-1)
        // For Face:   idx0=face index, idx1=i (1..p-1), idx2=j (Tri: 1..p-1-i; Quad: 1..p-1)
        // For Volume: idx0=i, idx1=j, idx2=k (1..p-1)
        int idx0 = 0, idx1 = 0, idx2 = 0;
    };

    struct HighOrderVTKPattern {
        HighOrderKind kind = HighOrderKind::Lagrange;
        int order = 2;
        std::vector<HighOrderNodeRole> sequence; // in VTK expected order
    };

    // Returns an abstract description of VTK's high-order node ordering for a given family/order/kind.
    // This does not include global vertex IDs — it provides the structural sequence (corners/edges/faces/volume).
    static HighOrderVTKPattern vtk_high_order_pattern(CellFamily family, int p, HighOrderKind kind = HighOrderKind::Lagrange);

    // Helper: infer Lagrange order p from total node count (best-effort). Returns -1 if unknown.
    static int infer_lagrange_order(CellFamily family, size_t node_count);
    // Helper: infer Serendipity order p from total node count (best-effort). Returns -1 if unknown.
    static int infer_serendipity_order(CellFamily family, size_t node_count);

    // ----------------------
    // Variable-arity families
    // ----------------------
    // Polygon (2D): vertices 0..m-1 (CCW). Faces = edges in CCW.
    // Views use the following vertex indexing conventions for derived 3D cells:
    //  - Prism(m): base 0..m-1, top m..2m-1 (paired i <-> i+m)
    //  - Pyramid(m): base 0..m-1, apex m
    // Oriented faces follow right-hand rule and outward normals.

    // Polygon (n-gon) oriented/canonical face and edge views
    static FaceListView get_polygon_faces_view(int m);
    static FaceListView get_polygon_faces_canonical_view(int m);
    static EdgeListView get_polygon_edges_view(int m);

    // m-gonal Prism oriented/canonical faces and edges
    static FaceListView get_prism_faces_view(int m);
    static FaceListView get_prism_faces_canonical_view(int m);
    static EdgeListView get_prism_edges_view(int m);

    // m-gonal Pyramid oriented/canonical faces and edges
    static FaceListView get_pyramid_faces_view(int m);
    static FaceListView get_pyramid_faces_canonical_view(int m);
    static EdgeListView get_pyramid_edges_view(int m);

    // ----------------------------
    // Higher-order node ordering
    // ----------------------------
    // Helpers to assemble quadratic element node ordering using the views above.
    // Ordering: corners -> edge midpoints (edge view order) -> face midpoints (face view order) -> body center (optional)
    struct HighOrderOrdering {
        // Fixed families
        static std::vector<index_t> assemble_quadratic(
            CellFamily family,
            const std::vector<index_t>& corners,
            const std::vector<index_t>& edge_mids,
            const std::vector<index_t>& face_mids = {},
            bool include_center = false,
            index_t center = 0);

        // Variable-arity families
        static std::vector<index_t> assemble_quadratic_polygon(
            int m,
            const std::vector<index_t>& corners,
            const std::vector<index_t>& edge_mids);

        static std::vector<index_t> assemble_quadratic_prism(
            int m,
            const std::vector<index_t>& corners,           // size = 2m (base then top)
            const std::vector<index_t>& edge_mids,         // size = 3m
            const std::vector<index_t>& face_mids = {},    // size = 2 + m (optional)
            bool include_center = false,
            index_t center = 0);

        static std::vector<index_t> assemble_quadratic_pyramid(
            int m,
            const std::vector<index_t>& corners,           // size = m+1 (base then apex)
            const std::vector<index_t>& edge_mids,         // size = m + m
            const std::vector<index_t>& face_mids = {},    // size = 1 + m (optional)
            bool include_center = false,
            index_t center = 0);
    };

    /**
     * @brief Get face-to-edge connectivity for a cell type
     *
     * Returns which edges form each face (useful for face traversal).
     *
     * @param family Cell type
     * @return Vector where element i contains edge indices forming face i
     * @throws std::runtime_error if cell type not supported
     */
    static std::vector<std::vector<int>> get_face_edges(CellFamily family);

private:
    // ---- 3D cell face definitions ----
    static std::vector<std::vector<index_t>> tet_faces_canonical();
    static std::vector<std::vector<index_t>> tet_faces_oriented();

    static std::vector<std::vector<index_t>> hex_faces_canonical();
    static std::vector<std::vector<index_t>> hex_faces_oriented();

    static std::vector<std::vector<index_t>> wedge_faces_canonical();
    static std::vector<std::vector<index_t>> wedge_faces_oriented();

    static std::vector<std::vector<index_t>> pyramid_faces_canonical();
    static std::vector<std::vector<index_t>> pyramid_faces_oriented();

    // ---- 2D cell edge definitions (boundary of 2D cells) ----
    static std::vector<std::vector<index_t>> tri_edges_canonical();
    static std::vector<std::vector<index_t>> tri_edges_oriented();

    static std::vector<std::vector<index_t>> quad_edges_canonical();
    static std::vector<std::vector<index_t>> quad_edges_oriented();

    // ---- 3D cell edge definitions ----
    static std::vector<std::array<index_t, 2>> tet_edges();
    static std::vector<std::array<index_t, 2>> hex_edges();
    static std::vector<std::array<index_t, 2>> wedge_edges();
    static std::vector<std::array<index_t, 2>> pyramid_edges();

    // ---- 2D cell edge definitions ----
    static std::vector<std::array<index_t, 2>> tri_edges();
    static std::vector<std::array<index_t, 2>> quad_edges();
};

} // namespace svmp

#endif // SVMP_CELL_TOPOLOGY_H

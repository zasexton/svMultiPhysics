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

    /**
     * @brief Get edge definitions for a cell type
     *
     * @param family Cell type
     * @return Vector of edges, each edge is a pair of local vertex indices
     * @throws std::runtime_error if cell type not supported
     */
    static std::vector<std::array<index_t, 2>> get_edges(CellFamily family);

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

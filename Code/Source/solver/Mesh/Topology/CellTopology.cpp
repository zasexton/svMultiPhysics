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

#include "CellTopology.h"
#include <algorithm>

namespace svmp {

// ==========================================
// Public Interface
// ==========================================

std::vector<std::vector<index_t>> CellTopology::get_boundary_faces(CellFamily family) {
    switch (family) {
        case CellFamily::Tetra:
            return tet_faces_canonical();
        case CellFamily::Hex:
            return hex_faces_canonical();
        case CellFamily::Wedge:
            return wedge_faces_canonical();
        case CellFamily::Pyramid:
            return pyramid_faces_canonical();
        case CellFamily::Triangle:
            return tri_edges_canonical();
        case CellFamily::Quad:
            return quad_edges_canonical();
        default:
            throw std::runtime_error("Unsupported cell family for boundary face extraction");
    }
}

std::vector<std::vector<index_t>> CellTopology::get_oriented_boundary_faces(CellFamily family) {
    switch (family) {
        case CellFamily::Tetra:
            return tet_faces_oriented();
        case CellFamily::Hex:
            return hex_faces_oriented();
        case CellFamily::Wedge:
            return wedge_faces_oriented();
        case CellFamily::Pyramid:
            return pyramid_faces_oriented();
        case CellFamily::Triangle:
            return tri_edges_oriented();
        case CellFamily::Quad:
            return quad_edges_oriented();
        default:
            throw std::runtime_error("Unsupported cell family for oriented boundary face extraction");
    }
}

std::vector<std::array<index_t, 2>> CellTopology::get_edges(CellFamily family) {
    switch (family) {
        case CellFamily::Tetra:
            return tet_edges();
        case CellFamily::Hex:
            return hex_edges();
        case CellFamily::Wedge:
            return wedge_edges();
        case CellFamily::Pyramid:
            return pyramid_edges();
        case CellFamily::Triangle:
            return tri_edges();
        case CellFamily::Quad:
            return quad_edges();
        default:
            throw std::runtime_error("Unsupported cell family for edge extraction");
    }
}

std::vector<std::vector<int>> CellTopology::get_face_edges(CellFamily family) {
    // Face-to-edge connectivity (which edges bound each face)
    // This is useful for face traversal algorithms
    switch (family) {
        case CellFamily::Tetra:
            return {{0, 1, 2}, {0, 3, 4}, {1, 4, 5}, {2, 3, 5}};
        case CellFamily::Hex:
            return {{0, 1, 2, 3}, {4, 5, 6, 7}, {0, 4, 8, 9}, {2, 6, 10, 11}, {3, 7, 8, 11}, {1, 5, 9, 10}};
        case CellFamily::Triangle:
            return {{0}, {1}, {2}};
        case CellFamily::Quad:
            return {{0}, {1}, {2}, {3}};
        default:
            throw std::runtime_error("Face-to-edge connectivity not implemented for this cell type");
    }
}

// ==========================================
// Tetrahedron (4 vertices: 0,1,2,3)
// ==========================================

std::vector<std::vector<index_t>> CellTopology::tet_faces_canonical() {
    // Faces sorted for canonical comparison
    std::vector<std::vector<index_t>> faces = {
        {0, 1, 2},
        {0, 1, 3},
        {0, 2, 3},
        {1, 2, 3}
    };
    for (auto& face : faces) {
        std::sort(face.begin(), face.end());
    }
    return faces;
}

std::vector<std::vector<index_t>> CellTopology::tet_faces_oriented() {
    // Faces with right-hand rule ordering (outward normals)
    return {
        {0, 2, 1},  // Face opposite vertex 3 (CCW from outside)
        {0, 1, 3},  // Face opposite vertex 2
        {0, 3, 2},  // Face opposite vertex 1
        {1, 2, 3}   // Face opposite vertex 0
    };
}

std::vector<std::array<index_t, 2>> CellTopology::tet_edges() {
    return {
        {0, 1}, {0, 2}, {0, 3},
        {1, 2}, {1, 3}, {2, 3}
    };
}

// ==========================================
// Hexahedron (8 vertices: standard ordering)
// ==========================================

std::vector<std::vector<index_t>> CellTopology::hex_faces_canonical() {
    std::vector<std::vector<index_t>> faces = {
        {0, 1, 2, 3},  // bottom
        {4, 5, 6, 7},  // top
        {0, 1, 5, 4},  // front
        {2, 3, 7, 6},  // back
        {0, 3, 7, 4},  // left
        {1, 2, 6, 5}   // right
    };
    for (auto& face : faces) {
        std::sort(face.begin(), face.end());
    }
    return faces;
}

std::vector<std::vector<index_t>> CellTopology::hex_faces_oriented() {
    // Right-hand rule ordering (outward normals)
    return {
        {0, 3, 2, 1},  // bottom (z-)
        {4, 5, 6, 7},  // top (z+)
        {0, 1, 5, 4},  // front (y-)
        {2, 3, 7, 6},  // back (y+)
        {0, 4, 7, 3},  // left (x-)
        {1, 2, 6, 5}   // right (x+)
    };
}

std::vector<std::array<index_t, 2>> CellTopology::hex_edges() {
    return {
        // Bottom face edges
        {0, 1}, {1, 2}, {2, 3}, {3, 0},
        // Top face edges
        {4, 5}, {5, 6}, {6, 7}, {7, 4},
        // Vertical edges
        {0, 4}, {1, 5}, {2, 6}, {3, 7}
    };
}

// ==========================================
// Wedge/Prism (6 vertices)
// ==========================================

std::vector<std::vector<index_t>> CellTopology::wedge_faces_canonical() {
    std::vector<std::vector<index_t>> faces = {
        {0, 1, 2},            // bottom triangle
        {3, 4, 5},            // top triangle
        {0, 1, 4, 3},         // front quad
        {1, 2, 5, 4},         // right quad
        {2, 0, 3, 5}          // left quad
    };
    for (auto& face : faces) {
        std::sort(face.begin(), face.end());
    }
    return faces;
}

std::vector<std::vector<index_t>> CellTopology::wedge_faces_oriented() {
    return {
        {0, 2, 1},            // bottom triangle
        {3, 4, 5},            // top triangle
        {0, 1, 4, 3},         // front quad
        {1, 2, 5, 4},         // right quad
        {2, 0, 3, 5}          // left quad
    };
}

std::vector<std::array<index_t, 2>> CellTopology::wedge_edges() {
    return {
        // Bottom triangle edges
        {0, 1}, {1, 2}, {2, 0},
        // Top triangle edges
        {3, 4}, {4, 5}, {5, 3},
        // Vertical edges
        {0, 3}, {1, 4}, {2, 5}
    };
}

// ==========================================
// Pyramid (5 vertices: 4-vertex base + apex)
// ==========================================

std::vector<std::vector<index_t>> CellTopology::pyramid_faces_canonical() {
    std::vector<std::vector<index_t>> faces = {
        {0, 1, 2, 3},  // quad base
        {0, 1, 4},     // triangle
        {1, 2, 4},     // triangle
        {2, 3, 4},     // triangle
        {3, 0, 4}      // triangle
    };
    for (auto& face : faces) {
        std::sort(face.begin(), face.end());
    }
    return faces;
}

std::vector<std::vector<index_t>> CellTopology::pyramid_faces_oriented() {
    return {
        {0, 3, 2, 1},  // quad base
        {0, 1, 4},     // triangle
        {1, 2, 4},     // triangle
        {2, 3, 4},     // triangle
        {3, 0, 4}      // triangle
    };
}

std::vector<std::array<index_t, 2>> CellTopology::pyramid_edges() {
    return {
        // Base edges
        {0, 1}, {1, 2}, {2, 3}, {3, 0},
        // Lateral edges to apex
        {0, 4}, {1, 4}, {2, 4}, {3, 4}
    };
}

// ==========================================
// Triangle (3 vertices)
// ==========================================

std::vector<std::vector<index_t>> CellTopology::tri_edges_canonical() {
    std::vector<std::vector<index_t>> edges = {
        {0, 1},
        {1, 2},
        {2, 0}
    };
    for (auto& edge : edges) {
        std::sort(edge.begin(), edge.end());
    }
    return edges;
}

std::vector<std::vector<index_t>> CellTopology::tri_edges_oriented() {
    // CCW ordering for outward normals (2D)
    return {
        {0, 1},
        {1, 2},
        {2, 0}
    };
}

std::vector<std::array<index_t, 2>> CellTopology::tri_edges() {
    return {
        {0, 1}, {1, 2}, {2, 0}
    };
}

// ==========================================
// Quadrilateral (4 vertices)
// ==========================================

std::vector<std::vector<index_t>> CellTopology::quad_edges_canonical() {
    std::vector<std::vector<index_t>> edges = {
        {0, 1},
        {1, 2},
        {2, 3},
        {3, 0}
    };
    for (auto& edge : edges) {
        std::sort(edge.begin(), edge.end());
    }
    return edges;
}

std::vector<std::vector<index_t>> CellTopology::quad_edges_oriented() {
    // CCW ordering for outward normals (2D)
    return {
        {0, 1},
        {1, 2},
        {2, 3},
        {3, 0}
    };
}

std::vector<std::array<index_t, 2>> CellTopology::quad_edges() {
    return {
        {0, 1}, {1, 2}, {2, 3}, {3, 0}
    };
}

} // namespace svmp

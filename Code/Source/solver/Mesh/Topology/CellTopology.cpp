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
#include <cmath>
#include <memory>
#include <unordered_map>

namespace svmp {
namespace {
// Static topology tables for fixed families (prototype: Tet/Hex)

// Tetrahedron: oriented faces (outward, right-hand rule)
// Faces: {1,2,3}, {0,3,2}, {0,1,3}, {0,2,1}
// Reference: VTK tetrahedron corner/face ordering (vtkTetra), VTK File Formats doc.
constexpr index_t TET_FACES_ORIENTED_IDX[] = {
    1,2,3,  0,3,2,  0,1,3,  0,2,1
};
constexpr int TET_FACES_ORIENTED_OFF[] = {0,3,6,9,12};

// Tetrahedron edges (6)
constexpr index_t TET_EDGES_FLAT[] = {
    0,1, 0,2, 0,3, 1,2, 1,3, 2,3
};

// Hexahedron: oriented faces (outward, right-hand rule)
// Chosen to ensure adjacent faces traverse shared edges in opposite directions.
// Faces: bottom (0,3,2,1), top (4,5,6,7), sides (0,1,5,4), (1,2,6,5), (2,3,7,6), (3,0,4,7)
// Reference: VTK hexahedron corner/face ordering (vtkHexahedron), VTK File Formats doc.
constexpr index_t HEX_FACES_ORIENTED_IDX[] = {
    0,3,2,1,  4,5,6,7,  0,1,5,4,  1,2,6,5,  2,3,7,6,  3,0,4,7
};
constexpr int HEX_FACES_ORIENTED_OFF[] = {0,4,8,12,16,20,24};

// Hexahedron edges (12)
constexpr index_t HEX_EDGES_FLAT[] = {
    0,1, 1,2, 2,3, 3,0,
    4,5, 5,6, 6,7, 7,4,
    0,4, 1,5, 2,6, 3,7
};

// =========================
// Triangle (2D cell) faces (edges) oriented CCW
// Reference: VTK triangle (vtkTriangle), edges oriented CCW.
constexpr index_t TRI_FACES_ORIENTED_IDX[] = {
    0,1,  1,2,  2,0
};
constexpr int TRI_FACES_ORIENTED_OFF[] = {0,2,4,6};

// Triangle edges (3)
constexpr index_t TRI_EDGES_FLAT[] = {
    0,1, 1,2, 2,0
};

// =========================
// Quad (2D cell) faces (edges) oriented CCW
// Reference: VTK quad (vtkQuad), edges oriented CCW.
constexpr index_t QUAD_FACES_ORIENTED_IDX[] = {
    0,1,  1,2,  2,3,  3,0
};
constexpr int QUAD_FACES_ORIENTED_OFF[] = {0,2,4,6,8};

// Quad edges (4)
constexpr index_t QUAD_EDGES_FLAT[] = {
    0,1, 1,2, 2,3, 3,0
};

// =========================
// Wedge (triangular prism) oriented faces
// Faces: bottom (0,2,1), top (3,4,5), sides (0,1,4,3), (1,2,5,4), (2,0,3,5)
// Reference: VTK wedge (vtkWedge) face ordering and orientation.
constexpr index_t WEDGE_FACES_ORIENTED_IDX[] = {
    0,2,1,   3,4,5,   0,1,4,3,   1,2,5,4,   2,0,3,5
};
constexpr int WEDGE_FACES_ORIENTED_OFF[] = {0,3,6,10,14,18};

// Wedge edges (9): bottom 3, top 3, vertical 3
constexpr index_t WEDGE_EDGES_FLAT[] = {
    0,1, 1,2, 2,0,
    3,4, 4,5, 5,3,
    0,3, 1,4, 2,5
};

// =========================
// Pyramid oriented faces
// Base (0,1,2,3), sides: (0,4,1), (1,4,2), (2,4,3), (3,4,0)
// Reference: VTK pyramid (vtkPyramid) face ordering and orientation.
constexpr index_t PYR_FACES_ORIENTED_IDX[] = {
    0,1,2,3,   0,4,1,   1,4,2,   2,4,3,   3,4,0
};
constexpr int PYR_FACES_ORIENTED_OFF[] = {0,4,7,10,13,16};

// Pyramid edges (8): base 4, apex connections 4
constexpr index_t PYR_EDGES_FLAT[] = {
    0,1, 1,2, 2,3, 3,0,
    0,4, 1,4, 2,4, 3,4
};
} // anonymous namespace

// begin public interface

// ==========================================
// Public Interface
// ==========================================

std::vector<std::vector<index_t>> CellTopology::get_boundary_faces(CellFamily family) {
    // Prefer canonical view (zero-allocation) for fixed families, fallback to original
    auto view = get_boundary_faces_canonical_view(family);
    if (view.indices && view.offsets && view.face_count > 0) {
        std::vector<std::vector<index_t>> faces(view.face_count);
        for (int f = 0; f < view.face_count; ++f) {
            int b = view.offsets[f];
            int e = view.offsets[f+1];
            faces[f].assign(view.indices + b, view.indices + e);
        }
        return faces;
    }
    switch (family) {
        case CellFamily::Tetra:   return tet_faces_canonical();
        case CellFamily::Hex:     return hex_faces_canonical();
        case CellFamily::Wedge:   return wedge_faces_canonical();
        case CellFamily::Pyramid: return pyramid_faces_canonical();
        case CellFamily::Triangle:return tri_edges_canonical();
        case CellFamily::Quad:    return quad_edges_canonical();
        default:
            throw std::runtime_error("Unsupported cell family for boundary face extraction");
    }
}

std::vector<std::vector<index_t>> CellTopology::get_oriented_boundary_faces(CellFamily family) {
    auto view = get_oriented_boundary_faces_view(family);
    if (view.indices && view.offsets && view.face_count > 0) {
        std::vector<std::vector<index_t>> faces(view.face_count);
        for (int f = 0; f < view.face_count; ++f) {
            int b = view.offsets[f];
            int e = view.offsets[f+1];
            faces[f].assign(view.indices + b, view.indices + e);
        }
        return faces;
    }
    // Fallback to original implementations if view unavailable
    switch (family) {
        case CellFamily::Wedge:   return wedge_faces_oriented();
        case CellFamily::Pyramid: return pyramid_faces_oriented();
        case CellFamily::Triangle:return tri_edges_oriented();
        case CellFamily::Quad:    return quad_edges_oriented();
        default:
            throw std::runtime_error("Unsupported cell family for oriented boundary face extraction");
    }
}

CellTopology::FaceListView CellTopology::get_oriented_boundary_faces_view(CellFamily family) {
    switch (family) {
        case CellFamily::Tetra:
            return {TET_FACES_ORIENTED_IDX, TET_FACES_ORIENTED_OFF, 4};
        case CellFamily::Hex:
            return {HEX_FACES_ORIENTED_IDX, HEX_FACES_ORIENTED_OFF, 6};
        case CellFamily::Triangle:
            return {TRI_FACES_ORIENTED_IDX, TRI_FACES_ORIENTED_OFF, 3};
        case CellFamily::Quad:
            return {QUAD_FACES_ORIENTED_IDX, QUAD_FACES_ORIENTED_OFF, 4};
        case CellFamily::Wedge:
            return {WEDGE_FACES_ORIENTED_IDX, WEDGE_FACES_ORIENTED_OFF, 5};
        case CellFamily::Pyramid:
            return {PYR_FACES_ORIENTED_IDX, PYR_FACES_ORIENTED_OFF, 5};
        default:
            return {nullptr, nullptr, 0};
    }
}

// ------------------------------
// Canonical (sorted) face views
// ------------------------------
namespace {
// Fixed-family canonical faces (sorted within-face) for topology detection

// Triangle canonical edges (faces in 2D): {0,1},{1,2},{0,2}
constexpr index_t TRI_FACES_CAN_IDX[] = { 0,1, 1,2, 0,2 };
constexpr int TRI_FACES_CAN_OFF[] = {0,2,4,6};

// Quad canonical edges: {0,1},{1,2},{2,3},{0,3}
constexpr index_t QUAD_FACES_CAN_IDX[] = { 0,1, 1,2, 2,3, 0,3 };
constexpr int QUAD_FACES_CAN_OFF[] = {0,2,4,6,8};

// Tetra canonical faces: {0,1,2}, {0,1,3}, {0,2,3}, {1,2,3}
constexpr index_t TET_FACES_CAN_IDX[] = { 0,1,2, 0,1,3, 0,2,3, 1,2,3 };
constexpr int TET_FACES_CAN_OFF[] = {0,3,6,9,12};

// Hex canonical faces: bottom, top, front, right, back, left (sorted within each face)
constexpr index_t HEX_FACES_CAN_IDX[] = {
    0,1,2,3,  4,5,6,7,  0,1,4,5,  1,2,5,6,  2,3,6,7,  0,3,4,7
};
constexpr int HEX_FACES_CAN_OFF[] = {0,4,8,12,16,20,24};

// Wedge canonical faces: bottom, top, and three quads (each sorted within-face)
constexpr index_t WEDGE_FACES_CAN_IDX[] = {
    0,1,2,   3,4,5,   0,1,3,4,   1,2,4,5,   0,2,3,5
};
constexpr int WEDGE_FACES_CAN_OFF[] = {0,3,6,10,14,18};

// Pyramid canonical faces: base and side triangles (sorted within-face)
constexpr index_t PYR_FACES_CAN_IDX[] = {
    0,1,2,3,  0,1,4,  1,2,4,  2,3,4,  0,3,4
};
constexpr int PYR_FACES_CAN_OFF[] = {0,4,7,10,13,16};
} // anonymous namespace

CellTopology::FaceListView CellTopology::get_boundary_faces_canonical_view(CellFamily family) {
    switch (family) {
        case CellFamily::Triangle: return {TRI_FACES_CAN_IDX,  TRI_FACES_CAN_OFF,  3};
        case CellFamily::Quad:     return {QUAD_FACES_CAN_IDX, QUAD_FACES_CAN_OFF, 4};
        case CellFamily::Tetra:    return {TET_FACES_CAN_IDX,  TET_FACES_CAN_OFF,  4};
        case CellFamily::Hex:      return {HEX_FACES_CAN_IDX,  HEX_FACES_CAN_OFF,  6};
        case CellFamily::Wedge:    return {WEDGE_FACES_CAN_IDX,WEDGE_FACES_CAN_OFF,5};
        case CellFamily::Pyramid:  return {PYR_FACES_CAN_IDX,  PYR_FACES_CAN_OFF,  5};
        default:                   return {nullptr, nullptr, 0};
    }
}

std::vector<std::array<index_t, 2>> CellTopology::get_edges(CellFamily family) {
    auto view = get_edges_view(family);
    if (view.pairs_flat && view.edge_count > 0) {
        std::vector<std::array<index_t,2>> edges;
        edges.reserve(static_cast<size_t>(view.edge_count));
        for (int i = 0; i < view.edge_count; ++i) {
            edges.push_back({view.pairs_flat[2*i+0], view.pairs_flat[2*i+1]});
        }
        return edges;
    }
    // Fallback to original implementations if view unavailable
    switch (family) {
        case CellFamily::Wedge:   return wedge_edges();
        case CellFamily::Pyramid: return pyramid_edges();
        case CellFamily::Triangle:return tri_edges();
        case CellFamily::Quad:    return quad_edges();
        default:
            throw std::runtime_error("Unsupported cell family for edge extraction");
    }
}

CellTopology::EdgeListView CellTopology::get_edges_view(CellFamily family) {
    switch (family) {
        case CellFamily::Tetra:
            return {TET_EDGES_FLAT, 6};
        case CellFamily::Hex:
            return {HEX_EDGES_FLAT, 12};
        case CellFamily::Triangle:
            return {TRI_EDGES_FLAT, 3};
        case CellFamily::Quad:
            return {QUAD_EDGES_FLAT, 4};
        case CellFamily::Wedge:
            return {WEDGE_EDGES_FLAT, 9};
        case CellFamily::Pyramid:
            return {PYR_EDGES_FLAT, 8};
        default:
            return {nullptr, 0};
    }
}

// ======================================================
// Variable-arity families: polygon, prism(m), pyramid(m)
// ======================================================
namespace {
struct FaceCacheEntry {
    std::vector<index_t> idx;   // flattened indices
    std::vector<int> off;       // CSR offsets
    int count = 0;              // number of faces
};
struct EdgeCacheEntry {
    std::vector<index_t> pairs; // flattened pairs [v0,v1,...]
    int count = 0;              // number of edges
};

// Caches keyed by m
static std::unordered_map<int, std::shared_ptr<FaceCacheEntry>> poly_face_cache;
static std::unordered_map<int, std::shared_ptr<FaceCacheEntry>> poly_face_can_cache;
static std::unordered_map<int, std::shared_ptr<EdgeCacheEntry>> poly_edge_cache;

static std::unordered_map<int, std::shared_ptr<FaceCacheEntry>> prism_face_cache;
static std::unordered_map<int, std::shared_ptr<FaceCacheEntry>> prism_face_can_cache;
static std::unordered_map<int, std::shared_ptr<EdgeCacheEntry>> prism_edge_cache;

static std::unordered_map<int, std::shared_ptr<FaceCacheEntry>> pyr_face_cache;
static std::unordered_map<int, std::shared_ptr<FaceCacheEntry>> pyr_face_can_cache;
static std::unordered_map<int, std::shared_ptr<EdgeCacheEntry>> pyr_edge_cache;

inline std::shared_ptr<FaceCacheEntry> build_polygon_faces_oriented(int m) {
    auto e = std::make_shared<FaceCacheEntry>();
    e->count = m;
    e->off.resize(m + 1);
    e->idx.reserve(2 * m);
    int off = 0;
    for (int i = 0; i < m; ++i) {
        e->off[i] = off;
        int j = (i + 1) % m;
        e->idx.push_back(i);
        e->idx.push_back(j);
        off += 2;
    }
    e->off[m] = off;
    return e;
}

inline std::shared_ptr<FaceCacheEntry> build_polygon_faces_canonical(int m) {
    auto e = std::make_shared<FaceCacheEntry>();
    e->count = m;
    e->off.resize(m + 1);
    e->idx.reserve(2 * m);
    int off = 0;
    for (int i = 0; i < m; ++i) {
        e->off[i] = off;
        int j = (i + 1) % m;
        int a = std::min(i, j);
        int b = std::max(i, j);
        e->idx.push_back(a);
        e->idx.push_back(b);
        off += 2;
    }
    e->off[m] = off;
    return e;
}

inline std::shared_ptr<EdgeCacheEntry> build_polygon_edges_oriented(int m) {
    auto e = std::make_shared<EdgeCacheEntry>();
    e->count = m;
    e->pairs.reserve(2 * m);
    for (int i = 0; i < m; ++i) {
        int j = (i + 1) % m;
        e->pairs.push_back(i);
        e->pairs.push_back(j);
    }
    return e;
}

inline std::shared_ptr<FaceCacheEntry> build_prism_faces_oriented(int m) {
    auto e = std::make_shared<FaceCacheEntry>();
    e->count = 2 + m; // bottom, top, m side quads
    e->off.clear();
    e->off.reserve(e->count + 1);
    e->idx.clear();
    e->idx.reserve(2 * m + 4 * m);
    int off = 0;
    // bottom (reverse base order): 0, m-1, ..., 1
    e->off.push_back(off);
    e->idx.push_back(0);
    for (int k = m - 1; k >= 1; --k) e->idx.push_back(k);
    off += m;
    e->off.push_back(off);
    // top in base order: m..2m-1
    for (int k = 0; k < m; ++k) e->idx.push_back(m + k);
    off += m;
    e->off.push_back(off);
    // side quads: (i, i+1, m+i+1, m+i)
    for (int i = 0; i < m; ++i) {
        int i1 = (i + 1) % m;
        e->idx.push_back(i);
        e->idx.push_back(i1);
        e->idx.push_back(m + i1);
        e->idx.push_back(m + i);
        off += 4;
        e->off.push_back(off);
    }
    return e;
}

inline std::shared_ptr<FaceCacheEntry> build_prism_faces_canonical(int m) {
    auto oriented = build_prism_faces_oriented(m);
    // sort indices within each face to create canonical version
    auto e = std::make_shared<FaceCacheEntry>(*oriented);
    for (int f = 0; f < e->count; ++f) {
        int b = e->off[f];
        int eoff = e->off[f+1];
        std::sort(e->idx.begin() + b, e->idx.begin() + eoff);
    }
    return e;
}

inline std::shared_ptr<EdgeCacheEntry> build_prism_edges_oriented(int m) {
    auto e = std::make_shared<EdgeCacheEntry>();
    e->count = 3 * m;
    e->pairs.reserve(2 * e->count);
    // bottom edges
    for (int i = 0; i < m; ++i) {
        int j = (i + 1) % m;
        e->pairs.push_back(i);
        e->pairs.push_back(j);
    }
    // top edges
    for (int i = 0; i < m; ++i) {
        int j = (i + 1) % m;
        e->pairs.push_back(m + i);
        e->pairs.push_back(m + j);
    }
    // vertical edges
    for (int i = 0; i < m; ++i) {
        e->pairs.push_back(i);
        e->pairs.push_back(m + i);
    }
    return e;
}

inline std::shared_ptr<FaceCacheEntry> build_pyramid_faces_oriented(int m) {
    auto e = std::make_shared<FaceCacheEntry>();
    e->count = 1 + m; // base + m sides
    e->off.reserve(e->count + 1);
    e->idx.reserve(m + 3 * m);
    int off = 0;
    // base CCW: 0..m-1
    e->off.push_back(off);
    for (int k = 0; k < m; ++k) e->idx.push_back(k);
    off += m;
    e->off.push_back(off);
    // sides: (i, apex, i+1)
    for (int i = 0; i < m; ++i) {
        int i1 = (i + 1) % m;
        e->idx.push_back(i);
        e->idx.push_back(m);
        e->idx.push_back(i1);
        off += 3;
        e->off.push_back(off);
    }
    return e;
}

inline std::shared_ptr<FaceCacheEntry> build_pyramid_faces_canonical(int m) {
    auto oriented = build_pyramid_faces_oriented(m);
    auto e = std::make_shared<FaceCacheEntry>(*oriented);
    for (int f = 0; f < e->count; ++f) {
        int b = e->off[f];
        int eoff = e->off[f+1];
        std::sort(e->idx.begin() + b, e->idx.begin() + eoff);
    }
    return e;
}

inline std::shared_ptr<EdgeCacheEntry> build_pyramid_edges_oriented(int m) {
    auto e = std::make_shared<EdgeCacheEntry>();
    e->count = m + m; // base edges + edges to apex
    e->pairs.reserve(2 * e->count);
    // base edges
    for (int i = 0; i < m; ++i) {
        int j = (i + 1) % m;
        e->pairs.push_back(i);
        e->pairs.push_back(j);
    }
    // edges to apex
    for (int i = 0; i < m; ++i) {
        e->pairs.push_back(i);
        e->pairs.push_back(m);
    }
    return e;
}
} // anonymous namespace (variable-arity caches)

CellTopology::FaceListView CellTopology::get_polygon_faces_view(int m) {
    if (m < 3) return {nullptr, nullptr, 0};
    auto& cache = poly_face_cache;
    auto it = cache.find(m);
    if (it == cache.end()) it = cache.emplace(m, build_polygon_faces_oriented(m)).first;
    auto& e = *it->second;
    return {e.idx.data(), e.off.data(), e.count};
}

CellTopology::FaceListView CellTopology::get_polygon_faces_canonical_view(int m) {
    if (m < 3) return {nullptr, nullptr, 0};
    auto& cache = poly_face_can_cache;
    auto it = cache.find(m);
    if (it == cache.end()) it = cache.emplace(m, build_polygon_faces_canonical(m)).first;
    auto& e = *it->second;
    return {e.idx.data(), e.off.data(), e.count};
}

CellTopology::EdgeListView CellTopology::get_polygon_edges_view(int m) {
    if (m < 3) return {nullptr, 0};
    auto& cache = poly_edge_cache;
    auto it = cache.find(m);
    if (it == cache.end()) it = cache.emplace(m, build_polygon_edges_oriented(m)).first;
    auto& e = *it->second;
    return {e.pairs.data(), e.count};
}

CellTopology::FaceListView CellTopology::get_prism_faces_view(int m) {
    if (m < 3) return {nullptr, nullptr, 0};
    auto& cache = prism_face_cache;
    auto it = cache.find(m);
    if (it == cache.end()) it = cache.emplace(m, build_prism_faces_oriented(m)).first;
    auto& e = *it->second;
    return {e.idx.data(), e.off.data(), e.count};
}

CellTopology::FaceListView CellTopology::get_prism_faces_canonical_view(int m) {
    if (m < 3) return {nullptr, nullptr, 0};
    auto& cache = prism_face_can_cache;
    auto it = cache.find(m);
    if (it == cache.end()) it = cache.emplace(m, build_prism_faces_canonical(m)).first;
    auto& e = *it->second;
    return {e.idx.data(), e.off.data(), e.count};
}

CellTopology::EdgeListView CellTopology::get_prism_edges_view(int m) {
    if (m < 3) return {nullptr, 0};
    auto& cache = prism_edge_cache;
    auto it = cache.find(m);
    if (it == cache.end()) it = cache.emplace(m, build_prism_edges_oriented(m)).first;
    auto& e = *it->second;
    return {e.pairs.data(), e.count};
}

CellTopology::FaceListView CellTopology::get_pyramid_faces_view(int m) {
    if (m < 3) return {nullptr, nullptr, 0};
    auto& cache = pyr_face_cache;
    auto it = cache.find(m);
    if (it == cache.end()) it = cache.emplace(m, build_pyramid_faces_oriented(m)).first;
    auto& e = *it->second;
    return {e.idx.data(), e.off.data(), e.count};
}

CellTopology::FaceListView CellTopology::get_pyramid_faces_canonical_view(int m) {
    if (m < 3) return {nullptr, nullptr, 0};
    auto& cache = pyr_face_can_cache;
    auto it = cache.find(m);
    if (it == cache.end()) it = cache.emplace(m, build_pyramid_faces_canonical(m)).first;
    auto& e = *it->second;
    return {e.idx.data(), e.off.data(), e.count};
}

CellTopology::EdgeListView CellTopology::get_pyramid_edges_view(int m) {
    if (m < 3) return {nullptr, 0};
    auto& cache = pyr_edge_cache;
    auto it = cache.find(m);
    if (it == cache.end()) it = cache.emplace(m, build_pyramid_edges_oriented(m)).first;
    auto& e = *it->second;
    return {e.pairs.data(), e.count};
}

// ------------------------------------------
// Higher-order node ordering (quadratic, VTK)
// ------------------------------------------
std::vector<index_t> CellTopology::HighOrderOrdering::assemble_quadratic(
    CellFamily family,
    const std::vector<index_t>& corners,
    const std::vector<index_t>& edge_mids,
    const std::vector<index_t>& face_mids,
    bool include_center,
    index_t center) {

    // Check counts against topology views
    auto eview = CellTopology::get_edges_view(family);
    auto fview = CellTopology::get_oriented_boundary_faces_view(family);
    if (eview.edge_count != static_cast<int>(edge_mids.size())) {
        throw std::invalid_argument("edge_mids size does not match edge count for family");
    }
    if (fview.face_count != 0 && !face_mids.empty() && fview.face_count != static_cast<int>(face_mids.size())) {
        throw std::invalid_argument("face_mids size does not match face count for family");
    }

    std::vector<index_t> out;
    out.reserve(corners.size() + edge_mids.size() + face_mids.size() + (include_center ? 1 : 0));
    out.insert(out.end(), corners.begin(), corners.end());
    out.insert(out.end(), edge_mids.begin(), edge_mids.end());
    out.insert(out.end(), face_mids.begin(), face_mids.end());
    if (include_center) out.push_back(center);
    return out;
}

std::vector<index_t> CellTopology::HighOrderOrdering::assemble_quadratic_polygon(
    int m,
    const std::vector<index_t>& corners,
    const std::vector<index_t>& edge_mids) {
    auto eview = CellTopology::get_polygon_edges_view(m);
    if (eview.edge_count != static_cast<int>(edge_mids.size())) {
        throw std::invalid_argument("edge_mids size does not match polygon edge count");
    }
    std::vector<index_t> out;
    out.reserve(corners.size() + edge_mids.size());
    out.insert(out.end(), corners.begin(), corners.end());
    out.insert(out.end(), edge_mids.begin(), edge_mids.end());
    return out;
}

std::vector<index_t> CellTopology::HighOrderOrdering::assemble_quadratic_prism(
    int m,
    const std::vector<index_t>& corners,
    const std::vector<index_t>& edge_mids,
    const std::vector<index_t>& face_mids,
    bool include_center,
    index_t center) {

    auto eview = CellTopology::get_prism_edges_view(m);
    auto fview = CellTopology::get_prism_faces_view(m);
    if (eview.edge_count != static_cast<int>(edge_mids.size())) {
        throw std::invalid_argument("edge_mids size does not match prism edge count");
    }
    if (!face_mids.empty() && fview.face_count != static_cast<int>(face_mids.size())) {
        throw std::invalid_argument("face_mids size does not match prism face count");
    }
    std::vector<index_t> out;
    out.reserve(corners.size() + edge_mids.size() + face_mids.size() + (include_center ? 1 : 0));
    out.insert(out.end(), corners.begin(), corners.end());
    out.insert(out.end(), edge_mids.begin(), edge_mids.end());
    out.insert(out.end(), face_mids.begin(), face_mids.end());
    if (include_center) out.push_back(center);
    return out;
}

std::vector<index_t> CellTopology::HighOrderOrdering::assemble_quadratic_pyramid(
    int m,
    const std::vector<index_t>& corners,
    const std::vector<index_t>& edge_mids,
    const std::vector<index_t>& face_mids,
    bool include_center,
    index_t center) {

    auto eview = CellTopology::get_pyramid_edges_view(m);
    auto fview = CellTopology::get_pyramid_faces_view(m);
    if (eview.edge_count != static_cast<int>(edge_mids.size())) {
        throw std::invalid_argument("edge_mids size does not match pyramid edge count");
    }
    if (!face_mids.empty() && fview.face_count != static_cast<int>(face_mids.size())) {
        throw std::invalid_argument("face_mids size does not match pyramid face count");
    }
    std::vector<index_t> out;
    out.reserve(corners.size() + edge_mids.size() + face_mids.size() + (include_center ? 1 : 0));
    out.insert(out.end(), corners.begin(), corners.end());
    out.insert(out.end(), edge_mids.begin(), edge_mids.end());
    out.insert(out.end(), face_mids.begin(), face_mids.end());
    if (include_center) out.push_back(center);
    return out;
}

std::vector<std::vector<int>> CellTopology::get_face_edges(CellFamily family) {
    // Face-to-edge connectivity (which edges bound each face)
    // Implementation derives mapping directly from the oriented (preferred) or canonical face views
    // and the edge view, avoiding hard-coded tables.
    // Reference for face orientation/ordering: VTK cell definitions (vtkTetra, vtkHexahedron,
    // vtkWedge, vtkPyramid, vtkTriangle, vtkQuad). See VTK Doxygen and the VTK File Formats
    // documentation for the expected corner ordering and outward face orientation.
    // This method ensures the mapping stays consistent with the tables above.

    // Try to use oriented face view; if unavailable, use canonical view.
    FaceListView fview = get_oriented_boundary_faces_view(family);
    bool used_oriented = true;
    if (!(fview.indices && fview.offsets && fview.face_count > 0)) {
        fview = get_boundary_faces_canonical_view(family);
        used_oriented = false;
    }
    // Get edges view
    EdgeListView eview = get_edges_view(family);
    if (!(fview.indices && fview.offsets && fview.face_count > 0 && eview.pairs_flat && eview.edge_count > 0)) {
        // 2D families (Triangle/Quad) still have oriented/canonical face views and edges
        // If we cannot get views, fall back to previous behavior
        switch (family) {
            case CellFamily::Triangle: return {{0}, {1}, {2}}; // faces are edges in 2D
            case CellFamily::Quad:     return {{0}, {1}, {2}, {3}};
            default:
                throw std::runtime_error("Face-to-edge connectivity not available for this cell type");
        }
    }

    // Build map from undirected edge key -> edge index
    std::unordered_map<long long, int> edge_map;
    edge_map.reserve(static_cast<size_t>(eview.edge_count) * 2);
    auto pack = [](index_t a, index_t b) -> long long {
        index_t lo = std::min(a,b);
        index_t hi = std::max(a,b);
        return (static_cast<long long>(lo) << 32) | static_cast<unsigned long long>(static_cast<uint32_t>(hi));
    };
    for (int ei = 0; ei < eview.edge_count; ++ei) {
        index_t a = eview.pairs_flat[2*ei+0];
        index_t b = eview.pairs_flat[2*ei+1];
        edge_map.emplace(pack(a,b), ei);
    }

    std::vector<std::vector<int>> face2edges;
    face2edges.resize(static_cast<size_t>(fview.face_count));

    for (int fi = 0; fi < fview.face_count; ++fi) {
        int b = fview.offsets[fi];
        int e = fview.offsets[fi+1];
        int fv = e - b;
        // Handle 2D faces specially: they are edges; return the single matching edge index
        if (fv == 2) {
            index_t u = fview.indices[b+0];
            index_t v = fview.indices[b+1];
            auto it = edge_map.find(pack(u,v));
            if (it == edge_map.end()) {
                throw std::runtime_error("Face-to-edge mapping failed: edge not found (2D)");
            }
            face2edges[fi] = { it->second };
            continue;
        }

        // For 3D faces (tri or quad), walk the face loop and map each boundary edge
        std::vector<int> edges_for_face;
        edges_for_face.reserve(fv);
        for (int k = 0; k < fv; ++k) {
            index_t u = fview.indices[b + k];
            index_t v = fview.indices[b + ((k + 1) % fv)];
            auto it = edge_map.find(pack(u,v));
            if (it == edge_map.end()) {
                throw std::runtime_error("Face-to-edge mapping failed: edge not found (3D)");
            }
            edges_for_face.push_back(it->second);
        }
        face2edges[fi] = std::move(edges_for_face);
    }

    return face2edges;
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
    // Ordering chosen to match a common dimension-agnostic simplex convention:
    // (1,2,3), (0,3,2), (0,1,3), (0,2,1) each listed CCW as seen from outside
    return {
        {1, 2, 3},  // opposite vertex 0
        {0, 3, 2},  // opposite vertex 1
        {0, 1, 3},  // opposite vertex 2
        {0, 2, 1}   // opposite vertex 3
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
    // Right-hand rule ordering (outward normals), matching the static
    // HEX_FACES_ORIENTED_IDX/OFF tables used by get_oriented_boundary_faces_view.
    // Reference: VTK hexahedron corner and face ordering (vtkHexahedron).
    return {
        {0,3,2,1},  // bottom (z-)
        {4,5,6,7},  // top (z+)
        {0,1,5,4},  // front
        {1,2,6,5},  // right
        {2,3,7,6},  // back
        {3,0,4,7}   // left
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
    // Base listed CCW as seen from outside; side triangles follow (i, apex, i+1)
    return {
        {0, 1, 2, 3},  // quad base (outward)
        {0, 4, 1},     // side triangles (wrap i+1 mod 4)
        {1, 4, 2},
        {2, 4, 3},
        {3, 4, 0}
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

// ==========================================
// High-order (p>2) scaffolding
// ==========================================

CellTopology::HighOrderPattern CellTopology::high_order_pattern(CellFamily family, int p, HighOrderKind kind) {
    HighOrderPattern pat; pat.kind = kind; pat.order = p;
    if (p <= 2) return pat; // handled by quadratic/linear paths

    // Corners first
    int nc = 0;
    switch (family) {
        case CellFamily::Triangle: nc = 3; break;
        case CellFamily::Quad:     nc = 4; break;
        case CellFamily::Tetra:    nc = 4; break;
        case CellFamily::Hex:      nc = 8; break;
        case CellFamily::Wedge:    nc = 6; break;
        case CellFamily::Pyramid:  nc = 5; break;
        default: break;
    }
    for (int i=0;i<nc;++i) pat.sequence.push_back({HONodeRole::Corner, i,0,0});

    // Edge nodes: enumerate topology edges and add (p-1) nodes per edge for Lagrange/Serendipity
    auto eview = get_edges_view(family);
    int edgeNodes = std::max(0, p-1);
    for (int ei=0; ei<eview.edge_count; ++ei) {
        for (int k=1; k<=edgeNodes; ++k) pat.sequence.push_back({HONodeRole::Edge, ei, k, edgeNodes});
    }

    // Face interior nodes (Lagrange only, Serendipity typically omits 2D interior for p=3 but may include for higher)
    if (kind == HighOrderKind::Lagrange) {
        if (family == CellFamily::Triangle || family == CellFamily::Quad || family == CellFamily::Tetra || family == CellFamily::Hex || family == CellFamily::Wedge || family == CellFamily::Pyramid) {
            // Faces for 3D cells; for 2D, faces are edges so skip here
            if (family == CellFamily::Triangle || family == CellFamily::Quad) {
                // 2D: no face interior concept beyond element interior; handled separately if needed
            } else {
                auto fview = get_oriented_boundary_faces_view(family);
                for (int fi=0; fi<fview.face_count; ++fi) {
                    int b = fview.offsets[fi], e = fview.offsets[fi+1];
                    int fv = e-b;
                    if (fv == 3) {
                        // Triangular face: barycentric i,j >=1, i+j <= p-2 (VTK ordering lexicographic)
                        for (int i=1; i<=p-2; ++i) {
                            for (int j=1; j<=p-1-i; ++j) {
                                pat.sequence.push_back({HONodeRole::Face, fi, i, j});
                            }
                        }
                    } else if (fv == 4) {
                        // Quadrilateral face: grid i=1..p-1, j=1..p-1 (row-major)
                        for (int i=1; i<=p-1; ++i) {
                            for (int j=1; j<=p-1; ++j) {
                                pat.sequence.push_back({HONodeRole::Face, fi, i, j});
                            }
                        }
                    }
                }
            }
        }
    }

    // Volume interior nodes (Lagrange only): for Hex and Tetra/Wedge/Pyramid (scaffolding)
    if (kind == HighOrderKind::Lagrange) {
        if (family == CellFamily::Hex) {
            for (int i=1;i<=p-1;++i)
                for (int j=1;j<=p-1;++j)
                    for (int k=1;k<=p-1;++k)
                        pat.sequence.push_back({HONodeRole::Volume, i,j,k});
        } else if (family == CellFamily::Tetra) {
            // VTK Lagrange tetrahedron interior nodes (barycentric enumeration):
            // i, j, k >= 1 with i + j + k <= p - 1 - 1 (equivalently, all four barycentric
            // coordinates >= 1). Total count = (p-1)(p-2)(p-3)/6.
            for (int i=1;i<=p-3;++i)
                for (int j=1;j<=p-2-i;++j)
                    for (int k=1;k<=p-1-i-j;++k)
                        pat.sequence.push_back({HONodeRole::Volume, i,j,k});
        } else if (family == CellFamily::Wedge) {
            // VTK Lagrange wedge interior nodes: two triangular directions (i,j) with
            // 1 <= i <= p-2, 1 <= j <= p-1-i (matching triangular face interior nodes),
            // and one linear index k = 1..p-1 along the third axis.
            for (int i=1;i<=p-2;++i)
                for (int j=1;j<=p-1-i;++j)
                    for (int k=1;k<=p-1;++k)
                        pat.sequence.push_back({HONodeRole::Volume, i,j,k});
        } else if (family == CellFamily::Pyramid) {
            // VTK Lagrange pyramid interior (volume) nodes:
            // Layered enumeration along the vertical index k = 1..p-2. At each layer k, the
            // in-layer grid has size n = (p+1 - k) and strictly interior points are (n-2)^2.
            // Total interior nodes sum to (p-2)(p-1)(2p-3)/6, matching
            // N_total - [corners + edge + face] where N_total = sum_{m=1}^{p+1} m^2.
            // Reference: VTK vtkLagrangePyramid ordering and counts.
            for (int k=1; k<=p-2; ++k) {
                int n = (p + 1) - k; // grid size at this layer
                for (int i=1; i<=n-2; ++i) {
                    for (int j=1; j<=n-2; ++j) {
                        pat.sequence.push_back({HONodeRole::Volume, i, j, k});
                    }
                }
            }
        }
    }

    return pat;
}

int CellTopology::infer_lagrange_order(CellFamily family, size_t node_count) {
    // infer p from total nodes n for Lagrange elements
    switch (family) {
        case CellFamily::Triangle: {
            // n = (p+1)(p+2)/2
            // solve p^2 + 3p + 2 - 2n = 0
            double D = 1.0 + 8.0*(double)node_count;
            if (D < 0) return -1;
            double p = (-3.0 + std::sqrt(D)) / 2.0;
            int pi = (int)std::round(p);
            if ((size_t)((pi+1)*(pi+2)/2) == node_count) return pi;
            return -1;
        }
        case CellFamily::Quad: {
            // n = (p+1)^2
            int pi = (int)std::lround(std::sqrt((double)node_count)) - 1;
            if ((size_t)((pi+1)*(pi+1)) == node_count) return pi;
            return -1;
        }
        case CellFamily::Tetra: {
            // n = (p+1)(p+2)(p+3)/6
            for (int pi=2; pi<=12; ++pi) {
                size_t n = (size_t)(pi+1)*(pi+2)*(pi+3)/6;
                if (n == node_count) return pi;
            }
            return -1;
        }
        case CellFamily::Hex: {
            // n = (p+1)^3
            int pi = (int)std::lround(std::cbrt((double)node_count)) - 1;
            if ((size_t)((pi+1)*(pi+1)*(pi+1)) == node_count) return pi;
            return -1;
        }
        case CellFamily::Wedge: {
            // Triangular prism (wedge): n = (p+1)*(p+1)*(p+2)/2
            // Reference: VTK Lagrange wedge point count; also consistent with CellShape::expected_vertices()
            for (int pi=2; pi<=12; ++pi) {
                size_t n = (size_t)(pi+1)*(pi+1)*(pi+2)/2;
                if (n == node_count) return pi;
            }
            // Also allow linear/quadratic quick matches
            if (node_count == 6) return 1;
            if (node_count == 18) return 2;
            return -1;
        }
        case CellFamily::Pyramid: {
            // Lagrange pyramid total nodes: N = sum_{m=1}^{p+1} m^2 = (p+1)(p+2)(2p+3)/6
            for (int pi=1; pi<=20; ++pi) {
                size_t n = (size_t)(pi+1)*(pi+2)*(2*pi+3)/6;
                if (n == node_count) return pi;
            }
            // Quadratic classic (non-Lagrange) special case: 13 nodes (no base center). Accept as p=2.
            if (node_count == 13) return 2;
            return -1;
        }
        default:
            return -1;
    }
}

int CellTopology::infer_serendipity_order(CellFamily family, size_t node_count) {
    // Tighten handling for Quad serendipity: recognize common Q8 and Q12 only.
    // Reference: standard 2D serendipity families omit interior nodes:
    //  - p=2: 8 nodes (edge midpoints only)
    //  - p=3: 12 nodes (two per edge)
    if (family == CellFamily::Quad) {
        // Common 2D serendipity pattern: corners + (p-1) nodes per edge, no interior nodes.
        // Node count n = 4 + 4*(p-1) for p >= 2.
        if (node_count >= 8 && (node_count - 4) % 4 == 0) {
            int pi = (int)((node_count - 4)/4) + 1;
            if (pi >= 2 && (size_t)(4 + 4*(pi-1)) == node_count) return pi;
        }
        return -1;
    }
    // Other families: unavailable
    return -1;
}

} // namespace svmp

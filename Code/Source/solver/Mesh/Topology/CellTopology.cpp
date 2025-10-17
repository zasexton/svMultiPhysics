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
#include <memory>

namespace svmp {
namespace {
// Static topology tables for fixed families (prototype: Tet/Hex)

// Tetrahedron: oriented faces (outward, right-hand rule)
// Faces: {1,2,3}, {0,3,2}, {0,1,3}, {0,2,1}
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
    // Right-hand rule ordering (outward normals), consistent with
    // dimension-agnostic orthotope rules and VTK-compatible face loops.
    // One valid outward set:
    //  - bottom: (0,1,2,3)
    //  - top   : (4,7,6,5)
    //  - sides : (0,1,5,4), (1,2,6,5), (2,3,7,6), (3,0,4,7)
    return {
        {0, 1, 2, 3},  // bottom (z-)
        {4, 7, 6, 5},  // top (z+)
        {0, 1, 5, 4},  // front (y-)
        {1, 2, 6, 5},  // right (x+)
        {2, 3, 7, 6},  // back  (y+)
        {3, 0, 4, 7}   // left  (x-)
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

} // namespace svmp

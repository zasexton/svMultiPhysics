/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Systems/MeshSearchAccess.h"

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH

#include "Mesh/Core/MeshBase.h"
#include "Mesh/Search/SearchBuilders.h"
#include "Mesh/Search/MeshSearch.h"
#include "Mesh/Search/SearchPrimitives.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>

namespace svmp {
namespace FE {
namespace systems {

namespace {

using AABB = svmp::search::AABB;

struct MarkerTriangle {
    std::array<Real, 3> a{};
    std::array<Real, 3> b{};
    std::array<Real, 3> c{};
    bool is_segment{false};
    GlobalIndex face_id{INVALID_GLOBAL_INDEX};
    AABB bounds{};
    std::array<Real, 3> centroid{};
};

struct MarkerBVHNode {
    AABB bounds{};
    int left{-1};
    int right{-1};
    int begin{0};
    int end{0}; // exclusive
    bool is_leaf{false};
};

class MarkerBoundaryBVH {
public:
    void build(const svmp::Mesh& mesh, int marker, svmp::Configuration cfg)
    {
        triangles_.clear();
        nodes_.clear();

        const auto faces = mesh.base().faces_with_label(static_cast<svmp::label_t>(marker));
        if (faces.empty()) {
            root_ = -1;
            built_ = true;
            marker_ = marker;
            cfg_ = cfg;
            return;
        }

        for (const auto face_id : faces) {
            const auto verts =
                svmp::search::SearchBuilders::get_face_vertex_coords(mesh.base(),
                                                                    static_cast<svmp::index_t>(face_id),
                                                                    cfg);

            if (verts.size() == 2u) {
                // 2D boundary face (edge): build a segment primitive.
                MarkerTriangle t;
                t.is_segment = true;
                t.a = {static_cast<Real>(verts[0][0]), static_cast<Real>(verts[0][1]), static_cast<Real>(verts[0][2])};
                t.b = {static_cast<Real>(verts[1][0]), static_cast<Real>(verts[1][1]), static_cast<Real>(verts[1][2])};
                t.c = t.b;
                t.face_id = static_cast<GlobalIndex>(face_id);
                t.bounds = AABB();
                t.bounds.include(t.a);
                t.bounds.include(t.b);
                t.centroid = t.bounds.center();
                triangles_.push_back(t);
                continue;
            }

            // Surface-like boundary face (triangle/quad): triangulate.
            const auto tris = svmp::search::SearchBuilders::triangulate_face(mesh.base(),
                                                                             static_cast<svmp::index_t>(face_id),
                                                                             cfg);
            for (const auto& tri : tris) {
                MarkerTriangle t;
                t.is_segment = false;
                t.a = {static_cast<Real>(tri[0][0]), static_cast<Real>(tri[0][1]), static_cast<Real>(tri[0][2])};
                t.b = {static_cast<Real>(tri[1][0]), static_cast<Real>(tri[1][1]), static_cast<Real>(tri[1][2])};
                t.c = {static_cast<Real>(tri[2][0]), static_cast<Real>(tri[2][1]), static_cast<Real>(tri[2][2])};
                t.face_id = static_cast<GlobalIndex>(face_id);
                t.bounds = AABB();
                t.bounds.include(t.a);
                t.bounds.include(t.b);
                t.bounds.include(t.c);
                t.centroid = t.bounds.center();
                triangles_.push_back(t);
            }
        }

        if (triangles_.empty()) {
            root_ = -1;
            built_ = true;
            marker_ = marker;
            cfg_ = cfg;
            return;
        }

        root_ = buildNode(/*begin=*/0, /*end=*/static_cast<int>(triangles_.size()));
        built_ = true;
        marker_ = marker;
        cfg_ = cfg;
    }

    [[nodiscard]] bool isBuiltFor(int marker, svmp::Configuration cfg) const noexcept
    {
        return built_ && marker == marker_ && cfg == cfg_;
    }

    [[nodiscard]] ISearchAccess::ClosestBoundaryPoint query(const std::array<Real, 3>& point,
                                                            Real max_distance) const
    {
        ISearchAccess::ClosestBoundaryPoint out;
        if (root_ < 0 || triangles_.empty() || nodes_.empty()) {
            return out;
        }

        Real best = max_distance;
        std::array<Real, 3> best_pt{0.0, 0.0, 0.0};
        GlobalIndex best_face = INVALID_GLOBAL_INDEX;

        std::vector<int> stack;
        stack.push_back(root_);

        while (!stack.empty()) {
            const int node_id = stack.back();
            stack.pop_back();
            const auto& node = nodes_[static_cast<std::size_t>(node_id)];

            const auto d_box = static_cast<Real>(svmp::search::point_aabb_distance(point, node.bounds));
            if (d_box > best) {
                continue;
            }

            if (node.is_leaf) {
                for (int i = node.begin; i < node.end; ++i) {
                    const auto& tri = triangles_[static_cast<std::size_t>(i)];
                    std::array<Real, 3> cp{};
                    const auto d = static_cast<Real>(tri.is_segment
                                                         ? svmp::search::point_segment_distance(point, tri.a, tri.b, &cp)
                                                         : svmp::search::point_triangle_distance(point,
                                                                                                 tri.a,
                                                                                                 tri.b,
                                                                                                 tri.c,
                                                                                                 &cp));
                    if (d < best) {
                        best = d;
                        best_pt = cp;
                        best_face = tri.face_id;
                    }
                }
            } else {
                if (node.left >= 0) stack.push_back(node.left);
                if (node.right >= 0) stack.push_back(node.right);
            }
        }

        if (best_face != INVALID_GLOBAL_INDEX && std::isfinite(best)) {
            out.found = true;
            out.face_id = best_face;
            out.closest_point = best_pt;
            out.distance = best;
        }

        return out;
    }

private:
    static constexpr int kLeafSize = 8;

    [[nodiscard]] static int splitAxis(const AABB& bounds)
    {
        const auto e = bounds.extents();
        if (e[0] >= e[1] && e[0] >= e[2]) return 0;
        if (e[1] >= e[0] && e[1] >= e[2]) return 1;
        return 2;
    }

    [[nodiscard]] int buildNode(int begin, int end)
    {
        MarkerBVHNode node;
        node.begin = begin;
        node.end = end;

        for (int i = begin; i < end; ++i) {
            node.bounds.include(triangles_[static_cast<std::size_t>(i)].bounds);
        }

        const int count = end - begin;
        if (count <= kLeafSize) {
            node.is_leaf = true;
            nodes_.push_back(node);
            return static_cast<int>(nodes_.size() - 1);
        }

        const int axis = splitAxis(node.bounds);
        const int mid = begin + count / 2;

        auto first = triangles_.begin() + begin;
        auto nth = triangles_.begin() + mid;
        auto last = triangles_.begin() + end;

        std::nth_element(first, nth, last, [axis](const MarkerTriangle& a, const MarkerTriangle& b) {
            return a.centroid[static_cast<std::size_t>(axis)] < b.centroid[static_cast<std::size_t>(axis)];
        });

        const int this_id = static_cast<int>(nodes_.size());
        nodes_.push_back(node);

        const int left = buildNode(begin, mid);
        const int right = buildNode(mid, end);

        auto& stored = nodes_[static_cast<std::size_t>(this_id)];
        stored.left = left;
        stored.right = right;
        stored.is_leaf = false;
        stored.begin = 0;
        stored.end = 0;

        // Tighten bounds from children.
        stored.bounds = AABB();
        if (left >= 0) stored.bounds.include(nodes_[static_cast<std::size_t>(left)].bounds);
        if (right >= 0) stored.bounds.include(nodes_[static_cast<std::size_t>(right)].bounds);

        return this_id;
    }

    std::vector<MarkerTriangle> triangles_{};
    std::vector<MarkerBVHNode> nodes_{};
    int root_{-1};
    bool built_{false};
    int marker_{0};
    svmp::Configuration cfg_{svmp::Configuration::Reference};
};

} // namespace

MeshSearchAccess::MeshSearchAccess(const svmp::Mesh& mesh)
    : mesh_(mesh)
{
}

MeshSearchAccess::MeshSearchAccess(const svmp::Mesh& mesh, svmp::Configuration cfg_override)
    : mesh_(mesh),
      coord_cfg_override_enabled_(true),
      coord_cfg_override_(cfg_override)
{
}

int MeshSearchAccess::dimension() const noexcept
{
    return mesh_.dim();
}

svmp::Configuration MeshSearchAccess::queryConfig() const noexcept
{
    return coord_cfg_override_enabled_ ? coord_cfg_override_ : mesh_.base().active_configuration();
}

void MeshSearchAccess::build() const
{
    svmp::MeshSearch::SearchConfig cfg;
    cfg.primary_use = svmp::MeshSearch::QueryType::PointLocation;
    mesh_.base().build_search_structure(cfg, queryConfig());
}

std::vector<GlobalIndex> MeshSearchAccess::verticesInRadius(
    const std::array<Real, 3>& point,
    Real radius) const
{
    const auto cfg = queryConfig();
    auto ids = svmp::MeshSearch::vertices_in_radius(mesh_.base(), point, radius, cfg);
    std::vector<GlobalIndex> out;
    out.reserve(ids.size());
    for (auto v : ids) {
        out.push_back(static_cast<GlobalIndex>(v));
    }
    return out;
}

ISearchAccess::PointLocation MeshSearchAccess::locatePoint(
    const std::array<Real, 3>& point,
    GlobalIndex hint_cell) const
{
    const auto cfg = queryConfig();
    const auto res = svmp::MeshSearch::locate_point(mesh_.base(),
                                                    point,
                                                    cfg,
                                                    static_cast<svmp::index_t>(hint_cell));

    ISearchAccess::PointLocation out;
    out.found = res.found;
    if (!res.found) {
        return out;
    }

    out.cell_id = static_cast<GlobalIndex>(res.cell_id);
    out.xi = {static_cast<Real>(res.xi[0]),
              static_cast<Real>(res.xi[1]),
              static_cast<Real>(res.xi[2])};
    return out;
}

ISearchAccess::NearestVertex MeshSearchAccess::nearestVertex(
    const std::array<Real, 3>& point) const
{
    const auto cfg = queryConfig();
    const auto [vid, dist] = svmp::MeshSearch::nearest_vertex(mesh_.base(), point, cfg);

    ISearchAccess::NearestVertex out;
    if (vid < 0) {
        return out;
    }
    out.found = true;
    out.vertex_id = static_cast<GlobalIndex>(vid);
    out.distance = static_cast<Real>(dist);
    return out;
}

std::vector<ISearchAccess::VertexNeighbor> MeshSearchAccess::kNearestVertices(
    const std::array<Real, 3>& point,
    std::size_t k) const
{
    if (k == 0u) return {};
    const auto cfg = queryConfig();
    const auto hits = svmp::MeshSearch::k_nearest_vertices(mesh_.base(), point, k, cfg);
    std::vector<ISearchAccess::VertexNeighbor> out;
    out.reserve(hits.size());
    for (const auto& [vid, dist] : hits) {
        out.push_back(ISearchAccess::VertexNeighbor{static_cast<GlobalIndex>(vid), static_cast<Real>(dist)});
    }
    return out;
}

ISearchAccess::NearestCell MeshSearchAccess::nearestCell(
    const std::array<Real, 3>& point) const
{
    const auto cfg = queryConfig();
    const auto [cid, dist] = svmp::MeshSearch::nearest_cell(mesh_.base(), point, cfg);

    ISearchAccess::NearestCell out;
    if (cid < 0) {
        return out;
    }
    out.found = true;
    out.cell_id = static_cast<GlobalIndex>(cid);
    out.distance = static_cast<Real>(dist);
    return out;
}

ISearchAccess::ClosestBoundaryPoint MeshSearchAccess::closestBoundaryPoint(
    const std::array<Real, 3>& point,
    Real max_distance) const
{
    const auto cfg = queryConfig();
    const auto res = svmp::MeshSearch::closest_boundary_point(mesh_.base(), point, cfg);
    const auto& cp = res.first;
    const auto face = static_cast<GlobalIndex>(res.second);

    ISearchAccess::ClosestBoundaryPoint out;
    if (face == INVALID_GLOBAL_INDEX) {
        return out;
    }

    const Real dx = static_cast<Real>(cp[0]) - point[0];
    const Real dy = static_cast<Real>(cp[1]) - point[1];
    const Real dz = static_cast<Real>(cp[2]) - point[2];
    const Real dist = std::sqrt(dx * dx + dy * dy + dz * dz);
    if (dist > max_distance) {
        return out;
    }

    out.found = true;
    out.face_id = face;
    out.closest_point = {static_cast<Real>(cp[0]), static_cast<Real>(cp[1]), static_cast<Real>(cp[2])};
    out.distance = dist;
    return out;
}

ISearchAccess::ClosestBoundaryPoint MeshSearchAccess::closestBoundaryPointOnMarker(
    int boundary_marker,
    const std::array<Real, 3>& point,
    Real max_distance) const
{
    const auto cfg = queryConfig();

    // Cache a BVH per (marker, config) for fast closest-point queries on the marker subset.
    static std::mutex mutex;
    static std::unordered_map<const svmp::Mesh*, std::unordered_map<int, MarkerBoundaryBVH>> cache;
    std::lock_guard<std::mutex> lock(mutex);
    auto& by_marker = cache[&mesh_];
    auto& entry = by_marker[boundary_marker];
    if (!entry.isBuiltFor(boundary_marker, cfg)) {
        entry.build(mesh_, boundary_marker, cfg);
    }
    return entry.query(point, max_distance);
}

} // namespace systems
} // namespace FE
} // namespace svmp

#endif // defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH

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

#include "BoundingVolume.h"
#include "../Core/MeshBase.h"
#include "MeshGeometry.h"
#include <cmath>
#include <algorithm>

#if __has_include(<Eigen/Dense>) && __has_include(<Eigen/Eigenvalues>)
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#define SVMP_HAVE_EIGEN 1
#else
#define SVMP_HAVE_EIGEN 0
#endif

namespace svmp {

//=============================================================================
// OBB Implementation
//=============================================================================

std::array<std::array<real_t, 3>, 8> OBB::get_corners() const {
    std::array<std::array<real_t, 3>, 8> corners;

    // Generate 8 corners from center + combinations of ±extents along axes
    int idx = 0;
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            for (int k = 0; k < 2; ++k) {
                real_t s0 = (i == 0 ? -extents[0] : extents[0]);
                real_t s1 = (j == 0 ? -extents[1] : extents[1]);
                real_t s2 = (k == 0 ? -extents[2] : extents[2]);

                corners[idx][0] = center[0] + s0*axes[0][0] + s1*axes[1][0] + s2*axes[2][0];
                corners[idx][1] = center[1] + s0*axes[0][1] + s1*axes[1][1] + s2*axes[2][1];
                corners[idx][2] = center[2] + s0*axes[0][2] + s1*axes[1][2] + s2*axes[2][2];
                ++idx;
            }
        }
    }

    return corners;
}

AABB OBB::to_aabb() const {
    auto corners = get_corners();
    AABB box(corners[0]);
    for (int i = 1; i < 8; ++i) {
        box.expand(corners[i]);
    }
    return box;
}

bool OBB::contains(const std::array<real_t, 3>& pt) const {
    // Transform point to OBB local frame
    real_t dx = pt[0] - center[0];
    real_t dy = pt[1] - center[1];
    real_t dz = pt[2] - center[2];

    // Project onto each axis
    for (int i = 0; i < 3; ++i) {
        real_t proj = dx*axes[i][0] + dy*axes[i][1] + dz*axes[i][2];
        if (std::abs(proj) > extents[i]) {
            return false;
        }
    }

    return true;
}

bool OBB::intersects(const OBB& other) const {
    // Separating Axis Theorem (SAT) test for OBB-OBB intersection
    // Test 15 potential separating axes: 3+3 face normals + 9 edge cross products

    auto test_axis = [](const std::array<real_t, 3>& axis,
                       const OBB& a, const OBB& b) -> bool {
        // Normalize axis
        real_t len_sq = axis[0]*axis[0] + axis[1]*axis[1] + axis[2]*axis[2];
        if (len_sq < 1e-20) return true; // Degenerate axis, skip

        real_t inv_len = 1.0 / std::sqrt(len_sq);
        std::array<real_t, 3> n = {axis[0]*inv_len, axis[1]*inv_len, axis[2]*inv_len};

        // Project centers
        real_t da = (b.center[0] - a.center[0])*n[0] +
                    (b.center[1] - a.center[1])*n[1] +
                    (b.center[2] - a.center[2])*n[2];

        // Project extents of a
        real_t ra = 0.0;
        for (int i = 0; i < 3; ++i) {
            real_t proj = std::abs(a.axes[i][0]*n[0] + a.axes[i][1]*n[1] + a.axes[i][2]*n[2]);
            ra += a.extents[i] * proj;
        }

        // Project extents of b
        real_t rb = 0.0;
        for (int i = 0; i < 3; ++i) {
            real_t proj = std::abs(b.axes[i][0]*n[0] + b.axes[i][1]*n[1] + b.axes[i][2]*n[2]);
            rb += b.extents[i] * proj;
        }

        return std::abs(da) <= ra + rb;
    };

    // Test face normals of this
    for (int i = 0; i < 3; ++i) {
        if (!test_axis(axes[i], *this, other)) return false;
    }

    // Test face normals of other
    for (int i = 0; i < 3; ++i) {
        if (!test_axis(other.axes[i], *this, other)) return false;
    }

    // Test edge cross products (9 axes)
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            std::array<real_t, 3> cross_axis = {
                axes[i][1]*other.axes[j][2] - axes[i][2]*other.axes[j][1],
                axes[i][2]*other.axes[j][0] - axes[i][0]*other.axes[j][2],
                axes[i][0]*other.axes[j][1] - axes[i][1]*other.axes[j][0]
            };
            if (!test_axis(cross_axis, *this, other)) return false;
        }
    }

    return true; // No separating axis found
}

//=============================================================================
// BoundingSphere Implementation
//=============================================================================

BoundingSphere::BoundingSphere(const AABB& box) {
    if (!box.is_valid()) {
        center = {0, 0, 0};
        radius = 0;
        return;
    }

    center = box.center();

    // Radius is distance from center to any corner
    real_t dx = box.max[0] - center[0];
    real_t dy = box.max[1] - center[1];
    real_t dz = box.max[2] - center[2];
    radius = std::sqrt(dx*dx + dy*dy + dz*dz);
}

void BoundingSphere::expand(const std::array<real_t, 3>& pt) {
    real_t dx = pt[0] - center[0];
    real_t dy = pt[1] - center[1];
    real_t dz = pt[2] - center[2];
    real_t dist = std::sqrt(dx*dx + dy*dy + dz*dz);

    if (dist > radius) {
        // Expand sphere to include point
        real_t new_radius = (radius + dist) * 0.5;
        real_t alpha = (new_radius - radius) / dist;
        center[0] += dx * alpha;
        center[1] += dy * alpha;
        center[2] += dz * alpha;
        radius = new_radius;
    }
}

void BoundingSphere::merge(const BoundingSphere& other) {
    real_t dx = other.center[0] - center[0];
    real_t dy = other.center[1] - center[1];
    real_t dz = other.center[2] - center[2];
    real_t dist = std::sqrt(dx*dx + dy*dy + dz*dz);

    if (dist + other.radius <= radius) {
        // Other is inside this
        return;
    }

    if (dist + radius <= other.radius) {
        // This is inside other
        *this = other;
        return;
    }

    // Compute new enclosing sphere
    real_t new_radius = (radius + dist + other.radius) * 0.5;
    real_t alpha = (new_radius - radius) / dist;
    center[0] += dx * alpha;
    center[1] += dy * alpha;
    center[2] += dz * alpha;
    radius = new_radius;
}

//=============================================================================
// BoundingVolumeBuilder Implementation
//=============================================================================

std::vector<AABB> BoundingVolumeBuilder::build_cell_aabbs(
    const MeshBase& mesh,
    Configuration cfg) {

    const index_t n_cells = mesh.n_cells();
    std::vector<AABB> boxes(n_cells);

    for (index_t cell = 0; cell < n_cells; ++cell) {
        auto bb = MeshGeometry::cell_bounding_box(mesh, cell, cfg);
        boxes[cell] = AABB(bb.min, bb.max);
    }

    return boxes;
}

std::vector<AABB> BoundingVolumeBuilder::build_face_aabbs(
    const MeshBase& mesh,
    Configuration cfg) {

    const index_t n_faces = mesh.n_faces();
    std::vector<AABB> boxes(n_faces);

    const auto& X = ((cfg == Configuration::Current || cfg == Configuration::Deformed) && mesh.has_current_coords())
                        ? mesh.X_cur()
                        : mesh.X_ref();
    const int dim = mesh.dim();
    auto get_coords = [&](index_t vid) -> std::array<real_t, 3> {
        std::array<real_t, 3> pt = {0.0, 0.0, 0.0};
        for (int d = 0; d < dim; ++d) {
            pt[d] = X[static_cast<size_t>(vid) * dim + d];
        }
        return pt;
    };

    for (index_t face = 0; face < n_faces; ++face) {
        auto vertices = mesh.face_vertices(face);

        if (vertices.empty()) {
            boxes[face] = AABB(); // Invalid box
            continue;
        }

        AABB box(get_coords(vertices[0]));
        for (size_t i = 1; i < vertices.size(); ++i) {
            box.expand(get_coords(vertices[i]));
        }

        boxes[face] = box;
    }

    return boxes;
}

std::vector<AABB> BoundingVolumeBuilder::build_edge_aabbs(
    const MeshBase& mesh,
    Configuration cfg) {

    const index_t n_edges = mesh.n_edges();
    std::vector<AABB> boxes(n_edges);

    const auto& X = ((cfg == Configuration::Current || cfg == Configuration::Deformed) && mesh.has_current_coords())
                        ? mesh.X_cur()
                        : mesh.X_ref();
    const int dim = mesh.dim();
    auto get_coords = [&](index_t vid) -> std::array<real_t, 3> {
        std::array<real_t, 3> pt = {0.0, 0.0, 0.0};
        for (int d = 0; d < dim; ++d) {
            pt[d] = X[static_cast<size_t>(vid) * dim + d];
        }
        return pt;
    };

    for (index_t edge = 0; edge < n_edges; ++edge) {
        auto vertices = mesh.edge_vertices(edge);

        if (vertices.size() < 2) {
            boxes[edge] = AABB(); // Invalid box
            continue;
        }

        AABB box(get_coords(vertices[0]));
        box.expand(get_coords(vertices[1]));

        // Include mid-edge nodes for high-order edges
        for (size_t i = 2; i < vertices.size(); ++i) {
            box.expand(get_coords(vertices[i]));
        }

        boxes[edge] = box;
    }

    return boxes;
}

AABB BoundingVolumeBuilder::build_mesh_aabb(
    const MeshBase& mesh,
    Configuration cfg) {

    auto bb = MeshGeometry::bounding_box(mesh, cfg);
    return AABB(bb.min, bb.max);
}

std::vector<BoundingSphere> BoundingVolumeBuilder::build_cell_spheres(
    const MeshBase& mesh,
    Configuration cfg) {

    const index_t n_cells = mesh.n_cells();
    std::vector<BoundingSphere> spheres(n_cells);

    for (index_t cell = 0; cell < n_cells; ++cell) {
        // Use AABB as initial approximation
        auto bb = MeshGeometry::cell_bounding_box(mesh, cell, cfg);
        spheres[cell] = BoundingSphere(AABB(bb.min, bb.max));

        // Could refine with Welzl's algorithm for tighter fit, but AABB sphere is sufficient
    }

    return spheres;
}

OBB BoundingVolumeBuilder::build_cell_obb_pca(
    const MeshBase& mesh,
    index_t cell,
    Configuration cfg) {

    // Get cell vertices
    auto vertices = mesh.cell_vertices(cell);

    if (vertices.empty()) {
        return OBB(); // Degenerate
    }

    // Get coordinate accessor
    auto get_coords = [&](index_t vid) -> std::array<real_t, 3> {
        const auto& X = ((cfg == Configuration::Current || cfg == Configuration::Deformed) && mesh.has_current_coords())
                            ? mesh.X_cur()
                            : mesh.X_ref();
        const int dim = mesh.dim();
        std::array<real_t, 3> pt = {0.0, 0.0, 0.0};
        for (int d = 0; d < dim; ++d) {
            pt[d] = X[static_cast<size_t>(vid) * dim + d];
        }
        return pt;
    };

    // Collect points
    std::vector<std::array<real_t, 3>> points;
    points.reserve(vertices.size());
    for (auto vid : vertices) {
        points.push_back(get_coords(vid));
    }

    // Compute centroid
    std::array<real_t, 3> centroid = {0, 0, 0};
    for (const auto& pt : points) {
        centroid[0] += pt[0];
        centroid[1] += pt[1];
        centroid[2] += pt[2];
    }
    centroid[0] /= points.size();
    centroid[1] /= points.size();
    centroid[2] /= points.size();

    // Compute covariance matrix
    auto cov = compute_covariance(points);

    // Compute principal axes via eigendecomposition
    auto axes = compute_principal_axes(cov);

    // Project points onto principal axes to find extents
    std::array<real_t, 3> min_proj = {std::numeric_limits<real_t>::max(),
                                       std::numeric_limits<real_t>::max(),
                                       std::numeric_limits<real_t>::max()};
    std::array<real_t, 3> max_proj = {std::numeric_limits<real_t>::lowest(),
                                       std::numeric_limits<real_t>::lowest(),
                                       std::numeric_limits<real_t>::lowest()};

    for (const auto& pt : points) {
        real_t dx = pt[0] - centroid[0];
        real_t dy = pt[1] - centroid[1];
        real_t dz = pt[2] - centroid[2];

        for (int i = 0; i < 3; ++i) {
            real_t proj = dx*axes[i][0] + dy*axes[i][1] + dz*axes[i][2];
            min_proj[i] = std::min(min_proj[i], proj);
            max_proj[i] = std::max(max_proj[i], proj);
        }
    }

    // Build OBB
    OBB obb;
    obb.axes = axes;
    obb.extents[0] = (max_proj[0] - min_proj[0]) * 0.5;
    obb.extents[1] = (max_proj[1] - min_proj[1]) * 0.5;
    obb.extents[2] = (max_proj[2] - min_proj[2]) * 0.5;

    // Adjust center to OBB center (not centroid)
    real_t mid_proj[3] = {
        (min_proj[0] + max_proj[0]) * 0.5,
        (min_proj[1] + max_proj[1]) * 0.5,
        (min_proj[2] + max_proj[2]) * 0.5
    };

    obb.center[0] = centroid[0] + mid_proj[0]*axes[0][0] + mid_proj[1]*axes[1][0] + mid_proj[2]*axes[2][0];
    obb.center[1] = centroid[1] + mid_proj[0]*axes[0][1] + mid_proj[1]*axes[1][1] + mid_proj[2]*axes[2][1];
    obb.center[2] = centroid[2] + mid_proj[0]*axes[0][2] + mid_proj[1]*axes[1][2] + mid_proj[2]*axes[2][2];

    return obb;
}

std::vector<OBB> BoundingVolumeBuilder::build_cell_obbs(
    const MeshBase& mesh,
    Configuration cfg) {

    const index_t n_cells = mesh.n_cells();
    std::vector<OBB> obbs(n_cells);

    for (index_t cell = 0; cell < n_cells; ++cell) {
        obbs[cell] = build_cell_obb_pca(mesh, cell, cfg);
    }

    return obbs;
}

std::vector<BoundingVolumeBuilder::AABBTreeNode>
BoundingVolumeBuilder::build_aabb_tree(
    const MeshBase& mesh,
    Configuration cfg,
    int max_depth,
    int max_cells_per_leaf) {

    // Build cell AABBs
    auto cell_boxes = build_cell_aabbs(mesh, cfg);

    // Create root node
    std::vector<AABBTreeNode> nodes;
    nodes.reserve(2 * cell_boxes.size()); // Rough estimate

    AABBTreeNode root;
    root.box = build_mesh_aabb(mesh, cfg);
    root.cell_indices.resize(cell_boxes.size());
    for (size_t i = 0; i < cell_boxes.size(); ++i) {
        root.cell_indices[i] = static_cast<index_t>(i);
    }

    nodes.push_back(root);

    // Recursively split
    split_aabb_tree(nodes, 0, cell_boxes, 0, max_depth, max_cells_per_leaf);

    return nodes;
}

//=============================================================================
// Private Helpers
//=============================================================================

std::array<std::array<real_t, 3>, 3>
BoundingVolumeBuilder::compute_covariance(
    const std::vector<std::array<real_t, 3>>& points) {

    // Compute centroid
    std::array<real_t, 3> centroid = {0, 0, 0};
    for (const auto& pt : points) {
        centroid[0] += pt[0];
        centroid[1] += pt[1];
        centroid[2] += pt[2];
    }
    centroid[0] /= points.size();
    centroid[1] /= points.size();
    centroid[2] /= points.size();

    // Compute covariance
    std::array<std::array<real_t, 3>, 3> cov = {{{0, 0, 0}, {0, 0, 0}, {0, 0, 0}}};

    for (const auto& pt : points) {
        real_t dx = pt[0] - centroid[0];
        real_t dy = pt[1] - centroid[1];
        real_t dz = pt[2] - centroid[2];

        cov[0][0] += dx * dx;
        cov[0][1] += dx * dy;
        cov[0][2] += dx * dz;
        cov[1][1] += dy * dy;
        cov[1][2] += dy * dz;
        cov[2][2] += dz * dz;
    }

    // Symmetric matrix
    cov[1][0] = cov[0][1];
    cov[2][0] = cov[0][2];
    cov[2][1] = cov[1][2];

    // Normalize
    real_t scale = 1.0 / points.size();
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            cov[i][j] *= scale;
        }
    }

    return cov;
}

std::array<std::array<real_t, 3>, 3>
BoundingVolumeBuilder::compute_principal_axes(
    const std::array<std::array<real_t, 3>, 3>& covariance) {

    std::array<std::array<real_t, 3>, 3> axes;

    // If Eigen is unavailable, fall back to axis-aligned directions.
#if SVMP_HAVE_EIGEN
    Eigen::Matrix3d C;
    C << covariance[0][0], covariance[0][1], covariance[0][2],
         covariance[1][0], covariance[1][1], covariance[1][2],
         covariance[2][0], covariance[2][1], covariance[2][2];

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(C);
    Eigen::Matrix3d eigenvectors = solver.eigenvectors();

    // Use eigenvectors in decreasing eigenvalue order (largest variance first)
    for (int i = 0; i < 3; ++i) {
        int col = 2 - i; // Reverse order (largest eigenvalue last in Eigen)
        axes[i][0] = eigenvectors(0, col);
        axes[i][1] = eigenvectors(1, col);
        axes[i][2] = eigenvectors(2, col);
    }
#else
    axes[0] = {1.0, 0.0, 0.0};
    axes[1] = {0.0, 1.0, 0.0};
    axes[2] = {0.0, 0.0, 1.0};
#endif

    return axes;
}

void BoundingVolumeBuilder::split_aabb_tree(
    std::vector<AABBTreeNode>& nodes,
    int node_idx,
    const std::vector<AABB>& cell_boxes,
    int depth,
    int max_depth,
    int max_cells_per_leaf) {

    AABBTreeNode& node = nodes[node_idx];

    // Check termination criteria
    if (depth >= max_depth || node.cell_indices.size() <= static_cast<size_t>(max_cells_per_leaf)) {
        return; // Leaf node
    }

    // Choose split axis (longest axis)
    int split_axis = node.box.longest_axis();

    // Sort cells by center along split axis
    std::vector<std::pair<real_t, index_t>> sorted_cells;
    sorted_cells.reserve(node.cell_indices.size());

    for (auto cell_id : node.cell_indices) {
        real_t center_coord = cell_boxes[cell_id].center()[split_axis];
        sorted_cells.push_back({center_coord, cell_id});
    }

    std::sort(sorted_cells.begin(), sorted_cells.end());

    // Split at median
    size_t mid = sorted_cells.size() / 2;

    // Create left child
    AABBTreeNode left;
    left.cell_indices.reserve(mid);
    for (size_t i = 0; i < mid; ++i) {
        left.cell_indices.push_back(sorted_cells[i].second);
        left.box.merge(cell_boxes[sorted_cells[i].second]);
    }

    // Create right child
    AABBTreeNode right;
    right.cell_indices.reserve(sorted_cells.size() - mid);
    for (size_t i = mid; i < sorted_cells.size(); ++i) {
        right.cell_indices.push_back(sorted_cells[i].second);
        right.box.merge(cell_boxes[sorted_cells[i].second]);
    }

    // Add children to node list
    int left_idx = nodes.size();
    nodes.push_back(left);
    int right_idx = nodes.size();
    nodes.push_back(right);

    // Update parent pointers
    nodes[node_idx].left_child = left_idx;
    nodes[node_idx].right_child = right_idx;
    nodes[node_idx].cell_indices.clear(); // Not a leaf anymore

    // Recurse
    split_aabb_tree(nodes, left_idx, cell_boxes, depth + 1, max_depth, max_cells_per_leaf);
    split_aabb_tree(nodes, right_idx, cell_boxes, depth + 1, max_depth, max_cells_per_leaf);
}

} // namespace svmp

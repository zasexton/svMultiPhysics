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

#ifndef SVMP_BOUNDING_VOLUME_H
#define SVMP_BOUNDING_VOLUME_H

#include "../Core/MeshTypes.h"
#include <array>
#include <vector>
#include <limits>
#include <algorithm>

namespace svmp {

// Forward declarations
class MeshBase;

/**
 * @brief Axis-Aligned Bounding Box (AABB)
 *
 * Standard AABB representation with min/max corners.
 * Provides intersection tests, merging, expansion, and containment queries.
 */
struct AABB {
    std::array<real_t, 3> min{std::numeric_limits<real_t>::max(),
                               std::numeric_limits<real_t>::max(),
                               std::numeric_limits<real_t>::max()};
    std::array<real_t, 3> max{std::numeric_limits<real_t>::lowest(),
                               std::numeric_limits<real_t>::lowest(),
                               std::numeric_limits<real_t>::lowest()};

    /// Default constructor creates invalid box (min > max)
    AABB() = default;

    /// Construct from min/max corners
    AABB(const std::array<real_t, 3>& min_pt, const std::array<real_t, 3>& max_pt)
        : min(min_pt), max(max_pt) {}

    /// Construct from single point
    explicit AABB(const std::array<real_t, 3>& pt) : min(pt), max(pt) {}

    /// Check if box is valid (min <= max in all dimensions)
    bool is_valid() const {
        return min[0] <= max[0] && min[1] <= max[1] && min[2] <= max[2];
    }

    /// Check if box is empty (min == max in any dimension)
    bool is_empty() const {
        return min[0] >= max[0] || min[1] >= max[1] || min[2] >= max[2];
    }

    /// Get box center
    std::array<real_t, 3> center() const {
        return {(min[0] + max[0]) * 0.5,
                (min[1] + max[1]) * 0.5,
                (min[2] + max[2]) * 0.5};
    }

    /// Get box extents (half-widths)
    std::array<real_t, 3> extents() const {
        return {(max[0] - min[0]) * 0.5,
                (max[1] - min[1]) * 0.5,
                (max[2] - min[2]) * 0.5};
    }

    /// Get box dimensions (full widths)
    std::array<real_t, 3> dimensions() const {
        return {max[0] - min[0], max[1] - min[1], max[2] - min[2]};
    }

    /// Get box volume
    real_t volume() const {
        if (!is_valid()) return 0.0;
        return (max[0] - min[0]) * (max[1] - min[1]) * (max[2] - min[2]);
    }

    /// Get box surface area
    real_t surface_area() const {
        if (!is_valid()) return 0.0;
        real_t dx = max[0] - min[0];
        real_t dy = max[1] - min[1];
        real_t dz = max[2] - min[2];
        return 2.0 * (dx*dy + dy*dz + dz*dx);
    }

    /// Expand box to include point
    void expand(const std::array<real_t, 3>& pt) {
        min[0] = std::min(min[0], pt[0]);
        min[1] = std::min(min[1], pt[1]);
        min[2] = std::min(min[2], pt[2]);
        max[0] = std::max(max[0], pt[0]);
        max[1] = std::max(max[1], pt[1]);
        max[2] = std::max(max[2], pt[2]);
    }

    /// Expand box by margin in all directions
    void expand(real_t margin) {
        min[0] -= margin;
        min[1] -= margin;
        min[2] -= margin;
        max[0] += margin;
        max[1] += margin;
        max[2] += margin;
    }

    /// Merge with another box
    void merge(const AABB& other) {
        if (!other.is_valid()) return;
        min[0] = std::min(min[0], other.min[0]);
        min[1] = std::min(min[1], other.min[1]);
        min[2] = std::min(min[2], other.min[2]);
        max[0] = std::max(max[0], other.max[0]);
        max[1] = std::max(max[1], other.max[1]);
        max[2] = std::max(max[2], other.max[2]);
    }

    /// Check if point is inside box (inclusive)
    bool contains(const std::array<real_t, 3>& pt) const {
        return pt[0] >= min[0] && pt[0] <= max[0] &&
               pt[1] >= min[1] && pt[1] <= max[1] &&
               pt[2] >= min[2] && pt[2] <= max[2];
    }

    /// Check if box intersects another box
    bool intersects(const AABB& other) const {
        return !(max[0] < other.min[0] || min[0] > other.max[0] ||
                 max[1] < other.min[1] || min[1] > other.max[1] ||
                 max[2] < other.min[2] || min[2] > other.max[2]);
    }

    /// Compute intersection of two boxes
    AABB intersection(const AABB& other) const {
        if (!intersects(other)) return AABB(); // Invalid box
        return AABB(
            {std::max(min[0], other.min[0]),
             std::max(min[1], other.min[1]),
             std::max(min[2], other.min[2])},
            {std::min(max[0], other.max[0]),
             std::min(max[1], other.max[1]),
             std::min(max[2], other.max[2])}
        );
    }

    /// Compute squared distance from point to box (0 if inside)
    real_t distance_squared(const std::array<real_t, 3>& pt) const {
        real_t dist_sq = 0.0;
        for (int i = 0; i < 3; ++i) {
            if (pt[i] < min[i]) {
                real_t d = min[i] - pt[i];
                dist_sq += d * d;
            } else if (pt[i] > max[i]) {
                real_t d = pt[i] - max[i];
                dist_sq += d * d;
            }
        }
        return dist_sq;
    }

    /// Get longest axis (0=x, 1=y, 2=z)
    int longest_axis() const {
        real_t dx = max[0] - min[0];
        real_t dy = max[1] - min[1];
        real_t dz = max[2] - min[2];
        if (dx >= dy && dx >= dz) return 0;
        if (dy >= dz) return 1;
        return 2;
    }
};

/**
 * @brief Oriented Bounding Box (OBB)
 *
 * OBB with arbitrary orientation defined by center, axes, and half-extents.
 * Tighter than AABB for rotated/elongated objects.
 */
struct OBB {
    std::array<real_t, 3> center{0, 0, 0};      // Box center
    std::array<std::array<real_t, 3>, 3> axes;  // Orthonormal axes (columns)
    std::array<real_t, 3> extents{0, 0, 0};     // Half-widths along each axis

    /// Default constructor creates degenerate box at origin
    OBB() {
        axes[0] = {1, 0, 0};
        axes[1] = {0, 1, 0};
        axes[2] = {0, 0, 1};
    }

    /// Get corner vertices (8 corners for 3D box)
    std::array<std::array<real_t, 3>, 8> get_corners() const;

    /// Convert to AABB (conservative bounding box)
    AABB to_aabb() const;

    /// Check if point is inside OBB
    bool contains(const std::array<real_t, 3>& pt) const;

    /// Check if OBB intersects another OBB (SAT test)
    bool intersects(const OBB& other) const;
};

/**
 * @brief Sphere bounding volume
 *
 * Simple sphere representation for efficient broad-phase testing.
 */
struct BoundingSphere {
    std::array<real_t, 3> center{0, 0, 0};
    real_t radius{0};

    BoundingSphere() = default;

    BoundingSphere(const std::array<real_t, 3>& c, real_t r)
        : center(c), radius(r) {}

    /// Construct from AABB (smallest enclosing sphere)
    explicit BoundingSphere(const AABB& box);

    /// Check if point is inside sphere
    bool contains(const std::array<real_t, 3>& pt) const {
        real_t dx = pt[0] - center[0];
        real_t dy = pt[1] - center[1];
        real_t dz = pt[2] - center[2];
        return dx*dx + dy*dy + dz*dz <= radius*radius;
    }

    /// Check if spheres intersect
    bool intersects(const BoundingSphere& other) const {
        real_t dx = other.center[0] - center[0];
        real_t dy = other.center[1] - center[1];
        real_t dz = other.center[2] - center[2];
        real_t dist_sq = dx*dx + dy*dy + dz*dz;
        real_t r_sum = radius + other.radius;
        return dist_sq <= r_sum * r_sum;
    }

    /// Expand to include point
    void expand(const std::array<real_t, 3>& pt);

    /// Merge with another sphere (not guaranteed to be minimal)
    void merge(const BoundingSphere& other);
};

/**
 * @brief Bounding volume utilities and batch builders
 *
 * Provides efficient batched construction of bounding volumes for all mesh entities.
 * SIMD-friendly loops for performance on large meshes.
 */
class BoundingVolumeBuilder {
public:
    /**
     * @brief Build AABBs for all cells in the mesh
     * @param mesh The mesh
     * @param cfg Reference or current configuration
     * @return Vector of AABBs, indexed by cell ID
     */
    static std::vector<AABB> build_cell_aabbs(
        const MeshBase& mesh,
        Configuration cfg = Configuration::Reference);

    /**
     * @brief Build AABBs for all faces in the mesh
     * @param mesh The mesh
     * @param cfg Reference or current configuration
     * @return Vector of AABBs, indexed by face ID
     */
    static std::vector<AABB> build_face_aabbs(
        const MeshBase& mesh,
        Configuration cfg = Configuration::Reference);

    /**
     * @brief Build AABBs for all edges in the mesh
     * @param mesh The mesh
     * @param cfg Reference or current configuration
     * @return Vector of AABBs, indexed by edge ID
     */
    static std::vector<AABB> build_edge_aabbs(
        const MeshBase& mesh,
        Configuration cfg = Configuration::Reference);

    /**
     * @brief Build global mesh AABB
     * @param mesh The mesh
     * @param cfg Reference or current configuration
     * @return Bounding box enclosing entire mesh
     */
    static AABB build_mesh_aabb(
        const MeshBase& mesh,
        Configuration cfg = Configuration::Reference);

    /**
     * @brief Build bounding spheres for all cells
     * @param mesh The mesh
     * @param cfg Reference or current configuration
     * @return Vector of bounding spheres, indexed by cell ID
     */
    static std::vector<BoundingSphere> build_cell_spheres(
        const MeshBase& mesh,
        Configuration cfg = Configuration::Reference);

    /**
     * @brief Build OBB for a single cell using PCA
     * @param mesh The mesh
     * @param cell Cell index
     * @param cfg Reference or current configuration
     * @return Oriented bounding box
     */
    static OBB build_cell_obb_pca(
        const MeshBase& mesh,
        index_t cell,
        Configuration cfg = Configuration::Reference);

    /**
     * @brief Build OBB for all cells using PCA
     * @param mesh The mesh
     * @param cfg Reference or current configuration
     * @return Vector of OBBs, indexed by cell ID
     */
    static std::vector<OBB> build_cell_obbs(
        const MeshBase& mesh,
        Configuration cfg = Configuration::Reference);

    /**
     * @brief Build hierarchical AABB tree for cells (for spatial search)
     * @param mesh The mesh
     * @param cfg Reference or current configuration
     * @param max_depth Maximum tree depth
     * @param max_cells_per_leaf Maximum cells per leaf node
     * @return Root AABB and tree structure (implementation-specific)
     */
    struct AABBTreeNode {
        AABB box;
        int left_child{-1};   // -1 if leaf
        int right_child{-1};
        std::vector<index_t> cell_indices;  // Non-empty only for leaves
    };

    static std::vector<AABBTreeNode> build_aabb_tree(
        const MeshBase& mesh,
        Configuration cfg = Configuration::Reference,
        int max_depth = 20,
        int max_cells_per_leaf = 10);

private:
    /// Helper: compute covariance matrix for PCA
    static std::array<std::array<real_t, 3>, 3> compute_covariance(
        const std::vector<std::array<real_t, 3>>& points);

    /// Helper: compute principal axes via eigendecomposition
    static std::array<std::array<real_t, 3>, 3> compute_principal_axes(
        const std::array<std::array<real_t, 3>, 3>& covariance);

    /// Helper: recursively split AABB tree
    static void split_aabb_tree(
        std::vector<AABBTreeNode>& nodes,
        int node_idx,
        const std::vector<AABB>& cell_boxes,
        int depth,
        int max_depth,
        int max_cells_per_leaf);
};

} // namespace svmp

#endif // SVMP_BOUNDING_VOLUME_H

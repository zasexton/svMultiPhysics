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

#ifndef SVMP_BOUNDARY_KEY_H
#define SVMP_BOUNDARY_KEY_H

#include "../Core/MeshTypes.h"
#include <vector>
#include <algorithm>
#include <functional>

namespace svmp {

/**
 * @brief Canonical boundary key for topological boundary identification
 *
 * Represents a boundary (of any dimension) by its vertex indices in a canonical (sorted) order.
 * Used for detecting topological boundaries by counting incidence.
 *
 * Examples:
 * - In 3D meshes: boundary faces (triangles, quads)
 * - In 2D meshes: boundary edges
 * - In 1D meshes: boundary vertices
 *
 * Key properties:
 * - Orientation-independent (sorted vertices)
 * - Hashable for use in unordered_map
 * - Comparable for use in ordered_map
 */
class BoundaryKey {
public:
    BoundaryKey() = default;

    /**
     * @brief Construct boundary key from vertex indices
     * @param vertices Vector of vertex indices defining the boundary
     */
    explicit BoundaryKey(const std::vector<index_t>& vertices)
        : vertices_(vertices) {
        // Canonicalize by sorting
        std::sort(vertices_.begin(), vertices_.end());
    }

    /**
     * @brief Construct boundary key from pointer and count
     * @param vertices Pointer to vertex array
     * @param n_vertices Number of vertices
     */
    BoundaryKey(const index_t* vertices, size_t n_vertices) {
        vertices_.assign(vertices, vertices + n_vertices);
        std::sort(vertices_.begin(), vertices_.end());
    }

    // Accessors
    const std::vector<index_t>& vertices() const { return vertices_; }
    size_t size() const { return vertices_.size(); }

    // Comparison operators
    bool operator==(const BoundaryKey& other) const {
        return vertices_ == other.vertices_;
    }

    bool operator!=(const BoundaryKey& other) const {
        return vertices_ != other.vertices_;
    }

    bool operator<(const BoundaryKey& other) const {
        return vertices_ < other.vertices_;
    }

    // Hash function for unordered containers
    struct Hash {
        size_t operator()(const BoundaryKey& key) const {
            size_t hash = 0;
            for (index_t v : key.vertices_) {
                // Simple hash combination
                hash ^= std::hash<index_t>()(v) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
            }
            return hash;
        }
    };

private:
    std::vector<index_t> vertices_;
};

/**
 * @brief Oriented boundary key that preserves vertex ordering
 *
 * Used when orientation information is needed (e.g., for chain complex approach).
 * Stores vertices in their original order along with a sign.
 */
class OrientedBoundaryKey {
public:
    OrientedBoundaryKey() = default;

    /**
     * @brief Construct oriented boundary key
     * @param vertices Ordered vertex indices
     * @param sign +1 or -1 indicating orientation
     */
    OrientedBoundaryKey(const std::vector<index_t>& vertices, int sign = 1)
        : vertices_(vertices), sign_(sign) {}

    OrientedBoundaryKey(const index_t* vertices, size_t n_vertices, int sign = 1)
        : vertices_(vertices, vertices + n_vertices), sign_(sign) {}

    // Get canonical (unoriented) version
    BoundaryKey canonical() const {
        return BoundaryKey(vertices_);
    }

    // Accessors
    const std::vector<index_t>& vertices() const { return vertices_; }
    int sign() const { return sign_; }
    size_t size() const { return vertices_.size(); }

    // Comparison
    bool operator==(const OrientedBoundaryKey& other) const {
        return vertices_ == other.vertices_ && sign_ == other.sign_;
    }

private:
    std::vector<index_t> vertices_;
    int sign_ = 1;
};

} // namespace svmp

#endif // SVMP_BOUNDARY_KEY_H

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

#ifndef SVMP_BOUNDARY_COMPONENT_H
#define SVMP_BOUNDARY_COMPONENT_H

#include "../Core/MeshTypes.h"
#include <vector>
#include <unordered_set>

namespace svmp {

/**
 * @brief Represents a connected component of the mesh boundary
 *
 * A boundary component is a maximal connected set of boundary (n-1) entities.
 * For example, in a 3D mesh, each separate surface patch is a component; in 2D,
 * a boundary curve is a component. Connectivity is determined through shared
 * (n-2) entities (edges in 3D, vertices in 2D).
 */
class BoundaryComponent {
public:
    BoundaryComponent() = default;
    explicit BoundaryComponent(int id) : id_(id) {}

    // ---- Basic properties ----
    int id() const { return id_; }
    void set_id(int id) { id_ = id; }

    size_t n_entities() const { return boundary_entities_.size(); }
    size_t n_vertices() const { return boundary_vertices_.size(); }

    // ---- Entity access (codimension-1) ----
    const std::vector<index_t>& entities() const { return boundary_entities_; }
    void add_entity(index_t entity_id) { boundary_entities_.push_back(entity_id); }
    void reserve_entities(size_t n) { boundary_entities_.reserve(n); }

    // ---- Vertex access ----
    const std::unordered_set<index_t>& vertices() const { return boundary_vertices_; }
    void add_vertex(index_t vertex_id) { boundary_vertices_.insert(vertex_id); }
    bool contains_vertex(index_t vertex_id) const {
        return boundary_vertices_.find(vertex_id) != boundary_vertices_.end();
    }

    // ---- Topology ----

    /**
     * @brief Check if this component is closed (manifold without boundary)
     * @return true if every edge is shared by exactly 2 faces
     */
    bool is_closed() const { return is_closed_; }
    void set_closed(bool closed) { is_closed_ = closed; }

    /**
     * @brief Check if this component is orientable
     * @return true if a consistent orientation exists
     */
    bool is_orientable() const { return is_orientable_; }
    void set_orientable(bool orientable) { is_orientable_ = orientable; }

    // ---- Geometric properties (computed separately) ----

    real_t area() const { return area_; }
    void set_area(real_t area) { area_ = area; }

    const std::array<real_t, 3>& centroid() const { return centroid_; }
    void set_centroid(const std::array<real_t, 3>& centroid) { centroid_ = centroid; }

    // ---- Utilities ----

    void clear() {
        boundary_entities_.clear();
        boundary_vertices_.clear();
        is_closed_ = false;
        is_orientable_ = true;
        area_ = 0.0;
        centroid_ = {0.0, 0.0, 0.0};
    }

    void shrink_to_fit() {
        boundary_entities_.shrink_to_fit();
    }

private:
    int id_ = -1;
    std::vector<index_t> boundary_entities_;
    std::unordered_set<index_t> boundary_vertices_;

    // Topological properties
    bool is_closed_ = false;
    bool is_orientable_ = true;

    // Geometric properties
    real_t area_ = 0.0;
    std::array<real_t, 3> centroid_ = {0.0, 0.0, 0.0};
};

} // namespace svmp

#endif // SVMP_BOUNDARY_COMPONENT_H

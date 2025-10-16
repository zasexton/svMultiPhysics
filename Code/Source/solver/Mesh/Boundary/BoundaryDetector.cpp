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

#include "BoundaryDetector.h"
#include "../Topology/CellTopology.h"
#include <queue>
#include <algorithm>
#include <stdexcept>

namespace svmp {

// ==========================================
// Construction
// ==========================================

BoundaryDetector::BoundaryDetector(const MeshBase& mesh)
    : mesh_(mesh) {
}

// ==========================================
// Main detection methods
// ==========================================

BoundaryDetector::BoundaryInfo BoundaryDetector::detect_boundary() {
    BoundaryInfo info;

    // Step 1: Compute boundary incidence
    auto boundary_incidence_map = compute_boundary_incidence();

    // Step 2: Classify boundaries and create reverse mapping
    std::unordered_map<BoundaryKey, index_t, BoundaryKey::Hash> boundary_key_to_index;
    std::vector<BoundaryKey> boundary_keys;

    index_t boundary_idx = 0;
    for (const auto& [key, incidence] : boundary_incidence_map) {
        boundary_key_to_index[key] = boundary_idx;
        boundary_keys.push_back(key);

        if (incidence.is_boundary()) {
            info.boundary_entities.push_back(boundary_idx);
            // Add vertices to boundary vertex set
            for (index_t vertex : key.vertices()) {
                info.boundary_vertices.insert(vertex);
            }
            // Store oriented vertices (right-hand rule, outward normal)
            auto orient = incidence.boundary_orientation();
            info.oriented_boundary_entities.push_back(orient);
        } else if (incidence.is_interior()) {
            info.interior_entities.push_back(boundary_idx);
        } else if (incidence.is_nonmanifold()) {
            info.nonmanifold_entities.push_back(boundary_idx);
        }

        boundary_idx++;
    }

    // Step 3: Populate boundary_types (aligned with boundary_entities)
    if (!info.boundary_entities.empty()) {
        EntityKind kind = (mesh_.dim() == 1) ? EntityKind::Vertex
                           : (mesh_.dim() == 2) ? EntityKind::Edge
                                                 : EntityKind::Face;
        info.boundary_types.resize(info.boundary_entities.size(), kind);
    }

    // Step 4: Extract connected components
    if (!info.boundary_entities.empty()) {
        info.components = extract_boundary_components(info.boundary_entities);
    }

    return info;
}

std::vector<index_t> BoundaryDetector::detect_boundary_chain_complex() {
    // Chain complex approach using Z2 arithmetic

    // Build boundary incidence
    auto boundary_incidence_map = compute_boundary_incidence();

    std::vector<index_t> boundary_faces;
    index_t boundary_idx = 0;

    for (const auto& [key, incidence] : boundary_incidence_map) {
        // Over Z2, boundary entities have odd incidence count
        if (incidence.count % 2 == 1) {
            boundary_faces.push_back(boundary_idx);
        }
        boundary_idx++;
    }

    return boundary_faces;
}

std::unordered_map<BoundaryKey, BoundaryDetector::BoundaryIncidence, BoundaryKey::Hash>
BoundaryDetector::compute_boundary_incidence() {
    std::unordered_map<BoundaryKey, BoundaryIncidence, BoundaryKey::Hash> boundary_map;

    const int dim = mesh_.dim();

    // 1D special-case: boundaries are vertices (0D). Use endpoints (corner vertices).
    if (dim == 1) {
        for (index_t c = 0; c < static_cast<index_t>(mesh_.n_cells()); ++c) {
            auto [vertices_ptr, n_vertices] = mesh_.cell_vertices_span(c);
            CellShape shape = mesh_.cell_shape(c);

                // Corner vertices are first 'num_corners' entries by convention
            const int nc = (shape.num_corners > 0) ? shape.num_corners : static_cast<int>(n_vertices);
            if (nc >= 1) {
                // Left endpoint
                std::vector<index_t> verts = {vertices_ptr[0]};
                BoundaryKey key(verts);
                auto& inc = boundary_map[key];
                inc.key = key;
                inc.incident_cells.push_back(c);
                inc.count++;
                inc.oriented_vertices.push_back(verts);
            }
            if (nc >= 2) {
                // Right endpoint (last corner)
                std::vector<index_t> verts = {vertices_ptr[nc - 1]};
                BoundaryKey key(verts);
                auto& inc = boundary_map[key];
                inc.key = key;
                inc.incident_cells.push_back(c);
                inc.count++;
                inc.oriented_vertices.push_back(verts);
            }
        }
        return boundary_map;
    }

    // Enumerate all n-cells and their (n-1)-faces for 2D/3D
    for (index_t c = 0; c < static_cast<index_t>(mesh_.n_cells()); ++c) {
        auto [vertices_ptr, n_vertices] = mesh_.cell_vertices_span(c);
        CellShape shape = mesh_.cell_shape(c);

        // Get face topology from CellTopology (no switch statements!)
        auto face_defs = CellTopology::get_boundary_faces(shape.family);
        auto oriented_face_defs = CellTopology::get_oriented_boundary_faces(shape.family);

        // Apply topology definitions to this cell's actual vertices
        for (size_t i = 0; i < face_defs.size(); ++i) {
            // Convert local indices to global vertex IDs
            std::vector<index_t> face_vertices;
            face_vertices.reserve(face_defs[i].size());
            for (index_t local_idx : face_defs[i]) {
                face_vertices.push_back(vertices_ptr[local_idx]);
            }

            // Create canonical key (sorted)
            BoundaryKey face_key(face_vertices);

            // Get oriented vertices (preserve ordering)
            std::vector<index_t> oriented_verts;
            oriented_verts.reserve(oriented_face_defs[i].size());
            for (index_t local_idx : oriented_face_defs[i]) {
                oriented_verts.push_back(vertices_ptr[local_idx]);
            }

            // Record incidence
            auto& incidence = boundary_map[face_key];
            incidence.key = face_key;
            incidence.incident_cells.push_back(c);
            incidence.count++;
            incidence.oriented_vertices.push_back(oriented_verts);
        }
    }

    return boundary_map;
}

// ==========================================
// Component extraction
// ==========================================

std::vector<BoundaryComponent> BoundaryDetector::extract_boundary_components(
    const std::vector<index_t>& boundary_entities) {

    if (boundary_entities.empty()) {
        return {};
    }

    // Build (n-2)-to-(n-1) adjacency for boundary entities
    std::unordered_map<BoundaryKey, std::vector<index_t>, BoundaryKey::Hash> sub_to_entities;

    // Get boundary incidence to retrieve boundary keys
    auto boundary_incidence_map = compute_boundary_incidence();
    std::vector<BoundaryKey> boundary_keys;
    for (const auto& [key, incidence] : boundary_incidence_map) {
        if (incidence.is_boundary()) {
            boundary_keys.push_back(key);
        }
    }

    // Build sub-entity adjacency
    for (size_t i = 0; i < boundary_entities.size(); ++i) {
        index_t entity_id = boundary_entities[i];
        const auto& boundary_key = boundary_keys[i];

        // Extract (n-2) sub-entities of this boundary
        auto subents = extract_codim2_from_codim1(boundary_key.vertices());

        for (const auto& se : subents) {
            sub_to_entities[se].push_back(entity_id);
        }
    }

    // BFS to find connected components
    std::unordered_set<index_t> visited;
    std::vector<BoundaryComponent> components;
    int component_id = 0;

    for (index_t entity_id : boundary_entities) {
        if (visited.find(entity_id) != visited.end()) {
            continue;
        }

        BoundaryComponent component(component_id++);
        bfs_boundary_component(entity_id, boundary_entities, sub_to_entities, visited, component);
        component.shrink_to_fit();
        components.push_back(std::move(component));
    }

    return components;
}

void BoundaryDetector::bfs_boundary_component(
    index_t start_entity,
    const std::vector<index_t>& boundary_entities,
    const std::unordered_map<BoundaryKey, std::vector<index_t>, BoundaryKey::Hash>& sub_to_entities,
    std::unordered_set<index_t>& visited,
    BoundaryComponent& component) const {

    std::queue<index_t> queue;
    queue.push(start_entity);
    visited.insert(start_entity);

    // Get boundary keys
    auto boundary_incidence_map = const_cast<BoundaryDetector*>(this)->compute_boundary_incidence();
    std::vector<BoundaryKey> boundary_keys;
    for (const auto& [key, incidence] : boundary_incidence_map) {
        if (incidence.is_boundary()) {
            boundary_keys.push_back(key);
        }
    }

    while (!queue.empty()) {
        index_t current_entity = queue.front();
        queue.pop();

        component.add_entity(current_entity);

        // Find boundary key for current face
        const BoundaryKey* current_key = nullptr;
        for (size_t i = 0; i < boundary_entities.size(); ++i) {
            if (boundary_entities[i] == current_entity) {
                current_key = &boundary_keys[i];
                break;
            }
        }

        if (!current_key) continue;

        // Add vertices to component
        for (index_t vertex : current_key->vertices()) {
            component.add_vertex(vertex);
        }

        // Find neighbors through shared (n-2) sub-entities
        auto subents = extract_codim2_from_codim1(current_key->vertices());

        for (const auto& se : subents) {
            auto it = sub_to_entities.find(se);
            if (it == sub_to_entities.end()) continue;

            for (index_t neighbor_entity : it->second) {
                if (neighbor_entity != current_entity && visited.find(neighbor_entity) == visited.end()) {
                    visited.insert(neighbor_entity);
                    queue.push(neighbor_entity);
                }
            }
        }
    }
}

// ==========================================
// Cell face extraction
// ==========================================

std::vector<BoundaryKey> BoundaryDetector::extract_cell_codim1(index_t cell_id) const {
    auto [vertices_ptr, n_vertices] = mesh_.cell_vertices_span(cell_id);
    CellShape shape = mesh_.cell_shape(cell_id);

    // Get face topology from CellTopology
    auto face_defs = CellTopology::get_boundary_faces(shape.family);

    // Convert local indices to global vertex IDs and create boundary keys
    std::vector<BoundaryKey> boundary_keys;
    boundary_keys.reserve(face_defs.size());

    for (const auto& face_def : face_defs) {
        std::vector<index_t> face_vertices;
        face_vertices.reserve(face_def.size());
        for (index_t local_idx : face_def) {
            face_vertices.push_back(vertices_ptr[local_idx]);
        }
        boundary_keys.emplace_back(face_vertices);
    }

    return boundary_keys;
}


// ==========================================
// Edge extraction from faces
// ==========================================

std::vector<BoundaryKey> BoundaryDetector::extract_codim2_from_codim1(
    const std::vector<index_t>& entity_vertices) const {

    return extract_codim2_from_ring(entity_vertices);
}

std::vector<BoundaryKey> BoundaryDetector::extract_codim2_from_ring(
    const std::vector<index_t>& vertices) const {

    std::vector<BoundaryKey> edges;
    edges.reserve(vertices.size());

    for (size_t i = 0; i < vertices.size(); ++i) {
        size_t next = (i + 1) % vertices.size();
        edges.emplace_back(std::vector<index_t>{vertices[i], vertices[next]});
    }

    return edges;
}

// ==========================================
// Utilities
// ==========================================

bool BoundaryDetector::is_closed_mesh() {
    auto boundary_incidence_map = compute_boundary_incidence();

    for (const auto& [key, incidence] : boundary_incidence_map) {
        if (incidence.count % 2 == 1) {
            return false;
        }
    }

    return true;
}

std::vector<index_t> BoundaryDetector::detect_nonmanifold_codim1() {
    auto boundary_incidence_map = compute_boundary_incidence();

    std::vector<index_t> nonmanifold_entities;
    index_t boundary_idx = 0;

    for (const auto& [key, incidence] : boundary_incidence_map) {
        if (incidence.is_nonmanifold()) {
            nonmanifold_entities.push_back(boundary_idx);
        }
        boundary_idx++;
    }

    return nonmanifold_entities;
}

} // namespace svmp

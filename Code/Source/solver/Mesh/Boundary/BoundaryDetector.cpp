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
namespace {

int cell_topological_dim(CellFamily family) {
    switch (family) {
        case CellFamily::Line:
            return 1;
        case CellFamily::Triangle:
        case CellFamily::Quad:
        case CellFamily::Polygon:
            return 2;
        case CellFamily::Tetra:
        case CellFamily::Hex:
        case CellFamily::Wedge:
        case CellFamily::Pyramid:
        case CellFamily::Polyhedron:
            return 3;
        case CellFamily::Point:
            return 0;
    }
    return 0;
}

int mesh_topological_dim(const MeshBase& mesh) {
    int tdim = 0;
    for (index_t c = 0; c < static_cast<index_t>(mesh.n_cells()); ++c) {
        tdim = std::max(tdim, cell_topological_dim(mesh.cell_shape(c).family));
    }
    return tdim;
}

EntityKind boundary_entity_kind(int topo_dim) {
    switch (topo_dim) {
        case 1: return EntityKind::Vertex;
        case 2: return EntityKind::Edge;
        case 3: return EntityKind::Face;
        default: return EntityKind::Vertex;
    }
}

struct OrderKind {
    int p = 1;
    CellTopology::HighOrderKind kind = CellTopology::HighOrderKind::Lagrange;
};

OrderKind deduce_order_kind(const CellShape& shape, size_t n_vertices) {
    // Prefer inference from node count (VTK-style ordering) when possible.
    const int p_ser = CellTopology::infer_serendipity_order(shape.family, n_vertices);
    if (p_ser > 0) {
        return {p_ser, CellTopology::HighOrderKind::Serendipity};
    }

    const int p_lag = CellTopology::infer_lagrange_order(shape.family, n_vertices);
    if (p_lag > 0) {
        return {p_lag, CellTopology::HighOrderKind::Lagrange};
    }

    if (shape.order > 1) {
        return {shape.order, CellTopology::HighOrderKind::Lagrange};
    }

    return {1, CellTopology::HighOrderKind::Lagrange};
}

std::vector<BoundaryKey> sorted_entity_keys(
    const std::unordered_map<BoundaryKey, BoundaryDetector::BoundaryIncidence, BoundaryKey::Hash>& incidence_map) {

    std::vector<BoundaryKey> keys;
    keys.reserve(incidence_map.size());
    for (const auto& kv : incidence_map) {
        keys.push_back(kv.first);
    }
    std::sort(keys.begin(), keys.end());
    return keys;
}

} // namespace

// ==========================================
// Construction
// ==========================================

BoundaryDetector::BoundaryDetector(const MeshBase& mesh)
    : mesh_(mesh),
      topo_dim_(mesh_topological_dim(mesh)) {
}

// ==========================================
// Main detection methods
// ==========================================

BoundaryDetector::BoundaryInfo BoundaryDetector::detect_boundary() {
    BoundaryInfo info;

    // Step 1: Compute boundary incidence
    auto boundary_incidence_map = compute_boundary_incidence();

    // Step 2: Classify entities in a stable (sorted) ordering.
    auto entity_keys = sorted_entity_keys(boundary_incidence_map);

    for (index_t entity_id = 0; entity_id < static_cast<index_t>(entity_keys.size()); ++entity_id) {
        const auto& key = entity_keys[static_cast<size_t>(entity_id)];
        const auto& incidence = boundary_incidence_map.at(key);

        if (incidence.is_boundary()) {
            info.boundary_entities.push_back(entity_id);
            for (index_t vertex : incidence.boundary_entity_vertices()) {
                info.boundary_vertices.insert(vertex);
            }
            info.oriented_boundary_entities.push_back(incidence.boundary_orientation());
        } else if (incidence.is_interior()) {
            info.interior_entities.push_back(entity_id);
        } else if (incidence.is_nonmanifold()) {
            info.nonmanifold_entities.push_back(entity_id);
        }
    }

    info.entity_keys = std::move(entity_keys);

    // Step 3: Populate boundary_types (aligned with boundary_entities)
    if (!info.boundary_entities.empty()) {
        EntityKind kind = boundary_entity_kind(topo_dim_);
        info.boundary_types.resize(info.boundary_entities.size(), kind);
    }

    // Step 4: Extract connected components
    if (!info.boundary_entities.empty()) {
        info.components = extract_boundary_components_impl(info.boundary_entities, info.entity_keys, boundary_incidence_map);
    }

    return info;
}

std::vector<index_t> BoundaryDetector::detect_boundary_chain_complex() {
    // Chain complex approach using Z2 arithmetic

    // Build boundary incidence
    auto boundary_incidence_map = compute_boundary_incidence();
    const auto entity_keys = sorted_entity_keys(boundary_incidence_map);

    std::vector<index_t> boundary_faces;
    for (index_t entity_id = 0; entity_id < static_cast<index_t>(entity_keys.size()); ++entity_id) {
        const auto& key = entity_keys[static_cast<size_t>(entity_id)];
        const auto& incidence = boundary_incidence_map.at(key);
        // Over Z2, boundary entities have odd incidence count
        if (incidence.count % 2 == 1) {
            boundary_faces.push_back(entity_id);
        }
    }

    return boundary_faces;
}

std::unordered_map<BoundaryKey, BoundaryDetector::BoundaryIncidence, BoundaryKey::Hash>
BoundaryDetector::compute_boundary_incidence() const {
    std::unordered_map<BoundaryKey, BoundaryIncidence, BoundaryKey::Hash> boundary_map;

    const int tdim = topo_dim_;

    if (tdim == 0) {
        return boundary_map;
    }

    // 1D special-case: boundaries are vertices (0D). Use endpoints (corner vertices).
    if (tdim == 1) {
        for (index_t c = 0; c < static_cast<index_t>(mesh_.n_cells()); ++c) {
            auto [vertices_ptr, n_vertices] = mesh_.cell_vertices_span(c);
            const CellShape& shape = mesh_.cell_shape(c);
            if (cell_topological_dim(shape.family) != tdim) {
                continue;
            }
            if (shape.family != CellFamily::Line) {
                continue;
            }

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

    // Enumerate all maximal n-cells and their (n-1)-entities for 2D/3D
    for (index_t c = 0; c < static_cast<index_t>(mesh_.n_cells()); ++c) {
        auto [vertices_ptr, n_vertices] = mesh_.cell_vertices_span(c);
        const CellShape& shape = mesh_.cell_shape(c);
        if (cell_topological_dim(shape.family) != tdim) {
            continue;
        }

        const auto ok = deduce_order_kind(shape, n_vertices);
        const int p = ok.p;
        const auto kind = ok.kind;
        const int edge_steps = std::max(0, p - 1);

        if (shape.family == CellFamily::Polygon && tdim == 2) {
            // 2D polygon: boundary entities are edges between consecutive corners.
            const int nc = (shape.num_corners > 0) ? shape.num_corners : static_cast<int>(n_vertices);
            if (nc < 2) {
                continue;
            }
            for (int i = 0; i < nc; ++i) {
                index_t a = vertices_ptr[i];
                index_t b = vertices_ptr[(i + 1) % nc];

                std::vector<index_t> edge_verts = {a, b};
                BoundaryKey edge_key(edge_verts);

                auto& incidence = boundary_map[edge_key];
                incidence.key = edge_key;
                incidence.incident_cells.push_back(c);
                incidence.count++;
                incidence.oriented_vertices.push_back(edge_verts);
                incidence.entity_vertices.push_back(edge_verts);
            }
            continue;
        }

        // Get codim-1 topology from CellTopology.
        auto face_defs = CellTopology::get_boundary_faces(shape.family);
        auto oriented_face_defs = CellTopology::get_oriented_boundary_faces(shape.family);
        if (face_defs.size() != oriented_face_defs.size()) {
            throw std::runtime_error("CellTopology returned mismatched canonical/oriented boundary face tables");
        }

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

            // Oriented vertices suitable for geometry: cyclic boundary ring for faces, polyline for edges.
            // Full entity vertices may include higher-order nodes (edge/face interior) if present.
            std::vector<index_t> ring_local;
            std::vector<index_t> entity_local;

            const int fv = static_cast<int>(oriented_face_defs[i].size());
            if (p <= 1) {
                ring_local = oriented_face_defs[i];
                entity_local = oriented_face_defs[i];
            } else {
                entity_local = CellTopology::high_order_face_local_nodes(shape.family, p,
                                                                         static_cast<int>(i), kind);

                if (fv == 2) {
                    // 2D: codim-1 entity is an edge.
                    // high_order_face_local_nodes() already returns [end0, interior..., end1] in the
                    // oriented edge direction, so it can be used directly as an oriented polyline.
                    ring_local = entity_local;
                } else {
                    // 3D: codim-1 entity is a face. Build a cyclic ring by interleaving edge nodes.
                    const size_t min_sz = static_cast<size_t>(fv) + static_cast<size_t>(fv) * static_cast<size_t>(edge_steps);
                    if (entity_local.size() < min_sz) {
                        throw std::runtime_error("BoundaryDetector: unexpected high-order face node list size");
                    }
                    ring_local.reserve(min_sz);
                    for (int j = 0; j < fv; ++j) {
                        ring_local.push_back(oriented_face_defs[i][static_cast<size_t>(j)]);
                        for (int k = 0; k < edge_steps; ++k) {
                            const size_t idx = static_cast<size_t>(fv) +
                                               static_cast<size_t>(j) * static_cast<size_t>(edge_steps) +
                                               static_cast<size_t>(k);
                            ring_local.push_back(entity_local[idx]);
                        }
                    }
                }
            }

            auto local_to_global = [&](index_t local_idx) -> index_t {
                const size_t idx = static_cast<size_t>(local_idx);
                if (idx >= n_vertices) {
                    throw std::runtime_error("BoundaryDetector: local node index out of range for cell connectivity");
                }
                return vertices_ptr[idx];
            };

            std::vector<index_t> oriented_verts;
            oriented_verts.reserve(ring_local.size());
            for (index_t li : ring_local) {
                oriented_verts.push_back(local_to_global(li));
            }

            std::vector<index_t> entity_verts;
            entity_verts.reserve(entity_local.size());
            for (index_t li : entity_local) {
                entity_verts.push_back(local_to_global(li));
            }

            // Record incidence
            auto& incidence = boundary_map[face_key];
            incidence.key = face_key;
            incidence.incident_cells.push_back(c);
            incidence.count++;
            incidence.oriented_vertices.push_back(std::move(oriented_verts));
            incidence.entity_vertices.push_back(std::move(entity_verts));
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

    auto boundary_incidence_map = compute_boundary_incidence();
    const auto entity_keys = sorted_entity_keys(boundary_incidence_map);
    return extract_boundary_components_impl(boundary_entities, entity_keys, boundary_incidence_map);
}

std::vector<BoundaryComponent> BoundaryDetector::extract_boundary_components_impl(
    const std::vector<index_t>& boundary_entities,
    const std::vector<BoundaryKey>& all_entities,
    const std::unordered_map<BoundaryKey, BoundaryIncidence, BoundaryKey::Hash>& incidence_map) const {

    if (boundary_entities.empty()) {
        return {};
    }

    // Build (n-2)-to-(n-1) adjacency for boundary entities.
    std::unordered_map<BoundaryKey, std::vector<index_t>, BoundaryKey::Hash> sub_to_entities;

    for (index_t entity_id : boundary_entities) {
        if (entity_id < 0 || static_cast<size_t>(entity_id) >= all_entities.size()) {
            throw std::out_of_range("extract_boundary_components: boundary entity id out of range");
        }

        const auto& entity_key = all_entities[static_cast<size_t>(entity_id)];
        auto it = incidence_map.find(entity_key);
        if (it == incidence_map.end()) {
            continue;
        }

        const auto& codim1_vertices = it->second.boundary_orientation();

        const auto subents = extract_codim2_from_codim1(codim1_vertices);
        for (const auto& se : subents) {
            sub_to_entities[se].push_back(entity_id);
        }
    }

    // BFS to find connected components.
    std::unordered_set<index_t> visited;
    visited.reserve(boundary_entities.size());

    std::vector<BoundaryComponent> components;
    int component_id = 0;

    for (index_t start_entity : boundary_entities) {
        if (visited.find(start_entity) != visited.end()) {
            continue;
        }

        BoundaryComponent component(component_id++);

        std::queue<index_t> queue;
        queue.push(start_entity);
        visited.insert(start_entity);

        while (!queue.empty()) {
            index_t current_entity = queue.front();
            queue.pop();

            component.add_entity(current_entity);

            const auto& entity_key = all_entities[static_cast<size_t>(current_entity)];
            auto it = incidence_map.find(entity_key);
            if (it != incidence_map.end()) {
                for (index_t vertex : it->second.boundary_entity_vertices()) {
                    component.add_vertex(vertex);
                }
            } else {
                for (index_t vertex : entity_key.vertices()) {
                    component.add_vertex(vertex);
                }
            }

            const auto& codim1_vertices =
                (it != incidence_map.end()) ? it->second.boundary_orientation() : entity_key.vertices();

            const auto subents = extract_codim2_from_codim1(codim1_vertices);
            for (const auto& se : subents) {
                auto sit = sub_to_entities.find(se);
                if (sit == sub_to_entities.end()) {
                    continue;
                }
                for (index_t neighbor_entity : sit->second) {
                    if (visited.insert(neighbor_entity).second) {
                        queue.push(neighbor_entity);
                    }
                }
            }
        }

        component.shrink_to_fit();
        components.push_back(std::move(component));
    }

    return components;
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

    // In a topologically 3D mesh, codim-1 entities are faces and codim-2 are edges.
    // In a topologically 2D mesh, codim-1 entities are edges and codim-2 are vertices.
    // In a topologically 1D mesh, codim-1 entities are vertices; we treat them as disconnected.
    if (topo_dim_ == 3) {
        // Requires vertices in cyclic order for polygonal faces.
        return extract_codim2_from_ring(entity_vertices);
    }

    if (topo_dim_ == 2) {
        std::vector<BoundaryKey> verts;
        verts.reserve(entity_vertices.size());
        for (index_t v : entity_vertices) {
            verts.emplace_back(std::vector<index_t>{v});
        }
        return verts;
    }

    return {};
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
    const auto entity_keys = sorted_entity_keys(boundary_incidence_map);

    std::vector<index_t> nonmanifold_entities;
    for (index_t entity_id = 0; entity_id < static_cast<index_t>(entity_keys.size()); ++entity_id) {
        const auto& key = entity_keys[static_cast<size_t>(entity_id)];
        const auto& incidence = boundary_incidence_map.at(key);
        if (incidence.is_nonmanifold()) {
            nonmanifold_entities.push_back(entity_id);
        }
    }

    return nonmanifold_entities;
}

} // namespace svmp

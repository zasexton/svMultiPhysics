/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "EntityDofMap.h"
#include <algorithm>
#include <unordered_set>

namespace svmp {
namespace FE {
namespace dofs {

// =============================================================================
// Construction
// =============================================================================

EntityDofMap::EntityDofMap() = default;
EntityDofMap::~EntityDofMap() = default;

EntityDofMap::EntityDofMap(EntityDofMap&&) noexcept = default;
EntityDofMap& EntityDofMap::operator=(EntityDofMap&&) noexcept = default;

// =============================================================================
// Setup
// =============================================================================

void EntityDofMap::reserve(GlobalIndex n_vertices, GlobalIndex n_edges,
                            GlobalIndex n_faces, GlobalIndex n_cells) {
    n_vertices_ = n_vertices;
    n_edges_ = n_edges;
    n_faces_ = n_faces;
    n_cells_ = n_cells;

    if (n_vertices > 0) {
        vertex_offsets_.resize(static_cast<std::size_t>(n_vertices) + 1, 0);
    }
    if (n_edges > 0) {
        edge_offsets_.resize(static_cast<std::size_t>(n_edges) + 1, 0);
    }
    if (n_faces > 0) {
        face_offsets_.resize(static_cast<std::size_t>(n_faces) + 1, 0);
    }
    if (n_cells > 0) {
        cell_interior_offsets_.resize(static_cast<std::size_t>(n_cells) + 1, 0);
    }
}

void EntityDofMap::setVertexDofs(GlobalIndex vertex_id, std::span<const GlobalIndex> dofs) {
    if (vertex_id < 0) {
        throw FEException("EntityDofMap::setVertexDofs: negative vertex_id");
    }

    auto idx = static_cast<std::size_t>(vertex_id);
    if (idx >= vertex_offsets_.size()) {
        vertex_offsets_.resize(idx + 2, 0);
        n_vertices_ = std::max(n_vertices_, vertex_id + 1);
    }

    vertex_offsets_[idx] = static_cast<GlobalIndex>(vertex_dofs_.size());
    vertex_dofs_.insert(vertex_dofs_.end(), dofs.begin(), dofs.end());
    vertex_offsets_[idx + 1] = static_cast<GlobalIndex>(vertex_dofs_.size());
}

void EntityDofMap::setEdgeDofs(GlobalIndex edge_id, std::span<const GlobalIndex> dofs) {
    if (edge_id < 0) {
        throw FEException("EntityDofMap::setEdgeDofs: negative edge_id");
    }

    auto idx = static_cast<std::size_t>(edge_id);
    if (idx >= edge_offsets_.size()) {
        edge_offsets_.resize(idx + 2, 0);
        n_edges_ = std::max(n_edges_, edge_id + 1);
    }

    edge_offsets_[idx] = static_cast<GlobalIndex>(edge_dofs_.size());
    edge_dofs_.insert(edge_dofs_.end(), dofs.begin(), dofs.end());
    edge_offsets_[idx + 1] = static_cast<GlobalIndex>(edge_dofs_.size());
}

void EntityDofMap::setFaceDofs(GlobalIndex face_id, std::span<const GlobalIndex> dofs) {
    if (face_id < 0) {
        throw FEException("EntityDofMap::setFaceDofs: negative face_id");
    }

    auto idx = static_cast<std::size_t>(face_id);
    if (idx >= face_offsets_.size()) {
        face_offsets_.resize(idx + 2, 0);
        n_faces_ = std::max(n_faces_, face_id + 1);
    }

    face_offsets_[idx] = static_cast<GlobalIndex>(face_dofs_.size());
    face_dofs_.insert(face_dofs_.end(), dofs.begin(), dofs.end());
    face_offsets_[idx + 1] = static_cast<GlobalIndex>(face_dofs_.size());
}

void EntityDofMap::setCellInteriorDofs(GlobalIndex cell_id, std::span<const GlobalIndex> dofs) {
    if (cell_id < 0) {
        throw FEException("EntityDofMap::setCellInteriorDofs: negative cell_id");
    }

    auto idx = static_cast<std::size_t>(cell_id);
    if (idx >= cell_interior_offsets_.size()) {
        cell_interior_offsets_.resize(idx + 2, 0);
        n_cells_ = std::max(n_cells_, cell_id + 1);
    }

    cell_interior_offsets_[idx] = static_cast<GlobalIndex>(cell_interior_dofs_.size());
    cell_interior_dofs_.insert(cell_interior_dofs_.end(), dofs.begin(), dofs.end());
    cell_interior_offsets_[idx + 1] = static_cast<GlobalIndex>(cell_interior_dofs_.size());
}

void EntityDofMap::buildReverseMapping() {
    dof_to_entity_.clear();

    // Add vertex DOFs
    for (GlobalIndex v = 0; v < n_vertices_; ++v) {
        for (auto dof : getVertexDofs(v)) {
            dof_to_entity_[dof] = {EntityKind::Vertex, v};
        }
    }

    // Add edge DOFs
    for (GlobalIndex e = 0; e < n_edges_; ++e) {
        for (auto dof : getEdgeDofs(e)) {
            dof_to_entity_[dof] = {EntityKind::Edge, e};
        }
    }

    // Add face DOFs
    for (GlobalIndex f = 0; f < n_faces_; ++f) {
        for (auto dof : getFaceDofs(f)) {
            dof_to_entity_[dof] = {EntityKind::Face, f};
        }
    }

    // Add cell interior DOFs
    for (GlobalIndex c = 0; c < n_cells_; ++c) {
        for (auto dof : getCellInteriorDofs(c)) {
            dof_to_entity_[dof] = {EntityKind::Cell, c};
        }
    }

    has_reverse_map_ = true;
}

void EntityDofMap::finalize() {
    // Ensure all offset arrays have proper final values
    if (!vertex_offsets_.empty()) {
        vertex_offsets_.back() = static_cast<GlobalIndex>(vertex_dofs_.size());
    }
    if (!edge_offsets_.empty()) {
        edge_offsets_.back() = static_cast<GlobalIndex>(edge_dofs_.size());
    }
    if (!face_offsets_.empty()) {
        face_offsets_.back() = static_cast<GlobalIndex>(face_dofs_.size());
    }
    if (!cell_interior_offsets_.empty()) {
        cell_interior_offsets_.back() = static_cast<GlobalIndex>(cell_interior_dofs_.size());
    }

    finalized_ = true;
}

// =============================================================================
// Entity -> DOF Queries
// =============================================================================

std::span<const GlobalIndex> EntityDofMap::getVertexDofs(GlobalIndex vertex_id) const {
    return getSpan(vertex_offsets_, vertex_dofs_, vertex_id);
}

std::span<const GlobalIndex> EntityDofMap::getEdgeDofs(GlobalIndex edge_id) const {
    return getSpan(edge_offsets_, edge_dofs_, edge_id);
}

std::span<const GlobalIndex> EntityDofMap::getFaceDofs(GlobalIndex face_id) const {
    return getSpan(face_offsets_, face_dofs_, face_id);
}

std::span<const GlobalIndex> EntityDofMap::getCellInteriorDofs(GlobalIndex cell_id) const {
    return getSpan(cell_interior_offsets_, cell_interior_dofs_, cell_id);
}

std::span<const GlobalIndex> EntityDofMap::getEntityDofs(EntityKind kind,
                                                          GlobalIndex id) const {
    switch (kind) {
        case EntityKind::Vertex:
            return getVertexDofs(id);
        case EntityKind::Edge:
            return getEdgeDofs(id);
        case EntityKind::Face:
            return getFaceDofs(id);
        case EntityKind::Cell:
            return getCellInteriorDofs(id);
        default:
            return {};
    }
}

// =============================================================================
// DOF -> Entity Queries
// =============================================================================

std::optional<EntityRef> EntityDofMap::getDofEntity(GlobalIndex dof_id) const {
    if (!has_reverse_map_) {
        throw FEException("EntityDofMap::getDofEntity: reverse mapping not built");
    }

    auto it = dof_to_entity_.find(dof_id);
    if (it != dof_to_entity_.end()) {
        return it->second;
    }
    return std::nullopt;
}

std::vector<EntityRef> EntityDofMap::getDofSupportEntities(GlobalIndex dof_id) const {
    std::vector<EntityRef> result;

    // Get the primary entity
    auto primary = getDofEntity(dof_id);
    if (primary) {
        result.push_back(*primary);
    }

    // For a complete implementation, we would need mesh connectivity
    // to find all higher-dimensional entities containing the primary entity.
    // This requires access to the mesh topology which we don't have here.
    // The caller should use mesh topology queries if they need this.

    return result;
}

// =============================================================================
// Batch Queries
// =============================================================================

std::vector<GlobalIndex> EntityDofMap::getBoundaryFaceDofs(
    int boundary_id,
    std::span<const int> face_boundary_labels) const {

    std::vector<GlobalIndex> result;
    std::unordered_set<GlobalIndex> seen;

    for (GlobalIndex f = 0; f < static_cast<GlobalIndex>(face_boundary_labels.size()); ++f) {
        if (face_boundary_labels[static_cast<std::size_t>(f)] == boundary_id) {
            for (auto dof : getFaceDofs(f)) {
                if (seen.insert(dof).second) {
                    result.push_back(dof);
                }
            }
        }
    }

    std::sort(result.begin(), result.end());
    return result;
}

std::vector<GlobalIndex> EntityDofMap::getInterfaceDofs() const {
    std::unordered_set<GlobalIndex> interface_set;

    // All vertex DOFs are interface DOFs
    for (auto dof : vertex_dofs_) {
        interface_set.insert(dof);
    }

    // All edge DOFs are interface DOFs
    for (auto dof : edge_dofs_) {
        interface_set.insert(dof);
    }

    // All face DOFs are interface DOFs
    for (auto dof : face_dofs_) {
        interface_set.insert(dof);
    }

    // Cell interior DOFs are NOT interface DOFs

    std::vector<GlobalIndex> result(interface_set.begin(), interface_set.end());
    std::sort(result.begin(), result.end());
    return result;
}

// =============================================================================
// Statistics
// =============================================================================

EntityDofMap::Statistics EntityDofMap::getStatistics() const {
    Statistics stats;
    stats.n_vertex_dofs = static_cast<GlobalIndex>(vertex_dofs_.size());
    stats.n_edge_dofs = static_cast<GlobalIndex>(edge_dofs_.size());
    stats.n_face_dofs = static_cast<GlobalIndex>(face_dofs_.size());
    stats.n_cell_interior_dofs = static_cast<GlobalIndex>(cell_interior_dofs_.size());
    stats.total_dofs = stats.n_vertex_dofs + stats.n_edge_dofs +
                       stats.n_face_dofs + stats.n_cell_interior_dofs;
    return stats;
}

// =============================================================================
// Internal Helpers
// =============================================================================

std::span<const GlobalIndex> EntityDofMap::getSpan(const std::vector<GlobalIndex>& offsets,
                                                    const std::vector<GlobalIndex>& data,
                                                    GlobalIndex id) const {
    if (id < 0) {
        return {};
    }

    auto idx = static_cast<std::size_t>(id);
    if (idx + 1 >= offsets.size()) {
        return {};
    }

    auto start = static_cast<std::size_t>(offsets[idx]);
    auto end = static_cast<std::size_t>(offsets[idx + 1]);

    if (start > data.size() || end > data.size() || start > end) {
        return {};
    }

    return {data.data() + start, end - start};
}

} // namespace dofs
} // namespace FE
} // namespace svmp

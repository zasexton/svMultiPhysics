/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_DOFS_ENTITYDOFMAP_H
#define SVMP_FE_DOFS_ENTITYDOFMAP_H

/**
 * @file EntityDofMap.h
 * @brief Entity-to-DOF associations (vertex/edge/face/cell DOFs)
 *
 * The EntityDofMap provides mapping from mesh entities (vertices, edges,
 * faces, cells) to their associated DOFs. This is essential for:
 *  - Applying Dirichlet BCs on boundaries (face/edge DOFs)
 *  - Periodic constraints (face-to-face DOF pairing)
 *  - Hanging node constraints (edge/face constraints)
 *  - Static condensation (cell interior vs interface DOFs)
 *  - Debugging (which entity does a DOF belong to?)
 *
 * The class stores entity->DOF mappings in CSR format for each entity kind.
 */

#include "DofMap.h"
#include "Core/Types.h"
#include "Core/FEException.h"

#include <vector>
#include <span>
#include <unordered_map>
#include <optional>

namespace svmp {
namespace FE {
namespace dofs {

/**
 * @brief Entity kind for DOF association
 */
enum class EntityKind : std::uint8_t {
    Vertex = 0,
    Edge = 1,
    Face = 2,
    Cell = 3
};

/**
 * @brief Entity reference (type + ID)
 */
struct EntityRef {
    EntityKind kind{EntityKind::Vertex};
    GlobalIndex id{-1};

    bool operator==(const EntityRef& other) const noexcept {
        return kind == other.kind && id == other.id;
    }
};

/**
 * @brief Hash function for EntityRef
 */
struct EntityRefHash {
    std::size_t operator()(const EntityRef& ref) const noexcept {
        return std::hash<GlobalIndex>()(ref.id) ^
               (std::hash<std::uint8_t>()(static_cast<std::uint8_t>(ref.kind)) << 16);
    }
};

/**
 * @brief Entity-to-DOF mapping for all entity types
 *
 * This class stores the DOFs associated with each mesh entity in a
 * compressed format. For high-order elements:
 * - Vertices have vertex DOFs
 * - Edges have edge-interior DOFs (not vertex DOFs)
 * - Faces have face-interior DOFs (not edge/vertex DOFs)
 * - Cells have cell-interior DOFs (not face/edge/vertex DOFs)
 *
 * The canonical ordering on shared entities (edges/faces) is based on
 * global vertex IDs, consistent with Spaces::OrientationManager.
 */
class EntityDofMap {
public:
    // =========================================================================
    // Construction
    // =========================================================================

    EntityDofMap();
    ~EntityDofMap();

    // Move semantics
    EntityDofMap(EntityDofMap&&) noexcept;
    EntityDofMap& operator=(EntityDofMap&&) noexcept;

    // No copy (large data)
    EntityDofMap(const EntityDofMap&) = delete;
    EntityDofMap& operator=(const EntityDofMap&) = delete;

    // =========================================================================
    // Setup
    // =========================================================================

    /**
     * @brief Reserve storage for expected entity counts
     *
     * @param n_vertices Number of vertices
     * @param n_edges Number of edges
     * @param n_faces Number of faces
     * @param n_cells Number of cells
     */
    void reserve(GlobalIndex n_vertices, GlobalIndex n_edges,
                 GlobalIndex n_faces, GlobalIndex n_cells);

    /**
     * @brief Set DOFs for a vertex
     */
    void setVertexDofs(GlobalIndex vertex_id, std::span<const GlobalIndex> dofs);

    /**
     * @brief Set DOFs for an edge
     */
    void setEdgeDofs(GlobalIndex edge_id, std::span<const GlobalIndex> dofs);

    /**
     * @brief Set DOFs for a face
     */
    void setFaceDofs(GlobalIndex face_id, std::span<const GlobalIndex> dofs);

    /**
     * @brief Set interior DOFs for a cell
     */
    void setCellInteriorDofs(GlobalIndex cell_id, std::span<const GlobalIndex> dofs);

    /**
     * @brief Build reverse mapping (DOF -> entity)
     *
     * Call after all entity DOFs are set. Enables getDofEntity() queries.
     */
    void buildReverseMapping();

    /**
     * @brief Finalize the entity DOF map
     */
    void finalize();

    // =========================================================================
    // Entity -> DOF Queries
    // =========================================================================

    /**
     * @brief Get DOFs for a vertex
     *
     * @param vertex_id Vertex index
     * @return DOFs on this vertex
     */
    [[nodiscard]] std::span<const GlobalIndex> getVertexDofs(GlobalIndex vertex_id) const;

    /**
     * @brief Get DOFs for an edge
     *
     * @param edge_id Edge index
     * @return DOFs on this edge (edge-interior only)
     */
    [[nodiscard]] std::span<const GlobalIndex> getEdgeDofs(GlobalIndex edge_id) const;

    /**
     * @brief Get DOFs for a face
     *
     * @param face_id Face index
     * @return DOFs on this face (face-interior only)
     */
    [[nodiscard]] std::span<const GlobalIndex> getFaceDofs(GlobalIndex face_id) const;

    /**
     * @brief Get cell-interior DOFs
     *
     * @param cell_id Cell index
     * @return Interior DOFs (not on any sub-entity)
     */
    [[nodiscard]] std::span<const GlobalIndex> getCellInteriorDofs(GlobalIndex cell_id) const;

    /**
     * @brief Get all DOFs for an entity
     *
     * @param kind Entity kind
     * @param id Entity index
     * @return DOFs on this entity
     */
    [[nodiscard]] std::span<const GlobalIndex> getEntityDofs(EntityKind kind,
                                                              GlobalIndex id) const;

    // =========================================================================
    // DOF -> Entity Queries (requires buildReverseMapping)
    // =========================================================================

    /**
     * @brief Get the entity a DOF belongs to
     *
     * Returns the "owning" entity - the lowest-dimensional entity
     * that contains this DOF:
     * - Vertex DOF -> vertex entity
     * - Edge DOF -> edge entity
     * - Face DOF -> face entity
     * - Cell DOF -> cell entity
     *
     * @param dof_id Global DOF index
     * @return Entity reference, or nullopt if unknown
     */
    [[nodiscard]] std::optional<EntityRef> getDofEntity(GlobalIndex dof_id) const;

    /**
     * @brief Get all entities that support a DOF
     *
     * For a vertex DOF, this includes the vertex plus all edges/faces/cells
     * that contain that vertex. Useful for debugging and constraints.
     *
     * @param dof_id Global DOF index
     * @return Vector of supporting entities (may be empty if DOF unknown)
     */
    [[nodiscard]] std::vector<EntityRef> getDofSupportEntities(GlobalIndex dof_id) const;

    // =========================================================================
    // Batch Queries
    // =========================================================================

    /**
     * @brief Get DOFs on a boundary (all faces with given boundary ID)
     *
     * @param boundary_id Boundary label
     * @param face_boundary_labels Mapping from face ID to boundary label
     * @return All DOFs on the boundary
     */
    [[nodiscard]] std::vector<GlobalIndex> getBoundaryFaceDofs(
        int boundary_id,
        std::span<const int> face_boundary_labels) const;

    /**
     * @brief Get interface DOFs (DOFs shared between cells)
     *
     * Returns DOFs on vertices, edges, and faces (not cell interiors).
     *
     * @return Interface DOF indices
     */
    [[nodiscard]] std::vector<GlobalIndex> getInterfaceDofs() const;

    // =========================================================================
    // Statistics
    // =========================================================================

    /**
     * @brief Get entity DOF map statistics
     */
    struct Statistics {
        GlobalIndex n_vertex_dofs{0};
        GlobalIndex n_edge_dofs{0};
        GlobalIndex n_face_dofs{0};
        GlobalIndex n_cell_interior_dofs{0};
        GlobalIndex total_dofs{0};
    };

    [[nodiscard]] Statistics getStatistics() const;

    // =========================================================================
    // Counts
    // =========================================================================

    [[nodiscard]] GlobalIndex numVertices() const noexcept { return n_vertices_; }
    [[nodiscard]] GlobalIndex numEdges() const noexcept { return n_edges_; }
    [[nodiscard]] GlobalIndex numFaces() const noexcept { return n_faces_; }
    [[nodiscard]] GlobalIndex numCells() const noexcept { return n_cells_; }

    [[nodiscard]] bool isFinalized() const noexcept { return finalized_; }
    [[nodiscard]] bool hasReverseMapping() const noexcept { return has_reverse_map_; }

private:
    // Helper to get span from CSR storage
    std::span<const GlobalIndex> getSpan(const std::vector<GlobalIndex>& offsets,
                                          const std::vector<GlobalIndex>& data,
                                          GlobalIndex id) const;

    // CSR storage for each entity type
    std::vector<GlobalIndex> vertex_offsets_;
    std::vector<GlobalIndex> vertex_dofs_;

    std::vector<GlobalIndex> edge_offsets_;
    std::vector<GlobalIndex> edge_dofs_;

    std::vector<GlobalIndex> face_offsets_;
    std::vector<GlobalIndex> face_dofs_;

    std::vector<GlobalIndex> cell_interior_offsets_;
    std::vector<GlobalIndex> cell_interior_dofs_;

    // Reverse mapping: DOF -> entity
    std::unordered_map<GlobalIndex, EntityRef> dof_to_entity_;

    // Counts
    GlobalIndex n_vertices_{0};
    GlobalIndex n_edges_{0};
    GlobalIndex n_faces_{0};
    GlobalIndex n_cells_{0};

    // State
    bool finalized_{false};
    bool has_reverse_map_{false};
};

} // namespace dofs
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_DOFS_ENTITYDOFMAP_H

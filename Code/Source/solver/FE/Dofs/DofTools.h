/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_DOFS_DOFTOOLS_H
#define SVMP_FE_DOFS_DOFTOOLS_H

/**
 * @file DofTools.h
 * @brief Common DOF selection and query utilities
 *
 * DofTools provides utility functions for extracting subsets of DOFs
 * based on various criteria:
 *  - Boundary DOFs (by boundary ID and field/component mask)
 *  - Subdomain DOFs (by subdomain labels)
 *  - Entity DOFs (vertices, edges, faces, cells)
 *  - Interface DOFs (partition boundaries)
 *  - Geometric predicates (DOFs in a region)
 *
 * These utilities prevent physics modules from reinventing DOF extraction
 * logic and provide a clean interface between boundary IDs and DOF IDs.
 */

#include "DofMap.h"
#include "DofIndexSet.h"
#include "EntityDofMap.h"
#include "Core/Types.h"
#include "Core/FEException.h"

#include <vector>
#include <span>
#include <functional>
#include <string>
#include <bitset>
#include <optional>

namespace svmp {
namespace FE {
namespace dofs {

// Forward declarations (avoid including DofHandler.h here)
struct MeshTopologyInfo;

/**
 * @brief Component mask for selecting DOF components in vector fields
 *
 * Used to select specific components (e.g., x,y,z for velocity) when
 * extracting DOFs. Supports up to 8 components.
 */
class ComponentMask {
public:
    static constexpr std::size_t MAX_COMPONENTS = 8;

    /** @brief Create mask selecting all components */
    static ComponentMask all() noexcept;

    /** @brief Create mask selecting no components */
    static ComponentMask none() noexcept;

    /** @brief Create mask selecting single component */
    static ComponentMask component(std::size_t idx);

    /** @brief Create mask from list of selected component indices */
    static ComponentMask selected(std::initializer_list<std::size_t> indices);

    /** @brief Default: select all components */
    ComponentMask() noexcept;

    /** @brief Check if component is selected */
    [[nodiscard]] bool isSelected(std::size_t component) const noexcept;

    /** @brief Get number of selected components */
    [[nodiscard]] std::size_t numSelected() const noexcept;

    /** @brief Get total number of components */
    [[nodiscard]] std::size_t size() const noexcept { return n_components_; }

    /** @brief Set number of components */
    void setSize(std::size_t n_components);

    /** @brief Select a component */
    void select(std::size_t component);

    /** @brief Deselect a component */
    void deselect(std::size_t component);

    /** @brief Boolean AND with another mask */
    ComponentMask operator&(const ComponentMask& other) const noexcept;

    /** @brief Boolean OR with another mask */
    ComponentMask operator|(const ComponentMask& other) const noexcept;

    /** @brief Check if all components are selected */
    [[nodiscard]] bool selectsAll() const noexcept;

    /** @brief Check if no components are selected */
    [[nodiscard]] bool selectsNone() const noexcept;

private:
    std::bitset<MAX_COMPONENTS> mask_;
    std::size_t n_components_{MAX_COMPONENTS};
};

/**
 * @brief Field mask for selecting specific fields in multi-field systems
 */
class FieldMask {
public:
    static constexpr std::size_t MAX_FIELDS = 16;

    /** @brief Create mask selecting all fields */
    static FieldMask all() noexcept;

    /** @brief Create mask selecting no fields */
    static FieldMask none() noexcept;

    /** @brief Create mask selecting single field by index */
    static FieldMask field(std::size_t idx);

    /** @brief Create mask selecting fields by name (requires field registry) */
    static FieldMask named(std::initializer_list<std::string> names);

    /** @brief Default: select all fields */
    FieldMask() noexcept;

    /** @brief Check if field is selected */
    [[nodiscard]] bool isSelected(std::size_t field_idx) const noexcept;

    /** @brief Select a field */
    void select(std::size_t field_idx);

    /** @brief Deselect a field */
    void deselect(std::size_t field_idx);

    /** @brief Get number of selected fields */
    [[nodiscard]] std::size_t numSelected() const noexcept;

    /** @brief Store field names for named access */
    void setFieldNames(std::span<const std::string> names);

    /** @brief Get field index by name (-1 if not found) */
    [[nodiscard]] int getFieldIndex(const std::string& name) const;

private:
    std::bitset<MAX_FIELDS> mask_;
    std::vector<std::string> field_names_;
};

/**
 * @brief Options for DOF extraction operations
 */
struct DofExtractionOptions {
    bool include_constrained{true};  ///< Include constrained DOFs
    bool sort_result{true};          ///< Sort result indices
    bool remove_duplicates{true};    ///< Remove duplicate DOFs
};

/**
 * @brief Geometric predicate type for DOF selection
 *
 * Takes (x, y, z) coordinates and returns true if the point is in the region.
 */
using GeometricPredicate = std::function<bool(double, double, double)>;

// =============================================================================
// DofTools Namespace - Free Functions
// =============================================================================

namespace DofTools {

// =========================================================================
// Boundary DOF Extraction
// =========================================================================

/**
 * @brief Extract DOFs on a boundary by boundary ID
 *
 * @param entity_map Entity-to-DOF associations
 * @param boundary_id Boundary label to extract
 * @param facet_boundary_labels Labels for each boundary facet (edge in 2D, face in 3D)
 * @param facet2vertex_offsets CSR offsets into facet2vertex_data (size = n_facets + 1)
 * @param facet2vertex_data CSR vertex indices for each facet (cyclic ordering for faces)
 * @param edge2vertex_data Edge endpoints (size = 2 * n_edges, required when edge DOFs exist)
 * @param options Extraction options
 * @return Vector of DOF indices on the boundary
 */
std::vector<GlobalIndex> extractBoundaryDofs(
    const EntityDofMap& entity_map,
    int boundary_id,
    std::span<const int> facet_boundary_labels,
    std::span<const GlobalIndex> facet2vertex_offsets,
    std::span<const GlobalIndex> facet2vertex_data,
    std::span<const GlobalIndex> edge2vertex_data,
    const DofExtractionOptions& options = {});

/**
 * @brief Extract specific field DOFs on a boundary
 *
 * @param entity_map Entity-to-DOF associations
 * @param boundary_id Boundary label
 * @param facet_boundary_labels Facet labels
 * @param facet2vertex_offsets CSR offsets into facet2vertex_data
 * @param facet2vertex_data CSR vertex indices for each facet
 * @param edge2vertex_data Edge endpoints (required when edge DOFs exist)
 * @param field_mask Which fields to include
 * @param dofs_per_field DOFs per field (for field separation)
 * @param options Extraction options
 * @return DOFs for selected fields on the boundary
 */
std::vector<GlobalIndex> extractBoundaryDofs(
    const EntityDofMap& entity_map,
    int boundary_id,
    std::span<const int> facet_boundary_labels,
    std::span<const GlobalIndex> facet2vertex_offsets,
    std::span<const GlobalIndex> facet2vertex_data,
    std::span<const GlobalIndex> edge2vertex_data,
    const FieldMask& field_mask,
    std::span<const GlobalIndex> dofs_per_field,
    const DofExtractionOptions& options = {});

/**
 * @brief Extract specific component DOFs on a boundary
 *
 * For vector-valued fields (e.g., velocity with u,v,w components),
 * this extracts only the specified components.
 *
 * This routine supports both common component layouts:
 * - Interleaved: `component = dof % n_components`
 * - Block-by-component: `dof = scalar + component * stride`
 *
 * The component layout is inferred from the `EntityDofMap` ordering. If no
 * consistent component pattern is found, an exception is thrown to avoid
 * silently returning incorrect DOF sets.
 *
 * @param entity_map Entity-to-DOF associations
 * @param boundary_id Boundary label
 * @param facet_boundary_labels Facet labels
 * @param facet2vertex_offsets CSR offsets into facet2vertex_data
 * @param facet2vertex_data CSR vertex indices for each facet
 * @param edge2vertex_data Edge endpoints (required when edge DOFs exist)
 * @param component_mask Which components to include
 * @param n_components Total number of components
 * @param options Extraction options
 * @return DOFs for selected components on the boundary
 */
std::vector<GlobalIndex> extractBoundaryDofs(
    const EntityDofMap& entity_map,
    int boundary_id,
    std::span<const int> facet_boundary_labels,
    std::span<const GlobalIndex> facet2vertex_offsets,
    std::span<const GlobalIndex> facet2vertex_data,
    std::span<const GlobalIndex> edge2vertex_data,
    const ComponentMask& component_mask,
    std::size_t n_components,
    const DofExtractionOptions& options = {});

/**
 * @brief Extract all DOFs on any boundary (union of all boundary faces)
 *
 * @param entity_map Entity-to-DOF associations
 * @param facet_boundary_labels Facet labels (non-zero = boundary)
 * @param facet2vertex_offsets CSR offsets into facet2vertex_data
 * @param facet2vertex_data CSR vertex indices for each facet
 * @param edge2vertex_data Edge endpoints (required when edge DOFs exist)
 * @return All boundary DOFs
 */
std::vector<GlobalIndex> extractAllBoundaryDofs(
    const EntityDofMap& entity_map,
    std::span<const int> facet_boundary_labels,
    std::span<const GlobalIndex> facet2vertex_offsets,
    std::span<const GlobalIndex> facet2vertex_data,
    std::span<const GlobalIndex> edge2vertex_data);

// =========================================================================
// Subdomain DOF Extraction
// =========================================================================

/**
 * @brief Extract DOFs in a subdomain (region) by label
 *
 * @param dof_map The DOF map
 * @param subdomain_id Subdomain label
 * @param cell_subdomain_labels Labels for each cell
 * @return IndexSet of DOFs in the subdomain
 */
IndexSet extractSubdomainDofs(
    const DofMap& dof_map,
    int subdomain_id,
    std::span<const int> cell_subdomain_labels);

/**
 * @brief Extract DOFs from multiple subdomains
 *
 * @param dof_map The DOF map
 * @param subdomain_ids List of subdomain labels
 * @param cell_subdomain_labels Labels for each cell
 * @return IndexSet of DOFs (union of all subdomains)
 */
IndexSet extractSubdomainDofs(
    const DofMap& dof_map,
    std::span<const int> subdomain_ids,
    std::span<const int> cell_subdomain_labels);

/**
 * @brief Extract interior DOFs (not on any boundary)
 *
 * @param entity_map Entity-to-DOF associations
 * @param face_boundary_labels Face labels (non-zero = boundary)
 * @return IndexSet of interior DOFs
 */
IndexSet extractInteriorDofs(
    const EntityDofMap& entity_map,
    std::span<const int> face_boundary_labels);

// =========================================================================
// Entity-Based Extraction
// =========================================================================

/**
 * @brief Extract DOFs associated with a range of entities
 *
 * @param entity_map Entity-to-DOF associations
 * @param kind Entity type (Vertex, Edge, Face, Cell)
 * @param entity_ids Entity indices
 * @return Vector of DOFs on those entities
 */
std::vector<GlobalIndex> extractEntityDofs(
    const EntityDofMap& entity_map,
    EntityKind kind,
    std::span<const GlobalIndex> entity_ids);

/**
 * @brief Extract DOFs on all vertices
 *
 * @param entity_map Entity-to-DOF associations
 * @return Vector of all vertex DOFs
 */
std::vector<GlobalIndex> extractAllVertexDofs(const EntityDofMap& entity_map);

/**
 * @brief Extract DOFs on all edges
 *
 * @param entity_map Entity-to-DOF associations
 * @return Vector of all edge-interior DOFs
 */
std::vector<GlobalIndex> extractAllEdgeDofs(const EntityDofMap& entity_map);

/**
 * @brief Extract DOFs on all faces
 *
 * @param entity_map Entity-to-DOF associations
 * @return Vector of all face-interior DOFs
 */
std::vector<GlobalIndex> extractAllFaceDofs(const EntityDofMap& entity_map);

/**
 * @brief Extract cell-interior DOFs (bubble DOFs)
 *
 * @param entity_map Entity-to-DOF associations
 * @return Vector of all cell-interior DOFs
 */
std::vector<GlobalIndex> extractAllCellInteriorDofs(const EntityDofMap& entity_map);

/**
 * @brief Extract interface DOFs (shared between cells)
 *
 * Returns DOFs on vertices, edges, and faces - not cell interiors.
 * In serial, these are DOFs that would be shared in a parallel decomposition.
 *
 * @param entity_map Entity-to-DOF associations
 * @return IndexSet of interface DOFs
 */
IndexSet extractInterfaceDofs(const EntityDofMap& entity_map);

/**
 * @brief Get all mesh entities supporting a DOF (topology-aware)
 *
 * Returns the primary entity of the DOF (from EntityDofMap reverse mapping)
 * plus higher-dimensional entities (edges/faces/cells) that contain it,
 * derived from the provided mesh topology. Intended for debugging and for
 * constraint bookkeeping where the full support set is needed.
 *
 * @param dof_id Global DOF index
 * @param entity_map Entity-to-DOF associations (requires reverse mapping)
 * @param topology Mesh topology connectivity (cell/edge/face incidence)
 * @return Sorted unique list of supporting entities
 */
std::vector<EntityRef> getDofSupportEntities(
    GlobalIndex dof_id,
    const EntityDofMap& entity_map,
    const MeshTopologyInfo& topology);

// =========================================================================
// Geometric Predicate Extraction
// =========================================================================

/**
 * @brief Extract DOFs where node coordinates satisfy a predicate
 *
 * @param dof_map The DOF map
 * @param vertex_dof_map Vertex-to-DOF associations
 * @param vertex_coords Vertex coordinates [n_vertices * dim]
 * @param dim Spatial dimension (2 or 3)
 * @param predicate Function returning true for points in region
 * @return Vector of DOFs in the region
 *
 * @note This works for vertex DOFs. For higher-order DOFs, the DOF location
 *       is approximated based on the supporting entity centroid.
 */
std::vector<GlobalIndex> extractDofsInRegion(
    const DofMap& dof_map,
    const EntityDofMap& vertex_dof_map,
    std::span<const double> vertex_coords,
    int dim,
    GeometricPredicate predicate);

/**
 * @brief Extract DOFs in a box region
 *
 * @param dof_map The DOF map
 * @param vertex_dof_map Vertex-to-DOF associations
 * @param vertex_coords Vertex coordinates
 * @param dim Spatial dimension
 * @param min_corner Minimum corner of box (x, y, z)
 * @param max_corner Maximum corner of box (x, y, z)
 * @return Vector of DOFs in the box
 */
std::vector<GlobalIndex> extractDofsInBox(
    const DofMap& dof_map,
    const EntityDofMap& vertex_dof_map,
    std::span<const double> vertex_coords,
    int dim,
    std::span<const double> min_corner,
    std::span<const double> max_corner);

/**
 * @brief Extract DOFs in a spherical region
 *
 * @param dof_map The DOF map
 * @param vertex_dof_map Vertex-to-DOF associations
 * @param vertex_coords Vertex coordinates
 * @param dim Spatial dimension
 * @param center Sphere center (x, y, z)
 * @param radius Sphere radius
 * @return Vector of DOFs in the sphere
 */
std::vector<GlobalIndex> extractDofsInSphere(
    const DofMap& dof_map,
    const EntityDofMap& vertex_dof_map,
    std::span<const double> vertex_coords,
    int dim,
    std::span<const double> center,
    double radius);

/**
 * @brief Extract DOFs on a plane (within tolerance)
 *
 * @param dof_map The DOF map
 * @param vertex_dof_map Vertex-to-DOF associations
 * @param vertex_coords Vertex coordinates
 * @param dim Spatial dimension
 * @param normal Plane normal (normalized)
 * @param point Point on plane
 * @param tolerance Distance tolerance
 * @return Vector of DOFs on the plane
 */
std::vector<GlobalIndex> extractDofsOnPlane(
    const DofMap& dof_map,
    const EntityDofMap& vertex_dof_map,
    std::span<const double> vertex_coords,
    int dim,
    std::span<const double> normal,
    std::span<const double> point,
    double tolerance = 1e-10);

// =========================================================================
// Partition Interface DOFs (Parallel)
// =========================================================================

/**
 * @brief Extract DOFs on partition interfaces (for parallel)
 *
 * These are DOFs that are shared between MPI ranks, identified
 * from ghost cell information.
 *
 * @param dof_map The DOF map
 * @param owned_cell_mask True for owned cells, false for ghost cells
 * @return IndexSet of partition interface DOFs
 */
IndexSet extractPartitionInterfaceDofs(
    const DofMap& dof_map,
    std::span<const bool> owned_cell_mask);

// =========================================================================
// DOF Statistics and Analysis
// =========================================================================

/**
 * @brief Count DOFs by entity type
 */
struct DofCountsByEntity {
    GlobalIndex vertex_dofs{0};
    GlobalIndex edge_dofs{0};
    GlobalIndex face_dofs{0};
    GlobalIndex cell_interior_dofs{0};
    GlobalIndex total{0};
};

/**
 * @brief Count DOFs by entity type
 *
 * @param entity_map Entity-to-DOF associations
 * @return Counts per entity type
 */
DofCountsByEntity countDofsByEntity(const EntityDofMap& entity_map);

/**
 * @brief Compute DOF distribution statistics
 */
struct DofDistributionStats {
    GlobalIndex total_dofs{0};
    double min_dofs_per_cell{0.0};
    double max_dofs_per_cell{0.0};
    double avg_dofs_per_cell{0.0};
    double std_dev_dofs_per_cell{0.0};
};

/**
 * @brief Compute DOF distribution statistics
 *
 * @param dof_map The DOF map
 * @return Distribution statistics
 */
DofDistributionStats computeDistributionStats(const DofMap& dof_map);

// =========================================================================
// Utility Functions
// =========================================================================

/**
 * @brief Sort and remove duplicates from DOF list
 *
 * @param dofs DOF list (modified in place)
 */
void sortAndUnique(std::vector<GlobalIndex>& dofs);

/**
 * @brief Convert vector to IndexSet
 *
 * @param dofs DOF vector
 * @return IndexSet containing the DOFs
 */
IndexSet toIndexSet(std::vector<GlobalIndex> dofs);

/**
 * @brief Convert IndexSet to vector
 *
 * @param index_set The IndexSet
 * @return Vector of DOF indices
 */
std::vector<GlobalIndex> toVector(const IndexSet& index_set);

/**
 * @brief Compute complement of DOF set
 *
 * @param dofs DOFs to exclude
 * @param n_total_dofs Total number of DOFs
 * @return IndexSet of DOFs not in the input set
 */
IndexSet complement(const IndexSet& dofs, GlobalIndex n_total_dofs);

} // namespace DofTools

} // namespace dofs
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_DOFS_DOFTOOLS_H

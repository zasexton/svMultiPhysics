#ifndef SVMP_FE_AUXILIARY_STATE_INDEXING_H
#define SVMP_FE_AUXILIARY_STATE_INDEXING_H

/**
 * @file AuxiliaryStateIndexing.h
 * @brief Scope-specific entity indexing helpers for auxiliary state blocks.
 *
 * Each auxiliary block scope has a different entity set.  This module
 * provides indexing descriptors that map from scope-specific entity
 * coordinates to flat storage indices.
 *
 * ## Scope → Entity mapping
 *
 * | Scope             | Entity set                          | Count source                |
 * |-------------------|-------------------------------------|-----------------------------|
 * | `Global`          | Single synthetic entity (id=0)      | 1                           |
 * | `Node`            | Mesh vertices (owned + ghost)       | `numVertices`               |
 * | `Cell`            | Mesh cells (owned)                  | `numOwnedCells`             |
 * | `QuadraturePoint` | QPs per cell × cells                | `sum(n_qp_per_cell)`        |
 * | `Region`          | Topology regions / components       | topology context            |
 * | `Boundary`        | One per named boundary collection   | 1                           |
 * | `Facet`           | Faces/edges on a named boundary     | boundary entity count       |
 *
 * ## Canonical entity ordering
 *
 * - Owned entities first, then ghosts (where applicable).
 * - Default ordering is `ByEntityThenComponent`.
 *
 * ## Distributed semantics
 *
 * - `Global` scope: single instance, no distribution.
 * - `Node` scope: owned nodes first, ghost nodes appended.
 * - `Cell` scope: owned cells only (ghosts are typically not needed).
 * - `QuadraturePoint` scope: inherits cell ownership.
 * - `Region` scope: owned topology-region entities only.
 * - `Boundary` scope: single instance on a named boundary, no distribution.
 * - `Facet` scope: stable entity ordering on boundary subsets.
 */

#include "Core/Types.h"
#include "Core/FEException.h"

#include "Auxiliary/AuxiliaryStateTypes.h"

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace svmp {
namespace FE {
namespace systems {

/**
 * @brief Indexing descriptor for an auxiliary block.
 *
 * Created by scope-specific factory functions.  Provides entity-count
 * and flat-index computation for a given block specification.
 */
class AuxiliaryBlockIndexing {
public:
    AuxiliaryBlockIndexing() = default;

    // -----------------------------------------------------------------
    //  Factory functions (scope-specific)
    // -----------------------------------------------------------------

    /**
     * @brief Create indexing for Global scope (1 entity).
     */
    static AuxiliaryBlockIndexing createGlobal(int component_stride);

    /**
     * @brief Create indexing for Node scope.
     *
     * @param n_owned_nodes Number of locally owned nodes.
     * @param n_ghost_nodes Number of ghost nodes (0 for serial).
     * @param component_stride Components per node.
     */
    static AuxiliaryBlockIndexing createNode(
        std::size_t n_owned_nodes,
        std::size_t n_ghost_nodes,
        int component_stride);

    /**
     * @brief Create indexing for Cell scope.
     *
     * @param n_owned_cells Number of locally owned cells.
     * @param component_stride Components per cell.
     */
    static AuxiliaryBlockIndexing createCell(
        std::size_t n_owned_cells,
        int component_stride);

    /**
     * @brief Create indexing for QuadraturePoint scope.
     *
     * @param qp_offsets CSR-style offsets: cell `i` has QPs in
     *                   `[qp_offsets[i], qp_offsets[i+1])`.
     *                   Size = `n_cells + 1`.  `qp_offsets[0]` must be 0.
     * @param component_stride Components per quadrature point.
     */
    static AuxiliaryBlockIndexing createQuadraturePoint(
        std::span<const std::size_t> qp_offsets,
        int component_stride);

    /**
     * @brief Create ragged indexing for Node scope.
     *
     * @param n_owned_nodes Number of locally owned nodes.
     * @param n_ghost_nodes Number of ghost nodes.
     * @param component_offsets Per-entity component offsets, size =
     *                          n_owned_nodes + n_ghost_nodes + 1.
     */
    static AuxiliaryBlockIndexing createRaggedNode(
        std::size_t n_owned_nodes,
        std::size_t n_ghost_nodes,
        std::span<const std::size_t> component_offsets);

    /**
     * @brief Create ragged indexing for Cell scope.
     */
    static AuxiliaryBlockIndexing createRaggedCell(
        std::span<const std::size_t> component_offsets);

    /**
     * @brief Create ragged indexing for QuadraturePoint scope.
     *
     * `qp_offsets` maps covered cells to flat QP entities.  `component_offsets`
     * maps each flat QP entity to its variable-size component slice.
     */
    static AuxiliaryBlockIndexing createRaggedQuadraturePoint(
        std::span<const std::size_t> qp_offsets,
        std::span<const std::size_t> component_offsets);

    /**
     * @brief Create indexing for Region scope.
     *
     * @param n_regions Number of topology regions owned/materialized locally.
     * @param component_stride Components per region.
     */
    static AuxiliaryBlockIndexing createRegion(
        std::size_t n_regions,
        int component_stride);

    /**
     * @brief Create ragged indexing for Region scope.
     */
    static AuxiliaryBlockIndexing createRaggedRegion(
        std::span<const std::size_t> component_offsets);

    /**
     * @brief Create indexing for Boundary scope (one instance per named boundary).
     *
     * @param component_stride Components per instance.
     */
    static AuxiliaryBlockIndexing createBoundary(int component_stride);

    /**
     * @brief Create indexing for Facet scope (one instance per boundary facet).
     *
     * @param n_boundary_entities Number of facets on the boundary.
     * @param component_stride Components per boundary facet.
     */
    static AuxiliaryBlockIndexing createFacet(
        std::size_t n_boundary_entities,
        int component_stride);

    // -----------------------------------------------------------------
    //  Properties
    // -----------------------------------------------------------------

    [[nodiscard]] AuxiliaryStateScope scope() const noexcept { return scope_; }
    [[nodiscard]] AuxiliaryLayoutMode layoutMode() const noexcept { return layout_mode_; }
    [[nodiscard]] std::size_t totalEntityCount() const noexcept { return total_entity_count_; }
    [[nodiscard]] std::size_t ownedEntityCount() const noexcept { return owned_entity_count_; }
    [[nodiscard]] std::size_t ghostEntityCount() const noexcept
    {
        return total_entity_count_ - owned_entity_count_;
    }
    [[nodiscard]] int componentStride() const noexcept { return component_stride_; }
    [[nodiscard]] std::size_t totalStorageSize() const noexcept { return total_storage_size_; }
    [[nodiscard]] std::size_t ownedStorageSize() const noexcept
    {
        if (layout_mode_ == AuxiliaryLayoutMode::Ragged) {
            return owned_entity_count_ < component_offsets_.size()
                ? component_offsets_[owned_entity_count_]
                : total_storage_size_;
        }
        return owned_entity_count_ * static_cast<std::size_t>(component_stride_);
    }

    // -----------------------------------------------------------------
    //  Index computation (ByEntityThenComponent)
    // -----------------------------------------------------------------

    /**
     * @brief Flat index for entity `i`, component `c`.
     *
     * For fixed-stride ByEntityThenComponent: `i * stride + c`.
     * For ragged: `component_offsets[i] + c`.
     */
    [[nodiscard]] std::size_t flatIndex(std::size_t entity_idx, int component) const noexcept
    {
        if (layout_mode_ == AuxiliaryLayoutMode::Ragged) {
            return component_offsets_[entity_idx] + static_cast<std::size_t>(component);
        }
        return entity_idx * static_cast<std::size_t>(component_stride_) +
               static_cast<std::size_t>(component);
    }

    /**
     * @brief Flat index for QP scope: cell `cell_idx`, local QP `local_qp`, component `c`.
     *
     * Only valid for QuadraturePoint scope.
     */
    [[nodiscard]] std::size_t qpFlatIndex(
        std::size_t cell_idx, std::size_t local_qp, int component) const noexcept
    {
        const auto qp_offset = qp_offsets_[cell_idx] + local_qp;
        return flatIndex(qp_offset, component);
    }

    /**
     * @brief Number of QPs for cell `cell_idx`.
     *
     * Only valid for QuadraturePoint scope.
     */
    [[nodiscard]] std::size_t qpsForCell(std::size_t cell_idx) const noexcept
    {
        return qp_offsets_[cell_idx + 1] - qp_offsets_[cell_idx];
    }

    /// QP offsets array (QuadraturePoint scope only).
    [[nodiscard]] std::span<const std::size_t> qpOffsets() const noexcept
    {
        return qp_offsets_;
    }

    /// Ragged component offsets array (ragged layout only).
    [[nodiscard]] std::span<const std::size_t> componentOffsets() const noexcept
    {
        return component_offsets_;
    }

    /// Flat storage offset for one entity.
    [[nodiscard]] std::size_t entityStorageOffset(std::size_t entity_idx) const noexcept
    {
        return layout_mode_ == AuxiliaryLayoutMode::Ragged
            ? component_offsets_[entity_idx]
            : entity_idx * static_cast<std::size_t>(component_stride_);
    }

    /// Number of stored components for one entity.
    [[nodiscard]] std::size_t entityComponentCount(std::size_t entity_idx) const noexcept
    {
        return layout_mode_ == AuxiliaryLayoutMode::Ragged
            ? component_offsets_[entity_idx + 1] - component_offsets_[entity_idx]
            : static_cast<std::size_t>(component_stride_);
    }

private:
    static AuxiliaryBlockIndexing createRaggedScoped_(
        AuxiliaryStateScope scope,
        std::size_t owned_entity_count,
        std::span<const std::size_t> component_offsets);

    AuxiliaryStateScope scope_{AuxiliaryStateScope::Global};
    AuxiliaryLayoutMode layout_mode_{AuxiliaryLayoutMode::FixedStride};
    std::size_t total_entity_count_{0};
    std::size_t owned_entity_count_{0};
    int component_stride_{0};
    std::size_t total_storage_size_{0};

    /// Component offsets for ragged layout (size = total_entity_count + 1).
    std::vector<std::size_t> component_offsets_{};

    /// QP offsets for QuadraturePoint scope (size = n_cells + 1).
    std::vector<std::size_t> qp_offsets_{};
};

} // namespace systems
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_AUXILIARY_STATE_INDEXING_H

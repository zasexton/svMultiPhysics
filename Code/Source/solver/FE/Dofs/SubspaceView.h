/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_DOFS_SUBSPACEVIEW_H
#define SVMP_FE_DOFS_SUBSPACEVIEW_H

/**
 * @file SubspaceView.h
 * @brief Lightweight views into DOF subsets by field/component
 *
 * SubspaceView provides a non-owning view into a subset of DOFs,
 * typically representing:
 *  - A single field (velocity, pressure, etc.)
 *  - Specific components of a vector field
 *  - DOFs on a boundary or subdomain
 *
 * Key features:
 *  - Non-owning: just indexes into parent DOF map
 *  - Block position tracking for block preconditioners
 *  - Subvector extraction from full vectors
 *  - Efficient iteration over DOF subset
 */

#include "DofIndexSet.h"
#include "Core/Types.h"
#include "Core/FEException.h"

#include <vector>
#include <span>
#include <string>
#include <optional>

namespace svmp {
namespace FE {
namespace dofs {

/**
 * @brief Non-owning view into a subset of DOFs
 *
 * This class provides efficient access to a subset of DOFs, supporting
 * operations like extracting subvectors and iterating over the subset.
 */
class SubspaceView {
public:
    // =========================================================================
    // Construction
    // =========================================================================

    /**
     * @brief Default constructor (empty view)
     */
    SubspaceView() = default;

    /**
     * @brief Construct from IndexSet
     *
     * @param dofs DOFs in this subspace
     * @param name Optional name for the subspace
     * @param block_index Optional block index
     */
    explicit SubspaceView(IndexSet dofs, std::string name = "",
                          int block_index = -1);

    /**
     * @brief Construct from contiguous range
     *
     * @param start First DOF in range
     * @param end One past last DOF in range
     * @param name Optional name
     * @param block_index Optional block index
     */
    SubspaceView(GlobalIndex start, GlobalIndex end, std::string name = "",
                 int block_index = -1);

    // Default copy/move
    SubspaceView(const SubspaceView&) = default;
    SubspaceView& operator=(const SubspaceView&) = default;
    SubspaceView(SubspaceView&&) noexcept = default;
    SubspaceView& operator=(SubspaceView&&) noexcept = default;

    ~SubspaceView() = default;

    // =========================================================================
    // Basic Queries
    // =========================================================================

    /**
     * @brief Get DOFs in this subspace
     */
    [[nodiscard]] const IndexSet& getDofs() const noexcept { return dofs_; }

    /**
     * @brief Get as IndexSet (alias for getDofs)
     */
    [[nodiscard]] const IndexSet& getIndexSet() const noexcept { return dofs_; }

    /**
     * @brief Get local size (number of DOFs in subspace)
     */
    [[nodiscard]] GlobalIndex getLocalSize() const noexcept {
        return dofs_.size();
    }

    /**
     * @brief Get global size (same as local for serial)
     */
    [[nodiscard]] GlobalIndex getGlobalSize() const noexcept {
        return global_size_ > 0 ? global_size_ : dofs_.size();
    }

    /**
     * @brief Set global size (for parallel)
     */
    void setGlobalSize(GlobalIndex size) { global_size_ = size; }

    /**
     * @brief Check if subspace is empty
     */
    [[nodiscard]] bool empty() const noexcept { return dofs_.empty(); }

    /**
     * @brief Get subspace name
     */
    [[nodiscard]] const std::string& name() const noexcept { return name_; }

    /**
     * @brief Set subspace name
     */
    void setName(const std::string& name) { name_ = name; }

    // =========================================================================
    // Block Information
    // =========================================================================

    /**
     * @brief Get block index in full system
     *
     * @return Block index, or -1 if not part of a block system
     */
    [[nodiscard]] int getBlockIndex() const noexcept { return block_index_; }

    /**
     * @brief Set block index
     */
    void setBlockIndex(int index) { block_index_ = index; }

    /**
     * @brief Check if this subspace is part of a block system
     */
    [[nodiscard]] bool hasBlockIndex() const noexcept { return block_index_ >= 0; }

    // =========================================================================
    // Range Information
    // =========================================================================

    /**
     * @brief Check if DOFs form a contiguous range
     */
    [[nodiscard]] bool isContiguous() const noexcept;

    /**
     * @brief Get contiguous range if applicable
     *
     * @return (start, end) if contiguous, nullopt otherwise
     */
    [[nodiscard]] std::optional<std::pair<GlobalIndex, GlobalIndex>>
    contiguousRange() const noexcept;

    // =========================================================================
    // DOF Access
    // =========================================================================

    /**
     * @brief Check if a DOF is in this subspace
     */
    [[nodiscard]] bool contains(GlobalIndex dof) const noexcept {
        return dofs_.contains(dof);
    }

    /**
     * @brief Get DOF indices as a vector
     */
    [[nodiscard]] std::vector<GlobalIndex> toVector() const {
        return dofs_.toVector();
    }

    /**
     * @brief Begin iterator
     */
    [[nodiscard]] auto begin() const { return dofs_.begin(); }

    /**
     * @brief End iterator
     */
    [[nodiscard]] auto end() const { return dofs_.end(); }

    // =========================================================================
    // Subvector Operations
    // =========================================================================

    /**
     * @brief Extract subvector values from full vector
     *
     * @param full_vector Full DOF vector
     * @return Values at subspace DOF locations
     */
    [[nodiscard]] std::vector<double> extractSubvector(
        std::span<const double> full_vector) const;

    /**
     * @brief Extract subvector into provided buffer
     *
     * @param full_vector Full DOF vector
     * @param output Buffer for extracted values (must have size() elements)
     */
    void extractSubvector(std::span<const double> full_vector,
                          std::span<double> output) const;

    /**
     * @brief Scatter subvector values to full vector
     *
     * @param subvector Values for subspace DOFs
     * @param full_vector Full DOF vector (modified)
     */
    void scatterToFull(std::span<const double> subvector,
                       std::span<double> full_vector) const;

    /**
     * @brief Scatter subvector values, accumulating
     *
     * @param subvector Values for subspace DOFs
     * @param full_vector Full DOF vector (modified, values added)
     */
    void scatterToFullAdd(std::span<const double> subvector,
                          std::span<double> full_vector) const;

    // =========================================================================
    // Mapping
    // =========================================================================

    /**
     * @brief Map local subspace index to global DOF
     *
     * @param local_idx Index within subspace [0, size())
     * @return Global DOF index
     */
    [[nodiscard]] GlobalIndex localToGlobal(GlobalIndex local_idx) const;

    /**
     * @brief Map global DOF to local subspace index
     *
     * @param global_dof Global DOF index
     * @return Local index, or -1 if not in subspace
     */
    [[nodiscard]] GlobalIndex globalToLocal(GlobalIndex global_dof) const;

    /**
     * @brief Build local-to-global mapping vector
     *
     * Useful for sparse matrix operations.
     */
    [[nodiscard]] std::vector<GlobalIndex> buildLocalToGlobalMap() const;

    // =========================================================================
    // Set Operations
    // =========================================================================

    /**
     * @brief Create intersection with another subspace
     */
    [[nodiscard]] SubspaceView intersection_with(const SubspaceView& other) const;

    /**
     * @brief Create union with another subspace
     */
    [[nodiscard]] SubspaceView union_with(const SubspaceView& other) const;

    /**
     * @brief Create difference (this - other)
     */
    [[nodiscard]] SubspaceView difference(const SubspaceView& other) const;

    /**
     * @brief Create complement in a given range
     *
     * @param total_dofs Total number of DOFs in full system
     */
    [[nodiscard]] SubspaceView complement(GlobalIndex total_dofs) const;

private:
    IndexSet dofs_;
    std::string name_;
    int block_index_{-1};
    GlobalIndex global_size_{0};

    // Cached local-to-global map for fast lookup
    mutable std::vector<GlobalIndex> local_to_global_;
    mutable bool local_to_global_built_{false};

    void buildLocalToGlobalIfNeeded() const;
};

} // namespace dofs
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_DOFS_SUBSPACEVIEW_H

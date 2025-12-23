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

#ifndef SVMP_FE_SPARSITY_PARALLEL_SPARSITY_H
#define SVMP_FE_SPARSITY_PARALLEL_SPARSITY_H

/**
 * @file ParallelSparsity.h
 * @brief MPI-parallel sparsity construction manager
 *
 * This header provides utilities and manager classes for constructing
 * distributed sparsity patterns in MPI-parallel environments. It handles:
 *
 * - DOF ownership and ghost DOF detection
 * - Communication patterns for off-processor entries
 * - Ghost row sparsity exchange
 * - Consistent global NNZ computation
 * - Parallel validation
 *
 * The ParallelSparsityManager orchestrates the construction of
 * DistributedSparsityPattern objects across MPI ranks, ensuring that
 * ghost column mappings are consistent and communication-minimized.
 *
 * @see DistributedSparsityPattern for the produced artifact
 * @see SparsityBuilder for element-based pattern construction
 */

#include "DistributedSparsityPattern.h"
#include "SparsityBuilder.h"
#include "Core/Types.h"
#include "Core/FEConfig.h"
#include "Core/FEException.h"

#include <vector>
#include <memory>
#include <functional>
#include <unordered_map>
#include <unordered_set>

#if FE_HAS_MPI
#include <mpi.h>
#endif

namespace svmp {
namespace FE {
namespace sparsity {

/**
 * @brief Ownership distribution for DOFs across MPI ranks
 *
 * Describes how global DOFs are distributed across ranks.
 *
 * @note Current distributed sparsity artifacts assume each rank owns a
 *       contiguous block of DOFs (PETSc/Trilinos-style row maps). Custom
 *       ownership functions must therefore describe a contiguous owned range
 *       for each rank.
 */
class DofOwnership {
public:
    /**
     * @brief Default constructor - serial mode (all DOFs owned by rank 0)
     */
    DofOwnership() = default;

    /**
     * @brief Construct for contiguous block distribution
     *
     * @param n_global_dofs Total number of global DOFs
     * @param n_ranks Number of MPI ranks
     * @param my_rank This rank's ID
     *
     * Creates an approximately equal distribution across ranks.
     */
    DofOwnership(GlobalIndex n_global_dofs, int n_ranks, int my_rank);

    /**
     * @brief Construct with explicit ownership ranges per rank
     *
     * @param rank_offsets Cumulative offsets: rank i owns [rank_offsets[i], rank_offsets[i+1])
     * @param my_rank This rank's ID
     *
     * rank_offsets must have size n_ranks + 1.
     */
    DofOwnership(std::span<const GlobalIndex> rank_offsets, int my_rank);

    /**
     * @brief Construct with custom ownership function
     *
     * @param n_global_dofs Total number of global DOFs
     * @param owner_func Function mapping global DOF -> owning rank
     * @param my_rank This rank's ID
     */
    DofOwnership(GlobalIndex n_global_dofs,
                 std::function<int(GlobalIndex)> owner_func,
                 int my_rank);

    // --- Queries ---

    /**
     * @brief Get total number of global DOFs
     */
    [[nodiscard]] GlobalIndex globalNumDofs() const noexcept { return n_global_dofs_; }

    /**
     * @brief Get number of locally owned DOFs
     */
    [[nodiscard]] GlobalIndex numOwnedDofs() const noexcept {
        return owned_range_.size();
    }

    /**
     * @brief Get owned DOF range
     */
    [[nodiscard]] const IndexRange& ownedRange() const noexcept { return owned_range_; }

    /**
     * @brief Get this rank's ID
     */
    [[nodiscard]] int myRank() const noexcept { return my_rank_; }

    /**
     * @brief Get total number of ranks
     */
    [[nodiscard]] int numRanks() const noexcept { return n_ranks_; }

    /**
     * @brief Check if a DOF is locally owned
     */
    [[nodiscard]] bool isOwned(GlobalIndex dof) const noexcept {
        return owned_range_.contains(dof);
    }

    /**
     * @brief Get the owner rank for a DOF
     *
     * @param dof Global DOF index
     * @return Owning rank ID
     */
    [[nodiscard]] int getOwner(GlobalIndex dof) const;

    /**
     * @brief Check if this is a serial (single-rank) distribution
     */
    [[nodiscard]] bool isSerial() const noexcept {
        return n_ranks_ == 1;
    }

    /**
     * @brief Get ownership range for a specific rank (block distribution only)
     *
     * @param rank Rank ID
     * @return Index range owned by that rank
     * @throws FEException if not using block distribution
     */
    [[nodiscard]] IndexRange getRankRange(int rank) const;

    // --- Ghost DOF management ---

    /**
     * @brief Register a ghost DOF (DOF used locally but owned elsewhere)
     *
     * @param dof Global DOF index
     */
    void addGhostDof(GlobalIndex dof);

    /**
     * @brief Get all registered ghost DOFs (sorted)
     */
    [[nodiscard]] std::span<const GlobalIndex> getGhostDofs() const noexcept {
        return ghost_dofs_;
    }

    /**
     * @brief Get number of ghost DOFs
     */
    [[nodiscard]] GlobalIndex numGhostDofs() const noexcept {
        return static_cast<GlobalIndex>(ghost_dofs_.size());
    }

    /**
     * @brief Finalize ghost DOF list (sort and deduplicate)
     */
    void finalizeGhosts();

    /**
     * @brief Clear ghost DOF list
     */
    void clearGhosts() {
        ghost_dofs_.clear();
        ghost_set_.clear();
    }

private:
    GlobalIndex n_global_dofs_{0};
    int n_ranks_{1};
    int my_rank_{0};
    IndexRange owned_range_;

    // For block distribution
    std::vector<GlobalIndex> rank_offsets_;

    // For custom distribution
    std::function<int(GlobalIndex)> owner_func_;
    bool using_custom_func_{false};

    // Ghost DOF tracking
    std::vector<GlobalIndex> ghost_dofs_;
    std::unordered_set<GlobalIndex> ghost_set_;
    bool ghosts_finalized_{false};
};

/**
 * @brief Manager for MPI-parallel sparsity pattern construction
 *
 * ParallelSparsityManager coordinates the construction of distributed
 * sparsity patterns across MPI ranks. It handles:
 *
 * 1. DOF ownership distribution
 * 2. Local pattern building (via SparsityBuilder)
 * 3. Ghost DOF detection and communication
 * 4. Ghost row sparsity exchange (optional)
 * 5. Final pattern assembly and validation
 *
 * The manager produces DistributedSparsityPattern objects that can be
 * used for PETSc MatMPIAIJ, Trilinos Tpetra::CrsMatrix, or similar
 * distributed sparse matrix formats.
 *
 * Usage:
 * @code
 * ParallelSparsityManager manager(MPI_COMM_WORLD);
 * manager.setOwnership(ownership);
 * manager.setDofMap(dof_map);
 * auto pattern = manager.build();
 * @endcode
 *
 * @note All operations except build() are local and do not require
 *       MPI communication. build() may perform collective operations
 *       if ghost row exchange is enabled.
 */
class ParallelSparsityManager {
public:
    // =========================================================================
    // Construction
    // =========================================================================

    /**
     * @brief Default constructor (serial mode)
     */
    ParallelSparsityManager();

#if FE_HAS_MPI
    /**
     * @brief Construct with MPI communicator
     *
     * @param comm MPI communicator
     */
    explicit ParallelSparsityManager(MPI_Comm comm);
#endif

    /**
     * @brief Construct for non-MPI parallel (e.g., shared memory)
     *
     * @param n_ranks Number of virtual ranks
     * @param my_rank This rank's ID
     */
    ParallelSparsityManager(int n_ranks, int my_rank);

    /// Destructor
    ~ParallelSparsityManager() = default;

    // Non-copyable
    ParallelSparsityManager(const ParallelSparsityManager&) = delete;
    ParallelSparsityManager& operator=(const ParallelSparsityManager&) = delete;

    // Movable
    ParallelSparsityManager(ParallelSparsityManager&&) = default;
    ParallelSparsityManager& operator=(ParallelSparsityManager&&) = default;

    // =========================================================================
    // Configuration
    // =========================================================================

    /**
     * @brief Set DOF ownership distribution
     *
     * @param ownership Ownership descriptor
     */
    void setOwnership(DofOwnership ownership);

    /**
     * @brief Set ownership for block (contiguous) distribution
     *
     * @param n_global_dofs Total global DOF count
     *
     * Automatically computes balanced block distribution.
     */
    void setBlockOwnership(GlobalIndex n_global_dofs);

    /**
     * @brief Set row DOF map for sparsity construction
     *
     * @param dof_map DOF map for row indices
     */
    void setRowDofMap(const dofs::DofMap& dof_map);

    /**
     * @brief Set row DOF map via interface
     *
     * @param dof_map_query DOF map query interface
     */
    void setRowDofMap(std::shared_ptr<IDofMapQuery> dof_map_query);

    /**
     * @brief Set column DOF map for rectangular patterns
     *
     * @param dof_map DOF map for column indices
     */
    void setColDofMap(const dofs::DofMap& dof_map);

    /**
     * @brief Set column DOF map via interface
     */
    void setColDofMap(std::shared_ptr<IDofMapQuery> dof_map_query);

    /**
     * @brief Set column ownership (for rectangular patterns)
     */
    void setColOwnership(DofOwnership ownership);

    /**
     * @brief Set build options
     */
    void setOptions(const SparsityBuildOptions& options) {
        options_ = options;
    }

    /**
     * @brief Enable/disable ghost row exchange
     *
     * @param enable If true, exchange ghost row sparsity between ranks
     *
     * Ghost row exchange is needed when ghost rows need complete
     * sparsity information (e.g., for matrix-vector products or
     * block preconditioners).
     *
     * @note Ghost row exchange requires an MPI communicator (not MPI_COMM_SELF)
     *       and more than one rank. In non-MPI/virtual-rank mode this option
     *       will throw to avoid silently producing incomplete ghost-row data.
     */
    void setGhostRowExchange(bool enable) {
        exchange_ghost_rows_ = enable;
    }

    // =========================================================================
    // Building
    // =========================================================================

    /**
     * @brief Build the distributed sparsity pattern
     *
     * This method:
     * 1. Constructs local pattern from DOF map
     * 2. Identifies ghost columns
     * 3. Optionally exchanges ghost row sparsity (collective)
     * 4. Creates DistributedSparsityPattern with diag/offdiag split
     *
     * @return Finalized distributed sparsity pattern
     * @throws FEException if not configured properly
     */
    [[nodiscard]] DistributedSparsityPattern build();

    /**
     * @brief Build pattern for a subset of cells
     *
     * @param cell_ids Local cell indices to include
     * @return Finalized distributed sparsity pattern
     */
    [[nodiscard]] DistributedSparsityPattern build(std::span<const GlobalIndex> cell_ids);

    // =========================================================================
    // Parallel utilities
    // =========================================================================

    /**
     * @brief Get this rank's ID
     */
    [[nodiscard]] int myRank() const noexcept { return my_rank_; }

    /**
     * @brief Get total number of ranks
     */
    [[nodiscard]] int numRanks() const noexcept { return n_ranks_; }

    /**
     * @brief Check if running in serial mode
     */
    [[nodiscard]] bool isSerial() const noexcept { return n_ranks_ == 1; }

    // =========================================================================
    // Communication plans (built after build())
    // =========================================================================

    /**
     * @brief Get the sorted list of ghost column global IDs used locally
     *
     * Populated after calling build().
     */
    [[nodiscard]] std::span<const GlobalIndex> ghostCols() const noexcept {
        return std::span<const GlobalIndex>(ghost_cols_.data(), ghost_cols_.size());
    }

    /**
     * @brief Get ghost columns grouped by owning rank
     *
     * @param owner_rank Owning rank ID
     * @return Span of ghost column global IDs owned by that rank (sorted)
     */
    [[nodiscard]] std::span<const GlobalIndex> ghostColsOwnedBy(int owner_rank) const;

#if FE_HAS_MPI
    /**
     * @brief Get MPI communicator
     */
    [[nodiscard]] MPI_Comm comm() const noexcept { return comm_; }
#endif

    // =========================================================================
    // Validation and statistics
    // =========================================================================

    /**
     * @brief Validate pattern consistency across ranks (collective)
     *
     * Checks:
     * - Ghost column lists are consistent
     * - Total NNZ matches across diag/offdiag
     * - No duplicate entries
     *
     * @return true if valid on all ranks
     */
    [[nodiscard]] bool validateParallel(const DistributedSparsityPattern& pattern) const;

    /**
     * @brief Compute global NNZ (collective reduction)
     *
     * @param pattern Pattern to analyze
     * @return Total global NNZ across all ranks
     */
    [[nodiscard]] GlobalIndex computeGlobalNnz(const DistributedSparsityPattern& pattern) const;

    /**
     * @brief Compute global statistics (collective reduction)
     *
     * @param pattern Pattern to analyze
     * @return Statistics with both local and global information
     */
    [[nodiscard]] DistributedSparsityStats computeGlobalStats(
        const DistributedSparsityPattern& pattern) const;

private:
    // MPI state
#if FE_HAS_MPI
    MPI_Comm comm_{MPI_COMM_SELF};
#endif
    int my_rank_{0};
    int n_ranks_{1};

    // Ownership
    DofOwnership row_ownership_;
    DofOwnership col_ownership_;
    bool col_ownership_set_{false};

    // DOF maps
    std::shared_ptr<IDofMapQuery> row_dof_map_;
    std::shared_ptr<IDofMapQuery> col_dof_map_;

    // Build options
    SparsityBuildOptions options_;
    bool exchange_ghost_rows_{false};

    // Cached communication plans (filled in build())
    std::vector<GlobalIndex> ghost_cols_;
    std::vector<std::vector<GlobalIndex>> ghost_cols_by_owner_;

    // Internal methods
    void exchangeGhostRowSparsity(DistributedSparsityPattern& pattern);
    void detectGhostDofs(const DistributedSparsityPattern& pattern, const DofOwnership& col_own);
    };

// ============================================================================
// Convenience functions
// ============================================================================

/**
 * @brief Build distributed sparsity pattern (convenience function)
 *
 * @param dof_map DOF map for both rows and columns
 * @param first_owned_dof First owned DOF index on this rank
 * @param n_owned_dofs Number of owned DOFs on this rank
 * @param global_n_dofs Total global DOF count
 * @param options Build options
 * @return Finalized distributed sparsity pattern
 */
[[nodiscard]] inline DistributedSparsityPattern buildDistributedPattern(
    const dofs::DofMap& dof_map,
    GlobalIndex first_owned_dof,
    GlobalIndex n_owned_dofs,
    GlobalIndex global_n_dofs,
    const SparsityBuildOptions& options = SparsityBuildOptions{}) {

    DistributedSparsityBuilder builder(dof_map, first_owned_dof, n_owned_dofs, global_n_dofs);
    builder.setOptions(options);
    return builder.build();
}

#if FE_HAS_MPI
/**
 * @brief Build distributed sparsity pattern with MPI (convenience function)
 *
 * @param dof_map DOF map for both rows and columns
 * @param comm MPI communicator
 * @param options Build options
 * @return Finalized distributed sparsity pattern
 */
[[nodiscard]] DistributedSparsityPattern buildDistributedPattern(
    const dofs::DofMap& dof_map,
    MPI_Comm comm,
    const SparsityBuildOptions& options = SparsityBuildOptions{});
#endif

} // namespace sparsity
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_SPARSITY_PARALLEL_SPARSITY_H

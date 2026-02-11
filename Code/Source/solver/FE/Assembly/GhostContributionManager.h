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

#ifndef SVMP_FE_ASSEMBLY_GHOST_CONTRIBUTION_MANAGER_H
#define SVMP_FE_ASSEMBLY_GHOST_CONTRIBUTION_MANAGER_H

/**
 * @file GhostContributionManager.h
 * @brief MPI ghost contribution accumulation and reverse scatter
 *
 * GhostContributionManager handles the communication of ghost DOF contributions
 * during parallel assembly. In distributed FEM, element matrices may contribute
 * to DOFs owned by other processes. This class manages:
 *
 * 1. Accumulation of ghost contributions locally
 * 2. Reverse scatter: send ghost contributions to owning processes
 * 3. Reception and accumulation of contributions from other processes
 *
 * Two ghost policies are supported:
 *
 * - OwnedRowsOnly: Only assemble to locally owned rows.
 *   Contributions to ghost rows are discarded. Simple but may lose accuracy
 *   at partition boundaries.
 *
 * - ReverseScatter: Assemble to all rows (owned + ghost), then reverse-scatter
 *   ghost contributions to owners. More accurate but requires communication.
 *
 * Determinism requirements:
 * - Ghost accumulation order must be deterministic (stable sorting by global row/col)
 * - Communication must be reproducible across runs with same partition
 *
 * @see ParallelAssembler for the parallel assembly orchestration
 */

#include "Core/Types.h"
#include "Core/FEConfig.h"
#include "Assembler.h"  // For GhostPolicy enum

#include <vector>
#include <span>
#include <unordered_map>
#include <memory>
#include <cstdint>
#include <chrono>

#if FE_HAS_MPI
#  include <mpi.h>
#endif

namespace svmp {
namespace FE {

// Forward declarations
namespace dofs {
    class DofMap;
    class DofPartition;
    class GhostDofManager;
}

namespace assembly {

// ============================================================================
// Ghost Contribution Entry
// ============================================================================

/**
 * @brief Single ghost contribution entry
 */
struct GhostContribution {
    GlobalIndex global_row;     ///< Global row index
    GlobalIndex global_col;     ///< Global column index (ignored for vectors)
    Real value;                 ///< Contribution value

    bool operator<(const GhostContribution& other) const noexcept {
        if (global_row != other.global_row) return global_row < other.global_row;
        return global_col < other.global_col;
    }
};

/**
 * @brief Single ghost vector contribution entry
 */
struct GhostVectorContribution {
    GlobalIndex global_row;  ///< Global row index
    Real value;              ///< Contribution value

    bool operator<(const GhostVectorContribution& other) const noexcept {
        return global_row < other.global_row;
    }
};

/**
 * @brief Buffer for ghost contributions destined for a single rank
 */
struct GhostBuffer {
    int dest_rank;                          ///< Destination rank
    std::vector<GhostContribution> entries; ///< Contributions to send
    std::vector<GhostVectorContribution> vector_entries; ///< Vector contributions to send

    void clear() {
        entries.clear();
        vector_entries.clear();
    }

    void reserve(std::size_t n_matrix, std::size_t n_vector) {
        entries.reserve(n_matrix);
        vector_entries.reserve(n_vector);
    }
};

// ============================================================================
// Ghost Contribution Manager
// ============================================================================

/**
 * @brief Manages ghost DOF contributions for parallel assembly
 *
 * Usage pattern:
 * @code
 *   GhostContributionManager ghost_manager(dof_map, mpi_comm);
 *   ghost_manager.setPolicy(GhostPolicy::ReverseScatter);
 *   ghost_manager.initialize();
 *
 *   // During assembly
 *   for (auto cell : local_cells) {
 *       auto dofs = getDofs(cell);
 *       auto local_mat = computeMatrix(cell);
 *
 *       // Add contributions - manager routes to local or ghost buffer
 *       for (int i = 0; i < n_dofs; ++i) {
 *           for (int j = 0; j < n_dofs; ++j) {
 *               ghost_manager.addMatrixContribution(
 *                   dofs[i], dofs[j], local_mat[i][j]);
 *           }
 *       }
 *   }
 *
 *   // After local assembly, exchange ghost contributions
 *   ghost_manager.exchangeContributions();
 *
 *   // Apply received contributions to global matrix
 *   ghost_manager.applyReceivedContributions(matrix_view);
 * @endcode
 */
class GhostContributionManager {
public:
    // =========================================================================
    // Construction
    // =========================================================================

    /**
     * @brief Default constructor
     */
    GhostContributionManager();

    /**
     * @brief Construct with DOF map
     */
    explicit GhostContributionManager(const dofs::DofMap& dof_map);

#if FE_HAS_MPI
    /**
     * @brief Construct with DOF map and MPI communicator
     */
    GhostContributionManager(const dofs::DofMap& dof_map, MPI_Comm comm);
#endif

    /**
     * @brief Destructor
     */
    ~GhostContributionManager();

    /**
     * @brief Move constructor
     */
    GhostContributionManager(GhostContributionManager&& other) noexcept;

    /**
     * @brief Move assignment
     */
    GhostContributionManager& operator=(GhostContributionManager&& other) noexcept;

    // Non-copyable
    GhostContributionManager(const GhostContributionManager&) = delete;
    GhostContributionManager& operator=(const GhostContributionManager&) = delete;

    // =========================================================================
    // Configuration
    // =========================================================================

    /**
     * @brief Set the DOF map
     */
    void setDofMap(const dofs::DofMap& dof_map);

    /**
     * @brief Set ghost DOF manager for ownership queries
     */
    void setGhostDofManager(const dofs::GhostDofManager& ghost_manager);

#if FE_HAS_MPI
    /**
     * @brief Set MPI communicator
     */
    void setComm(MPI_Comm comm);
#endif

    /**
     * @brief Set ghost policy
     */
    void setPolicy(GhostPolicy policy) { policy_ = policy; }

    /**
     * @brief Get current ghost policy
     */
    [[nodiscard]] GhostPolicy getPolicy() const noexcept { return policy_; }

    /**
     * @brief Enable deterministic mode (sort contributions before exchange)
     */
    void setDeterministic(bool deterministic) { deterministic_ = deterministic; }

    /**
     * @brief Set an offset applied to system DOF indices for ownership queries
     *
     * Some workflows assemble into a larger global system using a row DOF offset
     * (multi-field block assembly). In those cases the indices inserted into the
     * global matrix/vector are `row_offset + local_dof`, while DOF ownership is
     * often defined on the underlying row DofMap indexed by `local_dof`.
     *
     * This offset is subtracted from any row index passed to isOwned()/getOwnerRank()
     * before querying ownership.
     *
     * Default is 0 (no offset).
     */
    void setOwnershipOffset(GlobalIndex offset) noexcept { ownership_offset_ = offset; }

    /**
     * @brief Get the current ownership-query offset
     */
    [[nodiscard]] GlobalIndex ownershipOffset() const noexcept { return ownership_offset_; }

    // =========================================================================
    // Initialization
    // =========================================================================

    /**
     * @brief Initialize communication patterns
     *
     * Analyzes DOF ownership to determine:
     * - Which ranks own which ghost DOFs
     * - Communication graph for reverse scatter
     * - Buffer sizes for efficient communication
     */
    void initialize();

    /**
     * @brief Check if initialized
     */
    [[nodiscard]] bool isInitialized() const noexcept { return initialized_; }

    // =========================================================================
    // Contribution Accumulation
    // =========================================================================

    /**
     * @brief Add a matrix contribution
     *
     * If the row is locally owned, contribution is returned for direct insertion.
     * If the row is a ghost, contribution is buffered for later exchange.
     *
     * @param global_row Global row index
     * @param global_col Global column index
     * @param value Contribution value
     * @return true if locally owned (caller should insert), false if buffered
     */
    bool addMatrixContribution(GlobalIndex global_row, GlobalIndex global_col, Real value);

    /**
     * @brief Add a vector contribution
     *
     * @param global_row Global row index
     * @param value Contribution value
     * @return true if locally owned, false if buffered
     */
    bool addVectorContribution(GlobalIndex global_row, Real value);

    /**
     * @brief Add a batch of matrix contributions
     *
     * @param row_dofs Row DOF indices
     * @param col_dofs Column DOF indices
     * @param values Dense local matrix (row-major)
     * @param owned_contributions Output: contributions to owned rows
     */
    void addMatrixContributions(
        std::span<const GlobalIndex> row_dofs,
        std::span<const GlobalIndex> col_dofs,
        std::span<const Real> values,
        std::vector<GhostContribution>& owned_contributions);

    /**
     * @brief Check if a DOF is locally owned
     */
    [[nodiscard]] bool isOwned(GlobalIndex global_dof) const;

    /**
     * @brief Get owner rank for a DOF
     */
    [[nodiscard]] int getOwnerRank(GlobalIndex global_dof) const;

    // =========================================================================
    // Communication (MPI)
    // =========================================================================

    /**
     * @brief Exchange ghost contributions with other ranks
     *
     * Performs reverse scatter: sends buffered ghost contributions to owners,
     * receives contributions from other ranks.
     *
     * For determinism, contributions are sorted before exchange.
     */
    void exchangeContributions();

    /**
     * @brief Non-blocking exchange (start)
     *
     * Initiates non-blocking sends/receives. Call waitExchange() to complete.
     */
    void startExchange();

    /**
     * @brief Non-blocking exchange (complete)
     *
     * Waits for all exchanges to complete.
     */
    void waitExchange();

    /**
     * @brief Check if exchange is in progress
     */
    [[nodiscard]] bool isExchangeInProgress() const noexcept {
        return exchange_in_progress_;
    }

    // =========================================================================
    // Received Contribution Access
    // =========================================================================

    /**
     * @brief Get received matrix contributions
     *
     * After exchangeContributions(), this returns contributions received
     * from other ranks that should be added to the local matrix.
     */
    [[nodiscard]] std::span<const GhostContribution> getReceivedMatrixContributions() const;

    /**
     * @brief Get received vector contributions
     */
    [[nodiscard]] std::span<const GhostVectorContribution> getReceivedVectorContributions() const;

    /**
     * @brief Move received matrix contributions out of the manager
     *
     * This is primarily intended for orchestrators that overlap communication
     * with assembly and want to defer application until a later synchronization
     * point.
     */
    [[nodiscard]] std::vector<GhostContribution> takeReceivedMatrixContributions();

    /**
     * @brief Move received vector contributions out of the manager
     */
    [[nodiscard]] std::vector<GhostVectorContribution> takeReceivedVectorContributions();

    /**
     * @brief Clear received contributions after processing
     */
    void clearReceivedContributions();

    // =========================================================================
    // Buffer Management
    // =========================================================================

    /**
     * @brief Clear all send buffers for next assembly phase
     */
    void clearSendBuffers();

    /**
     * @brief Reserve buffer space
     *
     * @param entries_per_rank Expected entries per neighbor rank
     */
    void reserveBuffers(std::size_t entries_per_rank);

    /**
     * @brief Get total number of buffered matrix contributions
     */
    [[nodiscard]] std::size_t numBufferedMatrixContributions() const;

    /**
     * @brief Get total number of buffered vector contributions
     */
    [[nodiscard]] std::size_t numBufferedVectorContributions() const;

    // =========================================================================
    // Statistics
    // =========================================================================

    /**
     * @brief Get number of neighbor ranks
     */
    [[nodiscard]] int numNeighbors() const noexcept {
        return static_cast<int>(neighbor_ranks_.size());
    }

    /**
     * @brief Get neighbor rank list
     */
    [[nodiscard]] std::span<const int> getNeighborRanks() const noexcept {
        return neighbor_ranks_;
    }

    /**
     * @brief Statistics from last exchange
     */
    struct ExchangeStats {
        std::size_t bytes_sent{0};
        std::size_t bytes_received{0};
        std::size_t matrix_entries_sent{0};
        std::size_t matrix_entries_received{0};
        std::size_t vector_entries_sent{0};
        std::size_t vector_entries_received{0};
        double exchange_time_seconds{0.0};
    };

    [[nodiscard]] const ExchangeStats& getLastExchangeStats() const noexcept {
        return last_stats_;
    }

private:
    // =========================================================================
    // Internal Helpers
    // =========================================================================

    void buildCommunicationGraph();
    void sortBuffersForDeterminism();
    int findNeighborIndex(int rank) const;

    // Configuration
    const dofs::DofMap* dof_map_{nullptr};
    const dofs::GhostDofManager* ghost_dof_manager_{nullptr};
    GlobalIndex ownership_offset_{0};
    GhostPolicy policy_{GhostPolicy::ReverseScatter};
    bool deterministic_{AssemblyOptions{}.deterministic};

    // MPI info
#if FE_HAS_MPI
    MPI_Comm comm_{MPI_COMM_WORLD};
#endif
    int my_rank_{0};
    int world_size_{1};

    // Neighbor information
    std::vector<int> neighbor_ranks_;
    std::unordered_map<int, int> rank_to_neighbor_idx_;

    // Ownership lookup (ghost DOF -> owner rank)
    std::unordered_map<GlobalIndex, int> ghost_owner_rank_;

    // Send buffers (one per neighbor)
    std::vector<GhostBuffer> send_buffers_;

    // Receive buffers
    std::vector<GhostContribution> received_matrix_;
    std::vector<GhostVectorContribution> received_vector_;

    // State
    bool initialized_{false};
    bool exchange_in_progress_{false};

    // Statistics
    ExchangeStats last_stats_;

#if FE_HAS_MPI
    // MPI requests for non-blocking communication
    std::vector<std::vector<char>> send_buffers_raw_;
    std::vector<int> send_sizes_;
    std::vector<int> recv_sizes_;
    std::vector<MPI_Request> size_requests_;
    std::vector<MPI_Request> data_send_requests_;
    std::vector<MPI_Request> data_recv_requests_;
    std::vector<std::vector<char>> recv_buffers_raw_;
#endif

    std::chrono::steady_clock::time_point exchange_start_time_{};

    // Note: ownership is queried via dof_map_->isOwnedDof().
};

} // namespace assembly
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_ASSEMBLY_GHOST_CONTRIBUTION_MANAGER_H

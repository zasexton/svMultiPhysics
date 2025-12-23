/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_DOFS_DOFGRAPH_H
#define SVMP_FE_DOFS_DOFGRAPH_H

/**
 * @file DofGraph.h
 * @brief DOF connectivity graph for renumbering and sparsity
 *
 * The DofGraph represents the adjacency structure of DOFs - which DOFs
 * are connected through elements. This is essential for:
 *  - Matrix sparsity pattern generation
 *  - DOF renumbering algorithms (Cuthill-McKee, nested dissection)
 *  - AMG setup (strength of connection)
 *  - Bandwidth and fill-in estimation
 *
 * The graph is stored in CSR format for efficient traversal.
 */

#include "DofMap.h"
#include "Core/Types.h"
#include "Core/FEException.h"

#include <vector>
#include <span>
#include <memory>

namespace svmp {
namespace FE {
namespace dofs {

/**
 * @brief Coupling pattern for DOF graph construction
 */
enum class CouplingPattern : std::uint8_t {
    /**
     * @brief Cell-only coupling (CG elements)
     *
     * DOFs are connected if they appear in the same cell.
     * Standard for continuous Galerkin methods.
     */
    CellOnly,

    /**
     * @brief Cell plus face coupling (DG with face integrals)
     *
     * DOFs are connected if they are in the same cell OR
     * in neighboring cells (across shared faces).
     */
    CellPlusFace,

    /**
     * @brief Full DG coupling (all trace DOFs)
     *
     * DOFs on cell faces are connected to all DOFs on
     * adjacent cell faces.
     */
    DGCoupling,

    /**
     * @brief Block diagonal (no inter-cell coupling)
     *
     * DOFs are only connected within their own cell.
     * Used for block-Jacobi preconditioners.
     */
    BlockDiagonal
};

/**
 * @brief Options for DOF graph construction
 */
struct DofGraphOptions {
    CouplingPattern pattern{CouplingPattern::CellOnly};
    bool symmetric{true};           ///< Force symmetric graph
    bool include_diagonal{true};    ///< Include self-loops
    bool remove_duplicates{true};   ///< Remove duplicate edges
};

/**
 * @brief DOF connectivity graph in CSR format
 *
 * Represents the adjacency structure of DOFs derived from element connectivity.
 * The graph can be used for sparsity pattern generation and renumbering.
 */
class DofGraph {
public:
    // =========================================================================
    // Construction
    // =========================================================================

    DofGraph();
    ~DofGraph();

    // Move semantics
    DofGraph(DofGraph&&) noexcept;
    DofGraph& operator=(DofGraph&&) noexcept;

    // No copy (large data)
    DofGraph(const DofGraph&) = delete;
    DofGraph& operator=(const DofGraph&) = delete;

    // =========================================================================
    // Graph Building
    // =========================================================================

    /**
     * @brief Build graph from DOF map
     *
     * @param dof_map The DOF map providing cell->DOF connectivity
     * @param options Graph construction options
     */
    void build(const DofMap& dof_map, const DofGraphOptions& options = {});

    /**
     * @brief Build graph with face connectivity for DG
     *
     * @param dof_map The DOF map
     * @param cell_neighbors Cell neighbor information [n_cells]
     *        Each entry lists neighboring cell indices (-1 for boundary)
     * @param cell_neighbor_offsets CSR offsets into cell_neighbors
     * @param options Graph construction options
     */
    void buildWithFaces(const DofMap& dof_map,
                        std::span<const GlobalIndex> cell_neighbors,
                        std::span<const GlobalIndex> cell_neighbor_offsets,
                        const DofGraphOptions& options = {});

    /**
     * @brief Build from explicit adjacency
     *
     * @param n_dofs Number of DOFs
     * @param row_offsets CSR row offsets [n_dofs+1]
     * @param col_indices CSR column indices
     */
    void buildFromCSR(GlobalIndex n_dofs,
                      std::span<const GlobalIndex> row_offsets,
                      std::span<const GlobalIndex> col_indices);

    /**
     * @brief Invalidate the graph (mark as needing rebuild)
     */
    void invalidate();

    // =========================================================================
    // Graph Access
    // =========================================================================

    /**
     * @brief Check if graph is valid (built)
     */
    [[nodiscard]] bool isValid() const noexcept { return valid_; }

    /**
     * @brief Get number of DOFs (vertices)
     */
    [[nodiscard]] GlobalIndex numDofs() const noexcept { return n_dofs_; }

    /**
     * @brief Get number of edges (nonzeros)
     */
    [[nodiscard]] GlobalIndex numEdges() const noexcept {
        return static_cast<GlobalIndex>(col_indices_.size());
    }

    /**
     * @brief Get CSR row offsets
     */
    [[nodiscard]] std::span<const GlobalIndex> getAdjOffsets() const noexcept {
        return row_offsets_;
    }

    /**
     * @brief Get CSR column indices
     */
    [[nodiscard]] std::span<const GlobalIndex> getAdjIndices() const noexcept {
        return col_indices_;
    }

    /**
     * @brief Get neighbors of a DOF
     *
     * @param dof DOF index
     * @return Span of neighbor DOF indices
     */
    [[nodiscard]] std::span<const GlobalIndex> getNeighbors(GlobalIndex dof) const;

    /**
     * @brief Get degree (number of neighbors) of a DOF
     */
    [[nodiscard]] GlobalIndex getDegree(GlobalIndex dof) const;

    // =========================================================================
    // Graph Statistics
    // =========================================================================

    /**
     * @brief Get matrix bandwidth
     *
     * bandwidth = max over all edges |i - j|
     */
    [[nodiscard]] GlobalIndex getBandwidth() const;

    /**
     * @brief Get maximum row nonzeros
     */
    [[nodiscard]] GlobalIndex getMaxRowNnz() const;

    /**
     * @brief Get average row nonzeros
     */
    [[nodiscard]] double getAvgRowNnz() const;

    /**
     * @brief Check if graph is symmetric
     */
    [[nodiscard]] bool isSymmetric() const;

    /**
     * @brief Get profile (envelope size)
     *
     * profile = sum over rows of (max_col_in_row - min_col_in_row)
     */
    [[nodiscard]] GlobalIndex getProfile() const;

    /**
     * @brief Graph statistics structure
     */
    struct Statistics {
        GlobalIndex n_dofs{0};
        GlobalIndex n_edges{0};
        GlobalIndex bandwidth{0};
        GlobalIndex max_degree{0};
        double avg_degree{0.0};
        GlobalIndex profile{0};
        bool symmetric{false};
    };

    /**
     * @brief Get all statistics
     */
    [[nodiscard]] Statistics getStatistics() const;

    // =========================================================================
    // Graph Manipulation
    // =========================================================================

    /**
     * @brief Make graph symmetric (add missing reverse edges)
     */
    void symmetrize();

    /**
     * @brief Remove duplicate edges
     */
    void removeDuplicates();

    /**
     * @brief Sort column indices within each row
     */
    void sortIndices();

    /**
     * @brief Apply permutation to graph
     *
     * Creates new graph with permuted vertex indices.
     *
     * @param permutation new_index = permutation[old_index]
     * @return New permuted graph
     */
    [[nodiscard]] DofGraph applyPermutation(std::span<const GlobalIndex> permutation) const;

    // =========================================================================
    // Subgraph Extraction
    // =========================================================================

    /**
     * @brief Extract subgraph for subset of DOFs
     *
     * @param dof_subset DOF indices to include
     * @return Subgraph (renumbered to 0..n-1)
     */
    [[nodiscard]] DofGraph extractSubgraph(std::span<const GlobalIndex> dof_subset) const;

private:
    // CSR storage
    std::vector<GlobalIndex> row_offsets_;
    std::vector<GlobalIndex> col_indices_;

    // Metadata
    GlobalIndex n_dofs_{0};
    bool valid_{false};
    bool symmetric_{false};

    // Build helpers
    void buildCellOnly(const DofMap& dof_map, const DofGraphOptions& options);
};

// =============================================================================
// Utility Functions
// =============================================================================

/**
 * @brief Compute sparsity pattern from DOF graph
 *
 * @param graph DOF graph
 * @param row_offsets Output CSR row offsets
 * @param col_indices Output CSR column indices
 */
void computeSparsityPattern(const DofGraph& graph,
                            std::vector<GlobalIndex>& row_offsets,
                            std::vector<GlobalIndex>& col_indices);

/**
 * @brief Estimate fill-in for Cholesky factorization
 *
 * Uses symbolic factorization to count fill-in entries.
 *
 * @param graph DOF graph (must be symmetric)
 * @return Estimated number of fill-in entries
 */
GlobalIndex estimateFillIn(const DofGraph& graph);

} // namespace dofs
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_DOFS_DOFGRAPH_H

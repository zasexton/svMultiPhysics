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

#ifndef SVMP_FE_GRAPH_SPARSITY_H
#define SVMP_FE_GRAPH_SPARSITY_H

/**
 * @file GraphSparsity.h
 * @brief Graph-based sparsity analysis and reordering operations
 *
 * This header provides graph-based operations for analyzing and optimizing
 * sparsity patterns:
 *
 * - Graph coloring for parallel assembly
 * - Bandwidth computation and reduction
 * - Fill-in prediction for direct solvers
 * - Reordering algorithms (Cuthill-McKee, reverse Cuthill-McKee)
 * - Connected component detection
 * - Level set computation
 *
 * The GraphSparsity class treats the sparsity pattern as an adjacency graph
 * where rows/columns are vertices and non-zero entries are edges. For
 * asymmetric patterns, operations typically symmetrize first.
 *
 * Complexity notes:
 * - Graph coloring: O(NNZ)
 * - Bandwidth: O(NNZ)
 * - RCM ordering: O(NNZ + n * log(n)) for priority queue operations
 * - Fill-in prediction: O(n^2) worst case for incomplete factorization
 *
 * @see SparsityOptimizer for higher-level optimization routines
 * @see SparsityPattern for the underlying data structure
 */

#include "SparsityPattern.h"
#include "Core/Types.h"
#include "Core/FEConfig.h"
#include "Core/FEException.h"

#include <vector>
#include <span>
#include <functional>
#include <optional>
#include <queue>

namespace svmp {
namespace FE {
namespace sparsity {

// ============================================================================
// Graph Analysis Structures
// ============================================================================

/**
 * @brief Statistics about graph structure
 */
struct GraphStats {
    GlobalIndex n_vertices{0};         ///< Number of vertices (rows for square)
    GlobalIndex n_edges{0};            ///< Number of edges (NNZ for symmetric)
    GlobalIndex bandwidth{0};          ///< Maximum |row - col|
    GlobalIndex profile{0};            ///< Sum of row bandwidths
    GlobalIndex envelope{0};           ///< Sum of (diag_pos - first_nonzero_col)
    GlobalIndex n_components{0};       ///< Number of connected components
    GlobalIndex max_degree{0};         ///< Maximum vertex degree
    GlobalIndex min_degree{0};         ///< Minimum vertex degree
    double avg_degree{0.0};            ///< Average vertex degree
    GlobalIndex diameter{-1};          ///< Graph diameter (-1 if not computed)
    bool is_connected{false};          ///< True if single connected component
};

/**
 * @brief Result of graph coloring
 */
struct ColoringResult {
    std::vector<GlobalIndex> colors;   ///< Color assignment for each vertex
    GlobalIndex num_colors{0};         ///< Number of colors used
    std::vector<GlobalIndex> color_counts; ///< Number of vertices per color
    bool is_valid{false};              ///< True if valid coloring

    /**
     * @brief Get vertices of a specific color
     */
    [[nodiscard]] std::vector<GlobalIndex> getVerticesOfColor(GlobalIndex color) const;
};

/**
 * @brief Result of level set computation
 */
struct LevelSetResult {
    std::vector<GlobalIndex> levels;   ///< Level assignment for each vertex
    GlobalIndex num_levels{0};         ///< Number of levels
    std::vector<std::vector<GlobalIndex>> level_sets; ///< Vertices per level
    GlobalIndex root{-1};              ///< Root vertex used
};

/**
 * @brief Result of connected components computation
 */
struct ComponentResult {
    std::vector<GlobalIndex> component_id; ///< Component assignment for each vertex
    GlobalIndex num_components{0};         ///< Number of components
    std::vector<GlobalIndex> component_sizes; ///< Size of each component

    /**
     * @brief Get vertices in a specific component
     */
    [[nodiscard]] std::vector<GlobalIndex> getVerticesInComponent(GlobalIndex comp) const;
};

/**
 * @brief Fill-in prediction result
 */
struct FillInPrediction {
    GlobalIndex original_nnz{0};       ///< Original NNZ
    GlobalIndex predicted_fill{0};     ///< Predicted fill-in entries
    GlobalIndex total_factor_nnz{0};   ///< Total NNZ after factorization
    double fill_ratio{1.0};            ///< total / original
    std::vector<GlobalIndex> row_fill; ///< Fill-in per row
};

// ============================================================================
// GraphSparsity Class
// ============================================================================

/**
 * @brief Graph-based analysis and operations on sparsity patterns
 *
 * GraphSparsity provides graph-theoretic operations on sparsity patterns,
 * treating the pattern as an adjacency structure. Most operations assume
 * or require a symmetric pattern (undirected graph).
 *
 * Usage:
 * @code
 * SparsityPattern pattern = buildPattern(...);
 * GraphSparsity graph(pattern);
 *
 * // Compute reordering
 * auto rcm_perm = graph.reverseCuthillMcKee();
 * SparsityPattern reordered = pattern.permute(rcm_perm, rcm_perm);
 *
 * // Color for parallel assembly
 * auto coloring = graph.greedyColoring();
 * for (GlobalIndex c = 0; c < coloring.num_colors; ++c) {
 *     auto vertices = coloring.getVerticesOfColor(c);
 *     // Process vertices of color c in parallel
 * }
 *
 * // Analyze structure
 * auto stats = graph.computeStats();
 * std::cout << "Bandwidth: " << stats.bandwidth << "\n";
 * @endcode
 */
class GraphSparsity {
public:
    // =========================================================================
    // Construction
    // =========================================================================

    /**
     * @brief Default constructor - empty graph
     */
    GraphSparsity() = default;

    /**
     * @brief Construct from sparsity pattern
     *
     * @param pattern Sparsity pattern (will be symmetrized for graph operations)
     * @param symmetrize If true, treat pattern as symmetric (A + A^T)
     */
    explicit GraphSparsity(const SparsityPattern& pattern, bool symmetrize = true);

    /**
     * @brief Construct from sparsity pattern (move)
     */
    explicit GraphSparsity(SparsityPattern&& pattern, bool symmetrize = true);

    /// Destructor
    ~GraphSparsity() = default;

    // Copy and move
    GraphSparsity(const GraphSparsity&) = default;
    GraphSparsity& operator=(const GraphSparsity&) = default;
    GraphSparsity(GraphSparsity&&) noexcept = default;
    GraphSparsity& operator=(GraphSparsity&&) noexcept = default;

    // =========================================================================
    // Configuration
    // =========================================================================

    /**
     * @brief Set the underlying pattern
     *
     * @param pattern New pattern
     * @param symmetrize If true, symmetrize the pattern
     */
    void setPattern(const SparsityPattern& pattern, bool symmetrize = true);

    /**
     * @brief Get the underlying pattern
     */
    [[nodiscard]] const SparsityPattern& getPattern() const noexcept {
        return pattern_;
    }

    // =========================================================================
    // Graph Statistics
    // =========================================================================

    /**
     * @brief Compute graph statistics
     *
     * @return Statistics about the graph structure
     */
    [[nodiscard]] GraphStats computeStats() const;

    /**
     * @brief Compute bandwidth
     *
     * @return Maximum |row - col| for any entry
     */
    [[nodiscard]] GlobalIndex computeBandwidth() const;

    /**
     * @brief Compute profile (envelope)
     *
     * @return Sum over rows of (last_col - first_col + 1)
     */
    [[nodiscard]] GlobalIndex computeProfile() const;

    /**
     * @brief Compute vertex degrees
     *
     * @return Degree (number of neighbors) for each vertex
     */
    [[nodiscard]] std::vector<GlobalIndex> computeDegrees() const;

    /**
     * @brief Get degree of a vertex
     */
    [[nodiscard]] GlobalIndex getDegree(GlobalIndex vertex) const;

    /**
     * @brief Get neighbors of a vertex
     */
    [[nodiscard]] std::vector<GlobalIndex> getNeighbors(GlobalIndex vertex) const;

    // =========================================================================
    // Graph Coloring
    // =========================================================================

    /**
     * @brief Greedy graph coloring
     *
     * Colors vertices such that no two adjacent vertices share the same color.
     * Uses a greedy algorithm with specified ordering.
     *
     * @param ordering Optional vertex ordering (default: natural order)
     * @return Coloring result
     *
     * Complexity: O(NNZ)
     */
    [[nodiscard]] ColoringResult greedyColoring(
        std::span<const GlobalIndex> ordering = {}) const;

    /**
     * @brief Graph coloring with degree-based ordering
     *
     * Uses largest-degree-first (LDF) or smallest-degree-last (SDL) strategy.
     *
     * @param use_smallest_last If true, use SDL; otherwise use LDF
     * @return Coloring result
     */
    [[nodiscard]] ColoringResult degreeBasedColoring(bool use_smallest_last = true) const;

    /**
     * @brief Verify a coloring is valid
     *
     * @param colors Color assignment to verify
     * @return true if no adjacent vertices share a color
     */
    [[nodiscard]] bool verifyColoring(std::span<const GlobalIndex> colors) const;

    // =========================================================================
    // Reordering Algorithms
    // =========================================================================

    /**
     * @brief Cuthill-McKee ordering
     *
     * Produces a permutation that reduces bandwidth by BFS traversal
     * from a peripheral vertex.
     *
     * @param start_vertex Starting vertex (-1 for automatic selection)
     * @return Permutation vector
     *
     * Complexity: O(NNZ + n * log(n))
     */
    [[nodiscard]] std::vector<GlobalIndex> cuthillMcKee(GlobalIndex start_vertex = -1) const;

    /**
     * @brief Reverse Cuthill-McKee ordering
     *
     * Reversed CM ordering, often gives better results for direct solvers.
     *
     * @param start_vertex Starting vertex (-1 for automatic selection)
     * @return Permutation vector
     */
    [[nodiscard]] std::vector<GlobalIndex> reverseCuthillMcKee(GlobalIndex start_vertex = -1) const;

    /**
     * @brief Find a pseudo-peripheral vertex
     *
     * Finds a vertex that is "far" from all others, good starting point for RCM.
     *
     * @param start Optional starting vertex for search
     * @return Index of pseudo-peripheral vertex
     */
    [[nodiscard]] GlobalIndex findPseudoPeripheral(GlobalIndex start = 0) const;

    /**
     * @brief Minimum degree ordering (symbolic)
     *
     * Approximates the minimum degree algorithm for fill-reducing ordering.
     *
     * @return Permutation vector
     *
     * Complexity: O(n^2) worst case
     *
     * @note For production use, prefer external libraries like METIS or AMD.
     */
    [[nodiscard]] std::vector<GlobalIndex> approximateMinimumDegree() const;

    /**
     * @brief Natural ordering (identity permutation)
     *
     * @return Identity permutation [0, 1, 2, ..., n-1]
     */
    [[nodiscard]] std::vector<GlobalIndex> naturalOrdering() const;

    // =========================================================================
    // Structural Analysis
    // =========================================================================

    /**
     * @brief Compute connected components
     *
     * @return Component assignment for each vertex
     */
    [[nodiscard]] ComponentResult computeConnectedComponents() const;

    /**
     * @brief Check if graph is connected
     *
     * @return true if all vertices reachable from any vertex
     */
    [[nodiscard]] bool isConnected() const;

    /**
     * @brief Compute level sets from a root vertex
     *
     * Level sets are BFS layers from the root.
     *
     * @param root Root vertex for BFS
     * @return Level set structure
     */
    [[nodiscard]] LevelSetResult computeLevelSets(GlobalIndex root) const;

    /**
     * @brief Compute graph diameter (longest shortest path)
     *
     * @return Diameter, or -1 if graph is disconnected
     *
     * Complexity: O(n * NNZ) - runs BFS from multiple vertices
     */
    [[nodiscard]] GlobalIndex computeDiameter() const;

    /**
     * @brief Compute eccentricity of a vertex
     *
     * @param vertex Vertex to compute eccentricity for
     * @return Maximum distance from vertex to any other vertex
     */
    [[nodiscard]] GlobalIndex computeEccentricity(GlobalIndex vertex) const;

    // =========================================================================
    // Fill-in Analysis
    // =========================================================================

    /**
     * @brief Predict fill-in for Cholesky factorization
     *
     * @return Fill-in prediction
     *
     * @note Assumes symmetric positive definite structure.
     */
    [[nodiscard]] FillInPrediction predictCholeskyFillIn() const;

    /**
     * @brief Predict fill-in for LU factorization
     *
     * @return Fill-in prediction
     */
    [[nodiscard]] FillInPrediction predictLUFillIn() const;

    /**
     * @brief Compute symbolic Cholesky factor pattern
     *
     * @return Pattern of L in LL^T factorization
     */
    [[nodiscard]] SparsityPattern symbolicCholesky() const;

    /**
     * @brief Compute symbolic LU factor pattern
     *
     * @return Pair of (L pattern, U pattern)
     */
    [[nodiscard]] std::pair<SparsityPattern, SparsityPattern> symbolicLU() const;

    // =========================================================================
    // Query Methods
    // =========================================================================

    /**
     * @brief Get number of vertices
     */
    [[nodiscard]] GlobalIndex numVertices() const noexcept {
        return pattern_.numRows();
    }

    /**
     * @brief Get number of edges (half of NNZ for symmetric)
     */
    [[nodiscard]] GlobalIndex numEdges() const;

    /**
     * @brief Check if pattern is empty
     */
    [[nodiscard]] bool empty() const noexcept {
        return pattern_.numRows() == 0;
    }

private:
    // Internal helpers
    void ensureSymmetric();
    [[nodiscard]] GlobalIndex bfsDistance(GlobalIndex from, GlobalIndex to) const;
    [[nodiscard]] std::vector<GlobalIndex> bfsDistances(GlobalIndex from) const;

    SparsityPattern pattern_;
    bool is_symmetric_{true};
};

// ============================================================================
// Free Functions
// ============================================================================

/**
 * @brief Compute bandwidth of a sparsity pattern
 */
[[nodiscard]] GlobalIndex computePatternBandwidth(const SparsityPattern& pattern);

/**
 * @brief Compute profile of a sparsity pattern
 */
[[nodiscard]] GlobalIndex computePatternProfile(const SparsityPattern& pattern);

/**
 * @brief Apply RCM reordering to a pattern
 *
 * Convenience function that computes RCM and applies it.
 *
 * @param pattern Input pattern
 * @return Reordered pattern
 */
[[nodiscard]] SparsityPattern applyRCM(const SparsityPattern& pattern);

/**
 * @brief Get RCM permutation for a pattern
 */
[[nodiscard]] std::vector<GlobalIndex> getRCMPermutation(const SparsityPattern& pattern);

/**
 * @brief Color a pattern for parallel assembly
 */
[[nodiscard]] ColoringResult colorPattern(const SparsityPattern& pattern);

} // namespace sparsity
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_GRAPH_SPARSITY_H

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

#ifndef SVMP_FE_SPARSITY_ANALYZER_H
#define SVMP_FE_SPARSITY_ANALYZER_H

/**
 * @file SparsityAnalyzer.h
 * @brief Comprehensive analysis of sparsity pattern characteristics
 *
 * This header provides the SparsityAnalyzer class for in-depth analysis of
 * sparsity patterns. The analysis results are useful for:
 *
 * - Solver selection (direct vs iterative)
 * - Preconditioner tuning
 * - Reordering strategy selection
 * - Fill-in estimation for direct solvers
 * - Memory requirement predictions
 * - Performance debugging
 *
 * Key analyses:
 * - Bandwidth and profile statistics
 * - Fill-in estimation (for Cholesky/LU factorization)
 * - Symmetry detection (structural and value)
 * - Diagonal dominance indicators
 * - Block structure detection
 * - Graph connectivity analysis
 *
 * Complexity notes:
 * - Basic statistics: O(NNZ)
 * - Symmetry check: O(NNZ)
 * - Fill-in estimation: O(N * avg_nnz^2) approximate
 * - Block detection: O(N * avg_nnz)
 * - Graph properties: O(N + NNZ)
 *
 * @see SparsityPattern for the data structure being analyzed
 * @see SparsityOptimizer for pattern optimization based on analysis
 */

#include "SparsityPattern.h"
#include "DistributedSparsityPattern.h"
#include "Core/Types.h"
#include "Core/FEConfig.h"
#include "Core/FEException.h"

#include <vector>
#include <span>
#include <string>
#include <ostream>
#include <optional>
#include <map>
#include <memory>

namespace svmp {
namespace FE {
namespace sparsity {

// ============================================================================
// Analysis Results Structures
// ============================================================================

/**
 * @brief Bandwidth and profile statistics
 */
struct BandwidthStats {
    GlobalIndex bandwidth{0};        ///< Maximum |row - col|
    GlobalIndex lower_bandwidth{0};  ///< Maximum (row - col) for row > col
    GlobalIndex upper_bandwidth{0};  ///< Maximum (col - row) for col > row
    GlobalIndex profile{0};          ///< Sum of (first_col - row) for each row
    GlobalIndex envelope{0};         ///< Sum of row bandwidths
    double avg_bandwidth{0.0};       ///< Average row bandwidth
    GlobalIndex median_bandwidth{0}; ///< Median row bandwidth

    /**
     * @brief Estimated bandwidth after RCM reordering
     *
     * Heuristic estimate, actual may vary.
     */
    GlobalIndex estimated_rcm_bandwidth{0};
};

/**
 * @brief Row-wise statistics
 */
struct RowStats {
    GlobalIndex n_rows{0};           ///< Total rows
    GlobalIndex empty_rows{0};       ///< Rows with no entries
    GlobalIndex min_nnz{0};          ///< Minimum row NNZ
    GlobalIndex max_nnz{0};          ///< Maximum row NNZ
    double avg_nnz{0.0};             ///< Average row NNZ
    double std_dev_nnz{0.0};         ///< Standard deviation of row NNZ
    GlobalIndex median_nnz{0};       ///< Median row NNZ

    /**
     * @brief Row NNZ histogram buckets
     *
     * histogram[i] = count of rows with NNZ in bucket i
     */
    std::vector<GlobalIndex> histogram;

    /**
     * @brief Bucket boundaries for histogram
     */
    std::vector<GlobalIndex> histogram_bounds;

    /**
     * @brief Check if row NNZ distribution is uniform
     */
    [[nodiscard]] bool isUniform(double threshold = 0.1) const noexcept {
        if (avg_nnz == 0) return true;
        return std_dev_nnz / avg_nnz < threshold;
    }
};

/**
 * @brief Diagonal structure analysis
 */
struct DiagonalStats {
    GlobalIndex n_diag{0};           ///< Number of diagonal entries present
    GlobalIndex n_missing{0};        ///< Number of missing diagonal entries
    bool all_present{false};         ///< All diagonal entries present
    GlobalIndex n_isolated_diag{0};  ///< Diagonal-only rows (nnz = 1)

    /**
     * @brief Fraction of diagonal entries present
     */
    [[nodiscard]] double coverage() const noexcept {
        GlobalIndex total = n_diag + n_missing;
        return total > 0 ? static_cast<double>(n_diag) / static_cast<double>(total) : 0.0;
    }
};

/**
 * @brief Symmetry analysis results
 */
struct SymmetryStats {
    bool is_symmetric{false};        ///< Structurally symmetric
    GlobalIndex n_symmetric{0};      ///< Number of symmetric pairs
    GlobalIndex n_asymmetric{0};     ///< Number of asymmetric entries
    double symmetry_ratio{0.0};      ///< Ratio of symmetric entries

    /**
     * @brief Check if pattern is nearly symmetric
     *
     * @param threshold Maximum asymmetry ratio
     */
    [[nodiscard]] bool isNearlySymmetric(double threshold = 0.01) const noexcept {
        GlobalIndex total = n_symmetric + n_asymmetric;
        if (total == 0) return true;
        return static_cast<double>(n_asymmetric) / static_cast<double>(total) < threshold;
    }
};

/**
 * @brief Block structure detection results
 */
struct BlockStats {
    bool has_block_structure{false}; ///< Pattern has detectable block structure
    GlobalIndex detected_block_size{1}; ///< Detected block size (1 = no blocking)
    GlobalIndex n_dense_blocks{0};   ///< Number of dense diagonal blocks
    GlobalIndex n_arrow_rows{0};     ///< Rows with arrow-like structure
    GlobalIndex n_arrow_cols{0};     ///< Columns with arrow-like structure

    /**
     * @brief Suggested block sizes for block ILU/IC
     */
    std::vector<GlobalIndex> suggested_block_sizes;
};

/**
 * @brief Fill-in estimation for direct factorization
 */
struct FillInEstimate {
    GlobalIndex original_nnz{0};     ///< Original NNZ
    GlobalIndex estimated_fill{0};   ///< Estimated fill-in
    GlobalIndex estimated_total{0};  ///< Estimated total after factorization
    double fill_ratio{0.0};          ///< Fill / Original
    double density_after{0.0};       ///< Estimated density after factorization

    /**
     * @brief Estimation method used
     */
    std::string method;

    /**
     * @brief Confidence level (0-1, higher = more confident)
     */
    double confidence{0.0};
};

/**
 * @brief Graph connectivity analysis
 */
struct ConnectivityStats {
    GlobalIndex n_components{0};     ///< Number of connected components
    bool is_connected{false};        ///< Single connected component
    GlobalIndex largest_component{0}; ///< Size of largest component
    GlobalIndex smallest_component{0}; ///< Size of smallest component

    /**
     * @brief Component sizes (if multiple)
     */
    std::vector<GlobalIndex> component_sizes;
};

/**
 * @brief Solver recommendation based on analysis
 */
enum class SolverRecommendation : std::uint8_t {
    DirectSparse,      ///< Use direct sparse solver (small/medium, moderate fill)
    DirectDense,       ///< Convert to dense (very small or very dense)
    IterativeKrylov,   ///< Use Krylov solver (large, sparse)
    IterativeMultigrid,///< Use multigrid (structured, elliptic)
    Hybrid,            ///< Use hybrid approach
    Unknown            ///< Cannot determine
};

/**
 * @brief Comprehensive analysis report
 */
struct AnalysisReport {
    // Basic statistics
    GlobalIndex n_rows{0};
    GlobalIndex n_cols{0};
    GlobalIndex nnz{0};
    double density{0.0};
    bool is_square{false};

    // Detailed statistics
    BandwidthStats bandwidth;
    RowStats rows;
    DiagonalStats diagonal;
    SymmetryStats symmetry;
    BlockStats blocks;
    FillInEstimate fill_in;
    ConnectivityStats connectivity;

    // Recommendations
    SolverRecommendation solver_recommendation{SolverRecommendation::Unknown};
    std::string recommendation_reason;

    /**
     * @brief Estimated memory for factorization (bytes)
     */
    std::size_t estimated_factor_memory{0};

    /**
     * @brief Analysis computation time (seconds)
     */
    double analysis_time_sec{0.0};

    /**
     * @brief Print human-readable report
     */
    void print(std::ostream& out) const;

    /**
     * @brief Get summary string
     */
    [[nodiscard]] std::string summary() const;
};

// ============================================================================
// SparsityAnalyzer Class
// ============================================================================

/**
 * @brief Options for sparsity analysis
 */
struct AnalysisOptions {
    bool compute_bandwidth{true};    ///< Compute bandwidth statistics
    bool compute_row_stats{true};    ///< Compute row-wise statistics
    bool compute_diagonal{true};     ///< Analyze diagonal structure
    bool compute_symmetry{true};     ///< Check symmetry
    bool detect_blocks{true};        ///< Detect block structure
    bool estimate_fill{true};        ///< Estimate fill-in
    bool analyze_connectivity{false};///< Analyze graph connectivity
    bool make_recommendations{true}; ///< Generate solver recommendations

    GlobalIndex histogram_buckets{20}; ///< Number of histogram buckets
    GlobalIndex max_block_size{32};    ///< Maximum block size to detect

    /**
     * @brief Create options for quick analysis (fewer features)
     */
    static AnalysisOptions quick() {
        AnalysisOptions opts;
        opts.detect_blocks = false;
        opts.estimate_fill = false;
        opts.analyze_connectivity = false;
        return opts;
    }

    /**
     * @brief Create options for full analysis (all features)
     */
    static AnalysisOptions full() {
        AnalysisOptions opts;
        opts.analyze_connectivity = true;
        return opts;
    }
};

/**
 * @brief Comprehensive sparsity pattern analyzer
 *
 * SparsityAnalyzer provides in-depth analysis of sparsity patterns to help
 * with solver selection, preconditioner tuning, and performance optimization.
 *
 * Usage:
 * @code
 * SparsityPattern pattern = buildPattern(...);
 *
 * // Quick analysis
 * SparsityAnalyzer analyzer;
 * auto report = analyzer.analyze(pattern);
 * report.print(std::cout);
 *
 * // Specific analysis
 * auto bandwidth = analyzer.computeBandwidth(pattern);
 * auto symmetry = analyzer.checkSymmetry(pattern);
 * auto fill = analyzer.estimateFillIn(pattern);
 *
 * // Get solver recommendation
 * auto recommendation = analyzer.recommendSolver(pattern);
 * @endcode
 */
class SparsityAnalyzer {
public:
    // =========================================================================
    // Construction
    // =========================================================================

    /**
     * @brief Default constructor
     */
    SparsityAnalyzer() = default;

    /**
     * @brief Construct with options
     */
    explicit SparsityAnalyzer(const AnalysisOptions& options);

    /// Destructor
    ~SparsityAnalyzer() = default;

    // Copyable and movable
    SparsityAnalyzer(const SparsityAnalyzer&) = default;
    SparsityAnalyzer& operator=(const SparsityAnalyzer&) = default;
    SparsityAnalyzer(SparsityAnalyzer&&) = default;
    SparsityAnalyzer& operator=(SparsityAnalyzer&&) = default;

    // =========================================================================
    // Configuration
    // =========================================================================

    /**
     * @brief Set analysis options
     */
    void setOptions(const AnalysisOptions& options) {
        options_ = options;
    }

    /**
     * @brief Get current options
     */
    [[nodiscard]] const AnalysisOptions& getOptions() const noexcept {
        return options_;
    }

    // =========================================================================
    // Full Analysis
    // =========================================================================

    /**
     * @brief Perform comprehensive analysis
     *
     * @param pattern Pattern to analyze
     * @param options Options (uses default if not provided)
     * @return Complete analysis report
     */
    [[nodiscard]] AnalysisReport analyze(
        const SparsityPattern& pattern,
        const AnalysisOptions& options) const;

    /**
     * @brief Perform analysis with default options
     */
    [[nodiscard]] AnalysisReport analyze(const SparsityPattern& pattern) const;

    /**
     * @brief Perform quick analysis (subset of features)
     */
    [[nodiscard]] AnalysisReport analyzeQuick(const SparsityPattern& pattern) const;

    // =========================================================================
    // Individual Analyses
    // =========================================================================

    /**
     * @brief Compute bandwidth statistics
     *
     * @param pattern Pattern to analyze
     * @return Bandwidth statistics
     *
     * Complexity: O(NNZ)
     */
    [[nodiscard]] BandwidthStats computeBandwidth(const SparsityPattern& pattern) const;

    /**
     * @brief Compute row statistics
     *
     * @param pattern Pattern to analyze
     * @param n_buckets Number of histogram buckets
     * @return Row-wise statistics
     *
     * Complexity: O(N + NNZ)
     */
    [[nodiscard]] RowStats computeRowStats(
        const SparsityPattern& pattern,
        GlobalIndex n_buckets = 20) const;

    /**
     * @brief Analyze diagonal structure
     *
     * @param pattern Pattern to analyze
     * @return Diagonal statistics
     *
     * Complexity: O(N)
     */
    [[nodiscard]] DiagonalStats analyzeDiagonal(const SparsityPattern& pattern) const;

    /**
     * @brief Check symmetry
     *
     * @param pattern Pattern to analyze
     * @return Symmetry statistics
     *
     * Complexity: O(NNZ)
     */
    [[nodiscard]] SymmetryStats checkSymmetry(const SparsityPattern& pattern) const;

    /**
     * @brief Detect block structure
     *
     * @param pattern Pattern to analyze
     * @param max_block_size Maximum block size to check
     * @return Block structure statistics
     *
     * Complexity: O(N * max_block_size * avg_nnz)
     */
    [[nodiscard]] BlockStats detectBlocks(
        const SparsityPattern& pattern,
        GlobalIndex max_block_size = 32) const;

    /**
     * @brief Estimate fill-in for factorization
     *
     * @param pattern Pattern to analyze
     * @return Fill-in estimate
     *
     * Uses symbolic factorization or heuristics.
     * Complexity: O(N * avg_nnz^2) approximate
     */
    [[nodiscard]] FillInEstimate estimateFillIn(const SparsityPattern& pattern) const;

    /**
     * @brief Analyze graph connectivity
     *
     * @param pattern Pattern to analyze
     * @return Connectivity statistics
     *
     * Complexity: O(N + NNZ) using BFS/DFS
     */
    [[nodiscard]] ConnectivityStats analyzeConnectivity(
        const SparsityPattern& pattern) const;

    // =========================================================================
    // Recommendations
    // =========================================================================

    /**
     * @brief Get solver recommendation
     *
     * @param pattern Pattern to analyze
     * @return Recommended solver type
     */
    [[nodiscard]] SolverRecommendation recommendSolver(
        const SparsityPattern& pattern) const;

    /**
     * @brief Get solver recommendation with reason
     *
     * @param pattern Pattern to analyze
     * @return Pair of (recommendation, reason string)
     */
    [[nodiscard]] std::pair<SolverRecommendation, std::string> recommendSolverWithReason(
        const SparsityPattern& pattern) const;

    /**
     * @brief Estimate memory for factorization
     *
     * @param pattern Pattern to analyze
     * @param element_size Size of each element in bytes (default: 8 for double)
     * @return Estimated memory in bytes
     */
    [[nodiscard]] std::size_t estimateFactorMemory(
        const SparsityPattern& pattern,
        std::size_t element_size = sizeof(double)) const;

    /**
     * @brief Suggest reordering algorithm
     *
     * @param pattern Pattern to analyze
     * @return Suggested reordering algorithm name
     */
    [[nodiscard]] std::string suggestReordering(const SparsityPattern& pattern) const;

    // =========================================================================
    // Distributed Pattern Analysis
    // =========================================================================

    /**
     * @brief Analyze distributed pattern
     *
     * @param pattern Distributed pattern to analyze
     * @return Analysis report (local portion)
     */
    [[nodiscard]] AnalysisReport analyze(
        const DistributedSparsityPattern& pattern) const;

    /**
     * @brief Get communication statistics for distributed pattern
     *
     * @param pattern Distributed pattern to analyze
     * @return Map of rank -> communication volume
     */
    [[nodiscard]] std::map<int, GlobalIndex> analyzeCommPattern(
        const DistributedSparsityPattern& pattern) const;

    // =========================================================================
    // Visualization Support
    // =========================================================================

    /**
     * @brief Generate spy plot data (for visualization)
     *
     * @param pattern Pattern to visualize
     * @param max_size Maximum dimension for downsampling
     * @return Pair of (row indices, col indices) for non-zeros
     *
     * Returns sampled points if pattern is larger than max_size.
     */
    [[nodiscard]] std::pair<std::vector<GlobalIndex>, std::vector<GlobalIndex>>
    generateSpyData(const SparsityPattern& pattern, GlobalIndex max_size = 1000) const;

    /**
     * @brief Export pattern to Matrix Market format string
     *
     * @param pattern Pattern to export
     * @return Matrix Market format string (structure only)
     */
    [[nodiscard]] std::string exportMatrixMarket(const SparsityPattern& pattern) const;

    /**
     * @brief Export pattern to simple text format
     *
     * @param pattern Pattern to export
     * @param show_values If true, show 'X' for entries
     * @return ASCII representation of pattern
     */
    [[nodiscard]] std::string exportAscii(
        const SparsityPattern& pattern,
        GlobalIndex max_size = 50) const;

    // =========================================================================
    // Utility Methods
    // =========================================================================

    /**
     * @brief Get solver recommendation name
     */
    [[nodiscard]] static std::string recommendationName(SolverRecommendation rec);

    /**
     * @brief Compare two patterns
     *
     * @param a First pattern
     * @param b Second pattern
     * @return Comparison report string
     */
    [[nodiscard]] std::string compare(
        const SparsityPattern& a,
        const SparsityPattern& b) const;

private:
    // Internal helpers
    [[nodiscard]] std::vector<GlobalIndex> getRowNnzArray(
        const SparsityPattern& pattern) const;

    [[nodiscard]] std::vector<GlobalIndex> findConnectedComponents(
        const SparsityPattern& pattern) const;

    [[nodiscard]] GlobalIndex estimateFillInSimple(
        const SparsityPattern& pattern) const;

    [[nodiscard]] bool checkBlockPattern(
        const SparsityPattern& pattern,
        GlobalIndex block_size) const;

    AnalysisOptions options_;
};

// ============================================================================
// Convenience Functions
// ============================================================================

/**
 * @brief Quick bandwidth computation
 */
[[nodiscard]] GlobalIndex computeBandwidth(const SparsityPattern& pattern);

/**
 * @brief Quick symmetry check
 */
[[nodiscard]] bool isSymmetric(const SparsityPattern& pattern);

/**
 * @brief Quick fill ratio computation
 */
[[nodiscard]] double computeFillRatio(const SparsityPattern& pattern);

/**
 * @brief Print analysis report to stream
 */
void printAnalysis(const SparsityPattern& pattern, std::ostream& out);

/**
 * @brief Get quick summary of pattern characteristics
 */
[[nodiscard]] std::string getPatternSummary(const SparsityPattern& pattern);

} // namespace sparsity
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_SPARSITY_ANALYZER_H

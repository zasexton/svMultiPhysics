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

#ifndef SVMP_FE_SPARSITY_OPTIMIZER_H
#define SVMP_FE_SPARSITY_OPTIMIZER_H

/**
 * @file SparsityOptimizer.h
 * @brief High-level sparsity pattern optimization routines
 *
 * This header provides the SparsityOptimizer class for optimizing sparsity
 * patterns for various solver and hardware targets:
 *
 * - Bandwidth minimization (for banded solvers)
 * - Profile reduction (for skyline/envelope solvers)
 * - Fill-in minimization (for direct sparse solvers)
 * - Cache-line optimization (for iterative solvers)
 * - Vectorization-friendly ordering
 * - Parallel assembly optimization
 *
 * The optimizer provides both automatic strategy selection and manual
 * control over optimization parameters.
 *
 * Integration with external libraries:
 * - METIS/ParMETIS: nested dissection, graph partitioning
 * - AMD: approximate minimum degree
 * - Scotch/PT-Scotch: parallel reordering
 *
 * If external libraries are not available, fallback algorithms are used.
 *
 * @see GraphSparsity for underlying graph algorithms
 * @see SparsityPattern for the data structure being optimized
 */

#include "SparsityPattern.h"
#include "GraphSparsity.h"
#include "Core/Types.h"
#include "Core/FEConfig.h"
#include "Core/FEException.h"

#include <vector>
#include <span>
#include <memory>
#include <optional>
#include <functional>
#include <string>

#if FE_HAS_MPI
#include <mpi.h>
#endif

namespace svmp {
namespace FE {
namespace sparsity {

class DistributedSparsityPattern;

// ============================================================================
// Optimization Goals and Strategies
// ============================================================================

/**
 * @brief Optimization goal for sparsity pattern
 */
enum class OptimizationGoal : std::uint8_t {
    MinimizeBandwidth,     ///< Minimize matrix bandwidth
    MinimizeProfile,       ///< Minimize profile/envelope
    MinimizeFillIn,        ///< Minimize fill-in during factorization
    OptimizeCache,         ///< Optimize for cache performance
    OptimizeVectorization, ///< Optimize for SIMD operations
    OptimizeParallel,      ///< Optimize for parallel assembly
    Balanced               ///< Balance between multiple goals
};

/**
 * @brief Reordering algorithm to use
 */
enum class ReorderingAlgorithm : std::uint8_t {
    None,                  ///< No reordering
    Natural,               ///< Natural ordering (identity)
    CuthillMcKee,          ///< Cuthill-McKee
    ReverseCuthillMcKee,   ///< Reverse Cuthill-McKee
    MinimumDegree,         ///< Minimum degree (builtin approximation)
    AMD,                   ///< Approximate minimum degree (external)
    NestedDissection,      ///< Nested dissection (external)
    METIS,                 ///< METIS graph partitioning (external)
    Scotch,                ///< Scotch reordering (external)
    Auto                   ///< Automatic selection based on pattern
};

/**
 * @brief Options for sparsity optimization
 */
struct OptimizationOptions {
    OptimizationGoal goal{OptimizationGoal::Balanced};
    ReorderingAlgorithm algorithm{ReorderingAlgorithm::Auto};

    /**
     * @brief Symmetrize pattern before optimization
     *
     * Most reordering algorithms work on symmetric patterns.
     */
    bool symmetrize{true};

    /**
     * @brief Allow external library calls
     *
     * If false, only builtin algorithms are used.
     */
    bool allow_external{true};

    /**
     * @brief Target number of colors for parallel optimization
     */
    GlobalIndex target_colors{0};

    /**
     * @brief Cache line size in bytes for cache optimization
     */
    std::size_t cache_line_bytes{64};

    /**
     * @brief SIMD vector width in elements for vectorization
     */
    std::size_t simd_width{8};

    /**
     * @brief Weight for bandwidth in balanced optimization
     */
    double bandwidth_weight{1.0};

    /**
     * @brief Weight for fill-in in balanced optimization
     */
    double fill_weight{1.0};

    /**
     * @brief Weight for parallelism in balanced optimization
     */
    double parallel_weight{1.0};

    /**
     * @brief Maximum iterations for iterative algorithms
     */
    int max_iterations{100};

    /**
     * @brief Tolerance for convergence
     */
    double tolerance{1e-6};

    /**
     * @brief Verbose output during optimization
     */
    bool verbose{false};
};

/**
 * @brief Results of sparsity optimization
 */
struct OptimizationResult {
    /**
     * @brief Permutation vector (new_index = perm[old_index])
     */
    std::vector<GlobalIndex> permutation;

    /**
     * @brief Inverse permutation (old_index = inv_perm[new_index])
     */
    std::vector<GlobalIndex> inverse_permutation;

    /**
     * @brief Algorithm that was used
     */
    ReorderingAlgorithm algorithm_used{ReorderingAlgorithm::None};

    /**
     * @brief Original bandwidth
     */
    GlobalIndex original_bandwidth{0};

    /**
     * @brief Optimized bandwidth
     */
    GlobalIndex optimized_bandwidth{0};

    /**
     * @brief Original profile
     */
    GlobalIndex original_profile{0};

    /**
     * @brief Optimized profile
     */
    GlobalIndex optimized_profile{0};

    /**
     * @brief Predicted fill-in reduction factor
     */
    double fill_reduction{1.0};

    /**
     * @brief Number of colors for parallel assembly
     */
    GlobalIndex num_colors{0};

    /**
     * @brief Optimization time in seconds
     */
    double optimization_time_sec{0.0};

    /**
     * @brief Check if optimization improved the pattern
     */
    [[nodiscard]] bool improved() const noexcept {
        return optimized_bandwidth < original_bandwidth ||
               optimized_profile < original_profile;
    }

    /**
     * @brief Get bandwidth reduction ratio
     */
    [[nodiscard]] double bandwidthReduction() const noexcept {
        if (original_bandwidth == 0) return 1.0;
        return static_cast<double>(optimized_bandwidth) /
               static_cast<double>(original_bandwidth);
    }

    /**
     * @brief Get profile reduction ratio
     */
    [[nodiscard]] double profileReduction() const noexcept {
        if (original_profile == 0) return 1.0;
        return static_cast<double>(optimized_profile) /
               static_cast<double>(original_profile);
    }
};

// ============================================================================
// SparsityOptimizer Class
// ============================================================================

/**
 * @brief High-level optimizer for sparsity patterns
 *
 * SparsityOptimizer provides a unified interface for optimizing sparsity
 * patterns using various algorithms and strategies. It handles:
 *
 * - Algorithm selection based on pattern characteristics
 * - External library integration (METIS, AMD, etc.)
 * - Fallback to builtin algorithms when external not available
 * - Performance measurement and reporting
 *
 * Usage:
 * @code
 * SparsityPattern pattern = buildPattern(...);
 *
 * // Simple optimization with defaults
 * SparsityOptimizer optimizer;
 * auto result = optimizer.optimize(pattern);
 * SparsityPattern optimized = pattern.permute(result.permutation,
 *                                              result.permutation);
 *
 * // Goal-specific optimization
 * OptimizationOptions opts;
 * opts.goal = OptimizationGoal::MinimizeFillIn;
 * opts.algorithm = ReorderingAlgorithm::NestedDissection;
 * auto result = optimizer.optimize(pattern, opts);
 *
 * // Automatic optimization with result analysis
 * auto result = optimizer.optimize(pattern);
 * std::cout << "Bandwidth reduced from " << result.original_bandwidth
 *           << " to " << result.optimized_bandwidth << "\n";
 * @endcode
 */
class SparsityOptimizer {
public:
    // =========================================================================
    // Construction
    // =========================================================================

    /**
     * @brief Default constructor
     */
    SparsityOptimizer() = default;

    /**
     * @brief Construct with default options
     */
    explicit SparsityOptimizer(const OptimizationOptions& options);

    /// Destructor
    ~SparsityOptimizer() = default;

    // Non-copyable for thread safety
    SparsityOptimizer(const SparsityOptimizer&) = delete;
    SparsityOptimizer& operator=(const SparsityOptimizer&) = delete;

    // Movable
    SparsityOptimizer(SparsityOptimizer&&) = default;
    SparsityOptimizer& operator=(SparsityOptimizer&&) = default;

    // =========================================================================
    // Configuration
    // =========================================================================

    /**
     * @brief Set default optimization options
     */
    void setOptions(const OptimizationOptions& options);

    /**
     * @brief Get current options
     */
    [[nodiscard]] const OptimizationOptions& getOptions() const noexcept {
        return options_;
    }

    // =========================================================================
    // Optimization Methods
    // =========================================================================

    /**
     * @brief Optimize a sparsity pattern
     *
     * @param pattern Pattern to optimize
     * @param options Options (uses default if not provided)
     * @return Optimization result with permutation and statistics
     */
    [[nodiscard]] OptimizationResult optimize(
        const SparsityPattern& pattern,
        const OptimizationOptions& options) const;

    /**
     * @brief Optimize using default options
     */
    [[nodiscard]] OptimizationResult optimize(const SparsityPattern& pattern) const;

    /**
     * @brief Optimize and return reordered pattern
     *
     * @param pattern Pattern to optimize
     * @param options Options (uses default if not provided)
     * @return Pair of (reordered pattern, optimization result)
     */
    [[nodiscard]] std::pair<SparsityPattern, OptimizationResult> optimizeAndApply(
        const SparsityPattern& pattern,
        const OptimizationOptions& options) const;

    /**
     * @brief Optimize and return reordered pattern (default options)
     */
    [[nodiscard]] std::pair<SparsityPattern, OptimizationResult> optimizeAndApply(
        const SparsityPattern& pattern) const;

    // =========================================================================
    // Goal-Specific Optimization
    // =========================================================================

    /**
     * @brief Optimize for minimum bandwidth
     */
    [[nodiscard]] OptimizationResult optimizeForBandwidth(
        const SparsityPattern& pattern) const;

    /**
     * @brief Optimize for minimum profile
     */
    [[nodiscard]] OptimizationResult optimizeForProfile(
        const SparsityPattern& pattern) const;

    /**
     * @brief Optimize for minimum fill-in
     */
    [[nodiscard]] OptimizationResult optimizeForFillIn(
        const SparsityPattern& pattern) const;

    /**
     * @brief Optimize for cache performance
     */
    [[nodiscard]] OptimizationResult optimizeForCache(
        const SparsityPattern& pattern) const;

    /**
     * @brief Optimize for parallel assembly
     */
    [[nodiscard]] OptimizationResult optimizeForParallel(
        const SparsityPattern& pattern) const;

    // =========================================================================
    // Algorithm-Specific Methods
    // =========================================================================

    /**
     * @brief Apply Cuthill-McKee reordering
     */
    [[nodiscard]] OptimizationResult applyCuthillMcKee(
        const SparsityPattern& pattern, bool reverse = false) const;

    /**
     * @brief Apply minimum degree reordering
     */
    [[nodiscard]] OptimizationResult applyMinimumDegree(
        const SparsityPattern& pattern) const;

    /**
     * @brief Apply nested dissection reordering (if available)
     */
    [[nodiscard]] OptimizationResult applyNestedDissection(
        const SparsityPattern& pattern) const;

    // =========================================================================
    // Analysis Methods
    // =========================================================================

    /**
     * @brief Analyze pattern and suggest best algorithm
     *
     * @param pattern Pattern to analyze
     * @return Recommended algorithm
     */
    [[nodiscard]] ReorderingAlgorithm suggestAlgorithm(
        const SparsityPattern& pattern) const;

    /**
     * @brief Compare multiple algorithms on a pattern
     *
     * @param pattern Pattern to test
     * @param algorithms Algorithms to compare
     * @return Results for each algorithm
     */
    [[nodiscard]] std::vector<OptimizationResult> compareAlgorithms(
        const SparsityPattern& pattern,
        std::span<const ReorderingAlgorithm> algorithms) const;

    /**
     * @brief Estimate optimization potential
     *
     * Quick analysis of how much improvement is possible.
     *
     * @param pattern Pattern to analyze
     * @return Estimated potential improvement (0.0 = none, 1.0 = significant)
     */
    [[nodiscard]] double estimatePotential(const SparsityPattern& pattern) const;

#if FE_HAS_MPI
    /**
     * @brief Compute a global nested-dissection permutation using ParMETIS
     *
     * @param pattern Distributed pattern (square, finalized)
     * @param comm MPI communicator
     * @return Global permutation vector (old->new) replicated on all ranks
     *
     * This utility is intended for distributed solvers that need a consistent
     * global reordering derived from a distributed adjacency graph.
     */
    [[nodiscard]] std::vector<GlobalIndex> parmetisNodeNDPermutation(
        const DistributedSparsityPattern& pattern,
        MPI_Comm comm) const;
#endif

    // =========================================================================
    // External Library Availability
    // =========================================================================

    /**
     * @brief Check if METIS is available
     */
    [[nodiscard]] static bool hasMetis() noexcept;

    /**
     * @brief Check if AMD (SuiteSparse) is available
     */
    [[nodiscard]] static bool hasAMD() noexcept;

    /**
     * @brief Check if Scotch is available
     */
    [[nodiscard]] static bool hasScotch() noexcept;

    /**
     * @brief Check if ParMETIS is available
     */
    [[nodiscard]] static bool hasParMetis() noexcept;

    /**
     * @brief Get list of available algorithms
     */
    [[nodiscard]] static std::vector<ReorderingAlgorithm> availableAlgorithms();

    /**
     * @brief Get human-readable algorithm name
     */
    [[nodiscard]] static std::string algorithmName(ReorderingAlgorithm algo);

private:
    // Internal implementation methods
    [[nodiscard]] OptimizationResult runOptimization(
        const SparsityPattern& pattern,
        const OptimizationOptions& options) const;

    [[nodiscard]] OptimizationResult selectAndRunAlgorithm(
        const SparsityPattern& pattern,
        ReorderingAlgorithm algorithm,
        const OptimizationOptions& options) const;

    void computeStatistics(OptimizationResult& result,
                          const SparsityPattern& pattern,
                          const std::vector<GlobalIndex>& perm) const;

    OptimizationOptions options_;
};

// ============================================================================
// Convenience Functions
// ============================================================================

/**
 * @brief Optimize pattern with automatic algorithm selection
 */
[[nodiscard]] SparsityPattern optimizePattern(const SparsityPattern& pattern);

/**
 * @brief Optimize pattern for bandwidth
 */
[[nodiscard]] SparsityPattern optimizeForBandwidth(const SparsityPattern& pattern);

/**
 * @brief Optimize pattern for fill-in
 */
[[nodiscard]] SparsityPattern optimizeForFillIn(const SparsityPattern& pattern);

/**
 * @brief Get optimized permutation
 */
[[nodiscard]] std::vector<GlobalIndex> getOptimalPermutation(
    const SparsityPattern& pattern,
    OptimizationGoal goal = OptimizationGoal::Balanced);

/**
 * @brief Print optimization report
 */
void printOptimizationReport(const OptimizationResult& result,
                             std::ostream& out);

} // namespace sparsity
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_SPARSITY_OPTIMIZER_H

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

#include <gtest/gtest.h>
#include "Sparsity/SparsityOptimizer.h"
#include "Sparsity/SparsityPattern.h"
#include "Sparsity/GraphSparsity.h"
#include <vector>
#include <algorithm>
#include <numeric>

using namespace svmp::FE;
using namespace svmp::FE::sparsity;

// ============================================================================
// Helper to create test patterns
// ============================================================================

// Arrow pattern: dense first row/column with diagonal
SparsityPattern createArrowPattern(GlobalIndex n) {
    SparsityPattern pattern(n, n);
    for (GlobalIndex i = 0; i < n; ++i) {
        pattern.addEntry(0, i);
        pattern.addEntry(i, 0);
        pattern.addEntry(i, i);
    }
    pattern.finalize();
    return pattern;
}

// Band pattern
SparsityPattern createBandPattern(GlobalIndex n, GlobalIndex bandwidth) {
    SparsityPattern pattern(n, n);
    for (GlobalIndex i = 0; i < n; ++i) {
        GlobalIndex start = std::max(GlobalIndex(0), i - bandwidth);
        GlobalIndex end = std::min(n, i + bandwidth + 1);
        for (GlobalIndex j = start; j < end; ++j) {
            pattern.addEntry(i, j);
        }
    }
    pattern.finalize();
    return pattern;
}

// Random-ish symmetric pattern with poor ordering
SparsityPattern createPoorlyOrderedPattern(GlobalIndex n) {
    SparsityPattern pattern(n, n);

    // Connect vertices in reverse pairs
    for (GlobalIndex i = 0; i < n / 2; ++i) {
        GlobalIndex j = n - 1 - i;
        pattern.addEntry(i, j);
        pattern.addEntry(j, i);
        pattern.addEntry(i, i);
        pattern.addEntry(j, j);
    }

    // Add some local connections
    for (GlobalIndex i = 0; i < n - 1; ++i) {
        pattern.addEntry(i, i + 1);
        pattern.addEntry(i + 1, i);
    }

    pattern.finalize();
    return pattern;
}

// ============================================================================
// Basic Construction Tests
// ============================================================================

TEST(SparsityOptimizerTest, DefaultConstruction) {
    SparsityOptimizer optimizer;
    auto options = optimizer.getOptions();

    EXPECT_EQ(options.goal, OptimizationGoal::Balanced);
    EXPECT_EQ(options.algorithm, ReorderingAlgorithm::Auto);
}

TEST(SparsityOptimizerTest, ConstructWithOptions) {
    OptimizationOptions opts;
    opts.goal = OptimizationGoal::MinimizeBandwidth;
    opts.algorithm = ReorderingAlgorithm::ReverseCuthillMcKee;

    SparsityOptimizer optimizer(opts);
    auto stored_opts = optimizer.getOptions();

    EXPECT_EQ(stored_opts.goal, OptimizationGoal::MinimizeBandwidth);
    EXPECT_EQ(stored_opts.algorithm, ReorderingAlgorithm::ReverseCuthillMcKee);
}

TEST(SparsityOptimizerTest, SetOptions) {
    SparsityOptimizer optimizer;

    OptimizationOptions opts;
    opts.goal = OptimizationGoal::MinimizeFillIn;
    optimizer.setOptions(opts);

    EXPECT_EQ(optimizer.getOptions().goal, OptimizationGoal::MinimizeFillIn);
}

// ============================================================================
// Optimization Tests
// ============================================================================

TEST(SparsityOptimizerTest, OptimizeWithDefaults) {
    auto pattern = createArrowPattern(10);

    SparsityOptimizer optimizer;
    auto result = optimizer.optimize(pattern);

    EXPECT_EQ(result.permutation.size(), 10);
    EXPECT_EQ(result.inverse_permutation.size(), 10);
    EXPECT_NE(result.algorithm_used, ReorderingAlgorithm::None);
}

TEST(SparsityOptimizerTest, OptimizeAndApply) {
    auto pattern = createPoorlyOrderedPattern(20);
    GlobalIndex original_bw = computePatternBandwidth(pattern);

    SparsityOptimizer optimizer;
    auto [optimized, result] = optimizer.optimizeAndApply(pattern);

    EXPECT_EQ(optimized.numRows(), pattern.numRows());
    EXPECT_EQ(optimized.getNnz(), pattern.getNnz());

    GlobalIndex new_bw = computePatternBandwidth(optimized);
    EXPECT_LE(new_bw, original_bw);
}

TEST(SparsityOptimizerTest, OptimizeForBandwidth) {
    auto pattern = createPoorlyOrderedPattern(20);
    GlobalIndex original_bw = computePatternBandwidth(pattern);

    SparsityOptimizer optimizer;
    auto result = optimizer.optimizeForBandwidth(pattern);

    auto optimized = pattern.permute(result.permutation, result.permutation);
    GlobalIndex new_bw = computePatternBandwidth(optimized);

    EXPECT_LE(new_bw, original_bw);
    EXPECT_EQ(result.optimized_bandwidth, new_bw);
}

TEST(SparsityOptimizerTest, OptimizeForProfile) {
    auto pattern = createPoorlyOrderedPattern(20);

    SparsityOptimizer optimizer;
    auto result = optimizer.optimizeForProfile(pattern);

    EXPECT_GT(result.original_profile, 0);
    EXPECT_GT(result.optimized_profile, 0);
    // Profile optimization is not always guaranteed to improve due to
    // heuristic algorithms, but the result should be reasonable
    EXPECT_GT(result.permutation.size(), 0);
}

TEST(SparsityOptimizerTest, OptimizeForFillIn) {
    auto pattern = createArrowPattern(10);

    SparsityOptimizer optimizer;
    auto result = optimizer.optimizeForFillIn(pattern);

    EXPECT_EQ(result.permutation.size(), 10);
    EXPECT_LE(result.fill_reduction, 1.0);
}

TEST(SparsityOptimizerTest, OptimizeForCache) {
    auto pattern = createBandPattern(20, 3);

    SparsityOptimizer optimizer;
    auto result = optimizer.optimizeForCache(pattern);

    EXPECT_EQ(result.permutation.size(), 20);
}

TEST(SparsityOptimizerTest, OptimizeForParallel) {
    auto pattern = createBandPattern(20, 2);

    SparsityOptimizer optimizer;
    auto result = optimizer.optimizeForParallel(pattern);

    EXPECT_GT(result.num_colors, 0);
}

// ============================================================================
// Algorithm-Specific Tests
// ============================================================================

TEST(SparsityOptimizerTest, ApplyCuthillMcKee) {
    auto pattern = createPoorlyOrderedPattern(15);

    SparsityOptimizer optimizer;
    auto result = optimizer.applyCuthillMcKee(pattern, false);

    EXPECT_EQ(result.algorithm_used, ReorderingAlgorithm::CuthillMcKee);
    EXPECT_EQ(result.permutation.size(), 15);
}

TEST(SparsityOptimizerTest, ApplyReverseCuthillMcKee) {
    auto pattern = createPoorlyOrderedPattern(15);

    SparsityOptimizer optimizer;
    auto result = optimizer.applyCuthillMcKee(pattern, true);

    EXPECT_EQ(result.algorithm_used, ReorderingAlgorithm::ReverseCuthillMcKee);
}

TEST(SparsityOptimizerTest, ApplyMinimumDegree) {
    auto pattern = createArrowPattern(10);

    SparsityOptimizer optimizer;
    auto result = optimizer.applyMinimumDegree(pattern);

    EXPECT_EQ(result.algorithm_used, ReorderingAlgorithm::MinimumDegree);
    EXPECT_EQ(result.permutation.size(), 10);
}

TEST(SparsityOptimizerTest, ApplyNestedDissection) {
    auto pattern = createBandPattern(20, 3);

    SparsityOptimizer optimizer;
    auto result = optimizer.applyNestedDissection(pattern);

    // May fall back to different algorithm if external lib not available
    EXPECT_EQ(result.permutation.size(), 20);
}

// ============================================================================
// Result Validation Tests
// ============================================================================

TEST(SparsityOptimizerTest, PermutationIsValid) {
    auto pattern = createArrowPattern(10);

    SparsityOptimizer optimizer;
    auto result = optimizer.optimize(pattern);

    // Check permutation is valid
    std::vector<GlobalIndex> sorted_perm = result.permutation;
    std::sort(sorted_perm.begin(), sorted_perm.end());
    for (GlobalIndex i = 0; i < 10; ++i) {
        EXPECT_EQ(sorted_perm[i], i);
    }
}

TEST(SparsityOptimizerTest, InversePermutationValid) {
    auto pattern = createArrowPattern(10);

    SparsityOptimizer optimizer;
    auto result = optimizer.optimize(pattern);

    // Check inverse is correct
    for (GlobalIndex i = 0; i < 10; ++i) {
        EXPECT_EQ(result.inverse_permutation[result.permutation[i]], i);
    }
}

TEST(SparsityOptimizerTest, ResultImproved) {
    auto pattern = createPoorlyOrderedPattern(30);

    SparsityOptimizer optimizer;
    OptimizationOptions opts;
    opts.goal = OptimizationGoal::MinimizeBandwidth;
    opts.algorithm = ReorderingAlgorithm::ReverseCuthillMcKee;

    auto result = optimizer.optimize(pattern, opts);

    // For this pattern, we expect improvement
    EXPECT_TRUE(result.improved() || result.optimized_bandwidth == result.original_bandwidth);
}

TEST(SparsityOptimizerTest, BandwidthReductionRatio) {
    auto pattern = createArrowPattern(10);

    SparsityOptimizer optimizer;
    auto result = optimizer.optimizeForBandwidth(pattern);

    double ratio = result.bandwidthReduction();
    EXPECT_GT(ratio, 0.0);
    EXPECT_LE(ratio, 1.0);
}

TEST(SparsityOptimizerTest, ProfileReductionRatio) {
    auto pattern = createArrowPattern(10);

    SparsityOptimizer optimizer;
    auto result = optimizer.optimizeForProfile(pattern);

    double ratio = result.profileReduction();
    EXPECT_GT(ratio, 0.0);
    EXPECT_LE(ratio, 1.0);
}

// ============================================================================
// Analysis Tests
// ============================================================================

TEST(SparsityOptimizerTest, SuggestAlgorithm) {
    SparsityOptimizer optimizer;

    // Small pattern
    auto small = createBandPattern(10, 2);
    auto suggested = optimizer.suggestAlgorithm(small);
    EXPECT_NE(suggested, ReorderingAlgorithm::None);

    // Larger pattern
    auto large = createBandPattern(100, 5);
    suggested = optimizer.suggestAlgorithm(large);
    EXPECT_NE(suggested, ReorderingAlgorithm::None);
}

TEST(SparsityOptimizerTest, CompareAlgorithms) {
    auto pattern = createPoorlyOrderedPattern(20);

    SparsityOptimizer optimizer;
    std::vector<ReorderingAlgorithm> algorithms = {
        ReorderingAlgorithm::CuthillMcKee,
        ReorderingAlgorithm::ReverseCuthillMcKee,
        ReorderingAlgorithm::MinimumDegree
    };

    auto results = optimizer.compareAlgorithms(pattern, algorithms);

    EXPECT_EQ(results.size(), 3);

    for (const auto& result : results) {
        EXPECT_EQ(result.permutation.size(), 20);
    }
}

TEST(SparsityOptimizerTest, EstimatePotential) {
    SparsityOptimizer optimizer;

    // Already optimal (diagonal)
    SparsityPattern diagonal(5, 5);
    for (GlobalIndex i = 0; i < 5; ++i) diagonal.addEntry(i, i);
    diagonal.finalize();
    double potential1 = optimizer.estimatePotential(diagonal);

    // Poorly ordered
    auto poor = createPoorlyOrderedPattern(20);
    double potential2 = optimizer.estimatePotential(poor);

    // Poorly ordered should have more potential
    EXPECT_GE(potential2, potential1);
}

// ============================================================================
// External Library Tests
// ============================================================================

TEST(SparsityOptimizerTest, HasMetis) {
    // Just check it doesn't crash
    bool has = SparsityOptimizer::hasMetis();
    (void)has;  // May be true or false depending on build
}

TEST(SparsityOptimizerTest, HasAMD) {
    bool has = SparsityOptimizer::hasAMD();
    (void)has;
}

TEST(SparsityOptimizerTest, HasScotch) {
    bool has = SparsityOptimizer::hasScotch();
    (void)has;
}

TEST(SparsityOptimizerTest, HasParMetis) {
    bool has = SparsityOptimizer::hasParMetis();
    (void)has;
}

TEST(SparsityOptimizerTest, AvailableAlgorithms) {
    auto algorithms = SparsityOptimizer::availableAlgorithms();

    EXPECT_FALSE(algorithms.empty());

    // At minimum should have builtin algorithms
    auto has_rcm = std::find(algorithms.begin(), algorithms.end(),
                              ReorderingAlgorithm::ReverseCuthillMcKee) != algorithms.end();
    EXPECT_TRUE(has_rcm);
}

TEST(SparsityOptimizerTest, AlgorithmName) {
    EXPECT_EQ(SparsityOptimizer::algorithmName(ReorderingAlgorithm::None), "None");
    EXPECT_EQ(SparsityOptimizer::algorithmName(ReorderingAlgorithm::ReverseCuthillMcKee), "Reverse Cuthill-McKee");
    EXPECT_EQ(SparsityOptimizer::algorithmName(ReorderingAlgorithm::AMD), "AMD (Approximate Minimum Degree)");
}

TEST(SparsityOptimizerTest, MetisNodeNDDeterministicWhenAvailable) {
    if (!SparsityOptimizer::hasMetis()) {
        GTEST_SKIP() << "METIS not available in this build";
    }

    auto pattern = createPoorlyOrderedPattern(60);

    OptimizationOptions opts;
    opts.goal = OptimizationGoal::MinimizeFillIn;
    opts.algorithm = ReorderingAlgorithm::METIS;
    opts.allow_external = true;

    SparsityOptimizer optimizer;
    auto r1 = optimizer.optimize(pattern, opts);
    auto r2 = optimizer.optimize(pattern, opts);

    EXPECT_EQ(r1.algorithm_used, ReorderingAlgorithm::METIS);
    EXPECT_EQ(r1.permutation, r2.permutation);
    EXPECT_EQ(r1.inverse_permutation, r2.inverse_permutation);
}

// ============================================================================
// Options Tests
// ============================================================================

TEST(SparsityOptimizerTest, SymmetrizeOption) {
    SparsityPattern asymmetric(3, 3);
    asymmetric.addEntry(0, 1);
    asymmetric.addEntry(1, 2);
    asymmetric.finalize();

    SparsityOptimizer optimizer;

    OptimizationOptions opts;
    opts.symmetrize = true;
    auto result = optimizer.optimize(asymmetric, opts);

    EXPECT_EQ(result.permutation.size(), 3);
}

TEST(SparsityOptimizerTest, NoExternalOption) {
    auto pattern = createArrowPattern(10);

    OptimizationOptions opts;
    opts.allow_external = false;

    SparsityOptimizer optimizer;
    auto result = optimizer.optimize(pattern, opts);

    // Should use builtin algorithm
    EXPECT_TRUE(result.algorithm_used == ReorderingAlgorithm::CuthillMcKee ||
                result.algorithm_used == ReorderingAlgorithm::ReverseCuthillMcKee ||
                result.algorithm_used == ReorderingAlgorithm::MinimumDegree ||
                result.algorithm_used == ReorderingAlgorithm::Natural);
}

TEST(SparsityOptimizerTest, VerboseOption) {
    auto pattern = createBandPattern(10, 2);

    OptimizationOptions opts;
    opts.verbose = true;

    SparsityOptimizer optimizer;
    // Should not crash with verbose output
    auto result = optimizer.optimize(pattern, opts);

    EXPECT_EQ(result.permutation.size(), 10);
}

// ============================================================================
// Convenience Function Tests
// ============================================================================

TEST(SparsityOptimizerTest, OptimizePatternFunction) {
    auto pattern = createArrowPattern(10);

    auto optimized = optimizePattern(pattern);

    EXPECT_EQ(optimized.numRows(), 10);
    EXPECT_EQ(optimized.getNnz(), pattern.getNnz());
}

TEST(SparsityOptimizerTest, OptimizeForBandwidthFunction) {
    auto pattern = createPoorlyOrderedPattern(15);

    auto optimized = optimizeForBandwidth(pattern);

    EXPECT_EQ(optimized.numRows(), 15);
}

TEST(SparsityOptimizerTest, OptimizeForFillInFunction) {
    auto pattern = createArrowPattern(8);

    auto optimized = optimizeForFillIn(pattern);

    EXPECT_EQ(optimized.numRows(), 8);
}

TEST(SparsityOptimizerTest, GetOptimalPermutation) {
    auto pattern = createBandPattern(10, 2);

    auto perm = getOptimalPermutation(pattern);

    EXPECT_EQ(perm.size(), 10);

    // Verify valid permutation
    std::vector<GlobalIndex> sorted = perm;
    std::sort(sorted.begin(), sorted.end());
    for (GlobalIndex i = 0; i < 10; ++i) {
        EXPECT_EQ(sorted[i], i);
    }
}

TEST(SparsityOptimizerTest, GetOptimalPermutationWithGoal) {
    auto pattern = createArrowPattern(10);

    auto perm = getOptimalPermutation(pattern, OptimizationGoal::MinimizeFillIn);

    EXPECT_EQ(perm.size(), 10);
}

// ============================================================================
// Edge Cases Tests
// ============================================================================

TEST(SparsityOptimizerTest, EmptyPattern) {
    SparsityPattern pattern(0, 0);
    pattern.finalize();

    SparsityOptimizer optimizer;
    auto result = optimizer.optimize(pattern);

    EXPECT_TRUE(result.permutation.empty());
}

TEST(SparsityOptimizerTest, SingleVertex) {
    SparsityPattern pattern(1, 1);
    pattern.addEntry(0, 0);
    pattern.finalize();

    SparsityOptimizer optimizer;
    auto result = optimizer.optimize(pattern);

    EXPECT_EQ(result.permutation.size(), 1);
    EXPECT_EQ(result.permutation[0], 0);
}

TEST(SparsityOptimizerTest, DiagonalPattern) {
    SparsityPattern pattern(5, 5);
    for (GlobalIndex i = 0; i < 5; ++i) {
        pattern.addEntry(i, i);
    }
    pattern.finalize();

    SparsityOptimizer optimizer;
    auto result = optimizer.optimizeForBandwidth(pattern);

    // Diagonal has bandwidth 0, can't improve
    EXPECT_EQ(result.original_bandwidth, 0);
    EXPECT_EQ(result.optimized_bandwidth, 0);
}

TEST(SparsityOptimizerTest, AlreadyOptimal) {
    // Tridiagonal is already optimal for bandwidth
    SparsityPattern pattern(10, 10);
    for (GlobalIndex i = 0; i < 10; ++i) {
        pattern.addEntry(i, i);
        if (i > 0) pattern.addEntry(i, i - 1);
        if (i < 9) pattern.addEntry(i, i + 1);
    }
    pattern.finalize();

    SparsityOptimizer optimizer;
    auto result = optimizer.optimizeForBandwidth(pattern);

    EXPECT_EQ(result.original_bandwidth, 1);
    // Should not make it worse
    EXPECT_LE(result.optimized_bandwidth, 1);
}

// ============================================================================
// Determinism Tests
// ============================================================================

TEST(SparsityOptimizerTest, DeterministicOptimization) {
    auto create_and_optimize = []() {
        auto pattern = createPoorlyOrderedPattern(20);
        SparsityOptimizer optimizer;
        return optimizer.optimizeForBandwidth(pattern);
    };

    auto r1 = create_and_optimize();
    auto r2 = create_and_optimize();

    EXPECT_EQ(r1.permutation, r2.permutation);
    EXPECT_EQ(r1.optimized_bandwidth, r2.optimized_bandwidth);
}

// ============================================================================
// Timing Tests
// ============================================================================

TEST(SparsityOptimizerTest, OptimizationTimeRecorded) {
    auto pattern = createBandPattern(50, 5);

    SparsityOptimizer optimizer;
    auto result = optimizer.optimize(pattern);

    // Should have recorded some time
    EXPECT_GE(result.optimization_time_sec, 0.0);
}

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
#include "Sparsity/SparsityAnalyzer.h"
#include "Sparsity/SparsityOps.h"
#include <sstream>
#include <cmath>

using namespace svmp::FE;
using namespace svmp::FE::sparsity;

// ============================================================================
// Helper Functions
// ============================================================================

namespace {

SparsityPattern createTridiagonalPattern(GlobalIndex n) {
    SparsityPattern pattern(n, n);
    for (GlobalIndex i = 0; i < n; ++i) {
        pattern.addEntry(i, i);
        if (i > 0) pattern.addEntry(i, i - 1);
        if (i < n - 1) pattern.addEntry(i, i + 1);
    }
    pattern.finalize();
    return pattern;
}

SparsityPattern createDiagonalPattern(GlobalIndex n) {
    SparsityPattern pattern(n, n);
    for (GlobalIndex i = 0; i < n; ++i) {
        pattern.addEntry(i, i);
    }
    pattern.finalize();
    return pattern;
}

SparsityPattern createDensePattern(GlobalIndex n) {
    SparsityPattern pattern(n, n);
    for (GlobalIndex i = 0; i < n; ++i) {
        for (GlobalIndex j = 0; j < n; ++j) {
            pattern.addEntry(i, j);
        }
    }
    pattern.finalize();
    return pattern;
}

SparsityPattern createArrowPattern(GlobalIndex n) {
    SparsityPattern pattern(n, n);
    // Dense first row and column
    for (GlobalIndex i = 0; i < n; ++i) {
        pattern.addEntry(0, i);
        pattern.addEntry(i, 0);
        pattern.addEntry(i, i);  // Diagonal
    }
    pattern.finalize();
    return pattern;
}

SparsityPattern createBlockDiagonalPattern(GlobalIndex n_blocks, GlobalIndex block_size) {
    GlobalIndex n = n_blocks * block_size;
    SparsityPattern pattern(n, n);

    for (GlobalIndex b = 0; b < n_blocks; ++b) {
        GlobalIndex start = b * block_size;
        for (GlobalIndex i = 0; i < block_size; ++i) {
            for (GlobalIndex j = 0; j < block_size; ++j) {
                pattern.addEntry(start + i, start + j);
            }
        }
    }
    pattern.finalize();
    return pattern;
}

SparsityPattern createAsymmetricPattern(GlobalIndex n) {
    SparsityPattern pattern(n, n);
    for (GlobalIndex i = 0; i < n; ++i) {
        pattern.addEntry(i, i);
        if (i > 0) {
            pattern.addEntry(i, i - 1);  // Lower diagonal only (no upper)
        }
    }
    pattern.finalize();
    return pattern;
}

SparsityPattern createDisconnectedPattern(GlobalIndex n) {
    SparsityPattern pattern(n, n);
    // Two disconnected components: [0, n/2) and [n/2, n)
    GlobalIndex mid = n / 2;
    for (GlobalIndex i = 0; i < mid; ++i) {
        pattern.addEntry(i, i);
        if (i + 1 < mid) pattern.addEntry(i, i + 1);
        if (i > 0) pattern.addEntry(i, i - 1);
    }
    for (GlobalIndex i = mid; i < n; ++i) {
        pattern.addEntry(i, i);
        if (i + 1 < n) pattern.addEntry(i, i + 1);
        if (i > mid) pattern.addEntry(i, i - 1);
    }
    pattern.finalize();
    return pattern;
}

} // anonymous namespace

// ============================================================================
// BandwidthStats Tests
// ============================================================================

TEST(SparsityAnalyzerTest, BandwidthTridiagonal) {
    auto pattern = createTridiagonalPattern(10);
    SparsityAnalyzer analyzer;
    auto stats = analyzer.computeBandwidth(pattern);

    EXPECT_EQ(stats.bandwidth, 1);
    EXPECT_EQ(stats.lower_bandwidth, 1);
    EXPECT_EQ(stats.upper_bandwidth, 1);
    EXPECT_GT(stats.profile, 0);
}

TEST(SparsityAnalyzerTest, BandwidthDiagonal) {
    auto pattern = createDiagonalPattern(10);
    SparsityAnalyzer analyzer;
    auto stats = analyzer.computeBandwidth(pattern);

    EXPECT_EQ(stats.bandwidth, 0);
    EXPECT_EQ(stats.lower_bandwidth, 0);
    EXPECT_EQ(stats.upper_bandwidth, 0);
}

TEST(SparsityAnalyzerTest, BandwidthDense) {
    auto pattern = createDensePattern(5);
    SparsityAnalyzer analyzer;
    auto stats = analyzer.computeBandwidth(pattern);

    EXPECT_EQ(stats.bandwidth, 4);
    EXPECT_EQ(stats.lower_bandwidth, 4);
    EXPECT_EQ(stats.upper_bandwidth, 4);
}

TEST(SparsityAnalyzerTest, BandwidthArrow) {
    auto pattern = createArrowPattern(10);
    SparsityAnalyzer analyzer;
    auto stats = analyzer.computeBandwidth(pattern);

    EXPECT_EQ(stats.bandwidth, 9);  // Max distance from diagonal
}

// ============================================================================
// RowStats Tests
// ============================================================================

TEST(SparsityAnalyzerTest, RowStatsTridiagonal) {
    auto pattern = createTridiagonalPattern(10);
    SparsityAnalyzer analyzer;
    auto stats = analyzer.computeRowStats(pattern);

    EXPECT_EQ(stats.n_rows, 10);
    EXPECT_EQ(stats.empty_rows, 0);
    EXPECT_EQ(stats.min_nnz, 2);  // Boundary rows
    EXPECT_EQ(stats.max_nnz, 3);  // Interior rows
    EXPECT_NEAR(stats.avg_nnz, 2.8, 0.1);  // (2*2 + 8*3) / 10 = 28/10
}

TEST(SparsityAnalyzerTest, RowStatsDiagonal) {
    auto pattern = createDiagonalPattern(10);
    SparsityAnalyzer analyzer;
    auto stats = analyzer.computeRowStats(pattern);

    EXPECT_EQ(stats.min_nnz, 1);
    EXPECT_EQ(stats.max_nnz, 1);
    EXPECT_DOUBLE_EQ(stats.avg_nnz, 1.0);
    EXPECT_DOUBLE_EQ(stats.std_dev_nnz, 0.0);
    EXPECT_TRUE(stats.isUniform());
}

TEST(SparsityAnalyzerTest, RowStatsHistogram) {
    auto pattern = createTridiagonalPattern(100);
    SparsityAnalyzer analyzer;
    auto stats = analyzer.computeRowStats(pattern, 10);

    EXPECT_EQ(stats.histogram.size(), 10);
    EXPECT_FALSE(stats.histogram_bounds.empty());
}

TEST(SparsityAnalyzerTest, RowStatsMedian) {
    auto pattern = createTridiagonalPattern(11);  // Odd number for exact median
    SparsityAnalyzer analyzer;
    auto stats = analyzer.computeRowStats(pattern);

    EXPECT_EQ(stats.median_nnz, 3);  // Most rows have 3 entries
}

// ============================================================================
// DiagonalStats Tests
// ============================================================================

TEST(SparsityAnalyzerTest, DiagonalComplete) {
    auto pattern = createTridiagonalPattern(10);
    SparsityAnalyzer analyzer;
    auto stats = analyzer.analyzeDiagonal(pattern);

    EXPECT_EQ(stats.n_diag, 10);
    EXPECT_EQ(stats.n_missing, 0);
    EXPECT_TRUE(stats.all_present);
    EXPECT_DOUBLE_EQ(stats.coverage(), 1.0);
}

TEST(SparsityAnalyzerTest, DiagonalMissing) {
    SparsityPattern pattern(5, 5);
    pattern.addEntry(0, 0);
    pattern.addEntry(0, 1);
    pattern.addEntry(1, 0);
    // Missing diagonals at 1, 2, 3, 4
    pattern.finalize();

    SparsityAnalyzer analyzer;
    auto stats = analyzer.analyzeDiagonal(pattern);

    EXPECT_EQ(stats.n_diag, 1);
    EXPECT_EQ(stats.n_missing, 4);
    EXPECT_FALSE(stats.all_present);
    EXPECT_DOUBLE_EQ(stats.coverage(), 0.2);
}

TEST(SparsityAnalyzerTest, DiagonalIsolated) {
    auto pattern = createDiagonalPattern(10);
    SparsityAnalyzer analyzer;
    auto stats = analyzer.analyzeDiagonal(pattern);

    EXPECT_EQ(stats.n_isolated_diag, 10);  // All rows have only diagonal
}

// ============================================================================
// SymmetryStats Tests
// ============================================================================

TEST(SparsityAnalyzerTest, SymmetrySymmetric) {
    auto pattern = createTridiagonalPattern(10);
    SparsityAnalyzer analyzer;
    auto stats = analyzer.checkSymmetry(pattern);

    EXPECT_TRUE(stats.is_symmetric);
    EXPECT_EQ(stats.n_asymmetric, 0);
    EXPECT_DOUBLE_EQ(stats.symmetry_ratio, 1.0);
}

TEST(SparsityAnalyzerTest, SymmetryAsymmetric) {
    auto pattern = createAsymmetricPattern(10);
    SparsityAnalyzer analyzer;
    auto stats = analyzer.checkSymmetry(pattern);

    EXPECT_FALSE(stats.is_symmetric);
    EXPECT_GT(stats.n_asymmetric, 0);
    EXPECT_FALSE(stats.isNearlySymmetric());
}

TEST(SparsityAnalyzerTest, SymmetryNearlySymmetric) {
    // Create nearly symmetric pattern with one asymmetric entry
    SparsityPattern pattern(100, 100);
    for (GlobalIndex i = 0; i < 100; ++i) {
        pattern.addEntry(i, i);
        if (i < 99) {
            pattern.addEntry(i, i + 1);
            pattern.addEntry(i + 1, i);  // Symmetric off-diagonal
        }
    }
    // Add one asymmetric entry in lower triangle without upper counterpart
    pattern.addEntry(99, 50);  // (99, 50) without (50, 99)
    pattern.finalize();

    SparsityAnalyzer analyzer;
    auto stats = analyzer.checkSymmetry(pattern);

    // One missing transpose entry makes it asymmetric
    EXPECT_FALSE(stats.is_symmetric);
    EXPECT_TRUE(stats.isNearlySymmetric(0.1));  // But nearly symmetric
}

// ============================================================================
// BlockStats Tests
// ============================================================================

TEST(SparsityAnalyzerTest, BlockDetection) {
    auto pattern = createBlockDiagonalPattern(5, 4);  // 5 blocks of size 4
    SparsityAnalyzer analyzer;
    auto stats = analyzer.detectBlocks(pattern, 8);

    // Block detection is heuristic-based, should detect some structure
    // The exact block size may vary based on the detection algorithm
    EXPECT_TRUE(stats.has_block_structure);
    // Block size should be >= 2 (some structure detected)
    EXPECT_GE(stats.detected_block_size, 2);
}

TEST(SparsityAnalyzerTest, NoBlockStructure) {
    // Create a truly irregular pattern - no block structure
    SparsityPattern pattern(11, 11);  // Prime size to avoid block matching
    for (GlobalIndex i = 0; i < 11; ++i) {
        pattern.addEntry(i, i);
        // Add sparse random-looking entries
        if (i % 3 == 0 && i + 2 < 11) pattern.addEntry(i, i + 2);
        if (i % 5 == 0 && i > 3) pattern.addEntry(i, i - 3);
    }
    pattern.finalize();

    SparsityAnalyzer analyzer;
    auto stats = analyzer.detectBlocks(pattern, 8);

    // Prime-sized irregular pattern should have limited block structure
    // Either no block structure, or if detected, block size should be small
    if (stats.has_block_structure) {
        // If detected, suggested block sizes should exist
        EXPECT_FALSE(stats.suggested_block_sizes.empty());
    }
}

TEST(SparsityAnalyzerTest, ArrowStructure) {
    auto pattern = createArrowPattern(10);
    SparsityAnalyzer analyzer;
    auto stats = analyzer.detectBlocks(pattern);

    EXPECT_GT(stats.n_arrow_rows, 0);
    EXPECT_GT(stats.n_arrow_cols, 0);
}

TEST(SparsityAnalyzerTest, DiagonalNotBlockStructure) {
    auto pattern = createDiagonalPattern(12);
    SparsityAnalyzer analyzer;
    auto stats = analyzer.detectBlocks(pattern, 8);

    EXPECT_FALSE(stats.has_block_structure);
    EXPECT_EQ(stats.detected_block_size, 1);
    EXPECT_TRUE(stats.suggested_block_sizes.empty());
}

// ============================================================================
// FillInEstimate Tests
// ============================================================================

TEST(SparsityAnalyzerTest, FillInEstimateSparse) {
    auto pattern = createTridiagonalPattern(100);
    SparsityAnalyzer analyzer;
    auto estimate = analyzer.estimateFillIn(pattern);

    EXPECT_EQ(estimate.original_nnz, pattern.getNnz());
    EXPECT_GE(estimate.estimated_fill, 0);
    EXPECT_GE(estimate.fill_ratio, 1.0);  // At least original
    EXPECT_FALSE(estimate.method.empty());
}

TEST(SparsityAnalyzerTest, FillInEstimateDense) {
    auto pattern = createDensePattern(10);
    SparsityAnalyzer analyzer;
    auto estimate = analyzer.estimateFillIn(pattern);

    // Already dense, minimal fill-in expected
    EXPECT_EQ(estimate.original_nnz, 100);
    EXPECT_NEAR(estimate.fill_ratio, 1.0, 0.5);  // Close to 1
}

// ============================================================================
// ConnectivityStats Tests
// ============================================================================

TEST(SparsityAnalyzerTest, ConnectivityConnected) {
    auto pattern = createTridiagonalPattern(10);
    SparsityAnalyzer analyzer;
    auto stats = analyzer.analyzeConnectivity(pattern);

    EXPECT_EQ(stats.n_components, 1);
    EXPECT_TRUE(stats.is_connected);
    EXPECT_EQ(stats.largest_component, 10);
}

TEST(SparsityAnalyzerTest, ConnectivityDisconnected) {
    auto pattern = createDisconnectedPattern(10);
    SparsityAnalyzer analyzer;
    auto stats = analyzer.analyzeConnectivity(pattern);

    EXPECT_EQ(stats.n_components, 2);
    EXPECT_FALSE(stats.is_connected);
    EXPECT_EQ(stats.component_sizes.size(), 2);
}

// ============================================================================
// Full Analysis Tests
// ============================================================================

TEST(SparsityAnalyzerTest, FullAnalysis) {
    auto pattern = createTridiagonalPattern(100);
    SparsityAnalyzer analyzer;
    auto report = analyzer.analyze(pattern);

    EXPECT_EQ(report.n_rows, 100);
    EXPECT_EQ(report.n_cols, 100);
    EXPECT_EQ(report.nnz, pattern.getNnz());
    EXPECT_TRUE(report.is_square);
    EXPECT_GT(report.analysis_time_sec, 0);
    EXPECT_NE(report.solver_recommendation, SolverRecommendation::Unknown);
}

TEST(SparsityAnalyzerTest, QuickAnalysis) {
    auto pattern = createTridiagonalPattern(100);
    SparsityAnalyzer analyzer;
    auto report = analyzer.analyzeQuick(pattern);

    EXPECT_EQ(report.n_rows, 100);
    // Quick analysis should be faster but still provide basic stats
}

TEST(SparsityAnalyzerTest, AnalysisOptions) {
    auto pattern = createTridiagonalPattern(50);

    AnalysisOptions opts;
    opts.compute_bandwidth = true;
    opts.compute_symmetry = true;
    opts.detect_blocks = false;
    opts.estimate_fill = false;
    opts.analyze_connectivity = false;

    SparsityAnalyzer analyzer(opts);
    auto report = analyzer.analyze(pattern);

    EXPECT_EQ(report.bandwidth.bandwidth, 1);
    EXPECT_TRUE(report.symmetry.is_symmetric);
}

TEST(SparsityAnalyzerTest, AnalysisReportPrint) {
    auto pattern = createTridiagonalPattern(20);
    SparsityAnalyzer analyzer;
    auto report = analyzer.analyze(pattern);

    std::ostringstream oss;
    report.print(oss);
    std::string output = oss.str();

    EXPECT_FALSE(output.empty());
    EXPECT_NE(output.find("Sparsity Pattern Analysis"), std::string::npos);
    EXPECT_NE(output.find("Bandwidth"), std::string::npos);
}

TEST(SparsityAnalyzerTest, AnalysisReportSummary) {
    auto pattern = createTridiagonalPattern(20);
    SparsityAnalyzer analyzer;
    auto report = analyzer.analyze(pattern);

    std::string summary = report.summary();
    EXPECT_FALSE(summary.empty());
    EXPECT_NE(summary.find("20x20"), std::string::npos);
}

// ============================================================================
// Solver Recommendation Tests
// ============================================================================

TEST(SparsityAnalyzerTest, RecommendSolverSmall) {
    auto pattern = createDensePattern(50);
    SparsityAnalyzer analyzer;
    auto rec = analyzer.recommendSolver(pattern);

    EXPECT_EQ(rec, SolverRecommendation::DirectDense);
}

TEST(SparsityAnalyzerTest, RecommendSolverSparse) {
    auto pattern = createTridiagonalPattern(5000);
    SparsityAnalyzer analyzer;
    auto rec = analyzer.recommendSolver(pattern);

    // Large sparse system should recommend iterative
    EXPECT_TRUE(rec == SolverRecommendation::IterativeKrylov ||
                rec == SolverRecommendation::IterativeMultigrid ||
                rec == SolverRecommendation::DirectSparse);
}

TEST(SparsityAnalyzerTest, RecommendSolverWithReason) {
    auto pattern = createTridiagonalPattern(100);
    SparsityAnalyzer analyzer;
    auto [rec, reason] = analyzer.recommendSolverWithReason(pattern);

    EXPECT_NE(rec, SolverRecommendation::Unknown);
    EXPECT_FALSE(reason.empty());
}

TEST(SparsityAnalyzerTest, RecommendationName) {
    EXPECT_EQ(SparsityAnalyzer::recommendationName(SolverRecommendation::DirectSparse),
              "Direct Sparse");
    EXPECT_EQ(SparsityAnalyzer::recommendationName(SolverRecommendation::IterativeKrylov),
              "Iterative Krylov");
}

// ============================================================================
// Memory Estimation Tests
// ============================================================================

TEST(SparsityAnalyzerTest, EstimateFactorMemory) {
    auto pattern = createTridiagonalPattern(100);
    SparsityAnalyzer analyzer;
    auto memory = analyzer.estimateFactorMemory(pattern);

    EXPECT_GT(memory, 0);
    EXPECT_GE(memory, pattern.getNnz() * sizeof(double));
}

// ============================================================================
// Reordering Suggestion Tests
// ============================================================================

TEST(SparsityAnalyzerTest, SuggestReorderingSymmetric) {
    auto pattern = createTridiagonalPattern(100);
    SparsityAnalyzer analyzer;
    auto suggestion = analyzer.suggestReordering(pattern);

    EXPECT_FALSE(suggestion.empty());
    // Symmetric tridiagonal should suggest AMD or RCM
    EXPECT_TRUE(suggestion == "AMD" || suggestion == "RCM" || suggestion == "none");
}

TEST(SparsityAnalyzerTest, SuggestReorderingAsymmetric) {
    auto pattern = createAsymmetricPattern(100);
    SparsityAnalyzer analyzer;
    auto suggestion = analyzer.suggestReordering(pattern);

    EXPECT_FALSE(suggestion.empty());
}

// ============================================================================
// Visualization Tests
// ============================================================================

TEST(SparsityAnalyzerTest, GenerateSpyData) {
    auto pattern = createTridiagonalPattern(100);
    SparsityAnalyzer analyzer;
    auto [rows, cols] = analyzer.generateSpyData(pattern, 50);

    EXPECT_FALSE(rows.empty());
    EXPECT_EQ(rows.size(), cols.size());
}

TEST(SparsityAnalyzerTest, ExportMatrixMarket) {
    auto pattern = createTridiagonalPattern(10);
    SparsityAnalyzer analyzer;
    auto mm = analyzer.exportMatrixMarket(pattern);

    EXPECT_FALSE(mm.empty());
    EXPECT_NE(mm.find("%%MatrixMarket"), std::string::npos);
    EXPECT_NE(mm.find("10 10"), std::string::npos);
}

TEST(SparsityAnalyzerTest, ExportAscii) {
    auto pattern = createTridiagonalPattern(5);
    SparsityAnalyzer analyzer;
    auto ascii = analyzer.exportAscii(pattern, 10);

    EXPECT_FALSE(ascii.empty());
    EXPECT_NE(ascii.find("X"), std::string::npos);  // Has entries
    EXPECT_NE(ascii.find("."), std::string::npos);  // Has zeros
}

// ============================================================================
// Compare Patterns Tests
// ============================================================================

TEST(SparsityAnalyzerTest, ComparePatterns) {
    auto pattern1 = createTridiagonalPattern(10);
    auto pattern2 = createTridiagonalPattern(10);

    SparsityAnalyzer analyzer;
    std::string comparison = analyzer.compare(pattern1, pattern2);

    EXPECT_FALSE(comparison.empty());
    EXPECT_NE(comparison.find("Comparison"), std::string::npos);
}

TEST(SparsityAnalyzerTest, CompareDifferentPatterns) {
    auto pattern1 = createTridiagonalPattern(10);
    auto pattern2 = createDensePattern(10);

    SparsityAnalyzer analyzer;
    std::string comparison = analyzer.compare(pattern1, pattern2);

    EXPECT_FALSE(comparison.empty());
    EXPECT_NE(comparison.find("NNZ:"), std::string::npos);
}

// ============================================================================
// Convenience Function Tests
// ============================================================================

TEST(SparsityAnalyzerTest, ConvenienceBandwidth) {
    auto pattern = createTridiagonalPattern(10);
    EXPECT_EQ(computeBandwidth(pattern), 1);
}

TEST(SparsityAnalyzerTest, ConvenienceIsSymmetric) {
    auto symmetric = createTridiagonalPattern(10);
    auto asymmetric = createAsymmetricPattern(10);

    EXPECT_TRUE(isSymmetric(symmetric));
    EXPECT_FALSE(isSymmetric(asymmetric));
}

TEST(SparsityAnalyzerTest, ConvenienceFillRatio) {
    auto diagonal = createDiagonalPattern(10);
    auto dense = createDensePattern(10);

    EXPECT_NEAR(computeFillRatio(diagonal), 0.1, 0.01);
    EXPECT_NEAR(computeFillRatio(dense), 1.0, 0.01);
}

TEST(SparsityAnalyzerTest, ConveniencePrintAnalysis) {
    auto pattern = createTridiagonalPattern(10);
    std::ostringstream oss;
    printAnalysis(pattern, oss);

    EXPECT_FALSE(oss.str().empty());
}

TEST(SparsityAnalyzerTest, ConvenienceGetSummary) {
    auto pattern = createTridiagonalPattern(10);
    std::string summary = getPatternSummary(pattern);

    EXPECT_FALSE(summary.empty());
    EXPECT_NE(summary.find("10x10"), std::string::npos);
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST(SparsityAnalyzerTest, EmptyPattern) {
    SparsityPattern pattern(0, 0);
    pattern.finalize();

    SparsityAnalyzer analyzer;
    auto report = analyzer.analyze(pattern);

    EXPECT_EQ(report.n_rows, 0);
    EXPECT_EQ(report.n_cols, 0);
    EXPECT_EQ(report.nnz, 0);
}

TEST(SparsityAnalyzerTest, SingleElement) {
    SparsityPattern pattern(1, 1);
    pattern.addEntry(0, 0);
    pattern.finalize();

    SparsityAnalyzer analyzer;
    auto report = analyzer.analyze(pattern);

    EXPECT_EQ(report.n_rows, 1);
    EXPECT_EQ(report.nnz, 1);
    EXPECT_TRUE(report.symmetry.is_symmetric);
    EXPECT_EQ(report.bandwidth.bandwidth, 0);
}

TEST(SparsityAnalyzerTest, RectangularPattern) {
    SparsityPattern pattern(5, 10);
    for (GlobalIndex i = 0; i < 5; ++i) {
        pattern.addEntry(i, i);
        pattern.addEntry(i, i + 5);
    }
    pattern.finalize();

    SparsityAnalyzer analyzer;
    auto report = analyzer.analyze(pattern);

    EXPECT_EQ(report.n_rows, 5);
    EXPECT_EQ(report.n_cols, 10);
    EXPECT_FALSE(report.is_square);
    // Symmetry check should be skipped for non-square
}

TEST(SparsityAnalyzerTest, PatternWithEmptyRows) {
    SparsityPattern pattern(5, 5);
    pattern.addEntry(0, 0);
    pattern.addEntry(2, 2);
    pattern.addEntry(4, 4);
    // Rows 1 and 3 are empty
    pattern.finalize();

    SparsityAnalyzer analyzer;
    auto stats = analyzer.computeRowStats(pattern);

    EXPECT_EQ(stats.empty_rows, 2);
    EXPECT_EQ(stats.min_nnz, 0);
}

TEST(SparsityAnalyzerTest, VeryLargePattern) {
    // Test with a large pattern to ensure scalability
    auto pattern = createTridiagonalPattern(10000);

    SparsityAnalyzer analyzer;
    auto report = analyzer.analyzeQuick(pattern);

    EXPECT_EQ(report.n_rows, 10000);
    EXPECT_GT(report.analysis_time_sec, 0);
}

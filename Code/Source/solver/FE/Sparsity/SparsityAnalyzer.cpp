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

#include "SparsityAnalyzer.h"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <queue>
#include <set>

namespace svmp {
namespace FE {
namespace sparsity {

// ============================================================================
// AnalysisReport Implementation
// ============================================================================

void AnalysisReport::print(std::ostream& out) const {
    out << "\n";
    out << "========================================\n";
    out << "     Sparsity Pattern Analysis Report\n";
    out << "========================================\n\n";

    // Basic info
    out << "Basic Statistics:\n";
    out << "  Dimensions:     " << n_rows << " x " << n_cols;
    if (is_square) out << " (square)";
    out << "\n";
    out << "  Non-zeros:      " << nnz << "\n";
    out << "  Density:        " << std::fixed << std::setprecision(6)
        << density * 100.0 << "%\n\n";

    // Bandwidth
    out << "Bandwidth Statistics:\n";
    out << "  Bandwidth:      " << bandwidth.bandwidth << "\n";
    out << "  Lower BW:       " << bandwidth.lower_bandwidth << "\n";
    out << "  Upper BW:       " << bandwidth.upper_bandwidth << "\n";
    out << "  Profile:        " << bandwidth.profile << "\n";
    out << "  Avg Row BW:     " << std::fixed << std::setprecision(2)
        << bandwidth.avg_bandwidth << "\n";
    if (bandwidth.estimated_rcm_bandwidth > 0) {
        out << "  Est. RCM BW:    " << bandwidth.estimated_rcm_bandwidth << "\n";
    }
    out << "\n";

    // Row statistics
    out << "Row Statistics:\n";
    out << "  Empty rows:     " << rows.empty_rows << "\n";
    out << "  Min NNZ/row:    " << rows.min_nnz << "\n";
    out << "  Max NNZ/row:    " << rows.max_nnz << "\n";
    out << "  Avg NNZ/row:    " << std::fixed << std::setprecision(2)
        << rows.avg_nnz << "\n";
    out << "  Std Dev:        " << std::fixed << std::setprecision(2)
        << rows.std_dev_nnz << "\n";
    out << "  Distribution:   " << (rows.isUniform() ? "Uniform" : "Non-uniform") << "\n\n";

    // Diagonal
    out << "Diagonal Structure:\n";
    out << "  Present:        " << diagonal.n_diag << "/" << n_rows
        << " (" << std::fixed << std::setprecision(1)
        << diagonal.coverage() * 100 << "%)\n";
    out << "  All present:    " << (diagonal.all_present ? "Yes" : "No") << "\n";
    out << "  Isolated diag:  " << diagonal.n_isolated_diag << "\n\n";

    // Symmetry
    out << "Symmetry:\n";
    out << "  Symmetric:      " << (symmetry.is_symmetric ? "Yes" : "No") << "\n";
    out << "  Symmetric pairs: " << symmetry.n_symmetric << "\n";
    out << "  Asymmetric:     " << symmetry.n_asymmetric << "\n";
    out << "  Symmetry ratio: " << std::fixed << std::setprecision(2)
        << symmetry.symmetry_ratio * 100 << "%\n\n";

    // Block structure
    if (blocks.has_block_structure) {
        out << "Block Structure:\n";
        out << "  Detected:       Yes\n";
        out << "  Block size:     " << blocks.detected_block_size << "\n";
        out << "  Dense blocks:   " << blocks.n_dense_blocks << "\n\n";
    }

    // Fill-in estimation
    out << "Fill-in Estimation:\n";
    out << "  Original NNZ:   " << fill_in.original_nnz << "\n";
    out << "  Est. fill-in:   " << fill_in.estimated_fill << "\n";
    out << "  Est. total:     " << fill_in.estimated_total << "\n";
    out << "  Fill ratio:     " << std::fixed << std::setprecision(2)
        << fill_in.fill_ratio << "x\n";
    out << "  Method:         " << fill_in.method << "\n\n";

    // Connectivity (if analyzed)
    if (connectivity.n_components > 0) {
        out << "Connectivity:\n";
        out << "  Components:     " << connectivity.n_components << "\n";
        out << "  Connected:      " << (connectivity.is_connected ? "Yes" : "No") << "\n";
        if (!connectivity.is_connected) {
            out << "  Largest comp:   " << connectivity.largest_component << "\n";
            out << "  Smallest comp:  " << connectivity.smallest_component << "\n";
        }
        out << "\n";
    }

    // Recommendation
    out << "Recommendation:\n";
    out << "  Solver type:    " << SparsityAnalyzer::recommendationName(solver_recommendation) << "\n";
    if (!recommendation_reason.empty()) {
        out << "  Reason:         " << recommendation_reason << "\n";
    }
    out << "  Est. memory:    " << (estimated_factor_memory / (1024.0 * 1024.0))
        << " MB\n\n";

    out << "Analysis time: " << std::fixed << std::setprecision(3)
        << analysis_time_sec * 1000 << " ms\n";
    out << "========================================\n";
}

std::string AnalysisReport::summary() const {
    std::ostringstream oss;
    oss << n_rows << "x" << n_cols << ", NNZ=" << nnz
        << ", density=" << std::fixed << std::setprecision(4) << density * 100 << "%"
        << ", BW=" << bandwidth.bandwidth
        << ", " << (symmetry.is_symmetric ? "symmetric" : "asymmetric");
    return oss.str();
}

// ============================================================================
// SparsityAnalyzer Implementation
// ============================================================================

SparsityAnalyzer::SparsityAnalyzer(const AnalysisOptions& options)
    : options_(options)
{
}

AnalysisReport SparsityAnalyzer::analyze(
    const SparsityPattern& pattern,
    const AnalysisOptions& options) const {

    auto start_time = std::chrono::steady_clock::now();

    AnalysisReport report;

    // Basic statistics
    report.n_rows = pattern.numRows();
    report.n_cols = pattern.numCols();
    report.nnz = pattern.getNnz();
    report.is_square = pattern.isSquare();

    if (report.n_rows > 0 && report.n_cols > 0) {
        report.density = static_cast<double>(report.nnz) /
            (static_cast<double>(report.n_rows) * static_cast<double>(report.n_cols));
    }

    // Bandwidth analysis
    if (options.compute_bandwidth) {
        report.bandwidth = computeBandwidth(pattern);
    }

    // Row statistics
    if (options.compute_row_stats) {
        report.rows = computeRowStats(pattern, options.histogram_buckets);
    }

    // Diagonal analysis
    if (options.compute_diagonal) {
        report.diagonal = analyzeDiagonal(pattern);
    }

    // Symmetry analysis
    if (options.compute_symmetry && report.is_square) {
        report.symmetry = checkSymmetry(pattern);
    }

    // Block detection
    if (options.detect_blocks && report.is_square) {
        report.blocks = detectBlocks(pattern, options.max_block_size);
    }

    // Fill-in estimation
    if (options.estimate_fill && report.is_square) {
        report.fill_in = estimateFillIn(pattern);
    }

    // Connectivity analysis
    if (options.analyze_connectivity) {
        report.connectivity = analyzeConnectivity(pattern);
    }

    // Generate recommendation
    if (options.make_recommendations) {
        auto [rec, reason] = recommendSolverWithReason(pattern);
        report.solver_recommendation = rec;
        report.recommendation_reason = reason;
        report.estimated_factor_memory = estimateFactorMemory(pattern);
    }

    auto end_time = std::chrono::steady_clock::now();
    report.analysis_time_sec = std::chrono::duration<double>(end_time - start_time).count();

    return report;
}

AnalysisReport SparsityAnalyzer::analyze(const SparsityPattern& pattern) const {
    return analyze(pattern, options_);
}

AnalysisReport SparsityAnalyzer::analyzeQuick(const SparsityPattern& pattern) const {
    return analyze(pattern, AnalysisOptions::quick());
}

// ============================================================================
// Individual Analyses
// ============================================================================

BandwidthStats SparsityAnalyzer::computeBandwidth(const SparsityPattern& pattern) const {
    BandwidthStats stats;

    if (pattern.numRows() == 0) {
        return stats;
    }

    GlobalIndex n_rows = pattern.numRows();
    std::vector<GlobalIndex> row_bandwidths;
    row_bandwidths.reserve(static_cast<std::size_t>(n_rows));

    GlobalIndex profile = 0;

    for (GlobalIndex row = 0; row < n_rows; ++row) {
        GlobalIndex row_nnz = pattern.getRowNnz(row);
        if (row_nnz == 0) {
            row_bandwidths.push_back(0);
            continue;
        }

        GlobalIndex first_col = row;  // Default to diagonal
        GlobalIndex last_col = row;

        if (pattern.isFinalized()) {
            auto row_span = pattern.getRowSpan(row);
            if (!row_span.empty()) {
                first_col = row_span.front();
                last_col = row_span.back();

                for (GlobalIndex col : row_span) {
                    GlobalIndex diff = std::abs(row - col);
                    stats.bandwidth = std::max(stats.bandwidth, diff);

                    if (row > col) {
                        stats.lower_bandwidth = std::max(stats.lower_bandwidth, row - col);
                    } else if (col > row) {
                        stats.upper_bandwidth = std::max(stats.upper_bandwidth, col - row);
                    }
                }
            }
        } else {
            // Pattern not finalized - iterate over hasEntry (slower)
            for (GlobalIndex col = 0; col < pattern.numCols(); ++col) {
                if (pattern.hasEntry(row, col)) {
                    first_col = std::min(first_col, col);
                    last_col = std::max(last_col, col);

                    GlobalIndex diff = std::abs(row - col);
                    stats.bandwidth = std::max(stats.bandwidth, diff);

                    if (row > col) {
                        stats.lower_bandwidth = std::max(stats.lower_bandwidth, row - col);
                    } else if (col > row) {
                        stats.upper_bandwidth = std::max(stats.upper_bandwidth, col - row);
                    }
                }
            }
        }

        GlobalIndex row_bw = last_col - first_col;
        row_bandwidths.push_back(row_bw);
        stats.envelope += row_bw;

        // Profile: distance from diagonal to first non-zero in row
        if (first_col <= row) {
            profile += row - first_col;
        }
    }

    stats.profile = profile;

    // Compute average and median bandwidth
    if (!row_bandwidths.empty()) {
        GlobalIndex sum = 0;
        for (GlobalIndex bw : row_bandwidths) {
            sum += bw;
        }
        stats.avg_bandwidth = static_cast<double>(sum) /
                              static_cast<double>(row_bandwidths.size());

        // Median
        std::sort(row_bandwidths.begin(), row_bandwidths.end());
        std::size_t mid = row_bandwidths.size() / 2;
        if (row_bandwidths.size() % 2 == 0) {
            stats.median_bandwidth = (row_bandwidths[mid-1] + row_bandwidths[mid]) / 2;
        } else {
            stats.median_bandwidth = row_bandwidths[mid];
        }
    }

    // Estimate RCM improvement (heuristic: typically 20-40% reduction)
    stats.estimated_rcm_bandwidth = static_cast<GlobalIndex>(
        static_cast<double>(stats.bandwidth) * 0.6);

    return stats;
}

RowStats SparsityAnalyzer::computeRowStats(
    const SparsityPattern& pattern,
    GlobalIndex n_buckets) const {

    RowStats stats;
    stats.n_rows = pattern.numRows();

    if (stats.n_rows == 0) {
        return stats;
    }

    // Collect row NNZ values
    std::vector<GlobalIndex> row_nnz = getRowNnzArray(pattern);

    // Basic statistics
    stats.min_nnz = *std::min_element(row_nnz.begin(), row_nnz.end());
    stats.max_nnz = *std::max_element(row_nnz.begin(), row_nnz.end());

    GlobalIndex sum = 0;
    for (GlobalIndex nnz : row_nnz) {
        sum += nnz;
        if (nnz == 0) {
            ++stats.empty_rows;
        }
    }
    stats.avg_nnz = static_cast<double>(sum) / static_cast<double>(row_nnz.size());

    // Standard deviation
    double sum_sq_diff = 0.0;
    for (GlobalIndex nnz : row_nnz) {
        double diff = static_cast<double>(nnz) - stats.avg_nnz;
        sum_sq_diff += diff * diff;
    }
    stats.std_dev_nnz = std::sqrt(sum_sq_diff / static_cast<double>(row_nnz.size()));

    // Median
    std::sort(row_nnz.begin(), row_nnz.end());
    std::size_t mid = row_nnz.size() / 2;
    if (row_nnz.size() % 2 == 0) {
        stats.median_nnz = (row_nnz[mid-1] + row_nnz[mid]) / 2;
    } else {
        stats.median_nnz = row_nnz[mid];
    }

    // Histogram
    if (n_buckets > 0 && stats.max_nnz > stats.min_nnz) {
        stats.histogram.resize(static_cast<std::size_t>(n_buckets), 0);
        stats.histogram_bounds.resize(static_cast<std::size_t>(n_buckets) + 1);

        GlobalIndex range = stats.max_nnz - stats.min_nnz;
        GlobalIndex bucket_size = (range + n_buckets - 1) / n_buckets;

        for (GlobalIndex i = 0; i <= n_buckets; ++i) {
            stats.histogram_bounds[static_cast<std::size_t>(i)] =
                stats.min_nnz + i * bucket_size;
        }

        for (GlobalIndex nnz : row_nnz) {
            GlobalIndex bucket = (nnz - stats.min_nnz) / bucket_size;
            if (bucket >= n_buckets) bucket = n_buckets - 1;
            ++stats.histogram[static_cast<std::size_t>(bucket)];
        }
    }

    return stats;
}

DiagonalStats SparsityAnalyzer::analyzeDiagonal(const SparsityPattern& pattern) const {
    DiagonalStats stats;

    if (!pattern.isSquare()) {
        return stats;
    }

    GlobalIndex n = pattern.numRows();
    stats.all_present = true;

    for (GlobalIndex i = 0; i < n; ++i) {
        bool has_diag = pattern.hasDiagonal(i);
        if (has_diag) {
            ++stats.n_diag;

            // Check if isolated (diagonal only)
            if (pattern.getRowNnz(i) == 1) {
                ++stats.n_isolated_diag;
            }
        } else {
            ++stats.n_missing;
            stats.all_present = false;
        }
    }

    return stats;
}

SymmetryStats SparsityAnalyzer::checkSymmetry(const SparsityPattern& pattern) const {
    SymmetryStats stats;

    if (!pattern.isSquare()) {
        return stats;
    }

    GlobalIndex n_rows = pattern.numRows();
    stats.is_symmetric = true;

    for (GlobalIndex row = 0; row < n_rows; ++row) {
        if (pattern.isFinalized()) {
            auto row_span = pattern.getRowSpan(row);
            for (GlobalIndex col : row_span) {
                if (col >= row) continue;  // Only check lower triangle

                bool has_transpose = pattern.hasEntry(col, row);
                if (has_transpose) {
                    stats.n_symmetric += 2;  // Count both (i,j) and (j,i)
                } else {
                    ++stats.n_asymmetric;
                    stats.is_symmetric = false;
                }
            }
            // Count diagonal
            if (pattern.hasEntry(row, row)) {
                ++stats.n_symmetric;
            }
        } else {
            for (GlobalIndex col = 0; col <= row; ++col) {
                if (pattern.hasEntry(row, col)) {
                    if (row == col) {
                        ++stats.n_symmetric;
                    } else {
                        bool has_transpose = pattern.hasEntry(col, row);
                        if (has_transpose) {
                            stats.n_symmetric += 2;
                        } else {
                            ++stats.n_asymmetric;
                            stats.is_symmetric = false;
                        }
                    }
                }
            }
        }
    }

    GlobalIndex total = stats.n_symmetric + stats.n_asymmetric;
    if (total > 0) {
        stats.symmetry_ratio = static_cast<double>(stats.n_symmetric) /
                               static_cast<double>(total);
    } else {
        stats.symmetry_ratio = 1.0;
    }

    return stats;
}

BlockStats SparsityAnalyzer::detectBlocks(
    const SparsityPattern& pattern,
    GlobalIndex max_block_size) const {

    BlockStats stats;

    if (!pattern.isSquare() || pattern.numRows() < 2) {
        return stats;
    }

    // Try to detect block structure by checking if pattern is consistent
    // with blocked indexing
    for (GlobalIndex bs = 2; bs <= max_block_size; ++bs) {
        if (pattern.numRows() % bs != 0) continue;

        if (checkBlockPattern(pattern, bs)) {
            stats.has_block_structure = true;
            stats.detected_block_size = bs;
            stats.suggested_block_sizes.push_back(bs);
        }
    }

    // Count dense diagonal blocks
    if (stats.has_block_structure) {
        GlobalIndex bs = stats.detected_block_size;
        GlobalIndex n_blocks = pattern.numRows() / bs;

        for (GlobalIndex b = 0; b < n_blocks; ++b) {
            GlobalIndex start = b * bs;
            GlobalIndex end = start + bs;

            // Check if diagonal block is dense
            bool is_dense = true;
            for (GlobalIndex i = start; i < end && is_dense; ++i) {
                for (GlobalIndex j = start; j < end && is_dense; ++j) {
                    if (!pattern.hasEntry(i, j)) {
                        is_dense = false;
                    }
                }
            }
            if (is_dense) {
                ++stats.n_dense_blocks;
            }
        }
    }

    // Check for arrow structure (dense first row/col)
    if (pattern.numRows() > 0) {
        GlobalIndex first_row_nnz = pattern.getRowNnz(0);
        if (first_row_nnz > pattern.numCols() / 2) {
            ++stats.n_arrow_rows;
        }

        // Check first column density
        GlobalIndex first_col_count = 0;
        for (GlobalIndex i = 0; i < pattern.numRows(); ++i) {
            if (pattern.hasEntry(i, 0)) {
                ++first_col_count;
            }
        }
        if (first_col_count > pattern.numRows() / 2) {
            ++stats.n_arrow_cols;
        }
    }

    return stats;
}

FillInEstimate SparsityAnalyzer::estimateFillIn(const SparsityPattern& pattern) const {
    FillInEstimate estimate;

    if (!pattern.isSquare()) {
        return estimate;
    }

    estimate.original_nnz = pattern.getNnz();

    // Use simple heuristic estimation
    estimate.estimated_fill = estimateFillInSimple(pattern);
    estimate.estimated_total = estimate.original_nnz + estimate.estimated_fill;

    if (estimate.original_nnz > 0) {
        estimate.fill_ratio = static_cast<double>(estimate.estimated_total) /
                              static_cast<double>(estimate.original_nnz);
    }

    GlobalIndex n = pattern.numRows();
    if (n > 0) {
        estimate.density_after = static_cast<double>(estimate.estimated_total) /
                                 (static_cast<double>(n) * static_cast<double>(n));
    }

    estimate.method = "heuristic";
    estimate.confidence = 0.6;  // Heuristic estimate

    return estimate;
}

ConnectivityStats SparsityAnalyzer::analyzeConnectivity(
    const SparsityPattern& pattern) const {

    ConnectivityStats stats;

    if (pattern.numRows() == 0) {
        return stats;
    }

    std::vector<GlobalIndex> component_ids = findConnectedComponents(pattern);

    // Count unique components
    std::set<GlobalIndex> unique_components(component_ids.begin(), component_ids.end());
    stats.n_components = static_cast<GlobalIndex>(unique_components.size());
    stats.is_connected = (stats.n_components == 1);

    // Count component sizes
    std::map<GlobalIndex, GlobalIndex> comp_sizes;
    for (GlobalIndex comp : component_ids) {
        ++comp_sizes[comp];
    }

    stats.component_sizes.reserve(comp_sizes.size());
    for (const auto& [comp, size] : comp_sizes) {
        stats.component_sizes.push_back(size);
    }

    if (!stats.component_sizes.empty()) {
        stats.largest_component = *std::max_element(
            stats.component_sizes.begin(), stats.component_sizes.end());
        stats.smallest_component = *std::min_element(
            stats.component_sizes.begin(), stats.component_sizes.end());
    }

    return stats;
}

// ============================================================================
// Recommendations
// ============================================================================

SolverRecommendation SparsityAnalyzer::recommendSolver(
    const SparsityPattern& pattern) const {
    return recommendSolverWithReason(pattern).first;
}

std::pair<SolverRecommendation, std::string> SparsityAnalyzer::recommendSolverWithReason(
    const SparsityPattern& pattern) const {

    GlobalIndex n = pattern.numRows();
    GlobalIndex nnz = pattern.getNnz();

    if (n == 0) {
        return {SolverRecommendation::Unknown, "Empty matrix"};
    }

    double density = static_cast<double>(nnz) /
                     (static_cast<double>(n) * static_cast<double>(n));

    // Very small or very dense -> direct dense
    if (n < 100 || density > 0.5) {
        return {SolverRecommendation::DirectDense,
                "Small size or high density"};
    }

    // Small to medium with moderate sparsity -> direct sparse
    if (n < 10000 && density < 0.1) {
        return {SolverRecommendation::DirectSparse,
                "Moderate size with good sparsity"};
    }

    // Medium size with good sparsity
    if (n < 100000) {
        auto fill = estimateFillIn(pattern);
        if (fill.fill_ratio < 10.0) {
            return {SolverRecommendation::DirectSparse,
                    "Acceptable fill-in estimate"};
        }
    }

    // Large problems -> iterative
    double avg_nnz = static_cast<double>(nnz) / static_cast<double>(n);
    if (avg_nnz < 20) {
        return {SolverRecommendation::IterativeKrylov,
                "Large sparse system, low connectivity"};
    }

    // Check for structured pattern (might benefit from multigrid)
    auto bandwidth = computeBandwidth(pattern);
    if (bandwidth.bandwidth < n / 10) {
        return {SolverRecommendation::IterativeMultigrid,
                "Banded structure suggests structured problem"};
    }

    return {SolverRecommendation::IterativeKrylov,
            "Large system with moderate connectivity"};
}

std::size_t SparsityAnalyzer::estimateFactorMemory(
    const SparsityPattern& pattern,
    std::size_t element_size) const {

    if (!pattern.isSquare()) {
        return 0;
    }

    auto fill = estimateFillIn(pattern);
    return static_cast<std::size_t>(fill.estimated_total) * element_size;
}

std::string SparsityAnalyzer::suggestReordering(const SparsityPattern& pattern) const {
    if (!pattern.isSquare()) {
        return "none";
    }

    GlobalIndex n = pattern.numRows();

    // Small problems don't need reordering
    if (n < 100) {
        return "none";
    }

    auto symmetry = checkSymmetry(pattern);

    // Symmetric problems -> AMD or RCM
    if (symmetry.is_symmetric || symmetry.isNearlySymmetric()) {
        auto bandwidth = computeBandwidth(pattern);
        if (bandwidth.bandwidth > n / 10) {
            return "RCM";  // High bandwidth, try RCM first
        }
        return "AMD";  // Low bandwidth, AMD for fill-in reduction
    }

    // Asymmetric -> nested dissection or column AMD
    return "NestedDissection";
}

// ============================================================================
// Distributed Pattern Analysis
// ============================================================================

AnalysisReport SparsityAnalyzer::analyze(
    const DistributedSparsityPattern& pattern) const {

    // Analyze the local portion
    AnalysisReport report;

    report.n_rows = pattern.numOwnedRows();
    report.n_cols = pattern.globalCols();
    report.nnz = pattern.getLocalNnz();
    report.is_square = (pattern.globalRows() == pattern.globalCols());

    if (report.n_rows > 0 && report.n_cols > 0) {
        report.density = static_cast<double>(report.nnz) /
            (static_cast<double>(report.n_rows) * static_cast<double>(report.n_cols));
    }

    // Additional distributed-specific info could be added here

    return report;
}

std::map<int, GlobalIndex> SparsityAnalyzer::analyzeCommPattern(
    const DistributedSparsityPattern& /*pattern*/) const {

    // This would require MPI communication to be fully implemented
    // For now, return empty map
    std::map<int, GlobalIndex> comm_pattern;
    return comm_pattern;
}

// ============================================================================
// Visualization Support
// ============================================================================

std::pair<std::vector<GlobalIndex>, std::vector<GlobalIndex>>
SparsityAnalyzer::generateSpyData(
    const SparsityPattern& pattern,
    GlobalIndex max_size) const {

    std::vector<GlobalIndex> rows, cols;

    GlobalIndex n_rows = pattern.numRows();
    GlobalIndex n_cols = pattern.numCols();
    GlobalIndex nnz = pattern.getNnz();

    // If small enough, return all points
    if (n_rows <= max_size && n_cols <= max_size) {
        rows.reserve(static_cast<std::size_t>(nnz));
        cols.reserve(static_cast<std::size_t>(nnz));

        for (GlobalIndex row = 0; row < n_rows; ++row) {
            if (pattern.isFinalized()) {
                auto row_span = pattern.getRowSpan(row);
                for (GlobalIndex col : row_span) {
                    rows.push_back(row);
                    cols.push_back(col);
                }
            }
        }
    } else {
        // Sample points for large patterns
        double row_scale = static_cast<double>(max_size) / static_cast<double>(n_rows);
        double col_scale = static_cast<double>(max_size) / static_cast<double>(n_cols);

        std::set<std::pair<GlobalIndex, GlobalIndex>> sampled;

        for (GlobalIndex row = 0; row < n_rows; ++row) {
            GlobalIndex sampled_row = static_cast<GlobalIndex>(
                static_cast<double>(row) * row_scale);

            if (pattern.isFinalized()) {
                auto row_span = pattern.getRowSpan(row);
                for (GlobalIndex col : row_span) {
                    GlobalIndex sampled_col = static_cast<GlobalIndex>(
                        static_cast<double>(col) * col_scale);
                    sampled.emplace(sampled_row, sampled_col);
                }
            }
        }

        rows.reserve(sampled.size());
        cols.reserve(sampled.size());
        for (const auto& [r, c] : sampled) {
            rows.push_back(r);
            cols.push_back(c);
        }
    }

    return {rows, cols};
}

std::string SparsityAnalyzer::exportMatrixMarket(const SparsityPattern& pattern) const {
    std::ostringstream oss;

    oss << "%%MatrixMarket matrix coordinate pattern general\n";
    oss << "% Generated by SparsityAnalyzer\n";
    oss << pattern.numRows() << " " << pattern.numCols() << " " << pattern.getNnz() << "\n";

    for (GlobalIndex row = 0; row < pattern.numRows(); ++row) {
        if (pattern.isFinalized()) {
            auto row_span = pattern.getRowSpan(row);
            for (GlobalIndex col : row_span) {
                oss << (row + 1) << " " << (col + 1) << "\n";  // 1-based indexing
            }
        }
    }

    return oss.str();
}

std::string SparsityAnalyzer::exportAscii(
    const SparsityPattern& pattern,
    GlobalIndex max_size) const {

    std::ostringstream oss;

    GlobalIndex n_rows = std::min(pattern.numRows(), max_size);
    GlobalIndex n_cols = std::min(pattern.numCols(), max_size);

    // Header
    oss << "    ";
    for (GlobalIndex c = 0; c < n_cols; ++c) {
        oss << (c % 10);
    }
    oss << "\n";
    oss << "    ";
    for (GlobalIndex c = 0; c < n_cols; ++c) {
        oss << "-";
    }
    oss << "\n";

    // Rows
    for (GlobalIndex r = 0; r < n_rows; ++r) {
        oss << std::setw(3) << r << "|";
        for (GlobalIndex c = 0; c < n_cols; ++c) {
            if (pattern.hasEntry(r, c)) {
                oss << "X";
            } else {
                oss << ".";
            }
        }
        oss << "\n";
    }

    if (pattern.numRows() > max_size || pattern.numCols() > max_size) {
        oss << "\n(truncated to " << max_size << "x" << max_size << ")\n";
    }

    return oss.str();
}

// ============================================================================
// Utility Methods
// ============================================================================

std::string SparsityAnalyzer::recommendationName(SolverRecommendation rec) {
    switch (rec) {
        case SolverRecommendation::DirectSparse: return "Direct Sparse";
        case SolverRecommendation::DirectDense: return "Direct Dense";
        case SolverRecommendation::IterativeKrylov: return "Iterative Krylov";
        case SolverRecommendation::IterativeMultigrid: return "Iterative Multigrid";
        case SolverRecommendation::Hybrid: return "Hybrid";
        default: return "Unknown";
    }
}

std::string SparsityAnalyzer::compare(
    const SparsityPattern& a,
    const SparsityPattern& b) const {

    std::ostringstream oss;
    oss << "Pattern Comparison:\n";
    oss << "  Dimensions: " << a.numRows() << "x" << a.numCols()
        << " vs " << b.numRows() << "x" << b.numCols() << "\n";
    oss << "  NNZ: " << a.getNnz() << " vs " << b.getNnz() << "\n";

    if (a.numRows() == b.numRows() && a.numCols() == b.numCols()) {
        // Count differences
        GlobalIndex only_in_a = 0, only_in_b = 0, in_both = 0;
        for (GlobalIndex row = 0; row < a.numRows(); ++row) {
            if (a.isFinalized() && b.isFinalized()) {
                auto span_a = a.getRowSpan(row);
                auto span_b = b.getRowSpan(row);

                std::set<GlobalIndex> set_a(span_a.begin(), span_a.end());
                std::set<GlobalIndex> set_b(span_b.begin(), span_b.end());

                for (GlobalIndex col : set_a) {
                    if (set_b.count(col)) {
                        ++in_both;
                    } else {
                        ++only_in_a;
                    }
                }
                for (GlobalIndex col : set_b) {
                    if (!set_a.count(col)) {
                        ++only_in_b;
                    }
                }
            }
        }
        oss << "  In both: " << in_both << "\n";
        oss << "  Only in A: " << only_in_a << "\n";
        oss << "  Only in B: " << only_in_b << "\n";
    }

    return oss.str();
}

// ============================================================================
// Internal Helpers
// ============================================================================

std::vector<GlobalIndex> SparsityAnalyzer::getRowNnzArray(
    const SparsityPattern& pattern) const {

    std::vector<GlobalIndex> row_nnz;
    row_nnz.reserve(static_cast<std::size_t>(pattern.numRows()));

    for (GlobalIndex row = 0; row < pattern.numRows(); ++row) {
        row_nnz.push_back(pattern.getRowNnz(row));
    }

    return row_nnz;
}

std::vector<GlobalIndex> SparsityAnalyzer::findConnectedComponents(
    const SparsityPattern& pattern) const {

    GlobalIndex n = pattern.numRows();
    std::vector<GlobalIndex> component(static_cast<std::size_t>(n), -1);
    GlobalIndex current_component = 0;

    for (GlobalIndex start = 0; start < n; ++start) {
        if (component[static_cast<std::size_t>(start)] >= 0) continue;

        // BFS from this node
        std::queue<GlobalIndex> queue;
        queue.push(start);
        component[static_cast<std::size_t>(start)] = current_component;

        while (!queue.empty()) {
            GlobalIndex node = queue.front();
            queue.pop();

            // Visit neighbors (symmetric treatment for undirected graph)
            if (pattern.isFinalized()) {
                auto row_span = pattern.getRowSpan(node);
                for (GlobalIndex neighbor : row_span) {
                    if (neighbor < n &&
                        component[static_cast<std::size_t>(neighbor)] < 0) {
                        component[static_cast<std::size_t>(neighbor)] = current_component;
                        queue.push(neighbor);
                    }
                }
            }

            // Also check column entries (for non-symmetric patterns)
            for (GlobalIndex row = 0; row < n; ++row) {
                if (pattern.hasEntry(row, node) &&
                    component[static_cast<std::size_t>(row)] < 0) {
                    component[static_cast<std::size_t>(row)] = current_component;
                    queue.push(row);
                }
            }
        }

        ++current_component;
    }

    return component;
}

GlobalIndex SparsityAnalyzer::estimateFillInSimple(
    const SparsityPattern& pattern) const {

    // Simple heuristic based on average row connectivity
    GlobalIndex n = pattern.numRows();
    GlobalIndex nnz = pattern.getNnz();

    if (n == 0 || nnz == 0) return 0;

    double avg_nnz = static_cast<double>(nnz) / static_cast<double>(n);

    // Heuristic: fill-in is approximately (avg_nnz)^2 * n / 2 for sparse matrices
    // This is a rough estimate based on typical FEM patterns
    double estimated = avg_nnz * avg_nnz * static_cast<double>(n) * 0.3;

    // Clamp to reasonable bounds
    GlobalIndex max_fill = n * n - nnz;
    GlobalIndex result = static_cast<GlobalIndex>(estimated);
    return std::min(result, max_fill);
}

bool SparsityAnalyzer::checkBlockPattern(
    const SparsityPattern& pattern,
    GlobalIndex block_size) const {

    FE_CHECK_ARG(block_size > 0, "Block size must be positive");

    if (block_size == 1) {
        return true;
    }

    const SparsityPattern* p = &pattern;
    std::optional<SparsityPattern> finalized;
    if (!pattern.isFinalized()) {
        finalized = pattern.cloneFinalized();
        p = &(*finalized);
    }

    const GlobalIndex n = p->numRows();
    if (!p->isSquare() || p->numCols() != n) {
        return false;
    }
    if (n % block_size != 0) {
        return false;
    }

    const GlobalIndex n_blocks = n / block_size;
    FE_CHECK_ARG(n_blocks > 0, "Invalid block decomposition");

    // Strict block compatibility: any nonzero block must be fully populated.
    // This is the prerequisite for lossless representation in dense-block
    // formats (e.g., MatBAIJ), and helps avoid false positives in detection.
    const GlobalIndex full_block_nnz = block_size * block_size;
    std::vector<GlobalIndex> block_counts(
        static_cast<std::size_t>(n_blocks * n_blocks), 0);

    for (GlobalIndex row = 0; row < n; ++row) {
        const GlobalIndex br = row / block_size;
        const auto row_span = p->getRowSpan(row);
        for (GlobalIndex col : row_span) {
            const GlobalIndex bc = col / block_size;
            const std::size_t idx = static_cast<std::size_t>(br * n_blocks + bc);
            ++block_counts[idx];
        }
    }

    for (GlobalIndex cnt : block_counts) {
        if (cnt != 0 && cnt != full_block_nnz) {
            return false;
        }
    }

    return true;
}

// ============================================================================
// Convenience Functions
// ============================================================================

GlobalIndex computeBandwidth(const SparsityPattern& pattern) {
    SparsityAnalyzer analyzer;
    return analyzer.computeBandwidth(pattern).bandwidth;
}

bool isSymmetric(const SparsityPattern& pattern) {
    SparsityAnalyzer analyzer;
    return analyzer.checkSymmetry(pattern).is_symmetric;
}

double computeFillRatio(const SparsityPattern& pattern) {
    if (pattern.numRows() == 0 || pattern.numCols() == 0) {
        return 0.0;
    }
    return static_cast<double>(pattern.getNnz()) /
           (static_cast<double>(pattern.numRows()) *
            static_cast<double>(pattern.numCols()));
}

void printAnalysis(const SparsityPattern& pattern, std::ostream& out) {
    SparsityAnalyzer analyzer;
    auto report = analyzer.analyze(pattern);
    report.print(out);
}

std::string getPatternSummary(const SparsityPattern& pattern) {
    SparsityAnalyzer analyzer;
    auto report = analyzer.analyzeQuick(pattern);
    return report.summary();
}

} // namespace sparsity
} // namespace FE
} // namespace svmp

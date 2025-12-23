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

#include "SparsityOptimizer.h"
#include "SparsityOps.h"
#include "DistributedSparsityPattern.h"
#include <chrono>
#include <array>
#include <algorithm>
#include <numeric>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <limits>

#ifdef SVMP_HAS_METIS
#include <metis.h>
#endif

#ifdef SVMP_HAS_PARMETIS
#include <parmetis.h>
#endif

namespace svmp {
namespace FE {
namespace sparsity {

// ============================================================================
// Construction
// ============================================================================

SparsityOptimizer::SparsityOptimizer(const OptimizationOptions& options)
    : options_(options)
{
}

void SparsityOptimizer::setOptions(const OptimizationOptions& options) {
    options_ = options;
}

// ============================================================================
// Optimization Methods
// ============================================================================

OptimizationResult SparsityOptimizer::optimize(
    const SparsityPattern& pattern,
    const OptimizationOptions& options) const
{
    return runOptimization(pattern, options);
}

OptimizationResult SparsityOptimizer::optimize(const SparsityPattern& pattern) const {
    return runOptimization(pattern, options_);
}

std::pair<SparsityPattern, OptimizationResult> SparsityOptimizer::optimizeAndApply(
    const SparsityPattern& pattern,
    const OptimizationOptions& options) const
{
    auto result = optimize(pattern, options);

    if (result.permutation.empty() || !result.improved()) {
        // No improvement or no permutation - return copy of original
        SparsityPattern copy(pattern);
        if (!copy.isFinalized()) {
            copy.finalize();
        }
        return {std::move(copy), std::move(result)};
    }

    auto reordered = pattern.permute(result.permutation, result.permutation);
    return {std::move(reordered), std::move(result)};
}

std::pair<SparsityPattern, OptimizationResult> SparsityOptimizer::optimizeAndApply(
    const SparsityPattern& pattern) const
{
    return optimizeAndApply(pattern, options_);
}

// ============================================================================
// Goal-Specific Optimization
// ============================================================================

OptimizationResult SparsityOptimizer::optimizeForBandwidth(
    const SparsityPattern& pattern) const
{
    OptimizationOptions opts = options_;
    opts.goal = OptimizationGoal::MinimizeBandwidth;
    opts.algorithm = ReorderingAlgorithm::ReverseCuthillMcKee;
    return optimize(pattern, opts);
}

OptimizationResult SparsityOptimizer::optimizeForProfile(
    const SparsityPattern& pattern) const
{
    OptimizationOptions opts = options_;
    opts.goal = OptimizationGoal::MinimizeProfile;
    opts.algorithm = ReorderingAlgorithm::ReverseCuthillMcKee;
    return optimize(pattern, opts);
}

OptimizationResult SparsityOptimizer::optimizeForFillIn(
    const SparsityPattern& pattern) const
{
    OptimizationOptions opts = options_;
    opts.goal = OptimizationGoal::MinimizeFillIn;

    // Prefer external libraries for fill-in minimization
    if (hasMetis() && opts.allow_external) {
        opts.algorithm = ReorderingAlgorithm::METIS;
    } else if (hasAMD() && opts.allow_external) {
        opts.algorithm = ReorderingAlgorithm::AMD;
    } else {
        opts.algorithm = ReorderingAlgorithm::MinimumDegree;
    }

    return optimize(pattern, opts);
}

OptimizationResult SparsityOptimizer::optimizeForCache(
    const SparsityPattern& pattern) const
{
    OptimizationOptions opts = options_;
    opts.goal = OptimizationGoal::OptimizeCache;
    // RCM tends to produce good cache locality
    opts.algorithm = ReorderingAlgorithm::ReverseCuthillMcKee;
    return optimize(pattern, opts);
}

OptimizationResult SparsityOptimizer::optimizeForParallel(
    const SparsityPattern& pattern) const
{
    OptimizationOptions opts = options_;
    opts.goal = OptimizationGoal::OptimizeParallel;

    // For parallel, we want good coloring rather than specific reordering
    // But RCM can help group related rows together
    opts.algorithm = ReorderingAlgorithm::ReverseCuthillMcKee;

    auto result = optimize(pattern, opts);

    // Compute coloring information
    GraphSparsity graph(pattern);
    auto coloring = graph.degreeBasedColoring();
    result.num_colors = coloring.num_colors;

    return result;
}

// ============================================================================
// Algorithm-Specific Methods
// ============================================================================

OptimizationResult SparsityOptimizer::applyCuthillMcKee(
    const SparsityPattern& pattern, bool reverse) const
{
    return selectAndRunAlgorithm(pattern,
        reverse ? ReorderingAlgorithm::ReverseCuthillMcKee
                : ReorderingAlgorithm::CuthillMcKee,
        options_);
}

OptimizationResult SparsityOptimizer::applyMinimumDegree(
    const SparsityPattern& pattern) const
{
    return selectAndRunAlgorithm(pattern, ReorderingAlgorithm::MinimumDegree, options_);
}

OptimizationResult SparsityOptimizer::applyNestedDissection(
    const SparsityPattern& pattern) const
{
    if (!options_.allow_external || !hasMetis()) {
        // Fallback to approximate minimum degree
        return selectAndRunAlgorithm(pattern, ReorderingAlgorithm::MinimumDegree, options_);
    }
    return selectAndRunAlgorithm(pattern, ReorderingAlgorithm::NestedDissection, options_);
}

// ============================================================================
// Analysis Methods
// ============================================================================

ReorderingAlgorithm SparsityOptimizer::suggestAlgorithm(
    const SparsityPattern& pattern) const
{
    GlobalIndex n = pattern.numRows();
    GlobalIndex nnz = pattern.getNnz();

    if (n == 0) return ReorderingAlgorithm::None;

    double density = static_cast<double>(nnz) / (static_cast<double>(n) * static_cast<double>(n));

    // Very small matrices: natural ordering is fine
    if (n < 100) {
        return ReorderingAlgorithm::Natural;
    }

    // Dense matrices: limited benefit from reordering
    if (density > 0.5) {
        return ReorderingAlgorithm::Natural;
    }

    // Large sparse matrices: prefer fill-reducing orderings
    if (n > 10000 && density < 0.01) {
        if (hasMetis() && options_.allow_external) {
            return ReorderingAlgorithm::METIS;
        }
        if (hasAMD() && options_.allow_external) {
            return ReorderingAlgorithm::AMD;
        }
        return ReorderingAlgorithm::MinimumDegree;
    }

    // Medium-sized sparse matrices: RCM is usually good
    return ReorderingAlgorithm::ReverseCuthillMcKee;
}

std::vector<OptimizationResult> SparsityOptimizer::compareAlgorithms(
    const SparsityPattern& pattern,
    std::span<const ReorderingAlgorithm> algorithms) const
{
    std::vector<OptimizationResult> results;
    results.reserve(algorithms.size());

    for (ReorderingAlgorithm algo : algorithms) {
        results.push_back(selectAndRunAlgorithm(pattern, algo, options_));
    }

    return results;
}

double SparsityOptimizer::estimatePotential(const SparsityPattern& pattern) const {
    GlobalIndex n = pattern.numRows();
    if (n < 10) return 0.0;  // Too small to benefit

    GlobalIndex bandwidth = pattern.computeBandwidth();
    auto stats = pattern.computeStats();

    // Estimate potential based on how far bandwidth is from ideal
    // Ideal bandwidth for many FEM matrices is O(sqrt(n))
    double ideal_bandwidth = std::sqrt(static_cast<double>(n));
    double current_bandwidth = static_cast<double>(bandwidth);

    if (current_bandwidth <= ideal_bandwidth * 1.5) {
        return 0.1;  // Already near optimal
    }

    double potential = 1.0 - (ideal_bandwidth / current_bandwidth);
    return std::min(1.0, std::max(0.0, potential));
}

#if FE_HAS_MPI
std::vector<GlobalIndex> SparsityOptimizer::parmetisNodeNDPermutation(
    const DistributedSparsityPattern& pattern,
    MPI_Comm comm) const
{
#ifndef SVMP_HAS_PARMETIS
    (void)pattern;
    (void)comm;
    FE_THROW(NotImplementedException,
             "ParMETIS support is not enabled (SVMP_HAS_PARMETIS not defined)");
#else
    FE_CHECK_ARG(comm != MPI_COMM_NULL, "MPI communicator must not be MPI_COMM_NULL");
    FE_THROW_IF(!pattern.isFinalized(), InvalidArgumentException,
                "ParMETIS ordering requires a finalized distributed pattern");
    FE_CHECK_ARG(pattern.isSquare(), "ParMETIS ordering requires a square pattern");

    int my_rank = 0;
    int n_ranks = 1;
    MPI_Comm_rank(comm, &my_rank);
    MPI_Comm_size(comm, &n_ranks);

    const IndexRange owned = pattern.ownedRows();
    const GlobalIndex n_global = pattern.globalRows();
    FE_CHECK_ARG(owned.first >= 0 && owned.last >= owned.first, "Invalid owned row range");
    FE_CHECK_ARG(n_global >= 0, "Invalid global size");

    FE_THROW_IF(owned.size() < 1, InvalidArgumentException,
                "ParMETIS requires each rank to own at least one vertex");

    // Gather global owned ranges to build/validate vtxdist.
    std::vector<GlobalIndex> ranges(static_cast<std::size_t>(2 * n_ranks), 0);
    const GlobalIndex send_range[2] = {owned.first, owned.last};
    MPI_Allgather(send_range, 2, MPI_INT64_T,
                  ranges.data(), 2, MPI_INT64_T, comm);

    std::vector<GlobalIndex> offsets(static_cast<std::size_t>(n_ranks) + 1, 0);
    FE_CHECK_ARG(ranges[0] == 0, "ParMETIS requires rank 0 ownership to start at 0");
    for (int r = 0; r < n_ranks; ++r) {
        const GlobalIndex first = ranges[static_cast<std::size_t>(2 * r)];
        const GlobalIndex last = ranges[static_cast<std::size_t>(2 * r + 1)];
        FE_CHECK_ARG(first >= 0 && last >= first, "Invalid owned range gathered for rank " + std::to_string(r));
        if (r == 0) {
            offsets[0] = first;
        } else {
            FE_CHECK_ARG(first == offsets[static_cast<std::size_t>(r)],
                         "ParMETIS requires contiguous ownership ranges across ranks");
        }
        FE_CHECK_ARG(last > first, "ParMETIS requires each rank to own at least one vertex");
        offsets[static_cast<std::size_t>(r) + 1] = last;
    }
    FE_CHECK_ARG(offsets.front() == 0, "ParMETIS ownership must start at 0");
    FE_CHECK_ARG(offsets.back() == n_global, "ParMETIS ownership does not match global size");

    FE_THROW_IF(n_global > static_cast<GlobalIndex>(std::numeric_limits<idx_t>::max()),
                InvalidArgumentException,
                "ParMETIS build is configured for 32-bit indices; problem is too large");

    std::vector<idx_t> vtxdist(static_cast<std::size_t>(n_ranks) + 1, 0);
    for (int r = 0; r <= n_ranks; ++r) {
        vtxdist[static_cast<std::size_t>(r)] = static_cast<idx_t>(offsets[static_cast<std::size_t>(r)]);
    }

    const GlobalIndex local_n_global = owned.size();
    FE_THROW_IF(local_n_global > static_cast<GlobalIndex>(std::numeric_limits<idx_t>::max()),
                InvalidArgumentException,
                "ParMETIS build is configured for 32-bit indices; local problem is too large");

    const idx_t local_n = static_cast<idx_t>(local_n_global);
    std::vector<idx_t> xadj(static_cast<std::size_t>(local_n) + 1, 0);
    std::vector<idx_t> adjncy;
    adjncy.reserve(static_cast<std::size_t>(pattern.getLocalNnz()));

    xadj[0] = 0;
    for (GlobalIndex global_row = owned.first; global_row < owned.last; ++global_row) {
        const auto cols = pattern.getOwnedRowGlobalCols(global_row);
        for (GlobalIndex col : cols) {
            if (col == global_row) continue;
            FE_THROW_IF(col < 0 || col >= n_global, InvalidArgumentException,
                        "Distributed adjacency contains out-of-range column index");
            FE_THROW_IF(col > static_cast<GlobalIndex>(std::numeric_limits<idx_t>::max()),
                        InvalidArgumentException,
                        "ParMETIS build is configured for 32-bit indices; adjacency index is too large");
            adjncy.push_back(static_cast<idx_t>(col));
        }
        const std::size_t local_row = static_cast<std::size_t>(global_row - owned.first);
        FE_THROW_IF(adjncy.size() > static_cast<std::size_t>(std::numeric_limits<idx_t>::max()),
                    InvalidArgumentException,
                    "ParMETIS build is configured for 32-bit indices; adjacency is too large");
        xadj[local_row + 1] = static_cast<idx_t>(adjncy.size());
    }

    std::vector<idx_t> order(static_cast<std::size_t>(local_n), 0);
    std::vector<idx_t> sizes(static_cast<std::size_t>(2 * n_ranks), 0);

    idx_t numflag = 0; // 0-based
    std::array<idx_t, 4> options{{1, 0, 1, 0}}; // use options, dbglvl=0, seed=1, ipart/psr=0
    MPI_Comm comm_copy = comm;

    const int status = ParMETIS_V3_NodeND(vtxdist.data(),
                                          xadj.data(),
                                          adjncy.data(),
                                          &numflag,
                                          options.data(),
                                          order.data(),
                                          sizes.data(),
                                          &comm_copy);

    FE_THROW_IF(status != METIS_OK, FEException,
                "ParMETIS_V3_NodeND failed with status " + std::to_string(status));

    std::vector<GlobalIndex> local_perm(static_cast<std::size_t>(local_n), 0);
    for (std::size_t i = 0; i < local_perm.size(); ++i) {
        local_perm[i] = static_cast<GlobalIndex>(order[i]);
    }

    // Gather local old->new values into a global vector indexed by old ID.
    std::vector<int> recvcounts(static_cast<std::size_t>(n_ranks), 0);
    std::vector<int> displs(static_cast<std::size_t>(n_ranks), 0);
    for (int r = 0; r < n_ranks; ++r) {
        const GlobalIndex count = offsets[static_cast<std::size_t>(r) + 1] - offsets[static_cast<std::size_t>(r)];
        FE_THROW_IF(count > std::numeric_limits<int>::max(), InvalidArgumentException,
                    "MPI_Allgatherv counts exceed INT_MAX");
        FE_THROW_IF(offsets[static_cast<std::size_t>(r)] > std::numeric_limits<int>::max(), InvalidArgumentException,
                    "MPI_Allgatherv displacements exceed INT_MAX");
        recvcounts[static_cast<std::size_t>(r)] = static_cast<int>(count);
        displs[static_cast<std::size_t>(r)] = static_cast<int>(offsets[static_cast<std::size_t>(r)]);
    }

    FE_THROW_IF(local_n_global > std::numeric_limits<int>::max(), InvalidArgumentException,
                "MPI_Allgatherv sendcount exceeds INT_MAX");

    std::vector<GlobalIndex> global_perm(static_cast<std::size_t>(n_global), 0);
    MPI_Allgatherv(local_perm.data(),
                   static_cast<int>(local_n_global),
                   MPI_INT64_T,
                   global_perm.data(),
                   recvcounts.data(),
                   displs.data(),
                   MPI_INT64_T,
                   comm);

    return global_perm;
#endif
}
#endif

// ============================================================================
// External Library Availability
// ============================================================================

bool SparsityOptimizer::hasMetis() noexcept {
    // Check at compile time if METIS is available
    #ifdef SVMP_HAS_METIS
    return true;
    #else
    return false;
    #endif
}

bool SparsityOptimizer::hasAMD() noexcept {
    // Check at compile time if AMD (SuiteSparse) is available
    #ifdef SVMP_HAS_SUITESPARSE
    return true;
    #else
    return false;
    #endif
}

bool SparsityOptimizer::hasScotch() noexcept {
    // Check at compile time if Scotch is available
    #ifdef SVMP_HAS_SCOTCH
    return true;
    #else
    return false;
    #endif
}

bool SparsityOptimizer::hasParMetis() noexcept {
    // Check at compile time if ParMETIS is available
    #ifdef SVMP_HAS_PARMETIS
    return true;
    #else
    return false;
    #endif
}

std::vector<ReorderingAlgorithm> SparsityOptimizer::availableAlgorithms() {
    std::vector<ReorderingAlgorithm> algos = {
        ReorderingAlgorithm::None,
        ReorderingAlgorithm::Natural,
        ReorderingAlgorithm::CuthillMcKee,
        ReorderingAlgorithm::ReverseCuthillMcKee,
        ReorderingAlgorithm::MinimumDegree
    };

    if (hasAMD()) {
        algos.push_back(ReorderingAlgorithm::AMD);
    }
    if (hasMetis()) {
        algos.push_back(ReorderingAlgorithm::NestedDissection);
        algos.push_back(ReorderingAlgorithm::METIS);
    }
    if (hasScotch()) {
        algos.push_back(ReorderingAlgorithm::Scotch);
    }

    return algos;
}

std::string SparsityOptimizer::algorithmName(ReorderingAlgorithm algo) {
    switch (algo) {
        case ReorderingAlgorithm::None: return "None";
        case ReorderingAlgorithm::Natural: return "Natural";
        case ReorderingAlgorithm::CuthillMcKee: return "Cuthill-McKee";
        case ReorderingAlgorithm::ReverseCuthillMcKee: return "Reverse Cuthill-McKee";
        case ReorderingAlgorithm::MinimumDegree: return "Minimum Degree";
        case ReorderingAlgorithm::AMD: return "AMD (Approximate Minimum Degree)";
        case ReorderingAlgorithm::NestedDissection: return "Nested Dissection";
        case ReorderingAlgorithm::METIS: return "METIS";
        case ReorderingAlgorithm::Scotch: return "Scotch";
        case ReorderingAlgorithm::Auto: return "Auto";
        default: return "Unknown";
    }
}

// ============================================================================
// Internal Implementation
// ============================================================================

OptimizationResult SparsityOptimizer::runOptimization(
    const SparsityPattern& pattern,
    const OptimizationOptions& options) const
{
    auto start_time = std::chrono::high_resolution_clock::now();

    OptimizationResult result;
    result.original_bandwidth = pattern.computeBandwidth();
    result.original_profile = computePatternProfile(pattern);

    GlobalIndex n = pattern.numRows();
    if (n == 0) {
        result.algorithm_used = ReorderingAlgorithm::None;
        return result;
    }

    // Determine algorithm
    ReorderingAlgorithm algo = options.algorithm;
    if (algo == ReorderingAlgorithm::Auto) {
        const auto suggest_for_options = [&](bool allow_external) -> ReorderingAlgorithm {
            const GlobalIndex n_rows = pattern.numRows();
            const GlobalIndex nnz = pattern.getNnz();

            if (n_rows == 0) return ReorderingAlgorithm::None;
            const double density = static_cast<double>(nnz) /
                                   (static_cast<double>(n_rows) * static_cast<double>(n_rows));

            if (n_rows < 100) return ReorderingAlgorithm::Natural;
            if (density > 0.5) return ReorderingAlgorithm::Natural;

            if (n_rows > 10000 && density < 0.01) {
                if (allow_external && hasMetis()) return ReorderingAlgorithm::METIS;
                if (allow_external && hasAMD()) return ReorderingAlgorithm::AMD;
                return ReorderingAlgorithm::MinimumDegree;
            }

            return ReorderingAlgorithm::ReverseCuthillMcKee;
        };

        algo = suggest_for_options(options.allow_external);
    }

    // Run the selected algorithm
    result = selectAndRunAlgorithm(pattern, algo, options);
    result.original_bandwidth = pattern.computeBandwidth();
    result.original_profile = computePatternProfile(pattern);

    auto end_time = std::chrono::high_resolution_clock::now();
    result.optimization_time_sec = std::chrono::duration<double>(
        end_time - start_time).count();

    if (options.verbose) {
        std::cout << "Optimization completed in " << std::fixed << std::setprecision(3)
                  << result.optimization_time_sec << " seconds\n";
        std::cout << "Algorithm: " << algorithmName(result.algorithm_used) << "\n";
        std::cout << "Bandwidth: " << result.original_bandwidth << " -> "
                  << result.optimized_bandwidth << "\n";
        std::cout << "Profile: " << result.original_profile << " -> "
                  << result.optimized_profile << "\n";
    }

    return result;
}

OptimizationResult SparsityOptimizer::selectAndRunAlgorithm(
    const SparsityPattern& pattern,
    ReorderingAlgorithm algorithm,
    const OptimizationOptions& options) const
{
    OptimizationResult result;
    result.algorithm_used = algorithm;

    GlobalIndex n = pattern.numRows();
    if (n == 0) return result;

    GraphSparsity graph(pattern, true);

    switch (algorithm) {
        case ReorderingAlgorithm::None:
            // Identity permutation
            result.permutation.resize(static_cast<std::size_t>(n));
            std::iota(result.permutation.begin(), result.permutation.end(), GlobalIndex{0});
            break;

        case ReorderingAlgorithm::Natural:
            result.permutation.resize(static_cast<std::size_t>(n));
            std::iota(result.permutation.begin(), result.permutation.end(), GlobalIndex{0});
            break;

        case ReorderingAlgorithm::CuthillMcKee:
            result.permutation = graph.cuthillMcKee();
            break;

        case ReorderingAlgorithm::ReverseCuthillMcKee:
            result.permutation = graph.reverseCuthillMcKee();
            break;

        case ReorderingAlgorithm::MinimumDegree:
            result.permutation = graph.approximateMinimumDegree();
            break;

        case ReorderingAlgorithm::AMD:
            // Fallback to builtin minimum degree if AMD not available
            if (!options.allow_external || !hasAMD()) {
                result.permutation = graph.approximateMinimumDegree();
                result.algorithm_used = ReorderingAlgorithm::MinimumDegree;
            } else {
                // Would call external AMD here
                result.permutation = graph.approximateMinimumDegree();
            }
            break;

        case ReorderingAlgorithm::NestedDissection:
        case ReorderingAlgorithm::METIS:
            // Fallback to minimum degree if METIS not available
            if (!options.allow_external || !hasMetis()) {
                result.permutation = graph.approximateMinimumDegree();
                result.algorithm_used = ReorderingAlgorithm::MinimumDegree;
            } else {
                #ifdef SVMP_HAS_METIS
                const auto& sym_pattern = graph.getPattern();
                const GlobalIndex nn = sym_pattern.numRows();

                if (nn <= 0) {
                    result.algorithm_used = ReorderingAlgorithm::None;
                    break;
                }

                FE_THROW_IF(nn > static_cast<GlobalIndex>(std::numeric_limits<idx_t>::max()),
                            InvalidArgumentException,
                            "METIS requires 32-bit vertex indices; problem is too large");

                const idx_t nvtxs = static_cast<idx_t>(nn);
                std::vector<idx_t> xadj(static_cast<std::size_t>(nvtxs) + 1);
                std::vector<idx_t> adjncy;

                xadj[0] = 0;
                adjncy.reserve(static_cast<std::size_t>(sym_pattern.getNnz()));

                for (GlobalIndex row = 0; row < nn; ++row) {
                    for (GlobalIndex col : sym_pattern.getRowSpan(row)) {
                        if (col == row) continue;
                        FE_THROW_IF(col < 0 || col >= nn, InvalidArgumentException,
                                    "METIS graph adjacency contains out-of-range column index");
                        FE_THROW_IF(col > static_cast<GlobalIndex>(std::numeric_limits<idx_t>::max()),
                                    InvalidArgumentException,
                                    "METIS requires 32-bit adjacency indices; problem is too large");
                        adjncy.push_back(static_cast<idx_t>(col));
                    }
                    xadj[static_cast<std::size_t>(row) + 1] =
                        static_cast<idx_t>(adjncy.size());
                }

                std::array<idx_t, METIS_NOPTIONS> metis_options{};
                METIS_SetDefaultOptions(metis_options.data());
                metis_options[METIS_OPTION_NUMBERING] = 0;  // 0-based
                metis_options[METIS_OPTION_SEED] = 0;       // deterministic
                metis_options[METIS_OPTION_DBGLVL] = 0;

                std::vector<idx_t> perm(static_cast<std::size_t>(nvtxs));
                std::vector<idx_t> iperm(static_cast<std::size_t>(nvtxs));

                idx_t nvtxs_copy = nvtxs;
                const int status = METIS_NodeND(&nvtxs_copy,
                                                xadj.data(),
                                                adjncy.data(),
                                                /*vwgt=*/nullptr,
                                                metis_options.data(),
                                                perm.data(),
                                                iperm.data());
                if (status != METIS_OK) {
                    result.permutation = graph.approximateMinimumDegree();
                    result.algorithm_used = ReorderingAlgorithm::MinimumDegree;
                } else {
                    result.permutation.resize(static_cast<std::size_t>(nvtxs));
                    for (std::size_t i = 0; i < result.permutation.size(); ++i) {
                        result.permutation[i] = static_cast<GlobalIndex>(iperm[i]);
                    }
                }
                #else
                result.permutation = graph.approximateMinimumDegree();
                result.algorithm_used = ReorderingAlgorithm::MinimumDegree;
                #endif
            }
            break;

        case ReorderingAlgorithm::Scotch:
            // Fallback to RCM if Scotch not available
            if (!options.allow_external || !hasScotch()) {
                result.permutation = graph.reverseCuthillMcKee();
                result.algorithm_used = ReorderingAlgorithm::ReverseCuthillMcKee;
            } else {
                // Would call external Scotch here
                result.permutation = graph.reverseCuthillMcKee();
            }
            break;

        case ReorderingAlgorithm::Auto:
            // Should have been resolved before this point
            result.permutation = graph.reverseCuthillMcKee();
            result.algorithm_used = ReorderingAlgorithm::ReverseCuthillMcKee;
            break;
    }

    // Compute inverse permutation
    result.inverse_permutation = invertPermutation(result.permutation);

    // Compute statistics on reordered pattern
    computeStatistics(result, pattern, result.permutation);

    return result;
}

void SparsityOptimizer::computeStatistics(OptimizationResult& result,
                                          const SparsityPattern& pattern,
                                          const std::vector<GlobalIndex>& perm) const
{
    // Apply permutation to compute new bandwidth/profile
    auto reordered = pattern.permute(perm, perm);

    result.optimized_bandwidth = reordered.computeBandwidth();
    result.optimized_profile = computePatternProfile(reordered);

    // Estimate fill reduction
    if (result.original_bandwidth > 0) {
        result.fill_reduction = static_cast<double>(result.optimized_bandwidth) /
                               static_cast<double>(result.original_bandwidth);
    }
}

// ============================================================================
// Convenience Functions
// ============================================================================

SparsityPattern optimizePattern(const SparsityPattern& pattern) {
    SparsityOptimizer optimizer;
    auto [optimized, result] = optimizer.optimizeAndApply(pattern);
    return optimized;
}

SparsityPattern optimizeForBandwidth(const SparsityPattern& pattern) {
    SparsityOptimizer optimizer;
    auto result = optimizer.optimizeForBandwidth(pattern);
    return pattern.permute(result.permutation, result.permutation);
}

SparsityPattern optimizeForFillIn(const SparsityPattern& pattern) {
    SparsityOptimizer optimizer;
    auto result = optimizer.optimizeForFillIn(pattern);
    return pattern.permute(result.permutation, result.permutation);
}

std::vector<GlobalIndex> getOptimalPermutation(const SparsityPattern& pattern,
                                                OptimizationGoal goal)
{
    OptimizationOptions opts;
    opts.goal = goal;

    SparsityOptimizer optimizer(opts);
    auto result = optimizer.optimize(pattern, opts);
    return result.permutation;
}

void printOptimizationReport(const OptimizationResult& result, std::ostream& out) {
    out << "=== Sparsity Optimization Report ===\n";
    out << "Algorithm: " << SparsityOptimizer::algorithmName(result.algorithm_used) << "\n";
    out << "\n";
    out << "Bandwidth:\n";
    out << "  Original:  " << result.original_bandwidth << "\n";
    out << "  Optimized: " << result.optimized_bandwidth << "\n";
    out << "  Reduction: " << std::fixed << std::setprecision(1)
        << (1.0 - result.bandwidthReduction()) * 100.0 << "%\n";
    out << "\n";
    out << "Profile:\n";
    out << "  Original:  " << result.original_profile << "\n";
    out << "  Optimized: " << result.optimized_profile << "\n";
    out << "  Reduction: " << std::fixed << std::setprecision(1)
        << (1.0 - result.profileReduction()) * 100.0 << "%\n";
    out << "\n";
    if (result.num_colors > 0) {
        out << "Coloring: " << result.num_colors << " colors\n";
    }
    out << "Time: " << std::fixed << std::setprecision(3)
        << result.optimization_time_sec << " seconds\n";
    out << "=====================================\n";
}

} // namespace sparsity
} // namespace FE
} // namespace svmp

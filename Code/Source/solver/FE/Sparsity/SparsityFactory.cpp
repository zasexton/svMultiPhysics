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

/**
 * @file SparsityFactory.cpp
 * @brief Implementation of SparsityFactory class
 */

#include "SparsityFactory.h"
#include "SparsityTwoPassBuilder.h"
#include "CompressedSparsity.h"
#include "SparsityOps.h"
#include "Dofs/DofMap.h"

#include <chrono>
#include <algorithm>
#include <numeric>
#include <sstream>
#include <functional>

namespace svmp {
namespace FE {
namespace sparsity {

// ============================================================================
// SparsityFactory Construction
// ============================================================================

SparsityFactory::SparsityFactory(const FactoryOptions& options)
    : options_(options)
{
}

// ============================================================================
// Configuration
// ============================================================================

void SparsityFactory::setOptions(const FactoryOptions& options) {
    options_ = options;
}

void SparsityFactory::clearCache() {
    cache_.clear();
    dist_cache_.clear();
    block_cache_.clear();
    cache_hits_ = 0;
    cache_misses_ = 0;
}

std::pair<std::size_t, std::size_t> SparsityFactory::getCacheStats() const {
    return {cache_hits_, cache_misses_};
}

void SparsityFactory::setConstraints(std::shared_ptr<IConstraintQuery> constraint_query) {
    constraint_query_ = std::move(constraint_query);
}

void SparsityFactory::setConstraints(const dofs::DofConstraints& constraints) {
    constraint_query_ = std::make_shared<DofConstraintsAdapter>(constraints);
}

void SparsityFactory::clearConstraints() {
    constraint_query_.reset();
}

bool SparsityFactory::hasConstraints() const noexcept {
    return constraint_query_ != nullptr && constraint_query_->numConstraints() > 0;
}

// ============================================================================
// Helper to get max DOFs per cell from DofMap
// ============================================================================

namespace {

// Helper to compute max DOFs per cell by sampling
GlobalIndex getMaxDofsPerCell(const dofs::DofMap& dof_map) {
    GlobalIndex n_cells = dof_map.getNumCells();
    if (n_cells == 0) return 0;

    GlobalIndex max_dofs = 0;
    // Sample first few cells or all if small
    GlobalIndex sample_size = std::min(n_cells, GlobalIndex(100));
    for (GlobalIndex i = 0; i < sample_size; ++i) {
        auto cell_dofs = dof_map.getCellDofs(i);
        max_dofs = std::max(max_dofs, static_cast<GlobalIndex>(cell_dofs.size()));
    }
    return max_dofs;
}

inline std::size_t hashCombine(std::size_t seed, std::size_t value) {
    return seed ^ (value + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2));
}

std::size_t hashDofMapConnectivity(const dofs::DofMap& dof_map) {
    std::size_t seed = 0;
    seed = hashCombine(seed, std::hash<GlobalIndex>{}(dof_map.getNumDofs()));
    seed = hashCombine(seed, std::hash<GlobalIndex>{}(dof_map.getNumCells()));

    auto offsets = dof_map.getOffsets();
    auto indices = dof_map.getDofIndices();
    seed = hashCombine(seed, std::hash<std::size_t>{}(offsets.size()));
    seed = hashCombine(seed, std::hash<std::size_t>{}(indices.size()));

    for (GlobalIndex v : offsets) {
        seed = hashCombine(seed, std::hash<GlobalIndex>{}(v));
    }
    for (GlobalIndex v : indices) {
        seed = hashCombine(seed, std::hash<GlobalIndex>{}(v));
    }
    return seed;
}

std::size_t hashConstraintQuery(const IConstraintQuery& query) {
    std::size_t seed = 0;
    auto constrained = query.getAllConstrainedDofs();
    std::sort(constrained.begin(), constrained.end());
    seed = hashCombine(seed, std::hash<std::size_t>{}(constrained.size()));

    for (GlobalIndex slave : constrained) {
        seed = hashCombine(seed, std::hash<GlobalIndex>{}(slave));
        auto masters = query.getMasterDofs(slave);
        std::sort(masters.begin(), masters.end());
        seed = hashCombine(seed, std::hash<std::size_t>{}(masters.size()));
        for (GlobalIndex master : masters) {
            seed = hashCombine(seed, std::hash<GlobalIndex>{}(master));
        }
    }
    return seed;
}

} // anonymous namespace

// ============================================================================
// Pattern Creation - DOF Map Based
// ============================================================================

FactoryResult SparsityFactory::create(
    const dofs::DofMap& dof_map,
    const FactoryOptions& options) const
{
    auto start_time = std::chrono::high_resolution_clock::now();

    FactoryResult result;

    if (options.apply_constraints) {
        FE_CHECK_ARG(constraint_query_ != nullptr,
                     "apply_constraints requested but no constraints are configured; "
                     "call SparsityFactory::setConstraints(...) or apply constraints manually");
    }

    // Check cache if enabled
    std::string cache_key;
    if (options.enable_caching) {
        cache_key = computeCacheKey(dof_map, options);
        auto cached = getCachedPattern(cache_key);
        if (cached) {
            result.pattern = std::make_unique<SparsityPattern>(cached->cloneFinalized());
            result.from_cache = true;
            result.cache_key = cache_key;
            ++cache_hits_;

            // Preserve the strategy selection metadata for reporting.
            GlobalIndex n_dofs = dof_map.getNumDofs();
            GlobalIndex n_cells = dof_map.getNumCells();
            GlobalIndex avg_dofs = getMaxDofsPerCell(dof_map);
            GlobalIndex estimated_nnz = estimateNnz(n_cells, avg_dofs);
            result.strategy_used = selectStrategy(n_dofs, estimated_nnz, options);

            auto end_time = std::chrono::high_resolution_clock::now();
            result.construction_time_sec =
                std::chrono::duration<double>(end_time - start_time).count();
            return result;
        }
        ++cache_misses_;
    }

    // Create pattern based on type
    result = createInternal(dof_map, options);
    result.cache_key = cache_key;  // Preserve cache key after createInternal

    // Apply constraints if requested
    if (options.apply_constraints && result.pattern) {
        ConstraintSparsityAugmenter augmenter(constraint_query_);
        AugmentationOptions aug_opts;
        aug_opts.mode = options.constraint_mode;
        aug_opts.ensure_diagonal = options.ensure_diagonal;
        augmenter.setOptions(aug_opts);

        if (options.constraint_mode == AugmentationMode::ReducedSystem) {
            auto reduced = augmenter.buildReducedPattern(*result.pattern);
            result.pattern = std::make_unique<SparsityPattern>(std::move(reduced));
        } else {
            SparsityPattern augmented = *result.pattern;  // Copy reconstructs to Building state
            augmenter.augment(augmented, options.constraint_mode);
            if (options.ensure_diagonal && augmented.isSquare()) {
                augmented.ensureDiagonal();
            }
            if (options.ensure_non_empty_rows) {
                augmented.ensureNonEmptyRows();
            }
            augmented.finalize();
            result.pattern = std::make_unique<SparsityPattern>(std::move(augmented));
        }
    }

    // Optimize if requested
    if (options.optimize && result.pattern) {
        SparsityOptimizer optimizer;
        OptimizationOptions opt_opts;
        opt_opts.goal = options.optimization_goal;

        auto [optimized, opt_result] = optimizer.optimizeAndApply(*result.pattern, opt_opts);
        result.pattern = std::make_unique<SparsityPattern>(std::move(optimized));
        result.optimization_result = std::move(opt_result);
    }

    // Cache result if enabled
    if (options.enable_caching && result.pattern && !result.cache_key.empty()) {
        cachePattern(result.cache_key,
                     std::make_shared<SparsityPattern>(result.pattern->cloneFinalized()));
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    result.construction_time_sec =
        std::chrono::duration<double>(end_time - start_time).count();

    return result;
}

FactoryResult SparsityFactory::create(const dofs::DofMap& dof_map) const {
    return create(dof_map, options_);
}

SparsityPattern SparsityFactory::createFromDofMap(const dofs::DofMap& dof_map) const {
    auto result = create(dof_map);
    if (!result.pattern) {
        FE_THROW(InvalidArgumentException, "Failed to create pattern from DOF map");
    }
    return std::move(*result.pattern);
}

FactoryResult SparsityFactory::createRectangular(
    const dofs::DofMap& row_dof_map,
    const dofs::DofMap& col_dof_map,
    const FactoryOptions& options) const
{
    auto start_time = std::chrono::high_resolution_clock::now();

    FactoryResult result;

    GlobalIndex n_rows = row_dof_map.getNumDofs();
    GlobalIndex n_cols = col_dof_map.getNumDofs();
    GlobalIndex n_cells = row_dof_map.getNumCells();
    FE_CHECK_ARG(col_dof_map.getNumCells() == n_cells,
                 "Row and column DOF maps must have the same number of cells");

    // Estimate NNZ for strategy selection
    GlobalIndex avg_row_dofs = getMaxDofsPerCell(row_dof_map);
    GlobalIndex avg_col_dofs = getMaxDofsPerCell(col_dof_map);
    GlobalIndex estimated_nnz = n_cells * avg_row_dofs * avg_col_dofs;

    ConstructionStrategy strategy = selectStrategy(n_rows, estimated_nnz, options);
    result.strategy_used = strategy;

    if (strategy == ConstructionStrategy::TwoPass) {
        // Use two-pass builder for large problems
        TwoPassBuildOptions tp_opts;
        tp_opts.num_threads = 1;
        tp_opts.ensure_diagonal = options.ensure_diagonal;
        tp_opts.ensure_non_empty_rows = options.ensure_non_empty_rows;

        SparsityTwoPassBuilder builder(n_rows, n_cols, tp_opts);

        // Count pass
        for (GlobalIndex cell = 0; cell < n_cells; ++cell) {
            auto row_dofs = row_dof_map.getCellDofs(cell);
            auto col_dofs = col_dof_map.getCellDofs(cell);
            builder.countElementCouplings(row_dofs, col_dofs);
        }

        builder.finalizeCount();

        // Fill pass
        for (GlobalIndex cell = 0; cell < n_cells; ++cell) {
            auto row_dofs = row_dof_map.getCellDofs(cell);
            auto col_dofs = col_dof_map.getCellDofs(cell);
            builder.addElementCouplings(row_dofs, col_dofs);
        }

        result.pattern = std::make_unique<SparsityPattern>(builder.finalize());
    } else {
        // Standard construction
        SparsityPattern pattern(n_rows, n_cols);

        for (GlobalIndex cell = 0; cell < n_cells; ++cell) {
            auto row_dofs = row_dof_map.getCellDofs(cell);
            auto col_dofs = col_dof_map.getCellDofs(cell);
            for (GlobalIndex row : row_dofs) {
                for (GlobalIndex col : col_dofs) {
                    pattern.addEntry(row, col);
                }
            }
        }

        if (options.ensure_diagonal && n_rows == n_cols) {
            pattern.ensureDiagonal();
        }
        if (options.ensure_non_empty_rows) {
            pattern.ensureNonEmptyRows();
        }

        pattern.finalize();
        result.pattern = std::make_unique<SparsityPattern>(std::move(pattern));
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    result.construction_time_sec =
        std::chrono::duration<double>(end_time - start_time).count();

    return result;
}

FactoryResult SparsityFactory::createRectangular(
    const dofs::DofMap& row_dof_map,
    const dofs::DofMap& col_dof_map) const
{
    return createRectangular(row_dof_map, col_dof_map, options_);
}

// ============================================================================
// Pattern Creation - Block Structured
// ============================================================================

FactoryResult SparsityFactory::createBlockPattern(
    std::span<const dofs::DofMap*> field_dof_maps,
    std::span<const std::vector<bool>> coupling_matrix,
    const FactoryOptions& options) const
{
    auto start_time = std::chrono::high_resolution_clock::now();

    FactoryResult result;

    FE_CHECK_ARG(!field_dof_maps.empty(),
                "Must provide at least one field DOF map");

    const std::size_t n_fields = field_dof_maps.size();

    if (!coupling_matrix.empty()) {
        FE_CHECK_ARG(coupling_matrix.size() == n_fields,
                    "Coupling matrix rows must match number of fields");
    }

    std::string cache_key;
    if (options.enable_caching) {
        cache_key = computeBlockCacheKey(field_dof_maps, coupling_matrix, options);
        auto cached_monolithic = getCachedPattern(cache_key);
        auto cached_block = getCachedBlockPattern(cache_key);
        if (cached_monolithic && cached_block) {
            result.pattern = std::make_unique<SparsityPattern>(cached_monolithic->cloneFinalized());
            result.block_pattern = std::make_unique<BlockSparsity>(cached_block->cloneFinalized());
            result.from_cache = true;
            result.cache_key = cache_key;
            ++cache_hits_;

            auto end_time = std::chrono::high_resolution_clock::now();
            result.construction_time_sec =
                std::chrono::duration<double>(end_time - start_time).count();
            return result;
        }
        ++cache_misses_;
        result.cache_key = cache_key;
    }

    // Compute total DOFs and offsets
    std::vector<GlobalIndex> field_offsets(n_fields + 1, 0);
    for (std::size_t i = 0; i < n_fields; ++i) {
        field_offsets[i + 1] = field_offsets[i] + field_dof_maps[i]->getNumDofs();
    }
    GlobalIndex total_dofs = field_offsets[n_fields];

    // Create monolithic pattern
    SparsityPattern pattern(total_dofs, total_dofs);

    // Build pattern for each field pair
    for (std::size_t row_field = 0; row_field < n_fields; ++row_field) {
        for (std::size_t col_field = 0; col_field < n_fields; ++col_field) {
            // Check if fields are coupled
            bool coupled = coupling_matrix.empty();
            if (!coupling_matrix.empty()) {
                const auto& row = coupling_matrix[row_field];
                if (col_field < row.size()) {
                    coupled = row[col_field];
                }
            }

            if (!coupled) continue;

            const auto* row_map = field_dof_maps[row_field];
            const auto* col_map = field_dof_maps[col_field];
            GlobalIndex row_offset = field_offsets[row_field];
            GlobalIndex col_offset = field_offsets[col_field];

            // Add couplings from element connectivity
            GlobalIndex n_cells = row_map->getNumCells();
            FE_CHECK_ARG(col_map->getNumCells() == n_cells,
                         "Field DOF maps must have the same number of cells");
            for (GlobalIndex cell = 0; cell < n_cells; ++cell) {
                auto row_dofs = row_map->getCellDofs(cell);
                auto col_dofs = col_map->getCellDofs(cell);

                for (GlobalIndex local_row : row_dofs) {
                    GlobalIndex global_row = row_offset + local_row;
                    for (GlobalIndex local_col : col_dofs) {
                        GlobalIndex global_col = col_offset + local_col;
                        pattern.addEntry(global_row, global_col);
                    }
                }
            }
        }
    }

    // Ensure diagonal if requested
    if (options.ensure_diagonal) {
        for (GlobalIndex i = 0; i < total_dofs; ++i) {
            pattern.addEntry(i, i);
        }
    }

    if (options.ensure_non_empty_rows) {
        pattern.ensureNonEmptyRows();
    }

    pattern.finalize();

    // Create block pattern wrapper
    result.pattern = std::make_unique<SparsityPattern>(std::move(pattern));

    // Also create BlockSparsity populated from the monolithic pattern
    std::vector<GlobalIndex> block_sizes(n_fields);
    for (std::size_t i = 0; i < n_fields; ++i) {
        block_sizes[i] = field_dof_maps[i]->getNumDofs();
    }
    result.block_pattern = std::make_unique<BlockSparsity>(
        BlockSparsity::fromMonolithic(*result.pattern, block_sizes, block_sizes));

    if (options.enable_caching && !cache_key.empty() && result.pattern && result.block_pattern) {
        cachePattern(cache_key,
                     std::make_shared<SparsityPattern>(result.pattern->cloneFinalized()));
        cacheBlockPattern(cache_key,
                          std::make_shared<BlockSparsity>(result.block_pattern->cloneFinalized()));
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    result.construction_time_sec =
        std::chrono::duration<double>(end_time - start_time).count();

    return result;
}

FactoryResult SparsityFactory::createBlockPattern(
    std::span<const dofs::DofMap*> field_dof_maps,
    const FactoryOptions& options) const
{
    // Create full coupling matrix
    const std::size_t n_fields = field_dof_maps.size();
    std::vector<std::vector<bool>> full_coupling(n_fields,
                                                  std::vector<bool>(n_fields, true));
    return createBlockPattern(field_dof_maps, full_coupling, options);
}

// ============================================================================
// Pattern Creation - DG
// ============================================================================

FactoryResult SparsityFactory::createDGPattern(
    const dofs::DofMap& dof_map,
    const std::vector<std::pair<GlobalIndex, GlobalIndex>>& face_adjacency,
    const FactoryOptions& options) const
{
    auto start_time = std::chrono::high_resolution_clock::now();

    FactoryResult result;

    GlobalIndex n_dofs = dof_map.getNumDofs();
    GlobalIndex n_cells = dof_map.getNumCells();

    SparsityPattern pattern(n_dofs, n_dofs);

    // Add element self-couplings (all DOFs in element couple with each other)
    for (GlobalIndex cell = 0; cell < n_cells; ++cell) {
        auto cell_dofs = dof_map.getCellDofs(cell);
        for (GlobalIndex row : cell_dofs) {
            for (GlobalIndex col : cell_dofs) {
                pattern.addEntry(row, col);
            }
        }
    }

    // Add face couplings (neighbor element DOFs)
    for (const auto& [elem1, elem2] : face_adjacency) {
        if (elem1 < 0 || elem1 >= n_cells ||
            elem2 < 0 || elem2 >= n_cells) {
            continue; // Skip invalid elements (e.g., boundary faces)
        }

        auto dofs1 = dof_map.getCellDofs(elem1);
        auto dofs2 = dof_map.getCellDofs(elem2);

        // Add bidirectional coupling
        for (GlobalIndex row : dofs1) {
            for (GlobalIndex col : dofs2) {
                pattern.addEntry(row, col);
            }
        }
        for (GlobalIndex row : dofs2) {
            for (GlobalIndex col : dofs1) {
                pattern.addEntry(row, col);
            }
        }
    }

    // Ensure diagonal if requested
    if (options.ensure_diagonal) {
        for (GlobalIndex i = 0; i < n_dofs; ++i) {
            pattern.addEntry(i, i);
        }
    }

    if (options.ensure_non_empty_rows) {
        pattern.ensureNonEmptyRows();
    }

    pattern.finalize();
    result.pattern = std::make_unique<SparsityPattern>(std::move(pattern));

    auto end_time = std::chrono::high_resolution_clock::now();
    result.construction_time_sec =
        std::chrono::duration<double>(end_time - start_time).count();

    return result;
}

// ============================================================================
// Pattern Creation - Distributed
// ============================================================================

FactoryResult SparsityFactory::createDistributed(
    const dofs::DofMap& dof_map,
    std::pair<GlobalIndex, GlobalIndex> owned_range,
    GlobalIndex global_size,
    const FactoryOptions& options) const
{
    auto start_time = std::chrono::high_resolution_clock::now();

    FactoryResult result;

    if (options.apply_constraints) {
        FE_CHECK_ARG(constraint_query_ != nullptr,
                     "apply_constraints requested but no constraints are configured; "
                     "call SparsityFactory::setConstraints(...) or apply constraints manually");
    }

    GlobalIndex owned_start = owned_range.first;
    GlobalIndex owned_end = owned_range.second;
    GlobalIndex n_owned = owned_end - owned_start;

    FE_CHECK_ARG(owned_start >= 0 && owned_start <= owned_end,
                "Invalid owned range");
    FE_CHECK_ARG(owned_end <= global_size,
                "Owned range exceeds global size");

    std::string cache_key;
    if (options.enable_caching) {
        cache_key = computeDistributedCacheKey(dof_map, owned_range, global_size, options);
        auto cached = getCachedDistributedPattern(cache_key);
        if (cached) {
            result.distributed_pattern =
                std::make_unique<DistributedSparsityPattern>(cached->cloneFinalized());
            result.from_cache = true;
            result.cache_key = cache_key;
            ++cache_hits_;

            auto end_time = std::chrono::high_resolution_clock::now();
            result.construction_time_sec =
                std::chrono::duration<double>(end_time - start_time).count();
            return result;
        }
        ++cache_misses_;
        result.cache_key = cache_key;
    }

    DistributedSparsityPattern pattern(
        owned_start, n_owned, owned_start, n_owned, global_size, global_size);

    GlobalIndex n_cells = dof_map.getNumCells();

    // Build pattern from element connectivity
    for (GlobalIndex cell = 0; cell < n_cells; ++cell) {
        auto cell_dofs = dof_map.getCellDofs(cell);

        for (GlobalIndex row : cell_dofs) {
            // Only process owned rows
            if (row >= owned_start && row < owned_end) {
                for (GlobalIndex col : cell_dofs) {
                    if (!options.include_ghost_rows) {
                        // Build diagonal block only (owned columns)
                        if (col < owned_start || col >= owned_end) {
                            continue;
                        }
                    }
                    pattern.addEntry(row, col);
                }
            }
        }
    }

    if (options.apply_constraints) {
        ConstraintSparsityAugmenter augmenter(constraint_query_);
        AugmentationOptions aug_opts;
        aug_opts.mode = options.constraint_mode;
        aug_opts.ensure_diagonal = options.ensure_diagonal;
        aug_opts.include_ghost_columns = options.include_ghost_rows;
        augmenter.setOptions(aug_opts);

        if (options.constraint_mode == AugmentationMode::ReducedSystem) {
            // Without an MPI communicator, only support the trivial single-rank
            // case where this rank owns the full system.
            FE_CHECK_ARG(owned_start == 0 && owned_end == global_size,
                         "ReducedSystem mode for distributed patterns requires an MPI-aware "
                         "reduced-pattern builder. Use ConstraintSparsityAugmenter::buildReducedDistributedPattern().");

            pattern.finalize();

            SparsityPattern full(global_size, global_size);
            for (GlobalIndex row = 0; row < global_size; ++row) {
                auto cols = pattern.getOwnedRowGlobalCols(row);
                for (GlobalIndex col : cols) {
                    full.addEntry(row, col);
                }
            }
            full.finalize();

            auto reduced = augmenter.buildReducedPattern(full);
            const GlobalIndex n_reduced = reduced.numRows();

            DistributedSparsityPattern reduced_dist(
                IndexRange{0, n_reduced},
                IndexRange{0, n_reduced},
                n_reduced,
                n_reduced);

            if (n_reduced > 0) {
                for (GlobalIndex row = 0; row < n_reduced; ++row) {
                    auto row_span = reduced.getRowSpan(row);
                    for (GlobalIndex col : row_span) {
                        reduced_dist.addEntry(row, col);
                    }
                }
            }

            if (options.ensure_diagonal) {
                reduced_dist.ensureDiagonal();
            }
            if (options.ensure_non_empty_rows) {
                reduced_dist.ensureNonEmptyRows();
            }

            reduced_dist.finalize();
            result.distributed_pattern = std::make_unique<DistributedSparsityPattern>(std::move(reduced_dist));

            if (options.enable_caching && !cache_key.empty() && result.distributed_pattern) {
                cacheDistributedPattern(
                    cache_key,
                    std::make_shared<DistributedSparsityPattern>(result.distributed_pattern->cloneFinalized()));
            }

            auto end_time = std::chrono::high_resolution_clock::now();
            result.construction_time_sec =
                std::chrono::duration<double>(end_time - start_time).count();
            return result;
        } else {
            augmenter.augment(pattern, options.constraint_mode);
        }
    }

    // Ensure diagonal for owned rows if requested
    if (options.ensure_diagonal) {
        pattern.ensureDiagonal();
    }

    if (options.ensure_non_empty_rows) {
        pattern.ensureNonEmptyRows();
    }

    if (!pattern.isFinalized()) {
        pattern.finalize();
    }

    if (!result.distributed_pattern) {
        result.distributed_pattern = std::make_unique<DistributedSparsityPattern>(std::move(pattern));
    }

    if (options.enable_caching && !cache_key.empty() && result.distributed_pattern) {
        cacheDistributedPattern(
            cache_key,
            std::make_shared<DistributedSparsityPattern>(result.distributed_pattern->cloneFinalized()));
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    result.construction_time_sec =
        std::chrono::duration<double>(end_time - start_time).count();

    return result;
}

FactoryResult SparsityFactory::createDistributed(
    const dofs::DofMap& dof_map,
    std::pair<GlobalIndex, GlobalIndex> owned_range,
    GlobalIndex global_size) const
{
    return createDistributed(dof_map, owned_range, global_size, options_);
}

// ============================================================================
// Pattern Creation - Low Level
// ============================================================================

FactoryResult SparsityFactory::createFromArrays(
    GlobalIndex n_rows,
    GlobalIndex n_cols,
    GlobalIndex n_elements,
    std::span<const GlobalIndex> elem_offsets,
    std::span<const GlobalIndex> elem_dofs,
    const FactoryOptions& options) const
{
    auto start_time = std::chrono::high_resolution_clock::now();

    FactoryResult result;

    FE_CHECK_ARG(n_rows > 0 && n_cols > 0, "Dimensions must be positive");
    FE_CHECK_ARG(static_cast<GlobalIndex>(elem_offsets.size()) == n_elements + 1,
                "Element offsets size mismatch");
    FE_CHECK_ARG(n_elements >= 0, "Number of elements must be non-negative");

    // Estimate NNZ for strategy selection
    GlobalIndex max_dofs_per_elem = 0;
    const auto n_elements_u = static_cast<std::size_t>(n_elements);
    for (std::size_t elem = 0; elem < n_elements_u; ++elem) {
        GlobalIndex dofs_in_elem = elem_offsets[elem + 1] - elem_offsets[elem];
        max_dofs_per_elem = std::max(max_dofs_per_elem, dofs_in_elem);
    }
    GlobalIndex estimated_nnz = n_elements * max_dofs_per_elem * max_dofs_per_elem;

    ConstructionStrategy strategy = selectStrategy(n_rows, estimated_nnz, options);
    result.strategy_used = strategy;

    if (strategy == ConstructionStrategy::TwoPass) {
        TwoPassBuildOptions tp_opts;
        tp_opts.ensure_diagonal = options.ensure_diagonal;
        tp_opts.ensure_non_empty_rows = options.ensure_non_empty_rows;
        SparsityTwoPassBuilder builder(n_rows, n_cols, tp_opts);

        // Count pass
        for (std::size_t elem = 0; elem < n_elements_u; ++elem) {
            GlobalIndex start = elem_offsets[elem];
            GlobalIndex end = elem_offsets[elem + 1];
            FE_CHECK_ARG(start >= 0 && end >= start, "Invalid element offsets");
            const auto n = static_cast<std::size_t>(end - start);
            std::span<const GlobalIndex> dofs_span(elem_dofs.data() + static_cast<std::ptrdiff_t>(start), n);
            builder.countElementCouplings(dofs_span);
        }

        builder.finalizeCount();

        // Fill pass
        for (std::size_t elem = 0; elem < n_elements_u; ++elem) {
            GlobalIndex start = elem_offsets[elem];
            GlobalIndex end = elem_offsets[elem + 1];
            FE_CHECK_ARG(start >= 0 && end >= start, "Invalid element offsets");
            const auto n = static_cast<std::size_t>(end - start);
            std::span<const GlobalIndex> dofs_span(elem_dofs.data() + static_cast<std::ptrdiff_t>(start), n);
            builder.addElementCouplings(dofs_span);
        }

        result.pattern = std::make_unique<SparsityPattern>(builder.finalize());
    } else {
        SparsityPattern pattern(n_rows, n_cols);

        for (std::size_t elem = 0; elem < n_elements_u; ++elem) {
            GlobalIndex start = elem_offsets[elem];
            GlobalIndex end = elem_offsets[elem + 1];
            FE_CHECK_ARG(start >= 0 && end >= start, "Invalid element offsets");

            for (GlobalIndex i = start; i < end; ++i) {
                GlobalIndex row = elem_dofs[static_cast<std::size_t>(i)];
                if (row >= 0 && row < n_rows) {
                    for (GlobalIndex j = start; j < end; ++j) {
                        GlobalIndex col = elem_dofs[static_cast<std::size_t>(j)];
                        if (col >= 0 && col < n_cols) {
                            pattern.addEntry(row, col);
                        }
                    }
                }
            }
        }

        if (options.ensure_diagonal && n_rows == n_cols) {
            pattern.ensureDiagonal();
        }

        if (options.ensure_non_empty_rows) {
            pattern.ensureNonEmptyRows();
        }

        pattern.finalize();
        result.pattern = std::make_unique<SparsityPattern>(std::move(pattern));
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    result.construction_time_sec =
        std::chrono::duration<double>(end_time - start_time).count();

    return result;
}

FactoryResult SparsityFactory::createFromCallback(
    GlobalIndex n_rows,
    GlobalIndex n_cols,
    std::function<std::vector<GlobalIndex>(GlobalIndex row)> row_entries,
    const FactoryOptions& options) const
{
    auto start_time = std::chrono::high_resolution_clock::now();

    FactoryResult result;

    FE_CHECK_ARG(n_rows > 0 && n_cols > 0, "Dimensions must be positive");
    FE_CHECK_ARG(row_entries != nullptr, "Row entries callback required");

    SparsityPattern pattern(n_rows, n_cols);

    for (GlobalIndex row = 0; row < n_rows; ++row) {
        std::vector<GlobalIndex> cols = row_entries(row);
        for (GlobalIndex col : cols) {
            if (col >= 0 && col < n_cols) {
                pattern.addEntry(row, col);
            }
        }
    }

    if (options.ensure_diagonal && n_rows == n_cols) {
        pattern.ensureDiagonal();
    }

    if (options.ensure_non_empty_rows) {
        pattern.ensureNonEmptyRows();
    }

    pattern.finalize();
    result.pattern = std::make_unique<SparsityPattern>(std::move(pattern));

    auto end_time = std::chrono::high_resolution_clock::now();
    result.construction_time_sec =
        std::chrono::duration<double>(end_time - start_time).count();

    return result;
}

// ============================================================================
// Pattern Creation - Standard Patterns
// ============================================================================

SparsityPattern SparsityFactory::createLaplacianPattern(
    std::span<const GlobalIndex> dims,
    bool periodic) const
{
    FE_CHECK_ARG(!dims.empty() && dims.size() <= 3,
                "Dimensions must be 1D, 2D, or 3D");

    for (auto d : dims) {
        FE_CHECK_ARG(d > 0, "All dimensions must be positive");
    }

    // Compute total size
    GlobalIndex n = 1;
    for (auto d : dims) {
        n *= d;
    }

    SparsityPattern pattern(n, n);

    auto linear_index = [&](std::vector<GlobalIndex>& idx) -> GlobalIndex {
        GlobalIndex lin = 0;
        GlobalIndex stride = 1;
        for (std::size_t d = 0; d < dims.size(); ++d) {
            lin += idx[d] * stride;
            stride *= dims[d];
        }
        return lin;
    };

    auto add_neighbor = [&](std::vector<GlobalIndex>& idx, std::size_t dim,
                            GlobalIndex offset, GlobalIndex row) {
        GlobalIndex orig = idx[dim];
        GlobalIndex new_val = orig + offset;

        if (periodic) {
            new_val = (new_val + dims[dim]) % dims[dim];
        } else if (new_val < 0 || new_val >= dims[dim]) {
            return; // Out of bounds, no neighbor
        }

        idx[dim] = new_val;
        GlobalIndex col = linear_index(idx);
        pattern.addEntry(row, col);
        idx[dim] = orig;
    };

    std::vector<GlobalIndex> idx(dims.size(), 0);

    for (GlobalIndex row = 0; row < n; ++row) {
        // Compute multi-index
        GlobalIndex temp = row;
        for (std::size_t d = 0; d < dims.size(); ++d) {
            idx[d] = temp % dims[d];
            temp /= dims[d];
        }

        // Diagonal
        pattern.addEntry(row, row);

        // Neighbors in each dimension
        for (std::size_t d = 0; d < dims.size(); ++d) {
            add_neighbor(idx, d, -1, row);
            add_neighbor(idx, d, +1, row);
        }
    }

    pattern.finalize();
    return pattern;
}

SparsityPattern SparsityFactory::createBandPattern(
    GlobalIndex n,
    GlobalIndex lower_bandwidth,
    GlobalIndex upper_bandwidth) const
{
    FE_CHECK_ARG(n > 0, "Size must be positive");
    FE_CHECK_ARG(lower_bandwidth >= 0 && upper_bandwidth >= 0,
                "Bandwidths must be non-negative");

    SparsityPattern pattern(n, n);

    for (GlobalIndex row = 0; row < n; ++row) {
        GlobalIndex col_start = std::max(GlobalIndex(0), row - lower_bandwidth);
        GlobalIndex col_end = std::min(n, row + upper_bandwidth + 1);

        for (GlobalIndex col = col_start; col < col_end; ++col) {
            pattern.addEntry(row, col);
        }
    }

    pattern.finalize();
    return pattern;
}

SparsityPattern SparsityFactory::createDiagonalPattern(GlobalIndex n) const {
    FE_CHECK_ARG(n > 0, "Size must be positive");

    SparsityPattern pattern(n, n);
    for (GlobalIndex i = 0; i < n; ++i) {
        pattern.addEntry(i, i);
    }
    pattern.finalize();
    return pattern;
}

SparsityPattern SparsityFactory::createDensePattern(
    GlobalIndex n_rows, GlobalIndex n_cols) const
{
    FE_CHECK_ARG(n_rows > 0 && n_cols > 0, "Dimensions must be positive");

    SparsityPattern pattern(n_rows, n_cols);
    for (GlobalIndex row = 0; row < n_rows; ++row) {
        for (GlobalIndex col = 0; col < n_cols; ++col) {
            pattern.addEntry(row, col);
        }
    }
    pattern.finalize();
    return pattern;
}

// ============================================================================
// Constraint Application
// ============================================================================

SparsityPattern SparsityFactory::applyConstraints(
    const SparsityPattern& pattern,
    const std::vector<SparsityConstraint>& constraints,
    AugmentationMode mode) const
{
    ConstraintSparsityAugmenter augmenter(constraints);
    if (mode == AugmentationMode::ReducedSystem) {
        return augmenter.buildReducedPattern(pattern);
    }

    SparsityPattern result = pattern;  // Copy reconstructs to Building state
    augmenter.augment(result, mode);
    result.finalize();
    return result;
}

// ============================================================================
// Analysis and Recommendations
// ============================================================================

ConstructionStrategy SparsityFactory::suggestStrategy(
    GlobalIndex n_dofs,
    GlobalIndex n_elements,
    GlobalIndex avg_dofs_per_element) const
{
    GlobalIndex estimated_nnz = estimateNnz(n_elements, avg_dofs_per_element);
    return selectStrategy(n_dofs, estimated_nnz, options_);
}

GlobalIndex SparsityFactory::estimateNnz(
    GlobalIndex n_elements,
    GlobalIndex avg_dofs_per_element) const
{
    // Upper bound: each element contributes avg_dofs^2 entries
    // Actual will be less due to shared DOFs
    return n_elements * avg_dofs_per_element * avg_dofs_per_element;
}

std::size_t SparsityFactory::estimateMemoryBytes(
    GlobalIndex n_dofs,
    GlobalIndex estimated_nnz,
    ConstructionStrategy strategy) const
{
    std::size_t bytes = 0;

    switch (strategy) {
        case ConstructionStrategy::Standard:
            // Set of sets: n_dofs sets, each with avg nnz/n_dofs entries
            // Plus overhead for set nodes
            bytes = static_cast<std::size_t>(n_dofs) * 64;  // Set overhead
            bytes += static_cast<std::size_t>(estimated_nnz) * 40;  // Set node overhead
            break;

        case ConstructionStrategy::TwoPass:
            // Two arrays: row_ptr (n+1) and col_ind (nnz)
            bytes = static_cast<std::size_t>(n_dofs + 1) * sizeof(GlobalIndex);
            bytes += static_cast<std::size_t>(estimated_nnz) * sizeof(GlobalIndex);
            break;

        case ConstructionStrategy::Compressed:
            // Compressed buffer with deferred sort
            bytes = static_cast<std::size_t>(estimated_nnz) * 2 * sizeof(GlobalIndex);
            break;

        case ConstructionStrategy::Auto:
            // Use two-pass estimate as default
            bytes = static_cast<std::size_t>(n_dofs + 1) * sizeof(GlobalIndex);
            bytes += static_cast<std::size_t>(estimated_nnz) * sizeof(GlobalIndex);
            break;
    }

    return bytes;
}

// ============================================================================
// Internal Implementation
// ============================================================================

FactoryResult SparsityFactory::createInternal(
    const dofs::DofMap& dof_map,
    const FactoryOptions& options) const
{
    FactoryResult result;

    GlobalIndex n_dofs = dof_map.getNumDofs();
    GlobalIndex n_cells = dof_map.getNumCells();
    GlobalIndex avg_dofs = getMaxDofsPerCell(dof_map);
    GlobalIndex estimated_nnz = estimateNnz(n_cells, avg_dofs);

    ConstructionStrategy strategy = selectStrategy(n_dofs, estimated_nnz, options);
    result.strategy_used = strategy;

    switch (options.type) {
        case PatternType::Standard:
        {
            if (strategy == ConstructionStrategy::TwoPass) {
                TwoPassBuildOptions tp_opts;
                tp_opts.ensure_diagonal = options.ensure_diagonal;
                tp_opts.ensure_non_empty_rows = options.ensure_non_empty_rows;
                SparsityTwoPassBuilder builder(n_dofs, n_dofs, tp_opts);

                // Count pass
                for (GlobalIndex cell = 0; cell < n_cells; ++cell) {
                    auto cell_dofs = dof_map.getCellDofs(cell);
                    builder.countElementCouplings(cell_dofs);
                }

                builder.finalizeCount();

                // Fill pass
                for (GlobalIndex cell = 0; cell < n_cells; ++cell) {
                    auto cell_dofs = dof_map.getCellDofs(cell);
                    builder.addElementCouplings(cell_dofs);
                }

                result.pattern = std::make_unique<SparsityPattern>(builder.finalize());
            } else {
                SparsityPattern pattern(n_dofs, n_dofs);

                for (GlobalIndex cell = 0; cell < n_cells; ++cell) {
                    auto cell_dofs = dof_map.getCellDofs(cell);
                    for (GlobalIndex row : cell_dofs) {
                        for (GlobalIndex col : cell_dofs) {
                            pattern.addEntry(row, col);
                        }
                    }
                }

                if (options.ensure_diagonal) {
                    pattern.ensureDiagonal();
                }

                if (options.ensure_non_empty_rows) {
                    pattern.ensureNonEmptyRows();
                }

                pattern.finalize();
                result.pattern = std::make_unique<SparsityPattern>(std::move(pattern));
            }
            break;
        }

        case PatternType::Custom:
            FE_THROW(InvalidArgumentException,
                     "PatternType::Custom requires an explicit callback; use "
                     "SparsityFactory::createFromCallback(...) instead.");

        case PatternType::DG: {
            FE_THROW(InvalidArgumentException,
                     "PatternType::DG requires face adjacency; use "
                     "SparsityFactory::createDGPattern(...) instead.");
        }

        case PatternType::Mixed:
        case PatternType::BlockStructured:
        case PatternType::Rectangular: {
            FE_THROW(InvalidArgumentException,
                     "Requested pattern type requires additional inputs; use the dedicated "
                     "factory entry points (createRectangular, createBlockPattern, createDGPattern).");
        }
    }

    return result;
}

ConstructionStrategy SparsityFactory::selectStrategy(
    GlobalIndex n_dofs,
    GlobalIndex estimated_nnz,
    const FactoryOptions& options) const
{
    if (options.strategy != ConstructionStrategy::Auto) {
        return options.strategy;
    }

    // Heuristic: use two-pass for large problems
    if (estimated_nnz > options.two_pass_threshold) {
        return ConstructionStrategy::TwoPass;
    }

    // Use compressed for moderate size problems with high density
    double density = static_cast<double>(estimated_nnz) /
                    (static_cast<double>(n_dofs) * static_cast<double>(n_dofs));
    if (n_dofs > 10000 && density > 0.1) {
        return ConstructionStrategy::Compressed;
    }

    return ConstructionStrategy::Standard;
}

std::string SparsityFactory::computeCacheKey(
    const dofs::DofMap& dof_map,
    const FactoryOptions& options) const
{
    std::size_t dof_hash = hashDofMapConnectivity(dof_map);
    std::size_t constraint_hash = 0;
    if (options.apply_constraints && constraint_query_ != nullptr &&
        constraint_query_->numConstraints() > 0) {
        constraint_hash = hashConstraintQuery(*constraint_query_);
    }

    std::ostringstream oss;
    oss << "pattern_" << dof_map.getNumDofs() << "_" << dof_map.getNumCells()
        << "_h" << dof_hash
        << "_t" << static_cast<int>(options.type)
        << "_bk" << static_cast<int>(options.backend)
        << "_s" << static_cast<int>(options.strategy)
        << "_ed" << options.ensure_diagonal
        << "_en" << options.ensure_non_empty_rows
        << "_opt" << options.optimize
        << "_og" << static_cast<int>(options.optimization_goal)
        << "_dg" << options.dg_include_penalty
        << "_gh" << options.include_ghost_rows
        << "_bs" << options.block_size
        << "_ac" << options.apply_constraints
        << "_cm" << static_cast<int>(options.constraint_mode)
        << "_ch" << constraint_hash;
    return oss.str();
}

std::string SparsityFactory::computeDistributedCacheKey(
    const dofs::DofMap& dof_map,
    std::pair<GlobalIndex, GlobalIndex> owned_range,
    GlobalIndex global_size,
    const FactoryOptions& options) const
{
    std::size_t dof_hash = hashDofMapConnectivity(dof_map);
    std::size_t constraint_hash = 0;
    if (options.apply_constraints && constraint_query_ != nullptr &&
        constraint_query_->numConstraints() > 0) {
        constraint_hash = hashConstraintQuery(*constraint_query_);
    }

    std::ostringstream oss;
    oss << "distpattern_" << global_size << "_" << dof_map.getNumCells()
        << "_own" << owned_range.first << "_" << owned_range.second
        << "_h" << dof_hash
        << "_ed" << options.ensure_diagonal
        << "_en" << options.ensure_non_empty_rows
        << "_gh" << options.include_ghost_rows
        << "_ac" << options.apply_constraints
        << "_cm" << static_cast<int>(options.constraint_mode)
        << "_ch" << constraint_hash;
    return oss.str();
}

std::string SparsityFactory::computeBlockCacheKey(
    std::span<const dofs::DofMap*> field_dof_maps,
    std::span<const std::vector<bool>> coupling_matrix,
    const FactoryOptions& options) const
{
    std::size_t seed = 0;
    seed = hashCombine(seed, std::hash<std::size_t>{}(field_dof_maps.size()));

    for (const auto* map : field_dof_maps) {
        FE_CHECK_ARG(map != nullptr, "Null field DOF map");
        seed = hashCombine(seed, hashDofMapConnectivity(*map));
    }

    seed = hashCombine(seed, std::hash<std::size_t>{}(coupling_matrix.size()));
    for (const auto& row : coupling_matrix) {
        seed = hashCombine(seed, std::hash<std::size_t>{}(row.size()));
        for (bool v : row) {
            seed = hashCombine(seed, std::hash<int>{}(v ? 1 : 0));
        }
    }

    std::size_t constraint_hash = 0;
    if (options.apply_constraints && constraint_query_ != nullptr &&
        constraint_query_->numConstraints() > 0) {
        constraint_hash = hashConstraintQuery(*constraint_query_);
    }

    std::ostringstream oss;
    oss << "blockpattern_" << field_dof_maps.size()
        << "_h" << seed
        << "_ed" << options.ensure_diagonal
        << "_en" << options.ensure_non_empty_rows
        << "_ac" << options.apply_constraints
        << "_cm" << static_cast<int>(options.constraint_mode)
        << "_ch" << constraint_hash;
    return oss.str();
}

void SparsityFactory::cachePattern(
    const std::string& key,
    std::shared_ptr<SparsityPattern> pattern) const
{
    cache_[key] = std::move(pattern);
}

std::shared_ptr<SparsityPattern> SparsityFactory::getCachedPattern(
    const std::string& key) const
{
    auto it = cache_.find(key);
    if (it != cache_.end()) {
        return it->second;
    }
    return nullptr;
}

void SparsityFactory::cacheDistributedPattern(
    const std::string& key,
    std::shared_ptr<DistributedSparsityPattern> pattern) const
{
    dist_cache_[key] = std::move(pattern);
}

std::shared_ptr<DistributedSparsityPattern> SparsityFactory::getCachedDistributedPattern(
    const std::string& key) const
{
    auto it = dist_cache_.find(key);
    if (it != dist_cache_.end()) {
        return it->second;
    }
    return nullptr;
}

void SparsityFactory::cacheBlockPattern(
    const std::string& key,
    std::shared_ptr<BlockSparsity> pattern) const
{
    block_cache_[key] = std::move(pattern);
}

std::shared_ptr<BlockSparsity> SparsityFactory::getCachedBlockPattern(
    const std::string& key) const
{
    auto it = block_cache_.find(key);
    if (it != block_cache_.end()) {
        return it->second;
    }
    return nullptr;
}

// ============================================================================
// Convenience Functions
// ============================================================================

SparsityPattern createPattern(const dofs::DofMap& dof_map) {
    SparsityFactory factory;
    return factory.createFromDofMap(dof_map);
}

SparsityPattern createOptimizedPattern(const dofs::DofMap& dof_map) {
    SparsityFactory factory;
    FactoryOptions opts;
    opts.optimize = true;
    opts.optimization_goal = OptimizationGoal::Balanced;

    auto result = factory.create(dof_map, opts);
    if (!result.pattern) {
        FE_THROW(InvalidArgumentException, "Failed to create optimized pattern");
    }
    return std::move(*result.pattern);
}

SparsityPattern createDGPatternFromMap(
    const dofs::DofMap& dof_map,
    const std::vector<std::pair<GlobalIndex, GlobalIndex>>& face_adjacency)
{
    SparsityFactory factory;
    FactoryOptions opts;
    opts.type = PatternType::DG;

    auto result = factory.createDGPattern(dof_map, face_adjacency, opts);
    if (!result.pattern) {
        FE_THROW(InvalidArgumentException, "Failed to create DG pattern");
    }
    return std::move(*result.pattern);
}

FactoryOptions recommendOptions(
    GlobalIndex n_dofs,
    GlobalIndex n_elements,
    bool is_distributed,
    bool has_constraints)
{
    FactoryOptions opts;

    // Select construction strategy based on size
    GlobalIndex estimated_nnz = n_elements * 64;  // Rough estimate
    if (estimated_nnz > 1000000) {
        opts.strategy = ConstructionStrategy::TwoPass;
    } else if (estimated_nnz > 100000) {
        opts.strategy = ConstructionStrategy::Compressed;
    } else {
        opts.strategy = ConstructionStrategy::Standard;
    }

    // Backend recommendation
    if (is_distributed) {
        opts.backend = TargetBackend::PETSc;
        opts.include_ghost_rows = true;
    } else {
        opts.backend = TargetBackend::Generic;
    }

    // Constraint handling
    opts.apply_constraints = has_constraints;
    if (has_constraints) {
        opts.constraint_mode = AugmentationMode::EliminationFill;
    }

    // Always ensure diagonal for solver compatibility
    opts.ensure_diagonal = true;

    // Enable optimization for larger problems
    if (n_dofs > 10000) {
        opts.optimize = true;
        opts.optimization_goal = OptimizationGoal::MinimizeBandwidth;
    }

    return opts;
}

} // namespace sparsity
} // namespace FE
} // namespace svmp

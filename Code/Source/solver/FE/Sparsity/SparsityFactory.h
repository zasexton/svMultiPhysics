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

#ifndef SVMP_FE_SPARSITY_FACTORY_H
#define SVMP_FE_SPARSITY_FACTORY_H

/**
 * @file SparsityFactory.h
 * @brief Factory for creating sparsity patterns from various inputs
 *
 * This header provides the SparsityFactory class for creating sparsity patterns
 * in a unified way. It handles:
 *
 * - Pattern creation from DOF maps and mesh topology
 * - Pattern type selection (CG, DG, mixed, block)
 * - Automatic algorithm selection based on problem size
 * - Backend compatibility hints
 * - Pattern caching and reuse
 *
 * The factory provides sensible defaults while allowing full customization
 * for advanced use cases.
 *
 * @see SparsityPattern for the output data structure
 * @see SparsityBuilder for lower-level construction
 */

#include "SparsityPattern.h"
#include "DistributedSparsityPattern.h"
#include "BlockSparsity.h"
#include "SparsityBuilder.h"
#include "DGSparsityBuilder.h"
#include "ConstraintSparsityAugmenter.h"
#include "SparsityOptimizer.h"
#include "Core/Types.h"
#include "Core/FEConfig.h"
#include "Core/FEException.h"

#include <memory>
#include <vector>
#include <span>
#include <functional>
#include <optional>
#include <string>
#include <unordered_map>

namespace svmp {
namespace FE {

// Forward declarations
namespace dofs {
class DofMap;
class DofConstraints;
}

namespace sparsity {

// ============================================================================
// Pattern Types and Options
// ============================================================================

/**
 * @brief Type of sparsity pattern to create
 */
enum class PatternType : std::uint8_t {
    Standard,          ///< Standard CG pattern from element connectivity
    DG,                ///< DG pattern with face couplings
    Mixed,             ///< Mixed CG/DG pattern
    BlockStructured,   ///< Block-structured pattern for multi-field
    Rectangular,       ///< Rectangular pattern (different row/col spaces)
    Custom             ///< Custom pattern from user-defined callback
};

/**
 * @brief Target backend for pattern compatibility
 */
enum class TargetBackend : std::uint8_t {
    Generic,           ///< Generic pattern, no backend-specific optimizations
    PETSc,             ///< Optimize for PETSc (diag/offdiag preallocation)
    Trilinos,          ///< Optimize for Trilinos/Tpetra (static profile)
    Eigen,             ///< Optimize for Eigen (CSR/CSC compatible)
    Custom             ///< Custom backend requirements
};

/**
 * @brief Construction strategy for pattern building
 */
enum class ConstructionStrategy : std::uint8_t {
    Standard,          ///< Standard set-of-sets approach
    TwoPass,           ///< Two-pass counting/filling for large problems
    Compressed,        ///< Compressed construction with deferred deduplication
    Auto               ///< Automatic selection based on problem size
};

/**
 * @brief Options for factory pattern creation
 */
struct FactoryOptions {
    PatternType type{PatternType::Standard};
    TargetBackend backend{TargetBackend::Generic};
    ConstructionStrategy strategy{ConstructionStrategy::Auto};

    /**
     * @brief Enable constraint augmentation
     *
     * When true, SparsityFactory::create() applies constraint-induced fill
     * using constraints configured via setConstraints(...).
     */
    bool apply_constraints{false};

    /**
     * @brief Constraint augmentation mode
     */
    AugmentationMode constraint_mode{AugmentationMode::EliminationFill};

    /**
     * @brief Enable pattern optimization
     */
    bool optimize{false};

    /**
     * @brief Optimization goal if optimize=true
     */
    OptimizationGoal optimization_goal{OptimizationGoal::Balanced};

    /**
     * @brief Ensure diagonal entries exist
     */
    bool ensure_diagonal{true};

    /**
     * @brief Ensure no empty rows
     */
    bool ensure_non_empty_rows{true};

    /**
     * @brief Include couplings to non-owned (ghost) DOFs in distributed patterns
     *
     * When false, distributed patterns are built with only owned-by-owned
     * couplings (diagonal block), excluding the off-diagonal block.
     */
    bool include_ghost_rows{true};

    /**
     * @brief Block size for block-structured patterns
     */
    GlobalIndex block_size{1};

    /**
     * @brief DG penalty coefficient type affects pattern
     */
    bool dg_include_penalty{true};

    /**
     * @brief Threshold for switching to two-pass construction
     */
    GlobalIndex two_pass_threshold{100000};

    /**
     * @brief Enable caching of created patterns
     */
    bool enable_caching{false};

    /**
     * @brief Verbose output during construction
     */
    bool verbose{false};
};

/**
 * @brief Result from factory pattern creation
 */
struct FactoryResult {
    /**
     * @brief The created pattern
     */
    std::unique_ptr<SparsityPattern> pattern;

    /**
     * @brief Distributed pattern (if applicable)
     */
    std::unique_ptr<DistributedSparsityPattern> distributed_pattern;

    /**
     * @brief Block pattern (if applicable)
     */
    std::unique_ptr<BlockSparsity> block_pattern;

    /**
     * @brief Optimization result (if optimization was applied)
     */
    std::optional<OptimizationResult> optimization_result;

    /**
     * @brief Strategy that was used
     */
    ConstructionStrategy strategy_used{ConstructionStrategy::Standard};

    /**
     * @brief Construction time in seconds
     */
    double construction_time_sec{0.0};

    /**
     * @brief Cache key if caching was used
     */
    std::string cache_key;

    /**
     * @brief Whether result was retrieved from cache
     */
    bool from_cache{false};
};

// ============================================================================
// SparsityFactory Class
// ============================================================================

/**
 * @brief Factory for creating sparsity patterns
 *
 * SparsityFactory provides a high-level interface for creating sparsity
 * patterns from mesh and DOF information. It handles algorithm selection,
 * backend compatibility, and optional optimization.
 *
 * Usage:
 * @code
 * // Simple pattern creation
 * SparsityFactory factory;
 * auto pattern = factory.createFromDofMap(dof_map);
 *
 * // With options
 * FactoryOptions opts;
 * opts.type = PatternType::DG;
 * opts.optimize = true;
 * auto result = factory.create(dof_map, opts);
 *
 * // Block-structured pattern for Stokes
 * FactoryOptions block_opts;
 * block_opts.type = PatternType::BlockStructured;
 * block_opts.block_size = 3;  // velocity components
 * auto stokes_pattern = factory.createBlockPattern(
 *     {velocity_dofs, pressure_dofs}, block_opts);
 *
 * // Distributed pattern for MPI
 * auto dist_pattern = factory.createDistributed(
 *     dof_map, ownership_range, opts);
 * @endcode
 */
class SparsityFactory {
public:
    // =========================================================================
    // Construction
    // =========================================================================

    /**
     * @brief Default constructor
     */
    SparsityFactory() = default;

    /**
     * @brief Construct with default options
     */
    explicit SparsityFactory(const FactoryOptions& options);

    /// Destructor
    ~SparsityFactory() = default;

    // Non-copyable for cache integrity
    SparsityFactory(const SparsityFactory&) = delete;
    SparsityFactory& operator=(const SparsityFactory&) = delete;

    // Movable
    SparsityFactory(SparsityFactory&&) = default;
    SparsityFactory& operator=(SparsityFactory&&) = default;

    // =========================================================================
    // Configuration
    // =========================================================================

    /**
     * @brief Set default options
     */
    void setOptions(const FactoryOptions& options);

    /**
     * @brief Get current options
     */
    [[nodiscard]] const FactoryOptions& getOptions() const noexcept {
        return options_;
    }

    /**
     * @brief Clear pattern cache
     */
    void clearCache();

    /**
     * @brief Get cache statistics
     */
    [[nodiscard]] std::pair<std::size_t, std::size_t> getCacheStats() const;

    /**
     * @brief Set constraint query interface used by apply_constraints
     *
     * The caller is responsible for ensuring the provided query remains valid
     * for the lifetime of the factory or until replaced/cleared.
     */
    void setConstraints(std::shared_ptr<IConstraintQuery> constraint_query);

    /**
     * @brief Convenience overload: use FE/Dofs constraints with an adapter
     *
     * Note: The referenced DofConstraints object must outlive the factory (or
     * until constraints are cleared/replaced), since the adapter stores a pointer.
     */
    void setConstraints(const dofs::DofConstraints& constraints);

    /**
     * @brief Clear configured constraints
     */
    void clearConstraints();

    /**
     * @brief Check whether constraints are configured
     */
    [[nodiscard]] bool hasConstraints() const noexcept;

    // =========================================================================
    // Pattern Creation - DOF Map Based
    // =========================================================================

    /**
     * @brief Create pattern from DOF map
     *
     * @param dof_map DOF map with element connectivity
     * @param options Creation options
     * @return Factory result with created pattern
     */
    [[nodiscard]] FactoryResult create(
        const dofs::DofMap& dof_map,
        const FactoryOptions& options) const;

    /**
     * @brief Create pattern from DOF map with default options
     */
    [[nodiscard]] FactoryResult create(const dofs::DofMap& dof_map) const;

    /**
     * @brief Create simple pattern (convenience method)
     */
    [[nodiscard]] SparsityPattern createFromDofMap(const dofs::DofMap& dof_map) const;

    /**
     * @brief Create rectangular pattern from two DOF maps
     *
     * @param row_dof_map DOF map for rows
     * @param col_dof_map DOF map for columns
     * @param options Creation options
     * @return Factory result
     */
    [[nodiscard]] FactoryResult createRectangular(
        const dofs::DofMap& row_dof_map,
        const dofs::DofMap& col_dof_map,
        const FactoryOptions& options) const;

    /**
     * @brief Create rectangular pattern with default options
     */
    [[nodiscard]] FactoryResult createRectangular(
        const dofs::DofMap& row_dof_map,
        const dofs::DofMap& col_dof_map) const;

    // =========================================================================
    // Pattern Creation - Block Structured
    // =========================================================================

    /**
     * @brief Create block-structured pattern
     *
     * @param field_dof_maps DOF maps for each field
     * @param coupling_matrix Which fields couple (true = coupled)
     * @param options Creation options
     * @return Factory result with block pattern
     */
    [[nodiscard]] FactoryResult createBlockPattern(
        std::span<const dofs::DofMap*> field_dof_maps,
        std::span<const std::vector<bool>> coupling_matrix,
        const FactoryOptions& options) const;

    /**
     * @brief Create block pattern with full coupling
     */
    [[nodiscard]] FactoryResult createBlockPattern(
        std::span<const dofs::DofMap*> field_dof_maps,
        const FactoryOptions& options) const;

    // =========================================================================
    // Pattern Creation - DG
    // =========================================================================

    /**
     * @brief Create DG pattern with face couplings
     *
     * @param dof_map DOF map
     * @param face_adjacency Face neighbor information
     * @param options Creation options
     * @return Factory result
     */
    [[nodiscard]] FactoryResult createDGPattern(
        const dofs::DofMap& dof_map,
        const std::vector<std::pair<GlobalIndex, GlobalIndex>>& face_adjacency,
        const FactoryOptions& options) const;

    // =========================================================================
    // Pattern Creation - Distributed
    // =========================================================================

    /**
     * @brief Create distributed pattern
     *
     * @param dof_map DOF map with ghost information
     * @param owned_range Range of owned rows [first, last)
     * @param global_size Total global size
     * @param options Creation options
     * @return Factory result with distributed pattern
     */
    [[nodiscard]] FactoryResult createDistributed(
        const dofs::DofMap& dof_map,
        std::pair<GlobalIndex, GlobalIndex> owned_range,
        GlobalIndex global_size,
        const FactoryOptions& options) const;

    /**
     * @brief Create distributed pattern with default options
     */
    [[nodiscard]] FactoryResult createDistributed(
        const dofs::DofMap& dof_map,
        std::pair<GlobalIndex, GlobalIndex> owned_range,
        GlobalIndex global_size) const;

    // =========================================================================
    // Pattern Creation - Low Level
    // =========================================================================

    /**
     * @brief Create pattern from connectivity arrays
     *
     * @param n_rows Number of rows
     * @param n_cols Number of columns
     * @param n_elements Number of elements
     * @param elem_offsets CSR-style offsets for DOFs per element
     * @param elem_dofs Flat array of DOFs
     * @param options Creation options
     * @return Factory result
     */
    [[nodiscard]] FactoryResult createFromArrays(
        GlobalIndex n_rows,
        GlobalIndex n_cols,
        GlobalIndex n_elements,
        std::span<const GlobalIndex> elem_offsets,
        std::span<const GlobalIndex> elem_dofs,
        const FactoryOptions& options) const;

    /**
     * @brief Create pattern from callback function
     *
     * @param n_rows Number of rows
     * @param n_cols Number of columns
     * @param row_entries Callback returning column indices for each row
     * @param options Creation options
     * @return Factory result
     */
    [[nodiscard]] FactoryResult createFromCallback(
        GlobalIndex n_rows,
        GlobalIndex n_cols,
        std::function<std::vector<GlobalIndex>(GlobalIndex row)> row_entries,
        const FactoryOptions& options) const;

    // =========================================================================
    // Pattern Creation - Standard Patterns
    // =========================================================================

    /**
     * @brief Create Laplacian stencil pattern (5-point in 2D, 7-point in 3D)
     *
     * @param dims Grid dimensions
     * @param periodic Whether boundaries are periodic
     * @return Pattern for Laplacian stencil
     */
    [[nodiscard]] SparsityPattern createLaplacianPattern(
        std::span<const GlobalIndex> dims,
        bool periodic = false) const;

    /**
     * @brief Create band pattern
     *
     * @param n Size of square matrix
     * @param lower_bandwidth Lower bandwidth
     * @param upper_bandwidth Upper bandwidth
     * @return Band pattern
     */
    [[nodiscard]] SparsityPattern createBandPattern(
        GlobalIndex n,
        GlobalIndex lower_bandwidth,
        GlobalIndex upper_bandwidth) const;

    /**
     * @brief Create diagonal pattern
     *
     * @param n Size
     * @return Diagonal-only pattern
     */
    [[nodiscard]] SparsityPattern createDiagonalPattern(GlobalIndex n) const;

    /**
     * @brief Create dense/full pattern
     *
     * @param n_rows Number of rows
     * @param n_cols Number of columns
     * @return Full pattern (all entries non-zero)
     */
    [[nodiscard]] SparsityPattern createDensePattern(
        GlobalIndex n_rows, GlobalIndex n_cols) const;

    // =========================================================================
    // Constraint Application
    // =========================================================================

    /**
     * @brief Apply constraint fill-in to a pattern
     *
     * @param pattern Pattern to augment
     * @param constraints Constraint definitions
     * @param mode Augmentation mode
     * @return Augmented pattern
     */
    [[nodiscard]] SparsityPattern applyConstraints(
        const SparsityPattern& pattern,
        const std::vector<SparsityConstraint>& constraints,
        AugmentationMode mode = AugmentationMode::EliminationFill) const;

    // =========================================================================
    // Analysis and Recommendations
    // =========================================================================

    /**
     * @brief Suggest construction strategy based on problem size
     *
     * @param n_dofs Number of DOFs
     * @param n_elements Number of elements
     * @param avg_dofs_per_element Average DOFs per element
     * @return Recommended strategy
     */
    [[nodiscard]] ConstructionStrategy suggestStrategy(
        GlobalIndex n_dofs,
        GlobalIndex n_elements,
        GlobalIndex avg_dofs_per_element) const;

    /**
     * @brief Estimate NNZ for pattern
     */
    [[nodiscard]] GlobalIndex estimateNnz(
        GlobalIndex n_elements,
        GlobalIndex avg_dofs_per_element) const;

    /**
     * @brief Estimate memory usage for pattern construction
     */
    [[nodiscard]] std::size_t estimateMemoryBytes(
        GlobalIndex n_dofs,
        GlobalIndex estimated_nnz,
        ConstructionStrategy strategy) const;

private:
    // Internal implementation
    [[nodiscard]] FactoryResult createInternal(
        const dofs::DofMap& dof_map,
        const FactoryOptions& options) const;

    [[nodiscard]] ConstructionStrategy selectStrategy(
        GlobalIndex n_dofs,
        GlobalIndex estimated_nnz,
        const FactoryOptions& options) const;

    [[nodiscard]] std::string computeCacheKey(
        const dofs::DofMap& dof_map,
        const FactoryOptions& options) const;

    [[nodiscard]] std::string computeDistributedCacheKey(
        const dofs::DofMap& dof_map,
        std::pair<GlobalIndex, GlobalIndex> owned_range,
        GlobalIndex global_size,
        const FactoryOptions& options) const;

    [[nodiscard]] std::string computeBlockCacheKey(
        std::span<const dofs::DofMap*> field_dof_maps,
        std::span<const std::vector<bool>> coupling_matrix,
        const FactoryOptions& options) const;

    void cachePattern(const std::string& key,
                      std::shared_ptr<SparsityPattern> pattern) const;

    [[nodiscard]] std::shared_ptr<SparsityPattern> getCachedPattern(
        const std::string& key) const;

    void cacheDistributedPattern(const std::string& key,
                                 std::shared_ptr<DistributedSparsityPattern> pattern) const;

    [[nodiscard]] std::shared_ptr<DistributedSparsityPattern> getCachedDistributedPattern(
        const std::string& key) const;

    void cacheBlockPattern(const std::string& key,
                           std::shared_ptr<BlockSparsity> pattern) const;

    [[nodiscard]] std::shared_ptr<BlockSparsity> getCachedBlockPattern(
        const std::string& key) const;

    FactoryOptions options_;

    // Optional constraint query for apply_constraints workflows
    std::shared_ptr<IConstraintQuery> constraint_query_;

    // Pattern cache (mutable for const methods)
    mutable std::unordered_map<std::string, std::shared_ptr<SparsityPattern>> cache_;
    mutable std::unordered_map<std::string, std::shared_ptr<DistributedSparsityPattern>> dist_cache_;
    mutable std::unordered_map<std::string, std::shared_ptr<BlockSparsity>> block_cache_;
    mutable std::size_t cache_hits_{0};
    mutable std::size_t cache_misses_{0};
};

// ============================================================================
// Convenience Functions
// ============================================================================

/**
 * @brief Create pattern from DOF map (simplest interface)
 */
[[nodiscard]] SparsityPattern createPattern(const dofs::DofMap& dof_map);

/**
 * @brief Create pattern with optimization
 */
[[nodiscard]] SparsityPattern createOptimizedPattern(const dofs::DofMap& dof_map);

/**
 * @brief Create DG pattern
 */
[[nodiscard]] SparsityPattern createDGPatternFromMap(
    const dofs::DofMap& dof_map,
    const std::vector<std::pair<GlobalIndex, GlobalIndex>>& face_adjacency);

/**
 * @brief Get recommended factory options for a problem size
 */
[[nodiscard]] FactoryOptions recommendOptions(
    GlobalIndex n_dofs,
    GlobalIndex n_elements,
    bool is_distributed = false,
    bool has_constraints = false);

} // namespace sparsity
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_SPARSITY_FACTORY_H

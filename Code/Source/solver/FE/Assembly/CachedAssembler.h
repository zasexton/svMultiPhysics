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

#ifndef SVMP_FE_ASSEMBLY_CACHED_ASSEMBLER_H
#define SVMP_FE_ASSEMBLY_CACHED_ASSEMBLER_H

/**
 * @file CachedAssembler.h
 * @brief Assembly with element matrix caching for linear problems
 *
 * CachedAssembler optimizes linear problem assembly by caching computed
 * element matrices and reusing them across multiple assembly operations.
 * This is particularly effective for:
 *
 * - Linear problems with constant coefficients
 * - Time-stepping methods where stiffness/mass matrices don't change
 * - Parameter studies where only the RHS changes
 * - Iterative solvers requiring multiple matrix-vector products
 *
 * Caching Strategy:
 *
 * 1. FULL CACHE: Store complete dense local matrices for each element
 *    - Pros: Fastest reassembly, reuses entire computation
 *    - Cons: High memory usage (O(n_elements * dofs^2))
 *
 * 2. REFERENCE ELEMENT CACHE: Store per-element-type on reference element
 *    - Pros: Minimal memory, good for uniform meshes
 *    - Cons: Requires geometry scaling at assembly time
 *
 * 3. GEOMETRIC FACTOR CACHE: Store element-specific geometric factors
 *    - Pros: Balance between memory and speed
 *    - Cons: Still requires some computation
 *
 * Cache Invalidation:
 * - Manual invalidation via invalidateCache()
 * - Automatic invalidation when mesh geometry changes
 * - Per-element invalidation for local mesh modifications
 *
 * Memory Management:
 * - Configurable memory budget
 * - LRU eviction when budget exceeded
 * - Option to serialize cache to disk
 *
 * Thread Safety:
 * - Cache is populated in parallel (first assembly)
 * - Cache reads are thread-safe (subsequent assemblies)
 * - Invalidation is synchronized
 *
 * @see StandardAssembler for baseline non-caching implementation
 */

#include "Core/Types.h"
#include "Assembly/DecoratorAssembler.h"
#include "Assembly/AssemblyKernel.h"
#include "Assembly/GlobalSystemView.h"

#include <vector>
#include <unordered_map>
#include <memory>
#include <mutex>
#include <atomic>
#include <optional>
#include <functional>

namespace svmp {
namespace FE {

// Forward declarations
namespace dofs {
    class DofMap;
    class DofHandler;
}

namespace spaces {
    class FunctionSpace;
}

namespace sparsity {
    class SparsityPattern;
}

namespace constraints {
    class AffineConstraints;
    class ConstraintDistributor;
}

namespace assembly {

// ============================================================================
// Cache Configuration
// ============================================================================

/**
 * @brief Caching strategy selection
 */
enum class CacheStrategy {
    /**
     * @brief Store full local matrix per element
     */
    FullMatrix,

    /**
     * @brief Store reference element matrices (scaled at assembly)
     */
    ReferenceElement,

    /**
     * @brief Store geometric factors only
     */
    GeometricFactors,

    /**
     * @brief No caching (delegate to StandardAssembler)
     */
    None
};

/**
 * @brief Cache eviction policy
 */
enum class EvictionPolicy {
    /**
     * @brief Least Recently Used
     */
    LRU,

    /**
     * @brief First In First Out
     */
    FIFO,

    /**
     * @brief No eviction (grow unbounded or fail)
     */
    None
};

/**
 * @brief Configuration options for CachedAssembler
 */
struct CacheOptions {
    /**
     * @brief Caching strategy to use
     */
    CacheStrategy strategy{CacheStrategy::FullMatrix};

    /**
     * @brief Eviction policy when memory limit reached
     */
    EvictionPolicy eviction{EvictionPolicy::LRU};

    /**
     * @brief Maximum memory for cache (bytes, 0 = unlimited)
     */
    std::size_t max_memory_bytes{0};

    /**
     * @brief Maximum number of cached elements (0 = unlimited)
     */
    std::size_t max_elements{0};

    /**
     * @brief Enable automatic invalidation on mesh changes
     */
    bool auto_invalidate_on_mesh_change{true};

    /**
     * @brief Enable thread-safe parallel cache population
     */
    bool parallel_population{true};

    /**
     * @brief Number of threads for parallel population
     */
    int num_threads{4};

    /**
     * @brief Verify cached values periodically (debugging)
     */
    bool verify_cache{false};

    /**
     * @brief Relative tolerance for cache verification
     */
    Real verify_tolerance{1e-12};
};

/**
 * @brief Statistics about cache usage
 */
struct CacheStats {
    std::size_t total_elements{0};       ///< Total elements in cache
    std::size_t memory_bytes{0};         ///< Current memory usage
    std::size_t cache_hits{0};           ///< Number of cache hits
    std::size_t cache_misses{0};         ///< Number of cache misses
    std::size_t evictions{0};            ///< Number of evictions
    std::size_t invalidations{0};        ///< Number of invalidations
    double population_seconds{0.0};       ///< Time to populate cache
    double hit_rate{0.0};                 ///< Hit rate (hits / (hits + misses))

    void updateHitRate() {
        std::size_t total = cache_hits + cache_misses;
        hit_rate = total > 0 ? static_cast<double>(cache_hits) / static_cast<double>(total) : 0.0;
    }
};

// ============================================================================
// Cached Element Data
// ============================================================================

/**
 * @brief Cached data for a single element
 */
struct CachedElementData {
    /**
     * @brief Local element matrix (row-major)
     */
    std::vector<Real> local_matrix;

    /**
     * @brief Local element vector (for source terms)
     */
    std::vector<Real> local_vector;

    /**
     * @brief Row DOF indices
     */
    std::vector<GlobalIndex> row_dofs;

    /**
     * @brief Column DOF indices
     */
    std::vector<GlobalIndex> col_dofs;

    /**
     * @brief Element type
     */
    ElementType element_type{ElementType::Unknown};

    /**
     * @brief Whether matrix is cached
     */
    bool has_matrix{false};

    /**
     * @brief Whether vector is cached
     */
    bool has_vector{false};

    /**
     * @brief Timestamp for LRU tracking
     */
    std::size_t last_access{0};

    /**
     * @brief Monotonic insertion order (for FIFO eviction)
     */
    std::size_t insertion_order{0};

    /**
     * @brief Estimated memory usage in bytes
     */
    [[nodiscard]] std::size_t memoryBytes() const {
        return local_matrix.capacity() * sizeof(Real) +
               local_vector.capacity() * sizeof(Real) +
               row_dofs.capacity() * sizeof(GlobalIndex) +
               col_dofs.capacity() * sizeof(GlobalIndex) +
               sizeof(*this);
    }
};

// ============================================================================
// Element Matrix Cache
// ============================================================================

/**
 * @brief Thread-safe cache for element matrices
 */
class ElementMatrixCache {
public:
    ElementMatrixCache();
    explicit ElementMatrixCache(const CacheOptions& options);
    ~ElementMatrixCache();

    // =========================================================================
    // Configuration
    // =========================================================================

    /**
     * @brief Set cache options
     */
    void setOptions(const CacheOptions& options);

    /**
     * @brief Get current options
     */
    [[nodiscard]] const CacheOptions& getOptions() const noexcept;

    // =========================================================================
    // Cache Operations
    // =========================================================================

    /**
     * @brief Check if element is cached
     */
    [[nodiscard]] bool contains(GlobalIndex cell_id) const;

    /**
     * @brief Get cached element data (nullptr if not cached)
     */
    [[nodiscard]] const CachedElementData* get(GlobalIndex cell_id);

    /**
     * @brief Insert element data into cache
     *
     * @return true if inserted, false if evicted or rejected
     */
    bool insert(GlobalIndex cell_id, CachedElementData data);

    /**
     * @brief Invalidate specific element
     */
    void invalidate(GlobalIndex cell_id);

    /**
     * @brief Invalidate all cached elements
     */
    void invalidateAll();

    /**
     * @brief Clear the entire cache
     */
    void clear();

    // =========================================================================
    // Statistics
    // =========================================================================

    /**
     * @brief Get cache statistics
     */
    [[nodiscard]] CacheStats getStats() const;

    /**
     * @brief Reset statistics counters
     */
    void resetStats();

    // =========================================================================
    // Memory Management
    // =========================================================================

    /**
     * @brief Get current memory usage
     */
    [[nodiscard]] std::size_t memoryUsage() const;

    /**
     * @brief Trigger cache eviction to meet memory limit
     */
    void evictToLimit();

    /**
     * @brief Reserve capacity for expected number of elements
     */
    void reserve(std::size_t num_elements);

private:
    /**
     * @brief Evict one element based on policy
     */
    void evictOne();

    CacheOptions options_;
    std::unordered_map<GlobalIndex, CachedElementData> cache_;
    mutable std::mutex mutex_;
    mutable CacheStats stats_;
    std::atomic<std::size_t> access_counter_{0};
    std::atomic<std::size_t> insertion_counter_{0};
};

// ============================================================================
// CachedAssembler
// ============================================================================

/**
 * @brief Assembler with element matrix caching for linear problems
 *
 * CachedAssembler extends StandardAssembler with caching capabilities.
 *
 * Usage for time-stepping:
 * @code
 *   CachedAssembler assembler;
 *   assembler.setDofMap(dof_map);
 *   assembler.setCacheOptions({.strategy = CacheStrategy::FullMatrix});
 *
 *   // First assembly populates cache
 *   assembler.assembleMatrix(mesh, space, space, stiffness_kernel, K);
 *
 *   // Subsequent assemblies use cache (fast)
 *   for (int step = 0; step < num_steps; ++step) {
 *       // Matrix assembly is fast (from cache)
 *       assembler.assembleMatrixFromCache(K);
 *
 *       // RHS assembly (not cached, depends on time)
 *       assembler.assembleVector(mesh, space, rhs_kernel, F);
 *
 *       solve(K, F, u);
 *   }
 *
 *   // Check cache effectiveness
 *   auto stats = assembler.getCacheStats();
 *   std::cout << "Hit rate: " << stats.hit_rate * 100 << "%" << std::endl;
 * @endcode
 *
 * Usage for parameter studies:
 * @code
 *   CachedAssembler assembler;
 *   // ... configuration ...
 *
 *   // Cache the geometry-dependent part
 *   assembler.populateCache(mesh, space, geometric_kernel);
 *
 *   // Vary material parameters
 *   for (double param : parameters) {
 *       material_kernel.setParameter(param);
 *       assembler.assembleWithCachedGeometry(mesh, space, material_kernel, K);
 *       solve(K, F, u);
 *   }
 * @endcode
 */
class CachedAssembler : public DecoratorAssembler {
public:
    // =========================================================================
    // Construction
    // =========================================================================

    CachedAssembler();
    explicit CachedAssembler(const CacheOptions& options);
    explicit CachedAssembler(std::unique_ptr<Assembler> base,
                             const CacheOptions& options = {});
    ~CachedAssembler() override;

    CachedAssembler(CachedAssembler&& other) noexcept;
    CachedAssembler& operator=(CachedAssembler&& other) noexcept;

    // Non-copyable
    CachedAssembler(const CachedAssembler&) = delete;
    CachedAssembler& operator=(const CachedAssembler&) = delete;

    [[nodiscard]] std::string name() const override { return "Cached(" + base().name() + ")"; }

    // =========================================================================
    // Cache-Specific Configuration
    // =========================================================================

    /**
     * @brief Set cache options
     */
    void setCacheOptions(const CacheOptions& options);

    /**
     * @brief Get current cache options
     */
    [[nodiscard]] const CacheOptions& getCacheOptions() const noexcept;

    /**
     * @brief Get cache statistics
     */
    [[nodiscard]] CacheStats getCacheStats() const;

    /**
     * @brief Reset cache statistics
     */
    void resetCacheStats();

    // =========================================================================
    // Lifecycle
    // =========================================================================

    void reset() override;

    // =========================================================================
    // Assembly Operations (Assembler interface)
    // =========================================================================

    AssemblyResult assembleMatrix(
        const IMeshAccess& mesh,
        const spaces::FunctionSpace& test_space,
        const spaces::FunctionSpace& trial_space,
        AssemblyKernel& kernel,
        GlobalSystemView& matrix_view) override;

    AssemblyResult assembleVector(
        const IMeshAccess& mesh,
        const spaces::FunctionSpace& space,
        AssemblyKernel& kernel,
        GlobalSystemView& vector_view) override;

    AssemblyResult assembleBoth(
        const IMeshAccess& mesh,
        const spaces::FunctionSpace& test_space,
        const spaces::FunctionSpace& trial_space,
        AssemblyKernel& kernel,
        GlobalSystemView& matrix_view,
        GlobalSystemView& vector_view) override;

    AssemblyResult assembleBoundaryFaces(
        const IMeshAccess& mesh,
        int boundary_marker,
        const spaces::FunctionSpace& space,
        AssemblyKernel& kernel,
        GlobalSystemView* matrix_view,
        GlobalSystemView* vector_view) override;

    AssemblyResult assembleInteriorFaces(
        const IMeshAccess& mesh,
        const spaces::FunctionSpace& test_space,
        const spaces::FunctionSpace& trial_space,
        AssemblyKernel& kernel,
        GlobalSystemView& matrix_view,
        GlobalSystemView* vector_view) override;

    // =========================================================================
    // Cache-Specific Assembly Operations
    // =========================================================================

    /**
     * @brief Populate cache without assembling into global system
     *
     * Useful for pre-computing element matrices before multiple assemblies.
     */
    AssemblyResult populateCache(
        const IMeshAccess& mesh,
        const spaces::FunctionSpace& test_space,
        const spaces::FunctionSpace& trial_space,
        AssemblyKernel& kernel);

    /**
     * @brief Assemble global matrix purely from cached element matrices
     *
     * Fails if any element is not cached.
     */
    AssemblyResult assembleMatrixFromCache(GlobalSystemView& matrix_view);

    /**
     * @brief Assemble with hybrid strategy: cache hits use cache, misses compute
     */
    AssemblyResult assembleMatrixHybrid(
        const IMeshAccess& mesh,
        const spaces::FunctionSpace& test_space,
        const spaces::FunctionSpace& trial_space,
        AssemblyKernel& kernel,
        GlobalSystemView& matrix_view);

    // =========================================================================
    // Cache Management
    // =========================================================================

    /**
     * @brief Invalidate entire cache
     */
    void invalidateCache();

    /**
     * @brief Invalidate specific elements
     */
    void invalidateElements(std::span<const GlobalIndex> cell_ids);

    /**
     * @brief Check if cache is populated
     */
    [[nodiscard]] bool isCachePopulated() const noexcept;

    /**
     * @brief Get number of cached elements
     */
    [[nodiscard]] std::size_t numCachedElements() const;

    /**
     * @brief Get cache memory usage
     */
    [[nodiscard]] std::size_t cacheMemoryUsage() const;

    /**
     * @brief Set callback for cache invalidation events
     */
    void setInvalidationCallback(std::function<void(GlobalIndex)> callback);

private:
    CacheOptions cache_options_;
    std::unique_ptr<ElementMatrixCache> cache_;
    bool cache_populated_{false};
    std::function<void(GlobalIndex)> invalidation_callback_;
};

// ============================================================================
// Factory Functions
// ============================================================================

/**
 * @brief Create CachedAssembler with default options
 */
std::unique_ptr<Assembler> createCachedAssembler();

/**
 * @brief Create CachedAssembler with specified cache options
 */
std::unique_ptr<Assembler> createCachedAssembler(const CacheOptions& options);

} // namespace assembly
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_ASSEMBLY_CACHED_ASSEMBLER_H

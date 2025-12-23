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

#ifndef SVMP_FE_SPARSITY_CACHE_H
#define SVMP_FE_SPARSITY_CACHE_H

/**
 * @file SparsityCache.h
 * @brief Cache and reuse sparsity patterns for FEM matrix assembly
 *
 * This header provides the SparsityCache class for caching and reusing
 * sparsity patterns. In many FEM workflows, the same sparsity pattern is
 * needed across multiple time steps or Newton iterations. Caching avoids
 * redundant pattern construction.
 *
 * Key features:
 * - LRU (Least Recently Used) eviction policy
 * - Thread-safe access with read-write locking
 * - Configurable memory limits
 * - Signature-based lookup for pattern matching
 * - Statistics for cache performance monitoring
 *
 * Performance notes:
 * - Lookup: O(1) average with hash table
 * - Eviction: O(1) with doubly-linked list for LRU
 * - Memory: Configurable limit with automatic eviction
 * - Thread safety: Read-write lock for concurrent access
 *
 * Determinism: Pattern storage and retrieval are deterministic.
 * The same signature always maps to the same pattern.
 *
 * @see SparsityPattern for the cached data structure
 * @see SparsityFactory for pattern creation with cache integration
 */

#include "SparsityPattern.h"
#include "Core/Types.h"
#include "Core/FEConfig.h"
#include "Core/FEException.h"

#include <memory>
#include <shared_mutex>
#include <unordered_map>
#include <list>
#include <string>
#include <chrono>
#include <optional>
#include <functional>
#include <atomic>

namespace svmp {
namespace FE {
namespace sparsity {

// ============================================================================
// Cache Configuration
// ============================================================================

/**
 * @brief Configuration options for SparsityCache
 */
struct CacheConfig {
    /**
     * @brief Maximum memory usage in bytes
     *
     * When exceeded, oldest entries are evicted.
     * Default: 1 GB
     */
    std::size_t max_memory_bytes{1024 * 1024 * 1024};

    /**
     * @brief Maximum number of cached patterns
     *
     * When exceeded, oldest entries are evicted.
     * Default: 100 patterns
     */
    std::size_t max_entries{100};

    /**
     * @brief Enable automatic eviction
     *
     * If false, cache will not evict entries automatically.
     * Insert will fail when limits are exceeded.
     */
    bool auto_evict{true};

    /**
     * @brief Enable thread-safe access
     *
     * If false, caller is responsible for synchronization.
     * Slightly faster for single-threaded use.
     */
    bool thread_safe{true};

    /**
     * @brief Enable statistics collection
     *
     * Small performance overhead when enabled.
     */
    bool collect_stats{true};

    /**
     * @brief Time-to-live for entries in seconds
     *
     * Entries older than this are considered stale.
     * 0 = no expiration (default)
     */
    std::chrono::seconds ttl{0};

    /**
     * @brief Enable pattern validation on retrieval
     *
     * Validates pattern integrity when retrieved from cache.
     * Small performance overhead.
     */
    bool validate_on_get{false};
};

/**
 * @brief Statistics for cache performance monitoring
 */
struct CacheStats {
    std::size_t hits{0};           ///< Number of cache hits
    std::size_t misses{0};         ///< Number of cache misses
    std::size_t evictions{0};      ///< Number of evictions
    std::size_t insertions{0};     ///< Number of insertions
    std::size_t current_entries{0}; ///< Current number of entries
    std::size_t current_bytes{0};  ///< Current memory usage
    std::size_t peak_entries{0};   ///< Peak number of entries
    std::size_t peak_bytes{0};     ///< Peak memory usage
    std::size_t expired{0};        ///< Number of expired entries removed

    /**
     * @brief Get hit rate (0.0 to 1.0)
     */
    [[nodiscard]] double hitRate() const noexcept {
        std::size_t total = hits + misses;
        return total > 0 ? static_cast<double>(hits) / static_cast<double>(total) : 0.0;
    }

    /**
     * @brief Get miss rate (0.0 to 1.0)
     */
    [[nodiscard]] double missRate() const noexcept {
        return 1.0 - hitRate();
    }

    /**
     * @brief Reset statistics
     */
    void reset() noexcept {
        hits = misses = evictions = insertions = expired = 0;
    }
};

/**
 * @brief Signature for identifying cached patterns
 *
 * A signature uniquely identifies a sparsity pattern based on:
 * - Problem configuration (mesh, DOF map, constraints)
 * - Build options
 * - Any other distinguishing parameters
 */
struct PatternSignature {
    std::string key;             ///< Primary hash key
    GlobalIndex n_rows{0};       ///< Expected number of rows
    GlobalIndex n_cols{0};       ///< Expected number of columns
    std::size_t hash{0};         ///< Precomputed hash value

    PatternSignature() = default;

    /**
     * @brief Construct from string key
     */
    explicit PatternSignature(std::string k)
        : key(std::move(k)) {
        computeHash();
    }

    /**
     * @brief Construct from key and dimensions
     */
    PatternSignature(std::string k, GlobalIndex rows, GlobalIndex cols)
        : key(std::move(k)), n_rows(rows), n_cols(cols) {
        computeHash();
    }

    /**
     * @brief Equality comparison
     */
    bool operator==(const PatternSignature& other) const noexcept {
        return key == other.key && n_rows == other.n_rows && n_cols == other.n_cols;
    }

    /**
     * @brief Check if signature is valid
     */
    [[nodiscard]] bool isValid() const noexcept {
        return !key.empty();
    }

private:
    void computeHash() {
        std::hash<std::string> str_hash;
        hash = str_hash(key);
        hash ^= std::hash<GlobalIndex>{}(n_rows) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        hash ^= std::hash<GlobalIndex>{}(n_cols) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
    }
};

/**
 * @brief Hash functor for PatternSignature
 */
struct SignatureHash {
    std::size_t operator()(const PatternSignature& sig) const noexcept {
        return sig.hash;
    }
};

// ============================================================================
// Cache Entry
// ============================================================================

/**
 * @brief Internal cache entry with metadata
 */
struct CacheEntry {
    std::shared_ptr<SparsityPattern> pattern;
    PatternSignature signature;
    std::chrono::steady_clock::time_point created;
    std::chrono::steady_clock::time_point last_access;
    std::size_t access_count{0};
    std::size_t memory_bytes{0};

    CacheEntry() = default;

    CacheEntry(std::shared_ptr<SparsityPattern> p, PatternSignature sig)
        : pattern(std::move(p)),
          signature(std::move(sig)),
          created(std::chrono::steady_clock::now()),
          last_access(created),
          access_count(1) {
        if (pattern) {
            memory_bytes = pattern->memoryUsageBytes();
        }
    }

    /**
     * @brief Update access metadata
     */
    void touch() {
        last_access = std::chrono::steady_clock::now();
        ++access_count;
    }

    /**
     * @brief Check if entry is expired
     */
    [[nodiscard]] bool isExpired(std::chrono::seconds ttl) const {
        if (ttl.count() == 0) return false;
        auto age = std::chrono::steady_clock::now() - created;
        return age > ttl;
    }
};

// ============================================================================
// SparsityCache Class
// ============================================================================

/**
 * @brief Thread-safe LRU cache for sparsity patterns
 *
 * SparsityCache provides efficient storage and retrieval of sparsity patterns
 * with configurable memory limits and LRU eviction policy.
 *
 * Usage:
 * @code
 * // Create cache with default config
 * SparsityCache cache;
 *
 * // Or with custom config
 * CacheConfig config;
 * config.max_memory_bytes = 512 * 1024 * 1024;  // 512 MB
 * config.max_entries = 50;
 * SparsityCache cache(config);
 *
 * // Generate signature for pattern
 * PatternSignature sig("mesh_v1_dofmap_p2", n_rows, n_cols);
 *
 * // Try to get from cache
 * auto pattern = cache.get(sig);
 * if (!pattern) {
 *     // Build pattern
 *     pattern = std::make_shared<SparsityPattern>(buildPattern(...));
 *     cache.put(sig, pattern);
 * }
 *
 * // Check statistics
 * auto stats = cache.getStats();
 * std::cout << "Hit rate: " << stats.hitRate() * 100 << "%" << std::endl;
 * @endcode
 *
 * Thread safety:
 * - All public methods are thread-safe when config.thread_safe is true
 * - Uses shared_mutex for read-write locking
 * - Multiple readers can access concurrently
 * - Writers have exclusive access
 */
class SparsityCache {
public:
    // =========================================================================
    // Construction
    // =========================================================================

    /**
     * @brief Default constructor with default configuration
     */
    SparsityCache();

    /**
     * @brief Construct with custom configuration
     *
     * @param config Cache configuration options
     */
    explicit SparsityCache(const CacheConfig& config);

    /// Destructor
    ~SparsityCache() = default;

    // Non-copyable (contains mutex)
    SparsityCache(const SparsityCache&) = delete;
    SparsityCache& operator=(const SparsityCache&) = delete;

    // Movable
    SparsityCache(SparsityCache&& other) noexcept;
    SparsityCache& operator=(SparsityCache&& other) noexcept;

    // =========================================================================
    // Configuration
    // =========================================================================

    /**
     * @brief Get current configuration
     */
    [[nodiscard]] const CacheConfig& getConfig() const noexcept {
        return config_;
    }

    /**
     * @brief Update configuration
     *
     * @param config New configuration
     *
     * @note May trigger evictions if new limits are lower than current usage.
     */
    void setConfig(const CacheConfig& config);

    /**
     * @brief Set maximum memory limit
     *
     * @param max_bytes Maximum memory in bytes
     */
    void setMaxMemory(std::size_t max_bytes);

    /**
     * @brief Set maximum entry limit
     *
     * @param max_entries Maximum number of entries
     */
    void setMaxEntries(std::size_t max_entries);

    // =========================================================================
    // Core Operations
    // =========================================================================

    /**
     * @brief Get pattern from cache
     *
     * @param signature Pattern signature
     * @return Shared pointer to pattern, or nullptr if not found
     *
     * Thread-safe. Updates LRU order on hit.
     */
    [[nodiscard]] std::shared_ptr<SparsityPattern> get(const PatternSignature& signature);

    /**
     * @brief Try to get pattern, return optional
     *
     * @param signature Pattern signature
     * @return Optional containing shared_ptr on hit, empty on miss
     */
    [[nodiscard]] std::optional<std::shared_ptr<SparsityPattern>> tryGet(
        const PatternSignature& signature);

    /**
     * @brief Put pattern into cache
     *
     * @param signature Pattern signature
     * @param pattern Pattern to cache
     * @return true if inserted, false if failed (e.g., too large)
     *
     * Thread-safe. May evict older entries if limits are exceeded.
     */
    bool put(const PatternSignature& signature,
             std::shared_ptr<SparsityPattern> pattern);

    /**
     * @brief Get or create pattern
     *
     * @param signature Pattern signature
     * @param factory Function to create pattern if not cached
     * @return Shared pointer to pattern (from cache or newly created)
     *
     * Thread-safe. Atomically checks cache and creates if needed.
     */
    [[nodiscard]] std::shared_ptr<SparsityPattern> getOrCreate(
        const PatternSignature& signature,
        std::function<std::shared_ptr<SparsityPattern>()> factory);

    /**
     * @brief Check if pattern exists in cache
     *
     * @param signature Pattern signature
     * @return true if pattern is cached
     *
     * Does not update LRU order.
     */
    [[nodiscard]] bool contains(const PatternSignature& signature) const;

    /**
     * @brief Remove pattern from cache
     *
     * @param signature Pattern signature
     * @return true if removed, false if not found
     */
    bool remove(const PatternSignature& signature);

    /**
     * @brief Clear all cached patterns
     */
    void clear();

    /**
     * @brief Remove expired entries (if TTL is configured)
     *
     * @return Number of entries removed
     */
    std::size_t purgeExpired();

    // =========================================================================
    // Eviction
    // =========================================================================

    /**
     * @brief Evict least recently used entry
     *
     * @return true if an entry was evicted
     */
    bool evictOne();

    /**
     * @brief Evict entries until memory is below limit
     *
     * @param target_bytes Target memory usage
     * @return Number of entries evicted
     */
    std::size_t evictToMemory(std::size_t target_bytes);

    /**
     * @brief Evict entries until count is below limit
     *
     * @param target_count Target entry count
     * @return Number of entries evicted
     */
    std::size_t evictToCount(std::size_t target_count);

    // =========================================================================
    // Statistics and Introspection
    // =========================================================================

    /**
     * @brief Get cache statistics
     *
     * @return Copy of current statistics
     */
    [[nodiscard]] CacheStats getStats() const;

    /**
     * @brief Reset statistics
     */
    void resetStats();

    /**
     * @brief Get number of cached patterns
     */
    [[nodiscard]] std::size_t size() const noexcept;

    /**
     * @brief Check if cache is empty
     */
    [[nodiscard]] bool empty() const noexcept;

    /**
     * @brief Get current memory usage in bytes
     */
    [[nodiscard]] std::size_t memoryUsage() const noexcept;

    /**
     * @brief Get remaining memory capacity
     */
    [[nodiscard]] std::size_t remainingMemory() const noexcept;

    /**
     * @brief Get list of all cached signatures
     *
     * @return Vector of signatures
     */
    [[nodiscard]] std::vector<PatternSignature> getCachedSignatures() const;

    /**
     * @brief Get entry metadata (for debugging)
     *
     * @param signature Pattern signature
     * @return Optional entry metadata
     */
    [[nodiscard]] std::optional<CacheEntry> getEntryInfo(
        const PatternSignature& signature) const;

    // =========================================================================
    // Validation
    // =========================================================================

    /**
     * @brief Validate all cached patterns
     *
     * @return true if all patterns are valid
     */
    [[nodiscard]] bool validateAll() const;

    /**
     * @brief Remove invalid entries
     *
     * @return Number of invalid entries removed
     */
    std::size_t purgeInvalid();

private:
    // LRU list type: stores iterators to map entries for O(1) reordering
    using LRUList = std::list<PatternSignature>;
    using LRUIterator = LRUList::iterator;

    // Map from signature to (entry, LRU iterator)
    struct CacheNode {
        CacheEntry entry;
        LRUIterator lru_iter;
    };
    using CacheMap = std::unordered_map<PatternSignature, CacheNode, SignatureHash>;

    // Internal implementation
    void evictIfNeeded();
    void updateLRU(typename CacheMap::iterator it);
    void updateStats(bool hit);

    // Data members
    CacheConfig config_;
    CacheMap cache_map_;
    LRUList lru_list_;
    mutable CacheStats stats_;

    // Synchronization
    mutable std::shared_mutex mutex_;

    // Current memory usage
    std::atomic<std::size_t> current_memory_{0};
};

// ============================================================================
// Global Cache Instance
// ============================================================================

/**
 * @brief Get the global sparsity cache instance
 *
 * Thread-safe singleton access to a shared cache.
 */
[[nodiscard]] SparsityCache& getGlobalCache();

/**
 * @brief Set global cache configuration
 *
 * @param config New configuration
 */
void configureGlobalCache(const CacheConfig& config);

/**
 * @brief Clear the global cache
 */
void clearGlobalCache();

// ============================================================================
// Signature Builders
// ============================================================================

/**
 * @brief Build signature from DOF map properties
 *
 * @param n_dofs Number of DOFs
 * @param n_elements Number of elements
 * @param avg_dofs_per_elem Average DOFs per element
 * @param version Optional version string
 * @return Pattern signature
 */
[[nodiscard]] PatternSignature buildSignature(
    GlobalIndex n_dofs,
    GlobalIndex n_elements,
    GlobalIndex avg_dofs_per_elem,
    const std::string& version = "");

/**
 * @brief Build signature from mesh and DOF map hash
 *
 * @param mesh_hash Hash of mesh topology
 * @param dof_hash Hash of DOF configuration
 * @param n_rows Expected rows
 * @param n_cols Expected columns
 * @return Pattern signature
 */
[[nodiscard]] PatternSignature buildSignature(
    std::size_t mesh_hash,
    std::size_t dof_hash,
    GlobalIndex n_rows,
    GlobalIndex n_cols);

/**
 * @brief Compute hash for array of indices
 *
 * @param indices Index array
 * @return Hash value
 */
[[nodiscard]] std::size_t computeIndexHash(std::span<const GlobalIndex> indices);

} // namespace sparsity
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_SPARSITY_CACHE_H

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

#include "SparsityCache.h"
#include <mutex>
#include <sstream>
#include <iomanip>

namespace svmp {
namespace FE {
namespace sparsity {

// ============================================================================
// SparsityCache Implementation
// ============================================================================

SparsityCache::SparsityCache()
    : config_()
{
}

SparsityCache::SparsityCache(const CacheConfig& config)
    : config_(config)
{
}

SparsityCache::SparsityCache(SparsityCache&& other) noexcept
    : config_(std::move(other.config_)),
      cache_map_(std::move(other.cache_map_)),
      lru_list_(std::move(other.lru_list_)),
      stats_(std::move(other.stats_)),
      current_memory_(other.current_memory_.load())
{
    other.current_memory_ = 0;
}

SparsityCache& SparsityCache::operator=(SparsityCache&& other) noexcept {
    if (this != &other) {
        std::unique_lock lock(mutex_);
        std::unique_lock other_lock(other.mutex_);

        config_ = std::move(other.config_);
        cache_map_ = std::move(other.cache_map_);
        lru_list_ = std::move(other.lru_list_);
        stats_ = std::move(other.stats_);
        current_memory_ = other.current_memory_.load();
        other.current_memory_ = 0;
    }
    return *this;
}

// ============================================================================
// Configuration
// ============================================================================

void SparsityCache::setConfig(const CacheConfig& config) {
    std::unique_lock lock(mutex_);
    config_ = config;

    // Enforce new limits
    while (cache_map_.size() > config_.max_entries ||
           current_memory_ > config_.max_memory_bytes) {
        if (!evictOne()) break;
    }
}

void SparsityCache::setMaxMemory(std::size_t max_bytes) {
    std::unique_lock lock(mutex_);
    config_.max_memory_bytes = max_bytes;

    // Enforce new limit
    while (current_memory_ > config_.max_memory_bytes) {
        if (!evictOne()) break;
    }
}

void SparsityCache::setMaxEntries(std::size_t max_entries) {
    std::unique_lock lock(mutex_);
    config_.max_entries = max_entries;

    // Enforce new limit
    while (cache_map_.size() > config_.max_entries) {
        if (!evictOne()) break;
    }
}

// ============================================================================
// Core Operations
// ============================================================================

std::shared_ptr<SparsityPattern> SparsityCache::get(const PatternSignature& signature) {
    if (config_.thread_safe) {
        // First try with shared lock (read)
        {
            std::shared_lock read_lock(mutex_);
            auto it = cache_map_.find(signature);
            if (it == cache_map_.end()) {
                if (config_.collect_stats) {
                    ++stats_.misses;
                }
                return nullptr;
            }

            // Check expiration
            if (it->second.entry.isExpired(config_.ttl)) {
                // Need to upgrade to exclusive lock to remove
                read_lock.unlock();
                std::unique_lock write_lock(mutex_);
                // Re-check after acquiring write lock
                it = cache_map_.find(signature);
                if (it != cache_map_.end() && it->second.entry.isExpired(config_.ttl)) {
                    lru_list_.erase(it->second.lru_iter);
                    current_memory_ -= it->second.entry.memory_bytes;
                    cache_map_.erase(it);
                    if (config_.collect_stats) {
                        ++stats_.expired;
                        ++stats_.misses;
                    }
                }
                return nullptr;
            }
        }

        // Need exclusive lock to update LRU
        std::unique_lock write_lock(mutex_);
        auto it = cache_map_.find(signature);
        if (it == cache_map_.end()) {
            if (config_.collect_stats) {
                ++stats_.misses;
            }
            return nullptr;
        }

        // Validate if configured
        if (config_.validate_on_get && it->second.entry.pattern) {
            if (!it->second.entry.pattern->validate()) {
                // Invalid entry - remove it
                lru_list_.erase(it->second.lru_iter);
                current_memory_ -= it->second.entry.memory_bytes;
                cache_map_.erase(it);
                if (config_.collect_stats) {
                    ++stats_.misses;
                }
                return nullptr;
            }
        }

        // Update LRU
        updateLRU(it);
        it->second.entry.touch();

        if (config_.collect_stats) {
            ++stats_.hits;
        }

        return it->second.entry.pattern;
    } else {
        // Non-thread-safe path
        auto it = cache_map_.find(signature);
        if (it == cache_map_.end()) {
            if (config_.collect_stats) {
                ++stats_.misses;
            }
            return nullptr;
        }

        // Check expiration
        if (it->second.entry.isExpired(config_.ttl)) {
            lru_list_.erase(it->second.lru_iter);
            current_memory_ -= it->second.entry.memory_bytes;
            cache_map_.erase(it);
            if (config_.collect_stats) {
                ++stats_.expired;
                ++stats_.misses;
            }
            return nullptr;
        }

        // Validate if configured
        if (config_.validate_on_get && it->second.entry.pattern) {
            if (!it->second.entry.pattern->validate()) {
                lru_list_.erase(it->second.lru_iter);
                current_memory_ -= it->second.entry.memory_bytes;
                cache_map_.erase(it);
                if (config_.collect_stats) {
                    ++stats_.misses;
                }
                return nullptr;
            }
        }

        // Update LRU
        updateLRU(it);
        it->second.entry.touch();

        if (config_.collect_stats) {
            ++stats_.hits;
        }

        return it->second.entry.pattern;
    }
}

std::optional<std::shared_ptr<SparsityPattern>> SparsityCache::tryGet(
    const PatternSignature& signature) {
    auto result = get(signature);
    if (result) {
        return result;
    }
    return std::nullopt;
}

bool SparsityCache::put(const PatternSignature& signature,
                        std::shared_ptr<SparsityPattern> pattern) {
    if (!pattern) {
        return false;
    }

    std::size_t pattern_size = pattern->memoryUsageBytes();

    // Check if pattern is too large to cache
    if (pattern_size > config_.max_memory_bytes) {
        return false;
    }

    std::unique_lock<std::shared_mutex> lock_guard;
    if (config_.thread_safe) {
        lock_guard = std::unique_lock<std::shared_mutex>(mutex_);
    }

    // Check if already exists - update if so
    auto existing = cache_map_.find(signature);
    if (existing != cache_map_.end()) {
        // Update existing entry
        current_memory_ -= existing->second.entry.memory_bytes;
        existing->second.entry = CacheEntry(pattern, signature);
        current_memory_ += pattern_size;
        updateLRU(existing);
        return true;
    }

    // Evict if needed
    if (config_.auto_evict) {
        evictIfNeeded();

        // Make room for new entry
        while (current_memory_ + pattern_size > config_.max_memory_bytes ||
               cache_map_.size() >= config_.max_entries) {
            if (!evictOne()) {
                // Can't evict anymore
                if (!config_.auto_evict) {
                    return false;
                }
                break;
            }
        }
    } else {
        // Check if we have room
        if (current_memory_ + pattern_size > config_.max_memory_bytes ||
            cache_map_.size() >= config_.max_entries) {
            return false;
        }
    }

    // Insert new entry
    lru_list_.push_front(signature);
    CacheNode node;
    node.entry = CacheEntry(pattern, signature);
    node.lru_iter = lru_list_.begin();
    cache_map_.emplace(signature, std::move(node));

    current_memory_ += pattern_size;

    if (config_.collect_stats) {
        ++stats_.insertions;
        stats_.current_entries = cache_map_.size();
        stats_.current_bytes = current_memory_;
        if (stats_.current_entries > stats_.peak_entries) {
            stats_.peak_entries = stats_.current_entries;
        }
        if (stats_.current_bytes > stats_.peak_bytes) {
            stats_.peak_bytes = stats_.current_bytes;
        }
    }

    return true;
}

std::shared_ptr<SparsityPattern> SparsityCache::getOrCreate(
    const PatternSignature& signature,
    std::function<std::shared_ptr<SparsityPattern>()> factory) {

    // Try to get from cache first
    auto cached = get(signature);
    if (cached) {
        return cached;
    }

    // Create new pattern
    auto pattern = factory();
    if (!pattern) {
        return nullptr;
    }

    // Try to cache it
    put(signature, pattern);

    return pattern;
}

bool SparsityCache::contains(const PatternSignature& signature) const {
    if (config_.thread_safe) {
        std::shared_lock lock(mutex_);
        auto it = cache_map_.find(signature);
        if (it == cache_map_.end()) return false;
        if (it->second.entry.isExpired(config_.ttl)) return false;
        return true;
    } else {
        auto it = cache_map_.find(signature);
        if (it == cache_map_.end()) return false;
        if (it->second.entry.isExpired(config_.ttl)) return false;
        return true;
    }
}

bool SparsityCache::remove(const PatternSignature& signature) {
    std::unique_lock<std::shared_mutex> lock_guard;
    if (config_.thread_safe) {
        lock_guard = std::unique_lock<std::shared_mutex>(mutex_);
    }

    auto it = cache_map_.find(signature);
    if (it == cache_map_.end()) {
        return false;
    }

    lru_list_.erase(it->second.lru_iter);
    current_memory_ -= it->second.entry.memory_bytes;
    cache_map_.erase(it);

    if (config_.collect_stats) {
        stats_.current_entries = cache_map_.size();
        stats_.current_bytes = current_memory_;
    }

    return true;
}

void SparsityCache::clear() {
    std::unique_lock<std::shared_mutex> lock_guard;
    if (config_.thread_safe) {
        lock_guard = std::unique_lock<std::shared_mutex>(mutex_);
    }

    cache_map_.clear();
    lru_list_.clear();
    current_memory_ = 0;

    if (config_.collect_stats) {
        stats_.current_entries = 0;
        stats_.current_bytes = 0;
    }
}

std::size_t SparsityCache::purgeExpired() {
    if (config_.ttl.count() == 0) {
        return 0;
    }

    std::unique_lock<std::shared_mutex> lock_guard;
    if (config_.thread_safe) {
        lock_guard = std::unique_lock<std::shared_mutex>(mutex_);
    }

    std::size_t removed = 0;
    auto it = cache_map_.begin();
    while (it != cache_map_.end()) {
        if (it->second.entry.isExpired(config_.ttl)) {
            lru_list_.erase(it->second.lru_iter);
            current_memory_ -= it->second.entry.memory_bytes;
            it = cache_map_.erase(it);
            ++removed;
        } else {
            ++it;
        }
    }

    if (config_.collect_stats) {
        stats_.expired += removed;
        stats_.current_entries = cache_map_.size();
        stats_.current_bytes = current_memory_;
    }

    return removed;
}

// ============================================================================
// Eviction
// ============================================================================

bool SparsityCache::evictOne() {
    // Must be called with lock held
    if (lru_list_.empty()) {
        return false;
    }

    // Get least recently used (back of list)
    const auto& signature = lru_list_.back();
    auto it = cache_map_.find(signature);
    if (it != cache_map_.end()) {
        current_memory_ -= it->second.entry.memory_bytes;
        cache_map_.erase(it);
    }
    lru_list_.pop_back();

    if (config_.collect_stats) {
        ++stats_.evictions;
        stats_.current_entries = cache_map_.size();
        stats_.current_bytes = current_memory_;
    }

    return true;
}

std::size_t SparsityCache::evictToMemory(std::size_t target_bytes) {
    std::unique_lock<std::shared_mutex> lock_guard;
    if (config_.thread_safe) {
        lock_guard = std::unique_lock<std::shared_mutex>(mutex_);
    }

    std::size_t evicted = 0;
    while (current_memory_ > target_bytes && !lru_list_.empty()) {
        if (evictOne()) {
            ++evicted;
        } else {
            break;
        }
    }
    return evicted;
}

std::size_t SparsityCache::evictToCount(std::size_t target_count) {
    std::unique_lock<std::shared_mutex> lock_guard;
    if (config_.thread_safe) {
        lock_guard = std::unique_lock<std::shared_mutex>(mutex_);
    }

    std::size_t evicted = 0;
    while (cache_map_.size() > target_count && !lru_list_.empty()) {
        if (evictOne()) {
            ++evicted;
        } else {
            break;
        }
    }
    return evicted;
}

void SparsityCache::evictIfNeeded() {
    // Called with lock held
    // Remove expired entries first
    if (config_.ttl.count() > 0) {
        auto it = cache_map_.begin();
        while (it != cache_map_.end()) {
            if (it->second.entry.isExpired(config_.ttl)) {
                lru_list_.erase(it->second.lru_iter);
                current_memory_ -= it->second.entry.memory_bytes;
                it = cache_map_.erase(it);
                if (config_.collect_stats) {
                    ++stats_.expired;
                }
            } else {
                ++it;
            }
        }
    }
}

void SparsityCache::updateLRU(typename CacheMap::iterator it) {
    // Move to front of LRU list
    lru_list_.erase(it->second.lru_iter);
    lru_list_.push_front(it->first);
    it->second.lru_iter = lru_list_.begin();
}

void SparsityCache::updateStats(bool hit) {
    if (!config_.collect_stats) return;

    if (hit) {
        ++stats_.hits;
    } else {
        ++stats_.misses;
    }
}

// ============================================================================
// Statistics and Introspection
// ============================================================================

CacheStats SparsityCache::getStats() const {
    if (config_.thread_safe) {
        std::shared_lock lock(mutex_);
        CacheStats result = stats_;
        result.current_entries = cache_map_.size();
        result.current_bytes = current_memory_;
        return result;
    } else {
        CacheStats result = stats_;
        result.current_entries = cache_map_.size();
        result.current_bytes = current_memory_;
        return result;
    }
}

void SparsityCache::resetStats() {
    std::unique_lock<std::shared_mutex> lock_guard;
    if (config_.thread_safe) {
        lock_guard = std::unique_lock<std::shared_mutex>(mutex_);
    }

    stats_.reset();
    stats_.current_entries = cache_map_.size();
    stats_.current_bytes = current_memory_;
    stats_.peak_entries = stats_.current_entries;
    stats_.peak_bytes = stats_.current_bytes;
}

std::size_t SparsityCache::size() const noexcept {
    if (config_.thread_safe) {
        std::shared_lock lock(mutex_);
        return cache_map_.size();
    }
    return cache_map_.size();
}

bool SparsityCache::empty() const noexcept {
    if (config_.thread_safe) {
        std::shared_lock lock(mutex_);
        return cache_map_.empty();
    }
    return cache_map_.empty();
}

std::size_t SparsityCache::memoryUsage() const noexcept {
    return current_memory_;
}

std::size_t SparsityCache::remainingMemory() const noexcept {
    std::size_t current = current_memory_;
    if (current >= config_.max_memory_bytes) {
        return 0;
    }
    return config_.max_memory_bytes - current;
}

std::vector<PatternSignature> SparsityCache::getCachedSignatures() const {
    if (config_.thread_safe) {
        std::shared_lock lock(mutex_);
        std::vector<PatternSignature> result;
        result.reserve(cache_map_.size());
        for (const auto& [sig, node] : cache_map_) {
            result.push_back(sig);
        }
        return result;
    } else {
        std::vector<PatternSignature> result;
        result.reserve(cache_map_.size());
        for (const auto& [sig, node] : cache_map_) {
            result.push_back(sig);
        }
        return result;
    }
}

std::optional<CacheEntry> SparsityCache::getEntryInfo(
    const PatternSignature& signature) const {
    if (config_.thread_safe) {
        std::shared_lock lock(mutex_);
        auto it = cache_map_.find(signature);
        if (it == cache_map_.end()) {
            return std::nullopt;
        }
        return it->second.entry;
    } else {
        auto it = cache_map_.find(signature);
        if (it == cache_map_.end()) {
            return std::nullopt;
        }
        return it->second.entry;
    }
}

// ============================================================================
// Validation
// ============================================================================

bool SparsityCache::validateAll() const {
    if (config_.thread_safe) {
        std::shared_lock lock(mutex_);
        for (const auto& [sig, node] : cache_map_) {
            if (node.entry.pattern && !node.entry.pattern->validate()) {
                return false;
            }
        }
        return true;
    } else {
        for (const auto& [sig, node] : cache_map_) {
            if (node.entry.pattern && !node.entry.pattern->validate()) {
                return false;
            }
        }
        return true;
    }
}

std::size_t SparsityCache::purgeInvalid() {
    std::unique_lock<std::shared_mutex> lock_guard;
    if (config_.thread_safe) {
        lock_guard = std::unique_lock<std::shared_mutex>(mutex_);
    }

    std::size_t removed = 0;
    auto it = cache_map_.begin();
    while (it != cache_map_.end()) {
        bool invalid = !it->second.entry.pattern ||
                       !it->second.entry.pattern->validate();
        if (invalid) {
            lru_list_.erase(it->second.lru_iter);
            current_memory_ -= it->second.entry.memory_bytes;
            it = cache_map_.erase(it);
            ++removed;
        } else {
            ++it;
        }
    }

    if (config_.collect_stats) {
        stats_.current_entries = cache_map_.size();
        stats_.current_bytes = current_memory_;
    }

    return removed;
}

// ============================================================================
// Global Cache Instance
// ============================================================================

namespace {

// Thread-safe singleton using Meyer's singleton pattern
SparsityCache& globalCacheInstance() {
    static SparsityCache instance;
    return instance;
}

} // anonymous namespace

SparsityCache& getGlobalCache() {
    return globalCacheInstance();
}

void configureGlobalCache(const CacheConfig& config) {
    getGlobalCache().setConfig(config);
}

void clearGlobalCache() {
    getGlobalCache().clear();
}

// ============================================================================
// Signature Builders
// ============================================================================

PatternSignature buildSignature(
    GlobalIndex n_dofs,
    GlobalIndex n_elements,
    GlobalIndex avg_dofs_per_elem,
    const std::string& version) {

    std::ostringstream oss;
    oss << "sparsity_" << n_dofs << "_" << n_elements << "_" << avg_dofs_per_elem;
    if (!version.empty()) {
        oss << "_" << version;
    }

    return PatternSignature(oss.str(), n_dofs, n_dofs);
}

PatternSignature buildSignature(
    std::size_t mesh_hash,
    std::size_t dof_hash,
    GlobalIndex n_rows,
    GlobalIndex n_cols) {

    std::ostringstream oss;
    oss << std::hex << std::setfill('0');
    oss << "sparsity_m" << std::setw(16) << mesh_hash;
    oss << "_d" << std::setw(16) << dof_hash;

    return PatternSignature(oss.str(), n_rows, n_cols);
}

std::size_t computeIndexHash(std::span<const GlobalIndex> indices) {
    std::size_t hash = 0;
    for (GlobalIndex idx : indices) {
        // FNV-1a hash variant
        hash ^= static_cast<std::size_t>(idx);
        hash *= 0x100000001b3ULL;
    }
    return hash;
}

} // namespace sparsity
} // namespace FE
} // namespace svmp

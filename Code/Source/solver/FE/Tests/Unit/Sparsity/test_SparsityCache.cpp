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
#include "Sparsity/SparsityCache.h"
#include <thread>
#include <vector>
#include <atomic>

using namespace svmp::FE;
using namespace svmp::FE::sparsity;

// ============================================================================
// Helper Functions
// ============================================================================

std::shared_ptr<SparsityPattern> createTestPattern(GlobalIndex n, GlobalIndex nnz_per_row = 5) {
    auto pattern = std::make_shared<SparsityPattern>(n, n);
    for (GlobalIndex row = 0; row < n; ++row) {
        pattern->addEntry(row, row);  // Diagonal
        for (GlobalIndex k = 1; k < nnz_per_row && row + k < n; ++k) {
            pattern->addEntry(row, row + k);
            pattern->addEntry(row + k, row);
        }
    }
    pattern->finalize();
    return pattern;
}

// ============================================================================
// PatternSignature Tests
// ============================================================================

TEST(PatternSignatureTest, DefaultConstruction) {
    PatternSignature sig;
    EXPECT_TRUE(sig.key.empty());
    EXPECT_FALSE(sig.isValid());
}

TEST(PatternSignatureTest, StringConstruction) {
    PatternSignature sig("test_key");
    EXPECT_EQ(sig.key, "test_key");
    EXPECT_TRUE(sig.isValid());
}

TEST(PatternSignatureTest, FullConstruction) {
    PatternSignature sig("mesh_v1", 100, 100);
    EXPECT_EQ(sig.key, "mesh_v1");
    EXPECT_EQ(sig.n_rows, 100);
    EXPECT_EQ(sig.n_cols, 100);
    EXPECT_TRUE(sig.isValid());
}

TEST(PatternSignatureTest, Equality) {
    PatternSignature sig1("key", 10, 10);
    PatternSignature sig2("key", 10, 10);
    PatternSignature sig3("key", 10, 20);
    PatternSignature sig4("other_key", 10, 10);

    EXPECT_TRUE(sig1 == sig2);
    EXPECT_FALSE(sig1 == sig3);
    EXPECT_FALSE(sig1 == sig4);
}

TEST(PatternSignatureTest, Hash) {
    PatternSignature sig1("key", 10, 10);
    PatternSignature sig2("key", 10, 10);
    PatternSignature sig3("other_key", 10, 10);

    SignatureHash hasher;
    EXPECT_EQ(hasher(sig1), hasher(sig2));
    EXPECT_NE(hasher(sig1), hasher(sig3));
}

// ============================================================================
// SparsityCache Basic Tests
// ============================================================================

TEST(SparsityCacheTest, DefaultConstruction) {
    SparsityCache cache;
    EXPECT_TRUE(cache.empty());
    EXPECT_EQ(cache.size(), 0);
    EXPECT_EQ(cache.memoryUsage(), 0);
}

TEST(SparsityCacheTest, CustomConfiguration) {
    CacheConfig config;
    config.max_memory_bytes = 1024 * 1024;  // 1 MB
    config.max_entries = 10;
    config.auto_evict = true;

    SparsityCache cache(config);
    EXPECT_EQ(cache.getConfig().max_memory_bytes, 1024 * 1024);
    EXPECT_EQ(cache.getConfig().max_entries, 10);
}

TEST(SparsityCacheTest, PutAndGet) {
    SparsityCache cache;

    PatternSignature sig("test_pattern", 10, 10);
    auto pattern = createTestPattern(10);

    EXPECT_TRUE(cache.put(sig, pattern));
    EXPECT_EQ(cache.size(), 1);
    EXPECT_TRUE(cache.contains(sig));

    auto retrieved = cache.get(sig);
    ASSERT_NE(retrieved, nullptr);
    EXPECT_EQ(retrieved->numRows(), 10);
    EXPECT_EQ(retrieved->numCols(), 10);
}

TEST(SparsityCacheTest, CacheMiss) {
    SparsityCache cache;

    PatternSignature sig("nonexistent", 10, 10);
    auto retrieved = cache.get(sig);
    EXPECT_EQ(retrieved, nullptr);
}

TEST(SparsityCacheTest, TryGet) {
    SparsityCache cache;

    PatternSignature sig("test_pattern", 10, 10);
    auto pattern = createTestPattern(10);
    cache.put(sig, pattern);

    auto result = cache.tryGet(sig);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ((*result)->numRows(), 10);

    PatternSignature missing_sig("missing", 5, 5);
    auto missing_result = cache.tryGet(missing_sig);
    EXPECT_FALSE(missing_result.has_value());
}

TEST(SparsityCacheTest, UpdateExisting) {
    SparsityCache cache;

    PatternSignature sig("pattern", 10, 10);
    auto pattern1 = createTestPattern(10, 3);
    auto pattern2 = createTestPattern(10, 5);

    cache.put(sig, pattern1);
    GlobalIndex nnz1 = cache.get(sig)->getNnz();

    cache.put(sig, pattern2);
    GlobalIndex nnz2 = cache.get(sig)->getNnz();

    EXPECT_EQ(cache.size(), 1);  // Still one entry
    EXPECT_NE(nnz1, nnz2);  // Different NNZ
}

TEST(SparsityCacheTest, Remove) {
    SparsityCache cache;

    PatternSignature sig("pattern", 10, 10);
    auto pattern = createTestPattern(10);
    cache.put(sig, pattern);

    EXPECT_TRUE(cache.contains(sig));
    EXPECT_TRUE(cache.remove(sig));
    EXPECT_FALSE(cache.contains(sig));
    EXPECT_FALSE(cache.remove(sig));  // Already removed
}

TEST(SparsityCacheTest, Clear) {
    SparsityCache cache;

    for (int i = 0; i < 5; ++i) {
        PatternSignature sig("pattern_" + std::to_string(i), 10, 10);
        cache.put(sig, createTestPattern(10));
    }

    EXPECT_EQ(cache.size(), 5);
    cache.clear();
    EXPECT_EQ(cache.size(), 0);
    EXPECT_TRUE(cache.empty());
}

// ============================================================================
// GetOrCreate Tests
// ============================================================================

TEST(SparsityCacheTest, GetOrCreateCacheHit) {
    SparsityCache cache;

    PatternSignature sig("pattern", 10, 10);
    auto original = createTestPattern(10);
    cache.put(sig, original);

    bool factory_called = false;
    auto result = cache.getOrCreate(sig, [&]() {
        factory_called = true;
        return createTestPattern(10);
    });

    EXPECT_FALSE(factory_called);
    EXPECT_EQ(result, original);
}

TEST(SparsityCacheTest, GetOrCreateCacheMiss) {
    SparsityCache cache;

    PatternSignature sig("pattern", 10, 10);
    bool factory_called = false;

    auto result = cache.getOrCreate(sig, [&]() {
        factory_called = true;
        return createTestPattern(10);
    });

    EXPECT_TRUE(factory_called);
    EXPECT_NE(result, nullptr);
    EXPECT_TRUE(cache.contains(sig));
}

// ============================================================================
// LRU Eviction Tests
// ============================================================================

TEST(SparsityCacheTest, LRUEviction) {
    CacheConfig config;
    config.max_entries = 3;
    config.auto_evict = true;
    SparsityCache cache(config);

    PatternSignature sig1("pattern_1", 10, 10);
    PatternSignature sig2("pattern_2", 10, 10);
    PatternSignature sig3("pattern_3", 10, 10);
    PatternSignature sig4("pattern_4", 10, 10);

    cache.put(sig1, createTestPattern(10));
    cache.put(sig2, createTestPattern(10));
    cache.put(sig3, createTestPattern(10));

    EXPECT_EQ(cache.size(), 3);

    // Adding a 4th should evict sig1 (oldest)
    cache.put(sig4, createTestPattern(10));

    EXPECT_EQ(cache.size(), 3);
    EXPECT_FALSE(cache.contains(sig1));  // Evicted
    EXPECT_TRUE(cache.contains(sig2));
    EXPECT_TRUE(cache.contains(sig3));
    EXPECT_TRUE(cache.contains(sig4));
}

TEST(SparsityCacheTest, LRUOrderUpdate) {
    CacheConfig config;
    config.max_entries = 3;
    config.auto_evict = true;
    SparsityCache cache(config);

    PatternSignature sig1("pattern_1", 10, 10);
    PatternSignature sig2("pattern_2", 10, 10);
    PatternSignature sig3("pattern_3", 10, 10);
    PatternSignature sig4("pattern_4", 10, 10);

    cache.put(sig1, createTestPattern(10));
    cache.put(sig2, createTestPattern(10));
    cache.put(sig3, createTestPattern(10));

    // Access sig1 to make it most recently used
    cache.get(sig1);

    // Adding sig4 should evict sig2 (now oldest)
    cache.put(sig4, createTestPattern(10));

    EXPECT_TRUE(cache.contains(sig1));   // Still there (was accessed)
    EXPECT_FALSE(cache.contains(sig2));  // Evicted (was oldest)
    EXPECT_TRUE(cache.contains(sig3));
    EXPECT_TRUE(cache.contains(sig4));
}

TEST(SparsityCacheTest, ManualEviction) {
    SparsityCache cache;

    for (int i = 0; i < 5; ++i) {
        PatternSignature sig("pattern_" + std::to_string(i), 10, 10);
        cache.put(sig, createTestPattern(10));
    }

    EXPECT_EQ(cache.size(), 5);

    cache.evictOne();
    EXPECT_EQ(cache.size(), 4);

    cache.evictToCount(2);
    EXPECT_EQ(cache.size(), 2);
}

TEST(SparsityCacheTest, MemoryBasedEviction) {
    CacheConfig config;
    config.max_memory_bytes = 10000;  // Small limit
    config.auto_evict = true;
    SparsityCache cache(config);

    // Keep adding patterns until eviction happens
    int eviction_count = 0;
    for (int i = 0; i < 10; ++i) {
        PatternSignature sig("pattern_" + std::to_string(i), 100, 100);
        bool inserted = cache.put(sig, createTestPattern(100, 10));

        if (cache.memoryUsage() > config.max_memory_bytes) {
            // Should trigger eviction
            ++eviction_count;
        }
    }

    // Memory should be under limit
    EXPECT_LE(cache.memoryUsage(), config.max_memory_bytes);
}

// ============================================================================
// Statistics Tests
// ============================================================================

TEST(SparsityCacheTest, HitMissStatistics) {
    CacheConfig config;
    config.collect_stats = true;
    SparsityCache cache(config);

    PatternSignature sig1("pattern_1", 10, 10);
    PatternSignature sig2("pattern_2", 10, 10);
    cache.put(sig1, createTestPattern(10));

    // Hit
    cache.get(sig1);
    cache.get(sig1);

    // Miss
    cache.get(sig2);
    cache.get(sig2);

    auto stats = cache.getStats();
    EXPECT_EQ(stats.hits, 2);
    EXPECT_EQ(stats.misses, 2);
    EXPECT_DOUBLE_EQ(stats.hitRate(), 0.5);
}

TEST(SparsityCacheTest, ResetStatistics) {
    CacheConfig config;
    config.collect_stats = true;
    SparsityCache cache(config);

    PatternSignature sig("pattern", 10, 10);
    cache.put(sig, createTestPattern(10));
    cache.get(sig);

    auto stats1 = cache.getStats();
    EXPECT_GT(stats1.hits, 0);

    cache.resetStats();
    auto stats2 = cache.getStats();
    EXPECT_EQ(stats2.hits, 0);
    EXPECT_EQ(stats2.misses, 0);
}

// ============================================================================
// Signature Builder Tests
// ============================================================================

TEST(SparsityCacheTest, BuildSignatureFromDofProperties) {
    auto sig = buildSignature(1000, 500, 8, "v1");

    EXPECT_TRUE(sig.isValid());
    EXPECT_EQ(sig.n_rows, 1000);
    EXPECT_EQ(sig.n_cols, 1000);
    EXPECT_NE(sig.key.find("1000"), std::string::npos);
}

TEST(SparsityCacheTest, BuildSignatureFromHashes) {
    auto sig = buildSignature(0x12345678, 0x87654321, 100, 200);

    EXPECT_TRUE(sig.isValid());
    EXPECT_EQ(sig.n_rows, 100);
    EXPECT_EQ(sig.n_cols, 200);
}

TEST(SparsityCacheTest, ComputeIndexHash) {
    std::vector<GlobalIndex> indices1 = {1, 2, 3, 4, 5};
    std::vector<GlobalIndex> indices2 = {1, 2, 3, 4, 5};
    std::vector<GlobalIndex> indices3 = {5, 4, 3, 2, 1};

    auto hash1 = computeIndexHash(indices1);
    auto hash2 = computeIndexHash(indices2);
    auto hash3 = computeIndexHash(indices3);

    EXPECT_EQ(hash1, hash2);  // Same content, same hash
    EXPECT_NE(hash1, hash3);  // Different order, different hash
}

// ============================================================================
// Validation Tests
// ============================================================================

TEST(SparsityCacheTest, ValidateAll) {
    SparsityCache cache;

    PatternSignature sig1("pattern_1", 10, 10);
    PatternSignature sig2("pattern_2", 10, 10);
    cache.put(sig1, createTestPattern(10));
    cache.put(sig2, createTestPattern(10));

    EXPECT_TRUE(cache.validateAll());
}

TEST(SparsityCacheTest, GetCachedSignatures) {
    SparsityCache cache;

    PatternSignature sig1("pattern_1", 10, 10);
    PatternSignature sig2("pattern_2", 10, 10);
    cache.put(sig1, createTestPattern(10));
    cache.put(sig2, createTestPattern(10));

    auto signatures = cache.getCachedSignatures();
    EXPECT_EQ(signatures.size(), 2);
}

TEST(SparsityCacheTest, GetEntryInfo) {
    SparsityCache cache;

    PatternSignature sig("pattern", 10, 10);
    cache.put(sig, createTestPattern(10));

    auto info = cache.getEntryInfo(sig);
    ASSERT_TRUE(info.has_value());
    EXPECT_EQ(info->signature.key, "pattern");
    EXPECT_GT(info->memory_bytes, 0);
    EXPECT_EQ(info->access_count, 1);
}

// ============================================================================
// Global Cache Tests
// ============================================================================

TEST(SparsityCacheTest, GlobalCacheAccess) {
    clearGlobalCache();

    PatternSignature sig("global_pattern", 10, 10);
    getGlobalCache().put(sig, createTestPattern(10));

    EXPECT_TRUE(getGlobalCache().contains(sig));

    clearGlobalCache();
    EXPECT_FALSE(getGlobalCache().contains(sig));
}

TEST(SparsityCacheTest, ConfigureGlobalCache) {
    CacheConfig config;
    config.max_entries = 50;
    configureGlobalCache(config);

    EXPECT_EQ(getGlobalCache().getConfig().max_entries, 50);

    // Reset to defaults
    configureGlobalCache(CacheConfig{});
}

// ============================================================================
// Thread Safety Tests
// ============================================================================

TEST(SparsityCacheTest, ConcurrentReads) {
    CacheConfig config;
    config.thread_safe = true;
    SparsityCache cache(config);

    PatternSignature sig("pattern", 10, 10);
    cache.put(sig, createTestPattern(10));

    std::atomic<int> success_count{0};
    std::vector<std::thread> threads;

    for (int i = 0; i < 10; ++i) {
        threads.emplace_back([&]() {
            for (int j = 0; j < 100; ++j) {
                auto pattern = cache.get(sig);
                if (pattern != nullptr) {
                    ++success_count;
                }
            }
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    EXPECT_EQ(success_count, 1000);
}

TEST(SparsityCacheTest, ConcurrentReadWrite) {
    CacheConfig config;
    config.thread_safe = true;
    config.max_entries = 100;
    SparsityCache cache(config);

    std::atomic<int> read_success{0};
    std::atomic<int> write_success{0};
    std::vector<std::thread> threads;

    // Writer threads
    for (int i = 0; i < 5; ++i) {
        threads.emplace_back([&, i]() {
            for (int j = 0; j < 20; ++j) {
                PatternSignature sig("pattern_" + std::to_string(i) + "_" + std::to_string(j), 10, 10);
                if (cache.put(sig, createTestPattern(10))) {
                    ++write_success;
                }
            }
        });
    }

    // Reader threads
    for (int i = 0; i < 5; ++i) {
        threads.emplace_back([&, i]() {
            for (int j = 0; j < 50; ++j) {
                PatternSignature sig("pattern_" + std::to_string(i % 5) + "_" + std::to_string(j % 20), 10, 10);
                auto pattern = cache.get(sig);
                if (pattern != nullptr) {
                    ++read_success;
                }
            }
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    EXPECT_GT(write_success, 0);
    // Read success may vary depending on timing
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST(SparsityCacheTest, NullPatternRejected) {
    SparsityCache cache;

    PatternSignature sig("pattern", 10, 10);
    std::shared_ptr<SparsityPattern> null_pattern = nullptr;

    EXPECT_FALSE(cache.put(sig, null_pattern));
    EXPECT_EQ(cache.size(), 0);
}

TEST(SparsityCacheTest, InvalidSignature) {
    SparsityCache cache;

    PatternSignature invalid_sig;  // Empty key
    EXPECT_FALSE(invalid_sig.isValid());

    cache.put(invalid_sig, createTestPattern(10));
    // Should work but with empty key
}

TEST(SparsityCacheTest, VeryLargePattern) {
    CacheConfig config;
    config.max_memory_bytes = 1000;  // Very small limit
    SparsityCache cache(config);

    PatternSignature sig("large_pattern", 1000, 1000);
    auto pattern = createTestPattern(1000);

    // Pattern is too large to cache
    EXPECT_FALSE(cache.put(sig, pattern));
    EXPECT_FALSE(cache.contains(sig));
}

TEST(SparsityCacheTest, NoAutoEvict) {
    CacheConfig config;
    config.max_entries = 2;
    config.auto_evict = false;
    SparsityCache cache(config);

    cache.put(PatternSignature("p1", 10, 10), createTestPattern(10));
    cache.put(PatternSignature("p2", 10, 10), createTestPattern(10));

    // Third insert should fail (no auto eviction)
    EXPECT_FALSE(cache.put(PatternSignature("p3", 10, 10), createTestPattern(10)));
    EXPECT_EQ(cache.size(), 2);
}

TEST(SparsityCacheTest, RemainingMemory) {
    CacheConfig config;
    config.max_memory_bytes = 100000;
    SparsityCache cache(config);

    EXPECT_EQ(cache.remainingMemory(), 100000);

    cache.put(PatternSignature("p1", 10, 10), createTestPattern(10));
    EXPECT_LT(cache.remainingMemory(), 100000);
}

// ============================================================================
// Move Semantics Tests
// ============================================================================

TEST(SparsityCacheTest, MoveConstruction) {
    SparsityCache cache1;
    PatternSignature sig("pattern", 10, 10);
    cache1.put(sig, createTestPattern(10));

    SparsityCache cache2(std::move(cache1));

    EXPECT_TRUE(cache2.contains(sig));
    EXPECT_EQ(cache1.size(), 0);  // Moved from
}

TEST(SparsityCacheTest, MoveAssignment) {
    SparsityCache cache1;
    SparsityCache cache2;

    PatternSignature sig("pattern", 10, 10);
    cache1.put(sig, createTestPattern(10));

    cache2 = std::move(cache1);

    EXPECT_TRUE(cache2.contains(sig));
    EXPECT_EQ(cache1.size(), 0);  // Moved from
}

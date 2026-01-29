/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include <gtest/gtest.h>

#include "Forms/FormExpr.h"
#include "Forms/JIT/JITCompiler.h"
#include "Tests/Unit/Forms/JITTestHelpers.h"
#include "Tests/Unit/Forms/PerfTestHelpers.h"

#include <algorithm>
#include <cstdint>
#include <iostream>

namespace svmp {
namespace FE {
namespace forms {
namespace test {

TEST(JITCacheRegression, CacheHitAtLeast100xFasterThanCompile)
{
    requireLLVMJITOrSkip();
    if (!perfTestsEnabled()) {
        GTEST_SKIP() << "Set SVMP_FE_RUN_PERF_TESTS=1 to enable";
    }

    const int hit_iters = std::max(1, detail::getenvInt("SVMP_FE_PERF_ITERS_CACHE_HIT", 2000));
    const double min_ratio = detail::getenvDouble("SVMP_FE_PERF_MIN_CACHE_RATIO", 100.0);

    auto options = makeUnitTestJITOptions();
    options.optimization_level = 2;
    options.vectorize = true;
    options.dump_directory = "svmp_fe_jit_dumps_perf_cache_regression";

    auto compiler = jit::JITCompiler::getOrCreate(options);
    ASSERT_NE(compiler, nullptr);

    compiler->resetCacheStats();

    jit::ValidationOptions v;
    v.strictness = jit::Strictness::Strict;

    const auto integrand = FormExpr::parameterRef(1234567);

    const double compile_time = detail::timeSeconds([&]() {
        const auto r = compiler->compileFunctional(integrand, IntegralDomain::Cell, v);
        ASSERT_TRUE(r.ok) << r.message;
        ASSERT_FALSE(r.kernels.empty());
        ASSERT_NE(r.kernels[0].address, 0u);
    });

    const double hit_time_total = detail::timeSeconds([&]() {
        for (int i = 0; i < hit_iters; ++i) {
            const auto r = compiler->compileFunctional(integrand, IntegralDomain::Cell, v);
            ASSERT_TRUE(r.ok) << r.message;
        }
    });

    const double hit_time = hit_time_total / static_cast<double>(hit_iters);
    const double ratio = (hit_time > 0.0) ? (compile_time / hit_time) : 0.0;

    std::cerr << "JITCacheRegression.CacheHitLatency: compile_ms=" << (1e3 * compile_time)
              << " hit_us/call=" << (1e6 * hit_time)
              << " ratio=" << ratio << "x (min=" << min_ratio << "x)"
              << " hit_iters=" << hit_iters << "\n";

    EXPECT_GE(ratio, min_ratio)
        << "Cache hit latency is too slow vs compile. compile=" << compile_time << "s hit=" << hit_time << "s";

    const auto stats = compiler->cacheStats();
    EXPECT_EQ(stats.kernel.misses, 1u);
    EXPECT_EQ(stats.kernel.stores, 1u);
    EXPECT_EQ(stats.kernel.hits, static_cast<std::uint64_t>(hit_iters));
}

TEST(JITCacheRegression, CacheKeyDifferentiatesDistinctFunctionalIntegrands)
{
    requireLLVMJITOrSkip();
    if (!perfTestsEnabled()) {
        GTEST_SKIP() << "Set SVMP_FE_RUN_PERF_TESTS=1 to enable";
    }

    auto options = makeUnitTestJITOptions();
    options.optimization_level = 2;
    options.vectorize = true;
    options.dump_directory = "svmp_fe_jit_dumps_perf_cache_key_regression";

    auto compiler = jit::JITCompiler::getOrCreate(options);
    ASSERT_NE(compiler, nullptr);

    compiler->resetCacheStats();

    jit::ValidationOptions v;
    v.strictness = jit::Strictness::Strict;

    const auto a = FormExpr::parameterRef(1);
    const auto b = FormExpr::parameterRef(2);

    const auto r1 = compiler->compileFunctional(a, IntegralDomain::Cell, v);
    ASSERT_TRUE(r1.ok) << r1.message;
    const auto r2 = compiler->compileFunctional(b, IntegralDomain::Cell, v);
    ASSERT_TRUE(r2.ok) << r2.message;

    const auto stats = compiler->cacheStats();
    EXPECT_EQ(stats.kernel.hits, 0u);
    EXPECT_EQ(stats.kernel.misses, 2u);
    EXPECT_EQ(stats.kernel.stores, 2u);
}

} // namespace test
} // namespace forms
} // namespace FE
} // namespace svmp

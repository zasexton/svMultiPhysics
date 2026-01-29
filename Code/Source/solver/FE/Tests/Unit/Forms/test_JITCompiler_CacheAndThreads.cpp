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

#include <cstddef>
#include <cstdint>
#include <string>
#include <thread>
#include <vector>

namespace svmp {
namespace FE {
namespace forms {
namespace test {

TEST(JITCompilerCache, CacheHitReturnsSameAddress)
{
    requireLLVMJITOrSkip();

    auto options = makeUnitTestJITOptions();
    options.dump_directory = "svmp_fe_jit_dumps_tests_cachehit";
    auto compiler = jit::JITCompiler::getOrCreate(options);
    ASSERT_NE(compiler, nullptr);

    compiler->resetCacheStats();

    jit::ValidationOptions v;
    v.strictness = jit::Strictness::Strict;

    const auto integrand = FormExpr::parameterRef(123);

    const auto r1 = compiler->compileFunctional(integrand, IntegralDomain::Cell, v);
    ASSERT_TRUE(r1.ok) << r1.message;
    ASSERT_FALSE(r1.kernels.empty());
    ASSERT_NE(r1.kernels[0].address, 0u);

    const auto stats1 = compiler->cacheStats();
    EXPECT_EQ(stats1.kernel.misses, 1u);
    EXPECT_EQ(stats1.kernel.stores, 1u);

    const auto r2 = compiler->compileFunctional(integrand, IntegralDomain::Cell, v);
    ASSERT_TRUE(r2.ok) << r2.message;
    ASSERT_FALSE(r2.kernels.empty());

    EXPECT_EQ(r2.kernels[0].address, r1.kernels[0].address);

    const auto stats2 = compiler->cacheStats();
    EXPECT_EQ(stats2.kernel.hits, 1u);
    EXPECT_EQ(stats2.kernel.misses, 1u);
    EXPECT_EQ(stats2.kernel.stores, 1u);
}

TEST(JITCompilerCache, CacheDisabledCompilesNewInstance)
{
    requireLLVMJITOrSkip();

    auto options = makeUnitTestJITOptions();
    options.cache_kernels = false;
    options.dump_directory = "svmp_fe_jit_dumps_tests_cache_disabled";

    auto compiler = jit::JITCompiler::getOrCreate(options);
    ASSERT_NE(compiler, nullptr);

    jit::ValidationOptions v;
    v.strictness = jit::Strictness::Strict;

    const auto integrand = FormExpr::parameterRef(124);

    const auto r1 = compiler->compileFunctional(integrand, IntegralDomain::Cell, v);
    ASSERT_TRUE(r1.ok) << r1.message;
    ASSERT_FALSE(r1.kernels.empty());
    ASSERT_NE(r1.kernels[0].address, 0u);

    const auto r2 = compiler->compileFunctional(integrand, IntegralDomain::Cell, v);
    ASSERT_TRUE(r2.ok) << r2.message;
    ASSERT_FALSE(r2.kernels.empty());
    ASSERT_NE(r2.kernels[0].address, 0u);

    EXPECT_NE(r2.kernels[0].address, r1.kernels[0].address);
}

TEST(JITCompilerThreadSafety, ConcurrentCompileSameFormCompilesOnce)
{
    requireLLVMJITOrSkip();

    auto options = makeUnitTestJITOptions();
    options.dump_directory = "svmp_fe_jit_dumps_tests_concurrent_compile";
    auto compiler = jit::JITCompiler::getOrCreate(options);
    ASSERT_NE(compiler, nullptr);

    compiler->resetCacheStats();

    jit::ValidationOptions v;
    v.strictness = jit::Strictness::Strict;

    const auto integrand = FormExpr::parameterRef(456);

    struct Result {
        bool ok{false};
        std::string message{};
        std::uintptr_t addr{0u};
    };

    constexpr std::size_t n_threads = 8;
    std::vector<Result> results(n_threads);
    std::vector<std::thread> threads;
    threads.reserve(n_threads);

    for (std::size_t i = 0; i < n_threads; ++i) {
        threads.emplace_back([&, i] {
            const auto r = compiler->compileFunctional(integrand, IntegralDomain::Cell, v);
            results[i].ok = r.ok && !r.kernels.empty() && r.kernels[0].address != 0u;
            results[i].message = r.message;
            results[i].addr = (r.ok && !r.kernels.empty()) ? r.kernels[0].address : 0u;
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    for (std::size_t i = 0; i < n_threads; ++i) {
        ASSERT_TRUE(results[i].ok) << results[i].message;
        ASSERT_NE(results[i].addr, 0u);
        EXPECT_EQ(results[i].addr, results[0].addr);
    }

    const auto stats = compiler->cacheStats();
    EXPECT_EQ(stats.kernel.misses, 1u);
    EXPECT_EQ(stats.kernel.stores, 1u);
    EXPECT_EQ(stats.kernel.hits, static_cast<std::uint64_t>(n_threads - 1u));
}

} // namespace test
} // namespace forms
} // namespace FE
} // namespace svmp

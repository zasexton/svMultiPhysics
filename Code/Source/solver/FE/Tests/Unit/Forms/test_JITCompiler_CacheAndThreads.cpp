/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include <gtest/gtest.h>

#include "Forms/FormCompiler.h"
#include "Forms/FormExpr.h"
#include "Forms/FormKernels.h"
#include "Forms/JIT/JITCompiler.h"
#include "Forms/JIT/JITKernelWrapper.h"
#include "Spaces/H1Space.h"
#include "Tests/Unit/Forms/JITTestHelpers.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace svmp {
namespace FE {
namespace forms {
namespace test {

namespace {

[[nodiscard]] std::string hexString(std::uint64_t value)
{
    std::ostringstream os;
    os << std::hex << value;
    return os.str();
}

[[nodiscard]] bool isLowerHexString(std::string_view text)
{
    return !text.empty() &&
           std::all_of(text.begin(), text.end(), [](char c) {
               return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
           });
}

[[nodiscard]] std::filesystem::path makeUniqueTempDir(std::string_view label)
{
    static std::atomic<std::uint64_t> counter{0u};
    const auto stamp = static_cast<std::uint64_t>(
        std::chrono::high_resolution_clock::now().time_since_epoch().count());
    const auto id = counter.fetch_add(1u, std::memory_order_relaxed);
    auto dir = std::filesystem::temp_directory_path() /
               ("svmp_fe_" + std::string(label) + "_" + std::to_string(stamp) + "_" + std::to_string(id));
    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
    std::filesystem::create_directories(dir, ec);
    return dir;
}

[[nodiscard]] std::vector<std::filesystem::path> objectCacheFiles(const std::filesystem::path& dir)
{
    std::vector<std::filesystem::path> files;
    std::error_code ec;
    if (!std::filesystem::exists(dir, ec) || ec) {
        return files;
    }
    for (std::filesystem::recursive_directory_iterator it(dir, ec), end; !ec && it != end; it.increment(ec)) {
        if (it->is_regular_file(ec) && it->path().extension() == ".objcache") {
            files.push_back(it->path());
        }
    }
    std::sort(files.begin(), files.end());
    return files;
}

[[nodiscard]] std::filesystem::path objectCacheFileForSymbol(const std::filesystem::path& dir,
                                                             std::string_view symbol)
{
    const auto expected = std::string(symbol) + ".objcache";
    for (const auto& path : objectCacheFiles(dir)) {
        if (path.filename().string() == expected) {
            return path;
        }
    }
    return {};
}

class ScopedTempDir {
public:
    explicit ScopedTempDir(std::string_view label)
        : path_(makeUniqueTempDir(label))
    {
    }

    ~ScopedTempDir()
    {
        std::error_code ec;
        std::filesystem::remove_all(path_, ec);
    }

    [[nodiscard]] const std::filesystem::path& path() const noexcept { return path_; }

private:
    std::filesystem::path path_;
};

} // namespace

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

TEST(JITCompilerCache, RegistryDistinguishesSIMDBatchAndFastMathMode)
{
    requireLLVMJITOrSkip();

    auto base = makeUnitTestJITOptions();
    base.dump_directory = "svmp_fe_jit_dumps_tests_registry_key_base";
    base.simd_batch = false;
    base.fast_math_mode = JITFastMathMode::Strict;

    auto simd = base;
    simd.simd_batch = true;

    auto relaxed = base;
    relaxed.fast_math_mode = JITFastMathMode::Relaxed;

    const auto compiler_base = jit::JITCompiler::getOrCreate(base);
    const auto compiler_simd = jit::JITCompiler::getOrCreate(simd);
    const auto compiler_relaxed = jit::JITCompiler::getOrCreate(relaxed);

    ASSERT_NE(compiler_base, nullptr);
    ASSERT_NE(compiler_simd, nullptr);
    ASSERT_NE(compiler_relaxed, nullptr);
    EXPECT_NE(compiler_base.get(), compiler_simd.get());
    EXPECT_NE(compiler_base.get(), compiler_relaxed.get());
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

TEST(JITCompilerCache, StableSymbolMatchesCacheKeyAndOptionsAffectKey)
{
    requireLLVMJITOrSkip();

    jit::ValidationOptions v;
    v.strictness = jit::Strictness::Strict;

    auto options_o0 = makeUnitTestJITOptions();
    options_o0.optimization_level = 0;
    options_o0.dump_directory = "svmp_fe_jit_dumps_tests_key_o0";
    auto compiler_o0 = jit::JITCompiler::getOrCreate(options_o0);
    ASSERT_NE(compiler_o0, nullptr);

    const auto integrand = FormExpr::parameterRef(777);
    const auto r_o0 = compiler_o0->compileFunctional(integrand, IntegralDomain::Cell, v);
    ASSERT_TRUE(r_o0.ok) << r_o0.message;
    ASSERT_EQ(r_o0.kernels.size(), 1u);

    const auto& k_o0 = r_o0.kernels.front();
    ASSERT_NE(k_o0.cache_key, 0u);
    constexpr std::string_view prefix = "svmp_fe_jit_kernel_";
    ASSERT_EQ(k_o0.symbol.rfind(prefix, 0u), 0u);
    const auto suffix = std::string_view(k_o0.symbol).substr(prefix.size());
    EXPECT_TRUE(isLowerHexString(suffix));
    EXPECT_EQ(k_o0.symbol, std::string(prefix) + hexString(k_o0.cache_key));

    auto options_o2 = makeUnitTestJITOptions();
    options_o2.optimization_level = 2;
    options_o2.dump_directory = "svmp_fe_jit_dumps_tests_key_o2";
    auto compiler_o2 = jit::JITCompiler::getOrCreate(options_o2);
    ASSERT_NE(compiler_o2, nullptr);

    const auto r_o2 = compiler_o2->compileFunctional(integrand, IntegralDomain::Cell, v);
    ASSERT_TRUE(r_o2.ok) << r_o2.message;
    ASSERT_EQ(r_o2.kernels.size(), 1u);

    EXPECT_NE(r_o2.kernels.front().cache_key, k_o0.cache_key);
    EXPECT_NE(r_o2.kernels.front().symbol, k_o0.symbol);
}

TEST(JITCompilerCache, ObjectCacheLoadsFromDiskAcrossCompilerInstances)
{
    requireLLVMJITOrSkip();

    ScopedTempDir cache_dir("jit_objcache_disk");

    auto options_first = makeUnitTestJITOptions();
    options_first.cache_directory = cache_dir.path().string();
    options_first.dump_directory = (cache_dir.path() / "dump_first").string();
    auto compiler_first = jit::JITCompiler::getOrCreate(options_first);
    ASSERT_NE(compiler_first, nullptr);
    compiler_first->resetCacheStats();

    jit::ValidationOptions v;
    v.strictness = jit::Strictness::Strict;

    const auto integrand = FormExpr::parameterRef(778);
    const auto first = compiler_first->compileFunctional(integrand, IntegralDomain::Cell, v);
    ASSERT_TRUE(first.ok) << first.message;
    ASSERT_EQ(first.kernels.size(), 1u);
    ASSERT_NE(first.kernels.front().address, 0u);

    const auto first_stats = compiler_first->cacheStats();
    EXPECT_GE(first_stats.object.notify_compiled, 1u);
    EXPECT_GT(first_stats.object.bytes_written, 0u);
    ASSERT_FALSE(objectCacheFiles(cache_dir.path()).empty());

    auto options_second = options_first;
    options_second.dump_directory = (cache_dir.path() / "dump_second").string();
    auto compiler_second = jit::JITCompiler::getOrCreate(options_second);
    ASSERT_NE(compiler_second, nullptr);
    compiler_second->resetCacheStats();

    const auto second = compiler_second->compileFunctional(integrand, IntegralDomain::Cell, v);
    ASSERT_TRUE(second.ok) << second.message;
    ASSERT_EQ(second.kernels.size(), 1u);
    ASSERT_NE(second.kernels.front().address, 0u);

    EXPECT_EQ(second.kernels.front().cache_key, first.kernels.front().cache_key);
    EXPECT_EQ(second.kernels.front().symbol, first.kernels.front().symbol);

    const auto second_stats = compiler_second->cacheStats();
    EXPECT_GE(second_stats.object.disk_hits, 1u);
    EXPECT_GT(second_stats.object.bytes_read, 0u);
}

TEST(JITCompilerCache, ColocatedObjectCacheLoadsFromDiskAcrossCompilerInstances)
{
    requireLLVMJITOrSkip();

    ScopedTempDir cache_dir("jit_objcache_colocated_disk");

    spaces::H1Space space(ElementType::Tetra4, 1);
    const auto u = FormExpr::trialFunction(space, "u");
    const auto v = FormExpr::testFunction(space, "v");

    FormCompiler form_compiler;
    auto mass_ir = form_compiler.compileBilinear((u * v).dx());
    auto stiffness_ir = form_compiler.compileBilinear(inner(grad(u), grad(v)).dx());

    std::vector<jit::JITCompiler::ColocatedKernelSpec> specs;
    specs.push_back(jit::JITCompiler::ColocatedKernelSpec{
        .ir = &mass_ir,
        .domain = IntegralDomain::Cell,
    });
    specs.push_back(jit::JITCompiler::ColocatedKernelSpec{
        .ir = &stiffness_ir,
        .domain = IntegralDomain::Cell,
    });

    auto options_first = makeUnitTestJITOptions();
    options_first.cache_directory = cache_dir.path().string();
    options_first.dump_directory = (cache_dir.path() / "dump_first").string();
    auto compiler_first = jit::JITCompiler::getOrCreate(options_first);
    ASSERT_NE(compiler_first, nullptr);
    compiler_first->resetCacheStats();

    std::vector<jit::JITCompiler::ColocatedKernelResult> first_results;
    const auto first = compiler_first->compileColocated(specs, first_results);
    ASSERT_TRUE(first.ok) << first.message;
    ASSERT_EQ(first_results.size(), specs.size());
    ASSERT_NE(first_results[0].address, 0u);
    ASSERT_NE(first_results[1].address, 0u);
    ASSERT_NE(first_results[0].symbol, first_results[1].symbol);
    ASSERT_FALSE(objectCacheFiles(cache_dir.path()).empty());

    const auto first_stats = compiler_first->cacheStats();
    EXPECT_GE(first_stats.object.notify_compiled, 1u);
    EXPECT_GT(first_stats.object.bytes_written, 0u);

    auto options_second = options_first;
    options_second.dump_directory = (cache_dir.path() / "dump_second").string();
    auto compiler_second = jit::JITCompiler::getOrCreate(options_second);
    ASSERT_NE(compiler_second, nullptr);
    compiler_second->resetCacheStats();

    std::vector<jit::JITCompiler::ColocatedKernelResult> second_results;
    const auto second = compiler_second->compileColocated(specs, second_results);
    ASSERT_TRUE(second.ok) << second.message;
    ASSERT_EQ(second_results.size(), specs.size());
    ASSERT_NE(second_results[0].address, 0u);
    ASSERT_NE(second_results[1].address, 0u);
    EXPECT_EQ(second_results[0].symbol, first_results[0].symbol);
    EXPECT_EQ(second_results[1].symbol, first_results[1].symbol);

    const auto second_stats = compiler_second->cacheStats();
    EXPECT_GE(second_stats.object.disk_hits, 1u);
    EXPECT_GT(second_stats.object.bytes_read, 0u);
}

TEST(JITCompilerCache, CorruptDiskObjectFallsBackToRecompile)
{
    requireLLVMJITOrSkip();

    ScopedTempDir cache_dir("jit_objcache_corrupt");

    auto options_first = makeUnitTestJITOptions();
    options_first.cache_directory = cache_dir.path().string();
    options_first.dump_directory = (cache_dir.path() / "dump_first").string();
    auto compiler_first = jit::JITCompiler::getOrCreate(options_first);
    ASSERT_NE(compiler_first, nullptr);

    jit::ValidationOptions v;
    v.strictness = jit::Strictness::Strict;

    const auto integrand = FormExpr::parameterRef(779);
    const auto first = compiler_first->compileFunctional(integrand, IntegralDomain::Cell, v);
    ASSERT_TRUE(first.ok) << first.message;
    ASSERT_EQ(first.kernels.size(), 1u);

    const auto files = objectCacheFiles(cache_dir.path());
    ASSERT_FALSE(files.empty());
    {
        std::ofstream out(files.front(), std::ios::binary | std::ios::trunc);
        ASSERT_TRUE(out.good());
        out << "not an object file";
    }

    auto options_second = options_first;
    options_second.dump_directory = (cache_dir.path() / "dump_second").string();
    auto compiler_second = jit::JITCompiler::getOrCreate(options_second);
    ASSERT_NE(compiler_second, nullptr);
    compiler_second->resetCacheStats();

    const auto second = compiler_second->compileFunctional(integrand, IntegralDomain::Cell, v);
    ASSERT_TRUE(second.ok) << second.message;
    ASSERT_EQ(second.kernels.size(), 1u);
    EXPECT_NE(second.kernels.front().address, 0u);
    EXPECT_EQ(second.kernels.front().cache_key, first.kernels.front().cache_key);

    const auto stats = compiler_second->cacheStats();
    EXPECT_GE(stats.object.misses, 1u);
    EXPECT_GE(stats.object.notify_compiled, 1u);
}

TEST(JITCompilerCache, CacheDirectoriesAreIsolated)
{
    requireLLVMJITOrSkip();

    ScopedTempDir cache_a("jit_objcache_isolated_a");
    ScopedTempDir cache_b("jit_objcache_isolated_b");

    jit::ValidationOptions v;
    v.strictness = jit::Strictness::Strict;
    const auto integrand = FormExpr::parameterRef(780);

    auto options_a = makeUnitTestJITOptions();
    options_a.cache_directory = cache_a.path().string();
    options_a.dump_directory = (cache_a.path() / "dump").string();
    auto compiler_a = jit::JITCompiler::getOrCreate(options_a);
    ASSERT_NE(compiler_a, nullptr);
    compiler_a->resetCacheStats();

    const auto first = compiler_a->compileFunctional(integrand, IntegralDomain::Cell, v);
    ASSERT_TRUE(first.ok) << first.message;
    ASSERT_EQ(first.kernels.size(), 1u);
    ASSERT_FALSE(objectCacheFiles(cache_a.path()).empty());

    const auto first_stats = compiler_a->cacheStats();
    EXPECT_GE(first_stats.object.notify_compiled, 1u);
    EXPECT_GT(first_stats.object.bytes_written, 0u);

    auto options_b = makeUnitTestJITOptions();
    options_b.cache_directory = cache_b.path().string();
    options_b.dump_directory = (cache_b.path() / "dump").string();
    auto compiler_b = jit::JITCompiler::getOrCreate(options_b);
    ASSERT_NE(compiler_b, nullptr);
    compiler_b->resetCacheStats();

    const auto second = compiler_b->compileFunctional(integrand, IntegralDomain::Cell, v);
    ASSERT_TRUE(second.ok) << second.message;
    ASSERT_EQ(second.kernels.size(), 1u);
    ASSERT_FALSE(objectCacheFiles(cache_b.path()).empty());

    EXPECT_EQ(second.kernels.front().cache_key, first.kernels.front().cache_key);
    EXPECT_EQ(second.kernels.front().symbol, first.kernels.front().symbol);

    const auto second_stats = compiler_b->cacheStats();
    EXPECT_EQ(second_stats.object.disk_hits, 0u);
    EXPECT_GE(second_stats.object.misses, 1u);
    EXPECT_GE(second_stats.object.notify_compiled, 1u);
    EXPECT_GT(second_stats.object.bytes_written, 0u);
}

TEST(JITCompilerCache, WrongSymbolDiskObjectIsRejectedAndRecompiled)
{
    requireLLVMJITOrSkip();

    ScopedTempDir cache_dir("jit_objcache_wrong_symbol");

    jit::ValidationOptions v;
    v.strictness = jit::Strictness::Strict;
    const auto integrand_a = FormExpr::parameterRef(781);
    const auto integrand_b = FormExpr::parameterRef(782);

    auto options_a = makeUnitTestJITOptions();
    options_a.cache_directory = cache_dir.path().string();
    options_a.dump_directory = (cache_dir.path() / "dump_a").string();
    auto compiler_a = jit::JITCompiler::getOrCreate(options_a);
    ASSERT_NE(compiler_a, nullptr);
    const auto first_a = compiler_a->compileFunctional(integrand_a, IntegralDomain::Cell, v);
    ASSERT_TRUE(first_a.ok) << first_a.message;
    ASSERT_EQ(first_a.kernels.size(), 1u);

    auto options_b_seed = makeUnitTestJITOptions();
    options_b_seed.cache_directory = cache_dir.path().string();
    options_b_seed.dump_directory = (cache_dir.path() / "dump_b_seed").string();
    auto compiler_b_seed = jit::JITCompiler::getOrCreate(options_b_seed);
    ASSERT_NE(compiler_b_seed, nullptr);
    const auto first_b = compiler_b_seed->compileFunctional(integrand_b, IntegralDomain::Cell, v);
    ASSERT_TRUE(first_b.ok) << first_b.message;
    ASSERT_EQ(first_b.kernels.size(), 1u);
    ASSERT_NE(first_a.kernels.front().symbol, first_b.kernels.front().symbol);

    const auto a_path = objectCacheFileForSymbol(cache_dir.path(), first_a.kernels.front().symbol);
    const auto b_path = objectCacheFileForSymbol(cache_dir.path(), first_b.kernels.front().symbol);
    ASSERT_FALSE(a_path.empty());
    ASSERT_FALSE(b_path.empty());
    std::filesystem::copy_file(a_path, b_path, std::filesystem::copy_options::overwrite_existing);

    auto options_probe = makeUnitTestJITOptions();
    options_probe.cache_directory = cache_dir.path().string();
    options_probe.dump_directory = (cache_dir.path() / "dump_b_probe").string();
    auto compiler_probe = jit::JITCompiler::getOrCreate(options_probe);
    ASSERT_NE(compiler_probe, nullptr);
    compiler_probe->resetCacheStats();

    const auto repaired_b = compiler_probe->compileFunctional(integrand_b, IntegralDomain::Cell, v);
    ASSERT_TRUE(repaired_b.ok) << repaired_b.message;
    ASSERT_EQ(repaired_b.kernels.size(), 1u);
    EXPECT_EQ(repaired_b.kernels.front().cache_key, first_b.kernels.front().cache_key);
    EXPECT_EQ(repaired_b.kernels.front().symbol, first_b.kernels.front().symbol);
    EXPECT_NE(repaired_b.kernels.front().address, 0u);

    const auto stats = compiler_probe->cacheStats();
    EXPECT_GE(stats.object.misses, 1u);
    EXPECT_GE(stats.object.notify_compiled, 1u);
}

TEST(JITKernelWrapper, EnsureCompiledIsExplicitAndIdempotent)
{
    requireLLVMJITOrSkip();

    spaces::H1Space space(ElementType::Tetra4, 1);
    const auto u = FormExpr::trialFunction(space, "u");
    const auto v = FormExpr::testFunction(space, "v");

    FormCompiler form_compiler;
    auto kernel = std::make_shared<FormKernel>(form_compiler.compileBilinear((u * v).dx()));

    auto options = makeUnitTestJITOptions();
    options.dump_directory = "svmp_fe_jit_dumps_tests_wrapper_ensure";
    jit::JITKernelWrapper wrapper(kernel, options);

    EXPECT_FALSE(wrapper.isJITReady());
    wrapper.ensureCompiled();
    EXPECT_TRUE(wrapper.isJITReady());
    wrapper.ensureCompiled();
    EXPECT_TRUE(wrapper.isJITReady());
}

} // namespace test
} // namespace forms
} // namespace FE
} // namespace svmp

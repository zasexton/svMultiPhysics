/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include <gtest/gtest.h>

#include "Forms/FormCompiler.h"
#include "Forms/Index.h"
#include "Forms/JIT/JITCompiler.h"
#include "Spaces/H1Space.h"
#include "Spaces/ProductSpace.h"
#include "Tests/Unit/Forms/JITTestHelpers.h"
#include "Tests/Unit/Forms/PerfTestHelpers.h"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace svmp {
namespace FE {
namespace forms {
namespace test {

namespace {

[[nodiscard]] std::filesystem::path makeUniqueCacheDir(std::string_view tag)
{
    const auto now = std::chrono::steady_clock::now().time_since_epoch().count();
    const std::string name = "svmp_fe_jit_objcache_perf_" + std::string(tag) + "_" + std::to_string(now);
    return std::filesystem::temp_directory_path() / name;
}

[[nodiscard]] std::size_t countRegularFilesRecursive(const std::filesystem::path& dir)
{
    std::error_code ec;
    if (!std::filesystem::exists(dir, ec)) return 0u;
    std::size_t count = 0u;
    for (auto it = std::filesystem::recursive_directory_iterator(dir, ec);
         it != std::filesystem::recursive_directory_iterator(); it.increment(ec)) {
        if (ec) break;
        if (it->is_regular_file(ec)) ++count;
    }
    return count;
}

} // namespace

TEST(JITDiskCache, ObjectCacheSecondCompilerFasterThanColdCompile)
{
    requireLLVMJITOrSkip();
    if (!perfTestsEnabled()) {
        GTEST_SKIP() << "Set SVMP_FE_RUN_PERF_TESTS=1 to enable";
    }

    const double min_ratio = detail::getenvDouble("SVMP_FE_PERF_MIN_DISK_CACHE_RATIO", 1.1);

    auto base = std::make_shared<spaces::H1Space>(ElementType::Tetra4, 3);
    spaces::ProductSpace space(base, 3);

    SymbolicOptions sym_opts;
    sym_opts.jit.enable = true;
    FormCompiler form_compiler(sym_opts);

    const auto u = FormExpr::trialFunction(space, "u");
    const auto v = FormExpr::testFunction(space, "v");
    const Index i("i");
    const Index j("j");
    const auto form = (grad(u)(i, j) * grad(v)(i, j)).dx();
    const auto ir = form_compiler.compileBilinear(form);

    const auto cache_dir = makeUniqueCacheDir("forms");
    std::filesystem::create_directories(cache_dir);

    jit::ValidationOptions vopt;
    vopt.strictness = jit::Strictness::Strict;

    auto opt_a = makeUnitTestJITOptions();
    opt_a.optimization_level = 2;
    opt_a.vectorize = true;
    opt_a.cache_kernels = true;
    opt_a.cache_directory = cache_dir.string();
    opt_a.dump_directory = "svmp_fe_jit_dumps_perf_objcache_a";

    auto compiler_a = jit::JITCompiler::getOrCreate(opt_a);
    ASSERT_NE(compiler_a, nullptr);

    // Warm up engine init on this compiler instance (different kernel).
    {
        const auto warm = compiler_a->compileFunctional(FormExpr::parameterRef(1), IntegralDomain::Cell, vopt);
        ASSERT_TRUE(warm.ok) << warm.message;
    }

    const double sec_cold = detail::timeSeconds([&]() {
        const auto r = compiler_a->compile(ir, vopt);
        ASSERT_TRUE(r.ok) << r.message;
        ASSERT_FALSE(r.kernels.empty());
    });

    auto opt_b = opt_a;
    opt_b.dump_directory = "svmp_fe_jit_dumps_perf_objcache_b"; // force a new compiler+engine instance

    auto compiler_b = jit::JITCompiler::getOrCreate(opt_b);
    ASSERT_NE(compiler_b, nullptr);

    // Warm up engine init on this compiler instance (different kernel).
    {
        const auto warm = compiler_b->compileFunctional(FormExpr::parameterRef(2), IntegralDomain::Cell, vopt);
        ASSERT_TRUE(warm.ok) << warm.message;
    }

    const double sec_disk = detail::timeSeconds([&]() {
        const auto r = compiler_b->compile(ir, vopt);
        ASSERT_TRUE(r.ok) << r.message;
        ASSERT_FALSE(r.kernels.empty());
    });

    const double ratio = (sec_disk > 0.0) ? (sec_cold / sec_disk) : 0.0;
    const std::size_t files = countRegularFilesRecursive(cache_dir);

    std::cerr << "JITDiskCache.ObjectCache: cache_dir=" << cache_dir.string()
              << " files=" << files
              << " cold_ms=" << (1e3 * sec_cold)
              << " disk_ms=" << (1e3 * sec_disk)
              << " ratio=" << ratio << "x (min=" << min_ratio << "x)\n";

    EXPECT_GE(files, 1u) << "Object cache directory appears empty: " << cache_dir.string();
    EXPECT_GE(ratio, min_ratio) << "Disk cache did not speed up compile enough. cold=" << sec_cold << "s disk="
                                << sec_disk << "s";
}

} // namespace test
} // namespace forms
} // namespace FE
} // namespace svmp

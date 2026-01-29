/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include <gtest/gtest.h>

#include "Assembly/StandardAssembler.h"
#include "Forms/FormCompiler.h"
#include "Forms/FormKernels.h"
#include "Forms/Index.h"
#include "Forms/JIT/JITKernelWrapper.h"
#include "Spaces/H1Space.h"
#include "Spaces/ProductSpace.h"
#include "Tests/Unit/Forms/FormsTestHelpers.h"
#include "Tests/Unit/Forms/JITTestHelpers.h"
#include "Tests/Unit/Forms/PerfTestHelpers.h"

#include <algorithm>
#include <iostream>
#include <memory>
#include <vector>

#if defined(__unix__) || defined(__APPLE__)
#include <sys/resource.h>
#endif

namespace svmp {
namespace FE {
namespace forms {
namespace test {

namespace {

#if defined(__unix__) || defined(__APPLE__)
[[nodiscard]] std::size_t peakRssKilobytes()
{
    struct rusage usage {};
    if (getrusage(RUSAGE_SELF, &usage) != 0) {
        return 0u;
    }
    // On Linux, ru_maxrss is kilobytes. On macOS it is bytes (we still report the raw value).
    return static_cast<std::size_t>(usage.ru_maxrss);
}
#endif

dofs::DofMap makeContiguousSingleCellDofMap(GlobalIndex num_dofs)
{
    dofs::DofMap dof_map(1, num_dofs, static_cast<LocalIndex>(num_dofs));
    std::vector<GlobalIndex> cell_dofs(static_cast<std::size_t>(num_dofs));
    for (GlobalIndex i = 0; i < num_dofs; ++i) cell_dofs[static_cast<std::size_t>(i)] = i;
    dof_map.setCellDofs(0, cell_dofs);
    dof_map.setNumDofs(num_dofs);
    dof_map.setNumLocalDofs(num_dofs);
    dof_map.finalize();
    return dof_map;
}

} // namespace

TEST(JITMemoryOverhead, PeakRssIncreaseUnderConfiguredBound)
{
    requireLLVMJITOrSkip();
    if (!perfTestsEnabled()) {
        GTEST_SKIP() << "Set SVMP_FE_RUN_PERF_TESTS=1 to enable";
    }

#if !defined(__unix__) && !defined(__APPLE__)
    GTEST_SKIP() << "Peak RSS measurement not supported on this platform";
#else
    const std::size_t max_delta_kb = static_cast<std::size_t>(
        std::max(0, detail::getenvInt("SVMP_FE_PERF_MAX_RSS_DELTA_KB", 1024 * 1024)));

    const std::size_t rss0 = peakRssKilobytes();

    SingleTetraMeshAccess mesh;
    auto base = std::make_shared<spaces::H1Space>(ElementType::Tetra4, 3);
    spaces::ProductSpace space(base, 3);
    auto dof_map = makeContiguousSingleCellDofMap(static_cast<GlobalIndex>(space.dofs_per_element()));

    SymbolicOptions sym_opts;
    sym_opts.jit.enable = true;
    FormCompiler compiler(sym_opts);

    const auto u = FormExpr::trialFunction(space, "u");
    const auto v = FormExpr::testFunction(space, "v");
    const Index i("i");
    const Index j("j");
    const auto form = (grad(u)(i, j) * grad(v)(i, j)).dx();

    auto ir = compiler.compileBilinear(form);
    auto fallback = std::make_shared<FormKernel>(std::move(ir));

    forms::JITOptions jit_opts;
    jit_opts.enable = true;
    jit_opts.optimization_level = 2;
    jit_opts.vectorize = true;
    jit_opts.cache_kernels = true;
    jit_opts.specialization.enable = false;

    forms::jit::JITKernelWrapper jit_kernel(fallback, jit_opts);

    assembly::StandardAssembler assembler;
    assembler.setDofMap(dof_map);

    assembly::DenseMatrixView mat(static_cast<GlobalIndex>(dof_map.getNumDofs()));
    mat.zero();

    // Trigger LLVM init + compilation.
    (void)assembler.assembleMatrix(mesh, space, space, jit_kernel, mat);

    const std::size_t rss1 = peakRssKilobytes();
    const std::size_t delta = (rss1 > rss0) ? (rss1 - rss0) : 0u;

    std::cerr << "JITMemoryOverhead.PeakRSS: rss0=" << rss0 << " rss1=" << rss1 << " delta=" << delta
              << " (max_delta_kb=" << max_delta_kb << ")\n";

    EXPECT_LE(delta, max_delta_kb)
        << "Peak RSS increase exceeded configured bound. rss0=" << rss0 << " rss1=" << rss1 << " delta=" << delta;
#endif
}

} // namespace test
} // namespace forms
} // namespace FE
} // namespace svmp


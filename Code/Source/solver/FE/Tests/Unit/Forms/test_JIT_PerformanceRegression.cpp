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

#include <iostream>
#include <memory>
#include <vector>

namespace svmp {
namespace FE {
namespace forms {
namespace test {

namespace {

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

TEST(JITPerformanceRegression, CellAssembly_JITAtLeast2xFasterThanInterpreter)
{
    requireLLVMJITOrSkip();
    if (!perfTestsEnabled()) {
        GTEST_SKIP() << "Set SVMP_FE_RUN_PERF_TESTS=1 to enable";
    }

    const int iters = detail::getenvInt("SVMP_FE_PERF_ITERS_CELL", 200);
    const int repeats = std::max(1, detail::getenvInt("SVMP_FE_PERF_REPEATS", 1));
    const double min_speedup = detail::getenvDouble("SVMP_FE_PERF_MIN_SPEEDUP_CELL", 2.0);

    SingleTetraMeshAccess mesh;
    auto base = std::make_shared<spaces::H1Space>(ElementType::Tetra4, 1);
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

    auto ir_interp = compiler.compileBilinear(form);
    auto ir_jit = compiler.compileBilinear(form);

    auto interp_kernel = std::make_shared<FormKernel>(std::move(ir_interp));
    auto jit_fallback = std::make_shared<FormKernel>(std::move(ir_jit));

    forms::JITOptions jit_opts;
    jit_opts.enable = true;
    jit_opts.optimization_level = 2;
    jit_opts.vectorize = true;
    jit_opts.cache_kernels = true;
    jit_opts.specialization.enable = false;

    forms::jit::JITKernelWrapper jit_kernel(jit_fallback, jit_opts);

    assembly::StandardAssembler assembler;
    assembler.setDofMap(dof_map);

    assembly::DenseMatrixView mat(static_cast<GlobalIndex>(dof_map.getNumDofs()));
    mat.zero();

    // Warmup + JIT compile.
    (void)assembler.assembleMatrix(mesh, space, space, *interp_kernel, mat);
    (void)assembler.assembleMatrix(mesh, space, space, jit_kernel, mat);

    const auto run_interp = [&]() {
        for (int k = 0; k < iters; ++k) {
            mat.zero();
            (void)assembler.assembleMatrix(mesh, space, space, *interp_kernel, mat);
        }
    };
    const auto run_jit = [&]() {
        for (int k = 0; k < iters; ++k) {
            mat.zero();
            (void)assembler.assembleMatrix(mesh, space, space, jit_kernel, mat);
        }
    };

    const double sec_interp = detail::bestOfSeconds(repeats, run_interp);
    const double sec_jit = detail::bestOfSeconds(repeats, run_jit);

    const double speedup = (sec_jit > 0.0) ? (sec_interp / sec_jit) : 0.0;
    std::cerr << "JITPerformanceRegression.CellAssembly: iters=" << iters
              << " repeats=" << repeats
              << " interp_ms/call=" << (1e3 * sec_interp / static_cast<double>(iters))
              << " jit_ms/call=" << (1e3 * sec_jit / static_cast<double>(iters))
              << " speedup=" << speedup << "x (min=" << min_speedup << "x)\n";

    EXPECT_GE(speedup, min_speedup)
        << "JIT speedup fell below threshold. interp=" << sec_interp << "s jit=" << sec_jit << "s";
}

TEST(JITPerformanceRegression, HighOrderP3_JITAtLeast4xFasterThanInterpreter)
{
    requireLLVMJITOrSkip();
    if (!perfTestsEnabled()) {
        GTEST_SKIP() << "Set SVMP_FE_RUN_PERF_TESTS=1 to enable";
    }

    const int iters = detail::getenvInt("SVMP_FE_PERF_ITERS_P3", 60);
    const int repeats = std::max(1, detail::getenvInt("SVMP_FE_PERF_REPEATS", 1));
    const double min_speedup = detail::getenvDouble("SVMP_FE_PERF_MIN_SPEEDUP_P3", 4.0);

    SingleTetraMeshAccess mesh;
    spaces::H1Space space(ElementType::Tetra4, 3);
    auto dof_map = makeContiguousSingleCellDofMap(static_cast<GlobalIndex>(space.dofs_per_element()));

    FormCompiler compiler;

    const auto u = FormExpr::trialFunction(space, "u");
    const auto v = FormExpr::testFunction(space, "v");
    const auto form = inner(grad(u), grad(v)).dx();

    auto ir_interp = compiler.compileBilinear(form);
    auto ir_jit = compiler.compileBilinear(form);

    auto interp_kernel = std::make_shared<FormKernel>(std::move(ir_interp));
    auto jit_fallback = std::make_shared<FormKernel>(std::move(ir_jit));

    forms::JITOptions jit_opts;
    jit_opts.enable = true;
    jit_opts.optimization_level = 2;
    jit_opts.vectorize = true;
    jit_opts.cache_kernels = true;
    jit_opts.specialization.enable = false;

    forms::jit::JITKernelWrapper jit_kernel(jit_fallback, jit_opts);

    assembly::StandardAssembler assembler;
    assembler.setDofMap(dof_map);

    assembly::DenseMatrixView mat(static_cast<GlobalIndex>(dof_map.getNumDofs()));
    mat.zero();

    // Warmup + JIT compile.
    (void)assembler.assembleMatrix(mesh, space, space, *interp_kernel, mat);
    (void)assembler.assembleMatrix(mesh, space, space, jit_kernel, mat);

    const auto run_interp = [&]() {
        for (int k = 0; k < iters; ++k) {
            mat.zero();
            (void)assembler.assembleMatrix(mesh, space, space, *interp_kernel, mat);
        }
    };
    const auto run_jit = [&]() {
        for (int k = 0; k < iters; ++k) {
            mat.zero();
            (void)assembler.assembleMatrix(mesh, space, space, jit_kernel, mat);
        }
    };

    const double sec_interp = detail::bestOfSeconds(repeats, run_interp);
    const double sec_jit = detail::bestOfSeconds(repeats, run_jit);

    const double speedup = (sec_jit > 0.0) ? (sec_interp / sec_jit) : 0.0;
    std::cerr << "JITPerformanceRegression.HighOrderP3: iters=" << iters
              << " repeats=" << repeats
              << " interp_ms/call=" << (1e3 * sec_interp / static_cast<double>(iters))
              << " jit_ms/call=" << (1e3 * sec_jit / static_cast<double>(iters))
              << " speedup=" << speedup << "x (min=" << min_speedup << "x)\n";

    EXPECT_GE(speedup, min_speedup)
        << "JIT speedup fell below threshold. interp=" << sec_interp << "s jit=" << sec_jit << "s";
}

} // namespace test
} // namespace forms
} // namespace FE
} // namespace svmp

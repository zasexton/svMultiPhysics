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
#include "Forms/JIT/JITCompiler.h"
#include "Forms/JIT/JITKernelWrapper.h"
#include "Spaces/H1Space.h"
#include "Tests/Unit/Forms/FormsTestHelpers.h"
#include "Tests/Unit/Forms/JITTestHelpers.h"
#include "Tests/Unit/Forms/PerfTestHelpers.h"

#include <algorithm>
#include <iostream>
#include <memory>
#include <vector>

namespace svmp {
namespace FE {
namespace forms {
namespace test {

namespace {

[[nodiscard]] std::vector<Real> defaultSolution(LocalIndex n)
{
    std::vector<Real> u(static_cast<std::size_t>(n), 0.0);
    for (LocalIndex i = 0; i < n; ++i) {
        u[static_cast<std::size_t>(i)] = 0.1 + 0.01 * static_cast<Real>(i);
    }
    return u;
}

} // namespace

TEST(JITPerformanceRegression, NonlinearAssembly_JITAtLeast1p5xFasterThanAD)
{
    requireLLVMJITOrSkip();
    if (!perfTestsEnabled()) {
        GTEST_SKIP() << "Set SVMP_FE_RUN_PERF_TESTS=1 to enable";
    }

    const int iters = std::max(1, detail::getenvInt("SVMP_FE_PERF_ITERS_NONLINEAR", 250));
    const int repeats = std::max(1, detail::getenvInt("SVMP_FE_PERF_REPEATS", 1));
    const double min_speedup = detail::getenvDouble("SVMP_FE_PERF_MIN_SPEEDUP_NONLINEAR", 1.5);

    SingleTetraMeshAccess mesh;
    spaces::H1Space space(ElementType::Tetra4, 1);
    auto dof_map = createSingleTetraDofMap();

    FormCompiler compiler;
    const auto u = FormExpr::trialFunction(space, "u");
    const auto v = FormExpr::testFunction(space, "v");
    const auto residual = (inner(grad(u), grad(v)) + pow(u, FormExpr::constant(3.0)) * v).dx();

    auto ir_ad = compiler.compileResidual(residual);
    auto ir_sym = compiler.compileResidual(residual);

    auto ad_kernel =
        std::make_shared<NonlinearFormKernel>(std::move(ir_ad), ADMode::Forward, NonlinearKernelOutput::Both);

    auto sym_kernel = std::make_shared<SymbolicNonlinearFormKernel>(std::move(ir_sym), NonlinearKernelOutput::Both);
    sym_kernel->resolveInlinableConstitutives();

    forms::JITOptions jit_opts;
    jit_opts.enable = true;
    jit_opts.optimization_level = 2;
    jit_opts.vectorize = true;
    jit_opts.cache_kernels = true;
    jit_opts.specialization.enable = false;
    auto jit_kernel = forms::jit::JITKernelWrapper(sym_kernel, jit_opts);

    // Ensure JIT compilation is viable (residual + tangent).
    {
        auto c = forms::jit::JITCompiler::getOrCreate(jit_opts);
        ASSERT_NE(c, nullptr);
        jit::ValidationOptions vopt;
        vopt.strictness = jit::Strictness::AllowExternalCalls;

        const auto r_res = c->compile(sym_kernel->residualIR(), vopt);
        ASSERT_TRUE(r_res.ok) << r_res.message;
        const auto r_tan = c->compile(sym_kernel->tangentIR(), vopt);
        ASSERT_TRUE(r_tan.ok) << r_tan.message;
    }

    assembly::StandardAssembler assembler;
    assembler.setDofMap(dof_map);
    assembler.setCurrentSolution(defaultSolution(static_cast<LocalIndex>(dof_map.getNumDofs())));

    assembly::DenseMatrixView J(static_cast<GlobalIndex>(dof_map.getNumDofs()));
    assembly::DenseVectorView R(static_cast<GlobalIndex>(dof_map.getNumDofs()));
    J.zero();
    R.zero();

    (void)assembler.assembleBoth(mesh, space, space, *ad_kernel, J, R);
    (void)assembler.assembleBoth(mesh, space, space, jit_kernel, J, R);

    const auto run_ad = [&]() {
        for (int k = 0; k < iters; ++k) {
            J.zero();
            R.zero();
            (void)assembler.assembleBoth(mesh, space, space, *ad_kernel, J, R);
        }
    };
    const auto run_jit = [&]() {
        for (int k = 0; k < iters; ++k) {
            J.zero();
            R.zero();
            (void)assembler.assembleBoth(mesh, space, space, jit_kernel, J, R);
        }
    };

    const double sec_ad = detail::bestOfSeconds(repeats, run_ad);
    const double sec_jit = detail::bestOfSeconds(repeats, run_jit);
    const double speedup = (sec_jit > 0.0) ? (sec_ad / sec_jit) : 0.0;

    std::cerr << "JITPerformanceRegression.NonlinearAssembly: iters=" << iters
              << " repeats=" << repeats
              << " ad_ms/call=" << (1e3 * sec_ad / static_cast<double>(iters))
              << " jit_ms/call=" << (1e3 * sec_jit / static_cast<double>(iters))
              << " speedup=" << speedup << "x (min=" << min_speedup << "x)\n";

    EXPECT_GE(speedup, min_speedup) << "Nonlinear JIT speedup fell below threshold.";
}

TEST(JITPerformanceRegression, SymbolicJacobian_JITFasterThanAD)
{
    requireLLVMJITOrSkip();
    if (!perfTestsEnabled()) {
        GTEST_SKIP() << "Set SVMP_FE_RUN_PERF_TESTS=1 to enable";
    }

    const int iters = std::max(1, detail::getenvInt("SVMP_FE_PERF_ITERS_JAC", 300));
    const int repeats = std::max(1, detail::getenvInt("SVMP_FE_PERF_REPEATS", 1));
    const double min_speedup = detail::getenvDouble("SVMP_FE_PERF_MIN_SPEEDUP_JAC", 1.2);

    SingleTetraMeshAccess mesh;
    spaces::H1Space space(ElementType::Tetra4, 1);
    auto dof_map = createSingleTetraDofMap();

    FormCompiler compiler;
    const auto u = FormExpr::trialFunction(space, "u");
    const auto v = FormExpr::testFunction(space, "v");
    const auto residual = (inner(grad(u), grad(v)) + pow(u, FormExpr::constant(3.0)) * v).dx();

    auto ir_ad = compiler.compileResidual(residual);
    auto ir_sym = compiler.compileResidual(residual);

    auto ad_kernel =
        std::make_shared<NonlinearFormKernel>(std::move(ir_ad), ADMode::Forward, NonlinearKernelOutput::MatrixOnly);

    auto sym_kernel =
        std::make_shared<SymbolicNonlinearFormKernel>(std::move(ir_sym), NonlinearKernelOutput::MatrixOnly);
    sym_kernel->resolveInlinableConstitutives();

    forms::JITOptions jit_opts;
    jit_opts.enable = true;
    jit_opts.optimization_level = 2;
    jit_opts.vectorize = true;
    jit_opts.cache_kernels = true;
    jit_opts.specialization.enable = false;
    auto jit_kernel = forms::jit::JITKernelWrapper(sym_kernel, jit_opts);

    // Ensure JIT compilation is viable (tangent only is sufficient, but compile both for safety).
    {
        auto c = forms::jit::JITCompiler::getOrCreate(jit_opts);
        ASSERT_NE(c, nullptr);
        jit::ValidationOptions vopt;
        vopt.strictness = jit::Strictness::AllowExternalCalls;

        const auto r_tan = c->compile(sym_kernel->tangentIR(), vopt);
        ASSERT_TRUE(r_tan.ok) << r_tan.message;
    }

    assembly::StandardAssembler assembler;
    assembler.setDofMap(dof_map);
    assembler.setCurrentSolution(defaultSolution(static_cast<LocalIndex>(dof_map.getNumDofs())));

    assembly::DenseMatrixView J(static_cast<GlobalIndex>(dof_map.getNumDofs()));
    assembly::DenseVectorView R(static_cast<GlobalIndex>(dof_map.getNumDofs()));
    J.zero();
    R.zero();

    (void)assembler.assembleBoth(mesh, space, space, *ad_kernel, J, R);
    (void)assembler.assembleBoth(mesh, space, space, jit_kernel, J, R);

    const auto run_ad = [&]() {
        for (int k = 0; k < iters; ++k) {
            J.zero();
            (void)assembler.assembleBoth(mesh, space, space, *ad_kernel, J, R);
        }
    };
    const auto run_jit = [&]() {
        for (int k = 0; k < iters; ++k) {
            J.zero();
            (void)assembler.assembleBoth(mesh, space, space, jit_kernel, J, R);
        }
    };

    const double sec_ad = detail::bestOfSeconds(repeats, run_ad);
    const double sec_jit = detail::bestOfSeconds(repeats, run_jit);
    const double speedup = (sec_jit > 0.0) ? (sec_ad / sec_jit) : 0.0;

    std::cerr << "JITPerformanceRegression.SymbolicJacobian: iters=" << iters
              << " repeats=" << repeats
              << " ad_ms/call=" << (1e3 * sec_ad / static_cast<double>(iters))
              << " jit_ms/call=" << (1e3 * sec_jit / static_cast<double>(iters))
              << " speedup=" << speedup << "x (min=" << min_speedup << "x)\n";

    EXPECT_GE(speedup, min_speedup) << "Symbolic Jacobian JIT speedup fell below threshold.";
}

} // namespace test
} // namespace forms
} // namespace FE
} // namespace svmp


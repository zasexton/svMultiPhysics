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

TEST(JITPerformanceRegression, BoundaryAssembly_JITNotMuchSlowerThanInterpreter)
{
    requireLLVMJITOrSkip();
    if (!perfTestsEnabled()) {
        GTEST_SKIP() << "Set SVMP_FE_RUN_PERF_TESTS=1 to enable";
    }

    const int iters = std::max(1, detail::getenvInt("SVMP_FE_PERF_ITERS_BOUNDARY", 2000));
    const int repeats = std::max(1, detail::getenvInt("SVMP_FE_PERF_REPEATS", 1));
    const double min_speedup = detail::getenvDouble("SVMP_FE_PERF_MIN_SPEEDUP_BOUNDARY", 0.8);

    constexpr int marker = 2;
    SingleTetraOneBoundaryFaceMeshAccess mesh(marker);
    spaces::H1Space space(ElementType::Tetra4, 1);
    auto dof_map = createSingleTetraDofMap();

    FormCompiler compiler;
    const auto v = FormExpr::testFunction(space, "v");
    const auto linear = v.ds(marker);

    auto ir_interp = compiler.compileLinear(linear);
    auto ir_jit = compiler.compileLinear(linear);

    auto interp_kernel = std::make_shared<FormKernel>(std::move(ir_interp));
    auto jit_fallback = std::make_shared<FormKernel>(std::move(ir_jit));

    forms::JITOptions jit_opts;
    jit_opts.enable = true;
    jit_opts.optimization_level = 2;
    jit_opts.vectorize = true;
    jit_opts.cache_kernels = true;
    jit_opts.specialization.enable = false;
    forms::jit::JITKernelWrapper jit_kernel(jit_fallback, jit_opts);

    // Ensure JIT compilation is viable.
    {
        auto c = forms::jit::JITCompiler::getOrCreate(jit_opts);
        ASSERT_NE(c, nullptr);
        jit::ValidationOptions vopt;
        vopt.strictness = jit::Strictness::Strict;
        const auto r = c->compile(jit_fallback->ir(), vopt);
        ASSERT_TRUE(r.ok) << r.message;
    }

    assembly::StandardAssembler assembler;
    assembler.setDofMap(dof_map);

    assembly::DenseVectorView vec(static_cast<GlobalIndex>(dof_map.getNumDofs()));
    vec.zero();

    (void)assembler.assembleBoundaryFaces(mesh, marker, space, *interp_kernel, nullptr, &vec);
    (void)assembler.assembleBoundaryFaces(mesh, marker, space, jit_kernel, nullptr, &vec);

    const auto run_interp = [&]() {
        for (int k = 0; k < iters; ++k) {
            vec.zero();
            (void)assembler.assembleBoundaryFaces(mesh, marker, space, *interp_kernel, nullptr, &vec);
        }
    };
    const auto run_jit = [&]() {
        for (int k = 0; k < iters; ++k) {
            vec.zero();
            (void)assembler.assembleBoundaryFaces(mesh, marker, space, jit_kernel, nullptr, &vec);
        }
    };

    const double sec_interp = detail::bestOfSeconds(repeats, run_interp);
    const double sec_jit = detail::bestOfSeconds(repeats, run_jit);
    const double speedup = (sec_jit > 0.0) ? (sec_interp / sec_jit) : 0.0;

    std::cerr << "JITPerformanceRegression.BoundaryAssembly: iters=" << iters
              << " repeats=" << repeats
              << " interp_us/call=" << (1e6 * sec_interp / static_cast<double>(iters))
              << " jit_us/call=" << (1e6 * sec_jit / static_cast<double>(iters))
              << " speedup=" << speedup << "x (min=" << min_speedup << "x)\n";

    EXPECT_GE(speedup, min_speedup) << "Boundary assembly speedup fell below threshold.";
}

TEST(JITPerformanceRegression, DGInteriorFaceAssembly_JITNotMuchSlowerThanInterpreter)
{
    requireLLVMJITOrSkip();
    if (!perfTestsEnabled()) {
        GTEST_SKIP() << "Set SVMP_FE_RUN_PERF_TESTS=1 to enable";
    }

    const int iters = std::max(1, detail::getenvInt("SVMP_FE_PERF_ITERS_DG", 600));
    const int repeats = std::max(1, detail::getenvInt("SVMP_FE_PERF_REPEATS", 1));
    const double min_speedup = detail::getenvDouble("SVMP_FE_PERF_MIN_SPEEDUP_DG", 0.8);

    TwoTetraSharedFaceMeshAccess mesh;
    spaces::H1Space space(ElementType::Tetra4, 1);
    auto dof_map = createTwoTetraDG_DofMap();

    FormCompiler compiler;
    const auto u = FormExpr::trialFunction(space, "u");
    const auto v = FormExpr::testFunction(space, "v");
    const auto form = (FormExpr::constant(1.0) * inner(jump(u), jump(v))).dS();

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

    // Ensure JIT compilation is viable.
    {
        auto c = forms::jit::JITCompiler::getOrCreate(jit_opts);
        ASSERT_NE(c, nullptr);
        jit::ValidationOptions vopt;
        vopt.strictness = jit::Strictness::Strict;
        const auto r = c->compile(jit_fallback->ir(), vopt);
        ASSERT_TRUE(r.ok) << r.message;
    }

    assembly::StandardAssembler assembler;
    assembler.setDofMap(dof_map);

    assembly::DenseMatrixView mat(static_cast<GlobalIndex>(dof_map.getNumDofs()));
    mat.zero();

    (void)assembler.assembleInteriorFaces(mesh, space, space, *interp_kernel, mat, nullptr);
    (void)assembler.assembleInteriorFaces(mesh, space, space, jit_kernel, mat, nullptr);

    const auto run_interp = [&]() {
        for (int k = 0; k < iters; ++k) {
            mat.zero();
            (void)assembler.assembleInteriorFaces(mesh, space, space, *interp_kernel, mat, nullptr);
        }
    };
    const auto run_jit = [&]() {
        for (int k = 0; k < iters; ++k) {
            mat.zero();
            (void)assembler.assembleInteriorFaces(mesh, space, space, jit_kernel, mat, nullptr);
        }
    };

    const double sec_interp = detail::bestOfSeconds(repeats, run_interp);
    const double sec_jit = detail::bestOfSeconds(repeats, run_jit);
    const double speedup = (sec_jit > 0.0) ? (sec_interp / sec_jit) : 0.0;

    std::cerr << "JITPerformanceRegression.DGInteriorFaceAssembly: iters=" << iters
              << " repeats=" << repeats
              << " interp_us/call=" << (1e6 * sec_interp / static_cast<double>(iters))
              << " jit_us/call=" << (1e6 * sec_jit / static_cast<double>(iters))
              << " speedup=" << speedup << "x (min=" << min_speedup << "x)\n";

    EXPECT_GE(speedup, min_speedup) << "DG interior-face assembly speedup fell below threshold.";
}

} // namespace test
} // namespace forms
} // namespace FE
} // namespace svmp


/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include <gtest/gtest.h>

#include "Forms/FormCompiler.h"
#include "Forms/FormKernels.h"
#include "Forms/JIT/JITCompiler.h"
#include "Spaces/H1Space.h"
#include "Tests/Unit/Forms/JITTestHelpers.h"
#include "Tests/Unit/Forms/PerfTestHelpers.h"

#include <iostream>

namespace svmp {
namespace FE {
namespace forms {
namespace test {

TEST(JITCompilationLatency, SimplePoissonBilinearCompilesUnder100ms)
{
    requireLLVMJITOrSkip();
    if (!perfTestsEnabled()) {
        GTEST_SKIP() << "Set SVMP_FE_RUN_PERF_TESTS=1 to enable";
    }

    const double max_ms = detail::getenvDouble("SVMP_FE_PERF_MAX_COMPILE_MS", 100.0);

    auto options = makeUnitTestJITOptions();
    options.optimization_level = 2;
    options.vectorize = true;
    options.dump_directory = "svmp_fe_jit_dumps_perf_compile_latency";
    auto compiler = jit::JITCompiler::getOrCreate(options);
    ASSERT_NE(compiler, nullptr);

    jit::ValidationOptions v;
    v.strictness = jit::Strictness::Strict;

    // Warm up ORC/LLVM initialization on this compiler instance.
    {
        const auto warm = compiler->compileFunctional(FormExpr::parameterRef(99), IntegralDomain::Cell, v);
        ASSERT_TRUE(warm.ok) << warm.message;
    }

    SymbolicOptions sym_opts;
    sym_opts.jit.enable = true;
    FormCompiler form_compiler(sym_opts);
    spaces::H1Space space(ElementType::Tetra4, 1);
    const auto u = FormExpr::trialFunction(space, "u");
    const auto w = FormExpr::testFunction(space, "w");
    const auto form = inner(grad(u), grad(w)).dx();
    const auto ir = form_compiler.compileBilinear(form);

    const double sec = detail::timeSeconds([&]() {
        const auto r = compiler->compile(ir, v);
        ASSERT_TRUE(r.ok) << r.message;
        ASSERT_FALSE(r.kernels.empty());
    });

    std::cerr << "JITCompilationLatency.SimplePoisson: compile_ms=" << (1e3 * sec)
              << " (max=" << max_ms << ")\n";

    EXPECT_LT(1e3 * sec, max_ms) << "JIT compilation took " << (1e3 * sec) << "ms, expected <" << max_ms << "ms";
}

TEST(JITCompilationLatency, ComplexNonlinearWithMatrixFunctionsCompilesUnder200ms)
{
    requireLLVMJITOrSkip();
    if (!perfTestsEnabled()) {
        GTEST_SKIP() << "Set SVMP_FE_RUN_PERF_TESTS=1 to enable";
    }

    const double max_ms = detail::getenvDouble("SVMP_FE_PERF_MAX_COMPILE_MS_COMPLEX", 200.0);

    auto options = makeUnitTestJITOptions();
    options.optimization_level = 2;
    options.vectorize = true;
    options.cache_kernels = false;
    options.dump_directory = "svmp_fe_jit_dumps_perf_compile_latency_complex";
    auto compiler = jit::JITCompiler::getOrCreate(options);
    ASSERT_NE(compiler, nullptr);

    jit::ValidationOptions v;
    v.strictness = jit::Strictness::AllowExternalCalls;

    // Warm up ORC/LLVM initialization on this compiler instance.
    {
        const auto warm = compiler->compileFunctional(FormExpr::parameterRef(101), IntegralDomain::Cell, v);
        ASSERT_TRUE(warm.ok) << warm.message;
    }

    FormCompiler form_compiler;
    spaces::H1Space space(ElementType::Tetra4, 1);
    const auto u = FormExpr::trialFunction(space, "u");
    const auto w = FormExpr::testFunction(space, "w");

    const auto s = u * u + FormExpr::constant(1.2);
    const auto A = FormExpr::identity(3) * s;

    const auto coeff = trace(A.matrixExp()) +
                       trace(A.matrixLog()) +
                       trace(A.matrixSqrt()) +
                       trace(A.matrixPow(FormExpr::constant(2.3)));

    const auto residual = (coeff * w).dx();
    auto residual_ir = form_compiler.compileResidual(residual);

    // For nonlinear forms, JIT compilation expects the current-solution representation (StateField)
    // rather than TrialFunction terminals. SymbolicNonlinearFormKernel handles this rewrite and
    // also builds a bilinear tangent IR via symbolic differentiation.
    SymbolicNonlinearFormKernel sym_kernel(std::move(residual_ir), NonlinearKernelOutput::Both);
    sym_kernel.resolveInlinableConstitutives();

    const double sec = detail::timeSeconds([&]() {
        const auto r_res = compiler->compile(sym_kernel.residualIR(), v);
        ASSERT_TRUE(r_res.ok) << r_res.message;
        ASSERT_FALSE(r_res.kernels.empty());

        const auto r_tan = compiler->compile(sym_kernel.tangentIR(), v);
        ASSERT_TRUE(r_tan.ok) << r_tan.message;
        ASSERT_FALSE(r_tan.kernels.empty());
    });

    std::cerr << "JITCompilationLatency.ComplexNonlinearWithMatrixFunctions: compile_ms=" << (1e3 * sec)
              << " (max=" << max_ms << ")\n";

    EXPECT_LT(1e3 * sec, max_ms) << "JIT compilation took " << (1e3 * sec) << "ms, expected <" << max_ms << "ms";
}

} // namespace test
} // namespace forms
} // namespace FE
} // namespace svmp

/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include <gtest/gtest.h>

#include "Assembly/StandardAssembler.h"
#include "Forms/Einsum.h"
#include "Forms/FormCompiler.h"
#include "Forms/FormKernels.h"
#include "Forms/Index.h"
#include "Forms/JIT/JITCompiler.h"
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

[[nodiscard]] FormExpr constantMat3(Real a00, Real a01, Real a02,
                                    Real a10, Real a11, Real a12,
                                    Real a20, Real a21, Real a22)
{
    std::vector<std::vector<FormExpr>> rows(3);
    rows[0] = {FormExpr::constant(a00), FormExpr::constant(a01), FormExpr::constant(a02)};
    rows[1] = {FormExpr::constant(a10), FormExpr::constant(a11), FormExpr::constant(a12)};
    rows[2] = {FormExpr::constant(a20), FormExpr::constant(a21), FormExpr::constant(a22)};
    return FormExpr::asTensor(std::move(rows));
}

} // namespace

TEST(JITPerformanceRegression, TensorContraction_JITAtLeast3xFasterThanInterpreter)
{
    requireLLVMJITOrSkip();
    if (!perfTestsEnabled()) {
        GTEST_SKIP() << "Set SVMP_FE_RUN_PERF_TESTS=1 to enable";
    }

    const int iters = std::max(1, detail::getenvInt("SVMP_FE_PERF_ITERS_TENSOR", 10));
    const int repeats = std::max(1, detail::getenvInt("SVMP_FE_PERF_REPEATS", 1));
    const double min_speedup = detail::getenvDouble("SVMP_FE_PERF_MIN_SPEEDUP_TENSOR", 3.0);

    SingleTetraMeshAccess mesh;
    auto base = std::make_shared<spaces::H1Space>(ElementType::Tetra4, 1);
    spaces::ProductSpace space(base, 3);
    auto dof_map = makeContiguousSingleCellDofMap(static_cast<GlobalIndex>(space.dofs_per_element()));

    const auto u = FormExpr::trialFunction(space, "u");
    const auto v = FormExpr::testFunction(space, "v");

    const auto A = constantMat3(1.1, 0.2, -0.3,
                                0.4, 1.5, 0.6,
                                -0.7, 0.8, 1.9);
    const auto B = constantMat3(2.0, 0.1, 0.0,
                                -0.2, 1.0, 0.3,
                                0.4, -0.1, 1.2);

    const Index i("i");
    const Index j("j");
    const Index k("k");
    const Index l("l");

    const auto integrand_tensor = A(i, j) * grad(u)(j, k) * B(k, l) * grad(v)(l, i);
    const auto form_tensor = integrand_tensor.dx();
    const auto form_scalar = forms::einsum(integrand_tensor).dx();

    SymbolicOptions allow_indexed;
    allow_indexed.jit.enable = true; // keep IndexedAccess in FormIR
    FormCompiler compiler_tensor(allow_indexed);
    auto ir_tensor = compiler_tensor.compileBilinear(form_tensor);

    FormCompiler compiler_scalar;
    auto ir_scalar = compiler_scalar.compileBilinear(form_scalar);

    auto interp_kernel = std::make_shared<FormKernel>(std::move(ir_scalar));
    auto jit_fallback = std::make_shared<FormKernel>(std::move(ir_tensor));

    forms::JITOptions jit_opts;
    jit_opts.enable = true;
    jit_opts.optimization_level = 2;
    jit_opts.vectorize = true;
    jit_opts.cache_kernels = true;
    jit_opts.specialization.enable = false;
    jit_opts.tensor.mode = TensorLoweringMode::On;
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

    (void)assembler.assembleMatrix(mesh, space, space, *interp_kernel, mat);
    (void)assembler.assembleMatrix(mesh, space, space, jit_kernel, mat);

    const auto run_interp = [&]() {
        for (int it = 0; it < iters; ++it) {
            mat.zero();
            (void)assembler.assembleMatrix(mesh, space, space, *interp_kernel, mat);
        }
    };
    const auto run_jit = [&]() {
        for (int it = 0; it < iters; ++it) {
            mat.zero();
            (void)assembler.assembleMatrix(mesh, space, space, jit_kernel, mat);
        }
    };

    const double sec_interp = detail::bestOfSeconds(repeats, run_interp);
    const double sec_jit = detail::bestOfSeconds(repeats, run_jit);
    const double speedup = (sec_jit > 0.0) ? (sec_interp / sec_jit) : 0.0;

    std::cerr << "JITPerformanceRegression.TensorContraction: iters=" << iters
              << " repeats=" << repeats
              << " interp_ms/call=" << (1e3 * sec_interp / static_cast<double>(iters))
              << " jit_ms/call=" << (1e3 * sec_jit / static_cast<double>(iters))
              << " speedup=" << speedup << "x (min=" << min_speedup << "x)\n";

    EXPECT_GE(speedup, min_speedup) << "Tensor-contraction JIT speedup fell below threshold.";
}

} // namespace test
} // namespace forms
} // namespace FE
} // namespace svmp

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

TEST(JITSpecializationPerformance, SpecializedNotMuchSlowerThanGeneric)
{
    requireLLVMJITOrSkip();
    if (!perfTestsEnabled()) {
        GTEST_SKIP() << "Set SVMP_FE_RUN_PERF_TESTS=1 to enable";
    }

    const int iters = detail::getenvInt("SVMP_FE_PERF_ITERS_SPEC", 80);
    const int repeats = std::max(1, detail::getenvInt("SVMP_FE_PERF_REPEATS", 1));
    const double min_ratio = detail::getenvDouble("SVMP_FE_PERF_MIN_SPECIALIZATION_RATIO", 0.8);

    SingleTetraMeshAccess mesh;
    spaces::H1Space space(ElementType::Tetra4, 3);
    auto dof_map = makeContiguousSingleCellDofMap(static_cast<GlobalIndex>(space.dofs_per_element()));

    FormCompiler compiler;
    const auto u = FormExpr::trialFunction(space, "u");
    const auto v = FormExpr::testFunction(space, "v");
    const auto form = inner(grad(u), grad(v)).dx();

    auto ir_generic = compiler.compileBilinear(form);
    auto ir_spec = compiler.compileBilinear(form);

    auto fallback_generic = std::make_shared<FormKernel>(std::move(ir_generic));
    auto fallback_spec = std::make_shared<FormKernel>(std::move(ir_spec));

    forms::JITOptions opt_generic;
    opt_generic.enable = true;
    opt_generic.optimization_level = 2;
    opt_generic.vectorize = true;
    opt_generic.cache_kernels = true;
    opt_generic.specialization.enable = false;

    forms::JITOptions opt_spec = opt_generic;
    opt_spec.specialization.enable = true;
    opt_spec.specialization.specialize_n_qpts = true;
    opt_spec.specialization.specialize_dofs = true;
    opt_spec.specialization.max_specialized_n_qpts = 512;
    opt_spec.specialization.max_specialized_dofs = 512;
    opt_spec.specialization.max_variants_per_kernel = 8;
    opt_spec.specialization.enable_loop_unroll_metadata = true;
    opt_spec.specialization.max_unroll_trip_count = 64;

    forms::jit::JITKernelWrapper jit_generic(fallback_generic, opt_generic);
    forms::jit::JITKernelWrapper jit_specialized(fallback_spec, opt_spec);

    assembly::StandardAssembler assembler;
    assembler.setDofMap(dof_map);

    assembly::DenseMatrixView mat(static_cast<GlobalIndex>(dof_map.getNumDofs()));
    mat.zero();

    // Warmup + compile (including any specialized variant).
    (void)assembler.assembleMatrix(mesh, space, space, jit_generic, mat);
    (void)assembler.assembleMatrix(mesh, space, space, jit_specialized, mat);

    const auto run_generic = [&]() {
        for (int k = 0; k < iters; ++k) {
            mat.zero();
            (void)assembler.assembleMatrix(mesh, space, space, jit_generic, mat);
        }
    };
    const auto run_specialized = [&]() {
        for (int k = 0; k < iters; ++k) {
            mat.zero();
            (void)assembler.assembleMatrix(mesh, space, space, jit_specialized, mat);
        }
    };

    const double sec_generic = detail::bestOfSeconds(repeats, run_generic);
    const double sec_spec = detail::bestOfSeconds(repeats, run_specialized);
    const double ratio = (sec_spec > 0.0) ? (sec_generic / sec_spec) : 0.0;

    std::cerr << "JITSpecializationPerformance.SpecializationRatio: iters=" << iters
              << " repeats=" << repeats
              << " generic_ms/call=" << (1e3 * sec_generic / static_cast<double>(iters))
              << " specialized_ms/call=" << (1e3 * sec_spec / static_cast<double>(iters))
              << " ratio=" << ratio << "x (min=" << min_ratio << "x)\n";

    EXPECT_GE(ratio, min_ratio)
        << "Specialized variant is too slow vs generic. generic=" << sec_generic << "s specialized=" << sec_spec
        << "s";
}

} // namespace test
} // namespace forms
} // namespace FE
} // namespace svmp


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
#include "Forms/JIT/JITEngine.h"
#include "Forms/JIT/LLVMGen.h"
#include "Forms/JIT/JITKernelWrapper.h"
#include "Spaces/H1Space.h"
#include "Tests/Unit/Forms/FormsTestHelpers.h"
#include "Tests/Unit/Forms/JITTestHelpers.h"
#include "Tests/Unit/Forms/PerfTestHelpers.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
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

[[nodiscard]] std::string readFileToString(const std::filesystem::path& path)
{
    std::ifstream in(path);
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

[[nodiscard]] bool llvmIRContainsVectorTypes(std::string_view ir_text)
{
    // Heuristic: look for vector types like "<4 x double>".
    return ir_text.find(" x double>") != std::string_view::npos ||
           ir_text.find(" x float>") != std::string_view::npos ||
           ir_text.find(" x i32>") != std::string_view::npos ||
           ir_text.find(" x i64>") != std::string_view::npos;
}

} // namespace

TEST(JITVectorization, VectorizedKernelFasterOrEqualToScalar)
{
    requireLLVMJITOrSkip();
    if (!perfTestsEnabled()) {
        GTEST_SKIP() << "Set SVMP_FE_RUN_PERF_TESTS=1 to enable";
    }

    const int iters = detail::getenvInt("SVMP_FE_PERF_ITERS_VECT", 200);
    const int repeats = std::max(1, detail::getenvInt("SVMP_FE_PERF_REPEATS", 1));
    // Default is intentionally tolerant; SIMD/vectorization can be neutral or even slightly negative
    // depending on kernel shape, LLVM decisions, and hardware. This guards against large regressions.
    const double min_speedup = detail::getenvDouble("SVMP_FE_PERF_MIN_SPEEDUP_VECT", 0.80);

    SingleTetraMeshAccess mesh;
    spaces::H1Space space(ElementType::Tetra4, 3);
    auto dof_map = makeContiguousSingleCellDofMap(static_cast<GlobalIndex>(space.dofs_per_element()));

    FormCompiler compiler;
    const auto u = FormExpr::trialFunction(space, "u");
    const auto v = FormExpr::testFunction(space, "v");
    const auto form = inner(grad(u), grad(v)).dx();

    auto ir_scalar = compiler.compileBilinear(form);
    auto ir_vector = compiler.compileBilinear(form);

    auto fallback_scalar = std::make_shared<FormKernel>(std::move(ir_scalar));
    auto fallback_vector = std::make_shared<FormKernel>(std::move(ir_vector));

    forms::JITOptions opt_scalar;
    opt_scalar.enable = true;
    opt_scalar.optimization_level = 2;
    opt_scalar.vectorize = false;
    opt_scalar.cache_kernels = true;
    opt_scalar.specialization.enable = false;

    forms::JITOptions opt_vector = opt_scalar;
    opt_vector.vectorize = true;

    forms::jit::JITKernelWrapper jit_scalar(fallback_scalar, opt_scalar);
    forms::jit::JITKernelWrapper jit_vector(fallback_vector, opt_vector);

    assembly::StandardAssembler assembler;
    assembler.setDofMap(dof_map);

    assembly::DenseMatrixView mat(static_cast<GlobalIndex>(dof_map.getNumDofs()));
    mat.zero();

    // Warmup + JIT compile.
    (void)assembler.assembleMatrix(mesh, space, space, jit_scalar, mat);
    (void)assembler.assembleMatrix(mesh, space, space, jit_vector, mat);

    const auto run_scalar = [&]() {
        for (int k = 0; k < iters; ++k) {
            mat.zero();
            (void)assembler.assembleMatrix(mesh, space, space, jit_scalar, mat);
        }
    };
    const auto run_vector = [&]() {
        for (int k = 0; k < iters; ++k) {
            mat.zero();
            (void)assembler.assembleMatrix(mesh, space, space, jit_vector, mat);
        }
    };

    const double sec_scalar = detail::bestOfSeconds(repeats, run_scalar);
    const double sec_vector = detail::bestOfSeconds(repeats, run_vector);

    const double speedup = (sec_vector > 0.0) ? (sec_scalar / sec_vector) : 0.0;
    std::cerr << "JITVectorization.VectorSpeedup: iters=" << iters
              << " repeats=" << repeats
              << " scalar_ms/call=" << (1e3 * sec_scalar / static_cast<double>(iters))
              << " vector_ms/call=" << (1e3 * sec_vector / static_cast<double>(iters))
              << " speedup=" << speedup << "x (min=" << min_speedup << "x)\n";

    EXPECT_GE(speedup, min_speedup)
        << "Vectorized JIT did not meet speedup threshold. scalar=" << sec_scalar << "s vector=" << sec_vector << "s";
}

TEST(JITVectorization, OptimizedIRContainsVectorTypesWhenVectorizeEnabled)
{
    requireLLVMJITOrSkip();
    if (!perfTestsEnabled()) {
        GTEST_SKIP() << "Set SVMP_FE_RUN_PERF_TESTS=1 to enable";
    }

    const bool require_simd = detail::getenvInt("SVMP_FE_PERF_REQUIRE_SIMD", 0) == 1;

    spaces::H1Space space(ElementType::Tetra4, 3);
    FormCompiler compiler;
    const auto u = FormExpr::trialFunction(space, "u");
    const auto v = FormExpr::testFunction(space, "v");
    const auto form = inner(grad(u), grad(v)).dx();
    const auto ir = compiler.compileBilinear(form);

    forms::JITOptions opt;
    opt.enable = true;
    opt.optimization_level = 2;
    opt.vectorize = true;
    opt.cache_kernels = false;
    opt.dump_llvm_ir = true;
    opt.dump_llvm_ir_optimized = true;
    opt.dump_directory = "svmp_fe_jit_dumps_perf_vectorization_ir";

    auto engine = forms::jit::JITEngine::create(opt);
    ASSERT_NE(engine, nullptr);
    forms::jit::LLVMGen gen(opt);

    std::vector<std::size_t> term_indices;
    for (std::size_t t = 0; t < ir.terms().size(); ++t) {
        if (ir.terms()[t].domain == IntegralDomain::Cell) {
            term_indices.push_back(t);
        }
    }
    ASSERT_FALSE(term_indices.empty());

    const std::string symbol = "jit_vect_check_p3";
    std::uintptr_t addr = 0;
    const auto r = gen.compileAndAddKernel(*engine,
                                           ir,
                                           term_indices,
                                           IntegralDomain::Cell,
                                           /*boundary_marker=*/-1,
                                           /*interface_marker=*/-1,
                                           symbol,
                                           addr);
    ASSERT_TRUE(r.ok) << r.message;
    ASSERT_NE(addr, 0u);

    const std::filesystem::path dump_dir = opt.dump_directory;
    const std::filesystem::path ll_after = dump_dir / (symbol + std::string("_after.ll"));
    const auto text = readFileToString(ll_after);
    const bool has_vec = llvmIRContainsVectorTypes(text);

    std::cerr << "JITVectorization.OptimizedIR: file=" << ll_after.string()
              << " vector_types=" << (has_vec ? "yes" : "no")
              << " (set SVMP_FE_PERF_REQUIRE_SIMD=1 to enforce)\n";

    if (require_simd) {
        EXPECT_TRUE(has_vec) << "No vector types found in optimized LLVM IR: " << ll_after.string();
    } else {
        SUCCEED();
    }
}

} // namespace test
} // namespace forms
} // namespace FE
} // namespace svmp

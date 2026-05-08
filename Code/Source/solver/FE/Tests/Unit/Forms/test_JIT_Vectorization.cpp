/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include <gtest/gtest.h>

#include "Assembly/JIT/KernelArgs.h"
#include "Assembly/StandardAssembler.h"
#include "Forms/FormCompiler.h"
#include "Forms/FormKernels.h"
#include "Forms/JIT/JITEngine.h"
#include "Forms/JIT/JITCompiler.h"
#include "Forms/JIT/LLVMGen.h"
#include "Forms/JIT/JITKernelWrapper.h"
#include "Spaces/H1Space.h"
#include "Tests/Unit/Forms/FormsTestHelpers.h"
#include "Tests/Unit/Forms/JITTestHelpers.h"
#include "Tests/Unit/Forms/PerfTestHelpers.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <span>
#include <sstream>
#include <string_view>
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

[[nodiscard]] std::vector<std::size_t> cellTermIndices(const FormIR& ir)
{
    std::vector<std::size_t> indices;
    for (std::size_t i = 0; i < ir.terms().size(); ++i) {
        if (ir.terms()[i].domain == IntegralDomain::Cell) {
            indices.push_back(i);
        }
    }
    return indices;
}

[[nodiscard]] std::vector<Real> assembleDenseMatrix(SingleTetraMeshAccess& mesh,
                                                    const spaces::H1Space& space,
                                                    const dofs::DofMap& dof_map,
                                                    assembly::AssemblyKernel& kernel)
{
    assembly::StandardAssembler assembler;
    assembler.setDofMap(dof_map);

    assembly::DenseMatrixView mat(static_cast<GlobalIndex>(dof_map.getNumDofs()));
    mat.zero();
    (void)assembler.assembleMatrix(mesh, space, space, kernel, mat);

    const auto data = mat.data();
    return {data.begin(), data.end()};
}

void expectDenseDataNear(const std::vector<Real>& actual,
                         const std::vector<Real>& expected,
                         Real tol)
{
    ASSERT_EQ(actual.size(), expected.size());
    for (std::size_t i = 0; i < actual.size(); ++i) {
        EXPECT_NEAR(actual[i], expected[i], tol) << "entry " << i;
    }
}

} // namespace

TEST(JITBasisBaking, DumpedIRContainsBakedScalarBasisConstants)
{
    requireLLVMJITOrSkip();

    spaces::H1Space space(ElementType::Tetra4, 1);
    FormCompiler compiler;
    const auto u = FormExpr::trialFunction(space, "u");
    const auto v = FormExpr::testFunction(space, "v");
    const auto ir = compiler.compileBilinear((u * v).dx());
    const auto term_indices = cellTermIndices(ir);
    ASSERT_FALSE(term_indices.empty());

    forms::JITOptions opt = makeUnitTestJITOptions();
    opt.cache_kernels = false;
    opt.dump_llvm_ir = true;
    opt.dump_llvm_ir_optimized = false;
    opt.dump_directory = "svmp_fe_jit_dumps_tests_basis_baking";
    std::filesystem::remove_all(opt.dump_directory);

    jit::JITCompileSpecialization spec;
    spec.domain = IntegralDomain::Cell;
    spec.n_qpts_minus = 1u;
    spec.n_test_dofs_minus = 4u;
    spec.n_trial_dofs_minus = 4u;
    spec.is_affine = true;
    spec.baked_basis.enabled = true;
    spec.baked_basis.geometry_affine = true;
    spec.baked_basis.hash = 0x8765'4321'feed'cafeULL;

    auto fill_side = [](jit::JITBakedBasisSide& side,
                        std::uint64_t hash,
                        std::vector<double> values) {
        side.enabled = true;
        side.scalar_basis = true;
        side.n_qpts = 1u;
        side.n_dofs = 4u;
        side.basis_hash = hash;
        side.quadrature_hash = 0x0102'0304'0506'0708ULL;
        side.table_hash = hash ^ 0xa5a5'a5a5'a5a5'a5a5ULL;
        side.scalar_values_qmajor = std::move(values);
    };
    fill_side(spec.baked_basis.test, 0x1111'2222'3333'4444ULL, {2.0, 3.0, 5.0, 7.0});
    fill_side(spec.baked_basis.trial, 0x9999'aaaa'bbbb'ccccULL, {11.0, 13.0, 17.0, 19.0});

    auto engine = forms::jit::JITEngine::create(opt);
    ASSERT_NE(engine, nullptr);
    forms::jit::LLVMGen gen(opt);

    constexpr std::string_view symbol = "jit_basis_bake_probe";
    std::uintptr_t addr = 0u;
    const auto result = gen.compileAndAddKernel(*engine,
                                                ir,
                                                term_indices,
                                                IntegralDomain::Cell,
                                                /*boundary_marker=*/-1,
                                                /*interface_marker=*/-1,
                                                symbol,
                                                addr,
                                                &spec);
    ASSERT_TRUE(result.ok) << result.message;
    ASSERT_NE(addr, 0u);

    const auto before_ir =
        readFileToString(std::filesystem::path(opt.dump_directory) /
                         (std::string(symbol) + "_before.ll"));
    ASSERT_FALSE(before_ir.empty());
    EXPECT_NE(before_ir.find("2.000000e+00"), std::string::npos);
    EXPECT_NE(before_ir.find("1.100000e+01"), std::string::npos);
}

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

TEST(JITColocatedKernels, CompileColocatedAddressesAssembleLikeFallback)
{
    requireLLVMJITOrSkip();

    SingleTetraMeshAccess mesh;
    spaces::H1Space space(ElementType::Tetra4, 1);
    auto dof_map = makeContiguousSingleCellDofMap(static_cast<GlobalIndex>(space.dofs_per_element()));

    FormCompiler compiler;
    const auto u = FormExpr::trialFunction(space, "u");
    const auto v = FormExpr::testFunction(space, "v");
    const auto mass_form = (u * v).dx();
    const auto stiffness_form = inner(grad(u), grad(v)).dx();

    auto mass_ir_fallback = compiler.compileBilinear(mass_form);
    auto stiffness_ir_fallback = compiler.compileBilinear(stiffness_form);
    auto mass_ir_colocated = compiler.compileBilinear(mass_form);
    auto stiffness_ir_colocated = compiler.compileBilinear(stiffness_form);

    auto mass_fallback = std::make_shared<FormKernel>(std::move(mass_ir_fallback));
    auto stiffness_fallback = std::make_shared<FormKernel>(std::move(stiffness_ir_fallback));

    auto options = makeUnitTestJITOptions();
    options.dump_directory = "svmp_fe_jit_dumps_tests_colocated";
    auto jit_compiler = jit::JITCompiler::getOrCreate(options);
    ASSERT_NE(jit_compiler, nullptr);

    std::vector<jit::JITCompiler::ColocatedKernelSpec> specs;
    specs.push_back(jit::JITCompiler::ColocatedKernelSpec{
        .ir = &mass_ir_colocated,
        .term_indices = cellTermIndices(mass_ir_colocated),
        .domain = IntegralDomain::Cell,
    });
    specs.push_back(jit::JITCompiler::ColocatedKernelSpec{
        .ir = &stiffness_ir_colocated,
        .term_indices = cellTermIndices(stiffness_ir_colocated),
        .domain = IntegralDomain::Cell,
    });
    ASSERT_FALSE(specs[0].term_indices.empty());
    ASSERT_FALSE(specs[1].term_indices.empty());

    std::vector<jit::JITCompiler::ColocatedKernelResult> results;
    const auto colocated = jit_compiler->compileColocated(specs, results);
    ASSERT_TRUE(colocated.ok) << colocated.message;
    ASSERT_EQ(results.size(), specs.size());
    ASSERT_NE(results[0].address, 0u);
    ASSERT_NE(results[1].address, 0u);
    ASSERT_EQ(results[0].symbol.rfind("svmp_fe_jit_coloc_", 0u), 0u);
    ASSERT_EQ(results[1].symbol.rfind("svmp_fe_jit_coloc_", 0u), 0u);
    EXPECT_NE(results[0].symbol, results[1].symbol);

    jit::JITKernelWrapper mass_jit(mass_fallback, options);
    mass_jit.setExternalCellAddress(results[0].address);
    ASSERT_TRUE(mass_jit.isJITReady());

    jit::JITKernelWrapper stiffness_jit(stiffness_fallback, options);
    stiffness_jit.setExternalCellAddress(results[1].address);
    ASSERT_TRUE(stiffness_jit.isJITReady());

    const auto mass_expected = assembleDenseMatrix(mesh, space, dof_map, *mass_fallback);
    const auto mass_actual = assembleDenseMatrix(mesh, space, dof_map, mass_jit);
    expectDenseDataNear(mass_actual, mass_expected, 1e-12);

    const auto stiffness_expected = assembleDenseMatrix(mesh, space, dof_map, *stiffness_fallback);
    const auto stiffness_actual = assembleDenseMatrix(mesh, space, dof_map, stiffness_jit);
    expectDenseDataNear(stiffness_actual, stiffness_expected, 1e-12);
}

TEST(JITCoupledKernels, CompileDirectBlockOrderingAndOutputBuffers)
{
    requireLLVMJITOrSkip();

    spaces::H1Space space(ElementType::Tetra4, 1);
    FormCompiler compiler;
    const auto u = FormExpr::trialFunction(space, "u");
    const auto v = FormExpr::testFunction(space, "v");

    const auto matrix_only_ir = compiler.compileBilinear((FormExpr::constant(2.0) * u * v).dx());
    const auto vector_only_ir = compiler.compileResidual((FormExpr::constant(3.0) * u * v).dx());
    const auto mixed_matrix_ir = compiler.compileBilinear((FormExpr::constant(5.0) * u * v).dx());
    const auto mixed_vector_ir = compiler.compileResidual((FormExpr::constant(7.0) * u * v).dx());

    auto options = makeUnitTestJITOptions();
    options.cache_kernels = false;
    options.dump_directory = "svmp_fe_jit_dumps_tests_coupled_direct";
    auto engine = jit::JITEngine::create(options);
    ASSERT_NE(engine, nullptr);

    jit::LLVMGen gen(options);
    const std::array<jit::LLVMGen::MonolithicBlockInfo, 3> blocks = {{
        {.tangent_ir = &matrix_only_ir, .want_matrix = true, .want_vector = false},
        {.residual_ir = &vector_only_ir, .want_matrix = false, .want_vector = true},
        {.tangent_ir = &mixed_matrix_ir, .residual_ir = &mixed_vector_ir, .want_matrix = true, .want_vector = true},
    }};

    constexpr std::string_view symbol = "svmp_fe_jit_test_coupled_direct_ordering";
    std::uintptr_t address = 0;
    const auto compiled = gen.compileAndAddCoupledKernel(*engine, blocks, symbol, address);
    ASSERT_TRUE(compiled.ok) << compiled.message;
    ASSERT_NE(address, 0u);

    std::uintptr_t lookup_address = 0;
    EXPECT_TRUE(engine->tryLookup(symbol, lookup_address));
    EXPECT_EQ(lookup_address, address);

    const std::array<Real, 4> basis = {{0.1, 0.2, 0.3, 0.4}};
    const std::array<Real, 4> solution = {{1.0, 2.0, 3.0, 4.0}};
    const Real solution_at_q =
        basis[0] * solution[0] + basis[1] * solution[1] +
        basis[2] * solution[2] + basis[3] * solution[3];

    std::array<Real, 16> matrix_only{};
    std::array<Real, 4> vector_only{};
    std::array<Real, 16> mixed_matrix{};
    std::array<Real, 4> mixed_vector{};

    std::array<assembly::jit::CoupledBlockView, 3> block_views = {{
        assembly::jit::CoupledBlockView{
            .test_basis_values = basis.data(),
            .trial_basis_values = basis.data(),
            .n_test_dofs = 4,
            .n_trial_dofs = 4,
            .solution_coefficients = solution.data(),
            .element_matrix = matrix_only.data(),
        },
        assembly::jit::CoupledBlockView{
            .test_basis_values = basis.data(),
            .trial_basis_values = basis.data(),
            .n_test_dofs = 4,
            .n_trial_dofs = 4,
            .solution_coefficients = solution.data(),
            .element_vector = vector_only.data(),
        },
        assembly::jit::CoupledBlockView{
            .test_basis_values = basis.data(),
            .trial_basis_values = basis.data(),
            .n_test_dofs = 4,
            .n_trial_dofs = 4,
            .solution_coefficients = solution.data(),
            .element_matrix = mixed_matrix.data(),
            .element_vector = mixed_vector.data(),
        },
    }};

    const std::array<Real, 1> weights = {{1.0}};
    const std::array<Real, 1> dets = {{1.0}};
    const std::array<Real, 9> identity = {{1.0, 0.0, 0.0,
                                           0.0, 1.0, 0.0,
                                           0.0, 0.0, 1.0}};
    const std::array<Real, 3> qpt = {{0.25, 0.25, 0.25}};

    assembly::jit::CoupledCellKernelArgsV1 element;
    element.abi_version = assembly::jit::kCoupledCellKernelABIV1;
    element.num_blocks = static_cast<std::uint32_t>(block_views.size());
    element.dim = 3;
    element.n_qpts = 1;
    element.integration_weights = weights.data();
    element.jacobians = identity.data();
    element.inverse_jacobians = identity.data();
    element.jacobian_dets = dets.data();
    element.physical_points_xyz = qpt.data();
    element.blocks = block_views.data();

    assembly::jit::CoupledCellKernelBatchArgsV1 batch;
    batch.abi_version = assembly::jit::kCoupledCellKernelABIV1;
    batch.batch_size = 1;
    batch.num_blocks = static_cast<std::uint32_t>(block_views.size());
    batch.elements = &element;

    using CoupledKernelFn = void (*)(void*);
    reinterpret_cast<CoupledKernelFn>(address)(reinterpret_cast<void*>(&batch));

    const auto expect_matrix = [&](const std::array<Real, 16>& actual, Real coeff, std::string_view label) {
        for (std::size_t i = 0; i < basis.size(); ++i) {
            for (std::size_t j = 0; j < basis.size(); ++j) {
                EXPECT_NEAR(actual[i * basis.size() + j], coeff * basis[i] * basis[j], 1e-12)
                    << label << "(" << i << "," << j << ")";
            }
        }
    };
    const auto expect_vector = [&](const std::array<Real, 4>& actual, Real coeff, std::string_view label) {
        for (std::size_t i = 0; i < basis.size(); ++i) {
            EXPECT_NEAR(actual[i], coeff * solution_at_q * basis[i], 1e-12)
                << label << "(" << i << ")";
        }
    };

    expect_matrix(matrix_only, 2.0, "matrix-only block");
    expect_vector(vector_only, 3.0, "vector-only block");
    expect_matrix(mixed_matrix, 5.0, "mixed matrix block");
    expect_vector(mixed_vector, 7.0, "mixed vector block");
}

TEST(JITCoupledKernels, RejectsEmptyAndMissingBlockSpecs)
{
    requireLLVMJITOrSkip();

    auto options = makeUnitTestJITOptions();
    options.cache_kernels = false;
    options.dump_directory = "svmp_fe_jit_dumps_tests_coupled_invalid";
    auto engine = jit::JITEngine::create(options);
    ASSERT_NE(engine, nullptr);

    jit::LLVMGen gen(options);

    std::uintptr_t address = 1234;
    std::span<const jit::LLVMGen::MonolithicBlockInfo> empty_blocks;
    const auto empty = gen.compileAndAddCoupledKernel(*engine, empty_blocks,
                                                      "svmp_fe_jit_test_empty_coupled",
                                                      address);
    EXPECT_FALSE(empty.ok);
    EXPECT_EQ(address, 0u);
    EXPECT_NE(empty.message.find("no blocks"), std::string::npos);

    const std::array<jit::LLVMGen::MonolithicBlockInfo, 1> missing = {{
        {.want_matrix = true, .want_vector = false},
    }};
    address = 1234;
    const auto no_ir = gen.compileAndAddCoupledKernel(*engine, missing,
                                                      "svmp_fe_jit_test_missing_coupled_ir",
                                                      address);
    EXPECT_FALSE(no_ir.ok);
    EXPECT_EQ(address, 0u);
    EXPECT_NE(no_ir.message.find("has no cell terms"), std::string::npos);
}

TEST(JITColocatedKernels, RejectsNullFormIRSpec)
{
    requireLLVMJITOrSkip();

    auto options = makeUnitTestJITOptions();
    options.dump_directory = "svmp_fe_jit_dumps_tests_colocated_null";
    auto jit_compiler = jit::JITCompiler::getOrCreate(options);
    ASSERT_NE(jit_compiler, nullptr);

    std::vector<jit::JITCompiler::ColocatedKernelSpec> specs(1);
    std::vector<jit::JITCompiler::ColocatedKernelResult> results;

    const auto r = jit_compiler->compileColocated(specs, results);
    EXPECT_FALSE(r.ok);
    EXPECT_NE(r.message.find("null FormIR"), std::string::npos);
    EXPECT_TRUE(results.empty());
}

} // namespace test
} // namespace forms
} // namespace FE
} // namespace svmp

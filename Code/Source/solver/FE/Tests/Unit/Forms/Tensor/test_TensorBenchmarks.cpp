/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

/**
 * @file test_TensorBenchmarks.cpp
 * @brief Micro-benchmarks for tensor-calculus lowering/evaluation (disabled by default)
 *
 * Enable with `--gtest_also_run_disabled_tests`.
 */

#include <gtest/gtest.h>

#include "Assembly/StandardAssembler.h"
#include "Forms/Einsum.h"
#include "Forms/FormCompiler.h"
#include "Forms/FormKernels.h"
#include "Forms/Index.h"
#include "Forms/JIT/KernelIR.h"
#include "Spaces/H1Space.h"
#include "Spaces/ProductSpace.h"
#include "Tests/Unit/Forms/FormsTestHelpers.h"

#include <chrono>
#include <memory>
#include <vector>

#ifndef SVMP_FE_ENABLE_LLVM_JIT
#define SVMP_FE_ENABLE_LLVM_JIT 0
#endif

#if SVMP_FE_ENABLE_LLVM_JIT
#include "Forms/JIT/JITCompiler.h"
#include "Forms/JIT/JITValidation.h"
#include "Forms/JIT/LLVMGen.h"

#include <filesystem>
#include <fstream>
#include <sstream>
#endif

namespace svmp::FE::forms::tensor {

namespace {

using Clock = std::chrono::steady_clock;

template <class Fn>
double timeSeconds(Fn&& fn)
{
    const auto t0 = Clock::now();
    fn();
    const auto t1 = Clock::now();
    return std::chrono::duration_cast<std::chrono::duration<double>>(t1 - t0).count();
}

[[nodiscard]] std::size_t countExprNodes(const std::shared_ptr<FormExprNode>& node)
{
    if (!node) return 0u;
    std::size_t count = 0u;
    const auto visit = [&](const auto& self, const std::shared_ptr<FormExprNode>& n) -> void {
        if (!n) return;
        ++count;
        for (const auto& c : n->childrenShared()) {
            self(self, c);
        }
    };
    visit(visit, node);
    return count;
}

[[nodiscard]] dofs::DofMap makeSingleCellDofMap(GlobalIndex n_dofs)
{
    dofs::DofMap dof_map(1, n_dofs, static_cast<LocalIndex>(n_dofs));
    std::vector<GlobalIndex> cell_dofs(static_cast<std::size_t>(n_dofs));
    for (GlobalIndex i = 0; i < n_dofs; ++i) {
        cell_dofs[static_cast<std::size_t>(i)] = i;
    }
    dof_map.setCellDofs(0, cell_dofs);
    dof_map.setNumDofs(n_dofs);
    dof_map.setNumLocalDofs(n_dofs);
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

TEST(TensorBenchmarks, DISABLED_ExpressionAndKernelIRSize)
{
    auto base = std::make_shared<spaces::H1Space>(ElementType::Tetra4, 1);
    spaces::ProductSpace space(base, 3);

    const auto u = FormExpr::trialFunction(space, "u");
    const auto v = FormExpr::testFunction(space, "v");

    const auto A = constantMat3(1.1, 0.2, -0.3,
                                0.4, 1.5, 0.6,
                                -0.7, 0.8, 1.9);
    const auto B = constantMat3(2.0, 0.1, 0.0,
                                -0.2, 1.0, 0.3,
                                0.4, -0.1, 1.2);

    const forms::Index i("i");
    const forms::Index j("j");
    const forms::Index k("k");
    const forms::Index l("l");

    // A(i,j) * grad(u)(j,k) * B(k,l) * grad(v)(l,i) -> scalar (dim^4 = 81 terms in 3D)
    const auto integrand_tensor = A(i, j) * grad(u)(j, k) * B(k, l) * grad(v)(l, i);
    const auto integrand_scalar = forms::einsum(integrand_tensor);

    const std::size_t nodes_tensor = countExprNodes(integrand_tensor.nodeShared());
    const std::size_t nodes_scalar = countExprNodes(integrand_scalar.nodeShared());

    const auto kir_tensor = forms::jit::lowerToKernelIR(integrand_tensor);
    const auto kir_scalar = forms::jit::lowerToKernelIR(integrand_scalar);

    SUCCEED() << "FormExpr nodes: tensor=" << nodes_tensor << " scalar_expanded=" << nodes_scalar
              << " | KernelIR ops: tensor=" << kir_tensor.ir.opCount() << " scalar_expanded=" << kir_scalar.ir.opCount();
}

TEST(TensorBenchmarks, DISABLED_InterpreterAssemblyThroughput)
{
    forms::test::SingleTetraMeshAccess mesh;
    auto base = std::make_shared<spaces::H1Space>(ElementType::Tetra4, 1);
    spaces::ProductSpace space(base, 3);
    auto dof_map = makeSingleCellDofMap(static_cast<GlobalIndex>(space.dofs_per_element()));

    const auto u = FormExpr::trialFunction(space, "u");
    const auto v = FormExpr::testFunction(space, "v");

    const auto A = constantMat3(1.1, 0.2, -0.3,
                                0.4, 1.5, 0.6,
                                -0.7, 0.8, 1.9);
    const auto B = constantMat3(2.0, 0.1, 0.0,
                                -0.2, 1.0, 0.3,
                                0.4, -0.1, 1.2);

    const forms::Index i("i");
    const forms::Index j("j");
    const forms::Index k("k");
    const forms::Index l("l");
    const auto form = (A(i, j) * grad(u)(j, k) * B(k, l) * grad(v)(l, i)).dx();

    SymbolicOptions opts;
    opts.jit.enable = true; // keep IndexedAccess in FormIR
    FormCompiler compiler(opts);

    auto ir_a = compiler.compileBilinear(form);
    auto ir_b = compiler.compileBilinear(form);

    FormKernel kernel_scalar(std::move(ir_a)); // scalar-expanded via einsum() at eval time
    FormKernel kernel_tensor(std::move(ir_b));
    TensorJITOptions tensor_opts;
    tensor_opts.mode = TensorLoweringMode::On;
    kernel_tensor.setTensorInterpreterOptions(tensor_opts);

    assembly::StandardAssembler assembler;
    assembler.setDofMap(dof_map);

    assembly::DenseMatrixView mat(static_cast<GlobalIndex>(space.dofs_per_element()));
    mat.zero();

    constexpr int kIters = 400;
    (void)assembler.assembleMatrix(mesh, space, space, kernel_scalar, mat); // warmup
    const double sec_scalar = timeSeconds([&]() {
        for (int it = 0; it < kIters; ++it) {
            mat.zero();
            (void)assembler.assembleMatrix(mesh, space, space, kernel_scalar, mat);
        }
    });

    (void)assembler.assembleMatrix(mesh, space, space, kernel_tensor, mat); // warmup
    const double sec_tensor = timeSeconds([&]() {
        for (int it = 0; it < kIters; ++it) {
            mat.zero();
            (void)assembler.assembleMatrix(mesh, space, space, kernel_tensor, mat);
        }
    });

    const double eps_scalar = (sec_scalar > 0.0) ? (static_cast<double>(kIters) / sec_scalar) : 0.0;
    const double eps_tensor = (sec_tensor > 0.0) ? (static_cast<double>(kIters) / sec_tensor) : 0.0;
    SUCCEED() << "Interpreter elem/s: scalar_expanded=" << eps_scalar << " tensor_loops=" << eps_tensor;
}

#if SVMP_FE_ENABLE_LLVM_JIT

namespace {

[[nodiscard]] std::size_t countLLVMInstructionsText(std::string_view ir_text)
{
    std::size_t count = 0u;
    bool in_define = false;
    for (std::size_t pos = 0; pos < ir_text.size();) {
        const std::size_t eol = ir_text.find('\n', pos);
        const std::size_t len = (eol == std::string_view::npos) ? (ir_text.size() - pos) : (eol - pos);
        const auto line = ir_text.substr(pos, len);
        pos = (eol == std::string_view::npos) ? ir_text.size() : (eol + 1u);

        if (!in_define) {
            if (line.rfind("define ", 0) == 0) {
                in_define = true;
            }
            continue;
        }

        if (line == "}") {
            in_define = false;
            continue;
        }

        if (line.size() >= 2 && line[0] == ' ' && line[1] == ' ') {
            // Skip labels and comments.
            if (!line.empty() && line.back() == ':') continue;
            if (line.size() >= 3 && line[2] == ';') continue;
            ++count;
        }
    }
    return count;
}

[[nodiscard]] std::string readFileToString(const std::filesystem::path& path)
{
    std::ifstream in(path);
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

} // namespace

TEST(TensorBenchmarks, DISABLED_JITCompileTimeCacheHitAndLLVMIRSize)
{
    auto base = std::make_shared<spaces::H1Space>(ElementType::Tetra4, 1);
    spaces::ProductSpace space(base, 3);

    const auto u = FormExpr::trialFunction(space, "u");
    const auto v = FormExpr::testFunction(space, "v");

    const auto A = constantMat3(1.1, 0.2, -0.3,
                                0.4, 1.5, 0.6,
                                -0.7, 0.8, 1.9);
    const auto B = constantMat3(2.0, 0.1, 0.0,
                                -0.2, 1.0, 0.3,
                                0.4, -0.1, 1.2);

    const forms::Index i("i");
    const forms::Index j("j");
    const forms::Index k("k");
    const forms::Index l("l");

    const auto integrand_tensor = A(i, j) * grad(u)(j, k) * B(k, l) * grad(v)(l, i);
    const auto form_tensor = integrand_tensor.dx();
    const auto form_scalar = forms::einsum(integrand_tensor).dx();

    // Compile IRs.
    SymbolicOptions allow_indexed;
    allow_indexed.jit.enable = true;
    FormCompiler compiler_indexed(allow_indexed);
    auto ir_tensor = compiler_indexed.compileBilinear(form_tensor);

    FormCompiler compiler_scalar;
    auto ir_scalar = compiler_scalar.compileBilinear(form_scalar);

    // JIT compile time: first compile vs cache hit.
    forms::JITOptions jit_opts;
    jit_opts.enable = true;
    jit_opts.optimization_level = 2;
    jit_opts.cache_kernels = true;
    jit_opts.vectorize = true;
    jit_opts.tensor.mode = TensorLoweringMode::On;

    auto compiler = forms::jit::JITCompiler::getOrCreate(jit_opts);
    compiler->resetCacheStats();

    const double sec_first = timeSeconds([&]() {
        const auto r = compiler->compile(ir_tensor);
        ASSERT_TRUE(r.ok) << r.message;
    });

    const double sec_hit = timeSeconds([&]() {
        const auto r = compiler->compile(ir_tensor);
        ASSERT_TRUE(r.ok) << r.message;
    });

    // LLVM IR instruction count proxy (text-based) for tensor vs scalar-expanded lowering.
    //
    // Use LLVMGen directly so the symbol and dump path are deterministic.
    auto buildAndCount = [&](const FormIR& ir, const forms::JITOptions& opt, std::string_view tag) -> std::size_t {
        namespace fs = std::filesystem;
        fs::path dump_dir = fs::path(opt.dump_directory) / std::string(tag);
        fs::create_directories(dump_dir);

        forms::JITOptions local = opt;
        local.dump_llvm_ir = true;
        local.dump_directory = dump_dir.string();

        auto engine = forms::jit::JITEngine::create(local);
        if (!engine) {
            ADD_FAILURE() << "TensorBenchmarks: failed to create JITEngine";
            return 0u;
        }
        forms::jit::LLVMGen gen(local);

        std::vector<std::size_t> term_indices;
        for (std::size_t t = 0; t < ir.terms().size(); ++t) {
            if (ir.terms()[t].domain == IntegralDomain::Cell) {
                term_indices.push_back(t);
            }
        }
        if (term_indices.empty()) {
            ADD_FAILURE() << "TensorBenchmarks: no Cell terms found in FormIR";
            return 0u;
        }

        std::uintptr_t addr = 0;
        const auto r = gen.compileAndAddKernel(*engine,
                                               ir,
                                               term_indices,
                                               IntegralDomain::Cell,
                                               /*boundary_marker=*/-1,
                                               /*interface_marker=*/-1,
                                               std::string("tensor_bench_") + std::string(tag),
                                               addr);
        if (!r.ok) {
            ADD_FAILURE() << r.message;
            return 0u;
        }

        const fs::path ll = dump_dir / (std::string("tensor_bench_") + std::string(tag) + "_before.ll");
        const auto text = readFileToString(ll);
        return countLLVMInstructionsText(text);
    };

    forms::JITOptions opt_tensor = jit_opts;
    opt_tensor.tensor.mode = TensorLoweringMode::On;
    const std::size_t inst_tensor = buildAndCount(ir_tensor, opt_tensor, "tensor");

    forms::JITOptions opt_scalar = jit_opts;
    opt_scalar.tensor.mode = TensorLoweringMode::Off;
    const std::size_t inst_scalar = buildAndCount(ir_scalar, opt_scalar, "scalar");

    SUCCEED() << "JIT compile sec: first=" << sec_first << " cache_hit=" << sec_hit
              << " | LLVM IR inst (approx): tensor=" << inst_tensor << " scalar_expanded=" << inst_scalar;
}

#endif // SVMP_FE_ENABLE_LLVM_JIT

} // namespace svmp::FE::forms::tensor

/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

/**
 * @file test_SymbolicAssembler.cpp
 * @brief Unit tests for SymbolicAssembler and form expression DSL
 */

#include <gtest/gtest.h>

#include "Assembly/SymbolicAssembler.h"
#include "Assembly/AssemblyContext.h"

#include <cmath>
#include <stdexcept>

namespace svmp {
namespace FE {
namespace assembly {
namespace test {

// ============================================================================
// FormExprType Tests
// ============================================================================

TEST(FormExprTypeTest, EnumValues) {
    // Test function types
    EXPECT_NE(FormExprType::TestFunction, FormExprType::TrialFunction);
    EXPECT_NE(FormExprType::TrialFunction, FormExprType::Coefficient);

    // Derivative types
    EXPECT_NE(FormExprType::Gradient, FormExprType::Divergence);
    EXPECT_NE(FormExprType::Curl, FormExprType::Hessian);

    // Operation types
    EXPECT_NE(FormExprType::InnerProduct, FormExprType::OuterProduct);
    EXPECT_NE(FormExprType::Multiply, FormExprType::Add);

    // Integration types
    EXPECT_NE(FormExprType::CellIntegral, FormExprType::BoundaryIntegral);
}

// ============================================================================
// ADMode Tests
// ============================================================================

TEST(ADModeTest, EnumValues) {
    EXPECT_NE(ADMode::None, ADMode::Forward);
    EXPECT_NE(ADMode::Forward, ADMode::Reverse);
    EXPECT_NE(ADMode::Reverse, ADMode::Taylor);
}

// ============================================================================
// JITOptions Tests
// ============================================================================

TEST(JITOptionsTest, Defaults) {
    JITOptions options;

    EXPECT_FALSE(options.enable);
    EXPECT_EQ(options.optimization_level, 2);
    EXPECT_TRUE(options.cache_kernels);
    EXPECT_TRUE(options.vectorize);
    EXPECT_TRUE(options.cache_directory.empty());
}

TEST(JITOptionsTest, CustomValues) {
    JITOptions options;
    options.enable = true;
    options.optimization_level = 3;
    options.cache_directory = "/tmp/jit_cache";

    EXPECT_TRUE(options.enable);
    EXPECT_EQ(options.optimization_level, 3);
    EXPECT_EQ(options.cache_directory, "/tmp/jit_cache");
}

// ============================================================================
// SymbolicOptions Tests
// ============================================================================

TEST(SymbolicOptionsTest, Defaults) {
    SymbolicOptions options;

    EXPECT_EQ(options.ad_mode, ADMode::None);
    EXPECT_FALSE(options.jit.enable);
    EXPECT_TRUE(options.simplify_expressions);
    EXPECT_TRUE(options.exploit_sparsity);
    EXPECT_TRUE(options.cache_expressions);
    EXPECT_FALSE(options.verbose);
}

TEST(SymbolicOptionsTest, WithAD) {
    SymbolicOptions options;
    options.ad_mode = ADMode::Forward;

    EXPECT_EQ(options.ad_mode, ADMode::Forward);
}

TEST(SymbolicOptionsTest, WithJIT) {
    SymbolicOptions options;
    options.jit.enable = true;
    options.jit.optimization_level = 3;

    EXPECT_TRUE(options.jit.enable);
    EXPECT_EQ(options.jit.optimization_level, 3);
}

// ============================================================================
// FormExpr Tests
// ============================================================================

TEST(FormExprTest, DefaultConstruction) {
    FormExpr expr;

    EXPECT_FALSE(expr.isValid());
}

TEST(FormExprTest, TestFunction) {
    FormExpr v = FormExpr::testFunction("v");

    EXPECT_TRUE(v.isValid());
    EXPECT_TRUE(v.hasTest());
    EXPECT_FALSE(v.hasTrial());
    EXPECT_FALSE(v.isBilinear());
    EXPECT_TRUE(v.isLinear());
}

TEST(FormExprTest, TrialFunction) {
    FormExpr u = FormExpr::trialFunction("u");

    EXPECT_TRUE(u.isValid());
    EXPECT_FALSE(u.hasTest());
    EXPECT_TRUE(u.hasTrial());
}

TEST(FormExprTest, Constant) {
    FormExpr c = FormExpr::constant(3.14);

    EXPECT_TRUE(c.isValid());
    EXPECT_FALSE(c.hasTest());
    EXPECT_FALSE(c.hasTrial());
}

TEST(FormExprTest, Identity) {
    FormExpr I = FormExpr::identity(3);

    EXPECT_TRUE(I.isValid());
}

TEST(FormExprTest, Normal) {
    FormExpr n = FormExpr::normal();

    EXPECT_TRUE(n.isValid());
}

TEST(FormExprTest, Coefficient) {
    auto func = [](Real x, Real y, Real z) { return x + y + z; };
    FormExpr c = FormExpr::coefficient("f", func);

    EXPECT_TRUE(c.isValid());
    EXPECT_FALSE(c.hasTest());
    EXPECT_FALSE(c.hasTrial());
}

TEST(FormExprTest, Gradient) {
    FormExpr u = FormExpr::trialFunction("u");
    FormExpr grad_u = u.grad();

    EXPECT_TRUE(grad_u.isValid());
    EXPECT_TRUE(grad_u.hasTrial());
}

TEST(FormExprTest, GradFreeFunction) {
    FormExpr u = FormExpr::trialFunction("u");
    FormExpr grad_u = grad(u);

    EXPECT_TRUE(grad_u.isValid());
}

TEST(FormExprTest, Divergence) {
    FormExpr u = FormExpr::trialFunction("u");
    FormExpr div_u = u.div();

    EXPECT_TRUE(div_u.isValid());
}

TEST(FormExprTest, Curl) {
    FormExpr u = FormExpr::trialFunction("u");
    FormExpr curl_u = u.curl();

    EXPECT_TRUE(curl_u.isValid());
}

TEST(FormExprTest, Hessian) {
    FormExpr u = FormExpr::trialFunction("u");
    FormExpr H_u = u.hessian();

    EXPECT_TRUE(H_u.isValid());
}

TEST(FormExprTest, Jump) {
    FormExpr u = FormExpr::trialFunction("u");
    FormExpr jump_u = u.jump();

    EXPECT_TRUE(jump_u.isValid());
}

TEST(FormExprTest, Average) {
    FormExpr u = FormExpr::trialFunction("u");
    FormExpr avg_u = u.avg();

    EXPECT_TRUE(avg_u.isValid());
}

TEST(FormExprTest, JumpFreeFunction) {
    FormExpr u = FormExpr::trialFunction("u");
    FormExpr jump_u = jump(u);

    EXPECT_TRUE(jump_u.isValid());
}

TEST(FormExprTest, AverageFreeFunction) {
    FormExpr u = FormExpr::trialFunction("u");
    FormExpr avg_u = avg(u);

    EXPECT_TRUE(avg_u.isValid());
}

TEST(FormExprTest, Negation) {
    FormExpr u = FormExpr::trialFunction("u");
    FormExpr neg_u = -u;

    EXPECT_TRUE(neg_u.isValid());
}

TEST(FormExprTest, Addition) {
    FormExpr u = FormExpr::trialFunction("u");
    FormExpr v = FormExpr::testFunction("v");
    FormExpr sum = u + v;

    EXPECT_TRUE(sum.isValid());
    EXPECT_TRUE(sum.hasTest());
    EXPECT_TRUE(sum.hasTrial());
}

TEST(FormExprTest, Subtraction) {
    FormExpr u = FormExpr::trialFunction("u");
    FormExpr v = FormExpr::testFunction("v");
    FormExpr diff = u - v;

    EXPECT_TRUE(diff.isValid());
}

TEST(FormExprTest, Multiplication) {
    FormExpr u = FormExpr::trialFunction("u");
    FormExpr v = FormExpr::testFunction("v");
    FormExpr prod = u * v;

    EXPECT_TRUE(prod.isValid());
}

TEST(FormExprTest, ScalarMultiplication) {
    FormExpr u = FormExpr::trialFunction("u");
    FormExpr scaled = u * 2.0;

    EXPECT_TRUE(scaled.isValid());

    // Test commutative form
    FormExpr scaled2 = 2.0 * u;
    EXPECT_TRUE(scaled2.isValid());
}

TEST(FormExprTest, InnerProduct) {
    FormExpr u = FormExpr::trialFunction("u");
    FormExpr v = FormExpr::testFunction("v");
    FormExpr inner_prod = u.inner(v);

    EXPECT_TRUE(inner_prod.isValid());
}

TEST(FormExprTest, InnerFreeFunction) {
    FormExpr u = FormExpr::trialFunction("u");
    FormExpr v = FormExpr::testFunction("v");
    FormExpr inner_prod = inner(u, v);

    EXPECT_TRUE(inner_prod.isValid());
}

TEST(FormExprTest, OuterProduct) {
    FormExpr u = FormExpr::trialFunction("u");
    FormExpr v = FormExpr::testFunction("v");
    FormExpr outer_prod = u.outer(v);

    EXPECT_TRUE(outer_prod.isValid());
}

TEST(FormExprTest, CellIntegral) {
    FormExpr u = FormExpr::trialFunction("u");
    FormExpr v = FormExpr::testFunction("v");
    FormExpr form = (u * v).dx();

    EXPECT_TRUE(form.isValid());
    EXPECT_TRUE(form.isBilinear());
}

TEST(FormExprTest, BoundaryIntegral) {
    FormExpr u = FormExpr::trialFunction("u");
    FormExpr v = FormExpr::testFunction("v");
    FormExpr form = (u * v).ds();

    EXPECT_TRUE(form.isValid());
}

TEST(FormExprTest, BoundaryIntegralWithMarker) {
    FormExpr u = FormExpr::trialFunction("u");
    FormExpr v = FormExpr::testFunction("v");
    FormExpr form = (u * v).ds(1);  // Boundary marker 1

    EXPECT_TRUE(form.isValid());
}

TEST(FormExprTest, InteriorFaceIntegral) {
    FormExpr u = FormExpr::trialFunction("u");
    FormExpr v = FormExpr::testFunction("v");
    FormExpr form = (jump(u) * avg(v)).dS();

    EXPECT_TRUE(form.isValid());
}

TEST(FormExprTest, LaplaceForm) {
    // a(u,v) = integral grad(u) . grad(v) dx
    FormExpr u = FormExpr::trialFunction("u");
    FormExpr v = FormExpr::testFunction("v");
    FormExpr a = inner(grad(u), grad(v)).dx();

    EXPECT_TRUE(a.isValid());
    EXPECT_TRUE(a.isBilinear());
}

TEST(FormExprTest, MassForm) {
    // m(u,v) = integral u * v dx
    FormExpr u = FormExpr::trialFunction("u");
    FormExpr v = FormExpr::testFunction("v");
    FormExpr m = (u * v).dx();

    EXPECT_TRUE(m.isValid());
    EXPECT_TRUE(m.isBilinear());
}

TEST(FormExprTest, SourceForm) {
    // L(v) = integral f * v dx
    auto f = [](Real /*x*/, Real /*y*/, Real /*z*/) { return 1.0; };
    FormExpr coeff = FormExpr::coefficient("f", f);
    FormExpr v = FormExpr::testFunction("v");
    FormExpr L = (coeff * v).dx();

    EXPECT_TRUE(L.isValid());
    EXPECT_TRUE(L.isLinear());
}

TEST(FormExprTest, ToString) {
    FormExpr u = FormExpr::trialFunction("u");
    FormExpr v = FormExpr::testFunction("v");
    FormExpr a = inner(grad(u), grad(v)).dx();

    std::string str = a.toString();
    EXPECT_FALSE(str.empty());
}

// ============================================================================
// FormIR Tests
// ============================================================================

TEST(FormIRTest, DefaultConstruction) {
    FormIR ir;

    EXPECT_FALSE(ir.isCompiled());
}

TEST(FormIRTest, MoveConstruction) {
    FormIR ir1;
    FormIR ir2(std::move(ir1));

    // Should not throw
    SUCCEED();
}

TEST(FormIRTest, Queries) {
    FormIR ir;

    // Uncompiled IR should have safe defaults
    EXPECT_FALSE(ir.isBilinear());
    EXPECT_FALSE(ir.isLinear());
    EXPECT_FALSE(ir.hasCellTerms());
    EXPECT_FALSE(ir.hasBoundaryTerms());
    EXPECT_FALSE(ir.hasFaceTerms());
}

// ============================================================================
// FormCompiler Tests
// ============================================================================

TEST(FormCompilerTest, DefaultConstruction) {
    FormCompiler compiler;

    // Should construct successfully
    SUCCEED();
}

TEST(FormCompilerTest, ConstructionWithOptions) {
    SymbolicOptions options;
    options.simplify_expressions = false;

    FormCompiler compiler(options);

    // Should construct successfully
    SUCCEED();
}

TEST(FormCompilerTest, SetOptions) {
    FormCompiler compiler;

    SymbolicOptions options;
    options.verbose = true;

    compiler.setOptions(options);

    // Should not throw
    SUCCEED();
}

TEST(FormCompilerTest, CompileBilinear) {
    FormCompiler compiler;

    FormExpr u = FormExpr::trialFunction("u");
    FormExpr v = FormExpr::testFunction("v");
    FormExpr a = inner(grad(u), grad(v)).dx();

    FormIR ir = compiler.compileBilinear(a);

    // Should compile without error
    SUCCEED();
}

TEST(FormCompilerTest, CompileLinear) {
    FormCompiler compiler;

    FormExpr v = FormExpr::testFunction("v");
    FormExpr c = FormExpr::constant(1.0);
    FormExpr L = (c * v).dx();

    FormIR ir = compiler.compileLinear(L);

    // Should compile without error
    SUCCEED();
}

TEST(FormCompilerTest, CompileGeneric) {
    FormCompiler compiler;

    FormExpr u = FormExpr::trialFunction("u");
    FormExpr v = FormExpr::testFunction("v");
    FormExpr form = (u * v).dx();

    FormIR ir = compiler.compile(form);

    // Should compile without error
    SUCCEED();
}

TEST(FormCompilerTest, GetLastStats) {
    FormCompiler compiler;

    FormExpr u = FormExpr::trialFunction("u");
    FormExpr v = FormExpr::testFunction("v");
    FormExpr a = (u * v).dx();

    compiler.compile(a);

    const auto& stats = compiler.getLastStats();
    EXPECT_GE(stats.compile_seconds, 0.0);
}

// ============================================================================
// SymbolicKernel Tests
// ============================================================================

TEST(SymbolicKernelTest, Construction) {
    FormCompiler compiler;

    FormExpr u = FormExpr::trialFunction("u");
    FormExpr v = FormExpr::testFunction("v");
    FormExpr form = inner(grad(u), grad(v)).dx();

    FormIR ir = compiler.compile(form);
    SymbolicKernel kernel(std::move(ir));

    // Should construct successfully
    SUCCEED();
}

TEST(SymbolicKernelTest, GetRequiredData) {
    FormCompiler compiler;

    FormExpr u = FormExpr::trialFunction("u");
    FormExpr v = FormExpr::testFunction("v");
    FormExpr form = inner(grad(u), grad(v)).dx();

    FormIR ir = compiler.compile(form);
    SymbolicKernel kernel(std::move(ir));

    RequiredData required = kernel.getRequiredData();
    // Should have some required data flags set
    (void)required;
    SUCCEED();
}

TEST(SymbolicKernelTest, HasCell) {
    FormCompiler compiler;

    FormExpr u = FormExpr::trialFunction("u");
    FormExpr v = FormExpr::testFunction("v");
    FormExpr form = (u * v).dx();

    FormIR ir = compiler.compile(form);
    SymbolicKernel kernel(std::move(ir));

    // Cell integral form should have cell terms
    EXPECT_TRUE(kernel.hasCell());
}

TEST(SymbolicKernelTest, HasBoundaryFace) {
    FormCompiler compiler;

    FormExpr u = FormExpr::trialFunction("u");
    FormExpr v = FormExpr::testFunction("v");
    FormExpr form = (u * v).ds();

    FormIR ir = compiler.compile(form);
    SymbolicKernel kernel(std::move(ir));

    // Boundary integral form should have boundary face terms
    EXPECT_TRUE(kernel.hasBoundaryFace());
}

// ============================================================================
// SymbolicAssembler Tests
// ============================================================================

TEST(SymbolicAssemblerTest, DefaultConstruction) {
    SymbolicAssembler assembler;

    EXPECT_FALSE(assembler.isConfigured());
}

TEST(SymbolicAssemblerTest, ConstructionWithOptions) {
    SymbolicOptions options;
    options.ad_mode = ADMode::Forward;

    SymbolicAssembler assembler(options);

    EXPECT_EQ(assembler.getSymbolicOptions().ad_mode, ADMode::Forward);
}

TEST(SymbolicAssemblerTest, SetSymbolicOptions) {
    SymbolicAssembler assembler;

    SymbolicOptions options;
    options.verbose = true;

    assembler.setSymbolicOptions(options);

    EXPECT_TRUE(assembler.getSymbolicOptions().verbose);
}

TEST(SymbolicAssemblerTest, SetAssemblyOptions) {
    SymbolicAssembler assembler;

    AssemblyOptions options;
    options.deterministic = false;

    assembler.setOptions(options);

    EXPECT_FALSE(assembler.getOptions().deterministic);
}

TEST(SymbolicAssemblerTest, Initialize) {
    SymbolicAssembler assembler;

    EXPECT_THROW(assembler.initialize(), std::runtime_error);
}

TEST(SymbolicAssemblerTest, Reset) {
    SymbolicAssembler assembler;
    EXPECT_NO_THROW(assembler.reset());
}

TEST(SymbolicAssemblerTest, ClearCache) {
    SymbolicAssembler assembler;

    assembler.clearCache();

    // Should not throw
    SUCCEED();
}

TEST(SymbolicAssemblerTest, Precompile) {
    SymbolicAssembler assembler;

    FormExpr u = FormExpr::trialFunction("u");
    FormExpr v = FormExpr::testFunction("v");
    FormExpr form = (u * v).dx();

    auto kernel = assembler.precompile(form);

    EXPECT_NE(kernel, nullptr);
}

TEST(SymbolicAssemblerTest, MoveConstruction) {
    SymbolicOptions options;
    options.cache_expressions = false;

    SymbolicAssembler assembler1(options);
    SymbolicAssembler assembler2(std::move(assembler1));

    EXPECT_FALSE(assembler2.getSymbolicOptions().cache_expressions);
}

TEST(SymbolicAssemblerTest, MoveAssignment) {
    SymbolicOptions options;
    options.simplify_expressions = false;

    SymbolicAssembler assembler1(options);
    SymbolicAssembler assembler2;

    assembler2 = std::move(assembler1);

    EXPECT_FALSE(assembler2.getSymbolicOptions().simplify_expressions);
}

// ============================================================================
// Factory Tests
// ============================================================================

TEST(SymbolicAssemblerFactoryTest, CreateDefault) {
    auto assembler = createSymbolicAssembler();

    EXPECT_NE(assembler, nullptr);
}

TEST(SymbolicAssemblerFactoryTest, CreateWithOptions) {
    SymbolicOptions options;
    options.ad_mode = ADMode::Reverse;

    auto assembler = createSymbolicAssembler(options);

    EXPECT_NE(assembler, nullptr);

    auto* sym = dynamic_cast<SymbolicAssembler*>(assembler.get());
    if (sym) {
        EXPECT_EQ(sym->getSymbolicOptions().ad_mode, ADMode::Reverse);
    }
}

// ============================================================================
// Expression Building Tests
// ============================================================================

TEST(FormExprBuildingTest, StokesFlow) {
    // Stokes equations: -mu * lap(u) + grad(p) = f, div(u) = 0
    FormExpr u = FormExpr::trialFunction("u");
    FormExpr p = FormExpr::trialFunction("p");
    FormExpr v = FormExpr::testFunction("v");
    FormExpr q = FormExpr::testFunction("q");

    // Momentum equation bilinear form
    FormExpr mu = FormExpr::constant(1.0);
    FormExpr a_momentum = (mu * inner(grad(u), grad(v))).dx();

    // Pressure gradient coupling
    FormExpr b = (p * div(v)).dx();

    // Continuity equation
    FormExpr c = (div(u) * q).dx();

    EXPECT_TRUE(a_momentum.isValid());
    EXPECT_TRUE(b.isValid());
    EXPECT_TRUE(c.isValid());
}

TEST(FormExprBuildingTest, DGPenalty) {
    // DG penalty term: sum_F eta/h * integral [[u]] [[v]] dS
    FormExpr u = FormExpr::trialFunction("u");
    FormExpr v = FormExpr::testFunction("v");
    FormExpr eta = FormExpr::constant(10.0);

    FormExpr penalty = (eta * inner(jump(u), jump(v))).dS();

    EXPECT_TRUE(penalty.isValid());
}

TEST(FormExprBuildingTest, Nitsche) {
    // Nitsche BC: integral_Gamma (u - g) * (grad v . n) + (grad u . n) * v + eta/h * (u - g) * v ds
    FormExpr u = FormExpr::trialFunction("u");
    FormExpr v = FormExpr::testFunction("v");
    FormExpr n = FormExpr::normal();
    FormExpr eta = FormExpr::constant(100.0);

    FormExpr consistency = (inner(grad(u), n) * v).ds();
    FormExpr adjoint = (u * inner(grad(v), n)).ds();
    FormExpr penalty = (eta * u * v).ds();

    FormExpr nitsche = consistency + adjoint + penalty;

    EXPECT_TRUE(nitsche.isValid());
}

} // namespace test
} // namespace assembly
} // namespace FE
} // namespace svmp

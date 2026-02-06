/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

/**
 * @file test_Complex.cpp
 * @brief Unit tests for FE/Forms complex-valued vocabulary helpers
 */

#include <gtest/gtest.h>

#include "Assembly/AssemblyContext.h"
#include "Assembly/JIT/KernelArgs.h"
#include "Assembly/StandardAssembler.h"
#include "Forms/Complex.h"
#include "Forms/FormCompiler.h"
#include "Forms/FormKernels.h"
#include "Forms/JIT/JITCompiler.h"
#include "Spaces/H1Space.h"
#include "Tests/Unit/Forms/FormsTestHelpers.h"

#include <array>
#include <cmath>
#include <complex>
#include <vector>

namespace svmp {
namespace FE {
namespace forms {
namespace test {

#ifndef SVMP_FE_ENABLE_LLVM_JIT
#define SVMP_FE_ENABLE_LLVM_JIT 0
#endif

namespace {

Real singleTetraVolume()
{
    return 1.0 / 6.0;
}

Real singleTetraP1BasisIntegral()
{
    return singleTetraVolume() / 4.0;
}

assembly::DenseVectorView assembleCellLinear(const FormExpr& scalar_expr,
                                             dofs::DofMap& dof_map,
                                             const assembly::IMeshAccess& mesh,
                                             const spaces::FunctionSpace& space)
{
    FormCompiler compiler;
    const auto v = FormExpr::testFunction(space, "v");
    const auto form = (scalar_expr * v).dx();
    auto ir = compiler.compileLinear(form);
    FormKernel kernel(std::move(ir));

    assembly::StandardAssembler assembler;
    assembler.setDofMap(dof_map);

    assembly::DenseVectorView vec(static_cast<GlobalIndex>(dof_map.getNumDofs()));
    vec.zero();
    (void)assembler.assembleVector(mesh, space, kernel, vec);
    return vec;
}

assembly::DenseMatrixView assembleCellBilinear(const FormExpr& bilinear_form,
                                               dofs::DofMap& dof_map,
                                               const assembly::IMeshAccess& mesh,
                                               const spaces::FunctionSpace& space)
{
    FormCompiler compiler;
    auto ir = compiler.compileBilinear(bilinear_form);
    FormKernel kernel(std::move(ir));

    assembly::StandardAssembler assembler;
    assembler.setDofMap(dof_map);

    assembly::DenseMatrixView mat(static_cast<GlobalIndex>(dof_map.getNumDofs()));
    mat.zero();
    (void)assembler.assembleMatrix(mesh, space, space, kernel, mat);
    return mat;
}

} // namespace

#if SVMP_FE_ENABLE_LLVM_JIT
namespace {

using JITFn = void (*)(const void*);

inline void callJIT(std::uintptr_t addr, const void* args) noexcept
{
    if (addr == 0 || args == nullptr) {
        return;
    }
    reinterpret_cast<JITFn>(addr)(args);
}

static void setupTwoDofTwoQptScalarContext(assembly::AssemblyContext& ctx)
{
    std::vector<assembly::AssemblyContext::Point3D> quad_pts = {{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}};
    std::vector<Real> weights = {0.5, 0.5};
    ctx.setQuadratureData(std::span<const assembly::AssemblyContext::Point3D>(quad_pts.data(), quad_pts.size()),
                          std::span<const Real>(weights.data(), weights.size()));

    const LocalIndex n_dofs = 2;

    std::vector<Real> values = {
        1.0, 0.5, // phi_0 at q=0,1
        0.0, 0.5  // phi_1 at q=0,1
    };

    std::vector<assembly::AssemblyContext::Vector3D> grads = {
        {-1.0, 0.0, 0.0}, {-1.0, 0.0, 0.0},
        { 1.0, 0.0, 0.0}, { 1.0, 0.0, 0.0}
    };

    ctx.setTestBasisData(n_dofs,
                         std::span<const Real>(values.data(), values.size()),
                         std::span<const assembly::AssemblyContext::Vector3D>(grads.data(), grads.size()));
    ctx.setPhysicalGradients(std::span<const assembly::AssemblyContext::Vector3D>(grads.data(), grads.size()),
                             std::span<const assembly::AssemblyContext::Vector3D>(grads.data(), grads.size()));

    std::vector<Real> int_wts = {0.25, 0.25};
    ctx.setIntegrationWeights(std::span<const Real>(int_wts.data(), int_wts.size()));
}

static void expectKernelMatrixNear(const assembly::KernelOutput& A,
                                   const assembly::KernelOutput& B,
                                   Real tol)
{
    ASSERT_TRUE(A.has_matrix);
    ASSERT_TRUE(B.has_matrix);
    ASSERT_FALSE(A.has_vector);
    ASSERT_FALSE(B.has_vector);
    ASSERT_EQ(A.n_test_dofs, B.n_test_dofs);
    ASSERT_EQ(A.n_trial_dofs, B.n_trial_dofs);
    ASSERT_EQ(A.local_matrix.size(), B.local_matrix.size());
    for (std::size_t i = 0; i < A.local_matrix.size(); ++i) {
        EXPECT_NEAR(A.local_matrix[i], B.local_matrix[i], tol);
    }
}

static assembly::KernelOutput evalCellMatrixInterpreter(const FormIR& ir,
                                                        const assembly::AssemblyContext& ctx)
{
    FormKernel kernel(ir.clone());
    assembly::KernelOutput out;
    out.reserve(ctx.numTestDofs(), ctx.numTrialDofs(), /*need_matrix=*/true, /*need_vector=*/false);
    kernel.computeCell(ctx, out);
    return out;
}

static assembly::KernelOutput evalCellMatrixJIT(const FormIR& ir,
                                                const assembly::AssemblyContext& ctx,
                                                const forms::JITOptions& jit_opts)
{
    auto compiler = forms::jit::JITCompiler::getOrCreate(jit_opts);
    const auto r = compiler->compile(ir);
    ASSERT_TRUE(r.ok) << r.message;
    ASSERT_FALSE(r.kernels.empty());

    std::uintptr_t addr = 0;
    for (const auto& k : r.kernels) {
        if (k.domain == IntegralDomain::Cell) {
            addr = k.address;
            break;
        }
    }
    ASSERT_NE(addr, 0u);

    assembly::KernelOutput out;
    out.reserve(ctx.numTestDofs(), ctx.numTrialDofs(), /*need_matrix=*/true, /*need_vector=*/false);
    out.clear();

    assembly::jit::PackingChecks checks;
    checks.validate_alignment = true;
    const auto args = assembly::jit::packCellKernelArgsV6(ctx, out, checks);
    callJIT(addr, &args);
    return out;
}

} // namespace
#endif // SVMP_FE_ENABLE_LLVM_JIT

TEST(ComplexTest, ComplexScalarConstantReal)
{
    const auto z = ComplexScalar::constant(1.25, -2.5);
    ASSERT_TRUE(z.isValid());
    ASSERT_NE(z.re.node(), nullptr);
    ASSERT_NE(z.im.node(), nullptr);

    EXPECT_EQ(z.re.node()->type(), FormExprType::Constant);
    EXPECT_EQ(z.im.node()->type(), FormExprType::Constant);

    ASSERT_TRUE(z.re.node()->constantValue().has_value());
    ASSERT_TRUE(z.im.node()->constantValue().has_value());
    EXPECT_DOUBLE_EQ(*z.re.node()->constantValue(), 1.25);
    EXPECT_DOUBLE_EQ(*z.im.node()->constantValue(), -2.5);
}

TEST(ComplexTest, ComplexScalarConstantStdComplex)
{
    const std::complex<Real> z0{3.0, -4.0};
    const auto z = ComplexScalar::constant(z0);
    ASSERT_TRUE(z.isValid());

    ASSERT_TRUE(z.re.node()->constantValue().has_value());
    ASSERT_TRUE(z.im.node()->constantValue().has_value());
    EXPECT_DOUBLE_EQ(*z.re.node()->constantValue(), 3.0);
    EXPECT_DOUBLE_EQ(*z.im.node()->constantValue(), -4.0);
}

TEST(ComplexTest, ImaginaryUnit)
{
    const auto z = I();
    ASSERT_TRUE(z.isValid());

    ASSERT_TRUE(z.re.node()->constantValue().has_value());
    ASSERT_TRUE(z.im.node()->constantValue().has_value());
    EXPECT_DOUBLE_EQ(*z.re.node()->constantValue(), 0.0);
    EXPECT_DOUBLE_EQ(*z.im.node()->constantValue(), 1.0);
}

TEST(ComplexTest, ComplexConjugate)
{
    SingleTetraMeshAccess mesh;
    auto dof_map = createSingleTetraDofMap();
    spaces::H1Space space(ElementType::Tetra4, 1);

    const auto z = ComplexScalar::constant(2.0, 3.0);
    const auto c = conj(z);

    const Real scale = singleTetraP1BasisIntegral();
    auto re = assembleCellLinear(c.re, dof_map, mesh, space);
    auto im = assembleCellLinear(c.im, dof_map, mesh, space);

    for (GlobalIndex i = 0; i < 4; ++i) {
        EXPECT_NEAR(re.getVectorEntry(i), 2.0 * scale, 1e-12);
        EXPECT_NEAR(im.getVectorEntry(i), -3.0 * scale, 1e-12);
    }
}

TEST(ComplexTest, ComplexAdditionSubtractionNegation)
{
    SingleTetraMeshAccess mesh;
    auto dof_map = createSingleTetraDofMap();
    spaces::H1Space space(ElementType::Tetra4, 1);

    const auto a = ComplexScalar::constant(2.0, 3.0);
    const auto b = ComplexScalar::constant(4.0, -5.0);

    const auto sum = a + b;
    const auto diff = a - b;
    const auto neg_a = -a;

    const Real scale = singleTetraP1BasisIntegral();

    auto sum_re = assembleCellLinear(sum.re, dof_map, mesh, space);
    auto sum_im = assembleCellLinear(sum.im, dof_map, mesh, space);
    auto diff_re = assembleCellLinear(diff.re, dof_map, mesh, space);
    auto diff_im = assembleCellLinear(diff.im, dof_map, mesh, space);
    auto neg_re = assembleCellLinear(neg_a.re, dof_map, mesh, space);
    auto neg_im = assembleCellLinear(neg_a.im, dof_map, mesh, space);

    for (GlobalIndex i = 0; i < 4; ++i) {
        EXPECT_NEAR(sum_re.getVectorEntry(i), (2.0 + 4.0) * scale, 1e-12);
        EXPECT_NEAR(sum_im.getVectorEntry(i), (3.0 - 5.0) * scale, 1e-12);
        EXPECT_NEAR(diff_re.getVectorEntry(i), (2.0 - 4.0) * scale, 1e-12);
        EXPECT_NEAR(diff_im.getVectorEntry(i), (3.0 + 5.0) * scale, 1e-12);
        EXPECT_NEAR(neg_re.getVectorEntry(i), (-2.0) * scale, 1e-12);
        EXPECT_NEAR(neg_im.getVectorEntry(i), (-3.0) * scale, 1e-12);
    }
}

TEST(ComplexTest, ComplexMultiplication)
{
    SingleTetraMeshAccess mesh;
    auto dof_map = createSingleTetraDofMap();
    spaces::H1Space space(ElementType::Tetra4, 1);

    const auto a = ComplexScalar::constant(2.0, 3.0);
    const auto b = ComplexScalar::constant(4.0, -5.0);
    const auto prod = a * b;

    const Real expected_re = 2.0 * 4.0 - 3.0 * (-5.0);
    const Real expected_im = 2.0 * (-5.0) + 3.0 * 4.0;
    const Real scale = singleTetraP1BasisIntegral();

    auto re = assembleCellLinear(prod.re, dof_map, mesh, space);
    auto im = assembleCellLinear(prod.im, dof_map, mesh, space);

    for (GlobalIndex i = 0; i < 4; ++i) {
        EXPECT_NEAR(re.getVectorEntry(i), expected_re * scale, 1e-12);
        EXPECT_NEAR(im.getVectorEntry(i), expected_im * scale, 1e-12);
    }
}

TEST(ComplexTest, ComplexScalarTimesFormExprAndReal)
{
    SingleTetraMeshAccess mesh;
    auto dof_map = createSingleTetraDofMap();
    spaces::H1Space space(ElementType::Tetra4, 1);

    const auto z = ComplexScalar::constant(2.0, 3.0);
    const auto s = FormExpr::constant(4.0);

    const auto left = z * s;
    const auto right = s * z;
    const auto rscale_left = 3.0 * z;
    const auto rscale_right = z * 3.0;

    const Real scale = singleTetraP1BasisIntegral();

    auto left_re = assembleCellLinear(left.re, dof_map, mesh, space);
    auto left_im = assembleCellLinear(left.im, dof_map, mesh, space);
    auto right_re = assembleCellLinear(right.re, dof_map, mesh, space);
    auto right_im = assembleCellLinear(right.im, dof_map, mesh, space);
    auto rsl_re = assembleCellLinear(rscale_left.re, dof_map, mesh, space);
    auto rsl_im = assembleCellLinear(rscale_left.im, dof_map, mesh, space);
    auto rsr_re = assembleCellLinear(rscale_right.re, dof_map, mesh, space);
    auto rsr_im = assembleCellLinear(rscale_right.im, dof_map, mesh, space);

    for (GlobalIndex i = 0; i < 4; ++i) {
        EXPECT_NEAR(left_re.getVectorEntry(i), (2.0 * 4.0) * scale, 1e-12);
        EXPECT_NEAR(left_im.getVectorEntry(i), (3.0 * 4.0) * scale, 1e-12);
        EXPECT_NEAR(right_re.getVectorEntry(i), (2.0 * 4.0) * scale, 1e-12);
        EXPECT_NEAR(right_im.getVectorEntry(i), (3.0 * 4.0) * scale, 1e-12);

        EXPECT_NEAR(rsl_re.getVectorEntry(i), (3.0 * 2.0) * scale, 1e-12);
        EXPECT_NEAR(rsl_im.getVectorEntry(i), (3.0 * 3.0) * scale, 1e-12);
        EXPECT_NEAR(rsr_re.getVectorEntry(i), (3.0 * 2.0) * scale, 1e-12);
        EXPECT_NEAR(rsr_im.getVectorEntry(i), (3.0 * 3.0) * scale, 1e-12);
    }
}

TEST(ComplexTest, ComplexLinearFormValidationAndToRealBlock2x1)
{
    const ComplexLinearForm f_good{FormExpr::constant(1.0), FormExpr::constant(2.0)};
    EXPECT_TRUE(f_good.isValid());

    const ComplexLinearForm f_bad{FormExpr::constant(1.0), FormExpr{}};
    EXPECT_FALSE(f_bad.isValid());

    const auto blocks = toRealBlock2x1(f_good);
    EXPECT_EQ(blocks.numTestFields(), 2u);
    EXPECT_TRUE(blocks.hasBlock(0));
    EXPECT_TRUE(blocks.hasBlock(1));
    EXPECT_EQ(blocks.block(0).toString(), f_good.re.toString());
    EXPECT_EQ(blocks.block(1).toString(), f_good.im.toString());
}

#if SVMP_FE_ENABLE_LLVM_JIT
TEST(ComplexTest, ToRealBlock2x2JITMatchesInterpreter)
{
    assembly::AssemblyContext ctx;
    ctx.reserve(/*max_dofs=*/8, /*max_qpts=*/8, /*dim=*/3);
    setupTwoDofTwoQptScalarContext(ctx);

    FormExprNode::SpaceSignature signature{};
    signature.space_type = spaces::SpaceType::H1;
    signature.field_type = FieldType::Scalar;
    signature.continuity = Continuity::C0;
    signature.value_dimension = 1;
    signature.topological_dimension = 3;
    signature.polynomial_order = 1;
    signature.element_type = ElementType::Tetra4;

    const auto u = FormExpr::trialFunction(signature, "u");
    const auto v = FormExpr::testFunction(signature, "v");

    const auto a_re = (u * v).dx();
    const auto a_im = (FormExpr::constant(2.0) * u * v).dx();
    const ComplexBilinearForm a{a_re, a_im};
    const auto blocks = toRealBlock2x2(a);

    SymbolicOptions sym_opts;
    sym_opts.jit.enable = true;
    FormCompiler form_compiler(sym_opts);

    forms::JITOptions jit_opts;
    jit_opts.enable = true;
    jit_opts.optimization_level = 2;
    jit_opts.cache_kernels = false;
    jit_opts.vectorize = true;

    const std::array<std::pair<int, int>, 4> ij = {{{0, 0}, {0, 1}, {1, 0}, {1, 1}}};
    for (const auto& [i, j] : ij) {
        const auto expr = blocks.block(static_cast<std::size_t>(i), static_cast<std::size_t>(j));
        const auto ir = form_compiler.compileBilinear(expr);

        const auto interp = evalCellMatrixInterpreter(ir, ctx);
        const auto jit = evalCellMatrixJIT(ir, ctx, jit_opts);
        expectKernelMatrixNear(jit, interp, 1e-12);
    }
}
#endif // SVMP_FE_ENABLE_LLVM_JIT

TEST(ComplexTest, ToRealBlock2x2Symmetry)
{
    SingleTetraMeshAccess mesh;
    auto dof_map = createSingleTetraDofMap();
    spaces::H1Space space(ElementType::Tetra4, 1);

    const auto u = FormExpr::trialFunction(space, "u");
    const auto v = FormExpr::testFunction(space, "v");

    const auto a_re = (u * v).dx();
    const auto a_im = (FormExpr::constant(2.0) * u * v).dx();
    const ComplexBilinearForm a{a_re, a_im};
    const auto blocks = toRealBlock2x2(a);

    const auto A_re = assembleCellBilinear(a_re, dof_map, mesh, space);
    const auto A_im = assembleCellBilinear(a_im, dof_map, mesh, space);

    const auto A00 = assembleCellBilinear(blocks.block(0, 0), dof_map, mesh, space);
    const auto A01 = assembleCellBilinear(blocks.block(0, 1), dof_map, mesh, space);
    const auto A10 = assembleCellBilinear(blocks.block(1, 0), dof_map, mesh, space);
    const auto A11 = assembleCellBilinear(blocks.block(1, 1), dof_map, mesh, space);

    for (GlobalIndex r = 0; r < 4; ++r) {
        for (GlobalIndex c = 0; c < 4; ++c) {
            EXPECT_NEAR(A00.getMatrixEntry(r, c), A_re.getMatrixEntry(r, c), 1e-12);
            EXPECT_NEAR(A11.getMatrixEntry(r, c), A_re.getMatrixEntry(r, c), 1e-12);
            EXPECT_NEAR(A10.getMatrixEntry(r, c), A_im.getMatrixEntry(r, c), 1e-12);
            EXPECT_NEAR(A01.getMatrixEntry(r, c), -A_im.getMatrixEntry(r, c), 1e-12);
        }
    }
}

} // namespace test
} // namespace forms
} // namespace FE
} // namespace svmp

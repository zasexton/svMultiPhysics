/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

/**
 * @file test_FormCompiler.cpp
 * @brief Unit tests for FE/Forms compilation (FormExpr -> FormIR)
 */

#include <gtest/gtest.h>

#include "Forms/BlockForm.h"
#include "Forms/FormCompiler.h"
#include "Forms/FormKernels.h"
#include "Spaces/HCurlSpace.h"
#include "Spaces/HDivSpace.h"
#include "Spaces/H1Space.h"
#include "Spaces/L2Space.h"
#include "Spaces/ProductSpace.h"
#include "Systems/FESystem.h"
#include "Tests/Unit/Forms/FormsTestHelpers.h"

#include <memory>
#include <stdexcept>

namespace svmp {
namespace FE {
namespace forms {
namespace test {

TEST(FormCompilerTest, RequiresTopLevelIntegrals)
{
    FormCompiler compiler;
    spaces::H1Space space(ElementType::Tetra4, 1);
    const auto u = FormExpr::trialFunction(space, "u");
    const auto v = FormExpr::testFunction(space, "v");
    const auto not_integrated = u * v;

    EXPECT_THROW((void)compiler.compileBilinear(not_integrated), std::invalid_argument);
}

TEST(FormCompilerTest, CompileLinearRejectsTrial)
{
    FormCompiler compiler;
    spaces::H1Space space(ElementType::Tetra4, 1);
    const auto u = FormExpr::trialFunction(space, "u");
    const auto v = FormExpr::testFunction(space, "v");
    EXPECT_THROW((void)compiler.compileLinear((u * v).dx()), std::invalid_argument);
}

TEST(FormCompilerTest, CompileBilinearRejectsMissingSymbols)
{
    FormCompiler compiler;
    spaces::H1Space space(ElementType::Tetra4, 1);
    const auto v = FormExpr::testFunction(space, "v");
    EXPECT_THROW((void)compiler.compileBilinear(v.dx()), std::invalid_argument);
}

TEST(FormCompilerTest, CompileResidualRequiresTrialAndTest)
{
    FormCompiler compiler;
    spaces::H1Space space(ElementType::Tetra4, 1);
    const auto v = FormExpr::testFunction(space, "v");
    EXPECT_THROW((void)compiler.compileResidual(v.dx()), std::invalid_argument);
}

TEST(FormCompilerTest, CompilesMultipleTermsAndMarkers)
{
    FormCompiler compiler;

    spaces::H1Space space(ElementType::Tetra4, 1);
    const auto u = FormExpr::trialFunction(space, "u");
    const auto v = FormExpr::testFunction(space, "v");

    const auto ir = compiler.compileBilinear((u * v).dx());

    EXPECT_TRUE(ir.isCompiled());
    EXPECT_EQ(ir.kind(), FormKind::Bilinear);
    EXPECT_TRUE(ir.hasCellTerms());
    EXPECT_FALSE(ir.hasBoundaryTerms());
    EXPECT_FALSE(ir.hasInteriorFaceTerms());

    const auto linear_form = v.ds(1) + (FormExpr::constant(2.0) * v).ds(2);
    const auto ir_linear = compiler.compileLinear(linear_form);
    EXPECT_TRUE(ir_linear.isCompiled());
    EXPECT_EQ(ir_linear.kind(), FormKind::Linear);
    EXPECT_TRUE(ir_linear.hasBoundaryTerms());
    EXPECT_FALSE(ir_linear.hasCellTerms());
    EXPECT_FALSE(ir_linear.hasInteriorFaceTerms());
    ASSERT_EQ(ir_linear.terms().size(), 2u);
    EXPECT_EQ(ir_linear.terms()[0].domain, IntegralDomain::Boundary);
    EXPECT_EQ(ir_linear.terms()[1].domain, IntegralDomain::Boundary);
    EXPECT_EQ(ir_linear.terms()[0].boundary_marker, 1);
    EXPECT_EQ(ir_linear.terms()[1].boundary_marker, 2);
}

TEST(FormCompilerTest, TimeDerivativeTermsSplitWithinIntegral)
{
    FormCompiler compiler;
    spaces::H1Space space(ElementType::Tetra4, 1);
    const auto u = FormExpr::trialFunction(space, "u");
    const auto v = FormExpr::testFunction(space, "v");

    const auto form = (dt(u) * v + inner(grad(u), grad(v))).dx();
    const auto ir = compiler.compileBilinear(form);

    EXPECT_TRUE(ir.isCompiled());
    EXPECT_EQ(ir.maxTimeDerivativeOrder(), 1);
    ASSERT_EQ(ir.terms().size(), 2u);
    EXPECT_EQ(ir.terms()[0].domain, IntegralDomain::Cell);
    EXPECT_EQ(ir.terms()[1].domain, IntegralDomain::Cell);
    EXPECT_TRUE((ir.terms()[0].time_derivative_order == 1) ^ (ir.terms()[1].time_derivative_order == 1));
}

TEST(FormCompilerTest, TimeDerivativeOrderExtracted)
{
    FormCompiler compiler;
    spaces::H1Space space(ElementType::Tetra4, 1);
    const auto u = FormExpr::trialFunction(space, "u");
    const auto v = FormExpr::testFunction(space, "v");

    const auto form = (dt(u, 2) * v).dx();
    const auto ir = compiler.compileBilinear(form);
    EXPECT_EQ(ir.maxTimeDerivativeOrder(), 2);
    ASSERT_EQ(ir.terms().size(), 1u);
    EXPECT_EQ(ir.terms()[0].time_derivative_order, 2);
}

TEST(FormCompilerTest, TimeDerivativeRejectsNonTrialOperands)
{
    FormCompiler compiler;
    spaces::H1Space space(ElementType::Tetra4, 1);
    const auto v = FormExpr::testFunction(space, "v");
    EXPECT_THROW((void)compiler.compileLinear(dt(v).dx()), std::invalid_argument);
}

TEST(FormCompilerTest, MultipleDtFactorsRejected)
{
    FormCompiler compiler;
    spaces::H1Space space(ElementType::Tetra4, 1);
    const auto u = FormExpr::trialFunction(space, "u");
    const auto v = FormExpr::testFunction(space, "v");

    EXPECT_THROW((void)compiler.compileBilinear((dt(u) * dt(u) * v).dx()), std::invalid_argument);
}

TEST(FormCompilerTest, MultipleDtOccurrencesInAdditiveContextAllowed)
{
    FormCompiler compiler;
    spaces::H1Space space(ElementType::Tetra4, 1);
    const auto u = FormExpr::trialFunction(space, "u");
    const auto v = FormExpr::testFunction(space, "v");

    // Ensure dt(u) can appear multiple times as long as it is not multiplied by itself.
    // The scalar factor intentionally prevents collectIntegralTerms() from splitting the addends.
    const auto form = (FormExpr::constant(2.0) * (dt(u) * v + dt(u) * v)).dx();
    EXPECT_NO_THROW((void)compiler.compileBilinear(form));

    const auto ir = compiler.compileBilinear(form);
    EXPECT_EQ(ir.maxTimeDerivativeOrder(), 1);
    ASSERT_EQ(ir.terms().size(), 1u);
    EXPECT_EQ(ir.terms()[0].time_derivative_order, 1);
}

TEST(FormCompilerTest, MultipleDtFactorsAllowedInResidual)
{
    FormCompiler compiler;
    spaces::H1Space space(ElementType::Tetra4, 1);
    const auto u = FormExpr::trialFunction(space, "u");
    const auto v = FormExpr::testFunction(space, "v");

    EXPECT_NO_THROW((void)compiler.compileResidual((dt(u) * dt(u) * v).dx()));

    const auto ir = compiler.compileResidual((dt(u) * dt(u) * v).dx());
    EXPECT_TRUE(ir.isCompiled());
    EXPECT_EQ(ir.kind(), FormKind::Residual);
    EXPECT_EQ(ir.maxTimeDerivativeOrder(), 1);
    ASSERT_EQ(ir.terms().size(), 1u);
    EXPECT_EQ(ir.terms()[0].time_derivative_order, 1);
}

TEST(FormCompilerTest, TemporalOrderSignalsThroughKernelAndSystem)
{
    auto mesh = std::make_shared<SingleTetraMeshAccess>();
    auto space = std::make_shared<spaces::H1Space>(ElementType::Tetra4, 1);

    systems::FESystem sys(mesh);
    auto u_field = sys.addField(systems::FieldSpec{.name = "u", .space = space, .components = 1});
    sys.addOperator("op");

    FormCompiler compiler;
    const auto u = FormExpr::trialFunction(*space, "u");
    const auto v = FormExpr::testFunction(*space, "v");
    auto ir = compiler.compileBilinear((dt(u, 2) * v).dx());

    auto kernel = std::make_shared<FormKernel>(std::move(ir));
    EXPECT_EQ(kernel->maxTemporalDerivativeOrder(), 2);

    sys.addCellKernel("op", u_field, u_field, kernel);
    EXPECT_EQ(sys.temporalOrder(), 2);
    EXPECT_TRUE(sys.isTransient());
}

TEST(FormCompilerTest, HessianSetsRequiredDataFlags)
{
    FormCompiler compiler;
    spaces::H1Space space(ElementType::Tetra4, 2);
    const auto u = FormExpr::trialFunction(space, "u");
    const auto v = FormExpr::testFunction(space, "v");

    const auto ir_bilinear = compiler.compileBilinear(inner(u.hessian(), v.hessian()).dx());
    EXPECT_TRUE(assembly::hasFlag(ir_bilinear.requiredData(), assembly::RequiredData::BasisHessians));
    EXPECT_FALSE(assembly::hasFlag(ir_bilinear.requiredData(), assembly::RequiredData::SolutionHessians));

    const auto ir_residual = compiler.compileResidual(inner(u.hessian(), v.hessian()).dx());
    EXPECT_TRUE(assembly::hasFlag(ir_residual.requiredData(), assembly::RequiredData::BasisHessians));
    EXPECT_TRUE(assembly::hasFlag(ir_residual.requiredData(), assembly::RequiredData::SolutionHessians));
}

TEST(FormCompilerTest, ComponentHessianOfVectorTrialSetsRequiredDataFlags)
{
    FormCompiler compiler;
    auto scalar = std::make_shared<spaces::H1Space>(ElementType::Tetra4, 2);
    spaces::ProductSpace vec(scalar, /*components=*/2);
    const auto u = FormExpr::trialFunction(vec, "u");
    const auto v = FormExpr::testFunction(vec, "v");

    const auto expr = inner(component(u, 0).hessian(), component(v, 0).hessian()).dx();

    const auto ir_bilinear = compiler.compileBilinear(expr);
    EXPECT_TRUE(assembly::hasFlag(ir_bilinear.requiredData(), assembly::RequiredData::BasisHessians));
    EXPECT_FALSE(assembly::hasFlag(ir_bilinear.requiredData(), assembly::RequiredData::SolutionHessians));

    const auto ir_residual = compiler.compileResidual(expr);
    EXPECT_TRUE(assembly::hasFlag(ir_residual.requiredData(), assembly::RequiredData::BasisHessians));
    EXPECT_TRUE(assembly::hasFlag(ir_residual.requiredData(), assembly::RequiredData::SolutionHessians));
}

TEST(FormCompilerTest, HCurlCurlRequestsBasisCurlsNotGradients)
{
    FormCompiler compiler;
    spaces::HCurlSpace space(ElementType::Tetra4, /*order=*/0);

    const auto u = FormExpr::trialFunction(space, "u");
    const auto v = FormExpr::testFunction(space, "v");
    const auto expr = inner(curl(u), curl(v)).dx();

    const auto ir_bilinear = compiler.compileBilinear(expr);
    EXPECT_TRUE(assembly::hasFlag(ir_bilinear.requiredData(), assembly::RequiredData::BasisCurls));
    EXPECT_FALSE(assembly::hasFlag(ir_bilinear.requiredData(), assembly::RequiredData::PhysicalGradients));
    EXPECT_FALSE(assembly::hasFlag(ir_bilinear.requiredData(), assembly::RequiredData::SolutionGradients));

    const auto ir_residual = compiler.compileResidual(expr);
    EXPECT_TRUE(assembly::hasFlag(ir_residual.requiredData(), assembly::RequiredData::BasisCurls));
    EXPECT_TRUE(assembly::hasFlag(ir_residual.requiredData(), assembly::RequiredData::SolutionValues));
    EXPECT_FALSE(assembly::hasFlag(ir_residual.requiredData(), assembly::RequiredData::SolutionGradients));
}

TEST(FormCompilerTest, HDivDivRequestsBasisDivergencesNotGradients)
{
    FormCompiler compiler;
    spaces::HDivSpace space(ElementType::Tetra4, /*order=*/0);

    const auto u = FormExpr::trialFunction(space, "u");
    const auto v = FormExpr::testFunction(space, "v");
    const auto expr = (div(u) * div(v)).dx();

    const auto ir_bilinear = compiler.compileBilinear(expr);
    EXPECT_TRUE(assembly::hasFlag(ir_bilinear.requiredData(), assembly::RequiredData::BasisDivergences));
    EXPECT_FALSE(assembly::hasFlag(ir_bilinear.requiredData(), assembly::RequiredData::PhysicalGradients));
    EXPECT_FALSE(assembly::hasFlag(ir_bilinear.requiredData(), assembly::RequiredData::SolutionGradients));

    const auto ir_residual = compiler.compileResidual(expr);
    EXPECT_TRUE(assembly::hasFlag(ir_residual.requiredData(), assembly::RequiredData::BasisDivergences));
    EXPECT_TRUE(assembly::hasFlag(ir_residual.requiredData(), assembly::RequiredData::SolutionValues));
    EXPECT_FALSE(assembly::hasFlag(ir_residual.requiredData(), assembly::RequiredData::SolutionGradients));
}

TEST(FormCompilerTest, BlockLinearCompilationSkipsEmptyBlocks)
{
    FormCompiler compiler;
    spaces::H1Space space(ElementType::Tetra4, 1);

    BlockLinearForm L(2);
    L.setBlock(0, FormExpr::testFunction(space, "v").dx());
    // Leave block 1 unset.

    auto blocks = compiler.compileLinear(L);
    ASSERT_EQ(blocks.size(), 2u);
    ASSERT_TRUE(blocks[0].has_value());
    EXPECT_EQ(blocks[0]->kind(), FormKind::Linear);
    EXPECT_FALSE(blocks[1].has_value());
}

TEST(FormCompilerTest, BlockBilinearAndResidualCompileEachBlockIndependently)
{
    FormCompiler compiler;

    spaces::H1Space V(ElementType::Tetra4, 1);
    spaces::L2Space Q(ElementType::Tetra4, 1);

    const auto u = FormExpr::trialFunction(V, "u");
    const auto v = FormExpr::testFunction(V, "v");
    const auto p = FormExpr::trialFunction(Q, "p");
    const auto q = FormExpr::testFunction(Q, "q");

    BlockBilinearForm A(2, 2);
    A.setBlock(0, 0, (u * v).dx());
    A.setBlock(0, 1, (p * v).dx());
    A.setBlock(1, 0, (u * q).dx());
    A.setBlock(1, 1, (p * q).dx());

    {
        auto blocks = compiler.compileBilinear(A);
        ASSERT_EQ(blocks.size(), 2u);
        ASSERT_EQ(blocks[0].size(), 2u);
        ASSERT_EQ(blocks[1].size(), 2u);
        for (std::size_t i = 0; i < 2; ++i) {
            for (std::size_t j = 0; j < 2; ++j) {
                ASSERT_TRUE(blocks[i][j].has_value());
                EXPECT_EQ(blocks[i][j]->kind(), FormKind::Bilinear);
            }
        }
    }

    {
        auto blocks = compiler.compileResidual(A);
        ASSERT_EQ(blocks.size(), 2u);
        ASSERT_EQ(blocks[0].size(), 2u);
        ASSERT_EQ(blocks[1].size(), 2u);
        for (std::size_t i = 0; i < 2; ++i) {
            for (std::size_t j = 0; j < 2; ++j) {
                ASSERT_TRUE(blocks[i][j].has_value());
                EXPECT_EQ(blocks[i][j]->kind(), FormKind::Residual);
            }
        }
    }
}

} // namespace test
} // namespace forms
} // namespace FE
} // namespace svmp

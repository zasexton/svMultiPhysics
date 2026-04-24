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
#include "Forms/Index.h"
#include "Spaces/HCurlSpace.h"
#include "Spaces/HDivSpace.h"
#include "Spaces/H1Space.h"
#include "Spaces/L2Space.h"
#include "Spaces/ProductSpace.h"
#include "Systems/FESystem.h"
#include "Tests/Unit/Forms/FormsTestHelpers.h"

#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace svmp {
namespace FE {
namespace forms {
namespace test {

TEST(FormCompilerTest, ConstructorAndSetOptionsPreserveSymbolicTangentOptions)
{
    SymbolicOptions opts;
    opts.ad_mode = ADMode::Forward;
    opts.use_symbolic_tangent = true;
    opts.geometry_sensitivity.mode = GeometrySensitivityMode::MeshMotionUnknowns;
    opts.geometry_sensitivity.mesh_motion_field = 3;
    opts.simplify_expressions = false;
    opts.exploit_sparsity = false;
    opts.cache_expressions = false;
    opts.verbose = true;
    opts.jit.enable = true;
    opts.jit.optimization_level = 3;
    opts.jit.vectorize = false;
    opts.jit.tensor.mode = TensorLoweringMode::Auto;

    FormCompiler compiler(opts);
    EXPECT_EQ(compiler.options().ad_mode, ADMode::Forward);
    EXPECT_TRUE(compiler.options().use_symbolic_tangent);
    EXPECT_EQ(compiler.options().geometry_sensitivity.mode,
              GeometrySensitivityMode::MeshMotionUnknowns);
    EXPECT_EQ(compiler.options().geometry_sensitivity.mesh_motion_field, 3);
    EXPECT_FALSE(compiler.options().simplify_expressions);
    EXPECT_FALSE(compiler.options().exploit_sparsity);
    EXPECT_FALSE(compiler.options().cache_expressions);
    EXPECT_TRUE(compiler.options().verbose);
    EXPECT_TRUE(compiler.options().jit.enable);
    EXPECT_EQ(compiler.options().jit.optimization_level, 3);
    EXPECT_FALSE(compiler.options().jit.vectorize);
    EXPECT_EQ(compiler.options().jit.tensor.mode, TensorLoweringMode::Auto);

    spaces::H1Space space(ElementType::Tetra4, 1);
    const auto u = FormExpr::trialFunction(space, "u");
    const auto v = FormExpr::testFunction(space, "v");
    const auto ir = compiler.compileResidual((u * u * v).dx());
    EXPECT_TRUE(ir.isCompiled());
    EXPECT_EQ(ir.kind(), FormKind::Residual);
    EXPECT_TRUE(compiler.options().use_symbolic_tangent);
    EXPECT_EQ(compiler.options().geometry_sensitivity.mode,
              GeometrySensitivityMode::MeshMotionUnknowns);

    SymbolicOptions reset;
    reset.use_symbolic_tangent = false;
    reset.jit.enable = false;
    compiler.setOptions(reset);
    EXPECT_FALSE(compiler.options().use_symbolic_tangent);
    EXPECT_EQ(compiler.options().geometry_sensitivity.mode,
              GeometrySensitivityMode::GeometryConstant);
    EXPECT_FALSE(compiler.options().jit.enable);
}

TEST(FormCompilerTest, DefaultAndMoveConstructorsPreserveOptionsAndUsability)
{
    FormCompiler default_compiler;
    const SymbolicOptions defaults;
    EXPECT_EQ(default_compiler.options().ad_mode, defaults.ad_mode);
    EXPECT_EQ(default_compiler.options().use_symbolic_tangent, defaults.use_symbolic_tangent);
    EXPECT_EQ(default_compiler.options().geometry_sensitivity.mode,
              defaults.geometry_sensitivity.mode);
    EXPECT_EQ(default_compiler.options().geometry_sensitivity.mesh_motion_field,
              defaults.geometry_sensitivity.mesh_motion_field);
    EXPECT_EQ(default_compiler.options().jit.enable, defaults.jit.enable);
    EXPECT_EQ(default_compiler.options().jit.optimization_level, defaults.jit.optimization_level);
    EXPECT_EQ(default_compiler.options().jit.tensor.mode, defaults.jit.tensor.mode);

    SymbolicOptions opts;
    opts.ad_mode = ADMode::Reverse;
    opts.use_symbolic_tangent = true;
    opts.geometry_sensitivity.mode = GeometrySensitivityMode::MeshMotionUnknowns;
    opts.geometry_sensitivity.mesh_motion_field = 4;
    opts.jit.enable = true;
    opts.jit.optimization_level = 1;
    opts.jit.simd_batch = false;

    FormCompiler configured(opts);
    FormCompiler moved_constructed(std::move(configured));
    EXPECT_EQ(moved_constructed.options().ad_mode, ADMode::Reverse);
    EXPECT_TRUE(moved_constructed.options().use_symbolic_tangent);
    EXPECT_EQ(moved_constructed.options().geometry_sensitivity.mode,
              GeometrySensitivityMode::MeshMotionUnknowns);
    EXPECT_EQ(moved_constructed.options().geometry_sensitivity.mesh_motion_field, 4);
    EXPECT_TRUE(moved_constructed.options().jit.enable);
    EXPECT_EQ(moved_constructed.options().jit.optimization_level, 1);
    EXPECT_FALSE(moved_constructed.options().jit.simd_batch);

    FormCompiler moved_assigned;
    moved_assigned = std::move(moved_constructed);
    EXPECT_EQ(moved_assigned.options().ad_mode, ADMode::Reverse);
    EXPECT_TRUE(moved_assigned.options().use_symbolic_tangent);
    EXPECT_EQ(moved_assigned.options().geometry_sensitivity.mesh_motion_field, 4);

    spaces::H1Space space(ElementType::Tetra4, 1);
    const auto u = FormExpr::trialFunction(space, "u");
    const auto v = FormExpr::testFunction(space, "v");
    const auto ir = moved_assigned.compileBilinear((u * v).dx());
    EXPECT_TRUE(ir.isCompiled());
    EXPECT_EQ(ir.kind(), FormKind::Bilinear);
}

TEST(FormCompilerTest, GeometrySensitivityModeFailsClosedForGeometryTerminals)
{
    SymbolicOptions opts;
    opts.geometry_sensitivity.mode = GeometrySensitivityMode::MeshMotionUnknowns;
    opts.geometry_sensitivity.mesh_motion_field = 2;

    FormCompiler compiler(opts);
    spaces::H1Space space(ElementType::Tetra4, 1);
    const auto u = FormExpr::trialFunction(space, "u");
    const auto v = FormExpr::testFunction(space, "v");

    EXPECT_THROW(
        (void)compiler.compileResidual(
            (component(FormExpr::currentCoordinate(), 0) * u * v).dx()),
        std::invalid_argument);

    const auto ir = compiler.compileResidual((u * v).dx());
    EXPECT_TRUE(ir.isCompiled());
}

TEST(FormCompilerTest, GeometrySensitivityModeRequiresMeshMotionField)
{
    SymbolicOptions opts;
    opts.geometry_sensitivity.mode = GeometrySensitivityMode::MeshMotionUnknowns;

    FormCompiler compiler(opts);
    spaces::H1Space space(ElementType::Tetra4, 1);
    const auto u = FormExpr::trialFunction(space, "u");
    const auto v = FormExpr::testFunction(space, "v");

    EXPECT_THROW((void)compiler.compileResidual((u * v).dx()), std::invalid_argument);
}

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

TEST(FormCompilerTest, CompileImplRejectsUnresolvedCoupledAndAuxiliaryPlaceholders)
{
    FormCompiler compiler;
    spaces::H1Space space(ElementType::Tetra4, 1);
    const auto u = FormExpr::trialFunction(space, "u");
    const auto v = FormExpr::testFunction(space, "v");

    const std::vector<FormExpr> placeholders = {
        FormExpr::auxiliaryState("aux_state"),
        FormExpr::auxiliaryInput("aux_input"),
        FormExpr::auxiliaryOutput("aux_output"),
        FormExpr::boundaryIntegral(v, 2, "wall_reduction"),
    };

    for (const auto& placeholder : placeholders) {
        try {
            (void)compiler.compileBilinear((u * placeholder * v).dx());
            FAIL() << "Expected unresolved placeholder rejection for " << placeholder.toString();
        } catch (const std::invalid_argument& e) {
            const std::string msg = e.what();
            EXPECT_NE(msg.find("unresolved placeholder terminal"), std::string::npos);
            EXPECT_NE(msg.find(placeholder.toString()), std::string::npos);
        }
    }
}

TEST(FormCompilerTest, CompileImplRejectsUnloweredIndexedAccessWhenJITDisabled)
{
    spaces::H1Space space(ElementType::Tetra4, 1);
    const auto u = FormExpr::trialFunction(space, "u");
    const auto v = FormExpr::testFunction(space, "v");
    const Index i("i");

    SymbolicOptions no_jit;
    no_jit.jit.enable = false;
    FormCompiler interpreter_compiler(no_jit);
    EXPECT_THROW((void)interpreter_compiler.compileBilinear((grad(u)(i) * grad(v)(i)).dx()),
                 std::invalid_argument);

    SymbolicOptions jit_enabled;
    jit_enabled.jit.enable = true;
    FormCompiler jit_compiler(jit_enabled);
    EXPECT_NO_THROW((void)jit_compiler.compileBilinear((grad(u)(i) * grad(v)(i)).dx()));
}

TEST(FormCompilerTest, CompileImplMarksFaceGeometryAndNeighborRequirements)
{
    FormCompiler compiler;
    spaces::H1Space space(ElementType::Tetra4, 1);
    const auto u = FormExpr::trialFunction(space, "u");
    const auto v = FormExpr::testFunction(space, "v");

    const auto form = (jump(u) * jump(v)).dS() + (minus(u) * minus(v)).dI(7);
    const auto ir = compiler.compileBilinear(form);

    EXPECT_TRUE(ir.isCompiled());
    EXPECT_TRUE(ir.hasInteriorFaceTerms());
    EXPECT_TRUE(ir.hasInterfaceFaceTerms());
    EXPECT_TRUE(assembly::hasFlag(ir.requiredData(), assembly::RequiredData::Normals));
    EXPECT_TRUE(assembly::hasFlag(ir.requiredData(), assembly::RequiredData::NeighborData));
    EXPECT_TRUE(assembly::hasFlag(ir.requiredData(), assembly::RequiredData::FaceOrientations));

    ASSERT_EQ(ir.terms().size(), 2u);
    bool saw_interior = false;
    bool saw_interface = false;
    for (const auto& term : ir.terms()) {
        if (term.domain == IntegralDomain::InteriorFace) {
            saw_interior = true;
            EXPECT_EQ(term.interface_marker, -1);
            EXPECT_TRUE(assembly::hasFlag(term.required_data, assembly::RequiredData::NeighborData));
        }
        if (term.domain == IntegralDomain::InterfaceFace) {
            saw_interface = true;
            EXPECT_EQ(term.interface_marker, 7);
            EXPECT_TRUE(assembly::hasFlag(term.required_data, assembly::RequiredData::FaceOrientations));
        }
    }
    EXPECT_TRUE(saw_interior);
    EXPECT_TRUE(saw_interface);
}

TEST(FormCompilerTest, CompileImplPopulatesDumpAndSortedFieldRequirements)
{
    FormCompiler compiler;
    spaces::H1Space space(ElementType::Tetra4, 1);
    const auto u = FormExpr::trialFunction(space, "u");
    const auto v = FormExpr::testFunction(space, "v");

    constexpr FieldId low_field = 3;
    constexpr FieldId high_field = 9;
    const auto low = FormExpr::discreteField(low_field, space, "low");
    const auto high = FormExpr::stateField(high_field, space, "high");

    const auto residual = (low * u * v + inner(grad(high), grad(v))).dx();
    const auto ir = compiler.compileResidual(residual);

    EXPECT_TRUE(ir.isCompiled());
    EXPECT_EQ(ir.kind(), FormKind::Residual);
    EXPECT_EQ(ir.maxTimeDerivativeOrder(), 0);

    const auto dump = ir.dump();
    EXPECT_NE(dump.find("kind: residual"), std::string::npos);
    EXPECT_NE(dump.find("terms:"), std::string::npos);
    EXPECT_NE(dump.find("dx"), std::string::npos);

    const auto& reqs = ir.fieldRequirements();
    ASSERT_EQ(reqs.size(), 2u);
    EXPECT_EQ(reqs[0].field, low_field);
    EXPECT_TRUE(assembly::hasFlag(reqs[0].required, assembly::RequiredData::SolutionValues));
    EXPECT_FALSE(assembly::hasFlag(reqs[0].required, assembly::RequiredData::SolutionGradients));

    EXPECT_EQ(reqs[1].field, high_field);
    EXPECT_TRUE(assembly::hasFlag(reqs[1].required, assembly::RequiredData::SolutionValues));
    EXPECT_TRUE(assembly::hasFlag(reqs[1].required, assembly::RequiredData::SolutionGradients));
}

TEST(FormCompilerTest, CompileResidualPreservesMultiDomainMetadataAndRequirements)
{
    FormCompiler compiler;
    spaces::H1Space space(ElementType::Tetra4, 1);
    const auto u = FormExpr::trialFunction(space, "u");
    const auto v = FormExpr::testFunction(space, "v");

    const auto residual =
        (u * u * v).dx() +
        (u * v).ds(4) +
        (minus(u) * minus(v)).dI(8);
    const auto ir = compiler.compileResidual(residual);

    EXPECT_TRUE(ir.isCompiled());
    EXPECT_EQ(ir.kind(), FormKind::Residual);
    EXPECT_TRUE(ir.hasCellTerms());
    EXPECT_TRUE(ir.hasBoundaryTerms());
    EXPECT_TRUE(ir.hasInterfaceFaceTerms());
    EXPECT_TRUE(assembly::hasFlag(ir.requiredData(), assembly::RequiredData::SolutionValues));
    EXPECT_TRUE(assembly::hasFlag(ir.requiredData(), assembly::RequiredData::Normals));
    EXPECT_TRUE(assembly::hasFlag(ir.requiredData(), assembly::RequiredData::NeighborData));
    EXPECT_TRUE(assembly::hasFlag(ir.requiredData(), assembly::RequiredData::FaceOrientations));

    ASSERT_EQ(ir.terms().size(), 3u);
    bool saw_cell = false;
    bool saw_boundary = false;
    bool saw_interface = false;
    for (const auto& term : ir.terms()) {
        if (term.domain == IntegralDomain::Cell) {
            saw_cell = true;
            EXPECT_EQ(term.boundary_marker, -1);
            EXPECT_EQ(term.interface_marker, -1);
        } else if (term.domain == IntegralDomain::Boundary) {
            saw_boundary = true;
            EXPECT_EQ(term.boundary_marker, 4);
            EXPECT_TRUE(assembly::hasFlag(term.required_data, assembly::RequiredData::Normals));
        } else if (term.domain == IntegralDomain::InterfaceFace) {
            saw_interface = true;
            EXPECT_EQ(term.interface_marker, 8);
            EXPECT_TRUE(assembly::hasFlag(term.required_data, assembly::RequiredData::NeighborData));
            EXPECT_TRUE(assembly::hasFlag(term.required_data, assembly::RequiredData::FaceOrientations));
        }
    }
    EXPECT_TRUE(saw_cell);
    EXPECT_TRUE(saw_boundary);
    EXPECT_TRUE(saw_interface);
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

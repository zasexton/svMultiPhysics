/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include <gtest/gtest.h>

#include "Assembly/CutIntegrationContext.h"
#include "Assembly/GlobalSystemView.h"
#include "Assembly/StandardAssembler.h"
#include "Assembly/TimeIntegrationContext.h"
#include "Dofs/DofMap.h"
#include "Forms/FormCompiler.h"
#include "Forms/FormKernels.h"
#include "Forms/JIT/JITCompiler.h"
#include "Forms/JIT/JITKernelWrapper.h"
#include "Forms/JIT/JITValidation.h"
#include "Forms/Vocabulary.h"
#include "Geometry/CutQuadrature.h"
#include "Spaces/H1Space.h"
#include "Spaces/ProductSpace.h"
#include "Tests/Unit/Forms/FormsTestHelpers.h"
#include "Tests/Unit/Forms/JITTestHelpers.h"

#include <cmath>
#include <memory>
#include <utility>
#include <vector>

namespace svmp {
namespace FE {
namespace forms {
namespace test {

namespace {

struct StokesVMSParams {
    Real rho{1.0};
    Real mu{0.01};
    Real ct_m{1.0};
    Real ct_c{36.0};
    Real eps{1e-12};
};

FormExpr makeStokesVMSMomentumForm(const spaces::FunctionSpace& velocity_space,
                                  const FormExpr& u,
                                  const FormExpr& p,
                                  int dim,
                                  const StokesVMSParams& params,
                                  int cut_volume_marker = -1)
{
    using namespace svmp::FE::forms;

    const auto v = TestFunction(velocity_space, "v");

    const auto rho = FormExpr::constant(params.rho);
    const auto mu = FormExpr::constant(params.mu);

    // Stokes: convection disabled (a = 0), body force f = 0.
    std::vector<FormExpr> zero_vec;
    zero_vec.reserve(static_cast<std::size_t>(dim));
    for (int d = 0; d < dim; ++d) {
        zero_vec.push_back(FormExpr::constant(0.0));
    }
    const auto a = FormExpr::asVector(zero_vec);
    const auto f = FormExpr::asVector(std::move(zero_vec));

    const auto stress = FormExpr::constant(2.0) * mu * sym(grad(u));
    const auto r_m = rho * (dt(u) + grad(u) * a - f) + grad(p) - div(stress);

    // Galerkin terms.
    const auto inertia = rho * inner(dt(u), v);
    const auto viscous = FormExpr::constant(2.0) * mu * inner(sym(grad(u)), sym(grad(v)));
    const auto forcing = -rho * inner(f, v);

    // Residual-based VMS static subscales.
    const auto eps = FormExpr::constant(params.eps);
    const auto dt_step = FormExpr::effectiveTimeStep();
    const auto ct_m = FormExpr::constant(params.ct_m);
    const auto ct_c = FormExpr::constant(params.ct_c);

    const auto Jinv_expr = Jinv();
    const auto K = transpose(Jinv_expr) * Jinv_expr;
    const auto nu = mu / rho;

    const auto kT = FormExpr::constant(4.0) * (ct_m * ct_m) / (dt_step * dt_step);
    const auto kU = inner(a, K * a);
    const auto kS = ct_c * doubleContraction(K, K) * (nu * nu);
    const auto tau_m = FormExpr::constant(1.0) / (rho * sqrt(kT + kU + kS + eps));
    const auto tau_c = FormExpr::constant(1.0) / (tau_m * trace(K) + eps);

    const auto u_sub = -tau_m * r_m;
    const auto p_sub = -tau_c * div(u);

    const auto p_adv = p + p_sub;
    const auto pressure_adv = -p_adv * div(v);

    // Convection-related terms are zero for Stokes (a = 0).
    const auto convection_adv = rho * inner(grad(u) * a, v);
    const auto supg = -rho * inner(grad(v) * a, u_sub);

    const auto integrand =
        inertia + convection_adv + viscous + pressure_adv + forcing + supg;
    if (cut_volume_marker >= 0) {
        return integrand.dCutVolume(cut_volume_marker,
                                    CutVolumeSide::Negative);
    }
    return integrand.dx();
}

FormExpr makeStokesVMSContinuityForm(const spaces::FunctionSpace& velocity_space,
                                     const spaces::FunctionSpace& pressure_space,
                                     const FormExpr& u,
                                     const FormExpr& p,
                                     int dim,
                                     const StokesVMSParams& params,
                                     int cut_volume_marker = -1)
{
    using namespace svmp::FE::forms;
    (void)velocity_space;

    const auto q = TestFunction(pressure_space, "q");

    const auto rho = FormExpr::constant(params.rho);
    const auto mu = FormExpr::constant(params.mu);

    // Stokes: convection disabled (a = 0), body force f = 0.
    std::vector<FormExpr> zero_vec;
    zero_vec.reserve(static_cast<std::size_t>(dim));
    for (int d = 0; d < dim; ++d) {
        zero_vec.push_back(FormExpr::constant(0.0));
    }
    const auto a = FormExpr::asVector(zero_vec);
    const auto f = FormExpr::asVector(std::move(zero_vec));

    const auto stress = FormExpr::constant(2.0) * mu * sym(grad(u));
    const auto r_m = rho * (dt(u) + grad(u) * a - f) + grad(p) - div(stress);

    // Residual-based VMS static subscales.
    const auto eps = FormExpr::constant(params.eps);
    const auto dt_step = FormExpr::effectiveTimeStep();
    const auto ct_m = FormExpr::constant(params.ct_m);
    const auto ct_c = FormExpr::constant(params.ct_c);

    const auto Jinv_expr = Jinv();
    const auto K = transpose(Jinv_expr) * Jinv_expr;
    const auto nu = mu / rho;

    const auto kT = FormExpr::constant(4.0) * (ct_m * ct_m) / (dt_step * dt_step);
    const auto kU = inner(a, K * a);
    const auto kS = ct_c * K.doubleContraction(K) * (nu * nu);
    const auto tau_m = FormExpr::constant(1.0) / (rho * sqrt(kT + kU + kS + eps));

    const auto u_sub = -tau_m * r_m;

    const auto integrand = q * div(u) - inner(grad(q), u_sub);
    if (cut_volume_marker >= 0) {
        return integrand.dCutVolume(cut_volume_marker,
                                    CutVolumeSide::Negative);
    }
    return integrand.dx();
}

void expectDenseNear(const assembly::DenseMatrixView& A,
                     const assembly::DenseMatrixView& B,
                     Real tol)
{
    ASSERT_EQ(A.numRows(), B.numRows());
    ASSERT_EQ(A.numCols(), B.numCols());
    for (GlobalIndex i = 0; i < A.numRows(); ++i) {
        for (GlobalIndex j = 0; j < A.numCols(); ++j) {
            EXPECT_NEAR(A.getMatrixEntry(i, j), B.getMatrixEntry(i, j), tol);
        }
    }
}

void expectDenseNear(const assembly::DenseVectorView& a,
                     const assembly::DenseVectorView& b,
                     Real tol)
{
    ASSERT_EQ(a.numRows(), b.numRows());
    for (GlobalIndex i = 0; i < a.numRows(); ++i) {
        EXPECT_NEAR(a.getVectorEntry(i), b.getVectorEntry(i), tol);
    }
}

} // namespace

TEST(JITCompilerStokesVMS, TangentCellCompilesForMomentumBlocks)
{
    requireLLVMJITOrSkip();

    constexpr int dim = 2;
    const auto base = std::make_shared<spaces::H1Space>(ElementType::Quad4, /*order=*/1);
    const auto vel_space = std::make_shared<spaces::ProductSpace>(base, dim);
    const auto p_space = std::make_shared<spaces::H1Space>(ElementType::Quad4, /*order=*/1);

    const FieldId u_id = static_cast<FieldId>(0);
    const FieldId p_id = static_cast<FieldId>(1);

    SymbolicOptions compiler_opts;
    compiler_opts.jit.enable = true;
    FormCompiler compiler(compiler_opts);

    const auto jit_opts = makeUnitTestJITOptions();
    auto jit_compiler = jit::JITCompiler::getOrCreate(jit_opts);
    ASSERT_TRUE(jit_compiler);

    jit::ValidationOptions vopt;
    vopt.strictness = jit::Strictness::AllowExternalCalls;

    const StokesVMSParams params{};

    // (1) Momentum Jacobian block dR_u/du: u is TrialFunction, p is DiscreteField.
    {
        const auto u = TrialFunction(*vel_space, "u");
        const auto p = FormExpr::discreteField(p_id, *p_space, "p");
        const auto form = makeStokesVMSMomentumForm(*vel_space, u, p, dim, params);
        auto ir = compiler.compileResidual(form);
        SymbolicNonlinearFormKernel k(std::move(ir), NonlinearKernelOutput::MatrixOnly);
        k.resolveInlinableConstitutives();

        const auto r = jit_compiler->compile(k.tangentIR(), vopt);
        EXPECT_TRUE(r.ok) << r.message;
    }

    // (2) Momentum Jacobian block dR_u/dp: p is TrialFunction, u is DiscreteField.
    {
        const auto u = FormExpr::discreteField(u_id, *vel_space, "u");
        const auto p = TrialFunction(*p_space, "p");
        const auto form = makeStokesVMSMomentumForm(*vel_space, u, p, dim, params);
        auto ir = compiler.compileResidual(form);
        SymbolicNonlinearFormKernel k(std::move(ir), NonlinearKernelOutput::MatrixOnly);
        k.resolveInlinableConstitutives();

        const auto r = jit_compiler->compile(k.tangentIR(), vopt);
        EXPECT_TRUE(r.ok) << r.message;
    }
}

TEST(JITCompilerStokesVMS, Triangle3_CellAssembly_JITMatchesInterpreter_ForAllBlocks)
{
    requireLLVMJITOrSkip();

    constexpr int dim = 2;
    const StokesVMSParams params{};

    SingleTriangleMeshAccess mesh;

    auto base = std::make_shared<spaces::H1Space>(ElementType::Triangle3, /*order=*/1);
    spaces::ProductSpace vel_space(base, dim);
    spaces::H1Space p_space(ElementType::Triangle3, /*order=*/1);

    // Primary DofMaps for block assembly use local indexing (0..n-1).
    dofs::DofMap dof_map_u(/*n_cells=*/1, /*n_dofs_total=*/6, /*dofs_per_cell=*/6);
    const std::array<GlobalIndex, 6> u_dofs = {0, 1, 2, 3, 4, 5};
    dof_map_u.setCellDofs(0, u_dofs);
    dof_map_u.setNumDofs(6);
    dof_map_u.setNumLocalDofs(6);
    dof_map_u.finalize();

    auto dof_map_p = createSingleTriangleDofMap(); // [0,1,2]

    // Time integration: simple first-derivative stencil so dt(u) and effectiveTimeStep() are defined.
    constexpr Real dt = Real(0.05);
    assembly::TimeIntegrationContext ti;
    ti.integrator_name = "unit";
    ti.dt1 = assembly::TimeDerivativeStencil{.order = 1, .a = {Real(1.0) / dt, -Real(1.0) / dt}};

    // Solution layouts:
    //  - u-primary: [u(6) | p(3)]
    //  - p-primary: [p(3) | u(6)]
    const std::vector<Real> u_coeffs = {0.1, -0.2, 0.3, -0.1, 0.05, -0.03};
    const std::vector<Real> p_coeffs = {0.2, -0.1, 0.05};

    std::vector<Real> sol_u_primary(9, 0.0);
    std::copy(u_coeffs.begin(), u_coeffs.end(), sol_u_primary.begin());
    std::copy(p_coeffs.begin(), p_coeffs.end(), sol_u_primary.begin() + 6);

    std::vector<Real> sol_p_primary(9, 0.0);
    std::copy(p_coeffs.begin(), p_coeffs.end(), sol_p_primary.begin());
    std::copy(u_coeffs.begin(), u_coeffs.end(), sol_p_primary.begin() + 3);

    std::vector<Real> prev_u_primary(9, 0.0);
    std::vector<Real> prev_p_primary(9, 0.0);

    const FieldId u_id = static_cast<FieldId>(0);
    const FieldId p_id = static_cast<FieldId>(1);

    FormCompiler compiler;

    const forms::JITOptions jit_opts = [] {
        forms::JITOptions opt;
        opt.enable = true;
        opt.optimization_level = 2;
        opt.vectorize = true;
        return opt;
    }();

    const auto compareKernel = [&](const FormExpr& residual_form,
                                   std::string_view label,
                                   const spaces::FunctionSpace& test_space,
                                   const spaces::FunctionSpace& trial_space,
                                   const dofs::DofMap& row_map,
                                   const dofs::DofMap& col_map,
                                   std::span<const Real> current,
                                   std::span<const Real> previous,
                                   std::span<const assembly::FieldSolutionAccess> field_access,
                                   GlobalIndex n_rows,
                                   GlobalIndex n_cols) {
        SCOPED_TRACE(std::string(label));
        auto ir_interp = compiler.compileResidual(residual_form);
        auto ir_jit = compiler.compileResidual(residual_form);

        SymbolicNonlinearFormKernel interp_kernel(std::move(ir_interp), NonlinearKernelOutput::Both);
        interp_kernel.resolveInlinableConstitutives();

        auto jit_fallback =
            std::make_shared<SymbolicNonlinearFormKernel>(std::move(ir_jit), NonlinearKernelOutput::Both);
        forms::jit::JITKernelWrapper jit_kernel(jit_fallback, jit_opts);
        jit_kernel.resolveInlinableConstitutives();
        jit_kernel.ensureCompiled();
        ASSERT_TRUE(jit_kernel.isJITReady())
            << "selective-weight parity must exercise LLVM, not compile fallback";

        assembly::StandardAssembler assembler;
        assembler.setRowDofMap(row_map);
        assembler.setColDofMap(col_map);
        assembler.setTimeIntegrationContext(&ti);
        assembler.setTime(0.0);
        assembler.setTimeStep(dt);
        assembler.setCurrentSolution(current);
        assembler.setPreviousSolution(previous);
        assembler.setFieldSolutionAccess(field_access);

        assembly::DenseMatrixView J_interp(n_rows, n_cols);
        assembly::DenseVectorView R_interp(n_rows);
        J_interp.zero();
        R_interp.zero();
        (void)assembler.assembleBoth(mesh, test_space, trial_space, interp_kernel, J_interp, R_interp);

        assembly::DenseMatrixView J_jit(n_rows, n_cols);
        assembly::DenseVectorView R_jit(n_rows);
        J_jit.zero();
        R_jit.zero();
        (void)assembler.assembleBoth(mesh, test_space, trial_space, jit_kernel, J_jit, R_jit);

        expectDenseNear(R_jit, R_interp, 1e-12);
        expectDenseNear(J_jit, J_interp, 1e-12);

        // A consistent first-rate solve separates the algebraic residual from
        // the differential mass operator with selective term weights.  VMS
        // expressions contain dt(u) below nonlinear subscale operations, so
        // exercise both compiled selections explicitly. LLVM emits a control-
        // flow skip for an exactly-zero complete-term weight before evaluating
        // that term; finite interpreter parity here protects that contract.
        const auto saved_time_weight = ti.time_derivative_term_weight;
        const auto saved_non_time_weight =
            ti.non_time_derivative_term_weight;
        const auto compareSelectiveWeights = [&](Real time_weight,
                                                 Real non_time_weight) {
            ti.time_derivative_term_weight = time_weight;
            ti.non_time_derivative_term_weight = non_time_weight;

            J_interp.zero();
            R_interp.zero();
            (void)assembler.assembleBoth(mesh, test_space, trial_space,
                                         interp_kernel, J_interp, R_interp);

            J_jit.zero();
            R_jit.zero();
            (void)assembler.assembleBoth(mesh, test_space, trial_space,
                                         jit_kernel, J_jit, R_jit);

            for (GlobalIndex i = 0; i < n_rows; ++i) {
                EXPECT_TRUE(std::isfinite(R_interp.getVectorEntry(i)));
                EXPECT_TRUE(std::isfinite(R_jit.getVectorEntry(i)));
            }
            for (GlobalIndex i = 0; i < n_rows; ++i) {
                for (GlobalIndex j = 0; j < n_cols; ++j) {
                    EXPECT_TRUE(std::isfinite(J_interp.getMatrixEntry(i, j)));
                    EXPECT_TRUE(std::isfinite(J_jit.getMatrixEntry(i, j)));
                }
            }
            expectDenseNear(R_jit, R_interp, 1e-12);
            expectDenseNear(J_jit, J_interp, 1e-12);
        };

        compareSelectiveWeights(Real{0.0}, Real{1.0});
        compareSelectiveWeights(Real{1.0}, Real{0.0});
        ti.time_derivative_term_weight = saved_time_weight;
        ti.non_time_derivative_term_weight = saved_non_time_weight;
    };

    // dR_u/du: u TrialFunction, p DiscreteField.
    {
        const auto u = TrialFunction(vel_space, "u");
        const auto p = FormExpr::discreteField(p_id, p_space, "p");
        const auto form = makeStokesVMSMomentumForm(vel_space, u, p, dim, params);

        const std::array<assembly::FieldSolutionAccess, 1> fields = {{
            {.field = p_id, .space = &p_space, .dof_map = &dof_map_p, .dof_offset = 6},
        }};

        compareKernel(form, "dR_u/du (Triangle3)", vel_space, vel_space, dof_map_u, dof_map_u,
                      sol_u_primary, prev_u_primary, fields, /*n_rows=*/6, /*n_cols=*/6);
    }

    // dR_u/dp: p TrialFunction, u DiscreteField.
    {
        const auto u = FormExpr::discreteField(u_id, vel_space, "u");
        const auto p = TrialFunction(p_space, "p");
        const auto form = makeStokesVMSMomentumForm(vel_space, u, p, dim, params);

        const std::array<assembly::FieldSolutionAccess, 1> fields = {{
            {.field = u_id, .space = &vel_space, .dof_map = &dof_map_u, .dof_offset = 3},
        }};

        compareKernel(form, "dR_u/dp (Triangle3)", vel_space, p_space, dof_map_u, dof_map_p,
                      sol_p_primary, prev_p_primary, fields, /*n_rows=*/6, /*n_cols=*/3);
    }

    // dR_p/du: u TrialFunction, p DiscreteField.
    {
        const auto u = TrialFunction(vel_space, "u");
        const auto p = FormExpr::discreteField(p_id, p_space, "p");
        const auto form = makeStokesVMSContinuityForm(vel_space, p_space, u, p, dim, params);

        const std::array<assembly::FieldSolutionAccess, 1> fields = {{
            {.field = p_id, .space = &p_space, .dof_map = &dof_map_p, .dof_offset = 6},
        }};

        compareKernel(form, "dR_p/du (Triangle3)", p_space, vel_space, dof_map_p, dof_map_u,
                      sol_u_primary, prev_u_primary, fields, /*n_rows=*/3, /*n_cols=*/6);
    }

    // dR_p/dp: p TrialFunction, u DiscreteField.
    {
        const auto u = FormExpr::discreteField(u_id, vel_space, "u");
        const auto p = TrialFunction(p_space, "p");
        const auto form = makeStokesVMSContinuityForm(vel_space, p_space, u, p, dim, params);

        const std::array<assembly::FieldSolutionAccess, 1> fields = {{
            {.field = u_id, .space = &vel_space, .dof_map = &dof_map_u, .dof_offset = 3},
        }};

        compareKernel(form, "dR_p/dp (Triangle3)", p_space, p_space, dof_map_p, dof_map_p,
                      sol_p_primary, prev_p_primary, fields, /*n_rows=*/3, /*n_cols=*/3);
    }
}

TEST(JITCompilerStokesVMS,
     Triangle3_CutVolumeSelectiveWeights_JITMatchInterpreter)
{
    requireLLVMJITOrSkip();

    constexpr int dim = 2;
    constexpr int marker = 91;
    constexpr Real dt = Real{0.05};
    const StokesVMSParams params{};
    SingleTriangleMeshAccess mesh;

    auto base =
        std::make_shared<spaces::H1Space>(ElementType::Triangle3, /*order=*/1);
    spaces::ProductSpace velocity_space(base, dim);
    spaces::H1Space pressure_space(ElementType::Triangle3, /*order=*/1);

    dofs::DofMap velocity_map(
        /*n_cells=*/1, /*n_dofs_total=*/6, /*dofs_per_cell=*/6);
    const std::array<GlobalIndex, 6> velocity_dofs = {0, 1, 2, 3, 4, 5};
    velocity_map.setCellDofs(0, velocity_dofs);
    velocity_map.setNumDofs(6);
    velocity_map.setNumLocalDofs(6);
    velocity_map.finalize();
    auto pressure_map = createSingleTriangleDofMap();

    const FieldId pressure_id = static_cast<FieldId>(1);
    const auto velocity = TrialFunction(velocity_space, "u");
    const auto pressure =
        FormExpr::discreteField(pressure_id, pressure_space, "p");
    const auto residual = makeStokesVMSMomentumForm(
        velocity_space, velocity, pressure, dim, params, marker);

    FormCompiler compiler;
    auto interpreter_ir = compiler.compileResidual(residual);
    auto jit_ir = compiler.compileResidual(residual);
    SymbolicNonlinearFormKernel interpreter_kernel(
        std::move(interpreter_ir), NonlinearKernelOutput::Both);
    interpreter_kernel.resolveInlinableConstitutives();

    auto jit_fallback = std::make_shared<SymbolicNonlinearFormKernel>(
        std::move(jit_ir), NonlinearKernelOutput::Both);
    auto jit_options = makeUnitTestJITOptions();
    jit_options.enable = true;
    jit_options.optimization_level = 2;
    forms::jit::JITKernelWrapper jit_kernel(jit_fallback, jit_options);
    jit_kernel.resolveInlinableConstitutives();
    jit_kernel.ensureCompiled();
    ASSERT_TRUE(jit_kernel.isJITReady())
        << "CutVolume selective-weight parity must exercise LLVM";
    ASSERT_TRUE(jit_kernel.hasCompiledTangentDispatch(
        IntegralDomain::CutVolume, marker))
        << "CutVolume tangent did not receive a compiled marker dispatch";

    geometry::CutQuadratureRule rule;
    rule.kind = geometry::CutQuadratureKind::Volume;
    rule.side = geometry::CutIntegrationSide::Negative;
    rule.parent_measure = Real{0.5};
    rule.measure = Real{0.25};
    rule.volume_fraction = Real{0.5};
    rule.frame = geometry::CutGeometryFrame::Reference;
    rule.provenance.parent_entity = 0;
    rule.provenance.marker = marker;
    rule.points.push_back(geometry::CutQuadraturePoint{
        .point = {{Real{1.0 / 6.0}, Real{1.0 / 6.0}, Real{0.0}}},
        .weight = rule.measure,
    });
    assembly::CutCellAssemblyMetadata metadata;
    metadata.parent_entity = 0;
    metadata.side = geometry::CutIntegrationSide::Negative;
    metadata.volume_fraction = rule.volume_fraction;
    assembly::CutIntegrationContext cut_context;
    cut_context.addGeneratedVolumeRule(marker, metadata, rule);

    assembly::TimeIntegrationContext time_context;
    time_context.integrator_name = "unit";
    time_context.dt1 = assembly::TimeDerivativeStencil{
        .order = 1,
        .a = {Real{1.0} / dt, -Real{1.0} / dt},
    };

    const std::vector<Real> velocity_coefficients = {
        Real{0.1}, Real{-0.2}, Real{0.3},
        Real{-0.1}, Real{0.05}, Real{-0.03}};
    const std::vector<Real> pressure_coefficients = {
        Real{0.2}, Real{-0.1}, Real{0.05}};
    std::vector<Real> current(9, Real{0.0});
    std::copy(velocity_coefficients.begin(), velocity_coefficients.end(),
              current.begin());
    std::copy(pressure_coefficients.begin(), pressure_coefficients.end(),
              current.begin() + 6);
    const std::vector<Real> previous(9, Real{0.0});
    const std::array<assembly::FieldSolutionAccess, 1> field_access = {{
        {.field = pressure_id,
         .space = &pressure_space,
         .dof_map = &pressure_map,
         .dof_offset = 6},
    }};

    assembly::StandardAssembler assembler;
    assembler.setRowDofMap(velocity_map);
    assembler.setColDofMap(velocity_map);
    assembler.setTimeIntegrationContext(&time_context);
    assembler.setTime(0.0);
    assembler.setTimeStep(dt);
    assembler.setCurrentSolution(current);
    assembler.setPreviousSolution(previous);
    assembler.setFieldSolutionAccess(field_access);

    const auto assemble = [&](assembly::AssemblyKernel& kernel,
                              assembly::DenseMatrixView& matrix,
                              assembly::DenseVectorView& vector) {
        matrix.zero();
        vector.zero();
        const auto result = assembler.assembleCutVolumes(
            mesh, cut_context, marker, geometry::CutIntegrationSide::Negative,
            velocity_space, velocity_space, kernel, &matrix, &vector,
            /*assemble_matrix=*/true, /*assemble_vector=*/true);
        ASSERT_TRUE(result.success) << result.error_message;
        ASSERT_EQ(result.elements_assembled, 1);
    };

    const auto compare = [&](Real time_weight, Real non_time_weight) {
        SCOPED_TRACE("time_weight=" + std::to_string(time_weight) +
                     " non_time_weight=" + std::to_string(non_time_weight));
        time_context.time_derivative_term_weight = time_weight;
        time_context.non_time_derivative_term_weight = non_time_weight;

        assembly::DenseMatrixView interpreter_matrix(6, 6);
        assembly::DenseVectorView interpreter_vector(6);
        assemble(interpreter_kernel, interpreter_matrix, interpreter_vector);

        assembly::DenseMatrixView jit_matrix(6, 6);
        assembly::DenseVectorView jit_vector(6);
        assemble(jit_kernel, jit_matrix, jit_vector);

        Real max_interpreter_magnitude = Real{0.0};
        for (GlobalIndex row = 0; row < 6; ++row) {
            EXPECT_TRUE(std::isfinite(interpreter_vector.getVectorEntry(row)));
            EXPECT_TRUE(std::isfinite(jit_vector.getVectorEntry(row)));
            max_interpreter_magnitude = std::max(
                max_interpreter_magnitude,
                std::abs(interpreter_vector.getVectorEntry(row)));
            for (GlobalIndex column = 0; column < 6; ++column) {
                EXPECT_TRUE(std::isfinite(
                    interpreter_matrix.getMatrixEntry(row, column)));
                EXPECT_TRUE(
                    std::isfinite(jit_matrix.getMatrixEntry(row, column)));
                max_interpreter_magnitude = std::max(
                    max_interpreter_magnitude,
                    std::abs(interpreter_matrix.getMatrixEntry(row, column)));
            }
        }
        EXPECT_GT(max_interpreter_magnitude, Real{1.0e-10})
            << "selective branch was not exercised by a nonzero operator";
        expectDenseNear(jit_vector, interpreter_vector, Real{1.0e-12});
        expectDenseNear(jit_matrix, interpreter_matrix, Real{1.0e-12});
    };

    compare(Real{0.0}, Real{1.0});
    compare(Real{1.0}, Real{0.0});
}

TEST(JITCompilerVectorOps, Triangle3_DivU_JITMatchesInterpreter)
{
    requireLLVMJITOrSkip();

    constexpr int dim = 2;
    SingleTriangleMeshAccess mesh;

    auto base = std::make_shared<spaces::H1Space>(ElementType::Triangle3, /*order=*/1);
    spaces::ProductSpace vel_space(base, dim);
    spaces::H1Space p_space(ElementType::Triangle3, /*order=*/1);

    // Local DofMaps.
    dofs::DofMap dof_map_u(/*n_cells=*/1, /*n_dofs_total=*/6, /*dofs_per_cell=*/6);
    const std::array<GlobalIndex, 6> u_dofs = {0, 1, 2, 3, 4, 5};
    dof_map_u.setCellDofs(0, u_dofs);
    dof_map_u.setNumDofs(6);
    dof_map_u.setNumLocalDofs(6);
    dof_map_u.finalize();

    auto dof_map_p = createSingleTriangleDofMap();

    FormCompiler compiler;
    const forms::JITOptions jit_opts = [] {
        forms::JITOptions opt;
        opt.enable = true;
        opt.optimization_level = 2;
        opt.vectorize = true;
        return opt;
    }();

    const auto u = TrialFunction(vel_space, "u");
    const auto q = TestFunction(p_space, "q");
    const auto form = (q * div(u)).dx();

    auto ir_interp = compiler.compileResidual(form);
    auto ir_jit = compiler.compileResidual(form);

    SymbolicNonlinearFormKernel interp_kernel(std::move(ir_interp), NonlinearKernelOutput::Both);
    interp_kernel.resolveInlinableConstitutives();

    auto jit_fallback =
        std::make_shared<SymbolicNonlinearFormKernel>(std::move(ir_jit), NonlinearKernelOutput::Both);
    forms::jit::JITKernelWrapper jit_kernel(jit_fallback, jit_opts);
    jit_kernel.resolveInlinableConstitutives();

    assembly::StandardAssembler assembler;
    assembler.setRowDofMap(dof_map_p);
    assembler.setColDofMap(dof_map_u);

    const std::vector<Real> sol_u = {0.1, -0.2, 0.3, -0.1, 0.05, -0.03};
    assembler.setCurrentSolution(sol_u);

    assembly::DenseMatrixView J_interp(/*n_rows=*/3, /*n_cols=*/6);
    assembly::DenseVectorView R_interp(/*n_rows=*/3);
    J_interp.zero();
    R_interp.zero();
    (void)assembler.assembleBoth(mesh, p_space, vel_space, interp_kernel, J_interp, R_interp);

    assembly::DenseMatrixView J_jit(/*n_rows=*/3, /*n_cols=*/6);
    assembly::DenseVectorView R_jit(/*n_rows=*/3);
    J_jit.zero();
    R_jit.zero();
    (void)assembler.assembleBoth(mesh, p_space, vel_space, jit_kernel, J_jit, R_jit);

    expectDenseNear(R_jit, R_interp, 1e-12);
    expectDenseNear(J_jit, J_interp, 1e-12);
}

} // namespace test
} // namespace forms
} // namespace FE
} // namespace svmp

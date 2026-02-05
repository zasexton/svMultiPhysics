/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

/**
 * @file test_JIT_ExtendedParity.cpp
 * @brief Numerical parity checks between interpreter and LLVM JIT for extended operator coverage.
 */

#include <gtest/gtest.h>

#include "Assembly/GlobalSystemView.h"
#include "Assembly/StandardAssembler.h"
#include "Assembly/TimeIntegrationContext.h"
#include "Forms/FormCompiler.h"
#include "Forms/FormKernels.h"
#include "Forms/JIT/JITKernelWrapper.h"
#include "Forms/Vocabulary.h"
#include "Spaces/H1Space.h"
#include "Tests/Unit/Forms/FormsTestHelpers.h"

#include <memory>
#include <optional>
#include <vector>

namespace svmp {
namespace FE {
namespace forms {
namespace test {

#if SVMP_FE_ENABLE_LLVM_JIT
namespace {

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

[[nodiscard]] forms::JITOptions defaultJitOptions()
{
    forms::JITOptions jit_opts;
    jit_opts.enable = true;
    jit_opts.optimization_level = 2;
    jit_opts.vectorize = true;
    return jit_opts;
}

struct CompareOptions {
    const assembly::TimeIntegrationContext* time_ctx{nullptr};
    Real time_step{0.0};
    std::optional<std::vector<Real>> prev_solution{};
    std::optional<std::vector<Real>> prev_solution2{};
};

void expectJitMatchesInterpreterCell(const assembly::IMeshAccess& mesh,
                                    const dofs::DofMap& dof_map,
                                    const spaces::FunctionSpace& space,
                                    const FormExpr& residual,
                                    const std::vector<Real>& U,
                                    Real vec_tol,
                                    Real mat_tol,
                                    const CompareOptions& options = {})
{
    FormCompiler compiler;
    auto ir_interp = compiler.compileResidual(residual);
    auto ir_jit = compiler.compileResidual(residual);

    auto interp_kernel =
        std::make_shared<SymbolicNonlinearFormKernel>(std::move(ir_interp), NonlinearKernelOutput::Both);
    interp_kernel->resolveInlinableConstitutives();

    auto jit_fallback =
        std::make_shared<SymbolicNonlinearFormKernel>(std::move(ir_jit), NonlinearKernelOutput::Both);
    forms::jit::JITKernelWrapper jit_kernel(jit_fallback, defaultJitOptions());
    jit_kernel.resolveInlinableConstitutives();

    assembly::StandardAssembler assembler;
    assembler.setDofMap(dof_map);
    assembler.setCurrentSolution(U);
    assembler.setTimeIntegrationContext(options.time_ctx);
    assembler.setTimeStep(options.time_step);
    if (options.prev_solution.has_value()) {
        assembler.setPreviousSolution(*options.prev_solution);
    }
    if (options.prev_solution2.has_value()) {
        assembler.setPreviousSolution2(*options.prev_solution2);
    }

    const GlobalIndex n = dof_map.getNumDofs();

    assembly::DenseMatrixView J_interp(n);
    assembly::DenseVectorView R_interp(n);
    J_interp.zero();
    R_interp.zero();
    (void)assembler.assembleBoth(mesh, space, space, *interp_kernel, J_interp, R_interp);

    assembly::DenseMatrixView J_jit(n);
    assembly::DenseVectorView R_jit(n);
    J_jit.zero();
    R_jit.zero();
    (void)assembler.assembleBoth(mesh, space, space, jit_kernel, J_jit, R_jit);

    expectDenseNear(R_jit, R_interp, vec_tol);
    expectDenseNear(J_jit, J_interp, mat_tol);
}

void expectJitMatchesInterpreterBoundary(const assembly::IMeshAccess& mesh,
                                        int boundary_marker,
                                        const dofs::DofMap& dof_map,
                                        const spaces::FunctionSpace& space,
                                        const FormExpr& residual,
                                        const std::vector<Real>& U,
                                        Real vec_tol,
                                        Real mat_tol)
{
    FormCompiler compiler;
    auto ir_interp = compiler.compileResidual(residual);
    auto ir_jit = compiler.compileResidual(residual);

    auto interp_kernel =
        std::make_shared<SymbolicNonlinearFormKernel>(std::move(ir_interp), NonlinearKernelOutput::Both);
    interp_kernel->resolveInlinableConstitutives();

    auto jit_fallback =
        std::make_shared<SymbolicNonlinearFormKernel>(std::move(ir_jit), NonlinearKernelOutput::Both);
    forms::jit::JITKernelWrapper jit_kernel(jit_fallback, defaultJitOptions());
    jit_kernel.resolveInlinableConstitutives();

    assembly::StandardAssembler assembler;
    assembler.setDofMap(dof_map);
    assembler.setCurrentSolution(U);

    const GlobalIndex n = dof_map.getNumDofs();

    assembly::DenseMatrixView J_interp(n);
    assembly::DenseVectorView R_interp(n);
    J_interp.zero();
    R_interp.zero();
    (void)assembler.assembleBoundaryFaces(mesh, boundary_marker, space, *interp_kernel, &J_interp, &R_interp);

    assembly::DenseMatrixView J_jit(n);
    assembly::DenseVectorView R_jit(n);
    J_jit.zero();
    R_jit.zero();
    (void)assembler.assembleBoundaryFaces(mesh, boundary_marker, space, jit_kernel, &J_jit, &R_jit);

    expectDenseNear(R_jit, R_interp, vec_tol);
    expectDenseNear(J_jit, J_interp, mat_tol);
}

void expectJitMatchesInterpreterInteriorFacesBilinear(const assembly::IMeshAccess& mesh,
                                                      const dofs::DofMap& dof_map,
                                                      const spaces::FunctionSpace& test_space,
                                                      const spaces::FunctionSpace& trial_space,
                                                      const FormExpr& bilinear,
                                                      Real mat_tol)
{
    FormCompiler compiler;
    auto ir_interp = compiler.compileBilinear(bilinear);
    auto ir_jit = compiler.compileBilinear(bilinear);

    auto interp_kernel = std::make_shared<FormKernel>(std::move(ir_interp));
    auto jit_fallback = std::make_shared<FormKernel>(std::move(ir_jit));

    forms::jit::JITKernelWrapper jit_kernel(jit_fallback, defaultJitOptions());

    assembly::StandardAssembler assembler;
    assembler.setDofMap(dof_map);

    const GlobalIndex n = dof_map.getNumDofs();

    assembly::DenseMatrixView A_interp(n);
    A_interp.zero();
    (void)assembler.assembleInteriorFaces(mesh, test_space, trial_space, *interp_kernel, A_interp, nullptr);

    assembly::DenseMatrixView A_jit(n);
    A_jit.zero();
    (void)assembler.assembleInteriorFaces(mesh, test_space, trial_space, jit_kernel, A_jit, nullptr);

    expectDenseNear(A_jit, A_interp, mat_tol);
}

FormExpr spd2x2FromGradU(const FormExpr& u)
{
    const auto I = FormExpr::identity(2);
    const auto g = grad(u);
    return (2.0 * I) + outer(g, g);
}

} // namespace

TEST(JITExtendedParityTest, SmoothHeavisideCellMatchesInterpreter)
{
    SingleTetraMeshAccess mesh;
    auto dof_map = createSingleTetraDofMap();
    spaces::H1Space space(ElementType::Tetra4, /*order=*/1);

    const auto u = TrialFunction(space, "u");
    const auto v = TestFunction(space, "v");
    const auto eps = FormExpr::constant(Real(0.2));
    const auto residual = inner(smoothHeaviside(u, eps) * grad(u), grad(v)).dx();

    const std::vector<Real> U = {-0.2, 0.1, 0.25, -0.05};
    expectJitMatchesInterpreterCell(mesh, dof_map, space, residual, U, /*vec_tol=*/1e-12, /*mat_tol=*/1e-12);
}

TEST(JITExtendedParityTest, MatrixExp2DCellMatchesInterpreter)
{
    SingleTriangleMeshAccess mesh;
    auto dof_map = createSingleTriangleDofMap();
    spaces::H1Space space(ElementType::Triangle3, /*order=*/1);

    const auto u = TrialFunction(space, "u");
    const auto v = TestFunction(space, "v");
    const auto A = spd2x2FromGradU(u);
    const auto residual = (A.matrixExp().trace() * v).dx();

    const std::vector<Real> U = {0.12, -0.08, 0.15};
    expectJitMatchesInterpreterCell(mesh, dof_map, space, residual, U, /*vec_tol=*/1e-12, /*mat_tol=*/1e-12);
}

TEST(JITExtendedParityTest, DGInteriorPenaltyMatchesInterpreter)
{
    TwoTetraSharedFaceMeshAccess mesh;
    auto dof_map = createTwoTetraDG_DofMap();
    spaces::H1Space space(ElementType::Tetra4, /*order=*/1);

    const auto u = FormExpr::trialFunction(space, "u");
    const auto v = FormExpr::testFunction(space, "v");

    const auto eta = FormExpr::constant(Real(10.0));
    const auto form = (eta * inner(jump(u), jump(v))).dS();

    expectJitMatchesInterpreterInteriorFacesBilinear(mesh, dof_map, space, space, form, /*mat_tol=*/1e-12);
}

TEST(JITExtendedParityTest, NitscheBoundaryMatchesInterpreter)
{
    SingleTetraOneBoundaryFaceMeshAccess mesh(/*boundary_marker=*/2);
    auto dof_map = createSingleTetraDofMap();
    spaces::H1Space space(ElementType::Tetra4, /*order=*/1);

    const auto u = FormExpr::trialFunction(space, "u");
    const auto v = FormExpr::testFunction(space, "v");

    const auto n = FormExpr::normal();
    const auto gamma = FormExpr::constant(Real(25.0));
    const auto residual =
        (-inner(grad(u), n) * v - u * inner(grad(v), n) + (gamma / h()) * u * v).ds(2);

    const std::vector<Real> U = {0.1, -0.2, 0.3, -0.1};
    expectJitMatchesInterpreterBoundary(mesh, /*boundary_marker=*/2, dof_map, space, residual, U, /*vec_tol=*/1e-12,
                                        /*mat_tol=*/1e-12);
}

TEST(JITExtendedParityTest, HistoryConvolutionMatchesInterpreter)
{
    SingleTetraMeshAccess mesh;
    auto dof_map = createSingleTetraDofMap();
    spaces::H1Space space(ElementType::Tetra4, /*order=*/1);

    constexpr Real w1 = 0.25;
    constexpr Real w2 = -1.75;

    const auto u = TrialFunction(space, "u");
    const auto v = TestFunction(space, "v");
    const auto hist = FormExpr::historyConvolution({FormExpr::constant(w1), FormExpr::constant(w2)});
    const auto residual = ((u + hist) * v).dx();

    // Dummy transient context so StandardAssembler populates previous-solution quadrature data.
    assembly::TimeIntegrationContext ti;
    ti.integrator_name = "unit_history_parity";
    assembly::TimeDerivativeStencil dt1;
    dt1.order = 1;
    dt1.a = {Real(1.0), Real(-1.0), Real(1.0)};
    ti.dt1 = dt1;

    CompareOptions options{};
    options.time_ctx = &ti;

    const std::vector<Real> U = {0.1, -0.2, 0.3, -0.1};
    options.prev_solution = std::vector<Real>{0.07, -0.01, 0.02, 0.05};
    options.prev_solution2 = std::vector<Real>{0.09, 0.03, -0.04, 0.01};

    expectJitMatchesInterpreterCell(mesh, dof_map, space, residual, U, /*vec_tol=*/1e-12, /*mat_tol=*/1e-12, options);
}

#endif // SVMP_FE_ENABLE_LLVM_JIT

} // namespace test
} // namespace forms
} // namespace FE
} // namespace svmp

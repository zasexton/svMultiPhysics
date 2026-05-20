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
#include "Assembly/CutIntegrationContext.h"
#include "Assembly/StandardAssembler.h"
#include "Assembly/TimeIntegrationContext.h"
#include "Forms/BoundaryConditions.h"
#include "Forms/CutCellForms.h"
#include "Forms/FormCompiler.h"
#include "Forms/FormKernels.h"
#include "Forms/JIT/JITKernelWrapper.h"
#include "Forms/Vocabulary.h"
#include "Spaces/HCurlSpace.h"
#include "Spaces/HDivSpace.h"
#include "Spaces/H1Space.h"
#include "Tests/Unit/Forms/FormsTestHelpers.h"

#include <array>
#include <memory>
#include <optional>
#include <string>
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

[[nodiscard]] dofs::DofMap createSingleTetraDofMap(LocalIndex n_dofs)
{
    dofs::DofMap dof_map(1, static_cast<GlobalIndex>(n_dofs), n_dofs);
    std::vector<GlobalIndex> cell_dofs(static_cast<std::size_t>(n_dofs));
    for (LocalIndex i = 0; i < n_dofs; ++i) {
        cell_dofs[static_cast<std::size_t>(i)] = static_cast<GlobalIndex>(i);
    }
    dof_map.setCellDofs(0, cell_dofs);
    dof_map.setNumDofs(static_cast<GlobalIndex>(n_dofs));
    dof_map.setNumLocalDofs(static_cast<GlobalIndex>(n_dofs));
    dof_map.finalize();
    return dof_map;
}

[[nodiscard]] dofs::DofMap createTwoTetraContinuousDofMap()
{
    dofs::DofMap dof_map(/*n_cells=*/2, /*n_dofs_total=*/5, /*dofs_per_cell=*/4);
    dof_map.setCellDofs(0, std::vector<GlobalIndex>{0, 1, 2, 3});
    dof_map.setCellDofs(1, std::vector<GlobalIndex>{1, 2, 3, 4});
    dof_map.setNumDofs(5);
    dof_map.setNumLocalDofs(5);
    dof_map.finalize();
    return dof_map;
}

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

void expectJitMatchesInterpreterLinearFieldCell(const assembly::IMeshAccess& mesh,
                                                const dofs::DofMap& dof_map,
                                                const spaces::FunctionSpace& space,
                                                FieldId field,
                                                const FormExpr& linear_form,
                                                const std::vector<Real>& field_coefficients,
                                                Real vec_tol)
{
    FormCompiler compiler;
    auto ir_interp = compiler.compileLinear(linear_form);
    auto ir_jit = compiler.compileLinear(linear_form);

    auto interp_kernel = std::make_shared<FormKernel>(std::move(ir_interp));
    auto jit_fallback = std::make_shared<FormKernel>(std::move(ir_jit));
    forms::jit::JITKernelWrapper jit_kernel(jit_fallback, defaultJitOptions());

    const std::array<assembly::FieldSolutionAccess, 1> field_access = {{
        assembly::FieldSolutionAccess{
            .field = field,
            .space = &space,
            .dof_map = &dof_map,
            .dof_offset = 0,
        },
    }};

    assembly::StandardAssembler assembler;
    assembler.setDofMap(dof_map);
    assembler.setFieldSolutionAccess(field_access);
    assembler.setCurrentSolution(field_coefficients);

    const GlobalIndex n = dof_map.getNumDofs();

    assembly::DenseVectorView R_interp(n);
    R_interp.zero();
    (void)assembler.assembleVector(mesh, space, *interp_kernel, R_interp);

    assembly::DenseVectorView R_jit(n);
    R_jit.zero();
    (void)assembler.assembleVector(mesh, space, jit_kernel, R_jit);

    expectDenseNear(R_jit, R_interp, vec_tol);
}

void expectJitCellAssemblyThrowsContaining(const assembly::IMeshAccess& mesh,
                                           const dofs::DofMap& dof_map,
                                           const spaces::FunctionSpace& space,
                                           const FormExpr& residual,
                                           const std::vector<Real>& U,
                                           std::initializer_list<const char*> snippets)
{
    FormCompiler compiler;
    auto ir = compiler.compileResidual(residual);

    auto fallback =
        std::make_shared<SymbolicNonlinearFormKernel>(std::move(ir), NonlinearKernelOutput::Both);
    forms::jit::JITKernelWrapper jit_kernel(fallback, defaultJitOptions());
    jit_kernel.resolveInlinableConstitutives();

    assembly::StandardAssembler assembler;
    assembler.setDofMap(dof_map);
    assembler.setCurrentSolution(U);

    const GlobalIndex n = dof_map.getNumDofs();
    assembly::DenseMatrixView J(n);
    assembly::DenseVectorView R(n);
    J.zero();
    R.zero();

    try {
        (void)assembler.assembleBoth(mesh, space, space, jit_kernel, J, R);
        FAIL() << "Expected FEException";
    } catch (const FEException& ex) {
        const std::string msg = ex.what();
        for (const auto* snippet : snippets) {
            EXPECT_NE(msg.find(snippet), std::string::npos)
                << "missing diagnostic snippet: " << snippet << "\nmessage: " << msg;
        }
    }
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

void expectJitMatchesInterpreterInteriorFacesResidual(
    const assembly::IMeshAccess& mesh,
    const dofs::DofMap& dof_map,
    const spaces::FunctionSpace& space,
    const FormExpr& residual,
    const std::vector<Real>& U,
    int marker,
    const assembly::CutIntegrationContext* cut_context,
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
    assembler.setCutIntegrationContext(cut_context);
    assembler.setCurrentSolution(U);

    const GlobalIndex n = dof_map.getNumDofs();

    assembly::DenseMatrixView J_interp(n);
    assembly::DenseVectorView R_interp(n);
    J_interp.zero();
    R_interp.zero();
    (void)assembler.assembleInteriorFaces(mesh, space, space, *interp_kernel, J_interp, &R_interp, marker);

    assembly::DenseMatrixView J_jit(n);
    assembly::DenseVectorView R_jit(n);
    J_jit.zero();
    R_jit.zero();
    (void)assembler.assembleInteriorFaces(mesh, space, space, jit_kernel, J_jit, &R_jit, marker);

    expectDenseNear(R_jit, R_interp, vec_tol);
    expectDenseNear(J_jit, J_interp, mat_tol);
}

FormExpr spd2x2FromGradU(const FormExpr& u)
{
    const auto I = FormExpr::identity(2);
    const auto g = grad(u);
    return (2.0 * I) + outer(g, g);
}

struct CurvedPiolaGradientCase {
    const char* name;
    ElementType geometry_type;
    ElementType space_type;
};

[[nodiscard]] std::vector<CurvedPiolaGradientCase> supportedCurvedVolumePiolaCases()
{
    return {
        {"Tetra10", ElementType::Tetra10, ElementType::Tetra4},
        {"Hex20", ElementType::Hex20, ElementType::Hex8},
        {"Hex27", ElementType::Hex27, ElementType::Hex8},
        {"Wedge15", ElementType::Wedge15, ElementType::Wedge6},
        {"Wedge18", ElementType::Wedge18, ElementType::Wedge6},
        {"Pyramid13", ElementType::Pyramid13, ElementType::Pyramid5},
        {"Pyramid14", ElementType::Pyramid14, ElementType::Pyramid5},
    };
}

[[nodiscard]] std::vector<CurvedPiolaGradientCase> lowerDimensionalCurvedPiolaCases()
{
    return {
        {"Triangle6", ElementType::Triangle6, ElementType::Triangle3},
        {"Quad8", ElementType::Quad8, ElementType::Quad4},
        {"Quad9", ElementType::Quad9, ElementType::Quad4},
    };
}

[[nodiscard]] std::vector<Real> deterministicCoefficients(std::size_t n)
{
    std::vector<Real> coeffs(n);
    for (std::size_t i = 0; i < n; ++i) {
        const Real sign = (i % 2u == 0u) ? Real(1) : Real(-1);
        coeffs[i] = sign * (Real(0.07) + Real(0.013) * static_cast<Real>(i));
    }
    return coeffs;
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

TEST(JITExtendedParityTest, IntrinsicVectorBasisGradientMatchesInterpreter)
{
    SingleTetraMeshAccess mesh;
    spaces::HCurlSpace space(ElementType::Tetra4, /*order=*/0, BasisType::Nedelec);
    auto dof_map = createSingleTetraDofMap(static_cast<LocalIndex>(space.dofs_per_element()));

    const auto u = TrialFunction(space, "u");
    const auto v = TestFunction(space, "v");
    const auto residual = (inner(sym(grad(u)), sym(grad(v))) + Real(0.25) * inner(grad(u), grad(v))).dx();

    const std::vector<Real> U = {0.12, -0.08, 0.15, -0.03, 0.21, -0.11};
    expectJitMatchesInterpreterCell(mesh, dof_map, space, residual, U, /*vec_tol=*/1e-12, /*mat_tol=*/1e-12);
}

TEST(JITExtendedParityTest, CurvedIntrinsicVectorBasisGradientMatchesInterpreter)
{
    CurvedTetra10MeshAccess mesh;

    {
        spaces::HCurlSpace space(ElementType::Tetra4, /*order=*/0, BasisType::Nedelec);
        auto dof_map = createSingleTetraDofMap(static_cast<LocalIndex>(space.dofs_per_element()));

        const auto u = TrialFunction(space, "u");
        const auto v = TestFunction(space, "v");
        const auto residual = (inner(sym(grad(u)), sym(grad(v))) +
                               Real(0.25) * inner(grad(u), grad(v))).dx();

        const std::vector<Real> U = {0.12, -0.08, 0.15, -0.03, 0.21, -0.11};
        expectJitMatchesInterpreterCell(mesh, dof_map, space, residual, U,
                                        /*vec_tol=*/1e-12, /*mat_tol=*/1e-12);
    }

    {
        spaces::HDivSpace space(ElementType::Tetra4, /*order=*/0, BasisType::RaviartThomas);
        auto dof_map = createSingleTetraDofMap(static_cast<LocalIndex>(space.dofs_per_element()));

        const auto u = TrialFunction(space, "u");
        const auto v = TestFunction(space, "v");
        const auto residual = inner(grad(u), grad(v)).dx();

        const std::vector<Real> U = {0.2, -0.1, 0.05, 0.16};
        expectJitMatchesInterpreterCell(mesh, dof_map, space, residual, U,
                                        /*vec_tol=*/1e-12, /*mat_tol=*/1e-12);
    }
}

TEST(JITExtendedParityTest, CurvedVolumeIntrinsicVectorBasisGradientsCoverAllEnabledGeometryFamilies)
{
    for (const auto& c : supportedCurvedVolumePiolaCases()) {
        SCOPED_TRACE(c.name);
        CurvedSingleCellMeshAccess mesh(c.geometry_type, c.name);

        {
            spaces::HCurlSpace space(c.space_type, /*order=*/0, BasisType::Nedelec);
            auto dof_map = createSingleTetraDofMap(static_cast<LocalIndex>(space.dofs_per_element()));

            const auto u = TrialFunction(space, "u");
            const auto v = TestFunction(space, "v");
            const auto residual = (inner(sym(grad(u)), sym(grad(v))) +
                                   Real(0.25) * inner(grad(u), grad(v))).dx();

            expectJitMatchesInterpreterCell(mesh,
                                            dof_map,
                                            space,
                                            residual,
                                            deterministicCoefficients(space.dofs_per_element()),
                                            /*vec_tol=*/2e-10,
                                            /*mat_tol=*/2e-10);
        }

        {
            spaces::HDivSpace space(c.space_type, /*order=*/0, BasisType::RaviartThomas);
            auto dof_map = createSingleTetraDofMap(static_cast<LocalIndex>(space.dofs_per_element()));

            const auto u = TrialFunction(space, "u");
            const auto v = TestFunction(space, "v");
            const auto residual = inner(grad(u), grad(v)).dx();

            expectJitMatchesInterpreterCell(mesh,
                                            dof_map,
                                            space,
                                            residual,
                                            deterministicCoefficients(space.dofs_per_element()),
                                            /*vec_tol=*/2e-10,
                                            /*mat_tol=*/2e-10);
        }
    }
}

TEST(JITExtendedParityTest, CurvedTetra10BDMHDivVectorBasisGradientMatchesInterpreter)
{
    CurvedSingleCellMeshAccess mesh(ElementType::Tetra10, "Tetra10-BDM");
    spaces::HDivSpace space(ElementType::Tetra4, /*order=*/1, BasisType::BDM);
    auto dof_map = createSingleTetraDofMap(static_cast<LocalIndex>(space.dofs_per_element()));

    const auto u = TrialFunction(space, "u");
    const auto v = TestFunction(space, "v");
    const auto residual = inner(grad(u), grad(v)).dx();

    expectJitMatchesInterpreterCell(mesh,
                                    dof_map,
                                    space,
                                    residual,
                                    deterministicCoefficients(space.dofs_per_element()),
                                    /*vec_tol=*/2e-10,
                                    /*mat_tol=*/2e-10);
}

TEST(JITExtendedParityTest, CurvedTetra10HigherOrderRTAndNedelecVectorBasisGradientMatchesInterpreter)
{
    CurvedSingleCellMeshAccess mesh(ElementType::Tetra10, "Tetra10-higher-order-RT-Nedelec");

    {
        spaces::HDivSpace space(ElementType::Tetra4, /*order=*/1, BasisType::RaviartThomas);
        auto dof_map = createSingleTetraDofMap(static_cast<LocalIndex>(space.dofs_per_element()));

        const auto u = TrialFunction(space, "u");
        const auto v = TestFunction(space, "v");
        const auto residual = inner(grad(u), grad(v)).dx();

        expectJitMatchesInterpreterCell(mesh,
                                        dof_map,
                                        space,
                                        residual,
                                        deterministicCoefficients(space.dofs_per_element()),
                                        /*vec_tol=*/2e-10,
                                        /*mat_tol=*/2e-10);
    }

    {
        spaces::HCurlSpace space(ElementType::Tetra4, /*order=*/1, BasisType::Nedelec);
        auto dof_map = createSingleTetraDofMap(static_cast<LocalIndex>(space.dofs_per_element()));

        const auto u = TrialFunction(space, "u");
        const auto v = TestFunction(space, "v");
        const auto residual = (inner(sym(grad(u)), sym(grad(v))) +
                               Real(0.25) * inner(grad(u), grad(v))).dx();

        expectJitMatchesInterpreterCell(mesh,
                                        dof_map,
                                        space,
                                        residual,
                                        deterministicCoefficients(space.dofs_per_element()),
                                        /*vec_tol=*/2e-10,
                                        /*mat_tol=*/2e-10);
    }
}

TEST(JITExtendedParityTest, CurvedVectorBasisFieldGradientMatchesInterpreter)
{
    constexpr FieldId kCurvedVectorField = 9201;
    CurvedSingleCellMeshAccess mesh(ElementType::Tetra10, "Tetra10-field-gradient-JIT");

    auto check_field_gradient = [&](const spaces::FunctionSpace& space, const FormExpr& field_expr) {
        auto dof_map = createSingleTetraDofMap(static_cast<LocalIndex>(space.dofs_per_element()));
        const auto v = TestFunction(space, "v");
        const auto linear_form = inner(grad(field_expr), grad(v)).dx();

        expectJitMatchesInterpreterLinearFieldCell(mesh,
                                                  dof_map,
                                                  space,
                                                  kCurvedVectorField,
                                                  linear_form,
                                                  deterministicCoefficients(space.dofs_per_element()),
                                                  /*vec_tol=*/2e-10);
    };

    {
        spaces::HDivSpace space(ElementType::Tetra4, /*order=*/0, BasisType::RaviartThomas);
        check_field_gradient(space, FormExpr::discreteField(kCurvedVectorField, space, "hdiv_discrete"));
        check_field_gradient(space, FormExpr::stateField(kCurvedVectorField, space, "hdiv_state"));
    }

    {
        spaces::HCurlSpace space(ElementType::Tetra4, /*order=*/0, BasisType::Nedelec);
        check_field_gradient(space, FormExpr::discreteField(kCurvedVectorField, space, "hcurl_discrete"));
        check_field_gradient(space, FormExpr::stateField(kCurvedVectorField, space, "hcurl_state"));
    }
}

TEST(JITExtendedParityTest, CurvedHigherOrderVectorBasisFieldGradientMatchesInterpreter)
{
    constexpr FieldId kCurvedVectorField = 9202;
    CurvedSingleCellMeshAccess mesh(ElementType::Tetra10, "Tetra10-higher-order-field-gradient-JIT");

    auto check_field_gradient = [&](const spaces::FunctionSpace& space, const FormExpr& field_expr) {
        auto dof_map = createSingleTetraDofMap(static_cast<LocalIndex>(space.dofs_per_element()));
        const auto v = TestFunction(space, "v");
        const auto linear_form = inner(grad(field_expr), grad(v)).dx();

        expectJitMatchesInterpreterLinearFieldCell(mesh,
                                                  dof_map,
                                                  space,
                                                  kCurvedVectorField,
                                                  linear_form,
                                                  deterministicCoefficients(space.dofs_per_element()),
                                                  /*vec_tol=*/3e-10);
    };

    {
        spaces::HDivSpace space(ElementType::Tetra4, /*order=*/1, BasisType::RaviartThomas);
        check_field_gradient(space, FormExpr::discreteField(kCurvedVectorField, space, "rt1_discrete"));
        check_field_gradient(space, FormExpr::stateField(kCurvedVectorField, space, "rt1_state"));
    }

    {
        spaces::HDivSpace space(ElementType::Tetra4, /*order=*/1, BasisType::BDM);
        check_field_gradient(space, FormExpr::discreteField(kCurvedVectorField, space, "bdm1_discrete"));
        check_field_gradient(space, FormExpr::stateField(kCurvedVectorField, space, "bdm1_state"));
    }

    {
        spaces::HCurlSpace space(ElementType::Tetra4, /*order=*/1, BasisType::Nedelec);
        check_field_gradient(space, FormExpr::discreteField(kCurvedVectorField, space, "nedelec1_discrete"));
        check_field_gradient(space, FormExpr::stateField(kCurvedVectorField, space, "nedelec1_state"));
    }
}

TEST(JITExtendedParityTest, LowerDimensionalCurvedPiolaVectorBasisGradientsFailClosed)
{
    for (const auto& c : lowerDimensionalCurvedPiolaCases()) {
        SCOPED_TRACE(c.name);
        CurvedSingleCellMeshAccess mesh(c.geometry_type, c.name);

        {
            spaces::HDivSpace space(c.space_type, /*order=*/0, BasisType::RaviartThomas);
            auto dof_map = createSingleTetraDofMap(static_cast<LocalIndex>(space.dofs_per_element()));
            const auto u = TrialFunction(space, "u");
            const auto v = TestFunction(space, "v");
            const auto residual = inner(grad(u), grad(v)).dx();
            expectJitCellAssemblyThrowsContaining(
                mesh,
                dof_map,
                space,
                residual,
                deterministicCoefficients(space.dofs_per_element()),
                {"curved Piola", "non-affine 3D volume mappings", "lower-dimensional curved mappings"});
        }

        {
            spaces::HCurlSpace space(c.space_type, /*order=*/0, BasisType::Nedelec);
            auto dof_map = createSingleTetraDofMap(static_cast<LocalIndex>(space.dofs_per_element()));
            const auto u = TrialFunction(space, "u");
            const auto v = TestFunction(space, "v");
            const auto residual = inner(grad(u), grad(v)).dx();
            expectJitCellAssemblyThrowsContaining(
                mesh,
                dof_map,
                space,
                residual,
                deterministicCoefficients(space.dofs_per_element()),
                {"curved Piola", "non-affine 3D volume mappings", "lower-dimensional curved mappings"});
        }
    }
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

TEST(JITExtendedParityTest, CutAdjacentGradientPenaltyContinuousH1MatchesInterpreter)
{
    constexpr int marker = 12;

    TwoTetraSharedFaceMeshAccess mesh;
    auto dof_map = createTwoTetraContinuousDofMap();
    spaces::H1Space space(ElementType::Tetra4, /*order=*/1);

    const auto u = TrialFunction(space, "u");
    const auto v = TestFunction(space, "v");
    const auto h_face = avg(hNormal());
    const auto h3 = h_face * h_face * h_face;
    const auto residual = cutAdjacentFacetIntegral(
        cutStabilizationScale() * h3 *
            inner(cutAdjacentFacetGradientJump(u),
                  cutAdjacentFacetGradientJump(v)),
        marker);

    assembly::CutIntegrationContext cut_context;
    assembly::CutFacetSetHandle handle;
    handle.marker = marker;
    handle.name = "continuous-h1-cut-adjacent-gradient-penalty";
    handle.facets = {0};
    handle.facet_metadata = {assembly::CutFacetSetFacetMetadata{
        .facet = 0,
        .first_cell = 0,
        .second_cell = 1,
        .stabilization_scale = Real{1.75},
        .stable_id = 17}};
    cut_context.addFacetSetHandle(std::move(handle));

    const std::vector<Real> U = {0.12, -0.05, 0.08, 0.02, -0.07};
    expectJitMatchesInterpreterInteriorFacesResidual(
        mesh,
        dof_map,
        space,
        residual,
        U,
        marker,
        &cut_context,
        /*vec_tol=*/1e-12,
        /*mat_tol=*/1e-12);
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

TEST(JITExtendedParityTest, TraceNitscheBoundaryMatchesInterpreter)
{
    SingleTetraOneBoundaryFaceMeshAccess mesh(/*boundary_marker=*/2);
    auto dof_map = createSingleTetraDofMap();
    spaces::H1Space space(ElementType::Tetra4, /*order=*/1);

    const auto u = FormExpr::trialFunction(space, "u");
    const auto v = FormExpr::testFunction(space, "v");
    const auto n = FormExpr::normal();
    auto residual = (u * v).dx();
    residual = bc::applyTraceNitsche(std::move(residual),
                                     u,
                                     v,
                                     /*boundary_marker=*/2,
                                     FormExpr::constant(0.0),
                                     inner(grad(u), n),
                                     inner(grad(v), n),
                                     FormExpr::constant(1.0) / h(),
                                     bc::ScalarTraceOperator::Identity);

    const std::vector<Real> U = {0.1, -0.2, 0.3, -0.1};
    expectJitMatchesInterpreterBoundary(mesh, /*boundary_marker=*/2, dof_map, space, residual, U,
                                        /*vec_tol=*/1e-12, /*mat_tol=*/1e-12);
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

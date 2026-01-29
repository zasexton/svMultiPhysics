/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include <gtest/gtest.h>

#include "Assembly/FunctionalAssembler.h"
#include "Forms/FormExpr.h"
#include "Forms/JIT/JITFunctionalKernelWrapper.h"
#include "Spaces/H1Space.h"
#include "Spaces/ProductSpace.h"
#include "Tests/Unit/Forms/FormsTestHelpers.h"
#include "Tests/Unit/Forms/JITTestHelpers.h"

#include <array>
#include <vector>

namespace svmp {
namespace FE {
namespace forms {
namespace test {
namespace {

[[nodiscard]] dofs::DofMap makeSingleCellDofMap(GlobalIndex n_dofs)
{
    dofs::DofMap dof_map(1, n_dofs, n_dofs);
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

Real assembleJIT(const FormExpr& integrand,
                 assembly::FunctionalAssembler& assembler,
                 bool has_cell,
                 bool has_boundary)
{
    auto throw_fallback = makeThrowingTotalKernelFor(integrand, has_cell, has_boundary);
    jit::JITFunctionalKernelWrapper jit_kernel(
        throw_fallback, integrand,
        has_cell ? jit::JITFunctionalKernelWrapper::Domain::Cell : jit::JITFunctionalKernelWrapper::Domain::BoundaryFace,
        makeUnitTestJITOptions());
    return has_cell ? assembler.assembleScalar(jit_kernel) : assembler.assembleBoundaryScalar(jit_kernel, /*boundary_marker=*/2);
}

Real assembleInterp(const FormExpr& integrand,
                    assembly::FunctionalAssembler& assembler,
                    FunctionalFormKernel::Domain domain,
                    int boundary_marker)
{
    auto interp = makeFunctionalFormKernel(integrand, domain);
    return (domain == FunctionalFormKernel::Domain::Cell) ? assembler.assembleScalar(*interp)
                                                          : assembler.assembleBoundaryScalar(*interp, boundary_marker);
}

} // namespace

TEST(LLVMGenDifferentialOps, Gradient_ComputesCorrectly)
{
    requireLLVMJITOrSkip();

    constexpr FieldId field = 0;
    SingleTetraMeshAccess mesh;
    const auto dof_map = createSingleTetraDofMap();
    spaces::H1Space space(ElementType::Tetra4, 1);

    // Represent u_h(x,y,z) = x by nodal values on the reference tetra.
    std::vector<Real> u(4, Real(0.0));
    u[0] = 0.0; // (0,0,0)
    u[1] = 1.0; // (1,0,0)
    u[2] = 0.0; // (0,1,0)
    u[3] = 0.0; // (0,0,1)

    assembly::FunctionalAssembler assembler;
    assembler.setMesh(mesh);
    assembler.setDofMap(dof_map);
    assembler.setSpace(space);
    assembler.setPrimaryField(field);
    assembler.setSolution(u);

    const auto one = FormExpr::constant(Real(1.0));
    const Real volume = assembler.assembleScalar(*makeFunctionalFormKernel(one, FunctionalFormKernel::Domain::Cell));
    ASSERT_GT(volume, Real(0.0));

    const auto integrand =
        FormExpr::discreteField(field, space, "u").grad().component(0); // du/dx = 1

    const Real ref = assembleInterp(integrand, assembler, FunctionalFormKernel::Domain::Cell, /*boundary_marker=*/-1);
    const Real jit = assembleJIT(integrand, assembler, /*has_cell=*/true, /*has_boundary=*/false);

    EXPECT_NEAR(jit, ref, 1e-12);
    EXPECT_NEAR(jit / volume, Real(1.0), 1e-12);
}

TEST(LLVMGenDifferentialOps, Divergence_ComputesCorrectly)
{
    requireLLVMJITOrSkip();

    constexpr FieldId field = 0;
    SingleTetraMeshAccess mesh;
    auto base = std::make_shared<spaces::H1Space>(ElementType::Tetra4, 1);
    spaces::ProductSpace space(base, 3);

    const auto n_dofs = static_cast<GlobalIndex>(space.dofs_per_element());
    const auto dof_map = makeSingleCellDofMap(n_dofs);

    // Represent u_h = [x, y, z] on the reference tetra (P1, component-wise ordering).
    std::vector<Real> u(static_cast<std::size_t>(n_dofs), Real(0.0));
    // component 0 (x): [0,1,0,0]
    u[0] = 0.0;
    u[1] = 1.0;
    u[2] = 0.0;
    u[3] = 0.0;
    // component 1 (y): [0,0,1,0]
    u[4] = 0.0;
    u[5] = 0.0;
    u[6] = 1.0;
    u[7] = 0.0;
    // component 2 (z): [0,0,0,1]
    u[8] = 0.0;
    u[9] = 0.0;
    u[10] = 0.0;
    u[11] = 1.0;

    assembly::FunctionalAssembler assembler;
    assembler.setMesh(mesh);
    assembler.setDofMap(dof_map);
    assembler.setSpace(space);
    assembler.setPrimaryField(field);
    assembler.setSolution(u);

    const auto one = FormExpr::constant(Real(1.0));
    const Real volume = assembler.assembleScalar(*makeFunctionalFormKernel(one, FunctionalFormKernel::Domain::Cell));
    ASSERT_GT(volume, Real(0.0));

    const auto integrand = FormExpr::discreteField(field, space, "u").div(); // div([x,y,z]) = 3

    const Real ref = assembleInterp(integrand, assembler, FunctionalFormKernel::Domain::Cell, /*boundary_marker=*/-1);
    const Real jit = assembleJIT(integrand, assembler, /*has_cell=*/true, /*has_boundary=*/false);

    EXPECT_NEAR(jit, ref, 1e-12);
    EXPECT_NEAR(jit / volume, Real(3.0), 1e-12);
}

TEST(LLVMGenDifferentialOps, Curl_ComputesCorrectly)
{
    requireLLVMJITOrSkip();

    constexpr FieldId field = 0;
    SingleTetraMeshAccess mesh;
    auto base = std::make_shared<spaces::H1Space>(ElementType::Tetra4, 1);
    spaces::ProductSpace space(base, 3);

    const auto n_dofs = static_cast<GlobalIndex>(space.dofs_per_element());
    const auto dof_map = makeSingleCellDofMap(n_dofs);

    // u_h = [x, y, z] => curl(u) = 0.
    std::vector<Real> u(static_cast<std::size_t>(n_dofs), Real(0.0));
    u[1] = 1.0;
    u[6] = 1.0;
    u[11] = 1.0;

    assembly::FunctionalAssembler assembler;
    assembler.setMesh(mesh);
    assembler.setDofMap(dof_map);
    assembler.setSpace(space);
    assembler.setPrimaryField(field);
    assembler.setSolution(u);

    const auto one = FormExpr::constant(Real(1.0));
    const Real volume = assembler.assembleScalar(*makeFunctionalFormKernel(one, FunctionalFormKernel::Domain::Cell));
    ASSERT_GT(volume, Real(0.0));

    const auto integrand =
        FormExpr::discreteField(field, space, "u").curl().component(0) +
        FormExpr::discreteField(field, space, "u").curl().component(1) * Real(10.0) +
        FormExpr::discreteField(field, space, "u").curl().component(2) * Real(100.0);

    const Real ref = assembleInterp(integrand, assembler, FunctionalFormKernel::Domain::Cell, /*boundary_marker=*/-1);
    const Real jit = assembleJIT(integrand, assembler, /*has_cell=*/true, /*has_boundary=*/false);

    EXPECT_NEAR(jit, ref, 1e-12);
    EXPECT_NEAR(jit / volume, Real(0.0), 1e-12);
}

TEST(LLVMGenDifferentialOps, Hessian_ComputesSecondDerivatives)
{
    requireLLVMJITOrSkip();

    constexpr FieldId field = 0;
    SingleTetraMeshAccess mesh;
    const auto dof_map = createSingleTetraDofMap();
    spaces::H1Space space(ElementType::Tetra4, 1);

    // u_h(x,y,z) = x (P1) => Hessian(u_h) = 0.
    std::vector<Real> u(4, Real(0.0));
    u[0] = 0.0; // (0,0,0)
    u[1] = 1.0; // (1,0,0)
    u[2] = 0.0; // (0,1,0)
    u[3] = 0.0; // (0,0,1)

    assembly::FunctionalAssembler assembler;
    assembler.setMesh(mesh);
    assembler.setDofMap(dof_map);
    assembler.setSpace(space);
    assembler.setPrimaryField(field);
    assembler.setSolution(u);

    const auto one = FormExpr::constant(Real(1.0));
    const Real volume = assembler.assembleScalar(*makeFunctionalFormKernel(one, FunctionalFormKernel::Domain::Cell));
    ASSERT_GT(volume, Real(0.0));

    const auto H = FormExpr::discreteField(field, space, "u").hessian();
    const auto integrand =
        H.component(0, 0) + H.component(1, 1) * Real(10.0) + H.component(2, 2) * Real(100.0);

    const Real ref = assembleInterp(integrand, assembler, FunctionalFormKernel::Domain::Cell, /*boundary_marker=*/-1);
    const Real jit = assembleJIT(integrand, assembler, /*has_cell=*/true, /*has_boundary=*/false);

    EXPECT_NEAR(jit, ref, 1e-12);
    EXPECT_NEAR(jit / volume, Real(0.0), 1e-12);
}

} // namespace test
} // namespace forms
} // namespace FE
} // namespace svmp

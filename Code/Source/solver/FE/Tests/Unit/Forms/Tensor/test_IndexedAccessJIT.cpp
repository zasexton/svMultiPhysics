/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include <gtest/gtest.h>

#include "Assembly/GlobalSystemView.h"
#include "Assembly/StandardAssembler.h"
#include "Forms/FormCompiler.h"
#include "Forms/FormKernels.h"
#include "Forms/Index.h"
#include "Forms/JIT/JITKernelWrapper.h"
#include "Spaces/H1Space.h"
#include "Spaces/ProductSpace.h"
#include "Tests/Unit/Forms/FormsTestHelpers.h"

namespace svmp {
namespace FE {
namespace forms {
namespace test {

#ifndef SVMP_FE_ENABLE_LLVM_JIT
#define SVMP_FE_ENABLE_LLVM_JIT 0
#endif

#if SVMP_FE_ENABLE_LLVM_JIT
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

void expectDenseNear(const assembly::DenseMatrixView& A,
                     const assembly::DenseMatrixView& B,
                     double tol)
{
    ASSERT_EQ(A.numRows(), B.numRows());
    ASSERT_EQ(A.numCols(), B.numCols());
    for (GlobalIndex i = 0; i < A.numRows(); ++i) {
        for (GlobalIndex j = 0; j < A.numCols(); ++j) {
            SCOPED_TRACE(::testing::Message() << "i=" << i << " j=" << j);
            EXPECT_NEAR(A.getMatrixEntry(i, j), B.getMatrixEntry(i, j), tol);
        }
    }
}

} // namespace

TEST(IndexedAccessJIT, CellMatrixContractionMatchesInterpreter)
{
    SingleTetraMeshAccess mesh;
    auto base = std::make_shared<spaces::H1Space>(ElementType::Tetra4, 1);
    spaces::ProductSpace space(base, 3);
    auto dof_map = makeSingleCellDofMap(static_cast<GlobalIndex>(space.dofs_per_element()));

    SymbolicOptions sym_opts;
    sym_opts.jit.enable = true;

    FormCompiler compiler(sym_opts);

    const auto u = FormExpr::trialFunction(space, "u");
    const auto v = FormExpr::testFunction(space, "v");

    const Index i("i");
    const Index j("j");
    const auto form = (grad(u)(i, j) * grad(v)(i, j)).dx();

    auto ir_interp = compiler.compileBilinear(form);
    auto ir_jit = compiler.compileBilinear(form);

    auto interp_kernel = std::make_shared<FormKernel>(std::move(ir_interp));

    auto jit_fallback = std::make_shared<FormKernel>(std::move(ir_jit));
    forms::JITOptions jit_opts;
    jit_opts.enable = true;
    jit_opts.optimization_level = 2;
    jit_opts.vectorize = true;
    forms::jit::JITKernelWrapper jit_kernel(jit_fallback, jit_opts);

    assembly::StandardAssembler assembler;
    assembler.setDofMap(dof_map);

    assembly::DenseMatrixView A_interp(static_cast<GlobalIndex>(space.dofs_per_element()));
    A_interp.zero();
    (void)assembler.assembleMatrix(mesh, space, space, *interp_kernel, A_interp);

    assembly::DenseMatrixView A_jit(static_cast<GlobalIndex>(space.dofs_per_element()));
    A_jit.zero();
    (void)assembler.assembleMatrix(mesh, space, space, jit_kernel, A_jit);

    expectDenseNear(A_jit, A_interp, 1e-12);
}

TEST(IndexedAccessJIT, InteriorFaceVectorContractionMatchesInterpreter)
{
    TwoTetraSharedFaceMeshAccess mesh;
    auto dof_map = createTwoTetraDG_DofMap();
    spaces::H1Space space(ElementType::Tetra4, 1);

    SymbolicOptions sym_opts;
    sym_opts.jit.enable = true;

    FormCompiler compiler(sym_opts);

    const auto u = FormExpr::trialFunction(space, "u");
    const auto v = FormExpr::testFunction(space, "v");

    const Index i("i");
    const auto form = (FormExpr::constant(2.5) * jump(grad(u))(i) * jump(grad(v))(i)).dS();

    auto ir_interp = compiler.compileBilinear(form);
    auto ir_jit = compiler.compileBilinear(form);

    auto interp_kernel = std::make_shared<FormKernel>(std::move(ir_interp));

    auto jit_fallback = std::make_shared<FormKernel>(std::move(ir_jit));
    forms::JITOptions jit_opts;
    jit_opts.enable = true;
    jit_opts.optimization_level = 2;
    jit_opts.vectorize = true;
    forms::jit::JITKernelWrapper jit_kernel(jit_fallback, jit_opts);

    assembly::StandardAssembler assembler;
    assembler.setDofMap(dof_map);

    assembly::DenseMatrixView A_interp(8);
    A_interp.zero();
    (void)assembler.assembleInteriorFaces(mesh, space, space, *interp_kernel, A_interp, nullptr);

    assembly::DenseMatrixView A_jit(8);
    A_jit.zero();
    (void)assembler.assembleInteriorFaces(mesh, space, space, jit_kernel, A_jit, nullptr);

    expectDenseNear(A_jit, A_interp, 1e-12);
}
#endif

} // namespace test
} // namespace forms
} // namespace FE
} // namespace svmp


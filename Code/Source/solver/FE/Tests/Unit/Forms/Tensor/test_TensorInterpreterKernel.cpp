/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

/**
 * @file test_TensorInterpreterKernel.cpp
 * @brief End-to-end verification: tensor-calculus interpreter vs scalar-expanded interpreter
 */

#include <gtest/gtest.h>

#include "Assembly/StandardAssembler.h"
#include "Forms/FormCompiler.h"
#include "Forms/FormKernels.h"
#include "Forms/Index.h"
#include "Spaces/H1Space.h"
#include "Spaces/ProductSpace.h"
#include "Tests/Unit/Forms/FormsTestHelpers.h"

#include <memory>

namespace svmp::FE::forms::tensor {

namespace {

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

void expectDenseNear(const assembly::DenseMatrixView& A,
                     const assembly::DenseMatrixView& B,
                     Real tol)
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

TEST(TensorInterpreterKernel, CellMatrixContractionMatchesScalarExpandedInterpreter)
{
    forms::test::SingleTetraMeshAccess mesh;
    auto base = std::make_shared<spaces::H1Space>(ElementType::Tetra4, 1);
    spaces::ProductSpace space(base, 3);
    auto dof_map = makeSingleCellDofMap(static_cast<GlobalIndex>(space.dofs_per_element()));

    SymbolicOptions opts;
    opts.jit.enable = true; // allow IndexedAccess in FormIR
    FormCompiler compiler(opts);

    const auto u = FormExpr::trialFunction(space, "u");
    const auto v = FormExpr::testFunction(space, "v");
    const forms::Index i("i");
    const forms::Index j("j");
    const auto form = (grad(u)(i, j) * grad(v)(i, j)).dx();

    auto ir_scalar = compiler.compileBilinear(form);
    auto ir_tensor = compiler.compileBilinear(form);

    FormKernel kernel_scalar(std::move(ir_scalar)); // scalar-expanded via einsum() during evaluation

    FormKernel kernel_tensor(std::move(ir_tensor));
    TensorJITOptions tensor_opts;
    tensor_opts.mode = TensorLoweringMode::On; // force tensor loop interpreter
    kernel_tensor.setTensorInterpreterOptions(tensor_opts);

    assembly::StandardAssembler assembler;
    assembler.setDofMap(dof_map);

    assembly::DenseMatrixView A_scalar(static_cast<GlobalIndex>(space.dofs_per_element()));
    A_scalar.zero();
    (void)assembler.assembleMatrix(mesh, space, space, kernel_scalar, A_scalar);

    assembly::DenseMatrixView A_tensor(static_cast<GlobalIndex>(space.dofs_per_element()));
    A_tensor.zero();
    (void)assembler.assembleMatrix(mesh, space, space, kernel_tensor, A_tensor);

    expectDenseNear(A_tensor, A_scalar, 1e-12);
}

} // namespace svmp::FE::forms::tensor

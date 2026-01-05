/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

/**
 * @file test_FormKernel_Cell.cpp
 * @brief Unit tests for FE/Forms cell (dx) assembly
 */

#include <gtest/gtest.h>

#include "Assembly/GlobalSystemView.h"
#include "Assembly/StandardAssembler.h"
#include "Forms/FormCompiler.h"
#include "Forms/FormKernels.h"
#include "Spaces/H1Space.h"
#include "Tests/Unit/Forms/FormsTestHelpers.h"

namespace svmp {
namespace FE {
namespace forms {
namespace test {

TEST(FormKernelCellTest, LinearDxIntegratesBasisFunctions)
{
    SingleTetraMeshAccess mesh;
    auto dof_map = createSingleTetraDofMap();
    spaces::H1Space space(ElementType::Tetra4, 1);

    FormCompiler compiler;
    const auto v = FormExpr::testFunction(space, "v");
    const auto form = v.dx();

    auto ir = compiler.compileLinear(form);
    FormKernel kernel(std::move(ir));

    assembly::StandardAssembler assembler;
    assembler.setDofMap(dof_map);

    assembly::DenseVectorView vec(4);
    vec.zero();

    (void)assembler.assembleVector(mesh, space, kernel, vec);

    const Real V = 1.0 / 6.0;
    const Real expected = V / 4.0;

    for (GlobalIndex i = 0; i < 4; ++i) {
        EXPECT_NEAR(vec.getVectorEntry(i), expected, 1e-12);
    }
}

TEST(FormKernelCellTest, LinearDxWithConstantScaling)
{
    SingleTetraMeshAccess mesh;
    auto dof_map = createSingleTetraDofMap();
    spaces::H1Space space(ElementType::Tetra4, 1);

    FormCompiler compiler;
    const auto v = FormExpr::testFunction(space, "v");
    const auto form = (FormExpr::constant(2.0) * v).dx();

    auto ir = compiler.compileLinear(form);
    FormKernel kernel(std::move(ir));

    assembly::StandardAssembler assembler;
    assembler.setDofMap(dof_map);

    assembly::DenseVectorView vec(4);
    vec.zero();

    (void)assembler.assembleVector(mesh, space, kernel, vec);

    const Real V = 1.0 / 6.0;
    const Real expected = 2.0 * (V / 4.0);

    for (GlobalIndex i = 0; i < 4; ++i) {
        EXPECT_NEAR(vec.getVectorEntry(i), expected, 1e-12);
    }
}

TEST(FormKernelCellTest, DtBilinearRequiresTransientContextAndSignalsTemporalOrder)
{
    SingleTetraMeshAccess mesh;
    auto dof_map = createSingleTetraDofMap();
    spaces::H1Space space(ElementType::Tetra4, 1);

    FormCompiler compiler;
    const auto u = FormExpr::trialFunction(space, "u");
    const auto v = FormExpr::testFunction(space, "v");
    const auto form = (dt(u, 2) * v).dx();

    auto ir = compiler.compileBilinear(form);
    FormKernel kernel(std::move(ir));
    EXPECT_EQ(kernel.maxTemporalDerivativeOrder(), 2);

    assembly::StandardAssembler assembler;
    assembler.setDofMap(dof_map);

    assembly::DenseMatrixView mat(4);
    mat.zero();

    try {
        (void)assembler.assembleMatrix(mesh, space, space, kernel, mat);
        FAIL() << "Expected assembly to fail without a transient time-integration context";
    } catch (const svmp::FE::FEException& e) {
        const std::string msg = e.what();
        EXPECT_NE(msg.find("dt(...) operator requires a transient time-integration context"), std::string::npos);
    }
}

} // namespace test
} // namespace forms
} // namespace FE
} // namespace svmp

/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

/**
 * @file test_Einsum.cpp
 * @brief Unit tests for FE/Forms Einstein index notation lowering (einsum)
 */

#include <gtest/gtest.h>

#include "Assembly/StandardAssembler.h"
#include "Forms/Einsum.h"
#include "Forms/FormCompiler.h"
#include "Forms/FormKernels.h"
#include "Forms/Index.h"
#include "Spaces/H1Space.h"
#include "Tests/Unit/Forms/FormsTestHelpers.h"

#include <string>

namespace svmp {
namespace FE {
namespace forms {
namespace test {

namespace {

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

TEST(EinsumTest, LowersGradInnerProductToExplicitComponentSum)
{
    SingleTetraMeshAccess mesh;
    auto dof_map = createSingleTetraDofMap();
    spaces::H1Space space(ElementType::Tetra4, 1);

    const auto u = FormExpr::trialFunction(space, "u");
    const auto v = FormExpr::testFunction(space, "v");

    const auto a_ref = inner(grad(u), grad(v)).dx();

    const Index i("i");
    const auto a_idx = einsum(grad(u)(i) * grad(v)(i)).dx();

    const auto A_ref = assembleCellBilinear(a_ref, dof_map, mesh, space);
    const auto A_idx = assembleCellBilinear(a_idx, dof_map, mesh, space);

    for (GlobalIndex r = 0; r < 4; ++r) {
        for (GlobalIndex c = 0; c < 4; ++c) {
            EXPECT_NEAR(A_ref.getMatrixEntry(r, c), A_idx.getMatrixEntry(r, c), 1e-12);
        }
    }
}

TEST(EinsumTest, CompileRejectsUnloweredIndexedAccess)
{
    spaces::H1Space space(ElementType::Tetra4, 1);
    const auto u = FormExpr::trialFunction(space, "u");
    const auto v = FormExpr::testFunction(space, "v");
    const Index i("i");

    FormCompiler compiler;

    try {
        (void)compiler.compileBilinear((grad(u)(i) * grad(v)(i)).dx());
        FAIL() << "Expected FormCompiler to reject unlowered indexed access";
    } catch (const std::invalid_argument& e) {
        const std::string msg = e.what();
        EXPECT_NE(msg.find("forms::einsum"), std::string::npos);
    }
}

TEST(EinsumTest, EinsumRejectsFreeIndex)
{
    spaces::H1Space space(ElementType::Tetra4, 1);
    const auto u = FormExpr::trialFunction(space, "u");
    const Index i("i");

    try {
        (void)einsum(grad(u)(i));
        FAIL() << "Expected einsum to reject a free index";
    } catch (const std::invalid_argument& e) {
        const std::string msg = e.what();
        EXPECT_NE(msg.find("free indices"), std::string::npos);
    }
}

TEST(EinsumTest, EinsumRejectsTripleIndexUse)
{
    spaces::H1Space space(ElementType::Tetra4, 1);
    const auto u = FormExpr::trialFunction(space, "u");
    const auto v = FormExpr::testFunction(space, "v");
    const Index i("i");

    // i appears 3 times.
    const auto expr = (grad(u)(i) * grad(v)(i)) * grad(v)(i);

    try {
        (void)einsum(expr);
        FAIL() << "Expected einsum to reject an index used more than twice";
    } catch (const std::invalid_argument& e) {
        const std::string msg = e.what();
        EXPECT_NE(msg.find("exactly twice"), std::string::npos);
    }
}

} // namespace test
} // namespace forms
} // namespace FE
} // namespace svmp


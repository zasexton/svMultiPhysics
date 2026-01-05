/**
 * @file test_PrimitiveTypes.cpp
 * @brief Unit tests for newly added FE/Forms primitive mathematical types
 */

#include <gtest/gtest.h>

#include "Assembly/GlobalSystemView.h"
#include "Assembly/StandardAssembler.h"
#include "Forms/BlockForm.h"
#include "Forms/Complex.h"
#include "Forms/FormCompiler.h"
#include "Forms/FormKernels.h"
#include "Forms/Vocabulary.h"
#include "Spaces/H1Space.h"
#include "Tests/Unit/Forms/FormsTestHelpers.h"

#include <array>
#include <complex>
#include <string>

namespace svmp {
namespace FE {
namespace forms {
namespace test {

namespace {

Real singleTetraVolume()
{
    return 1.0 / 6.0;
}

Real singleTetraP1BasisIntegral()
{
    return singleTetraVolume() / 4.0;
}

assembly::DenseVectorView assembleCellLinear(const FormExpr& scalar_expr,
                                             dofs::DofMap& dof_map,
                                             const assembly::IMeshAccess& mesh,
                                             const spaces::FunctionSpace& space)
{
    FormCompiler compiler;
    const auto v = FormExpr::testFunction(space, "v");
    const auto form = (scalar_expr * v).dx();
    auto ir = compiler.compileLinear(form);
    FormKernel kernel(std::move(ir));

    assembly::StandardAssembler assembler;
    assembler.setDofMap(dof_map);

    assembly::DenseVectorView vec(static_cast<GlobalIndex>(dof_map.getNumDofs()));
    vec.zero();
    (void)assembler.assembleVector(mesh, space, kernel, vec);
    return vec;
}

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

constexpr std::size_t idx4(int i, int j, int k, int l) noexcept
{
    return static_cast<std::size_t>((((i * 3) + j) * 3 + k) * 3 + l);
}

} // namespace

TEST(PrimitiveTypes, SymmetricAndSkewTensorMatchDefinitions)
{
    SingleTetraMeshAccess mesh;
    auto dof_map = createSingleTetraDofMap();
    spaces::H1Space space(ElementType::Tetra4, 1);

    const auto A = FormExpr::coefficient(
        "A",
        [](Real, Real, Real) {
            return std::array<std::array<Real, 3>, 3>{
                std::array<Real, 3>{1.0, 2.0, 3.0},
                std::array<Real, 3>{4.0, 5.0, 6.0},
                std::array<Real, 3>{7.0, 8.0, 9.0},
            };
        });

    const auto sym01 = component(sym(A), 0, 1);
    const auto skew01 = component(skew(A), 0, 1);
    const auto sym01_wrap = component(SymmetricTensor(A), 0, 1);
    const auto skew01_wrap = component(SkewTensor(A), 0, 1);

    const Real expected_sym01 = 0.5 * (2.0 + 4.0);
    const Real expected_skew01 = 0.5 * (2.0 - 4.0);
    const Real scale = singleTetraP1BasisIntegral();

    {
        auto vec = assembleCellLinear(sym01, dof_map, mesh, space);
        for (GlobalIndex i = 0; i < 4; ++i) {
            EXPECT_NEAR(vec.getVectorEntry(i), expected_sym01 * scale, 1e-12);
        }
    }
    {
        auto vec = assembleCellLinear(skew01, dof_map, mesh, space);
        for (GlobalIndex i = 0; i < 4; ++i) {
            EXPECT_NEAR(vec.getVectorEntry(i), expected_skew01 * scale, 1e-12);
        }
    }
    {
        auto vec = assembleCellLinear(sym01_wrap, dof_map, mesh, space);
        for (GlobalIndex i = 0; i < 4; ++i) {
            EXPECT_NEAR(vec.getVectorEntry(i), expected_sym01 * scale, 1e-12);
        }
    }
    {
        auto vec = assembleCellLinear(skew01_wrap, dof_map, mesh, space);
        for (GlobalIndex i = 0; i < 4; ++i) {
            EXPECT_NEAR(vec.getVectorEntry(i), expected_skew01 * scale, 1e-12);
        }
    }
}

TEST(PrimitiveTypes, FourthOrderTensorDoubleContractionIdentityActsAsIdentity)
{
    SingleTetraMeshAccess mesh;
    auto dof_map = createSingleTetraDofMap();
    spaces::H1Space space(ElementType::Tetra4, 1);

    const auto A = FormExpr::coefficient(
        "A",
        [](Real, Real, Real) {
            return std::array<std::array<Real, 3>, 3>{
                std::array<Real, 3>{1.0, 2.0, 0.0},
                std::array<Real, 3>{3.0, 4.0, 0.0},
                std::array<Real, 3>{0.0, 0.0, 5.0},
            };
        });

    const auto I4 = FormExpr::coefficient(
        "I4",
        [](Real, Real, Real) {
            std::array<Real, 81> C{};
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    for (int k = 0; k < 3; ++k) {
                        for (int l = 0; l < 3; ++l) {
                            C[idx4(i, j, k, l)] = (i == k && j == l) ? 1.0 : 0.0;
                        }
                    }
                }
            }
            return C;
        });

    const auto B = doubleContraction(I4, A);
    const auto err = norm(B - A);

    auto vec = assembleCellLinear(err, dof_map, mesh, space);
    for (GlobalIndex i = 0; i < 4; ++i) {
        EXPECT_NEAR(vec.getVectorEntry(i), 0.0, 1e-12);
    }

    const auto B2 = doubleContraction(FormExpr::constant(2.0) * I4, A);
    const auto err2 = norm(B2 - (FormExpr::constant(2.0) * A));
    auto vec2 = assembleCellLinear(err2, dof_map, mesh, space);
    for (GlobalIndex i = 0; i < 4; ++i) {
        EXPECT_NEAR(vec2.getVectorEntry(i), 0.0, 1e-12);
    }
}

TEST(PrimitiveTypes, ComplexBlockLiftingProducesExpected2x2Structure)
{
    SingleTetraMeshAccess mesh;
    auto dof_map = createSingleTetraDofMap();
    spaces::H1Space space(ElementType::Tetra4, 1);

    const auto u = FormExpr::trialFunction(space, "u");
    const auto v = FormExpr::testFunction(space, "v");

    const auto a_re = (u * v).dx();
    const auto a_im = (FormExpr::constant(2.0) * u * v).dx();

    const ComplexBilinearForm a{a_re, a_im};
    const auto blocks = toRealBlock2x2(a);

    const auto A_re = assembleCellBilinear(a_re, dof_map, mesh, space);
    const auto A_im = assembleCellBilinear(a_im, dof_map, mesh, space);

    const auto A00 = assembleCellBilinear(blocks.block(0, 0), dof_map, mesh, space);
    const auto A01 = assembleCellBilinear(blocks.block(0, 1), dof_map, mesh, space);
    const auto A10 = assembleCellBilinear(blocks.block(1, 0), dof_map, mesh, space);
    const auto A11 = assembleCellBilinear(blocks.block(1, 1), dof_map, mesh, space);

    for (GlobalIndex r = 0; r < 4; ++r) {
        for (GlobalIndex c = 0; c < 4; ++c) {
            EXPECT_NEAR(A00.getMatrixEntry(r, c), A_re.getMatrixEntry(r, c), 1e-12);
            EXPECT_NEAR(A11.getMatrixEntry(r, c), A_re.getMatrixEntry(r, c), 1e-12);
            EXPECT_NEAR(A10.getMatrixEntry(r, c), A_im.getMatrixEntry(r, c), 1e-12);
            EXPECT_NEAR(A01.getMatrixEntry(r, c), -A_im.getMatrixEntry(r, c), 1e-12);
        }
    }
}

TEST(PrimitiveTypes, BlockFormContainersIndexAndStoreBlocks)
{
    spaces::H1Space space(ElementType::Tetra4, 1);
    const auto u = FormExpr::trialFunction(space, "u");
    const auto v = FormExpr::testFunction(space, "v");
    const auto a = (u * v).dx();
    const auto l = v.dx();

    BlockBilinearForm A(2, 3);
    EXPECT_FALSE(A.hasBlock(1, 2));
    A.setBlock(1, 2, a);
    EXPECT_TRUE(A.hasBlock(1, 2));
    EXPECT_EQ(A.block(1, 2).toString(), a.toString());

    BlockLinearForm b(2);
    EXPECT_FALSE(b.hasBlock(0));
    b.setBlock(0, l);
    EXPECT_TRUE(b.hasBlock(0));
    EXPECT_EQ(b.block(0).toString(), l.toString());

    EXPECT_THROW((void)A.block(2, 0), std::out_of_range);
    EXPECT_THROW((void)A.block(0, 3), std::out_of_range);
    EXPECT_THROW((void)b.block(2), std::out_of_range);
}

} // namespace test
} // namespace forms
} // namespace FE
} // namespace svmp


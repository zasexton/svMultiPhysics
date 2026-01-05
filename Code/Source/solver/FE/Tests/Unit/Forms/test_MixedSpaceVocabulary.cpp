/**
 * @file test_MixedSpaceVocabulary.cpp
 * @brief Unit tests for MixedSpace-related Forms vocabulary helpers
 */

#include <gtest/gtest.h>

#include "Forms/BlockForm.h"
#include "Forms/FormCompiler.h"
#include "Forms/Vocabulary.h"

#include "Spaces/H1Space.h"
#include "Spaces/L2Space.h"
#include "Spaces/MixedSpace.h"

#include <stdexcept>

using svmp::FE::ElementType;

namespace svmp {
namespace FE {
namespace forms {
namespace test {

TEST(MixedSpaceVocabulary, TrialAndTestFunctionsBindToComponentSpaces)
{
    auto V = std::make_shared<spaces::H1Space>(ElementType::Tetra4, 1);
    auto Q = std::make_shared<spaces::L2Space>(ElementType::Tetra4, 1);

    spaces::MixedSpace W;
    W.add_component("u", V);
    W.add_component("p", Q);

    const auto U = TrialFunctions(W);
    const auto Vt = TestFunctions(W);

    ASSERT_EQ(U.size(), 2u);
    ASSERT_EQ(Vt.size(), 2u);

    EXPECT_EQ(U[0].toString(), "u");
    EXPECT_EQ(U[1].toString(), "p");
    EXPECT_EQ(Vt[0].toString(), "v0");
    EXPECT_EQ(Vt[1].toString(), "v1");

    ASSERT_NE(U[0].node(), nullptr);
    ASSERT_NE(U[1].node(), nullptr);
    ASSERT_NE(Vt[0].node(), nullptr);
    ASSERT_NE(Vt[1].node(), nullptr);

    ASSERT_NE(U[0].node()->spaceSignature(), nullptr);
    ASSERT_NE(U[1].node()->spaceSignature(), nullptr);
    ASSERT_NE(Vt[0].node()->spaceSignature(), nullptr);
    ASSERT_NE(Vt[1].node()->spaceSignature(), nullptr);

    EXPECT_EQ(U[0].node()->spaceSignature()->space_type, V->space_type());
    EXPECT_EQ(U[1].node()->spaceSignature()->space_type, Q->space_type());
    EXPECT_EQ(Vt[0].node()->spaceSignature()->space_type, V->space_type());
    EXPECT_EQ(Vt[1].node()->spaceSignature()->space_type, Q->space_type());

    EXPECT_THROW((void)TrialFunctions(W, {"only_one_name"}), std::invalid_argument);
    EXPECT_THROW((void)TestFunctions(W, {"only_one_name"}), std::invalid_argument);
}

TEST(MixedSpaceVocabulary, BlockBilinearCompilesPerBlock)
{
    auto V = std::make_shared<spaces::H1Space>(ElementType::Tetra4, 1);
    auto Q = std::make_shared<spaces::L2Space>(ElementType::Tetra4, 1);

    spaces::MixedSpace W;
    W.add_component("u", V);
    W.add_component("p", Q);

    const auto U = TrialFunctions(W);
    const auto Vt = TestFunctions(W);

    BlockBilinearForm A(/*num_test_fields=*/2, /*num_trial_fields=*/2);
    A.setBlock(0, 0, (U[0] * Vt[0]).dx());
    A.setBlock(0, 1, (U[1] * Vt[0]).dx());
    A.setBlock(1, 0, (U[0] * Vt[1]).dx());
    A.setBlock(1, 1, (U[1] * Vt[1]).dx());

    FormCompiler compiler;
    const auto blocks = compiler.compileBilinear(A);

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

} // namespace test
} // namespace forms
} // namespace FE
} // namespace svmp


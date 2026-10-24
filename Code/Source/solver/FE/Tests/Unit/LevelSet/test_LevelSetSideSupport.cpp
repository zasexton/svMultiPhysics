/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include <gtest/gtest.h>

#include "LevelSet/LevelSetSideSupport.h"
#include "Systems/FESystem.h"

#include "Tests/Unit/Forms/FormsTestHelpers.h"

#include <memory>
#include <vector>

namespace {

using svmp::FE::level_set::LevelSetSideSupportQuery;
using svmp::FE::level_set::LevelSetSupportSide;
using svmp::FE::level_set::levelSetSideSupportVertices;
using svmp::FE::level_set::vertexOnLevelSetSide;

} // namespace

TEST(LevelSetSideSupport, VertexSidesAreInclusiveOfTheIsovalue)
{
    EXPECT_TRUE(vertexOnLevelSetSide(-1.0, 0.0, LevelSetSupportSide::Negative));
    EXPECT_TRUE(vertexOnLevelSetSide(0.0, 0.0, LevelSetSupportSide::Negative));
    EXPECT_FALSE(vertexOnLevelSetSide(0.5, 0.0, LevelSetSupportSide::Negative));
    EXPECT_TRUE(vertexOnLevelSetSide(0.5, 0.0, LevelSetSupportSide::Positive));
    EXPECT_TRUE(vertexOnLevelSetSide(0.25, 0.25, LevelSetSupportSide::Positive));
    EXPECT_FALSE(vertexOnLevelSetSide(-0.5, 0.0, LevelSetSupportSide::Positive));
    EXPECT_TRUE(vertexOnLevelSetSide(-0.5, 0.0, LevelSetSupportSide::Everywhere));
}

TEST(LevelSetSideSupport, WithoutANativeMeshTheSupportIsTheVerticesOnTheSide)
{
    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraMeshAccess>();
    svmp::FE::systems::FESystem sys(mesh);
    // Two components per vertex: only the first is the level set.
    const std::vector<svmp::FE::Real> phi{-1.0, 9.0, 0.0, 9.0, 0.5, 9.0, 2.0, 9.0};
    LevelSetSideSupportQuery query{};
    query.level_set = {phi.data(), 2u, 4u};
    query.isovalue = 0.0;
    query.side = LevelSetSupportSide::Negative;
    query.message_prefix = "unit";
    EXPECT_EQ(levelSetSideSupportVertices(sys, query, 4),
              (std::vector<unsigned char>{1, 1, 0, 0}));
    query.side = LevelSetSupportSide::Positive;
    EXPECT_EQ(levelSetSideSupportVertices(sys, query, 4),
              (std::vector<unsigned char>{0, 1, 1, 1}));
}

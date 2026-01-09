/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include <gtest/gtest.h>

#include "Physics/Core/Domain.h"
#include "Physics/Core/ParameterSchema.h"

#include "FE/Core/FEException.h"
#include "FE/Systems/SystemState.h"

namespace svmp {
namespace Physics {
namespace test {

TEST(MarkerSet, EmptyIsAll)
{
    MarkerSet s;
    EXPECT_TRUE(s.isAll());
    EXPECT_TRUE(s.contains(0));
    EXPECT_TRUE(s.contains(123));
    EXPECT_TRUE(s.contains(-7));
}

TEST(MarkerSet, NegativeMarkerIsAll)
{
    MarkerSet s;
    s.markers = {kAllBoundaryMarkers};
    EXPECT_TRUE(s.isAll());
    EXPECT_TRUE(s.contains(5));
}

TEST(MarkerSet, ContainsSpecificMarkers)
{
    MarkerSet s;
    s.markers = {1, 3, 7};
    EXPECT_FALSE(s.isAll());
    EXPECT_TRUE(s.contains(1));
    EXPECT_TRUE(s.contains(7));
    EXPECT_FALSE(s.contains(2));
}

TEST(ParameterSchema, ValidateHonorsDefaults)
{
    ParameterSchema schema;

    FE::params::Spec s;
    s.key = "alpha";
    s.type = FE::params::ValueType::Real;
    s.required = true;
    s.default_value = FE::params::Value{FE::Real(2.0)};
    schema.add(std::move(s), "test");

    FE::systems::SystemStateView state;
    EXPECT_NO_THROW(schema.validate(state));
}

TEST(ParameterSchema, ValidateThrowsWhenMissingRequired)
{
    ParameterSchema schema;

    FE::params::Spec s;
    s.key = "alpha";
    s.type = FE::params::ValueType::Real;
    s.required = true;
    schema.add(std::move(s), "test");

    FE::systems::SystemStateView state;
    EXPECT_THROW(schema.validate(state), FE::InvalidArgumentException);
}

TEST(ParameterSchema, FindAndClear)
{
    ParameterSchema schema;

    FE::params::Spec s;
    s.key = "alpha";
    s.type = FE::params::ValueType::Real;
    s.required = false;
    schema.add(std::move(s), "test");

    ASSERT_NE(schema.find("alpha"), nullptr);
    EXPECT_EQ(schema.find("alpha")->type, FE::params::ValueType::Real);

    schema.clear();
    EXPECT_TRUE(schema.specs().empty());
    EXPECT_EQ(schema.find("alpha"), nullptr);
}

} // namespace test
} // namespace Physics
} // namespace svmp


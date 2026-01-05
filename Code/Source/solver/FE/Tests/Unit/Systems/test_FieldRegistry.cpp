/**
 * @file test_FieldRegistry.cpp
 * @brief Unit tests for Systems FieldRegistry
 */

#include <gtest/gtest.h>

#include "Systems/FieldRegistry.h"
#include "Spaces/H1Space.h"

using svmp::FE::ElementType;
using svmp::FE::INVALID_FIELD_ID;
using svmp::FE::FieldId;
using svmp::FE::systems::FieldRegistry;
using svmp::FE::systems::FieldSpec;
using svmp::FE::spaces::H1Space;

TEST(FieldRegistry, AddsFindsAndRejectsDuplicates)
{
    FieldRegistry reg;
    auto space = std::make_shared<H1Space>(ElementType::Quad4, /*order=*/1);

    FieldId u = reg.add(FieldSpec{.name = "u", .space = space, .components = 1});
    EXPECT_NE(u, INVALID_FIELD_ID);
    EXPECT_TRUE(reg.has(u));
    EXPECT_EQ(reg.findByName("u"), u);
    EXPECT_EQ(reg.get(u).name, "u");

    EXPECT_THROW(reg.add(FieldSpec{.name = "u", .space = space, .components = 1}), svmp::FE::FEException);
}


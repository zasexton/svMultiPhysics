/**
 * @file test_FieldRegistry.cpp
 * @brief Unit tests for Systems FieldRegistry
 */

#include <gtest/gtest.h>

#include "Systems/FieldRegistry.h"
#include "Spaces/H1Space.h"

#include <string>
#include <vector>

using svmp::FE::ElementType;
using svmp::FE::INVALID_FIELD_ID;
using svmp::FE::FieldId;
using svmp::FE::systems::FieldRegistry;
using svmp::FE::systems::FieldSpec;
using svmp::FE::systems::FieldShapeKind;
using svmp::FE::systems::INVALID_STATE_GROUP_ID;
using svmp::FE::systems::StateGroupKind;
using svmp::FE::systems::StateGroupSpec;
using svmp::FE::systems::StateGroupSumConstraint;
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

TEST(FieldRegistry, AddsAndQueriesStateGroups)
{
    FieldRegistry reg;
    auto space = std::make_shared<H1Space>(ElementType::Quad4, /*order=*/1);

    FieldId c0 = reg.add(FieldSpec{.name = "c0", .space = space, .components = 1});
    FieldId c1 = reg.add(FieldSpec{.name = "c1", .space = space, .components = 1});

    auto group = reg.addStateGroup(StateGroupSpec{
        .name = "species",
        .kind = StateGroupKind::ConservedComponents,
        .shape = FieldShapeKind::IndexedScalarSet,
        .fields = {c0, c1},
        .component_count = 2,
        .conserved_quantity_name = "total",
        .sum_constraint = StateGroupSumConstraint{.target = 1.0, .tolerance = 1.0e-12, .equality = true},
        .analysis_tags = {"bounded", "transport"}});

    EXPECT_NE(group, INVALID_STATE_GROUP_ID);
    EXPECT_TRUE(reg.hasStateGroup(group));
    EXPECT_EQ(reg.findStateGroupByName("species"), group);
    ASSERT_EQ(reg.stateGroupCount(), 1u);

    const auto& record = reg.getStateGroup(group);
    EXPECT_EQ(record.name, "species");
    EXPECT_EQ(record.kind, StateGroupKind::ConservedComponents);
    EXPECT_EQ(record.shape, FieldShapeKind::IndexedScalarSet);
    EXPECT_EQ(record.fields, std::vector<FieldId>({c0, c1}));
    EXPECT_EQ(record.field_names, std::vector<std::string>({"c0", "c1"}));
    EXPECT_EQ(record.component_count, 2);
    ASSERT_TRUE(record.conserved_quantity_name.has_value());
    EXPECT_EQ(*record.conserved_quantity_name, "total");
    ASSERT_TRUE(record.sum_constraint.has_value());
    EXPECT_DOUBLE_EQ(record.sum_constraint->target, 1.0);
    EXPECT_EQ(record.analysis_tags.size(), 2u);
}

TEST(FieldRegistry, ResolvesStateGroupsByFieldName)
{
    FieldRegistry reg;
    auto space = std::make_shared<H1Space>(ElementType::Quad4, /*order=*/1);

    FieldId a = reg.add(FieldSpec{.name = "a", .space = space, .components = 1});
    FieldId b = reg.add(FieldSpec{.name = "b", .space = space, .components = 1});

    auto group = reg.addStateGroup(StateGroupSpec{
        .name = "named",
        .kind = StateGroupKind::IndependentFields,
        .shape = FieldShapeKind::MixedFieldGroup,
        .field_names = {"a", "b"}});

    const auto& record = reg.getStateGroup(group);
    EXPECT_EQ(record.fields, std::vector<FieldId>({a, b}));
    EXPECT_EQ(record.component_count, 2);
}

TEST(FieldRegistry, RejectsInvalidStateGroups)
{
    FieldRegistry reg;
    auto space = std::make_shared<H1Space>(ElementType::Quad4, /*order=*/1);
    FieldId field = reg.add(FieldSpec{.name = "field", .space = space, .components = 1});

    EXPECT_THROW(reg.addStateGroup(StateGroupSpec{}), svmp::FE::FEException);
    EXPECT_THROW(reg.addStateGroup(StateGroupSpec{.name = "empty"}), svmp::FE::FEException);
    EXPECT_THROW(reg.addStateGroup(StateGroupSpec{.name = "unknown", .field_names = {"missing"}}),
                 svmp::FE::FEException);
    EXPECT_THROW(reg.addStateGroup(StateGroupSpec{.name = "bad_bounds",
                                                  .fields = {field},
                                                  .bounds = {.lower = 2.0, .upper = 1.0}}),
                 svmp::FE::FEException);

    auto group = reg.addStateGroup(StateGroupSpec{.name = "ok", .fields = {field}});
    EXPECT_NE(group, INVALID_STATE_GROUP_ID);
    EXPECT_THROW(reg.addStateGroup(StateGroupSpec{.name = "ok", .fields = {field}}),
                 svmp::FE::FEException);
}

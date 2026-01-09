/**
 * @file test_OperatorRegistry.cpp
 * @brief Unit tests for Systems OperatorRegistry
 */

#include <gtest/gtest.h>

#include "Systems/OperatorRegistry.h"

#include "Assembly/Assembler.h"
#include "Assembly/AssemblyKernel.h"
#include "Systems/GlobalKernel.h"

#include <memory>
#include <string>

using svmp::FE::FieldId;
using svmp::FE::INVALID_FIELD_ID;
using svmp::FE::assembly::MassKernel;
using svmp::FE::systems::AssemblyRequest;
using svmp::FE::systems::FESystem;
using svmp::FE::systems::GlobalKernel;
using svmp::FE::systems::OperatorRegistry;
using svmp::FE::systems::OperatorTag;
using svmp::FE::systems::SystemStateView;

namespace {

class DummyGlobalKernel final : public GlobalKernel {
public:
    [[nodiscard]] std::string name() const override { return "DummyGlobalKernel"; }

    svmp::FE::assembly::AssemblyResult assemble(const FESystem&,
                                                const AssemblyRequest&,
                                                const SystemStateView&,
                                                svmp::FE::assembly::GlobalSystemView*,
                                                svmp::FE::assembly::GlobalSystemView*) override
    {
        return {};
    }
};

} // namespace

TEST(OperatorRegistry, OperatorRegistry_AddOperator_CreatesEmptyDefinition)
{
    OperatorRegistry reg;
    reg.addOperator("mass");

    const auto& def = reg.get("mass");
    EXPECT_EQ(def.tag, "mass");
    EXPECT_TRUE(def.cells.empty());
    EXPECT_TRUE(def.boundary.empty());
    EXPECT_TRUE(def.interior.empty());
    EXPECT_TRUE(def.global.empty());
}

TEST(OperatorRegistry, OperatorRegistry_Has_ReturnsTrueForExistingOperator)
{
    OperatorRegistry reg;
    reg.addOperator("op");
    EXPECT_TRUE(reg.has("op"));
}

TEST(OperatorRegistry, OperatorRegistry_Has_ReturnsFalseForNonExistentOperator)
{
    OperatorRegistry reg;
    EXPECT_FALSE(reg.has("missing"));
}

TEST(OperatorRegistry, OperatorRegistry_Get_ReturnsCorrectDefinition)
{
    OperatorRegistry reg;
    reg.addOperator("op");

    auto& def = reg.get("op");
    def.cells.push_back({/*test_field=*/FieldId{0}, /*trial_field=*/FieldId{0}, std::make_shared<MassKernel>(1.0)});
    def.boundary.push_back({/*marker=*/7, /*test_field=*/FieldId{0}, /*trial_field=*/FieldId{0}, std::make_shared<MassKernel>(2.0)});
    def.interior.push_back({/*test_field=*/FieldId{0}, /*trial_field=*/FieldId{0}, std::make_shared<MassKernel>(3.0)});
    def.global.push_back(std::make_shared<DummyGlobalKernel>());

    const auto& got = reg.get("op");
    ASSERT_EQ(got.cells.size(), 1u);
    ASSERT_EQ(got.boundary.size(), 1u);
    ASSERT_EQ(got.interior.size(), 1u);
    ASSERT_EQ(got.global.size(), 1u);
    EXPECT_EQ(got.cells[0].test_field, FieldId{0});
    EXPECT_EQ(got.boundary[0].marker, 7);
    EXPECT_EQ(got.global[0]->name(), "DummyGlobalKernel");
}

TEST(OperatorRegistry, OperatorRegistry_Get_ThrowsForNonExistentOperator)
{
    OperatorRegistry reg;
    EXPECT_THROW((void)reg.get("missing"), svmp::FE::InvalidArgumentException);
}

TEST(OperatorRegistry, OperatorRegistry_List_ReturnsAllOperatorTags)
{
    OperatorRegistry reg;
    reg.addOperator("b");
    reg.addOperator("a");
    reg.addOperator("c");

    const auto tags = reg.list();
    ASSERT_EQ(tags.size(), 3u);
    EXPECT_EQ(tags[0], "a");
    EXPECT_EQ(tags[1], "b");
    EXPECT_EQ(tags[2], "c");
}

TEST(OperatorRegistry, OperatorRegistry_AddOperator_RejectsDuplicateTag)
{
    OperatorRegistry reg;
    reg.addOperator("op");
    EXPECT_THROW(reg.addOperator("op"), svmp::FE::InvalidArgumentException);
}

TEST(OperatorRegistry, OperatorRegistry_EmptyRegistry_ListReturnsEmpty)
{
    OperatorRegistry reg;
    const auto tags = reg.list();
    EXPECT_TRUE(tags.empty());
}

TEST(OperatorRegistry, OperatorRegistry_Get_ConstAndNonConstVersionsConsistent)
{
    OperatorRegistry reg;
    reg.addOperator("op");

    auto& def = reg.get("op");
    def.cells.push_back({/*test_field=*/FieldId{1}, /*trial_field=*/FieldId{2}, std::make_shared<MassKernel>(1.0)});

    const OperatorRegistry& cref = reg;
    const auto& def_const = cref.get("op");
    EXPECT_EQ(def_const.tag, def.tag);
    ASSERT_EQ(def_const.cells.size(), def.cells.size());
    EXPECT_EQ(def_const.cells[0].test_field, def.cells[0].test_field);
    EXPECT_EQ(def_const.cells[0].trial_field, def.cells[0].trial_field);
}


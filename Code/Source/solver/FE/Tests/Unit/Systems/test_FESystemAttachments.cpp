/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include <gtest/gtest.h>

#include "Core/FEException.h"
#include "Systems/FESystem.h"

#include "Tests/Unit/Forms/FormsTestHelpers.h"

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace {

using svmp::FE::systems::FESystem;

std::shared_ptr<svmp::FE::forms::test::SingleTetraMeshAccess> tetraMesh()
{
    return std::make_shared<svmp::FE::forms::test::SingleTetraMeshAccess>();
}

/// Counts constructions and destructions of its objects.
struct CountedRecords {
    static inline int constructed = 0;
    static inline int destroyed = 0;

    CountedRecords() { ++constructed; }
    ~CountedRecords() { ++destroyed; }
    CountedRecords(const CountedRecords&) = delete;
    CountedRecords& operator=(const CountedRecords&) = delete;

    std::vector<int> values{};
};

struct OtherRecords {
    std::string name{"other"};
};

/// An attachment whose destructor records the order of destruction.
struct DestructionOrder {
    std::vector<std::string>* log{nullptr};
    std::string label{};
    ~DestructionOrder()
    {
        if (log != nullptr) {
            log->push_back(label);
        }
    }
};

struct FirstOrderProbe : DestructionOrder {};
struct SecondOrderProbe : DestructionOrder {};

/// The message of the InvalidArgumentException that @p call throws.
std::string invalidArgumentMessage(const std::function<void()>& call)
{
    try {
        call();
    } catch (const svmp::FE::InvalidArgumentException& error) {
        return error.what();
    }
    return {};
}

} // namespace

TEST(FESystemAttachments, AttachmentIsCreatedOnceAndFoundByType)
{
    FESystem system(tetraMesh());
    const FESystem& const_system = system;
    EXPECT_EQ(system.findAttachment<CountedRecords>(), nullptr);
    EXPECT_EQ(const_system.findAttachment<CountedRecords>(), nullptr);

    CountedRecords::constructed = 0;
    auto& records = system.attachment<CountedRecords>();
    EXPECT_EQ(CountedRecords::constructed, 1);
    records.values.push_back(7);

    auto& again = system.attachment<CountedRecords>();
    EXPECT_EQ(&again, &records);
    EXPECT_EQ(CountedRecords::constructed, 1);
    EXPECT_EQ(system.findAttachment<CountedRecords>(), &records);
    ASSERT_NE(const_system.findAttachment<CountedRecords>(), nullptr);
    EXPECT_EQ(const_system.findAttachment<CountedRecords>()->values,
              std::vector<int>{7});
}

TEST(FESystemAttachments, EachTypeHasItsOwnAttachment)
{
    FESystem system(tetraMesh());
    auto& counted = system.attachment<CountedRecords>();
    EXPECT_EQ(system.findAttachment<OtherRecords>(), nullptr);
    auto& other = system.attachment<OtherRecords>();
    EXPECT_EQ(other.name, "other");
    EXPECT_NE(static_cast<void*>(&counted), static_cast<void*>(&other));
    EXPECT_EQ(system.findAttachment<CountedRecords>(), &counted);
    EXPECT_EQ(system.findAttachment<OtherRecords>(), &other);

    // Attachments of different systems are distinct.
    FESystem second(tetraMesh());
    EXPECT_EQ(second.findAttachment<CountedRecords>(), nullptr);
    EXPECT_NE(&second.attachment<CountedRecords>(), &counted);
}

TEST(FESystemAttachments, AttachmentsLiveAsLongAsTheSystemAndMoveWithIt)
{
    CountedRecords::constructed = 0;
    CountedRecords::destroyed = 0;
    {
        auto system = std::make_unique<FESystem>(tetraMesh());
        auto& records = system->attachment<CountedRecords>();
        records.values = {1, 2, 3};
        EXPECT_EQ(CountedRecords::destroyed, 0);

        FESystem moved(std::move(*system));
        EXPECT_EQ(CountedRecords::destroyed, 0);
        ASSERT_NE(moved.findAttachment<CountedRecords>(), nullptr);
        EXPECT_EQ(moved.findAttachment<CountedRecords>(), &records);
        EXPECT_EQ(moved.findAttachment<CountedRecords>()->values,
                  (std::vector<int>{1, 2, 3}));

        system.reset();
        EXPECT_EQ(CountedRecords::destroyed, 0);
    }
    EXPECT_EQ(CountedRecords::constructed, 1);
    EXPECT_EQ(CountedRecords::destroyed, 1);
}

TEST(FESystemAttachments, EveryAttachmentIsDestroyedWithTheSystem)
{
    std::vector<std::string> log;
    {
        FESystem system(tetraMesh());
        auto& first = system.attachment<FirstOrderProbe>();
        first.log = &log;
        first.label = "first";
        auto& second = system.attachment<SecondOrderProbe>();
        second.log = &log;
        second.label = "second";
        EXPECT_TRUE(log.empty());
    }
    ASSERT_EQ(log.size(), 2u);
    EXPECT_NE(log[0], log[1]);
}

TEST(FESystemAttachments, HasFieldByIdentifier)
{
    FESystem system(tetraMesh());
    EXPECT_FALSE(system.hasField(svmp::FE::FieldId{0}));
    EXPECT_FALSE(system.hasField(svmp::FE::INVALID_FIELD_ID));
}

TEST(FESystemAttachments, FrozenMeshNormalDeclarationsRejectChanges)
{
    using svmp::FE::systems::MeshNormalBoundaryConstraintDeclaration;
    FESystem system(tetraMesh());
    const auto declare = [&] {
        system.declareMeshNormalBoundaryConstraint(
            MeshNormalBoundaryConstraintDeclaration{});
    };
    const auto bind = [&] {
        system.bindMeshNormalBoundaryConstraintConsumer(
            svmp::FE::INVALID_FIELD_ID, 1, "op", "mesh row", std::nullopt);
    };
    const std::string frozen_declaration =
        "declarations cannot change after mesh-boundary history";
    const std::string frozen_binding =
        "consumer provenance cannot change after mesh-boundary history";

    // Before the freeze, the declaration checks themselves reject these.
    const auto declare_before = invalidArgumentMessage(declare);
    const auto bind_before = invalidArgumentMessage(bind);
    ASSERT_FALSE(declare_before.empty());
    ASSERT_FALSE(bind_before.empty());
    EXPECT_EQ(declare_before.find(frozen_declaration), std::string::npos);
    EXPECT_EQ(bind_before.find(frozen_binding), std::string::npos);

    system.freezeMeshNormalBoundaryConstraintDeclarations();
    EXPECT_NE(invalidArgumentMessage(declare).find(frozen_declaration),
              std::string::npos);
    EXPECT_NE(invalidArgumentMessage(bind).find(frozen_binding),
              std::string::npos);
    EXPECT_TRUE(system.meshNormalBoundaryConstraints().empty());
    EXPECT_FALSE(system.meshBoundaryHistoryTransactionActive());
}

/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include <gtest/gtest.h>

#include "Core/FEException.h"
#include "Systems/AcceptedStageRecords.h"
#include "Systems/FESystem.h"

#include "Tests/Unit/Forms/FormsTestHelpers.h"

#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using svmp::FE::systems::AcceptedStageRecordStore;
using svmp::FE::systems::AcceptedStageRecords;
using svmp::FE::systems::FESystem;
using svmp::FE::systems::acceptedStageRecords;
using svmp::FE::systems::runCollectiveRecordPhase;
namespace record_support = svmp::FE::systems::record_support;

struct TestDeclaration {
    int marker{-1};
    std::string owner{};
};

struct TestRecord {
    int marker{-1};
    int step{0};
    double value{0.0};
};

using TestStore = AcceptedStageRecordStore<TestDeclaration, TestRecord>;
struct OtherTag {};
using OtherStore =
    AcceptedStageRecordStore<TestDeclaration, TestRecord, OtherTag>;

std::shared_ptr<svmp::FE::forms::test::SingleTetraMeshAccess> tetraMesh()
{
    return std::make_shared<svmp::FE::forms::test::SingleTetraMeshAccess>();
}

/// A one-rank collective: every phase agrees locally.
record_support::MeshBoundaryCollectiveContext serialCollective()
{
    record_support::MeshBoundaryCollectiveContext collective{};
    collective.size = 1;
    return collective;
}

std::vector<TestRecord> group(int step)
{
    return {TestRecord{.marker = 1, .step = step, .value = 0.5 * step},
            TestRecord{.marker = 2, .step = step, .value = -1.0 * step}};
}

} // namespace

TEST(AcceptedStageRecords, StoreIsAnAttachmentSharedByHandlesAndKeyedByTag)
{
    FESystem system(tetraMesh());
    const FESystem& const_system = system;
    EXPECT_FALSE(acceptedStageRecords<TestStore>(const_system).exists());
    EXPECT_TRUE(acceptedStageRecords<TestStore>(const_system).declarations().empty());
    EXPECT_TRUE(acceptedStageRecords<TestStore>(const_system).history().empty());

    acceptedStageRecords<TestStore>(system).declare(
        TestDeclaration{.marker = 3, .owner = "first"});
    auto second_handle = acceptedStageRecords<TestStore>(system);
    ASSERT_EQ(second_handle.declarations().size(), 1u);
    EXPECT_EQ(second_handle.declarations()[0].owner, "first");
    EXPECT_EQ(&second_handle.store(), system.findAttachment<TestStore>());

    const auto view = acceptedStageRecords<TestStore>(const_system);
    ASSERT_TRUE(view.exists());
    EXPECT_EQ(view.declarations().size(), 1u);

    // A different tag is a different family.
    EXPECT_EQ(system.findAttachment<OtherStore>(), nullptr);
    EXPECT_TRUE(acceptedStageRecords<OtherStore>(system).declarations().empty());
    EXPECT_NE(static_cast<const void*>(system.findAttachment<OtherStore>()),
              static_cast<const void*>(system.findAttachment<TestStore>()));
}

TEST(AcceptedStageRecords, StageReplacesThePendingGroupAndDiscardEndsTheAttempt)
{
    TestStore store;
    AcceptedStageRecords<TestStore> records(store);
    records.declare(TestDeclaration{.marker = 1, .owner = "a"});
    records.declare(TestDeclaration{.marker = 2, .owner = "b"});
    EXPECT_FALSE(records.declarationsFrozen());
    records.freezeDeclarations();
    EXPECT_TRUE(records.declarationsFrozen());

    records.beginTransaction();
    EXPECT_TRUE(records.transactionActive());
    records.stagePending(group(1));
    records.endTransaction();
    EXPECT_FALSE(records.transactionActive());
    ASSERT_EQ(records.pending().size(), 2u);
    EXPECT_EQ(records.pending()[1].value, -1.0);

    // A second stage of the same attempt replaces the group.
    records.stagePending(group(2));
    ASSERT_EQ(records.pending().size(), 2u);
    EXPECT_EQ(records.pending()[0].step, 2);

    // Owners bind more evidence into the pending records.
    records.mutablePending()[0].value = 7.0;
    EXPECT_EQ(store.pending[0].value, 7.0);

    records.beginTransaction();
    records.discardPending();
    EXPECT_TRUE(records.pending().empty());
    EXPECT_FALSE(records.transactionActive());
    EXPECT_TRUE(records.history().empty());
    EXPECT_TRUE(records.declarationsFrozen());
}

TEST(AcceptedStageRecords, PublishMovesThePendingGroupToTheHistory)
{
    TestStore store;
    AcceptedStageRecords<TestStore> records(store);
    const auto collective = serialCollective();

    records.stagePending(group(1));
    EXPECT_EQ(records.publishPendingCollectively(
                  collective, "test/publish", "remote failure"),
              0u);
    EXPECT_TRUE(records.pending().empty());
    EXPECT_FALSE(records.transactionActive());
    ASSERT_EQ(records.history().size(), 2u);

    records.stagePending(group(2));
    EXPECT_EQ(records.publishPendingCollectively(
                  collective, "test/publish", "remote failure"),
              2u);
    ASSERT_EQ(records.history().size(), 4u);
    EXPECT_EQ(records.history()[2].step, 2);
    EXPECT_EQ(records.history()[3].marker, 2);

    // An identical replay drops its group without publishing.
    records.stagePending(group(2));
    records.clearPending();
    EXPECT_TRUE(records.pending().empty());
    EXPECT_EQ(records.history().size(), 4u);

    // An empty pending group publishes nothing.
    EXPECT_EQ(records.publishPendingCollectively(
                  collective, "test/publish", "remote failure"),
              4u);
    EXPECT_EQ(records.history().size(), 4u);
}

TEST(AcceptedStageRecords, HistoryAppendsAndGrowsGeometrically)
{
    TestStore store;
    AcceptedStageRecords<TestStore> records(store);
    records.reserveHistory(3u);
    EXPECT_GE(store.history.capacity(), 3u);
    for (int step = 0; step < 3; ++step) {
        records.appendHistory(TestRecord{.marker = 1, .step = step});
    }
    const auto capacity = store.history.capacity();
    ASSERT_EQ(records.history().size(), 3u);

    // Within capacity: no reallocation.
    records.growHistoryFor(capacity - 3u);
    EXPECT_EQ(store.history.capacity(), capacity);

    // Beyond capacity: at least twice the capacity.
    records.growHistoryFor(capacity - 3u + 1u);
    EXPECT_GE(store.history.capacity(), 2u * capacity);
    EXPECT_EQ(records.history()[2].step, 2);
}

TEST(AcceptedStageRecords, CollectivePhaseRethrowsTheLocalFailureAfterTheFailureAction)
{
    const auto collective = serialCollective();
    int calls = 0;
    EXPECT_NO_THROW(runCollectiveRecordPhase(
        collective, "test/phase", "remote failure", [&] { ++calls; }));
    EXPECT_EQ(calls, 1);

    bool rolled_back = false;
    try {
        runCollectiveRecordPhase(
            collective,
            "test/phase",
            "remote failure",
            [] { throw std::runtime_error("local failure"); },
            [&] { rolled_back = true; });
        FAIL() << "the local failure was not rethrown";
    } catch (const std::runtime_error& error) {
        EXPECT_EQ(std::string(error.what()), "local failure");
    }
    EXPECT_TRUE(rolled_back);

    // Without a failure, the failure action does not run.
    rolled_back = false;
    runCollectiveRecordPhase(
        collective, "test/phase", "remote failure", [] {}, [&] {
            rolled_back = true;
        });
    EXPECT_FALSE(rolled_back);
}

/**
 * @file test_AuxiliaryStateManager.cpp
 * @brief Unit tests for AuxiliaryStateManager — distributed ownership, sync, restart, transfer
 */

#include <gtest/gtest.h>

#include "Auxiliary/AuxiliaryStateManager.h"

#include <cmath>
#include <numeric>
#include <vector>

using svmp::FE::Real;
using namespace svmp::FE::systems;

// ---------------------------------------------------------------------------
//  Helper: make a spec with given scope and size
// ---------------------------------------------------------------------------

static AuxiliaryStateSpec makeSpec(const std::string& name, int size,
                                   AuxiliaryStateScope scope)
{
    AuxiliaryStateSpec spec;
    spec.name = name;
    spec.size = size;
    spec.scope = scope;
    return spec;
}

// ---------------------------------------------------------------------------
//  Block registration and access
// ---------------------------------------------------------------------------

TEST(AuxiliaryStateManager, RegisterAndAccessBlocks)
{
    AuxiliaryStateManager mgr;

    auto s1 = makeSpec("lumped", 3, AuxiliaryStateScope::Global);
    auto s2 = makeSpec("ionic", 4, AuxiliaryStateScope::Node);
    auto s3 = makeSpec("damage", 1, AuxiliaryStateScope::Cell);

    mgr.registerBlock(s1, 1);
    mgr.registerBlock(s2, 100);
    mgr.registerBlock(s3, 500);

    EXPECT_EQ(mgr.blockCount(), 3u);
    EXPECT_TRUE(mgr.hasBlock("lumped"));
    EXPECT_TRUE(mgr.hasBlock("ionic"));
    EXPECT_TRUE(mgr.hasBlock("damage"));
    EXPECT_FALSE(mgr.hasBlock("nonexistent"));

    EXPECT_EQ(mgr.getBlock("lumped").scope(), AuxiliaryStateScope::Global);
    EXPECT_EQ(mgr.getBlock("ionic").entityCount(), 100u);
    EXPECT_EQ(mgr.getBlock("damage").storageSize(), 500u);
}

TEST(AuxiliaryStateManager, RegisterWithInitialValues)
{
    AuxiliaryStateManager mgr;

    auto spec = makeSpec("X", 2, AuxiliaryStateScope::Global);
    const std::vector<Real> init = {3.0, 7.0};
    mgr.registerBlock(spec, 1, init);

    EXPECT_DOUBLE_EQ(mgr.getBlock("X").work()[0], 3.0);
    EXPECT_DOUBLE_EQ(mgr.getBlock("X").work()[1], 7.0);
}

TEST(AuxiliaryStateManager, DuplicateBlockThrows)
{
    AuxiliaryStateManager mgr;
    auto spec = makeSpec("X", 1, AuxiliaryStateScope::Global);
    mgr.registerBlock(spec, 1);
    EXPECT_THROW(mgr.registerBlock(spec, 1), svmp::FE::InvalidArgumentException);
}

TEST(AuxiliaryStateManager, RegisterBlockRagged)
{
    AuxiliaryStateManager mgr;

    AuxiliaryStateSpec spec;
    spec.name = "ragged";
    spec.size = 0;
    spec.scope = AuxiliaryStateScope::Cell;
    spec.layout_mode = AuxiliaryLayoutMode::Ragged;

    const std::vector<std::size_t> offsets = {0, 3, 5, 10};
    mgr.registerBlockRagged(spec, offsets);

    EXPECT_EQ(mgr.getBlock("ragged").entityCount(), 3u);
    EXPECT_EQ(mgr.getBlock("ragged").storageSize(), 10u);
}

// ---------------------------------------------------------------------------
//  Indexing access
// ---------------------------------------------------------------------------

TEST(AuxiliaryStateManager, GetIndexingForScopes)
{
    AuxiliaryStateManager mgr;

    mgr.registerBlock(makeSpec("G", 2, AuxiliaryStateScope::Global), 1);
    mgr.registerBlock(makeSpec("N", 4, AuxiliaryStateScope::Node), 50);
    mgr.registerBlock(makeSpec("C", 1, AuxiliaryStateScope::Cell), 200);
    mgr.registerBlock(makeSpec("B", 3, AuxiliaryStateScope::Facet), 30);

    auto& gi = mgr.getIndexing("G");
    EXPECT_EQ(gi.scope(), AuxiliaryStateScope::Global);
    EXPECT_EQ(gi.totalEntityCount(), 1u);
    EXPECT_EQ(gi.totalStorageSize(), 2u);

    auto& ni = mgr.getIndexing("N");
    EXPECT_EQ(ni.scope(), AuxiliaryStateScope::Node);
    EXPECT_EQ(ni.totalEntityCount(), 50u);
    EXPECT_EQ(ni.totalStorageSize(), 200u);

    auto& ci = mgr.getIndexing("C");
    EXPECT_EQ(ci.totalEntityCount(), 200u);

    auto& bi = mgr.getIndexing("B");
    EXPECT_EQ(bi.scope(), AuxiliaryStateScope::Facet);
    EXPECT_EQ(bi.totalEntityCount(), 30u);
}

TEST(AuxiliaryStateManager, GetSpecByName)
{
    AuxiliaryStateManager mgr;
    auto spec = makeSpec("ionic", 4, AuxiliaryStateScope::Node);
    spec.history_mode = AuxiliaryHistoryMode::MultiStep;
    spec.history_depth = 3;
    mgr.registerBlock(spec, 100);

    const auto& retrieved = mgr.getSpec("ionic");
    EXPECT_EQ(retrieved.name, "ionic");
    EXPECT_EQ(retrieved.size, 4);
    EXPECT_EQ(retrieved.scope, AuxiliaryStateScope::Node);
    EXPECT_EQ(retrieved.history_mode, AuxiliaryHistoryMode::MultiStep);
    EXPECT_EQ(retrieved.history_depth, 3);
}

TEST(AuxiliaryStateManager, RegisterNodeBlockWithOwnedGhostSplit)
{
    AuxiliaryStateManager mgr;

    auto spec = makeSpec("ionic", 2, AuxiliaryStateScope::Node);
    spec.sync_policy = AuxiliarySyncPolicy::OwnedAndGhost;

    mgr.registerBlock(spec, 8, 5);

    const auto& indexing = mgr.getIndexing("ionic");
    EXPECT_EQ(indexing.totalEntityCount(), 8u);
    EXPECT_EQ(indexing.ownedEntityCount(), 5u);
    EXPECT_EQ(indexing.ghostEntityCount(), 3u);
    EXPECT_EQ(indexing.totalStorageSize(), 16u);
    EXPECT_EQ(indexing.ownedStorageSize(), 10u);

    const auto& blk = mgr.getBlock("ionic");
    EXPECT_EQ(blk.entityCount(), 8u);
    EXPECT_EQ(blk.ownedEntityCount(), 5u);

    const auto layout = blk.blockLayout();
    EXPECT_EQ(layout.entity_count, 8u);
    EXPECT_EQ(layout.local_storage_size, 16u);
    EXPECT_EQ(layout.owned_entity_count, 5u);
    EXPECT_EQ(layout.owned_storage_size, 10u);

    EXPECT_NO_THROW(mgr.validate());
}

TEST(AuxiliaryStateManager, GhostSplitRejectedForNonNodeScope)
{
    AuxiliaryStateManager mgr;
    auto spec = makeSpec("cell_data", 1, AuxiliaryStateScope::Cell);
    EXPECT_THROW(mgr.registerBlock(spec, 6, 4), svmp::FE::InvalidArgumentException);
}

// ---------------------------------------------------------------------------
//  Ghost synchronization
// ---------------------------------------------------------------------------

TEST(AuxiliaryStateManager, GhostSyncHookCalledForOwnedAndGhost)
{
    AuxiliaryStateManager mgr;

    AuxiliaryStateSpec spec;
    spec.name = "synced";
    spec.size = 1;
    spec.scope = AuxiliaryStateScope::Node;
    spec.sync_policy = AuxiliarySyncPolicy::OwnedAndGhost;
    mgr.registerBlock(spec, 10);

    bool hook_called = false;
    mgr.setGhostSyncHook("synced", [&](std::string_view name, std::span<Real> buf) {
        hook_called = true;
        EXPECT_EQ(name, "synced");
        EXPECT_EQ(buf.size(), 10u);
    });

    mgr.syncGhosts();
    EXPECT_TRUE(hook_called);
}

TEST(AuxiliaryStateManager, GhostSyncSkippedForNonePolicy)
{
    AuxiliaryStateManager mgr;

    AuxiliaryStateSpec spec;
    spec.name = "local";
    spec.size = 1;
    spec.scope = AuxiliaryStateScope::Cell;
    spec.sync_policy = AuxiliarySyncPolicy::None;
    mgr.registerBlock(spec, 10);

    bool hook_called = false;
    mgr.setGhostSyncHook("local", [&](std::string_view, std::span<Real>) {
        hook_called = true;
    });

    mgr.syncGhosts();
    EXPECT_FALSE(hook_called); // sync_policy is None
}

TEST(AuxiliaryStateManager, GhostSyncPerBlock)
{
    AuxiliaryStateManager mgr;

    AuxiliaryStateSpec spec;
    spec.name = "synced";
    spec.size = 1;
    spec.scope = AuxiliaryStateScope::Node;
    spec.sync_policy = AuxiliarySyncPolicy::OwnedAndGhost;
    mgr.registerBlock(spec, 5);

    bool hook_called = false;
    mgr.setGhostSyncHook("synced", [&](std::string_view, std::span<Real>) {
        hook_called = true;
    });

    mgr.syncGhosts("synced");
    EXPECT_TRUE(hook_called);
}

// ---------------------------------------------------------------------------
//  Lifecycle
// ---------------------------------------------------------------------------

TEST(AuxiliaryStateManager, CommitAndRollback)
{
    AuxiliaryStateManager mgr;
    mgr.registerBlock(makeSpec("X", 1, AuxiliaryStateScope::Global), 1,
                      std::vector<Real>{0.0});

    mgr.getBlock("X").work()[0] = 10.0;
    mgr.commitAll(0.1);

    EXPECT_DOUBLE_EQ(mgr.getBlock("X").committed()[0], 10.0);

    mgr.getBlock("X").work()[0] = 99.0;
    mgr.rollbackAll();

    EXPECT_DOUBLE_EQ(mgr.getBlock("X").work()[0], 10.0);
}

TEST(AuxiliaryStateManager, ResetAllToCommitted)
{
    AuxiliaryStateManager mgr;
    mgr.registerBlock(makeSpec("X", 1, AuxiliaryStateScope::Global), 1,
                      std::vector<Real>{5.0});

    mgr.getBlock("X").work()[0] = 99.0;
    mgr.resetAllToCommitted();

    EXPECT_DOUBLE_EQ(mgr.getBlock("X").work()[0], 5.0);
}

TEST(AuxiliaryStateManager, ResetAndRollbackRefreshGhostWorkValues)
{
    AuxiliaryStateManager mgr;

    AuxiliaryStateSpec spec;
    spec.name = "ghosted";
    spec.size = 1;
    spec.scope = AuxiliaryStateScope::Node;
    spec.sync_policy = AuxiliarySyncPolicy::OwnedAndGhost;
    mgr.registerBlock(spec, 5, 3, std::vector<Real>{1.0, 2.0, 3.0, 90.0, 91.0});

    mgr.setGhostSyncHook("ghosted", [&](std::string_view, std::span<Real> values) {
        values[3] = values[0] + 100.0;
        values[4] = values[1] + 100.0;
    });

    auto& blk = mgr.getBlock("ghosted");
    blk.work()[0] = -1.0;
    blk.work()[1] = -2.0;
    blk.work()[2] = -3.0;
    blk.work()[3] = -4.0;
    blk.work()[4] = -5.0;

    mgr.resetAllToCommitted();
    EXPECT_DOUBLE_EQ(blk.work()[0], 1.0);
    EXPECT_DOUBLE_EQ(blk.work()[1], 2.0);
    EXPECT_DOUBLE_EQ(blk.work()[2], 3.0);
    EXPECT_DOUBLE_EQ(blk.work()[3], 101.0);
    EXPECT_DOUBLE_EQ(blk.work()[4], 102.0);

    blk.work()[0] = 7.0;
    blk.work()[1] = 8.0;
    blk.work()[2] = 9.0;
    blk.work()[3] = -40.0;
    blk.work()[4] = -50.0;
    mgr.commitAll(0.25);

    blk.work()[0] = -7.0;
    blk.work()[1] = -8.0;
    blk.work()[2] = -9.0;
    blk.work()[3] = -60.0;
    blk.work()[4] = -70.0;
    mgr.rollbackAll();

    EXPECT_DOUBLE_EQ(blk.work()[0], 7.0);
    EXPECT_DOUBLE_EQ(blk.work()[1], 8.0);
    EXPECT_DOUBLE_EQ(blk.work()[2], 9.0);
    EXPECT_DOUBLE_EQ(blk.work()[3], 107.0);
    EXPECT_DOUBLE_EQ(blk.work()[4], 108.0);
}

TEST(AuxiliaryStateManager, ClearRemovesEverything)
{
    AuxiliaryStateManager mgr;
    mgr.registerBlock(makeSpec("X", 1, AuxiliaryStateScope::Global), 1);
    mgr.clear();

    EXPECT_EQ(mgr.blockCount(), 0u);
    EXPECT_FALSE(mgr.hasBlock("X"));
}

TEST(AuxiliaryStateManager, InvalidateSetupClearsHooks)
{
    AuxiliaryStateManager mgr;

    AuxiliaryStateSpec spec;
    spec.name = "synced";
    spec.size = 1;
    spec.scope = AuxiliaryStateScope::Node;
    spec.sync_policy = AuxiliarySyncPolicy::OwnedAndGhost;
    mgr.registerBlock(spec, 5);

    bool hook_called = false;
    mgr.setGhostSyncHook("synced", [&](std::string_view, std::span<Real>) {
        hook_called = true;
    });

    mgr.invalidateSetup();

    // Hook should be cleared
    mgr.syncGhosts();
    EXPECT_FALSE(hook_called);

    // But block data is preserved
    EXPECT_TRUE(mgr.hasBlock("synced"));
    EXPECT_EQ(mgr.getBlock("synced").storageSize(), 5u);
}

// ---------------------------------------------------------------------------
//  Pack / Unpack (checkpoint / restart)
// ---------------------------------------------------------------------------

TEST(AuxiliaryStateManager, PackUnpackSingleBlock)
{
    AuxiliaryStateManager mgr;
    mgr.registerBlock(makeSpec("X", 2, AuxiliaryStateScope::Global), 1);

    // Set work and committed differently
    mgr.getBlock("X").work()[0] = 1.0;
    mgr.getBlock("X").work()[1] = 2.0;
    mgr.commitAll(0.1);
    mgr.getBlock("X").work()[0] = 3.0;
    mgr.getBlock("X").work()[1] = 4.0;

    auto packed = mgr.packBlock("X");
    ASSERT_EQ(packed.size(), 4u); // 2 committed + 2 work

    // Create a fresh manager and unpack
    AuxiliaryStateManager mgr2;
    mgr2.registerBlock(makeSpec("X", 2, AuxiliaryStateScope::Global), 1);

    mgr2.unpackBlock("X", packed);

    EXPECT_DOUBLE_EQ(mgr2.getBlock("X").committed()[0], 1.0);
    EXPECT_DOUBLE_EQ(mgr2.getBlock("X").committed()[1], 2.0);
    EXPECT_DOUBLE_EQ(mgr2.getBlock("X").work()[0], 3.0);
    EXPECT_DOUBLE_EQ(mgr2.getBlock("X").work()[1], 4.0);
}

TEST(AuxiliaryStateManager, UnpackRestoresGhostedCommittedAndWorkViaSyncHook)
{
    AuxiliaryStateManager mgr;
    AuxiliaryStateSpec spec;
    spec.name = "ghosted";
    spec.size = 1;
    spec.scope = AuxiliaryStateScope::Node;
    spec.sync_policy = AuxiliarySyncPolicy::OwnedAndGhost;
    mgr.registerBlock(spec, 5, 3, std::vector<Real>{1.0, 2.0, 3.0, 90.0, 91.0});
    mgr.setGhostSyncHook("ghosted", [&](std::string_view, std::span<Real> values) {
        values[3] = values[0] + 10.0;
        values[4] = values[1] + 10.0;
    });

    auto packed = mgr.packAll();

    AuxiliaryStateManager restored;
    restored.registerBlock(spec, 5, 3, std::vector<Real>{0.0, 0.0, 0.0, -1.0, -1.0});
    restored.setGhostSyncHook("ghosted", [&](std::string_view, std::span<Real> values) {
        values[3] = values[0] + 10.0;
        values[4] = values[1] + 10.0;
    });

    restored.unpackAll(packed);

    const auto committed = restored.getBlock("ghosted").committed();
    const auto work = restored.getBlock("ghosted").work();
    EXPECT_DOUBLE_EQ(committed[0], 1.0);
    EXPECT_DOUBLE_EQ(committed[1], 2.0);
    EXPECT_DOUBLE_EQ(committed[2], 3.0);
    EXPECT_DOUBLE_EQ(committed[3], 11.0);
    EXPECT_DOUBLE_EQ(committed[4], 12.0);
    EXPECT_DOUBLE_EQ(work[3], 11.0);
    EXPECT_DOUBLE_EQ(work[4], 12.0);
}

TEST(AuxiliaryStateManager, PackUnpackAllBlocks)
{
    AuxiliaryStateManager mgr;
    mgr.registerBlock(makeSpec("A", 1, AuxiliaryStateScope::Global), 1,
                      std::vector<Real>{10.0});
    mgr.registerBlock(makeSpec("B", 2, AuxiliaryStateScope::Global), 1,
                      std::vector<Real>{20.0, 30.0});

    mgr.getBlock("A").work()[0] = 11.0;
    mgr.getBlock("B").work()[0] = 21.0;
    mgr.getBlock("B").work()[1] = 31.0;

    auto packed = mgr.packAll();

    // Unpack into fresh manager
    AuxiliaryStateManager mgr2;
    mgr2.registerBlock(makeSpec("A", 1, AuxiliaryStateScope::Global), 1);
    mgr2.registerBlock(makeSpec("B", 2, AuxiliaryStateScope::Global), 1);

    mgr2.unpackAll(packed);

    EXPECT_DOUBLE_EQ(mgr2.getBlock("A").committed()[0], 10.0);
    EXPECT_DOUBLE_EQ(mgr2.getBlock("A").work()[0], 11.0);
    EXPECT_DOUBLE_EQ(mgr2.getBlock("B").committed()[0], 20.0);
    EXPECT_DOUBLE_EQ(mgr2.getBlock("B").committed()[1], 30.0);
    EXPECT_DOUBLE_EQ(mgr2.getBlock("B").work()[0], 21.0);
    EXPECT_DOUBLE_EQ(mgr2.getBlock("B").work()[1], 31.0);
}

TEST(AuxiliaryStateManager, UnpackSizeMismatchThrows)
{
    AuxiliaryStateManager mgr;
    mgr.registerBlock(makeSpec("X", 2, AuxiliaryStateScope::Global), 1);

    std::vector<Real> bad_packed = {1.0}; // too small
    EXPECT_THROW(mgr.unpackBlock("X", bad_packed), svmp::FE::InvalidArgumentException);
}

// ---------------------------------------------------------------------------
//  Transfer / remap
// ---------------------------------------------------------------------------

TEST(AuxiliaryStateManager, TransferBlockWithHook)
{
    AuxiliaryStateManager mgr;
    mgr.registerBlock(makeSpec("D", 1, AuxiliaryStateScope::Cell), 3,
                      std::vector<Real>{1.0, 2.0, 3.0});

    // Custom hook: double each value, fill new entities with -1
    mgr.setTransferHook("D", [](std::span<const Real> old_data,
                                std::size_t old_count,
                                std::size_t new_count,
                                std::span<Real> output) {
        for (std::size_t i = 0; i < std::min(old_count, new_count); ++i) {
            output[i] = old_data[i] * 2.0;
        }
        for (std::size_t i = old_count; i < new_count; ++i) {
            output[i] = -1.0;
        }
    });

    mgr.transferBlock("D", 5);

    auto& blk = mgr.getBlock("D");
    EXPECT_EQ(blk.entityCount(), 5u);
    EXPECT_DOUBLE_EQ(blk.work()[0], 2.0);
    EXPECT_DOUBLE_EQ(blk.work()[1], 4.0);
    EXPECT_DOUBLE_EQ(blk.work()[2], 6.0);
    EXPECT_DOUBLE_EQ(blk.work()[3], -1.0);
    EXPECT_DOUBLE_EQ(blk.work()[4], -1.0);
}

TEST(AuxiliaryStateManager, TransferBlockDefaultResize)
{
    AuxiliaryStateManager mgr;
    mgr.registerBlock(makeSpec("D", 2, AuxiliaryStateScope::Cell), 3);

    mgr.getBlock("D").work()[0] = 5.0;
    mgr.getBlock("D").work()[1] = 6.0;

    mgr.transferBlock("D", 5);

    auto& blk = mgr.getBlock("D");
    EXPECT_EQ(blk.entityCount(), 5u);
    EXPECT_DOUBLE_EQ(blk.work()[0], 5.0); // preserved
    EXPECT_DOUBLE_EQ(blk.work()[1], 6.0); // preserved
}

TEST(AuxiliaryStateManager, ReinitializeBlockZeroFills)
{
    AuxiliaryStateManager mgr;
    mgr.registerBlock(makeSpec("D", 1, AuxiliaryStateScope::Cell), 3,
                      std::vector<Real>{1.0, 2.0, 3.0});

    mgr.reinitializeBlock("D", 5);

    auto& blk = mgr.getBlock("D");
    EXPECT_EQ(blk.entityCount(), 5u);
    for (std::size_t i = 0; i < 5; ++i) {
        EXPECT_DOUBLE_EQ(blk.work()[i], 0.0);
    }
}

// ---------------------------------------------------------------------------
//  Validation
// ---------------------------------------------------------------------------

TEST(AuxiliaryStateManager, ValidatePassesOnConsistentState)
{
    AuxiliaryStateManager mgr;
    mgr.registerBlock(makeSpec("A", 2, AuxiliaryStateScope::Global), 1);
    mgr.registerBlock(makeSpec("B", 3, AuxiliaryStateScope::Node), 50);

    EXPECT_NO_THROW(mgr.validate());
}

// ---------------------------------------------------------------------------
//  Storage summary
// ---------------------------------------------------------------------------

TEST(AuxiliaryStateManager, StorageSummary)
{
    AuxiliaryStateManager mgr;
    mgr.registerBlock(makeSpec("A", 3, AuxiliaryStateScope::Global), 1);
    mgr.registerBlock(makeSpec("B", 2, AuxiliaryStateScope::Node), 100);

    auto summary = mgr.storageSummary();
    EXPECT_EQ(summary.block_count, 2u);
    EXPECT_EQ(summary.total_work_storage, 203u);
}

// ---------------------------------------------------------------------------
//  All scopes in one manager
// ---------------------------------------------------------------------------

TEST(AuxiliaryStateManager, AllSevenScopesRegistered)
{
    AuxiliaryStateManager mgr;

    mgr.registerBlock(makeSpec("global", 2, AuxiliaryStateScope::Global), 1);
    mgr.registerBlock(makeSpec("boundary", 3, AuxiliaryStateScope::Boundary), 1);
    mgr.registerBlock(makeSpec("node", 4, AuxiliaryStateScope::Node), 50);
    mgr.registerBlock(makeSpec("cell", 1, AuxiliaryStateScope::Cell), 200);
    mgr.registerBlock(makeSpec("qp", 6, AuxiliaryStateScope::QuadraturePoint), 800);
    mgr.registerBlock(makeSpec("region", 5, AuxiliaryStateScope::Region), 4);
    mgr.registerBlock(makeSpec("facet", 2, AuxiliaryStateScope::Facet), 30);

    EXPECT_EQ(mgr.blockCount(), 7u);

    EXPECT_EQ(mgr.getBlock("global").scope(), AuxiliaryStateScope::Global);
    EXPECT_EQ(mgr.getBlock("boundary").scope(), AuxiliaryStateScope::Boundary);
    EXPECT_EQ(mgr.getBlock("node").scope(), AuxiliaryStateScope::Node);
    EXPECT_EQ(mgr.getBlock("cell").scope(), AuxiliaryStateScope::Cell);
    EXPECT_EQ(mgr.getBlock("qp").scope(), AuxiliaryStateScope::QuadraturePoint);
    EXPECT_EQ(mgr.getBlock("region").scope(), AuxiliaryStateScope::Region);
    EXPECT_EQ(mgr.getBlock("facet").scope(), AuxiliaryStateScope::Facet);

    auto summary = mgr.storageSummary();
    EXPECT_EQ(summary.total_work_storage,
              2u + 3u + 200u + 200u + 4800u + 20u + 60u); // 5285

    EXPECT_NO_THROW(mgr.validate());
}

TEST(AuxiliaryStateManager, RegisterRegionBlockPreservesIndexing)
{
    AuxiliaryStateManager mgr;

    mgr.registerBlock(makeSpec("region", 2, AuxiliaryStateScope::Region), 3);

    const auto& idx = mgr.getIndexing("region");
    EXPECT_EQ(idx.scope(), AuxiliaryStateScope::Region);
    EXPECT_EQ(idx.totalEntityCount(), 3u);
    EXPECT_EQ(idx.ownedEntityCount(), 3u);
    EXPECT_EQ(idx.flatIndex(2, 1), 5u);
    EXPECT_NO_THROW(mgr.validate());
}

TEST(AuxiliaryStateManager, RegisterQuadraturePointBlockWithOffsetsPreservesIndexing)
{
    AuxiliaryStateManager mgr;

    const std::vector<std::size_t> qp_offsets{0, 4, 7, 9};
    mgr.registerBlockWithQPOffsets(
        makeSpec("qp", 2, AuxiliaryStateScope::QuadraturePoint), qp_offsets);

    const auto& idx = mgr.getIndexing("qp");
    EXPECT_EQ(idx.scope(), AuxiliaryStateScope::QuadraturePoint);
    EXPECT_EQ(idx.totalEntityCount(), 9u);
    EXPECT_EQ(idx.qpOffsets().size(), qp_offsets.size());
    EXPECT_EQ(idx.qpOffsets()[1], 4u);
    EXPECT_EQ(idx.qpsForCell(1), 3u);
    EXPECT_EQ(idx.qpFlatIndex(1, 2, 1), (4u + 2u) * 2u + 1u);
}

// ---------------------------------------------------------------------------
//  Ownership rules for Monolithic blocks
// ---------------------------------------------------------------------------

TEST(AuxiliaryStateManager, MonolithicBlockUsesAuxLayouts)
{
    AuxiliaryStateManager mgr;

    AuxiliaryStateSpec spec;
    spec.name = "mono_aux";
    spec.size = 3;
    spec.scope = AuxiliaryStateScope::Node;
    spec.solve_mode = AuxiliarySolveMode::Monolithic;
    mgr.registerBlock(spec, 100);

    const auto& retrieved_spec = mgr.getSpec("mono_aux");
    EXPECT_EQ(retrieved_spec.solve_mode, AuxiliarySolveMode::Monolithic);

    // Block is stored with auxiliary-specific layout, not FE DOF maps
    auto& blk = mgr.getBlock("mono_aux");
    EXPECT_EQ(blk.storageSize(), 300u); // 100 nodes × 3 components
}

// ---------------------------------------------------------------------------
//  Formulation-selectable sync policy
// ---------------------------------------------------------------------------

TEST(AuxiliaryStateManager, SyncPolicySelectablePerBlock)
{
    AuxiliaryStateManager mgr;

    AuxiliaryStateSpec s1;
    s1.name = "owned_only";
    s1.size = 1;
    s1.scope = AuxiliaryStateScope::Node;
    s1.sync_policy = AuxiliarySyncPolicy::OwnedOnly;
    mgr.registerBlock(s1, 10);

    AuxiliaryStateSpec s2;
    s2.name = "full_sync";
    s2.size = 1;
    s2.scope = AuxiliaryStateScope::Node;
    s2.sync_policy = AuxiliarySyncPolicy::OwnedAndGhost;
    mgr.registerBlock(s2, 10);

    EXPECT_EQ(mgr.getSpec("owned_only").sync_policy, AuxiliarySyncPolicy::OwnedOnly);
    EXPECT_EQ(mgr.getSpec("full_sync").sync_policy, AuxiliarySyncPolicy::OwnedAndGhost);

    // Only OwnedAndGhost triggers sync hooks
    int sync_count = 0;
    mgr.setGhostSyncHook("owned_only", [&](std::string_view, std::span<Real>) {
        sync_count++;
    });
    mgr.setGhostSyncHook("full_sync", [&](std::string_view, std::span<Real>) {
        sync_count++;
    });

    mgr.syncGhosts();
    EXPECT_EQ(sync_count, 1); // Only full_sync's hook called
}

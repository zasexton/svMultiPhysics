/**
 * @file test_AuxiliaryStateStorage.cpp
 * @brief Unit tests for AuxiliaryBlockStorage — per-block storage backend
 */

#include <gtest/gtest.h>

#include "Auxiliary/AuxiliaryState.h"
#include "Auxiliary/AuxiliaryStateStorage.h"

#include <numeric>
#include <vector>

using svmp::FE::Real;
using namespace svmp::FE::systems;

// ---------------------------------------------------------------------------
//  Helper: make a spec with given scope
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
//  Fixed-stride setup
// ---------------------------------------------------------------------------

TEST(AuxiliaryBlockStorage, DefaultNotSetUp)
{
    AuxiliaryBlockStorage storage;
    EXPECT_FALSE(storage.isSetup());
}

TEST(AuxiliaryBlockStorage, FixedStrideSetup_Global)
{
    AuxiliaryBlockStorage storage;
    auto spec = makeSpec("lumped", 3, AuxiliaryStateScope::Global);

    storage.setupFixedStride(spec, 1);

    EXPECT_TRUE(storage.isSetup());
    EXPECT_EQ(storage.name(), "lumped");
    EXPECT_EQ(storage.scope(), AuxiliaryStateScope::Global);
    EXPECT_EQ(storage.layoutMode(), AuxiliaryLayoutMode::FixedStride);
    EXPECT_EQ(storage.componentStride(), 3);
    EXPECT_EQ(storage.entityCount(), 1u);
    EXPECT_EQ(storage.storageSize(), 3u);
}

TEST(AuxiliaryBlockStorage, FixedStrideSetup_Boundary)
{
    AuxiliaryBlockStorage storage;
    auto spec = makeSpec("outlet_rcr", 3, AuxiliaryStateScope::Boundary);

    storage.setupFixedStride(spec, 1);

    EXPECT_TRUE(storage.isSetup());
    EXPECT_EQ(storage.scope(), AuxiliaryStateScope::Boundary);
    EXPECT_EQ(storage.entityCount(), 1u);
    EXPECT_EQ(storage.storageSize(), 3u);
}

TEST(AuxiliaryBlockStorage, FixedStrideSetup_Node)
{
    AuxiliaryBlockStorage storage;
    auto spec = makeSpec("ionic", 4, AuxiliaryStateScope::Node);

    storage.setupFixedStride(spec, 100); // 100 nodes

    EXPECT_EQ(storage.entityCount(), 100u);
    EXPECT_EQ(storage.storageSize(), 400u); // 100 * 4
    EXPECT_EQ(storage.work().size(), 400u);
    EXPECT_EQ(storage.committed().size(), 400u);
}

TEST(AuxiliaryBlockStorage, FixedStrideSetup_Cell)
{
    AuxiliaryBlockStorage storage;
    auto spec = makeSpec("damage", 1, AuxiliaryStateScope::Cell);

    storage.setupFixedStride(spec, 500);

    EXPECT_EQ(storage.scope(), AuxiliaryStateScope::Cell);
    EXPECT_EQ(storage.entityCount(), 500u);
    EXPECT_EQ(storage.storageSize(), 500u);
}

TEST(AuxiliaryBlockStorage, FixedStrideSetup_QuadraturePoint)
{
    AuxiliaryBlockStorage storage;
    auto spec = makeSpec("plastic_strain", 6, AuxiliaryStateScope::QuadraturePoint);

    // 50 cells × 4 QPs = 200 QPs
    storage.setupFixedStride(spec, 200);

    EXPECT_EQ(storage.scope(), AuxiliaryStateScope::QuadraturePoint);
    EXPECT_EQ(storage.entityCount(), 200u);
    EXPECT_EQ(storage.storageSize(), 1200u); // 200 * 6
}

TEST(AuxiliaryBlockStorage, FixedStrideSetup_Region)
{
    AuxiliaryBlockStorage storage;
    auto spec = makeSpec("region_state", 2, AuxiliaryStateScope::Region);

    storage.setupFixedStride(spec, 3);

    EXPECT_EQ(storage.scope(), AuxiliaryStateScope::Region);
    EXPECT_EQ(storage.entityCount(), 3u);
    EXPECT_EQ(storage.storageSize(), 6u);
}

TEST(AuxiliaryBlockStorage, FixedStrideSetup_Facet)
{
    AuxiliaryBlockStorage storage;
    auto spec = makeSpec("bc_state", 2, AuxiliaryStateScope::Facet);

    storage.setupFixedStride(spec, 30); // 30 boundary faces

    EXPECT_EQ(storage.scope(), AuxiliaryStateScope::Facet);
    EXPECT_EQ(storage.entityCount(), 30u);
    EXPECT_EQ(storage.storageSize(), 60u);
}

TEST(AuxiliaryBlockStorage, FixedStrideRejectsRaggedLayoutSpec)
{
    AuxiliaryBlockStorage storage;
    auto spec = makeSpec("wrong_layout", 2, AuxiliaryStateScope::Cell);
    spec.layout_mode = AuxiliaryLayoutMode::Ragged;

    EXPECT_THROW(storage.setupFixedStride(spec, 4),
                 svmp::FE::InvalidArgumentException);
    EXPECT_FALSE(storage.isSetup());
}

// ---------------------------------------------------------------------------
//  Work buffer access
// ---------------------------------------------------------------------------

TEST(AuxiliaryBlockStorage, WorkBufferReadWrite)
{
    AuxiliaryBlockStorage storage;
    auto spec = makeSpec("v", 2, AuxiliaryStateScope::Node);
    storage.setupFixedStride(spec, 3);

    // Write via span
    auto w = storage.work();
    ASSERT_EQ(w.size(), 6u);
    w[0] = 1.0; w[1] = 2.0;
    w[2] = 3.0; w[3] = 4.0;
    w[4] = 5.0; w[5] = 6.0;

    // Read back
    EXPECT_DOUBLE_EQ(storage.work()[0], 1.0);
    EXPECT_DOUBLE_EQ(storage.work()[5], 6.0);
}

TEST(AuxiliaryBlockStorage, EntityViewFixedStride)
{
    AuxiliaryBlockStorage storage;
    auto spec = makeSpec("v", 3, AuxiliaryStateScope::Node);
    storage.setupFixedStride(spec, 4);

    // Set entity 2 values
    auto w = storage.work();
    w[6] = 10.0; w[7] = 20.0; w[8] = 30.0;

    auto entity2 = storage.workEntity(2);
    ASSERT_EQ(entity2.size(), 3u);
    EXPECT_DOUBLE_EQ(entity2[0], 10.0);
    EXPECT_DOUBLE_EQ(entity2[1], 20.0);
    EXPECT_DOUBLE_EQ(entity2[2], 30.0);
}

TEST(AuxiliaryBlockStorage, EntityViewRejectsComponentMajorFixedStride)
{
    AuxiliaryBlockStorage storage;
    auto spec = makeSpec("v", 2, AuxiliaryStateScope::Node);
    spec.ordering = AuxiliaryEntityOrdering::ByComponentThenEntity;
    storage.setupFixedStride(spec, 3);

    auto w = storage.work();
    w[0] = 10.0; w[1] = 20.0; w[2] = 30.0;
    w[3] = 11.0; w[4] = 21.0; w[5] = 31.0;
    storage.initialize(w);

    EXPECT_THROW(storage.workEntity(1), svmp::FE::systems::InvalidStateException);

    const auto& const_storage = storage;
    EXPECT_THROW(const_storage.workEntity(1), svmp::FE::systems::InvalidStateException);
    EXPECT_THROW(const_storage.committedEntity(1), svmp::FE::systems::InvalidStateException);

    const auto gathered = storage.gatherEntityWork(1);
    ASSERT_EQ(gathered.size(), 2u);
    EXPECT_DOUBLE_EQ(gathered[0], 20.0);
    EXPECT_DOUBLE_EQ(gathered[1], 21.0);
}

// ---------------------------------------------------------------------------
//  Committed buffer
// ---------------------------------------------------------------------------

TEST(AuxiliaryBlockStorage, CommittedStartsZero)
{
    AuxiliaryBlockStorage storage;
    auto spec = makeSpec("x", 2, AuxiliaryStateScope::Global);
    storage.setupFixedStride(spec, 1);

    auto c = storage.committed();
    ASSERT_EQ(c.size(), 2u);
    EXPECT_DOUBLE_EQ(c[0], 0.0);
    EXPECT_DOUBLE_EQ(c[1], 0.0);
}

// ---------------------------------------------------------------------------
//  Initialize
// ---------------------------------------------------------------------------

TEST(AuxiliaryBlockStorage, InitializeSetsWorkAndCommitted)
{
    AuxiliaryBlockStorage storage;
    auto spec = makeSpec("x", 2, AuxiliaryStateScope::Global);
    storage.setupFixedStride(spec, 1);

    const std::vector<Real> init = {5.0, 7.0};
    storage.initialize(init);

    EXPECT_DOUBLE_EQ(storage.work()[0], 5.0);
    EXPECT_DOUBLE_EQ(storage.work()[1], 7.0);
    EXPECT_DOUBLE_EQ(storage.committed()[0], 5.0);
    EXPECT_DOUBLE_EQ(storage.committed()[1], 7.0);
}

// ---------------------------------------------------------------------------
//  Commit / reset / rollback
// ---------------------------------------------------------------------------

TEST(AuxiliaryBlockStorage, CommitAndResetCycle)
{
    AuxiliaryBlockStorage storage;
    auto spec = makeSpec("x", 1, AuxiliaryStateScope::Global);
    storage.setupFixedStride(spec, 1);

    storage.work()[0] = 10.0;
    storage.commitTimeStep(0.1);

    EXPECT_DOUBLE_EQ(storage.committed()[0], 10.0);

    // Modify work
    storage.work()[0] = 20.0;
    EXPECT_DOUBLE_EQ(storage.work()[0], 20.0);

    // Reset reverts work to committed
    storage.resetToCommitted();
    EXPECT_DOUBLE_EQ(storage.work()[0], 10.0);
}

TEST(AuxiliaryBlockStorage, RollbackEqualsReset)
{
    AuxiliaryBlockStorage storage;
    auto spec = makeSpec("x", 1, AuxiliaryStateScope::Global);
    storage.setupFixedStride(spec, 1);

    storage.work()[0] = 5.0;
    storage.commitTimeStep(0.0);

    storage.work()[0] = 99.0;
    storage.rollback();
    EXPECT_DOUBLE_EQ(storage.work()[0], 5.0);
}

TEST(AuxiliaryBlockStorage, CommitPushesHistory)
{
    AuxiliaryBlockStorage storage;
    auto spec = makeSpec("x", 1, AuxiliaryStateScope::Global);
    spec.history_mode = AuxiliaryHistoryMode::MultiStep;
    spec.history_depth = 3;
    storage.setupFixedStride(spec, 1);

    // Initial state
    storage.initialize(std::vector<Real>{0.0});

    // Step 1
    storage.work()[0] = 1.0;
    storage.commitTimeStep(0.1);

    // Step 2
    storage.work()[0] = 2.0;
    storage.commitTimeStep(0.2);

    // History should have the committed states before each commit
    EXPECT_EQ(storage.history().depth(), 2u);
    EXPECT_DOUBLE_EQ(storage.history().snapshot(0)[0], 1.0);  // pushed at t=0.2
    EXPECT_DOUBLE_EQ(storage.history().snapshot(1)[0], 0.0);  // pushed at t=0.1
}

TEST(AuxiliaryBlockStorage, GhostedCommitOnlyCopiesOwnedPrefix)
{
    AuxiliaryBlockStorage storage;
    auto spec = makeSpec("ghosted", 1, AuxiliaryStateScope::Node);
    spec.history_mode = AuxiliaryHistoryMode::SingleStep;
    storage.setupFixedStride(spec, 5);
    storage.setOwnedEntityCount(3);
    storage.initialize(std::vector<Real>{1.0, 2.0, 3.0, 90.0, 91.0});

    auto work = storage.work();
    work[0] = 10.0;
    work[1] = 20.0;
    work[2] = 30.0;
    work[3] = 40.0;
    work[4] = 50.0;

    storage.commitTimeStep(0.25);

    const auto committed = storage.committed();
    EXPECT_EQ(storage.ownedEntityCount(), 3u);
    EXPECT_DOUBLE_EQ(committed[0], 10.0);
    EXPECT_DOUBLE_EQ(committed[1], 20.0);
    EXPECT_DOUBLE_EQ(committed[2], 30.0);
    EXPECT_DOUBLE_EQ(committed[3], 90.0);
    EXPECT_DOUBLE_EQ(committed[4], 91.0);

    ASSERT_EQ(storage.history().depth(), 1u);
    const auto hist = storage.history().snapshot(0);
    EXPECT_DOUBLE_EQ(hist[0], 1.0);
    EXPECT_DOUBLE_EQ(hist[1], 2.0);
    EXPECT_DOUBLE_EQ(hist[2], 3.0);
    EXPECT_DOUBLE_EQ(hist[3], 90.0);
    EXPECT_DOUBLE_EQ(hist[4], 91.0);
}

TEST(AuxiliaryBlockStorage, GhostedResetOnlyRestoresOwnedPrefix)
{
    AuxiliaryBlockStorage storage;
    auto spec = makeSpec("ghosted", 1, AuxiliaryStateScope::Node);
    storage.setupFixedStride(spec, 5);
    storage.setOwnedEntityCount(3);
    storage.initialize(std::vector<Real>{1.0, 2.0, 3.0, 90.0, 91.0});

    auto work = storage.work();
    work[0] = -1.0;
    work[1] = -2.0;
    work[2] = -3.0;
    work[3] = -4.0;
    work[4] = -5.0;

    storage.resetToCommitted();

    EXPECT_DOUBLE_EQ(storage.work()[0], 1.0);
    EXPECT_DOUBLE_EQ(storage.work()[1], 2.0);
    EXPECT_DOUBLE_EQ(storage.work()[2], 3.0);
    EXPECT_DOUBLE_EQ(storage.work()[3], -4.0);
    EXPECT_DOUBLE_EQ(storage.work()[4], -5.0);
}

// ---------------------------------------------------------------------------
//  Ragged layout
// ---------------------------------------------------------------------------

TEST(AuxiliaryBlockStorage, RaggedLayout)
{
    AuxiliaryBlockStorage storage;
    AuxiliaryStateSpec spec;
    spec.name = "ragged_block";
    spec.size = 0; // Not used for ragged
    spec.scope = AuxiliaryStateScope::Cell;
    spec.layout_mode = AuxiliaryLayoutMode::Ragged;

    // 3 entities: sizes 2, 5, 3 → total 10
    const std::vector<std::size_t> offsets = {0, 2, 7, 10};
    storage.setupRagged(spec, offsets);

    EXPECT_TRUE(storage.isSetup());
    EXPECT_EQ(storage.layoutMode(), AuxiliaryLayoutMode::Ragged);
    EXPECT_EQ(storage.entityCount(), 3u);
    EXPECT_EQ(storage.storageSize(), 10u);

    // Entity views
    auto e0 = storage.workEntity(0);
    EXPECT_EQ(e0.size(), 2u);

    auto e1 = storage.workEntity(1);
    EXPECT_EQ(e1.size(), 5u);

    auto e2 = storage.workEntity(2);
    EXPECT_EQ(e2.size(), 3u);

    // Write and read back
    e1[0] = 42.0;
    EXPECT_DOUBLE_EQ(storage.work()[2], 42.0); // offset 2 for entity 1
}

TEST(AuxiliaryBlockStorage, RaggedLayoutRejectsNonmonotoneOffsets)
{
    AuxiliaryBlockStorage storage;
    AuxiliaryStateSpec spec;
    spec.name = "ragged_block";
    spec.size = 0;
    spec.scope = AuxiliaryStateScope::Cell;
    spec.layout_mode = AuxiliaryLayoutMode::Ragged;

    EXPECT_THROW(storage.setupRagged(spec, std::vector<std::size_t>{0, 4, 3}),
                 svmp::FE::InvalidArgumentException);
    EXPECT_FALSE(storage.isSetup());
}

TEST(AuxiliaryBlockStorage, RaggedRejectsFixedStrideLayoutSpec)
{
    AuxiliaryBlockStorage storage;
    auto spec = makeSpec("wrong_layout", 2, AuxiliaryStateScope::Cell);

    EXPECT_THROW(storage.setupRagged(spec, std::vector<std::size_t>{0, 2, 5}),
                 svmp::FE::InvalidArgumentException);
    EXPECT_FALSE(storage.isSetup());
}

// ---------------------------------------------------------------------------
//  Resize (fixed-stride)
// ---------------------------------------------------------------------------

TEST(AuxiliaryBlockStorage, ResizePreservesData)
{
    AuxiliaryBlockStorage storage;
    auto spec = makeSpec("x", 2, AuxiliaryStateScope::Node);
    storage.setupFixedStride(spec, 3);

    storage.work()[0] = 1.0;
    storage.work()[1] = 2.0;
    storage.work()[4] = 9.0;

    storage.resize(5); // grow from 3 to 5 entities

    EXPECT_EQ(storage.entityCount(), 5u);
    EXPECT_EQ(storage.storageSize(), 10u);
    EXPECT_DOUBLE_EQ(storage.work()[0], 1.0); // preserved
    EXPECT_DOUBLE_EQ(storage.work()[1], 2.0); // preserved
    EXPECT_DOUBLE_EQ(storage.work()[4], 9.0); // preserved
    EXPECT_DOUBLE_EQ(storage.work()[8], 0.0); // new entry = 0
}

// ---------------------------------------------------------------------------
//  Clear
// ---------------------------------------------------------------------------

TEST(AuxiliaryBlockStorage, ClearResetsEverything)
{
    AuxiliaryBlockStorage storage;
    auto spec = makeSpec("x", 2, AuxiliaryStateScope::Global);
    storage.setupFixedStride(spec, 1);

    storage.work()[0] = 5.0;
    storage.clear();

    EXPECT_FALSE(storage.isSetup());
    EXPECT_EQ(storage.storageSize(), 0u);
}

// ---------------------------------------------------------------------------
//  Block layout summary
// ---------------------------------------------------------------------------

TEST(AuxiliaryBlockStorage, BlockLayoutReturnsCorrectValues)
{
    AuxiliaryBlockStorage storage;
    auto spec = makeSpec("ionic", 4, AuxiliaryStateScope::Node);
    spec.history_mode = AuxiliaryHistoryMode::SingleStep;
    storage.setupFixedStride(spec, 50);

    // Push some history
    storage.work()[0] = 1.0;
    storage.commitTimeStep(0.1);

    auto layout = storage.blockLayout();
    EXPECT_EQ(layout.component_stride, 4);
    EXPECT_EQ(layout.entity_count, 50u);
    EXPECT_EQ(layout.local_storage_size, 200u);
    EXPECT_EQ(layout.owned_entity_count, 50u);
    EXPECT_EQ(layout.owned_storage_size, 200u);
    EXPECT_EQ(layout.history_storage_size, 200u); // 1 snapshot × 200
}

// ---------------------------------------------------------------------------
//  AuxiliaryState multi-block API
// ---------------------------------------------------------------------------

TEST(AuxiliaryStateMultiBlock, RegisterMultipleBlocksDifferentScopes)
{
    AuxiliaryState state;

    AuxiliaryStateSpec global_spec;
    global_spec.name = "lumped_rcr";
    global_spec.size = 3;
    global_spec.scope = AuxiliaryStateScope::Global;

    AuxiliaryStateSpec node_spec;
    node_spec.name = "ionic_gates";
    node_spec.size = 4;
    node_spec.scope = AuxiliaryStateScope::Node;

    AuxiliaryStateSpec cell_spec;
    cell_spec.name = "damage";
    cell_spec.size = 1;
    cell_spec.scope = AuxiliaryStateScope::Cell;

    state.registerBlock(global_spec, 1);
    state.registerBlock(node_spec, 100);
    state.registerBlock(cell_spec, 500);

    EXPECT_EQ(state.blockCount(), 3u);
    EXPECT_TRUE(state.hasBlock("lumped_rcr"));
    EXPECT_TRUE(state.hasBlock("ionic_gates"));
    EXPECT_TRUE(state.hasBlock("damage"));
    EXPECT_FALSE(state.hasBlock("nonexistent"));

    auto& lumped = state.getBlock("lumped_rcr");
    EXPECT_EQ(lumped.scope(), AuxiliaryStateScope::Global);
    EXPECT_EQ(lumped.entityCount(), 1u);
    EXPECT_EQ(lumped.storageSize(), 3u);

    auto& ionic = state.getBlock("ionic_gates");
    EXPECT_EQ(ionic.scope(), AuxiliaryStateScope::Node);
    EXPECT_EQ(ionic.storageSize(), 400u);

    auto& dmg = state.getBlock("damage");
    EXPECT_EQ(dmg.scope(), AuxiliaryStateScope::Cell);
    EXPECT_EQ(dmg.storageSize(), 500u);
}

TEST(AuxiliaryStateMultiBlock, BlockLookupByIndex)
{
    AuxiliaryState state;

    AuxiliaryStateSpec spec;
    spec.name = "A";
    spec.size = 1;
    auto idx = state.registerBlock(spec, 1);
    EXPECT_EQ(idx, 0u);

    spec.name = "B";
    idx = state.registerBlock(spec, 1);
    EXPECT_EQ(idx, 1u);

    EXPECT_EQ(state.block(0).name(), "A");
    EXPECT_EQ(state.block(1).name(), "B");
}

TEST(AuxiliaryStateMultiBlock, DuplicateBlockNameThrows)
{
    AuxiliaryState state;

    AuxiliaryStateSpec spec;
    spec.name = "X";
    spec.size = 1;
    state.registerBlock(spec, 1);

    EXPECT_THROW(state.registerBlock(spec, 1), svmp::FE::InvalidArgumentException);
}

TEST(AuxiliaryStateMultiBlock, InvalidFixedStrideRegistrationLeavesStateUnchanged)
{
    AuxiliaryState state;

    AuxiliaryStateSpec spec;
    spec.name = "bad";
    spec.size = 0;
    spec.scope = AuxiliaryStateScope::Cell;

    EXPECT_THROW(state.registerBlock(spec, 3), svmp::FE::InvalidArgumentException);
    EXPECT_EQ(state.blockCount(), 0u);
    EXPECT_FALSE(state.hasBlock("bad"));

    spec.size = 1;
    EXPECT_NO_THROW(state.registerBlock(spec, 3));
    EXPECT_EQ(state.blockCount(), 1u);
    EXPECT_TRUE(state.hasBlock("bad"));
}

TEST(AuxiliaryStateMultiBlock, RegisterBlockWithInitialValues)
{
    AuxiliaryState state;

    AuxiliaryStateSpec spec;
    spec.name = "X";
    spec.size = 2;
    spec.scope = AuxiliaryStateScope::Global;

    const std::vector<Real> init = {3.0, 7.0};
    state.registerBlock(spec, 1, init);

    auto& blk = state.getBlock("X");
    EXPECT_DOUBLE_EQ(blk.work()[0], 3.0);
    EXPECT_DOUBLE_EQ(blk.work()[1], 7.0);
    EXPECT_DOUBLE_EQ(blk.committed()[0], 3.0);
}

TEST(AuxiliaryStateMultiBlock, RegisterBlockRagged)
{
    AuxiliaryState state;

    AuxiliaryStateSpec spec;
    spec.name = "ragged";
    spec.size = 0;
    spec.scope = AuxiliaryStateScope::Cell;
    spec.layout_mode = AuxiliaryLayoutMode::Ragged;

    const std::vector<std::size_t> offsets = {0, 2, 5};
    state.registerBlockRagged(spec, offsets);

    auto& blk = state.getBlock("ragged");
    EXPECT_EQ(blk.layoutMode(), AuxiliaryLayoutMode::Ragged);
    EXPECT_EQ(blk.entityCount(), 2u);
    EXPECT_EQ(blk.storageSize(), 5u);
}

TEST(AuxiliaryStateMultiBlock, InvalidRaggedRegistrationLeavesStateUnchanged)
{
    AuxiliaryState state;

    AuxiliaryStateSpec spec;
    spec.name = "ragged";
    spec.size = 0;
    spec.scope = AuxiliaryStateScope::Cell;
    spec.layout_mode = AuxiliaryLayoutMode::Ragged;

    EXPECT_THROW(state.registerBlockRagged(spec, std::vector<std::size_t>{0, 4, 3}),
                 svmp::FE::InvalidArgumentException);
    EXPECT_EQ(state.blockCount(), 0u);
    EXPECT_FALSE(state.hasBlock("ragged"));

    EXPECT_NO_THROW(state.registerBlockRagged(spec, std::vector<std::size_t>{0, 4, 4}));
    EXPECT_EQ(state.blockCount(), 1u);
    EXPECT_TRUE(state.hasBlock("ragged"));
}

TEST(AuxiliaryStateMultiBlock, RaggedRegistrationRequiresRaggedSpec)
{
    AuxiliaryState state;

    AuxiliaryStateSpec spec;
    spec.name = "ragged";
    spec.size = 0;
    spec.scope = AuxiliaryStateScope::Cell;

    EXPECT_THROW(state.registerBlockRagged(spec, std::vector<std::size_t>{0, 2}),
                 svmp::FE::InvalidArgumentException);
    EXPECT_EQ(state.blockCount(), 0u);
}

TEST(AuxiliaryStateMultiBlock, BlockNames)
{
    AuxiliaryState state;

    AuxiliaryStateSpec s1; s1.name = "alpha"; s1.size = 1;
    AuxiliaryStateSpec s2; s2.name = "beta";  s2.size = 1;
    AuxiliaryStateSpec s3; s3.name = "gamma"; s3.size = 1;

    state.registerBlock(s1, 1);
    state.registerBlock(s2, 1);
    state.registerBlock(s3, 1);

    auto names = state.blockNames();
    ASSERT_EQ(names.size(), 3u);
    EXPECT_EQ(names[0], "alpha");
    EXPECT_EQ(names[1], "beta");
    EXPECT_EQ(names[2], "gamma");
}

// ---------------------------------------------------------------------------
//  Bulk block operations
// ---------------------------------------------------------------------------

TEST(AuxiliaryStateMultiBlock, ResetAllBlocks)
{
    AuxiliaryState state;

    AuxiliaryStateSpec s1; s1.name = "A"; s1.size = 1;
    AuxiliaryStateSpec s2; s2.name = "B"; s2.size = 1;
    state.registerBlock(s1, 1, std::vector<Real>{1.0});
    state.registerBlock(s2, 1, std::vector<Real>{2.0});

    // Modify work
    state.getBlock("A").work()[0] = 99.0;
    state.getBlock("B").work()[0] = 99.0;

    state.resetAllBlocks();

    EXPECT_DOUBLE_EQ(state.getBlock("A").work()[0], 1.0);
    EXPECT_DOUBLE_EQ(state.getBlock("B").work()[0], 2.0);
}

TEST(AuxiliaryStateMultiBlock, CommitAndRollbackAllBlocks)
{
    AuxiliaryState state;

    AuxiliaryStateSpec s1; s1.name = "A"; s1.size = 1;
    state.registerBlock(s1, 1, std::vector<Real>{0.0});

    state.getBlock("A").work()[0] = 10.0;
    state.commitAllBlocks(1.0);

    EXPECT_DOUBLE_EQ(state.getBlock("A").committed()[0], 10.0);

    state.getBlock("A").work()[0] = 99.0;
    state.rollbackAllBlocks();

    EXPECT_DOUBLE_EQ(state.getBlock("A").work()[0], 10.0);
}

// ---------------------------------------------------------------------------
//  Storage summary
// ---------------------------------------------------------------------------

TEST(AuxiliaryStateMultiBlock, StorageSummary)
{
    AuxiliaryState state;

    AuxiliaryStateSpec s1; s1.name = "A"; s1.size = 3;
    s1.scope = AuxiliaryStateScope::Global;
    AuxiliaryStateSpec s2; s2.name = "B"; s2.size = 2;
    s2.scope = AuxiliaryStateScope::Node;

    state.registerBlock(s1, 1);    // 3 values
    state.registerBlock(s2, 100);  // 200 values

    auto summary = state.storageSummary();
    EXPECT_EQ(summary.block_count, 2u);
    EXPECT_EQ(summary.total_work_storage, 203u);
    EXPECT_EQ(summary.total_committed_storage, 203u);
    EXPECT_EQ(summary.total_history_storage, 0u);
}

// ---------------------------------------------------------------------------
//  History queries through blocks
// ---------------------------------------------------------------------------

TEST(AuxiliaryStateMultiBlock, BlockHistoryQuery)
{
    AuxiliaryState state;

    AuxiliaryStateSpec spec;
    spec.name = "X";
    spec.size = 2;
    spec.scope = AuxiliaryStateScope::Global;
    spec.history_mode = AuxiliaryHistoryMode::MultiStep;
    spec.history_depth = 5;

    state.registerBlock(spec, 1, std::vector<Real>{0.0, 0.0});

    auto& blk = state.getBlock("X");

    // Step 1
    blk.work()[0] = 1.0; blk.work()[1] = 10.0;
    blk.commitTimeStep(0.1);

    // Step 2
    blk.work()[0] = 2.0; blk.work()[1] = 20.0;
    blk.commitTimeStep(0.2);

    // Current committed
    EXPECT_DOUBLE_EQ(blk.committed()[0], 2.0);

    // History
    EXPECT_EQ(blk.history().depth(), 2u);
    EXPECT_DOUBLE_EQ(blk.history().snapshot(0)[0], 1.0);  // t=0.2 push
    EXPECT_DOUBLE_EQ(blk.history().snapshot(1)[0], 0.0);  // t=0.1 push
    EXPECT_DOUBLE_EQ(blk.history().snapshotTime(0), 0.2);
    EXPECT_DOUBLE_EQ(blk.history().snapshotTime(1), 0.1);
}

TEST(AuxiliaryStateMultiBlock, ClearRemovesBlocks)
{
    AuxiliaryState state;

    AuxiliaryStateSpec blk; blk.name = "Y"; blk.size = 1;
    state.registerBlock(blk, 1);

    state.clear();

    EXPECT_EQ(state.blockCount(), 0u);
    EXPECT_FALSE(state.hasBlock("Y"));
}

// ---------------------------------------------------------------------------
//  Layout-aware gather/scatter tests
// ---------------------------------------------------------------------------

TEST(AuxiliaryBlockStorage, GatherScatter_ByEntityThenComponent)
{
    AuxiliaryBlockStorage storage;
    AuxiliaryStateSpec spec;
    spec.name = "v";
    spec.size = 2;
    spec.ordering = AuxiliaryEntityOrdering::ByEntityThenComponent;
    storage.setupFixedStride(spec, 3);

    // Write: entity0=[1,2], entity1=[3,4], entity2=[5,6]
    auto w = storage.work();
    w[0]=1; w[1]=2; w[2]=3; w[3]=4; w[4]=5; w[5]=6;

    auto g0 = storage.gatherEntityWork(0);
    ASSERT_EQ(g0.size(), 2u);
    EXPECT_DOUBLE_EQ(g0[0], 1.0);
    EXPECT_DOUBLE_EQ(g0[1], 2.0);

    auto g2 = storage.gatherEntityWork(2);
    EXPECT_DOUBLE_EQ(g2[0], 5.0);
    EXPECT_DOUBLE_EQ(g2[1], 6.0);

    // Scatter back
    storage.scatterEntityWork(1, std::vector<Real>{30.0, 40.0});
    EXPECT_DOUBLE_EQ(storage.work()[2], 30.0);
    EXPECT_DOUBLE_EQ(storage.work()[3], 40.0);
}

TEST(AuxiliaryBlockStorage, GatherScatter_ByComponentThenEntity)
{
    AuxiliaryBlockStorage storage;
    AuxiliaryStateSpec spec;
    spec.name = "v";
    spec.size = 2;
    spec.ordering = AuxiliaryEntityOrdering::ByComponentThenEntity;
    storage.setupFixedStride(spec, 3);

    // Layout: [comp0_e0, comp0_e1, comp0_e2, comp1_e0, comp1_e1, comp1_e2]
    auto w = storage.work();
    w[0]=10; w[1]=20; w[2]=30; // component 0 for entities 0,1,2
    w[3]=11; w[4]=21; w[5]=31; // component 1 for entities 0,1,2

    auto g0 = storage.gatherEntityWork(0);
    ASSERT_EQ(g0.size(), 2u);
    EXPECT_DOUBLE_EQ(g0[0], 10.0); // comp0
    EXPECT_DOUBLE_EQ(g0[1], 11.0); // comp1

    auto g1 = storage.gatherEntityWork(1);
    EXPECT_DOUBLE_EQ(g1[0], 20.0);
    EXPECT_DOUBLE_EQ(g1[1], 21.0);

    auto g2 = storage.gatherEntityWork(2);
    EXPECT_DOUBLE_EQ(g2[0], 30.0);
    EXPECT_DOUBLE_EQ(g2[1], 31.0);

    // Scatter: write entity 1 = [99, 88]
    storage.scatterEntityWork(1, std::vector<Real>{99.0, 88.0});
    EXPECT_DOUBLE_EQ(storage.work()[1], 99.0); // comp0_e1
    EXPECT_DOUBLE_EQ(storage.work()[4], 88.0); // comp1_e1

    // Committed gather
    storage.initialize(storage.work()); // copy work→committed
    auto gc = storage.gatherEntityCommitted(1);
    EXPECT_DOUBLE_EQ(gc[0], 99.0);
    EXPECT_DOUBLE_EQ(gc[1], 88.0);
}

TEST(AuxiliaryBlockStorage, GatherScatter_Ragged)
{
    AuxiliaryBlockStorage storage;
    AuxiliaryStateSpec spec;
    spec.name = "ragged";
    spec.size = 0;
    spec.scope = AuxiliaryStateScope::Cell;
    spec.layout_mode = AuxiliaryLayoutMode::Ragged;

    // 3 entities: sizes 2, 3, 1 → total 6
    const std::vector<std::size_t> offsets = {0, 2, 5, 6};
    storage.setupRagged(spec, offsets);

    auto w = storage.work();
    w[0]=10; w[1]=20; // entity 0
    w[2]=30; w[3]=40; w[4]=50; // entity 1
    w[5]=60; // entity 2

    auto g0 = storage.gatherEntityWork(0);
    ASSERT_EQ(g0.size(), 2u);
    EXPECT_DOUBLE_EQ(g0[0], 10.0);
    EXPECT_DOUBLE_EQ(g0[1], 20.0);

    auto g1 = storage.gatherEntityWork(1);
    ASSERT_EQ(g1.size(), 3u);
    EXPECT_DOUBLE_EQ(g1[0], 30.0);
    EXPECT_DOUBLE_EQ(g1[2], 50.0);

    auto g2 = storage.gatherEntityWork(2);
    ASSERT_EQ(g2.size(), 1u);
    EXPECT_DOUBLE_EQ(g2[0], 60.0);

    // Scatter
    storage.scatterEntityWork(1, std::vector<Real>{300, 400, 500});
    EXPECT_DOUBLE_EQ(storage.work()[2], 300.0);
    EXPECT_DOUBLE_EQ(storage.work()[4], 500.0);
    EXPECT_THROW(storage.scatterEntityWork(1, std::vector<Real>{1.0, 2.0}),
                 svmp::FE::InvalidArgumentException);
}

TEST(AuxiliaryBlockStorage, GatherHistory_ByComponentThenEntity)
{
    AuxiliaryBlockStorage storage;
    AuxiliaryStateSpec spec;
    spec.name = "v";
    spec.size = 2;
    spec.ordering = AuxiliaryEntityOrdering::ByComponentThenEntity;
    spec.history_mode = AuxiliaryHistoryMode::SingleStep;
    storage.setupFixedStride(spec, 3);

    // Initialize with values, then commit twice to push history.
    auto w = storage.work();
    w[0]=10; w[1]=20; w[2]=30; w[3]=11; w[4]=21; w[5]=31;
    // First commit: pushes zeros (initial committed) into history, copies work→committed.
    storage.commitTimeStep(0.1);
    // Second commit: pushes the [10,20,30,11,21,31] committed into history.
    w = storage.work();
    w[0]=100; w[1]=200; w[2]=300; w[3]=110; w[4]=210; w[5]=310;
    storage.commitTimeStep(0.2);

    // History[0] should be the state committed at t=0.2 (which was the work at t=0.1).
    auto h1 = storage.gatherEntityHistory(0, 1);
    ASSERT_EQ(h1.size(), 2u);
    EXPECT_DOUBLE_EQ(h1[0], 20.0); // comp0_e1 from first committed
    EXPECT_DOUBLE_EQ(h1[1], 21.0); // comp1_e1 from first committed
}

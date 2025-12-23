/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

/**
 * @file test_WorkStreamAssembler.cpp
 * @brief Unit tests for WorkStreamAssembler and Scratch/Copy pattern assembly
 */

#include <gtest/gtest.h>

#include "Assembly/WorkStreamAssembler.h"
#include "Assembly/AssemblyContext.h"

#include <vector>
#include <atomic>
#include <thread>
#include <stdexcept>

namespace svmp {
namespace FE {
namespace assembly {
namespace test {

// ============================================================================
// ScratchData Tests
// ============================================================================

TEST(ScratchDataTest, DefaultConstruction) {
    ScratchData scratch;
    // Default construction should succeed
    EXPECT_TRUE(scratch.scratchMatrix().empty());
    EXPECT_TRUE(scratch.scratchVector().empty());
}

TEST(ScratchDataTest, ParameterizedConstruction) {
    ScratchData scratch(10, 4, 3);  // max_dofs=10, max_qpts=4, dim=3

    // Should have reserved storage
    SUCCEED();
}

TEST(ScratchDataTest, Reserve) {
    ScratchData scratch;
    scratch.reserve(8, 6, 2);  // max_dofs=8, max_qpts=6, dim=2

    // After reserve, internal storage should be ready
    SUCCEED();
}

TEST(ScratchDataTest, Clear) {
    ScratchData scratch(10, 4, 3);
    scratch.clear();

    // Clear should not throw and should allow reuse
    SUCCEED();
}

TEST(ScratchDataTest, ContextAccess) {
    ScratchData scratch;

    AssemblyContext& ctx = scratch.context();
    const ScratchData& const_scratch = scratch;
    [[maybe_unused]] const AssemblyContext& const_ctx = const_scratch.context();

    // Context should be accessible
    SUCCEED();
}

TEST(ScratchDataTest, ScratchMatrixAccess) {
    ScratchData scratch;

    std::vector<Real>& mat = scratch.scratchMatrix();
    mat.resize(16, 0.0);
    mat[0] = 1.0;

    EXPECT_DOUBLE_EQ(scratch.scratchMatrix()[0], 1.0);
}

TEST(ScratchDataTest, ScratchVectorAccess) {
    ScratchData scratch;

    std::vector<Real>& vec = scratch.scratchVector();
    vec.resize(4, 0.0);
    vec[2] = 3.14;

    EXPECT_DOUBLE_EQ(scratch.scratchVector()[2], 3.14);
}

TEST(ScratchDataTest, UserData) {
    ScratchData scratch;

    auto user = std::make_shared<int>(42);
    scratch.setUserData(user);

    int* retrieved = scratch.getUserData<int>();
    ASSERT_NE(retrieved, nullptr);
    EXPECT_EQ(*retrieved, 42);
}

TEST(ScratchDataTest, CopyConstruction) {
    ScratchData scratch1(10, 4, 3);
    scratch1.scratchMatrix().resize(16, 1.5);

    ScratchData scratch2(scratch1);

    EXPECT_EQ(scratch2.scratchMatrix().size(), scratch1.scratchMatrix().size());
}

TEST(ScratchDataTest, MoveConstruction) {
    ScratchData scratch1(10, 4, 3);
    scratch1.scratchMatrix().resize(16, 2.5);

    ScratchData scratch2(std::move(scratch1));

    EXPECT_EQ(scratch2.scratchMatrix().size(), 16u);
}

// ============================================================================
// CopyData Tests
// ============================================================================

TEST(CopyDataTest, DefaultConstruction) {
    CopyData copy;

    EXPECT_FALSE(copy.isValid());
    EXPECT_FALSE(copy.has_matrix);
    EXPECT_FALSE(copy.has_vector);
    EXPECT_EQ(copy.cell_id, -1);
}

TEST(CopyDataTest, Reserve) {
    CopyData copy;
    copy.reserve(10);  // max_dofs = 10

    // Reserve should not throw
    SUCCEED();
}

TEST(CopyDataTest, Clear) {
    CopyData copy;
    copy.local_matrix.resize(16, 1.0);
    copy.has_matrix = true;
    copy.setValid(true);

    copy.clear();

    EXPECT_FALSE(copy.isValid());
}

TEST(CopyDataTest, ValidityFlag) {
    CopyData copy;

    EXPECT_FALSE(copy.isValid());

    copy.setValid(true);
    EXPECT_TRUE(copy.isValid());

    copy.setValid(false);
    EXPECT_FALSE(copy.isValid());
}

TEST(CopyDataTest, LocalMatrixStorage) {
    CopyData copy;

    // Store a 4x4 local matrix
    copy.local_matrix.resize(16, 0.0);
    for (int i = 0; i < 4; ++i) {
        copy.local_matrix[static_cast<std::size_t>(i * 4 + i)] = 1.0;  // Identity
    }
    copy.has_matrix = true;

    EXPECT_TRUE(copy.has_matrix);
    EXPECT_EQ(copy.local_matrix.size(), 16u);
    EXPECT_DOUBLE_EQ(copy.local_matrix[0], 1.0);
    EXPECT_DOUBLE_EQ(copy.local_matrix[5], 1.0);
}

TEST(CopyDataTest, LocalVectorStorage) {
    CopyData copy;

    copy.local_vector = {1.0, 2.0, 3.0, 4.0};
    copy.has_vector = true;

    EXPECT_TRUE(copy.has_vector);
    EXPECT_EQ(copy.local_vector.size(), 4u);
}

TEST(CopyDataTest, DofStorage) {
    CopyData copy;

    copy.row_dofs = {0, 1, 2, 3};
    copy.col_dofs = {0, 1, 2, 3};

    EXPECT_EQ(copy.row_dofs.size(), 4u);
    EXPECT_EQ(copy.col_dofs.size(), 4u);
}

TEST(CopyDataTest, CellId) {
    CopyData copy;

    copy.cell_id = 42;

    EXPECT_EQ(copy.cell_id, 42);
}

TEST(CopyDataTest, FaceBlocks) {
    CopyData copy;

    // Add face blocks for DG assembly
    CopyData::FaceBlock block;
    block.matrix.resize(16, 0.5);
    block.row_dofs = {0, 1, 2, 3};
    block.col_dofs = {4, 5, 6, 7};

    copy.face_blocks.push_back(block);

    EXPECT_EQ(copy.face_blocks.size(), 1u);
    EXPECT_EQ(copy.face_blocks[0].matrix.size(), 16u);
}

// ============================================================================
// WorkStreamOptions Tests
// ============================================================================

TEST(WorkStreamOptionsTest, Defaults) {
    WorkStreamOptions options;

    EXPECT_EQ(options.num_threads, 4);
    EXPECT_EQ(options.chunk_size, 64u);
    EXPECT_EQ(options.max_queue_depth, 1024u);
    EXPECT_TRUE(options.deterministic_copier);
    EXPECT_FALSE(options.enable_profiling);
    EXPECT_TRUE(options.use_constraints);
}

TEST(WorkStreamOptionsTest, CustomValues) {
    WorkStreamOptions options;
    options.num_threads = 8;
    options.chunk_size = 128;
    options.deterministic_copier = false;

    EXPECT_EQ(options.num_threads, 8);
    EXPECT_EQ(options.chunk_size, 128u);
    EXPECT_FALSE(options.deterministic_copier);
}

// ============================================================================
// WorkStreamStats Tests
// ============================================================================

TEST(WorkStreamStatsTest, DefaultValues) {
    WorkStreamStats stats;

    EXPECT_EQ(stats.elements_processed, 0);
    EXPECT_EQ(stats.faces_processed, 0);
    EXPECT_DOUBLE_EQ(stats.total_seconds, 0.0);
    EXPECT_DOUBLE_EQ(stats.worker_seconds, 0.0);
    EXPECT_DOUBLE_EQ(stats.copier_seconds, 0.0);
    EXPECT_EQ(stats.queue_highwater, 0u);
}

// ============================================================================
// WorkStreamAssembler Tests
// ============================================================================

TEST(WorkStreamAssemblerTest, DefaultConstruction) {
    WorkStreamAssembler assembler;

    EXPECT_FALSE(assembler.isConfigured());
}

TEST(WorkStreamAssemblerTest, ConstructionWithOptions) {
    WorkStreamOptions ws_options;
    ws_options.num_threads = 16;
    ws_options.deterministic_copier = false;

    WorkStreamAssembler assembler(ws_options);
    const auto& opts = assembler.getWorkStreamOptions();

    EXPECT_EQ(opts.num_threads, 16);
    EXPECT_FALSE(opts.deterministic_copier);
}

TEST(WorkStreamAssemblerTest, SetWorkStreamOptions) {
    WorkStreamAssembler assembler;

    WorkStreamOptions options;
    options.num_threads = 2;
    options.chunk_size = 32;

    assembler.setWorkStreamOptions(options);

    EXPECT_EQ(assembler.getWorkStreamOptions().num_threads, 2);
    EXPECT_EQ(assembler.getWorkStreamOptions().chunk_size, 32u);
}

TEST(WorkStreamAssemblerTest, SetAssemblyOptions) {
    WorkStreamAssembler assembler;

    AssemblyOptions options;
    options.deterministic = false;

    assembler.setOptions(options);

    EXPECT_FALSE(assembler.getOptions().deterministic);
}

TEST(WorkStreamAssemblerTest, MoveConstruction) {
    WorkStreamOptions options;
    options.num_threads = 6;

    WorkStreamAssembler assembler1(options);
    WorkStreamAssembler assembler2(std::move(assembler1));

    EXPECT_EQ(assembler2.getWorkStreamOptions().num_threads, 6);
}

TEST(WorkStreamAssemblerTest, MoveAssignment) {
    WorkStreamOptions options;
    options.num_threads = 3;

    WorkStreamAssembler assembler1(options);
    WorkStreamAssembler assembler2;

    assembler2 = std::move(assembler1);

    EXPECT_EQ(assembler2.getWorkStreamOptions().num_threads, 3);
}

TEST(WorkStreamAssemblerTest, Initialize) {
    WorkStreamAssembler assembler;

    EXPECT_THROW(assembler.initialize(), std::runtime_error);
}

TEST(WorkStreamAssemblerTest, Reset) {
    WorkStreamAssembler assembler;
    EXPECT_NO_THROW(assembler.reset());
}

TEST(WorkStreamAssemblerTest, GetLastStats) {
    WorkStreamAssembler assembler;

    const WorkStreamStats& stats = assembler.getLastStats();

    // Initial stats should be zeroed
    EXPECT_EQ(stats.elements_processed, 0);
}

TEST(WorkStreamAssemblerTest, ScratchDataFactory) {
    WorkStreamAssembler assembler;

    bool factory_called = false;
    assembler.setScratchDataFactory([&factory_called]() {
        factory_called = true;
        return ScratchData(8, 4, 3);
    });

    // Factory should be stored (will be called during assembly)
    SUCCEED();
}

// ============================================================================
// Factory Tests
// ============================================================================

TEST(WorkStreamAssemblerFactoryTest, CreateDefault) {
    auto assembler = createWorkStreamAssembler();

    EXPECT_NE(assembler, nullptr);
}

TEST(WorkStreamAssemblerFactoryTest, CreateWithOptions) {
    WorkStreamOptions options;
    options.num_threads = 12;

    auto assembler = createWorkStreamAssembler(options);

    EXPECT_NE(assembler, nullptr);

    // Cast to WorkStreamAssembler to check options
    auto* ws = dynamic_cast<WorkStreamAssembler*>(assembler.get());
    if (ws) {
        EXPECT_EQ(ws->getWorkStreamOptions().num_threads, 12);
    }
}

// ============================================================================
// Thread Safety Tests (Basic)
// ============================================================================

TEST(WorkStreamAssemblerTest, ConcurrentScratchDataCreation) {
    // Test that ScratchData can be created in multiple threads
    std::atomic<int> count{0};
    std::vector<std::thread> threads;

    for (int i = 0; i < 4; ++i) {
        threads.emplace_back([&count]() {
            ScratchData scratch(10, 4, 3);
            scratch.reserve(10, 4, 3);
            count++;
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    EXPECT_EQ(count.load(), 4);
}

TEST(WorkStreamAssemblerTest, ConcurrentCopyDataCreation) {
    // Test that CopyData can be created in multiple threads
    std::atomic<int> count{0};
    std::vector<std::thread> threads;

    for (int i = 0; i < 4; ++i) {
        threads.emplace_back([&count]() {
            CopyData copy;
            copy.reserve(8);
            copy.local_matrix.resize(64, 1.0);
            copy.setValid(true);
            count++;
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    EXPECT_EQ(count.load(), 4);
}

} // namespace test
} // namespace assembly
} // namespace FE
} // namespace svmp

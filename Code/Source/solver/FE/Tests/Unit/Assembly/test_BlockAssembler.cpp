/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

/**
 * @file test_BlockAssembler.cpp
 * @brief Unit tests for BlockAssembler
 */

#include <gtest/gtest.h>
#include "Assembly/BlockAssembler.h"
#include "Core/Types.h"

#include <vector>
#include <memory>
#include <cmath>
#include <string>

namespace svmp {
namespace FE {
namespace assembly {
namespace testing {

namespace {

// ============================================================================
// Mock Classes
// ============================================================================

/**
 * @brief Mock mesh access for block assembly testing
 */
class MockMeshAccess : public IMeshAccess {
public:
    MockMeshAccess(GlobalIndex num_cells = 50)
        : num_cells_(num_cells)
    {
    }

    GlobalIndex numCells() const override { return num_cells_; }
    GlobalIndex numOwnedCells() const override { return num_cells_; }
    GlobalIndex numBoundaryFaces() const override { return 0; }
    GlobalIndex numInteriorFaces() const override { return 0; }
    int dimension() const override { return 3; }

    bool isOwnedCell(GlobalIndex /*cell_id*/) const override { return true; }

    ElementType getCellType(GlobalIndex /*cell_id*/) const override {
        return ElementType::Tetra4;
    }

    void getCellNodes(GlobalIndex cell_id, std::vector<GlobalIndex>& nodes) const override {
        nodes.clear();
        nodes.push_back(cell_id * 4);
        nodes.push_back(cell_id * 4 + 1);
        nodes.push_back(cell_id * 4 + 2);
        nodes.push_back(cell_id * 4 + 3);
    }

    [[nodiscard]] std::array<Real, 3> getNodeCoordinates(GlobalIndex node_id) const override {
        return {
            static_cast<Real>(node_id % 10),
            static_cast<Real>((node_id / 10) % 10),
            static_cast<Real>(node_id / 100)
        };
    }

    void getCellCoordinates(GlobalIndex cell_id,
                           std::vector<std::array<Real, 3>>& coords) const override {
        coords.clear();
        for (int i = 0; i < 4; ++i) {
            coords.push_back(getNodeCoordinates(cell_id * 4 + i));
        }
    }

    [[nodiscard]] LocalIndex getLocalFaceIndex(GlobalIndex /*face_id*/,
                                               GlobalIndex /*cell_id*/) const override {
        return 0;
    }

    int getBoundaryFaceMarker(GlobalIndex /*face_id*/) const override {
        return 0;
    }

    std::pair<GlobalIndex, GlobalIndex> getInteriorFaceCells(GlobalIndex /*face_id*/) const override {
        return {-1, -1};
    }

    void forEachCell(std::function<void(GlobalIndex)> callback) const override {
        for (GlobalIndex i = 0; i < num_cells_; ++i) {
            callback(i);
        }
    }

    void forEachOwnedCell(std::function<void(GlobalIndex)> callback) const override {
        forEachCell(callback);
    }

    void forEachBoundaryFace(int /*marker*/,
        std::function<void(GlobalIndex, GlobalIndex)> /*callback*/) const override {}

    void forEachInteriorFace(
        std::function<void(GlobalIndex, GlobalIndex, GlobalIndex)> /*callback*/) const override {}

private:
    GlobalIndex num_cells_;
};

/**
 * @brief Simple mock block kernel for testing
 *
 * Implements a 2-field system (like Stokes)
 */
class MockBlockKernel : public IBlockKernel {
public:
    void computeBlock(
        AssemblyContext& /*context*/,
        FieldId row_field,
        FieldId col_field,
        KernelOutput& output) override
    {
        // Simple identity-like blocks
        LocalIndex n_row = (row_field == 0) ? 12 : 4;  // velocity: 3*4=12, pressure: 4
        LocalIndex n_col = (col_field == 0) ? 12 : 4;

        output.local_matrix.resize(static_cast<std::size_t>(n_row * n_col), 0.0);

        // Fill with identifiable values
        for (LocalIndex i = 0; i < n_row && i < n_col; ++i) {
            output.local_matrix[static_cast<std::size_t>(i * n_col + i)] =
                1.0 + row_field * 10 + col_field;
        }
    }

    void computeRhs(
        AssemblyContext& /*context*/,
        FieldId field,
        KernelOutput& output) override
    {
        LocalIndex n = (field == 0) ? 12 : 4;
        output.local_vector.assign(static_cast<std::size_t>(n), 1.0 + field);
    }

    [[nodiscard]] bool hasBlock(FieldId row_field, FieldId col_field) const override {
        // 2x2 block system
        return row_field < 2 && col_field < 2;
    }

    [[nodiscard]] int numFields() const override { return 2; }
};

} // namespace

// ============================================================================
// Test Fixtures
// ============================================================================

class BlockAssemblerTest : public ::testing::Test {
protected:
    void SetUp() override {
        mesh_ = std::make_unique<MockMeshAccess>(50);
        kernel_ = std::make_unique<MockBlockKernel>();
    }

    std::unique_ptr<MockMeshAccess> mesh_;
    std::unique_ptr<MockBlockKernel> kernel_;
};

// ============================================================================
// Construction Tests
// ============================================================================

TEST_F(BlockAssemblerTest, DefaultConstruction) {
    BlockAssembler assembler;
    EXPECT_FALSE(assembler.isConfigured());
    EXPECT_EQ(assembler.numFields(), 0);
}

TEST_F(BlockAssemblerTest, ConstructionWithOptions) {
    BlockAssemblerOptions options;
    options.mode = BlockAssemblyMode::Block;
    options.num_threads = 4;
    options.apply_constraints = false;

    BlockAssembler assembler(options);

    EXPECT_EQ(assembler.getOptions().mode, BlockAssemblyMode::Block);
    EXPECT_EQ(assembler.getOptions().num_threads, 4);
    EXPECT_FALSE(assembler.getOptions().apply_constraints);
}

// ============================================================================
// Configuration Tests
// ============================================================================

TEST_F(BlockAssemblerTest, SetMesh) {
    BlockAssembler assembler;
    assembler.setMesh(*mesh_);
    EXPECT_FALSE(assembler.isConfigured());  // Needs fields and kernel
}

TEST_F(BlockAssemblerTest, SetKernel) {
    BlockAssembler assembler;
    assembler.setKernel(*kernel_);
    EXPECT_FALSE(assembler.isConfigured());  // Needs mesh and fields
}

TEST_F(BlockAssemblerTest, SetOptions) {
    BlockAssembler assembler;

    BlockAssemblerOptions options;
    options.num_threads = 8;
    options.mode = BlockAssemblyMode::Segregated;

    assembler.setOptions(options);

    EXPECT_EQ(assembler.getOptions().num_threads, 8);
    EXPECT_EQ(assembler.getOptions().mode, BlockAssemblyMode::Segregated);
}

// ============================================================================
// Block Assembly Mode Tests
// ============================================================================

TEST_F(BlockAssemblerTest, BlockAssemblyModeEnum) {
    EXPECT_NE(BlockAssemblyMode::Monolithic, BlockAssemblyMode::Block);
    EXPECT_NE(BlockAssemblyMode::Block, BlockAssemblyMode::Segregated);
}

// ============================================================================
// Options Tests
// ============================================================================

TEST_F(BlockAssemblerTest, BlockAssemblerOptionsDefaults) {
    BlockAssemblerOptions options;

    EXPECT_EQ(options.mode, BlockAssemblyMode::Monolithic);
    EXPECT_EQ(options.num_threads, 0);  // Auto
    EXPECT_TRUE(options.apply_constraints);
    EXPECT_FALSE(options.verbose);
}

// ============================================================================
// Block Index Tests
// ============================================================================

TEST_F(BlockAssemblerTest, BlockIndexConstruction) {
    BlockIndex idx(1, 2);

    EXPECT_EQ(idx.row_field, 1);
    EXPECT_EQ(idx.col_field, 2);
}

TEST_F(BlockAssemblerTest, BlockIndexEquality) {
    BlockIndex a(0, 1);
    BlockIndex b(0, 1);
    BlockIndex c(1, 0);

    EXPECT_TRUE(a == b);
    EXPECT_FALSE(a == c);
}

TEST_F(BlockAssemblerTest, BlockIndexLessThan) {
    BlockIndex a(0, 0);
    BlockIndex b(0, 1);
    BlockIndex c(1, 0);

    EXPECT_TRUE(a < b);
    EXPECT_TRUE(a < c);
    EXPECT_TRUE(b < c);
}

TEST_F(BlockAssemblerTest, BlockIndexIsDiagonal) {
    BlockIndex diag(1, 1);
    BlockIndex off(0, 1);

    EXPECT_TRUE(diag.isDiagonal());
    EXPECT_FALSE(off.isDiagonal());
}

// ============================================================================
// Field Configuration Tests
// ============================================================================

TEST_F(BlockAssemblerTest, FieldConfigDefaults) {
    FieldConfig config;

    EXPECT_EQ(config.space, nullptr);
    EXPECT_EQ(config.dof_map, nullptr);
    EXPECT_EQ(config.constraints, nullptr);
    EXPECT_EQ(config.components, 1);
    EXPECT_FALSE(config.is_pressure_like);
}

TEST_F(BlockAssemblerTest, FieldConfigPopulation) {
    FieldConfig config;
    config.id = 0;
    config.name = "velocity";
    config.components = 3;
    config.is_pressure_like = false;

    EXPECT_EQ(config.id, 0);
    EXPECT_EQ(config.name, "velocity");
    EXPECT_EQ(config.components, 3);
}

// ============================================================================
// Block System Configuration Tests
// ============================================================================

TEST_F(BlockAssemblerTest, BlockSystemConfigDefaults) {
    BlockSystemConfig config;
    EXPECT_EQ(config.numFields(), 0);
    EXPECT_TRUE(config.fields.empty());
}

TEST_F(BlockAssemblerTest, BlockSystemConfigGetField) {
    BlockSystemConfig config;

    // Empty config should return nullptr
    auto field = config.getField(0);
    EXPECT_EQ(field, nullptr);
}

// ============================================================================
// Statistics Tests
// ============================================================================

TEST_F(BlockAssemblerTest, BlockAssemblyStatsDefaults) {
    BlockAssemblyStats stats;

    EXPECT_EQ(stats.num_cells, 0);
    EXPECT_DOUBLE_EQ(stats.total_seconds, 0.0);
    EXPECT_TRUE(stats.block_assembly_seconds.empty());
    EXPECT_TRUE(stats.block_nnz.empty());
}

TEST_F(BlockAssemblerTest, BlockAssemblyStatsPopulation) {
    BlockAssemblyStats stats;

    stats.num_cells = 100;
    stats.total_seconds = 0.5;
    stats.block_assembly_seconds[BlockIndex(0, 0)] = 0.1;
    stats.block_assembly_seconds[BlockIndex(0, 1)] = 0.2;
    stats.block_nnz[BlockIndex(0, 0)] = 1000;

    EXPECT_EQ(stats.num_cells, 100);
    EXPECT_DOUBLE_EQ(stats.total_seconds, 0.5);
    EXPECT_EQ(stats.block_assembly_seconds.size(), 2u);
    EXPECT_EQ(stats.block_nnz.size(), 1u);
}

// ============================================================================
// Kernel Interface Tests
// ============================================================================

TEST_F(BlockAssemblerTest, MockKernelNumFields) {
    EXPECT_EQ(kernel_->numFields(), 2);
}

TEST_F(BlockAssemblerTest, MockKernelHasBlock) {
    EXPECT_TRUE(kernel_->hasBlock(0, 0));
    EXPECT_TRUE(kernel_->hasBlock(0, 1));
    EXPECT_TRUE(kernel_->hasBlock(1, 0));
    EXPECT_TRUE(kernel_->hasBlock(1, 1));
    EXPECT_FALSE(kernel_->hasBlock(2, 0));  // Out of range
}

TEST_F(BlockAssemblerTest, MockKernelComputeBlock) {
    AssemblyContext context;
    KernelOutput output;

    kernel_->computeBlock(context, 0, 0, output);

    // Velocity-velocity block: 12x12
    EXPECT_EQ(output.local_matrix.size(), 144u);

    // Check diagonal has expected value: 1.0 + 0*10 + 0 = 1.0
    EXPECT_DOUBLE_EQ(output.local_matrix[0], 1.0);
}

TEST_F(BlockAssemblerTest, MockKernelComputeRhs) {
    AssemblyContext context;
    KernelOutput output;

    // Velocity RHS
    kernel_->computeRhs(context, 0, output);
    EXPECT_EQ(output.local_vector.size(), 12u);
    EXPECT_DOUBLE_EQ(output.local_vector[0], 1.0);  // 1.0 + 0

    // Pressure RHS
    kernel_->computeRhs(context, 1, output);
    EXPECT_EQ(output.local_vector.size(), 4u);
    EXPECT_DOUBLE_EQ(output.local_vector[0], 2.0);  // 1.0 + 1
}

TEST_F(BlockAssemblerTest, MockKernelGetRequiredData) {
    auto required = kernel_->getRequiredData(0, 0);
    EXPECT_TRUE((required & RequiredData::BasisValues) != RequiredData::None);
}

// ============================================================================
// Move Semantics Tests
// ============================================================================

TEST_F(BlockAssemblerTest, MoveConstruction) {
    BlockAssemblerOptions options;
    options.num_threads = 4;

    BlockAssembler assembler1(options);
    assembler1.setMesh(*mesh_);

    BlockAssembler assembler2(std::move(assembler1));

    EXPECT_EQ(assembler2.getOptions().num_threads, 4);
}

TEST_F(BlockAssemblerTest, MoveAssignment) {
    BlockAssemblerOptions options;
    options.num_threads = 8;

    BlockAssembler assembler1(options);

    BlockAssembler assembler2;
    assembler2 = std::move(assembler1);

    EXPECT_EQ(assembler2.getOptions().num_threads, 8);
}

// ============================================================================
// Factory Function Tests
// ============================================================================

TEST_F(BlockAssemblerTest, CreateBlockAssembler) {
    auto assembler = createBlockAssembler();
    EXPECT_NE(assembler, nullptr);
    EXPECT_FALSE(assembler->isConfigured());
}

TEST_F(BlockAssemblerTest, CreateBlockAssemblerWithOptions) {
    BlockAssemblerOptions options;
    options.mode = BlockAssemblyMode::Segregated;

    auto assembler = createBlockAssembler(options);
    EXPECT_NE(assembler, nullptr);
    EXPECT_EQ(assembler->getOptions().mode, BlockAssemblyMode::Segregated);
}

// ============================================================================
// Last Stats Tests
// ============================================================================

TEST_F(BlockAssemblerTest, GetLastStats) {
    BlockAssembler assembler;

    auto stats = assembler.getLastStats();
    EXPECT_EQ(stats.num_cells, 0);
    EXPECT_DOUBLE_EQ(stats.total_seconds, 0.0);
}

// ============================================================================
// Number of Fields Tests
// ============================================================================

TEST_F(BlockAssemblerTest, NumFieldsEmpty) {
    BlockAssembler assembler;
    EXPECT_EQ(assembler.numFields(), 0);
}

// ============================================================================
// Get Config Tests
// ============================================================================

TEST_F(BlockAssemblerTest, GetConfigEmpty) {
    BlockAssembler assembler;

    auto config = assembler.getConfig();
    EXPECT_EQ(config.numFields(), 0);
}

} // namespace testing
} // namespace assembly
} // namespace FE
} // namespace svmp

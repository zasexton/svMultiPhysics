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
#include "Assembly/StandardAssembler.h"
#include "Basis/BasisFunction.h"
#include "Dofs/DofMap.h"
#include "Elements/Element.h"
#include "Spaces/FunctionSpace.h"

#include <algorithm>
#include <array>
#include <memory>
#include <set>
#include <string>
#include <vector>

namespace svmp {
namespace FE {
namespace assembly {
namespace testing {

namespace {

// ============================================================================
// Mock Mesh Access
// ============================================================================

class MockMeshAccess : public IMeshAccess {
public:
    explicit MockMeshAccess(GlobalIndex num_cells = 2)
        : num_cells_(num_cells)
    {
    }

    [[nodiscard]] GlobalIndex numCells() const override { return num_cells_; }
    [[nodiscard]] GlobalIndex numOwnedCells() const override { return num_cells_; }
    [[nodiscard]] GlobalIndex numBoundaryFaces() const override { return 0; }
    [[nodiscard]] GlobalIndex numInteriorFaces() const override { return 0; }
    [[nodiscard]] int dimension() const override { return 3; }

    [[nodiscard]] bool isOwnedCell(GlobalIndex /*cell_id*/) const override { return true; }

    [[nodiscard]] ElementType getCellType(GlobalIndex /*cell_id*/) const override {
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

    void getCellCoordinates(GlobalIndex cell_id, std::vector<std::array<Real, 3>>& coords) const override {
        coords.clear();
        for (int i = 0; i < 4; ++i) {
            coords.push_back(getNodeCoordinates(cell_id * 4 + i));
        }
    }

    [[nodiscard]] LocalIndex getLocalFaceIndex(GlobalIndex /*face_id*/,
                                               GlobalIndex /*cell_id*/) const override
    {
        return 0;
    }

    [[nodiscard]] int getBoundaryFaceMarker(GlobalIndex /*face_id*/) const override { return 0; }

    [[nodiscard]] std::pair<GlobalIndex, GlobalIndex> getInteriorFaceCells(GlobalIndex /*face_id*/) const override
    {
        return {-1, -1};
    }

    void forEachCell(std::function<void(GlobalIndex)> callback) const override {
        for (GlobalIndex i = 0; i < num_cells_; ++i) {
            callback(i);
        }
    }

    void forEachOwnedCell(std::function<void(GlobalIndex)> callback) const override { forEachCell(callback); }

    void forEachBoundaryFace(int /*marker*/,
                             std::function<void(GlobalIndex, GlobalIndex)> /*callback*/) const override
    {
    }

    void forEachInteriorFace(
        std::function<void(GlobalIndex, GlobalIndex, GlobalIndex)> /*callback*/) const override
    {
    }

private:
    GlobalIndex num_cells_{0};
};

// ============================================================================
// Minimal FunctionSpace / Element for BlockAssembler field registration
// ============================================================================

inline dofs::DofMap createTestDofMap5()
{
    dofs::DofMap dof_map(2, 5, 4);  // 2 cells, 5 total DOFs, 4 dofs per cell

    const std::vector<GlobalIndex> cell0_dofs = {0, 1, 2, 3};
    const std::vector<GlobalIndex> cell1_dofs = {1, 2, 3, 4};

    dof_map.setCellDofs(0, cell0_dofs);
    dof_map.setCellDofs(1, cell1_dofs);
    dof_map.setNumDofs(5);
    dof_map.setNumLocalDofs(5);
    dof_map.finalize();

    return dof_map;
}

inline dofs::DofMap createTestDofMap4()
{
    dofs::DofMap dof_map(2, 4, 4);  // 2 cells, 4 total DOFs, 4 dofs per cell

    const std::vector<GlobalIndex> cell0_dofs = {0, 1, 2, 3};
    const std::vector<GlobalIndex> cell1_dofs = {0, 1, 2, 3};

    dof_map.setCellDofs(0, cell0_dofs);
    dof_map.setCellDofs(1, cell1_dofs);
    dof_map.setNumDofs(4);
    dof_map.setNumLocalDofs(4);
    dof_map.finalize();

    return dof_map;
}

class MockBasis final : public basis::BasisFunction {
public:
    [[nodiscard]] BasisType basis_type() const noexcept override { return BasisType::Lagrange; }
    [[nodiscard]] ElementType element_type() const noexcept override { return ElementType::Tetra4; }
    [[nodiscard]] int dimension() const noexcept override { return 3; }
    [[nodiscard]] int order() const noexcept override { return 1; }
    [[nodiscard]] std::size_t size() const noexcept override { return 4; }

    void evaluate_values(const math::Vector<Real, 3>& xi, std::vector<Real>& values) const override
    {
        values.resize(4);
        values[0] = 1.0 - xi[0] - xi[1] - xi[2];
        values[1] = xi[0];
        values[2] = xi[1];
        values[3] = xi[2];
    }
};

class MockElement final : public elements::Element {
public:
    MockElement()
        : basis_(std::make_shared<MockBasis>())
    {
    }

    [[nodiscard]] elements::ElementInfo info() const noexcept override
    {
        return {ElementType::Tetra4, FieldType::Scalar, Continuity::C0, 1};
    }

    [[nodiscard]] int dimension() const noexcept override { return 3; }
    [[nodiscard]] std::size_t num_dofs() const noexcept override { return 4; }
    [[nodiscard]] std::size_t num_nodes() const noexcept override { return 4; }

    [[nodiscard]] const basis::BasisFunction& basis() const noexcept override { return *basis_; }
    [[nodiscard]] std::shared_ptr<const basis::BasisFunction> basis_ptr() const noexcept override { return basis_; }

    [[nodiscard]] std::shared_ptr<const quadrature::QuadratureRule> quadrature() const noexcept override
    {
        return quad_;
    }

private:
    std::shared_ptr<basis::BasisFunction> basis_;
    std::shared_ptr<const quadrature::QuadratureRule> quad_{};
};

class MockFunctionSpace final : public spaces::FunctionSpace {
public:
    MockFunctionSpace()
        : element_(std::make_shared<MockElement>())
    {
    }

    [[nodiscard]] spaces::SpaceType space_type() const noexcept override { return spaces::SpaceType::H1; }
    [[nodiscard]] FieldType field_type() const noexcept override { return FieldType::Scalar; }
    [[nodiscard]] Continuity continuity() const noexcept override { return Continuity::C0; }
    [[nodiscard]] int value_dimension() const noexcept override { return 1; }
    [[nodiscard]] int topological_dimension() const noexcept override { return 3; }
    [[nodiscard]] int polynomial_order() const noexcept override { return 1; }
    [[nodiscard]] ElementType element_type() const noexcept override { return ElementType::Tetra4; }

    [[nodiscard]] const elements::Element& element() const noexcept override { return *element_; }
    [[nodiscard]] std::shared_ptr<const elements::Element> element_ptr() const noexcept override { return element_; }

    const elements::Element& getElement(ElementType /*type*/, GlobalIndex /*cell_id*/) const noexcept override
    {
        return *element_;
    }

private:
    std::shared_ptr<MockElement> element_;
};

class DummyKernel final : public AssemblyKernel {
public:
    [[nodiscard]] RequiredData getRequiredData() const override { return RequiredData::None; }

    void computeCell(const AssemblyContext& /*ctx*/, KernelOutput& output) override
    {
        output.local_matrix.clear();
        output.local_vector.clear();
        output.has_matrix = false;
        output.has_vector = false;
    }
};

} // namespace

// ============================================================================
// Test Fixtures
// ============================================================================

class BlockAssemblerTest : public ::testing::Test {
protected:
    void SetUp() override
    {
        mesh_ = std::make_unique<MockMeshAccess>(2);
        space_ = std::make_unique<MockFunctionSpace>();
        dof_u_ = createTestDofMap5();
        dof_p_ = createTestDofMap4();
    }

    std::unique_ptr<MockMeshAccess> mesh_;
    std::unique_ptr<MockFunctionSpace> space_;
    dofs::DofMap dof_u_;
    dofs::DofMap dof_p_;
};

// ============================================================================
// Construction Tests
// ============================================================================

TEST_F(BlockAssemblerTest, DefaultConstruction)
{
    BlockAssembler assembler;
    EXPECT_FALSE(assembler.isConfigured());
    EXPECT_EQ(assembler.numFields(), 0);
}

TEST_F(BlockAssemblerTest, ConstructionWithOptions)
{
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

TEST_F(BlockAssemblerTest, SetMesh)
{
    BlockAssembler assembler;
    assembler.setMesh(*mesh_);
    EXPECT_FALSE(assembler.isConfigured());  // Needs fields
}

TEST_F(BlockAssemblerTest, SetOptions)
{
    BlockAssembler assembler;

    BlockAssemblerOptions options;
    options.num_threads = 8;
    options.mode = BlockAssemblyMode::Segregated;

    assembler.setOptions(options);

    EXPECT_EQ(assembler.getOptions().num_threads, 8);
    EXPECT_EQ(assembler.getOptions().mode, BlockAssemblyMode::Segregated);
}

TEST_F(BlockAssemblerTest, ConfigureFieldsComputesOffsets)
{
    BlockAssembler assembler;
    assembler.setMesh(*mesh_);

    assembler.addField(0, "u", *space_, dof_u_);
    assembler.addField(1, "p", *space_, dof_p_);

    EXPECT_TRUE(assembler.isConfigured());
    EXPECT_EQ(assembler.numFields(), 2);

    const auto [off00_r, off00_c] = assembler.getBlockOffset(0, 0);
    EXPECT_EQ(off00_r, 0);
    EXPECT_EQ(off00_c, 0);

    const auto [off01_r, off01_c] = assembler.getBlockOffset(0, 1);
    EXPECT_EQ(off01_r, 0);
    EXPECT_EQ(off01_c, dof_u_.getNumDofs());

    const auto [off10_r, off10_c] = assembler.getBlockOffset(1, 0);
    EXPECT_EQ(off10_r, dof_u_.getNumDofs());
    EXPECT_EQ(off10_c, 0);

    const auto [sz01_r, sz01_c] = assembler.getBlockSize(0, 1);
    EXPECT_EQ(sz01_r, dof_u_.getNumDofs());
    EXPECT_EQ(sz01_c, dof_p_.getNumDofs());

    EXPECT_EQ(assembler.totalSize(), dof_u_.getNumDofs() + dof_p_.getNumDofs());
}

TEST_F(BlockAssemblerTest, BlockKernelAssignmentAndQuery)
{
    BlockAssembler assembler;
    assembler.setMesh(*mesh_);
    assembler.addField(0, "u", *space_, dof_u_);
    assembler.addField(1, "p", *space_, dof_p_);

    EXPECT_FALSE(assembler.hasBlockKernel(0, 0));

    assembler.setBlockKernel(0, 0, std::make_shared<DummyKernel>());
    assembler.setBlockKernel(0, 1, std::make_shared<DummyKernel>());

    EXPECT_TRUE(assembler.hasBlockKernel(0, 0));
    EXPECT_TRUE(assembler.hasBlockKernel(0, 1));
    EXPECT_FALSE(assembler.hasBlockKernel(1, 0));

    const auto blocks = assembler.getNonZeroBlocks();
    EXPECT_EQ(blocks.size(), 2u);
    EXPECT_NE(std::find(blocks.begin(), blocks.end(), BlockIndex(0, 0)), blocks.end());
    EXPECT_NE(std::find(blocks.begin(), blocks.end(), BlockIndex(0, 1)), blocks.end());

    assembler.setBlockKernel(0, 0, nullptr);
    EXPECT_FALSE(assembler.hasBlockKernel(0, 0));
}

TEST_F(BlockAssemblerTest, BlockAssemblerAssignmentAndDefault)
{
    BlockAssembler assembler;
    assembler.setMesh(*mesh_);
    assembler.addField(0, "u", *space_, dof_u_);
    assembler.addField(1, "p", *space_, dof_p_);

    auto custom = std::make_shared<StandardAssembler>();
    assembler.setBlockAssembler(0, 1, custom);

    EXPECT_TRUE(assembler.hasBlockAssembler(0, 1));
    EXPECT_EQ(&assembler.getBlockAssembler(0, 1), custom.get());

    auto& def00 = assembler.getBlockAssembler(0, 0);
    auto& def11 = assembler.getBlockAssembler(1, 1);
    EXPECT_NE(&def00, custom.get());
    EXPECT_EQ(&def00, &def11);
}

// ============================================================================
// Block Assembly Mode Tests
// ============================================================================

TEST_F(BlockAssemblerTest, BlockAssemblyModeEnum)
{
    EXPECT_NE(BlockAssemblyMode::Monolithic, BlockAssemblyMode::Block);
    EXPECT_NE(BlockAssemblyMode::Block, BlockAssemblyMode::Segregated);
}

// ============================================================================
// Options Tests
// ============================================================================

TEST_F(BlockAssemblerTest, BlockAssemblerOptionsDefaults)
{
    BlockAssemblerOptions options;

    EXPECT_EQ(options.mode, BlockAssemblyMode::Monolithic);
    EXPECT_EQ(options.num_threads, 0);  // Auto
    EXPECT_TRUE(options.apply_constraints);
    EXPECT_FALSE(options.verbose);
}

// ============================================================================
// Block Index Tests
// ============================================================================

TEST_F(BlockAssemblerTest, BlockIndexConstruction)
{
    BlockIndex idx(1, 2);

    EXPECT_EQ(idx.row_field, 1);
    EXPECT_EQ(idx.col_field, 2);
}

TEST_F(BlockAssemblerTest, BlockIndexEquality)
{
    BlockIndex a(0, 1);
    BlockIndex b(0, 1);
    BlockIndex c(1, 0);

    EXPECT_TRUE(a == b);
    EXPECT_FALSE(a == c);
}

TEST_F(BlockAssemblerTest, BlockIndexLessThan)
{
    BlockIndex a(0, 0);
    BlockIndex b(0, 1);
    BlockIndex c(1, 0);

    EXPECT_TRUE(a < b);
    EXPECT_TRUE(a < c);
    EXPECT_TRUE(b < c);
}

TEST_F(BlockAssemblerTest, BlockIndexIsDiagonal)
{
    BlockIndex diag(1, 1);
    BlockIndex off(0, 1);

    EXPECT_TRUE(diag.isDiagonal());
    EXPECT_FALSE(off.isDiagonal());
}

// ============================================================================
// Field Configuration Tests
// ============================================================================

TEST_F(BlockAssemblerTest, FieldConfigDefaults)
{
    FieldConfig config;

    EXPECT_EQ(config.space, nullptr);
    EXPECT_EQ(config.dof_map, nullptr);
    EXPECT_EQ(config.constraints, nullptr);
    EXPECT_EQ(config.components, 1);
    EXPECT_FALSE(config.is_pressure_like);
}

TEST_F(BlockAssemblerTest, FieldConfigPopulation)
{
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

TEST_F(BlockAssemblerTest, BlockSystemConfigDefaults)
{
    BlockSystemConfig config;
    EXPECT_EQ(config.numFields(), 0);
    EXPECT_TRUE(config.fields.empty());
}

TEST_F(BlockAssemblerTest, BlockSystemConfigGetField)
{
    BlockSystemConfig config;

    // Empty config should return nullptr
    auto field = config.getField(0);
    EXPECT_EQ(field, nullptr);
}

// ============================================================================
// Statistics Tests
// ============================================================================

TEST_F(BlockAssemblerTest, BlockAssemblyStatsDefaults)
{
    BlockAssemblyStats stats;

    EXPECT_EQ(stats.num_cells, 0);
    EXPECT_DOUBLE_EQ(stats.total_seconds, 0.0);
    EXPECT_TRUE(stats.block_assembly_seconds.empty());
    EXPECT_TRUE(stats.block_nnz.empty());
}

TEST_F(BlockAssemblerTest, BlockAssemblyStatsPopulation)
{
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
// Move Semantics Tests
// ============================================================================

TEST_F(BlockAssemblerTest, MoveConstruction)
{
    BlockAssemblerOptions options;
    options.num_threads = 4;

    BlockAssembler assembler1(options);
    assembler1.setMesh(*mesh_);

    BlockAssembler assembler2(std::move(assembler1));

    EXPECT_EQ(assembler2.getOptions().num_threads, 4);
}

TEST_F(BlockAssemblerTest, MoveAssignment)
{
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

TEST_F(BlockAssemblerTest, CreateBlockAssembler)
{
    auto assembler = createBlockAssembler();
    EXPECT_NE(assembler, nullptr);
    EXPECT_FALSE(assembler->isConfigured());
}

TEST_F(BlockAssemblerTest, CreateBlockAssemblerWithOptions)
{
    BlockAssemblerOptions options;
    options.mode = BlockAssemblyMode::Segregated;

    auto assembler = createBlockAssembler(options);
    EXPECT_NE(assembler, nullptr);
    EXPECT_EQ(assembler->getOptions().mode, BlockAssemblyMode::Segregated);
}

// ============================================================================
// Last Stats Tests
// ============================================================================

TEST_F(BlockAssemblerTest, GetLastStats)
{
    BlockAssembler assembler;

    auto stats = assembler.getLastStats();
    EXPECT_EQ(stats.num_cells, 0);
    EXPECT_DOUBLE_EQ(stats.total_seconds, 0.0);
}

// ============================================================================
// Number of Fields Tests
// ============================================================================

TEST_F(BlockAssemblerTest, NumFieldsEmpty)
{
    BlockAssembler assembler;
    EXPECT_EQ(assembler.numFields(), 0);
}

// ============================================================================
// Get Config Tests
// ============================================================================

TEST_F(BlockAssemblerTest, GetConfigEmpty)
{
    BlockAssembler assembler;

    auto config = assembler.getConfig();
    EXPECT_EQ(config.numFields(), 0);
}

} // namespace testing
} // namespace assembly
} // namespace FE
} // namespace svmp


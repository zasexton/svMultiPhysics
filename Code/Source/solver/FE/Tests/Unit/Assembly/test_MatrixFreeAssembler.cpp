/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

/**
 * @file test_MatrixFreeAssembler.cpp
 * @brief Unit tests for MatrixFreeAssembler
 */

#include <gtest/gtest.h>
#include "Assembly/MatrixFreeAssembler.h"
#include "Assembly/GlobalSystemView.h"
#include "Assembly/StandardAssembler.h"
#include "Core/Types.h"
#include "Dofs/DofMap.h"
#include "Spaces/H1Space.h"

#include <vector>
#include <memory>
#include <cmath>
#include <array>

namespace svmp {
namespace FE {
namespace assembly {
namespace testing {

namespace {

// ============================================================================
// Mock Classes
// ============================================================================

/**
 * @brief Mock mesh access for matrix-free testing
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
 * @brief Simple mock matrix-free kernel for testing
 *
 * Implements local identity operator: y_local = x_local
 */
class MockMatrixFreeKernel : public IMatrixFreeKernel {
public:
    void applyLocal(
        const AssemblyContext& /*context*/,
        std::span<const Real> x_local,
        std::span<Real> y_local) override
    {
        // Simple identity operation
        for (std::size_t i = 0; i < x_local.size(); ++i) {
            y_local[i] = x_local[i];
        }
        ++apply_count_;
    }

    [[nodiscard]] RequiredData getRequiredData() const noexcept override {
        return RequiredData::BasisValues;
    }

    [[nodiscard]] bool supportsBatched() const noexcept override { return false; }

    // Test helper
    int getApplyCount() const { return apply_count_; }
    void resetApplyCount() { apply_count_ = 0; }

private:
    mutable int apply_count_{0};
};

/**
 * @brief Mock kernel with cached data support
 */
class MockCachingKernel : public IMatrixFreeKernel {
public:
    void applyLocal(
        const AssemblyContext& /*context*/,
        std::span<const Real> x_local,
        std::span<Real> y_local) override
    {
        // Scale by 2
        for (std::size_t i = 0; i < x_local.size(); ++i) {
            y_local[i] = 2.0 * x_local[i];
        }
    }

    void setupElement(
        const AssemblyContext& /*context*/,
        std::vector<Real>& element_data) override
    {
        // Cache a scaling factor
        element_data.push_back(2.0);
        ++setup_count_;
    }

    void applyLocalCached(
        const AssemblyContext& /*context*/,
        std::span<const Real> element_data,
        std::span<const Real> x_local,
        std::span<Real> y_local) override
    {
        Real scale = element_data[0];
        for (std::size_t i = 0; i < x_local.size(); ++i) {
            y_local[i] = scale * x_local[i];
        }
    }

    [[nodiscard]] std::size_t elementDataSize(
        const AssemblyContext& /*context*/) const noexcept override
    {
        return 1;  // One Real for scaling factor
    }

    int getSetupCount() const { return setup_count_; }

private:
    int setup_count_{0};
};

class ValidTetraMeshAccess final : public IMeshAccess {
public:
    explicit ValidTetraMeshAccess(GlobalIndex num_cells) : num_cells_(num_cells) {}

    [[nodiscard]] GlobalIndex numCells() const override { return num_cells_; }
    [[nodiscard]] GlobalIndex numOwnedCells() const override { return num_cells_; }
    [[nodiscard]] GlobalIndex numBoundaryFaces() const override { return 0; }
    [[nodiscard]] GlobalIndex numInteriorFaces() const override { return 0; }
    [[nodiscard]] int dimension() const override { return 3; }

    [[nodiscard]] bool isOwnedCell(GlobalIndex /*cell_id*/) const override { return true; }
    [[nodiscard]] ElementType getCellType(GlobalIndex /*cell_id*/) const override { return ElementType::Tetra4; }

    void getCellNodes(GlobalIndex cell_id, std::vector<GlobalIndex>& nodes) const override
    {
        nodes.resize(4);
        const GlobalIndex base = cell_id * 4;
        nodes[0] = base + 0;
        nodes[1] = base + 1;
        nodes[2] = base + 2;
        nodes[3] = base + 3;
    }

    [[nodiscard]] std::array<Real, 3> getNodeCoordinates(GlobalIndex node_id) const override
    {
        const GlobalIndex cell_id = node_id / 4;
        const GlobalIndex local_id = node_id % 4;
        const Real offset = static_cast<Real>(cell_id) * 2.0;
        switch (local_id) {
        case 0:
            return {offset, 0.0, 0.0};
        case 1:
            return {offset + 1.0, 0.0, 0.0};
        case 2:
            return {offset, 1.0, 0.0};
        default:
            return {offset, 0.0, 1.0};
        }
    }

    void getCellCoordinates(GlobalIndex cell_id, std::vector<std::array<Real, 3>>& coords) const override
    {
        coords.resize(4);
        const Real offset = static_cast<Real>(cell_id) * 2.0;
        coords[0] = {offset, 0.0, 0.0};
        coords[1] = {offset + 1.0, 0.0, 0.0};
        coords[2] = {offset, 1.0, 0.0};
        coords[3] = {offset, 0.0, 1.0};
    }

    [[nodiscard]] LocalIndex getLocalFaceIndex(GlobalIndex /*face_id*/, GlobalIndex /*cell_id*/) const override
    {
        return 0;
    }

    [[nodiscard]] int getBoundaryFaceMarker(GlobalIndex /*face_id*/) const override { return 0; }

    [[nodiscard]] std::pair<GlobalIndex, GlobalIndex> getInteriorFaceCells(GlobalIndex /*face_id*/) const override
    {
        return {-1, -1};
    }

    void forEachCell(std::function<void(GlobalIndex)> callback) const override
    {
        for (GlobalIndex c = 0; c < num_cells_; ++c) {
            callback(c);
        }
    }

    void forEachOwnedCell(std::function<void(GlobalIndex)> callback) const override
    {
        forEachCell(std::move(callback));
    }

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

static dofs::DofMap buildDisjointTetra4DofMap(GlobalIndex num_cells)
{
    dofs::DofMap dof_map(num_cells, /*n_dofs_total=*/num_cells * 4, /*dofs_per_cell=*/4);
    for (GlobalIndex cell = 0; cell < num_cells; ++cell) {
        const std::array<GlobalIndex, 4> dofs = {
            cell * 4 + 0,
            cell * 4 + 1,
            cell * 4 + 2,
            cell * 4 + 3,
        };
        dof_map.setCellDofs(cell, dofs);
    }
    dof_map.setNumDofs(num_cells * 4);
    dof_map.setNumLocalDofs(num_cells * 4);
    dof_map.finalize();
    return dof_map;
}

} // namespace

// ============================================================================
// Test Fixtures
// ============================================================================

class MatrixFreeAssemblerTest : public ::testing::Test {
protected:
    void SetUp() override {
        mesh_ = std::make_unique<MockMeshAccess>(50);
        kernel_ = std::make_unique<MockMatrixFreeKernel>();
    }

    std::unique_ptr<MockMeshAccess> mesh_;
    std::unique_ptr<MockMatrixFreeKernel> kernel_;
};

// ============================================================================
// Construction Tests
// ============================================================================

TEST_F(MatrixFreeAssemblerTest, DefaultConstruction) {
    MatrixFreeAssembler assembler;
    EXPECT_FALSE(assembler.isConfigured());
    EXPECT_FALSE(assembler.isSetup());
}

TEST_F(MatrixFreeAssemblerTest, ConstructionWithOptions) {
    MatrixFreeOptions options;
    options.assembly_level = AssemblyLevel::Partial;
    options.num_threads = 4;
    options.vectorize = true;
    options.batch_size = 8;

    MatrixFreeAssembler assembler(options);

    EXPECT_EQ(assembler.getOptions().assembly_level, AssemblyLevel::Partial);
    EXPECT_EQ(assembler.getOptions().num_threads, 4);
    EXPECT_TRUE(assembler.getOptions().vectorize);
    EXPECT_EQ(assembler.getOptions().batch_size, 8);
}

// ============================================================================
// Configuration Tests
// ============================================================================

TEST_F(MatrixFreeAssemblerTest, SetMesh) {
    MatrixFreeAssembler assembler;
    assembler.setMesh(*mesh_);
    EXPECT_FALSE(assembler.isConfigured());  // Needs DofMap, Space, Kernel
}

TEST_F(MatrixFreeAssemblerTest, SetKernel) {
    MatrixFreeAssembler assembler;
    assembler.setKernel(*kernel_);
    EXPECT_FALSE(assembler.isConfigured());
}

TEST_F(MatrixFreeAssemblerTest, SetOptions) {
    MatrixFreeAssembler assembler;

    MatrixFreeOptions options;
    options.num_threads = 8;
    options.cache_geometry = true;
    options.cache_basis = true;

    assembler.setOptions(options);

    EXPECT_EQ(assembler.getOptions().num_threads, 8);
    EXPECT_TRUE(assembler.getOptions().cache_geometry);
    EXPECT_TRUE(assembler.getOptions().cache_basis);
}

// ============================================================================
// Assembly Level Tests
// ============================================================================

TEST_F(MatrixFreeAssemblerTest, AssemblyLevelEnum) {
    EXPECT_NE(AssemblyLevel::None, AssemblyLevel::Element);
    EXPECT_NE(AssemblyLevel::Element, AssemblyLevel::Partial);
    EXPECT_NE(AssemblyLevel::Partial, AssemblyLevel::Full);
}

// ============================================================================
// Options Tests
// ============================================================================

TEST_F(MatrixFreeAssemblerTest, MatrixFreeOptionsDefaults) {
    MatrixFreeOptions options;

    EXPECT_EQ(options.assembly_level, AssemblyLevel::Partial);
    EXPECT_EQ(options.num_threads, 0);  // Auto
    EXPECT_TRUE(options.vectorize);
    EXPECT_EQ(options.batch_size, 4);
    EXPECT_TRUE(options.apply_constraints);
    EXPECT_TRUE(options.cache_geometry);
    EXPECT_TRUE(options.cache_basis);
    EXPECT_FALSE(options.verbose);
}

// ============================================================================
// Statistics Tests
// ============================================================================

TEST_F(MatrixFreeAssemblerTest, MatrixFreeStatsDefaults) {
    MatrixFreeStats stats;

    EXPECT_DOUBLE_EQ(stats.setup_seconds, 0.0);
    EXPECT_EQ(stats.cached_bytes, 0u);
    EXPECT_EQ(stats.num_applies, 0);
    EXPECT_DOUBLE_EQ(stats.total_apply_seconds, 0.0);
    EXPECT_DOUBLE_EQ(stats.avg_apply_seconds, 0.0);
    EXPECT_DOUBLE_EQ(stats.last_apply_seconds, 0.0);
    EXPECT_DOUBLE_EQ(stats.gflops, 0.0);
    EXPECT_DOUBLE_EQ(stats.bandwidth_gb_s, 0.0);
}

TEST_F(MatrixFreeAssemblerTest, GetStatsInitial) {
    MatrixFreeAssembler assembler;

    auto stats = assembler.getStats();
    EXPECT_EQ(stats.num_applies, 0);
}

TEST_F(MatrixFreeAssemblerTest, ResetStats) {
    MatrixFreeAssembler assembler;
    assembler.resetStats();

    auto stats = assembler.getStats();
    EXPECT_EQ(stats.num_applies, 0);
    EXPECT_DOUBLE_EQ(stats.total_apply_seconds, 0.0);
}

// ============================================================================
// Setup State Tests
// ============================================================================

TEST_F(MatrixFreeAssemblerTest, IsSetupInitiallyFalse) {
    MatrixFreeAssembler assembler;
    EXPECT_FALSE(assembler.isSetup());
}

TEST_F(MatrixFreeAssemblerTest, InvalidateSetup) {
    MatrixFreeAssembler assembler;
    assembler.invalidateSetup();
    EXPECT_FALSE(assembler.isSetup());
}

// ============================================================================
// Dimensions Tests
// ============================================================================

TEST_F(MatrixFreeAssemblerTest, DimensionsBeforeConfiguration) {
    MatrixFreeAssembler assembler;

    // Before configuration, dimensions should be 0
    EXPECT_EQ(assembler.numRows(), 0);
    EXPECT_EQ(assembler.numCols(), 0);
}

// ============================================================================
// Actual Operation Tests
// ============================================================================

TEST(MatrixFreeAssemblerOperationTest, ApplyIdentityOperatorDisjointDofs) {
    const GlobalIndex num_cells = 3;
    ValidTetraMeshAccess mesh(num_cells);
    auto dof_map = buildDisjointTetra4DofMap(num_cells);
    spaces::H1Space space(ElementType::Tetra4, /*order=*/1);

    MockMatrixFreeKernel kernel;
    MatrixFreeAssembler assembler;
    assembler.setMesh(mesh);
    assembler.setDofMap(dof_map);
    assembler.setSpace(space);
    assembler.setKernel(kernel);

    std::vector<Real> x(static_cast<std::size_t>(dof_map.getNumDofs()));
    std::vector<Real> y(x.size(), 0.0);
    for (std::size_t i = 0; i < x.size(); ++i) {
        x[i] = static_cast<Real>(i + 1);
    }

    assembler.apply(x, y);

    for (std::size_t i = 0; i < x.size(); ++i) {
        EXPECT_DOUBLE_EQ(y[i], x[i]);
    }
}

TEST(MatrixFreeAssemblerOperationTest, ApplyScalingOperatorDisjointDofs) {
    const GlobalIndex num_cells = 2;
    ValidTetraMeshAccess mesh(num_cells);
    auto dof_map = buildDisjointTetra4DofMap(num_cells);
    spaces::H1Space space(ElementType::Tetra4, /*order=*/1);

    MockCachingKernel kernel;
    MatrixFreeAssembler assembler;
    assembler.setMesh(mesh);
    assembler.setDofMap(dof_map);
    assembler.setSpace(space);
    assembler.setKernel(kernel);

    std::vector<Real> x(static_cast<std::size_t>(dof_map.getNumDofs()));
    std::vector<Real> y(x.size(), 0.0);
    for (std::size_t i = 0; i < x.size(); ++i) {
        x[i] = static_cast<Real>(1.0 + 0.25 * static_cast<Real>(i));
    }

    assembler.apply(x, y);
    for (std::size_t i = 0; i < x.size(); ++i) {
        EXPECT_NEAR(y[i], 2.0 * x[i], 1e-12);
    }
}

TEST(MatrixFreeAssemblerOperationTest, ApplyDirectMatchesApply) {
    const GlobalIndex num_cells = 2;
    ValidTetraMeshAccess mesh(num_cells);
    auto dof_map = buildDisjointTetra4DofMap(num_cells);
    spaces::H1Space space(ElementType::Tetra4, /*order=*/1);

    MockMatrixFreeKernel kernel;
    MatrixFreeAssembler assembler;
    assembler.setMesh(mesh);
    assembler.setDofMap(dof_map);
    assembler.setSpace(space);
    assembler.setKernel(kernel);

    std::vector<Real> x(static_cast<std::size_t>(dof_map.getNumDofs()), 1.0);
    std::vector<Real> y(static_cast<std::size_t>(dof_map.getNumDofs()), 0.0);
    std::vector<Real> y_direct(static_cast<std::size_t>(dof_map.getNumDofs()), 0.0);

    assembler.apply(x, y);
    assembler.applyDirect(x, y_direct);

    for (std::size_t i = 0; i < y.size(); ++i) {
        EXPECT_NEAR(y_direct[i], y[i], 1e-12);
    }
}

TEST(MatrixFreeAssemblerOperationTest, GetDiagonalIdentity) {
    const GlobalIndex num_cells = 1;
    ValidTetraMeshAccess mesh(num_cells);
    auto dof_map = buildDisjointTetra4DofMap(num_cells);
    spaces::H1Space space(ElementType::Tetra4, /*order=*/1);

    MockMatrixFreeKernel kernel;
    MatrixFreeAssembler assembler;
    assembler.setMesh(mesh);
    assembler.setDofMap(dof_map);
    assembler.setSpace(space);
    assembler.setKernel(kernel);

    std::vector<Real> diag(static_cast<std::size_t>(dof_map.getNumDofs()), 0.0);
    assembler.getDiagonal(diag);

    for (Real v : diag) {
        EXPECT_NEAR(v, 1.0, 1e-12);
    }
}

TEST(MatrixFreeAssemblerOperationTest, ApplyMatchesExplicitStiffnessAssemblySingleCell) {
    const GlobalIndex num_cells = 1;
    ValidTetraMeshAccess mesh(num_cells);
    auto dof_map = buildDisjointTetra4DofMap(num_cells);
    spaces::H1Space space(ElementType::Tetra4, /*order=*/1);

    StiffnessKernel stiffness_kernel(1.0);
    auto wrapped = wrapAsMatrixFreeKernel(stiffness_kernel);

    MatrixFreeAssembler mf;
    mf.setMesh(mesh);
    mf.setDofMap(dof_map);
    mf.setSpace(space);
    mf.setKernel(*wrapped);

    DenseMatrixView A(dof_map.getNumDofs());
    StandardAssembler explicit_asm;
    explicit_asm.setDofMap(dof_map);
    explicit_asm.assembleMatrix(mesh, space, space, stiffness_kernel, A);

    std::vector<Real> x(static_cast<std::size_t>(dof_map.getNumDofs()));
    for (std::size_t i = 0; i < x.size(); ++i) {
        x[i] = static_cast<Real>(i + 1);
    }

    std::vector<Real> y_mf(x.size(), 0.0);
    std::vector<Real> y_exp(x.size(), 0.0);
    mf.apply(x, y_mf);

    for (GlobalIndex i = 0; i < dof_map.getNumDofs(); ++i) {
        Real sum = 0.0;
        for (GlobalIndex j = 0; j < dof_map.getNumDofs(); ++j) {
            sum += A.getMatrixEntry(i, j) * x[static_cast<std::size_t>(j)];
        }
        y_exp[static_cast<std::size_t>(i)] = sum;
    }

    for (std::size_t i = 0; i < y_mf.size(); ++i) {
        EXPECT_NEAR(y_mf[i], y_exp[i], 1e-10);
    }
}

// ============================================================================
// Kernel Interface Tests
// ============================================================================

TEST_F(MatrixFreeAssemblerTest, MockKernelApplyLocal) {
    AssemblyContext context;

    std::vector<Real> x_local = {1.0, 2.0, 3.0, 4.0};
    std::vector<Real> y_local(4, 0.0);

    kernel_->applyLocal(context, x_local, y_local);

    // Identity kernel
    for (std::size_t i = 0; i < 4; ++i) {
        EXPECT_DOUBLE_EQ(y_local[i], x_local[i]);
    }
}

TEST_F(MatrixFreeAssemblerTest, MockKernelApplyCount) {
    kernel_->resetApplyCount();
    EXPECT_EQ(kernel_->getApplyCount(), 0);

    AssemblyContext context;
    std::vector<Real> x(4, 1.0);
    std::vector<Real> y(4, 0.0);

    kernel_->applyLocal(context, x, y);
    EXPECT_EQ(kernel_->getApplyCount(), 1);

    kernel_->applyLocal(context, x, y);
    EXPECT_EQ(kernel_->getApplyCount(), 2);
}

TEST_F(MatrixFreeAssemblerTest, MockKernelRequiredData) {
    auto required = kernel_->getRequiredData();
    EXPECT_TRUE((required & RequiredData::BasisValues) != RequiredData::None);
}

TEST_F(MatrixFreeAssemblerTest, MockKernelSupportsBatched) {
    EXPECT_FALSE(kernel_->supportsBatched());
}

// ============================================================================
// Caching Kernel Tests
// ============================================================================

TEST_F(MatrixFreeAssemblerTest, CachingKernelSetupElement) {
    MockCachingKernel caching_kernel;
    AssemblyContext context;

    std::vector<Real> element_data;
    caching_kernel.setupElement(context, element_data);

    EXPECT_EQ(element_data.size(), 1u);
    EXPECT_DOUBLE_EQ(element_data[0], 2.0);
    EXPECT_EQ(caching_kernel.getSetupCount(), 1);
}

TEST_F(MatrixFreeAssemblerTest, CachingKernelApplyLocalCached) {
    MockCachingKernel caching_kernel;
    AssemblyContext context;

    std::vector<Real> element_data = {3.0};  // Scale factor
    std::vector<Real> x = {1.0, 2.0, 3.0};
    std::vector<Real> y(3, 0.0);

    caching_kernel.applyLocalCached(context, element_data, x, y);

    EXPECT_DOUBLE_EQ(y[0], 3.0);
    EXPECT_DOUBLE_EQ(y[1], 6.0);
    EXPECT_DOUBLE_EQ(y[2], 9.0);
}

TEST_F(MatrixFreeAssemblerTest, CachingKernelElementDataSize) {
    MockCachingKernel caching_kernel;
    AssemblyContext context;

    auto size = caching_kernel.elementDataSize(context);
    EXPECT_EQ(size, 1u);
}

// ============================================================================
// CachedElementData Tests
// ============================================================================

TEST_F(MatrixFreeAssemblerTest, CachedElementDataDefaults) {
    MatrixFreeElementData data;

    EXPECT_EQ(data.cell_id, -1);
    EXPECT_EQ(data.cell_type, ElementType::Unknown);
    EXPECT_TRUE(data.dofs.empty());
    EXPECT_TRUE(data.jacobians.empty());
    EXPECT_TRUE(data.det_jacobians.empty());
    EXPECT_TRUE(data.basis_values.empty());
    EXPECT_TRUE(data.kernel_data.empty());
}

TEST_F(MatrixFreeAssemblerTest, CachedElementDataPopulation) {
    MatrixFreeElementData data;

    data.cell_id = 42;
    data.cell_type = ElementType::Hex8;
    data.dofs = {0, 1, 2, 3, 4, 5, 6, 7};
    data.det_jacobians = {0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125};

    EXPECT_EQ(data.cell_id, 42);
    EXPECT_EQ(data.cell_type, ElementType::Hex8);
    EXPECT_EQ(data.dofs.size(), 8u);
    EXPECT_EQ(data.det_jacobians.size(), 8u);
}

// ============================================================================
// Move Semantics Tests
// ============================================================================

TEST_F(MatrixFreeAssemblerTest, MoveConstruction) {
    MatrixFreeOptions options;
    options.num_threads = 4;
    options.batch_size = 16;

    MatrixFreeAssembler assembler1(options);
    assembler1.setMesh(*mesh_);

    MatrixFreeAssembler assembler2(std::move(assembler1));

    EXPECT_EQ(assembler2.getOptions().num_threads, 4);
    EXPECT_EQ(assembler2.getOptions().batch_size, 16);
}

TEST_F(MatrixFreeAssemblerTest, MoveAssignment) {
    MatrixFreeOptions options;
    options.num_threads = 8;

    MatrixFreeAssembler assembler1(options);

    MatrixFreeAssembler assembler2;
    assembler2 = std::move(assembler1);

    EXPECT_EQ(assembler2.getOptions().num_threads, 8);
}

// ============================================================================
// Factory Function Tests
// ============================================================================

TEST_F(MatrixFreeAssemblerTest, CreateMatrixFreeAssembler) {
    auto assembler = createMatrixFreeAssembler();
    EXPECT_NE(assembler, nullptr);
    EXPECT_FALSE(assembler->isConfigured());
}

TEST_F(MatrixFreeAssemblerTest, CreateMatrixFreeAssemblerWithOptions) {
    MatrixFreeOptions options;
    options.assembly_level = AssemblyLevel::Element;

    auto assembler = createMatrixFreeAssembler(options);
    EXPECT_NE(assembler, nullptr);
    EXPECT_EQ(assembler->getOptions().assembly_level, AssemblyLevel::Element);
}

// ============================================================================
// Current Solution Tests
// ============================================================================

TEST_F(MatrixFreeAssemblerTest, SetCurrentSolution) {
    MatrixFreeAssembler assembler;

    std::vector<Real> solution(100, 1.0);
    assembler.setCurrentSolution(solution);

    // Should not throw - solution is stored for residual evaluation
    EXPECT_TRUE(true);
}

} // namespace testing
} // namespace assembly
} // namespace FE
} // namespace svmp

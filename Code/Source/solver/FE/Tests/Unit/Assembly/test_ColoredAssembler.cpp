/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

/**
 * @file test_ColoredAssembler.cpp
 * @brief Unit tests for ColoredAssembler
 */

#include <gtest/gtest.h>
#include "Assembly/ColoredAssembler.h"
#include "Assembly/BlockAssembler.h"
#include "Assembly/GlobalSystemView.h"
#include "Assembly/StandardAssembler.h"
#include "Constraints/AffineConstraints.h"
#include "Dofs/DofMap.h"
#include "Spaces/H1Space.h"
#include "Spaces/L2Space.h"

#include <atomic>
#include <thread>
#include "Core/Types.h"

#include <vector>
#include <memory>
#include <cmath>
#include <algorithm>
#include <numeric>

namespace svmp {
namespace FE {
namespace assembly {
namespace testing {

namespace {

// ============================================================================
// Mock Classes
// ============================================================================

/**
 * @brief Mock mesh access for colored assembly testing
 */
class MockMeshAccess : public IMeshAccess {
public:
    MockMeshAccess(GlobalIndex num_cells = 100)
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

class DisjointTets final : public MockMeshAccess {
public:
    using MockMeshAccess::MockMeshAccess;
    std::array<Real, 3> getNodeCoordinates(GlobalIndex d) const override {
        const auto i = d % 4;
        return {static_cast<Real>(2 * (d/4) + (i == 1)), Real(i == 2), Real(i == 3)};
    }
};

class ObservedPoisson final : public AssemblyKernel {
public:
    [[nodiscard]] RequiredData getRequiredData() const override { return poisson_.getRequiredData(); }
    void computeCell(const AssemblyContext& ctx, KernelOutput& output) override {
        if (std::this_thread::get_id() != caller_) ++worker_calls;
        poisson_.computeCell(ctx, output);
    }
    std::atomic<unsigned> worker_calls{0};
private:
    PoissonKernel poisson_{2.};
    const std::thread::id caller_{std::this_thread::get_id()};
};

} // namespace

// ============================================================================
// Test Fixtures
// ============================================================================

class ColoredAssemblerTest : public ::testing::Test {
protected:
    void SetUp() override {
        mesh_ = std::make_unique<MockMeshAccess>(100);
    }

    std::unique_ptr<MockMeshAccess> mesh_;
};

// ============================================================================
// Construction Tests
// ============================================================================

TEST_F(ColoredAssemblerTest, PoissonAssemblyMatchesStandardWithSharedAffineMasters)
{
    constexpr int cells = 16;
    constexpr GlobalIndex shared = 4 * cells;
    DisjointTets mesh(cells);
    dofs::DofMap map(cells, shared + 1, 4);
    constraints::AffineConstraints constraints;
    for (int c = 0; c < cells; ++c) {
        const GlobalIndex d = 4*c;
        map.setCellDofs(c, std::array<GlobalIndex,4>{d, d+1, d+2, d+3});
        constraints.addLine(d);
        constraints.addEntry(d, shared, 1.);
        constraints.setInhomogeneity(d, 1.);
    }
    constraints.addDirichlet(1, 0.25);
    constraints.close();
    map.finalize();
    spaces::H1Space space(ElementType::Tetra4, 1);
    ObservedPoisson kernel;
    for (int threads : {1, 4}) {
        AssemblyOptions options;
        options.num_threads = threads;
        options.use_coloring = true;
        ColoredAssembler colored;
        StandardAssembler standard;
        for (Assembler* assembler : std::array<Assembler*,2>{&colored, &standard}) {
            assembler->setOptions(options);
            assembler->setDofMap(map);
            assembler->setConstraints(&constraints);
        }
        for (int operation = 0; operation < 3; ++operation) {
            DenseSystemView expected(shared+1), actual(shared+1);
            auto assemble = [&](Assembler& assembler, DenseSystemView& out) {
                if (operation == 0) return assembler.assembleBoth(mesh, space, space, kernel, out, out);
                if (operation == 1) return assembler.assembleMatrix(mesh, space, space, kernel, out);
                return assembler.assembleVector(mesh, space, kernel, out);
            };
            ASSERT_TRUE(assemble(standard, expected));
            ASSERT_TRUE(assemble(colored, actual));
            if (operation != 2) EXPECT_NEAR(expected.getMatrixEntry(shared, shared), cells/2., 1.e-12);
            if (operation == 0) EXPECT_NEAR(expected.getVectorEntry(shared), -5.*cells/12. + 1./24., 1.e-12);
            if (operation == 2) EXPECT_NEAR(expected.getVectorEntry(shared), cells/12., 1.e-12);
            Real error = 0.;
            for (GlobalIndex i = 0; i <= shared; ++i) {
                error = std::max(error, std::abs(expected.getVectorEntry(i) - actual.getVectorEntry(i)));
                for (GlobalIndex j = 0; j <= shared; ++j) {
                    error = std::max(error, std::abs(expected.getMatrixEntry(i,j) - actual.getMatrixEntry(i,j)));
                }
            }
            EXPECT_NEAR(error, 0., 1.e-12) << "threads=" << threads << " operation=" << operation;
        }
        colored.setConstraints(nullptr);
        DenseSystemView unconstrained(shared+1);
        ASSERT_TRUE(colored.assembleBoth(mesh, space, space, kernel, unconstrained, unconstrained));
        EXPECT_NEAR(unconstrained.getMatrixEntry(0,0), .5, 1.e-13);
        EXPECT_NEAR(unconstrained.getVectorEntry(0), 1./12., 1.e-13);
        EXPECT_EQ(unconstrained.getMatrixEntry(shared,shared), 0.);
        EXPECT_EQ(kernel.worker_calls.load() > 0, threads > 1);
    }
}

TEST_F(ColoredAssemblerTest, RectangularConstraintAssemblyUsesBothFieldOffsets)
{
    DisjointTets mesh(2);
    spaces::H1Space test_space(ElementType::Tetra4, 1);
    spaces::L2Space trial_space(ElementType::Tetra4, 0);
    dofs::DofMap row_map(2, 8, 4), col_map(2, 2, 1);
    row_map.setCellDofs(0, std::array<GlobalIndex,4>{0, 1, 2, 3});
    row_map.setCellDofs(1, std::array<GlobalIndex,4>{4, 5, 6, 7});
    col_map.setCellDofs(0, std::array<GlobalIndex,1>{0});
    col_map.setCellDofs(1, std::array<GlobalIndex,1>{1});
    row_map.finalize();
    col_map.finalize();
    constraints::AffineConstraints ac;
    ac.addLine(3);
    ac.addEntry(3, 0, .5);
    ac.addLine(15);
    ac.addEntry(15, 19, -2.);
    ac.close();
    MassKernel kernel;
    StandardAssembler standard;
    ColoredAssembler colored;
    DenseSystemView reference(11, 20), result(11, 20);
    auto assemble = [&](Assembler& assembler, DenseSystemView& out) {
        AssemblyOptions options;
        options.num_threads = 2;
        assembler.setOptions(options);
        assembler.setRowDofMap(row_map, 3);
        assembler.setColDofMap(col_map, 15);
        assembler.setConstraints(&ac);
        return assembler.assembleMatrix(mesh, test_space, trial_space, kernel, out);
    };
    ASSERT_TRUE(assemble(standard, reference));
    ASSERT_TRUE(assemble(colored, result));
    EXPECT_NEAR(reference.getMatrixEntry(0, 19), -1./24., 1.e-12);
    for (GlobalIndex i = 0; i < 11; ++i) {
        for (GlobalIndex j = 0; j < 20; ++j) {
            EXPECT_NEAR(result.getMatrixEntry(i,j), reference.getMatrixEntry(i,j), 1.e-12);
        }
    }
}

TEST_F(ColoredAssemblerTest, DefaultConstruction) {
    ColoredAssembler assembler;
    EXPECT_FALSE(assembler.isConfigured());
    EXPECT_FALSE(assembler.hasColoring());
}

TEST_F(ColoredAssemblerTest, ConstructionWithColoringOptions) {
    ColoringOptions options;
    options.algorithm = ColoringAlgorithm::Greedy;
    options.max_colors = 64;
    options.balance_colors = false;
    options.verbose = true;

    ColoredAssembler assembler(options);

    EXPECT_EQ(assembler.name(), "Colored(StandardAssembler)");
    EXPECT_FALSE(assembler.hasColoring());
}

// ============================================================================
// Configuration Tests
// ============================================================================

TEST_F(ColoredAssemblerTest, SetOptionsInvalidatesColoringAndForwardsToBase) {
    ColoredAssembler assembler;

    // Seed a coloring state
    std::vector<int> colors(100, 0);
    assembler.setColoring(colors, 1);
    EXPECT_TRUE(assembler.hasColoring());

    AssemblyOptions options{};
    options.num_threads = 8;
    options.use_coloring = true;
    options.coloring.algorithm = ColoringAlgorithm::DSatur;
    options.coloring.max_colors = 128; // trigger invalidation
    options.coloring.balance_colors = true;

    assembler.setOptions(options);

    EXPECT_EQ(assembler.getOptions().num_threads, 8);
    EXPECT_FALSE(assembler.hasColoring());
}

// ============================================================================
// Coloring Options Tests
// ============================================================================

TEST_F(ColoredAssemblerTest, ColoringAlgorithmEnum) {
    EXPECT_NE(ColoringAlgorithm::Greedy, ColoringAlgorithm::DSatur);
    EXPECT_NE(ColoringAlgorithm::DSatur, ColoringAlgorithm::LargestFirst);
    EXPECT_NE(ColoringAlgorithm::LargestFirst, ColoringAlgorithm::SmallestLast);
}

TEST_F(ColoredAssemblerTest, ColoringOptionsDefaults) {
    ColoringOptions options;

    EXPECT_EQ(options.algorithm, ColoringAlgorithm::DSatur);
    EXPECT_EQ(options.max_colors, 256);
    EXPECT_TRUE(options.balance_colors);
    EXPECT_FALSE(options.reorder_elements);
    EXPECT_FALSE(options.verbose);
}

// ============================================================================
// Coloring Statistics Tests
// ============================================================================

TEST_F(ColoredAssemblerTest, ColoringStatsDefaults) {
    ColoringStats stats;

    EXPECT_EQ(stats.num_colors, 0);
    EXPECT_EQ(stats.num_elements, 0);
    EXPECT_EQ(stats.min_color_size, 0);
    EXPECT_EQ(stats.max_color_size, 0);
    EXPECT_DOUBLE_EQ(stats.avg_color_size, 0.0);
    EXPECT_DOUBLE_EQ(stats.coloring_seconds, 0.0);
    EXPECT_TRUE(stats.color_sizes.empty());
}

// ============================================================================
// Element Graph Tests
// ============================================================================

TEST_F(ColoredAssemblerTest, ElementGraphConstruction) {
    ElementGraph graph;
    EXPECT_EQ(graph.numElements(), 0);
    EXPECT_EQ(graph.numEdges(), 0);
}

TEST_F(ColoredAssemblerTest, ElementGraphWithSize) {
    ElementGraph graph(10);
    EXPECT_EQ(graph.numElements(), 10);
}

TEST_F(ColoredAssemblerTest, ElementGraphClear) {
    ElementGraph graph(10);
    graph.clear();
    EXPECT_EQ(graph.numElements(), 0);
    EXPECT_EQ(graph.numEdges(), 0);
}

// ============================================================================
// Coloring Management Tests
// ============================================================================

TEST_F(ColoredAssemblerTest, NoColoringByDefault) {
    ColoredAssembler assembler;
    EXPECT_FALSE(assembler.hasColoring());
    EXPECT_EQ(assembler.numColors(), 0);
}

TEST_F(ColoredAssemblerTest, SetColoring) {
    ColoredAssembler assembler;

    std::vector<int> colors(100);
    for (std::size_t i = 0; i < colors.size(); ++i) {
        colors[i] = static_cast<int>(i % 5);  // 5 colors
    }

    assembler.setColoring(colors, 5);

    EXPECT_TRUE(assembler.hasColoring());
    EXPECT_EQ(assembler.numColors(), 5);
}

TEST_F(ColoredAssemblerTest, SetColoringReturnsColors) {
    ColoredAssembler assembler;

    std::vector<int> colors(100);
    for (std::size_t i = 0; i < colors.size(); ++i) {
        colors[i] = static_cast<int>(i % 3);  // 3 colors
    }

    assembler.setColoring(colors, 3);

    auto returned_colors = assembler.getColors();
    EXPECT_EQ(returned_colors.size(), 100u);

    // Verify colors match
    for (std::size_t i = 0; i < 100; ++i) {
        EXPECT_EQ(returned_colors[i], colors[i]);
    }
}

// ============================================================================
// Block Index Tests
// ============================================================================

TEST_F(ColoredAssemblerTest, BlockIndexDefault) {
    BlockIndex idx;
    // Default-initialized, values unspecified
    EXPECT_TRUE(true);
}

TEST_F(ColoredAssemblerTest, BlockIndexConstruction) {
    BlockIndex idx(2, 3);
    EXPECT_EQ(idx.row_field, 2);
    EXPECT_EQ(idx.col_field, 3);
}

TEST_F(ColoredAssemblerTest, BlockIndexEquality) {
    BlockIndex idx1(1, 2);
    BlockIndex idx2(1, 2);
    BlockIndex idx3(1, 3);

    EXPECT_TRUE(idx1 == idx2);
    EXPECT_FALSE(idx1 == idx3);
}

TEST_F(ColoredAssemblerTest, BlockIndexOrdering) {
    BlockIndex idx1(0, 0);
    BlockIndex idx2(0, 1);
    BlockIndex idx3(1, 0);

    EXPECT_TRUE(idx1 < idx2);
    EXPECT_TRUE(idx1 < idx3);
    EXPECT_TRUE(idx2 < idx3);
}

TEST_F(ColoredAssemblerTest, BlockIndexIsDiagonal) {
    BlockIndex diag(2, 2);
    BlockIndex off_diag(2, 3);

    EXPECT_TRUE(diag.isDiagonal());
    EXPECT_FALSE(off_diag.isDiagonal());
}

// ============================================================================
// Move Semantics Tests
// ============================================================================

TEST_F(ColoredAssemblerTest, MoveConstruction) {
    ColoringOptions coloring;
    coloring.max_colors = 64;

    ColoredAssembler assembler1(coloring);

    AssemblyOptions options{};
    options.num_threads = 4;
    options.use_coloring = true;
    options.coloring = coloring;
    assembler1.setOptions(options);

    std::vector<int> colors(50, 0);
    for (std::size_t i = 0; i < colors.size(); ++i) {
        colors[i] = static_cast<int>(i % 3);
    }
    assembler1.setColoring(colors, 3);

    ColoredAssembler assembler2(std::move(assembler1));

    EXPECT_TRUE(assembler2.hasColoring());
    EXPECT_EQ(assembler2.numColors(), 3);
    EXPECT_EQ(assembler2.getOptions().num_threads, 4);
}

TEST_F(ColoredAssemblerTest, MoveAssignment) {
    ColoredAssembler assembler1;

    AssemblyOptions options{};
    options.num_threads = 4;
    options.use_coloring = true;
    assembler1.setOptions(options);

    std::vector<int> colors(50, 0);
    assembler1.setColoring(colors, 2);

    ColoredAssembler assembler2;
    assembler2 = std::move(assembler1);

    EXPECT_TRUE(assembler2.hasColoring());
    EXPECT_EQ(assembler2.numColors(), 2);
}

// ============================================================================
// Coloring Utility Function Tests
// ============================================================================

TEST_F(ColoredAssemblerTest, VerifyColoringValidEmptyGraph) {
    ElementGraph graph;
    std::vector<int> colors;

    // Empty graph/colors should be valid
    EXPECT_TRUE(verifyColoring(graph, colors));
}

// ============================================================================
// Coloring Statistics After SetColoring
// ============================================================================

TEST_F(ColoredAssemblerTest, GetColoringStatsAfterSetColoring) {
    ColoredAssembler assembler;

    std::vector<int> colors(100);
    // Create uneven distribution: 50 elements color 0, 30 color 1, 20 color 2
    for (std::size_t i = 0; i < 50; ++i) colors[i] = 0;
    for (std::size_t i = 50; i < 80; ++i) colors[i] = 1;
    for (std::size_t i = 80; i < 100; ++i) colors[i] = 2;

    assembler.setColoring(colors, 3);

    EXPECT_TRUE(assembler.hasColoring());
    EXPECT_EQ(assembler.numColors(), 3);
}

} // namespace testing
} // namespace assembly
} // namespace FE
} // namespace svmp

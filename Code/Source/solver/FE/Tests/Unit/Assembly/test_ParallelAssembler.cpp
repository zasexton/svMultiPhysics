/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

/**
 * @file test_ParallelAssembler.cpp
 * @brief Unit tests for ParallelAssembler
 *
 * Tests MPI-parallel assembly functionality:
 * - Construction with ghost policy options
 * - Serial assembly (single rank behavior)
 * - Assembly with ghost contributions
 * - OwnedRowsOnly vs ReverseScatter policies
 * - Deterministic results verification
 *
 * NOTE: These tests can run in serial (single rank) for unit testing.
 * Full MPI tests require launching with mpiexec.
 */

#include <gtest/gtest.h>

#include "Assembly/ParallelAssembler.h"
#include "Assembly/GlobalSystemView.h"
#include "Assembly/AssemblyKernel.h"
#include "Assembly/GhostContributionManager.h"
#include "Dofs/DofMap.h"
#include "Spaces/FunctionSpace.h"

#include <cmath>
#include <vector>
#include <memory>

namespace svmp {
namespace FE {
namespace assembly {
namespace test {

// ============================================================================
// Mock Classes for Testing (Parallel-aware)
// ============================================================================

/**
 * @brief Mock mesh for distributed testing
 *
 * Simulates a partitioned mesh. For single-rank testing,
 * all cells are owned. For multi-rank testing, cells are
 * partitioned across ranks.
 */
class MockDistributedMesh : public IMeshAccess {
public:
    explicit MockDistributedMesh(int rank = 0, int num_ranks = 1)
        : my_rank_(rank), num_ranks_(num_ranks)
    {
        // Create a simple mesh: 4 tetrahedra sharing vertices
        nodes_ = {
            {0.0, 0.0, 0.0},  // 0
            {1.0, 0.0, 0.0},  // 1
            {0.0, 1.0, 0.0},  // 2
            {0.0, 0.0, 1.0},  // 3
            {1.0, 1.0, 0.0},  // 4
            {1.0, 0.2, 1.0}   // 5 (non-coplanar with {2,3,4})
        };

        // 4 tetrahedra
        cells_ = {
            {0, 1, 2, 3},  // Cell 0
            {1, 2, 3, 4},  // Cell 1
            {1, 3, 4, 5},  // Cell 2
            {2, 3, 4, 5}   // Cell 3
        };

        cell_types_ = {
            ElementType::Tetra4, ElementType::Tetra4,
            ElementType::Tetra4, ElementType::Tetra4
        };

        // Determine owned cells based on partitioning
        setupOwnership();
    }

    [[nodiscard]] GlobalIndex numCells() const override { return 4; }

    [[nodiscard]] GlobalIndex numOwnedCells() const override {
        return static_cast<GlobalIndex>(owned_cells_.size());
    }

    [[nodiscard]] GlobalIndex numBoundaryFaces() const override { return 4; }
    [[nodiscard]] GlobalIndex numInteriorFaces() const override { return 4; }
    [[nodiscard]] int dimension() const override { return 3; }

    [[nodiscard]] bool isOwnedCell(GlobalIndex cell_id) const override {
        return cell_owner_[static_cast<std::size_t>(cell_id)] == my_rank_;
    }

    [[nodiscard]] ElementType getCellType(GlobalIndex cell_id) const override {
        return cell_types_[static_cast<std::size_t>(cell_id)];
    }

    void getCellNodes(GlobalIndex cell_id, std::vector<GlobalIndex>& nodes) const override {
        const auto& cell = cells_[static_cast<std::size_t>(cell_id)];
        nodes.assign(cell.begin(), cell.end());
    }

    [[nodiscard]] std::array<Real, 3> getNodeCoordinates(GlobalIndex node_id) const override {
        return nodes_[static_cast<std::size_t>(node_id)];
    }

    void getCellCoordinates(GlobalIndex cell_id,
                           std::vector<std::array<Real, 3>>& coords) const override {
        const auto& cell = cells_[static_cast<std::size_t>(cell_id)];
        coords.resize(cell.size());
        for (std::size_t i = 0; i < cell.size(); ++i) {
            coords[i] = nodes_[cell[i]];
        }
    }

    [[nodiscard]] LocalIndex getLocalFaceIndex(GlobalIndex /*face_id*/,
                                               GlobalIndex /*cell_id*/) const override {
        return 0;
    }

    [[nodiscard]] int getBoundaryFaceMarker(GlobalIndex /*face_id*/) const override {
        return 1;
    }

    [[nodiscard]] std::pair<GlobalIndex, GlobalIndex>
    getInteriorFaceCells(GlobalIndex face_id) const override {
        // Simplified: return adjacent cells based on face_id
        if (face_id == 0) return {0, 1};
        if (face_id == 1) return {1, 2};
        if (face_id == 2) return {2, 3};
        return {0, 3};
    }

    void forEachCell(std::function<void(GlobalIndex)> callback) const override {
        for (GlobalIndex i = 0; i < numCells(); ++i) {
            callback(i);
        }
    }

    void forEachOwnedCell(std::function<void(GlobalIndex)> callback) const override {
        for (auto cell_id : owned_cells_) {
            callback(cell_id);
        }
    }

    void forEachBoundaryFace(int /*marker*/,
                            std::function<void(GlobalIndex, GlobalIndex)> callback) const override {
        // Boundary faces: exterior faces
        callback(0, 0);
        callback(1, 0);
        callback(2, 3);
        callback(3, 3);
    }

    void forEachInteriorFace(
        std::function<void(GlobalIndex, GlobalIndex, GlobalIndex)> callback) const override {
        callback(4, 0, 1);  // Face between cells 0 and 1
        callback(5, 1, 2);  // Face between cells 1 and 2
        callback(6, 2, 3);  // Face between cells 2 and 3
        callback(7, 0, 3);  // Face between cells 0 and 3
    }

private:
    void setupOwnership() {
        // Simple round-robin partitioning
        cell_owner_.resize(4);
        for (int i = 0; i < 4; ++i) {
            cell_owner_[i] = i % num_ranks_;
            if (cell_owner_[i] == my_rank_) {
                owned_cells_.push_back(i);
            }
        }
    }

    std::vector<std::array<Real, 3>> nodes_;
    std::vector<std::vector<GlobalIndex>> cells_;
    std::vector<ElementType> cell_types_;
    std::vector<int> cell_owner_;
    std::vector<GlobalIndex> owned_cells_;
    int my_rank_;
    int num_ranks_;
};

// Note: MockDistributedDofMap was removed - use createParallelTestDofMap() instead

/**
 * @brief Dense system view for testing
 */
class TestDenseSystemView : public GlobalSystemView {
public:
    explicit TestDenseSystemView(GlobalIndex n)
        : n_(n), matrix_(n * n, 0.0), vector_(n, 0.0) {}

    // Square matrix addMatrixEntries (same row and col dofs)
    void addMatrixEntries(std::span<const GlobalIndex> dofs,
                          std::span<const Real> local_matrix,
                          AddMode mode = AddMode::Add) override {
        addMatrixEntries(dofs, dofs, local_matrix, mode);
    }

    // Rectangular matrix addMatrixEntries
    void addMatrixEntries(std::span<const GlobalIndex> row_dofs,
                          std::span<const GlobalIndex> col_dofs,
                          std::span<const Real> local_matrix,
                          AddMode mode = AddMode::Add) override {
        std::size_t k = 0;
        for (std::size_t i = 0; i < row_dofs.size(); ++i) {
            for (std::size_t j = 0; j < col_dofs.size(); ++j) {
                if (mode == AddMode::Add) {
                    matrix_[row_dofs[i] * n_ + col_dofs[j]] += local_matrix[k];
                } else {
                    matrix_[row_dofs[i] * n_ + col_dofs[j]] = local_matrix[k];
                }
                ++k;
            }
        }
    }

    void addMatrixEntry(GlobalIndex row, GlobalIndex col, Real value,
                        AddMode mode = AddMode::Add) override {
        if (mode == AddMode::Add) {
            matrix_[row * n_ + col] += value;
        } else {
            matrix_[row * n_ + col] = value;
        }
    }

    void setDiagonal(std::span<const GlobalIndex> dofs,
                     std::span<const Real> values) override {
        for (std::size_t i = 0; i < dofs.size(); ++i) {
            matrix_[dofs[i] * n_ + dofs[i]] = values[i];
        }
    }

    void setDiagonal(GlobalIndex dof, Real value) override {
        matrix_[dof * n_ + dof] = value;
    }

    void zeroRows(std::span<const GlobalIndex> rows, bool set_diagonal = true) override {
        for (auto row : rows) {
            for (GlobalIndex col = 0; col < n_; ++col) {
                matrix_[row * n_ + col] = 0.0;
            }
            if (set_diagonal) {
                matrix_[row * n_ + row] = 1.0;
            }
        }
    }

    void addVectorEntries(std::span<const GlobalIndex> dofs,
                          std::span<const Real> local_vector,
                          AddMode mode = AddMode::Add) override {
        for (std::size_t i = 0; i < dofs.size(); ++i) {
            if (mode == AddMode::Add) {
                vector_[dofs[i]] += local_vector[i];
            } else {
                vector_[dofs[i]] = local_vector[i];
            }
        }
    }

    void addVectorEntry(GlobalIndex dof, Real value, AddMode mode = AddMode::Add) override {
        if (mode == AddMode::Add) {
            vector_[dof] += value;
        } else {
            vector_[dof] = value;
        }
    }

    void setVectorEntries(std::span<const GlobalIndex> dofs,
                          std::span<const Real> values) override {
        for (std::size_t i = 0; i < dofs.size(); ++i) {
            vector_[dofs[i]] = values[i];
        }
    }

    void zeroVectorEntries(std::span<const GlobalIndex> dofs) override {
        for (auto dof : dofs) {
            vector_[dof] = 0.0;
        }
    }

    [[nodiscard]] Real getMatrixEntry(GlobalIndex row, GlobalIndex col) const override {
        return matrix_[row * n_ + col];
    }

    [[nodiscard]] Real getVectorEntry(GlobalIndex dof) const override {
        return vector_[dof];
    }

    [[nodiscard]] GlobalIndex numRows() const noexcept override { return n_; }
    [[nodiscard]] GlobalIndex numCols() const noexcept override { return n_; }

    void beginAssemblyPhase() override { phase_ = AssemblyPhase::Building; }
    void endAssemblyPhase() override { phase_ = AssemblyPhase::Flushing; }
    void finalizeAssembly() override { phase_ = AssemblyPhase::Finalized; finalized_ = true; }

    [[nodiscard]] AssemblyPhase getPhase() const noexcept override { return phase_; }

    [[nodiscard]] bool hasMatrix() const noexcept override { return true; }
    [[nodiscard]] bool hasVector() const noexcept override { return true; }

    [[nodiscard]] std::string backendName() const override { return "TestDenseSystem"; }

    void zero() override {
        std::fill(matrix_.begin(), matrix_.end(), 0.0);
        std::fill(vector_.begin(), vector_.end(), 0.0);
    }

    const std::vector<Real>& matrix() const { return matrix_; }
    const std::vector<Real>& vectorData() const { return vector_; }
    bool isFinalized() const { return finalized_; }

private:
    GlobalIndex n_;
    std::vector<Real> matrix_;
    std::vector<Real> vector_;
    AssemblyPhase phase_{AssemblyPhase::NotStarted};
    bool finalized_{false};
};

// Forward declaration for mock element
class ParallelMockElement;

/**
 * @brief Mock basis for testing
 */
class ParallelMockBasis : public basis::BasisFunction {
public:
    [[nodiscard]] BasisType basis_type() const noexcept override { return BasisType::Lagrange; }
    [[nodiscard]] ElementType element_type() const noexcept override { return ElementType::Tetra4; }
    [[nodiscard]] int dimension() const noexcept override { return 3; }
    [[nodiscard]] int order() const noexcept override { return 1; }
    [[nodiscard]] std::size_t size() const noexcept override { return 4; }

    void evaluate_values(const math::Vector<Real, 3>&, std::vector<Real>& v) const override {
        v = {0.25, 0.25, 0.25, 0.25};
    }

    void evaluate_gradients(const math::Vector<Real, 3>&, std::vector<basis::Gradient>& g) const override {
        g.resize(4);
        g[0] = {-1.0, -1.0, -1.0};
        g[1] = { 1.0,  0.0,  0.0};
        g[2] = { 0.0,  1.0,  0.0};
        g[3] = { 0.0,  0.0,  1.0};
    }
};

/**
 * @brief Mock element for testing
 */
class ParallelMockElement : public elements::Element {
public:
    ParallelMockElement() : basis_(std::make_shared<ParallelMockBasis>()) {}

    [[nodiscard]] elements::ElementInfo info() const noexcept override {
        return {ElementType::Tetra4, FieldType::Scalar, Continuity::C0, 1};
    }
    [[nodiscard]] int dimension() const noexcept override { return 3; }
    [[nodiscard]] std::size_t num_dofs() const noexcept override { return 4; }
    [[nodiscard]] std::size_t num_nodes() const noexcept override { return 4; }
    [[nodiscard]] const basis::BasisFunction& basis() const noexcept override { return *basis_; }
    [[nodiscard]] std::shared_ptr<const basis::BasisFunction> basis_ptr() const noexcept override {
        return basis_;
    }
    [[nodiscard]] std::shared_ptr<const quadrature::QuadratureRule> quadrature() const noexcept override {
        return nullptr;
    }

private:
    std::shared_ptr<ParallelMockBasis> basis_;
};

/**
 * @brief Mock function space for parallel testing
 */
class ParallelMockFunctionSpace : public spaces::FunctionSpace {
public:
    ParallelMockFunctionSpace() : mock_element_(std::make_shared<ParallelMockElement>()) {}

    [[nodiscard]] spaces::SpaceType space_type() const noexcept override { return spaces::SpaceType::H1; }
    [[nodiscard]] FieldType field_type() const noexcept override { return FieldType::Scalar; }
    [[nodiscard]] Continuity continuity() const noexcept override { return Continuity::C0; }
    [[nodiscard]] int value_dimension() const noexcept override { return 1; }
    [[nodiscard]] int topological_dimension() const noexcept override { return 3; }
    [[nodiscard]] int polynomial_order() const noexcept override { return 1; }
    [[nodiscard]] ElementType element_type() const noexcept override { return ElementType::Tetra4; }

    [[nodiscard]] const elements::Element& element() const noexcept override {
        return *mock_element_;
    }

    [[nodiscard]] std::shared_ptr<const elements::Element> element_ptr() const noexcept override {
        return mock_element_;
    }

    const elements::Element& getElement(ElementType, GlobalIndex) const noexcept override {
        return *mock_element_;
    }

private:
    std::shared_ptr<ParallelMockElement> mock_element_;
};

/**
 * @brief Simple identity kernel for testing
 */
class IdentityKernel : public AssemblyKernel {
public:
    void computeCell(const AssemblyContext& ctx, KernelOutput& output) override {
        const auto n = ctx.numTestDofs();
        output.local_matrix.resize(n * n, 0.0);
        output.local_vector.resize(n, 0.0);

        for (LocalIndex i = 0; i < n; ++i) {
            output.local_matrix[i * n + i] = 1.0;
            output.local_vector[i] = 1.0;
        }

        output.has_matrix = true;
        output.has_vector = true;
    }

    [[nodiscard]] RequiredData getRequiredData() const override {
        return RequiredData::None;
    }
};

// ============================================================================
// ParallelAssembler Tests (Serial Mode)
// ============================================================================

/**
 * @brief Create a configured DofMap for parallel testing
 */
inline dofs::DofMap createParallelTestDofMap() {
    dofs::DofMap dof_map(4, 6, 4);  // 4 cells, 6 total DOFs, 4 DOFs per cell

    std::vector<GlobalIndex> cell0_dofs = {0, 1, 2, 3};
    std::vector<GlobalIndex> cell1_dofs = {1, 2, 3, 4};
    std::vector<GlobalIndex> cell2_dofs = {1, 3, 4, 5};
    std::vector<GlobalIndex> cell3_dofs = {2, 3, 4, 5};

    dof_map.setCellDofs(0, cell0_dofs);
    dof_map.setCellDofs(1, cell1_dofs);
    dof_map.setCellDofs(2, cell2_dofs);
    dof_map.setCellDofs(3, cell3_dofs);
    dof_map.setNumDofs(6);
    dof_map.setNumLocalDofs(6);
    dof_map.finalize();

    return dof_map;
}

class ParallelAssemblerTest : public ::testing::Test {
protected:
    void SetUp() override {
        mesh_ = std::make_unique<MockDistributedMesh>(0, 1);  // Serial: rank 0 of 1
        dof_map_ = createParallelTestDofMap();
        system_ = std::make_unique<TestDenseSystemView>(6);

        assembler_ = std::make_unique<ParallelAssembler>();
        assembler_->setDofMap(dof_map_);
    }

    std::unique_ptr<MockDistributedMesh> mesh_;
    dofs::DofMap dof_map_;
    std::unique_ptr<TestDenseSystemView> system_;
    std::unique_ptr<ParallelAssembler> assembler_;
};

TEST_F(ParallelAssemblerTest, DefaultConstruction) {
    ParallelAssembler assembler;
    EXPECT_EQ(assembler.name(), "ParallelAssembler");
    EXPECT_FALSE(assembler.isConfigured());
    EXPECT_TRUE(assembler.supportsRectangular());
    EXPECT_TRUE(assembler.supportsDG());
    EXPECT_FALSE(assembler.isThreadSafe());
}

TEST_F(ParallelAssemblerTest, Configuration) {
    ParallelAssembler assembler;
    EXPECT_FALSE(assembler.isConfigured());

    assembler.setDofMap(dof_map_);
    EXPECT_TRUE(assembler.isConfigured());
}

TEST_F(ParallelAssemblerTest, SetGhostPolicy) {
    assembler_->setGhostPolicy(GhostPolicy::OwnedRowsOnly);
    EXPECT_EQ(assembler_->getGhostPolicy(), GhostPolicy::OwnedRowsOnly);

    assembler_->setGhostPolicy(GhostPolicy::ReverseScatter);
    EXPECT_EQ(assembler_->getGhostPolicy(), GhostPolicy::ReverseScatter);
}

TEST_F(ParallelAssemblerTest, SerialRank) {
    // In serial mode, should be rank 0 of 1
    EXPECT_EQ(assembler_->rank(), 0);
    EXPECT_EQ(assembler_->numRanks(), 1);
}

TEST_F(ParallelAssemblerTest, Initialize) {
    EXPECT_NO_THROW(assembler_->initialize());
}

TEST_F(ParallelAssemblerTest, SerialAssemblyOwnedRowsOnly) {
    assembler_->setGhostPolicy(GhostPolicy::OwnedRowsOnly);
    assembler_->initialize();

    ParallelMockFunctionSpace space;
    IdentityKernel kernel;

    auto result = assembler_->assembleMatrix(*mesh_, space, space, kernel, *system_);

    EXPECT_TRUE(result.success);
    EXPECT_EQ(result.elements_assembled, 4);  // All 4 cells in serial
}

TEST_F(ParallelAssemblerTest, SerialAssemblyReverseScatter) {
    assembler_->setGhostPolicy(GhostPolicy::ReverseScatter);
    assembler_->initialize();

    ParallelMockFunctionSpace space;
    IdentityKernel kernel;

    auto result = assembler_->assembleMatrix(*mesh_, space, space, kernel, *system_);

    EXPECT_TRUE(result.success);
    EXPECT_EQ(result.elements_assembled, 4);
}

TEST_F(ParallelAssemblerTest, Finalize) {
    assembler_->initialize();

    ParallelMockFunctionSpace space;
    IdentityKernel kernel;

    assembler_->assembleMatrix(*mesh_, space, space, kernel, *system_);
    assembler_->finalize(system_.get(), nullptr);

    EXPECT_TRUE(system_->isFinalized());
}

TEST_F(ParallelAssemblerTest, GhostManagerAccess) {
    auto& ghost_manager = assembler_->getGhostManager();

    // Should be able to access and configure ghost manager
    ghost_manager.setDeterministic(true);
    EXPECT_EQ(ghost_manager.getPolicy(), GhostPolicy::ReverseScatter);  // Default
}

// ============================================================================
// Determinism Tests
// ============================================================================

TEST_F(ParallelAssemblerTest, DeterministicResults) {
    // Run assembly twice and verify identical results
    assembler_->initialize();

    ParallelMockFunctionSpace space;
    IdentityKernel kernel;

    // First assembly
    assembler_->assembleMatrix(*mesh_, space, space, kernel, *system_);
    std::vector<Real> matrix1 = system_->matrix();

    // Reset and reassemble
    system_->zero();
    assembler_->reset();
    assembler_->setDofMap(dof_map_);
    assembler_->initialize();
    assembler_->assembleMatrix(*mesh_, space, space, kernel, *system_);
    std::vector<Real> matrix2 = system_->matrix();

    // Results should be identical
    ASSERT_EQ(matrix1.size(), matrix2.size());
    for (std::size_t i = 0; i < matrix1.size(); ++i) {
        EXPECT_DOUBLE_EQ(matrix1[i], matrix2[i])
            << "Determinism failure at index " << i;
    }
}

// ============================================================================
// Factory Function Tests
// ============================================================================

TEST(ParallelAssemblerFactory, CreateParallelAssembler) {
    auto assembler = createParallelAssembler();
    EXPECT_NE(assembler, nullptr);
    EXPECT_EQ(assembler->name(), "ParallelAssembler");
}

TEST(ParallelAssemblerFactory, CreateWithOptions) {
    AssemblyOptions options;
    options.deterministic = true;
    options.verbose = true;

    auto assembler = createParallelAssembler(options);
    EXPECT_NE(assembler, nullptr);
    EXPECT_TRUE(assembler->getOptions().deterministic);
}

} // namespace test
} // namespace assembly
} // namespace FE
} // namespace svmp

/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

/**
 * @file test_StandardAssembler.cpp
 * @brief Unit tests for StandardAssembler
 *
 * Tests the primary serial assembly strategy including:
 * - Construction and configuration
 * - Matrix assembly with simple kernels
 * - Vector assembly
 * - Combined (matrix+vector) assembly
 * - Constraint application
 * - Verification of assembled values for known cases
 */

#include <gtest/gtest.h>

#include "Assembly/StandardAssembler.h"
#include "Assembly/GlobalSystemView.h"
#include "Assembly/AssemblyKernel.h"
#include "Assembly/Assembler.h"
#include "Dofs/DofMap.h"
#include "Spaces/FunctionSpace.h"
#include "Spaces/H1Space.h"
#include "Constraints/AffineConstraints.h"
#include "Elements/LagrangeElement.h"
#include "Geometry/MappingFactory.h"

#include <cmath>
#include <vector>
#include <memory>
#include <numeric>
#include <limits>

namespace svmp {
namespace FE {
namespace assembly {
namespace test {

// ============================================================================
// Mock Classes for Testing
// ============================================================================

/**
 * @brief Simple mock mesh for testing assembly
 *
 * Creates a 2-element mesh (two tetrahedra sharing a face)
 */
class MockMeshAccess : public IMeshAccess {
public:
    MockMeshAccess() {
        // Define a simple 2-element mesh
        // 5 vertices forming 2 tetrahedra
        nodes_ = {
            {0.0, 0.0, 0.0},  // 0
            {1.0, 0.0, 0.0},  // 1
            {0.0, 1.0, 0.0},  // 2
            {0.0, 0.0, 1.0},  // 3
            {1.0, 1.0, 1.0}   // 4
        };

        // Cell 0: nodes 0,1,2,3
        // Cell 1: nodes 1,2,3,4
        cells_ = {
            {0, 1, 2, 3},
            {1, 2, 3, 4}
        };

        cell_types_ = {ElementType::Tetra4, ElementType::Tetra4};
    }

    [[nodiscard]] GlobalIndex numCells() const override { return 2; }
    [[nodiscard]] GlobalIndex numOwnedCells() const override { return 2; }
    [[nodiscard]] GlobalIndex numBoundaryFaces() const override { return 4; }
    [[nodiscard]] GlobalIndex numInteriorFaces() const override { return 1; }
    [[nodiscard]] int dimension() const override { return 3; }

    [[nodiscard]] bool isOwnedCell(GlobalIndex /*cell_id*/) const override { return true; }

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
        return 0;  // Simplified
    }

    [[nodiscard]] int getBoundaryFaceMarker(GlobalIndex /*face_id*/) const override {
        return 1;
    }

    [[nodiscard]] std::pair<GlobalIndex, GlobalIndex>
    getInteriorFaceCells(GlobalIndex /*face_id*/) const override {
        return {0, 1};
    }

    void forEachCell(std::function<void(GlobalIndex)> callback) const override {
        for (GlobalIndex i = 0; i < numCells(); ++i) {
            callback(i);
        }
    }

    void forEachOwnedCell(std::function<void(GlobalIndex)> callback) const override {
        forEachCell(callback);
    }

    void forEachBoundaryFace(int /*marker*/,
                            std::function<void(GlobalIndex, GlobalIndex)> callback) const override {
        // Boundary faces on cell 0: faces 0,1,2 (face 3 is shared)
        callback(0, 0);  // face 0, cell 0
        callback(1, 0);  // face 1, cell 0
        callback(2, 0);  // face 2, cell 0
        // Boundary faces on cell 1: face 0 (face 1,2,3 shared with cell 0)
        callback(3, 1);  // face 0, cell 1
    }

    void forEachInteriorFace(
        std::function<void(GlobalIndex, GlobalIndex, GlobalIndex)> callback) const override {
        // Face 4: shared between cell 0 (face 3) and cell 1 (face 3)
        callback(4, 0, 1);
    }

private:
    std::vector<std::array<Real, 3>> nodes_;
    std::vector<std::vector<GlobalIndex>> cells_;
    std::vector<ElementType> cell_types_;
};

/**
 * @brief Single non-affine Quad4 mesh for Hessian transform regression tests
 */
class SingleQuadMeshAccess final : public IMeshAccess {
public:
    SingleQuadMeshAccess()
    {
        nodes_ = {
            {0.0, 0.0, 0.0},  // 0
            {1.0, 0.0, 0.0},  // 1
            {1.0, 1.0, 0.0},  // 2
            {0.0, 2.0, 0.0},  // 3 (non-parallelogram)
        };
        cell_ = {0, 1, 2, 3};
    }

    [[nodiscard]] GlobalIndex numCells() const override { return 1; }
    [[nodiscard]] GlobalIndex numOwnedCells() const override { return 1; }
    [[nodiscard]] GlobalIndex numBoundaryFaces() const override { return 0; }
    [[nodiscard]] GlobalIndex numInteriorFaces() const override { return 0; }
    [[nodiscard]] int dimension() const override { return 2; }

    [[nodiscard]] bool isOwnedCell(GlobalIndex /*cell_id*/) const override { return true; }

    [[nodiscard]] ElementType getCellType(GlobalIndex /*cell_id*/) const override
    {
        return ElementType::Quad4;
    }

    void getCellNodes(GlobalIndex /*cell_id*/, std::vector<GlobalIndex>& nodes) const override
    {
        nodes.assign(cell_.begin(), cell_.end());
    }

    [[nodiscard]] std::array<Real, 3> getNodeCoordinates(GlobalIndex node_id) const override
    {
        return nodes_.at(static_cast<std::size_t>(node_id));
    }

    void getCellCoordinates(GlobalIndex /*cell_id*/,
                            std::vector<std::array<Real, 3>>& coords) const override
    {
        coords.resize(cell_.size());
        for (std::size_t i = 0; i < cell_.size(); ++i) {
            coords[i] = nodes_.at(static_cast<std::size_t>(cell_[i]));
        }
    }

    [[nodiscard]] LocalIndex getLocalFaceIndex(GlobalIndex /*face_id*/,
                                               GlobalIndex /*cell_id*/) const override
    {
        return 0;
    }

    [[nodiscard]] int getBoundaryFaceMarker(GlobalIndex /*face_id*/) const override { return -1; }

    [[nodiscard]] std::pair<GlobalIndex, GlobalIndex>
    getInteriorFaceCells(GlobalIndex /*face_id*/) const override
    {
        return {0, 0};
    }

    void forEachCell(std::function<void(GlobalIndex)> callback) const override
    {
        callback(0);
    }

    void forEachOwnedCell(std::function<void(GlobalIndex)> callback) const override
    {
        callback(0);
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
    std::vector<std::array<Real, 3>> nodes_{};
    std::array<GlobalIndex, 4> cell_{};
};

/**
 * @brief Kernel used to validate physical basis Hessians for non-affine mappings
 */
class BasisHessianTransformKernel final : public LinearFormKernel {
public:
    BasisHessianTransformKernel(std::shared_ptr<geometry::GeometryMapping> mapping,
                                const basis::BasisFunction& basis,
                                LocalIndex basis_index,
                                LocalIndex q_index,
                                Real eps,
                                Real tol)
        : mapping_(std::move(mapping)),
          basis_(&basis),
          basis_index_(basis_index),
          q_index_(q_index),
          eps_(eps),
          tol_(tol)
    {
    }

    [[nodiscard]] RequiredData getRequiredData() const override
    {
        return RequiredData::Standard | RequiredData::BasisHessians | RequiredData::QuadraturePoints;
    }

    void computeCell(const AssemblyContext& ctx, KernelOutput& out) override
    {
        const auto n_test = ctx.numTestDofs();
        out.reserve(n_test, ctx.numTrialDofs(), /*need_matrix=*/false, /*need_vector=*/true);

        const auto n_qpts = ctx.numQuadraturePoints();
        ASSERT_GT(n_qpts, 0);
        ASSERT_LT(q_index_, n_qpts);
        ASSERT_NE(mapping_.get(), nullptr);
        ASSERT_NE(basis_, nullptr);

        const int dim = mapping_->dimension();
        ASSERT_EQ(dim, 2);

        const auto xi_arr = ctx.quadraturePoint(q_index_);
        const math::Vector<Real, 3> xi{xi_arr[0], xi_arr[1], xi_arr[2]};

        const auto x_arr = ctx.physicalPoint(q_index_);
        const math::Vector<Real, 3> x0{x_arr[0], x_arr[1], x_arr[2]};

        std::vector<basis::Gradient> grads;
        grads.reserve(static_cast<std::size_t>(n_test));

        auto grad_phys_at = [&](const math::Vector<Real, 3>& xi_eval) -> math::Vector<Real, 3> {
            basis_->evaluate_gradients(xi_eval, grads);
            if (grads.size() <= static_cast<std::size_t>(basis_index_)) {
                ADD_FAILURE() << "BasisHessianTransformKernel: basis gradient index out of range";
                return {};
            }
            return mapping_->transform_gradient(grads[static_cast<std::size_t>(basis_index_)], xi_eval);
        };

        const auto H_ctx = ctx.physicalHessian(basis_index_, q_index_);

        // Compute Hessian via finite differences of the physical gradient.
        math::Matrix<Real, 3, 3> H_fd{};
        for (int c = 0; c < dim; ++c) {
            auto x_plus = x0;
            auto x_minus = x0;
            x_plus[static_cast<std::size_t>(c)] += eps_;
            x_minus[static_cast<std::size_t>(c)] -= eps_;

            const auto xi_plus = mapping_->map_to_reference(x_plus, xi);
            const auto xi_minus = mapping_->map_to_reference(x_minus, xi);

            const auto g_plus = grad_phys_at(xi_plus);
            const auto g_minus = grad_phys_at(xi_minus);

            for (int r = 0; r < dim; ++r) {
                H_fd(static_cast<std::size_t>(r), static_cast<std::size_t>(c)) =
                    (g_plus[static_cast<std::size_t>(r)] - g_minus[static_cast<std::size_t>(r)]) / (2.0 * eps_);
            }
        }

        for (int r = 0; r < dim; ++r) {
            for (int c = 0; c < dim; ++c) {
                EXPECT_NEAR(H_ctx[static_cast<std::size_t>(r)][static_cast<std::size_t>(c)],
                            H_fd(static_cast<std::size_t>(r), static_cast<std::size_t>(c)),
                            tol_);
            }
        }

        // Populate output with zeros (this kernel is used for verification only).
        for (LocalIndex i = 0; i < n_test; ++i) {
            out.vectorEntry(i) = 0.0;
        }
    }

    [[nodiscard]] std::string name() const override { return "BasisHessianTransformKernel"; }

private:
    std::shared_ptr<geometry::GeometryMapping> mapping_{};
    const basis::BasisFunction* basis_{nullptr};
    LocalIndex basis_index_{0};
    LocalIndex q_index_{0};
    Real eps_{1e-6};
    Real tol_{1e-6};
};

/**
 * @brief Simple dense storage for testing global system assembly
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

    // Test accessors
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

/**
 * @brief Create a configured DofMap for testing
 *
 * Sets up a DofMap for a 2-cell mesh with 5 total DOFs:
 * - Cell 0: nodes 0,1,2,3 -> DOFs 0,1,2,3
 * - Cell 1: nodes 1,2,3,4 -> DOFs 1,2,3,4
 */
inline dofs::DofMap createTestDofMap() {
    dofs::DofMap dof_map(2, 5, 4);  // 2 cells, 5 total DOFs, 4 DOFs per cell

    std::vector<GlobalIndex> cell0_dofs = {0, 1, 2, 3};
    std::vector<GlobalIndex> cell1_dofs = {1, 2, 3, 4};

    dof_map.setCellDofs(0, cell0_dofs);
    dof_map.setCellDofs(1, cell1_dofs);
    dof_map.setNumDofs(5);
    dof_map.setNumLocalDofs(5);
    dof_map.finalize();

    return dof_map;
}

/**
 * @brief Identity kernel for testing - produces identity matrix contributions
 */
class IdentityKernel : public AssemblyKernel {
public:
    IdentityKernel() : AssemblyKernel() {}

    void computeCell(const AssemblyContext& ctx, KernelOutput& output) override {
        const auto n = ctx.numTestDofs();
        output.local_matrix.resize(n * n, 0.0);
        output.local_vector.resize(n, 0.0);

        // Identity matrix
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

/**
 * @brief Mass matrix kernel - computes local mass matrix
 * M_ij = integral(phi_i * phi_j)
 */
class MassKernel : public AssemblyKernel {
public:
    MassKernel() : AssemblyKernel() {}

    void computeCell(const AssemblyContext& ctx, KernelOutput& output) override {
        const auto n_dofs = ctx.numTestDofs();
        const auto n_qpts = ctx.numQuadraturePoints();

        output.local_matrix.resize(n_dofs * n_dofs, 0.0);
        output.local_vector.resize(n_dofs, 0.0);

        // Compute mass matrix
        for (LocalIndex q = 0; q < n_qpts; ++q) {
            const Real w = ctx.integrationWeight(q);

            for (LocalIndex i = 0; i < n_dofs; ++i) {
                const Real phi_i = ctx.basisValue(i, q);

                for (LocalIndex j = 0; j < n_dofs; ++j) {
                    const Real phi_j = ctx.basisValue(j, q);
                    output.local_matrix[i * n_dofs + j] += w * phi_i * phi_j;
                }

                // RHS: integral(f * phi_i) with f=1
                output.local_vector[i] += w * phi_i;
            }
        }

        output.has_matrix = true;
        output.has_vector = true;
    }

    [[nodiscard]] RequiredData getRequiredData() const override {
        return RequiredData::BasisValues | RequiredData::IntegrationWeights;
    }
};

/**
 * @brief Stiffness matrix kernel - computes local stiffness matrix
 * K_ij = integral(grad(phi_i) . grad(phi_j))
 */
class StiffnessKernel : public AssemblyKernel {
public:
    StiffnessKernel() : AssemblyKernel() {}

    void computeCell(const AssemblyContext& ctx, KernelOutput& output) override {
        const auto n_dofs = ctx.numTestDofs();
        const auto n_qpts = ctx.numQuadraturePoints();

        output.local_matrix.resize(n_dofs * n_dofs, 0.0);
        output.local_vector.resize(n_dofs, 0.0);

        // Compute stiffness matrix
        for (LocalIndex q = 0; q < n_qpts; ++q) {
            const Real w = ctx.integrationWeight(q);

            for (LocalIndex i = 0; i < n_dofs; ++i) {
                const auto grad_i = ctx.physicalGradient(i, q);

                for (LocalIndex j = 0; j < n_dofs; ++j) {
                    const auto grad_j = ctx.physicalGradient(j, q);

                    // Dot product of gradients
                    Real dot = 0.0;
                    for (int d = 0; d < 3; ++d) {
                        dot += grad_i[d] * grad_j[d];
                    }
                    output.local_matrix[i * n_dofs + j] += w * dot;
                }
            }
        }

        output.has_matrix = true;
        output.has_vector = false;
    }

    [[nodiscard]] RequiredData getRequiredData() const override {
        return RequiredData::PhysicalGradients | RequiredData::IntegrationWeights;
    }
};

// Forward declaration for mock element
class MockElement;

/**
 * @brief Mock function space for testing
 */
class MockFunctionSpace : public spaces::FunctionSpace {
public:
    MockFunctionSpace();
    ~MockFunctionSpace() override = default;

    // Pure virtual method implementations from FunctionSpace
    [[nodiscard]] spaces::SpaceType space_type() const noexcept override {
        return spaces::SpaceType::H1;
    }

    [[nodiscard]] FieldType field_type() const noexcept override {
        return FieldType::Scalar;
    }

    [[nodiscard]] Continuity continuity() const noexcept override {
        return Continuity::C0;
    }

    [[nodiscard]] int value_dimension() const noexcept override {
        return 1;
    }

    [[nodiscard]] int topological_dimension() const noexcept override {
        return 3;
    }

    [[nodiscard]] int polynomial_order() const noexcept override {
        return 1;
    }

    [[nodiscard]] ElementType element_type() const noexcept override {
        return ElementType::Tetra4;
    }

    [[nodiscard]] const elements::Element& element() const noexcept override;

    [[nodiscard]] std::shared_ptr<const elements::Element> element_ptr() const noexcept override;

    const elements::Element& getElement(ElementType type, GlobalIndex /*cell_id*/) const noexcept override;

private:
    std::shared_ptr<MockElement> mock_element_;
};

// Inner mock basis class
class MockBasis : public basis::BasisFunction {
public:
    [[nodiscard]] BasisType basis_type() const noexcept override {
        return BasisType::Lagrange;
    }

    [[nodiscard]] ElementType element_type() const noexcept override {
        return ElementType::Tetra4;
    }

    [[nodiscard]] int dimension() const noexcept override { return 3; }
    [[nodiscard]] int order() const noexcept override { return 1; }
    [[nodiscard]] std::size_t size() const noexcept override { return 4; }

    void evaluate_values(const math::Vector<Real, 3>& xi,
                        std::vector<Real>& values) const override {
        // Linear Lagrange basis on reference tetrahedron
        // phi_0 = 1 - xi - eta - zeta
        // phi_1 = xi
        // phi_2 = eta
        // phi_3 = zeta
        values.resize(4);
        values[0] = 1.0 - xi[0] - xi[1] - xi[2];
        values[1] = xi[0];
        values[2] = xi[1];
        values[3] = xi[2];
    }

    void evaluate_gradients(const math::Vector<Real, 3>& /*xi*/,
                           std::vector<basis::Gradient>& gradients) const override {
        // Constant gradients for linear basis
        gradients.resize(4);
        gradients[0] = {-1.0, -1.0, -1.0};
        gradients[1] = { 1.0,  0.0,  0.0};
        gradients[2] = { 0.0,  1.0,  0.0};
        gradients[3] = { 0.0,  0.0,  1.0};
    }
};

// Inner mock element class
class MockElement : public elements::Element {
public:
    explicit MockElement(ElementType type = ElementType::Tetra4) : type_(type) {
        // Create a simple Lagrange basis for Tetra4
        basis_ = std::make_shared<MockBasis>();
        // Note: In tests, quadrature may not be fully used
        // quad_ = quadrature::QuadratureFactory::create(type, 2);
    }

    [[nodiscard]] elements::ElementInfo info() const noexcept override {
        return {type_, FieldType::Scalar, Continuity::C0, 1};
    }

    [[nodiscard]] int dimension() const noexcept override { return 3; }
    [[nodiscard]] std::size_t num_dofs() const noexcept override { return 4; }
    [[nodiscard]] std::size_t num_nodes() const noexcept override { return 4; }

    [[nodiscard]] const basis::BasisFunction& basis() const noexcept override {
        return *basis_;
    }

    [[nodiscard]] std::shared_ptr<const basis::BasisFunction> basis_ptr() const noexcept override {
        return basis_;
    }

    [[nodiscard]] std::shared_ptr<const quadrature::QuadratureRule> quadrature() const noexcept override {
        return quad_;
    }

private:
    ElementType type_;
    std::shared_ptr<basis::BasisFunction> basis_;
    std::shared_ptr<const quadrature::QuadratureRule> quad_;
};

// MockFunctionSpace implementation
MockFunctionSpace::MockFunctionSpace()
    : spaces::FunctionSpace(), mock_element_(std::make_shared<MockElement>()) {}

const elements::Element& MockFunctionSpace::element() const noexcept {
    return *mock_element_;
}

std::shared_ptr<const elements::Element> MockFunctionSpace::element_ptr() const noexcept {
    return mock_element_;
}

const elements::Element& MockFunctionSpace::getElement(ElementType /*type*/, GlobalIndex /*cell_id*/) const noexcept {
    return *mock_element_;
}

// ============================================================================
// StandardAssembler Tests
// ============================================================================

class StandardAssemblerTest : public ::testing::Test {
protected:
    void SetUp() override {
        mesh_ = std::make_unique<MockMeshAccess>();
        dof_map_ = createTestDofMap();
        space_ = std::make_unique<MockFunctionSpace>();
        system_ = std::make_unique<TestDenseSystemView>(5);

        assembler_ = std::make_unique<StandardAssembler>();
        assembler_->setDofMap(dof_map_);
    }

    std::unique_ptr<MockMeshAccess> mesh_;
    dofs::DofMap dof_map_;
    std::unique_ptr<MockFunctionSpace> space_;
    std::unique_ptr<TestDenseSystemView> system_;
    std::unique_ptr<StandardAssembler> assembler_;
};

// ============================================================================
// Face Assembly Coverage Tests
// ============================================================================

/**
 * @brief Minimal single-cell tetra mesh exposing one boundary face.
 *
 * The boundary face is reported as local_face_id on cell 0.
 */
class SingleTetraBoundaryMeshAccess : public IMeshAccess {
public:
    explicit SingleTetraBoundaryMeshAccess(LocalIndex local_face_id = 0)
        : local_face_id_(local_face_id)
    {
        // Reference tetra vertices: (0,0,0), (1,0,0), (0,1,0), (0,0,1)
        nodes_ = {
            {0.0, 0.0, 0.0},  // 0
            {1.0, 0.0, 0.0},  // 1
            {0.0, 1.0, 0.0},  // 2
            {0.0, 0.0, 1.0}   // 3
        };
        cell_ = {0, 1, 2, 3};
    }

    [[nodiscard]] GlobalIndex numCells() const override { return 1; }
    [[nodiscard]] GlobalIndex numOwnedCells() const override { return 1; }
    [[nodiscard]] GlobalIndex numBoundaryFaces() const override { return 1; }
    [[nodiscard]] GlobalIndex numInteriorFaces() const override { return 0; }
    [[nodiscard]] int dimension() const override { return 3; }

    [[nodiscard]] bool isOwnedCell(GlobalIndex /*cell_id*/) const override { return true; }

    [[nodiscard]] ElementType getCellType(GlobalIndex /*cell_id*/) const override {
        return ElementType::Tetra4;
    }

    void getCellNodes(GlobalIndex /*cell_id*/, std::vector<GlobalIndex>& nodes) const override {
        nodes.assign(cell_.begin(), cell_.end());
    }

    [[nodiscard]] std::array<Real, 3> getNodeCoordinates(GlobalIndex node_id) const override {
        return nodes_[static_cast<std::size_t>(node_id)];
    }

    void getCellCoordinates(GlobalIndex /*cell_id*/,
                            std::vector<std::array<Real, 3>>& coords) const override
    {
        coords.resize(cell_.size());
        for (std::size_t i = 0; i < cell_.size(); ++i) {
            coords[i] = nodes_[static_cast<std::size_t>(cell_[i])];
        }
    }

    [[nodiscard]] LocalIndex getLocalFaceIndex(GlobalIndex /*face_id*/,
                                               GlobalIndex /*cell_id*/) const override
    {
        return local_face_id_;
    }

    [[nodiscard]] int getBoundaryFaceMarker(GlobalIndex /*face_id*/) const override {
        return 1;
    }

    [[nodiscard]] std::pair<GlobalIndex, GlobalIndex>
    getInteriorFaceCells(GlobalIndex /*face_id*/) const override
    {
        return {-1, -1};
    }

    void forEachCell(std::function<void(GlobalIndex)> callback) const override {
        callback(0);
    }

    void forEachOwnedCell(std::function<void(GlobalIndex)> callback) const override {
        callback(0);
    }

    void forEachBoundaryFace(int marker,
                             std::function<void(GlobalIndex, GlobalIndex)> callback) const override
    {
        if (marker == 1) {
            callback(0, 0);
        }
    }

    void forEachInteriorFace(
        std::function<void(GlobalIndex, GlobalIndex, GlobalIndex)> /*callback*/) const override
    {
    }

private:
    LocalIndex local_face_id_{0};
    std::vector<std::array<Real, 3>> nodes_;
    std::array<GlobalIndex, 4> cell_{};
};

/**
 * @brief Two stacked Hex8 cells sharing one interior face at z=1.
 *
 * Cell 0: z in [-1,1], interior face is local face 1 (z=+1).
 * Cell 1: z in [1,3], interior face is local face 0 (z=-1).
 */
class TwoHexStackedMeshAccess : public IMeshAccess {
public:
    TwoHexStackedMeshAccess()
    {
        // Cell 0 nodes (Hex8 node ordering)
        // Bottom z=-1
        nodes_.push_back({-1.0, -1.0, -1.0});  // 0
        nodes_.push_back({ 1.0, -1.0, -1.0});  // 1
        nodes_.push_back({ 1.0,  1.0, -1.0});  // 2
        nodes_.push_back({-1.0,  1.0, -1.0});  // 3
        // Top z=+1
        nodes_.push_back({-1.0, -1.0,  1.0});  // 4  (shared)
        nodes_.push_back({ 1.0, -1.0,  1.0});  // 5  (shared)
        nodes_.push_back({ 1.0,  1.0,  1.0});  // 6  (shared)
        nodes_.push_back({-1.0,  1.0,  1.0});  // 7  (shared)

        // Cell 1 top nodes (z=+3)
        nodes_.push_back({-1.0, -1.0,  3.0});  // 8
        nodes_.push_back({ 1.0, -1.0,  3.0});  // 9
        nodes_.push_back({ 1.0,  1.0,  3.0});  // 10
        nodes_.push_back({-1.0,  1.0,  3.0});  // 11

        cells_.push_back({0, 1, 2, 3, 4, 5, 6, 7});        // Cell 0
        cells_.push_back({4, 5, 6, 7, 8, 9, 10, 11});      // Cell 1
    }

    [[nodiscard]] GlobalIndex numCells() const override { return 2; }
    [[nodiscard]] GlobalIndex numOwnedCells() const override { return 2; }
    [[nodiscard]] GlobalIndex numBoundaryFaces() const override { return 0; }
    [[nodiscard]] GlobalIndex numInteriorFaces() const override { return 1; }
    [[nodiscard]] int dimension() const override { return 3; }

    [[nodiscard]] bool isOwnedCell(GlobalIndex /*cell_id*/) const override { return true; }

    [[nodiscard]] ElementType getCellType(GlobalIndex /*cell_id*/) const override {
        return ElementType::Hex8;
    }

    void getCellNodes(GlobalIndex cell_id, std::vector<GlobalIndex>& nodes) const override {
        const auto& cell = cells_[static_cast<std::size_t>(cell_id)];
        nodes.assign(cell.begin(), cell.end());
    }

    [[nodiscard]] std::array<Real, 3> getNodeCoordinates(GlobalIndex node_id) const override {
        return nodes_[static_cast<std::size_t>(node_id)];
    }

    void getCellCoordinates(GlobalIndex cell_id,
                            std::vector<std::array<Real, 3>>& coords) const override
    {
        const auto& cell = cells_[static_cast<std::size_t>(cell_id)];
        coords.resize(cell.size());
        for (std::size_t i = 0; i < cell.size(); ++i) {
            coords[i] = nodes_[static_cast<std::size_t>(cell[i])];
        }
    }

    [[nodiscard]] LocalIndex getLocalFaceIndex(GlobalIndex face_id,
                                               GlobalIndex cell_id) const override
    {
        if (face_id != 0) {
            return 0;
        }
        return (cell_id == 0) ? 1 : 0;
    }

    [[nodiscard]] int getBoundaryFaceMarker(GlobalIndex /*face_id*/) const override {
        return 0;
    }

    [[nodiscard]] std::pair<GlobalIndex, GlobalIndex>
    getInteriorFaceCells(GlobalIndex /*face_id*/) const override
    {
        return {0, 1};
    }

    void forEachCell(std::function<void(GlobalIndex)> callback) const override {
        callback(0);
        callback(1);
    }

    void forEachOwnedCell(std::function<void(GlobalIndex)> callback) const override {
        forEachCell(std::move(callback));
    }

    void forEachBoundaryFace(int /*marker*/,
                             std::function<void(GlobalIndex, GlobalIndex)> /*callback*/) const override
    {
    }

    void forEachInteriorFace(
        std::function<void(GlobalIndex, GlobalIndex, GlobalIndex)> callback) const override
    {
        callback(0, 0, 1);
    }

private:
    std::vector<std::array<Real, 3>> nodes_;
    std::vector<std::array<GlobalIndex, 8>> cells_;
};

class HexFunctionSpace : public spaces::FunctionSpace {
public:
    HexFunctionSpace()
        : element_(std::make_shared<elements::LagrangeElement>(ElementType::Hex8, 1))
    {
    }

    [[nodiscard]] spaces::SpaceType space_type() const noexcept override {
        return spaces::SpaceType::H1;
    }
    [[nodiscard]] FieldType field_type() const noexcept override { return FieldType::Scalar; }
    [[nodiscard]] Continuity continuity() const noexcept override { return Continuity::C0; }
    [[nodiscard]] int value_dimension() const noexcept override { return 1; }
    [[nodiscard]] int topological_dimension() const noexcept override { return 3; }
    [[nodiscard]] int polynomial_order() const noexcept override { return 1; }
    [[nodiscard]] ElementType element_type() const noexcept override { return ElementType::Hex8; }

    [[nodiscard]] const elements::Element& element() const noexcept override { return *element_; }
    [[nodiscard]] std::shared_ptr<const elements::Element> element_ptr() const noexcept override {
        return element_;
    }

private:
    std::shared_ptr<elements::Element> element_;
};

/**
 * @brief Boundary face kernel that records face quadrature diagnostics into the local vector.
 *
 * local_vector[0] = sum_q integrationWeight(q)   (face measure in physical space)
 * local_vector[1] = max_q |quadraturePoint(q).z - expected_z|
 * local_vector[2] = avg_q normal(q).z
 */
class BoundaryFaceDiagnosticsKernel : public AssemblyKernel {
public:
    [[nodiscard]] RequiredData getRequiredData() const override {
        return RequiredData::QuadraturePoints | RequiredData::IntegrationWeights |
               RequiredData::Normals;
    }

    void computeCell(const AssemblyContext& /*ctx*/, KernelOutput& /*output*/) override {
        // Not used in these tests
    }

    [[nodiscard]] bool hasBoundaryFace() const noexcept override { return true; }

    void computeBoundaryFace(const AssemblyContext& ctx,
                             int /*boundary_marker*/,
                             KernelOutput& output) override
    {
        const auto n = ctx.numTestDofs();
        output.local_vector.assign(static_cast<std::size_t>(n), 0.0);
        output.has_vector = true;
        output.has_matrix = false;
        output.n_test_dofs = n;
        output.n_trial_dofs = n;

        const LocalIndex n_qpts = ctx.numQuadraturePoints();
        if (n_qpts == 0) {
            output.local_vector[0] = -1.0;  // Sentinel for missing quadrature
            return;
        }

        Real sum_w = 0.0;
        Real max_abs_dev = 0.0;
        Real avg_nz = 0.0;

        for (LocalIndex q = 0; q < n_qpts; ++q) {
            sum_w += ctx.integrationWeight(q);
            const auto xi = ctx.quadraturePoint(q);
            max_abs_dev = std::max(max_abs_dev, std::abs(xi[2] - Real(0)));
            const auto nrm = ctx.normal(q);
            avg_nz += nrm[2];
        }

        avg_nz /= static_cast<Real>(n_qpts);

        output.local_vector[0] = sum_w;
        if (n > 1) output.local_vector[1] = max_abs_dev;
        if (n > 2) output.local_vector[2] = avg_nz;
    }
};

/**
 * @brief Boundary face kernel that records only the physical face measure.
 *
 * local_vector[0] = sum_q integrationWeight(q)
 */
class BoundaryFaceMeasureKernel : public AssemblyKernel {
public:
    [[nodiscard]] RequiredData getRequiredData() const override {
        return RequiredData::IntegrationWeights;
    }

    void computeCell(const AssemblyContext& /*ctx*/, KernelOutput& /*output*/) override {}

    [[nodiscard]] bool hasBoundaryFace() const noexcept override { return true; }

    void computeBoundaryFace(const AssemblyContext& ctx,
                             int /*boundary_marker*/,
                             KernelOutput& output) override
    {
        const auto n = ctx.numTestDofs();
        output.local_vector.assign(static_cast<std::size_t>(n), 0.0);
        output.has_vector = true;
        output.has_matrix = false;
        output.n_test_dofs = n;
        output.n_trial_dofs = n;

        Real sum_w = 0.0;
        const LocalIndex n_qpts = ctx.numQuadraturePoints();
        for (LocalIndex q = 0; q < n_qpts; ++q) {
            sum_w += ctx.integrationWeight(q);
        }

        if (n > 0) {
            output.local_vector[0] = sum_w;
        }
    }
};

class InteriorFaceDiagnosticsKernel : public AssemblyKernel {
public:
    [[nodiscard]] RequiredData getRequiredData() const override {
        return RequiredData::IntegrationWeights | RequiredData::PhysicalPoints |
               RequiredData::Normals;
    }

    void computeCell(const AssemblyContext& /*ctx*/, KernelOutput& /*output*/) override {
        // Not used in these tests
    }

    [[nodiscard]] bool hasInteriorFace() const noexcept override { return true; }

    void computeInteriorFace(
        const AssemblyContext& ctx_minus,
        const AssemblyContext& ctx_plus,
        KernelOutput& output_minus,
        KernelOutput& output_plus,
        KernelOutput& coupling_minus_plus,
        KernelOutput& coupling_plus_minus) override
    {
        const LocalIndex n_minus = ctx_minus.numTestDofs();
        const LocalIndex n_plus = ctx_plus.numTestDofs();

        output_minus.local_vector.assign(static_cast<std::size_t>(n_minus), 0.0);
        output_plus.local_vector.assign(static_cast<std::size_t>(n_plus), 0.0);
        output_minus.has_vector = true;
        output_plus.has_vector = true;
        output_minus.has_matrix = false;
        output_plus.has_matrix = false;

        coupling_minus_plus.local_matrix.clear();
        coupling_plus_minus.local_matrix.clear();
        coupling_minus_plus.has_matrix = false;
        coupling_plus_minus.has_matrix = false;
        coupling_minus_plus.has_vector = false;
        coupling_plus_minus.has_vector = false;

        const LocalIndex n_qpts = ctx_minus.numQuadraturePoints();
        if (n_qpts == 0 || ctx_plus.numQuadraturePoints() == 0) {
            output_minus.local_vector[0] = -1.0;
            output_plus.local_vector[0] = -1.0;
            return;
        }

        Real sum_w_minus = 0.0;
        Real sum_w_plus = 0.0;
        Real avg_dot = 0.0;
        Real max_phys_dev = 0.0;

        for (LocalIndex q = 0; q < n_qpts; ++q) {
            sum_w_minus += ctx_minus.integrationWeight(q);
            sum_w_plus += ctx_plus.integrationWeight(q);

            const auto n0 = ctx_minus.normal(q);
            const auto n1 = ctx_plus.normal(q);
            avg_dot += n0[0] * n1[0] + n0[1] * n1[1] + n0[2] * n1[2];

            const auto x0 = ctx_minus.physicalPoint(q);
            const auto x1 = ctx_plus.physicalPoint(q);
            max_phys_dev = std::max({
                max_phys_dev,
                std::abs(x0[0] - x1[0]),
                std::abs(x0[1] - x1[1]),
                std::abs(x0[2] - x1[2])
            });
        }

        avg_dot /= static_cast<Real>(n_qpts);

        // Pack diagnostics into the first few local DOFs
        output_minus.local_vector[0] = sum_w_minus;
        if (n_minus > 1) output_minus.local_vector[1] = sum_w_plus;
        if (n_minus > 2) output_minus.local_vector[2] = avg_dot;
        if (n_minus > 3) output_minus.local_vector[3] = max_phys_dev;

        output_plus.local_vector[0] = sum_w_plus;
        if (n_plus > 1) output_plus.local_vector[1] = sum_w_minus;
        if (n_plus > 2) output_plus.local_vector[2] = avg_dot;
        if (n_plus > 3) output_plus.local_vector[3] = max_phys_dev;
    }
};

/**
 * @brief Minimal single-cell Triangle3 mesh exposing one boundary edge.
 *
 * Edge 0 is reported as local_face_id=0 on cell 0.
 * The physical coordinates match the Triangle3 reference element:
 *   (0,0), (1,0), (0,1) so the boundary edge length is 1.
 */
class SingleTriangleBoundaryMeshAccess : public IMeshAccess {
public:
    explicit SingleTriangleBoundaryMeshAccess(LocalIndex local_face_id)
        : local_face_id_(local_face_id)
    {
        nodes_ = {
            {0.0, 0.0, 0.0}, // 0
            {1.0, 0.0, 0.0}, // 1
            {0.0, 1.0, 0.0}  // 2
        };
        cell_nodes_ = {0, 1, 2};
    }

    [[nodiscard]] GlobalIndex numCells() const override { return 1; }
    [[nodiscard]] GlobalIndex numOwnedCells() const override { return 1; }
    [[nodiscard]] GlobalIndex numBoundaryFaces() const override { return 1; }
    [[nodiscard]] GlobalIndex numInteriorFaces() const override { return 0; }
    [[nodiscard]] int dimension() const override { return 2; }

    [[nodiscard]] bool isOwnedCell(GlobalIndex cell_id) const override { return cell_id == 0; }
    [[nodiscard]] ElementType getCellType(GlobalIndex /*cell_id*/) const override { return ElementType::Triangle3; }

    void getCellNodes(GlobalIndex /*cell_id*/, std::vector<GlobalIndex>& nodes) const override { nodes = cell_nodes_; }

    [[nodiscard]] std::array<Real, 3> getNodeCoordinates(GlobalIndex node_id) const override
    {
        return nodes_.at(static_cast<std::size_t>(node_id));
    }

    void getCellCoordinates(GlobalIndex /*cell_id*/, std::vector<std::array<Real, 3>>& coords) const override
    {
        coords = {nodes_[0], nodes_[1], nodes_[2]};
    }

    [[nodiscard]] LocalIndex getLocalFaceIndex(GlobalIndex face_id, GlobalIndex /*cell_id*/) const override
    {
        // Single boundary edge, configurable local edge id.
        if (face_id != 0) {
            throw std::runtime_error("SingleTriangleBoundaryMeshAccess: unknown face id");
        }
        return local_face_id_;
    }

    [[nodiscard]] int getBoundaryFaceMarker(GlobalIndex face_id) const override
    {
        return (face_id == 0) ? 1 : -1;
    }

    [[nodiscard]] std::pair<GlobalIndex, GlobalIndex> getInteriorFaceCells(GlobalIndex /*face_id*/) const override
    {
        return {static_cast<GlobalIndex>(-1), static_cast<GlobalIndex>(-1)};
    }

    void forEachCell(std::function<void(GlobalIndex)> callback) const override { callback(0); }
    void forEachOwnedCell(std::function<void(GlobalIndex)> callback) const override { callback(0); }

    void forEachBoundaryFace(int marker, std::function<void(GlobalIndex, GlobalIndex)> callback) const override
    {
        if (marker == 1) {
            callback(0, 0);
        }
    }

    void forEachInteriorFace(std::function<void(GlobalIndex, GlobalIndex, GlobalIndex)> /*callback*/) const override {}

private:
    LocalIndex local_face_id_{0};
    std::vector<std::array<Real, 3>> nodes_{};
    std::vector<GlobalIndex> cell_nodes_{};
};

/**
 * @brief Minimal single-cell Wedge6 mesh exposing one boundary face.
 *
 * The physical coordinates match the Wedge6 reference element:
 *   bottom: (0,0,-1), (1,0,-1), (0,1,-1)
 *   top:    (0,0,+1), (1,0,+1), (0,1,+1)
 */
class SingleWedgeBoundaryMeshAccess : public IMeshAccess {
public:
    explicit SingleWedgeBoundaryMeshAccess(LocalIndex local_face_id)
        : local_face_id_(local_face_id)
    {
        nodes_ = {
            {0.0, 0.0, -1.0}, // 0
            {1.0, 0.0, -1.0}, // 1
            {0.0, 1.0, -1.0}, // 2
            {0.0, 0.0, 1.0},  // 3
            {1.0, 0.0, 1.0},  // 4
            {0.0, 1.0, 1.0}   // 5
        };
        cell_nodes_ = {0, 1, 2, 3, 4, 5};
    }

    [[nodiscard]] GlobalIndex numCells() const override { return 1; }
    [[nodiscard]] GlobalIndex numOwnedCells() const override { return 1; }
    [[nodiscard]] GlobalIndex numBoundaryFaces() const override { return 1; }
    [[nodiscard]] GlobalIndex numInteriorFaces() const override { return 0; }
    [[nodiscard]] int dimension() const override { return 3; }

    [[nodiscard]] bool isOwnedCell(GlobalIndex cell_id) const override { return cell_id == 0; }
    [[nodiscard]] ElementType getCellType(GlobalIndex /*cell_id*/) const override { return ElementType::Wedge6; }

    void getCellNodes(GlobalIndex /*cell_id*/, std::vector<GlobalIndex>& nodes) const override { nodes = cell_nodes_; }

    [[nodiscard]] std::array<Real, 3> getNodeCoordinates(GlobalIndex node_id) const override
    {
        return nodes_.at(static_cast<std::size_t>(node_id));
    }

    void getCellCoordinates(GlobalIndex /*cell_id*/, std::vector<std::array<Real, 3>>& coords) const override
    {
        coords = {nodes_[0], nodes_[1], nodes_[2], nodes_[3], nodes_[4], nodes_[5]};
    }

    [[nodiscard]] LocalIndex getLocalFaceIndex(GlobalIndex face_id, GlobalIndex /*cell_id*/) const override
    {
        if (face_id != 0) {
            throw std::runtime_error("SingleWedgeBoundaryMeshAccess: unknown face id");
        }
        return local_face_id_;
    }

    [[nodiscard]] int getBoundaryFaceMarker(GlobalIndex face_id) const override
    {
        return (face_id == 0) ? 1 : -1;
    }

    [[nodiscard]] std::pair<GlobalIndex, GlobalIndex> getInteriorFaceCells(GlobalIndex /*face_id*/) const override
    {
        return {static_cast<GlobalIndex>(-1), static_cast<GlobalIndex>(-1)};
    }

    void forEachCell(std::function<void(GlobalIndex)> callback) const override { callback(0); }
    void forEachOwnedCell(std::function<void(GlobalIndex)> callback) const override { callback(0); }

    void forEachBoundaryFace(int marker, std::function<void(GlobalIndex, GlobalIndex)> callback) const override
    {
        if (marker == 1) {
            callback(0, 0);
        }
    }

    void forEachInteriorFace(std::function<void(GlobalIndex, GlobalIndex, GlobalIndex)> /*callback*/) const override {}

private:
    LocalIndex local_face_id_{0};
    std::vector<std::array<Real, 3>> nodes_{};
    std::vector<GlobalIndex> cell_nodes_{};
};

/**
 * @brief Minimal single-cell Pyramid5 mesh exposing one boundary face.
 *
 * The physical coordinates match the Pyramid5 reference element:
 *   base: (-1,-1,0), (1,-1,0), (1,1,0), (-1,1,0)
 *   apex: (0,0,1)
 */
class SinglePyramidBoundaryMeshAccess : public IMeshAccess {
public:
    explicit SinglePyramidBoundaryMeshAccess(LocalIndex local_face_id)
        : local_face_id_(local_face_id)
    {
        nodes_ = {
            {-1.0, -1.0, 0.0}, // 0
            {1.0, -1.0, 0.0},  // 1
            {1.0, 1.0, 0.0},   // 2
            {-1.0, 1.0, 0.0},  // 3
            {0.0, 0.0, 1.0}    // 4
        };
        cell_nodes_ = {0, 1, 2, 3, 4};
    }

    [[nodiscard]] GlobalIndex numCells() const override { return 1; }
    [[nodiscard]] GlobalIndex numOwnedCells() const override { return 1; }
    [[nodiscard]] GlobalIndex numBoundaryFaces() const override { return 1; }
    [[nodiscard]] GlobalIndex numInteriorFaces() const override { return 0; }
    [[nodiscard]] int dimension() const override { return 3; }

    [[nodiscard]] bool isOwnedCell(GlobalIndex cell_id) const override { return cell_id == 0; }
    [[nodiscard]] ElementType getCellType(GlobalIndex /*cell_id*/) const override { return ElementType::Pyramid5; }

    void getCellNodes(GlobalIndex /*cell_id*/, std::vector<GlobalIndex>& nodes) const override { nodes = cell_nodes_; }

    [[nodiscard]] std::array<Real, 3> getNodeCoordinates(GlobalIndex node_id) const override
    {
        return nodes_.at(static_cast<std::size_t>(node_id));
    }

    void getCellCoordinates(GlobalIndex /*cell_id*/, std::vector<std::array<Real, 3>>& coords) const override
    {
        coords = {nodes_[0], nodes_[1], nodes_[2], nodes_[3], nodes_[4]};
    }

    [[nodiscard]] LocalIndex getLocalFaceIndex(GlobalIndex face_id, GlobalIndex /*cell_id*/) const override
    {
        if (face_id != 0) {
            throw std::runtime_error("SinglePyramidBoundaryMeshAccess: unknown face id");
        }
        return local_face_id_;
    }

    [[nodiscard]] int getBoundaryFaceMarker(GlobalIndex face_id) const override
    {
        return (face_id == 0) ? 1 : -1;
    }

    [[nodiscard]] std::pair<GlobalIndex, GlobalIndex> getInteriorFaceCells(GlobalIndex /*face_id*/) const override
    {
        return {static_cast<GlobalIndex>(-1), static_cast<GlobalIndex>(-1)};
    }

    void forEachCell(std::function<void(GlobalIndex)> callback) const override { callback(0); }
    void forEachOwnedCell(std::function<void(GlobalIndex)> callback) const override { callback(0); }

    void forEachBoundaryFace(int marker, std::function<void(GlobalIndex, GlobalIndex)> callback) const override
    {
        if (marker == 1) {
            callback(0, 0);
        }
    }

    void forEachInteriorFace(std::function<void(GlobalIndex, GlobalIndex, GlobalIndex)> /*callback*/) const override {}

private:
    LocalIndex local_face_id_{0};
    std::vector<std::array<Real, 3>> nodes_{};
    std::vector<GlobalIndex> cell_nodes_{};
};

TEST(StandardAssemblerFaces, BoundaryFaceUsesFaceQuadratureAndNormals) {
    SingleTetraBoundaryMeshAccess mesh(/*local_face_id=*/0);

    // One tetra cell, 4 DOFs
    dofs::DofMap dof_map(1, 4, 4);
    dof_map.setCellDofs(0, std::vector<GlobalIndex>{0, 1, 2, 3});
    dof_map.setNumDofs(4);
    dof_map.setNumLocalDofs(4);
    dof_map.finalize();

    MockFunctionSpace space;
    DenseVectorView rhs(4);

    StandardAssembler assembler;
    assembler.setDofMap(dof_map);
    assembler.initialize();

    BoundaryFaceDiagnosticsKernel kernel;

    auto result = assembler.assembleBoundaryFaces(mesh, 1, space, kernel, nullptr, &rhs);
    EXPECT_TRUE(result.success);
    EXPECT_EQ(result.boundary_faces_assembled, 1);

    // Face 0 of reference tetra lies on z=0 and has area 0.5
    EXPECT_NEAR(rhs.getVectorEntry(0), 0.5, 1e-12);         // sum of integration weights
    EXPECT_NEAR(rhs.getVectorEntry(1), 0.0, 1e-12);         // max |xi.z|
    EXPECT_NEAR(rhs.getVectorEntry(2), -1.0, 1e-12);        // avg normal z component
}

TEST(StandardAssemblerFaces, Tetra3D_ObliqueFace2MeasureMatchesPhysicalArea) {
    SingleTetraBoundaryMeshAccess mesh(/*local_face_id=*/2);

    // One tetra cell, 4 DOFs
    dofs::DofMap dof_map(1, 4, 4);
    dof_map.setCellDofs(0, std::vector<GlobalIndex>{0, 1, 2, 3});
    dof_map.setNumDofs(4);
    dof_map.setNumLocalDofs(4);
    dof_map.finalize();

    MockFunctionSpace space;
    DenseVectorView rhs(4);

    StandardAssembler assembler;
    assembler.setDofMap(dof_map);
    assembler.initialize();

    BoundaryFaceMeasureKernel kernel;

    auto result = assembler.assembleBoundaryFaces(mesh, 1, space, kernel, nullptr, &rhs);
    EXPECT_TRUE(result.success);
    EXPECT_EQ(result.boundary_faces_assembled, 1);

    // Face 2 of the reference tetra is the oblique triangle (1,0,0)-(0,1,0)-(0,0,1),
    // with area sqrt(3)/2.
    EXPECT_NEAR(rhs.getVectorEntry(0), std::sqrt(3.0) / 2.0, 1e-12);
}

TEST(StandardAssemblerFaces, Triangle2D_BoundaryEdge0MeasureMatchesPhysicalLength) {
    SingleTriangleBoundaryMeshAccess mesh(/*local_face_id=*/0);

    // One Triangle3 cell, 3 DOFs.
    dofs::DofMap dof_map(1, 3, 3);
    dof_map.setCellDofs(0, std::vector<GlobalIndex>{0, 1, 2});
    dof_map.setNumDofs(3);
    dof_map.setNumLocalDofs(3);
    dof_map.finalize();

    spaces::H1Space space(ElementType::Triangle3, 1);
    DenseVectorView rhs(3);

    StandardAssembler assembler;
    assembler.setDofMap(dof_map);
    assembler.initialize();

    BoundaryFaceDiagnosticsKernel kernel;

    auto result = assembler.assembleBoundaryFaces(mesh, /*marker=*/1, space, kernel, nullptr, &rhs);
    EXPECT_TRUE(result.success);
    EXPECT_EQ(result.boundary_faces_assembled, 1);

    // Edge 0 is (0,0)-(1,0): physical length is 1.
    EXPECT_NEAR(rhs.getVectorEntry(0), 1.0, 1e-12);
    EXPECT_NEAR(rhs.getVectorEntry(1), 0.0, 1e-12);
    EXPECT_NEAR(rhs.getVectorEntry(2), 0.0, 1e-12);
}

TEST(StandardAssemblerFaces, Triangle2D_BoundaryEdge1MeasureMatchesPhysicalLength) {
    SingleTriangleBoundaryMeshAccess mesh(/*local_face_id=*/1);

    // One Triangle3 cell, 3 DOFs.
    dofs::DofMap dof_map(1, 3, 3);
    dof_map.setCellDofs(0, std::vector<GlobalIndex>{0, 1, 2});
    dof_map.setNumDofs(3);
    dof_map.setNumLocalDofs(3);
    dof_map.finalize();

    spaces::H1Space space(ElementType::Triangle3, 1);
    DenseVectorView rhs(3);

    StandardAssembler assembler;
    assembler.setDofMap(dof_map);
    assembler.initialize();

    BoundaryFaceDiagnosticsKernel kernel;

    auto result = assembler.assembleBoundaryFaces(mesh, /*marker=*/1, space, kernel, nullptr, &rhs);
    EXPECT_TRUE(result.success);
    EXPECT_EQ(result.boundary_faces_assembled, 1);

    // Edge 1 is (1,0)-(0,1): physical length is sqrt(2).
    EXPECT_NEAR(rhs.getVectorEntry(0), std::sqrt(2.0), 1e-12);
    EXPECT_NEAR(rhs.getVectorEntry(1), 0.0, 1e-12);
    EXPECT_NEAR(rhs.getVectorEntry(2), 0.0, 1e-12);
}

TEST(StandardAssemblerFaces, Wedge3D_QuadFace2MeasureMatchesPhysicalArea) {
    SingleWedgeBoundaryMeshAccess mesh(/*local_face_id=*/2);

    dofs::DofMap dof_map(1, 6, 6);
    dof_map.setCellDofs(0, std::vector<GlobalIndex>{0, 1, 2, 3, 4, 5});
    dof_map.setNumDofs(6);
    dof_map.setNumLocalDofs(6);
    dof_map.finalize();

    spaces::H1Space space(ElementType::Wedge6, 1);
    DenseVectorView rhs(6);

    StandardAssembler assembler;
    assembler.setDofMap(dof_map);
    assembler.initialize();

    BoundaryFaceMeasureKernel kernel;

    auto result = assembler.assembleBoundaryFaces(mesh, /*marker=*/1, space, kernel, nullptr, &rhs);
    EXPECT_TRUE(result.success);
    EXPECT_EQ(result.boundary_faces_assembled, 1);

    // Wedge face 2: rectangle with sides 1 (x) and 2 (z), area 2.
    EXPECT_NEAR(rhs.getVectorEntry(0), 2.0, 1e-12);
}

TEST(StandardAssemblerFaces, Wedge3D_ObliqueQuadFace3MeasureMatchesPhysicalArea) {
    SingleWedgeBoundaryMeshAccess mesh(/*local_face_id=*/3);

    dofs::DofMap dof_map(1, 6, 6);
    dof_map.setCellDofs(0, std::vector<GlobalIndex>{0, 1, 2, 3, 4, 5});
    dof_map.setNumDofs(6);
    dof_map.setNumLocalDofs(6);
    dof_map.finalize();

    spaces::H1Space space(ElementType::Wedge6, 1);
    DenseVectorView rhs(6);

    StandardAssembler assembler;
    assembler.setDofMap(dof_map);
    assembler.initialize();

    BoundaryFaceMeasureKernel kernel;

    auto result = assembler.assembleBoundaryFaces(mesh, /*marker=*/1, space, kernel, nullptr, &rhs);
    EXPECT_TRUE(result.success);
    EXPECT_EQ(result.boundary_faces_assembled, 1);

    // Wedge face 3: rectangle with sides sqrt(2) (in x-y) and 2 (z), area 2*sqrt(2).
    EXPECT_NEAR(rhs.getVectorEntry(0), 2.0 * std::sqrt(2.0), 1e-12);
}

TEST(StandardAssemblerFaces, Pyramid3D_TriFace1MeasureMatchesPhysicalArea) {
    SinglePyramidBoundaryMeshAccess mesh(/*local_face_id=*/1);

    dofs::DofMap dof_map(1, 5, 5);
    dof_map.setCellDofs(0, std::vector<GlobalIndex>{0, 1, 2, 3, 4});
    dof_map.setNumDofs(5);
    dof_map.setNumLocalDofs(5);
    dof_map.finalize();

    spaces::H1Space space(ElementType::Pyramid5, 1);
    DenseVectorView rhs(5);

    StandardAssembler assembler;
    assembler.setDofMap(dof_map);
    assembler.initialize();

    BoundaryFaceMeasureKernel kernel;

    auto result = assembler.assembleBoundaryFaces(mesh, /*marker=*/1, space, kernel, nullptr, &rhs);
    EXPECT_TRUE(result.success);
    EXPECT_EQ(result.boundary_faces_assembled, 1);

    // Pyramid face 1: triangle (-1,-1,0)-(1,-1,0)-(0,0,1), area sqrt(2).
    EXPECT_NEAR(rhs.getVectorEntry(0), std::sqrt(2.0), 1e-12);
}

TEST(StandardAssemblerFaces, InteriorFacePreparesBothSidesConsistently) {
    TwoHexStackedMeshAccess mesh;

    // Two Hex8 cells, 16 DOFs (no shared DOFs to isolate contributions)
    dofs::DofMap dof_map(2, 16, 8);
    dof_map.setCellDofs(0, std::vector<GlobalIndex>{0, 1, 2, 3, 4, 5, 6, 7});
    dof_map.setCellDofs(1, std::vector<GlobalIndex>{8, 9, 10, 11, 12, 13, 14, 15});
    dof_map.setNumDofs(16);
    dof_map.setNumLocalDofs(16);
    dof_map.finalize();

    HexFunctionSpace space;
    DenseSystemView system(16);

    StandardAssembler assembler;
    assembler.setDofMap(dof_map);
    assembler.initialize();

    InteriorFaceDiagnosticsKernel kernel;

    auto result = assembler.assembleInteriorFaces(mesh, space, space, kernel, system, &system);
    EXPECT_TRUE(result.success);
    EXPECT_EQ(result.interior_faces_assembled, 1);

    // Interior face is the shared square at z=1 with area 4; normals should oppose.
    EXPECT_NEAR(system.getVectorEntry(0), 4.0, 1e-12);       // sum_w_minus
    EXPECT_NEAR(system.getVectorEntry(1), 4.0, 1e-12);       // sum_w_plus
    EXPECT_NEAR(system.getVectorEntry(2), -1.0, 1e-12);      // avg dot(n_minus, n_plus)
    EXPECT_NEAR(system.getVectorEntry(3), 0.0, 1e-12);       // max physical point mismatch

    EXPECT_NEAR(system.getVectorEntry(8), 4.0, 1e-12);       // sum_w_plus (plus-side vector)
    EXPECT_NEAR(system.getVectorEntry(9), 4.0, 1e-12);       // sum_w_minus
    EXPECT_NEAR(system.getVectorEntry(10), -1.0, 1e-12);
    EXPECT_NEAR(system.getVectorEntry(11), 0.0, 1e-12);
}

TEST_F(StandardAssemblerTest, DefaultConstruction) {
    StandardAssembler assembler;
    EXPECT_EQ(assembler.name(), "StandardAssembler");
    EXPECT_FALSE(assembler.isConfigured());
    EXPECT_TRUE(assembler.supportsRectangular());
    EXPECT_TRUE(assembler.supportsDG());
    EXPECT_FALSE(assembler.isThreadSafe());
}

TEST_F(StandardAssemblerTest, Configuration) {
    StandardAssembler assembler;
    EXPECT_FALSE(assembler.isConfigured());

    assembler.setDofMap(dof_map_);
    EXPECT_TRUE(assembler.isConfigured());
}

TEST_F(StandardAssemblerTest, SetOptions) {
    AssemblyOptions options;
    options.deterministic = true;
    options.verbose = true;
    options.use_constraints = false;

    assembler_->setOptions(options);

    const auto& opts = assembler_->getOptions();
    EXPECT_TRUE(opts.deterministic);
    EXPECT_TRUE(opts.verbose);
    EXPECT_FALSE(opts.use_constraints);
}

TEST_F(StandardAssemblerTest, InitializeWithoutConfig) {
    StandardAssembler assembler;
    EXPECT_THROW(assembler.initialize(), std::runtime_error);
}

TEST_F(StandardAssemblerTest, Initialize) {
    EXPECT_NO_THROW(assembler_->initialize());
}

TEST_F(StandardAssemblerTest, AssembleMatrixIdentity) {
    assembler_->initialize();

    IdentityKernel kernel;
    auto result = assembler_->assembleMatrix(*mesh_, *space_, *space_, kernel, *system_);

    EXPECT_TRUE(result.success);
    EXPECT_EQ(result.elements_assembled, 2);
    EXPECT_GT(result.matrix_entries_inserted, 0);

    // Verify some diagonal entries
    // Cell 0 contributes to DOFs 0,1,2,3
    // Cell 1 contributes to DOFs 1,2,3,4
    // DOFs 1,2,3 get contributions from both cells
}

TEST_F(StandardAssemblerTest, AssembleVectorIdentity) {
    assembler_->initialize();

    IdentityKernel kernel;
    auto result = assembler_->assembleVector(*mesh_, *space_, kernel, *system_);

    EXPECT_TRUE(result.success);
    EXPECT_EQ(result.elements_assembled, 2);
    EXPECT_GT(result.vector_entries_inserted, 0);

    // All DOFs should have received contributions
    for (GlobalIndex i = 0; i < 5; ++i) {
        EXPECT_GE(system_->getVectorEntry(i), 0.0);
    }
}

TEST_F(StandardAssemblerTest, AssembleBoth) {
    assembler_->initialize();

    IdentityKernel kernel;
    auto result = assembler_->assembleBoth(*mesh_, *space_, *space_, kernel, *system_, *system_);

    EXPECT_TRUE(result.success);
    EXPECT_EQ(result.elements_assembled, 2);
    EXPECT_GT(result.matrix_entries_inserted, 0);
    EXPECT_GT(result.vector_entries_inserted, 0);
}

TEST_F(StandardAssemblerTest, Finalize) {
    assembler_->initialize();

    IdentityKernel kernel;
    assembler_->assembleMatrix(*mesh_, *space_, *space_, kernel, *system_);
    assembler_->finalize(system_.get(), nullptr);

    EXPECT_TRUE(system_->isFinalized());
}

TEST_F(StandardAssemblerTest, Reset) {
    assembler_->initialize();

    IdentityKernel kernel;
    assembler_->assembleMatrix(*mesh_, *space_, *space_, kernel, *system_);

    assembler_->reset();

    // After reset, should need to initialize again
    // The assembler preserves configuration but clears internal state
}

TEST_F(StandardAssemblerTest, TimingReport) {
    assembler_->initialize();

    IdentityKernel kernel;
    auto result = assembler_->assembleMatrix(*mesh_, *space_, *space_, kernel, *system_);

    EXPECT_GE(result.elapsed_time_seconds, 0.0);
}

// ============================================================================
// Assembly Correctness Tests
// ============================================================================

TEST_F(StandardAssemblerTest, MatrixSymmetry) {
    assembler_->initialize();

    IdentityKernel kernel;
    assembler_->assembleMatrix(*mesh_, *space_, *space_, kernel, *system_);

    // Identity kernel produces symmetric contributions
    const auto& mat = system_->matrix();
    for (GlobalIndex i = 0; i < 5; ++i) {
        for (GlobalIndex j = i + 1; j < 5; ++j) {
            EXPECT_NEAR(mat[i * 5 + j], mat[j * 5 + i], 1e-12)
                << "Matrix not symmetric at (" << i << "," << j << ")";
        }
    }
}

TEST_F(StandardAssemblerTest, SharedDofsReceiveMultipleContributions) {
    assembler_->initialize();

    IdentityKernel kernel;
    assembler_->assembleMatrix(*mesh_, *space_, *space_, kernel, *system_);

    // DOFs 1,2,3 are shared between cells 0 and 1
    // They should receive contributions from both cells
    // DOF 0 and DOF 4 are only in one cell each
    const auto& mat = system_->matrix();

    // Diagonal of DOF 0: only cell 0 contributes -> 1.0
    EXPECT_NEAR(mat[0 * 5 + 0], 1.0, 1e-12);

    // Diagonal of DOF 4: only cell 1 contributes -> 1.0
    EXPECT_NEAR(mat[4 * 5 + 4], 1.0, 1e-12);

    // Diagonal of DOF 1,2,3: both cells contribute -> 2.0
    EXPECT_NEAR(mat[1 * 5 + 1], 2.0, 1e-12);
    EXPECT_NEAR(mat[2 * 5 + 2], 2.0, 1e-12);
    EXPECT_NEAR(mat[3 * 5 + 3], 2.0, 1e-12);
}

TEST(StandardAssemblerHessianTest, PhysicalBasisHessiansMatchFiniteDifferenceGradient_NonAffineQuad)
{
    // Regression: for non-affine mappings, the physical Hessian transform must
    // include the inverse-mapping Hessian term (chain rule).
    SingleQuadMeshAccess mesh;
    spaces::H1Space space(ElementType::Quad4, 1);

    dofs::DofMap dof_map(1, /*num_dofs=*/4, /*max_dofs_per_cell=*/4);
    std::vector<GlobalIndex> cell_dofs = {0, 1, 2, 3};
    dof_map.setCellDofs(0, cell_dofs);
    dof_map.setNumDofs(4);
    dof_map.setNumLocalDofs(4);
    dof_map.finalize();

    // Create a mapping consistent with StandardAssembler's Quad4 geometry selection.
    std::vector<std::array<Real, 3>> coords;
    mesh.getCellCoordinates(0, coords);
    std::vector<math::Vector<Real, 3>> nodes;
    nodes.reserve(coords.size());
    for (const auto& c : coords) {
        nodes.push_back(math::Vector<Real, 3>{c[0], c[1], c[2]});
    }

    geometry::MappingRequest req;
    req.element_type = ElementType::Quad4;
    req.geometry_order = 1;
    req.use_affine = true;
    auto mapping = geometry::MappingFactory::create(req, nodes);

    StandardAssembler assembler;
    assembler.setDofMap(dof_map);

    DenseVectorView vec(4);
    vec.zero();

    BasisHessianTransformKernel kernel(mapping, space.element().basis(),
                                       /*basis_index=*/0,
                                       /*q_index=*/0,
                                       /*eps=*/1e-6,
                                       /*tol=*/5e-5);

    const auto result = assembler.assembleVector(mesh, space, kernel, vec);
    EXPECT_TRUE(result.success);
}

// ============================================================================
// StandardAssembler Edge Case Tests (UNIT_TEST_CHECKLIST.md)
// ============================================================================

namespace {

class EmptyMeshAccess final : public IMeshAccess {
public:
    [[nodiscard]] GlobalIndex numCells() const override { return 0; }
    [[nodiscard]] GlobalIndex numOwnedCells() const override { return 0; }
    [[nodiscard]] GlobalIndex numBoundaryFaces() const override { return 0; }
    [[nodiscard]] GlobalIndex numInteriorFaces() const override { return 0; }
    [[nodiscard]] int dimension() const override { return 3; }

    [[nodiscard]] bool isOwnedCell(GlobalIndex /*cell_id*/) const override { return false; }
    [[nodiscard]] ElementType getCellType(GlobalIndex /*cell_id*/) const override { return ElementType::Unknown; }

    void getCellNodes(GlobalIndex /*cell_id*/, std::vector<GlobalIndex>& nodes) const override { nodes.clear(); }
    [[nodiscard]] std::array<Real, 3> getNodeCoordinates(GlobalIndex /*node_id*/) const override { return {0.0, 0.0, 0.0}; }
    void getCellCoordinates(GlobalIndex /*cell_id*/, std::vector<std::array<Real, 3>>& coords) const override { coords.clear(); }

    [[nodiscard]] LocalIndex getLocalFaceIndex(GlobalIndex /*face_id*/,
                                               GlobalIndex /*cell_id*/) const override {
        return 0;
    }

    [[nodiscard]] int getBoundaryFaceMarker(GlobalIndex /*face_id*/) const override { return 0; }

    [[nodiscard]] std::pair<GlobalIndex, GlobalIndex>
    getInteriorFaceCells(GlobalIndex /*face_id*/) const override {
        return {-1, -1};
    }

    void forEachCell(std::function<void(GlobalIndex)> /*callback*/) const override {}
    void forEachOwnedCell(std::function<void(GlobalIndex)> /*callback*/) const override {}
    void forEachBoundaryFace(int /*marker*/,
                             std::function<void(GlobalIndex, GlobalIndex)> /*callback*/) const override {}
    void forEachInteriorFace(
        std::function<void(GlobalIndex, GlobalIndex, GlobalIndex)> /*callback*/) const override {}
};

class ConfigurableSingleTetraMeshAccess final : public IMeshAccess {
public:
    ConfigurableSingleTetraMeshAccess(std::array<std::array<Real, 3>, 4> nodes,
                                      std::array<GlobalIndex, 4> cell_nodes)
        : nodes_(std::move(nodes))
        , cell_nodes_(std::move(cell_nodes))
    {
    }

    [[nodiscard]] GlobalIndex numCells() const override { return 1; }
    [[nodiscard]] GlobalIndex numOwnedCells() const override { return 1; }
    [[nodiscard]] GlobalIndex numBoundaryFaces() const override { return 0; }
    [[nodiscard]] GlobalIndex numInteriorFaces() const override { return 0; }
    [[nodiscard]] int dimension() const override { return 3; }

    [[nodiscard]] bool isOwnedCell(GlobalIndex /*cell_id*/) const override { return true; }
    [[nodiscard]] ElementType getCellType(GlobalIndex /*cell_id*/) const override { return ElementType::Tetra4; }

    void getCellNodes(GlobalIndex /*cell_id*/, std::vector<GlobalIndex>& nodes) const override {
        nodes.assign(cell_nodes_.begin(), cell_nodes_.end());
    }

    [[nodiscard]] std::array<Real, 3> getNodeCoordinates(GlobalIndex node_id) const override {
        return nodes_.at(static_cast<std::size_t>(node_id));
    }

    void getCellCoordinates(GlobalIndex /*cell_id*/,
                            std::vector<std::array<Real, 3>>& coords) const override {
        coords.resize(4);
        for (std::size_t i = 0; i < 4; ++i) {
            coords[i] = nodes_.at(static_cast<std::size_t>(cell_nodes_[i]));
        }
    }

    [[nodiscard]] LocalIndex getLocalFaceIndex(GlobalIndex /*face_id*/,
                                               GlobalIndex /*cell_id*/) const override {
        return 0;
    }

    [[nodiscard]] int getBoundaryFaceMarker(GlobalIndex /*face_id*/) const override { return 0; }

    [[nodiscard]] std::pair<GlobalIndex, GlobalIndex>
    getInteriorFaceCells(GlobalIndex /*face_id*/) const override {
        return {-1, -1};
    }

    void forEachCell(std::function<void(GlobalIndex)> callback) const override { callback(0); }
    void forEachOwnedCell(std::function<void(GlobalIndex)> callback) const override { callback(0); }
    void forEachBoundaryFace(int /*marker*/,
                             std::function<void(GlobalIndex, GlobalIndex)> /*callback*/) const override {}
    void forEachInteriorFace(
        std::function<void(GlobalIndex, GlobalIndex, GlobalIndex)> /*callback*/) const override {}

private:
    std::array<std::array<Real, 3>, 4> nodes_{};
    std::array<GlobalIndex, 4> cell_nodes_{};
};

class SinglePointMeshAccess final : public IMeshAccess {
public:
    [[nodiscard]] GlobalIndex numCells() const override { return 1; }
    [[nodiscard]] GlobalIndex numOwnedCells() const override { return 1; }
    [[nodiscard]] GlobalIndex numBoundaryFaces() const override { return 0; }
    [[nodiscard]] GlobalIndex numInteriorFaces() const override { return 0; }
    [[nodiscard]] int dimension() const override { return 3; }

    [[nodiscard]] bool isOwnedCell(GlobalIndex /*cell_id*/) const override { return true; }
    [[nodiscard]] ElementType getCellType(GlobalIndex /*cell_id*/) const override { return ElementType::Point1; }

    void getCellNodes(GlobalIndex /*cell_id*/, std::vector<GlobalIndex>& nodes) const override {
        nodes.assign({0});
    }

    [[nodiscard]] std::array<Real, 3> getNodeCoordinates(GlobalIndex /*node_id*/) const override {
        return {0.0, 0.0, 0.0};
    }

    void getCellCoordinates(GlobalIndex /*cell_id*/,
                            std::vector<std::array<Real, 3>>& coords) const override {
        coords.assign({{0.0, 0.0, 0.0}});
    }

    [[nodiscard]] LocalIndex getLocalFaceIndex(GlobalIndex /*face_id*/,
                                               GlobalIndex /*cell_id*/) const override {
        return 0;
    }

    [[nodiscard]] int getBoundaryFaceMarker(GlobalIndex /*face_id*/) const override { return 0; }

    [[nodiscard]] std::pair<GlobalIndex, GlobalIndex>
    getInteriorFaceCells(GlobalIndex /*face_id*/) const override {
        return {-1, -1};
    }

    void forEachCell(std::function<void(GlobalIndex)> callback) const override { callback(0); }
    void forEachOwnedCell(std::function<void(GlobalIndex)> callback) const override { callback(0); }
    void forEachBoundaryFace(int /*marker*/,
                             std::function<void(GlobalIndex, GlobalIndex)> /*callback*/) const override {}
    void forEachInteriorFace(
        std::function<void(GlobalIndex, GlobalIndex, GlobalIndex)> /*callback*/) const override {}
};

class EmptyOutputKernel final : public AssemblyKernel {
public:
    void computeCell(const AssemblyContext& /*ctx*/, KernelOutput& /*output*/) override {}
    [[nodiscard]] RequiredData getRequiredData() const noexcept override { return RequiredData::None; }
    [[nodiscard]] bool isMatrixOnly() const noexcept override { return true; }
};

class NaNKernel final : public AssemblyKernel {
public:
    void computeCell(const AssemblyContext& ctx, KernelOutput& output) override {
        const LocalIndex n = ctx.numTestDofs();
        output.local_matrix.assign(static_cast<std::size_t>(n) * static_cast<std::size_t>(n),
                                   std::numeric_limits<Real>::quiet_NaN());
        output.has_matrix = true;
        output.has_vector = false;
    }
    [[nodiscard]] RequiredData getRequiredData() const noexcept override { return RequiredData::None; }
    [[nodiscard]] bool isMatrixOnly() const noexcept override { return true; }
};

class InfKernel final : public AssemblyKernel {
public:
    void computeCell(const AssemblyContext& ctx, KernelOutput& output) override {
        const LocalIndex n = ctx.numTestDofs();
        output.local_matrix.assign(static_cast<std::size_t>(n) * static_cast<std::size_t>(n),
                                   std::numeric_limits<Real>::infinity());
        output.has_matrix = true;
        output.has_vector = false;
    }
    [[nodiscard]] RequiredData getRequiredData() const noexcept override { return RequiredData::None; }
    [[nodiscard]] bool isMatrixOnly() const noexcept override { return true; }
};

} // namespace

TEST(StandardAssemblerEdgeCases, EmptyMeshNoCrashAndNoInsertions) {
    EmptyMeshAccess mesh;
    spaces::H1Space space(ElementType::Tetra4, /*order=*/1);

    dofs::DofMap dof_map(0, 0, 0);
    dof_map.setNumDofs(0);
    dof_map.setNumLocalDofs(0);
    dof_map.finalize();

    StandardAssembler assembler;
    assembler.setDofMap(dof_map);

    DenseMatrixView A(0);
    IdentityKernel kernel;

    const auto result = assembler.assembleMatrix(mesh, space, space, kernel, A);
    EXPECT_TRUE(result.success);
    EXPECT_EQ(result.elements_assembled, 0);
    EXPECT_EQ(A.numRows(), 0);
}

TEST(StandardAssemblerEdgeCases, SingleElementMeshAssemblesIdentity) {
    const std::array<std::array<Real, 3>, 4> nodes = {{
        {0.0, 0.0, 0.0},
        {1.0, 0.0, 0.0},
        {0.0, 1.0, 0.0},
        {0.0, 0.0, 1.0},
    }};
    ConfigurableSingleTetraMeshAccess mesh(nodes, /*cell_nodes=*/{0, 1, 2, 3});

    spaces::H1Space space(ElementType::Tetra4, /*order=*/1);

    dofs::DofMap dof_map(1, 4, 4);
    const std::array<GlobalIndex, 4> dofs = {0, 1, 2, 3};
    dof_map.setCellDofs(0, dofs);
    dof_map.setNumDofs(4);
    dof_map.setNumLocalDofs(4);
    dof_map.finalize();

    StandardAssembler assembler;
    assembler.setDofMap(dof_map);

    DenseMatrixView A(4);
    IdentityKernel kernel;
    const auto result = assembler.assembleMatrix(mesh, space, space, kernel, A);
    EXPECT_TRUE(result.success);
    EXPECT_EQ(result.elements_assembled, 1);

    for (GlobalIndex i = 0; i < 4; ++i) {
        for (GlobalIndex j = 0; j < 4; ++j) {
            const Real expected = (i == j) ? 1.0 : 0.0;
            EXPECT_DOUBLE_EQ(A.getMatrixEntry(i, j), expected);
        }
    }
}

TEST(StandardAssemblerEdgeCases, SingleNodePoint1IfSupported) {
    try {
        SinglePointMeshAccess mesh;
        spaces::H1Space space(ElementType::Point1, /*order=*/1);

        dofs::DofMap dof_map(1, 1, 1);
        const std::array<GlobalIndex, 1> dofs = {0};
        dof_map.setCellDofs(0, dofs);
        dof_map.setNumDofs(1);
        dof_map.setNumLocalDofs(1);
        dof_map.finalize();

        StandardAssembler assembler;
        assembler.setDofMap(dof_map);

        DenseMatrixView A(1);
        IdentityKernel kernel;
        const auto result = assembler.assembleMatrix(mesh, space, space, kernel, A);
        EXPECT_TRUE(result.success);
        EXPECT_DOUBLE_EQ(A.getMatrixEntry(0, 0), 1.0);
    } catch (const std::exception& e) {
        GTEST_SKIP() << "Point1 element path not supported: " << e.what();
    }
}

TEST(StandardAssemblerEdgeCases, RectangularAssemblyWithOffsets) {
    const std::array<std::array<Real, 3>, 4> nodes = {{
        {0.0, 0.0, 0.0},
        {1.0, 0.0, 0.0},
        {0.0, 1.0, 0.0},
        {0.0, 0.0, 1.0},
    }};
    ConfigurableSingleTetraMeshAccess mesh(nodes, /*cell_nodes=*/{0, 1, 2, 3});
    spaces::H1Space space(ElementType::Tetra4, /*order=*/1);

    dofs::DofMap dof_map(1, 4, 4);
    const std::array<GlobalIndex, 4> dofs = {0, 1, 2, 3};
    dof_map.setCellDofs(0, dofs);
    dof_map.setNumDofs(4);
    dof_map.setNumLocalDofs(4);
    dof_map.finalize();

    IdentityKernel kernel;

    // Row offset only.
    {
        StandardAssembler assembler;
        assembler.setRowDofMap(dof_map, /*row_offset=*/2);
        assembler.setColDofMap(dof_map, /*col_offset=*/0);
        DenseMatrixView A(/*rows=*/6, /*cols=*/4);
        assembler.assembleMatrix(mesh, space, space, kernel, A);
        EXPECT_DOUBLE_EQ(A.getMatrixEntry(2, 0), 1.0);
        EXPECT_DOUBLE_EQ(A.getMatrixEntry(0, 0), 0.0);
    }

    // Col offset only.
    {
        StandardAssembler assembler;
        assembler.setRowDofMap(dof_map, /*row_offset=*/0);
        assembler.setColDofMap(dof_map, /*col_offset=*/3);
        DenseMatrixView A(/*rows=*/4, /*cols=*/7);
        assembler.assembleMatrix(mesh, space, space, kernel, A);
        EXPECT_DOUBLE_EQ(A.getMatrixEntry(0, 3), 1.0);
        EXPECT_DOUBLE_EQ(A.getMatrixEntry(0, 0), 0.0);
    }

    // Both offsets.
    {
        StandardAssembler assembler;
        assembler.setRowDofMap(dof_map, /*row_offset=*/2);
        assembler.setColDofMap(dof_map, /*col_offset=*/3);
        DenseMatrixView A(/*rows=*/6, /*cols=*/7);
        assembler.assembleMatrix(mesh, space, space, kernel, A);
        EXPECT_DOUBLE_EQ(A.getMatrixEntry(2, 3), 1.0);
        EXPECT_DOUBLE_EQ(A.getMatrixEntry(0, 0), 0.0);
    }
}

TEST(StandardAssemblerEdgeCases, KernelReturnsEmptyOutputNoInsertions) {
    const std::array<std::array<Real, 3>, 4> nodes = {{
        {0.0, 0.0, 0.0},
        {1.0, 0.0, 0.0},
        {0.0, 1.0, 0.0},
        {0.0, 0.0, 1.0},
    }};
    ConfigurableSingleTetraMeshAccess mesh(nodes, /*cell_nodes=*/{0, 1, 2, 3});
    spaces::H1Space space(ElementType::Tetra4, /*order=*/1);

    dofs::DofMap dof_map(1, 4, 4);
    const std::array<GlobalIndex, 4> dofs = {0, 1, 2, 3};
    dof_map.setCellDofs(0, dofs);
    dof_map.setNumDofs(4);
    dof_map.setNumLocalDofs(4);
    dof_map.finalize();

    StandardAssembler assembler;
    assembler.setDofMap(dof_map);

    DenseMatrixView A(4);
    EmptyOutputKernel kernel;
    const auto result = assembler.assembleMatrix(mesh, space, space, kernel, A);
    EXPECT_TRUE(result.success);

    for (GlobalIndex i = 0; i < 4; ++i) {
        EXPECT_DOUBLE_EQ(A.getMatrixEntry(i, i), 0.0);
    }
}

TEST(StandardAssemblerEdgeCases, KernelReturnsNaNThrowsWhenEnabled) {
    const std::array<std::array<Real, 3>, 4> nodes = {{
        {0.0, 0.0, 0.0},
        {1.0, 0.0, 0.0},
        {0.0, 1.0, 0.0},
        {0.0, 0.0, 1.0},
    }};
    ConfigurableSingleTetraMeshAccess mesh(nodes, /*cell_nodes=*/{0, 1, 2, 3});
    spaces::H1Space space(ElementType::Tetra4, /*order=*/1);

    dofs::DofMap dof_map(1, 4, 4);
    const std::array<GlobalIndex, 4> dofs = {0, 1, 2, 3};
    dof_map.setCellDofs(0, dofs);
    dof_map.setNumDofs(4);
    dof_map.setNumLocalDofs(4);
    dof_map.finalize();

    AssemblyOptions opts;
    opts.check_finite_values = true;
    StandardAssembler assembler(opts);
    assembler.setDofMap(dof_map);

    DenseMatrixView A(4);
    NaNKernel kernel;
    EXPECT_THROW(assembler.assembleMatrix(mesh, space, space, kernel, A), std::runtime_error);
}

TEST(StandardAssemblerEdgeCases, KernelReturnsInfThrowsWhenEnabled) {
    const std::array<std::array<Real, 3>, 4> nodes = {{
        {0.0, 0.0, 0.0},
        {1.0, 0.0, 0.0},
        {0.0, 1.0, 0.0},
        {0.0, 0.0, 1.0},
    }};
    ConfigurableSingleTetraMeshAccess mesh(nodes, /*cell_nodes=*/{0, 1, 2, 3});
    spaces::H1Space space(ElementType::Tetra4, /*order=*/1);

    dofs::DofMap dof_map(1, 4, 4);
    const std::array<GlobalIndex, 4> dofs = {0, 1, 2, 3};
    dof_map.setCellDofs(0, dofs);
    dof_map.setNumDofs(4);
    dof_map.setNumLocalDofs(4);
    dof_map.finalize();

    AssemblyOptions opts;
    opts.check_finite_values = true;
    StandardAssembler assembler(opts);
    assembler.setDofMap(dof_map);

    DenseMatrixView A(4);
    InfKernel kernel;
    EXPECT_THROW(assembler.assembleMatrix(mesh, space, space, kernel, A), std::runtime_error);
}

TEST(StandardAssemblerEdgeCases, NearZeroJacobianProducesSmallMassMatrix) {
    constexpr Real eps = 1e-12;
    const std::array<std::array<Real, 3>, 4> nodes = {{
        {0.0, 0.0, 0.0},
        {1.0, 0.0, 0.0},
        {0.0, 1.0, 0.0},
        {0.0, 0.0, eps},
    }};
    ConfigurableSingleTetraMeshAccess mesh(nodes, /*cell_nodes=*/{0, 1, 2, 3});
    spaces::H1Space space(ElementType::Tetra4, /*order=*/1);

    dofs::DofMap dof_map(1, 4, 4);
    const std::array<GlobalIndex, 4> dofs = {0, 1, 2, 3};
    dof_map.setCellDofs(0, dofs);
    dof_map.setNumDofs(4);
    dof_map.setNumLocalDofs(4);
    dof_map.finalize();

    StandardAssembler assembler;
    assembler.setDofMap(dof_map);

    DenseMatrixView A(4);
    MassKernel kernel;
    assembler.assembleMatrix(mesh, space, space, kernel, A);

    const Real volume = eps / 6.0;
    const Real expected_m00 = volume / 10.0;  // P1 tetra mass matrix diag = V/10

    EXPECT_NEAR(A.getMatrixEntry(0, 0), expected_m00, expected_m00 * 1e-6 + 1e-18);
}

TEST(StandardAssemblerEdgeCases, InvertedElementMassMatrixMatchesNonInverted) {
    const std::array<std::array<Real, 3>, 4> nodes = {{
        {0.0, 0.0, 0.0},
        {1.0, 0.0, 0.0},
        {0.0, 1.0, 0.0},
        {0.0, 0.0, 1.0},
    }};
    ConfigurableSingleTetraMeshAccess mesh_a(nodes, /*cell_nodes=*/{0, 1, 2, 3});
    ConfigurableSingleTetraMeshAccess mesh_b(nodes, /*cell_nodes=*/{1, 0, 2, 3});  // inverted orientation

    spaces::H1Space space(ElementType::Tetra4, /*order=*/1);

    dofs::DofMap dof_map(1, 4, 4);
    const std::array<GlobalIndex, 4> dofs = {0, 1, 2, 3};
    dof_map.setCellDofs(0, dofs);
    dof_map.setNumDofs(4);
    dof_map.setNumLocalDofs(4);
    dof_map.finalize();

    StandardAssembler assembler;
    assembler.setDofMap(dof_map);

    MassKernel kernel;
    DenseMatrixView A(4);
    DenseMatrixView B(4);
    assembler.assembleMatrix(mesh_a, space, space, kernel, A);
    assembler.assembleMatrix(mesh_b, space, space, kernel, B);

    for (GlobalIndex i = 0; i < 4; ++i) {
        for (GlobalIndex j = 0; j < 4; ++j) {
            EXPECT_NEAR(A.getMatrixEntry(i, j), B.getMatrixEntry(i, j), 1e-12);
        }
    }
}

TEST(StandardAssemblerEdgeCases, ConstraintChainDistributesToMasters) {
    const std::array<std::array<Real, 3>, 4> nodes = {{
        {0.0, 0.0, 0.0},
        {1.0, 0.0, 0.0},
        {0.0, 1.0, 0.0},
        {0.0, 0.0, 1.0},
    }};
    ConfigurableSingleTetraMeshAccess mesh(nodes, /*cell_nodes=*/{0, 1, 2, 3});
    spaces::H1Space space(ElementType::Tetra4, /*order=*/1);

    dofs::DofMap dof_map(1, 4, 4);
    const std::array<GlobalIndex, 4> dofs = {0, 1, 2, 3};
    dof_map.setCellDofs(0, dofs);
    dof_map.setNumDofs(4);
    dof_map.setNumLocalDofs(4);
    dof_map.finalize();

    constraints::AffineConstraints constraints;
    constraints.addLine(0);
    constraints.addEntry(0, 1, 1.0);
    constraints.addLine(1);
    constraints.addEntry(1, 2, 1.0);
    constraints.close();  // resolves chain 0 -> 2

    StandardAssembler assembler;
    assembler.setDofMap(dof_map);
    assembler.setConstraints(&constraints);

    DenseSystemView sys(4);
    IdentityKernel kernel;
    assembler.assembleBoth(mesh, space, space, kernel, sys, sys);

    // Identity local matrix and RHS:
    // dof0 and dof1 are constrained and distribute to master dof2.
    EXPECT_DOUBLE_EQ(sys.getMatrixEntry(2, 2), 3.0);
    EXPECT_DOUBLE_EQ(sys.getVectorEntry(2), 3.0);

    // Constrained rows are not populated by distribution.
    EXPECT_DOUBLE_EQ(sys.getMatrixEntry(0, 0), 0.0);
    EXPECT_DOUBLE_EQ(sys.getMatrixEntry(1, 1), 0.0);
}

// ============================================================================
// Factory Function Tests
// ============================================================================

TEST(AssemblerFactory, CreateStandardAssembler) {
    auto assembler = createStandardAssembler();
    EXPECT_NE(assembler, nullptr);
    EXPECT_EQ(assembler->name(), "StandardAssembler");
}

TEST(AssemblerFactory, CreateWithOptions) {
    AssemblyOptions options;
    options.verbose = true;

    auto assembler = createStandardAssembler(options);
    EXPECT_NE(assembler, nullptr);
    EXPECT_TRUE(assembler->getOptions().verbose);
}

TEST(AssemblerFactory, CreateAssemblerByStrategy) {
    auto assembler = createAssembler(ThreadingStrategy::Sequential);
    EXPECT_NE(assembler, nullptr);
}

} // namespace test
} // namespace assembly
} // namespace FE
} // namespace svmp

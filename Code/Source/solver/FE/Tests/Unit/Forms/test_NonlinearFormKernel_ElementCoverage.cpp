/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

/**
 * @file test_NonlinearFormKernel_ElementCoverage.cpp
 * @brief Expand Jacobian FD verification across element types and polynomial orders.
 */

#include <gtest/gtest.h>

#include "Assembly/GlobalSystemView.h"
#include "Assembly/StandardAssembler.h"
#include "Basis/NodeOrderingConventions.h"
#include "Basis/SerendipityBasis.h"
#include "Dofs/DofMap.h"
#include "Elements/ReferenceElement.h"
#include "Forms/FormCompiler.h"
#include "Forms/FormKernels.h"
#include "Forms/Vocabulary.h"
#include "Quadrature/QuadratureFactory.h"
#include "Spaces/H1Space.h"
#include "Spaces/IsogeometricSpace.h"
#include "Spaces/SpaceFactory.h"
#include "Tests/Unit/Forms/FormsTestHelpers.h"

#include <array>
#include <vector>

namespace svmp {
namespace FE {
namespace forms {
namespace test {

namespace {

[[nodiscard]] dofs::DofMap makeSingleCellDofMap(GlobalIndex n_dofs)
{
    dofs::DofMap dof_map(1, n_dofs, static_cast<LocalIndex>(n_dofs));
    std::vector<GlobalIndex> cell_dofs(static_cast<std::size_t>(n_dofs));
    for (GlobalIndex i = 0; i < n_dofs; ++i) {
        cell_dofs[static_cast<std::size_t>(i)] = i;
    }
    dof_map.setCellDofs(0, cell_dofs);
    dof_map.setNumDofs(n_dofs);
    dof_map.setNumLocalDofs(n_dofs);
    dof_map.finalize();
    return dof_map;
}

class SingleQuadMeshAccess final : public assembly::IMeshAccess {
public:
    SingleQuadMeshAccess()
    {
        nodes_ = {
            {-1.0, -1.0, 0.0}, // 0
            {1.0, -1.0, 0.0},  // 1
            {1.0, 1.0, 0.0},   // 2
            {-1.0, 1.0, 0.0}   // 3
        };
        cell_ = {0, 1, 2, 3};
    }

    [[nodiscard]] GlobalIndex numCells() const override { return 1; }
    [[nodiscard]] GlobalIndex numOwnedCells() const override { return 1; }
    [[nodiscard]] GlobalIndex numBoundaryFaces() const override { return 0; }
    [[nodiscard]] GlobalIndex numInteriorFaces() const override { return 0; }
    [[nodiscard]] int dimension() const override { return 2; }

    [[nodiscard]] bool isOwnedCell(GlobalIndex /*cell_id*/) const override { return true; }
    [[nodiscard]] ElementType getCellType(GlobalIndex /*cell_id*/) const override { return ElementType::Quad4; }

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

    [[nodiscard]] std::pair<GlobalIndex, GlobalIndex> getInteriorFaceCells(GlobalIndex /*face_id*/) const override
    {
        return {0, 0};
    }

    void forEachCell(std::function<void(GlobalIndex)> callback) const override { callback(0); }
    void forEachOwnedCell(std::function<void(GlobalIndex)> callback) const override { callback(0); }

    void forEachBoundaryFace(int /*marker*/,
                             std::function<void(GlobalIndex, GlobalIndex)> /*callback*/) const override
    {
    }

    void forEachInteriorFace(std::function<void(GlobalIndex, GlobalIndex, GlobalIndex)> /*callback*/) const override
    {
    }

private:
    std::vector<std::array<Real, 3>> nodes_;
    std::array<GlobalIndex, 4> cell_{};
};

class SingleHexMeshAccess final : public assembly::IMeshAccess {
public:
    SingleHexMeshAccess()
    {
        nodes_ = {
            {-1.0, -1.0, -1.0}, // 0
            {1.0, -1.0, -1.0},  // 1
            {1.0, 1.0, -1.0},   // 2
            {-1.0, 1.0, -1.0},  // 3
            {-1.0, -1.0, 1.0},  // 4
            {1.0, -1.0, 1.0},   // 5
            {1.0, 1.0, 1.0},    // 6
            {-1.0, 1.0, 1.0}    // 7
        };
        cell_ = {0, 1, 2, 3, 4, 5, 6, 7};
    }

    [[nodiscard]] GlobalIndex numCells() const override { return 1; }
    [[nodiscard]] GlobalIndex numOwnedCells() const override { return 1; }
    [[nodiscard]] GlobalIndex numBoundaryFaces() const override { return 0; }
    [[nodiscard]] GlobalIndex numInteriorFaces() const override { return 0; }
    [[nodiscard]] int dimension() const override { return 3; }

    [[nodiscard]] bool isOwnedCell(GlobalIndex /*cell_id*/) const override { return true; }
    [[nodiscard]] ElementType getCellType(GlobalIndex /*cell_id*/) const override { return ElementType::Hex8; }

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

    [[nodiscard]] std::pair<GlobalIndex, GlobalIndex> getInteriorFaceCells(GlobalIndex /*face_id*/) const override
    {
        return {0, 0};
    }

    void forEachCell(std::function<void(GlobalIndex)> callback) const override { callback(0); }
    void forEachOwnedCell(std::function<void(GlobalIndex)> callback) const override { callback(0); }

    void forEachBoundaryFace(int /*marker*/,
                             std::function<void(GlobalIndex, GlobalIndex)> /*callback*/) const override
    {
    }

    void forEachInteriorFace(std::function<void(GlobalIndex, GlobalIndex, GlobalIndex)> /*callback*/) const override
    {
    }

private:
    std::vector<std::array<Real, 3>> nodes_;
    std::array<GlobalIndex, 8> cell_{};
};

class SingleWedgeMeshAccess final : public assembly::IMeshAccess {
public:
    SingleWedgeMeshAccess()
    {
        nodes_ = {
            {0.0, 0.0, -1.0}, // 0
            {1.0, 0.0, -1.0}, // 1
            {0.0, 1.0, -1.0}, // 2
            {0.0, 0.0, 1.0},  // 3
            {1.0, 0.0, 1.0},  // 4
            {0.0, 1.0, 1.0}   // 5
        };
        cell_ = {0, 1, 2, 3, 4, 5};
    }

    [[nodiscard]] GlobalIndex numCells() const override { return 1; }
    [[nodiscard]] GlobalIndex numOwnedCells() const override { return 1; }
    [[nodiscard]] GlobalIndex numBoundaryFaces() const override { return 0; }
    [[nodiscard]] GlobalIndex numInteriorFaces() const override { return 0; }
    [[nodiscard]] int dimension() const override { return 3; }

    [[nodiscard]] bool isOwnedCell(GlobalIndex /*cell_id*/) const override { return true; }
    [[nodiscard]] ElementType getCellType(GlobalIndex /*cell_id*/) const override { return ElementType::Wedge6; }

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

    [[nodiscard]] std::pair<GlobalIndex, GlobalIndex> getInteriorFaceCells(GlobalIndex /*face_id*/) const override
    {
        return {0, 0};
    }

    void forEachCell(std::function<void(GlobalIndex)> callback) const override { callback(0); }
    void forEachOwnedCell(std::function<void(GlobalIndex)> callback) const override { callback(0); }

    void forEachBoundaryFace(int /*marker*/,
                             std::function<void(GlobalIndex, GlobalIndex)> /*callback*/) const override
    {
    }

    void forEachInteriorFace(std::function<void(GlobalIndex, GlobalIndex, GlobalIndex)> /*callback*/) const override
    {
    }

private:
    std::vector<std::array<Real, 3>> nodes_;
    std::array<GlobalIndex, 6> cell_{};
};

class SinglePyramidMeshAccess final : public assembly::IMeshAccess {
public:
    SinglePyramidMeshAccess()
    {
        nodes_ = {
            {-1.0, -1.0, 0.0}, // 0
            {1.0, -1.0, 0.0},  // 1
            {1.0, 1.0, 0.0},   // 2
            {-1.0, 1.0, 0.0},  // 3
            {0.0, 0.0, 1.0}    // 4
        };
        cell_ = {0, 1, 2, 3, 4};
    }

    [[nodiscard]] GlobalIndex numCells() const override { return 1; }
    [[nodiscard]] GlobalIndex numOwnedCells() const override { return 1; }
    [[nodiscard]] GlobalIndex numBoundaryFaces() const override { return 0; }
    [[nodiscard]] GlobalIndex numInteriorFaces() const override { return 0; }
    [[nodiscard]] int dimension() const override { return 3; }

    [[nodiscard]] bool isOwnedCell(GlobalIndex /*cell_id*/) const override { return true; }
    [[nodiscard]] ElementType getCellType(GlobalIndex /*cell_id*/) const override { return ElementType::Pyramid5; }

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

    [[nodiscard]] std::pair<GlobalIndex, GlobalIndex> getInteriorFaceCells(GlobalIndex /*face_id*/) const override
    {
        return {0, 0};
    }

    void forEachCell(std::function<void(GlobalIndex)> callback) const override { callback(0); }
    void forEachOwnedCell(std::function<void(GlobalIndex)> callback) const override { callback(0); }

    void forEachBoundaryFace(int /*marker*/,
                             std::function<void(GlobalIndex, GlobalIndex)> /*callback*/) const override
    {
    }

    void forEachInteriorFace(std::function<void(GlobalIndex, GlobalIndex, GlobalIndex)> /*callback*/) const override
    {
    }

private:
    std::vector<std::array<Real, 3>> nodes_;
    std::array<GlobalIndex, 5> cell_{};
};

class CanonicalSingleElementMeshAccess final : public assembly::IMeshAccess {
public:
    explicit CanonicalSingleElementMeshAccess(ElementType elem_type)
        : elem_type_(elem_type)
    {
        const auto ref = elements::ReferenceElement::create(elem_type_);
        dim_ = ref.dimension();

        const auto n_nodes = basis::NodeOrdering::num_nodes(elem_type_);
        nodes_.reserve(n_nodes);
        cell_.reserve(n_nodes);

        for (std::size_t i = 0; i < n_nodes; ++i) {
            const auto xi = basis::NodeOrdering::get_node_coords(elem_type_, i);
            nodes_.push_back({xi[0], xi[1], xi[2]});
            cell_.push_back(static_cast<GlobalIndex>(i));
        }
    }

    [[nodiscard]] GlobalIndex numCells() const override { return 1; }
    [[nodiscard]] GlobalIndex numOwnedCells() const override { return 1; }
    [[nodiscard]] GlobalIndex numBoundaryFaces() const override { return 0; }
    [[nodiscard]] GlobalIndex numInteriorFaces() const override { return 0; }
    [[nodiscard]] int dimension() const override { return dim_; }

    [[nodiscard]] bool isOwnedCell(GlobalIndex /*cell_id*/) const override { return true; }
    [[nodiscard]] ElementType getCellType(GlobalIndex /*cell_id*/) const override { return elem_type_; }

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

    [[nodiscard]] std::pair<GlobalIndex, GlobalIndex> getInteriorFaceCells(GlobalIndex /*face_id*/) const override
    {
        return {0, 0};
    }

    void forEachCell(std::function<void(GlobalIndex)> callback) const override { callback(0); }
    void forEachOwnedCell(std::function<void(GlobalIndex)> callback) const override { callback(0); }

    void forEachBoundaryFace(int /*marker*/,
                             std::function<void(GlobalIndex, GlobalIndex)> /*callback*/) const override
    {
    }

    void forEachInteriorFace(std::function<void(GlobalIndex, GlobalIndex, GlobalIndex)> /*callback*/) const override
    {
    }

private:
    ElementType elem_type_{ElementType::Unknown};
    int dim_{-1};
    std::vector<std::array<Real, 3>> nodes_;
    std::vector<GlobalIndex> cell_;
};

void expectJacobianMatchesCentralDifferences(const assembly::IMeshAccess& mesh,
                                             const spaces::FunctionSpace& space,
                                             const FormExpr& residual,
                                             const std::vector<Real>& U,
                                             Real eps,
                                             Real tol,
                                             ADMode ad_mode = ADMode::Forward)
{
    const auto n_dofs = static_cast<GlobalIndex>(space.dofs_per_element());
    ASSERT_EQ(static_cast<GlobalIndex>(U.size()), n_dofs);

    FormCompiler compiler;
    auto ir = compiler.compileResidual(residual);
    NonlinearFormKernel kernel(std::move(ir), ad_mode);

    auto dof_map = makeSingleCellDofMap(n_dofs);
    assembly::StandardAssembler assembler;
    assembler.setDofMap(dof_map);
    assembler.setCurrentSolution(U);

    assembly::DenseMatrixView J(n_dofs);
    assembly::DenseVectorView R(n_dofs);
    J.zero();
    R.zero();
    (void)assembler.assembleBoth(mesh, space, space, kernel, J, R);

    for (GlobalIndex j = 0; j < n_dofs; ++j) {
        auto U_plus = U;
        auto U_minus = U;
        U_plus[static_cast<std::size_t>(j)] += eps;
        U_minus[static_cast<std::size_t>(j)] -= eps;

        assembler.setCurrentSolution(U_plus);
        assembly::DenseVectorView Rp(n_dofs);
        Rp.zero();
        (void)assembler.assembleVector(mesh, space, kernel, Rp);

        assembler.setCurrentSolution(U_minus);
        assembly::DenseVectorView Rm(n_dofs);
        Rm.zero();
        (void)assembler.assembleVector(mesh, space, kernel, Rm);

        for (GlobalIndex i = 0; i < n_dofs; ++i) {
            const Real fd = (Rp.getVectorEntry(i) - Rm.getVectorEntry(i)) / (2.0 * eps);
            EXPECT_NEAR(J.getMatrixEntry(i, j), fd, tol);
        }
    }
}

} // namespace

TEST(NonlinearFormKernelElementCoverageTest, PoissonJacobian_Quad4_P1_MatchesCentralDifferences)
{
    SingleQuadMeshAccess mesh;
    spaces::H1Space space(ElementType::Quad4, /*order=*/1);

    const auto u = FormExpr::trialFunction(space, "u");
    const auto v = FormExpr::testFunction(space, "v");
    const auto residual = inner(grad(u), grad(v)).dx();

    std::vector<Real> U(static_cast<std::size_t>(space.dofs_per_element()));
    for (std::size_t i = 0; i < U.size(); ++i) {
        U[i] = Real(0.03) * static_cast<Real>(i + 1);
    }

    expectJacobianMatchesCentralDifferences(mesh, space, residual, U, /*eps=*/1e-6, /*tol=*/1e-10);
}

TEST(NonlinearFormKernelElementCoverageTest, PoissonJacobian_Hex8_Q1_MatchesCentralDifferences)
{
    SingleHexMeshAccess mesh;
    spaces::H1Space space(ElementType::Hex8, /*order=*/1);

    const auto u = FormExpr::trialFunction(space, "u");
    const auto v = FormExpr::testFunction(space, "v");
    const auto residual = inner(grad(u), grad(v)).dx();

    std::vector<Real> U(static_cast<std::size_t>(space.dofs_per_element()));
    for (std::size_t i = 0; i < U.size(); ++i) {
        U[i] = Real(-0.02) * static_cast<Real>(i + 1);
    }

    expectJacobianMatchesCentralDifferences(mesh, space, residual, U, /*eps=*/1e-6, /*tol=*/1e-10);
}

TEST(NonlinearFormKernelElementCoverageTest, PoissonJacobian_Wedge6_P1_MatchesCentralDifferences)
{
    SingleWedgeMeshAccess mesh;
    spaces::H1Space space(ElementType::Wedge6, /*order=*/1);

    const auto u = FormExpr::trialFunction(space, "u");
    const auto v = FormExpr::testFunction(space, "v");
    const auto residual = inner(grad(u), grad(v)).dx();

    std::vector<Real> U(static_cast<std::size_t>(space.dofs_per_element()));
    for (std::size_t i = 0; i < U.size(); ++i) {
        U[i] = Real(0.01) * static_cast<Real>(i + 1);
    }

    expectJacobianMatchesCentralDifferences(mesh, space, residual, U, /*eps=*/1e-6, /*tol=*/1e-10);
}

TEST(NonlinearFormKernelElementCoverageTest, PoissonJacobian_Pyramid5_P1_MatchesCentralDifferences)
{
    SinglePyramidMeshAccess mesh;
    spaces::H1Space space(ElementType::Pyramid5, /*order=*/1);

    const auto u = FormExpr::trialFunction(space, "u");
    const auto v = FormExpr::testFunction(space, "v");
    const auto residual = inner(grad(u), grad(v)).dx();

    std::vector<Real> U(static_cast<std::size_t>(space.dofs_per_element()));
    for (std::size_t i = 0; i < U.size(); ++i) {
        U[i] = Real(-0.015) * static_cast<Real>(i + 1);
    }

    expectJacobianMatchesCentralDifferences(mesh, space, residual, U, /*eps=*/1e-6, /*tol=*/1e-10);
}

TEST(NonlinearFormKernelElementCoverageTest, NonlinearDiffusion_Tetra4_P2_MatchesCentralDifferences)
{
    SingleTetraMeshAccess mesh;
    spaces::H1Space space(ElementType::Tetra4, /*order=*/2);

    const auto u = FormExpr::trialFunction(space, "u");
    const auto v = FormExpr::testFunction(space, "v");
    const auto residual =
        inner((FormExpr::constant(Real(1.0)) + u * u) * grad(u), grad(v)).dx();

    std::vector<Real> U(static_cast<std::size_t>(space.dofs_per_element()));
    for (std::size_t i = 0; i < U.size(); ++i) {
        U[i] = Real(0.02) * static_cast<Real>(i + 1);
    }

    expectJacobianMatchesCentralDifferences(mesh, space, residual, U, /*eps=*/2e-6, /*tol=*/5e-8);
}

TEST(NonlinearFormKernelElementCoverageTest, NonlinearDiffusion_Hex8_Q2_MatchesCentralDifferences)
{
    SingleHexMeshAccess mesh;
    spaces::H1Space space(ElementType::Hex8, /*order=*/2);

    const auto u = FormExpr::trialFunction(space, "u");
    const auto v = FormExpr::testFunction(space, "v");
    const auto residual =
        inner((FormExpr::constant(Real(1.0)) + u * u) * grad(u), grad(v)).dx();

    std::vector<Real> U(static_cast<std::size_t>(space.dofs_per_element()));
    for (std::size_t i = 0; i < U.size(); ++i) {
        U[i] = Real(-0.005) * static_cast<Real>(i + 1);
    }

    expectJacobianMatchesCentralDifferences(mesh, space, residual, U, /*eps=*/2e-6, /*tol=*/5e-8);
}

TEST(NonlinearFormKernelElementCoverageTest, PoissonJacobian_Triangle3_P2_MatchesCentralDifferences)
{
    SingleTriangleMeshAccess mesh;
    spaces::H1Space space(ElementType::Triangle3, /*order=*/2);

    const auto u = FormExpr::trialFunction(space, "u");
    const auto v = FormExpr::testFunction(space, "v");
    const auto residual = inner(grad(u), grad(v)).dx();

    std::vector<Real> U(static_cast<std::size_t>(space.dofs_per_element()));
    for (std::size_t i = 0; i < U.size(); ++i) {
        U[i] = Real(0.02) * static_cast<Real>(i + 1);
    }

    expectJacobianMatchesCentralDifferences(mesh, space, residual, U, /*eps=*/1e-6, /*tol=*/1e-10);
}

TEST(NonlinearFormKernelElementCoverageTest, PoissonJacobian_Quad4_Q2_MatchesCentralDifferences)
{
    SingleQuadMeshAccess mesh;
    spaces::H1Space space(ElementType::Quad4, /*order=*/2);

    const auto u = FormExpr::trialFunction(space, "u");
    const auto v = FormExpr::testFunction(space, "v");
    const auto residual = inner(grad(u), grad(v)).dx();

    std::vector<Real> U(static_cast<std::size_t>(space.dofs_per_element()));
    for (std::size_t i = 0; i < U.size(); ++i) {
        U[i] = Real(-0.01) * static_cast<Real>(i + 1);
    }

    expectJacobianMatchesCentralDifferences(mesh, space, residual, U, /*eps=*/1e-6, /*tol=*/1e-10);
}

TEST(NonlinearFormKernelElementCoverageTest, NonlinearDiffusion_Wedge6_P2_MatchesCentralDifferences)
{
    SingleWedgeMeshAccess mesh;
    spaces::H1Space space(ElementType::Wedge6, /*order=*/2);

    const auto u = FormExpr::trialFunction(space, "u");
    const auto v = FormExpr::testFunction(space, "v");
    const auto residual =
        inner((FormExpr::constant(Real(1.0)) + u * u) * grad(u), grad(v)).dx();

    std::vector<Real> U(static_cast<std::size_t>(space.dofs_per_element()));
    for (std::size_t i = 0; i < U.size(); ++i) {
        U[i] = Real(0.01) * static_cast<Real>(i + 1);
    }

    expectJacobianMatchesCentralDifferences(mesh, space, residual, U, /*eps=*/2e-6, /*tol=*/5e-8);
}

TEST(NonlinearFormKernelElementCoverageTest, NonlinearDiffusion_Pyramid14_P2_MatchesCentralDifferences)
{
    SinglePyramidMeshAccess mesh;
    spaces::H1Space space(ElementType::Pyramid14, /*order=*/2);

    const auto u = FormExpr::trialFunction(space, "u");
    const auto v = FormExpr::testFunction(space, "v");
    const auto residual =
        inner((FormExpr::constant(Real(1.0)) + u * u) * grad(u), grad(v)).dx();

    std::vector<Real> U(static_cast<std::size_t>(space.dofs_per_element()));
    for (std::size_t i = 0; i < U.size(); ++i) {
        U[i] = Real(-0.008) * static_cast<Real>(i + 1);
    }

    expectJacobianMatchesCentralDifferences(mesh, space, residual, U, /*eps=*/2e-6, /*tol=*/5e-8);
}

TEST(NonlinearFormKernelElementCoverageTest, Convection_Triangle3_P2_Vector_MatchesCentralDifferences)
{
    SingleTriangleMeshAccess mesh;
    auto vec_space = spaces::VectorSpace(spaces::SpaceType::H1,
                                         ElementType::Triangle3,
                                         /*order=*/2,
                                         /*components=*/2);
    ASSERT_TRUE(vec_space);

    const auto u = FormExpr::trialFunction(*vec_space, "u");
    const auto v = FormExpr::testFunction(*vec_space, "v");

    // Nonlinear vector convection: ∫ (∇u · u) · v dx
    const auto residual = inner(grad(u) * u, v).dx();

    const auto n_dofs = static_cast<GlobalIndex>(vec_space->dofs_per_element());
    std::vector<Real> U(static_cast<std::size_t>(n_dofs), 0.0);
    for (GlobalIndex i = 0; i < n_dofs; ++i) {
        U[static_cast<std::size_t>(i)] = Real(0.03) * static_cast<Real>(i + 1);
    }

    expectJacobianMatchesCentralDifferences(mesh, *vec_space, residual, U, /*eps=*/1e-6, /*tol=*/5e-8);
}

TEST(NonlinearFormKernelElementCoverageTest, PoissonJacobian_Hex20_MatchesCentralDifferences)
{
    CanonicalSingleElementMeshAccess mesh(ElementType::Hex20);
    auto basis = std::make_shared<basis::SerendipityBasis>(ElementType::Hex20, /*order=*/2, /*geometry_mode=*/false);
    const int qord = quadrature::QuadratureFactory::recommended_order(/*basis_order=*/2, /*is_mass_matrix=*/false);
    auto quad = quadrature::QuadratureFactory::create(ElementType::Hex20, qord);
    spaces::IsogeometricSpace space(std::move(basis), std::move(quad));

    const auto u = FormExpr::trialFunction(space, "u");
    const auto v = FormExpr::testFunction(space, "v");
    const auto residual = inner(grad(u), grad(v)).dx();

    std::vector<Real> U(static_cast<std::size_t>(space.dofs_per_element()));
    for (std::size_t i = 0; i < U.size(); ++i) {
        U[i] = Real(0.01) * static_cast<Real>(i + 1);
    }

    expectJacobianMatchesCentralDifferences(mesh, space, residual, U, /*eps=*/1e-6, /*tol=*/5e-10);
}

TEST(NonlinearFormKernelElementCoverageTest, PoissonJacobian_Quad8_MatchesCentralDifferences)
{
    CanonicalSingleElementMeshAccess mesh(ElementType::Quad8);
    auto basis = std::make_shared<basis::SerendipityBasis>(ElementType::Quad8, /*order=*/2, /*geometry_mode=*/false);
    const int qord = quadrature::QuadratureFactory::recommended_order(/*basis_order=*/2, /*is_mass_matrix=*/false);
    auto quad = quadrature::QuadratureFactory::create(ElementType::Quad8, qord);
    spaces::IsogeometricSpace space(std::move(basis), std::move(quad));

    const auto u = FormExpr::trialFunction(space, "u");
    const auto v = FormExpr::testFunction(space, "v");
    const auto residual = inner(grad(u), grad(v)).dx();

    std::vector<Real> U(static_cast<std::size_t>(space.dofs_per_element()));
    for (std::size_t i = 0; i < U.size(); ++i) {
        U[i] = Real(-0.02) * static_cast<Real>(i + 1);
    }

    expectJacobianMatchesCentralDifferences(mesh, space, residual, U, /*eps=*/1e-6, /*tol=*/1e-10);
}

TEST(NonlinearFormKernelElementCoverageTest, PoissonJacobian_Quad9_MatchesCentralDifferences)
{
    CanonicalSingleElementMeshAccess mesh(ElementType::Quad9);
    spaces::H1Space space(ElementType::Quad9, /*order=*/2);

    const auto u = FormExpr::trialFunction(space, "u");
    const auto v = FormExpr::testFunction(space, "v");
    const auto residual = inner(grad(u), grad(v)).dx();

    std::vector<Real> U(static_cast<std::size_t>(space.dofs_per_element()));
    for (std::size_t i = 0; i < U.size(); ++i) {
        U[i] = Real(0.015) * static_cast<Real>(i + 1);
    }

    expectJacobianMatchesCentralDifferences(mesh, space, residual, U, /*eps=*/1e-6, /*tol=*/1e-10);
}

TEST(NonlinearFormKernelElementCoverageTest, PoissonJacobian_Triangle6_MatchesCentralDifferences)
{
    CanonicalSingleElementMeshAccess mesh(ElementType::Triangle6);
    spaces::H1Space space(ElementType::Triangle6, /*order=*/2);

    const auto u = FormExpr::trialFunction(space, "u");
    const auto v = FormExpr::testFunction(space, "v");
    const auto residual = inner(grad(u), grad(v)).dx();

    std::vector<Real> U(static_cast<std::size_t>(space.dofs_per_element()));
    for (std::size_t i = 0; i < U.size(); ++i) {
        U[i] = Real(-0.03) * static_cast<Real>(i + 1);
    }

    expectJacobianMatchesCentralDifferences(mesh, space, residual, U, /*eps=*/1e-6, /*tol=*/1e-10);
}

TEST(NonlinearFormKernelElementCoverageTest, PoissonJacobian_Wedge15_MatchesCentralDifferences)
{
    CanonicalSingleElementMeshAccess mesh(ElementType::Wedge15);
    auto basis = std::make_shared<basis::SerendipityBasis>(ElementType::Wedge15, /*order=*/2, /*geometry_mode=*/false);
    const int qord = quadrature::QuadratureFactory::recommended_order(/*basis_order=*/2, /*is_mass_matrix=*/false);
    auto quad = quadrature::QuadratureFactory::create(ElementType::Wedge15, qord);
    spaces::IsogeometricSpace space(std::move(basis), std::move(quad));

    const auto u = FormExpr::trialFunction(space, "u");
    const auto v = FormExpr::testFunction(space, "v");
    const auto residual = inner(grad(u), grad(v)).dx();

    std::vector<Real> U(static_cast<std::size_t>(space.dofs_per_element()));
    for (std::size_t i = 0; i < U.size(); ++i) {
        U[i] = Real(0.008) * static_cast<Real>(i + 1);
    }

    expectJacobianMatchesCentralDifferences(mesh, space, residual, U, /*eps=*/1e-6, /*tol=*/1e-10);
}

} // namespace test
} // namespace forms
} // namespace FE
} // namespace svmp

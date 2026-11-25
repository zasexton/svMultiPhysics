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

#include "Assembly/FunctionalAssembler.h"
#include "Assembly/GlobalSystemView.h"
#include "Assembly/StandardAssembler.h"
#include "Basis/NodeOrderingConventions.h"
#include "Basis/SerendipityBasis.h"
#include "Dofs/DofMap.h"
#include "Forms/CellMetric.h"
#include "Forms/FormCompiler.h"
#include "Forms/FormKernels.h"
#include "Forms/Vocabulary.h"
#include "Quadrature/QuadratureFactory.h"
#include "Spaces/H1Space.h"
#include "Spaces/GenericBasisSpace.h"
#include "Spaces/SpaceFactory.h"
#include "Tests/Unit/Forms/FormsTestHelpers.h"
#include "Tests/Unit/Forms/JITTestHelpers.h"

#include <array>
#include <cmath>
#include <initializer_list>
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
    explicit CanonicalSingleElementMeshAccess(ElementType elem_type,
        const math::Matrix<Real, 3, 3>& map = math::Matrix<Real, 3, 3>::identity(), int ambient_dim = 0)
        : elem_type_(elem_type), dim_(ambient_dim > 0 ? ambient_dim : element_dimension(elem_type))
    {
        const auto n_nodes = basis::ReferenceNodeLayout::num_nodes(elem_type_);
        nodes_.reserve(n_nodes);
        cell_.reserve(n_nodes);

        for (std::size_t i = 0; i < n_nodes; ++i) {
            const auto xi = basis::ReferenceNodeLayout::get_node_coords(elem_type_, i);
            const auto x = map * xi;
            nodes_.push_back({x[0], x[1], x[2]});
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

using Matrix = math::Matrix<Real, 3, 3>;

void expectCellMetric(ElementType linear, ElementType chart, const Matrix& map, int ambient_dim,
                      const FormExpr& metric, const Matrix& expected,
                      std::initializer_list<std::array<Real, 3>> normals = {})
{
    SCOPED_TRACE(::testing::Message() << "ElementType " << static_cast<int>(chart) << ", dimension " << ambient_dim);
    CanonicalSingleElementMeshAccess mesh(chart, map, ambient_dim);
    spaces::H1Space space(linear, 1);
    auto dofs = makeSingleCellDofMap(space.dofs_per_element());
    assembly::FunctionalAssembler assembler;
    assembler.setMesh(mesh);
    assembler.setSpace(space);
    assembler.setDofMap(dofs);
    const auto integrate = [&](const FormExpr& expr) {
        auto kernel = makeFunctionalFormKernel(expr, FunctionalFormKernel::Domain::Cell);
        return assembler.assembleScalar(*kernel);
    };
    const double measure = integrate(FormExpr::constant(1.0));
    ASSERT_GT(measure, 0.0);
    for (int i = 0; i < ambient_dim; ++i) {
        for (int j = 0; j < ambient_dim; ++j) {
            EXPECT_NEAR(integrate(metric.component(i, j)) / measure, expected(i, j), 1.0e-12)
                << "entry (" << i << ", " << j << ")";
        }
    }
    for (const auto& normal : normals) {
        const auto n = constantVector(normal, ambient_dim);
        EXPECT_NEAR(integrate(inner(n, metric * n)) / measure, 0.0, 1e-12);
    }
}

} // namespace

TEST(CellMetric, PhysicalComponentsOnLinearAndQuadraticReferenceCells)
{
    // Map xi to (2 xi_x + xi_y, xi_y, xi_z / 2) in the active dimensions.
    const Matrix map{{2, 1, 0}, {0, 1, 0}, {0, 0, 0.5}};
    const auto check = [&](ElementType linear, std::initializer_list<ElementType> charts,
                           const Matrix& expected) {
        for (const auto chart : charts) {
            expectCellMetric(linear, chart, map, element_dimension(chart), cellMetric(chart), expected);
        }
    };

    using enum ElementType;
    const Matrix cartesian{{0.25, -0.25, 0}, {-0.25, 1.25, 0}, {0, 0, 4}};
    check(Line2, {Line2, Line3}, cartesian);
    check(Quad4, {Quad4, Quad8, Quad9}, cartesian);
    check(Hex8, {Hex8, Hex20, Hex27}, cartesian);
    check(Pyramid5, {Pyramid5, Pyramid13, Pyramid14}, cartesian);

    // The shear aligns the triangular reference metric with the physical axes.
    // The extruded triangle keeps the independent axial scale of the map.
    const double tri_scale = 1.0 / std::sqrt(3.0);
    const Matrix prism{{0.5 * tri_scale, 0, 0}, {0, 1.5 * tri_scale, 0}, {0, 0, 4}};
    check(Triangle3, {Triangle3, Triangle6}, prism);
    check(Wedge6, {Wedge6, Wedge15, Wedge18}, prism);
    const double tet_scale = 1.0 / std::cbrt(4.0);
    const Matrix tetra{{0.5 * tet_scale, 0, tet_scale}, {0, 1.5 * tet_scale, tet_scale},
                       {tet_scale, tet_scale, 8 * tet_scale}};
    check(Tetra4, {Tetra4, Tetra10}, tetra);
}

TEST(CellMetric, EmbeddedCellsUseOnlyTangentialInverseLengths)
{
    using enum ElementType;
    // The line tangent is (1,2) or (1,2,3). Its dual is t / (t.t).
    const Matrix plane{{1, 0, 0}, {2, 1, 0}, {0, 0, 0}};
    Matrix line2{{1, 2, 0}, {2, 4, 0}, {0, 0, 0}};
    line2 /= 25.0;
    expectCellMetric(Line2, Line3, plane, 2, cellMetric(Line3, 2), line2, {{2, -1, 0}});
    const Matrix space{{1, 0, 0}, {2, 1, 0}, {3, 2, 0}};
    Matrix line3{{1, 2, 3}, {2, 4, 6}, {3, 6, 9}};
    line3 /= 196.0;
    expectCellMetric(Line2, Line2, space, 3, cellMetric(Line2, 3), line3, {{2, -1, 0}, {3, 0, -1}});

    // For tangents (1,2,3) and (0,1,2), the duals within the plane are
    // (5,2,-1)/6 and (-8,-2,4)/6. The quad metric sums their outer products.
    Matrix quad{{89, 26, -37}, {26, 8, -10}, {-37, -10, 17}};
    quad /= 36.0;
    expectCellMetric(Quad4, Quad4, space, 3, cellMetric(Quad4, 3), quad, {{1, -2, 1}});
    // The third barycentric gradient is (3,0,-3)/6. The triangle uses all three.
    Matrix triangle{{98, 26, -46}, {26, 8, -10}, {-46, -10, 26}};
    triangle /= 36.0 * std::sqrt(3.0);
    expectCellMetric(Triangle3, Triangle6, space, 3, cellMetric(Triangle6, 3), triangle, {{1, -2, 1}});
}

TEST(CellMetric, RequiresAValidIntrinsicAndAmbientDimension)
{
    using enum ElementType;
    EXPECT_THROW((void)cellMetric(Triangle3, 1), std::invalid_argument);
    EXPECT_THROW((void)cellMetric(Line2, 4), std::invalid_argument);
    EXPECT_THROW((void)cellMetric(Point1, 3), std::invalid_argument);
    EXPECT_THROW((void)cellMetric(Unknown), std::invalid_argument);
}

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
    spaces::GenericBasisSpace space(std::move(basis), std::move(quad));

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
    spaces::GenericBasisSpace space(std::move(basis), std::move(quad));

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
    spaces::GenericBasisSpace space(std::move(basis), std::move(quad));

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

/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

/**
 * @file test_ManufacturedSolutionConvergence_ElementTypes.cpp
 * @brief End-to-end FE verification on non-Quad4 element types: assemble + constrain + solve + convergence rate.
 *
 * Motivation: the existing manufactured-solution convergence tests are Quad4-only.
 * This file adds convergence checks on simplex and 3D elements using small structured meshes.
 */

#include <gtest/gtest.h>

#include "Assembly/Assembler.h"
#include "Assembly/GlobalSystemView.h"
#include "Assembly/StandardAssembler.h"
#include "Basis/LagrangeBasis.h"
#include "Constraints/AffineConstraints.h"
#include "Core/FEException.h"
#include "Dofs/DofHandler.h"
#include "Forms/FormCompiler.h"
#include "Forms/FormKernels.h"
#include "Forms/Vocabulary.h"
#include "Geometry/GeometryMapping.h"
#include "Geometry/MappingFactory.h"
#include "Math/Vector.h"
#include "Quadrature/QuadratureFactory.h"
#include "Spaces/H1Space.h"
#include "Spaces/ProductSpace.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <limits>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace svmp::FE::assembly::testing {
namespace {

[[nodiscard]] bool near(Real a, Real b, Real tol = Real(1e-12))
{
    return std::abs(a - b) <= tol;
}

struct DenseSolveResult {
    bool ok{false};
    std::string message{};
    std::vector<Real> x{};
};

[[nodiscard]] DenseSolveResult solveDenseWithPartialPivoting(DenseMatrixView A,
                                                            std::vector<Real> b)
{
    const GlobalIndex n = A.numRows();
    if (A.numCols() != n) {
        return {false, "solveDenseWithPartialPivoting: expected square matrix", {}};
    }
    if (static_cast<GlobalIndex>(b.size()) != n) {
        return {false, "solveDenseWithPartialPivoting: RHS size mismatch", {}};
    }

    auto data = std::vector<Real>(A.data().begin(), A.data().end());
    auto idx = [n](GlobalIndex i, GlobalIndex j) { return static_cast<std::size_t>(i * n + j); };

    // Gaussian elimination with partial pivoting.
    for (GlobalIndex k = 0; k < n; ++k) {
        GlobalIndex pivot = k;
        Real pivot_val = std::abs(data[idx(k, k)]);
        for (GlobalIndex i = k + 1; i < n; ++i) {
            const Real v = std::abs(data[idx(i, k)]);
            if (v > pivot_val) {
                pivot = i;
                pivot_val = v;
            }
        }
        if (pivot_val == 0.0) {
            return {false, "solveDenseWithPartialPivoting: singular matrix", {}};
        }

        if (pivot != k) {
            for (GlobalIndex j = k; j < n; ++j) {
                std::swap(data[idx(k, j)], data[idx(pivot, j)]);
            }
            std::swap(b[static_cast<std::size_t>(k)], b[static_cast<std::size_t>(pivot)]);
        }

        const Real akk = data[idx(k, k)];
        for (GlobalIndex i = k + 1; i < n; ++i) {
            const Real factor = data[idx(i, k)] / akk;
            data[idx(i, k)] = 0.0;
            for (GlobalIndex j = k + 1; j < n; ++j) {
                data[idx(i, j)] -= factor * data[idx(k, j)];
            }
            b[static_cast<std::size_t>(i)] -= factor * b[static_cast<std::size_t>(k)];
        }
    }

    // Back-substitution.
    std::vector<Real> x(static_cast<std::size_t>(n), 0.0);
    for (GlobalIndex ii = 0; ii < n; ++ii) {
        const GlobalIndex i = n - 1 - ii;
        Real s = b[static_cast<std::size_t>(i)];
        for (GlobalIndex j = i + 1; j < n; ++j) {
            s -= data[idx(i, j)] * x[static_cast<std::size_t>(j)];
        }
        x[static_cast<std::size_t>(i)] = s / data[idx(i, i)];
    }

    return {true, {}, std::move(x)};
}

[[nodiscard]] std::shared_ptr<geometry::GeometryMapping> buildAffineMappingForCell(
    const IMeshAccess& mesh,
    GlobalIndex cell_id)
{
    const ElementType cell_type = mesh.getCellType(cell_id);

    std::vector<std::array<Real, 3>> coords;
    mesh.getCellCoordinates(cell_id, coords);
    FE_THROW_IF(coords.empty(), InvalidArgumentException,
                "buildAffineMappingForCell: expected non-empty cell coordinates");

    std::vector<math::Vector<Real, 3>> node_coords(coords.size());
    for (std::size_t i = 0; i < coords.size(); ++i) {
        node_coords[i] = math::Vector<Real, 3>{coords[i][0], coords[i][1], coords[i][2]};
    }

    geometry::MappingRequest req;
    req.element_type = cell_type;
    req.geometry_order = 1;
    req.use_affine = true;
    return geometry::MappingFactory::create(req, node_coords);
}

[[nodiscard]] std::vector<std::array<Real, 3>> computeDofCoordinates(const IMeshAccess& mesh,
                                                                     const spaces::FunctionSpace& space,
                                                                     const dofs::DofMap& dof_map)
{
    const GlobalIndex n_dofs = dof_map.getNumDofs();
    std::vector<std::array<Real, 3>> coords(static_cast<std::size_t>(n_dofs),
                                            {std::numeric_limits<Real>::quiet_NaN(),
                                             std::numeric_limits<Real>::quiet_NaN(),
                                             std::numeric_limits<Real>::quiet_NaN()});
    std::vector<char> has_coord(static_cast<std::size_t>(n_dofs), 0);

    const ElementType cell_type = mesh.getCellType(/*cell_id=*/0);
    const auto& element = space.getElement(cell_type, /*cell_id=*/0);
    const auto& basis = element.basis();
    const auto* lag = dynamic_cast<const basis::LagrangeBasis*>(&basis);
    FE_CHECK_NOT_NULL(lag, "computeDofCoordinates: expected LagrangeBasis for H1/ProductSpace");
    const auto& ref_nodes = lag->nodes();

    const GlobalIndex n_cells = dof_map.getNumCells();
    for (GlobalIndex c = 0; c < n_cells; ++c) {
        auto mapping = buildAffineMappingForCell(mesh, c);
        FE_CHECK_NOT_NULL(mapping.get(), "computeDofCoordinates: mapping");

        const auto cell_dofs = dof_map.getCellDofs(c);
        FE_CHECK_ARG(ref_nodes.size() == cell_dofs.size(), "computeDofCoordinates: basis node count mismatch");

        for (LocalIndex li = 0; li < static_cast<LocalIndex>(cell_dofs.size()); ++li) {
            const GlobalIndex gd = cell_dofs[static_cast<std::size_t>(li)];
            if (gd < 0 || gd >= n_dofs) continue;
            if (has_coord[static_cast<std::size_t>(gd)]) continue;

            const auto& Xi = ref_nodes[static_cast<std::size_t>(li)];
            const math::Vector<Real, 3> xi{Xi[0], Xi[1], Xi[2]};
            const auto x = mapping->map_to_physical(xi);
            coords[static_cast<std::size_t>(gd)] = {x[0], x[1], x[2]};
            has_coord[static_cast<std::size_t>(gd)] = 1;
        }
    }

    for (GlobalIndex i = 0; i < n_dofs; ++i) {
        FE_THROW_IF(has_coord[static_cast<std::size_t>(i)] == 0, FEException,
                    "computeDofCoordinates: missing coordinate for DOF");
    }

    return coords;
}

struct DofCoordinatesAndComponents {
    std::vector<std::array<Real, 3>> coords;
    std::vector<int> component;
};

[[nodiscard]] DofCoordinatesAndComponents computeDofCoordinatesProductSpace(
    const IMeshAccess& mesh,
    const spaces::FunctionSpace& space,
    const dofs::DofMap& dof_map)
{
    FE_CHECK_ARG(space.space_type() == spaces::SpaceType::Product,
                 "computeDofCoordinatesProductSpace: expected ProductSpace");

    const GlobalIndex n_dofs = dof_map.getNumDofs();
    std::vector<std::array<Real, 3>> coords(static_cast<std::size_t>(n_dofs),
                                            {std::numeric_limits<Real>::quiet_NaN(),
                                             std::numeric_limits<Real>::quiet_NaN(),
                                             std::numeric_limits<Real>::quiet_NaN()});
    std::vector<int> component(static_cast<std::size_t>(n_dofs), -1);
    std::vector<char> has_coord(static_cast<std::size_t>(n_dofs), 0);

    const int n_components = space.value_dimension();
    FE_THROW_IF(n_components <= 0, InvalidArgumentException,
                "computeDofCoordinatesProductSpace: value_dimension must be positive");

    const ElementType cell_type = mesh.getCellType(/*cell_id=*/0);
    const auto& element = space.getElement(cell_type, /*cell_id=*/0);
    const auto& basis = element.basis();
    const auto* lag = dynamic_cast<const basis::LagrangeBasis*>(&basis);
    FE_CHECK_NOT_NULL(lag, "computeDofCoordinatesProductSpace: expected LagrangeBasis");
    const auto& ref_nodes = lag->nodes();

    const std::size_t dofs_per_comp = space.dofs_per_element() / static_cast<std::size_t>(n_components);
    FE_CHECK_ARG(dofs_per_comp * static_cast<std::size_t>(n_components) == space.dofs_per_element(),
                 "computeDofCoordinatesProductSpace: dofs_per_element not divisible by components");
    FE_CHECK_ARG(dofs_per_comp == ref_nodes.size(),
                 "computeDofCoordinatesProductSpace: scalar DOFs per component mismatch");

    const GlobalIndex n_cells = dof_map.getNumCells();
    for (GlobalIndex c = 0; c < n_cells; ++c) {
        auto mapping = buildAffineMappingForCell(mesh, c);
        FE_CHECK_NOT_NULL(mapping.get(), "computeDofCoordinatesProductSpace: mapping");

        const auto cell_dofs = dof_map.getCellDofs(c);
        FE_CHECK_ARG(cell_dofs.size() == ref_nodes.size() * static_cast<std::size_t>(n_components),
                     "computeDofCoordinatesProductSpace: basis node count mismatch");

        for (LocalIndex li = 0; li < static_cast<LocalIndex>(cell_dofs.size()); ++li) {
            const GlobalIndex gd = cell_dofs[static_cast<std::size_t>(li)];
            if (gd < 0 || gd >= n_dofs) continue;

            const std::size_t comp = static_cast<std::size_t>(li) / dofs_per_comp;
            const std::size_t scalar_li = static_cast<std::size_t>(li) % dofs_per_comp;

            if (!has_coord[static_cast<std::size_t>(gd)]) {
                const auto& Xi = ref_nodes[scalar_li];
                const math::Vector<Real, 3> xi{Xi[0], Xi[1], Xi[2]};
                const auto x = mapping->map_to_physical(xi);
                coords[static_cast<std::size_t>(gd)] = {x[0], x[1], x[2]};
                has_coord[static_cast<std::size_t>(gd)] = 1;
            }

            if (component[static_cast<std::size_t>(gd)] == -1) {
                component[static_cast<std::size_t>(gd)] = static_cast<int>(comp);
            } else {
                FE_THROW_IF(component[static_cast<std::size_t>(gd)] != static_cast<int>(comp), FEException,
                            "computeDofCoordinatesProductSpace: inconsistent component index for DOF");
            }
        }
    }

    for (GlobalIndex i = 0; i < n_dofs; ++i) {
        FE_THROW_IF(has_coord[static_cast<std::size_t>(i)] == 0, FEException,
                    "computeDofCoordinatesProductSpace: missing coordinate for DOF");
        FE_THROW_IF(component[static_cast<std::size_t>(i)] < 0, FEException,
                    "computeDofCoordinatesProductSpace: missing component for DOF");
    }

    return {std::move(coords), std::move(component)};
}

[[nodiscard]] constraints::AffineConstraints makeDirichletOnBoundaryUnitSquare(
    const std::vector<std::array<Real, 3>>& dof_coords,
    const std::function<Real(Real, Real)>& value,
    Real tol = Real(1e-12))
{
    constraints::AffineConstraints c;
    for (GlobalIndex dof = 0; dof < static_cast<GlobalIndex>(dof_coords.size()); ++dof) {
        const auto& x = dof_coords[static_cast<std::size_t>(dof)];
        if (near(x[0], Real(0.0), tol) || near(x[0], Real(1.0), tol) ||
            near(x[1], Real(0.0), tol) || near(x[1], Real(1.0), tol)) {
            c.addDirichlet(dof, value(x[0], x[1]));
        }
    }
    c.close();
    return c;
}

[[nodiscard]] constraints::AffineConstraints makeDirichletOnBoundaryUnitCube(
    const std::vector<std::array<Real, 3>>& dof_coords,
    const std::function<Real(Real, Real, Real)>& value,
    Real tol = Real(1e-12))
{
    constraints::AffineConstraints c;
    for (GlobalIndex dof = 0; dof < static_cast<GlobalIndex>(dof_coords.size()); ++dof) {
        const auto& x = dof_coords[static_cast<std::size_t>(dof)];
        if (near(x[0], Real(0.0), tol) || near(x[0], Real(1.0), tol) ||
            near(x[1], Real(0.0), tol) || near(x[1], Real(1.0), tol) ||
            near(x[2], Real(0.0), tol) || near(x[2], Real(1.0), tol)) {
            c.addDirichlet(dof, value(x[0], x[1], x[2]));
        }
    }
    c.close();
    return c;
}

void addDirichletOnBoundaryVector(const DofCoordinatesAndComponents& dof_info,
                                 const std::function<std::array<Real, 2>(Real, Real)>& value,
                                 constraints::AffineConstraints& c,
                                 Real tol = Real(1e-12))
{
    FE_CHECK_ARG(dof_info.coords.size() == dof_info.component.size(),
                 "addDirichletOnBoundaryVector: size mismatch");

    for (GlobalIndex dof = 0; dof < static_cast<GlobalIndex>(dof_info.coords.size()); ++dof) {
        const auto& x = dof_info.coords[static_cast<std::size_t>(dof)];
        if (!(near(x[0], Real(0.0), tol) || near(x[0], Real(1.0), tol) ||
              near(x[1], Real(0.0), tol) || near(x[1], Real(1.0), tol))) {
            continue;
        }

        const auto u = value(x[0], x[1]);
        const int comp = dof_info.component[static_cast<std::size_t>(dof)];
        FE_THROW_IF(comp < 0 || comp >= 2, InvalidArgumentException,
                    "addDirichletOnBoundaryVector: component out of range");

        c.addDirichlet(dof, u[static_cast<std::size_t>(comp)]);
    }
}

[[nodiscard]] Real computeL2ErrorScalar(const IMeshAccess& mesh,
                                       const spaces::FunctionSpace& space,
                                       const dofs::DofMap& dof_map,
                                       std::span<const Real> u_h,
                                       const std::function<Real(std::array<Real, 3>)>& u_exact,
                                       int quad_order)
{
    FE_CHECK_ARG(space.space_type() == spaces::SpaceType::H1, "computeL2ErrorScalar: expected H1 space");

    const ElementType cell_type = mesh.getCellType(/*cell_id=*/0);
    auto quad = quadrature::QuadratureFactory::create(cell_type, quad_order);
    FE_CHECK_NOT_NULL(quad.get(), "computeL2ErrorScalar: quadrature");

    const auto& element = space.getElement(cell_type, /*cell_id=*/0);
    const auto& basis = element.basis();

    std::vector<Real> phi;

    Real sum = 0.0;
    const GlobalIndex n_cells = dof_map.getNumCells();
    for (GlobalIndex c = 0; c < n_cells; ++c) {
        auto mapping = buildAffineMappingForCell(mesh, c);
        FE_CHECK_NOT_NULL(mapping.get(), "computeL2ErrorScalar: mapping");

        const auto cell_dofs = dof_map.getCellDofs(c);
        for (LocalIndex q = 0; q < static_cast<LocalIndex>(quad->num_points()); ++q) {
            const auto& qp = quad->point(static_cast<std::size_t>(q));
            const math::Vector<Real, 3> xi{qp[0], qp[1], qp[2]};

            basis.evaluate_values(xi, phi);
            FE_CHECK_ARG(phi.size() == cell_dofs.size(), "computeL2ErrorScalar: basis value size mismatch");

            Real uh = 0.0;
            for (std::size_t i = 0; i < cell_dofs.size(); ++i) {
                const auto gd = cell_dofs[i];
                uh += u_h[static_cast<std::size_t>(gd)] * phi[i];
            }

            const auto x = mapping->map_to_physical(xi);
            const Real ue = u_exact({x[0], x[1], x[2]});
            const Real e = uh - ue;

            const Real detJ = mapping->jacobian_determinant(xi);
            sum += e * e * quad->weight(static_cast<std::size_t>(q)) * std::abs(detJ);
        }
    }

    return std::sqrt(sum);
}

[[nodiscard]] Real computeL2ErrorVector2(const IMeshAccess& mesh,
                                        const spaces::FunctionSpace& space,
                                        const dofs::DofMap& dof_map,
                                        std::span<const Real> u_h,
                                        const std::function<std::array<Real, 2>(Real, Real)>& u_exact,
                                        int quad_order)
{
    FE_CHECK_ARG(space.space_type() == spaces::SpaceType::Product, "computeL2ErrorVector2: expected ProductSpace");
    FE_CHECK_ARG(space.value_dimension() == 2, "computeL2ErrorVector2: expected 2D vector field");

    const ElementType cell_type = mesh.getCellType(/*cell_id=*/0);
    auto quad = quadrature::QuadratureFactory::create(cell_type, quad_order);
    FE_CHECK_NOT_NULL(quad.get(), "computeL2ErrorVector2: quadrature");

    std::vector<Real> cell_coeffs;

    Real sum = 0.0;
    const GlobalIndex n_cells = dof_map.getNumCells();
    for (GlobalIndex c = 0; c < n_cells; ++c) {
        auto mapping = buildAffineMappingForCell(mesh, c);
        FE_CHECK_NOT_NULL(mapping.get(), "computeL2ErrorVector2: mapping");

        const auto cell_dofs = dof_map.getCellDofs(c);
        cell_coeffs.resize(cell_dofs.size());
        for (std::size_t i = 0; i < cell_dofs.size(); ++i) {
            const auto gd = cell_dofs[i];
            cell_coeffs[i] = u_h[static_cast<std::size_t>(gd)];
        }

        for (LocalIndex q = 0; q < static_cast<LocalIndex>(quad->num_points()); ++q) {
            const auto& qp = quad->point(static_cast<std::size_t>(q));
            const math::Vector<Real, 3> xi{qp[0], qp[1], qp[2]};

            const auto uh = space.evaluate(xi, cell_coeffs);

            const auto x = mapping->map_to_physical(xi);
            const auto ue = u_exact(x[0], x[1]);
            const Real e0 = uh[0] - ue[0];
            const Real e1 = uh[1] - ue[1];

            const Real detJ = mapping->jacobian_determinant(xi);
            const Real e2 = e0 * e0 + e1 * e1;
            sum += e2 * quad->weight(static_cast<std::size_t>(q)) * std::abs(detJ);
        }
    }

    return std::sqrt(sum);
}

[[nodiscard]] std::vector<Real> convergenceRatesDyadic(const std::vector<Real>& errors)
{
    std::vector<Real> rates;
    if (errors.size() < 2u) return rates;
    rates.reserve(errors.size() - 1u);
    for (std::size_t i = 0; i + 1u < errors.size(); ++i) {
        rates.push_back(std::log(errors[i] / errors[i + 1u]) / std::log(Real(2.0)));
    }
    return rates;
}

// -----------------------------------------------------------------------------
// Mesh access helpers
// -----------------------------------------------------------------------------

class StructuredTriMeshAccess final : public IMeshAccess {
public:
    explicit StructuredTriMeshAccess(int n_cells_per_axis)
        : n_(n_cells_per_axis)
    {
        FE_THROW_IF(n_ <= 0, InvalidArgumentException,
                    "StructuredTriMeshAccess: n must be >= 1");

        const int n_nodes_1d = n_ + 1;
        nodes_.resize(static_cast<std::size_t>(n_nodes_1d * n_nodes_1d));
        for (int j = 0; j < n_nodes_1d; ++j) {
            for (int i = 0; i < n_nodes_1d; ++i) {
                const Real x = static_cast<Real>(i) / static_cast<Real>(n_);
                const Real y = static_cast<Real>(j) / static_cast<Real>(n_);
                nodes_[static_cast<std::size_t>(nodeId(i, j))] = {x, y, 0.0};
            }
        }

        // Two triangles per logical quad cell.
        cells_.resize(static_cast<std::size_t>(2 * n_ * n_));
        for (int j = 0; j < n_; ++j) {
            for (int i = 0; i < n_; ++i) {
                const GlobalIndex v0 = nodeId(i, j);
                const GlobalIndex v1 = nodeId(i + 1, j);
                const GlobalIndex v2 = nodeId(i + 1, j + 1);
                const GlobalIndex v3 = nodeId(i, j + 1);

                const GlobalIndex c0 = triCellId(i, j, /*which=*/0);
                const GlobalIndex c1 = triCellId(i, j, /*which=*/1);
                cells_[static_cast<std::size_t>(c0)] = {v0, v1, v2};
                cells_[static_cast<std::size_t>(c1)] = {v0, v2, v3};
            }
        }
    }

    [[nodiscard]] GlobalIndex numCells() const override { return static_cast<GlobalIndex>(cells_.size()); }
    [[nodiscard]] GlobalIndex numOwnedCells() const override { return numCells(); }
    [[nodiscard]] GlobalIndex numBoundaryFaces() const override { return 0; }
    [[nodiscard]] GlobalIndex numInteriorFaces() const override { return 0; }
    [[nodiscard]] int dimension() const override { return 2; }

    [[nodiscard]] bool isOwnedCell(GlobalIndex /*cell_id*/) const override { return true; }
    [[nodiscard]] ElementType getCellType(GlobalIndex /*cell_id*/) const override { return ElementType::Triangle3; }

    void getCellNodes(GlobalIndex cell_id, std::vector<GlobalIndex>& nodes) const override
    {
        const auto& cell = cells_.at(static_cast<std::size_t>(cell_id));
        nodes.assign(cell.begin(), cell.end());
    }

    [[nodiscard]] std::array<Real, 3> getNodeCoordinates(GlobalIndex node_id) const override
    {
        return nodes_.at(static_cast<std::size_t>(node_id));
    }

    void getCellCoordinates(GlobalIndex cell_id,
                            std::vector<std::array<Real, 3>>& coords) const override
    {
        const auto& cell = cells_.at(static_cast<std::size_t>(cell_id));
        coords.resize(cell.size());
        for (std::size_t i = 0; i < cell.size(); ++i) {
            coords[i] = nodes_.at(static_cast<std::size_t>(cell[i]));
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

    void forEachCell(std::function<void(GlobalIndex)> callback) const override
    {
        for (GlobalIndex c = 0; c < numCells(); ++c) callback(c);
    }

    void forEachOwnedCell(std::function<void(GlobalIndex)> callback) const override
    {
        forEachCell(std::move(callback));
    }

    void forEachBoundaryFace(int /*marker*/,
                             std::function<void(GlobalIndex, GlobalIndex)> /*callback*/) const override
    {
    }

    void forEachInteriorFace(std::function<void(GlobalIndex, GlobalIndex, GlobalIndex)> /*callback*/) const override
    {
    }

    [[nodiscard]] int cellsPerAxis() const noexcept { return n_; }

private:
    [[nodiscard]] GlobalIndex nodeId(int i, int j) const
    {
        const int n_nodes_1d = n_ + 1;
        return static_cast<GlobalIndex>(i + n_nodes_1d * j);
    }

    [[nodiscard]] GlobalIndex triCellId(int i, int j, int which) const
    {
        return static_cast<GlobalIndex>(2 * (i + n_ * j) + which);
    }

    int n_{1};
    std::vector<std::array<Real, 3>> nodes_{};
    std::vector<std::array<GlobalIndex, 3>> cells_{};
};

[[nodiscard]] dofs::MeshTopologyInfo buildTriGridTopology(const StructuredTriMeshAccess& mesh)
{
    const int n = mesh.cellsPerAxis();
    const GlobalIndex n_cells = static_cast<GlobalIndex>(2 * n * n);
    const GlobalIndex n_vertices = static_cast<GlobalIndex>((n + 1) * (n + 1));

    dofs::MeshTopologyInfo topo;
    topo.n_cells = n_cells;
    topo.n_vertices = n_vertices;
    topo.dim = 2;

    topo.cell2vertex_offsets.resize(static_cast<std::size_t>(n_cells) + 1u, 0);
    topo.cell2vertex_data.resize(static_cast<std::size_t>(n_cells) * 3u, 0);

    std::vector<GlobalIndex> cell_nodes;
    cell_nodes.reserve(3);
    for (GlobalIndex c = 0; c < n_cells; ++c) {
        topo.cell2vertex_offsets[static_cast<std::size_t>(c)] = static_cast<MeshOffset>(3u * c);
        mesh.getCellNodes(c, cell_nodes);
        FE_CHECK_ARG(cell_nodes.size() == 3u, "buildTriGridTopology: Triangle3 node count");
        for (std::size_t i = 0; i < 3u; ++i) {
            topo.cell2vertex_data[static_cast<std::size_t>(3u * c + static_cast<GlobalIndex>(i))] =
                static_cast<MeshIndex>(cell_nodes[i]);
        }
    }
    topo.cell2vertex_offsets[static_cast<std::size_t>(n_cells)] = static_cast<MeshOffset>(3u * n_cells);

    topo.vertex_gids.resize(static_cast<std::size_t>(n_vertices), 0);
    for (GlobalIndex v = 0; v < n_vertices; ++v) {
        topo.vertex_gids[static_cast<std::size_t>(v)] = static_cast<dofs::gid_t>(v);
    }

    topo.cell_gids.resize(static_cast<std::size_t>(n_cells), 0);
    topo.cell_owner_ranks.resize(static_cast<std::size_t>(n_cells), 0);
    for (GlobalIndex c = 0; c < n_cells; ++c) {
        topo.cell_gids[static_cast<std::size_t>(c)] = static_cast<dofs::gid_t>(c);
        topo.cell_owner_ranks[static_cast<std::size_t>(c)] = 0;
    }

    return topo;
}

class StructuredHexMeshAccess final : public IMeshAccess {
public:
    explicit StructuredHexMeshAccess(int n_cells_per_axis)
        : n_(n_cells_per_axis)
    {
        FE_THROW_IF(n_ <= 0, InvalidArgumentException,
                    "StructuredHexMeshAccess: n must be >= 1");

        const int n_nodes_1d = n_ + 1;
        nodes_.resize(static_cast<std::size_t>(n_nodes_1d * n_nodes_1d * n_nodes_1d));
        for (int k = 0; k < n_nodes_1d; ++k) {
            for (int j = 0; j < n_nodes_1d; ++j) {
                for (int i = 0; i < n_nodes_1d; ++i) {
                    const Real x = static_cast<Real>(i) / static_cast<Real>(n_);
                    const Real y = static_cast<Real>(j) / static_cast<Real>(n_);
                    const Real z = static_cast<Real>(k) / static_cast<Real>(n_);
                    nodes_[static_cast<std::size_t>(nodeId(i, j, k))] = {x, y, z};
                }
            }
        }

        cells_.resize(static_cast<std::size_t>(n_ * n_ * n_));
        for (int k = 0; k < n_; ++k) {
            for (int j = 0; j < n_; ++j) {
                for (int i = 0; i < n_; ++i) {
                    const GlobalIndex c = cellId(i, j, k);
                    const GlobalIndex v000 = nodeId(i, j, k);
                    const GlobalIndex v100 = nodeId(i + 1, j, k);
                    const GlobalIndex v110 = nodeId(i + 1, j + 1, k);
                    const GlobalIndex v010 = nodeId(i, j + 1, k);
                    const GlobalIndex v001 = nodeId(i, j, k + 1);
                    const GlobalIndex v101 = nodeId(i + 1, j, k + 1);
                    const GlobalIndex v111 = nodeId(i + 1, j + 1, k + 1);
                    const GlobalIndex v011 = nodeId(i, j + 1, k + 1);

                    // VTK/SimVascular Hex8 ordering:
                    // 0:(0,0,0),1:(1,0,0),2:(1,1,0),3:(0,1,0),4:(0,0,1),5:(1,0,1),6:(1,1,1),7:(0,1,1)
                    cells_[static_cast<std::size_t>(c)] = {v000, v100, v110, v010, v001, v101, v111, v011};
                }
            }
        }
    }

    [[nodiscard]] GlobalIndex numCells() const override { return static_cast<GlobalIndex>(cells_.size()); }
    [[nodiscard]] GlobalIndex numOwnedCells() const override { return numCells(); }
    [[nodiscard]] GlobalIndex numBoundaryFaces() const override { return 0; }
    [[nodiscard]] GlobalIndex numInteriorFaces() const override { return 0; }
    [[nodiscard]] int dimension() const override { return 3; }

    [[nodiscard]] bool isOwnedCell(GlobalIndex /*cell_id*/) const override { return true; }
    [[nodiscard]] ElementType getCellType(GlobalIndex /*cell_id*/) const override { return ElementType::Hex8; }

    void getCellNodes(GlobalIndex cell_id, std::vector<GlobalIndex>& nodes) const override
    {
        const auto& cell = cells_.at(static_cast<std::size_t>(cell_id));
        nodes.assign(cell.begin(), cell.end());
    }

    [[nodiscard]] std::array<Real, 3> getNodeCoordinates(GlobalIndex node_id) const override
    {
        return nodes_.at(static_cast<std::size_t>(node_id));
    }

    void getCellCoordinates(GlobalIndex cell_id,
                            std::vector<std::array<Real, 3>>& coords) const override
    {
        const auto& cell = cells_.at(static_cast<std::size_t>(cell_id));
        coords.resize(cell.size());
        for (std::size_t i = 0; i < cell.size(); ++i) {
            coords[i] = nodes_.at(static_cast<std::size_t>(cell[i]));
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

    void forEachCell(std::function<void(GlobalIndex)> callback) const override
    {
        for (GlobalIndex c = 0; c < numCells(); ++c) callback(c);
    }

    void forEachOwnedCell(std::function<void(GlobalIndex)> callback) const override
    {
        forEachCell(std::move(callback));
    }

    void forEachBoundaryFace(int /*marker*/,
                             std::function<void(GlobalIndex, GlobalIndex)> /*callback*/) const override
    {
    }

    void forEachInteriorFace(std::function<void(GlobalIndex, GlobalIndex, GlobalIndex)> /*callback*/) const override
    {
    }

    [[nodiscard]] int cellsPerAxis() const noexcept { return n_; }

private:
    [[nodiscard]] GlobalIndex nodeId(int i, int j, int k) const
    {
        const int n_nodes_1d = n_ + 1;
        return static_cast<GlobalIndex>(i + n_nodes_1d * (j + n_nodes_1d * k));
    }

    [[nodiscard]] GlobalIndex cellId(int i, int j, int k) const
    {
        return static_cast<GlobalIndex>(i + n_ * (j + n_ * k));
    }

    int n_{1};
    std::vector<std::array<Real, 3>> nodes_{};
    std::vector<std::array<GlobalIndex, 8>> cells_{};
};

[[nodiscard]] dofs::MeshTopologyInfo buildHexGridTopology(const StructuredHexMeshAccess& mesh)
{
    const int n = mesh.cellsPerAxis();
    const GlobalIndex n_cells = static_cast<GlobalIndex>(n * n * n);
    const GlobalIndex n_vertices = static_cast<GlobalIndex>((n + 1) * (n + 1) * (n + 1));

    dofs::MeshTopologyInfo topo;
    topo.n_cells = n_cells;
    topo.n_vertices = n_vertices;
    topo.dim = 3;

    topo.cell2vertex_offsets.resize(static_cast<std::size_t>(n_cells) + 1u, 0);
    topo.cell2vertex_data.resize(static_cast<std::size_t>(n_cells) * 8u, 0);

    std::vector<GlobalIndex> cell_nodes;
    cell_nodes.reserve(8);
    for (GlobalIndex c = 0; c < n_cells; ++c) {
        topo.cell2vertex_offsets[static_cast<std::size_t>(c)] = static_cast<MeshOffset>(8u * c);
        mesh.getCellNodes(c, cell_nodes);
        FE_CHECK_ARG(cell_nodes.size() == 8u, "buildHexGridTopology: Hex8 node count");
        for (std::size_t i = 0; i < 8u; ++i) {
            topo.cell2vertex_data[static_cast<std::size_t>(8u * c + static_cast<GlobalIndex>(i))] =
                static_cast<MeshIndex>(cell_nodes[i]);
        }
    }
    topo.cell2vertex_offsets[static_cast<std::size_t>(n_cells)] = static_cast<MeshOffset>(8u * n_cells);

    topo.vertex_gids.resize(static_cast<std::size_t>(n_vertices), 0);
    for (GlobalIndex v = 0; v < n_vertices; ++v) {
        topo.vertex_gids[static_cast<std::size_t>(v)] = static_cast<dofs::gid_t>(v);
    }

    topo.cell_gids.resize(static_cast<std::size_t>(n_cells), 0);
    topo.cell_owner_ranks.resize(static_cast<std::size_t>(n_cells), 0);
    for (GlobalIndex c = 0; c < n_cells; ++c) {
        topo.cell_gids[static_cast<std::size_t>(c)] = static_cast<dofs::gid_t>(c);
        topo.cell_owner_ranks[static_cast<std::size_t>(c)] = 0;
    }

    return topo;
}

class StructuredKuhnTetMeshAccess final : public IMeshAccess {
public:
    explicit StructuredKuhnTetMeshAccess(int n_cells_per_axis)
        : n_(n_cells_per_axis)
    {
        FE_THROW_IF(n_ <= 0, InvalidArgumentException,
                    "StructuredKuhnTetMeshAccess: n must be >= 1");

        const int n_nodes_1d = n_ + 1;
        nodes_.resize(static_cast<std::size_t>(n_nodes_1d * n_nodes_1d * n_nodes_1d));
        for (int k = 0; k < n_nodes_1d; ++k) {
            for (int j = 0; j < n_nodes_1d; ++j) {
                for (int i = 0; i < n_nodes_1d; ++i) {
                    const Real x = static_cast<Real>(i) / static_cast<Real>(n_);
                    const Real y = static_cast<Real>(j) / static_cast<Real>(n_);
                    const Real z = static_cast<Real>(k) / static_cast<Real>(n_);
                    nodes_[static_cast<std::size_t>(nodeId(i, j, k))] = {x, y, z};
                }
            }
        }

        // 6 tets per cube (Kuhn / Freudenthal triangulation).
        cells_.resize(static_cast<std::size_t>(6 * n_ * n_ * n_));
        for (int k = 0; k < n_; ++k) {
            for (int j = 0; j < n_; ++j) {
                for (int i = 0; i < n_; ++i) {
                    const GlobalIndex v000 = nodeId(i, j, k);
                    const GlobalIndex v100 = nodeId(i + 1, j, k);
                    const GlobalIndex v010 = nodeId(i, j + 1, k);
                    const GlobalIndex v110 = nodeId(i + 1, j + 1, k);
                    const GlobalIndex v001 = nodeId(i, j, k + 1);
                    const GlobalIndex v101 = nodeId(i + 1, j, k + 1);
                    const GlobalIndex v011 = nodeId(i, j + 1, k + 1);
                    const GlobalIndex v111 = nodeId(i + 1, j + 1, k + 1);

                    const GlobalIndex base = 6 * cubeId(i, j, k);
                    cells_[static_cast<std::size_t>(base + 0)] = {v000, v100, v110, v111};
                    cells_[static_cast<std::size_t>(base + 1)] = {v000, v100, v101, v111};
                    cells_[static_cast<std::size_t>(base + 2)] = {v000, v010, v110, v111};
                    cells_[static_cast<std::size_t>(base + 3)] = {v000, v010, v011, v111};
                    cells_[static_cast<std::size_t>(base + 4)] = {v000, v001, v101, v111};
                    cells_[static_cast<std::size_t>(base + 5)] = {v000, v001, v011, v111};
                }
            }
        }
    }

    [[nodiscard]] GlobalIndex numCells() const override { return static_cast<GlobalIndex>(cells_.size()); }
    [[nodiscard]] GlobalIndex numOwnedCells() const override { return numCells(); }
    [[nodiscard]] GlobalIndex numBoundaryFaces() const override { return 0; }
    [[nodiscard]] GlobalIndex numInteriorFaces() const override { return 0; }
    [[nodiscard]] int dimension() const override { return 3; }

    [[nodiscard]] bool isOwnedCell(GlobalIndex /*cell_id*/) const override { return true; }
    [[nodiscard]] ElementType getCellType(GlobalIndex /*cell_id*/) const override { return ElementType::Tetra4; }

    void getCellNodes(GlobalIndex cell_id, std::vector<GlobalIndex>& nodes) const override
    {
        const auto& cell = cells_.at(static_cast<std::size_t>(cell_id));
        nodes.assign(cell.begin(), cell.end());
    }

    [[nodiscard]] std::array<Real, 3> getNodeCoordinates(GlobalIndex node_id) const override
    {
        return nodes_.at(static_cast<std::size_t>(node_id));
    }

    void getCellCoordinates(GlobalIndex cell_id,
                            std::vector<std::array<Real, 3>>& coords) const override
    {
        const auto& cell = cells_.at(static_cast<std::size_t>(cell_id));
        coords.resize(cell.size());
        for (std::size_t i = 0; i < cell.size(); ++i) {
            coords[i] = nodes_.at(static_cast<std::size_t>(cell[i]));
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

    void forEachCell(std::function<void(GlobalIndex)> callback) const override
    {
        for (GlobalIndex c = 0; c < numCells(); ++c) callback(c);
    }

    void forEachOwnedCell(std::function<void(GlobalIndex)> callback) const override
    {
        forEachCell(std::move(callback));
    }

    void forEachBoundaryFace(int /*marker*/,
                             std::function<void(GlobalIndex, GlobalIndex)> /*callback*/) const override
    {
    }

    void forEachInteriorFace(std::function<void(GlobalIndex, GlobalIndex, GlobalIndex)> /*callback*/) const override
    {
    }

    [[nodiscard]] int cellsPerAxis() const noexcept { return n_; }

private:
    [[nodiscard]] GlobalIndex nodeId(int i, int j, int k) const
    {
        const int n_nodes_1d = n_ + 1;
        return static_cast<GlobalIndex>(i + n_nodes_1d * (j + n_nodes_1d * k));
    }

    [[nodiscard]] GlobalIndex cubeId(int i, int j, int k) const
    {
        return static_cast<GlobalIndex>(i + n_ * (j + n_ * k));
    }

    int n_{1};
    std::vector<std::array<Real, 3>> nodes_{};
    std::vector<std::array<GlobalIndex, 4>> cells_{};
};

[[nodiscard]] dofs::MeshTopologyInfo buildKuhnTetGridTopology(const StructuredKuhnTetMeshAccess& mesh)
{
    const int n = mesh.cellsPerAxis();
    const GlobalIndex n_cells = static_cast<GlobalIndex>(6 * n * n * n);
    const GlobalIndex n_vertices = static_cast<GlobalIndex>((n + 1) * (n + 1) * (n + 1));

    dofs::MeshTopologyInfo topo;
    topo.n_cells = n_cells;
    topo.n_vertices = n_vertices;
    topo.dim = 3;

    topo.cell2vertex_offsets.resize(static_cast<std::size_t>(n_cells) + 1u, 0);
    topo.cell2vertex_data.resize(static_cast<std::size_t>(n_cells) * 4u, 0);

    std::vector<GlobalIndex> cell_nodes;
    cell_nodes.reserve(4);
    for (GlobalIndex c = 0; c < n_cells; ++c) {
        topo.cell2vertex_offsets[static_cast<std::size_t>(c)] = static_cast<MeshOffset>(4u * c);
        mesh.getCellNodes(c, cell_nodes);
        FE_CHECK_ARG(cell_nodes.size() == 4u, "buildKuhnTetGridTopology: Tetra4 node count");
        for (std::size_t i = 0; i < 4u; ++i) {
            topo.cell2vertex_data[static_cast<std::size_t>(4u * c + static_cast<GlobalIndex>(i))] =
                static_cast<MeshIndex>(cell_nodes[i]);
        }
    }
    topo.cell2vertex_offsets[static_cast<std::size_t>(n_cells)] = static_cast<MeshOffset>(4u * n_cells);

    topo.vertex_gids.resize(static_cast<std::size_t>(n_vertices), 0);
    for (GlobalIndex v = 0; v < n_vertices; ++v) {
        topo.vertex_gids[static_cast<std::size_t>(v)] = static_cast<dofs::gid_t>(v);
    }

    topo.cell_gids.resize(static_cast<std::size_t>(n_cells), 0);
    topo.cell_owner_ranks.resize(static_cast<std::size_t>(n_cells), 0);
    for (GlobalIndex c = 0; c < n_cells; ++c) {
        topo.cell_gids[static_cast<std::size_t>(c)] = static_cast<dofs::gid_t>(c);
        topo.cell_owner_ranks[static_cast<std::size_t>(c)] = 0;
    }

    return topo;
}

// -----------------------------------------------------------------------------
// MMS definitions
// -----------------------------------------------------------------------------

[[nodiscard]] Real uExact2D(Real x, Real y)
{
    const Real gx = x * x * (Real(1.0) - x);
    const Real gy = y * y * (Real(1.0) - y);
    return gx + gy;
}

[[nodiscard]] forms::FormExpr fRhs2D()
{
    using forms::FormExpr;
    using forms::x;

    const auto X = x();
    const auto xx = X.component(0);
    const auto yy = X.component(1);

    // For u = g(x) + g(y), g(t)=t^2(1-t):
    //   f = -Δu = 6(x + y) - 4.
    return FormExpr::constant(Real(6.0)) * (xx + yy) - FormExpr::constant(Real(4.0));
}

[[nodiscard]] Real uExact3D(Real x, Real y, Real z)
{
    const Real gx = x * x * (Real(1.0) - x);
    const Real gy = y * y * (Real(1.0) - y);
    const Real gz = z * z * (Real(1.0) - z);
    return gx + gy + gz;
}

[[nodiscard]] forms::FormExpr fRhs3D()
{
    using forms::FormExpr;
    using forms::x;

    const auto X = x();
    const auto xx = X.component(0);
    const auto yy = X.component(1);
    const auto zz = X.component(2);

    // For u = g(x) + g(y) + g(z), g(t)=t^2(1-t):
    //   f = -Δu = 6(x + y + z) - 6.
    return FormExpr::constant(Real(6.0)) * (xx + yy + zz) - FormExpr::constant(Real(6.0));
}

// -----------------------------------------------------------------------------
// Linear solve helper (assemble J and RHS at U=0 and solve J*u = rhs)
// -----------------------------------------------------------------------------

[[nodiscard]] std::vector<Real> solveLinearSystem(const IMeshAccess& mesh,
                                                  const dofs::MeshTopologyInfo& topo,
                                                  const spaces::FunctionSpace& space,
                                                  const forms::FormExpr& residual,
                                                  constraints::AffineConstraints constraints)
{
    dofs::DofHandler dof_handler;
    dof_handler.distributeDofs(topo, space);
    dof_handler.finalize();

    const auto& dof_map = dof_handler.getDofMap();
    const GlobalIndex n = dof_map.getNumDofs();

    if (!constraints.isClosed()) {
        constraints.close();
    }

    forms::FormCompiler compiler;
    auto ir = compiler.compileResidual(residual);
    forms::SymbolicNonlinearFormKernel kernel(std::move(ir), forms::NonlinearKernelOutput::Both);
    kernel.resolveInlinableConstitutives();

    StandardAssembler assembler;
    assembler.setDofMap(dof_map);
    assembler.setConstraints(&constraints);

    std::vector<Real> U0(static_cast<std::size_t>(n), 0.0);
    assembler.setCurrentSolution(U0);

    DenseMatrixView J(n);
    DenseVectorView R(n);
    J.zero();
    R.zero();
    (void)assembler.assembleBoth(mesh, space, space, kernel, J, R);

    std::vector<Real> rhs(static_cast<std::size_t>(n), 0.0);
    for (GlobalIndex i = 0; i < n; ++i) {
        rhs[static_cast<std::size_t>(i)] = R.getVectorEntry(i);
    }

    auto solve = solveDenseWithPartialPivoting(std::move(J), std::move(rhs));
    FE_THROW_IF(!solve.ok, FEException, solve.message);
    FE_THROW_IF(static_cast<GlobalIndex>(solve.x.size()) != n, FEException,
                "solveLinearSystem: solution size mismatch");

    constraints.distribute(solve.x.data(), n);
    return solve.x;
}

[[nodiscard]] Real elasticityMmsErrorTri(int n_cells_per_axis, Real nu)
{
    StructuredTriMeshAccess mesh(n_cells_per_axis);
    const auto topo = buildTriGridTopology(mesh);

    const Real pi = std::acos(Real(-1.0));
    constexpr Real E = 1.0;
    const Real mu = E / (Real(2.0) * (Real(1.0) + nu));
    const Real lambda = E * nu / ((Real(1.0) + nu) * (Real(1.0) - Real(2.0) * nu));

    auto u_scalar = std::make_shared<spaces::H1Space>(ElementType::Triangle3, /*order=*/1);
    spaces::ProductSpace u_space(std::move(u_scalar), /*components=*/2);

    dofs::DofHandler dof_handler;
    dof_handler.distributeDofs(topo, u_space);
    dof_handler.finalize();

    const auto& dof_map = dof_handler.getDofMap();
    const GlobalIndex n = dof_map.getNumDofs();

    const auto dof_info = computeDofCoordinatesProductSpace(mesh, u_space, dof_map);

    const auto u_exact = [pi](Real x, Real y) -> std::array<Real, 2> {
        const Real g = std::sin(pi * x) * std::sin(pi * y);
        return {g, Real(0.5) * g};
    };

    constraints::AffineConstraints constraints;
    addDirichletOnBoundaryVector(dof_info, u_exact, constraints);
    constraints.close();

    forms::FormCompiler compiler;
    const auto u = forms::TrialFunction(u_space, "u");
    const auto v = forms::TestFunction(u_space, "v");

    const auto mu_c = forms::FormExpr::constant(mu);
    const auto lambda_c = forms::FormExpr::constant(lambda);

    const auto f = forms::FormExpr::coefficient("f", [pi, mu, lambda](Real x, Real y, Real /*z*/) {
        const Real g = std::sin(pi * x) * std::sin(pi * y);
        const Real c = std::cos(pi * x) * std::cos(pi * y);

        const Real f1 = pi * pi * ((Real(3.0) * mu + lambda) * g - Real(0.5) * (lambda + mu) * c);
        const Real f2 = pi * pi * ((Real(1.5) * mu + Real(0.5) * lambda) * g - (lambda + mu) * c);
        return std::array<Real, 3>{f1, f2, 0.0};
    });

    // Assemble as Ku = b (evaluate at U=0 so R == RHS).
    const auto residual =
        (forms::FormExpr::constant(2.0) * mu_c * forms::inner(forms::sym(forms::grad(u)), forms::sym(forms::grad(v))) +
         lambda_c * forms::div(u) * forms::div(v) +
         forms::inner(f, v))
            .dx();

    auto ir = compiler.compileResidual(residual);
    forms::SymbolicNonlinearFormKernel kernel(std::move(ir), forms::NonlinearKernelOutput::Both);
    kernel.resolveInlinableConstitutives();

    StandardAssembler assembler;
    assembler.setDofMap(dof_map);
    assembler.setConstraints(&constraints);

    std::vector<Real> U0(static_cast<std::size_t>(n), 0.0);
    assembler.setCurrentSolution(U0);

    DenseMatrixView J(n);
    DenseVectorView R(n);
    J.zero();
    R.zero();
    (void)assembler.assembleBoth(mesh, u_space, u_space, kernel, J, R);

    std::vector<Real> rhs(static_cast<std::size_t>(n), 0.0);
    for (GlobalIndex i = 0; i < n; ++i) {
        rhs[static_cast<std::size_t>(i)] = R.getVectorEntry(i);
    }

    auto solve = solveDenseWithPartialPivoting(std::move(J), std::move(rhs));
    FE_THROW_IF(!solve.ok, FEException, solve.message);
    FE_THROW_IF(static_cast<GlobalIndex>(solve.x.size()) != n, FEException,
                "elasticityMmsErrorTri: solution size mismatch");

    constraints.distribute(solve.x.data(), n);

    return computeL2ErrorVector2(mesh, u_space, dof_map, solve.x, u_exact, /*quad_order=*/10);
}

} // namespace

TEST(ManufacturedSolutionConvergenceElementTypes, PoissonTriangle3_P1_L2ConvergesAtOrderTwo)
{
    std::vector<Real> errors;
    const std::vector<int> Ns = {2, 4, 8};
    errors.reserve(Ns.size());

    for (const int N : Ns) {
        StructuredTriMeshAccess mesh(N);
        auto topo = buildTriGridTopology(mesh);

        spaces::H1Space space(ElementType::Triangle3, /*order=*/1);
        dofs::DofHandler dof_handler;
        dof_handler.distributeDofs(topo, space);
        dof_handler.finalize();

        const auto& dof_map = dof_handler.getDofMap();
        const auto dof_coords = computeDofCoordinates(mesh, space, dof_map);

        auto constraints = makeDirichletOnBoundaryUnitSquare(
            dof_coords, [](Real x, Real y) { return uExact2D(x, y); });

        const auto u = forms::TrialFunction(space, "u");
        const auto v = forms::TestFunction(space, "v");
        const auto residual = (forms::inner(forms::grad(u), forms::grad(v)) + fRhs2D() * v).dx();

        const auto sol = solveLinearSystem(mesh, topo, space, residual, std::move(constraints));

        const Real err = computeL2ErrorScalar(
            mesh, space, dof_map, sol,
            [](std::array<Real, 3> x) { return uExact2D(x[0], x[1]); },
            /*quad_order=*/10);

        errors.push_back(err);
    }

    ASSERT_EQ(errors.size(), Ns.size());
    EXPECT_GT(errors[0], errors[1]);
    EXPECT_GT(errors[1], errors[2]);

    const auto rates = convergenceRatesDyadic(errors);
    ASSERT_EQ(rates.size(), 2u);
    EXPECT_GT(rates[0], Real(1.8));
    EXPECT_GT(rates[1], Real(1.8));
}

TEST(ManufacturedSolutionConvergenceElementTypes, PoissonTriangle3_P2_L2ConvergesAtOrderThree)
{
    std::vector<Real> errors;
    const std::vector<int> Ns = {2, 4, 8};
    errors.reserve(Ns.size());

    for (const int N : Ns) {
        StructuredTriMeshAccess mesh(N);
        auto topo = buildTriGridTopology(mesh);

        spaces::H1Space space(ElementType::Triangle3, /*order=*/2);
        dofs::DofHandler dof_handler;
        dof_handler.distributeDofs(topo, space);
        dof_handler.finalize();

        const auto& dof_map = dof_handler.getDofMap();
        const auto dof_coords = computeDofCoordinates(mesh, space, dof_map);

        auto constraints = makeDirichletOnBoundaryUnitSquare(
            dof_coords, [](Real x, Real y) { return uExact2D(x, y); });

        const auto u = forms::TrialFunction(space, "u");
        const auto v = forms::TestFunction(space, "v");
        const auto residual = (forms::inner(forms::grad(u), forms::grad(v)) + fRhs2D() * v).dx();

        const auto sol = solveLinearSystem(mesh, topo, space, residual, std::move(constraints));

        const Real err = computeL2ErrorScalar(
            mesh, space, dof_map, sol,
            [](std::array<Real, 3> x) { return uExact2D(x[0], x[1]); },
            /*quad_order=*/12);

        errors.push_back(err);
    }

    ASSERT_EQ(errors.size(), Ns.size());
    EXPECT_GT(errors[0], errors[1]);
    EXPECT_GT(errors[1], errors[2]);

    const auto rates = convergenceRatesDyadic(errors);
    ASSERT_EQ(rates.size(), 2u);
    EXPECT_GT(rates[0], Real(2.6));
    EXPECT_GT(rates[1], Real(2.6));
}

TEST(ManufacturedSolutionConvergenceElementTypes, PoissonTetra4_P1_L2ConvergesAtOrderTwo)
{
    std::vector<Real> errors;
    const std::vector<int> Ns = {1, 2, 4};
    errors.reserve(Ns.size());

    for (const int N : Ns) {
        StructuredKuhnTetMeshAccess mesh(N);
        auto topo = buildKuhnTetGridTopology(mesh);

        spaces::H1Space space(ElementType::Tetra4, /*order=*/1);
        dofs::DofHandler dof_handler;
        dof_handler.distributeDofs(topo, space);
        dof_handler.finalize();

        const auto& dof_map = dof_handler.getDofMap();
        const auto dof_coords = computeDofCoordinates(mesh, space, dof_map);

        auto constraints = makeDirichletOnBoundaryUnitCube(
            dof_coords, [](Real x, Real y, Real z) { return uExact3D(x, y, z); });

        const auto u = forms::TrialFunction(space, "u");
        const auto v = forms::TestFunction(space, "v");
        const auto residual = (forms::inner(forms::grad(u), forms::grad(v)) + fRhs3D() * v).dx();

        const auto sol = solveLinearSystem(mesh, topo, space, residual, std::move(constraints));

        const Real err = computeL2ErrorScalar(
            mesh, space, dof_map, sol,
            [](std::array<Real, 3> x) { return uExact3D(x[0], x[1], x[2]); },
            /*quad_order=*/8);

        errors.push_back(err);
    }

    ASSERT_EQ(errors.size(), Ns.size());
    EXPECT_GT(errors[0], errors[1]);
    EXPECT_GT(errors[1], errors[2]);

    const auto rates = convergenceRatesDyadic(errors);
    ASSERT_EQ(rates.size(), 2u);
    // The N=1 -> N=2 step is very coarse; accept a slightly relaxed rate there.
    EXPECT_GT(rates[0], Real(1.5));
    EXPECT_GT(rates[1], Real(1.7));
}

TEST(ManufacturedSolutionConvergenceElementTypes, PoissonTetra4_P2_L2ConvergesAtOrderThree)
{
    std::vector<Real> errors;
    const std::vector<int> Ns = {1, 2};
    errors.reserve(Ns.size());

    for (const int N : Ns) {
        StructuredKuhnTetMeshAccess mesh(N);
        auto topo = buildKuhnTetGridTopology(mesh);

        spaces::H1Space space(ElementType::Tetra4, /*order=*/2);
        dofs::DofHandler dof_handler;
        dof_handler.distributeDofs(topo, space);
        dof_handler.finalize();

        const auto& dof_map = dof_handler.getDofMap();
        const auto dof_coords = computeDofCoordinates(mesh, space, dof_map);

        auto constraints = makeDirichletOnBoundaryUnitCube(
            dof_coords, [](Real x, Real y, Real z) { return uExact3D(x, y, z); });

        const auto u = forms::TrialFunction(space, "u");
        const auto v = forms::TestFunction(space, "v");
        const auto residual = (forms::inner(forms::grad(u), forms::grad(v)) + fRhs3D() * v).dx();

        const auto sol = solveLinearSystem(mesh, topo, space, residual, std::move(constraints));

        const Real err = computeL2ErrorScalar(
            mesh, space, dof_map, sol,
            [](std::array<Real, 3> x) { return uExact3D(x[0], x[1], x[2]); },
            /*quad_order=*/10);

        errors.push_back(err);
    }

    ASSERT_EQ(errors.size(), Ns.size());
    EXPECT_GT(errors[0], errors[1]);

    const auto rates = convergenceRatesDyadic(errors);
    ASSERT_EQ(rates.size(), 1u);
    EXPECT_GT(rates[0], Real(2.2));
}

TEST(ManufacturedSolutionConvergenceElementTypes, PoissonHex8_Q1_L2ConvergesAtOrderTwo)
{
    std::vector<Real> errors;
    const std::vector<int> Ns = {1, 2, 4};
    errors.reserve(Ns.size());

    for (const int N : Ns) {
        StructuredHexMeshAccess mesh(N);
        auto topo = buildHexGridTopology(mesh);

        spaces::H1Space space(ElementType::Hex8, /*order=*/1);
        dofs::DofHandler dof_handler;
        dof_handler.distributeDofs(topo, space);
        dof_handler.finalize();

        const auto& dof_map = dof_handler.getDofMap();
        const auto dof_coords = computeDofCoordinates(mesh, space, dof_map);

        auto constraints = makeDirichletOnBoundaryUnitCube(
            dof_coords, [](Real x, Real y, Real z) { return uExact3D(x, y, z); });

        const auto u = forms::TrialFunction(space, "u");
        const auto v = forms::TestFunction(space, "v");
        const auto residual = (forms::inner(forms::grad(u), forms::grad(v)) + fRhs3D() * v).dx();

        const auto sol = solveLinearSystem(mesh, topo, space, residual, std::move(constraints));

        const Real err = computeL2ErrorScalar(
            mesh, space, dof_map, sol,
            [](std::array<Real, 3> x) { return uExact3D(x[0], x[1], x[2]); },
            /*quad_order=*/8);

        errors.push_back(err);
    }

    ASSERT_EQ(errors.size(), Ns.size());
    EXPECT_GT(errors[0], errors[1]);
    EXPECT_GT(errors[1], errors[2]);

    const auto rates = convergenceRatesDyadic(errors);
    ASSERT_EQ(rates.size(), 2u);
    // The N=1 -> N=2 step is very coarse; accept a slightly relaxed rate there.
    EXPECT_GT(rates[0], Real(1.5));
    EXPECT_GT(rates[1], Real(1.7));
}

TEST(ManufacturedSolutionConvergenceElementTypes, ElasticityTriangle3_P1_L2ConvergesAtOrderTwo)
{
    std::vector<Real> errors;
    const std::vector<int> Ns = {2, 4, 8};
    errors.reserve(Ns.size());

    for (const int N : Ns) {
        errors.push_back(elasticityMmsErrorTri(N, /*nu=*/Real(0.3)));
    }

    ASSERT_EQ(errors.size(), Ns.size());
    EXPECT_GT(errors[0], errors[1]);
    EXPECT_GT(errors[1], errors[2]);

    const auto rates = convergenceRatesDyadic(errors);
    ASSERT_EQ(rates.size(), 2u);
    EXPECT_GT(rates[0], Real(1.7));
    EXPECT_GT(rates[1], Real(1.7));
}

} // namespace svmp::FE::assembly::testing

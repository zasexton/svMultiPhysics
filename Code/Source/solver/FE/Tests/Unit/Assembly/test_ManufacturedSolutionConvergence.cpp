/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

/**
 * @file test_ManufacturedSolutionConvergence.cpp
 * @brief End-to-end FE verification: assemble + constrain + solve + convergence rate.
 */

#include <gtest/gtest.h>

#include "Assembly/GlobalSystemView.h"
#include "Assembly/StandardAssembler.h"
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

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <limits>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace svmp {
namespace FE {
namespace assembly {
namespace testing {

namespace {

class StructuredQuadMeshAccess final : public IMeshAccess {
public:
    explicit StructuredQuadMeshAccess(int n_cells_per_axis)
        : n_(n_cells_per_axis)
    {
        FE_THROW_IF(n_ <= 0, InvalidArgumentException,
                    "StructuredQuadMeshAccess: n must be >= 1");

        const int n_nodes_1d = n_ + 1;
        nodes_.resize(static_cast<std::size_t>(n_nodes_1d * n_nodes_1d));
        for (int j = 0; j < n_nodes_1d; ++j) {
            for (int i = 0; i < n_nodes_1d; ++i) {
                const Real x = static_cast<Real>(i) / static_cast<Real>(n_);
                const Real y = static_cast<Real>(j) / static_cast<Real>(n_);
                nodes_[static_cast<std::size_t>(nodeId(i, j))] = {x, y, 0.0};
            }
        }

        cells_.resize(static_cast<std::size_t>(n_ * n_));
        for (int j = 0; j < n_; ++j) {
            for (int i = 0; i < n_; ++i) {
                const GlobalIndex c = cellId(i, j);
                cells_[static_cast<std::size_t>(c)] = {
                    nodeId(i, j),
                    nodeId(i + 1, j),
                    nodeId(i + 1, j + 1),
                    nodeId(i, j + 1)};
            }
        }
    }

    [[nodiscard]] GlobalIndex numCells() const override { return static_cast<GlobalIndex>(cells_.size()); }
    [[nodiscard]] GlobalIndex numOwnedCells() const override { return numCells(); }
    [[nodiscard]] GlobalIndex numBoundaryFaces() const override { return 0; }
    [[nodiscard]] GlobalIndex numInteriorFaces() const override { return 0; }
    [[nodiscard]] int dimension() const override { return 2; }

    [[nodiscard]] bool isOwnedCell(GlobalIndex /*cell_id*/) const override { return true; }
    [[nodiscard]] ElementType getCellType(GlobalIndex /*cell_id*/) const override { return ElementType::Quad4; }

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
        for (GlobalIndex c = 0; c < numCells(); ++c) {
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

    [[nodiscard]] GlobalIndex cellId(int i, int j) const
    {
        return static_cast<GlobalIndex>(i + n_ * j);
    }

    int n_{1};
    std::vector<std::array<Real, 3>> nodes_{};
    std::vector<std::array<GlobalIndex, 4>> cells_{};
};

[[nodiscard]] dofs::MeshTopologyInfo buildQuadGridTopology(const StructuredQuadMeshAccess& mesh)
{
    const int n = mesh.cellsPerAxis();
    const GlobalIndex n_cells = static_cast<GlobalIndex>(n * n);
    const GlobalIndex n_vertices = static_cast<GlobalIndex>((n + 1) * (n + 1));

    dofs::MeshTopologyInfo topo;
    topo.n_cells = n_cells;
    topo.n_vertices = n_vertices;
    topo.dim = 2;

    topo.cell2vertex_offsets.resize(static_cast<std::size_t>(n_cells) + 1u, 0);
    topo.cell2vertex_data.resize(static_cast<std::size_t>(n_cells) * 4u, 0);

    std::vector<GlobalIndex> cell_nodes;
    cell_nodes.reserve(4);
    for (GlobalIndex c = 0; c < n_cells; ++c) {
        topo.cell2vertex_offsets[static_cast<std::size_t>(c)] = static_cast<MeshOffset>(4u * c);
        mesh.getCellNodes(c, cell_nodes);
        FE_CHECK_ARG(cell_nodes.size() == 4u, "buildQuadGridTopology: Quad4 cell node count");
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

struct DenseSolveResult {
    bool ok{true};
    std::string message{};
    std::vector<Real> x{};
};

[[nodiscard]] DenseSolveResult solveDenseWithPartialPivoting(DenseMatrixView A, std::vector<Real> b)
{
    const GlobalIndex n = A.numRows();
    if (n != A.numCols()) {
        return {false, "solveDenseWithPartialPivoting: matrix is not square", {}};
    }
    if (static_cast<GlobalIndex>(b.size()) != n) {
        return {false, "solveDenseWithPartialPivoting: rhs size mismatch", {}};
    }

    auto data = A.dataMutable();
    auto idx = [n](GlobalIndex i, GlobalIndex j) {
        return static_cast<std::size_t>(i * n + j);
    };

    // Gaussian elimination with partial pivoting.
    for (GlobalIndex k = 0; k < n; ++k) {
        GlobalIndex pivot = k;
        Real max_abs = std::abs(data[idx(k, k)]);
        for (GlobalIndex i = k + 1; i < n; ++i) {
            const Real v = std::abs(data[idx(i, k)]);
            if (v > max_abs) {
                max_abs = v;
                pivot = i;
            }
        }
        if (max_abs < Real(1e-14)) {
            return {false, "solveDenseWithPartialPivoting: near-singular matrix", {}};
        }
        if (pivot != k) {
            for (GlobalIndex j = 0; j < n; ++j) {
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

[[nodiscard]] std::shared_ptr<geometry::GeometryMapping> buildAffineMappingForQuad4(
    const IMeshAccess& mesh,
    GlobalIndex cell_id)
{
    std::vector<std::array<Real, 3>> coords;
    mesh.getCellCoordinates(cell_id, coords);
    FE_CHECK_ARG(coords.size() == 4u, "buildAffineMappingForQuad4: expected 4 geometry nodes");

    std::vector<math::Vector<Real, 3>> node_coords(coords.size());
    for (std::size_t i = 0; i < coords.size(); ++i) {
        node_coords[i] = math::Vector<Real, 3>{coords[i][0], coords[i][1], coords[i][2]};
    }

    geometry::MappingRequest req;
    req.element_type = ElementType::Quad4;
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

    const auto& element = space.getElement(ElementType::Quad4, /*cell_id=*/0);
    const auto& basis = element.basis();
    const auto* lag = dynamic_cast<const basis::LagrangeBasis*>(&basis);
    FE_CHECK_NOT_NULL(lag, "computeDofCoordinates: expected LagrangeBasis for H1Space");
    const auto& ref_nodes = lag->nodes();

    const GlobalIndex n_cells = dof_map.getNumCells();
    for (GlobalIndex c = 0; c < n_cells; ++c) {
        auto mapping = buildAffineMappingForQuad4(mesh, c);
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

[[nodiscard]] bool near(Real a, Real b, Real tol = Real(1e-12))
{
    return std::abs(a - b) <= tol;
}

[[nodiscard]] Real uExactPolyBiharmonicLike(Real x, Real y)
{
    // u = x(1-x) y(1-y) on [0,1]^2 with homogeneous Dirichlet.
    return x * (Real(1.0) - x) * y * (Real(1.0) - y);
}

[[nodiscard]] Real uExactPolyNotInQ2(Real x, Real y)
{
    // u = x^3(1-x) y(1-y) on [0,1]^2 with homogeneous Dirichlet.
    return x * x * x * (Real(1.0) - x) * y * (Real(1.0) - y);
}

[[nodiscard]] forms::FormExpr fRhsPolyBiharmonicLike()
{
    using svmp::FE::forms::FormExpr;
    using svmp::FE::forms::x;

    const auto X = x();
    const auto xx = X.component(0);
    const auto yy = X.component(1);

    // f = -Δu = 2(x - x^2 + y - y^2) = 2(x(1-x) + y(1-y))
    return FormExpr::constant(Real(2.0)) *
           (xx * (FormExpr::constant(Real(1.0)) - xx) + yy * (FormExpr::constant(Real(1.0)) - yy));
}

[[nodiscard]] forms::FormExpr fRhsPolyNotInQ2()
{
    // For u = x^3(1-x) y(1-y):
    //   f = -Δu = -(g''(x) h(y) + g(x) h''(y)) = -g''(x) h(y) + 2 g(x),
    // where g(x) = x^3(1-x), h(y) = y(1-y), g''(x) = 6x - 12x^2, h''(y) = -2.
    using svmp::FE::forms::FormExpr;
    using svmp::FE::forms::x;

    const auto X = x();
    const auto xx = X.component(0);
    const auto yy = X.component(1);
    const auto one = FormExpr::constant(Real(1.0));

    const auto g = xx * xx * xx * (one - xx);
    const auto h = yy * (one - yy);
    const auto gpp = FormExpr::constant(Real(6.0)) * xx - FormExpr::constant(Real(12.0)) * xx * xx;

    return (-gpp * h) + FormExpr::constant(Real(2.0)) * g;
}

[[nodiscard]] Real uExactPeriodicInX_Q2(Real x, Real y)
{
    (void)x;
    // In Q2 space: u(y) = 1 + y(1-y)
    return Real(1.0) + y * (Real(1.0) - y);
}

[[nodiscard]] forms::FormExpr fRhsPeriodicInX_Q2()
{
    // For u(y) = 1 + y(1-y): -Δu = 2.
    return forms::FormExpr::constant(Real(2.0));
}

struct PoissonSolveOutput {
    std::vector<Real> u;
    std::vector<std::array<Real, 3>> dof_coords;
};

[[nodiscard]] PoissonSolveOutput solvePoissonOnQuadGrid(const IMeshAccess& mesh,
                                                        const dofs::MeshTopologyInfo& topo,
                                                        int order,
                                                        const forms::FormExpr& rhs_f,
                                                        constraints::AffineConstraints constraints)
{
    spaces::H1Space space(ElementType::Quad4, order);

    dofs::DofHandler dof_handler;
    dof_handler.distributeDofs(topo, space);
    dof_handler.finalize();

    const auto& dof_map = dof_handler.getDofMap();
    const GlobalIndex n_dofs = dof_map.getNumDofs();

    auto dof_coords = computeDofCoordinates(mesh, space, dof_map);

    // Constraints must be closed before assembly.
    if (!constraints.isClosed()) {
        constraints.close();
    }

    forms::FormCompiler compiler;
    const auto u = forms::TrialFunction(space, "u");
    const auto v = forms::TestFunction(space, "v");
    // Assemble as a linear system Ku = b (not a residual Ku - b), because the current
    // constraint distribution logic applies inhomogeneous constraints with RHS semantics.
    const auto residual = (forms::inner(forms::grad(u), forms::grad(v)) + rhs_f * v).dx();

    auto ir = compiler.compileResidual(residual);
    forms::SymbolicNonlinearFormKernel kernel(std::move(ir), forms::NonlinearKernelOutput::Both);
    kernel.resolveInlinableConstitutives();

    StandardAssembler assembler;
    assembler.setDofMap(dof_map);
    assembler.setConstraints(&constraints);

    std::vector<Real> U0(static_cast<std::size_t>(n_dofs), 0.0);
    assembler.setCurrentSolution(U0);

    DenseMatrixView J(n_dofs);
    DenseVectorView R(n_dofs);
    J.zero();
    R.zero();
    (void)assembler.assembleBoth(mesh, space, space, kernel, J, R);

    std::vector<Real> rhs(static_cast<std::size_t>(n_dofs), 0.0);
    for (GlobalIndex i = 0; i < n_dofs; ++i) {
        rhs[static_cast<std::size_t>(i)] = R.getVectorEntry(i);
    }

    // For multi-point constraints (periodic/MPC), the ConstraintDistributor condenses slave DOFs
    // by redistributing contributions to masters, leaving all-slave rows/cols zero in the
    // assembled matrix. Solve the reduced system (excluding non-Dirichlet constrained DOFs),
    // then recover constrained DOFs via AffineConstraints::distribute().
    std::vector<GlobalIndex> keep_dofs;
    keep_dofs.reserve(static_cast<std::size_t>(n_dofs));
    for (GlobalIndex d = 0; d < n_dofs; ++d) {
        if (!constraints.isConstrained(d)) {
            keep_dofs.push_back(d);
            continue;
        }
        const auto c = constraints.getConstraint(d);
        FE_THROW_IF(!c.has_value(), FEException, "solvePoissonOnQuadGrid: constrained DOF missing constraint view");
        if (c->isDirichlet()) {
            keep_dofs.push_back(d);
        }
    }
    FE_THROW_IF(keep_dofs.empty(), FEException, "solvePoissonOnQuadGrid: no unconstrained DOFs to solve");

    const GlobalIndex n_keep = static_cast<GlobalIndex>(keep_dofs.size());
    DenseMatrixView A_keep(n_keep);
    std::vector<Real> rhs_keep(static_cast<std::size_t>(n_keep), 0.0);
    for (GlobalIndex ii = 0; ii < n_keep; ++ii) {
        const auto gi = keep_dofs[static_cast<std::size_t>(ii)];
        rhs_keep[static_cast<std::size_t>(ii)] = rhs[static_cast<std::size_t>(gi)];
        for (GlobalIndex jj = 0; jj < n_keep; ++jj) {
            const auto gj = keep_dofs[static_cast<std::size_t>(jj)];
            A_keep.addMatrixEntry(ii, jj, J.getMatrixEntry(gi, gj), AddMode::Insert);
        }
    }

    auto solve = solveDenseWithPartialPivoting(std::move(A_keep), std::move(rhs_keep));
    FE_THROW_IF(!solve.ok, FEException, solve.message);
    FE_THROW_IF(static_cast<GlobalIndex>(solve.x.size()) != n_keep, FEException,
                "solvePoissonOnQuadGrid: reduced solution size");

    std::vector<Real> u_full(static_cast<std::size_t>(n_dofs), 0.0);
    for (GlobalIndex ii = 0; ii < n_keep; ++ii) {
        const auto gi = keep_dofs[static_cast<std::size_t>(ii)];
        u_full[static_cast<std::size_t>(gi)] = solve.x[static_cast<std::size_t>(ii)];
    }
    constraints.distribute(u_full.data(), n_dofs);

    return {std::move(u_full), std::move(dof_coords)};
}

[[nodiscard]] Real computeL2Error(const IMeshAccess& mesh,
                                  const spaces::FunctionSpace& space,
                                  const dofs::DofMap& dof_map,
                                  std::span<const Real> u_h,
                                  const std::function<Real(Real, Real)>& u_exact,
                                  int quad_order)
{
    FE_CHECK_ARG(space.space_type() == spaces::SpaceType::H1, "computeL2Error: expected H1 space");

    auto quad = quadrature::QuadratureFactory::create(ElementType::Quad4, quad_order);
    FE_CHECK_NOT_NULL(quad.get(), "computeL2Error: quadrature");

    const auto& element = space.getElement(ElementType::Quad4, /*cell_id=*/0);
    const auto& basis = element.basis();

    std::vector<Real> phi;

    Real sum = 0.0;
    const GlobalIndex n_cells = dof_map.getNumCells();
    for (GlobalIndex c = 0; c < n_cells; ++c) {
        auto mapping = buildAffineMappingForQuad4(mesh, c);
        FE_CHECK_NOT_NULL(mapping.get(), "computeL2Error: mapping");

        const auto cell_dofs = dof_map.getCellDofs(c);
        for (LocalIndex q = 0; q < static_cast<LocalIndex>(quad->num_points()); ++q) {
            const auto& qp = quad->point(static_cast<std::size_t>(q));
            const math::Vector<Real, 3> xi{qp[0], qp[1], qp[2]};

            basis.evaluate_values(xi, phi);
            FE_CHECK_ARG(phi.size() == cell_dofs.size(), "computeL2Error: basis value size mismatch");

            Real uh = 0.0;
            for (std::size_t i = 0; i < cell_dofs.size(); ++i) {
                const auto gd = cell_dofs[i];
                uh += u_h[static_cast<std::size_t>(gd)] * phi[i];
            }

            const auto x = mapping->map_to_physical(xi);
            const Real ue = u_exact(x[0], x[1]);
            const Real e = uh - ue;

            const Real detJ = mapping->jacobian_determinant(xi);
            sum += e * e * quad->weight(static_cast<std::size_t>(q)) * std::abs(detJ);
        }
    }

    return std::sqrt(sum);
}

[[nodiscard]] constraints::AffineConstraints makeDirichletOnBoundary(
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

[[nodiscard]] constraints::AffineConstraints makeDirichletOnYAndPeriodicInX(
    const std::vector<std::array<Real, 3>>& dof_coords,
    const std::function<Real(Real, Real)>& dirichlet_value,
    Real tol = Real(1e-12))
{
    constraints::AffineConstraints c;

    // Dirichlet on y=0 and y=1 (inhomogeneous allowed).
    for (GlobalIndex dof = 0; dof < static_cast<GlobalIndex>(dof_coords.size()); ++dof) {
        const auto& x = dof_coords[static_cast<std::size_t>(dof)];
        if (near(x[1], Real(0.0), tol) || near(x[1], Real(1.0), tol)) {
            c.addDirichlet(dof, dirichlet_value(x[0], x[1]));
        }
    }

    // Periodic in x: map x=1 DOFs to x=0 DOFs at matching y (exclude y-Dirichlet DOFs).
    struct Candidate {
        Real y{0.0};
        GlobalIndex dof{-1};
    };
    std::vector<Candidate> masters;
    masters.reserve(dof_coords.size());

    for (GlobalIndex dof = 0; dof < static_cast<GlobalIndex>(dof_coords.size()); ++dof) {
        const auto& x = dof_coords[static_cast<std::size_t>(dof)];
        if (near(x[0], Real(0.0), tol) &&
            !near(x[1], Real(0.0), tol) && !near(x[1], Real(1.0), tol)) {
            masters.push_back({x[1], dof});
        }
    }

    for (GlobalIndex dof = 0; dof < static_cast<GlobalIndex>(dof_coords.size()); ++dof) {
        const auto& x = dof_coords[static_cast<std::size_t>(dof)];
        if (!near(x[0], Real(1.0), tol) || near(x[1], Real(0.0), tol) || near(x[1], Real(1.0), tol)) {
            continue;
        }

        // Skip if already Dirichlet.
        if (c.isConstrained(dof)) {
            continue;
        }

        GlobalIndex master = -1;
        for (const auto& cand : masters) {
            if (near(cand.y, x[1], tol)) {
                master = cand.dof;
                break;
            }
        }
        FE_THROW_IF(master < 0, FEException,
                    "makeDirichletOnYAndPeriodicInX: missing periodic master DOF");

        c.addLine(dof);
        c.addEntry(dof, master, 1.0);
    }

    c.close();
    return c;
}

[[nodiscard]] std::vector<Real> convergenceRates(const std::vector<Real>& errors)
{
    std::vector<Real> rates;
    if (errors.size() < 2u) return rates;
    rates.reserve(errors.size() - 1u);
    for (std::size_t i = 0; i + 1u < errors.size(); ++i) {
        rates.push_back(std::log(errors[i] / errors[i + 1u]) / std::log(Real(2.0)));
    }
    return rates;
}

} // namespace

TEST(ManufacturedSolutionConvergence, PoissonQuad4_Q1_L2ConvergesAtOrderTwo)
{
    std::vector<Real> errors;
    std::vector<int> Ns = {2, 4, 8};
    errors.reserve(Ns.size());

    for (const int N : Ns) {
        StructuredQuadMeshAccess mesh(N);
        auto topo = buildQuadGridTopology(mesh);

        spaces::H1Space space(ElementType::Quad4, /*order=*/1);
        dofs::DofHandler dof_handler;
        dof_handler.distributeDofs(topo, space);
        dof_handler.finalize();

        const auto& dof_map = dof_handler.getDofMap();
        const auto dof_coords = computeDofCoordinates(mesh, space, dof_map);

        auto constraints = makeDirichletOnBoundary(
            dof_coords,
            [](Real x, Real y) { return uExactPolyBiharmonicLike(x, y); });

        auto out = solvePoissonOnQuadGrid(mesh, topo, /*order=*/1, fRhsPolyBiharmonicLike(), std::move(constraints));

        const Real err = computeL2Error(
            mesh,
            space,
            dof_map,
            out.u,
            [](Real x, Real y) { return uExactPolyBiharmonicLike(x, y); },
            /*quad_order=*/10);

        errors.push_back(err);
    }

    ASSERT_EQ(errors.size(), Ns.size());
    EXPECT_GT(errors[0], errors[1]);
    EXPECT_GT(errors[1], errors[2]);

    const auto rates = convergenceRates(errors);
    ASSERT_EQ(rates.size(), 2u);
    EXPECT_GT(rates[0], Real(1.8));
    EXPECT_GT(rates[1], Real(1.8));
}

TEST(ManufacturedSolutionConvergence, PoissonQuad4_Q2_L2ConvergesAtOrderThree)
{
    std::vector<Real> errors;
    std::vector<int> Ns = {2, 4, 8};
    errors.reserve(Ns.size());

    for (const int N : Ns) {
        StructuredQuadMeshAccess mesh(N);
        auto topo = buildQuadGridTopology(mesh);

        spaces::H1Space space(ElementType::Quad4, /*order=*/2);
        dofs::DofHandler dof_handler;
        dof_handler.distributeDofs(topo, space);
        dof_handler.finalize();

        const auto& dof_map = dof_handler.getDofMap();
        const auto dof_coords = computeDofCoordinates(mesh, space, dof_map);

        auto constraints = makeDirichletOnBoundary(
            dof_coords,
            [](Real x, Real y) { return uExactPolyNotInQ2(x, y); });

        auto out = solvePoissonOnQuadGrid(mesh, topo, /*order=*/2, fRhsPolyNotInQ2(), std::move(constraints));

        const Real err = computeL2Error(
            mesh,
            space,
            dof_map,
            out.u,
            [](Real x, Real y) { return uExactPolyNotInQ2(x, y); },
            /*quad_order=*/10);

        errors.push_back(err);
    }

    ASSERT_EQ(errors.size(), Ns.size());
    EXPECT_GT(errors[0], errors[1]);
    EXPECT_GT(errors[1], errors[2]);

    const auto rates = convergenceRates(errors);
    ASSERT_EQ(rates.size(), 2u);
    EXPECT_GT(rates[0], Real(2.6));
    EXPECT_GT(rates[1], Real(2.6));
}

TEST(ConstrainedSolve, PeriodicXAndInhomogeneousDirichletY_Q2PolynomialSolvesToRoundoff)
{
    constexpr int N = 4;

    StructuredQuadMeshAccess mesh(N);
    auto topo = buildQuadGridTopology(mesh);

    spaces::H1Space space(ElementType::Quad4, /*order=*/2);
    dofs::DofHandler dof_handler;
    dof_handler.distributeDofs(topo, space);
    dof_handler.finalize();

    const auto& dof_map = dof_handler.getDofMap();
    const auto dof_coords = computeDofCoordinates(mesh, space, dof_map);

    auto constraints = makeDirichletOnYAndPeriodicInX(
        dof_coords,
        [](Real x, Real y) { return uExactPeriodicInX_Q2(x, y); });

    auto out = solvePoissonOnQuadGrid(mesh, topo, /*order=*/2, fRhsPeriodicInX_Q2(), std::move(constraints));

    // Verify solution matches exact at DOF points (polynomial lies in Q2 space).
    const Real tol = 5e-11;
    for (GlobalIndex dof = 0; dof < static_cast<GlobalIndex>(out.u.size()); ++dof) {
        const auto& x = dof_coords[static_cast<std::size_t>(dof)];
        EXPECT_NEAR(out.u[static_cast<std::size_t>(dof)], uExactPeriodicInX_Q2(x[0], x[1]), tol);
    }

    // Verify periodic DOFs: x=1 matches x=0 for interior y.
    for (GlobalIndex dof_s = 0; dof_s < static_cast<GlobalIndex>(dof_coords.size()); ++dof_s) {
        const auto& x = dof_coords[static_cast<std::size_t>(dof_s)];
        if (!near(x[0], Real(1.0)) || near(x[1], Real(0.0)) || near(x[1], Real(1.0))) {
            continue;
        }

        GlobalIndex dof_m = -1;
        for (GlobalIndex dof = 0; dof < static_cast<GlobalIndex>(dof_coords.size()); ++dof) {
            const auto& xm = dof_coords[static_cast<std::size_t>(dof)];
            if (near(xm[0], Real(0.0)) && near(xm[1], x[1])) {
                dof_m = dof;
                break;
            }
        }
        ASSERT_GE(dof_m, 0);
        EXPECT_NEAR(out.u[static_cast<std::size_t>(dof_s)], out.u[static_cast<std::size_t>(dof_m)], tol);
    }
}

} // namespace testing
} // namespace assembly
} // namespace FE
} // namespace svmp

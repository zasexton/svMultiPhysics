/**
 * @file test_ParallelConvergenceRateMPI.cpp
 * @brief MPI end-to-end accuracy: Poisson convergence rates on distributed assembly.
 *
 * This test complements serial manufactured-solution convergence by exercising:
 * - distributed DOF numbering/ownership
 * - ParallelAssembler assembly + constraint distribution
 * - ghost policies (OwnedRowsOnly vs ReverseScatter) under constraints
 *
 * We keep the mesh/problem small so this runs quickly under CTest MPI (2 and 4 ranks).
 */

#include <gtest/gtest.h>

#include "Assembly/GlobalSystemView.h"
#include "Assembly/ParallelAssembler.h"
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

#include <mpi.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
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

int mpiRank(MPI_Comm comm)
{
    int r = 0;
    MPI_Comm_rank(comm, &r);
    return r;
}

int mpiSize(MPI_Comm comm)
{
    int s = 1;
    MPI_Comm_size(comm, &s);
    return s;
}

MPI_Datatype mpiRealType()
{
    if (sizeof(Real) == sizeof(double)) {
        return MPI_DOUBLE;
    }
    if (sizeof(Real) == sizeof(float)) {
        return MPI_FLOAT;
    }
    return MPI_LONG_DOUBLE;
}

std::vector<Real> allreduceSum(std::span<const Real> local, MPI_Comm comm)
{
    std::vector<Real> global(local.size(), Real(0.0));
    const int n = static_cast<int>(local.size());
    MPI_Allreduce(local.data(), global.data(), n, mpiRealType(), MPI_SUM, comm);
    return global;
}

Real maxAbsDiff(std::span<const Real> a, std::span<const Real> b)
{
    EXPECT_EQ(a.size(), b.size());
    Real m = 0.0;
    const std::size_t n = std::min(a.size(), b.size());
    for (std::size_t i = 0; i < n; ++i) {
        m = std::max(m, static_cast<Real>(std::abs(a[i] - b[i])));
    }
    return m;
}

std::vector<int> neighborRanks(int my_rank, int world_size)
{
    std::vector<int> neighbors;
    neighbors.reserve(static_cast<std::size_t>(std::max(0, world_size - 1)));
    for (int r = 0; r < world_size; ++r) {
        if (r != my_rank) {
            neighbors.push_back(r);
        }
    }
    return neighbors;
}

class StructuredQuadMeshAccess final : public IMeshAccess {
public:
    StructuredQuadMeshAccess(int n_cells_per_axis,
                             std::vector<int> cell_owner_ranks,
                             int my_rank)
        : n_(n_cells_per_axis),
          cell_owner_ranks_(std::move(cell_owner_ranks)),
          my_rank_(my_rank)
    {
        FE_THROW_IF(n_ <= 0, InvalidArgumentException, "StructuredQuadMeshAccess: n must be >= 1");

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

        owned_cells_.clear();
        owned_cells_.reserve(cells_.size());
        for (GlobalIndex c = 0; c < numCells(); ++c) {
            if (isOwnedCell(c)) {
                owned_cells_.push_back(c);
            }
        }
    }

    [[nodiscard]] GlobalIndex numCells() const override { return static_cast<GlobalIndex>(cells_.size()); }
    [[nodiscard]] GlobalIndex numOwnedCells() const override { return static_cast<GlobalIndex>(owned_cells_.size()); }
    [[nodiscard]] GlobalIndex numBoundaryFaces() const override { return 0; }
    [[nodiscard]] GlobalIndex numInteriorFaces() const override { return 0; }
    [[nodiscard]] int dimension() const override { return 2; }

    [[nodiscard]] bool isOwnedCell(GlobalIndex cell_id) const override
    {
        return cell_owner_ranks_.at(static_cast<std::size_t>(cell_id)) == my_rank_;
    }

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
        for (const auto c : owned_cells_) {
            callback(c);
        }
    }

    void forEachBoundaryFace(int /*marker*/,
                             std::function<void(GlobalIndex, GlobalIndex)> /*callback*/) const override
    {
    }

    void forEachInteriorFace(std::function<void(GlobalIndex, GlobalIndex, GlobalIndex)> /*callback*/) const override
    {
    }

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
    std::vector<int> cell_owner_ranks_{};
    int my_rank_{0};
    std::vector<GlobalIndex> owned_cells_{};
};

[[nodiscard]] std::vector<int> partitionCellsRoundRobin(GlobalIndex n_cells, int world_size)
{
    std::vector<int> owners(static_cast<std::size_t>(n_cells), 0);
    for (GlobalIndex c = 0; c < n_cells; ++c) {
        owners[static_cast<std::size_t>(c)] = (world_size > 0) ? static_cast<int>(c % world_size) : 0;
    }
    return owners;
}

[[nodiscard]] dofs::MeshTopologyInfo buildQuadGridTopology(int n_cells_per_axis,
                                                           std::span<const int> cell_owner_ranks,
                                                           int my_rank,
                                                           int world_size)
{
    const int n = n_cells_per_axis;
    const GlobalIndex n_cells = static_cast<GlobalIndex>(n * n);
    const GlobalIndex n_vertices = static_cast<GlobalIndex>((n + 1) * (n + 1));

    dofs::MeshTopologyInfo topo;
    topo.n_cells = n_cells;
    topo.n_vertices = n_vertices;
    topo.dim = 2;

    topo.cell2vertex_offsets.resize(static_cast<std::size_t>(n_cells) + 1u, 0);
    topo.cell2vertex_data.resize(static_cast<std::size_t>(n_cells) * 4u, 0);

    auto nodeId = [n](int i, int j) -> GlobalIndex {
        const int n_nodes_1d = n + 1;
        return static_cast<GlobalIndex>(i + n_nodes_1d * j);
    };
    auto cellId = [n](int i, int j) -> GlobalIndex { return static_cast<GlobalIndex>(i + n * j); };

    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i) {
            const GlobalIndex c = cellId(i, j);
            topo.cell2vertex_offsets[static_cast<std::size_t>(c)] = static_cast<MeshOffset>(4u * c);
            const auto base = static_cast<std::size_t>(4u * c);
            topo.cell2vertex_data[base + 0u] = static_cast<MeshIndex>(nodeId(i, j));
            topo.cell2vertex_data[base + 1u] = static_cast<MeshIndex>(nodeId(i + 1, j));
            topo.cell2vertex_data[base + 2u] = static_cast<MeshIndex>(nodeId(i + 1, j + 1));
            topo.cell2vertex_data[base + 3u] = static_cast<MeshIndex>(nodeId(i, j + 1));
        }
    }
    topo.cell2vertex_offsets[static_cast<std::size_t>(n_cells)] = static_cast<MeshOffset>(4u * n_cells);

    topo.vertex_gids.resize(static_cast<std::size_t>(n_vertices), 0);
    for (GlobalIndex v = 0; v < n_vertices; ++v) {
        topo.vertex_gids[static_cast<std::size_t>(v)] = static_cast<dofs::gid_t>(v);
    }

    topo.cell_gids.resize(static_cast<std::size_t>(n_cells), 0);
    topo.cell_owner_ranks.assign(cell_owner_ranks.begin(), cell_owner_ranks.end());
    for (GlobalIndex c = 0; c < n_cells; ++c) {
        topo.cell_gids[static_cast<std::size_t>(c)] = static_cast<dofs::gid_t>(c);
    }

    topo.neighbor_ranks = neighborRanks(my_rank, world_size);
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
    const auto one = FormExpr::constant(Real(1.0));

    // u = x(1-x)y(1-y)
    // -Δu = 2 y(1-y) + 2 x(1-x)
    const auto g = xx * (one - xx);
    const auto h = yy * (one - yy);
    return FormExpr::constant(Real(2.0)) * h + FormExpr::constant(Real(2.0)) * g;
}

[[nodiscard]] forms::FormExpr fRhsPolyNotInQ2()
{
    using svmp::FE::forms::FormExpr;
    using svmp::FE::forms::x;

    const auto X = x();
    const auto xx = X.component(0);
    const auto yy = X.component(1);
    const auto one = FormExpr::constant(Real(1.0));

    // u = x^3(1-x) y(1-y)
    // Let g(x)=x^3(1-x)=x^3-x^4, h(y)=y(1-y)
    // g''(x)=6x - 12x^2, h''(y)=-2
    // -Δu = -(g'' h + g h'') = -g'' h + 2 g
    const auto g = (xx * xx * xx) * (one - xx);
    const auto h = yy * (one - yy);
    const auto gpp = FormExpr::constant(Real(6.0)) * xx - FormExpr::constant(Real(12.0)) * xx * xx;
    return (-gpp * h) + FormExpr::constant(Real(2.0)) * g;
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

struct GlobalSystemData {
    std::vector<Real> matrix;
    std::vector<Real> vector;
};

[[nodiscard]] GlobalSystemData assemblePoissonSystemMPI(const IMeshAccess& mesh,
                                                        const dofs::DofHandler& dof_handler,
                                                        const spaces::FunctionSpace& space,
                                                        const constraints::AffineConstraints& constraints,
                                                        const forms::FormExpr& rhs_f,
                                                        MPI_Comm comm,
                                                        GhostPolicy policy)
{
    const auto n_dofs = dof_handler.getNumDofs();
    FE_THROW_IF(n_dofs <= 0, FEException, "assemblePoissonSystemMPI: invalid DOF count");

    forms::FormCompiler compiler;
    const auto u = forms::TrialFunction(space, "u");
    const auto v = forms::TestFunction(space, "v");
    const auto residual = (forms::inner(forms::grad(u), forms::grad(v)) + rhs_f * v).dx();

    auto ir = compiler.compileResidual(residual);
    forms::SymbolicNonlinearFormKernel kernel(std::move(ir), forms::NonlinearKernelOutput::Both);
    kernel.resolveInlinableConstitutives();

    ParallelAssembler assembler;
    assembler.setComm(comm);
    assembler.setDofHandler(dof_handler);
    assembler.setConstraints(&constraints);

    AssemblyOptions opts;
    opts.ghost_policy = policy;
    opts.deterministic = true;
    opts.overlap_communication = false;
    assembler.setOptions(opts);

    std::vector<Real> U0(static_cast<std::size_t>(n_dofs), 0.0);
    assembler.setCurrentSolution(U0);
    assembler.initialize();

    DenseMatrixView J_local(n_dofs);
    DenseVectorView R_local(n_dofs);
    J_local.zero();
    R_local.zero();
    (void)assembler.assembleBoth(mesh, space, space, kernel, J_local, R_local);
    assembler.finalize(&J_local, &R_local);

    return {
        allreduceSum(J_local.data(), comm),
        allreduceSum(R_local.data(), comm),
    };
}

[[nodiscard]] std::vector<Real> solvePoissonFromGlobalSystem(const GlobalSystemData& sys,
                                                             const constraints::AffineConstraints& constraints)
{
    const GlobalIndex n = static_cast<GlobalIndex>(std::sqrt(static_cast<double>(sys.matrix.size())));
    FE_THROW_IF(n * n != static_cast<GlobalIndex>(sys.matrix.size()), FEException,
                "solvePoissonFromGlobalSystem: matrix size not square");
    FE_THROW_IF(static_cast<GlobalIndex>(sys.vector.size()) != n, FEException,
                "solvePoissonFromGlobalSystem: rhs size mismatch");

    DenseMatrixView A(n);
    std::copy(sys.matrix.begin(), sys.matrix.end(), A.dataMutable().begin());

    auto solve = solveDenseWithPartialPivoting(std::move(A), sys.vector);
    FE_THROW_IF(!solve.ok, FEException, solve.message);
    FE_THROW_IF(static_cast<GlobalIndex>(solve.x.size()) != n, FEException, "solvePoissonFromGlobalSystem: solve size");

    constraints.distribute(solve.x.data(), n);
    return std::move(solve.x);
}

} // namespace

TEST(ParallelConvergenceRateMPI, PoissonQuad4_Q1Q2_L2ConvergesAtExpectedRatesAndGhostPoliciesAgree)
{
    MPI_Comm comm = MPI_COMM_WORLD;
    const int rank = mpiRank(comm);
    const int size = mpiSize(comm);
    if (size < 2) {
        GTEST_SKIP() << "Run with 2+ MPI ranks to enable this test";
    }

    auto run = [&](int order,
                   std::vector<int> Ns,
                   const std::function<Real(Real, Real)>& u_exact,
                   const forms::FormExpr& rhs_f,
                   Real rate_min) {
        std::vector<Real> errors;
        errors.reserve(Ns.size());

        for (const int N : Ns) {
            const GlobalIndex n_cells = static_cast<GlobalIndex>(N * N);
            auto owners = partitionCellsRoundRobin(n_cells, size);

            StructuredQuadMeshAccess mesh(N, owners, rank);
            auto topo = buildQuadGridTopology(N, owners, rank, size);

            spaces::H1Space space(ElementType::Quad4, order);
            dofs::DofHandler dof_handler;
            dofs::DofDistributionOptions dof_opts;
            // Keep topology requirements minimal for structured test meshes: for p>1 this avoids
            // requiring explicit edge_gids/face_gids while still exercising distributed assembly.
            dof_opts.global_numbering = dofs::GlobalNumberingMode::OwnerContiguous;
            dof_opts.ownership = dofs::OwnershipStrategy::VertexGID;
            dof_opts.my_rank = rank;
            dof_opts.world_size = size;
            dof_opts.mpi_comm = comm;
            dof_handler.distributeDofs(topo, space, dof_opts);
            dof_handler.finalize();

            const auto& dof_map = dof_handler.getDofMap();
            const auto dof_coords = computeDofCoordinates(mesh, space, dof_map);

            const auto constraints = makeDirichletOnBoundary(dof_coords, u_exact);

            const auto owned_rows = assemblePoissonSystemMPI(mesh, dof_handler, space, constraints, rhs_f, comm,
                                                             GhostPolicy::OwnedRowsOnly);
            const auto reverse_scatter = assemblePoissonSystemMPI(mesh, dof_handler, space, constraints, rhs_f, comm,
                                                                  GhostPolicy::ReverseScatter);

            constexpr Real tol = 1e-12;
            EXPECT_LT(maxAbsDiff(owned_rows.matrix, reverse_scatter.matrix), tol);
            EXPECT_LT(maxAbsDiff(owned_rows.vector, reverse_scatter.vector), tol);

            const auto u_h = solvePoissonFromGlobalSystem(owned_rows, constraints);

            const Real err = computeL2Error(mesh, space, dof_map, u_h, u_exact, /*quad_order=*/10);
            errors.push_back(err);
        }

        ASSERT_EQ(errors.size(), Ns.size());
        ASSERT_GE(errors.size(), 2u);
        EXPECT_GT(errors[0], errors[1]);

        const Real rate = std::log(errors[0] / errors[1]) / std::log(Real(2.0));
        EXPECT_GT(rate, rate_min);
    };

    run(/*order=*/1,
        /*Ns=*/{4, 8},
        [](Real x, Real y) { return uExactPolyBiharmonicLike(x, y); },
        fRhsPolyBiharmonicLike(),
        /*rate_min=*/Real(1.8));

    run(/*order=*/2,
        /*Ns=*/{2, 4},
        [](Real x, Real y) { return uExactPolyNotInQ2(x, y); },
        fRhsPolyNotInQ2(),
        /*rate_min=*/Real(2.6));
}

} // namespace testing
} // namespace assembly
} // namespace FE
} // namespace svmp

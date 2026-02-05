/**
 * @file test_JacobianVerificationMPI.cpp
 * @brief MPI regression: Jacobian-vector product matches finite-difference directional derivative under distributed transient assembly.
 *
 * Motivation: MPI-only nonlinear convergence failures can stem from subtle Jacobian/residual inconsistencies
 * (e.g., missing cross-rank contributions or stale history usage). This test validates that the assembled
 * Jacobian is the derivative of the assembled residual for a small distributed problem with dt(u) terms.
 */

#include <gtest/gtest.h>

#include "Assembly/Assembler.h"
#include "Assembly/GlobalSystemView.h"

#include "Core/FEException.h"
#include "Core/Types.h"

#include "Dofs/DofHandler.h"

#include "Forms/Forms.h"
#include "Forms/Vocabulary.h"

#include "Spaces/SpaceFactory.h"

#include "Systems/FESystem.h"
#include "Systems/FormsInstaller.h"
#include "Systems/TransientSystem.h"
#include "Systems/TimeIntegrator.h"

#include <mpi.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace svmp::FE::assembly::testing {
namespace {

using svmp::FE::GlobalIndex;
using svmp::FE::Real;

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
    if (sizeof(Real) == sizeof(double)) return MPI_DOUBLE;
    if (sizeof(Real) == sizeof(float)) return MPI_FLOAT;
    return MPI_LONG_DOUBLE;
}

std::vector<Real> allreduceSum(std::span<const Real> local, MPI_Comm comm)
{
    std::vector<Real> global(local.size(), Real(0.0));
    const int n = static_cast<int>(local.size());
    MPI_Allreduce(local.data(), global.data(), n, mpiRealType(), MPI_SUM, comm);
    return global;
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

Real l2Norm(std::span<const Real> v)
{
    long double s = 0.0L;
    for (const auto& x : v) {
        const long double xx = static_cast<long double>(x);
        s += xx * xx;
    }
    return static_cast<Real>(std::sqrt(s));
}

// 2D strip of Quad4 cells, with interleaved node IDs:
// x-index i has nodes {2*i (bottom), 2*i+1 (top)}.
class StripQuadMeshAccess final : public IMeshAccess {
public:
    StripQuadMeshAccess(int n_cells, int my_rank)
        : n_cells_(n_cells)
        , my_rank_(my_rank)
    {
        FE_THROW_IF(n_cells_ < 1, InvalidArgumentException, "StripQuadMeshAccess: n_cells must be >= 1");

        const int n_x = n_cells_ + 1;
        const int n_nodes = 2 * n_x;
        nodes_.resize(static_cast<std::size_t>(n_nodes));

        for (int i = 0; i < n_x; ++i) {
            const Real x = static_cast<Real>(i) / static_cast<Real>(n_cells_);
            nodes_[static_cast<std::size_t>(2 * i + 0)] = {x, 0.0, 0.0}; // bottom
            nodes_[static_cast<std::size_t>(2 * i + 1)] = {x, 1.0, 0.0}; // top
        }

        cells_.resize(static_cast<std::size_t>(n_cells_));
        for (int c = 0; c < n_cells_; ++c) {
            const GlobalIndex bl = static_cast<GlobalIndex>(2 * c + 0);
            const GlobalIndex br = static_cast<GlobalIndex>(2 * (c + 1) + 0);
            const GlobalIndex tr = static_cast<GlobalIndex>(2 * (c + 1) + 1);
            const GlobalIndex tl = static_cast<GlobalIndex>(2 * c + 1);
            cells_[static_cast<std::size_t>(c)] = {bl, br, tr, tl};
        }
    }

    [[nodiscard]] GlobalIndex numCells() const override { return static_cast<GlobalIndex>(cells_.size()); }
    [[nodiscard]] GlobalIndex numOwnedCells() const override { return 1; }
    [[nodiscard]] GlobalIndex numBoundaryFaces() const override { return 0; }
    [[nodiscard]] GlobalIndex numInteriorFaces() const override { return 0; }
    [[nodiscard]] int dimension() const override { return 2; }

    [[nodiscard]] bool isOwnedCell(GlobalIndex cell_id) const override
    {
        return static_cast<int>(cell_id) == my_rank_;
    }

    [[nodiscard]] ElementType getCellType(GlobalIndex /*cell_id*/) const override { return ElementType::Quad4; }

    void getCellNodes(GlobalIndex cell_id, std::vector<GlobalIndex>& nodes) const override
    {
        const auto& c = cells_.at(static_cast<std::size_t>(cell_id));
        nodes.assign(c.begin(), c.end());
    }

    [[nodiscard]] std::array<Real, 3> getNodeCoordinates(GlobalIndex node_id) const override
    {
        return nodes_.at(static_cast<std::size_t>(node_id));
    }

    void getCellCoordinates(GlobalIndex cell_id,
                            std::vector<std::array<Real, 3>>& coords) const override
    {
        const auto& c = cells_.at(static_cast<std::size_t>(cell_id));
        coords.resize(c.size());
        for (std::size_t i = 0; i < c.size(); ++i) {
            coords[i] = nodes_.at(static_cast<std::size_t>(c[i]));
        }
    }

    [[nodiscard]] LocalIndex getLocalFaceIndex(GlobalIndex /*face_id*/, GlobalIndex /*cell_id*/) const override { return 0; }
    [[nodiscard]] int getBoundaryFaceMarker(GlobalIndex /*face_id*/) const override { return -1; }
    [[nodiscard]] std::pair<GlobalIndex, GlobalIndex> getInteriorFaceCells(GlobalIndex /*face_id*/) const override { return {0, 0}; }

    void forEachCell(std::function<void(GlobalIndex)> callback) const override
    {
        for (GlobalIndex c = 0; c < numCells(); ++c) callback(c);
    }

    void forEachOwnedCell(std::function<void(GlobalIndex)> callback) const override
    {
        callback(static_cast<GlobalIndex>(my_rank_));
    }

    void forEachBoundaryFace(int /*marker*/,
                             std::function<void(GlobalIndex, GlobalIndex)> /*callback*/) const override
    {
    }

    void forEachInteriorFace(std::function<void(GlobalIndex, GlobalIndex, GlobalIndex)> /*callback*/) const override
    {
    }

private:
    int n_cells_{0};
    int my_rank_{0};
    std::vector<std::array<Real, 3>> nodes_{};
    std::vector<std::array<GlobalIndex, 4>> cells_{};
};

[[nodiscard]] dofs::MeshTopologyInfo buildStripTopology(int n_cells, int my_rank, int world_size)
{
    dofs::MeshTopologyInfo topo;
    topo.dim = 2;
    topo.n_cells = static_cast<GlobalIndex>(n_cells);
    topo.n_vertices = static_cast<GlobalIndex>(2 * (n_cells + 1));

    topo.cell2vertex_offsets.resize(static_cast<std::size_t>(topo.n_cells) + 1, 0);
    topo.cell2vertex_data.resize(static_cast<std::size_t>(topo.n_cells) * 4);
    for (int c = 0; c < n_cells; ++c) {
        const std::size_t off = static_cast<std::size_t>(4 * c);
        topo.cell2vertex_offsets[static_cast<std::size_t>(c)] = static_cast<MeshOffset>(off);
        topo.cell2vertex_data[off + 0] = static_cast<MeshIndex>(2 * c + 0);
        topo.cell2vertex_data[off + 1] = static_cast<MeshIndex>(2 * (c + 1) + 0);
        topo.cell2vertex_data[off + 2] = static_cast<MeshIndex>(2 * (c + 1) + 1);
        topo.cell2vertex_data[off + 3] = static_cast<MeshIndex>(2 * c + 1);
    }
    topo.cell2vertex_offsets[static_cast<std::size_t>(topo.n_cells)] =
        static_cast<MeshOffset>(topo.cell2vertex_data.size());

    topo.vertex_gids.resize(static_cast<std::size_t>(topo.n_vertices));
    for (GlobalIndex v = 0; v < topo.n_vertices; ++v) {
        topo.vertex_gids[static_cast<std::size_t>(v)] = static_cast<dofs::gid_t>(v);
    }

    topo.cell_gids.resize(static_cast<std::size_t>(topo.n_cells));
    topo.cell_owner_ranks.resize(static_cast<std::size_t>(topo.n_cells));
    for (int c = 0; c < n_cells; ++c) {
        topo.cell_gids[static_cast<std::size_t>(c)] = static_cast<dofs::gid_t>(c);
        topo.cell_owner_ranks[static_cast<std::size_t>(c)] = c;
    }

    topo.neighbor_ranks = neighborRanks(my_rank, world_size);
    return topo;
}

std::unique_ptr<systems::FESystem> buildNonlinearTransientSystem(std::shared_ptr<const IMeshAccess> mesh,
                                                                 const dofs::MeshTopologyInfo& topo,
                                                                 MPI_Comm comm,
                                                                 int rank,
                                                                 int size,
                                                                 GhostPolicy ghost_policy)
{
    const auto u_space = spaces::VectorSpace(spaces::SpaceType::H1,
                                             mesh,
                                             /*order=*/1,
                                             /*components=*/2);
    const auto p_space = spaces::Space(spaces::SpaceType::H1,
                                       mesh,
                                       /*order=*/1,
                                       /*components=*/1);

    auto sys = std::make_unique<systems::FESystem>(std::move(mesh));
    const auto u_field = sys->addField(systems::FieldSpec{.name = "u", .space = u_space, .components = 2});
    const auto p_field = sys->addField(systems::FieldSpec{.name = "p", .space = p_space, .components = 1});
    sys->addOperator("op");

    const auto u = forms::FormExpr::stateField(u_field, *u_space, "u");
    const auto p = forms::FormExpr::stateField(p_field, *p_space, "p");
    const auto v = forms::TestFunction(*u_space, "v");
    const auto q = forms::TestFunction(*p_space, "q");

    const auto one = forms::FormExpr::constant(Real(1.0));
    const auto lambda = forms::FormExpr::constant(Real(2.0));
    const auto eps = forms::FormExpr::constant(Real(0.05));
    const auto kappa = forms::FormExpr::constant(Real(1.0));

    forms::BlockLinearForm residual(/*tests=*/2);
    residual.setBlock(
        0,
        (forms::inner(u.dt(1), v) +
         lambda * forms::inner(u, v) +
         eps * (one + forms::inner(u, u)) * forms::inner(u, v))
            .dx());
    residual.setBlock(1, (kappa * p * q).dx());

    const std::array<FieldId, 2> fields = {u_field, p_field};
    (void)systems::installCoupledResidual(
        *sys, "op",
        fields,
        fields,
        residual,
        systems::FormInstallOptions{.ad_mode = forms::ADMode::Forward});

    systems::SetupOptions setup_opts;
    setup_opts.assembler_name = "StandardAssembler";
    setup_opts.assembly_options.ghost_policy = ghost_policy;
    setup_opts.assembly_options.deterministic = true;
    setup_opts.assembly_options.overlap_communication = false;

    setup_opts.dof_options.global_numbering = dofs::GlobalNumberingMode::OwnerContiguous;
    setup_opts.dof_options.ownership = dofs::OwnershipStrategy::VertexGID;
    setup_opts.dof_options.my_rank = rank;
    setup_opts.dof_options.world_size = size;
    setup_opts.dof_options.mpi_comm = comm;

    systems::SetupInputs inputs;
    inputs.topology_override = topo;
    sys->setup(setup_opts, inputs);

    return sys;
}

} // namespace

TEST(JacobianVerificationMPI, JacobianTimesVectorMatchesFiniteDifference)
{
    MPI_Comm comm = MPI_COMM_WORLD;
    const int rank = mpiRank(comm);
    const int size = mpiSize(comm);
    if (size < 2) {
        GTEST_SKIP() << "Run with 2+ MPI ranks to enable this test";
    }

    const int n_cells = size;
    auto mesh = std::make_shared<StripQuadMeshAccess>(n_cells, rank);
    const auto topo = buildStripTopology(n_cells, rank, size);

    auto sys = buildNonlinearTransientSystem(mesh, topo, comm, rank, size, GhostPolicy::ReverseScatter);
    ASSERT_TRUE(sys->isSetup());

    const auto n_dofs = sys->dofHandler().getNumDofs();
    ASSERT_GT(n_dofs, 0);

    // Base transient state (replicated across ranks for test simplicity).
    constexpr double dt = 0.1;
    std::vector<Real> U(static_cast<std::size_t>(n_dofs), 0.0);
    std::vector<Real> U_prev(static_cast<std::size_t>(n_dofs), 0.0);
    std::vector<Real> U_prev2(static_cast<std::size_t>(n_dofs), 0.0);
    std::vector<Real> du(static_cast<std::size_t>(n_dofs), 0.0);
    for (GlobalIndex i = 0; i < n_dofs; ++i) {
        const Real base = static_cast<Real>(0.01) * static_cast<Real>(i + 1);
        U[static_cast<std::size_t>(i)] = base;
        U_prev[static_cast<std::size_t>(i)] = static_cast<Real>(0.9) * base;
        U_prev2[static_cast<std::size_t>(i)] = static_cast<Real>(0.8) * base;
        du[static_cast<std::size_t>(i)] = static_cast<Real>(std::sin(static_cast<double>(i + 1)));
    }

    systems::SystemStateView state;
    state.time = 0.25;
    state.dt = dt;
    state.dt_prev = dt;
    state.u = U;
    state.u_prev = U_prev;
    state.u_prev2 = U_prev2;

    auto integrator = std::make_shared<systems::BDF2Integrator>();
    systems::TransientSystem transient(*sys, std::move(integrator));

    systems::AssemblyRequest req;
    req.op = "op";
    req.want_matrix = true;
    req.want_vector = true;

    DenseSystemView out_base(n_dofs);
    out_base.zero();
    (void)transient.assemble(req, state, &out_base, &out_base);
    const auto J = allreduceSum(out_base.matrixData(), comm);
    const auto r0 = allreduceSum(out_base.vectorData(), comm);

    // Perturbed residual r(U + eps*du).
    const Real eps_fd = static_cast<Real>(1e-7);
    std::vector<Real> U_pert = U;
    for (std::size_t i = 0; i < U_pert.size(); ++i) {
        U_pert[i] += eps_fd * du[i];
    }

    systems::SystemStateView state_pert = state;
    state_pert.u = U_pert;

    DenseSystemView out_pert(n_dofs);
    out_pert.zero();
    (void)transient.assemble(req, state_pert, &out_pert, &out_pert);
    const auto r1 = allreduceSum(out_pert.vectorData(), comm);

    std::vector<Real> fd(static_cast<std::size_t>(n_dofs), 0.0);
    for (GlobalIndex i = 0; i < n_dofs; ++i) {
        fd[static_cast<std::size_t>(i)] =
            (r1[static_cast<std::size_t>(i)] - r0[static_cast<std::size_t>(i)]) / eps_fd;
    }

    std::vector<Real> jdu(static_cast<std::size_t>(n_dofs), 0.0);
    for (GlobalIndex i = 0; i < n_dofs; ++i) {
        Real sum = 0.0;
        const std::size_t row = static_cast<std::size_t>(i);
        for (GlobalIndex j = 0; j < n_dofs; ++j) {
            sum += J[row * static_cast<std::size_t>(n_dofs) + static_cast<std::size_t>(j)] *
                du[static_cast<std::size_t>(j)];
        }
        jdu[row] = sum;
    }

    std::vector<Real> diff(static_cast<std::size_t>(n_dofs), 0.0);
    for (std::size_t i = 0; i < diff.size(); ++i) {
        diff[i] = jdu[i] - fd[i];
    }

    if (rank == 0) {
        const Real denom = std::max<Real>(l2Norm(fd), static_cast<Real>(1e-30));
        const Real rel = l2Norm(diff) / denom;
        EXPECT_LT(rel, 1e-6);
    }
}

} // namespace svmp::FE::assembly::testing

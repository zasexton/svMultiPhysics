/**
 * @file test_FESystemSerialParallelEquivalenceMPI.cpp
 * @brief MPI accuracy tests for FE/Systems end-to-end assembly: FESystem Standard vs Parallel.
 *
 * This is a regression test for multi-field nonlinear workflows (e.g. Navier–Stokes)
 * where MPI runs exhibited Newton convergence issues. It verifies that:
 * - FESystem assembled residual/Jacobian match a rank-0 "serial" reference
 * - OwnedRowsOnly and ReverseScatter ghost policies are numerically identical
 * - Each MPI rank writes only to its owned rows (distributed-backend requirement)
 */

#include <gtest/gtest.h>

#include "Assembly/GlobalSystemView.h"
#include "Forms/Vocabulary.h"
#include "Spaces/SpaceFactory.h"
#include "Systems/FESystem.h"
#include "Systems/OperatorBackends.h"
#include "Systems/CoupledBoundaryManager.h"
#include "Systems/FormsInstaller.h"

#include <mpi.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <functional>
#include <memory>
#include <span>
#include <vector>

namespace svmp::FE::assembly::testing {
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

Real maxAbsRow(const DenseSystemView& sys, GlobalIndex row)
{
    Real m = 0.0;
    for (GlobalIndex j = 0; j < sys.numCols(); ++j) {
        m = std::max(m, static_cast<Real>(std::abs(sys.getMatrixEntry(row, j))));
    }
    return m;
}

void expectOnlyOwnedRowsAreWritten(const dofs::DofMap& dof_map,
                                  const DenseSystemView& out,
                                  Real tol)
{
    ASSERT_EQ(out.numRows(), dof_map.getNumDofs());
    ASSERT_EQ(out.numCols(), dof_map.getNumDofs());

    for (GlobalIndex i = 0; i < dof_map.getNumDofs(); ++i) {
        if (dof_map.isOwnedDof(i)) {
            continue;
        }
        EXPECT_LT(maxAbsRow(out, i), tol) << "Non-owned matrix row " << i << " was written";
        EXPECT_LT(std::abs(out.getVectorEntry(i)), tol) << "Non-owned vector row " << i << " was written";
    }
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

class FourTetraMeshAccess final : public IMeshAccess {
public:
    FourTetraMeshAccess(std::vector<int> cell_owner_ranks, int my_rank)
        : cell_owner_ranks_(std::move(cell_owner_ranks)),
          my_rank_(my_rank)
    {
        nodes_ = {
            {0.0, 0.0, 0.0},  // 0
            {1.0, 0.0, 0.0},  // 1
            {0.0, 1.0, 0.0},  // 2
            {0.0, 0.0, 1.0},  // 3
            {1.0, 1.0, 0.0},  // 4
            {1.0, 0.2, 1.0}   // 5
        };

        cells_ = {
            std::array<GlobalIndex, 4>{0, 1, 2, 3},  // 0
            std::array<GlobalIndex, 4>{1, 2, 3, 4},  // 1
            std::array<GlobalIndex, 4>{1, 3, 4, 5},  // 2
            std::array<GlobalIndex, 4>{2, 3, 4, 5}   // 3
        };

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
    [[nodiscard]] int dimension() const override { return 3; }

    [[nodiscard]] bool isOwnedCell(GlobalIndex cell_id) const override
    {
        return cell_owner_ranks_.at(static_cast<std::size_t>(cell_id)) == my_rank_;
    }

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
    std::vector<std::array<Real, 3>> nodes_{};
    std::vector<std::array<GlobalIndex, 4>> cells_{};
    std::vector<int> cell_owner_ranks_{};
    int my_rank_{0};
    std::vector<GlobalIndex> owned_cells_{};
};

class AllCellsOwnedOnRank0MeshAccess final : public IMeshAccess {
public:
    AllCellsOwnedOnRank0MeshAccess(std::shared_ptr<const FourTetraMeshAccess> base, int my_rank)
        : base_(std::move(base)), my_rank_(my_rank)
    {
        FE_CHECK_NOT_NULL(base_.get(), "AllCellsOwnedOnRank0MeshAccess: base");
    }

    [[nodiscard]] GlobalIndex numCells() const override { return base_->numCells(); }
    [[nodiscard]] GlobalIndex numOwnedCells() const override { return (my_rank_ == 0) ? base_->numCells() : 0; }
    [[nodiscard]] GlobalIndex numBoundaryFaces() const override { return base_->numBoundaryFaces(); }
    [[nodiscard]] GlobalIndex numInteriorFaces() const override { return base_->numInteriorFaces(); }
    [[nodiscard]] int dimension() const override { return base_->dimension(); }

    [[nodiscard]] bool isOwnedCell(GlobalIndex /*cell_id*/) const override { return my_rank_ == 0; }
    [[nodiscard]] ElementType getCellType(GlobalIndex cell_id) const override { return base_->getCellType(cell_id); }

    void getCellNodes(GlobalIndex cell_id, std::vector<GlobalIndex>& nodes) const override
    {
        base_->getCellNodes(cell_id, nodes);
    }

    [[nodiscard]] std::array<Real, 3> getNodeCoordinates(GlobalIndex node_id) const override
    {
        return base_->getNodeCoordinates(node_id);
    }

    void getCellCoordinates(GlobalIndex cell_id,
                            std::vector<std::array<Real, 3>>& coords) const override
    {
        base_->getCellCoordinates(cell_id, coords);
    }

    [[nodiscard]] LocalIndex getLocalFaceIndex(GlobalIndex face_id,
                                               GlobalIndex cell_id) const override
    {
        return base_->getLocalFaceIndex(face_id, cell_id);
    }

    [[nodiscard]] int getBoundaryFaceMarker(GlobalIndex face_id) const override
    {
        return base_->getBoundaryFaceMarker(face_id);
    }

    [[nodiscard]] std::pair<GlobalIndex, GlobalIndex> getInteriorFaceCells(GlobalIndex face_id) const override
    {
        return base_->getInteriorFaceCells(face_id);
    }

    void forEachCell(std::function<void(GlobalIndex)> callback) const override { base_->forEachCell(std::move(callback)); }

    void forEachOwnedCell(std::function<void(GlobalIndex)> callback) const override
    {
        if (my_rank_ != 0) return;
        base_->forEachCell(std::move(callback));
    }

    void forEachBoundaryFace(int marker,
                             std::function<void(GlobalIndex, GlobalIndex)> callback) const override
    {
        base_->forEachBoundaryFace(marker, std::move(callback));
    }

    void forEachInteriorFace(std::function<void(GlobalIndex, GlobalIndex, GlobalIndex)> callback) const override
    {
        base_->forEachInteriorFace(std::move(callback));
    }

private:
    std::shared_ptr<const FourTetraMeshAccess> base_;
    int my_rank_{0};
};

dofs::MeshTopologyInfo buildFourTetraTopology(std::span<const int> cell_owner_ranks,
                                             int my_rank,
                                             int world_size)
{
    dofs::MeshTopologyInfo topo;
    topo.n_cells = 4;
    topo.n_vertices = 6;
    topo.dim = 3;
    topo.cell2vertex_offsets = {0, 4, 8, 12, 16};
    topo.cell2vertex_data = {
        0, 1, 2, 3,
        1, 2, 3, 4,
        1, 3, 4, 5,
        2, 3, 4, 5,
    };
    topo.vertex_gids = {0, 1, 2, 3, 4, 5};
    topo.cell_gids = {0, 1, 2, 3};
    topo.cell_owner_ranks.assign(cell_owner_ranks.begin(), cell_owner_ranks.end());
    topo.neighbor_ranks = neighborRanks(my_rank, world_size);
    return topo;
}

std::vector<int> partitionTetraCellsRoundRobin(int world_size)
{
    std::vector<int> owners(4, 0);
    for (int c = 0; c < 4; ++c) {
        owners[static_cast<std::size_t>(c)] = (world_size > 0) ? (c % world_size) : 0;
    }
    return owners;
}

systems::FESystem buildCoupledNavierStokesLikeSystem(std::shared_ptr<const IMeshAccess> mesh,
                                                     const dofs::MeshTopologyInfo& topo,
                                                     MPI_Comm comm,
                                                     int rank,
                                                     int size,
                                                     std::string assembler_name,
                                                     GhostPolicy ghost_policy)
{
    auto u_space = spaces::VectorSpace(spaces::SpaceType::H1,
                                       mesh,
                                       /*order=*/1,
                                       /*components=*/3);
    auto p_space = spaces::Space(spaces::SpaceType::H1,
                                 mesh,
                                 /*order=*/1,
                                 /*components=*/1);

    systems::FESystem sys(std::move(mesh));
    const auto u_field = sys.addField(systems::FieldSpec{.name = "u", .space = u_space, .components = 3});
    const auto p_field = sys.addField(systems::FieldSpec{.name = "p", .space = p_space, .components = 1});
    sys.addOperator("ns");

    const auto u_state = forms::FormExpr::stateField(u_field, *u_space, "u");
    const auto p_state = forms::FormExpr::stateField(p_field, *p_space, "p");

    const auto v = forms::TestFunction(*u_space, "v");
    const auto q = forms::TestFunction(*p_space, "q");

    const Real nu = 0.01;
    const auto nu_c = forms::FormExpr::constant(nu);

    // Momentum residual: convection + diffusion - pressure coupling.
    forms::BlockLinearForm residual(/*tests=*/2);
    residual.setBlock(
        0,
        (forms::inner(forms::grad(u_state) * u_state, v) +
         nu_c * forms::inner(forms::grad(u_state), forms::grad(v)) -
         p_state * forms::div(v))
            .dx());
    // Continuity residual.
    residual.setBlock(1, (q * forms::div(u_state)).dx());

    const std::array<FieldId, 2> fields = {u_field, p_field};
    (void)systems::installCoupledResidual(
        sys, "ns",
        fields,
        fields,
        residual,
        systems::FormInstallOptions{.ad_mode = forms::ADMode::Forward});

    systems::SetupOptions opts;
    opts.assembler_name = std::move(assembler_name);
    opts.assembly_options.ghost_policy = ghost_policy;
    opts.assembly_options.deterministic = true;
    opts.assembly_options.overlap_communication = false;

    opts.dof_options.global_numbering = dofs::GlobalNumberingMode::GlobalIds;
    opts.dof_options.ownership = dofs::OwnershipStrategy::VertexGID;
    opts.dof_options.my_rank = rank;
    opts.dof_options.world_size = size;
    opts.dof_options.mpi_comm = comm;

    systems::SetupInputs inputs;
    inputs.topology_override = topo;
    sys.setup(opts, inputs);

    return sys;
}

struct GlobalSystemData {
    std::vector<Real> matrix;
    std::vector<Real> vector;
};

GlobalSystemData assembleAndAllreduce(systems::FESystem& sys,
                                      const systems::SystemStateView& state,
                                      MPI_Comm comm,
                                      bool check_owned_rows)
{
    const auto n_dofs = sys.dofHandler().getNumDofs();
    DenseSystemView out(n_dofs);
    out.zero();

    systems::AssemblyRequest req;
    req.op = "ns";
    req.want_matrix = true;
    req.want_vector = true;

    const auto result = sys.assemble(req, state, &out, &out);
    EXPECT_TRUE(result.success);

    if (check_owned_rows) {
        expectOnlyOwnedRowsAreWritten(sys.dofHandler().getDofMap(), out, /*tol=*/1e-14);
    }

    return {
        allreduceSum(out.matrixData(), comm),
        allreduceSum(out.vectorData(), comm),
    };
}

} // namespace

TEST(FESystemSerialParallelEquivalenceMPI, CoupledNavierStokesLikeAssemblyMatchesSerialAndGhostPoliciesAgree)
{
    MPI_Comm comm = MPI_COMM_WORLD;
    const int rank = mpiRank(comm);
    const int size = mpiSize(comm);
    if (size < 2) {
        GTEST_SKIP() << "Run with 2+ MPI ranks to enable this test";
    }

    const auto cell_owners = partitionTetraCellsRoundRobin(size);
    auto mesh_parallel = std::make_shared<FourTetraMeshAccess>(cell_owners, rank);
    const auto topo = buildFourTetraTopology(cell_owners, rank, size);

    // Rank-0 "serial" reference: assemble all cells on rank 0 using StandardAssembler,
    // but keep the same distributed DOF numbering/ownership derived from topo.
    auto mesh_serial = std::make_shared<AllCellsOwnedOnRank0MeshAccess>(mesh_parallel, rank);
    auto sys_ref = buildCoupledNavierStokesLikeSystem(mesh_serial,
                                                      topo,
                                                      comm,
                                                      rank,
                                                      size,
                                                      /*assembler_name=*/"StandardAssembler",
                                                      /*ghost_policy=*/GhostPolicy::ReverseScatter);

    auto sys_owned = buildCoupledNavierStokesLikeSystem(mesh_parallel,
                                                        topo,
                                                        comm,
                                                        rank,
                                                        size,
                                                        /*assembler_name=*/"ParallelAssembler",
                                                        /*ghost_policy=*/GhostPolicy::OwnedRowsOnly);
    auto sys_scatter = buildCoupledNavierStokesLikeSystem(mesh_parallel,
                                                          topo,
                                                          comm,
                                                          rank,
                                                          size,
                                                          /*assembler_name=*/"ParallelAssembler",
                                                          /*ghost_policy=*/GhostPolicy::ReverseScatter);

    ASSERT_EQ(sys_ref.dofHandler().getNumDofs(), sys_owned.dofHandler().getNumDofs());
    ASSERT_EQ(sys_ref.dofHandler().getNumDofs(), sys_scatter.dofHandler().getNumDofs());

    const auto n_dofs = sys_ref.dofHandler().getNumDofs();
    ASSERT_EQ(n_dofs, 24);

    // Global state vector (replicated across ranks for test simplicity).
    std::vector<Real> U(static_cast<std::size_t>(n_dofs), 0.0);
    for (GlobalIndex i = 0; i < n_dofs; ++i) {
        U[static_cast<std::size_t>(i)] = static_cast<Real>(0.03) * static_cast<Real>(i + 1);
    }
    systems::SystemStateView state;
    state.u = U;

    const auto ref = assembleAndAllreduce(sys_ref, state, comm, /*check_owned_rows=*/false);

    const auto owned = assembleAndAllreduce(sys_owned, state, comm, /*check_owned_rows=*/true);
    const auto scatter = assembleAndAllreduce(sys_scatter, state, comm, /*check_owned_rows=*/true);

    if (rank == 0) {
        constexpr Real tol = 1e-12;
        EXPECT_LT(maxAbsDiff(ref.matrix, owned.matrix), tol);
        EXPECT_LT(maxAbsDiff(ref.vector, owned.vector), tol);
        EXPECT_LT(maxAbsDiff(ref.matrix, scatter.matrix), tol);
        EXPECT_LT(maxAbsDiff(ref.vector, scatter.vector), tol);

        EXPECT_LT(maxAbsDiff(owned.matrix, scatter.matrix), tol);
        EXPECT_LT(maxAbsDiff(owned.vector, scatter.vector), tol);
    }
}

} // namespace svmp::FE::assembly::testing

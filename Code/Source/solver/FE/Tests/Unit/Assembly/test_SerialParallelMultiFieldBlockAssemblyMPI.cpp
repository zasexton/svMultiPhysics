/**
 * @file test_SerialParallelMultiFieldBlockAssemblyMPI.cpp
 * @brief MPI accuracy tests: multi-field block assembly (row/col DOF offsets) matches serial.
 *
 * This test targets the Systems-style workflow where assembly is performed block-by-block via:
 *   assembler.setRowDofMap(field_map, field_offset);
 *   assembler.setColDofMap(field_map, field_offset);
 *   assembler.assemble*(...);
 * followed by a single finalize().
 *
 * It verifies:
 * - ReverseScatter and OwnedRowsOnly ghost policies assemble the same global matrix/vector
 * - Results match a rank-0 serial reference (same global numbering)
 * - Each MPI rank only writes to its owned rows (critical for distributed backends)
 */

#include <gtest/gtest.h>

#include "Assembly/GlobalSystemView.h"
#include "Assembly/ParallelAssembler.h"
#include "Assembly/StandardAssembler.h"
#include "Dofs/DofHandler.h"
#include "Forms/FormCompiler.h"
#include "Forms/FormKernels.h"
#include "Forms/Vocabulary.h"
#include "Spaces/SpaceFactory.h"
#include "Systems/FESystem.h"

#include <mpi.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <functional>
#include <numeric>
#include <span>
#include <vector>

using svmp::FE::ElementType;
using svmp::FE::GlobalIndex;
using svmp::FE::LocalIndex;
using svmp::FE::Real;

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

Real maxAbsRow(const DenseMatrixView& A, GlobalIndex row)
{
    Real m = 0.0;
    for (GlobalIndex j = 0; j < A.numCols(); ++j) {
        m = std::max(m, static_cast<Real>(std::abs(A.getMatrixEntry(row, j))));
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

struct AssemblyDenseResult {
    std::vector<Real> matrix;
    std::vector<Real> vector;
};

void assembleStokesLikeBlocks(Assembler& assembler,
                              const IMeshAccess& mesh,
                              const spaces::FunctionSpace& u_space,
                              const spaces::FunctionSpace& p_space,
                              const dofs::DofMap& u_map,
                              GlobalIndex u_offset,
                              const dofs::DofMap& p_map,
                              GlobalIndex p_offset,
                              forms::FormKernel& uu_kernel,
                              forms::FormKernel& up_kernel,
                              forms::FormKernel& pu_kernel,
                              forms::FormKernel& pp_kernel,
                              forms::FormKernel& bu_kernel,
                              forms::FormKernel& bp_kernel,
                              DenseMatrixView& A,
                              DenseVectorView& b)
{
    // u-u
    assembler.setRowDofMap(u_map, u_offset);
    assembler.setColDofMap(u_map, u_offset);
    (void)assembler.assembleMatrix(mesh, u_space, u_space, uu_kernel, A);

    // u-p
    assembler.setRowDofMap(u_map, u_offset);
    assembler.setColDofMap(p_map, p_offset);
    (void)assembler.assembleMatrix(mesh, u_space, p_space, up_kernel, A);

    // p-u
    assembler.setRowDofMap(p_map, p_offset);
    assembler.setColDofMap(u_map, u_offset);
    (void)assembler.assembleMatrix(mesh, p_space, u_space, pu_kernel, A);

    // p-p (simple mass/stabilization-like term so the block is nonzero)
    assembler.setRowDofMap(p_map, p_offset);
    assembler.setColDofMap(p_map, p_offset);
    (void)assembler.assembleMatrix(mesh, p_space, p_space, pp_kernel, A);

    // Vector RHS blocks (explicitly reset col maps to avoid stale rectangular config)
    assembler.setRowDofMap(u_map, u_offset);
    assembler.setColDofMap(u_map, u_offset);
    (void)assembler.assembleVector(mesh, u_space, bu_kernel, b);

    assembler.setRowDofMap(p_map, p_offset);
    assembler.setColDofMap(p_map, p_offset);
    (void)assembler.assembleVector(mesh, p_space, bp_kernel, b);
}

void expectOnlyOwnedRowsAreWritten(const dofs::DofMap& dof_map,
                                  const DenseMatrixView& A,
                                  const DenseVectorView& b,
                                  Real tol)
{
    ASSERT_EQ(A.numRows(), dof_map.getNumDofs());
    ASSERT_EQ(b.numRows(), dof_map.getNumDofs());

    for (GlobalIndex i = 0; i < dof_map.getNumDofs(); ++i) {
        if (dof_map.isOwnedDof(i)) {
            continue;
        }
        EXPECT_LT(maxAbsRow(A, i), tol) << "Non-owned matrix row " << i << " was written";
        EXPECT_LT(std::abs(b.getVectorEntry(i)), tol) << "Non-owned vector row " << i << " was written";
    }
}

} // namespace

TEST(SerialParallelMultiFieldBlockAssemblyMPI, StokesBlocksMatchSerialAndGhostPoliciesAndHonorOwnership)
{
    MPI_Comm comm = MPI_COMM_WORLD;
    const int rank = mpiRank(comm);
    const int size = mpiSize(comm);
    if (size < 2) {
        GTEST_SKIP() << "Run with 2+ MPI ranks to enable this test";
    }

    const auto cell_owners = partitionTetraCellsRoundRobin(size);
    auto mesh = std::make_shared<FourTetraMeshAccess>(cell_owners, rank);
    const auto topo = buildFourTetraTopology(cell_owners, rank, size);

    auto u_space_ptr = spaces::VectorSpace(spaces::SpaceType::H1,
                                           mesh,
                                           /*order=*/1,
                                           /*components=*/3);
    auto p_space_ptr = spaces::Space(spaces::SpaceType::H1,
                                     mesh,
                                     /*order=*/1,
                                     /*components=*/1);

    systems::FESystem sys(mesh);
    const auto u_field = sys.addField(systems::FieldSpec{.name = "u", .space = u_space_ptr, .components = 3});
    const auto p_field = sys.addField(systems::FieldSpec{.name = "p", .space = p_space_ptr, .components = 1});

    systems::SetupOptions setup_opts;
    setup_opts.dof_options.numbering = dofs::DofNumberingStrategy::Sequential;
    setup_opts.dof_options.global_numbering = dofs::GlobalNumberingMode::GlobalIds;
    setup_opts.dof_options.ownership = dofs::OwnershipStrategy::VertexGID;
    setup_opts.dof_options.my_rank = rank;
    setup_opts.dof_options.world_size = size;
    setup_opts.dof_options.mpi_comm = comm;

    systems::SetupInputs inputs;
    inputs.topology_override = topo;
    sys.setup(setup_opts, inputs);

    const auto n_dofs = sys.dofHandler().getNumDofs();
    ASSERT_EQ(n_dofs, 24) << "Unexpected monolithic DOF count for 4 tetra mesh with (u,p)=(P1^3,P1)";

    const auto& u_map = sys.fieldDofHandler(u_field).getDofMap();
    const auto& p_map = sys.fieldDofHandler(p_field).getDofMap();
    const auto u_offset = sys.fieldDofOffset(u_field);
    const auto p_offset = sys.fieldDofOffset(p_field);
    ASSERT_EQ(u_offset, 0);
    ASSERT_EQ(p_offset, u_map.getNumDofs());
    ASSERT_EQ(u_map.getNumDofs(), 18);
    ASSERT_EQ(p_map.getNumDofs(), 6);

    // Build representative linear multi-field blocks with nontrivial off-diagonals.
    forms::FormCompiler compiler;
    const auto u = forms::TrialFunction(*u_space_ptr, "u");
    const auto v = forms::TestFunction(*u_space_ptr, "v");
    const auto p = forms::TrialFunction(*p_space_ptr, "p");
    const auto q = forms::TestFunction(*p_space_ptr, "q");

    const Real nu = 0.01;
    const auto nu_c = forms::FormExpr::constant(nu);

    auto uu_ir = compiler.compileBilinear((nu_c * forms::inner(forms::grad(u), forms::grad(v))).dx());
    auto up_ir = compiler.compileBilinear((-p * forms::div(v)).dx());
    auto pu_ir = compiler.compileBilinear((q * forms::div(u)).dx());
    auto pp_ir = compiler.compileBilinear((forms::FormExpr::constant(Real(0.1)) * p * q).dx());

    const auto X = forms::x();
    const auto one = forms::FormExpr::constant(Real(1.0));
    const auto two = forms::FormExpr::constant(Real(2.0));
    const auto three = forms::FormExpr::constant(Real(3.0));
    const auto f = forms::as_vector({X.component(0) + one,
                                     X.component(1) + two,
                                     X.component(2) + three});
    auto bu_ir = compiler.compileLinear((forms::inner(f, v)).dx());
    auto bp_ir = compiler.compileLinear(((X.component(0) - X.component(1)) * q).dx());

    forms::FormKernel uu_kernel(std::move(uu_ir));
    forms::FormKernel up_kernel(std::move(up_ir));
    forms::FormKernel pu_kernel(std::move(pu_ir));
    forms::FormKernel pp_kernel(std::move(pp_ir));
    forms::FormKernel bu_kernel(std::move(bu_ir));
    forms::FormKernel bp_kernel(std::move(bp_ir));
    uu_kernel.resolveInlinableConstitutives();
    up_kernel.resolveInlinableConstitutives();
    pu_kernel.resolveInlinableConstitutives();
    pp_kernel.resolveInlinableConstitutives();
    bu_kernel.resolveInlinableConstitutives();
    bp_kernel.resolveInlinableConstitutives();

    // Serial reference on rank 0: treat all cells as owned by rank 0, but keep the same global numbering.
    AssemblyDenseResult ref;
    if (rank == 0) {
        std::vector<int> all_owned(cell_owners.size(), 0);
        FourTetraMeshAccess serial_mesh(all_owned, /*my_rank=*/0);

        DenseMatrixView A_ref(n_dofs);
        DenseVectorView b_ref(n_dofs);
        A_ref.zero();
        b_ref.zero();

        StandardAssembler assembler;
        assembler.setDofHandler(sys.dofHandler());
        assembleStokesLikeBlocks(assembler,
                                 serial_mesh,
                                 *u_space_ptr,
                                 *p_space_ptr,
                                 u_map,
                                 u_offset,
                                 p_map,
                                 p_offset,
                                 uu_kernel,
                                 up_kernel,
                                 pu_kernel,
                                 pp_kernel,
                                 bu_kernel,
                                 bp_kernel,
                                 A_ref,
                                 b_ref);
        assembler.finalize(&A_ref, &b_ref);

        ref.matrix.assign(A_ref.data().begin(), A_ref.data().end());
        ref.vector.assign(b_ref.data().begin(), b_ref.data().end());
    }

    auto assemble_parallel = [&](GhostPolicy policy) -> AssemblyDenseResult {
        ParallelAssembler assembler;
        assembler.setComm(comm);
        assembler.setDofHandler(sys.dofHandler());

        AssemblyOptions opts;
        opts.ghost_policy = policy;
        opts.deterministic = true;
        opts.overlap_communication = false;
        assembler.setOptions(opts);
        assembler.initialize();

        DenseMatrixView A_local(n_dofs);
        DenseVectorView b_local(n_dofs);
        A_local.zero();
        b_local.zero();

        assembleStokesLikeBlocks(assembler,
                                 *mesh,
                                 *u_space_ptr,
                                 *p_space_ptr,
                                 u_map,
                                 u_offset,
                                 p_map,
                                 p_offset,
                                 uu_kernel,
                                 up_kernel,
                                 pu_kernel,
                                 pp_kernel,
                                 bu_kernel,
                                 bp_kernel,
                                 A_local,
                                 b_local);
        assembler.finalize(&A_local, &b_local);

        // Local correctness: distributed backends assume non-owned rows were untouched.
        expectOnlyOwnedRowsAreWritten(sys.dofHandler().getDofMap(), A_local, b_local, /*tol=*/1e-14);

        return {
            allreduceSum(A_local.data(), comm),
            allreduceSum(b_local.data(), comm),
        };
    };

    const auto owned_rows = assemble_parallel(GhostPolicy::OwnedRowsOnly);
    const auto reverse_scatter = assemble_parallel(GhostPolicy::ReverseScatter);

    if (rank == 0) {
        constexpr Real tol = 1e-12;
        EXPECT_LT(maxAbsDiff(ref.matrix, owned_rows.matrix), tol);
        EXPECT_LT(maxAbsDiff(ref.vector, owned_rows.vector), tol);
        EXPECT_LT(maxAbsDiff(ref.matrix, reverse_scatter.matrix), tol);
        EXPECT_LT(maxAbsDiff(ref.vector, reverse_scatter.vector), tol);

        EXPECT_LT(maxAbsDiff(owned_rows.matrix, reverse_scatter.matrix), tol);
        EXPECT_LT(maxAbsDiff(owned_rows.vector, reverse_scatter.vector), tol);
    }
}

} // namespace svmp::FE::assembly::testing

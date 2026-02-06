/**
 * @file test_SerialParallelFaceIntegralsMPI.cpp
 * @brief MPI accuracy tests: serial vs parallel equivalence for boundary/interior-face assembly.
 */

#include <gtest/gtest.h>

#include "Assembly/GlobalSystemView.h"
#include "Assembly/ParallelAssembler.h"
#include "Assembly/StandardAssembler.h"
#include "Dofs/DofHandler.h"
#include "Forms/FormCompiler.h"
#include "Forms/FormKernels.h"
#include "Forms/Vocabulary.h"
#include "Spaces/H1Space.h"
#include "Spaces/L2Space.h"

#include <mpi.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <span>
#include <stdexcept>
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

class TwoTetraFaceMeshAccess final : public IMeshAccess {
public:
    TwoTetraFaceMeshAccess(std::vector<int> cell_owner_ranks,
                           int my_rank)
        : cell_owner_ranks_(std::move(cell_owner_ranks)),
          my_rank_(my_rank)
    {
        // Two tetrahedra sharing face {1,2,3}.
        nodes_ = {
            {0.0, 0.0, 0.0},  // 0
            {1.0, 0.0, 0.0},  // 1
            {0.0, 1.0, 0.0},  // 2
            {0.0, 0.0, 1.0},  // 3
            {1.0, 1.0, 1.0}   // 4
        };
        cells_ = {
            std::array<GlobalIndex, 4>{0, 1, 2, 3},  // cell 0
            std::array<GlobalIndex, 4>{1, 2, 3, 4}   // cell 1
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
    [[nodiscard]] GlobalIndex numBoundaryFaces() const override { return 2; }
    [[nodiscard]] GlobalIndex numInteriorFaces() const override { return 1; }
    [[nodiscard]] int dimension() const override { return 3; }

    [[nodiscard]] bool isOwnedCell(GlobalIndex cell_id) const override
    {
        return cell_owner_ranks_.at(static_cast<std::size_t>(cell_id)) == my_rank_;
    }

    [[nodiscard]] ElementType getCellType(GlobalIndex /*cell_id*/) const override
    {
        return ElementType::Tetra4;
    }

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

    [[nodiscard]] LocalIndex getLocalFaceIndex(GlobalIndex face_id,
                                               GlobalIndex cell_id) const override
    {
        // Interior face id = 0: shared face {1,2,3}.
        // - cell 0 local face 0
        // - cell 1 local face 3
        if (face_id == 0) {
            if (cell_id == 0) return 0;
            if (cell_id == 1) return 3;
        }

        // Boundary face id = 1: cell 0 local face 3 (nodes {0,1,2})
        if (face_id == 1 && cell_id == 0) return 3;
        // Boundary face id = 2: cell 1 local face 0 (nodes {2,3,4})
        if (face_id == 2 && cell_id == 1) return 0;

        throw std::runtime_error("TwoTetraFaceMeshAccess::getLocalFaceIndex: invalid face/cell pair");
    }

    [[nodiscard]] int getBoundaryFaceMarker(GlobalIndex face_id) const override
    {
        if (face_id == 1 || face_id == 2) {
            return 2;
        }
        return -1;
    }

    [[nodiscard]] std::pair<GlobalIndex, GlobalIndex> getInteriorFaceCells(GlobalIndex face_id) const override
    {
        if (face_id == 0) {
            return {0, 1};
        }
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

    void forEachBoundaryFace(int marker, std::function<void(GlobalIndex, GlobalIndex)> callback) const override
    {
        if (marker < 0 || marker == 2) {
            callback(1, 0);
            callback(2, 1);
        }
    }

    void forEachInteriorFace(std::function<void(GlobalIndex, GlobalIndex, GlobalIndex)> callback) const override
    {
        callback(0, 0, 1);
    }

private:
    std::vector<std::array<Real, 3>> nodes_{};
    std::vector<std::array<GlobalIndex, 4>> cells_{};
    std::vector<int> cell_owner_ranks_{};
    int my_rank_{0};
    std::vector<GlobalIndex> owned_cells_{};
};

class FourTetraChainFaceMeshAccess final : public IMeshAccess {
public:
    FourTetraChainFaceMeshAccess(std::vector<int> cell_owner_ranks,
                                 int my_rank)
        : cell_owner_ranks_(std::move(cell_owner_ranks)),
          my_rank_(my_rank)
    {
        // Four tetrahedra in a "chain" (cells share successive faces):
        //  c0: (0,1,2,3)
        //  c1: (1,2,3,4) shares face {1,2,3} with c0
        //  c2: (2,3,4,5) shares face {2,3,4} with c1
        //  c3: (3,4,5,6) shares face {3,4,5} with c2
        nodes_ = {
            {0.0, 0.0, 0.0},  // 0
            {1.0, 0.0, 0.0},  // 1
            {0.0, 1.0, 0.0},  // 2
            {0.0, 0.0, 1.0},  // 3
            {1.0, 1.0, 1.0},  // 4
            {2.0, 1.0, 1.0},  // 5
            {2.0, 2.0, 2.0},  // 6
        };

        cells_ = {
            std::array<GlobalIndex, 4>{0, 1, 2, 3},  // cell 0
            std::array<GlobalIndex, 4>{1, 2, 3, 4},  // cell 1
            std::array<GlobalIndex, 4>{2, 3, 4, 5},  // cell 2
            std::array<GlobalIndex, 4>{3, 4, 5, 6},  // cell 3
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
    [[nodiscard]] GlobalIndex numInteriorFaces() const override { return 3; }
    [[nodiscard]] int dimension() const override { return 3; }

    [[nodiscard]] bool isOwnedCell(GlobalIndex cell_id) const override
    {
        return cell_owner_ranks_.at(static_cast<std::size_t>(cell_id)) == my_rank_;
    }

    [[nodiscard]] ElementType getCellType(GlobalIndex /*cell_id*/) const override
    {
        return ElementType::Tetra4;
    }

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

    [[nodiscard]] LocalIndex getLocalFaceIndex(GlobalIndex face_id,
                                               GlobalIndex cell_id) const override
    {
        // Tetra4 reference face order: {0,1,2}, {0,1,3}, {1,2,3}, {0,2,3}.
        // Chain shared faces:
        //  f0: {1,2,3} between c0 and c1 -> c0 face 2, c1 face 0 (local {0,1,2})
        //  f1: {2,3,4} between c1 and c2 -> c1 face 2, c2 face 0
        //  f2: {3,4,5} between c2 and c3 -> c2 face 2, c3 face 0
        if (face_id == 0) {
            if (cell_id == 0) return 2;
            if (cell_id == 1) return 0;
        }
        if (face_id == 1) {
            if (cell_id == 1) return 2;
            if (cell_id == 2) return 0;
        }
        if (face_id == 2) {
            if (cell_id == 2) return 2;
            if (cell_id == 3) return 0;
        }
        throw std::runtime_error("FourTetraChainFaceMeshAccess::getLocalFaceIndex: invalid face/cell pair");
    }

    [[nodiscard]] int getBoundaryFaceMarker(GlobalIndex /*face_id*/) const override { return -1; }

    [[nodiscard]] std::pair<GlobalIndex, GlobalIndex> getInteriorFaceCells(GlobalIndex face_id) const override
    {
        if (face_id == 0) return {0, 1};
        if (face_id == 1) return {1, 2};
        if (face_id == 2) return {2, 3};
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

    void forEachBoundaryFace(int /*marker*/, std::function<void(GlobalIndex, GlobalIndex)> /*callback*/) const override
    {
    }

    void forEachInteriorFace(std::function<void(GlobalIndex, GlobalIndex, GlobalIndex)> callback) const override
    {
        callback(0, 0, 1);
        callback(1, 1, 2);
        callback(2, 2, 3);
    }

private:
    std::vector<std::array<Real, 3>> nodes_{};
    std::vector<std::array<GlobalIndex, 4>> cells_{};
    std::vector<int> cell_owner_ranks_{};
    int my_rank_{0};
    std::vector<GlobalIndex> owned_cells_{};
};

dofs::MeshTopologyInfo buildTwoTetraTopology(std::span<const int> cell_owner_ranks,
                                             int my_rank,
                                             int world_size)
{
    dofs::MeshTopologyInfo topo;
    topo.n_cells = 2;
    topo.n_vertices = 5;
    topo.dim = 3;
    topo.cell2vertex_offsets = {0, 4, 8};
    topo.cell2vertex_data = {
        0, 1, 2, 3,
        1, 2, 3, 4,
    };
    topo.vertex_gids = {0, 1, 2, 3, 4};
    topo.cell_gids = {0, 1};
    topo.cell_owner_ranks.assign(cell_owner_ranks.begin(), cell_owner_ranks.end());
    topo.neighbor_ranks = neighborRanks(my_rank, world_size);
    return topo;
}

[[nodiscard]] dofs::MeshTopologyInfo buildFourTetraChainTopology(std::span<const int> cell_owner_ranks,
                                                                 int my_rank,
                                                                 int world_size)
{
    dofs::MeshTopologyInfo topo;
    topo.n_cells = 4;
    topo.n_vertices = 7;
    topo.dim = 3;
    topo.cell2vertex_offsets = {0, 4, 8, 12, 16};
    topo.cell2vertex_data = {
        0, 1, 2, 3,
        1, 2, 3, 4,
        2, 3, 4, 5,
        3, 4, 5, 6,
    };
    topo.vertex_gids = {0, 1, 2, 3, 4, 5, 6};
    topo.cell_gids = {0, 1, 2, 3};
    topo.cell_owner_ranks.assign(cell_owner_ranks.begin(), cell_owner_ranks.end());
    topo.neighbor_ranks = neighborRanks(my_rank, world_size);
    return topo;
}

std::vector<int> partitionTwoCells(int world_size)
{
    const int owner_cell1 = (world_size > 1) ? 1 : 0;
    return {0, owner_cell1};
}

std::vector<int> partitionFourCellsBlock(int world_size)
{
    std::vector<int> owners(4, 0);
    if (world_size >= 4) {
        owners = {0, 1, 2, 3};
        return owners;
    }
    if (world_size == 3) {
        owners = {0, 1, 2, 2};
        return owners;
    }
    if (world_size == 2) {
        owners = {0, 0, 1, 1};
        return owners;
    }
    return owners;
}

struct GlobalAssemblyResult {
    std::vector<Real> matrix;
    std::vector<Real> vector;
};

} // namespace

TEST(SerialParallelFaceIntegralsMPI, BoundaryAssemblyMatchesSerialAndGhostPoliciesAgree)
{
    MPI_Comm comm = MPI_COMM_WORLD;
    const int rank = mpiRank(comm);
    const int size = mpiSize(comm);
    if (size < 2) {
        GTEST_SKIP() << "Run with 2+ MPI ranks to enable this test";
    }

    const auto cell_owners = partitionTwoCells(size);
    TwoTetraFaceMeshAccess mesh(cell_owners, rank);
    const auto topo = buildTwoTetraTopology(cell_owners, rank, size);

    spaces::H1Space space(ElementType::Tetra4, /*order=*/1);
    dofs::DofHandler dof_handler;
    dofs::DofDistributionOptions dof_opts;
    dof_opts.global_numbering = dofs::GlobalNumberingMode::GlobalIds;
    dof_opts.ownership = dofs::OwnershipStrategy::VertexGID;
    dof_opts.my_rank = rank;
    dof_opts.world_size = size;
    dof_opts.mpi_comm = comm;
    dof_handler.distributeDofs(topo, space, dof_opts);
    dof_handler.finalize();

    const GlobalIndex n_dofs = dof_handler.getNumDofs();
    ASSERT_EQ(n_dofs, 5);

    forms::FormCompiler compiler;
    const auto u = forms::TrialFunction(space, "u");
    const auto v = forms::TestFunction(space, "v");
    const auto n = forms::FormExpr::normal();
    const auto residual = (forms::inner(forms::grad(u), n) * v).ds(2);

    auto ir = compiler.compileResidual(residual);
    forms::NonlinearFormKernel kernel(std::move(ir), forms::ADMode::Forward, forms::NonlinearKernelOutput::Both);
    kernel.resolveInlinableConstitutives();

    std::vector<Real> U(static_cast<std::size_t>(n_dofs), Real(0.0));
    for (GlobalIndex i = 0; i < n_dofs; ++i) {
        U[static_cast<std::size_t>(i)] = static_cast<Real>(0.1) * static_cast<Real>(i + 1);
    }

    GlobalAssemblyResult ref;
    if (rank == 0) {
        std::vector<int> all_owned = {0, 0};
        TwoTetraFaceMeshAccess serial_mesh(all_owned, /*my_rank=*/0);

        DenseMatrixView J_ref(n_dofs);
        DenseVectorView R_ref(n_dofs);
        J_ref.zero();
        R_ref.zero();

        StandardAssembler assembler;
        assembler.setDofHandler(dof_handler);
        assembler.setCurrentSolution(U);
        (void)assembler.assembleBoundaryFaces(serial_mesh, /*marker=*/2, space, kernel, &J_ref, &R_ref);
        assembler.finalize(&J_ref, &R_ref);

        ref.matrix.assign(J_ref.data().begin(), J_ref.data().end());
        ref.vector.assign(R_ref.data().begin(), R_ref.data().end());
    }

    auto assemble_parallel = [&](GhostPolicy policy) -> GlobalAssemblyResult {
        ParallelAssembler assembler;
        assembler.setComm(comm);
        assembler.setDofHandler(dof_handler);

        AssemblyOptions opts;
        opts.ghost_policy = policy;
        opts.deterministic = true;
        opts.overlap_communication = false;
        assembler.setOptions(opts);
        assembler.setCurrentSolution(U);
        assembler.initialize();

        DenseMatrixView J_local(n_dofs);
        DenseVectorView R_local(n_dofs);
        J_local.zero();
        R_local.zero();
        (void)assembler.assembleBoundaryFaces(mesh, /*marker=*/2, space, kernel, &J_local, &R_local);
        assembler.finalize(&J_local, &R_local);

        return {
            allreduceSum(J_local.data(), comm),
            allreduceSum(R_local.data(), comm),
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

TEST(SerialParallelFaceIntegralsMPI, CombinedCellAndNitscheBoundaryAssemblyMatchesSerialAndGhostPoliciesAgree)
{
    MPI_Comm comm = MPI_COMM_WORLD;
    const int rank = mpiRank(comm);
    const int size = mpiSize(comm);
    if (size < 2) {
        GTEST_SKIP() << "Run with 2+ MPI ranks to enable this test";
    }

    const auto cell_owners = partitionTwoCells(size);
    TwoTetraFaceMeshAccess mesh(cell_owners, rank);
    const auto topo = buildTwoTetraTopology(cell_owners, rank, size);

    spaces::H1Space space(ElementType::Tetra4, /*order=*/1);
    dofs::DofHandler dof_handler;
    dofs::DofDistributionOptions dof_opts;
    dof_opts.global_numbering = dofs::GlobalNumberingMode::GlobalIds;
    dof_opts.ownership = dofs::OwnershipStrategy::VertexGID;
    dof_opts.my_rank = rank;
    dof_opts.world_size = size;
    dof_opts.mpi_comm = comm;
    dof_handler.distributeDofs(topo, space, dof_opts);
    dof_handler.finalize();

    const GlobalIndex n_dofs = dof_handler.getNumDofs();
    ASSERT_EQ(n_dofs, 5);

    forms::FormCompiler compiler;
    const auto u = forms::TrialFunction(space, "u");
    const auto v = forms::TestFunction(space, "v");
    const auto n = forms::FormExpr::normal();
    const auto gamma = forms::FormExpr::constant(Real(30.0));

    const auto residual =
        forms::inner(forms::grad(u), forms::grad(v)).dx() +
        (-forms::inner(forms::grad(u), n) * v - u * forms::inner(forms::grad(v), n) + (gamma / forms::h()) * u * v)
            .ds(2);

    auto ir = compiler.compileResidual(residual);
    forms::NonlinearFormKernel kernel(std::move(ir), forms::ADMode::Forward, forms::NonlinearKernelOutput::Both);
    kernel.resolveInlinableConstitutives();

    std::vector<Real> U(static_cast<std::size_t>(n_dofs), Real(0.0));
    for (GlobalIndex i = 0; i < n_dofs; ++i) {
        U[static_cast<std::size_t>(i)] = static_cast<Real>(0.05) * static_cast<Real>(i + 1);
    }

    GlobalAssemblyResult ref;
    if (rank == 0) {
        std::vector<int> all_owned = {0, 0};
        TwoTetraFaceMeshAccess serial_mesh(all_owned, /*my_rank=*/0);

        DenseMatrixView J_ref(n_dofs);
        DenseVectorView R_ref(n_dofs);
        J_ref.zero();
        R_ref.zero();

        StandardAssembler assembler;
        assembler.setDofHandler(dof_handler);
        assembler.setCurrentSolution(U);
        (void)assembler.assembleBoth(serial_mesh, space, space, kernel, J_ref, R_ref);
        (void)assembler.assembleBoundaryFaces(serial_mesh, /*marker=*/2, space, kernel, &J_ref, &R_ref);
        assembler.finalize(&J_ref, &R_ref);

        ref.matrix.assign(J_ref.data().begin(), J_ref.data().end());
        ref.vector.assign(R_ref.data().begin(), R_ref.data().end());
    }

    auto assemble_parallel = [&](GhostPolicy policy) -> GlobalAssemblyResult {
        ParallelAssembler assembler;
        assembler.setComm(comm);
        assembler.setDofHandler(dof_handler);

        AssemblyOptions opts;
        opts.ghost_policy = policy;
        opts.deterministic = true;
        opts.overlap_communication = false;
        assembler.setOptions(opts);
        assembler.setCurrentSolution(U);
        assembler.initialize();

        DenseMatrixView J_local(n_dofs);
        DenseVectorView R_local(n_dofs);
        J_local.zero();
        R_local.zero();
        (void)assembler.assembleBoth(mesh, space, space, kernel, J_local, R_local);
        (void)assembler.assembleBoundaryFaces(mesh, /*marker=*/2, space, kernel, &J_local, &R_local);
        assembler.finalize(&J_local, &R_local);

        return {
            allreduceSum(J_local.data(), comm),
            allreduceSum(R_local.data(), comm),
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

TEST(SerialParallelFaceIntegralsMPI, DGPenaltyInteriorFaceAssemblyMatchesSerialAndGhostPoliciesAgree)
{
    MPI_Comm comm = MPI_COMM_WORLD;
    const int rank = mpiRank(comm);
    const int size = mpiSize(comm);
    if (size < 2) {
        GTEST_SKIP() << "Run with 2+ MPI ranks to enable this test";
    }

    const auto cell_owners = partitionTwoCells(size);
    TwoTetraFaceMeshAccess mesh(cell_owners, rank);
    const auto topo = buildTwoTetraTopology(cell_owners, rank, size);

    spaces::L2Space space(ElementType::Tetra4, /*order=*/1);
    dofs::DofHandler dof_handler;
    dofs::DofDistributionOptions dof_opts;
    dof_opts.global_numbering = dofs::GlobalNumberingMode::GlobalIds;
    dof_opts.ownership = dofs::OwnershipStrategy::CellOwner;
    dof_opts.my_rank = rank;
    dof_opts.world_size = size;
    dof_opts.mpi_comm = comm;
    dof_handler.distributeDofs(topo, space, dof_opts);
    dof_handler.finalize();

    const GlobalIndex n_dofs = dof_handler.getNumDofs();
    ASSERT_EQ(n_dofs, 8);

    forms::FormCompiler compiler;
    const auto u = forms::TrialFunction(space, "u");
    const auto v = forms::TestFunction(space, "v");
    const auto eta = forms::FormExpr::constant(Real(12.0));
    auto ir = compiler.compileBilinear((eta * forms::inner(forms::jump(u), forms::jump(v))).dS());
    forms::FormKernel kernel(std::move(ir));
    kernel.resolveInlinableConstitutives();

    std::vector<Real> ref_matrix;
    if (rank == 0) {
        std::vector<int> all_owned = {0, 0};
        TwoTetraFaceMeshAccess serial_mesh(all_owned, /*my_rank=*/0);

        DenseMatrixView A_ref(n_dofs);
        A_ref.zero();

        StandardAssembler assembler;
        assembler.setDofHandler(dof_handler);
        (void)assembler.assembleInteriorFaces(serial_mesh, space, space, kernel, A_ref, nullptr);
        assembler.finalize(&A_ref, nullptr);

        ref_matrix.assign(A_ref.data().begin(), A_ref.data().end());
    }

    auto assemble_parallel = [&](GhostPolicy policy) -> std::vector<Real> {
        ParallelAssembler assembler;
        assembler.setComm(comm);
        assembler.setDofHandler(dof_handler);

        AssemblyOptions opts;
        opts.ghost_policy = policy;
        opts.deterministic = true;
        opts.overlap_communication = false;
        assembler.setOptions(opts);
        assembler.initialize();

        DenseMatrixView A_local(n_dofs);
        A_local.zero();
        (void)assembler.assembleInteriorFaces(mesh, space, space, kernel, A_local, nullptr);
        assembler.finalize(&A_local, nullptr);
        return allreduceSum(A_local.data(), comm);
    };

    const auto owned_rows = assemble_parallel(GhostPolicy::OwnedRowsOnly);
    const auto reverse_scatter = assemble_parallel(GhostPolicy::ReverseScatter);

    if (rank == 0) {
        constexpr Real tol = 1e-12;
        EXPECT_LT(maxAbsDiff(ref_matrix, owned_rows), tol);
        EXPECT_LT(maxAbsDiff(ref_matrix, reverse_scatter), tol);
        EXPECT_LT(maxAbsDiff(owned_rows, reverse_scatter), tol);
    }
}

TEST(SerialParallelFaceIntegralsMPI, DGPenaltyInteriorFaceAssemblyOnFourTetraChainMatchesSerial)
{
    MPI_Comm comm = MPI_COMM_WORLD;
    const int rank = mpiRank(comm);
    const int size = mpiSize(comm);
    if (size < 2) {
        GTEST_SKIP() << "Run with 2+ MPI ranks to enable this test";
    }

    const auto cell_owners = partitionFourCellsBlock(size);
    FourTetraChainFaceMeshAccess mesh(cell_owners, rank);
    const auto topo = buildFourTetraChainTopology(cell_owners, rank, size);

    spaces::L2Space space(ElementType::Tetra4, /*order=*/1);
    dofs::DofHandler dof_handler;
    dofs::DofDistributionOptions dof_opts;
    dof_opts.global_numbering = dofs::GlobalNumberingMode::GlobalIds;
    dof_opts.ownership = dofs::OwnershipStrategy::CellOwner;
    dof_opts.my_rank = rank;
    dof_opts.world_size = size;
    dof_opts.mpi_comm = comm;
    dof_handler.distributeDofs(topo, space, dof_opts);
    dof_handler.finalize();

    const GlobalIndex n_dofs = dof_handler.getNumDofs();
    ASSERT_EQ(n_dofs, 16);

    forms::FormCompiler compiler;
    const auto u = forms::TrialFunction(space, "u");
    const auto v = forms::TestFunction(space, "v");
    const auto eta = forms::FormExpr::constant(Real(12.0));
    auto ir = compiler.compileBilinear((eta * forms::inner(forms::jump(u), forms::jump(v))).dS());
    forms::FormKernel kernel(std::move(ir));
    kernel.resolveInlinableConstitutives();

    std::vector<Real> ref_matrix;
    if (rank == 0) {
        std::vector<int> all_owned = {0, 0, 0, 0};
        FourTetraChainFaceMeshAccess serial_mesh(all_owned, /*my_rank=*/0);

        DenseMatrixView A_ref(n_dofs);
        A_ref.zero();

        StandardAssembler assembler;
        assembler.setDofHandler(dof_handler);
        (void)assembler.assembleInteriorFaces(serial_mesh, space, space, kernel, A_ref, nullptr);
        assembler.finalize(&A_ref, nullptr);

        ref_matrix.assign(A_ref.data().begin(), A_ref.data().end());
    }

    auto assemble_parallel = [&](GhostPolicy policy) -> std::vector<Real> {
        ParallelAssembler assembler;
        assembler.setComm(comm);
        assembler.setDofHandler(dof_handler);

        AssemblyOptions opts;
        opts.ghost_policy = policy;
        opts.deterministic = true;
        opts.overlap_communication = false;
        assembler.setOptions(opts);
        assembler.initialize();

        DenseMatrixView A_local(n_dofs);
        A_local.zero();
        (void)assembler.assembleInteriorFaces(mesh, space, space, kernel, A_local, nullptr);
        assembler.finalize(&A_local, nullptr);
        return allreduceSum(A_local.data(), comm);
    };

    const auto owned_rows = assemble_parallel(GhostPolicy::OwnedRowsOnly);
    const auto reverse_scatter = assemble_parallel(GhostPolicy::ReverseScatter);

    if (rank == 0) {
        constexpr Real tol = 1e-12;
        EXPECT_LT(maxAbsDiff(ref_matrix, owned_rows), tol);
        EXPECT_LT(maxAbsDiff(ref_matrix, reverse_scatter), tol);
        EXPECT_LT(maxAbsDiff(owned_rows, reverse_scatter), tol);
    }
}

} // namespace testing
} // namespace assembly
} // namespace FE
} // namespace svmp

/**
 * @file test_SerialParallelEquivalenceMPI.cpp
 * @brief MPI accuracy tests: serial vs parallel equivalence for assembled matrix/vector.
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

#include <mpi.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <span>
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

Real frobeniusNorm(std::span<const Real> a)
{
    long double sum = 0.0L;
    for (const auto v : a) {
        sum += static_cast<long double>(v) * static_cast<long double>(v);
    }
    return static_cast<Real>(std::sqrt(sum));
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

dofs::MeshTopologyInfo buildQuadGridTopology(int n_cells_per_axis,
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

std::vector<int> partitionQuadCellsStripesX(int n_cells_per_axis, int world_size)
{
    const int n = n_cells_per_axis;
    std::vector<int> owners(static_cast<std::size_t>(n * n), 0);
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i) {
            const int owner = std::min(world_size - 1, (i * world_size) / n);
            owners[static_cast<std::size_t>(i + n * j)] = owner;
        }
    }
    return owners;
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
            {1.0, 0.2, 1.0}   // 5 (non-coplanar with {2,3,4})
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

struct GlobalAssemblyResult {
    std::vector<Real> matrix;
    std::vector<Real> vector;
};

GlobalAssemblyResult assembleGlobalSystem(Assembler& assembler,
                                         const IMeshAccess& mesh,
                                         const spaces::FunctionSpace& space,
                                         forms::FormKernel& matrix_kernel,
                                         forms::FormKernel& vector_kernel,
                                         GlobalIndex n_dofs,
                                         MPI_Comm comm)
{
    DenseMatrixView A_local(n_dofs);
    DenseVectorView b_local(n_dofs);
    A_local.zero();
    b_local.zero();

    (void)assembler.assembleMatrix(mesh, space, space, matrix_kernel, A_local);
    (void)assembler.assembleVector(mesh, space, vector_kernel, b_local);
    assembler.finalize(&A_local, &b_local);

    return {
        allreduceSum(A_local.data(), comm),
        allreduceSum(b_local.data(), comm),
    };
}

struct ReferenceAssemblyResult {
    std::vector<Real> matrix;
    std::vector<Real> vector;
};

ReferenceAssemblyResult assembleReferenceSerial(const IMeshAccess& mesh,
                                               const spaces::FunctionSpace& space,
                                               const dofs::DofHandler& dof_handler,
                                               forms::FormKernel& matrix_kernel,
                                               forms::FormKernel& vector_kernel)
{
    const auto n_dofs = dof_handler.getNumDofs();
    DenseMatrixView A(n_dofs);
    DenseVectorView b(n_dofs);
    A.zero();
    b.zero();

    StandardAssembler assembler;
    assembler.setDofHandler(dof_handler);

    (void)assembler.assembleMatrix(mesh, space, space, matrix_kernel, A);
    (void)assembler.assembleVector(mesh, space, vector_kernel, b);
    assembler.finalize(&A, &b);

    return {
        std::vector<Real>(A.data().begin(), A.data().end()),
        std::vector<Real>(b.data().begin(), b.data().end()),
    };
}

void compareAgainstReference(const ReferenceAssemblyResult& ref,
                             const GlobalAssemblyResult& test,
                             Real tol)
{
    const Real matrix_diff = maxAbsDiff(ref.matrix, test.matrix);
    const Real vector_diff = maxAbsDiff(ref.vector, test.vector);

    const Real matrix_norm = frobeniusNorm(ref.matrix);
    const Real vector_norm = frobeniusNorm(ref.vector);

    EXPECT_LT(matrix_diff, tol) << "||A||_F=" << matrix_norm;
    EXPECT_LT(vector_diff, tol) << "||b||_2=" << vector_norm;
}

} // namespace

TEST(SerialParallelEquivalenceMPI, Quad4MatrixAndVectorMatchSerialAndGhostPoliciesAgree)
{
    MPI_Comm comm = MPI_COMM_WORLD;
    const int rank = mpiRank(comm);
    const int size = mpiSize(comm);
    if (size < 2) {
        GTEST_SKIP() << "Run with 2+ MPI ranks to enable this test";
    }

    constexpr int n_cells_per_axis = 8;
    const auto cell_owners = partitionQuadCellsStripesX(n_cells_per_axis, size);

    StructuredQuadMeshAccess mesh(n_cells_per_axis, cell_owners, rank);
    const auto topo = buildQuadGridTopology(n_cells_per_axis, cell_owners, rank, size);

    spaces::H1Space space(ElementType::Quad4, /*order=*/1);
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
    ASSERT_GT(n_dofs, 0);

    // Matrix: Poisson stiffness. Vector: coordinate-dependent load.
    forms::FormCompiler compiler;
    const auto u = forms::TrialFunction(space, "u");
    const auto v = forms::TestFunction(space, "v");
    const auto X = forms::x();
    const auto f = X.component(0) + Real(2.0) * X.component(1);

    auto bilinear_ir = compiler.compileBilinear(forms::inner(forms::grad(u), forms::grad(v)).dx());
    auto linear_ir = compiler.compileLinear((f * v).dx());

    forms::FormKernel matrix_kernel(std::move(bilinear_ir));
    forms::FormKernel vector_kernel(std::move(linear_ir));
    matrix_kernel.resolveInlinableConstitutives();
    vector_kernel.resolveInlinableConstitutives();

    ReferenceAssemblyResult ref;
    if (rank == 0) {
        // Assemble a true serial reference by treating all cells as "owned" on rank 0.
        std::vector<int> all_owned(cell_owners.size(), 0);
        StructuredQuadMeshAccess serial_mesh(n_cells_per_axis, all_owned, /*my_rank=*/0);
        ref = assembleReferenceSerial(serial_mesh, space, dof_handler, matrix_kernel, vector_kernel);
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
        assembler.initialize();

        return assembleGlobalSystem(assembler, mesh, space, matrix_kernel, vector_kernel, n_dofs, comm);
    };

    const auto owned_rows = assemble_parallel(GhostPolicy::OwnedRowsOnly);
    const auto reverse_scatter = assemble_parallel(GhostPolicy::ReverseScatter);

    if (rank == 0) {
        constexpr Real tol = 1e-12;
        compareAgainstReference(ref, owned_rows, tol);
        compareAgainstReference(ref, reverse_scatter, tol);
        EXPECT_LT(maxAbsDiff(owned_rows.matrix, reverse_scatter.matrix), tol);
        EXPECT_LT(maxAbsDiff(owned_rows.vector, reverse_scatter.vector), tol);
    }
}

TEST(SerialParallelEquivalenceMPI, Tetra4MatrixAndVectorMatchSerialAndGhostPoliciesAgree)
{
    MPI_Comm comm = MPI_COMM_WORLD;
    const int rank = mpiRank(comm);
    const int size = mpiSize(comm);
    if (size < 2) {
        GTEST_SKIP() << "Run with 2+ MPI ranks to enable this test";
    }

    const auto cell_owners = partitionTetraCellsRoundRobin(size);
    FourTetraMeshAccess mesh(cell_owners, rank);
    const auto topo = buildFourTetraTopology(cell_owners, rank, size);

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
    ASSERT_EQ(n_dofs, 6);

    forms::FormCompiler compiler;
    const auto u = forms::TrialFunction(space, "u");
    const auto v = forms::TestFunction(space, "v");
    const auto X = forms::x();
    const auto f = X.component(0) - Real(0.5) * X.component(1) + Real(0.25) * X.component(2);

    auto bilinear_ir = compiler.compileBilinear(forms::inner(forms::grad(u), forms::grad(v)).dx());
    auto linear_ir = compiler.compileLinear((f * v).dx());

    forms::FormKernel matrix_kernel(std::move(bilinear_ir));
    forms::FormKernel vector_kernel(std::move(linear_ir));
    matrix_kernel.resolveInlinableConstitutives();
    vector_kernel.resolveInlinableConstitutives();

    ReferenceAssemblyResult ref;
    if (rank == 0) {
        // Assemble a true serial reference by treating all cells as "owned" on rank 0.
        std::vector<int> all_owned(cell_owners.size(), 0);
        FourTetraMeshAccess serial_mesh(all_owned, /*my_rank=*/0);
        ref = assembleReferenceSerial(serial_mesh, space, dof_handler, matrix_kernel, vector_kernel);
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
        assembler.initialize();

        return assembleGlobalSystem(assembler, mesh, space, matrix_kernel, vector_kernel, n_dofs, comm);
    };

    const auto owned_rows = assemble_parallel(GhostPolicy::OwnedRowsOnly);
    const auto reverse_scatter = assemble_parallel(GhostPolicy::ReverseScatter);

    if (rank == 0) {
        constexpr Real tol = 1e-12;
        compareAgainstReference(ref, owned_rows, tol);
        compareAgainstReference(ref, reverse_scatter, tol);
        EXPECT_LT(maxAbsDiff(owned_rows.matrix, reverse_scatter.matrix), tol);
        EXPECT_LT(maxAbsDiff(owned_rows.vector, reverse_scatter.vector), tol);
    }
}

TEST(SerialParallelEquivalenceMPI, NonlinearDiffusionJacobianAndResidualMatchSerialAndGhostPoliciesAgree)
{
    MPI_Comm comm = MPI_COMM_WORLD;
    const int rank = mpiRank(comm);
    const int size = mpiSize(comm);
    if (size < 2) {
        GTEST_SKIP() << "Run with 2+ MPI ranks to enable this test";
    }

    const auto cell_owners = partitionTetraCellsRoundRobin(size);
    FourTetraMeshAccess mesh(cell_owners, rank);
    const auto topo = buildFourTetraTopology(cell_owners, rank, size);

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
    ASSERT_EQ(n_dofs, 6);

    forms::FormCompiler compiler;
    const auto u = forms::TrialFunction(space, "u");
    const auto v = forms::TestFunction(space, "v");
    const auto residual =
        forms::inner((forms::FormExpr::constant(Real(1.0)) + u * u) * forms::grad(u), forms::grad(v)).dx();

    auto residual_ir = compiler.compileResidual(residual);
    forms::NonlinearFormKernel kernel(std::move(residual_ir), forms::ADMode::Forward, forms::NonlinearKernelOutput::Both);
    kernel.resolveInlinableConstitutives();

    // Global solution vector (replicated across ranks for test simplicity).
    const std::vector<Real> U = {0.2, -0.1, 0.05, 0.15, -0.2, 0.12};

    ReferenceAssemblyResult ref;
    if (rank == 0) {
        std::vector<int> all_owned(cell_owners.size(), 0);
        FourTetraMeshAccess serial_mesh(all_owned, /*my_rank=*/0);

        DenseMatrixView J(n_dofs);
        DenseVectorView R(n_dofs);
        J.zero();
        R.zero();

        StandardAssembler assembler;
        assembler.setDofHandler(dof_handler);
        assembler.setCurrentSolution(U);
        (void)assembler.assembleBoth(serial_mesh, space, space, kernel, J, R);
        assembler.finalize(&J, &R);

        ref.matrix.assign(J.data().begin(), J.data().end());
        ref.vector.assign(R.data().begin(), R.data().end());
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
        assembler.finalize(&J_local, &R_local);

        return {
            allreduceSum(J_local.data(), comm),
            allreduceSum(R_local.data(), comm),
        };
    };

    const auto owned_rows = assemble_parallel(GhostPolicy::OwnedRowsOnly);
    const auto reverse_scatter = assemble_parallel(GhostPolicy::ReverseScatter);

    if (rank == 0) {
        constexpr Real tol = 1e-10;
        compareAgainstReference(ref, owned_rows, tol);
        compareAgainstReference(ref, reverse_scatter, tol);
        EXPECT_LT(maxAbsDiff(owned_rows.matrix, reverse_scatter.matrix), tol);
        EXPECT_LT(maxAbsDiff(owned_rows.vector, reverse_scatter.vector), tol);
    }
}

} // namespace testing
} // namespace assembly
} // namespace FE
} // namespace svmp

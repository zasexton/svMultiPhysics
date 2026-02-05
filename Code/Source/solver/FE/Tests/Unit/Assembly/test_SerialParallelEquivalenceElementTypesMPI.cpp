/**
 * @file test_SerialParallelEquivalenceElementTypesMPI.cpp
 * @brief MPI accuracy tests: serial vs parallel equivalence across 3D element types.
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
#include <functional>
#include <numeric>
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

struct ReferenceAssemblyResult {
    std::vector<Real> matrix;
    std::vector<Real> vector;
};

struct GlobalAssemblyResult {
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

GlobalAssemblyResult assembleGlobalSystem(ParallelAssembler& assembler,
                                         const IMeshAccess& mesh,
                                         const spaces::FunctionSpace& space,
                                         forms::FormKernel& matrix_kernel,
                                         forms::FormKernel& vector_kernel,
                                         GlobalIndex n_dofs,
                                         const dofs::DofMap& dof_map,
                                         MPI_Comm comm)
{
    DenseMatrixView A_local(n_dofs);
    DenseVectorView b_local(n_dofs);
    A_local.zero();
    b_local.zero();

    (void)assembler.assembleMatrix(mesh, space, space, matrix_kernel, A_local);
    (void)assembler.assembleVector(mesh, space, vector_kernel, b_local);
    assembler.finalize(&A_local, &b_local);

    expectOnlyOwnedRowsAreWritten(dof_map, A_local, b_local, /*tol=*/1e-14);

    return {
        allreduceSum(A_local.data(), comm),
        allreduceSum(b_local.data(), comm),
    };
}

std::vector<int> partitionTwoCells(int world_size)
{
    std::vector<int> owners(2, 0);
    if (world_size > 1) {
        owners[1] = 1;
    }
    return owners;
}

class TwoHexMeshAccess final : public IMeshAccess {
public:
    TwoHexMeshAccess(std::vector<int> cell_owner_ranks, int my_rank)
        : cell_owner_ranks_(std::move(cell_owner_ranks)),
          my_rank_(my_rank)
    {
        // Structured 2x1x1 grid: vertices (i,j,k) with i=0..2, j=0..1, k=0..1.
        nodes_.resize(12);
        auto vid = [](int i, int j, int k) -> GlobalIndex {
            return static_cast<GlobalIndex>(i + 3 * (j + 2 * k));
        };
        for (int k = 0; k <= 1; ++k) {
            for (int j = 0; j <= 1; ++j) {
                for (int i = 0; i <= 2; ++i) {
                    const Real x = static_cast<Real>(i) / Real(2.0);
                    const Real y = static_cast<Real>(j);
                    const Real z = static_cast<Real>(k);
                    nodes_[static_cast<std::size_t>(vid(i, j, k))] = {x, y, z};
                }
            }
        }

        cells_ = {
            std::array<GlobalIndex, 8>{
                vid(0, 0, 0), vid(1, 0, 0), vid(1, 1, 0), vid(0, 1, 0),
                vid(0, 0, 1), vid(1, 0, 1), vid(1, 1, 1), vid(0, 1, 1)},
            std::array<GlobalIndex, 8>{
                vid(1, 0, 0), vid(2, 0, 0), vid(2, 1, 0), vid(1, 1, 0),
                vid(1, 0, 1), vid(2, 0, 1), vid(2, 1, 1), vid(1, 1, 1)},
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
    std::array<std::array<GlobalIndex, 8>, 2> cells_{};
    std::vector<int> cell_owner_ranks_{};
    int my_rank_{0};
    std::vector<GlobalIndex> owned_cells_{};
};

dofs::MeshTopologyInfo buildTwoHexTopology(std::span<const int> cell_owner_ranks, int my_rank, int world_size)
{
    dofs::MeshTopologyInfo topo;
    topo.n_cells = 2;
    topo.n_vertices = 12;
    topo.dim = 3;
    topo.cell2vertex_offsets = {0, 8, 16};

    TwoHexMeshAccess tmp(std::vector<int>(cell_owner_ranks.begin(), cell_owner_ranks.end()), my_rank);
    std::vector<GlobalIndex> nodes;
    nodes.reserve(8);
    topo.cell2vertex_data.resize(16);
    for (GlobalIndex c = 0; c < topo.n_cells; ++c) {
        tmp.getCellNodes(c, nodes);
        for (std::size_t i = 0; i < nodes.size(); ++i) {
            topo.cell2vertex_data[static_cast<std::size_t>(8u * c + i)] = static_cast<MeshIndex>(nodes[i]);
        }
    }

    topo.vertex_gids.resize(static_cast<std::size_t>(topo.n_vertices), 0);
    for (GlobalIndex v = 0; v < topo.n_vertices; ++v) {
        topo.vertex_gids[static_cast<std::size_t>(v)] = static_cast<dofs::gid_t>(v);
    }

    topo.cell_gids = {0, 1};
    topo.cell_owner_ranks.assign(cell_owner_ranks.begin(), cell_owner_ranks.end());
    topo.neighbor_ranks = neighborRanks(my_rank, world_size);
    return topo;
}

class TwoWedgeMeshAccess final : public IMeshAccess {
public:
    TwoWedgeMeshAccess(std::vector<int> cell_owner_ranks, int my_rank)
        : cell_owner_ranks_(std::move(cell_owner_ranks)),
          my_rank_(my_rank)
    {
        nodes_ = {
            {0.0, 0.0, 0.0}, // 0
            {1.0, 0.0, 0.0}, // 1
            {0.0, 1.0, 0.0}, // 2
            {0.0, 0.0, 1.0}, // 3
            {1.0, 0.0, 1.0}, // 4
            {0.0, 1.0, 1.0}, // 5
            {0.0, 0.0, 2.0}, // 6
            {1.0, 0.0, 2.0}, // 7
            {0.0, 1.0, 2.0}, // 8
        };

        cells_ = {
            std::array<GlobalIndex, 6>{0, 1, 2, 3, 4, 5},
            std::array<GlobalIndex, 6>{3, 4, 5, 6, 7, 8},
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

    [[nodiscard]] ElementType getCellType(GlobalIndex /*cell_id*/) const override { return ElementType::Wedge6; }

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
    std::array<std::array<GlobalIndex, 6>, 2> cells_{};
    std::vector<int> cell_owner_ranks_{};
    int my_rank_{0};
    std::vector<GlobalIndex> owned_cells_{};
};

dofs::MeshTopologyInfo buildTwoWedgeTopology(std::span<const int> cell_owner_ranks, int my_rank, int world_size)
{
    dofs::MeshTopologyInfo topo;
    topo.n_cells = 2;
    topo.n_vertices = 9;
    topo.dim = 3;
    topo.cell2vertex_offsets = {0, 6, 12};

    topo.cell2vertex_data = {
        0, 1, 2, 3, 4, 5,
        3, 4, 5, 6, 7, 8,
    };

    topo.vertex_gids.resize(static_cast<std::size_t>(topo.n_vertices), 0);
    for (GlobalIndex v = 0; v < topo.n_vertices; ++v) {
        topo.vertex_gids[static_cast<std::size_t>(v)] = static_cast<dofs::gid_t>(v);
    }

    topo.cell_gids = {0, 1};
    topo.cell_owner_ranks.assign(cell_owner_ranks.begin(), cell_owner_ranks.end());
    topo.neighbor_ranks = neighborRanks(my_rank, world_size);
    return topo;
}

class TwoPyramidMeshAccess final : public IMeshAccess {
public:
    TwoPyramidMeshAccess(std::vector<int> cell_owner_ranks, int my_rank)
        : cell_owner_ranks_(std::move(cell_owner_ranks)),
          my_rank_(my_rank)
    {
        nodes_ = {
            {0.0, 0.0, 0.0}, // 0
            {1.0, 0.0, 0.0}, // 1
            {1.0, 1.0, 0.0}, // 2
            {0.0, 1.0, 0.0}, // 3
            {0.5, 0.5, 1.0}, // 4
            {0.5, 0.5, -1.0}, // 5
        };

        cells_ = {
            std::array<GlobalIndex, 5>{0, 1, 2, 3, 4},
            std::array<GlobalIndex, 5>{0, 1, 2, 3, 5},
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

    [[nodiscard]] ElementType getCellType(GlobalIndex /*cell_id*/) const override { return ElementType::Pyramid5; }

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
    std::array<std::array<GlobalIndex, 5>, 2> cells_{};
    std::vector<int> cell_owner_ranks_{};
    int my_rank_{0};
    std::vector<GlobalIndex> owned_cells_{};
};

dofs::MeshTopologyInfo buildTwoPyramidTopology(std::span<const int> cell_owner_ranks, int my_rank, int world_size)
{
    dofs::MeshTopologyInfo topo;
    topo.n_cells = 2;
    topo.n_vertices = 6;
    topo.dim = 3;
    topo.cell2vertex_offsets = {0, 5, 10};
    topo.cell2vertex_data = {
        0, 1, 2, 3, 4,
        0, 1, 2, 3, 5,
    };

    topo.vertex_gids.resize(static_cast<std::size_t>(topo.n_vertices), 0);
    for (GlobalIndex v = 0; v < topo.n_vertices; ++v) {
        topo.vertex_gids[static_cast<std::size_t>(v)] = static_cast<dofs::gid_t>(v);
    }

    topo.cell_gids = {0, 1};
    topo.cell_owner_ranks.assign(cell_owner_ranks.begin(), cell_owner_ranks.end());
    topo.neighbor_ranks = neighborRanks(my_rank, world_size);
    return topo;
}

void runPoissonEquivalenceTest(ElementType element_type,
                               const IMeshAccess& serial_mesh,
                               const IMeshAccess& parallel_mesh,
                               const dofs::MeshTopologyInfo& topo,
                               MPI_Comm comm,
                               int rank,
                               int size)
{
    spaces::H1Space space(element_type, /*order=*/1);
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
    const auto& dof_map = dof_handler.getDofMap();

    forms::FormCompiler compiler;
    const auto u = forms::TrialFunction(space, "u");
    const auto v = forms::TestFunction(space, "v");
    const auto X = forms::x();
    const auto f = X.component(0) + Real(2.0) * X.component(1) + Real(3.0) * X.component(2);

    auto bilinear_ir = compiler.compileBilinear(forms::inner(forms::grad(u), forms::grad(v)).dx());
    auto linear_ir = compiler.compileLinear((f * v).dx());

    forms::FormKernel matrix_kernel(std::move(bilinear_ir));
    forms::FormKernel vector_kernel(std::move(linear_ir));
    matrix_kernel.resolveInlinableConstitutives();
    vector_kernel.resolveInlinableConstitutives();

    ReferenceAssemblyResult ref;
    if (rank == 0) {
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

        return assembleGlobalSystem(assembler, parallel_mesh, space, matrix_kernel, vector_kernel, n_dofs, dof_map, comm);
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

} // namespace

TEST(SerialParallelEquivalenceElementTypesMPI, Hex8MatrixAndVectorMatchSerial)
{
    MPI_Comm comm = MPI_COMM_WORLD;
    const int rank = mpiRank(comm);
    const int size = mpiSize(comm);
    if (size < 2) {
        GTEST_SKIP() << "Run with 2+ MPI ranks to enable this test";
    }

    const auto owners = partitionTwoCells(size);
    TwoHexMeshAccess mesh(owners, rank);
    TwoHexMeshAccess serial_mesh(std::vector<int>(owners.size(), 0), /*my_rank=*/0);
    const auto topo = buildTwoHexTopology(owners, rank, size);

    runPoissonEquivalenceTest(ElementType::Hex8, serial_mesh, mesh, topo, comm, rank, size);
}

TEST(SerialParallelEquivalenceElementTypesMPI, Wedge6MatrixAndVectorMatchSerial)
{
    MPI_Comm comm = MPI_COMM_WORLD;
    const int rank = mpiRank(comm);
    const int size = mpiSize(comm);
    if (size < 2) {
        GTEST_SKIP() << "Run with 2+ MPI ranks to enable this test";
    }

    const auto owners = partitionTwoCells(size);
    TwoWedgeMeshAccess mesh(owners, rank);
    TwoWedgeMeshAccess serial_mesh(std::vector<int>(owners.size(), 0), /*my_rank=*/0);
    const auto topo = buildTwoWedgeTopology(owners, rank, size);

    runPoissonEquivalenceTest(ElementType::Wedge6, serial_mesh, mesh, topo, comm, rank, size);
}

TEST(SerialParallelEquivalenceElementTypesMPI, Pyramid5MatrixAndVectorMatchSerial)
{
    MPI_Comm comm = MPI_COMM_WORLD;
    const int rank = mpiRank(comm);
    const int size = mpiSize(comm);
    if (size < 2) {
        GTEST_SKIP() << "Run with 2+ MPI ranks to enable this test";
    }

    const auto owners = partitionTwoCells(size);
    TwoPyramidMeshAccess mesh(owners, rank);
    TwoPyramidMeshAccess serial_mesh(std::vector<int>(owners.size(), 0), /*my_rank=*/0);
    const auto topo = buildTwoPyramidTopology(owners, rank, size);

    runPoissonEquivalenceTest(ElementType::Pyramid5, serial_mesh, mesh, topo, comm, rank, size);
}

} // namespace svmp::FE::assembly::testing
